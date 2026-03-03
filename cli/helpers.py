"""Shared CLI helpers — utilities used across multiple commands.

Extracted from cli/main.py to enable the modular command structure.
Contains:
- collect_files()   — Resolve paths (files, dirs, globs) into a flat file list
- resolve_output()  — Output path & format resolution for transcribe/export
- PhaseTracker      — Rich Live progress display for transcription phases
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from rich.live import Live
from rich.text import Text

from cli.theme import (
    ACCENT,
    DIM,
    FORMAT_TO_EXT,
    SUCCESS,
    console,
    detect_format_from_path,
    format_duration,
)


# ── File Collection ─────────────────────────────────────────


def collect_files(
    paths: tuple[str, ...],
    *,
    recursive: bool = False,
    extensions: str | None = None,
    from_file: str | None = None,
    exclude: str | None = None,
) -> list[Path]:
    """Resolve CLI input paths into a flat, deduplicated list of files.

    Handles:
    - Individual files passed directly
    - Directories (walks for supported audio/video files)
    - Recursive walking (-R / --recursive)
    - Extension filtering (--ext mp3,m4a)
    - Manifest file (--from-file list.txt)
    - Stdin piping (path "-" reads file paths from stdin)
    - Exclude patterns (--exclude "*_draft*")
    - Deduplication via resolved paths

    Args:
        paths: Tuple of file/directory paths from Click argument.
        recursive: Walk directories recursively.
        extensions: Comma-separated list of extensions to include (e.g. "mp3,m4a").
        from_file: Path to a manifest file containing one path per line.
        exclude: Comma-separated glob patterns to exclude.

    Returns:
        Sorted, deduplicated list of Path objects.
    """
    from src.audiobench.core.ffmpeg import ALL_SUPPORTED_FORMATS

    # Build the allowed extension set
    if extensions:
        allowed_exts = {e.strip().lstrip(".").lower() for e in extensions.split(",")}
    else:
        allowed_exts = ALL_SUPPORTED_FORMATS

    # Build exclude patterns
    exclude_patterns = []
    if exclude:
        exclude_patterns = [p.strip() for p in exclude.split(",")]

    def _is_excluded(p: Path) -> bool:
        """Check if a path matches any exclude pattern."""
        if not exclude_patterns:
            return False
        import fnmatch

        name = p.name
        return any(fnmatch.fnmatch(name, pat) for pat in exclude_patterns)

    def _is_supported(p: Path) -> bool:
        """Check if a file has a supported extension."""
        ext = p.suffix.lstrip(".").lower()
        return ext in allowed_exts

    def _walk_directory(dir_path: Path) -> list[Path]:
        """Collect supported files from a directory."""
        if recursive:
            return sorted(
                f
                for f in dir_path.rglob("*")
                if f.is_file() and _is_supported(f) and not _is_excluded(f)
            )
        else:
            return sorted(
                f
                for f in dir_path.iterdir()
                if f.is_file() and _is_supported(f) and not _is_excluded(f)
            )

    collected: list[Path] = []

    # Collect from manifest file
    if from_file:
        manifest = Path(from_file)
        if manifest.exists():
            for line in manifest.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    p = Path(line).expanduser()
                    if p.exists():
                        collected.append(p)

    # Collect from stdin ("-")
    for path_str in paths:
        if path_str == "-":
            for line in sys.stdin:
                line = line.strip()
                if line:
                    p = Path(line).expanduser()
                    if p.exists():
                        collected.append(p)
            continue

        p = Path(path_str)

        if p.is_dir():
            collected.extend(_walk_directory(p))
        elif p.is_file() and not _is_excluded(p):
            collected.append(p)
        # else: Click's exists=True already validated, but be safe

    # Deduplicate via resolved paths, preserving order
    seen: set[Path] = set()
    deduped: list[Path] = []
    for f in collected:
        resolved = f.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(f)

    return deduped


# ── Output Resolution ───────────────────────────────────────


def resolve_collision(path: str, strategy: str) -> str | None:
    """Apply collision strategy when output file already exists.

    Args:
        path: The target output path.
        strategy: One of 'overwrite', 'skip', 'rename'.

    Returns:
        Final path to write to, or None if 'skip'.
    """
    p = Path(path)
    if not p.exists():
        return path

    if strategy == "overwrite":
        return path
    elif strategy == "skip":
        return None  # Caller should skip this file
    elif strategy == "rename":
        # Auto-increment: file.srt → file_1.srt → file_2.srt
        stem = p.stem
        suffix = p.suffix
        parent = p.parent
        counter = 1
        while True:
            candidate = parent / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return str(candidate)
            counter += 1
    return path  # fallback


def resolve_output(
    input_path: str,
    output_path: str | None,
    output_format: str | None,
    default_format: str,
    *,
    input_base_dir: str | None = None,
    collision: str = "overwrite",
) -> tuple[str | None, str]:
    """Resolve output path and format from CLI args.

    Rules:
        1. -o path.srt              → auto-detect format from extension
        2. -f srt (no -o)           → <stem>.srt in same dir as input
        3. -o dir/ (existing dir)   → dir/<stem>.<fmt>
        3b. Mirror mode: if input_base_dir is set, preserves relative
            structure (e.g., base/sub/file.m4a → out/sub/file.srt)
        4. Neither -o nor -f        → None (print to stdout)

    Args:
        input_path: Path to the input audio file.
        output_path: User-provided output path (-o flag).
        output_format: User-provided format (-f flag).
        default_format: Fallback format from settings.
        input_base_dir: If set, enables mirror mode — preserves relative
            directory structure under the output directory.
        collision: Strategy for existing files: 'overwrite', 'skip', 'rename'.

    Returns:
        Tuple of (resolved_output_path or None, format_string).
    """
    input_p = Path(input_path)
    stem = input_p.stem

    # Rule 1: -o with extension → auto-detect format
    if output_path:
        out_p = Path(output_path)

        # Rule 3: output is a directory
        if out_p.is_dir() or output_path.endswith("/"):
            fmt = output_format or default_format
            ext = FORMAT_TO_EXT.get(fmt, f".{fmt}")
            out_p.mkdir(parents=True, exist_ok=True)

            # Mirror mode (C1): preserve relative directory structure
            if input_base_dir:
                try:
                    rel = Path(input_path).resolve().relative_to(Path(input_base_dir).resolve())
                    # Replace the file extension, keep the dir structure
                    mirrored = out_p / rel.with_suffix(ext)
                    mirrored.parent.mkdir(parents=True, exist_ok=True)
                    final = str(mirrored)
                except ValueError:
                    # input_path is not relative to input_base_dir, fallback
                    final = str(out_p / f"{stem}{ext}")
            else:
                final = str(out_p / f"{stem}{ext}")

            # C2: collision handling
            final = resolve_collision(final, collision)
            return final, fmt

        # Rule 1: detect format from output extension
        detected = detect_format_from_path(output_path)
        fmt = output_format or detected or default_format
        final = resolve_collision(output_path, collision)
        return final, fmt

    # Rule 2: -f specified but no -o → auto-name
    if output_format:
        ext = FORMAT_TO_EXT.get(output_format, f".{output_format}")
        auto_path = str(input_p.with_suffix(ext))
        final = resolve_collision(auto_path, collision)
        return final, output_format

    # Rule 4: neither → stdout
    return None, default_format


def parse_formats(format_str: str | None) -> list[str]:
    """Parse multi-format string like 'srt,json' or 'all'.

    Returns a list of format strings. If None, returns empty list
    (caller should use default_format).
    """
    valid_formats = {"txt", "srt", "vtt", "json"}

    if not format_str:
        return []

    if format_str.strip().lower() == "all":
        return sorted(valid_formats)

    formats = [f.strip().lower() for f in format_str.split(",")]
    invalid = [f for f in formats if f not in valid_formats]
    if invalid:
        from cli.theme import console, error_panel

        console.print(
            error_panel(
                "Invalid format",
                f"Unknown format(s): {', '.join(invalid)}. "
                f"Valid: {', '.join(sorted(valid_formats))}",
            )
        )
        return []  # Return empty to signal error

    return formats


# ── Phase Tracker ────────────────────────────────────────────


class PhaseTracker:
    """Renders phased progress using Rich Live display.

    Uses a two-mode approach:
    1. **Live mode** — During loading/converting, a Rich Live display
       shows animated spinners and progress. Uses transient=True so
       the frame vanishes when stopped.
    2. **Streaming mode** — When the first transcript segment arrives,
       Live stops, completed phases print statically at the top, and
       each new segment prints below. Text grows downward in real-time.

    The result: phases stay at the top, transcript builds below,
    summary appears at the very bottom when done.
    """

    PHASES = ["loading", "converting", "transcribing", "saving"]
    LABELS = {
        "loading": "Loading model",
        "converting": "Converting audio",
        "transcribing": "Transcribing",
        "saving": "Saving",
    }
    SPINNERS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        self.phase_times: dict[str, float] = {}
        self._current_phase: str | None = None
        self._phase_start: float = 0
        self._last_progress: float = 0
        self._spin_idx: int = 0
        # Accumulated segments for live preview + partial save
        self.segments: list = []
        # Rich Live display — handles smooth in-place terminal updates
        self._live: Live | None = None
        # Whether we've switched to streaming mode
        self._streaming: bool = False

    def start(self) -> None:
        """Start the Rich Live display. Call before first update."""
        if not self.quiet:
            self._live = Live(
                self,
                console=console,
                refresh_per_second=10,
                transient=True,  # Frame vanishes when stopped
            )
            self._live.start()

    def _enter_streaming(self) -> None:
        """Transition from Live mode to streaming mode.

        Stops the Live display (frame vanishes due to transient=True),
        then prints completed phases as static text. After this,
        segments print below via regular console.print().
        """
        if self._streaming:
            return
        self._streaming = True

        # Stop Live — transient=True means the frame disappears cleanly
        if self._live:
            self._live.stop()
            self._live = None

        # Print phases statically at the top
        for phase in self.PHASES:
            label = self.LABELS.get(phase, phase)
            if phase in self.phase_times:
                elapsed_str = format_duration(self.phase_times[phase])
                console.print(f"  [{SUCCESS}]✓[/]  {label:<24} [{DIM}]{elapsed_str}[/]")
            elif phase == self._current_phase:
                console.print(f"  [{ACCENT}]◐[/]  {label}...")
            else:
                console.print(f"  [{DIM}]·  {label}[/]")

        # Blank line separating phases from transcript text
        console.print()

    def on_segment(self, segment: object) -> None:
        """Called after each segment is transcribed.

        On first call, switches to streaming mode (phases at top).
        Then prints each segment below, growing the transcript.
        """
        self.segments.append(segment)
        if self.quiet:
            return

        # First segment → switch to streaming mode
        if not self._streaming:
            self._enter_streaming()

        text = getattr(segment, "text", "").strip()
        if text:
            console.print(text, highlight=False)

    def update(self, phase: str, message: str, progress: float | None) -> None:
        """Called by the pipeline on phase transitions."""
        if self.quiet:
            return

        # Record timing for previous phase
        if self._current_phase and self._current_phase != phase:
            elapsed = time.perf_counter() - self._phase_start
            self.phase_times[self._current_phase] = elapsed

        if phase != self._current_phase:
            self._current_phase = phase
            self._phase_start = time.perf_counter()

        if progress is not None:
            self._last_progress = progress

    def _build_display(self) -> Text:
        """Build the Live display (loading/converting phases only)."""

        self._spin_idx = (self._spin_idx + 1) % len(self.SPINNERS)
        spinner = self.SPINNERS[self._spin_idx]

        display = Text()
        for phase in self.PHASES:
            label = self.LABELS.get(phase, phase)

            if phase in self.phase_times:
                # ✓ Completed
                elapsed_str = format_duration(self.phase_times[phase])
                display.append("  ✓", style=SUCCESS)
                display.append(f"  {label:<24} ", style="")
                display.append(elapsed_str, style=DIM)
                display.append("\n")
            elif phase == self._current_phase:
                # ⠼ Active with spinner
                display.append(f"  {spinner}", style=ACCENT)
                display.append(f"  {label}", style="")
                display.append("...", style=DIM)
                display.append("\n")
            else:
                # · Pending
                display.append("  ·", style=DIM)
                display.append(f"  {label}", style=DIM)
                display.append("\n")

        return display

    def __rich_console__(self, rconsole, options):
        """Rich renderable protocol — called every refresh cycle."""
        yield self._build_display()

    def finalize(self) -> None:
        """Record final timing and print completion summary."""
        if self.quiet:
            return

        if self._current_phase:
            elapsed = time.perf_counter() - self._phase_start
            self.phase_times[self._current_phase] = elapsed

        if self._streaming:
            # Phases already printed at top — nothing more to do
            pass
        else:
            # Still in Live mode (no segments came) — stop normally
            if self._live:
                self._live.refresh()
                self._live.stop()
                self._live = None

    def save_partial(self, input_path: str) -> str | None:
        """Save accumulated segments to a .partial.txt file."""
        if not self.segments:
            return None
        partial_path = str(Path(input_path).with_suffix(".partial.txt"))
        lines = []
        for seg in self.segments:
            start = getattr(seg, "start", 0)
            text = getattr(seg, "text", "")
            minutes = int(start // 60)
            seconds = int(start % 60)
            lines.append(f"[{minutes}:{seconds:02d}] {text}")
        Path(partial_path).write_text("\n".join(lines), encoding="utf-8")
        return partial_path
