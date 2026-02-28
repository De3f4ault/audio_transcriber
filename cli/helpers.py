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


def resolve_output(
    input_path: str,
    output_path: str | None,
    output_format: str | None,
    default_format: str,
) -> tuple[str | None, str]:
    """Resolve output path and format from CLI args.

    Rules:
        1. -o path.srt              → auto-detect format from extension
        2. -f srt (no -o)           → <stem>.srt in same dir as input
        3. -o dir/ (existing dir)   → dir/<stem>.<fmt>
        4. Neither -o nor -f        → None (print to stdout)
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
            return str(out_p / f"{stem}{ext}"), fmt

        # Rule 1: detect format from output extension
        detected = detect_format_from_path(output_path)
        fmt = output_format or detected or default_format
        return output_path, fmt

    # Rule 2: -f specified but no -o → auto-name
    if output_format:
        ext = FORMAT_TO_EXT.get(output_format, f".{output_format}")
        return str(input_p.with_suffix(ext)), output_format

    # Rule 4: neither → stdout
    return None, default_format


# ── Phase Tracker ────────────────────────────────────────────


class PhaseTracker:
    """Renders phased progress using Rich Live display.

    Uses Rich's Live context to render a single persistent block
    that updates in-place — no flickering, no ANSI cursor hacks.

    Features:
    - Completed phases show ✓ with elapsed time
    - Active phase shows a spinner + progress bar (for transcription)
    - Future phases show · (dimmed)
    - Live segment preview below active transcription
    - Supports SIGINT graceful partial save
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
        self._last_segment_text: str = ""
        # Accumulated segments for live preview + partial save
        self.segments: list = []
        # Rich Live display — handles smooth in-place terminal updates
        self._live: Live | None = None

    def start(self) -> None:
        """Start the Rich Live display. Call before first update."""
        if not self.quiet:
            # Pass `self` as the renderable — Rich calls our
            # __rich_console__ every refresh cycle (10fps),
            # giving us continuous spinner animation.
            self._live = Live(
                self,
                console=console,
                refresh_per_second=10,
            )
            self._live.start()

    def on_segment(self, segment: object) -> None:
        """Called after each segment is transcribed."""
        self.segments.append(segment)
        if self.quiet:
            return
        text = getattr(segment, "text", "").strip()
        if text:
            if len(text) > 80:
                self._last_segment_text = "..." + text[-77:]
            else:
                self._last_segment_text = text

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
            self._last_segment_text = ""

        if progress is not None:
            self._last_progress = progress

    def _build_display(self) -> Text:
        """Build the current display as a Rich Text object."""

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
                if phase == "transcribing" and self._last_progress > 0:
                    pct = self._last_progress
                    elapsed_str = format_duration(time.perf_counter() - self._phase_start)
                    display.append(f"  {label:<16}", style="")
                    # Progress bar
                    filled = int(28 * pct / 100)
                    remaining = 28 - filled
                    display.append("━" * filled, style=ACCENT)
                    display.append("━" * remaining, style=DIM)
                    display.append(f"  {pct:.0f}%", style="bold")
                    display.append(f"  {elapsed_str}", style=DIM)
                    display.append("\n")
                    # Live segment preview
                    if self._last_segment_text:
                        display.append("     ╰ ", style=DIM)
                        preview = self._last_segment_text
                        display.append(f'"{preview}"', style=DIM)
                        display.append("\n")
                else:
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
        """Record final timing and stop the Live display."""
        if self.quiet:
            return

        if self._current_phase:
            elapsed = time.perf_counter() - self._phase_start
            self.phase_times[self._current_phase] = elapsed

        # Force one final refresh then stop Live
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
