"""Click CLI — user-facing command-line interface.

Commands:
    transcribe  — Transcribe audio/video files (supports batch/glob)
    history     — View transcription history
    search      — Search past transcriptions
    export      — Re-export a past transcription to a different format
    download    — Pre-download a Whisper model
    delete      — Remove transcription(s) from history
    info        — Show system info and settings

Usage:
    transcriber transcribe meeting.m4a               → stdout (txt)
    transcriber transcribe meeting.m4a -f srt         → meeting.srt (auto-named)
    transcriber transcribe meeting.m4a -o notes.srt   → format from extension
    transcriber transcribe *.m4a -o ./out/            → batch mode
    transcriber transcribe meeting.m4a --fast         → speed preset
    transcriber transcribe meeting.m4a -q             → quiet (for piping)
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

import click
from rich.live import Live
from rich.text import Text

from cli.theme import (
    ACCENT,
    APP_NAME,
    APP_VERSION,
    BOLD,
    DIM,
    FORMAT_TO_EXT,
    SUCCESS,
    console,
    detect_format_from_path,
    error_panel,
    format_duration,
    format_size,
    make_table,
    stdout,
    summary_panel,
)
from src.transcriber.config.logging_config import setup_logging
from src.transcriber.config.settings import get_settings


# ── Helpers ──────────────────────────────────────────────────


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


# ── CLI Group ────────────────────────────────────────────────


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Show detailed log output")
@click.option("--debug", is_flag=True, help="Debug logging")
@click.version_option(version=APP_VERSION, prog_name="transcriber")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, debug: bool) -> None:
    """🎙️ Audio Transcriber — professional-grade offline transcription.

    \b
    Transcribe files:
      transcriber transcribe meeting.m4a                 Print to stdout
      transcriber transcribe meeting.m4a -f srt          Save as meeting.srt
      transcriber transcribe meeting.m4a -o notes.srt    Auto-detect SRT format
      transcriber transcribe *.m4a -o ./out/             Batch to directory
      transcriber transcribe meeting.m4a --fast          Fast preset
      transcriber transcribe meeting.m4a -q | grep word  Pipe-friendly

    \b
    Manage:
      transcriber history                                Past transcriptions
      transcriber search "keyword"                       Search text
      transcriber export 3 -f vtt                        Re-export as VTT
      transcriber download large-v3-turbo                Pre-download model
      transcriber delete 3                               Remove from history
      transcriber info                                   System info
    """
    if debug:
        log_level = "DEBUG"
    elif verbose:
        log_level = "INFO"
    else:
        log_level = "WARNING"
    setup_logging(log_level)
    ctx.ensure_object(dict)


# ── Transcribe Command ──────────────────────────────────────


@cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True), required=True)
@click.option(
    "-f",
    "--format",
    "output_format",
    default=None,
    type=click.Choice(["txt", "srt", "vtt", "json"]),
    help="Output format",
)
@click.option("-o", "--output", "output_path", default=None, help="Output path (file or directory)")
@click.option(
    "-l", "--language", default=None, help="Language code (e.g., en, sw). Default: auto-detect"
)
@click.option(
    "-m",
    "--model",
    default=None,
    type=click.Choice(["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]),
    help="Whisper model",
)
@click.option("--fast", "speed_preset", flag_value="fast", help="⚡ Fast: beam=1, batch=4")
@click.option(
    "--balanced",
    "speed_preset",
    flag_value="balanced",
    default=True,
    help="⚖️ Balanced: beam=3, batch=4 (default)",
)
@click.option(
    "--accurate", "speed_preset", flag_value="accurate", help="🎯 Accurate: beam=5, sequential"
)
@click.option("--no-cache", is_flag=True, help="Re-transcribe even if cached")
@click.option("--no-timestamps", is_flag=True, help="Disable word timestamps")
@click.option("-q", "--quiet", is_flag=True, help="Quiet mode (raw output only, for piping)")
@click.option("--check", is_flag=True, help="Show file metadata only (no transcription)")
@click.option("--enhance", is_flag=True, help="Apply noise reduction + normalization filters")
@click.option("--filter", "audio_filter", default=None, help="Custom ffmpeg audio filter graph")
@click.option(
    "--prompt",
    "initial_prompt",
    default=None,
    help="Guide model with context (e.g., 'Conversation in Swahili and English')",
)
def transcribe(
    files: tuple[str, ...],
    output_format: str | None,
    output_path: str | None,
    language: str | None,
    model: str | None,
    speed_preset: str,
    no_cache: bool,
    no_timestamps: bool,
    quiet: bool,
    check: bool,
    enhance: bool,
    audio_filter: str | None,
    initial_prompt: str | None,
) -> None:
    """Transcribe audio/video files.

    \b
    Examples:
      transcriber transcribe meeting.m4a                  Print to stdout
      transcriber transcribe meeting.m4a -f srt           Auto-save meeting.srt
      transcriber transcribe meeting.m4a -o notes.srt     Format from extension
      transcriber transcribe *.m4a -o ./out/              Batch to directory
      transcriber transcribe --fast lecture.mp3            Fast preset
      transcriber transcribe -q meeting.m4a | grep word   Pipe-friendly
    """
    from src.transcriber.core.pipeline import TranscriptionPipeline

    settings = get_settings()
    if model:
        settings.model_name = model

    # Build filter list
    filters: list[str] | None = None
    if enhance:
        from src.transcriber.core.ffmpeg import ENHANCE_FILTERS

        filters = list(ENHANCE_FILTERS)
    if audio_filter:
        filters = audio_filter.split(",") if not filters else filters + audio_filter.split(",")

    # --check: show metadata only, no transcription
    if check:
        from src.transcriber.core.ffmpeg import probe

        for file_path in files:
            input_p = Path(file_path)
            info = probe(file_path)
            table = make_table(
                f"File: {input_p.name}",
                [
                    ("Property", {"style": BOLD}),
                    ("Value", {}),
                ],
            )
            table.add_row("Codec", info.codec)
            table.add_row("Duration", format_duration(info.duration))
            table.add_row("Sample Rate", f"{info.sample_rate} Hz")
            table.add_row("Channels", str(info.channels))
            if info.bitrate:
                table.add_row("Bitrate", f"{info.bitrate // 1000} kbps")
            table.add_row("Container", info.format_name)
            table.add_row("Size", format_size(input_p.stat().st_size))
            if filters:
                table.add_row("Filters", ", ".join(filters))
            console.print(table)
            console.print(f"  [{SUCCESS}]Ready to transcribe.[/]")
        return

    preset_icons = {"fast": "fast", "balanced": "balanced", "accurate": "accurate"}
    preset_label = preset_icons.get(speed_preset, speed_preset)

    pipeline = TranscriptionPipeline()
    results: list[dict] = []

    for file_path in files:
        input_p = Path(file_path)
        file_size = input_p.stat().st_size

        # Resolve output path and format
        resolved_output, resolved_format = resolve_output(
            file_path,
            output_path,
            output_format,
            settings.output_format,
        )

        # ── Header ──
        if not quiet:
            console.print()
            console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/]")
            console.print(f"  [{DIM}]{'─' * 44}[/]")
            console.print(f"    File:    [{ACCENT}]{input_p.name}[/] ({format_size(file_size)})")
            console.print(f"    Model:   {settings.model_name} | Preset: {preset_label}")
            if filters:
                console.print(f"    Filters: [{DIM}]{', '.join(filters)}[/]")
            if resolved_output:
                console.print(f"    Output:  [{DIM}]{resolved_output}[/]")
            console.print(f"  [{DIM}]{'─' * 44}[/]")

        start_time = time.perf_counter()
        tracker = PhaseTracker(quiet=quiet)
        tracker.start()

        # SIGINT handler for graceful partial save
        original_handler = signal.getsignal(signal.SIGINT)

        def handle_interrupt(
            signum: int,
            frame: object,
            _tracker: PhaseTracker = tracker,
            _file_path: str = file_path,
            _original: object = original_handler,
        ) -> None:
            partial_path = _tracker.save_partial(_file_path)
            if partial_path:
                console.print(
                    f"\n  [{ACCENT}]Interrupted. Partial transcript saved to: {partial_path}[/]"
                )
            else:
                console.print(f"\n  [{DIM}]Interrupted. No segments transcribed yet.[/]")
            signal.signal(signal.SIGINT, _original)  # restore
            sys.exit(130)

        signal.signal(signal.SIGINT, handle_interrupt)

        try:
            transcript = pipeline.transcribe_file(
                file_path=file_path,
                language=language,
                output_format=resolved_format,
                output_path=resolved_output,
                word_timestamps=not no_timestamps,
                skip_cache=no_cache,
                speed_preset=speed_preset,
                initial_prompt=initial_prompt,
                on_phase=tracker.update,
                on_segment=tracker.on_segment,
                filters=filters,
            )

            tracker.finalize()

            elapsed = time.perf_counter() - start_time
            speed_ratio = transcript.duration_seconds / elapsed if elapsed > 0 else 0

            # ── Output ──
            if quiet:
                from src.transcriber.output.base import get_formatter

                formatter = get_formatter(resolved_format)
                stdout.print(formatter.format(transcript), highlight=False)
            else:
                # Print transcript to terminal if no file output
                if not resolved_output:
                    from src.transcriber.output.base import get_formatter

                    formatter = get_formatter(resolved_format)
                    console.print()
                    console.print(formatter.format(transcript))

                # ── Summary ──
                console.print()
                console.print(
                    summary_panel(
                        [
                            f"  [{SUCCESS}]✓ Done in {format_duration(elapsed)}[/]"
                            f"  [{DIM}]•  {speed_ratio:.1f}x real-time[/]",
                            "",
                            f"  Language   [{BOLD}]{transcript.language}[/] "
                            f"({transcript.language_probability * 100:.0f}%)"
                            f"     Segments  {transcript.segment_count}",
                            f"  Words      {transcript.word_count}"
                            f"              Audio     "
                            f"{format_duration(transcript.duration_seconds)}",
                        ]
                    )
                )

                if resolved_output:
                    console.print(f"  [{DIM}]Saved → {resolved_output}[/]")

            results.append(
                {
                    "file": input_p.name,
                    "words": transcript.word_count,
                    "duration": transcript.duration_seconds,
                    "elapsed": elapsed,
                    "speed": speed_ratio,
                    "language": transcript.language,
                }
            )

        except Exception as e:
            if not quiet:
                tracker.finalize()
                console.print(error_panel(f"Failed: {input_p.name}", str(e)))
            else:
                print(f"Error: {input_p.name}: {e}", file=sys.stderr)

    # ── Batch summary ──
    if len(results) > 1 and not quiet:
        console.print()
        table = make_table(
            "Batch Summary",
            [
                ("File", {"style": ACCENT}),
                ("Words", {"justify": "right", "width": 6}),
                ("Duration", {"justify": "right", "width": 10}),
                ("Processed", {"justify": "right", "width": 10}),
                ("Speed", {"justify": "right", "width": 8}),
            ],
        )
        for r in results:
            table.add_row(
                r["file"],
                str(r["words"]),
                format_duration(r["duration"]),
                format_duration(r["elapsed"]),
                f"{r['speed']:.1f}x",
            )
        console.print(table)


# ── History Command ──────────────────────────────────────────


@cli.command()
@click.option("--limit", default=20, help="Number of records to show")
def history(limit: int) -> None:
    """View transcription history."""
    from src.transcriber.storage.database import init_db
    from src.transcriber.storage.repository import TranscriptionRepository

    init_db()
    repo = TranscriptionRepository()
    records = repo.get_history(limit=limit)

    if not records:
        console.print(f"  [{DIM}]No transcription history yet.[/]")
        return

    table = make_table(
        "📝 Transcription History",
        [
            ("#", {"style": DIM, "width": 4}),
            ("File", {"style": ACCENT}),
            ("Language", {"width": 8}),
            ("Model", {"width": 16}),
            ("Words", {"justify": "right", "width": 6}),
            ("Duration", {"justify": "right", "width": 10}),
            ("Date", {"style": DIM, "width": 10}),
            ("Preview", {"max_width": 35}),
        ],
    )

    for rec in records:
        dur = format_duration(rec["duration"]) if rec["duration"] else "–"
        date = rec["created_at"][:10] if rec["created_at"] else "–"
        table.add_row(
            str(rec["id"]),
            rec["file_name"],
            rec["language"],
            rec["model"],
            str(rec["word_count"]),
            dur,
            date,
            (rec["text_preview"] or "")[:35],
        )

    console.print(table)


# ── Search Command ───────────────────────────────────────────


@cli.command()
@click.argument("query")
@click.option("--limit", default=10, help="Max results")
def search(query: str, limit: int) -> None:
    """Search past transcriptions by text content."""
    from src.transcriber.storage.database import init_db
    from src.transcriber.storage.repository import TranscriptionRepository

    init_db()
    repo = TranscriptionRepository()
    results = repo.search(query, limit=limit)

    if not results:
        console.print(f'  [{DIM}]No results for "{query}"[/]')
        return

    for r in results:
        from rich.panel import Panel

        console.print(
            Panel(
                f"[{ACCENT}]{r['file_name']}[/] ({r['language']}) — "
                f"[{DIM}]{r['created_at'][:10]}[/]\n"
                f"{r['text_preview']}",
                title=f"[{DIM}]#{r['id']}[/]",
                border_style=DIM,
            )
        )


# ── Export Command ───────────────────────────────────────────


@cli.command()
@click.argument("transcription_id", type=int)
@click.option(
    "-f",
    "--format",
    "output_format",
    required=True,
    type=click.Choice(["txt", "srt", "vtt", "json"]),
    help="Output format",
)
@click.option("-o", "--output", "output_path", default=None, help="Output file path")
def export(
    transcription_id: int,
    output_format: str,
    output_path: str | None,
) -> None:
    """Re-export a past transcription to a different format."""
    from src.transcriber.core.models import Segment, Transcript
    from src.transcriber.output.base import get_formatter
    from src.transcriber.storage.database import init_db
    from src.transcriber.storage.repository import TranscriptionRepository

    init_db()
    repo = TranscriptionRepository()
    data = repo.get_by_id(transcription_id)

    if not data:
        console.print(error_panel(f"Transcription #{transcription_id} not found"))
        sys.exit(1)

    segments = [
        Segment(
            id=s["index"],
            text=s["text"],
            start=s["start"],
            end=s["end"],
            speaker=s.get("speaker"),
        )
        for s in data.get("segments", [])
    ]

    transcript = Transcript(
        segments=segments,
        language=data.get("language", "en"),
        language_probability=data.get("language_probability", 0.0),
        duration_seconds=data.get("duration", 0.0),
        engine=data.get("engine", "faster-whisper"),
        model_name=data.get("model", "unknown"),
    )

    formatter = get_formatter(output_format)
    content = formatter.format(transcript)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        console.print(
            f"  [{SUCCESS}]✓[/] Exported #{transcription_id} as "
            f"{output_format.upper()} → [{ACCENT}]{output_path}[/]"
        )
    else:
        stdout.print(content, highlight=False)


# ── Delete Command ───────────────────────────────────────────


@cli.command()
@click.argument("transcription_id", type=int, required=False)
@click.option("--all", "delete_all", is_flag=True, help="Delete all transcriptions")
@click.confirmation_option(prompt="Are you sure?")
def delete(transcription_id: int | None, delete_all: bool) -> None:
    """Delete transcription(s) from history."""
    from src.transcriber.storage.database import init_db
    from src.transcriber.storage.repository import TranscriptionRepository

    init_db()
    repo = TranscriptionRepository()

    if delete_all:
        count = repo.delete_all()
        console.print(f"  [{SUCCESS}]✓[/] Deleted {count} transcription(s)")
    elif transcription_id is not None:
        ok = repo.delete_by_id(transcription_id)
        if ok:
            console.print(f"  [{SUCCESS}]✓[/] Deleted transcription #{transcription_id}")
        else:
            console.print(error_panel(f"Transcription #{transcription_id} not found"))
            sys.exit(1)
    else:
        console.print(error_panel("Specify a transcription ID or --all"))
        sys.exit(1)


# ── Download Command ─────────────────────────────────────────


@cli.command()
@click.argument(
    "model_name",
    type=click.Choice(
        [
            "tiny",
            "base",
            "small",
            "medium",
            "large-v3",
            "large-v3-turbo",
        ]
    ),
)
def download(model_name: str) -> None:
    """Pre-download a Whisper model for offline use."""
    console.print(f"  [{ACCENT}]Downloading model:[/] {model_name} ...")

    try:
        from faster_whisper import WhisperModel

        WhisperModel(model_name, device="cpu", compute_type="int8")
        console.print(f"  [{SUCCESS}]✓[/] Model '{model_name}' downloaded and cached.")
    except Exception as e:
        console.print(error_panel(f"Download failed: {model_name}", str(e)))
        sys.exit(1)


# ── Info Command ─────────────────────────────────────────────


@cli.command()
def info() -> None:
    """Show system info and current settings."""
    settings = get_settings()

    cuda_available = False
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass

    device_str = settings.resolve_device()
    device_label = f"{device_str} ({'CUDA' if cuda_available else 'CPU only'})"

    table = make_table(
        f"⚙️  {APP_NAME} v{APP_VERSION}",
        [
            ("Setting", {"style": BOLD}),
            ("Value", {}),
        ],
    )

    table.add_row("Model", settings.model_name)
    table.add_row("Device", device_label)
    table.add_row("Compute Type", settings.resolve_compute_type())
    table.add_row("CPU Threads", str(settings.resolve_cpu_threads()))
    table.add_row("Speed Preset", settings.speed_preset)
    table.add_row("Beam Size", str(settings.resolve_beam_size()))
    table.add_row("Batch Size", str(settings.resolve_batch_size()))
    table.add_row("Language", settings.language or "auto-detect")
    table.add_row("Output Format", settings.output_format)
    table.add_row("Word Timestamps", str(settings.word_timestamps))
    table.add_row("Diarization", str(settings.enable_diarization))
    table.add_row("Database", settings.database_url)
    table.add_row("Models Dir", str(settings.models_dir))
    table.add_row("HF Token", "✓ set" if settings.hf_token else "– not set")
    table.add_row("Log Level", settings.log_level)

    console.print(table)

    # Engines
    from src.transcriber.engines.factory import list_engines

    console.print(f"  [{DIM}]Engines: {', '.join(list_engines())}[/]")

    # Formats
    from src.transcriber.core.ffmpeg import AudioLoader

    formats = AudioLoader.get_supported_formats()
    console.print(f"  [{DIM}]Audio: {', '.join(sorted(formats['audio']))}[/]")
    console.print(f"  [{DIM}]Video: {', '.join(sorted(formats['video']))}[/]")

    # Presets
    console.print()
    pt = make_table(
        "Speed Presets",
        [
            ("Preset", {}),
            ("Beam", {"justify": "center", "width": 6}),
            ("Batch", {"justify": "center", "width": 6}),
            ("Description", {}),
        ],
    )
    pt.add_row("⚡ fast", "1", "4", "Maximum speed, good quality")
    pt.add_row("⚖️ balanced", "3", "4", "Good balance (default)")
    pt.add_row("🎯 accurate", "5", "1", "Best quality, slower")
    console.print(pt)


if __name__ == "__main__":
    cli()
