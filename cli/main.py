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
    audiobench transcribe meeting.m4a               → stdout (txt)
    audiobench transcribe meeting.m4a -f srt         → meeting.srt (auto-named)
    audiobench transcribe meeting.m4a -o notes.srt   → format from extension
    audiobench transcribe *.m4a -o ./out/            → batch mode
    audiobench transcribe meeting.m4a --fast         → speed preset
    audiobench transcribe meeting.m4a -q             → quiet (for piping)
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

import click
from rich.live import Live
from rich.text import Text

from cli.custom_group import DefaultGroup
from cli.theme import (
    ACCENT,
    APP_NAME,
    APP_VERSION,
    BOLD,
    CHAT_CODE_THEME,
    DIM,
    FORMAT_TO_EXT,
    SUCCESS,
    chat_console,
    console,
    detect_format_from_path,
    error_panel,
    format_duration,
    format_size,
    make_table,
    stdout,
    summary_panel,
)
from src.audiobench.config.logging_config import setup_logging
from src.audiobench.config.settings import get_settings

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


@click.group(cls=DefaultGroup, default_command="transcribe", invoke_without_command=True)
@click.option("-v", "--verbose", is_flag=True, help="Show detailed log output")
@click.option("--debug", is_flag=True, help="Debug logging")
@click.version_option(version=APP_VERSION, prog_name="audiobench")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, debug: bool) -> None:
    """AudioBench — offline audio workbench.

    \b
    Transcribe files:
      audiobench transcribe meeting.m4a                 Print to stdout
      audiobench transcribe meeting.m4a -f srt          Save as meeting.srt
      audiobench transcribe meeting.m4a -o notes.srt    Auto-detect SRT format
      audiobench transcribe *.m4a -o ./out/             Batch to directory
      audiobench transcribe meeting.m4a --fast          Fast preset
      audiobench transcribe meeting.m4a -q | grep word  Pipe-friendly

    \b
    Manage:
      audiobench history                                Past transcriptions
      audiobench search "keyword"                       Search text
      audiobench export 3 -f vtt                        Re-export as VTT
      audiobench download large-v3-turbo                Pre-download model
      audiobench delete 3                               Remove from history
      audiobench info                                   System info
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
@click.option("--fast", "speed_preset", flag_value="fast", help=" Fast: beam=1, batch=4")
@click.option(
    "--balanced",
    "speed_preset",
    flag_value="balanced",
    default=True,
    help=" Balanced: beam=3, batch=4 (default)",
)
@click.option(
    "--accurate", "speed_preset", flag_value="accurate", help="Accurate: beam=5, sequential"
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
@click.option(
    "--translate",
    is_flag=True,
    help="Translate speech to English (any language → English)",
)
@click.option(
    "--diarize",
    is_flag=True,
    help="Identify speakers (requires pyannote.audio + HF token)",
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
    translate: bool,
    diarize: bool,
) -> None:
    """Transcribe audio/video files.

    \b
    Examples:
      audiobench transcribe meeting.m4a                  Print to stdout
      audiobench transcribe meeting.m4a -f srt           Auto-save meeting.srt
      audiobench transcribe meeting.m4a -o notes.srt     Format from extension
      audiobench transcribe *.m4a -o ./out/              Batch to directory
      audiobench transcribe --fast lecture.mp3            Fast preset
      audiobench transcribe --translate audio_sw.m4a      Translate to English
      audiobench transcribe --diarize meeting.m4a         Identify speakers
      audiobench transcribe -q meeting.m4a | grep word   Pipe-friendly
    """
    from src.audiobench.core.pipeline import TranscriptionPipeline

    settings = get_settings()
    if model:
        settings.model_name = model

    # Build filter list
    filters: list[str] | None = None
    if enhance:
        from src.audiobench.core.ffmpeg import ENHANCE_FILTERS

        filters = list(ENHANCE_FILTERS)
    if audio_filter:
        filters = audio_filter.split(",") if not filters else filters + audio_filter.split(",")

    # --check: show metadata only, no transcription
    if check:
        from src.audiobench.core.ffmpeg import probe

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
            if translate:
                console.print("    Task:    [bold]Translate → English[/]")
            if diarize:
                console.print("    Diarize: [bold]Speaker identification[/]")
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
                translate=translate,
                enable_diarization=diarize,
                on_phase=tracker.update,
                on_segment=tracker.on_segment,
                filters=filters,
            )

            tracker.finalize()

            elapsed = time.perf_counter() - start_time
            speed_ratio = transcript.duration_seconds / elapsed if elapsed > 0 else 0

            # ── Output ──
            if quiet:
                from src.audiobench.output.base import get_formatter

                formatter = get_formatter(resolved_format)
                stdout.print(formatter.format(transcript), highlight=False)
            else:
                # Print transcript to terminal if no file output
                if not resolved_output:
                    from src.audiobench.output.base import get_formatter

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


# ── Subtitle Command ────────────────────────────────────────


@cli.command()
@click.argument("video", type=click.Path(exists=True))
@click.option("-o", "--output", "output_path", default=None, help="Output video path")
@click.option(
    "--hard",
    "hard_burn",
    is_flag=True,
    help="Burn subtitles into video pixels (permanent)",
)
@click.option("-l", "--language", default=None, help="Language code (e.g., en, sw)")
@click.option("--translate", is_flag=True, help="Translate subtitles to English")
@click.option("-q", "--quiet", is_flag=True, help="Quiet mode")
def subtitle(
    video: str,
    output_path: str | None,
    hard_burn: bool,
    language: str | None,
    translate: bool,
    quiet: bool,
) -> None:
    """Transcribe a video and embed subtitles into it.

    \b
    Examples:
      audiobench subtitle lecture.mp4                    Soft-embed subtitles
      audiobench subtitle lecture.mp4 --hard             Burn into video pixels
      audiobench subtitle lecture.mp4 -o subtitled.mp4   Custom output path
      audiobench subtitle lecture.mp4 --translate        Subtitles in English
    """
    import tempfile

    from src.audiobench.core.ffmpeg import SUPPORTED_VIDEO_FORMATS, embed_subtitles
    from src.audiobench.core.pipeline import TranscriptionPipeline
    from src.audiobench.output.srt import SrtFormatter

    video_path = Path(video)
    ext = video_path.suffix.lstrip(".").lower()

    if ext not in SUPPORTED_VIDEO_FORMATS:
        console.print(
            error_panel(
                "Unsupported format",
                f".{ext} is not a supported video format. "
                f"Supported: {', '.join(sorted(SUPPORTED_VIDEO_FORMATS))}",
            )
        )
        return

    # Resolve output path
    out = Path(output_path) if output_path else video_path.with_stem(f"{video_path.stem}_subtitled")

    if not quiet:
        console.print()
        console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — Subtitle Embedding")
        console.print(f"  [{DIM}]{'─' * 44}[/]")
        console.print(f"    Video:   [{ACCENT}]{video_path.name}[/]")
        console.print(f"    Output:  [{DIM}]{out.name}[/]")
        mode_desc = "Hard burn (permanent)" if hard_burn else "Soft embed (selectable track)"
        console.print(f"    Mode:    {mode_desc}")
        if translate:
            console.print("    Task:    [bold]Translate → English[/]")
        console.print(f"  [{DIM}]{'─' * 44}[/]")

    start_time = time.perf_counter()

    try:
        # Step 1: Transcribe the video's audio track
        if not quiet:
            console.print(f"  [{DIM}]Transcribing audio track...[/]")

        pipeline = TranscriptionPipeline()
        transcript = pipeline.transcribe_file(
            file_path=video,
            language=language,
            output_format="srt",
            word_timestamps=True,
            translate=translate,
        )

        # Step 2: Generate temporary SRT file
        formatter = SrtFormatter()
        srt_content = formatter.format(transcript)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".srt", delete=False, prefix="audiobench_sub_"
        ) as tmp:
            tmp.write(srt_content)
            srt_path = tmp.name

        if not quiet:
            console.print(f"  [{DIM}]Generated {transcript.segment_count} subtitle segments[/]")
            console.print(f"  [{DIM}]Embedding subtitles...[/]")

        # Step 3: Embed subtitles into video
        embed_subtitles(video_path, srt_path, out, hard_burn=hard_burn)

        # Cleanup temp SRT
        import contextlib

        with contextlib.suppress(OSError):
            Path(srt_path).unlink()

        elapsed = time.perf_counter() - start_time

        if not quiet:
            out_size = out.stat().st_size
            console.print()
            console.print(f"  [{SUCCESS}]✓ Subtitles embedded successfully[/]")
            console.print(f"    Output:   [{ACCENT}]{out}[/] ({format_size(out_size)})")
            console.print(f"    Segments: {transcript.segment_count}")
            console.print(f"    Elapsed:  {format_duration(elapsed)}")
            console.print()

    except Exception as e:
        console.print(error_panel("Subtitle embedding failed", str(e)))


# ── Listen (Live STT) Command ───────────────────────────────


@cli.command()
@click.option(
    "-l", "--language", default="en", show_default=True, help="Language code (e.g., en, sw)"
)
@click.option("--translate", is_flag=True, help="Translate speech to English in real-time")
@click.option(
    "--save",
    "save_path",
    default=None,
    help="Also save transcript to a text file",
)
@click.option(
    "--model",
    "live_model",
    default="base",
    show_default=True,
    help="Whisper model for live mode (smaller = faster)",
)
@click.option("-q", "--quiet", is_flag=True, help="Quiet mode (raw text output, for piping)")
def listen(
    language: str | None,
    translate: bool,
    save_path: str | None,
    live_model: str,
    quiet: bool,
) -> None:
    """Live transcription from microphone.

    \b
    Opens your microphone and transcribes speech in real-time.
    Press Ctrl+C to stop. The transcript is saved to history.

    \b
    Models (speed on CPU, 4 threads):
      tiny    — Fastest, low accuracy
      base    — Real-time + good accuracy (default)
      small   — Slower, high accuracy

    \b
    Examples:
      audiobench listen                         Live transcribe (base)
      audiobench listen --model tiny            Fastest, lower accuracy
      audiobench listen --language en            Force English
      audiobench listen --translate              Translate to English
      audiobench listen --save meeting.txt       Also save to file
      audiobench listen -q >> notes.txt          Pipe mode
    """
    from src.audiobench.streaming.display import LiveDisplay
    from src.audiobench.streaming.session import LiveSession

    settings = get_settings()

    # Override model for live mode (smaller = faster)
    settings.model_name = live_model
    # Force beam_size=1 for speed in live mode
    settings.beam_size = 1

    display = LiveDisplay(quiet=quiet)

    if not quiet:
        console.print()
        console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — Live Transcription")
        console.print(f"  [{DIM}]{'─' * 44}[/]")
        console.print(f"    Model:   {settings.model_name}")
        if language:
            console.print(f"    Language: {language}")
        if translate:
            console.print("    Task:    [bold]Translate → English[/]")
        if save_path:
            console.print(f"    Save to: [{DIM}]{save_path}[/]")
        console.print(f"  [{DIM}]{'─' * 44}[/]")
        console.print()

    # Create live session with display callbacks
    session = LiveSession(
        settings=settings,
        on_text=display.append_text,
        on_recording_start=display.set_recording,
        on_recording_stop=display.set_processing,
        translate=translate,
        language=language,
    )

    import time as _time

    display.start()
    display.set_listening()

    transcript = None
    try:
        transcript = session.run()
    except (KeyboardInterrupt, SystemExit):
        pass  # Graceful — session.run() already builds transcript
    finally:
        display.stop()

    # If interrupted, get transcript from the session's internal state
    if transcript is None:
        elapsed = _time.perf_counter() - (session._start_time or _time.perf_counter())
        transcript = session._build_transcript(max(0.0, elapsed))

    # Post-session output
    if not quiet:
        console.print()
        console.print(f"  [{SUCCESS}]✓ Session complete[/]")
        console.print(f"    Segments: {transcript.segment_count}")
        console.print(f"    Words:    {transcript.word_count}")
        console.print(f"    Duration: {format_duration(transcript.duration_seconds)}")

    # Auto-save transcript to file
    if transcript.text.strip():
        if not save_path:
            sessions_dir = Path.home() / ".audiobench" / "sessions"
            sessions_dir.mkdir(parents=True, exist_ok=True)
            ts = _time.strftime("%Y%m%d_%H%M%S")
            save_path = str(sessions_dir / f"live_{ts}.txt")

        Path(save_path).write_text(transcript.text, encoding="utf-8")
        if not quiet:
            console.print(f"    Saved:    [{ACCENT}]{save_path}[/]")

    # Save to database
    if transcript.segments:
        try:
            from src.audiobench.storage.database import init_db
            from src.audiobench.storage.repository import TranscriptionRepository

            init_db()
            repo = TranscriptionRepository()
            repo.save_live_session(transcript)
            if not quiet:
                console.print(f"    [{DIM}]Saved to history[/]")
        except Exception:
            pass  # Don't fail on DB errors for live sessions

    if not quiet:
        console.print()


# ── Speak (TTS) Command ─────────────────────────────────────


@cli.command()
@click.argument("text_or_file", required=False, default=None)
@click.option(
    "--id", "transcript_id", type=int, default=None, help="Speak transcript # from history"
)
@click.option(
    "--voice",
    default=None,
    help="Piper voice name (default: from settings)",
)
@click.option(
    "-o", "--output", "output_path", default=None, help="Save to WAV file instead of playing"
)
@click.option("-q", "--quiet", is_flag=True, help="Quiet mode")
def speak(
    text_or_file: str | None,
    transcript_id: int | None,
    voice: str | None,
    output_path: str | None,
    quiet: bool,
) -> None:
    """Speak text aloud using Piper TTS.

    \b
    Examples:
      audiobench speak "Hello world"                 Speak text directly
      audiobench speak notes.txt                     Speak a text file
      audiobench speak --id 3                        Speak transcript from history
      audiobench speak "Hello" -o greeting.wav       Save to file
      audiobench speak --voice en_US-lessac-medium "Test"
    """
    from src.audiobench.tts.engine import PiperTTSEngine, TTSError

    settings = get_settings()
    voice_name = voice or settings.tts_voice

    # Determine text to speak
    if transcript_id is not None:
        # Speak from history
        from src.audiobench.storage.database import init_db
        from src.audiobench.storage.repository import TranscriptionRepository

        init_db()
        repo = TranscriptionRepository()
        record = repo.get_by_id(transcript_id)
        if not record:
            console.print(error_panel("Not found", f"Transcript #{transcript_id} not found"))
            return
        text = record["full_text"]
        if not quiet:
            fname = record.get("file_name", "unknown")
            console.print(f"  [{DIM}]Speaking transcript #{transcript_id}: {fname}[/]")

    elif text_or_file is not None:
        # Check if it's a file path (short strings only — long text can't be paths)
        maybe_file = Path(text_or_file) if len(text_or_file) < 256 else None
        try:
            is_file = maybe_file and maybe_file.exists() and maybe_file.is_file()
        except OSError:
            is_file = False
        if is_file:
            text = maybe_file.read_text(encoding="utf-8")
            if not quiet:
                console.print(f"  [{DIM}]Speaking file: {maybe_file.name}[/]")
        else:
            text = text_or_file
    else:
        # Read from stdin
        text = sys.stdin.read()
        if not text.strip():
            console.print(error_panel("No input", "Provide text, a file, --id, or pipe to stdin"))
            return

    if not quiet:
        console.print(f"  [{ACCENT}]Voice: {voice_name}[/]")
        preview = text[:80] + "..." if len(text) > 80 else text
        console.print(f"  [{DIM}]{preview}[/]")

    try:
        engine = PiperTTSEngine(voices_dir=settings.voices_dir)

        if output_path:
            result = engine.synthesize(text, voice=voice_name, output_path=output_path)
            if not quiet:
                console.print(f"  [{SUCCESS}]✓ Saved to: {result}[/]")
        else:
            engine.play(text, voice=voice_name)
            if not quiet:
                console.print(f"  [{SUCCESS}]✓ Playback complete[/]")

    except TTSError as e:
        console.print(error_panel("TTS Error", str(e)))


# ── Download Voice Command ──────────────────────────────────


@cli.command("download-voice")
@click.argument("voice_name")
def download_voice(voice_name: str) -> None:
    """Download a Piper TTS voice model.

    \b
    Examples:
      audiobench download-voice en_US-amy-medium
      audiobench download-voice en_US-lessac-high
      audiobench download-voice de_DE-thorsten-medium
    """
    from src.audiobench.tts.engine import PiperTTSEngine, TTSError

    settings = get_settings()

    console.print(f"  [{ACCENT}]Downloading voice: {voice_name}[/]")

    try:
        engine = PiperTTSEngine(voices_dir=settings.voices_dir)
        model_path = engine.download_voice(voice_name)
        console.print(f"  [{SUCCESS}]✓ Voice downloaded to: {model_path}[/]")
    except TTSError as e:
        console.print(error_panel("Download failed", str(e)))


# ── Summarize (AI) Command ──────────────────────────────────


@cli.command()
@click.argument("transcript_id", type=int)
@click.option("--model", default=None, help="Ollama model (default: from settings)")
@click.option(
    "--prompt",
    "custom_prompt",
    default=None,
    help="Custom instruction (e.g., 'Focus on action items')",
)
def summarize(transcript_id: int, model: str | None, custom_prompt: str | None) -> None:
    """Summarize a transcript using local AI (Ollama).

    \b
    Examples:
      audiobench summarize 3                         Summarize transcript #3
      audiobench summarize 3 --model deepseek-v3.2   Use a specific model
      audiobench summarize 3 --prompt "Focus on action items"
    """
    from src.audiobench.ai.ollama import AIError, OllamaClient
    from src.audiobench.ai.prompts import (
        TRANSCRIPT_SYSTEM,
        action_items,
    )
    from src.audiobench.ai.prompts import (
        summarize as summarize_prompt,
    )
    from src.audiobench.storage.database import init_db
    from src.audiobench.storage.repository import TranscriptionRepository

    settings = get_settings()
    model_name = model or settings.ollama_model

    # Fetch transcript
    init_db()
    repo = TranscriptionRepository()
    record = repo.get_by_id(transcript_id)
    if not record:
        console.print(error_panel("Not found", f"Transcript #{transcript_id} not found"))
        return

    console.print()
    console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — AI Summary")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print(f"    Source:  [{ACCENT}]#{transcript_id} {record['file_name']}[/]")
    console.print(f"    Model:   {model_name}")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print()

    # Build prompt
    if custom_prompt and "action" in custom_prompt.lower():
        prompt = action_items(record["full_text"])
    elif custom_prompt:
        prompt = f"{custom_prompt}\n\nTRANSCRIPT:\n{record['full_text']}"
    else:
        prompt = summarize_prompt(record["full_text"])

    # Stream response
    try:
        client = OllamaClient(
            base_url=settings.ollama_base_url,
            model=model_name,
        )

        if not client.is_available():
            console.print(
                error_panel(
                    "Ollama not running",
                    f"Start with: ollama serve\nThen pull the model: ollama pull {model_name}",
                )
            )
            return

        console.print(f"  [{DIM}]Generating...[/]")
        console.print()

        for token in client.stream(prompt, system_prompt=TRANSCRIPT_SYSTEM):
            console.print(token, end="")

        console.print()
        console.print()
        console.print(f"  [{SUCCESS}]✓ Summary complete[/]")

    except AIError as e:
        console.print(error_panel("AI Error", str(e)))


# ── Ask (AI Q&A) Command ────────────────────────────────────


@cli.command()
@click.argument("transcript_id", type=int)
@click.argument("question")
@click.option("--model", default=None, help="Ollama model (default: from settings)")
def ask(transcript_id: int, question: str, model: str | None) -> None:
    """Ask a question about a transcript using AI.

    \b
    Examples:
      audiobench ask 3 "What decisions were made?"
      audiobench ask 3 "Who is responsible for the API?"
      audiobench ask 3 "List all mentioned dates" --model deepseek-v3.2
    """
    from src.audiobench.ai.ollama import AIError, OllamaClient
    from src.audiobench.ai.prompts import TRANSCRIPT_SYSTEM, qa
    from src.audiobench.storage.database import init_db
    from src.audiobench.storage.repository import TranscriptionRepository

    settings = get_settings()
    model_name = model or settings.ollama_model

    # Fetch transcript
    init_db()
    repo = TranscriptionRepository()
    record = repo.get_by_id(transcript_id)
    if not record:
        console.print(error_panel("Not found", f"Transcript #{transcript_id} not found"))
        return

    console.print()
    console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — AI Q&A")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print(f"    Source:   [{ACCENT}]#{transcript_id} {record['file_name']}[/]")
    console.print(f"    Question: {question}")
    console.print(f"    Model:    {model_name}")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print()

    prompt = qa(record["full_text"], question)

    try:
        client = OllamaClient(
            base_url=settings.ollama_base_url,
            model=model_name,
        )

        if not client.is_available():
            console.print(
                error_panel(
                    "Ollama not running",
                    "Start with: ollama serve",
                )
            )
            return

        for token in client.stream(prompt, system_prompt=TRANSCRIPT_SYSTEM):
            console.print(token, end="")

        console.print()
        console.print()

    except AIError as e:
        console.print(error_panel("AI Error", str(e)))


# ── Chat (AI Interactive) Command ───────────────────────────


CHAT_HELP_TEXT = (
    "  [bold]Slash Commands[/]\n"
    "  ─────────────────────────────────────\n"
    "  /help              Show this help\n"
    "  /context           Show loaded transcripts\n"
    "  /load <ID>         Add a transcript to context\n"
    "  /clear             Clear conversation history\n"
    "  /model <name>      Switch model mid-chat\n"
    "  /think             Toggle thinking display\n"
    "  /retry             Regenerate last response\n"
    "  /export [file]     Export chat to markdown\n"
    "  /history           List past chat sessions\n"
    "  /save              Force-save conversation\n"
    "  /exit              Exit chat (also Ctrl+D)\n"
    "\n"
    "  [bold]Multi-line Input[/]\n"
    "  ─────────────────────────────────────\n"
    '  Type [bold]triple-quotes (\\"\\"\\")'
    "[/] to start/end a multi-line block.\n"
)


def _handle_slash_command(
    cmd: str,
    session,
    tx_repo,
    chat_repo,
    settings,
) -> bool:
    """Handle a slash command. Returns True if the REPL should exit."""
    parts = cmd.strip().split(None, 1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command in ("/exit", "/quit", "/q"):
        return True

    elif command == "/help":
        console.print()
        console.print(CHAT_HELP_TEXT)

    elif command == "/context":
        console.print()
        for line in session.get_context_summary():
            console.print(f"    {line}")
        console.print()

    elif command == "/load":
        if not arg or not arg.strip().isdigit():
            console.print(f"  [{DIM}]Usage: /load <transcript_id>[/]")
            return False
        tid = int(arg.strip())
        record = tx_repo.get_by_id(tid)
        if not record:
            console.print(f"  [{DIM}]Transcript #{tid} not found[/]")
            return False
        session.load_transcripts([record])
        console.print(
            f"  [{SUCCESS}]✓ Loaded #{tid} "
            f"{record['file_name']} "
            f"({record['word_count']:,} words)[/]"
        )

    elif command == "/clear":
        session.clear_history()
        console.print(
            f"  [{SUCCESS}]✓ Conversation cleared (new session #{session.conversation_id})[/]"
        )

    elif command == "/model":
        if not arg:
            console.print(f"  [{DIM}]Current model: {session.model}[/]")
            console.print(f"  [{DIM}]Usage: /model <name>[/]")
            return False
        session.switch_model(arg.strip())
        console.print(f"  [{SUCCESS}]✓ Switched to {arg.strip()}[/]")

    elif command == "/think":
        session.show_thinking = not session.show_thinking
        state = "on" if session.show_thinking else "off"
        console.print(f"  [{SUCCESS}]✓ Thinking display: {state}[/]")

    elif command == "/history":
        convs = chat_repo.list_conversations(limit=10)
        if not convs:
            console.print(f"  [{DIM}]No past conversations[/]")
            return False
        console.print()
        for c in convs:
            tid_list = c.get("transcript_ids", [])
            ctx = f" (transcripts: {tid_list})" if tid_list else ""
            console.print(
                f"    [{ACCENT}]#{c['id']}[/] "
                f"{c['title']} "
                f"[{DIM}]({c['message_count']} msgs, "
                f"{c['model']}){ctx}[/]"
            )
        console.print()

    elif command == "/save":
        console.print(f"  [{SUCCESS}]✓ Conversation #{session.conversation_id} saved[/]")

    elif command == "/export":
        import time as _time
        from pathlib import Path

        if not session.messages:
            console.print(f"  [{DIM}]Nothing to export yet[/]")
            return False
        fname = arg.strip() if arg.strip() else None
        if not fname:
            slug = f"chat_{session.conversation_id or 'new'}_{int(_time.time())}"
            fname = f"{slug}.md"
        path = Path(fname).expanduser()
        lines = [f"# Chat #{session.conversation_id or 'new'}\n"]
        lines.append(f"Model: {session.model}  \n")
        lines.append("---\n")
        for msg in session.messages:
            if msg["role"] == "user":
                lines.append(f"**You:** {msg['content']}\n")
            elif msg["role"] == "assistant":
                lines.append(f"**AI:**\n\n{msg['content']}\n")
            lines.append("---\n")
        path.write_text("\n".join(lines), encoding="utf-8")
        console.print(f"  [{SUCCESS}]✓ Exported to {path}[/]")

    elif command == "/retry":
        # Signal to the REPL that we want a retry
        # We store a flag on the session object
        session._retry_requested = True  # noqa: SLF001
        return False  # handled in the REPL loop

    else:
        console.print(f"  [{DIM}]Unknown command: {command} (type /help for commands)[/]")

    return False


@cli.command()
@click.argument("transcript_ids", nargs=-1, type=int)
@click.option(
    "--model",
    default=None,
    help="Ollama model (default: from settings)",
)
@click.option(
    "--temperature",
    default=0.3,
    type=float,
    help="Creativity level (0.0-1.0)",
)
@click.option(
    "--search",
    "search_query",
    default=None,
    help="Load transcripts matching this search",
)
@click.option(
    "--recent",
    default=None,
    type=int,
    help="Load N most recent transcripts as context",
)
@click.option(
    "--resume",
    "resume_id",
    default=None,
    type=int,
    help="Resume a previous conversation by ID",
)
@click.option(
    "--list",
    "list_chats",
    is_flag=True,
    help="List past chat conversations",
)
@click.option(
    "--delete",
    "delete_id",
    default=None,
    type=int,
    help="Delete a chat conversation by ID",
)
@click.option(
    "--think/--no-think",
    default=True,
    help="Show/hide model chain-of-thought",
)
def chat(
    transcript_ids: tuple[int, ...],
    model: str | None,
    temperature: float,
    search_query: str | None,
    recent: int | None,
    resume_id: int | None,
    list_chats: bool,
    delete_id: int | None,
    think: bool,
) -> None:
    """Interactive AI chat with transcript context.

    \b
    Examples:
      audiobench chat                           Chat freely
      audiobench chat 3                         Chat about transcript #3
      audiobench chat 3 5 7                     Chat with multiple transcripts
      audiobench chat --search "meeting"        Load matching transcripts
      audiobench chat --recent 5                Load 5 most recent
      audiobench chat --resume 2                Resume conversation #2
      audiobench chat --list                    List past conversations
      audiobench chat --delete 2                Delete conversation #2
      audiobench chat --model deepseek-v3.1:671b-cloud
    """
    from src.audiobench.ai.chat import ChatSession
    from src.audiobench.ai.ollama import AIError, OllamaClient
    from src.audiobench.storage.chat_repository import ChatRepository
    from src.audiobench.storage.database import init_db
    from src.audiobench.storage.repository import TranscriptionRepository

    settings = get_settings()
    model_name = model or settings.ollama_model
    init_db()

    chat_repo = ChatRepository()
    tx_repo = TranscriptionRepository()

    # ── Handle --list ──
    if list_chats:
        convs = chat_repo.list_conversations(limit=20)
        if not convs:
            console.print(f"  [{DIM}]No chat conversations yet[/]")
            return
        console.print()
        console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — Chat History")
        console.print(f"  [{DIM}]{'─' * 44}[/]")
        for c in convs:
            tid_list = c.get("transcript_ids", [])
            ctx = f" ctx:{tid_list}" if tid_list else ""
            console.print(
                f"    [{ACCENT}]#{c['id']}[/] "
                f"{c['title']} "
                f"[{DIM}]({c['message_count']} msgs"
                f"{ctx})[/]"
            )
        console.print()
        console.print(f"  [{DIM}]Resume with: audiobench chat --resume <ID>[/]")
        console.print()
        return

    # ── Handle --delete ──
    if delete_id is not None:
        if chat_repo.delete_conversation(delete_id):
            console.print(f"  [{SUCCESS}]✓ Deleted conversation #{delete_id}[/]")
        else:
            console.print(
                error_panel(
                    "Not found",
                    f"Conversation #{delete_id} not found",
                )
            )
        return

    # ── Check Ollama ──
    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=model_name,
    )
    if not client.is_available():
        console.print(
            error_panel(
                "Ollama not running",
                f"Start with: ollama serve\nPull model: ollama pull {model_name}",
            )
        )
        return

    # ── Create or resume session ──
    session = ChatSession(
        client=client,
        chat_repo=chat_repo,
        model=model_name,
        temperature=temperature,
        conversation_id=resume_id,
        show_thinking=think,
    )

    # Resume existing conversation
    if resume_id is not None and not session.restore_from_db():
        console.print(
            error_panel(
                "Not found",
                f"Conversation #{resume_id} not found",
            )
        )
        return

    # ── Load transcript context ──
    transcripts_to_load = []

    # By explicit IDs
    for tid in transcript_ids:
        record = tx_repo.get_by_id(tid)
        if record:
            transcripts_to_load.append(record)
        else:
            console.print(f"  [{DIM}]Transcript #{tid} not found, skipping[/]")

    # By search
    if search_query:
        results = tx_repo.search(search_query, limit=5)
        for r in results:
            full = tx_repo.get_by_id(r["id"])
            if full:
                transcripts_to_load.append(full)
        if not results:
            console.print(f"  [{DIM}]No transcripts matching '{search_query}'[/]")

    # By recent
    if recent:
        history_items = tx_repo.get_history(limit=recent)
        for h in history_items:
            full = tx_repo.get_by_id(h["id"])
            if full:
                transcripts_to_load.append(full)

    if transcripts_to_load:
        session.load_transcripts(transcripts_to_load)

    # ── Header ──
    console.print()
    conv_label = f" [#{resume_id}]" if resume_id else ""
    console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — AI Chat{conv_label}")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print(f"    Model:    {model_name}")
    ctx_lines = session.get_context_summary()
    console.print(f"    Context:  {ctx_lines[0]}")
    for line in ctx_lines[1:]:
        console.print(f"              {line}")
    think_label = "on" if think else "off"
    console.print(f"    Thinking: {think_label}")
    if resume_id and session.turn_count > 0:
        console.print(f"    Resumed:  {session.turn_count} previous turn(s)")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print()

    # ── Render past messages on resume ──
    import contextlib
    import readline as _readline
    import time as _time
    from pathlib import Path as _Path

    from rich.console import Group
    from rich.markdown import Markdown as RichMarkdown
    from rich.padding import Padding

    # ── Readline history setup ──
    _history_file = _Path.home() / ".cache" / "audiobench_chat_history"
    _history_file.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(FileNotFoundError):
        _readline.read_history_file(str(_history_file))
    _readline.set_history_length(500)

    def _save_readline_history() -> None:
        with contextlib.suppress(OSError):
            _readline.write_history_file(str(_history_file))

    # ── Render past messages on resume ──
    if resume_id and session.messages:
        console.print(f"  [{DIM}]─── Previous Messages ───[/]")
        console.print()
        for msg in session.messages:
            if msg["role"] == "user":
                console.print(f"  [{ACCENT}]>>> {msg['content']}[/]")
                console.print()
            elif msg["role"] == "assistant":
                md = RichMarkdown(
                    msg["content"],
                    code_theme=CHAT_CODE_THEME,
                )
                chat_console.print(Padding(md, (0, 2, 1, 2)))
        console.print(f"  [{DIM}]─── End of History ───[/]")
        console.print()

    # ── Helper: stream a message and render ──
    def _stream_and_render(user_text: str) -> None:
        """Send user input and render the streamed response."""
        console.print()
        try:
            thinking_parts: list[str] = []
            content_parts: list[str] = []
            token_count = 0
            t_start = _time.monotonic()

            with Live(
                console=chat_console,
                refresh_per_second=8,
                vertical_overflow="visible",
            ) as live:
                for chunk in session.send(user_text):
                    thinking = chunk.get("thinking", "")
                    content = chunk.get("content", "")

                    if thinking and session.show_thinking:
                        thinking_parts.append(thinking)

                    if content:
                        content_parts.append(content)
                        token_count += 1

                    # Build display
                    display_parts = []

                    if thinking_parts and session.show_thinking:
                        think_text = "".join(thinking_parts)
                        display_parts.append(
                            Text(f"[thinking] {think_text}\n", style="dim italic"),
                        )

                    if content_parts:
                        md_text = "".join(content_parts)
                        display_parts.append(
                            RichMarkdown(
                                md_text,
                                code_theme=CHAT_CODE_THEME,
                            )
                        )

                    if display_parts:
                        live.update(Group(*display_parts))

            # Token stats
            elapsed = _time.monotonic() - t_start
            if token_count > 0 and elapsed > 0:
                tps = token_count / elapsed
                console.print(
                    f"  [{DIM}]{token_count} tokens · {tps:.1f} tok/s · {elapsed:.1f}s[/]"
                )
            console.print()

        except KeyboardInterrupt:
            console.print()
            console.print(f"  [{DIM}]Generation interrupted[/]")
            console.print()

        except AIError as e:
            console.print(error_panel("AI Error", str(e)))
            console.print()

    # ── Multi-line input helper ──
    def _read_multiline() -> str:
        """Read lines until closing triple-quotes."""
        lines: list[str] = []
        console.print(f'  [{DIM}]Multi-line mode (type """ to end):[/]')
        while True:
            try:
                line = input("... ")
            except (EOFError, KeyboardInterrupt):
                break
            if line.strip() == '"""':
                break
            lines.append(line)
        return "\n".join(lines)

    # ── Interactive REPL ──
    last_user_input: str | None = None
    session._retry_requested = False  # noqa: SLF001

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            _save_readline_history()
            if session.conversation_id:
                console.print(
                    f"  [{SUCCESS}]✓ Conversation "
                    f"#{session.conversation_id} saved "
                    f"({session.turn_count * 2} messages)[/]"
                )
            console.print(f"  [{DIM}]Goodbye![/]")
            console.print()
            break

        if not user_input:
            continue

        # Multi-line input
        if user_input == '"""':
            user_input = _read_multiline()
            if not user_input.strip():
                continue

        # Slash commands (accept both / and \)
        if user_input.startswith("\\"):
            user_input = "/" + user_input[1:]
        if user_input.startswith("/"):
            should_exit = _handle_slash_command(
                user_input,
                session,
                tx_repo,
                chat_repo,
                settings,
            )

            # Handle /retry
            if getattr(session, "_retry_requested", False):
                session._retry_requested = False  # noqa: SLF001
                if last_user_input and session.messages:
                    # Remove last assistant + user message
                    session._messages = [m for m in session._messages if m != session._messages[-1]]
                    if session._messages and session._messages[-1]["role"] == "user":
                        session._messages.pop()
                    console.print(f"  [{DIM}]Regenerating...[/]")
                    _stream_and_render(last_user_input)
                else:
                    console.print(f"  [{DIM}]Nothing to retry[/]")
                continue

            if should_exit:
                _save_readline_history()
                if session.conversation_id:
                    console.print(
                        f"  [{SUCCESS}]✓ Conversation "
                        f"#{session.conversation_id} saved "
                        f"({session.turn_count * 2} messages)"
                        f"[/]"
                    )
                console.print(f"  [{DIM}]Goodbye![/]")
                console.print()
                break
            continue

        last_user_input = user_input
        _stream_and_render(user_input)


# ── History Command ──────────────────────────────────────────


@cli.command()
@click.option("--limit", default=20, help="Number of records to show")
def history(limit: int) -> None:
    """View transcription history."""
    from src.audiobench.storage.database import init_db
    from src.audiobench.storage.repository import TranscriptionRepository

    init_db()
    repo = TranscriptionRepository()
    records = repo.get_history(limit=limit)

    if not records:
        console.print(f"  [{DIM}]No transcription history yet.[/]")
        return

    table = make_table(
        "Transcription History",
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
    from src.audiobench.storage.database import init_db
    from src.audiobench.storage.repository import TranscriptionRepository

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
    from src.audiobench.core.models import Segment, Transcript
    from src.audiobench.output.base import get_formatter
    from src.audiobench.storage.database import init_db
    from src.audiobench.storage.repository import TranscriptionRepository

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
    from src.audiobench.storage.database import init_db
    from src.audiobench.storage.repository import TranscriptionRepository

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
        f"{APP_NAME} v{APP_VERSION}",
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
    from src.audiobench.engines.factory import list_engines

    console.print(f"  [{DIM}]Engines: {', '.join(list_engines())}[/]")

    # Formats
    from src.audiobench.core.ffmpeg import AudioLoader

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
    pt.add_row("fast", "1", "4", "Maximum speed, good quality")
    pt.add_row("balanced", "3", "4", "Good balance (default)")
    pt.add_row("accurate", "5", "1", "Best quality, slower")
    console.print(pt)


if __name__ == "__main__":
    cli()
