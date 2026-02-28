"""Transcribe + Subtitle commands."""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

import click

from cli.helpers import PhaseTracker, resolve_output
from cli.theme import (
    ACCENT,
    APP_NAME,
    BOLD,
    DIM,
    SUCCESS,
    console,
    error_panel,
    format_duration,
    format_size,
    make_table,
    stdout,
    summary_panel,
)
from src.audiobench.config.settings import get_settings


# ── Transcribe Command ──────────────────────────────────────


@click.command()
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


@click.command()
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
