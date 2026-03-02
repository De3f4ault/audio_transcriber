"""Transcribe + Subtitle commands."""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

import click

from cli.helpers import PhaseTracker, collect_files, parse_formats, resolve_output
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
@click.argument("files", nargs=-1, type=click.Path(), required=False)
@click.option(
    "-f",
    "--format",
    "output_format",
    default=None,
    help="Output format: txt, srt, vtt, json (or comma-separated: srt,json  or 'all')",
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
@click.option(
    "-R",
    "--recursive",
    is_flag=True,
    help="Recurse into subdirectories when input is a directory",
)
@click.option(
    "--ext",
    "extensions",
    default=None,
    help="Filter by extension (e.g., --ext mp3,m4a). Default: all supported",
)
@click.option(
    "--from-file",
    "from_file",
    default=None,
    type=click.Path(exists=True),
    help="Read input paths from a manifest file (one per line)",
)
@click.option(
    "--exclude",
    default=None,
    help='Exclude files matching glob patterns (e.g., --exclude "*_draft*,temp_*")',
)
@click.option(
    "--collision",
    type=click.Choice(["overwrite", "skip", "rename"]),
    default="overwrite",
    show_default=True,
    help="Strategy when output file already exists",
)
@click.option(
    "--mirror",
    is_flag=True,
    help="Preserve directory structure in output (dir→dir mirror mode)",
)
@click.option(
    "--preset",
    "preset_name",
    default=None,
    help="Load a saved preset (e.g., --preset meeting)",
)
@click.option(
    "--id-only",
    is_flag=True,
    help="Output only transcription IDs (for piping to other commands)",
)
@click.option(
    "--notify",
    is_flag=True,
    help="Send desktop notification when transcription completes",
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
    recursive: bool,
    extensions: str | None,
    from_file: str | None,
    exclude: str | None,
    collision: str,
    mirror: bool,
    preset_name: str | None,
    id_only: bool,
    notify: bool,
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

    \b
    Directory & batch input:
      audiobench transcribe ./audiobooks/                Walk a directory
      audiobench transcribe ./audiobooks/ -R             Recurse into subdirs
      audiobench transcribe ./recordings/ --ext mp3,m4a  Filter by extension
      audiobench transcribe --from-file list.txt         Read paths from file
      find . -name '*.m4a' | audiobench transcribe -     Read from stdin
      audiobench transcribe . --exclude '*_draft*'       Exclude patterns

    \b
    Output control:
      audiobench transcribe dir/ -o out/ --mirror        Preserve dir structure
      audiobench transcribe dir/ --collision skip        Skip existing outputs
      audiobench transcribe dir/ --collision rename      Auto-rename conflicts
      audiobench transcribe file.m4a -f srt,json         Export to both formats
      audiobench transcribe file.m4a -f all              Export to all 4 formats

    \b
    Presets & automation:
      audiobench preset create meeting --model large-v3 --speed accurate
      audiobench transcribe file.m4a --preset meeting    Use saved preset
      audiobench transcribe dir/ --id-only               Output IDs only (piping)
    """
    from src.audiobench.core.pipeline import TranscriptionPipeline

    # E1: Load preset defaults (CLI flags override)
    if preset_name:
        from cli.commands.config_cmd import _load_preset

        preset_data = _load_preset(preset_name)
        if not preset_data:
            console.print(error_panel(f"Preset '{preset_name}' not found"))
            return

        # Apply preset values only where CLI didn't specify
        if not model and "model" in preset_data:
            model = preset_data["model"]
        if speed_preset == "balanced" and "speed" in preset_data:
            speed_preset = preset_data["speed"]
        if not language and "language" in preset_data:
            language = preset_data["language"]
        if not output_format and "format" in preset_data:
            output_format = preset_data["format"]
        if not enhance and preset_data.get("enhance"):
            enhance = True
        if not translate and preset_data.get("translate"):
            translate = True
        if not diarize and preset_data.get("diarize"):
            diarize = True
        if not audio_filter and "filter" in preset_data:
            audio_filter = preset_data["filter"]
        if not initial_prompt and "prompt" in preset_data:
            initial_prompt = preset_data["prompt"]

        if not quiet:
            console.print(f"  [{DIM}]Using preset: {preset_name}[/]")

    # E2: --id-only implies quiet
    if id_only:
        quiet = True

    # ── Resolve input files ──
    if not files and not from_file:
        console.print(
            error_panel(
                "No input", "Provide files, directories, --from-file, or pipe paths via stdin (-)"
            )
        )
        return

    resolved_files = collect_files(
        files,
        recursive=recursive,
        extensions=extensions,
        from_file=from_file,
        exclude=exclude,
    )

    if not resolved_files:
        console.print(
            error_panel("No files found", "No supported audio/video files matched the input.")
        )
        return

    if not quiet and len(resolved_files) != len(files or ()):
        # Show discovery summary when directory/glob expanded the input
        console.print(f"  [{DIM}]Found {len(resolved_files)} file(s) to process[/]")

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

        for file_path in resolved_files:
            input_p = Path(str(file_path))
            info = probe(str(file_path))
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

    # C1: Determine base directory for mirror mode
    input_base_dir: str | None = None
    if mirror and files:
        # Use the first directory argument as the base
        for p in files:
            if Path(p).is_dir():
                input_base_dir = p
                break

    # C3: Parse multi-format string
    multi_formats = parse_formats(output_format)
    # If parse_formats returns formats, use the first as primary
    primary_format = multi_formats[0] if multi_formats else None
    extra_formats = multi_formats[1:] if len(multi_formats) > 1 else []

    pipeline = TranscriptionPipeline()
    results: list[dict] = []

    for file_path in resolved_files:
        input_p = Path(str(file_path))
        file_size = input_p.stat().st_size

        # Resolve output path and format
        resolved_output, resolved_format = resolve_output(
            str(file_path),
            output_path,
            primary_format,
            settings.output_format,
            input_base_dir=input_base_dir,
            collision=collision,
        )

        # C2: collision=skip → resolve_output returns None path
        if resolved_output is None and (output_path or primary_format) and collision == "skip":
            if not quiet:
                console.print(f"  [{DIM}]Skipped (exists): {input_p.name}[/]")
            continue

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
            if extra_formats:
                console.print(f"    Formats: [{DIM}]{', '.join(multi_formats)}[/]")
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
            _file_path: str = str(file_path),
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
                file_path=str(file_path),
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
                # Transcript text was already displayed progressively
                # during transcription via on_segment callbacks.
                # Just show the summary panel below.

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

            # C3: Multi-format — save additional formats
            if extra_formats and transcript:
                from src.audiobench.output.base import get_formatter as get_fmt

                for extra_fmt in extra_formats:
                    extra_out, _ = resolve_output(
                        str(file_path),
                        output_path,
                        extra_fmt,
                        extra_fmt,
                        input_base_dir=input_base_dir,
                        collision=collision,
                    )
                    if extra_out is None:
                        if not quiet:
                            console.print(f"  [{DIM}]Skipped {extra_fmt} (exists)[/]")
                        continue
                    fmt_obj = get_fmt(extra_fmt)
                    content = fmt_obj.format(transcript)
                    Path(extra_out).parent.mkdir(parents=True, exist_ok=True)
                    Path(extra_out).write_text(content, encoding="utf-8")
                    if not quiet:
                        console.print(f"  [{DIM}]Saved → {extra_out}[/]")

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

    # ── Desktop notification ──
    if notify and results:
        _send_notification(results)


def _send_notification(results: list[dict]) -> None:
    """Send a desktop notification on transcription completion."""
    import subprocess

    total_words = sum(r["words"] for r in results)
    total_dur = sum(r["duration"] for r in results)
    n_files = len(results)

    title = "AudioBench — Transcription Complete"
    if n_files == 1:
        body = f"{results[0]['file']}: {total_words:,} words, {format_duration(total_dur)}"
    else:
        body = (
            f"{n_files} files: "
            f"{total_words:,} total words, "
            f"{format_duration(total_dur)} total audio"
        )

    try:
        import sys as _sys

        if _sys.platform == "darwin":
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    f'display notification "{body}" with title "{title}"',
                ],
                check=False,
            )
        elif _sys.platform == "win32":
            pass  # Windows toast not implemented
        else:
            subprocess.run(
                ["notify-send", title, body, "-i", "audio-x-generic"],
                check=False,
            )
    except FileNotFoundError:
        pass  # notify-send not installed — silently skip


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
