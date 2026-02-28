"""Listen (Live STT) command."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from cli.theme import (
    ACCENT,
    DIM,
    SUCCESS,
    console,
    error_panel,
    format_duration,
)
from src.audiobench.config.settings import get_settings


@click.command()
@click.option(
    "-l", "--language", default="en", show_default=True, help="Language code (e.g., en, sw)"
)
@click.option("--translate", is_flag=True, help="Translate to English in real-time")
@click.option("--save", "save_path", default=None, help="Save transcript to file")
@click.option("--model", "model_size", default=None, help="Whisper model for live (default: base)")
@click.option("-q", "--quiet", is_flag=True, help="Quiet mode")
def listen(
    language: str,
    translate: bool,
    save_path: str | None,
    model_size: str | None,
    quiet: bool,
) -> None:
    """Live transcription from microphone.

    \b
    Examples:
      audiobench listen                          Live transcribe (English)
      audiobench listen -l sw                    Live transcribe (Swahili)
      audiobench listen --translate              Translate to English
      audiobench listen --save session.txt       Save to file
      audiobench listen --model small            Use 'small' model
    """
    try:
        from src.audiobench.streaming.display import LiveTranscriptDisplay
        from src.audiobench.streaming.session import LiveSession
    except ImportError as e:
        console.print(
            error_panel(
                "Missing dependencies",
                f"Install streaming extras: pip install -e '.[streaming]'\n{e}",
            )
        )
        return

    settings = get_settings()
    live_model = model_size or "base"

    if not quiet:
        from cli.theme import APP_NAME, BOLD

        console.print()
        console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — Live Transcription")
        console.print(f"  [{DIM}]{'─' * 44}[/]")
        console.print(f"    Model:    {live_model}")
        console.print(f"    Language: {language}")
        if translate:
            console.print("    Task:     [bold]Translate → English[/]")
        console.print(f"    Device:   {settings.resolve_device()}")
        console.print(f"  [{DIM}]{'─' * 44}[/]")
        console.print()
        console.print(f"  [{DIM}]Press Ctrl+C to stop[/]")
        console.print()

    display = LiveTranscriptDisplay(console=console, quiet=quiet)
    session = LiveSession(
        model_size=live_model,
        language=language,
        translate=translate,
        on_segment=display.on_segment,
        on_partial=display.on_partial,
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
