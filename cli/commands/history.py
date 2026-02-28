"""History + Search + Export + Delete (data management) commands."""

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
    make_table,
    stdout,
)


# ── History Command ─────────────────────────────────────────


@click.command()
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


# ── Search Command ──────────────────────────────────────────


@click.command()
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


# ── Export Command ──────────────────────────────────────────


@click.command()
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


# ── Delete Command ──────────────────────────────────────────


@click.command()
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
