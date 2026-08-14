"""Rename command for generating semantic titles."""

import json
from collections.abc import Sequence

import click
from rich.console import Console

from audiobench.core.db_session import get_session
from audiobench.storage.models import AudioFileRecord, TranscriptionRecord
from audiobench.transcribe.rename_service import generate_and_apply_title

console = Console()


@click.command(name="rename")
@click.argument("transcript_ids", type=int, nargs=-1)
@click.option(
    "--auto",
    is_flag=True,
    help="Automatically rescue and retry all files tagged with pending_auto_rename",
)
@click.option(
    "--gemini",
    is_flag=True,
    help="Force the use of Gemini (gemini-2.5-flash) instead of Ollama for title generation",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force physical disk renaming even if the current filename does not look like junk",
)
def rename(transcript_ids: Sequence[int], auto: bool, gemini: bool, force: bool) -> None:
    """
    Generate semantic titles and rename audio files using AI.
    
    If TRANSCRIPT_IDS are provided, it forces a rename on those specific transcriptions.
    If --auto is used, it automatically scans for and retries any files that failed 
    auto-renaming in the background.
    """
    if not transcript_ids and not auto:
        console.print("[yellow]Please provide transcript IDs to rename or use --auto.[/]")
        raise click.Abort()

    ids_to_process = list(transcript_ids)

    if auto:
        with get_session() as session:
            # Find all audio files that have the pending_auto_rename tag
            audio_records = session.query(AudioFileRecord).all()
            for record in audio_records:
                try:
                    tags = json.loads(record.tags) if record.tags else []
                except Exception:
                    tags = []
                if "pending_auto_rename" in tags:
                    # Find the primary transcription for this audio file
                    tx = session.query(TranscriptionRecord).filter_by(audio_file_id=record.id).first()
                    if tx and tx.id not in ids_to_process:
                        ids_to_process.append(tx.id)

        if not ids_to_process:
            console.print("[green]No files found waiting for auto-rename.[/]")
            return

        console.print(f"[bold blue]Found {len(ids_to_process)} file(s) pending auto-rename.[/]")

    console.print("\n  [bold]AudioBench — Auto Rename[/]")
    console.print("  ────────────────────────────────────────────")
    console.print(f"    Targets: {len(ids_to_process)} file(s)")
    console.print("  ────────────────────────────────────────────")

    success_count = 0
    for tx_id in ids_to_process:
        with console.status(f"[cyan]Generating title for #{tx_id}...[/]"):
            success, result = generate_and_apply_title(tx_id, force_gemini=gemini, force_disk=force)

        if success:
            console.print(f"[green]✓[/] #{tx_id} renamed to: [bold]{result}[/]")
            success_count += 1
        else:
            console.print(f"[red]✗[/] #{tx_id} failed: [dim]{result}[/]")

    console.print("\n  ────────────────────────────────────────────")
    console.print(f"  [green]✓ Renamed: {success_count}[/]  |  [red]✗ Failed: {len(ids_to_process) - success_count}[/]")
