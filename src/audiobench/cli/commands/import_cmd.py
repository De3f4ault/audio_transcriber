from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, SpinnerColumn, TaskID, TextColumn

from audiobench.cli.display.theme import ACCENT, BOLD, DIM, error_panel

console = Console()


def run_import_flow(session=None, restore_state=None) -> list[int]:
    """Run the import flow and return a list of newly imported or updated audio_file IDs."""
    try:
        import questionary
    except ImportError:
        console.print(
            error_panel("Dependencies missing", "Please install questionary and prompt_toolkit")
        )
        return []

    # 1 & 2. Full-Screen Directory Navigation & File Selection
    from audiobench.cli.tui.import_tui import launch_file_manager

    selected_files, state = launch_file_manager(start_state=restore_state)

    if session and getattr(session, "navigation_stack", None) and session.navigation_stack[-1].context == "import":
        session.navigation_stack[-1].state = state
        session._persist_stack()

    if selected_files == "LAUNCH_TRANSCRIPT_IMPORT":
        from audiobench.cli.tui.reimport_tui import ReimportTUI
        tui = ReimportTUI(batch=False)
        if not tui.run():
            console.print(f"\n  [{DIM}]Reimport cancelled.[/]")
        return []

    if not selected_files:
        console.print(f"\n  [{DIM}]Import cancelled or no files selected.[/]")
        return []

    # Sort files alphabetically for the allocation wizard
    selected_files.sort(key=lambda x: x.name.lower())

    # 3. Engine Allocation Wizard
    console.print(f"\n  [{DIM}]Allocating engines for {len(selected_files)} files...[/]")

    allocations = []
    for file in selected_files:
        engine = questionary.select(
            f"Engine for '{file.name}':", choices=["gemini", "whisper", "skip"]
        ).ask()

        if engine and engine != "skip":
            allocations.append({"file": file, "engine": engine})

    if not allocations:
        console.print(f"  [{DIM}]All files skipped. Import cancelled.[/]")
        return []

    # 4. Trigger Background Importer
    console.print(f"\n  [{BOLD} {ACCENT}]Ready to import {len(allocations)} files![/]")

    from audiobench.storage.importer import BackgroundImporter

    importer = BackgroundImporter(max_workers=4)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        transient=False,
    )

    # We will track overall progress and individual files
    job_tasks: dict[str, TaskID] = {}

    def on_progress(total_bytes: int, copied_bytes: int, filename: str) -> None:
        if filename not in job_tasks:
            task = progress.add_task(f"[cyan]{filename}", total=total_bytes)
            job_tasks[filename] = task
        else:
            progress.update(job_tasks[filename], completed=copied_bytes)

    def on_file_done(source: Path, dest: Path, engine: str) -> None:
        pass  # Optional hook to database

    importer.set_callbacks(progress=on_progress, file_done=on_file_done)

    with progress:
        results = importer.run_import_jobs(allocations)

    console.print(f"\n  [{BOLD} {ACCENT}]✓ Successfully imported {len(results)} files![/]")

    # 5. Register to Database
    from audiobench.core.db_session import get_session
    from audiobench.storage.models import AudioFileRecord

    imported_ids = []
    with get_session() as session:
        for res in results:
            dest: Path = res["destination"]
            engine: str = res["engine"]
            is_dup = res["is_duplicate"]
            file_hash = res["file_hash"]

            if is_dup:
                existing_id = res["existing_id"]
                audio_record = session.query(AudioFileRecord).get(existing_id)
                if audio_record:
                    audio_record.tags = f'["engine_preference: {engine}"]'
                    imported_ids.append(audio_record.id)
                    console.print(
                        f"  [{DIM}]Skipped duplicate data for '{dest.name}', but updated engine to {engine}.[/]"
                    )
            else:
                # Create AudioFileRecord for the newly imported file
                audio_record = AudioFileRecord(
                    file_path=str(dest),
                    file_name=dest.name,
                    file_size_bytes=dest.stat().st_size,
                    format=dest.suffix.lstrip("."),
                    file_hash=file_hash,
                    tags=f'["engine_preference: {engine}"]',
                )
                session.add(audio_record)
                session.flush()  # flush to get the id
                imported_ids.append(audio_record.id)
        session.commit()

    # Fire plugin hooks for each newly imported file
    try:
        from audiobench.events import get_bus
        _bus = get_bus()
        for res in results:
            if not res.get("is_duplicate"):
                _bus.emit(
                    "import.complete",
                    audio_file_id=res.get("audio_file_id") or imported_ids[results.index(res)],
                    file_path=str(res["destination"]),
                )
    except Exception:
        pass

    console.print(f"  [{DIM}]Files registered to library database. Ready for transcription![/]")
    return imported_ids


@click.command(name="import")
@click.option("-T", "--transcript", is_flag=True, help="Reverse import: select transcript file first")
@click.option("--batch", is_flag=True, help="Batch mode for --transcript (folder-to-folder mapping)")
def import_cmd(transcript: bool, batch: bool):
    """Import audio files into the internal library.

    \b
    Examples:
      audiobench import
      audiobench import --transcript
      audiobench import --transcript --batch
    """
    if transcript:
        from audiobench.cli.tui.reimport_tui import ReimportTUI
        tui = ReimportTUI(batch=batch)
        if not tui.run():
            console.print(f"\n  [{DIM}]Reimport cancelled.[/]")
        return
        
    if batch and not transcript:
        console.print(error_panel("Error", "--batch is only supported with --transcript"))
        return

    run_import_flow()
