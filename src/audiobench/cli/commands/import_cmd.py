from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, SpinnerColumn, TaskID, TextColumn

from audiobench.cli.display.theme import ACCENT, BOLD, DIM, error_panel

console = Console()


def extract_metadata(file_path: Path) -> tuple[str | None, str | None]:
    title, author = None, None
    try:
        import mutagen
        m = mutagen.File(file_path, easy=True)
        if m:
            title = m.get("title", [None])[0]
            if not title:
                title = m.get("album", [None])[0]
            author = m.get("artist", [None])[0]
            if not author:
                author = m.get("albumartist", [None])[0]
    except ImportError:
        pass
    except Exception:
        pass
        
    if not title:
        name = file_path.stem
        if " - " in name:
            parts = name.split(" - ", 1)
            if len(parts) >= 2:
                author = parts[0].strip()
                title = parts[1].strip()
    return title, author


def run_import_flow(session=None, restore_state=None, initial_mode="MODE_AUDIO", override_title=None, override_author=None) -> list[int]:
    """Run the import flow and return a list of newly imported or updated audio_file IDs."""
    try:
        import questionary
    except ImportError:
        console.print(
            error_panel("Dependencies missing", "Please install questionary and prompt_toolkit")
        )
        return []

    from audiobench.cli.wizard import prompt_menu
    from audiobench.cli.tui.import_tui import launch_file_manager
    from audiobench.cli.tui.reimport_tui import ReimportTUI

    current_mode = initial_mode
    shared_state = restore_state or {}
    
    # Check if we should show the main menu
    if current_mode == "main_menu":
        options = [
            ("Audio", "Import raw audio files", "MODE_AUDIO"),
            ("Auto-Detect", "Intelligently scan and pair transcripts/audio", "MODE_AUTO"),
            ("Manual Pair", "Manually link a single transcript to an audio file", "MODE_SINGLE_TX"),
            ("Batch Folders", "Map a folder of transcripts to a folder of audio", "MODE_BATCH_TX"),
            ("Exit", "Cancel and exit", "exit")
        ]
        current_mode = prompt_menu("Select Import Mode", options, default_idx=0)
        if not current_mode or current_mode == "exit":
            return []

    # Routing Loop
    while current_mode:
        if current_mode == "MODE_AUDIO":
            selected_files, new_state = launch_file_manager(start_state=shared_state)
            shared_state.update(new_state)
            
            if isinstance(selected_files, str) and selected_files.startswith("MODE_"):
                current_mode = selected_files
                continue
            
            if not selected_files:
                console.print(f"\n  [{DIM}]Import cancelled or no files selected.[/]")
                return []
            
            # Break out of loop to process audio files below
            break
            
        elif current_mode in ("MODE_AUTO", "MODE_SINGLE_TX", "MODE_BATCH_TX"):
            is_auto = (current_mode == "MODE_AUTO")
            is_batch = (current_mode == "MODE_BATCH_TX")
            
            tui = ReimportTUI(batch=is_batch, auto_detect=is_auto, shared_state=shared_state)
            result = tui.run()
            shared_state.update(tui.state_export())
            
            if isinstance(result, str) and result.startswith("MODE_"):
                current_mode = result
                continue
                
            if not result:
                console.print(f"\n  [{DIM}]Transcript import cancelled.[/]")
            # Transcript imports don't return audio IDs for staging
            return []
            
        else:
            break

    if session and getattr(session, "navigation_stack", None) and session.navigation_stack[-1].context == "import":
        session.navigation_stack[-1].state = shared_state
        session._persist_stack()

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
    from audiobench.storage.models import AudioFileRecord, WorkRecord
    from audiobench.observatory.context import log_event

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
                title, author = extract_metadata(dest)
                if override_title: title = override_title
                if override_author: author = override_author
                
                work_id = None
                if title:
                    work_record = session.query(WorkRecord).filter_by(title=title, author=author).first()
                    if not work_record:
                        work_record = WorkRecord(title=title, author=author)
                        session.add(work_record)
                        session.flush()
                    work_id = work_record.id

                # Create AudioFileRecord for the newly imported file
                audio_record = AudioFileRecord(
                    file_path=str(dest),
                    file_name=dest.name,
                    file_size_bytes=dest.stat().st_size,
                    format=dest.suffix.lstrip("."),
                    file_hash=file_hash,
                    tags=f'["engine_preference: {engine}"]',
                    work_id=work_id,
                )
                session.add(audio_record)
                session.flush()  # flush to get the id
                
                if not work_id:
                    log_event(
                        "import", 
                        "work_unassigned", 
                        f"Work could not be assigned for {dest.name}", 
                        entity_type="audio_file", 
                        entity_id=audio_record.id
                    )
                
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
@click.option("--auto-detect", is_flag=True, help="Auto-detect mode: recursively scan and pair audio and transcript files")
@click.option("--title", default=None, help="Force assign a work title to imported files")
@click.option("--author", default=None, help="Force assign a work author to imported files")
def import_cmd(transcript: bool, batch: bool, auto_detect: bool, title: str | None, author: str | None):
    """Import audio files into the internal library.

    \b
    Examples:
      audiobench import
      audiobench import --transcript
      audiobench import --transcript --batch
      audiobench import --auto-detect
    """
    initial_mode = "main_menu"
    
    if auto_detect:
        initial_mode = "MODE_AUTO"
    elif transcript and batch:
        initial_mode = "MODE_BATCH_TX"
    elif transcript:
        initial_mode = "MODE_SINGLE_TX"
    elif not any([transcript, batch, auto_detect]):
        initial_mode = "main_menu"

    run_import_flow(initial_mode=initial_mode, override_title=title, override_author=author)
