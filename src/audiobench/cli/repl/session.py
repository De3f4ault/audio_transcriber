"""REPL session state — holds mutable state across the interactive session.

Provides:
    - ReplSession: Mutable state container (context, history, navigation)
    - CONTEXT_AWARE_COMMANDS: Set of commands that accept a transcript ID
"""

from __future__ import annotations

import contextlib
import readline

import click

from audiobench.cli.display.theme import WARNING, console
from audiobench.core.focused_entity import FocusedEntity

# ── Commands that accept a transcript ID as first positional arg ──
# When context is set and no ID is given, the REPL auto-injects it.

CONTEXT_AWARE_COMMANDS = {
    "show",  # show <id>
    "ask",  # ask <id> <question>
    "summarize",  # summarize <id>
    "vocab",  # vocab <id>
    "export",  # export <id>
    "delete",  # delete <id>
    "chat",  # chat <id...>  (nargs=-1, optional)
    "speak",  # speak <id>
}


class ReplSession:
    """Holds mutable REPL state across the interactive session."""

    def __init__(self, cli_group: click.Group) -> None:
        self.cli_group = cli_group
        self.focus: FocusedEntity | None = None

        # Use settings.data_dir for REPL history (project-local)
        from audiobench.core.settings import get_settings

        self._history_file = get_settings().data_dir / "repl_history"
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        self._repo = None
        self._command_count = 0
        self._history_ids: list[int] = []  # for .next / .prev navigation
        self._history_cursor: int = -1

    def _get_repo(self):
        """Lazy-init the transcription repository."""
        if self._repo is None:
            from audiobench.core.db_engine import init_db
            from audiobench.storage.repository import TranscriptionRepository

            init_db()
            self._repo = TranscriptionRepository()
        return self._repo

    def _load_history_ids(self) -> None:
        """Load all transcript IDs for .next/.prev navigation."""
        try:
            repo = self._get_repo()
            records = repo.get_history(limit=500)
            # Oldest first so .next goes forward in time
            self._history_ids = [r["id"] for r in reversed(records)]
        except Exception:
            self._history_ids = []

    # ── Prompt ──

    @property
    def last_id(self) -> int | None:
        """Derived property returning the relevant transcript ID for the current focus."""
        if not self.focus:
            return None
        if self.focus.type == "transcript":
            return self.focus.id
        if self.focus.type == "file":
            repo = self._get_repo()
            
            # If a chapter is focused, get the transcript ID of THAT chapter!
            if self.focus.chapter_index is not None:
                from audiobench.storage.chapter_repository import get_chapter_repo
                from audiobench.storage.models import ChapterRecord
                from audiobench.core.db_session import get_session
                chap = get_chapter_repo().get_chapter_by_index(self.focus.id, self.focus.chapter_index)
                if chap and chap.id:
                    with get_session() as db:
                        rec = db.query(ChapterRecord).filter_by(id=chap.id).first()
                        if rec and rec.transcription_id:
                            return rec.transcription_id
                return None
                
            tx = repo.get_latest_transcript_for_file(self.focus.id)
            return tx["id"] if tx else None
        return None

    def focus_chapter(self, chapter_index: int, chapter_title: str) -> bool:
        """Set focus to a specific chapter of the currently focused file."""
        if not self.focus or self.focus.type != "file":
            console.print(f"  [{WARNING}]You must focus on an audio file first.[/]")
            return False
            
        self.focus.chapter_index = chapter_index
        self.focus.chapter_title = chapter_title
        return True
        
    def clear_chapter_focus(self) -> None:
        """Clear the currently focused chapter."""
        if self.focus:
            self.focus.chapter_index = None
            self.focus.chapter_title = None

    @property
    def prompt(self) -> str:
        # Check for active background jobs
        job_badge = ""
        try:
            from audiobench.jobs.repository import JobRepository
            running = JobRepository().get_running_jobs()
            if running:
                n = len(running)
                label = "job" if n == 1 else "jobs"
                job_badge = f"\001\033[33m\002[{n} {label}]\001\033[0m\002 "
        except Exception:
            pass

        if self.focus:
            label = self.focus.display_label
            return f"{job_badge}\001\033[1;36m\002{label}\001\033[0m\002 \001\033[1;35m\002❯\001\033[0m\002 "
        return f"{job_badge}\001\033[1;35m\002❯\001\033[0m\002 "

    # ── Variable expansion ──

    def expand_vars(self, args: list[str]) -> list[str]:
        """Replace $last / $last_id / $id with the current context ID."""
        if self.last_id is None:
            return args
        sid = str(self.last_id)
        return [sid if a in ("$last", "$last_id", "$id") else a for a in args]

    # ── Context auto-injection ──

    def auto_inject_id(self, args: list[str]) -> list[str]:
        """For context-aware commands, inject the current ID if missing."""
        if not args or self.last_id is None:
            return args

        cmd_name = args[0]
        rest = args[1:]

        # Auto-inject file path and chapter for 'transcribe' if focused on a file
        if cmd_name == "transcribe" and self.focus and self.focus.type == "file":
            # Check if user already provided files (those without a leading dash, before any double dash)
            has_files = False
            for arg in rest:
                if not arg.startswith("-"):
                    has_files = True
                    break
                    
            injections = []
            if not has_files:
                repo = self._get_repo()
                audio_file = repo.get_audio_file(self.focus.id)
                if audio_file and audio_file.file_path:
                    injections.append(str(audio_file.file_path))
                    
            if self.focus.chapter_index is not None and "--chapters" not in rest:
                injections.extend(["--chapters", str(self.focus.chapter_index)])
                
            return [cmd_name] + injections + rest

        if cmd_name not in CONTEXT_AWARE_COMMANDS:
            return args

        # If the next arg is already a number, user specified an ID
        if rest and rest[0].isdigit():
            return args

        # If next arg is a flag, or there's no next arg — inject ID
        return [cmd_name, str(self.last_id)] + rest

    # ── Set context ──

    def set_context(self, record_id: int) -> None:
        """Set the current context using a transcript ID."""
        repo = self._get_repo()
        record = repo.get_by_id(record_id)
        
        if not record:
            console.print(f"  [{WARNING}]Transcript #{record_id} not found[/]")
            return

        # If transcript is linked to a file, focus on the file instead
        audio_file_id = record.get("audio_file_id")
        if audio_file_id:
            audio_file = repo.get_audio_file(audio_file_id)
            if audio_file:
                self.focus = FocusedEntity(
                    type="file", 
                    id=audio_file_id, 
                    label=audio_file["file_name"]
                )
            else:
                # Fallback if DB is inconsistent
                self.focus = FocusedEntity(
                    type="transcript", 
                    id=record_id, 
                    label=record.get("file_name", f"Transcript #{record_id}")
                )
        else:
            self.focus = FocusedEntity(
                type="transcript", 
                id=record_id, 
                label=record.get("file_name", f"Transcript #{record_id}")
            )

        # Update navigation cursor
        if record_id in self._history_ids:
            self._history_cursor = self._history_ids.index(record_id)

    def clear_context(self) -> None:
        """Clear the current context, return to bare prompt."""
        self.focus = None
        self._history_cursor = -1

    def refresh_context(self) -> None:
        """Refresh the current context if needed."""
        # Refresh logic is implicit now since last_id fetches live from DB
        pass

    # ── Readline history ──

    def load_history(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            readline.read_history_file(str(self._history_file))
        readline.set_history_length(1000)

    def save_history(self) -> None:
        with contextlib.suppress(OSError):
            readline.write_history_file(str(self._history_file))
