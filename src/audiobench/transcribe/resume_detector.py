import os

from rich.console import Console
from rich.panel import Panel

from audiobench.jobs.repository import JobRepository


class ResumeDetector:
    """Detects orphaned transcription jobs and prompts the user for resumption."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self.repo = JobRepository()

    def get_orphaned_jobs(self) -> list[dict]:
        """Return a list of jobs stuck in 'running' state whose PID is dead."""
        orphans = []
        running_jobs = self.repo.get_running_jobs()
        for job in running_jobs:
            pid = job.get("pid")
            # If there's no PID, or the PID is no longer alive, it's an orphan
            if not pid or not self._is_pid_alive(pid):
                orphans.append(job)
        return orphans

    def _is_pid_alive(self, pid: int) -> bool:
        """Check if a process is alive (Unix/Linux)."""
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Alive, but we don't have permission to kill it (e.g. root process).
            return True
        except Exception:
            # Fallback for Windows or strange OS issues
            return False

    def mark_failed(self, job_ids: list[int]) -> None:
        """Mark these orphaned jobs as failed."""
        for jid in job_ids:
            self.repo.mark_job_failed(jid, exit_code=-1)

    def prompt_user_to_resume(self, orphans: list[dict]) -> list[dict]:
        """
        Prompt the user to resume orphaned jobs.
        Returns the list of jobs the user selected to resume.
        """
        if not orphans:
            return []

        import questionary

        self.console.print("\n  [bold yellow]⚠ Interrupted Jobs Detected[/]")
        self.console.print("  [dim]The following transcriptions were interrupted and can be resumed from the last checkpoint:[/]")
        
        choices = []
        for job in orphans:
            file_path = job.get("audio_file") or job.get("command", "Unknown File")
            # Create a short name
            short_name = file_path.split("/")[-1][:40]
            if len(file_path.split("/")[-1]) > 40:
                short_name += "..."
                
            choices.append(
                questionary.Choice(
                    title=f"{short_name} (Job #{job['id']})",
                    value=job,
                    checked=True
                )
            )

        try:
            selected_jobs = questionary.checkbox(
                "Select jobs to resume (they will be queued):",
                choices=choices,
                style=questionary.Style([("highlighted", "fg:#000000 bg:#00d7d7")])
            ).ask()
            
            if selected_jobs is None:
                selected_jobs = []
                
        except Exception:
            selected_jobs = []

        # Mark all unselected orphans as failed so they don't prompt again
        unselected_ids = [j["id"] for j in orphans if j not in selected_jobs]
        self.mark_failed(unselected_ids)

        return selected_jobs
