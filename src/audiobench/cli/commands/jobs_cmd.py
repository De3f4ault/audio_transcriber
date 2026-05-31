"""Jobs command — manage background transcriptions."""

from __future__ import annotations

import os
import signal
from pathlib import Path

import click

from audiobench.cli.display.theme import ACCENT, BOLD, DIM, SUCCESS, WARNING, console, error_panel, make_table
from audiobench.jobs.repository import JobRepository
from audiobench.jobs.runner import get_job_phase, is_alive, startup_recovery, watch_job


@click.group(invoke_without_command=True)
@click.pass_context
def jobs(ctx: click.Context) -> None:
    """Manage background transcription jobs.

    Run without arguments to list recent jobs.
    """
    from audiobench.core.platform import SUPPORTS_BACKGROUND_JOBS
    import sys
    
    if not SUPPORTS_BACKGROUND_JOBS:
        console.print(f"  [{WARNING}]Background jobs are only supported on Linux/macOS.[/]")
        sys.exit(1)

    startup_recovery()
    
    if ctx.invoked_subcommand is None:
        _list_jobs()


def _list_jobs() -> None:
    repo = JobRepository()
    all_jobs = repo.get_all_jobs(limit=20)
    
    if not all_jobs:
        console.print(f"  [{DIM}]No recent jobs found[/]")
        return
        
    table = make_table(
        "Background Jobs",
        [
            ("ID", {"width": 4, "justify": "right"}),
            ("Status", {"width": 10}),
            ("Command", {}),
            ("Phase", {"width": 20}),
            ("Started", {"style": DIM}),
        ]
    )
    
    for job in all_jobs:
        job_id = job["id"]
        status = job.get("status", "unknown")
        cmd_str = job.get("command", "")
        # truncate command
        if cmd_str.startswith("audiobench "):
            cmd_str = cmd_str[11:]
        if len(cmd_str) > 40:
            cmd_str = cmd_str[:37] + "..."
            
        started = str(job.get("started_at", ""))[:16]  # Trim seconds/microseconds
        
        # Colorize status
        if status == "running":
            status_disp = f"[{ACCENT}]running[/]"
        elif status == "done":
            status_disp = f"[{SUCCESS}]done[/]"
        elif status == "failed":
            status_disp = f"[{WARNING}]failed[/]"
        elif status == "cancelled":
            status_disp = f"[{WARNING}]cancelled[/]"
        else:
            status_disp = status
            
        phase = get_job_phase(job_id) if status == "running" else ""
        
        table.add_row(
            f"#{job_id}",
            status_disp,
            cmd_str,
            phase,
            started
        )
        
    console.print(table)


@jobs.command(name="fg")
@click.argument("job_id", type=int)
def watch(job_id: int) -> None:
    """Bring a background job to the foreground (tail its logs).

    Works for both running and finished jobs.
    """
    startup_recovery()
    repo = JobRepository()
    job = repo.get_job(job_id)
    if not job:
        console.print(error_panel("Not Found", f"Job #{job_id} does not exist"))
        return

    # watch_job now handles both running and finished jobs
    watch_job(job_id)


@jobs.command(name="cancel")
@click.argument("job_id", type=int)
def cancel(job_id: int) -> None:
    """Cancel a running background job."""
    startup_recovery()
    repo = JobRepository()
    job = repo.get_job(job_id)

    if not job:
        console.print(error_panel("Not Found", f"Job #{job_id} does not exist"))
        return

    if job.get("status") != "running":
        console.print(f"  [{DIM}]Job #{job_id} is already {job.get('status')}[/]")
        return

    pid = job.get("pid")
    if pid and is_alive(pid):
        try:
            # We used start_new_session=True, so the PID is the Process Group ID (PGID).
            # We MUST use os.killpg to kill the entire group (including ffmpeg/whisper child processes).
            # Otherwise we orphan the heavy compute processes!
            os.killpg(pid, signal.SIGINT)
            console.print(f"  [{SUCCESS}]Sent SIGINT to process group #{job_id} (PGID {pid})[/]")

            # Wait, then SIGKILL if still alive
            import time
            time.sleep(1)
            if is_alive(pid):
                os.killpg(pid, signal.SIGKILL)
                console.print(f"  [{WARNING}]Force killed process group #{job_id} (PGID {pid})[/]")
        except ProcessLookupError:
            pass
        except PermissionError:
            console.print(error_panel("Permission Denied", f"Cannot kill PID {pid}"))
            return

    # Mark as cancelled with exit_code=130 (SIGINT convention)
    with repo._get_conn() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'cancelled', ended_at = CURRENT_TIMESTAMP, exit_code = 130 WHERE id = ?",
            (job_id,)
        )
    console.print(f"  [{SUCCESS}]Job #{job_id} cancelled[/]")


@jobs.command(name="logs")
@click.argument("job_id", type=int)
def logs(job_id: int) -> None:
    """Show the full log output of a job (any status)."""
    startup_recovery()
    repo = JobRepository()
    job = repo.get_job(job_id)
    if not job:
        console.print(error_panel("Not Found", f"Job #{job_id} does not exist"))
        return
    watch_job(job_id)


@jobs.command(name="prune")
@click.option("--all", "prune_all", is_flag=True, help="Also remove running jobs (dangerous)")
def prune(prune_all: bool) -> None:
    """Remove finished job records and their log files."""
    repo = JobRepository()
    all_jobs = repo.get_all_jobs(limit=1000)

    removed = 0
    for job in all_jobs:
        status = job.get("status", "")
        if status in ("done", "failed", "cancelled") or (prune_all and status == "running"):
            # Delete log files
            for path_key in ("log_path", "events_path"):
                p = job.get(path_key)
                if p and Path(p).exists():
                    try:
                        Path(p).unlink()
                    except OSError:
                        pass
            # Delete DB record
            with repo._get_conn() as conn:
                conn.execute("DELETE FROM jobs WHERE id = ?", (job["id"],))
            removed += 1

    if removed:
        console.print(f"  [{SUCCESS}]✓[/] Pruned {removed} job record(s)")
    else:
        console.print(f"  [{DIM}]Nothing to prune[/]")
