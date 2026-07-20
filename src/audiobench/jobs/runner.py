"""Job runner — manages background execution of audiobench commands.

Spawns subprocesses that are immune to HUP (terminal closing).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from audiobench.core.settings import get_settings
from audiobench.jobs.repository import JobRepository


def is_alive(pid: int) -> bool:
    """Check if a process is alive using cross-platform os.kill."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def get_job_phase(job_id: int) -> str:
    """Read the last emitted phase/progress from the job's events file."""
    repo = JobRepository()
    job = repo.get_job(job_id)
    if not job or not job.get("events_path"):
        return "running"

    events_path = Path(job["events_path"])
    if not events_path.exists():
        return "starting"

    try:
        content = events_path.read_text().strip()
        if not content:
            return "starting"
        last_line = content.rsplit("\n", 1)[-1]
        parts = dict(p.split("=", 1) for p in last_line.split() if "=" in p)

        if "progress" in parts:
            return f"{parts.get('phase', 'transcribing')} {parts['progress']}%"
        return parts.get("phase", "running")
    except Exception:
        return "running"


def submit_job(args: list[str], audio_file: str | None = None) -> int:
    """Submit a command to run in the background."""
    repo = JobRepository()

    # Strip "audiobench" prefix if present
    if args and args[0] == "audiobench":
        args = args[1:]

    command_str = "audiobench " + " ".join(args)
    job_id = repo.create_job(command=command_str, audio_file=audio_file)

    settings = get_settings()
    log_dir = settings.data_dir / "job_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"job_{job_id}.log"

    # Pre-create log file
    log_path.touch()

    # Add --job-id argument to the command so child process can self-report
    # We insert it right after the subcommand, e.g. "transcribe --job-id 7 file.mp4"
    child_args = [sys.executable, "-m", "audiobench"]
    if args:
        child_args.append(args[0])  # subcommand
        child_args.extend(["--job-id", str(job_id)])
        child_args.extend(args[1:])

    # Open log file and pass it to child; close OUR copy immediately after Popen
    # so only the child holds the fd open. This prevents a handle leak in the
    # parent process.
    log_file = open(log_path, "w", buffering=1)

    # Windows doesn't support start_new_session, use creationflags instead if needed
    kwargs = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True
    else:
        # CREATE_NEW_PROCESS_GROUP = 0x00000200
        kwargs["creationflags"] = 0x00000200

    proc = subprocess.Popen(
        child_args, stdout=log_file, stderr=log_file, stdin=subprocess.DEVNULL, **kwargs
    )
    log_file.close()  # Parent closes its copy — child still has it open

    repo.update_job_started(job_id, proc.pid, str(log_path), events_path=None)
    return job_id


def watch_job(job_id: int) -> None:
    """Tail the log file of a job (running or finished)."""
    from audiobench.cli.display.theme import ACCENT, DIM, SUCCESS, WARNING, console

    repo = JobRepository()
    job = repo.get_job(job_id)

    if not job:
        console.print(f"  Job {job_id} not found.")
        return

    status = job.get("status", "unknown")
    if status == "running":
        console.print(f"  [{ACCENT}][Watching job #{job_id}][/] [{DIM}]Ctrl+C to detach[/]\n")
    else:
        console.print(f"  [{DIM}][Log for job #{job_id} — {status}][/]\n")

    log_path = job.get("log_path")
    if not log_path or not Path(log_path).exists():
        console.print(f"  [{WARNING}]Log file not found: {log_path}[/]")
        return

    try:
        with open(log_path) as f:
            # Dump anything already written
            for line in f:
                console.print(line, end="", highlight=False)

            if status != "running":
                # Job is already finished — just dump and return
                return

            # Job is still running — keep tailing
            while True:
                line = f.readline()
                if line:
                    console.print(line, end="", highlight=False)
                else:
                    current_job = repo.get_job(job_id)
                    if current_job and current_job.get("status") != "running":
                        final_status = current_job.get("status")
                        color = SUCCESS if final_status == "done" else WARNING
                        console.print(f"\n  [{color}][Job finished — {final_status}][/]")
                        break
                    time.sleep(0.15)
    except KeyboardInterrupt:
        console.print(f"\n  [{DIM}][Detached from job][/]")


def startup_recovery() -> None:
    """Mark stale running jobs as failed."""
    repo = JobRepository()
    running_jobs = repo.get_running_jobs()

    for job in running_jobs:
        pid = job.get("pid")
        if pid and not is_alive(pid):
            repo.mark_job_failed(job["id"], exit_code=-1)


def _pid_watcher_loop(iterations: int | None = None, interval: float = 30.0) -> None:
    """Poll running jobs and mark any with dead PIDs as failed.

    Args:
        iterations: How many poll cycles to run. ``None`` means run forever
            (the normal daemon case). Pass a small integer in tests to avoid
            an infinite loop.
        interval: Seconds to sleep between cycles. Pass 0 in tests.
    """
    import time as _time

    count = 0
    while iterations is None or count < iterations:
        try:
            repo = JobRepository()
            for job in repo.get_running_jobs():
                pid = job.get("pid")
                if pid and not is_alive(pid):
                    repo.mark_job_failed(job["id"], exit_code=-1)
        except Exception:
            pass  # Never crash the daemon over a watcher error
        count += 1
        if interval > 0:
            _time.sleep(interval)


def start_pid_watcher(interval: float = 30.0) -> "threading.Thread":
    """Spawn the continuous PID watcher as a background daemon thread.

    Returns the started thread so the caller can inspect it (e.g. in tests).
    """
    import threading

    t = threading.Thread(
        target=_pid_watcher_loop,
        kwargs={"interval": interval},
        name="pid-watcher",
        daemon=True,
    )
    t.start()
    return t
