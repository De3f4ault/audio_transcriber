"""Sequential background worker daemon using OS file locks.

This module provides the queue worker that processes items from the `job_queue`
table sequentially. It uses OS-level file locking to ensure only a single
worker is active at any time, preventing stale locks if the process crashes violently.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.storage.models import JobQueueItem

logger = get_logger("jobs.queue_worker")


def acquire_os_lock(lock_file_path: Path):
    """Acquire a non-blocking exclusive OS file lock.

    Returns the file object if successful, None if already locked.
    """
    try:
        f = open(lock_file_path, "w")
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f
    except (BlockingIOError, OSError):
        # Could not acquire lock, meaning another worker is running
        try:
            f.close()
        except:
            pass
        return None


def enqueue_batch(
    staged_files: list[str],
    engine: str | None = None,
    model: str | None = None,
    speed_preset: str | None = None,
):
    """Enqueue a batch of files and start the background worker if not running."""
    with get_session() as session:
        for file_path in staged_files:
            item = JobQueueItem(
                file_path=file_path,
                engine=engine,
                model_name=model,
                speed_preset=speed_preset,
                status="pending",
            )
            session.add(item)
        session.commit()

    _spawn_daemon()


def _spawn_daemon():
    """Spawn the worker process in the background using nohup equivalent."""
    # We invoke this module directly
    cmd = [sys.executable, "-m", "audiobench.jobs.queue_worker"]

    # Detach
    log_dir = Path(get_settings().data_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "worker.log"

    with open(log_file, "a") as out:
        subprocess.Popen(
            cmd,
            stdout=out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )


def process_queue(foreground: bool = False):
    """Main worker loop. Runs sequentially until the queue is empty."""
    settings = get_settings()
    lock_path = Path(settings.data_dir) / "worker.lock"

    lock_file = acquire_os_lock(lock_path)
    if not lock_file:
        logger.info("Worker is already running. Exiting.")
        return

    logger.info("Worker started, acquired OS lock.")
    processed_count = 0

    try:
        while True:
            with get_session() as session:
                item = (
                    session.query(JobQueueItem)
                    .filter_by(status="pending")
                    .order_by(JobQueueItem.id)
                    .first()
                )

                # Update the signal file for the REPL prompt
                active_file = Path(settings.data_dir) / "jobs.active"
                if item:
                    # Quick count for the signal file
                    total = session.query(JobQueueItem).filter(
                        JobQueueItem.status.in_(["pending", "processing"])
                    ).count()
                    try:
                        active_file.write_text(str(total))
                    except Exception:
                        pass
                else:
                    try:
                        active_file.unlink(missing_ok=True)
                    except Exception:
                        pass
                    break

                item.status = "processing"
                session.commit()

                # Copy properties
                file_path = item.file_path
                engine = item.engine
                model = item.model_name
                preset = item.speed_preset
                item_id = item.id

            logger.info("Processing job %s for %s", item_id, file_path)

            try:
                cmd = ["audiobench", "transcribe", file_path]
                if engine:
                    cmd.extend(["--engine", engine])
                if model:
                    cmd.extend(["--model", model])
                if preset:
                    cmd.append(f"--{preset}")

                if not foreground:
                    # We also run it quietly so it doesn't try to draw rich progress bars in a log file
                    cmd.append("-q")
                    result = subprocess.run(cmd, capture_output=True, text=True)
                else:
                    # Let the user see the progress bars
                    result = subprocess.run(cmd)

                with get_session() as session:
                    item = session.query(JobQueueItem).get(item_id)
                    if result.returncode == 0:
                        item.status = "done"
                    else:
                        item.status = "failed"
                        err_msg = result.stderr if result.stderr else "See console output"
                        logger.error("Job %s failed: %s", item_id, err_msg)
                    session.commit()

                processed_count += 1
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt during job %s", item_id)
                with get_session() as session:
                    item = session.query(JobQueueItem).get(item_id)
                    item.status = "pending" # Reset to pending
                    session.commit()
                if foreground:
                    import click
                    from audiobench.cli.display.theme import console, WARNING
                    console.print(f"\n  [{WARNING}]Transcription interrupted![/]")
                    if click.confirm("Send remaining jobs to background?"):
                        _spawn_daemon()
                    return
                else:
                    return
            except Exception as e:
                logger.error("Error processing job %s: %s", item_id, e)
                with get_session() as session:
                    item = session.query(JobQueueItem).get(item_id)
                    item.status = "failed"
                    session.commit()

        logger.info("Queue is empty. Worker shutting down.")

        # Notify the user that all batch jobs have completed
        if processed_count > 0:
            try:
                # Basic desktop notification via os / notify-send / osascript
                if sys.platform == "darwin":
                    subprocess.run(
                        [
                            "osascript",
                            "-e",
                            'display notification "All queued transcriptions have completed." with title "AudioBench"',
                        ]
                    )
                elif sys.platform == "linux":
                    subprocess.run(
                        ["notify-send", "AudioBench", "All queued transcriptions have completed."]
                    )
            except Exception:
                pass

    finally:
        # lock_file is automatically released when closed or process dies
        lock_file.close()


if __name__ == "__main__":
    process_queue()
