"""Tests for the Background Queue Worker daemon."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from audiobench.core.db_session import get_session
from audiobench.jobs.queue_worker import (
    acquire_os_lock,
    enqueue_batch,
    process_queue,
)
from audiobench.storage.models import JobQueueItem


@pytest.fixture
def mock_subprocess():
    """Mock subprocess to prevent actual execution."""
    with patch("audiobench.jobs.queue_worker.subprocess") as mock_sub:
        yield mock_sub


def test_acquire_os_lock(tmp_path):
    """Test the OS file locking logic."""
    lock_file = tmp_path / "test.lock"

    # Test Unix
    with patch("os.name", "posix"), patch("fcntl.flock") as mock_flock:
        f = acquire_os_lock(lock_file)
        assert f is not None
        assert mock_flock.called
        f.close()

    # Test Windows
    mock_msvcrt = MagicMock()
    with patch("os.name", "nt"), patch.dict("sys.modules", {"msvcrt": mock_msvcrt}):
        f = acquire_os_lock(lock_file)
        assert f is not None
        assert mock_msvcrt.locking.called
        f.close()

    # Test locked failure
    with patch("os.name", "posix"), patch("fcntl.flock", side_effect=BlockingIOError):
        f = acquire_os_lock(lock_file)
        assert f is None


@patch("audiobench.jobs.queue_worker._spawn_daemon")
def test_enqueue_batch(mock_spawn, test_db):
    """Test enqueuing batch adds to db and spawns daemon."""
    with get_session() as session:
        session.query(JobQueueItem).delete()
        session.commit()

    files = ["/fake/1.mp3", "/fake/2.mp3"]
    enqueue_batch(files, engine="gemini", model="pro", speed_preset="fast")

    with get_session() as session:
        items = session.query(JobQueueItem).order_by(JobQueueItem.id).all()
        assert len(items) == 2
        assert items[0].file_path == "/fake/1.mp3"
        assert items[0].status == "pending"

    mock_spawn.assert_called_once()


def test_process_queue(test_db, mock_subprocess, tmp_path):
    """Test process_queue processes all items and updates status."""
    with get_session() as session:
        session.query(JobQueueItem).delete()
        session.add(JobQueueItem(file_path="/fake/success.mp3", status="pending"))
        session.add(JobQueueItem(file_path="/fake/fail.mp3", status="pending"))
        session.commit()

    # Mock subprocess.run to return success for first, fail for second
    success_result = MagicMock()
    success_result.returncode = 0

    fail_result = MagicMock()
    fail_result.returncode = 1
    fail_result.stderr = "Error"

    mock_subprocess.run.side_effect = [success_result, fail_result]

    # Run the worker synchronously
    process_queue()

    with get_session() as session:
        items = session.query(JobQueueItem).order_by(JobQueueItem.id).all()
        assert items[0].status == "done"
        assert items[1].status == "failed"
