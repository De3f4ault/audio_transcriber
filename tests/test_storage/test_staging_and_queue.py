"""Tests for StagingCartItem and JobQueueItem models."""

from __future__ import annotations

from audiobench.core.db_session import get_session
from audiobench.storage.models import AudioFileRecord, JobQueueItem, StagingCartItem


def test_staging_cart_item_lifecycle(test_db):
    """Test creating, updating, and deleting StagingCartItem records."""
    with get_session() as session:
        # Create a parent AudioFileRecord
        audio = AudioFileRecord(
            file_path="/fake/path/audio.mp3",
            file_name="audio.mp3",
            file_size_bytes=1024,
            format="mp3",
        )
        session.add(audio)
        session.commit()

        audio_id = audio.id

        # Add to cart
        cart_item = StagingCartItem(
            audio_file_id=audio_id, engine="gemini", model_name="pro", speed_preset="fast"
        )
        session.add(cart_item)
        session.commit()

        # Query and verify
        fetched = session.query(StagingCartItem).filter_by(audio_file_id=audio_id).first()
        assert fetched is not None
        assert fetched.engine == "gemini"
        assert fetched.model_name == "pro"
        assert fetched.speed_preset == "fast"
        assert fetched.audio_file.file_name == "audio.mp3"

        # Test cascade delete
        session.delete(fetched.audio_file)
        session.commit()

        # Cart item should be deleted
        assert session.query(StagingCartItem).count() == 0


def test_job_queue_item_lifecycle(test_db):
    """Test creating and updating JobQueueItem records."""
    with get_session() as session:
        job = JobQueueItem(
            file_path="/fake/path/job.mp3",
            engine="whisper",
            model_name="large-v3",
            speed_preset="balanced",
        )
        session.add(job)
        session.commit()

        job_id = job.id

        fetched = session.query(JobQueueItem).get(job_id)
        assert fetched is not None
        assert fetched.status == "pending"
        assert fetched.file_path == "/fake/path/job.mp3"

        fetched.status = "processing"
        session.commit()

        assert session.query(JobQueueItem).get(job_id).status == "processing"
