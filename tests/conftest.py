"""Shared fixtures for the AudioBench test suite.

Provides:
    - runner: Click CliRunner for CLI tests
    - tmp_data_dir: Temporary data directory (isolates DB, presets, etc.)
    - test_settings: Patched settings pointing to tmp_data_dir
    - test_db: Initialized temp database with session factory
    - sample_audio_dir: Directory with fake audio files for collection tests
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory structure."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "plugins").mkdir()
    (data_dir / "presets").mkdir()
    (data_dir / "logs").mkdir()
    return data_dir


@pytest.fixture
def test_settings(tmp_data_dir, monkeypatch):
    """Patched settings that use a temp directory for all data.

    This ensures tests never touch the real database or user data.
    """
    from audiobench.core.settings import get_settings

    db_path = tmp_data_dir / "test.db"

    # Inject test paths into the environment so pydantic-settings picks them up.
    # This avoids aliasing bugs where modules import `get_settings` directly
    # before we can mock.patch it.
    monkeypatch.setenv("AUDIOBENCH_DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("AUDIOBENCH_DATA_DIR", str(tmp_data_dir))
    monkeypatch.setenv("AUDIOBENCH_MODELS_DIR", str(tmp_data_dir / "models"))

    # Clear the lru_cache so the next call rebuilds from the patched environment
    get_settings.cache_clear()
    settings = get_settings()

    yield settings

    get_settings.cache_clear()


@pytest.fixture
def test_db(test_settings):
    """Initialize a test database and return the settings.

    The DB is created fresh in a temp dir — destroyed after the test.

    Both module-level singletons must be reset: _engine in db_engine and
    _SessionLocal in db_session. Resetting only _engine leaves db_session
    bound to the previous test's SQLite file, causing cross-test state leakage.
    """
    import audiobench.core.db_engine as db_mod
    import audiobench.core.db_session as sess_mod

    old_engine = db_mod._engine
    old_session_factory = sess_mod._SessionLocal

    # Tear down both singletons so init_db builds fresh against the new tmp path
    db_mod._engine = None
    sess_mod._SessionLocal = None

    from audiobench.core.db_engine import init_db

    init_db()
    yield test_settings

    # Dispose the test engine so SQLite file handles are released before restoring
    if db_mod._engine is not None:
        db_mod._engine.dispose()

    # Restore originals (supports nested/parallel fixture usage)
    db_mod._engine = old_engine
    sess_mod._SessionLocal = old_session_factory


@pytest.fixture
def sample_audio_dir(tmp_path):
    """Create a directory tree with fake audio files for testing file_collector."""
    root = tmp_path / "audio"
    root.mkdir()

    # Flat files
    (root / "meeting.mp3").write_text("fake")
    (root / "podcast.m4a").write_text("fake")
    (root / "notes.txt").write_text("not audio")
    (root / "draft_take.mp3").write_text("fake")

    # Subdirectory
    sub = root / "sub"
    sub.mkdir()
    (sub / "interview.wav").write_text("fake")
    (sub / "backup.mp3").write_text("fake")

    return root
