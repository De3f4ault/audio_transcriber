"""SQLAlchemy database setup — engine, session, and table creation.

Supports SQLite (default, zero-config) and PostgreSQL (swap via URL).

Usage:
    from src.audiobench.storage.database import get_session, init_db

    init_db()  # Create tables
    with get_session() as session:
        session.add(...)
        session.commit()
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.audiobench.config.logging_config import get_logger
from src.audiobench.config.settings import get_settings

logger = get_logger("storage.database")


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""

    pass


# Module-level engine and session factory (lazy init)
_engine = None
_SessionLocal = None


def _get_engine():
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        url = settings.database_url

        # For SQLite, ensure the parent directory exists
        if url.startswith("sqlite"):
            db_path = url.replace("sqlite:///", "")
            if db_path and not db_path.startswith(":"):
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        _engine = create_engine(
            url,
            echo=False,
            pool_pre_ping=True,
        )
        logger.info("Database engine created: %s", url.split("@")[-1] if "@" in url else url)

    return _engine


def _get_session_factory():
    """Get or create the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=_get_engine(), expire_on_commit=False)
    return _SessionLocal


def init_db() -> None:
    """Create all database tables."""
    from src.audiobench.storage.models import Base  # noqa: F811

    engine = _get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Get a database session as a context manager.

    Usage:
        with get_session() as session:
            session.add(record)
            session.commit()
    """
    factory = _get_session_factory()
    session = factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
