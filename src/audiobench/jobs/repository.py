"""Database operations for background jobs."""

from __future__ import annotations

import sqlite3

from audiobench.core.settings import get_settings


def dict_factory(cursor: sqlite3.Cursor, row: tuple) -> dict:
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


class JobRepository:
    def __init__(self) -> None:
        self.db_url = get_settings().database_url
        if self.db_url.startswith("sqlite:///"):
            self.db_path = self.db_url.replace("sqlite:///", "")
        else:
            self.db_path = None

    def _get_conn(self) -> sqlite3.Connection:
        if not self.db_path:
            raise RuntimeError("JobRepository only supports SQLite")
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = dict_factory
        return conn

    def create_job(self, command: str, audio_file: str | None = None) -> int:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO jobs (command, audio_file, status, started_at) VALUES (?, ?, 'running', CURRENT_TIMESTAMP)",
                (command, audio_file),
            )
            return cursor.lastrowid

    def update_job_started(self, job_id: int, pid: int, log_path: str, events_path: str) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE jobs SET pid = ?, log_path = ?, events_path = ?, status = 'running' WHERE id = ?",
                (pid, log_path, events_path, job_id),
            )

    def mark_job_failed(self, job_id: int, exit_code: int | None = None) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'failed', ended_at = CURRENT_TIMESTAMP, exit_code = ? WHERE id = ?",
                (exit_code, job_id),
            )

    def mark_job_done(self, job_id: int) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'done', ended_at = CURRENT_TIMESTAMP, exit_code = 0 WHERE id = ?",
                (job_id,),
            )

    def get_job(self, job_id: int) -> dict | None:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            return cursor.fetchone()

    def get_all_jobs(self, limit: int = 50) -> list[dict]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))
            return cursor.fetchall()

    def get_running_jobs(self) -> list[dict]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs WHERE status = 'running'")
            return cursor.fetchall()
