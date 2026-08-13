"""
Write path: drain thread holds one persistent sqlite3.Connection. Never use this connection outside the drain thread.
Read path: get_journal_session() opens a short-lived read-only connection per query. Safe from any thread.
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from audiobench.core.settings import get_settings
from audiobench.observatory.types import EventPayload


def get_journal_db_path() -> Path:
    return Path(get_settings().data_dir) / "journal.db"

def get_journal_session() -> sqlite3.Connection:
    path = get_journal_db_path()
    # Read-only short-lived connection
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn

def init_journal_db() -> None:
    settings = get_settings()
    log_dir = Path(settings.data_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    db_path = get_journal_db_path()
    conn = sqlite3.connect(str(db_path))

    sql_path = Path(__file__).parent.parent / "storage" / "migrations" / "014_journal.sql"
    with open(sql_path, encoding="utf-8") as f:
        sql_text = f.read()

    conn.executescript(sql_text)

    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=OFF;")
    conn.execute("PRAGMA cache_size=-8000;")

    conn.execute(
        "INSERT OR IGNORE INTO journal_meta (key, value) VALUES (?, ?)",
        ("schema_version", "1")
    )
    conn.commit()
    conn.close()

    from audiobench.observatory.migrate_legacy import migrate_command_events, migrate_log_files
    migrate_command_events()
    migrate_log_files()

def write_events_conn(conn: sqlite3.Connection, payloads: list[EventPayload]) -> None:
    tuples = []
    for payload in payloads:
        ts = payload.get("ts") or datetime.utcnow().isoformat(timespec='microseconds')
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            metadata = json.dumps(metadata)
        tuples.append((
            ts,
            payload.get("level", "INFO"),
            payload.get("subsystem", "unknown"),
            payload.get("event_type", "unknown"),
            payload.get("entity_type"),
            payload.get("entity_id"),
            payload.get("trace_id"),
            payload.get("span_id"),
            payload.get("parent_span_id"),
            payload.get("message"),
            metadata,
            payload.get("duration_ms"),
            payload.get("session_id"),
            payload.get("process"),
            payload.get("source", "live")
        ))

    conn.executemany(
        """
        INSERT INTO system_events (
            ts, level, subsystem, event_type, entity_type, entity_id,
            trace_id, span_id, parent_span_id, message, metadata,
            duration_ms, session_id, process, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        tuples
    )

def write_event_conn(conn: sqlite3.Connection, payload: EventPayload) -> None:
    write_events_conn(conn, [payload])

def write_metric_conn(conn: sqlite3.Connection, subsystem: str, metric: str, value: float, labels: dict | None = None) -> None:
    ts = datetime.utcnow().isoformat(timespec='microseconds')
    labels_str = json.dumps(labels) if labels else None

    conn.execute(
        """
        INSERT INTO system_metrics (ts, subsystem, metric, value, labels)
        VALUES (?, ?, ?, ?, ?)
        """,
        (ts, subsystem, metric, value, labels_str)
    )

def upsert_process_conn(conn: sqlite3.Connection, name: str, state: str, **fields) -> None:
    updated_at = datetime.utcnow().isoformat(timespec='microseconds')

    # We will just construct an INSERT OR REPLACE that includes all possible fields.
    # Since we don't know if the process row already exists, we will read it first or construct it.
    # However, managed_processes is simple enough.

    # Actually, a better approach is to do a dynamic UPDATE or a full INSERT OR REPLACE.
    # The requirement is just INSERT OR REPLACE INTO managed_processes.

    # Read existing fields if any to preserve them if not provided.
    row = conn.execute("SELECT * FROM managed_processes WHERE name = ?", (name,)).fetchone()

    pid = fields.get("pid", row[2] if row else None)
    started_at = fields.get("started_at", row[3] if row else None)
    stopped_at = fields.get("stopped_at", row[4] if row else None)
    restart_count = fields.get("restart_count", row[5] if row else 0)
    last_exit_code = fields.get("last_exit_code", row[6] if row else None)
    last_error = fields.get("last_error", row[7] if row else None)

    conn.execute(
        """
        INSERT OR REPLACE INTO managed_processes (
            name, state, pid, started_at, stopped_at, 
            restart_count, last_exit_code, last_error, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (name, state, pid, started_at, stopped_at, restart_count, last_exit_code, last_error, updated_at)
    )

def query_events(
    subsystem: str | None = None,
    level: str | None = None,
    entity_type: str | None = None,
    entity_id: str | int | None = None,
    trace_id: str | None = None,
    session_id: str | None = None,
    since: str | None = None,
    limit: int = 100,
    id_gt: int | None = None
) -> list[EventPayload]:
    query = "SELECT * FROM system_events WHERE 1=1"
    params: list[str | int | float] = []

    if subsystem is not None:
        query += " AND subsystem = ?"
        params.append(subsystem)
    if level is not None:
        query += " AND level = ?"
        params.append(level)
    if entity_type is not None:
        query += " AND entity_type = ?"
        params.append(entity_type)
    if entity_id is not None:
        query += " AND entity_id = ?"
        params.append(str(entity_id))
    if trace_id is not None:
        query += " AND trace_id = ?"
        params.append(trace_id)
    if session_id is not None:
        query += " AND session_id = ?"
        params.append(session_id)
    if since is not None:
        query += " AND ts >= ?"
        params.append(since)
    if id_gt is not None:
        query += " AND id > ?"
        params.append(id_gt)

    if id_gt is not None:
        # Ascending: return new events in chronological order for live tail
        query += " ORDER BY id ASC LIMIT ?"
    else:
        # Descending: return most-recent events first for log viewer
        query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_journal_session() as conn:
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()

    result: list[EventPayload] = []
    for r in rows:
        evt: EventPayload = {
            "id": r["id"],
            "ts": r["ts"],
            "level": r["level"],
            "subsystem": r["subsystem"],
            "event_type": r["event_type"],
        }
        if r["entity_type"] is not None: evt["entity_type"] = r["entity_type"]
        if r["entity_id"] is not None: evt["entity_id"] = r["entity_id"]
        if r["trace_id"] is not None: evt["trace_id"] = r["trace_id"]
        if r["span_id"] is not None: evt["span_id"] = r["span_id"]
        if r["parent_span_id"] is not None: evt["parent_span_id"] = r["parent_span_id"]
        if r["message"] is not None: evt["message"] = r["message"]
        if r["metadata"] is not None:
            try:
                evt["metadata"] = json.loads(r["metadata"])
            except Exception:
                evt["metadata"] = r["metadata"]
        if r["duration_ms"] is not None: evt["duration_ms"] = r["duration_ms"]
        if r["session_id"] is not None: evt["session_id"] = r["session_id"]
        if r["process"] is not None: evt["process"] = r["process"]
        if r["source"] is not None: evt["source"] = r["source"]

        result.append(evt)

    return result
