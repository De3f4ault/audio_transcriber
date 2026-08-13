import json
import queue
import sqlite3
import sys
import threading
import time

from audiobench.observatory.db import get_journal_db_path, write_events_conn
from audiobench.observatory.logfmt import write_logfmt_line
from audiobench.observatory.metrics import write_metrics_conn
from audiobench.observatory.thresholds import check_thresholds

QUEUE_MAX_SIZE: int = 10_000
DRAIN_BATCH_SIZE: int = 200

class ObservabilitySubscriber:
    __slots__ = ('_queue', '_thread')

    def __init__(self, queue_max: int = QUEUE_MAX_SIZE):
        self._queue = queue.Queue(maxsize=queue_max)
        self._thread = threading.Thread(target=self._drain, name="obs-drain", daemon=True)
        self._thread.start()
        import atexit
        atexit.register(self.flush)

    def flush(self) -> None:
        """Wait for the queue to drain before process exit."""
        self._queue.put(None)  # Sentinel to tell drain thread to flush and exit
        self._thread.join(timeout=2.0)

    def record(self, **payload) -> None:
        if "message" in payload and isinstance(payload["message"], str) and len(payload["message"]) > 500:
            payload["message"] = payload["message"][:497] + "..."

        if "metadata" in payload and isinstance(payload["metadata"], dict):
            try:
                raw = json.dumps(payload["metadata"], sort_keys=True, ensure_ascii=False)
                payload["metadata"] = raw[:4096] if len(raw) > 4096 else raw
            except Exception:
                pass

        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            print(f"[observatory] queue full — dropped: {payload.get('event_type')}", file=sys.stderr)

    def _drain(self) -> None:
        conn = None
        should_exit = False
        while not should_exit:
            try:
                if conn is None:
                    conn = sqlite3.connect(str(get_journal_db_path()), check_same_thread=False, isolation_level=None)
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")

                batch = []
                try:
                    item = self._queue.get(timeout=1.0)
                    if item is None:
                        should_exit = True
                    else:
                        batch.append(item)
                        while len(batch) < DRAIN_BATCH_SIZE:
                            try:
                                item = self._queue.get_nowait()
                                if item is None:
                                    should_exit = True
                                    break
                                batch.append(item)
                            except queue.Empty:
                                break
                except queue.Empty:
                    pass

                if not batch:
                    continue

                conn.execute("BEGIN TRANSACTION")
                write_events_conn(conn, batch)
                conn.execute("COMMIT")

                for event in batch:
                    try:
                        write_logfmt_line(event)
                        write_metrics_conn(conn, event)
                        check_thresholds(event)

                        if event.get("event_type") == "process_state_changed" and event.get("process"):
                            meta = event.get("metadata") or {}
                            if isinstance(meta, str):
                                try:
                                    meta = json.loads(meta)
                                except Exception:
                                    meta = {}
                            from audiobench.observatory.db import upsert_process_conn
                            state = meta.pop("state", "unknown")
                            upsert_process_conn(conn, event["process"], state, **meta)
                    except Exception as exc:
                        print(f"[observatory] drain non-db error: {exc}", file=sys.stderr)

            except sqlite3.DatabaseError as exc:
                print(f"[observatory] drain db error: {exc}", file=sys.stderr)
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
                time.sleep(0.1)
            except BaseException as exc:
                print(f"[observatory] drain error: {exc}", file=sys.stderr)
                time.sleep(0.1)

        if conn is not None:
            conn.close()

_subscriber_lock = threading.Lock()
_subscriber: ObservabilitySubscriber | None = None

def get_subscriber() -> ObservabilitySubscriber:
    global _subscriber
    with _subscriber_lock:
        if _subscriber is None:
            _subscriber = ObservabilitySubscriber()
    return _subscriber
