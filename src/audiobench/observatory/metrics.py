import json
import sqlite3

from audiobench.observatory.db import write_metric_conn
from audiobench.observatory.types import EventPayload


def extract_metrics(payload: EventPayload) -> list[tuple[str, float, dict]]:
    results = []
    subsystem = payload.get("subsystem", "unknown")
    event_type = payload.get("event_type", "unknown")

    base_labels = {"subsystem": subsystem}

    metadata = payload.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}

    if "engine" in metadata: base_labels["engine"] = metadata["engine"]
    if "model" in metadata: base_labels["model"] = metadata["model"]

    if "duration_ms" in payload and payload["duration_ms"] is not None:
        results.append((f"{event_type}_duration_ms", float(payload["duration_ms"]), base_labels))

    if "confidence_avg" in metadata and metadata["confidence_avg"] is not None:
        results.append(("confidence_avg", float(metadata["confidence_avg"]), base_labels))

    if "top_score" in metadata and metadata["top_score"] is not None:
        results.append(("search_top_score", float(metadata["top_score"]), base_labels))

    if "result_count" in metadata and metadata["result_count"] is not None:
        results.append(("search_result_count", float(metadata["result_count"]), base_labels))

    if "segments_indexed" in metadata and metadata["segments_indexed"] is not None:
        results.append(("segments_indexed", float(metadata["segments_indexed"]), base_labels))

    if "segments_skipped" in metadata and metadata["segments_skipped"] is not None:
        results.append(("segments_skipped", float(metadata["segments_skipped"]), base_labels))

    return results

def write_metrics_conn(conn: sqlite3.Connection, payload: EventPayload) -> None:
    metrics = extract_metrics(payload)
    for m_name, m_value, m_labels in metrics:
        write_metric_conn(conn, payload.get("subsystem", "unknown"), m_name, m_value, m_labels)
