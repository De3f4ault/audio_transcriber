import dataclasses
import functools
import json

from audiobench.core.settings import get_settings
from audiobench.observatory.types import EventPayload


@dataclasses.dataclass(frozen=True)
class ThresholdRule:
    field: str
    operator: str
    value: float
    level: str

@functools.lru_cache(maxsize=1)
def get_threshold_rules() -> tuple[ThresholdRule, ...]:
    defaults = {
        "transcribe.confidence": ThresholdRule("transcribe.confidence", "lt", 0.70, "WARN"),
        "daemon.search_latency_ms": ThresholdRule("daemon.search_latency_ms", "gt", 1000, "WARN"),
        "supervisor.restart_count": ThresholdRule("supervisor.restart_count", "gt", 2, "CRITICAL"),
        "memory.embed_confidence": ThresholdRule("memory.embed_confidence", "lt", 0.60, "WARN"),
    }

    settings = get_settings()
    obs_settings = getattr(settings, "observatory", {})
    thresholds = obs_settings.get("thresholds", {})

    for k, v in thresholds.items():
        if isinstance(v, (list, tuple)) and len(v) == 3:
            defaults[k] = ThresholdRule(k, v[0], float(v[1]), v[2])

    return tuple(defaults.values())

def check_thresholds(payload: EventPayload) -> None:
    if payload.get("event_type") == "threshold_exceeded":
        return

    rules = get_threshold_rules()

    metadata = payload.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}

    numeric_fields = {}
    subsystem = payload.get("subsystem", "unknown")

    for k, v in payload.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            key = k if "." in k else f"{subsystem}.{k}"
            numeric_fields[key] = float(v)

    for k, v in metadata.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            key = k if "." in k else f"{subsystem}.{k}"
            numeric_fields[key] = float(v)

    for rule in rules:
        if rule.field in numeric_fields:
            observed = numeric_fields[rule.field]
            fired = False
            if rule.operator == "lt" and observed < rule.value or rule.operator == "gt" and observed > rule.value:
                fired = True

            if fired:
                from audiobench.observatory.subscriber import get_subscriber

                alert_payload: EventPayload = {
                    "level": rule.level,
                    "subsystem": subsystem,
                    "event_type": "threshold_exceeded",
                    "message": f"Threshold crossed: {rule.field}={observed}",
                    "metadata": {
                        "rule": rule.field,
                        "value": observed,
                        "threshold": rule.value,
                        "source_event_type": payload.get("event_type"),
                        "source_span_id": payload.get("span_id")
                    }
                }
                get_subscriber().record(**alert_payload)
