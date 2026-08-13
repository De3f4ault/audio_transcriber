from typing import Any, Protocol, TypedDict


class EventPayload(TypedDict, total=False):
    id: int
    ts: str
    level: str
    subsystem: str
    event_type: str
    entity_type: str
    entity_id: str | int
    trace_id: str
    span_id: str
    parent_span_id: str
    message: str
    metadata: dict | str
    duration_ms: float
    session_id: str
    process: str
    source: str

class SubscriberProtocol(Protocol):
    def record(self, **payload: Any) -> None:
        ...
