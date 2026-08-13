"""
ContextVar module for distributed tracing.
Cross-boundary rules:
1. Process boundary (daemon/worker): Pass trace_id and span_id explicitly via IPC/payload.
2. Executor boundary (asyncio/threads): contextvars propagate automatically to threads started via asyncio.to_thread or ThreadPoolExecutor.
"""
import contextlib
import contextvars
import uuid
from datetime import datetime

from audiobench.observatory.types import EventPayload

_current_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("_current_trace_id", default=None)
_current_span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("_current_span_id", default=None)
_current_parent_span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("_current_parent_span_id", default=None)

def start_trace() -> str:
    trace_id = str(uuid.uuid4())[:8]
    _current_trace_id.set(trace_id)
    _current_span_id.set(None)
    _current_parent_span_id.set(None)
    return trace_id

def start_span() -> tuple[str, contextvars.Token]:
    span_id = str(uuid.uuid4())[:8]
    _current_parent_span_id.set(_current_span_id.get())
    token = _current_span_id.set(span_id)
    return span_id, token

def end_span(token: contextvars.Token) -> None:
    _current_span_id.reset(token)
    # Note: parent_span_id isn't trivially resettable to a history stack using just ContextVars without a full list,
    # but the token resets the current_span_id correctly.

@contextlib.contextmanager
def span(name: str | None = None):
    span_id, token = start_span()
    try:
        yield span_id
    finally:
        end_span(token)

def current_trace_id() -> str | None:
    return _current_trace_id.get()

def current_span_id() -> str | None:
    return _current_span_id.get()

def log_event(
    subsystem: str,
    event_type: str,
    message: str,
    level: str = "INFO",
    *,
    entity_type: str | None = None,
    entity_id: int | str | None = None,
    duration_ms: float | None = None,
    metadata: dict | None = None,
    session_id: str | None = None,
    process: str | None = None
) -> None:
    from audiobench.observatory.subscriber import get_subscriber

    payload: EventPayload = {
        "ts": datetime.utcnow().isoformat(timespec='microseconds'),
        "level": level,
        "subsystem": subsystem,
        "event_type": event_type,
        "message": message,
    }

    t_id = _current_trace_id.get()
    s_id = _current_span_id.get()
    ps_id = _current_parent_span_id.get()

    if t_id is not None: payload["trace_id"] = t_id
    if s_id is not None: payload["span_id"] = s_id
    if ps_id is not None: payload["parent_span_id"] = ps_id
    if entity_type is not None: payload["entity_type"] = entity_type
    if entity_id is not None: payload["entity_id"] = entity_id
    if duration_ms is not None: payload["duration_ms"] = duration_ms
    if metadata is not None: payload["metadata"] = metadata
    if session_id is not None: payload["session_id"] = session_id
    if process is not None: payload["process"] = process

    get_subscriber().record(**payload)
