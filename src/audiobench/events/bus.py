"""AudioBench EventBus — lightweight synchronous pub/sub for plugin hooks.

Design principles:
  - Thread-safe: subscriber list is protected by a lock.
  - Synchronous: subscribers run in the emitting thread. Fire-and-forget threads
    are the subscriber's responsibility if they need to be non-blocking.
  - Error-isolated: a crashing subscriber never kills the emitter. Errors are
    logged and the remaining subscribers still execute.
  - Zero dependencies outside stdlib.

Usage — emitting (inside core pipeline code):
    from audiobench.events import get_bus

    get_bus().emit("transcription.complete", tx_id=42, file_path="/path/to.mp3")

Usage — subscribing (inside a plugin's register() or at module level):
    from audiobench.events import subscribe

    @subscribe("transcription.complete")
    def on_done(tx_id: int, file_path: str, **kw):
        print(f"Done: {file_path} → #{tx_id}")

Available events
----------------
transcription.complete
    tx_id: int                  — ID in the `transcriptions` table
    file_path: str              — original audio file path
    duration_seconds: float     — audio duration
    word_count: int             — number of words transcribed
    language: str | None        — detected language code

summary.complete
    tx_id: int                  — transcription ID that was summarised
    summary: str                — the summary text

import.complete
    audio_file_id: int          — ID in the `audio_files` table
    file_path: str              — path of imported file
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("audiobench.events")


class EventBus:
    """Process-wide synchronous pub/sub broker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._handlers: dict[str, list[Callable[..., Any]]] = defaultdict(list)

    # ── Registration ──────────────────────────────────────────────────────────

    def on(self, event: str, fn: Callable[..., Any]) -> None:
        """Register *fn* as a subscriber for *event*.

        Supports glob-style wildcard: ``"*"`` matches every event.
        Duplicate registrations are silently ignored.
        """
        with self._lock:
            if fn not in self._handlers[event]:
                self._handlers[event].append(fn)

    def off(self, event: str, fn: Callable[..., Any]) -> None:
        """Remove a previously registered subscriber."""
        with self._lock:
            try:
                self._handlers[event].remove(fn)
            except ValueError:
                pass

    # ── Emission ──────────────────────────────────────────────────────────────

    def emit(self, event: str, **payload: Any) -> None:
        """Fire *event* with *payload* kwargs.

        Calls every handler registered for *event* plus any ``"*"`` wildcards.
        Handler exceptions are caught, logged, and never propagated.
        """
        with self._lock:
            handlers = list(self._handlers.get(event, []))
            handlers += [h for h in self._handlers.get("*", []) if h not in handlers]

        for fn in handlers:
            try:
                fn(**payload)
            except Exception:
                logger.exception(
                    "EventBus: handler %r raised for event %r",
                    getattr(fn, "__qualname__", repr(fn)),
                    event,
                )

    def subscribe(self, event: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a method/function as a subscriber.
        
        Example::
        
            @bus.subscribe("summary.complete")
            def on_done(summary: str, **kw): ...
        """
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.on(event, fn)
            return fn

        return decorator

    # ── Introspection ─────────────────────────────────────────────────────────

    def listeners(self, event: str | None = None) -> dict[str, list[str]]:
        """Return a snapshot of registered handlers (for ``\\graph``/debug)."""
        with self._lock:
            if event:
                fns = self._handlers.get(event, [])
                return {event: [getattr(f, "__qualname__", repr(f)) for f in fns]}
            return {
                ev: [getattr(f, "__qualname__", repr(f)) for f in fns]
                for ev, fns in self._handlers.items()
                if fns
            }


# ── Process singleton ─────────────────────────────────────────────────────────

_bus: EventBus | None = None
_bus_lock = threading.Lock()


def get_bus() -> EventBus:
    """Return the process-level EventBus instance (created on first call)."""
    global _bus
    if _bus is None:
        with _bus_lock:
            if _bus is None:
                _bus = EventBus()
    return _bus


# ── Decorator helper ──────────────────────────────────────────────────────────

def subscribe(event: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers the wrapped function as a subscriber.

    Example::

        @subscribe("transcription.complete")
        def my_handler(tx_id: int, file_path: str, **kw):
            ...
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        get_bus().on(event, fn)
        return fn

    return decorator
