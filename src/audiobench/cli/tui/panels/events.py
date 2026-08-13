"""EventsPanel — live-tailing event stream, btop-style."""

from __future__ import annotations

import asyncio
from collections import deque

from rich.text import Text
from textual.binding import Binding
from textual.message import Message
from textual.widgets import RichLog

# ── Colour maps ────────────────────────────────────────────────────────────────

_LEVEL_STYLE: dict[str, str] = {
    "INFO":     "#64b5f6",   # cool blue
    "WARN":     "#ffb300",   # amber
    "WARNING":  "#ffb300",
    "ERROR":    "#ef5350",   # red
    "CRITICAL": "#b71c1c bold",
    "DEBUG":    "#78909c",   # muted blue-grey
}

_SUBSYSTEM_STYLE: dict[str, str] = {
    "supervisor": "#80cbc4",  # teal
    "storage":    "#a5d6a7",  # green
    "daemon":     "#ce93d8",  # purple
    "repl":       "#90caf9",  # light blue
    "events":     "#fff176",  # yellow
    "core":       "#ffcc80",  # orange
}

# Event types to suppress by default (TF/CUDA noise, etc.)
_SUPPRESSED_EVENTS = frozenset({"process_output"})


def _format_event(event: dict, raw: bool = False, suppress: bool = True) -> Text | None:
    """Format a single event row into a Rich Text object.

    Returns None if the event should be suppressed.
    """
    event_type = event.get("event_type", "")
    if suppress and event_type in _SUPPRESSED_EVENTS:
        return None

    ts = (event.get("ts") or "")[:19].replace("T", " ")
    level = (event.get("level") or "INFO").upper()
    subsystem = (event.get("subsystem") or "?")
    message = (event.get("message") or "")[:140]

    if raw:
        # logfmt-style
        return Text(
            f"ts={ts} level={level} sub={subsystem} type={event_type} msg={message!r}",
            no_wrap=True,
            overflow="crop",
        )

    lvl_style = _LEVEL_STYLE.get(level, "white")
    sub_style = _SUBSYSTEM_STYLE.get(subsystem, "#b0bec5")

    t = Text(no_wrap=True, overflow="crop")
    t.append(f"{ts[11:]}  ", style="dim")           # HH:MM:SS.mmm
    t.append(f"{level:5} ", style=lvl_style)
    t.append(f"{subsystem:12} ", style=sub_style)
    t.append(f"{event_type:28} ", style="bold #cfd8dc")
    t.append(message, style="#eceff1")
    return t


class LiveEventsPanel(RichLog):
    """Live-tailing events panel in btop style.

    • deque(maxlen=2000) for O(1) bounded memory
    • asyncio.to_thread() for non-blocking DB polls
    • Suppresses noisy process_output by default (toggle with 'n')
    """

    MAX_LINES = 2_000

    BINDINGS = [
        Binding("enter", "open_detail", "Detail View"),
    ]

    def __init__(
        self,
        subsystem_filter: str | None = None,
        level_filter: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(highlight=False, markup=False, **kwargs)
        self._rendered: deque[Text] = deque(maxlen=self.MAX_LINES)
        self._last_id: int = 0
        self._subsystem_filter = subsystem_filter
        self._level_filter = level_filter
        self._raw = False
        self._suppress_noise = True
        self._events_per_second: float = 0.0
        self._tick_count: int = 0

    class OpenDetail(Message):
        """Message emitted when user presses Enter to open detail view."""
        pass

    def action_open_detail(self) -> None:
        self.post_message(self.OpenDetail())

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self.set_interval(1.0, self._poll)

    # ── Polling ───────────────────────────────────────────────────────────────

    async def _poll(self) -> None:
        try:
            from audiobench.observatory.db import query_events
            events = await asyncio.to_thread(
                query_events,
                id_gt=self._last_id,
                limit=100,
                subsystem=self._subsystem_filter,
                level=self._level_filter,
            )
        except Exception:
            return

        new_count = 0
        for ev in events:
            line = _format_event(ev, raw=self._raw, suppress=self._suppress_noise)
            if line is not None:
                self._rendered.append(line)
                self.write(line)
                new_count += 1
            eid = ev.get("id")
            if eid is not None:
                self._last_id = max(self._last_id, eid)

        # Report rate to app for sparkline
        self._events_per_second = new_count
        self._tick_count += 1

        # Push to sparkline if app exposes one
        try:
            app = self.app
            if hasattr(app, "push_event_rate"):
                app.push_event_rate(new_count)
        except Exception:
            pass

    # ── Public API ────────────────────────────────────────────────────────────

    def set_filter(
        self,
        subsystem: str | None = None,
        level: str | None = None,
    ) -> None:
        """Change filters and replay from scratch."""
        self._subsystem_filter = subsystem
        self._level_filter = level
        self._last_id = 0
        self._rendered.clear()
        self.clear()

    def toggle_raw(self) -> bool:
        """Toggle logfmt view. Returns new state."""
        self._raw = not self._raw
        self._last_id = 0
        self._rendered.clear()
        self.clear()
        return self._raw

    def toggle_noise_suppression(self) -> bool:
        """Toggle suppression of process_output noise. Returns new state."""
        self._suppress_noise = not self._suppress_noise
        self._last_id = 0
        self._rendered.clear()
        self.clear()
        return self._suppress_noise
