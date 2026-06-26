"""ObservatoryApp — btop-faithful live monitoring TUI for AudioBench.

Layout:
┌─ header (1 row: title │ rate │ clock) ──────────────────────────────────────┐
├─ Events  sub:all  lvl:all  ⊘noise ──────────────────────────────────────────┤
│  HH:MM:SS  INFO  supervisor  state_changed  ...                             │
│  HH:MM:SS  WARN  daemon      process_output ...                             │
│  ...                                                                        │
├─────────────────┬──────────────────────────┬────────────────────────────────┤
│ Processes       │ Health                   │ Jobs                           │
│ ● daemon RUNNING│ transcriptions ▮▮▮░ MB   │ (empty)                        │
│                 │ journal.db     ▮░░ MB    │                                │
└─────────────────┴──────────────────────────┴────────────────────────────────┘
[ f Filter  n Noise  l Logfmt  s Subsystem  r Refresh  q Quit ]
"""

from __future__ import annotations

import datetime

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Footer, Static, Input, Label
from textual.containers import Container

from audiobench.cli.tui.panels.events import LiveEventsPanel
from audiobench.cli.tui.panels.health import HealthPanel
from audiobench.cli.tui.panels.processes import ProcessesPanel
from audiobench.cli.tui.panels.operations import OperationsPanel


# ── TCSS ──────────────────────────────────────────────────────────────────────
# Background #101010 matches btop's main_bg.
# Borders are `solid` (thin ┌─┐) not `round` (╭╮╰╯).
# Accent colors directly from btop's default theme:
#   hi_fg=#b54800  selected_bg=#006200  title=#eeeeee  main_fg=#cecfc9

CSS = """
/* ── Root ── */
Screen {
    background: #101010;
    color: #cecfc9;
}

/* ── 1-row custom header (no Textual Header widget) ── */
#header-bar {
    height: 1;
    background: #1c1c1c;
    layout: horizontal;
    border-bottom: tall #2a2a2a;
}
#header-title {
    width: 1fr;
    color: #b54800;
    text-style: bold;
    content-align: left middle;
    padding: 0 1;
}
#header-rate {
    width: auto;
    color: #3d9a50;
    content-align: center middle;
    padding: 0 2;
}
#header-clock {
    width: 10;
    color: #606060;
    content-align: right middle;
    padding: 0 1;
}

/* ── Outer split ── */
#outer {
    layout: vertical;
    height: 1fr;
}

/* ── Top half: events ── */
#events-box {
    height: 1fr;
    border: solid #2a2a2a;
    border-title-color: #606060;
    padding: 0;
    background: #101010;
}
LiveEventsPanel {
    background: #101010;
    height: 1fr;
    scrollbar-size: 1 1;
    scrollbar-color: #2a2a2a;
    scrollbar-background: #101010;
}

/* ── Bottom half ── */
#bottom-row {
    layout: horizontal;
    height: 14; /* Fixed height for the bottom panels */
}

#processes-box {
    width: 1fr;
    height: 1fr;
    border: solid #2a2a2a;
    border-title-color: #3d9a50;
    border-title-style: bold;
    padding: 0;
    background: #101010;
}
ProcessesPanel {
    background: #101010;
    height: 1fr;
}

#health-box {
    width: 1fr;
    height: 1fr;
    border: solid #2a2a2a;
    border-title-color: #1c6bb0;
    border-title-style: bold;
    background: #101010;
    overflow-y: auto;
    padding: 0;
}
HealthPanel {
    background: #101010;
    padding: 0 0 1 0;
}

#ops-box {
    width: 1fr;
    height: 1fr;
    border: solid #2a2a2a;
    border-title-color: #6040a0;
    padding: 0;
    background: #101010;
}
OperationsPanel {
    background: #101010;
    height: 1fr;
}

/* ── Footer ── */
Footer {
    background: #1c1c1c;
    color: #606060;
    height: 1;
}
Footer > .footer--key {
    background: #2a2a2a;
    color: #b54800;
}
Footer > .footer--highlight {
    background: #006200;
    color: #eeeeee;
}

/* ── Filter modal ── */
FilterModal > #filter-bg {
    align: center middle;
}
#filter-dialog {
    background: #1c1c1c;
    border: solid #b54800;
    padding: 1 2;
    width: 60;
    height: auto;
}
#filter-dialog Label {
    color: #606060;
    margin-bottom: 1;
}
#filter-dialog Input {
    background: #101010;
    border: solid #2a2a2a;
    color: #cecfc9;
    margin-bottom: 1;
}
#filter-hint {
    color: #b54800;
}

/* ── Confirm modal ── */
ConfirmModal {
    align: center middle;
}
#confirm-dialog {
    background: #1c1c1c;
    border: solid #ef5350;
    padding: 1 2;
    width: auto;
    min-width: 40;
    height: auto;
}
#confirm-message {
    color: #cecfc9;
    text-style: bold;
    margin-bottom: 1;
}
#confirm-hint {
    color: #ef5350;
}
"""


# ── Filter modal ──────────────────────────────────────────────────────────────

class FilterModal(ModalScreen):
    """Popup filter dialog."""

    BINDINGS = [Binding("escape", "dismiss(None)", "Cancel")]

    def __init__(
        self,
        current_subsystem: str | None,
        current_level: str | None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._cur_sub = current_subsystem or ""
        self._cur_lvl = current_level or ""

    def compose(self) -> ComposeResult:
        with Container(id="filter-dialog"):
            yield Label("  Filter Events  ", id="filter-title")
            yield Label("Subsystem  (leave empty for all)")
            yield Input(
                value=self._cur_sub,
                placeholder="supervisor  daemon  storage  repl  events …",
                id="sub-input",
            )
            yield Label("Level  (leave empty for all)")
            yield Input(
                value=self._cur_lvl,
                placeholder="INFO  WARN  ERROR  CRITICAL",
                id="lvl-input",
            )
            yield Label("[Enter] Apply    [Esc] Cancel", id="filter-hint")

    def on_input_submitted(self, _: Input.Submitted) -> None:
        sub = self.query_one("#sub-input", Input).value.strip() or None
        lvl = self.query_one("#lvl-input", Input).value.strip().upper() or None
        self.dismiss((sub, lvl))


# ── Main app ──────────────────────────────────────────────────────────────────

class ObservatoryApp(App):
    """AudioBench Observatory — btop-faithful live monitoring."""

    CSS = CSS
    TITLE = "AudioBench Observatory"

    # Remove default Header so we draw our own
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("f", "filter",          "Filter",    priority=True),
        Binding("n", "toggle_noise",    "Noise",     priority=True),
        Binding("l", "toggle_raw",      "Logfmt",    priority=True),
        Binding("s", "cycle_subsystem", "Subsystem", priority=True),
        Binding("r", "force_refresh",   "Refresh",   priority=True),
        Binding("q", "quit",            "Quit",      priority=True),
    ]

    _SUBSYSTEM_CYCLE = [None, "supervisor", "daemon", "storage", "repl", "events", "core"]

    def __init__(
        self,
        subsystem: str | None = None,
        level: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._subsystem = subsystem
        self._level = level
        self._sub_idx = 0
        self._suppress_noise = True
        if subsystem in self._SUBSYSTEM_CYCLE:
            self._sub_idx = self._SUBSYSTEM_CYCLE.index(subsystem)

    # ── Layout ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        # 1-row header (no Textual Header widget)
        with Horizontal(id="header-bar"):
            yield Static("◈  AudioBench Observatory", id="header-title")
            yield Static("", id="header-rate")
            yield Static("", id="header-clock")

        # Main split
        with Vertical(id="outer"):

            # Top half: event log panel
            with Vertical(id="events-box"):
                yield LiveEventsPanel(
                    subsystem_filter=self._subsystem,
                    level_filter=self._level,
                    id="events-panel",
                )

            # Bottom half
            with Horizontal(id="bottom-row"):
                with Vertical(id="processes-box"):
                    yield ProcessesPanel(id="processes-panel")
                with Vertical(id="health-box"):
                    yield HealthPanel(id="health-panel")
                with Vertical(id="ops-box"):
                    yield OperationsPanel(id="ops-panel")

        yield Footer()

    def on_mount(self) -> None:
        self._update_titles()
        self.set_interval(1.0, self._tick)

    # ── Tick ──────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        now = datetime.datetime.now().strftime("%H:%M:%S")
        self.query_one("#header-clock", Static).update(now)

    # ── Event feed (called by LiveEventsPanel each poll) ───────────────────────

    def push_event_rate(self, count: int) -> None:
        rate_str = f"  {count} evt/s  " if count else ""
        self.query_one("#header-rate", Static).update(rate_str)

    # ── Title helpers ─────────────────────────────────────────────────────────

    def _events_title(self) -> str:
        sub   = self._subsystem or "all"
        lvl   = self._level or "all"
        noise = "⊘noise" if self._suppress_noise else "noise"
        return f" Events  sub:{sub}  lvl:{lvl}  {noise} "

    def _update_titles(self) -> None:
        self.query_one("#events-box").border_title     = self._events_title()
        self.query_one("#processes-box").border_title  = " Processes "
        self.query_one("#health-box").border_title     = " Health "
        self.query_one("#ops-box").border_title        = " Jobs "

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_filter(self) -> None:
        def _apply(result) -> None:
            if result is None:
                return
            sub, lvl = result
            self._subsystem = sub
            self._level = lvl
            self.query_one("#events-panel", LiveEventsPanel).set_filter(
                subsystem=sub, level=lvl
            )
            self.query_one("#events-box").border_title = self._events_title()

        self.push_screen(
            FilterModal(
                current_subsystem=self._subsystem,
                current_level=self._level,
            ),
            callback=_apply,
        )

    def action_toggle_noise(self) -> None:
        panel = self.query_one("#events-panel", LiveEventsPanel)
        self._suppress_noise = panel.toggle_noise_suppression()
        self.query_one("#events-box").border_title = self._events_title()
        self.notify(
            "process_output hidden" if self._suppress_noise else "Showing all events",
            timeout=2,
        )

    def action_toggle_raw(self) -> None:
        raw = self.query_one("#events-panel", LiveEventsPanel).toggle_raw()
        self.notify("Logfmt view" if raw else "Rich view", timeout=1)

    def action_cycle_subsystem(self) -> None:
        self._sub_idx = (self._sub_idx + 1) % len(self._SUBSYSTEM_CYCLE)
        self._subsystem = self._SUBSYSTEM_CYCLE[self._sub_idx]
        self.query_one("#events-panel", LiveEventsPanel).set_filter(
            subsystem=self._subsystem, level=self._level
        )
        self.query_one("#events-box").border_title = self._events_title()
        label = self._subsystem or "all"
        self.notify(f"Subsystem → {label}", timeout=1)

    def action_force_refresh(self) -> None:
        self.query_one("#health-panel",    HealthPanel)._refresh()
        self.query_one("#processes-panel", ProcessesPanel)._refresh()
        self.query_one("#ops-panel",       OperationsPanel)._refresh()
        self.notify("Panels refreshed", timeout=1)

    def on_live_events_panel_open_detail(self, message: LiveEventsPanel.OpenDetail) -> None:
        from audiobench.cli.tui.screens.event_detail_screen import EventDetailScreen
        self.push_screen(EventDetailScreen(subsystem=self._subsystem, level=self._level))
