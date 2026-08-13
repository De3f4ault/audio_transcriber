"""CLI commands for the AudioBench Observatory.

Commands:
  audiobench obs                  — Launch four-panel live TUI
  audiobench obs --subsystem X    — Open filtered by subsystem
  audiobench obs --level WARN     — Open filtered by level
  audiobench logs                 — Print last 100 events (non-interactive)
  audiobench logs --subsystem X
  audiobench logs --level WARN
  audiobench logs --entity TYPE:ID
  audiobench logs --since "2 hours ago"
  audiobench logs --follow        — Poll loop, no TUI
  audiobench logs --session last  — Events from most recent session_id
"""

from __future__ import annotations

import click

from audiobench.cli.display.theme import DIM, console


# Make `audiobench obs` launch the TUI directly without subcommands
@click.command(name="obs")
@click.option("--subsystem", "-s", default=None, help="Filter by subsystem")
@click.option("--level", "-l", default=None, help="Filter by level (INFO/WARN/ERROR/CRITICAL)")
def obs_command(subsystem: str | None, level: str | None) -> None:
    """Launch the Observatory four-panel live TUI.

    \b
    Keybindings:
      f  Filter events (subsystem, level)
      e  Expand selected event (full metadata JSON)
      l  Toggle raw logfmt tail view
      s  Cycle subsystem focus
      p  Focus process table
      q  Quit
    """
    from audiobench.events import get_bus
    from audiobench.observatory.db import init_journal_db
    from audiobench.observatory.subscriber import get_subscriber

    init_journal_db()
    get_bus().on("*", get_subscriber().record)

    from audiobench.cli.tui.observatory_app import ObservatoryApp
    app = ObservatoryApp(subsystem=subsystem, level=level)
    app.run()


@click.command(name="logs")
@click.option("--subsystem", "-s", default=None, help="Filter by subsystem")
@click.option("--level", "-l", default=None, help="Filter by level")
@click.option("--entity", default=None, help="Filter by entity, e.g. audio_file:42")
@click.option("--since", default=None, help='Filter by time, e.g. "2 hours ago" or ISO datetime')
@click.option("--session", default=None, help="Filter by session_id; use 'last' for most recent")
@click.option("--follow", is_flag=True, help="Poll for new events (like tail -f)")
@click.option("--limit", default=100, show_default=True, help="Max events to show")
def logs_command(
    subsystem: str | None,
    level: str | None,
    entity: str | None,
    since: str | None,
    session: str | None,
    follow: bool,
    limit: int,
) -> None:
    """Print Observatory events (non-interactive log viewer).

    \b
    Examples:
      audiobench logs
      audiobench logs --level WARN
      audiobench logs --subsystem daemon --follow
      audiobench logs --entity audio_file:126
      audiobench logs --since "1 hour ago"
      audiobench logs --session last
    """
    import time

    from audiobench.observatory.db import get_journal_session, init_journal_db, query_events

    init_journal_db()

    # Parse --entity flag
    entity_type: str | None = None
    entity_id: str | int | None = None
    if entity:
        try:
            entity_type, entity_id_str = entity.split(":", 1)
            entity_id = int(entity_id_str)
        except (ValueError, AttributeError):
            console.print("[red]Invalid --entity format. Use TYPE:ID, e.g. audio_file:42[/]")
            return

    # Parse --since flag
    since_iso: str | None = None
    if since:
        since_iso = _parse_since(since)
        if since_iso is None:
            console.print(f"[red]Could not parse --since value: {since!r}[/]")
            return

    # Resolve --session last
    session_id: str | None = None
    if session == "last":
        with get_journal_session() as conn:
            row = conn.execute(
                "SELECT session_id FROM system_events WHERE session_id IS NOT NULL "
                "ORDER BY ts DESC LIMIT 1"
            ).fetchone()
        session_id = row["session_id"] if row else None
        if session_id is None:
            console.print(f"[{DIM}]No sessions found in Observatory.[/]")
            return
        console.print(f"[dim]Session: {session_id}[/]")
    elif session:
        session_id = session

    def _print_batch(events: list) -> None:
        for ev in reversed(events):  # oldest first for log view
            ts = (ev.get("ts") or "")[:19]
            lvl = ev.get("level", "INFO")
            sub = ev.get("subsystem", "?")
            etype = ev.get("event_type", "")
            msg = (ev.get("message") or "")[:200]

            level_colour = {
                "INFO": "cyan", "WARN": "yellow", "ERROR": "red",
                "CRITICAL": "bold red", "DEBUG": "dim",
            }.get(lvl, "white")

            console.print(
                f"[dim]{ts}[/] [{level_colour}]{lvl:5}[/] [bold]{sub:12}[/] {etype:30} {msg}"
            )

    if follow:
        console.print("[dim]Following Observatory events. Ctrl+C to stop.[/]\n")
        last_id = 0
        try:
            while True:
                events = query_events(
                    subsystem=subsystem, level=level,
                    entity_type=entity_type, entity_id=entity_id,
                    since=since_iso, session_id=session_id,
                    id_gt=last_id, limit=50,
                )
                if events:
                    _print_batch(events)
                    last_id = max(e.get("id", 0) for e in events)
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    else:
        events = query_events(
            subsystem=subsystem, level=level,
            entity_type=entity_type, entity_id=entity_id,
            since=since_iso, session_id=session_id,
            limit=limit,
        )
        if not events:
            console.print(f"[{DIM}]No events found.[/]")
            return
        _print_batch(list(reversed(events)))  # most-recent-first → oldest first display
        console.print(f"\n[{DIM}]{len(events)} event(s)[/]")


def _parse_since(value: str) -> str | None:
    """Parse a human-readable 'since' string into an ISO 8601 timestamp."""
    import datetime
    import re

    value = value.strip().lower()

    # Try ISO datetime first
    try:
        dt = datetime.datetime.fromisoformat(value)
        return dt.isoformat(timespec="microseconds")
    except ValueError:
        pass

    # Human patterns: "2 hours ago", "30 minutes ago", "1 day ago"
    m = re.match(r"^(\d+)\s*(second|minute|hour|day|week|month)s?\s*ago$", value)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        deltas = {
            "second": datetime.timedelta(seconds=n),
            "minute": datetime.timedelta(minutes=n),
            "hour": datetime.timedelta(hours=n),
            "day": datetime.timedelta(days=n),
            "week": datetime.timedelta(weeks=n),
            "month": datetime.timedelta(days=n * 30),
        }
        dt = datetime.datetime.utcnow() - deltas[unit]
        return dt.isoformat(timespec="microseconds")

    return None
