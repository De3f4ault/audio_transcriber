r"""Dot-commands for the AudioBench REPL.

Dot-commands (prefixed with `.`) are the semantic layer.

Registration is side-effect-based: importing this module populates
_DOT_HANDLERS via @register_dot.
"""

from __future__ import annotations

from audiobench.cli.display.theme import (
    ACCENT,
    BOLD,
    DIM,
    SUCCESS,
    WARNING,
    console,
)
from audiobench.cli.repl.session import ReplSession

# ── Registry ─────────────────────────────────────────────────

_DOT_HANDLERS: dict[str, callable] = {}


def register_dot(name: str):
    """Decorator that registers a function as a dot-command handler."""
    def decorator(fn):
        _DOT_HANDLERS[name.lower()] = fn
        return fn
    return decorator


def dispatch_dot(session: ReplSession, line: str) -> bool:
    """Dispatch a dot-command (without the leading `.`).

    Returns True if handled, False if unknown.
    """
    parts = line.strip().split(None, 1)
    if not parts:
        _print_dot_help()
        return True

    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    handler = _DOT_HANDLERS.get(cmd)
    if handler is None:
        console.print(f"  [{WARNING}]Unknown dot-command: .{cmd}[/]")
        console.print(f"  [{DIM}]Type [{ACCENT}].help[/] for available dot-commands.[/]")
        return False

    try:
        handler(session, args)
        return True
    except Exception as e:
        console.print(f"  [{WARNING}]Error in .{cmd}: {e}[/]")
        return False


def _print_dot_help() -> None:
    console.print(f"\n  [{BOLD}]Dot-commands[/]")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print(f"  [{ACCENT}].use[/] [{DIM}]<id>[/]         Set context to a transcript by ID")
    console.print(f"  [{ACCENT}].focus[/] [{DIM}]<id>[/]       Set focus to an audio file by ID")
    console.print(f"  [{ACCENT}].help[/]              This help\n")


# ── .help ─────────────────────────────────────────────────────

@register_dot("help")
def dot_help(session: ReplSession, args: str) -> None:
    """Show available dot-commands."""
    _print_dot_help()

@register_dot("use")
def dot_use(session: ReplSession, args: str) -> None:
    """Set the current context to a specific transcript ID."""
    from audiobench.core.focused_entity import FocusedEntity
    from audiobench.storage.repository import TranscriptionRepository
    from audiobench.cli.repl.dispatch import print_context_summary
    
    arg = args.strip()
    if not arg.isdigit():
        console.print(f"  [{WARNING}]Usage: .use <transcript_id>[/]")
        return
        
    tx_id = int(arg)
    repo = TranscriptionRepository()
    rec = repo.get_by_id(tx_id)
    if not rec:
        console.print(f"  [{WARNING}]Transcript #{tx_id} not found.[/]")
        return
        
    session.set_context(tx_id)
    
    # Also focus the associated audio file if possible
    audio_file_id = rec.get("audio_file_id")
    if audio_file_id:
        file_name = rec.get("file_name", f"File #{audio_file_id}")
        session.focus = FocusedEntity(type="file", id=audio_file_id, label=file_name)
        
    print_context_summary(session)

@register_dot("focus")
def dot_focus(session: ReplSession, args: str) -> None:
    """Alias for \\focus."""
    from audiobench.cli.repl.backslash_commands import cmd_focus
    cmd_focus(session, args)
