"""Command registration — imports and attaches all commands to the CLI group."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import click


def register_all(cli: click.Group) -> None:
    """Register all command modules with the CLI group.

    Each module exports one or more Click commands that get
    added via cli.add_command(). The order here determines
    the order in --help output.
    """
    from cli.commands.transcribe import subtitle, transcribe
    from cli.commands.listen import listen
    from cli.commands.speak import download_voice, speak
    from cli.commands.summarize import summarize
    from cli.commands.chat import ask, chat
    from cli.commands.history import delete, export, history, search
    from cli.commands.system import download, info

    # ── Core workflow ──
    cli.add_command(transcribe)
    cli.add_command(subtitle)
    cli.add_command(listen)
    cli.add_command(speak)
    cli.add_command(download_voice, "download-voice")

    # ── AI ──
    cli.add_command(summarize)
    cli.add_command(ask)
    cli.add_command(chat)

    # ── Data management ──
    cli.add_command(history)
    cli.add_command(search)
    cli.add_command(export)
    cli.add_command(delete)

    # ── System ──
    cli.add_command(download)
    cli.add_command(info)
