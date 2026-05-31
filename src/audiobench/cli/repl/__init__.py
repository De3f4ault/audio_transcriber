"""REPL — context-aware interactive shell for AudioBench.

The REPL is the nerve center of AudioBench. It provides:
  - Context-aware ID injection: `show` auto-uses current transcript
  - Dot-commands for quick actions on the active transcript
  - Auto-context: transcribing sets context automatically
  - Onboarding: shows recent history + tips on first launch
  - Full command dispatch: every CLI command works inside the REPL
  - Shell escape, tab completion, persistent history
  - .play, .edit, .find, .path, .open — deep system integration

Package modules:
  - session: ReplSession state management
  - dispatch: Command dispatch + context summary
  - dot_commands: .stats, .show, .play, etc.
  - slash_commands: /help, /exit, /commands
  - completion: Tab completion setup
  - banner: Banner, onboarding, help text, goodbye

Usage:
    $ audiobench repl
    $ audiobench repl 42     ← start with transcript #42 as context
"""

from __future__ import annotations

import difflib
import shlex
import subprocess

import click

from audiobench.cli.display.theme import ACCENT, DIM, WARNING, console, error_panel
from audiobench.cli.repl.banner import print_banner, print_goodbye, print_onboarding
from audiobench.cli.repl.completion import setup_completion
from audiobench.cli.repl.dispatch import dispatch_command, print_context_summary
from audiobench.cli.repl.dot_commands import handle_dot_command
from audiobench.cli.repl.session import ReplSession
from audiobench.cli.repl.slash_commands import ALIASES, handle_slash_command


@click.command("repl")
@click.argument("transcript_id", required=False, type=int, default=None)
def repl(transcript_id: int | None) -> None:
    """Start the interactive AudioBench shell.

    \\b
    Start fresh:
      audiobench repl

    \\b
    Start with a transcript loaded:
      audiobench repl 42

    \\b
    Inside the REPL, every command works without 'audiobench' prefix.
    Context-aware commands auto-fill the transcript ID.
    Type 'help' for the full guide.
    """
    ctx = click.get_current_context()
    cli_group = ctx.parent.command if ctx.parent else None

    if cli_group is None or not isinstance(cli_group, click.Group):
        console.print(error_panel("REPL Error", "Cannot find parent CLI group"))
        return

    session = ReplSession(cli_group)
    session.load_history()
    session._load_history_ids()
    setup_completion(session)
    print_banner(session)

    # Pre-load context if transcript ID given
    if transcript_id is not None:
        session.set_context(transcript_id)
        if session.context_record:
            print_context_summary(session)
            console.print()

    # Onboarding: show recent transcripts
    print_onboarding(session)

    # ── Input loop ──
    while True:
        try:
            user_input = input(session.prompt).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            session.save_history()
            print_goodbye(session)
            break

        if not user_input:
            continue

        # ── AI shorthand: ? what are the key points? ──
        if user_input.startswith("?"):
            question = user_input[1:].strip()
            if not question:
                console.print(f"  [{DIM}]Usage: ? what are the key points?[/]")
                continue

            tx_id = session.last_id
            if tx_id is None:
                # Offer to use the most recent transcript
                try:
                    repo = session._get_repo()
                    records = repo.get_history(limit=1)
                    if records:
                        recent = records[0]
                        recent_id = recent["id"]
                        recent_name = recent.get("file_name", f"#{recent_id}")
                        console.print(
                            f"  [{WARNING}]No context set.[/] "
                            f"Use most recent [{ACCENT}]#{recent_id}[/] ({recent_name})? "
                            f"[{DIM}][y/N][/]"
                        )
                        confirm = input("  → ").strip().lower()
                        if confirm in ("y", "yes"):
                            session.set_context(recent_id)
                            tx_id = session.last_id
                        else:
                            console.print(f"  [{DIM}]Set context with[/] [{ACCENT}]work <file>[/] [{DIM}]or[/] [{ACCENT}].use <ID>[/]")
                            continue
                    else:
                        console.print(f"  [{DIM}]No transcriptions yet. Transcribe a file first.[/]")
                        continue
                except Exception:
                    console.print(f"  [{WARNING}]Set context first ([/][{ACCENT}]work <file>[/][{WARNING}]) to use ? shorthand[/]")
                    continue

            if tx_id is not None:
                dispatch_command(session, ["ask", str(tx_id), question])
            continue

        # Shell escape: !command
        if user_input.startswith("!"):
            shell_cmd = user_input[1:].strip()
            if shell_cmd:
                try:
                    subprocess.run(shell_cmd, shell=True)
                except Exception as e:
                    console.print(f"  [{WARNING}]{e}[/]")
            continue

        # Dot commands: .stats, .show, etc.
        if user_input.startswith("."):
            handle_dot_command(user_input, session)
            continue

        # Slash commands: /help, /exit, etc.
        if user_input.startswith("/"):
            should_exit = handle_slash_command(user_input, session)
            if should_exit:
                session.save_history()
                print_goodbye(session)
                break
            continue

        # Bare aliases: help, exit, quit, clear, commands
        if user_input.lower() in ALIASES:
            mapped = ALIASES[user_input.lower()]
            if mapped == "/exit":
                session.save_history()
                print_goodbye(session)
                break
            handle_slash_command(mapped, session)
            continue

        # Parse command line
        try:
            args = shlex.split(user_input)
        except ValueError as e:
            console.print(f"  [{WARNING}]Parse error: {e}[/]")
            continue

        if not args:
            continue

        # Check for background execution syntax '&'
        if args[-1] == "&":
            from audiobench.core.platform import SUPPORTS_BACKGROUND_JOBS
            if not SUPPORTS_BACKGROUND_JOBS:
                console.print(f"  [{WARNING}]Background jobs are only supported on Linux/macOS.[/]")
                continue

            from audiobench.jobs.runner import submit_job
            
            args = args[:-1]
            if not args:
                continue
            
            # Auto-expand vars before submitting so context is resolved
            args = session.expand_vars(args)
            args = session.auto_inject_id(args)
            
            job_id = submit_job(args)
            cmd_str = " ".join(args)
            if len(cmd_str) > 40:
                cmd_str = cmd_str[:37] + "..."
            console.print(f"  [{ACCENT}][{job_id}][/] Job submitted — {cmd_str}")
            continue

        # Strip 'audiobench' prefix if user types it out of habit
        cmd_name = args[0]
        if cmd_name == "audiobench" and len(args) > 1:
            args = args[1:]
            cmd_name = args[0]

        # Block nested REPL
        if cmd_name == "repl":
            console.print(f"  [{DIM}]Already in REPL. Type help for usage or /exit to quit.[/]")
            continue

        # File focus command
        if cmd_name == "work":
            if len(args) < 2:
                console.print(f"  [{WARNING}]Usage: work <file_path>[/]")
                continue
            file_path = args[1]
            from audiobench.storage.repository import TranscriptionRepository
            from audiobench.core.focused_entity import FocusedEntity
            from pathlib import Path
            
            repo = TranscriptionRepository()
            file_id = repo.get_or_create_file(file_path)
            
            session.focus = FocusedEntity(
                type="file",
                id=file_id,
                label=Path(file_path).name
            )
            continue

        # Check if it's a known command
        if cmd_name not in session.cli_group.commands:
            close = difflib.get_close_matches(
                cmd_name,
                list(session.cli_group.commands.keys()),
                n=1,
                cutoff=0.5,
            )
            suggestion = f"  Did you mean: [{ACCENT}]{close[0]}[/]?" if close else ""
            console.print(
                f"  [{WARNING}]Unknown command:[/] {cmd_name}\n"
                f"  [{DIM}]Type[/] [{ACCENT}]help[/] "
                f"[{DIM}]for usage or[/] "
                f"[{ACCENT}]/commands[/] "
                f"[{DIM}]to see all commands[/]"
            )
            if suggestion:
                console.print(suggestion)
            continue

        dispatch_command(session, args)
