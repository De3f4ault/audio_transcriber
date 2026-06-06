"""Backslash meta-commands for the AudioBench REPL.

These are local, instant commands that run against the database without
requiring any AI or external service.

Registration is side-effect-based: importing this module populates the
_BACKSLASH_HANDLERS registry in dispatch.py via @register_backslash.

To add a new \\command, decorate a function here:

    @register_backslash("mycommand")
    def cmd_mycommand(session: ReplSession, args: str) -> None:
        ...
"""

from __future__ import annotations

from audiobench.cli.display.theme import (
    ACCENT,
    BOLD,
    DIM,
    SUCCESS,
    WARNING,
    console,
    format_duration,
    make_table,
)
from audiobench.cli.repl.dispatch import register_backslash, register_resume
from audiobench.cli.repl.session import ReplSession


# ── Navigation ───────────────────────────────────────────────


@register_backslash("back")
def cmd_back(session: ReplSession, args: str) -> None:
    """Return to the previous navigation context."""
    frame = session.pop_frame()
    if frame is None:
        console.print(f"  [{DIM}]Already at top level.[/]")
        return

    console.print(f"  [{SUCCESS}]← Back to {frame.context}[/]")

    # Try to resume via registry if a handler exists
    from audiobench.cli.repl.dispatch import _RESUME_HANDLERS
    if frame.context in _RESUME_HANDLERS:
        handler = _RESUME_HANDLERS[frame.context]
        handler(session, frame)


@register_backslash("home")
def cmd_home(session: ReplSession, args: str) -> None:
    """Clear the entire navigation stack and return to the top level."""
    count = len(session.navigation_stack)
    if count == 0:
        console.print(f"  [{DIM}]Already at top level.[/]")
        return
    session.navigation_stack.clear()
    session._persist_stack()
    console.print(f"  [{SUCCESS}]↑ Cleared {count} level(s). Back at top.[/]")


@register_backslash("stack")
def cmd_stack(session: ReplSession, args: str) -> None:
    """Show the current navigation stack depth and contents."""
    if not session.navigation_stack:
        console.print(f"  [{DIM}]Stack is empty — you are at the top level.[/]")
        return

    depth = len(session.navigation_stack)
    console.print(f"\n  [{BOLD}]Navigation stack[/] [{DIM}]({depth} level(s))[/]")
    console.print(f"  [{DIM}]{'─' * 40}[/]")
    for i, frame in enumerate(session.navigation_stack, 1):
        resumed_tag = f" [{DIM}][resumed][/]" if frame.resumed else ""
        state_hint = ""
        if frame.state:
            keys = list(frame.state.keys())[:3]
            state_hint = f"  [{DIM}]{{{', '.join(keys)}}}[/]"
        intent_str = f" — {frame.intent}" if frame.intent else ""
        console.print(f"    {i}. [{ACCENT}]{frame.context}[/]{intent_str}{state_hint}{resumed_tag}")
    console.print()


# ── Focus ────────────────────────────────────────────────────


@register_backslash("focus")
def cmd_focus(session: ReplSession, args: str) -> None:
    """Focus on an audio file by ID, or show the current focus."""
    from audiobench.cli.repl.dispatch import print_context_summary

    if not args.strip():
        if session.focus:
            print_context_summary(session)
        else:
            console.print(
                f"  [{DIM}]No focus set.[/]  "
                f"Use [{ACCENT}]\\focus <id>[/] or [{ACCENT}]\\ls[/] to pick a file."
            )
        return

    arg = args.strip()
    if not arg.isdigit():
        console.print(f"  [{WARNING}]Usage: \\focus <audio_file_id>[/]")
        return

    file_id = int(arg)
    try:
        from audiobench.core.db_engine import init_db
        from audiobench.core.focused_entity import FocusedEntity
        from audiobench.storage.repository import TranscriptionRepository

        init_db()
        repo = TranscriptionRepository()
        audio_file = repo.get_audio_file(file_id)
        if not audio_file:
            console.print(f"  [{WARNING}]Audio file #{file_id} not found.[/]")
            return

        session.focus = FocusedEntity(
            type="file",
            id=file_id,
            label=audio_file["file_name"],
        )
        print_context_summary(session)
    except Exception as e:
        console.print(f"  [{WARNING}]Could not focus on #{file_id}: {e}[/]")


@register_backslash("unfocus")
def cmd_unfocus(session: ReplSession, args: str) -> None:
    """Clear the current focus and return to the bare prompt."""
    if not session.focus:
        console.print(f"  [{DIM}]No focus to clear.[/]")
        return
    label = session.focus.display_label
    session.focus = None
    console.print(f"  [{DIM}]Focus cleared ({label}).[/]")


# ── View ─────────────────────────────────────────────────────


@register_backslash("ls")
def cmd_ls(session: ReplSession, args: str) -> None:
    """List recent audio files. Optionally pass a count: \\ls 20"""
    from audiobench.core.db_engine import init_db

    try:
        init_db()
        repo = session._get_repo()
        limit = int(args.strip()) if args.strip().isdigit() else 10
        records = repo.get_history(limit=limit)

        if not records:
            console.print(f"  [{DIM}]No transcriptions yet.[/]")
            return

        table = make_table(
            f"Library — {len(records)} recent",
            [
                ("#", {"style": ACCENT, "justify": "right", "width": 5}),
                ("File", {"no_wrap": True}),
                ("Words", {"justify": "right", "width": 8}),
                ("Duration", {"justify": "right", "width": 10}),
                ("Model", {"width": 14}),
            ],
        )
        for rec in records:
            dur = format_duration(rec.get("duration") or 0)
            table.add_row(
                str(rec["id"]),
                rec.get("file_name", "?"),
                f"{rec.get('word_count', 0):,}",
                dur,
                rec.get("model", "?") or "?",
            )
        console.print(table)
        console.print(
            f"  [{DIM}]Use [{ACCENT}]\\focus <id>[/] to work on a file.[/]"
        )
    except Exception as e:
        console.print(f"  [{WARNING}]Could not list files: {e}[/]")


@register_backslash("show")
def cmd_show(session: ReplSession, args: str) -> None:
    """Show a transcript by ID, or the currently focused one."""
    from audiobench.cli.repl.dispatch import dispatch_command

    if args.strip().isdigit():
        dispatch_command(session, ["show", args.strip()])
    elif session.last_id:
        dispatch_command(session, ["show", str(session.last_id)])
    else:
        console.print(
            f"  [{DIM}]Usage: [{ACCENT}]\\show <id>[/] or focus on a file first.[/]"
        )


@register_backslash("jobs")
def cmd_jobs(session: ReplSession, args: str) -> None:
    """List background jobs and their status."""
    from audiobench.cli.repl.dispatch import dispatch_command

    dispatch_command(session, ["jobs"])


@register_backslash("search")
def cmd_search(session: ReplSession, args: str) -> None:
    """Full-text search transcripts — SQLite FTS5, instant, exact text match.

    Usage:  \\search "quarterly budget"
            \\search meeting --limit 20

    For meaning-based cross-source search, use: .search "concept"
    """
    import shlex
    from audiobench.core.db_engine import init_db
    from audiobench.core.focused_entity import FocusedEntity
    from audiobench.cli.repl.session import NavigationFrame
    from audiobench.cli.repl.dispatch import print_context_summary
    from audiobench.storage.repository import TranscriptionRepository

    if not args.strip():
        console.print(f"  [{DIM}]Usage: [{ACCENT}]\\search \"query\"[/][/]")
        console.print(f"  [{DIM}]For semantic search across all sources: [{ACCENT}].search \"concept\"[/][/]")
        return

    # Parse: first token is query, optional --limit N
    try:
        parts = shlex.split(args)
    except ValueError:
        parts = args.split()

    query = parts[0]
    limit = 10

    i = 1
    while i < len(parts):
        if parts[i] == "--limit" and i + 1 < len(parts):
            try:
                limit = int(parts[i + 1])
            except ValueError:
                pass
            i += 2
        else:
            i += 1

    init_db()
    repo = TranscriptionRepository()
    results = repo.search(query, limit=limit)

    if not results:
        console.print(f"  [{DIM}]No results for \"{query}\"[/]")
        console.print(f"  [{DIM}]Try [{ACCENT}].search \"{query}\"[/] for semantic search.[/]")
        return

    # Display results
    console.print(f"\n  [{BOLD}]Text Search — SQLite FTS5[/]  [{DIM}]\"{query}\"[/]")
    console.print(f"  [{DIM}]{'─' * 50}[/]")
    for i, r in enumerate(results, 1):
        file_name = r.get("file_name", f"Transcript #{r.get('id', '?')}")
        tx_id = r.get("id", "?")
        snippet = (r.get("text_preview") or "").strip().replace("\n", " ")
        if len(snippet) > 110:
            snippet = snippet[:107] + "..."
        console.print(f"  [{ACCENT}][{i}][/] [{ACCENT}]◎[/] {file_name}  [{DIM}](#{tx_id})[/]")
        if snippet:
            console.print(f"       [{DIM}]\"{snippet}\"[/]")
        console.print()
    console.print(f"  [{DIM}]{'─' * 50}[/]")

    # Interactive selection
    while True:
        console.print(
            f"  [{DIM}]Select to focus [1-{len(results)}], "
            f"[{ACCENT}]n <num>[/] to capture, "
            f"[{ACCENT}]q[/] to quit, or [{ACCENT}].search \"{query}\"[/] for semantic results[/]"
        )
        try:
            choice = input("  → ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return

        if choice in ("q", "quit", ""):
            return

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(results):
                r = results[idx]
                # Search returns transcript records — focus on the audio file
                audio_file_id = r.get("audio_file_id")
                tx_id = r.get("id")
                file_name = r.get("file_name", f"#{tx_id}")

                if audio_file_id:
                    session.push_frame(
                        NavigationFrame(context="\\search", state={}, intent=f'focused from search "{query}"')
                    )
                    session.focus = FocusedEntity(type="file", id=audio_file_id, label=file_name)
                    print_context_summary(session)
                    return
                elif tx_id:
                    session.set_context(tx_id)
                    print_context_summary(session)
                    return
            else:
                console.print(f"  [{WARNING}]Out of range. Enter 1–{len(results)}.[/]")
        elif choice.startswith("n ") and choice[2:].isdigit():
            idx = int(choice[2:]) - 1
            if 0 <= idx < len(results):
                r = results[idx]
                tx_id = r.get("id")
                if tx_id:
                    from audiobench.core.db_session import get_session as _gs
                    from audiobench.storage.models import ExpressionRecord
                    from audiobench.memory.enums import SourceType
                    with _gs() as db:
                        expr = db.query(ExpressionRecord).filter_by(
                            source_id=tx_id,
                            source_type=SourceType.AUDIO_TRANSCRIPT.value
                        ).first()
                    if expr:
                        cmd_capture(session, f"expression:{expr.id}")
                        return
                    else:
                        console.print(f"  [{WARNING}]Could not find expression for transcript #{tx_id}[/]")
            else:
                console.print(f"  [{WARNING}]Out of range. Enter 1–{len(results)}.[/]")
        else:
            console.print(f"  [{WARNING}]Enter a number, n <num>, or q.[/]")


@register_backslash("history")
def cmd_history(session: ReplSession, args: str) -> None:
    """Show transcript history. Usage: \\history [--limit 5]"""
    from audiobench.cli.repl.dispatch import dispatch_command
    import shlex

    parsed_args = shlex.split(args)
    dispatch_command(session, ["history"] + parsed_args)


@register_backslash("stats")
def cmd_stats(session: ReplSession, args: str) -> None:
    """Show detailed stats for the current transcript."""
    from audiobench.cli.display.theme import ACCENT, BOLD, DIM, console
    from audiobench.cli.display.theme import format_duration
    
    if not session.last_id:
        console.print(f"  [{DIM}]No focused transcript. Use [{ACCENT}]\\focus <id>[/][/]")
        return
        
    repo = session._get_repo()
    rec = repo.get_by_id(session.last_id)
    if not rec:
        return

    console.print(f"\n  [{BOLD}]Transcription #{rec['id']}[/]")
    console.print(f"  [{DIM}]{'─' * 40}[/]")
    console.print(f"  [{DIM}]File:[/]       [{ACCENT}]{rec.get('file_name', 'Unknown')}[/]")
    
    duration = format_duration(rec.get("duration_seconds") or 0)
    console.print(f"  [{DIM}]Duration:[/]   {duration}")
    console.print(f"  [{DIM}]Words:[/]      {rec.get('word_count', 0):,}")
    
    metadata = rec.get("metadata") or {}
    console.print(f"  [{DIM}]Model:[/]      {metadata.get('model', 'Unknown')}")
    console.print(f"  [{DIM}]Engine:[/]     {metadata.get('engine', 'Unknown')}")
    console.print(f"  [{DIM}]Preset:[/]     {metadata.get('speed_preset', 'balanced')}")
    console.print()


@register_backslash("chat")
def cmd_chat(session: ReplSession, args: str) -> None:
    """Start an interactive chat with the current transcript."""
    from audiobench.cli.repl.dispatch import dispatch_command
    
    if not session.last_id:
        console.print(f"  [{DIM}]No focused transcript. Use [{ACCENT}]\\focus <id>[/][/]")
        return
        
    dispatch_command(session, ["chat", str(session.last_id)])


@register_backslash("ask")
def cmd_ask(session: ReplSession, args: str) -> None:
    """Ask a question about the current transcript."""
    from audiobench.cli.repl.dispatch import dispatch_command
    
    if not session.last_id:
        console.print(f"  [{DIM}]No focused transcript. Use [{ACCENT}]\\focus <id>[/][/]")
        return
        
    if not args.strip():
        dispatch_command(session, ["ask", str(session.last_id), "--interactive"])
    else:
        dispatch_command(session, ["ask", str(session.last_id), args.strip()])


@register_backslash("summarize")
def cmd_summarize(session: ReplSession, args: str) -> None:
    """Generate an AI summary of the current transcript."""
    from audiobench.cli.repl.dispatch import dispatch_command
    
    if not session.last_id:
        console.print(f"  [{DIM}]No focused transcript. Use [{ACCENT}]\\focus <id>[/][/]")
        return
        
    if not args.strip():
        dispatch_command(session, ["summarize", str(session.last_id), "--interactive"])
    else:
        dispatch_command(session, ["summarize", str(session.last_id)])

@register_backslash("import")
def cmd_import(session: ReplSession, args: str) -> None:
    """Import audio files into the internal library."""
    from audiobench.cli.repl.session import NavigationFrame
    session.push_frame(NavigationFrame(context="import", state={}, intent="import files"))
    
    from audiobench.cli.commands.import_cmd import run_import_flow
    run_import_flow(session=session)

@register_backslash("transcribe")
def cmd_transcribe(session: ReplSession, args: str) -> None:
    """Transcribe audio files (interactive wizard by default)."""
    from audiobench.cli.repl.session import NavigationFrame
    from audiobench.cli.repl.dispatch import dispatch_command
    import shlex
    
    session.push_frame(NavigationFrame(context="transcribe", state={}, intent="transcribe audio"))
    
    if not args.strip():
        dispatch_command(session, ["transcribe", "--interactive"])
    else:
        parsed_args = shlex.split(args)
        dispatch_command(session, ["transcribe"] + parsed_args)

@register_backslash("config")
def cmd_config(session: ReplSession, args: str) -> None:
    """Run configuration wizard."""
    from audiobench.cli.repl.session import NavigationFrame
    from audiobench.cli.repl.dispatch import dispatch_command
    session.push_frame(NavigationFrame(context="config", state={}, intent="configure app"))
    dispatch_command(session, ["config", "--interactive"])

# ── Notes & Capture ──────────────────────────────────────────


@register_backslash("note")
def cmd_note(session: ReplSession, args: str) -> None:
    """Manage and edit notes.
    
    Usage:
      \\note              → open new note (context-linked if focus set)
      \\note "Title"      → open new note with title
      \\note 5            → reopen note #5
      \\note --list       → list active notes
      \\note --inbox      → open global inbox
      \\note --capture-log→ show unprocessed captures
    """
    from audiobench.cli.display.theme import console, ACCENT, WARNING, SUCCESS, DIM, BOLD
    from audiobench.storage.note_repository import NoteRepository
    import shlex
    import os
    import tempfile
    import subprocess
    
    repo = NoteRepository()
    arg = args.strip()
    
    if arg == "--list":
        notes = repo.list_notes()
        if not notes:
            console.print(f"  [{DIM}]No active notes found.[/]")
            return
        from audiobench.cli.display.theme import make_table
        table = make_table("Active Notes", [("ID", {"style": ACCENT, "justify": "right"}), ("Title", {}), ("Created", {})])
        for n in notes:
            table.add_row(str(n.id), n.title, str(n.created_at)[:16])
        console.print(table)
        return
        
    if arg == "--inbox":
        note = repo.find_or_create_inbox()
    elif arg == "--capture-log":
        captures = repo.list_unprocessed_captures()
        if not captures:
            console.print(f"  [{DIM}]No captures found.[/]")
            return
        console.print(f"\n  [{BOLD}]Capture Log[/]")
        for c in captures:
            console.print(f"  [{DIM}]{c['timestamp']}[/] [{ACCENT}]note:{c['note_id']}[/] {c['text']}")
        return
    elif arg.isdigit():
        note = repo.get_by_id(int(arg))
        if not note:
            console.print(f"  [{WARNING}]Note #{arg} not found.[/]")
            return
    else:
        # Create new
        title = arg.strip('"\'') if arg else ("Untitled Note" if not session.focus else f"Notes on {session.focus.label}")
        audio_file_id = session.focus.id if session.focus and session.focus.type == "file" else None
        note = repo.create(title=title, audio_file_id=audio_file_id)
        if session.focus:
            console.print(f"  [{DIM}]Opening note linked to {session.focus.label}...[/]")
            
    # Open in Editor
    editor = os.environ.get("EDITOR", "nano")
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".md",
        prefix=f"audiobench_note_{note.id}_",
        delete=False
    ) as f:
        f.write(note.body or "")
        tmp_path = f.name
        
    try:
        subprocess.run([editor, tmp_path])
        with open(tmp_path, "r") as f:
            new_body = f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            
    if new_body != note.body:
        repo.save_body(note.id, new_body)
        console.print(f"  [{SUCCESS}]Note #{note.id} saved. Searchable via .search[/]")
        
        # Post-save ambient retrieval
        from audiobench.daemon.factory import get_daemon_client
        daemon = get_daemon_client()
        if daemon.is_alive() and new_body.strip():
            console.print(f"  [{DIM}]Fetching related ideas...[/]")
            results = daemon.search(query=new_body[-500:], top_k=5, use_bm25=False, use_dense=True, use_colbert=True)
            if results:
                from audiobench.storage.expression_repository import ExpressionRepository
                expr_repo = ExpressionRepository()
                for r in results:
                    expr = expr_repo.get_by_id(r["expression_id"])
                    if expr and expr.source_id != note.id:
                        snippet = expr.content[:80].replace('\\n', ' ')
                        console.print(f"    [{DIM}]~[/] {snippet}...")
    else:
        console.print(f"  [{DIM}]No changes made.[/]")


def _resolve_capture_destination(session: ReplSession, repo) -> int:
    """Tier 1: active note, Tier 2: context note, Tier 3: inbox"""
    # Tier 1
    active_id = session.navigation_stack[-1].state.get("note_id") if session.navigation_stack else None
    if active_id:
        return active_id
    # Tier 2
    if session.focus and session.focus.type == "file":
        note = repo.find_or_create_context_note(session.focus.id, session.focus.label)
        return note.id
    # Tier 3
    inbox = repo.find_or_create_inbox()
    return inbox.id


@register_backslash("capture")
def cmd_capture(session: ReplSession, args: str) -> None:
    """Capture a thought or expression to the current context note or inbox."""
    from audiobench.cli.display.theme import console, ACCENT, WARNING, SUCCESS, DIM
    from audiobench.storage.note_repository import NoteRepository
    
    arg = args.strip()
    if not arg:
        try:
            arg = input("  Capture text > ").strip()
            if not arg: return
        except (EOFError, KeyboardInterrupt):
            return

    repo = NoteRepository()
    note_id = _resolve_capture_destination(session, repo)
    
    expression_id = None
    if arg.startswith("expression:"):
        from audiobench.storage.expression_repository import ExpressionRepository
        expr_repo = ExpressionRepository()
        eid_str = arg.replace("expression:", "").strip()
        if eid_str.isdigit():
            expression_id = int(eid_str)
            expr = expr_repo.get_by_id(expression_id)
            if expr:
                arg = f"Captured expression: {expr.content[:100]}..."
                
    repo.append_capture(note_id, arg, expression_id)
    dest_note = repo.get_by_id(note_id)
    console.print(f"  [{SUCCESS}]→ Captured to: {dest_note.title} (#note:{dest_note.id})[/]")


# ── Help ─────────────────────────────────────────────────────


@register_backslash("help")
def cmd_help(session: ReplSession, args: str) -> None:
    """Show available backslash commands."""
    console.print(f"\n  [{BOLD}]AudioBench Shell[/]\n")
    console.print(f"  [{DIM}]Three execution contexts, one prompt:[/]\n")

    console.print(f"  [{ACCENT}]\\[/][{DIM}]command[/]   — local, instant (SQLite)")
    console.print(f"  [{DIM}]?[/] question  — AI/semantic (requires focus)")
    console.print(f"  [yellow]![/][{DIM}]command[/]   — OS shell passthrough")
    console.print(f"  [{DIM}]bare text  — implicit ask (requires focus)[/]\n")

    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print(f"  [{BOLD}]Navigation[/]")
    console.print(f"    [{ACCENT}]\\back[/]              Return to previous context")
    console.print(f"    [{ACCENT}]\\home[/]              Clear stack, return to top")
    console.print(f"    [{ACCENT}]\\stack[/]             Show navigation stack\n")

    console.print(f"  [{BOLD}]Focus & View[/]")
    console.print(f"    [{ACCENT}]\\focus <id>[/]        Focus on audio file by ID")
    console.print(f"    [{ACCENT}]\\unfocus[/]           Clear current focus")
    console.print(f"    [{ACCENT}]\\ls [n][/]            List recent files (default 10)")
    console.print(f"    [{ACCENT}]\\show [id][/]         Show transcript\n")

    console.print(f"  [{BOLD}]Workflows[/]")
    console.print(f"    [{ACCENT}]\\workflow save <name>[/]   Capture last N commands")
    console.print(f"    [{ACCENT}]\\workflow list[/]          Show all saved workflows")
    console.print(f"    [{ACCENT}]\\workflow run <name>[/]    Replay a workflow")
    console.print(f"    [{ACCENT}]\\workflow delete <name>[/] Delete a workflow\n")

    console.print(f"  [{BOLD}]Intelligence[/]")
    console.print(f"    [{ACCENT}]\\related[/]           Show related fragments from other files")
    console.print(f"    [{ACCENT}]\\name[/]              Rename a speaker globally (e.g. \\name \"Speaker 1\" \"John\")")
    console.print(f"    [{ACCENT}]\\graph[/]             Command frequency stats")
    console.print(f"    [{ACCENT}]\\suggest[/]           What to do next (based on history)\n")

    console.print(f"  [{BOLD}]Interactive Flows[/]")
    console.print(f"    [{ACCENT}]\\library[/]  \\import[/]  \\transcribe[/]  \\chat[/]  \\ask[/]  \\summarize[/]\n")
    console.print(f"  [{DIM}]Shell passthrough: [/]  !ls  !ffmpeg  !play  ...\n")

# ── Workflow ──────────────────────────────────────────────────

@register_backslash("workflow")
def cmd_workflow(session: ReplSession, args: str) -> None:
    """Manage named workflows. Usage: \\workflow save|list|run|delete [name]"""
    parts = args.strip().split(None, 1)
    subcmd = parts[0].lower() if parts else ""
    name = parts[1].strip() if len(parts) > 1 else ""

    if subcmd == "list" or not subcmd:
        _workflow_list(session)
    elif subcmd == "save":
        if not name:
            console.print(f"  [{WARNING}]Usage: [{ACCENT}]\\workflow save <name>[/][/]")
            return
        _workflow_save(session, name)
    elif subcmd == "run":
        if not name:
            console.print(f"  [{WARNING}]Usage: [{ACCENT}]\\workflow run <name>[/][/]")
            return
        _workflow_run(session, name)
    elif subcmd in ("delete", "del", "rm"):
        if not name:
            console.print(f"  [{WARNING}]Usage: [{ACCENT}]\\workflow delete <name>[/][/]")
            return
        _workflow_delete(session, name)
    elif subcmd == "show":
        _workflow_show(name) if name else console.print(
            f"  [{WARNING}]Usage: [{ACCENT}]\\workflow show <name>[/][/]"
        )
    else:
        console.print(f"  [{WARNING}]Unknown subcommand: {subcmd}[/]")
        console.print(f"  [{DIM}]Use: save | list | run | delete | show[/]")


def _workflow_list(session: ReplSession) -> None:
    try:
        from audiobench.storage.command_graph_repository import get_command_graph_repo
        repo = get_command_graph_repo()
        workflows = repo.list_workflows()
    except Exception as e:
        console.print(f"  [{WARNING}]Could not load workflows: {e}[/]")
        return

    if not workflows:
        console.print(
            f"  [{DIM}]No workflows yet. Run [{ACCENT}]\\workflow save <name>[/] "
            f"to capture the last few commands.[/]"
        )
        return

    table = make_table(
        "Saved Workflows",
        [
            ("Name", {"style": ACCENT}),
            ("Steps", {"width": 6, "justify": "right"}),
            ("Description", {}),
            ("Updated", {"width": 20}),
        ],
    )
    for wf in workflows:
        table.add_row(
            wf["name"],
            str(len(wf["steps"])),
            wf.get("description", ""),
            str(wf.get("updated_at", ""))[:19],
        )
    console.print(table)
    console.print(f"  [{DIM}]Run with [{ACCENT}]\\workflow run <name>[/][/]")


def _workflow_save(session: ReplSession, name: str) -> None:
    """Capture the last N commands from command_events as a named workflow."""
    try:
        from audiobench.storage.command_graph_repository import get_command_graph_repo
        repo = get_command_graph_repo()
        recent = repo.get_recent_commands(limit=10)
    except Exception as e:
        console.print(f"  [{WARNING}]Could not read command history: {e}[/]")
        return

    if not recent:
        console.print(f"  [{DIM}]No commands in history yet. Run some commands first.[/]")
        return

    import json as _json
    steps = [
        {"command": r["command"], "args": _json.loads(r.get("args_json", "[]"))}
        for r in reversed(recent)
    ]

    console.print(f"\n  [{BOLD}]Last {len(steps)} commands:[/]")
    for i, step in enumerate(steps, 1):
        args_str = " ".join(str(a) for a in step["args"]) if step["args"] else ""
        console.print(f"    [{ACCENT}]{i}.[/] {step['command']} [{DIM}]{args_str}[/]")

    try:
        keep_raw = input(
            f"\n  Keep all {len(steps)}? Enter indices to keep (e.g. 1,3,5) or Enter for all: "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        console.print(f"\n  [{DIM}]Cancelled.[/]")
        return

    if keep_raw:
        try:
            indices = [int(x.strip()) - 1 for x in keep_raw.split(",")]
            steps = [steps[i] for i in indices if 0 <= i < len(steps)]
        except (ValueError, IndexError):
            console.print(f"  [{WARNING}]Invalid indices — saving all steps.[/]")

    try:
        desc_raw = input("  Description (optional): ").strip()
    except (EOFError, KeyboardInterrupt):
        desc_raw = ""

    try:
        from audiobench.storage.command_graph_repository import get_command_graph_repo
        repo = get_command_graph_repo()
        repo.save_workflow(name, steps, description=desc_raw)
        console.print(f"\n  [{SUCCESS}]✓ Workflow [{ACCENT}]{name}[/] saved ({len(steps)} steps)[/]")
    except Exception as e:
        console.print(f"  [{WARNING}]Failed to save workflow: {e}[/]")


def _workflow_run(session: ReplSession, name: str) -> None:
    """Replay a named workflow step-by-step."""
    try:
        from audiobench.storage.command_graph_repository import get_command_graph_repo
        repo = get_command_graph_repo()
        wf = repo.get_workflow(name)
    except Exception as e:
        console.print(f"  [{WARNING}]Could not load workflow: {e}[/]")
        return

    if not wf:
        console.print(f"  [{WARNING}]Workflow [{ACCENT}]{name}[/] not found.[/]")
        return

    steps = wf["steps"]
    if not steps:
        console.print(f"  [{DIM}]Workflow is empty.[/]")
        return

    console.print(f"\n  [{BOLD}]Running workflow:[/] [{ACCENT}]{name}[/] ({len(steps)} steps)")
    console.print(f"  [{DIM}]{'─' * 40}[/]")

    from audiobench.cli.repl.dispatch import dispatch_command
    for i, step in enumerate(steps, 1):
        cmd = step["command"]
        args = step.get("args", [])
        full_args = [cmd] + [str(a) for a in args]
        args_display = " ".join(str(a) for a in args)
        console.print(f"  [{DIM}][{i}/{len(steps)}][/] [{ACCENT}]{cmd}[/] {args_display}")
        dispatch_command(session, full_args)

    console.print(f"\n  [{SUCCESS}]✓ Workflow [{ACCENT}]{name}[/] complete.[/]")


def _workflow_delete(session: ReplSession, name: str) -> None:
    try:
        from audiobench.storage.command_graph_repository import get_command_graph_repo
        repo = get_command_graph_repo()
        deleted = repo.delete_workflow(name)
    except Exception as e:
        console.print(f"  [{WARNING}]Could not delete: {e}[/]")
        return
    if deleted:
        console.print(f"  [{SUCCESS}]✓ Deleted workflow [{ACCENT}]{name}[/][/]")
    else:
        console.print(f"  [{WARNING}]Workflow [{ACCENT}]{name}[/] not found.[/]")


def _workflow_show(name: str) -> None:
    try:
        from audiobench.storage.command_graph_repository import get_command_graph_repo
        repo = get_command_graph_repo()
        wf = repo.get_workflow(name)
    except Exception as e:
        console.print(f"  [{WARNING}]Could not load workflow: {e}[/]")
        return
    if not wf:
        console.print(f"  [{WARNING}]Workflow [{ACCENT}]{name}[/] not found.[/]")
        return
    console.print(f"\n  [{BOLD}]Workflow:[/] [{ACCENT}]{name}[/]")
    if wf.get("description"):
        console.print(f"  [{DIM}]{wf['description']}[/]")
    console.print(f"  [{DIM}]{'─' * 40}[/]")
    for i, step in enumerate(wf["steps"], 1):
        args_str = " ".join(str(a) for a in step.get("args", []))
        console.print(f"    [{ACCENT}]{i}.[/] {step['command']} [{DIM}]{args_str}[/]")
    console.print()


# ── Intelligence ──────────────────────────────────────────────

@register_backslash("related")
def cmd_related(session: ReplSession, args: str) -> None:
    """Show semantically related fragments from other transcripts."""
    from audiobench.cli.display.theme import console, ACCENT, BOLD, DIM, WARNING
    if not session.last_id:
        console.print(f"  [{DIM}]No focused transcript. Use [{ACCENT}]\\focus <id>[/][/]")
        return
        
    repo = session._get_repo()
    rec = repo.get_by_id(session.last_id)
    if not rec:
        return
        
    # Get text to search. Summary if available, otherwise first 500 chars.
    text_to_search = rec.get("summary", "")
    if not text_to_search:
        text_to_search = rec.get("full_text", "")[:500]
        
    if not text_to_search.strip():
        console.print(f"  [{DIM}]Transcript is empty, cannot find related content.[/]")
        return
        
    console.print(f"  [{DIM}]Searching for related fragments across the library...[/]")
    
    from audiobench.daemon.factory import get_daemon_client
    daemon = get_daemon_client()
    
    try:
        # Ask daemon for search
        results = daemon.search(
            query=text_to_search,
            top_k=20,
            use_bm25=False,  # pure semantic relation
            use_dense=True,
            use_colbert=True,
        )
    except Exception as e:
        console.print(f"  [{WARNING}]Daemon search failed: {e}[/]")
        return
        
    if not results:
        console.print(f"  [{DIM}]No related fragments found.[/]")
        return
        
    from audiobench.storage.expression_repository import ExpressionRepository
    expr_repo = ExpressionRepository()
    
    # Filter out current source_id
    related_exprs = []
    seen_ids = set()
    for r in results:
        expr_id = r.get("expression_id")
        if expr_id:
            expr = expr_repo.get_by_id(expr_id)
            if expr and expr.source_id != session.last_id and expr.source_id not in seen_ids:
                seen_ids.add(expr.source_id)  # One fragment per related transcript
                related_exprs.append((expr, r.get("score", 0.0)))
                if len(related_exprs) >= 5:
                    break
                    
    if not related_exprs:
        console.print(f"  [{DIM}]No related fragments found outside of this transcript.[/]")
        return
        
    console.print(f"\n  [{BOLD}]Related Thoughts[/]")
    console.print(f"  [{DIM}]{'─' * 40}[/]")
    
    for i, (expr, score) in enumerate(related_exprs, 1):
        # get original audio file name
        audio_file = repo.get_by_id(expr.source_id)
        name = audio_file.get("file_name", f"Transcript #{expr.source_id}") if audio_file else f"Source #{expr.source_id}"
        
        console.print(f"  [{ACCENT}]{i}. {name}[/] [{DIM}](score: {score:.2f})[/]")
        content = expr.content.strip().replace('\n', ' ')
        if len(content) > 150:
            content = content[:147] + "..."
        console.print(f"     \"{content}\"")
    console.print()

@register_backslash("name")
def cmd_name(session: ReplSession, args: str) -> None:
    """Rename a speaker in the focused transcript and globally."""
    import shlex
    from audiobench.cli.display.theme import console, ACCENT, BOLD, DIM, WARNING
    
    if not session.last_id:
        console.print(f"  [{DIM}]No focused transcript. Use [{ACCENT}]\\focus <id>[/][/]")
        return
        
    try:
        parts = shlex.split(args)
    except ValueError as e:
        console.print(f"  [{WARNING}]Failed to parse arguments: {e}[/]")
        return
        
    if len(parts) != 2:
        console.print(f"  [{WARNING}]Usage: \\name \"Old Name\" \"New Name\"[/]")
        return
        
    old_name, new_name = parts[0], parts[1]
    repo = session._get_repo()
    rec = repo.get_by_id(session.last_id)
    if not rec:
        return
        
    # Update local segments
    from audiobench.core.db_session import get_session
    from audiobench.storage.models import SegmentRecord
    
    with get_session() as db:
        segments = db.query(SegmentRecord).filter(
            SegmentRecord.transcription_id == session.last_id,
            SegmentRecord.speaker == old_name
        ).all()
        
        if not segments:
            console.print(f"  [{WARNING}]Speaker '{old_name}' not found in this transcript.[/]")
            return
            
        for seg in segments:
            seg.speaker = new_name
            
        # Update speaker_map on the transcription record
        import json
        try:
            transcription = db.query(TranscriptionRecord).filter_by(id=session.last_id).first()
            if transcription and transcription.speaker_map:
                s_map = json.loads(transcription.speaker_map)
                # Reverse lookup the original pyannote label (e.g. SPEAKER_00)
                pyannote_labels = [k for k, v in s_map.items() if v == old_name]
                for p_label in pyannote_labels:
                    s_map[p_label] = new_name
                transcription.speaker_map = json.dumps(s_map)
        except Exception as e:
            console.print(f"  [{WARNING}]Failed to update speaker_map: {e}[/]")

        db.commit()
        
    # Update Global Speaker Profile
    try:
        from audiobench.memory.memory_store import SpeakerProfileStore
        profile_store = SpeakerProfileStore()
        
        # We need to find the profile with old_name.
        # But LanceDB search is vector-based. We can filter by name.
        results = profile_store.table.search().where(f"name = '{old_name}'").limit(1).to_list()
        if results:
            profile_id = results[0]["profile_id"]
            vector = results[0]["vector"]
            profile_store.save_speaker(profile_id, new_name, vector)
            console.print(f"  [{ACCENT}]Global speaker profile updated to '{new_name}'[/]")
        else:
            console.print(f"  [{DIM}]No global voice print found for '{old_name}' to rename.[/]")
    except Exception as e:
        console.print(f"  [{WARNING}]Failed to update global speaker profile: {e}[/]")

    console.print(f"  [{BOLD}]Renamed {len(segments)} segments to '{new_name}' in transcript #{session.last_id}[/]")


@register_backslash("graph")
def cmd_graph(session: ReplSession, args: str) -> None:
    """Show command frequency stats from the command graph."""
    try:
        from audiobench.storage.command_graph_repository import get_command_graph_repo
        repo = get_command_graph_repo()
        days = int(args.strip()) if args.strip().isdigit() else 30
        stats = repo.get_command_stats(days=days)
    except Exception as e:
        console.print(f"  [{WARNING}]Could not read command graph: {e}[/]")
        return

    if not stats:
        console.print(
            f"  [{DIM}]No command history yet. Run commands inside the REPL to build it.[/]"
        )
        return

    table = make_table(
        f"Command Frequency — last {days} days",
        [
            ("Command", {"style": ACCENT}),
            ("Count", {"width": 8, "justify": "right"}),
            ("Avg ms", {"width": 10, "justify": "right"}),
        ],
    )
    for row in stats:
        avg = f"{row['avg_ms']:.0f}" if row["avg_ms"] else "—"
        table.add_row(row["command"], str(row["count"]), avg)
    console.print(table)


@register_backslash("suggest")
def cmd_suggest(session: ReplSession, args: str) -> None:
    """Suggest next actions based on your command history."""
    try:
        from audiobench.storage.command_graph_repository import get_command_graph_repo
        repo = get_command_graph_repo()
        recent = repo.get_recent_commands(limit=5)
    except Exception as e:
        console.print(f"  [{WARNING}]Could not read command history: {e}[/]")
        return

    if not recent:
        console.print(f"  [{DIM}]No history yet. Run some commands first.[/]")
        return

    last_cmd = recent[0]["command"]
    suggestions = []
    try:
        suggestions = repo.get_next_command_suggestions(after_command=last_cmd, limit=5)
    except Exception:
        pass

    console.print(f"\n  [{BOLD}]Last command:[/] [{ACCENT}]{last_cmd}[/]")
    if suggestions:
        console.print(f"  [{DIM}]What you usually do next:[/]")
        for s in suggestions:
            console.print(f"    [{ACCENT}]\\{s['command']}[/] [{DIM}]({s['count']}×)[/]")
    else:
        console.print(f"  [{DIM}]Not enough history to suggest yet. Keep using AudioBench![/]")

    try:
        workflows = repo.list_workflows()
        if workflows:
            console.print(f"\n  [{DIM}]Saved workflows:[/]")
            for wf in workflows[:3]:
                console.print(
                    f"    [{ACCENT}]\\workflow run {wf['name']}[/] "
                    f"[{DIM}]({len(wf['steps'])} steps)[/]"
                )
    except Exception:
        pass
    console.print()


# ── Library ──────────────────────────────────────────────────

@register_backslash("library")
def cmd_library(session: ReplSession, args: str) -> None:
    """Launch the interactive Library Command Center."""
    from audiobench.cli.commands.library_cmd import run_library
    run_library(session)

@register_resume("library")
def resume_library(session: ReplSession, frame) -> None:
    """Resume the library TUI with its previous state."""
    from audiobench.cli.commands.library_cmd import run_library
    run_library(session, restore_state=frame.state)

@register_resume("chat")
def resume_chat(session: ReplSession, frame) -> None:
    """Resume a previous chat conversation."""
    conversation_id = frame.state.get("conversation_id")
    if conversation_id:
        from audiobench.cli.repl.dispatch import dispatch_command
        dispatch_command(session, ["chat", "--resume", str(conversation_id)])

@register_resume("import")
def resume_import(session: ReplSession, frame) -> None:
    """Resume the import flow."""
    from audiobench.cli.commands.import_cmd import run_import_flow
    run_import_flow(session=session, restore_state=frame.state)

@register_resume("transcribe")
def resume_transcribe(session: ReplSession, frame) -> None:
    """Resume the transcribe flow."""
    from audiobench.cli.repl.dispatch import dispatch_command
    dispatch_command(session, ["transcribe", "--interactive"])

@register_resume("config")
def resume_config(session: ReplSession, frame) -> None:
    """Resume the config flow."""
    from audiobench.cli.repl.dispatch import dispatch_command
    dispatch_command(session, ["config", "--interactive"])

