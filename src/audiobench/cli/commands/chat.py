"""Chat + Ask (AI interactive) commands."""

from __future__ import annotations

import click
from rich.live import Live
from rich.text import Text

from audiobench.cli.display.theme import (
    ACCENT,
    APP_NAME,
    BOLD,
    CHAT_CODE_THEME,
    DIM,
    PROMPT,
    SUCCESS,
    chat_console,
    console,
    error_panel,
)
from audiobench.core.settings import get_settings

# ── Chat Help Text ──────────────────────────────────────────

CHAT_HELP_TEXT = (
    "  [bold]Slash Commands[/]\n"
    "  ─────────────────────────────────────\n"
    "  /help              Show this help\n"
    "  /context [ID]      Show context, or add transcript by ID\n"
    "  /load <ID>         Add a transcript to context\n"
    "  /remove <ID>       Remove a transcript from context\n"
    "  /clear             Clear history and all context\n"
    "  /model <name>      Switch model mid-chat\n"
    "  /compare <model>     Toggle side-by-side comparison\n"
    "  /compare off         Disable comparison mode\n"
    "  /think             Toggle thinking display\n"
    "  /retry             Regenerate last response\n"
    "  /export [file]     Export chat to markdown\n"
    "  /bookmarks [ID]    List bookmarks for a transcript\n"
    "  /history           List past chat sessions\n"
    "  /save              Force-save conversation\n"
    "  /exit              Exit chat (also Ctrl+D)\n"
    "\n"
    "  [bold]Multi-line Input[/]\n"
    "  ─────────────────────────────────────\n"
    '  Type [bold]triple-quotes (\\"\\"\\")'
    "[/] to start/end a multi-line block.\n"
)


# ── Slash Command Handler ───────────────────────────────────


def _handle_slash_command(
    cmd: str,
    session,
    tx_repo,
    chat_repo,
    settings,
) -> bool:
    """Handle a slash command. Returns True if the REPL should exit."""
    parts = cmd.strip().split(None, 1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command in ("/exit", "/quit", "/q"):
        return True

    elif command == "/help":
        console.print()
        console.print(CHAT_HELP_TEXT)

    elif command == "/context":
        if arg and arg.strip().isdigit():
            # /context <ID> → shorthand for /load <ID>
            tid = int(arg.strip())
            record = tx_repo.get_by_id(tid)
            if not record:
                console.print(f"  [{DIM}]Transcript #{tid} not found[/]")
                return False
            session.load_transcripts([record])
            console.print(
                f"  [{SUCCESS}]✓ Loaded #{tid} "
                f"{record['file_name']} "
                f"({record['word_count']:,} words)[/]"
            )
        else:
            console.print()
            for line in session.get_context_summary():
                console.print(f"    {line}")
            console.print()

    elif command == "/load":
        if not arg or not arg.strip().isdigit():
            console.print(f"  [{DIM}]Usage: /load <transcript_id>[/]")
            return False
        tid = int(arg.strip())
        record = tx_repo.get_by_id(tid)
        if not record:
            console.print(f"  [{DIM}]Transcript #{tid} not found[/]")
            return False
        session.load_transcripts([record])
        console.print(
            f"  [{SUCCESS}]✓ Loaded #{tid} "
            f"{record['file_name']} "
            f"({record['word_count']:,} words)[/]"
        )

    elif command == "/clear":
        session.clear_history()
        console.print(
            f"  [{SUCCESS}]✓ Conversation cleared (new session #{session.conversation_id})[/]"
        )
        console.print(f"  [{DIM}]Context reset — use /load <ID> to add transcripts[/]")

    elif command == "/remove":
        if not arg or not arg.strip().isdigit():
            console.print(f"  [{DIM}]Usage: /remove <transcript_id>[/]")
            return False
        tid = int(arg.strip())
        if session.remove_transcript(tid):
            console.print(f"  [{SUCCESS}]✓ Removed transcript #{tid} from context[/]")
        else:
            console.print(f"  [{DIM}]Transcript #{tid} not in context[/]")

    elif command == "/model":
        if not arg:
            console.print(f"  [{DIM}]Current model: {session.model}[/]")
            console.print(f"  [{DIM}]Usage: /model <name>[/]")
            return False
        session.switch_model(arg.strip())
        console.print(f"  [{SUCCESS}]✓ Switched to {arg.strip()}[/]")

    elif command == "/think":
        session.show_thinking = not session.show_thinking
        state = "on" if session.show_thinking else "off"
        console.print(f"  [{SUCCESS}]✓ Thinking display: {state}[/]")

    elif command == "/history":
        convs = chat_repo.list_conversations(limit=10)
        if not convs:
            console.print(f"  [{DIM}]No past conversations[/]")
            return False
        console.print()
        for c in convs:
            tid_list = c.get("transcript_ids", [])
            ctx = f" (transcripts: {tid_list})" if tid_list else ""
            console.print(
                f"    [{ACCENT}]#{c['id']}[/] "
                f"{c['title']} "
                f"[{DIM}]({c['message_count']} msgs, "
                f"{c['model']}){ctx}[/]"
            )
        console.print()

    elif command == "/save":
        console.print(f"  [{SUCCESS}]✓ Conversation #{session.conversation_id} saved[/]")

    elif command == "/export":
        import time as _time
        from pathlib import Path

        if not session.messages:
            console.print(f"  [{DIM}]Nothing to export yet[/]")
            return False
        fname = arg.strip() if arg.strip() else None
        if not fname:
            slug = f"chat_{session.conversation_id or 'new'}_{int(_time.time())}"
            fname = f"{slug}.md"
        path = Path(fname).expanduser()
        lines = [f"# Chat #{session.conversation_id or 'new'}\n"]
        lines.append(f"Model: {session.model}  \n")
        lines.append("---\n")
        for msg in session.messages:
            if msg["role"] == "user":
                lines.append(f"**You:** {msg['content']}\n")
            elif msg["role"] == "assistant":
                lines.append(f"**AI:**\n\n{msg['content']}\n")
            lines.append("---\n")
        path.write_text("\n".join(lines), encoding="utf-8")
        console.print(f"  [{SUCCESS}]✓ Exported to {path}[/]")

    elif command == "/retry":
        # Signal to the REPL that we want a retry
        # We store a flag on the session object
        session._retry_requested = True  # noqa: SLF001
        return False  # handled in the REPL loop

    elif command == "/compare":
        if not arg:
            # Status: show current comparison state
            cmp_model = getattr(session, "_compare_model", None)
            if cmp_model:
                console.print(
                    f"  [{ACCENT}]⚡ Comparison mode ON[/]\n"
                    f"  [{DIM}]Primary:   {session.model}[/]\n"
                    f"  [{DIM}]Secondary: {cmp_model}[/]\n"
                    f"  [{DIM}]Use /compare off to disable[/]"
                )
            else:
                console.print(
                    f"  [{DIM}]Comparison mode is OFF[/]\n"
                    f"  [{DIM}]Usage: /compare <model> to enable[/]\n"
                    f"  [{DIM}]Example: /compare qwen3-next:80b-cloud[/]"
                )
            return False
        if arg.strip().lower() == "off":
            old = getattr(session, "_compare_model", None)
            session._compare_model = None  # noqa: SLF001
            if old:
                console.print(
                    f"  [{SUCCESS}]✓ Comparison mode OFF[/] "
                    f"[{DIM}](was comparing with {old})[/]"
                )
            else:
                console.print(f"  [{DIM}]Comparison mode was already off[/]")
            return False
        # Enable or switch comparison model
        new_model = arg.strip()
        old = getattr(session, "_compare_model", None)
        session._compare_model = new_model  # noqa: SLF001
        if old and old != new_model:
            console.print(
                f"  [{ACCENT}]⚡ Switched comparison:[/] "
                f"[{DIM}]{old}[/] → [{BOLD}]{new_model}[/]"
            )
        else:
            console.print(
                f"  [{ACCENT}]⚡ Comparison mode ON[/]\n"
                f"  [{DIM}]Every prompt will compare {session.model} vs {new_model}[/]\n"
                f"  [{DIM}]/compare off to disable[/]"
            )
        return False

    elif command == "/bookmarks":
        from audiobench.core.db_engine import init_db
        from audiobench.storage.bookmark_repository import (
            BOOKMARK_TYPES,
            BookmarkRepository,
            _format_timestamp as _bfmt,
        )

        init_db()
        bm_repo = BookmarkRepository()

        if arg and arg.strip().isdigit():
            # Show bookmarks for a specific transcript's audio file
            tid = int(arg.strip())
            record = tx_repo.get_by_id(tid)
            if not record:
                console.print(f"  [{DIM}]Transcript #{tid} not found[/]")
                return False
            audio_id = record.get("audio_file_id")
            if not audio_id:
                console.print(f"  [{DIM}]No audio file linked to #{tid}[/]")
                return False
            bookmarks = bm_repo.list_for_file(audio_id)
            label = f"#{tid} {record.get('file_name', '')}"
        else:
            bookmarks = bm_repo.list_all(limit=15)
            label = "All files"

        if not bookmarks:
            console.print(f"  [{DIM}]No bookmarks found[/]")
            return False

        console.print()
        console.print(f"  [{ACCENT}]Bookmarks — {label}[/]")
        for b in bookmarks:
            emoji = BOOKMARK_TYPES.get(b["bookmark_type"], "🔖")
            time_str = _bfmt(b["timestamp"])
            if b.get("is_region") and b.get("end_timestamp"):
                time_str += f"→{_bfmt(b['end_timestamp'])}"
            console.print(
                f"    [{DIM}]#{b['id']}[/] {emoji} {time_str}  {b['name'][:40]}"
            )
        console.print()

    else:
        console.print(f"  [{DIM}]Unknown command: {command} (type /help for commands)[/]")

    return False


# ── Ask Command ─────────────────────────────────────────────


@click.command()
@click.argument("transcript_id", type=int, required=False)
@click.argument("question", required=False)
@click.option("--model", default=None, help="Ollama model (default: from settings)")
@click.option("-i", "--interactive", "interactive_mode", is_flag=True, help="Interactive wizard")
@click.option("--chapter", type=int, default=None, help="Ask about a specific chapter")
@click.option("--log", is_flag=True, help="View the full ask log for an audio file")
def ask(transcript_id: int | None, question: str | None, model: str | None, interactive_mode: bool = False, chapter: int | None = None, log: bool = False) -> None:
    """Ask a question about a transcript using AI.

    \b
    Examples:
      audiobench ask 3 "What decisions were made?"
      audiobench ask 3 "Who is responsible for the API?"
      audiobench ask 3 "List all mentioned dates" --model deepseek-v3.2
    """
    from audiobench.chat.context_builder import TRANSCRIPT_SYSTEM, qa
    from audiobench.chat.providers.ollama_provider import AIError, OllamaClient
    from audiobench.core.db_engine import init_db
    from audiobench.storage.repository import TranscriptionRepository
    from audiobench.cli.display.theme import BOLD, ACCENT, DIM
    import sys

    settings = get_settings()
    model_name = model or settings.ollama_model

    if interactive_mode:
        from audiobench.cli.wizard import prompt_transcription, prompt_string
        try:
            if not transcript_id:
                transcript_id = prompt_transcription("Select a transcript to query")
            if not log and not question:
                question = prompt_string("What is your question?")
        except KeyboardInterrupt:
            sys.exit(0)

    if not transcript_id:
        console.print(error_panel("Usage: audiobench ask [ID] [QUESTION]\nOr use --interactive"))
        sys.exit(1)

    if not log and not question:
        console.print(error_panel("Usage: audiobench ask [ID] [QUESTION]\nOr use --interactive"))
        sys.exit(1)

    # Fetch transcript
    init_db()
    repo = TranscriptionRepository()
    record = repo.get_by_id(transcript_id)
    if not record:
        console.print(error_panel("Not found", f"Transcript #{transcript_id} not found"))
        return
        
    audio_file_id = record.get("audio_file_id")

    if log:
        from audiobench.core.db_session import get_session
        from audiobench.storage.models import AskLog, AskEntry
        from rich.table import Table
        
        with get_session() as session:
            ask_log = session.query(AskLog).filter_by(audio_file_id=audio_file_id).first()
            if not ask_log or not ask_log.entries:
                console.print(f"[dim]No ask log entries found for audio file #{audio_file_id}.[/]")
                return
                
            table = Table(title=f"Ask Log for Audio #{audio_file_id}")
            table.add_column("Date", style="dim")
            table.add_column("Model", style="blue")
            table.add_column("Question", style="green")
            table.add_column("Answer")
            
            for entry in ask_log.entries:
                table.add_row(
                    entry.created_at.strftime("%Y-%m-%d %H:%M"),
                    entry.model_name,
                    entry.question,
                    entry.answer[:100] + ("..." if len(entry.answer) > 100 else "")
                )
            
            console.print(table)
        return
        
    if chapter is not None:
        from audiobench.storage.chapter_repository import get_chapter_repo
        from audiobench.storage.models import ChapterRecord
        from audiobench.core.db_session import get_session
        audio_file_id = record.get("audio_file_id")
        if audio_file_id:
            chap = get_chapter_repo().get_chapter_by_index(audio_file_id, chapter)
            if chap and chap.id:
                with get_session() as db:
                    db_chap = db.query(ChapterRecord).filter_by(id=chap.id).first()
                    tx_id = db_chap.transcription_id if db_chap else None
                if tx_id:
                    chap_record = repo.get_by_id(tx_id)
                    if chap_record:
                        record = chap_record
                    else:
                        console.print(error_panel("Chapter Not Found", f"Chapter {chapter} transcript is missing."))
                    return
            else:
                console.print(error_panel("Chapter Not Ready", f"Chapter {chapter} is missing or not transcribed."))
                return

    console.print()
    console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — AI Q&A")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print(f"    Source:   [{ACCENT}]#{transcript_id} {record['file_name']}[/]")
    console.print(f"    Question: {question}")
    console.print(f"    Model:    {model_name}")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print()

    prompt = qa(record["full_text"], question)

    try:
        client = OllamaClient(
            base_url=settings.ollama_base_url,
            model=model_name,
        )

        if not client.is_available():
            console.print(
                error_panel(
                    "Ollama not running",
                    "Start with: ollama serve",
                )
            )
            return

        import time as _time
        from rich.live import Live
        from rich.markdown import Markdown as RichMarkdown
        from rich.padding import Padding
        from rich.console import Group
        from rich.text import Text
        from audiobench.cli.display.theme import CHAT_CODE_THEME, DIM
        
        content_parts = []
        token_count = 0
        t_start = _time.monotonic()
        
        with Live(
            console=console,
            refresh_per_second=8,
            transient=True,
        ) as live:
            for token in client.stream(prompt, system_prompt=TRANSCRIPT_SYSTEM):
                content_parts.append(token)
                token_count += 1
                
                full_text = "".join(content_parts)
                preview_lines = full_text.splitlines()
                display_parts = []
                
                if len(preview_lines) > 8:
                    preview = "\n".join(preview_lines[-8:])
                    display_parts.append(Text("  ⋮\n", style="dim"))
                else:
                    preview = full_text
                    
                display_parts.append(Text(preview))
                elapsed_so_far = _time.monotonic() - t_start
                tps_so_far = token_count / elapsed_so_far if elapsed_so_far > 0 else 0
                display_parts.append(
                    Text(
                        f"\n  ▍ {token_count} tokens · {tps_so_far:.0f} tok/s",
                        style="dim",
                    )
                )
                live.update(Group(*display_parts))

        final_md = "".join(content_parts)
        console.print(
            Padding(
                RichMarkdown(final_md, code_theme=CHAT_CODE_THEME),
                (0, 0, 0, 0),
            )
        )

        elapsed = _time.monotonic() - t_start
        if token_count > 0 and elapsed > 0:
            tps = token_count / elapsed
            console.print(
                f"  [{DIM}]{token_count} tokens · {tps:.1f} tok/s · {elapsed:.1f}s[/]"
            )
        console.print()

        # --- PHASE 5.3: Ask Log & Expression Wiring ---
        from audiobench.chat.chat_store import ChatRepository
        from audiobench.storage.expression_repository import ExpressionRepository
        from audiobench.memory.enums import SourceType, RelationType
        from audiobench.daemon.factory import get_daemon_client
        from audiobench.core.logger_factory import get_logger
        
        _logger = get_logger("cmd.ask")
        
        audio_file_id = record.get("audio_file_id")
        if audio_file_id:
            chat_repo = ChatRepository()
            log_id = chat_repo.get_or_create_ask_log(audio_file_id)
            
            expr_repo = ExpressionRepository()
            
            # 1. Register Query Expression
            q_expr = expr_repo.register(
                content=question,
                source_type=SourceType.ASK_QUERY.value,
                source_id=log_id,
            )
            
            # 2. Register Answer Expression
            a_expr = expr_repo.register(
                content=final_md,
                source_type=SourceType.ASK_ANSWER.value,
                source_id=log_id,
            )
            
            # 3. Link Query -> Answer (Wait, relation_type is source)
            # Query is the source of the answer? No, Answer is derived from Query.
            # Relation(from=a_expr, to=q_expr, type=source)
            expr_repo.link(from_id=a_expr.id, to_id=q_expr.id, relation_type=RelationType.SOURCE.value)
            
            # 4. Link Answer -> Transcript Expression?
            # To do this, we need the Tier 1 transcript expression. We don't have its ID immediately,
            # but we can look it up or we can just link to the transcript ID in source_id (already done).
            
            chat_repo.add_ask_entry(
                log_id=log_id,
                question=question,
                answer=final_md,
                model_name=model_name,
                question_expression_id=q_expr.id,
                answer_expression_id=a_expr.id,
            )
            
            try:
                daemon = get_daemon_client()
                daemon.embed(
                    expression_id=q_expr.id,
                    content=question,
                    source_type=SourceType.ASK_QUERY,
                )
                daemon.embed(
                    expression_id=a_expr.id,
                    content=final_md,
                    source_type=SourceType.ASK_ANSWER,
                )
            except Exception as ex:
                _logger.warning("Daemon not available for ask query embedding: %s", ex)

    except AIError as e:
        console.print(error_panel("AI Error", str(e)))


# ── Chat Command ────────────────────────────────────────────


@click.command()
@click.argument("transcript_ids", nargs=-1, type=int)
@click.option(
    "--model",
    default=None,
    help="Ollama model (default: from settings)",
)
@click.option(
    "--temperature",
    default=0.3,
    type=float,
    help="Creativity level (0.0-1.0)",
)
@click.option(
    "--search",
    "search_query",
    default=None,
    help="Load transcripts matching this search",
)
@click.option(
    "--recent",
    default=None,
    type=int,
    help="Load N most recent transcripts as context",
)
@click.option(
    "--resume",
    "resume_id",
    default=None,
    type=int,
    help="Resume a previous conversation by ID",
)
@click.option(
    "--list",
    "list_chats",
    is_flag=True,
    help="List past chat conversations",
)
@click.option(
    "--delete",
    "delete_id",
    default=None,
    type=int,
    help="Delete a chat conversation by ID",
)
@click.option(
    "--think/--no-think",
    default=True,
    help="Show/hide model chain-of-thought",
)
@click.option(
    "--chapter",
    type=int,
    default=None,
    help="Chat with a specific chapter",
)
@click.option(
    "--summary",
    type=int,
    default=None,
    help="View the session memoir for a conversation",
)
def chat(
    transcript_ids: tuple[int, ...],
    model: str | None,
    temperature: float,
    search_query: str | None,
    recent: int | None,
    resume_id: int | None,
    list_chats: bool,
    delete_id: int | None,
    think: bool,
    chapter: int | None,
    summary: int | None,
) -> None:
    """Interactive AI chat with transcript context.

    \b
    Examples:
      audiobench chat                           Chat freely
      audiobench chat 3                         Chat about transcript #3
      audiobench chat 3 5 7                     Chat with multiple transcripts
      audiobench chat --search "meeting"        Load matching transcripts
      audiobench chat --recent 5                Load 5 most recent
      audiobench chat --resume 2                Resume conversation #2
      audiobench chat --list                    List past conversations
      audiobench chat --delete 2                Delete conversation #2
      audiobench chat --model deepseek-v3.1:671b-cloud
    """
    from audiobench.chat.chat_session import ChatSession
    from audiobench.chat.chat_store import ChatRepository
    from audiobench.chat.providers.ollama_provider import AIError, OllamaClient
    from audiobench.core.db_engine import init_db
    from audiobench.storage.repository import TranscriptionRepository

    settings = get_settings()
    model_name = model or settings.ollama_model
    init_db()

    chat_repo = ChatRepository()
    tx_repo = TranscriptionRepository()

    # ── Handle --list ──
    if list_chats:
        convs = chat_repo.list_conversations(limit=20)
        if not convs:
            console.print(f"  [{DIM}]No chat conversations yet[/]")
            return
        console.print()
        console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — Chat History")
        console.print(f"  [{DIM}]{'─' * 44}[/]")
        for c in convs:
            tid_list = c.get("transcript_ids", [])
            ctx = f" ctx:{tid_list}" if tid_list else ""
            console.print(
                f"    [{ACCENT}]#{c['id']}[/] "
                f"{c['title']} "
                f"[{DIM}]({c['message_count']} msgs"
                f"{ctx})[/]"
            )
        console.print()
        console.print(f"  [{DIM}]Resume with: audiobench chat --resume <ID>[/]")
        console.print()
        return

    # ── Handle --delete ──
    if summary is not None:
        init_db()
        from audiobench.core.db_session import get_session
        from audiobench.storage.models import ConversationSummary
        from rich.panel import Panel
        from rich.markdown import Markdown

        with get_session() as session:
            record = session.query(ConversationSummary).filter_by(conversation_id=summary).first()
            if not record:
                console.print(error_panel("Not found", f"Session summary for conversation #{summary} not found."))
                import sys
                sys.exit(1)

            md = f"**Narrative**: {record.narrative}\n\n"
            md += f"**Key Insights**: {record.key_insights}\n\n"
            md += f"**Open Threads**: {record.open_threads}\n\n"
            
            console.print(Panel(Markdown(md), title=f"Session Summary #{summary}", border_style="blue"))
        import sys
        sys.exit(0)

    if delete_id is not None:
        if chat_repo.delete_conversation(delete_id):
            console.print(f"  [{SUCCESS}]✓ Deleted conversation #{delete_id}[/]")
        else:
            console.print(
                error_panel(
                    "Not found",
                    f"Conversation #{delete_id} not found",
                )
            )
        return

    # ── Check Ollama ──
    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=model_name,
    )
    if not client.is_available():
        console.print(
            error_panel(
                "Ollama not running",
                f"Start with: ollama serve\nPull model: ollama pull {model_name}",
            )
        )
        return

    # ── Create or resume session ──
    session = ChatSession(
        client=client,
        chat_repo=chat_repo,
        model=model_name,
        temperature=temperature,
        conversation_id=resume_id,
        show_thinking=think,
    )

    # Resume existing conversation
    if resume_id is not None and not session.restore_from_db():
        console.print(
            error_panel(
                "Not found",
                f"Conversation #{resume_id} not found",
            )
        )
        return

    # ── Load transcript context ──
    transcripts_to_load = []

    # By explicit IDs
    for tid in transcript_ids:
        record = tx_repo.get_by_id(tid)
        if record:
            if chapter is not None:
                from audiobench.storage.chapter_repository import get_chapter_repo
                from audiobench.storage.models import ChapterRecord
                from audiobench.core.db_session import get_session
                audio_file_id = record.get("audio_file_id")
                if audio_file_id:
                    chap = get_chapter_repo().get_chapter_by_index(audio_file_id, chapter)
                    if chap and chap.id:
                        with get_session() as db:
                            db_chap = db.query(ChapterRecord).filter_by(id=chap.id).first()
                            tx_id = db_chap.transcription_id if db_chap else None
                        if tx_id:
                            chap_record = tx_repo.get_by_id(tx_id)
                            if chap_record:
                                chap_record["file_name"] = f"{record['file_name']} (Chapter {chapter}: {chap.title})"
                                transcripts_to_load.append(chap_record)
                                continue
                            else:
                                console.print(f"  [{DIM}]Chapter {chapter} transcript not found, skipping[/]")
                                continue
                    else:
                        console.print(f"  [{DIM}]Chapter {chapter} not found or not transcribed, skipping[/]")
                        continue
            transcripts_to_load.append(record)
        else:
            console.print(f"  [{DIM}]Transcript #{tid} not found, skipping[/]")

    # By search
    if search_query:
        results = tx_repo.search(search_query, limit=5)
        for r in results:
            full = tx_repo.get_by_id(r["id"])
            if full:
                transcripts_to_load.append(full)
        if not results:
            console.print(f"  [{DIM}]No transcripts matching '{search_query}'[/]")

    # By recent
    if recent:
        history_items = tx_repo.get_history(limit=recent)
        for h in history_items:
            full = tx_repo.get_by_id(h["id"])
            if full:
                transcripts_to_load.append(full)

    if transcripts_to_load:
        session.load_transcripts(transcripts_to_load)

    # ── Header ──
    console.print()
    conv_label = f" [#{resume_id}]" if resume_id else ""
    console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — AI Chat{conv_label}")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print(f"    Model:    {model_name}")
    ctx_lines = session.get_context_summary()
    console.print(f"    Context:  {ctx_lines[0]}")
    for line in ctx_lines[1:]:
        console.print(f"              {line}")
    think_label = "on" if think else "off"
    console.print(f"    Thinking: {think_label}")
    if resume_id and session.turn_count > 0:
        console.print(f"    Resumed:  {session.turn_count} previous turn(s)")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print()

    # ── Render past messages on resume ──
    import time as _time
    from pathlib import Path as _Path

    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.history import FileHistory
    from rich.console import Group
    from rich.layout import Layout
    from rich.markdown import Markdown as RichMarkdown
    from rich.padding import Padding
    from rich.panel import Panel

    # ── prompt_toolkit session with persistent history ──
    _history_file = _Path.home() / ".cache" / "audiobench_chat_history"
    _history_file.parent.mkdir(parents=True, exist_ok=True)
    _pt_session: PromptSession = PromptSession(
        history=FileHistory(str(_history_file)),
    )
    _multiline_active = False  # toggled by """

    def _save_readline_history() -> None:
        pass  # prompt_toolkit auto-saves via FileHistory

    def _render_comparison_pair(msg_a: dict, msg_b: dict) -> None:
        """Render a comparison pair as side-by-side panels."""
        layout = Layout()
        layout.split_row(
            Layout(name="left"),
            Layout(name="right"),
        )
        for side, msg in [("left", msg_a), ("right", msg_b)]:
            parts = []
            if msg.get("thinking") and session.show_thinking:
                think_preview = msg["thinking"][:300]
                if len(msg["thinking"]) > 300:
                    think_preview += "…"
                parts.append(Text(f"💭 {think_preview}", style="dim italic"))
            parts.append(
                RichMarkdown(msg["content"], code_theme=CHAT_CODE_THEME)
            )
            model_label = msg.get("model_name") or "Model"
            border = "cyan" if side == "left" else "magenta"
            layout[side].update(
                Panel(Group(*parts), title=model_label, border_style=border)
            )
        console.print(layout)

    # ── Render past messages on resume ──
    if resume_id and session.messages:
        console.print(f"  [{DIM}]─── Previous Messages ───[/]")
        console.print()
        msgs = session.messages
        i = 0
        while i < len(msgs):
            msg = msgs[i]
            if msg["role"] == "user":
                console.print(f"  [{PROMPT}]>>> {msg['content']}[/]")
                console.print()
                i += 1
            elif msg["role"] == "assistant":
                # Detect comparison pair: two consecutive assistants with different models
                if (
                    i + 1 < len(msgs)
                    and msgs[i + 1]["role"] == "assistant"
                    and msg.get("model_name") != msgs[i + 1].get("model_name")
                ):
                    _render_comparison_pair(msg, msgs[i + 1])
                    console.print()
                    i += 2
                elif msg["content"].strip():
                    # Show thinking if present
                    if msg.get("thinking") and session.show_thinking:
                        think_preview = msg["thinking"][:200]
                        if len(msg["thinking"]) > 200:
                            think_preview += "…"
                        console.print(
                            Padding(
                                Text(f"💭 {think_preview}", style="dim italic"),
                                (0, 2, 0, 4),
                            )
                        )
                    md = RichMarkdown(
                        msg["content"],
                        code_theme=CHAT_CODE_THEME,
                    )
                    chat_console.print(Padding(md, (0, 2, 1, 2)))
                    console.print()
                    i += 1
                else:
                    i += 1
            else:
                i += 1
        console.print(f"  [{DIM}]─── End of History ───[/]")
        console.print()

    # ── Helper: stream a message and render ──
    def _stream_and_render(user_text: str) -> None:
        """Send user input and render the streamed response.

        Uses a compact tail-preview in Rich Live during streaming to
        avoid the scrollback-duplication problem that occurs when Live
        content exceeds the terminal viewport height.  The full formatted
        markdown is printed once after streaming completes.
        """
        console.print()
        try:
            thinking_parts: list[str] = []
            content_parts: list[str] = []
            token_count = 0
            t_start = _time.monotonic()

            with Live(
                console=chat_console,
                refresh_per_second=8,
                transient=True,
            ) as live:
                for chunk in session.send(user_text):
                    thinking = chunk.get("thinking", "")
                    content = chunk.get("content", "")

                    if thinking:
                        thinking_parts.append(thinking)

                    if content:
                        content_parts.append(content)

                    if content:
                        token_count += 1

                    # ── Compact streaming preview ──
                    # Only show the *tail* of thinking/content so the Live
                    # viewport never exceeds the terminal height.  This
                    # prevents Rich from pushing lines into the permanent
                    # scrollback buffer where they can't be erased.
                    display_parts = []

                    if thinking_parts and session.show_thinking:
                        think_text = "".join(thinking_parts)
                        think_lines = think_text.splitlines()
                        if len(think_lines) > 5:
                            think_text = "…\n" + "\n".join(think_lines[-5:])
                        display_parts.append(
                            Text(f"💭 {think_text}", style="dim italic"),
                        )

                    if content_parts:
                        full_text = "".join(content_parts)
                        preview_lines = full_text.splitlines()
                        if len(preview_lines) > 8:
                            preview = "\n".join(preview_lines[-8:])
                            display_parts.append(
                                Text("  ⋮\n", style="dim"),
                            )
                        else:
                            preview = full_text
                        display_parts.append(Text(preview))
                        # Token counter only during content streaming
                        elapsed_so_far = _time.monotonic() - t_start
                        tps_so_far = token_count / elapsed_so_far if elapsed_so_far > 0 else 0
                        display_parts.append(
                            Text(
                                f"\n  ▍ {token_count} tokens · {tps_so_far:.0f} tok/s",
                                style="dim",
                            ),
                        )

                    if display_parts:
                        live.update(Group(*display_parts))

            # ── Print full formatted content ──
            # Live was transient so its viewport is erased; we now
            # print the complete markdown-rendered response once.
            if content_parts:
                final_md = "".join(content_parts)
                chat_console.print(
                    Padding(
                        RichMarkdown(final_md, code_theme=CHAT_CODE_THEME),
                        (0, 0, 0, 0),
                    )
                )
            elif thinking_parts:
                # Some models (e.g. deepseek) return the entire response
                # in the "thinking" field with empty "content". Display
                # the thinking text as the response in that case.
                final_md = "".join(thinking_parts)
                chat_console.print(
                    Padding(
                        RichMarkdown(final_md, code_theme=CHAT_CODE_THEME),
                        (0, 0, 0, 0),
                    )
                )

            # Persist response + background title gen (non-blocking)
            session.finalize_response()

            # Token stats
            elapsed = _time.monotonic() - t_start
            if token_count > 0 and elapsed > 0:
                tps = token_count / elapsed
                console.print(
                    f"  [{DIM}]{token_count} tokens · {tps:.1f} tok/s · {elapsed:.1f}s[/]"
                )
            console.print()

        except KeyboardInterrupt:
            # Save partial response if anything was generated
            if content_parts:
                session.finalize_response()
            console.print()
            console.print(f"  [{DIM}]Generation interrupted[/]")
            console.print()

        except AIError as e:
            console.print(error_panel("AI Error", str(e)))
            console.print()

    # ── Helper: compare two models and render ──
    def _compare_and_render(user_text: str, compare_model: str) -> None:
        """Run comparison between primary and secondary model."""
        console.print()
        try:
            from audiobench.chat.compare import ModelComparison

            # Build messages for the comparison
            cmp_messages = session._build_api_messages()  # noqa: SLF001
            cmp_messages.append({"role": "user", "content": user_text})

            comparison = ModelComparison(
                client=client,
                messages=cmp_messages,
                model_a=session.model,
                model_b=compare_model,
                temperature=temperature,
                show_thinking=session.show_thinking,
            )
            result = comparison.run()

            # Ensure conversation exists
            if not session.conversation_id:
                session._conversation_id = chat_repo.create_conversation(  # noqa: SLF001
                    model=session.model,
                    title="Model Comparison",
                )

            # Save user message
            chat_repo.add_message(
                session.conversation_id, "user", user_text
            )
            session._messages.append(  # noqa: SLF001
                {"role": "user", "content": user_text}
            )

            # Save Model A response
            res_a = result["model_a"]
            chat_repo.add_message(
                session.conversation_id,
                "assistant",
                res_a["content"],
                thinking=res_a["thinking"],
                model_name=res_a["model_name"],
            )
            session._messages.append({  # noqa: SLF001
                "role": "assistant",
                "content": res_a["content"],
                "thinking": res_a["thinking"],
                "model_name": res_a["model_name"],
            })

            # Save Model B response
            res_b = result["model_b"]
            chat_repo.add_message(
                session.conversation_id,
                "assistant",
                res_b["content"],
                thinking=res_b["thinking"],
                model_name=res_b["model_name"],
            )
            session._messages.append({  # noqa: SLF001
                "role": "assistant",
                "content": res_b["content"],
                "thinking": res_b["thinking"],
                "model_name": res_b["model_name"],
            })

            # Stats
            elapsed = result["elapsed"]
            total_tokens = res_a["tokens"] + res_b["tokens"]
            tps = total_tokens / elapsed if elapsed > 0 else 0
            console.print(
                f"  [{DIM}]{total_tokens} tok · {tps:.0f} tok/s · {elapsed:.1f}s[/]"
            )
            console.print()

            # Trigger title generation if first turn
            if session.turn_count <= 1:
                session._generate_title_async()  # noqa: SLF001

        except KeyboardInterrupt:
            console.print()
            console.print(f"  [{DIM}]Comparison interrupted[/]")
            console.print()

        except Exception as e:
            console.print(error_panel("Comparison Error", str(e)))
            console.print()

    # ── Multi-line input helper ──
    def _read_multiline() -> str:
        """Read multi-line input via prompt_toolkit (Alt+Enter or \"\"\" to end)."""
        console.print(
            f'  [{DIM}]Multi-line mode — type \"\"\" on its own line or press '
            f'Alt+Enter to submit:[/]'
        )
        try:
            text = _pt_session.prompt(
                ANSI('\033[38;5;240m... \033[0m'),
                multiline=True,
            )
        except (EOFError, KeyboardInterrupt):
            return ""
        # Strip wrapping triple-quotes if user typed them
        text = text.strip()
        if text.startswith('"""'):
            text = text[3:]
        if text.endswith('"""'):
            text = text[:-3]
        return text.strip()

    # ── Interactive REPL ──
    last_user_input: str | None = None
    session._retry_requested = False  # noqa: SLF001
    if not hasattr(session, "_compare_model"):
        session._compare_model = None  # noqa: SLF001

    def _trigger_summary(conv_id: int, messages: list[dict], repo, settings) -> None:
        """Trigger summary generation in a background thread."""
        import threading
        
        def run_summary():
            from audiobench.chat.summary_generator import SummaryGenerator
            from audiobench.storage.expression_repository import ExpressionRepository
            from audiobench.memory.enums import SourceType
            from audiobench.daemon.factory import get_daemon_client
            import json
            
            gen = SummaryGenerator()
            result = gen.generate(messages)
            if not result:
                return
                
            # Update title
            if result.refined_title:
                repo.update_title(conv_id, result.refined_title)
                
            # Write Expression
            expr_repo = ExpressionRepository()
            expr = expr_repo.register(
                content=result.narrative,
                source_type=SourceType.SESSION_SUMMARY.value,
                source_id=conv_id,
                session_type="chat",
                session_id=conv_id,
            )
            
            # Save ConversationSummary record
            repo.save_summary(
                conversation_id=conv_id,
                narrative=result.narrative,
                drift_phases=json.dumps(result.drift_phases),
                key_insights=json.dumps(result.key_insights),
                open_threads=json.dumps(result.open_threads),
                refined_title=result.refined_title,
                generated_by=gen.model_name,
                expression_id=expr.id,
            )
            
            # Embed Expression
            try:
                daemon = get_daemon_client()
                daemon.embed(
                    expression_id=expr.id,
                    content=result.narrative,
                    source_type=SourceType.SESSION_SUMMARY,
                )
            except Exception as ex:
                pass # Already logged by DaemonFactory/Client
                
        thread = threading.Thread(target=run_summary, daemon=True)
        thread.start()

    while True:
        try:
            # Show comparison mode in prompt
            cmp_active = getattr(session, "_compare_model", None)
            if cmp_active:
                prompt_str = ANSI('\033[38;5;214m⚡ >>> \033[0m')
            else:
                prompt_str = ANSI('\033[38;5;48m>>> \033[0m')
            user_input = _pt_session.prompt(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            _save_readline_history()
            if session.conversation_id:
                console.print(
                    f"  [{SUCCESS}]✓ Conversation "
                    f"#{session.conversation_id} saved "
                    f"({session.turn_count * 2} messages)[/]"
                )
                # --- PHASE 6.2/6.3: Wire Summary Trigger ---
                if session.turn_count >= 3:
                    _trigger_summary(session.conversation_id, session.messages, chat_repo, settings)
            console.print(f"  [{DIM}]Goodbye![/]")
            console.print()
            break

        if not user_input:
            continue

        # Multi-line input
        if user_input == '"""':
            user_input = _read_multiline()
            if not user_input.strip():
                continue

        # Slash commands (accept both / and \)
        if user_input.startswith("\\"):
            user_input = "/" + user_input[1:]
        if user_input.startswith("/"):
            should_exit = _handle_slash_command(
                user_input,
                session,
                tx_repo,
                chat_repo,
                settings,
            )

            # Handle /retry
            if getattr(session, "_retry_requested", False):
                session._retry_requested = False  # noqa: SLF001
                if last_user_input and session.messages:
                    # Remove last assistant + user message
                    session._messages = [m for m in session._messages if m != session._messages[-1]]
                    if session._messages and session._messages[-1]["role"] == "user":
                        session._messages.pop()
                    console.print(f"  [{DIM}]Regenerating...[/]")
                    _stream_and_render(last_user_input)
                else:
                    console.print(f"  [{DIM}]Nothing to retry[/]")
                continue

            if should_exit:
                _save_readline_history()
                if session.conversation_id:
                    console.print(
                        f"  [{SUCCESS}]✓ Conversation "
                        f"#{session.conversation_id} saved "
                        f"({session.turn_count * 2} messages)"
                        f"[/]"
                    )
                    # --- PHASE 6.2/6.3: Wire Summary Trigger ---
                    if session.turn_count >= 3:
                        _trigger_summary(session.conversation_id, session.messages, chat_repo, settings)
                console.print(f"  [{DIM}]Goodbye![/]")
                console.print()
                break
            continue

        last_user_input = user_input

        # Route through comparison or single-model
        compare_model = getattr(session, "_compare_model", None)
        if compare_model:
            _compare_and_render(user_input, compare_model)
        else:
            _stream_and_render(user_input)
