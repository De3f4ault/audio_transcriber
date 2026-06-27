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

def _maybe_pick_chapters(audio_file_id: int, token_threshold: int = 80000) -> list[int] | None:
    """Check if an audio file's transcript is too large and prompt for chapters if so.
    
    Returns a list of chapter IDs if picked, or None if the file is small enough
    to load entirely.
    """
    from audiobench.core.db_session import get_session
    from audiobench.storage.models import ChapterRecord
    from audiobench.cli.tui.chapter_picker import pick_chapters
    
    with get_session() as session:
        chapters = session.query(ChapterRecord).filter_by(audio_file_id=audio_file_id).order_by(ChapterRecord.start_time).all()
        
    if not chapters:
        return None
        
    total_chars = sum((ch.transcript_length or 0) for ch in chapters)
    estimated_tokens = total_chars / 4
    
    if estimated_tokens > token_threshold:
        return pick_chapters(audio_file_id)
        
    return None

# ── Ask Command ─────────────────────────────────────────────


@click.command()
@click.argument("transcript_id", type=int, required=False)
@click.argument("question", required=False)
@click.option("--model", default=None, help="Ollama model (default: from settings)")
@click.option("-i", "--interactive", "interactive_mode", is_flag=True, help="Interactive wizard")
@click.option("--chapter", type=int, default=None, help="Ask about a specific chapter")
@click.option("--log", is_flag=True, help="View the full ask log for an audio file")
def ask(
    transcript_id: int | None,
    question: str | None,
    model: str | None,
    interactive_mode: bool = False,
    chapter: int | None = None,
    log: bool = False,
) -> None:
    """Ask a question about a transcript using AI.

    \b
    Examples:
      audiobench ask 3 "What decisions were made?"
      audiobench ask 3 "Who is responsible for the API?"
      audiobench ask 3 "List all mentioned dates" --model deepseek-v3.2
    """
    import sys

    from audiobench.chat.context_builder import TRANSCRIPT_SYSTEM, qa
    from audiobench.chat.providers.ollama_provider import AIError, OllamaClient
    from audiobench.cli.display.theme import ACCENT, BOLD, DIM
    from audiobench.core.db_engine import init_db
    from audiobench.storage.repository import TranscriptionRepository

    settings = get_settings()
    model_name = model or settings.ollama_model

    if interactive_mode:
        from audiobench.cli.wizard import prompt_string, prompt_transcription

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
        from rich.table import Table

        from audiobench.core.db_session import get_session
        from audiobench.storage.models import AskLog

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
                    entry.answer[:100] + ("..." if len(entry.answer) > 100 else ""),
                )

            console.print(table)
        return

    if chapter is not None:
        from audiobench.core.db_session import get_session
        from audiobench.storage.chapter_repository import get_chapter_repo
        from audiobench.storage.models import ChapterRecord

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
                        console.print(
                            error_panel(
                                "Chapter Not Found", f"Chapter {chapter} transcript is missing."
                            )
                        )
                    return
            else:
                console.print(
                    error_panel(
                        "Chapter Not Ready", f"Chapter {chapter} is missing or not transcribed."
                    )
                )
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

        from rich.console import Group
        from rich.live import Live
        from rich.markdown import Markdown as RichMarkdown
        from rich.padding import Padding
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
            console.print(f"  [{DIM}]{token_count} tokens · {tps:.1f} tok/s · {elapsed:.1f}s[/]")
        console.print()

        # --- PHASE 5.3: Ask Log & Expression Wiring ---
        from audiobench.chat.chat_store import ChatRepository
        from audiobench.core.logger_factory import get_logger
        from audiobench.memory.knowledge_ingester import KnowledgeIngester
        import threading

        _logger = get_logger("cmd.ask")

        audio_file_id = record.get("audio_file_id")
        if audio_file_id:
            chat_repo = ChatRepository()
            log_id = chat_repo.get_or_create_ask_log(audio_file_id)

            entry_id = chat_repo.add_ask_entry(
                log_id=log_id,
                question=question,
                answer=final_md,
                model_name=model_name,
            )

            try:
                from audiobench.core.db_session import get_session
                from audiobench.storage.models import AskEntry, AskLog
                
                with get_session() as session:
                    entry = session.query(AskEntry).filter_by(id=entry_id).first()
                    log = session.query(AskLog).filter_by(id=log_id).first()
                    if entry and log:
                        session.expunge(entry)
                        session.expunge(log)
                        ingester = KnowledgeIngester()
                        threading.Thread(
                            target=ingester.ingest_ask_entry,
                            args=(entry, log),
                            daemon=True
                        ).start()
            except Exception as ex:
                _logger.warning("Failed to spawn ingestion thread for ask entry %d: %s", entry_id, ex)

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
@click.option(
    "--project",
    "project_id",
    type=int,
    default=None,
    help="Start/resume a study project by project ID",
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
    project_id: int | None,
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
        from rich.markdown import Markdown
        from rich.panel import Panel

        from audiobench.core.db_session import get_session
        from audiobench.storage.models import ConversationSummary

        with get_session() as session:
            record = session.query(ConversationSummary).filter_by(conversation_id=summary).first()
            if not record:
                console.print(
                    error_panel(
                        "Not found", f"Session summary for conversation #{summary} not found."
                    )
                )
                import sys

                sys.exit(1)

            md = f"**Narrative**: {record.narrative}\n\n"
            md += f"**Key Insights**: {record.key_insights}\n\n"
            md += f"**Open Threads**: {record.open_threads}\n\n"

            console.print(
                Panel(Markdown(md), title=f"Session Summary #{summary}", border_style="blue")
            )
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

    # ── Study Project Integration ──
    current_session_number = None
    transcripts_to_load = []
    
    if project_id is not None:
        from audiobench.core.db_session import get_session
        from audiobench.storage.models import StudyProject, StudySession
        from audiobench.storage.chapter_repository import get_chapter_repo
        from audiobench.storage.models import ChapterRecord
        import json
        import sys

        with get_session() as db:
            project = db.query(StudyProject).filter_by(id=project_id).first()
            if not project:
                console.print(error_panel("Not found", f"Study project #{project_id} not found"))
                sys.exit(1)
            
            # Find the active session
            sess = (
                db.query(StudySession)
                .filter_by(project_id=project_id)
                .filter(StudySession.closed_at.is_(None))
                .first()
            )
            if not sess:
                console.print(error_panel("No active session", "Use 'audiobench study resume' to start a new session"))
                sys.exit(1)
            
            current_session_number = sess.session_number
            audio_file_id = project.audio_file_id
            
            if sess.conversation_id:
                resume_id = sess.conversation_id
            
            try:
                chap_list = json.loads(sess.chapter_ids)
            except Exception:
                chap_list = []
            
            if chap_list:
                # Load only these chapters
                for ch_idx in chap_list:
                    chap = get_chapter_repo().get_chapter_by_index(audio_file_id, ch_idx)
                    if chap and chap.id:
                        db_chap = db.query(ChapterRecord).filter_by(id=chap.id).first()
                        tx_id = db_chap.transcription_id if db_chap else None
                        if tx_id:
                            chap_record = tx_repo.get_by_id(tx_id)
                            if chap_record:
                                chap_record["file_name"] = f"{chap_record['file_name']} (Chapter {ch_idx}: {chap.title})"
                                transcripts_to_load.append(chap_record)
            else:
                # Load full file
                from audiobench.storage.models import TranscriptionRecord
                tx = db.query(TranscriptionRecord).filter_by(audio_file_id=audio_file_id).first()
                if tx:
                    record = tx_repo.get_by_id(tx.id)
                    if record:
                        transcripts_to_load.append(record)

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

    # ── Load standard transcript context (if not study project) ──
    if project_id is None:
        # By explicit IDs
        for tid in transcript_ids:
            record = tx_repo.get_by_id(tid)
            if record:
                if chapter is not None:
                    from audiobench.core.db_session import get_session
                    from audiobench.storage.chapter_repository import get_chapter_repo
                    from audiobench.storage.models import ChapterRecord

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
                                    chap_record["file_name"] = (
                                        f"{record['file_name']} (Chapter {chapter}: {chap.title})"
                                    )
                                    transcripts_to_load.append(chap_record)
                                    continue
                                else:
                                    console.print(
                                        f"  [{DIM}]Chapter {chapter} transcript not found, skipping[/]"
                                    )
                                    continue
                        else:
                            console.print(
                                f"  [{DIM}]Chapter {chapter} not found or not transcribed, skipping[/]"
                            )
                            continue
                else:
                    audio_file_id = record.get("audio_file_id")
                    if audio_file_id:
                        picked = _maybe_pick_chapters(audio_file_id)
                        if picked:
                            from audiobench.core.db_session import get_session
                            from audiobench.storage.chapter_repository import get_chapter_repo
                            from audiobench.storage.models import ChapterRecord
                            
                            for ch_idx in picked:
                                chap = get_chapter_repo().get_chapter_by_index(audio_file_id, ch_idx)
                                if chap and chap.id:
                                    with get_session() as db:
                                        db_chap = db.query(ChapterRecord).filter_by(id=chap.id).first()
                                        tx_id = db_chap.transcription_id if db_chap else None
                                    if tx_id:
                                        chap_record = tx_repo.get_by_id(tx_id)
                                        if chap_record:
                                            chap_record["file_name"] = f"{record['file_name']} (Chapter {ch_idx}: {chap.title})"
                                            transcripts_to_load.append(chap_record)
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

    # Link conversation to study session if new
    if project_id is not None and resume_id is None:
        from audiobench.core.db_session import get_session
        from audiobench.storage.models import StudySession
        
        conv_id = chat_repo.create_conversation(
            model=session.model,
            title=f"Study Project #{project_id} - Session {current_session_number}",
            session_type="study"
        )
        session._conversation_id = conv_id
        
        with get_session() as db:
            sess = (
                db.query(StudySession)
                .filter_by(project_id=project_id)
                .filter(StudySession.closed_at.is_(None))
                .first()
            )
            if sess:
                sess.conversation_id = conv_id
                db.commit()

    if transcripts_to_load:
        session.load_transcripts(transcripts_to_load)

    from audiobench.chat.chat_repl import ChatREPL
    repl = ChatREPL(
        session, 
        tx_repo, 
        chat_repo, 
        settings, 
        session_type="study" if project_id else "chat",
        project_id=project_id,
        current_session_number=current_session_number,
    )
    repl.run(resume_id=session.conversation_id)

