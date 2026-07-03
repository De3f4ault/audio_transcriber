"""Interactive Chat REPL Loop."""

from __future__ import annotations

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown as RichMarkdown
from rich.padding import Padding
from rich.panel import Panel
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
from audiobench.chat.providers.ollama_provider import AIError
from audiobench.core.settings import get_settings
from audiobench.core.db_engine import init_db
from audiobench.chat.chat_store import ChatRepository
from audiobench.storage.repository import TranscriptionRepository
from audiobench.chat.chat_session import ChatSession
from audiobench.chat.providers.ollama_provider import OllamaClient

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


class ChatREPL:
    """Manages the interactive chat loop, slash commands, and rendering."""

    def __init__(
        self,
        session=None,
        tx_repo=None,
        chat_repo=None,
        settings=None,
        session_type: str = "chat",
        preloaded_fragments=None,
        preloaded_title: str | None = None,
        model: str | None = None,
        temperature: float = 0.3,
        think: bool = True,
        resume_id: int | None = None,
        project_id: int | None = None,
        current_session_number: int | None = None,
    ):
        init_db()
        self.settings = settings or get_settings()
        self.tx_repo = tx_repo or TranscriptionRepository()
        self.chat_repo = chat_repo or ChatRepository()

        if session is None:
            model_name = model or self.settings.ollama_model
            client = OllamaClient(
                base_url=self.settings.ollama_base_url,
                model=model_name,
            )
            self.session = ChatSession(
                client=client,
                chat_repo=self.chat_repo,
                model=model_name,
                temperature=temperature,
                conversation_id=resume_id,
                show_thinking=think,
            )
            if resume_id is not None:
                self.session.restore_from_db()
        else:
            self.session = session

        self.client = self.session._client
        self.temperature = self.session._temperature
        self.session_type = session_type
        self.preloaded_fragments = preloaded_fragments
        self.preloaded_title = preloaded_title
        self.project_id = project_id
        self.current_session_number = current_session_number
        self._last_hint_at: float = 0.0  # epoch timestamp of last BM25 hint

        # Create the initial conversation entry with specific session type
        if not self.session.conversation_id:
            title = self.preloaded_title or "New Chat"
            self.session._conversation_id = self.chat_repo.create_conversation(
                model=self.session.model,
                title=title,
                session_type=self.session_type,
            )
        else:
            # Update session type if resuming and different (or just leave it)
            pass

        import time as _time
        from pathlib import Path as _Path
        self._time = _time
        self._Path = _Path

        _history_file = self._Path.home() / ".cache" / "audiobench_chat_history"
        _history_file.parent.mkdir(parents=True, exist_ok=True)
        self._pt_session: PromptSession = PromptSession(
            history=FileHistory(str(_history_file)),
        )

    def _build_study_context(self) -> str:
        """Build a study context block from prior session memoirs.

        Compression degrades by age:
          - N-1 session: FULL (narrative + insights + threads)
          - N-2 to N-3:  DIGEST (truncated narrative + insights + threads)
          - N-4+:        KEY_ONLY (insights + threads only)

        Open threads are ALWAYS included at every compression level.
        Estimated token budget: < 8000 words.
        """
        if self.project_id is None:
            return ""

        from audiobench.core.db_session import get_session as _db
        from audiobench.storage.models import StudySession, ConversationSummary
        from audiobench.memory.memoir_writer import Memoir, CompressionLevel, compress_memoir
        import json

        current_n = self.current_session_number or 1

        with _db() as db:
            # Fetch all closed sessions for this project, ordered by session number
            prior_sessions = (
                db.query(StudySession)
                .filter(
                    StudySession.project_id == self.project_id,
                    StudySession.closed_at.isnot(None),
                )
                .order_by(StudySession.id)
                .all()
            )
            # For each session, load its memoir via ConversationSummary
            session_memoirs: list[tuple[int, Memoir]] = []
            for idx, s in enumerate(prior_sessions, 1):
                if s.memoir_id is None:
                    continue
                # Find ConversationSummary that linked to this expression
                cs = db.query(ConversationSummary).filter_by(
                    expression_id=s.memoir_id
                ).first()
                if cs is None:
                    continue
                memoir = Memoir(
                    narrative=cs.narrative,
                    key_insights=cs.key_insights,
                    open_threads=cs.open_threads,
                    refined_title=cs.refined_title,
                )
                session_memoirs.append((idx, memoir))

        if not session_memoirs:
            return ""

        parts: list[str] = []
        parts.append("# Prior Study Sessions\n")

        for session_num, memoir in session_memoirs:
            age = current_n - session_num  # 1 = N-1, 2 = N-2, ...
            if age == 1:
                level = CompressionLevel.FULL
            elif age <= 3:
                level = CompressionLevel.DIGEST
            else:
                level = CompressionLevel.KEY_ONLY

            title = memoir.refined_title or f"Session {session_num}"
            compressed = compress_memoir(memoir, level)
            parts.append(f"## Session {session_num}: {title}\n\n{compressed}\n")

        return "\n".join(parts)

    def _fetch_memory_hints(
        self,
        user_text: str,
        debounce_seconds: float = 30.0,
        top_k: int = 3,
    ) -> list[str]:
        """Silently run a BM25 search and return matching segment texts as hints.

        Only fires when:
          1. user_text ends with '?' (looks like a question)
          2. Enough time has passed since the last hint (debounce)

        Returns a list of short text hints (or empty list if suppressed).
        Never prints to stdout. Never raises.
        """
        import time

        # Only fire on questions
        stripped = user_text.strip()
        if not stripped.endswith("?"):
            return []

        # Debounce: don't fetch if a hint was already fetched recently
        now = time.time()
        if now - self._last_hint_at < debounce_seconds:
            return []

        try:
            from audiobench.memory.retrieval_streams import FTS5Stream
            from audiobench.memory.query_reformulator import ReformulatedQuery

            # Minimal reformulation: use the raw question as BM25 keywords
            # (strip the '?' and punctuation)
            keywords = stripped.rstrip("?").strip()
            rq = ReformulatedQuery(
                original=stripped,
                bm25_keywords=keywords,
                semantic_query=stripped,
                hyde_anchor=stripped,
            )
            hits = FTS5Stream().retrieve(rq, top_k=top_k)
            self._last_hint_at = now
            return [h.text for h in hits]
        except Exception:
            # Must never propagate
            return []

    def _handle_slash_command(self, cmd: str) -> bool:
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
                tid = int(arg.strip())
                record = self.tx_repo.get_by_id(tid)
                if not record:
                    console.print(f"  [{DIM}]Transcript #{tid} not found[/]")
                    return False
                self.session.load_transcripts([record])
                console.print(
                    f"  [{SUCCESS}]✓ Loaded #{tid} "
                    f"{record['file_name']} "
                    f"({record['word_count']:,} words)[/]"
                )
            else:
                console.print()
                for line in self.session.get_context_summary():
                    console.print(f"    {line}")
                console.print()

        elif command == "/load":
            if not arg or not arg.strip().isdigit():
                console.print(f"  [{DIM}]Usage: /load <transcript_id>[/]")
                return False
            tid = int(arg.strip())
            record = self.tx_repo.get_by_id(tid)
            if not record:
                console.print(f"  [{DIM}]Transcript #{tid} not found[/]")
                return False
            self.session.load_transcripts([record])
            console.print(
                f"  [{SUCCESS}]✓ Loaded #{tid} "
                f"{record['file_name']} "
                f"({record['word_count']:,} words)[/]"
            )

        elif command == "/clear":
            self.session.clear_history()
            console.print(
                f"  [{SUCCESS}]✓ Conversation cleared (new session #{self.session.conversation_id})[/]"
            )
            console.print(f"  [{DIM}]Context reset — use /load <ID> to add transcripts[/]")

        elif command == "/remove":
            if not arg or not arg.strip().isdigit():
                console.print(f"  [{DIM}]Usage: /remove <transcript_id>[/]")
                return False
            tid = int(arg.strip())
            if self.session.remove_transcript(tid):
                console.print(f"  [{SUCCESS}]✓ Removed transcript #{tid} from context[/]")
            else:
                console.print(f"  [{DIM}]Transcript #{tid} not in context[/]")

        elif command == "/model":
            if not arg:
                console.print(f"  [{DIM}]Current model: {self.session.model}[/]")
                console.print(f"  [{DIM}]Usage: /model <name>[/]")
                return False
            self.session.switch_model(arg.strip())
            console.print(f"  [{SUCCESS}]✓ Switched to {arg.strip()}[/]")

        elif command == "/think":
            self.session.show_thinking = not self.session.show_thinking
            state = "on" if self.session.show_thinking else "off"
            console.print(f"  [{SUCCESS}]✓ Thinking display: {state}[/]")

        elif command == "/history":
            convs = self.chat_repo.list_conversations(limit=10)
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
            console.print(f"  [{SUCCESS}]✓ Conversation #{self.session.conversation_id} saved[/]")

        elif command == "/export":
            if not self.session.messages:
                console.print(f"  [{DIM}]Nothing to export yet[/]")
                return False
            fname = arg.strip() if arg.strip() else None
            if not fname:
                slug = f"chat_{self.session.conversation_id or 'new'}_{int(self._time.time())}"
                fname = f"{slug}.md"
            path = self._Path(fname).expanduser()
            lines = [f"# Chat #{self.session.conversation_id or 'new'}\n"]
            lines.append(f"Model: {self.session.model}  \n")
            lines.append("---\n")
            for msg in self.session.messages:
                if msg["role"] == "user":
                    lines.append(f"**You:** {msg['content']}\n")
                elif msg["role"] == "assistant":
                    lines.append(f"**AI:**\n\n{msg['content']}\n")
                lines.append("---\n")
            path.write_text("\n".join(lines), encoding="utf-8")
            console.print(f"  [{SUCCESS}]✓ Exported to {path}[/]")

        elif command == "/retry":
            self.session._retry_requested = True
            return False

        elif command == "/compare":
            if not arg:
                cmp_model = getattr(self.session, "_compare_model", None)
                if cmp_model:
                    console.print(
                        f"  [{ACCENT}]⚡ Comparison mode ON[/]\n"
                        f"  [{DIM}]Primary:   {self.session.model}[/]\n"
                        f"  [{DIM}]Secondary: {cmp_model}[/]\n"
                        f"  [{DIM}]Use /compare off to disable[/]"
                    )
                else:
                    console.print(
                        f"  [{DIM}]Comparison mode is OFF[/]\n"
                        f"  [{DIM}]Usage: /compare <model> to enable[/]\n"
                        f"  [{DIM}]Example: /compare qwen4-next:110b-cloud[/]"
                    )
                return False
            if arg.strip().lower() == "off":
                old = getattr(self.session, "_compare_model", None)
                self.session._compare_model = None
                if old:
                    console.print(
                        f"  [{SUCCESS}]✓ Comparison mode OFF[/] [{DIM}](was comparing with {old})[/]"
                    )
                else:
                    console.print(f"  [{DIM}]Comparison mode was already off[/]")
                return False
            new_model = arg.strip()
            old = getattr(self.session, "_compare_model", None)
            self.session._compare_model = new_model
            if old and old != new_model:
                console.print(
                    f"  [{ACCENT}]⚡ Switched comparison:[/] [{DIM}]{old}[/] → [{BOLD}]{new_model}[/]"
                )
            else:
                console.print(
                    f"  [{ACCENT}]⚡ Comparison mode ON[/]\n"
                    f"  [{DIM}]Every prompt will compare {self.session.model} vs {new_model}[/]\n"
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
                tid = int(arg.strip())
                record = self.tx_repo.get_by_id(tid)
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
                console.print(f"    [{DIM}]#{b['id']}[/] {emoji} {time_str}  {b['name'][:40]}")
            console.print()

        else:
            console.print(f"  [{DIM}]Unknown command: {command} (type /help for commands)[/]")

        return False

    def _render_comparison_pair(self, msg_a: dict, msg_b: dict) -> None:
        """Render a comparison pair as side-by-side panels."""
        layout = Layout()
        layout.split_row(
            Layout(name="left"),
            Layout(name="right"),
        )
        for side, msg in [("left", msg_a), ("right", msg_b)]:
            parts = []
            if msg.get("thinking") and self.session.show_thinking:
                think_preview = msg["thinking"][:300]
                if len(msg["thinking"]) > 300:
                    think_preview += "…"
                parts.append(Text(f"💭 {think_preview}", style="dim italic"))
            parts.append(RichMarkdown(msg["content"], code_theme=CHAT_CODE_THEME))
            model_label = msg.get("model_name") or "Model"
            border = "cyan" if side == "left" else "magenta"
            layout[side].update(Panel(Group(*parts), title=model_label, border_style=border))
        console.print(layout)

    def _stream_and_render(self, user_text: str) -> None:
        """Send user input and render the streamed response."""
        console.print()
        try:
            thinking_parts: list[str] = []
            content_parts: list[str] = []
            token_count = 0
            t_start = self._time.monotonic()

            with Live(
                console=chat_console,
                refresh_per_second=8,
                transient=True,
            ) as live:
                for chunk in self.session.send(user_text):
                    thinking = chunk.get("thinking", "")
                    content = chunk.get("content", "")

                    if thinking:
                        thinking_parts.append(thinking)

                    if content:
                        content_parts.append(content)
                        token_count += 1

                    display_parts = []

                    if thinking_parts and self.session.show_thinking:
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
                        elapsed_so_far = self._time.monotonic() - t_start
                        tps_so_far = token_count / elapsed_so_far if elapsed_so_far > 0 else 0
                        display_parts.append(
                            Text(
                                f"\n  ▍ {token_count} tokens · {tps_so_far:.0f} tok/s",
                                style="dim",
                            ),
                        )

                    if display_parts:
                        live.update(Group(*display_parts))

            if content_parts:
                final_md = "".join(content_parts)
                chat_console.print(
                    Padding(
                        RichMarkdown(final_md, code_theme=CHAT_CODE_THEME),
                        (0, 0, 0, 0),
                    )
                )
            elif thinking_parts:
                final_md = "".join(thinking_parts)
                chat_console.print(
                    Padding(
                        RichMarkdown(final_md, code_theme=CHAT_CODE_THEME),
                        (0, 0, 0, 0),
                    )
                )

            self.session.finalize_response()

            elapsed = self._time.monotonic() - t_start
            if token_count > 0 and elapsed > 0:
                tps = token_count / elapsed
                console.print(
                    f"  [{DIM}]{token_count} tokens · {tps:.1f} tok/s · {elapsed:.1f}s[/]"
                )
            console.print()

        except KeyboardInterrupt:
            if content_parts:
                self.session.finalize_response()
            console.print()
            console.print(f"  [{DIM}]Generation interrupted[/]")
            console.print()

        except AIError as e:
            console.print(error_panel("AI Error", str(e)))
            console.print()

    def _compare_and_render(self, user_text: str, compare_model: str) -> None:
        """Run comparison between primary and secondary model."""
        console.print()
        try:
            from audiobench.chat.compare import ModelComparison

            cmp_messages = self.session._build_api_messages()
            cmp_messages.append({"role": "user", "content": user_text})

            comparison = ModelComparison(
                client=self.client,
                messages=cmp_messages,
                model_a=self.session.model,
                model_b=compare_model,
                temperature=self.temperature,
                show_thinking=self.session.show_thinking,
            )
            result = comparison.run()

            if not self.session.conversation_id:
                self.session._conversation_id = self.chat_repo.create_conversation(
                    model=self.session.model,
                    title="Model Comparison",
                    session_type=self.session_type,
                )

            self.chat_repo.add_message(self.session.conversation_id, "user", user_text)
            self.session._messages.append({"role": "user", "content": user_text})

            for side in ("model_a", "model_b"):
                res = result[side]
                self.chat_repo.add_message(
                    self.session.conversation_id,
                    "assistant",
                    res["content"],
                    thinking=res["thinking"],
                    model_name=res["model_name"],
                )
                self.session._messages.append(
                    {
                        "role": "assistant",
                        "content": res["content"],
                        "thinking": res["thinking"],
                        "model_name": res["model_name"],
                    }
                )

            elapsed = result["elapsed"]
            total_tokens = result["model_a"]["tokens"] + result["model_b"]["tokens"]
            tps = total_tokens / elapsed if elapsed > 0 else 0
            console.print(f"  [{DIM}]{total_tokens} tok · {tps:.0f} tok/s · {elapsed:.1f}s[/]")
            console.print()

            if self.session.turn_count <= 1:
                self.session._generate_title_async()

        except KeyboardInterrupt:
            console.print()
            console.print(f"  [{DIM}]Comparison interrupted[/]")
            console.print()

        except Exception as e:
            console.print(error_panel("Comparison Error", str(e)))
            console.print()

    def _read_multiline(self) -> str:
        """Read multi-line input via prompt_toolkit (Alt+Enter or \"\"\" to end)."""
        console.print(
            f'  [{DIM}]Multi-line mode — type """ on its own line or press Alt+Enter to submit:[/]'
        )
        try:
            text = self._pt_session.prompt(
                ANSI("\033[38;5;240m... \033[0m"),
                multiline=True,
            )
        except (EOFError, KeyboardInterrupt):
            return ""
        text = text.strip()
        if text.startswith('"""'):
            text = text[3:]
        if text.endswith('"""'):
            text = text[:-3]
        return text.strip()

    def _trigger_summary(self) -> None:
        """Trigger summary generation in a background thread."""
        import threading

        def run_summary():
            import json
            from audiobench.chat.summary_generator import SummaryGenerator
            from audiobench.daemon.factory import get_daemon_client
            from audiobench.memory.enums import SourceType
            from audiobench.storage.expression_repository import ExpressionRepository

            gen = SummaryGenerator()
            result = gen.generate(self.session.messages)
            if not result:
                return

            if result.refined_title:
                self.chat_repo.update_title(self.session.conversation_id, result.refined_title)

            expr_repo = ExpressionRepository()
            expr = expr_repo.register(
                content=result.narrative,
                source_type=SourceType.SESSION_SUMMARY.value,
                source_id=self.session.conversation_id,
                session_type="chat",
                session_id=self.session.conversation_id,
            )

            self.chat_repo.save_summary(
                conversation_id=self.session.conversation_id,
                narrative=result.narrative,
                drift_phases=json.dumps(result.drift_phases),
                key_insights=json.dumps(result.key_insights),
                open_threads=json.dumps(result.open_threads),
                refined_title=result.refined_title,
                generated_by=gen.model_name,
                expression_id=expr.id,
            )

            try:
                daemon = get_daemon_client()
                daemon.embed(
                    expression_id=expr.id,
                    content=result.narrative,
                    source_type=SourceType.SESSION_SUMMARY,
                )
            except Exception:
                pass

        thread = threading.Thread(target=run_summary, daemon=True)
        thread.start()

    def _push_exit_frame(self) -> None:
        if not self.session.conversation_id:
            return
        try:
            from audiobench.cli.repl.session import NavigationFrame, ReplSession
            ctx = click.get_current_context(silent=True)
            repl_session: ReplSession | None = None
            while ctx is not None:
                obj = getattr(ctx, "obj", None)
                if isinstance(obj, ReplSession):
                    repl_session = obj
                    break
                ctx = getattr(ctx, "parent", None)
            if repl_session is not None:
                repl_session.push_frame(NavigationFrame(
                    context="chat",
                    state={"conversation_id": self.session.conversation_id},
                    intent="mid-conversation",
                ))
        except Exception:
            pass

    def _render_history(self) -> None:
        if self.session.messages:
            console.print(f"  [{DIM}]─── Previous Messages ───[/]")
            console.print()
            msgs = self.session.messages
            i = 0
            while i < len(msgs):
                msg = msgs[i]
                if msg["role"] == "user":
                    console.print(f"  [{PROMPT}]>>> {msg['content']}[/]")
                    console.print()
                    i += 1
                elif msg["role"] == "assistant":
                    if (
                        i + 1 < len(msgs)
                        and msgs[i + 1]["role"] == "assistant"
                        and msg.get("model_name") != msgs[i + 1].get("model_name")
                    ):
                        self._render_comparison_pair(msg, msgs[i + 1])
                        console.print()
                        i += 2
                    elif msg["content"].strip():
                        if msg.get("thinking") and self.session.show_thinking:
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

    def run(self, resume_id: int | None = None) -> None:
        console.print()
        conv_label = f" [#{resume_id}]" if resume_id else ""
        if self.preloaded_title:
            console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — {self.preloaded_title}{conv_label}")
        else:
            console.print(f"  [{BOLD} {ACCENT}]{APP_NAME}[/] — AI Chat{conv_label}")
        
        console.print(f"  [{DIM}]{'─' * 44}[/]")
        console.print(f"    Model:    {self.session.model}")
        ctx_lines = self.session.get_context_summary()
        console.print(f"    Context:  {ctx_lines[0]}")
        for line in ctx_lines[1:]:
            console.print(f"              {line}")
        think_label = "on" if self.session.show_thinking else "off"
        console.print(f"    Thinking: {think_label}")
        if resume_id and self.session.turn_count > 0:
            console.print(f"    Resumed:  {self.session.turn_count} previous turn(s)")
        console.print(f"  [{DIM}]{'─' * 44}[/]")
        console.print()

        if resume_id:
            self._render_history()

        last_user_input: str | None = None
        self.session._retry_requested = False
        if not hasattr(self.session, "_compare_model"):
            self.session._compare_model = None

        while True:
            try:
                cmp_active = getattr(self.session, "_compare_model", None)
                if cmp_active:
                    prompt_str = ANSI("\033[38;5;214m⚡ >>> \033[0m")
                else:
                    prompt_str = ANSI("\033[38;5;48m>>> \033[0m")
                user_input = self._pt_session.prompt(prompt_str).strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                if self.session.conversation_id:
                    console.print(
                        f"  [{SUCCESS}]✓ Conversation "
                        f"#{self.session.conversation_id} saved "
                        f"({self.session.turn_count * 2} messages)[/]"
                    )
                    if self.session.turn_count >= 3:
                        self._trigger_summary()
                self._push_exit_frame()
                console.print(f"  [{DIM}]Goodbye![/]")
                console.print()
                break

            if not user_input:
                continue

            if user_input == '"""':
                user_input = self._read_multiline()
                if not user_input.strip():
                    continue

            if user_input.startswith("\\"):
                user_input = "/" + user_input[1:]
            if user_input.startswith("/"):
                should_exit = self._handle_slash_command(user_input)

                if getattr(self.session, "_retry_requested", False):
                    self.session._retry_requested = False
                    if last_user_input and self.session.messages:
                        self.session._messages = [m for m in self.session._messages if m != self.session._messages[-1]]
                        if self.session._messages and self.session._messages[-1]["role"] == "user":
                            self.session._messages.pop()
                        console.print(f"  [{DIM}]Regenerating...[/]")
                        self._stream_and_render(last_user_input)
                    else:
                        console.print(f"  [{DIM}]Nothing to retry[/]")
                    continue

                if should_exit:
                    if self.session.conversation_id:
                        console.print(
                            f"  [{SUCCESS}]✓ Conversation "
                            f"#{self.session.conversation_id} saved "
                            f"({self.session.turn_count * 2} messages)"
                            f"[/]"
                        )
                        if self.session.turn_count >= 3:
                            self._trigger_summary()
                    self._push_exit_frame()
                    console.print(f"  [{DIM}]Goodbye![/]")
                    console.print()
                    break
                continue

            last_user_input = user_input

            compare_model = getattr(self.session, "_compare_model", None)
            if compare_model:
                self._compare_and_render(user_input, compare_model)
            else:
                self._stream_and_render(user_input)
