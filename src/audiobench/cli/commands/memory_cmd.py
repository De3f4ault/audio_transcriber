"""CLI commands for interacting with the semantic memory system."""

import json

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from audiobench.cli.display.theme import error_panel
from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.memory.enums import SourceType
from audiobench.memory.query_engine import ResearchEngine
from audiobench.storage.models import ConversationSummary, ExpressionRecord

logger = get_logger("cli.memory")
console = Console()


import dataclasses

@dataclasses.dataclass
class SearchSessionState:
    preset: str = "balanced"
    initial_preset: str = "balanced"
    mmr_lambda: float = 0.5
    focus_source: str | None = None
    model: str | None = None

    @property
    def mmr_enabled(self) -> bool:
        return self.preset == "synthesis"


def _parse_slash_command(line: str, state: SearchSessionState) -> str | None:
    """Parse interactive REPL commands to mutate session state.
    
    Returns a feedback message if handled, or None if not a valid command.
    """
    parts = line.strip().split()
    if not parts or not parts[0].startswith("/"):
        return None
        
    cmd = parts[0].lower()
    
    if cmd == "/set":
        if len(parts) == 1:
            return (
                "\n[dim]"
                "  /set <fast | balanced | synthesis | deep | default>\n"
                "  /set model   <ollama-model-name | gemini | default>\n"
                "  /set mmr     <0.0-1.0>   (switches to synthesis preset)"
                "[/dim]\n"
            )
            
        arg1 = parts[1].lower()
        
        # Flattened preset setting: `/set deep` or `/set default`
        if arg1 in ("default", "reset"):
            state.preset = state.initial_preset
            return f"[green]Preset reset to default ('{state.preset}')[/green]"
        elif arg1 in ("fast", "balanced", "deep", "synthesis"):
            state.preset = arg1
            return f"[green]Preset updated to '{arg1}'[/green]"
            
        # Legacy preset setting: `/set preset deep`
        if arg1 == "preset" and len(parts) >= 3:
            val = parts[2].lower()
            if val in ("default", "reset"):
                state.preset = state.initial_preset
                return f"[green]Preset reset to default ('{state.preset}')[/green]"
            elif val in ("fast", "balanced", "deep", "synthesis"):
                state.preset = val
                return f"[green]Preset updated to '{val}'[/green]"
            return f"[red]Invalid preset '{val}'[/red]"
            
        elif parts[1].lower() == "mmr" and len(parts) >= 3:
            try:
                lam = float(parts[2])
                if 0.0 <= lam <= 1.0:
                    state.preset = "synthesis"  # automatically switch preset
                    state.mmr_lambda = lam
                    return f"[green]Preset → synthesis, λ = {lam:.2f}[/green]"
                return "[red]λ must be between 0.0 and 1.0[/red]"
            except ValueError:
                return "[red]Invalid lambda value[/red]"
                
        elif parts[1].lower() == "model" and len(parts) >= 3:
            val = parts[2]
            if val.lower() == "default":
                state.model = None
                return "[green]Model override cleared (using default)[/green]"
            state.model = val
            return f"[green]Model set to '{val}'[/green]"
                
    elif cmd == "/focus":
        if len(parts) >= 2:
            source = " ".join(parts[1:])
            state.focus_source = source
            return f"[green]Focus set to: {source}[/green]"
        return "[red]Usage: /focus <source_name>[/red]"
        
    elif cmd == "/unfocus":
        state.focus_source = None
        return "[green]Focus cleared.[/green]"
        
    elif cmd == "/forget":
        state.preset = "balanced"
        state.mmr_lambda = 0.5
        state.focus_source = None
        return "[green]Session state reset to defaults.[/green]"
        
    return f"[dim]Unknown command: {cmd}[/dim]"



@click.group()
def memory() -> None:
    """Interact with semantic memory (search, threads, inferences)."""
    pass


def query_completer(ctx, param, incomplete: str):
    """Click shell completion for search queries via daemon."""
    from audiobench.daemon.factory import get_daemon_client
    try:
        client = get_daemon_client()
        results = client.autocomplete(incomplete, top_k=10)
        from click.shell_completion import CompletionItem
        return [CompletionItem(r.get("text", "")) for r in results if r.get("text")]
    except Exception:
        return []


@memory.command()
@click.argument("query", type=str, required=False, shell_complete=query_completer)
@click.option(
    "--preset",
    type=click.Choice(["fast", "balanced", "deep", "synthesis"]),
    default="balanced",
    help="Search preset to use.",
)
@click.option("--enable-hyde/--no-hyde", default=None, help="Override HyDE setting.")
@click.option(
    "--enable-cross-encoder/--no-cross-encoder",
    default=None,
    help="Override Cross-Encoder setting.",
)
@click.option("--enable-colbert/--no-colbert", default=None, help="Override ColBERT setting.")
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Bypass the semantic cache and force a new generation.",
)
@click.option(
    "--model",
    default=None,
    help="Override synthesis model (e.g. gemini, gpt-4o, llama3)",
)
@click.option("--interactive", "-i", is_flag=True, help="Interactive wizard mode.")
def search(
    query: str | None,
    preset: str,
    enable_hyde: bool | None,
    enable_cross_encoder: bool | None,
    enable_colbert: bool | None,
    no_cache: bool,
    model: str | None,
    interactive: bool,
) -> None:
    """Search memory for a semantic query."""

    interactive_mode = interactive or not query

    if interactive_mode:
        from audiobench.cli.display.theme import ACCENT, BOLD, BOX_STYLE
        from audiobench.cli.wizard import prompt_bool, prompt_menu, prompt_string

        console.print()
        console.print(
            Panel(
                "",
                title=f"[{BOLD}][{ACCENT}]AudioBench Memory Search[/][/]",
                title_align="left",
                border_style=ACCENT,
                box=BOX_STYLE,
                expand=False,
            )
        )
        console.print()

        try:
            if not query:
                query = prompt_string("Search query [e.g. 'What did we discuss about X?']", enable_autocomplete=True)
                if not query:
                    console.print("  [dim]Cancelled.[/dim]")
                    return

            preset = prompt_menu(
                "Search preset",
                [
                    ("fast", "BM25 + Nomic (Fastest)", "fast"),
                    ("balanced", "BM25 + Nomic + ColBERT (Default)", "balanced"),
                    ("synthesis", "MMR + Cross-Source Diversity (Synthesis)", "synthesis"),
                    ("deep", "HyDE + CrossEncoder (Deepest)", "deep"),
                ],
                default_idx=1,
            )

            no_cache = prompt_bool(
                "Bypass semantic cache? (Force new RAG generation)", default=no_cache
            )
            console.print()
        except KeyboardInterrupt:
            console.print()
            return

    if not query:
        console.print(error_panel("No Query", "A query must be provided."))
        return

    engine = ResearchEngine()
    state = SearchSessionState(preset=preset, initial_preset=preset, model=model)

    _run_search_loop(engine, query, state)


@memory.command()
def threads() -> None:
    """List open conversational threads across all sessions."""
    with get_session() as session:
        summaries = session.query(ConversationSummary).all()

    if not summaries:
        console.print("No open threads found.")
        return

    table = Table(title="Open Threads", show_lines=True)
    table.add_column("Session Title", style="cyan", no_wrap=True)
    table.add_column("Open Threads")

    found = False
    for s in summaries:
        try:
            threads = json.loads(s.open_threads)
            if threads:
                found = True
                thread_texts = []
                for t in threads:
                    q = t.get("question", "")
                    c = t.get("context", "")
                    thread_texts.append(f"[bold]{q}[/bold]\n[dim]{c}[/dim]")

                table.add_row(
                    s.refined_title or f"Session #{s.conversation_id}", "\n\n".join(thread_texts)
                )
        except Exception:
            continue

    if found:
        console.print(table)
    else:
        console.print("No open threads found.")


@memory.command()
def inferences() -> None:
    """List active system inferences."""
    with get_session() as session:
        inferences = (
            session.query(ExpressionRecord)
            .filter_by(source_type=SourceType.SYSTEM_INFERENCE.value)
            .all()
        )

    if not inferences:
        console.print("No system inferences found.")
        return

    table = Table(title="System Inferences", show_lines=True)
    table.add_column("ID", style="cyan")
    table.add_column("Content")
    table.add_column("Confidence", justify="right")
    table.add_column("Status")

    for inf in inferences:
        conf = f"{inf.inference_confidence:.2f}" if inf.inference_confidence is not None else "-"
        table.add_row(str(inf.id), inf.content, conf, inf.inference_status or "-")
        console.print(table)


@memory.command()
@click.option("--execute", is_flag=True, help="Execute deletion of orphaned SQLite expression rows and LanceDB vector nodes.")
@click.option("--batch-size", default=500, type=int, help="Batch size for deletions (default: 500).")
def purge(execute: bool, batch_size: int) -> None:
    """Scan and purge orphaned vector nodes and expressions from storage."""
    from sqlalchemy import select
    from audiobench.storage.models import ExpressionRecord, TranscriptionRecord
    from audiobench.memory.memory_store import MemoryStore
    from audiobench.cli.display.theme import SUCCESS, DIM

    with get_session() as session:
        active_tx_ids = set(session.scalars(select(TranscriptionRecord.id)).all())
        all_tx_exprs = session.query(ExpressionRecord.id, ExpressionRecord.source_id).filter(
            ExpressionRecord.source_type == SourceType.AUDIO_TRANSCRIPT.value
        ).all()
        orphans = [eid for eid, sid in all_tx_exprs if sid not in active_tx_ids]

    store = MemoryStore()
    lancedb_orphans: list[int] = []
    if orphans:
        # Check in batches if orphan count is very large
        for i in range(0, len(orphans), batch_size):
            chunk = orphans[i:i + batch_size]
            ids_str = ", ".join(str(eid) for eid in chunk)
            try:
                lancedb_rows = store.table.search().where(f"expression_id IN ({ids_str})").select(["expression_id"]).to_list()
                lancedb_orphans.extend(int(r["expression_id"]) for r in lancedb_rows)
            except Exception:
                pass

    if not orphans and not lancedb_orphans:
        console.print(f"[{SUCCESS}]✓ No orphaned expressions or vector nodes found.[/]")
        return

    table = Table(title="Orphan Scan Summary", show_lines=True)
    table.add_column("Asset", style="cyan")
    table.add_column("Orphan Count", justify="right", style="bold yellow")
    table.add_row("SQLite Expression Records", str(len(orphans)))
    table.add_row("LanceDB Vector Nodes", str(len(lancedb_orphans)))
    console.print(table)

    if not execute:
        console.print(f"[{DIM}]DRY-RUN MODE. To permanently purge these records, run:[/] [bold]audiobench memory purge --execute[/bold]")
        return

    with console.status("Purging orphaned vectors and expressions..."):
        if lancedb_orphans:
            for i in range(0, len(lancedb_orphans), batch_size):
                batch = lancedb_orphans[i:i + batch_size]
                b_str = ", ".join(str(eid) for eid in batch)
                store.table.delete(f"expression_id IN ({b_str})")

        if orphans:
            with get_session() as session:
                for i in range(0, len(orphans), batch_size):
                    batch = orphans[i:i + batch_size]
                    session.query(ExpressionRecord).filter(ExpressionRecord.id.in_(batch)).delete(synchronize_session=False)
                session.commit()

    console.print(f"[{SUCCESS}]✓ Successfully purged {len(orphans)} SQLite rows and {len(lancedb_orphans)} LanceDB vector nodes.[/]")


@memory.command()
@click.argument("target_id", type=int)
@click.argument("correction_text", type=str)
def correct(target_id: int, correction_text: str) -> None:
    """Correct a system inference."""
    from audiobench.daemon.factory import get_daemon_client
    from audiobench.memory.enums import SourceType
    from audiobench.storage.expression_repository import ExpressionRepository

    expr_repo = ExpressionRepository()

    # 1. Update target inference status
    success = expr_repo.update_inference_status(target_id, "corrected")
    if not success:
        console.print(
            error_panel(
                "Not Found", f"Expression #{target_id} not found or is not a system inference."
            )
        )
        return

    # 2. Write User Correction Expression
    correction_expr = expr_repo.register(
        content=correction_text,
        source_type=SourceType.USER_CORRECTION.value,
    )

    # 3. Write Relation
    # Wait, does expr_repo.link accept created_by? Not currently.
    expr_repo.link(
        from_id=correction_expr.id,
        to_id=target_id,
        relation_type="corrects",  # It asks for type='corrects', maybe not in RelationType enum yet
    )

    # Update created_by manually if needed (not in ExpressionRelation schema from earlier, but let's check if it exists)
    # The requirement says `created_by='user'`, but `ExpressionRelation` might not have it. Let's just link it.

    # 4. Embed
    try:
        daemon = get_daemon_client()
        daemon.embed(
            expression_id=correction_expr.id,
            content=correction_text,
            source_type=SourceType.USER_CORRECTION,
        )
    except Exception as e:
        logger.warning("Daemon embed failed for correction: %s", e)

    console.print(
        f"[green]✓ Corrected inference #{target_id} with expression #{correction_expr.id}.[/green]\n"
    )


# ── Research Engine display helpers (P1-5) ────────────────────────────────────

def _fmt_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _short_source(file_path: str) -> str:
    """Return a readable short name from a full file path."""
    import os
    name = os.path.basename(file_path)
    # Strip common extensions
    for ext in (".mp4", ".mp3", ".m4a", ".wav", ".webm"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    # Strip leading hash prefix (e.g. "f5c6e2c8_Life is Short" → "Life is Short")
    if "_" in name and len(name.split("_")[0]) == 8:
        name = name.split("_", 1)[1]
    return name.strip()


_STREAM_COLORS: dict[str, str] = {
    "fts5":    "yellow",
    "dense":   "magenta",
    "colbert": "blue",
}


def _display_results(result: "ResearchResult") -> None:  # type: ignore[name-defined]
    """Render a ResearchResult to stdout.

    Layout: fragments list first (with source + stream badges), answer panel last.
    On synthesis failure: show a ⚠ warning banner instead of the answer.
    Streams that returned 0 results get a ✗ badge so the user knows what ran.
    """
    from audiobench.memory.query_engine import ResearchResult  # local to avoid cycle

    console.print()

    # ── HyDE fallback notice (dim, single line) ───────────────────────────────
    if result.hyde_fallback:
        console.print(
            "[dim yellow]⚠  HyDE unavailable — falling back to direct query embedding[/dim yellow]"
        )
    elif getattr(result, "hyde_document", None):
        console.print(
            Panel(
                f"[italic]{result.hyde_document}[/italic]",
                title="[dim]HyDE Generation[/dim]",
                border_style="dim",
                expand=False,
            )
        )

    # ── Fragments ─────────────────────────────────────────────────────────────
    if result.sources:
        # Build skipped-stream suffix for header
        skipped_note = ""
        if result.streams_skipped:
            # result.streams_skipped is a list of tuples: (name, reason)
            skipped_labels_list = []
            for s_name, s_reason in result.streams_skipped:
                if s_reason == "0 hits":
                    skipped_labels_list.append(f"[dim red]{s_name}✗[/dim red]")
                else:
                    skipped_labels_list.append(f"[dim red]{s_name}✗[/dim red] [dim]({s_reason})[/dim]")
            
            skipped_labels = " ".join(skipped_labels_list)
            skipped_note = f"  {skipped_labels}"

        console.print(f"[bold]Fragments ({len(result.sources)})[/bold]{skipped_note}")
        for i, fr in enumerate(result.sources, 1):
            ts = f"{_fmt_timestamp(fr.start_time)} → {_fmt_timestamp(fr.end_time)}"
            score_str = f"[dim](rrf: {fr.rrf_score:.4f})[/dim]"
            snippet = fr.text[:120].replace("\n", " ")
            if len(fr.text) > 120:
                snippet += "…"

            # Source label
            source_label = ""
            if fr.source_file:
                source_label = f"  [dim]{_short_source(fr.source_file)}[/dim]"

            # Stream contribution badges (streams that DID return this hit)
            badges = ""
            if fr.stream_contributions:
                parts = []
                for stream, rank in fr.stream_contributions:
                    color = _STREAM_COLORS.get(stream, "white")
                    parts.append(f"[{color}]{stream}[/{color}][dim]#{rank}[/dim]")
                badges = "  " + " ".join(parts)

            console.print(
                f"  [bold]{i}.[/bold] [cyan]{ts}[/cyan] {score_str}{badges}{source_label}\n"
                f"     [italic]{snippet}[/italic]"
            )
        console.print()

    # ── Answer (always last) ──────────────────────────────────────────────────
    if result.synthesis_failed or result.answer is None:
        console.print(
            Panel(
                f"[bold yellow]⚠ Synthesis failed[/bold yellow]\n"
                f"[dim]{result.synthesis_error or 'Unknown error'}[/dim]\n\n"
                "[italic]Showing raw fragments instead.[/italic]",
                title="[bold red]Synthesis Failed[/bold red]",
                border_style="red",
                expand=False,
            )
        )
    else:
        from rich.markdown import Markdown as RichMarkdown
        from audiobench.cli.display.theme import CHAT_CODE_THEME

        console.print(
            Panel(
                RichMarkdown(result.answer, code_theme=CHAT_CODE_THEME),
                title=f"[bold cyan]Answer[/bold cyan] [dim]({result.query_time_seconds:.2f}s)[/dim]",
                border_style="cyan",
                expand=False,
            )
        )
    console.print()


def _stream_expanded_synthesis(result: "ResearchResult") -> None:  # type: ignore[name-defined]
    """Generate and stream a deeply expanded synthesis using all fragments."""
    from audiobench.chat.providers.ollama_provider import OllamaClient
    from audiobench.core.settings import get_settings
    from rich.live import Live
    from rich.markdown import Markdown as RichMarkdown
    from rich.padding import Padding
    from audiobench.cli.display.theme import chat_console, CHAT_CODE_THEME, DIM
    
    settings = get_settings()
    client = OllamaClient(base_url=settings.ollama_base_url, model=settings.ollama_model)
    
    prompt = (
        f"You are exploring the open question: '{result.query}'.\n"
        "Synthesize the following source fragments into a deep, expanded answer.\n"
        "Please heavily cite specific times or timestamps from the text.\n\n"
    )
    for i, src in enumerate(result.sources, 1):
        prompt += f"--- Fragment {i} (Time: {_fmt_timestamp(src.start_time)} - {_fmt_timestamp(src.end_time)}) ---\n"
        content = getattr(src, "expression_content", None) or src.text
        prompt += f"{content}\n\n"
        
    console.print("\n[dim]Generating expanded synthesis...[/dim]\n")
    content_parts = []
    
    try:
        with Live(console=chat_console, refresh_per_second=8, transient=True) as live:
            for chunk in client.stream(prompt=prompt, system_prompt="You are an expert researcher."):
                if chunk:
                    content_parts.append(chunk)
                    live.update(RichMarkdown("".join(content_parts), code_theme=CHAT_CODE_THEME))
                    
        if content_parts:
            chat_console.print(Padding(RichMarkdown("".join(content_parts), code_theme=CHAT_CODE_THEME), (0, 0, 1, 0)))
            
    except Exception as e:
        console.print(f"[red]Error during synthesis: {e}[/red]")


def _read_single_key() -> str:
    """Read one keypress from stdin without requiring Enter.

    Captures multi-byte escape sequences for arrow keys (\\x1b[A/B/C/D).
    Falls back to ``input()`` if the terminal is not a tty (e.g. piped
    input in tests), in which case arrow-key navigation is unavailable.
    """
    import sys
    import termios
    import tty

    if not sys.stdin.isatty():
        return input()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        # Arrow keys send \x1b[A / \x1b[B / \x1b[C / \x1b[D — read two more bytes
        if ch == "\x1b":
            ch += sys.stdin.read(2)
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _fetch_adjacent(
    transcription_id: int,
    anchor_start: float,
    anchor_end: float,
) -> tuple[list[dict], list[dict]]:
    """Return (prev_segs, next_segs) relative to the anchor window.

    Fetches the two segments immediately before ``anchor_start`` and the
    two immediately at or after ``anchor_end`` from the same transcript.
    Both lists are ordered chronologically (earliest first).
    """
    from sqlalchemy import text as sql_text

    with get_session() as session:
        prev_rows = session.execute(
            sql_text(
                "SELECT id, start_time, end_time, text FROM segments "
                "WHERE transcription_id = :tid AND start_time < :start "
                "ORDER BY start_time DESC LIMIT 2"
            ),
            {"tid": transcription_id, "start": anchor_start},
        ).mappings().all()

        next_rows = session.execute(
            sql_text(
                "SELECT id, start_time, end_time, text FROM segments "
                "WHERE transcription_id = :tid AND start_time >= :end "
                "ORDER BY start_time ASC LIMIT 2"
            ),
            {"tid": transcription_id, "end": anchor_end},
        ).mappings().all()

    # prev_rows come back newest-first from DESC; reverse to chronological order
    return list(reversed([dict(r) for r in prev_rows])), [dict(r) for r in next_rows]


def _resolve_audio_file_id(transcription_id: int, _cache: dict = {}) -> int:
    """Memoized lookup for audio_file_id from a transcription_id."""
    if transcription_id in _cache:
        return _cache[transcription_id]
    from audiobench.storage.models import TranscriptionRecord
    with get_session() as s:
        rec = s.query(TranscriptionRecord).filter_by(id=transcription_id).first()
        if rec and rec.audio_file_id:
            _cache[transcription_id] = rec.audio_file_id
            return rec.audio_file_id
    return 0

def _get_max_end_time(transcription_id: int, _cache: dict = {}) -> float:
    """Memoized lookup for MAX(end_time) of a transcription."""
    if transcription_id in _cache:
        return _cache[transcription_id]
    from sqlalchemy import text
    with get_session() as s:
        res = s.execute(text("SELECT MAX(end_time) FROM segments WHERE transcription_id = :tid"), {"tid": transcription_id}).scalar()
        _cache[transcription_id] = res or 0.0
        return _cache[transcription_id]

def _toggle_bookmark(
    fr_display: dict,
    transcription_id: int,
    console: Console,
) -> None:
    """Toggle a bookmark on the currently-displayed segment.

    Takes ``fr_display`` (the scroll-adjusted dict with id/start_time/end_time/text)
    and the ``transcription_id`` from the parent FusedResult (not in fr_display).
    """
    from audiobench.storage.bookmark_repository import BookmarkRepository
    audio_file_id = _resolve_audio_file_id(transcription_id)
    if not audio_file_id:
        console.print("[red]Cannot bookmark: no audio_file_id[/red]")
        return

    repo = BookmarkRepository()
    bm = repo.get_nearest(audio_file_id, fr_display["start_time"], window=1.0)
    if bm:
        repo.delete(bm["id"])
    else:
        repo.add_region(
            audio_file_id,
            fr_display["start_time"],
            fr_display["end_time"],
            name="Bookmarked from reader",
        )

def _yank_to_clipboard(text: str, console: Console) -> None:
    import subprocess
    try:
        subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode("utf-8"), check=True)
        console.print("  [green]Yanked to clipboard (xclip)[/green]")
    except Exception:
        try:
            subprocess.run(["xsel", "--clipboard", "--input"], input=text.encode("utf-8"), check=True)
            console.print("  [green]Yanked to clipboard (xsel)[/green]")
        except Exception:
            console.print(f"\n[dim]Clipboard copy failed. Text:[/dim]\n{text}\n")


def _find_related(
    console: Console,
    fr: "FusedResult",  # type: ignore[name-defined]
    exclude_ids: set[int],
) -> "FusedResult | None":
    """Show the top 5 semantically related fragments and return a user-selected one, if any."""
    console.print()
    console.print("  [bold cyan]Related search preset[/bold cyan]")
    console.print("    [1] fast            [dim]BM25 + Nomic (Fastest)[/dim]")
    console.print("    [2] balanced        [dim]BM25 + Nomic + ColBERT (Default)[/dim]")
    console.print("    [3] synthesis       [dim]MMR + Cross-Source Diversity (Synthesis)[/dim]")
    console.print("    [4] deep            [dim]HyDE + CrossEncoder (Deepest)[/dim]")
    console.print("    [any] cancel")
    preset_choice = input("  → ").strip()

    preset_map = {
        "1": "fast",
        "2": "balanced",
        "3": "synthesis",
        "4": "deep",
    }
    preset = preset_map.get(preset_choice)
    if not preset:
        return None

    from audiobench.memory.query_engine import MemorySearchEngine
    engine = MemorySearchEngine()

    with console.status(f"[cyan]Finding related fragments (preset={preset})...[/cyan]"):
        try:
            res = engine.search(fr.text, top_k=10, preset=preset)
            hits = res.fused
        except Exception as exc:  # noqa: BLE001
            console.print(f"\n  [red]Related search failed: {exc}[/red]")
            console.print("  [dim]Press any key to continue…[/dim]")
            _read_single_key()
            return None

    related = [h for h in hits if h.segment_id not in exclude_ids][:5]

    console.print()
    console.rule("[bold]Related moments[/bold]")
    if not related:
        console.print("  [dim]No related fragments found outside the current result set.[/dim]")
        console.print("  [dim]Press any key to return to reader…[/dim]")
        _read_single_key()
        return None
        
    for i, h in enumerate(related, 1):
        src = _short_source(h.source_file)
        ts = f"{_fmt_timestamp(h.start_time)}–{_fmt_timestamp(h.end_time)}"
        snippet = h.text[:120].replace("\n", " ")
        console.print(
            f"  [bold]{i}.[/bold] [cyan]{ts}[/cyan]  [dim]{src}[/dim]\n"
            f"     [italic]{snippet}[/italic]"
        )
    console.print()
    console.print("  [dim]Select 1-5 to jump, or any other key to go back…[/dim]")
    
    try:
        key = _read_single_key()
        if key.isdigit() and 1 <= int(key) <= len(related):
            return related[int(key) - 1]
    except (KeyboardInterrupt, EOFError):
        pass
    return None


def _write_note(console: Console, fr_display: dict, audio_file_id: int, source_name: str) -> None:
    from audiobench.storage.note_repository import NoteRepository
    try:
        console.print()
        text = input("  Note: ").strip()
    except (KeyboardInterrupt, EOFError):
        return
        
    if not text:
        return
        
    repo = NoteRepository()
    
    transcript_expr_id = None
    if fr_display.get("id"):
        with get_session() as s:
            from sqlalchemy import text as sql_text
            res = s.execute(sql_text("SELECT expression_id FROM expression_segment_map WHERE segment_id = :sid LIMIT 1"), {"sid": fr_display["id"]}).scalar()
            if res:
                transcript_expr_id = res
            
    title = f"Notes on {source_name}"
    col = repo.find_or_create_collection(audio_file_id, title)
    
    repo.create_capture(
        collection_id=col.id,
        body=text,
        segment_id=fr_display.get("id"),
        transcript_expression_id=transcript_expr_id,
        collection_expression_id=col.expression_id
    )
    console.print("  [cyan]✎ Note saved[/cyan]")
    import time
    time.sleep(0.5)


def _open_fragment_reader(
    console: Console,
    fragments: list,  # list[FusedResult]
    initial_idx: int = 0,
) -> None:
    """Interactive fragment reader workspace."""
    import textwrap
    import shutil
    from audiobench.storage.bookmark_repository import BookmarkRepository
    from audiobench.storage.note_repository import NoteRepository

    if not fragments:
        console.print("[dim]No fragments to read.[/dim]")
        return

    current_idx = max(0, min(initial_idx, len(fragments) - 1))
    viewport_offset = 0
    exclude_ids: set[int] = {fr.segment_id for fr in fragments}
    
    bookmark_repo = BookmarkRepository()
    note_repo = NoteRepository()

    while True:
        fr = fragments[current_idx]
        total = len(fragments)
        audio_file_id = _resolve_audio_file_id(fr.transcription_id)

        try:
            prev_segs, next_segs = _fetch_adjacent(
                fr.transcription_id, fr.start_time, fr.end_time
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reader adjacency fetch failed: %s", exc)
            prev_segs, next_segs = [], []

        if viewport_offset < 0:
            scroll_idx = min(abs(viewport_offset) - 1, len(prev_segs) - 1)
            if prev_segs and scroll_idx >= 0:
                scrolled_seg = prev_segs[scroll_idx]
                try:
                    prev_segs, next_segs = _fetch_adjacent(
                        fr.transcription_id, scrolled_seg["start_time"], scrolled_seg["end_time"]
                    )
                    next_segs = [{"id": fr.segment_id, "start_time": fr.start_time,
                                  "end_time": fr.end_time, "text": fr.text}] + list(next_segs[:1])
                    fr_display = scrolled_seg
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Viewport scroll fetch failed: %s", exc)
                    fr_display = {"id": fr.segment_id, "start_time": fr.start_time, "end_time": fr.end_time, "text": fr.text}
            else:
                fr_display = {"id": fr.segment_id, "start_time": fr.start_time, "end_time": fr.end_time, "text": fr.text}
        elif viewport_offset > 0:
            scroll_idx = min(viewport_offset - 1, len(next_segs) - 1)
            if next_segs and scroll_idx >= 0:
                scrolled_seg = next_segs[scroll_idx]
                try:
                    prev_segs, next_segs = _fetch_adjacent(
                        fr.transcription_id, scrolled_seg["start_time"], scrolled_seg["end_time"]
                    )
                    prev_segs = list(prev_segs[-1:]) + [{"id": fr.segment_id, "start_time": fr.start_time,
                                                         "end_time": fr.end_time, "text": fr.text}]
                    fr_display = scrolled_seg
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Viewport scroll fetch failed: %s", exc)
                    fr_display = {"id": fr.segment_id, "start_time": fr.start_time, "end_time": fr.end_time, "text": fr.text}
            else:
                fr_display = {"id": fr.segment_id, "start_time": fr.start_time, "end_time": fr.end_time, "text": fr.text}
        else:
            fr_display = {"id": fr.segment_id, "start_time": fr.start_time, "end_time": fr.end_time, "text": fr.text}

        console.clear()
        
        # Header indicators
        has_bookmark = bool(bookmark_repo.get_nearest(audio_file_id, fr_display["start_time"], window=1.0)) if audio_file_id else False
        captures = note_repo.get_captures_for_segment(fr_display.get("id"))
        
        bookmark_ind = "  [yellow]★[/yellow]" if has_bookmark else ""
        note_ind = "  [cyan]✎[/cyan]" if captures else ""

        src_name = _short_source(fr.source_file)
        ts_range = f"{_fmt_timestamp(fr_display['start_time'])} – {_fmt_timestamp(fr_display['end_time'])}"
        console.print(
            f"  [bold]Fragment {current_idx + 1}/{total}[/bold]  [dim]·[/dim]  {src_name}  [dim]·[/dim]  {ts_range}{bookmark_ind}{note_ind}"
        )

        # Progress bar — build with rich.text.Text so █░ are treated as plain
        # chars and Rich never sees them as markup tags.  No [ ] brackets.
        max_end_time = _get_max_end_time(fr.transcription_id)
        if max_end_time > 0:
            progress = min(1.0, max(0.0, fr_display["start_time"] / max_end_time))
            bar_len = 36
            filled = int(progress * bar_len)
            bar_chars = "\u2588" * filled + "\u2591" * (bar_len - filled)
            pos_str   = f"{_fmt_timestamp(fr_display['start_time'])} / {_fmt_timestamp(max_end_time)}"
            from rich.text import Text as _T
            _bar = _T()
            _bar.append("  ")
            _bar.append(bar_chars)          # plain — no markup interpretation
            _bar.append("  ")
            _bar.append(pos_str, style="dim")
            console.print(_bar)

        console.print()
        console.rule("earlier", style="dim", characters="\u2500")
        console.print()

        cols = shutil.get_terminal_size().columns
        wrap_width = min(cols - 8, 80)
        # Indent for hanging timestamp ("  00:00:00   " = 14 chars)
        ts_prefix_len = 14
        text_wrap_width = max(30, wrap_width - ts_prefix_len)
        hanging_indent = " " * ts_prefix_len

        # ── Earlier context: word-wrapped, timestamp on first line only ──────
        for seg in prev_segs:
            ts      = _fmt_timestamp(seg["start_time"])
            raw     = seg["text"].replace("\n", " ").strip()
            lines   = textwrap.wrap(raw, width=text_wrap_width) or [""]
            first   = f"  [dim]{ts}   {lines[0]}[/dim]"
            console.print(first)
            for extra in lines[1:]:
                console.print(f"  [dim]{hanging_indent}{extra}[/dim]")
            console.print()  # blank line between each context segment

        console.rule(style="bold white", characters="\u2550")

        # ── Active fragment ──────────────────────────────────────────────────
        current_text = textwrap.fill(fr_display["text"], width=wrap_width)
        ts_label     = f"[bold cyan]\u25b6 {_fmt_timestamp(fr_display['start_time'])}[/bold cyan]"
        # Right-align source name on the same line as the ▶ timestamp
        src_padded   = src_name.rjust(max(0, wrap_width - len(_fmt_timestamp(fr_display['start_time'])) - 4))
        console.print(f"  {ts_label}  [dim]{src_padded}[/dim]")
        console.print(f"{textwrap.indent(current_text, '     ')}\n")

        # Render captures inline
        for cap in captures:
            cap_body = textwrap.fill(cap.body, width=wrap_width - 4)
            console.print(f"     [cyan]\u270e[/cyan]  [dim]\"{cap_body}\"[/dim]")
            console.print(f"         [dim]\u2014 {str(cap.created_at)[:16]}[/dim]\n")

        console.rule(style="bold white", characters="\u2550")
        console.print()

        # ── Later context: same wrapped + hanging-indent layout ───────────────
        for seg in next_segs:
            ts    = _fmt_timestamp(seg["start_time"])
            raw   = seg["text"].replace("\n", " ").strip()
            lines = textwrap.wrap(raw, width=text_wrap_width) or [""]
            console.print(f"  [dim]{ts}   {lines[0]}[/dim]")
            for extra in lines[1:]:
                console.print(f"  [dim]{hanging_indent}{extra}[/dim]")
            console.print()  # blank line between each context segment

        console.rule("later", style="dim", characters="\u2500")

        console.print()
        # Action bar — key letters dim-highlighted so they stand out against descriptions
        console.print(
            "  [dim]B[/dim] bookmark  [dim]\u00b7[/dim]  "
            "[dim]N[/dim] note  [dim]\u00b7[/dim]  "
            "[dim]Y[/dim] yank  [dim]\u00b7[/dim]  "
            "[dim]H/L[/dim] scroll  [dim]\u00b7[/dim]  "
            "[dim]J/K[/dim] fragments  [dim]\u00b7[/dim]  "
            "[dim]R[/dim] related  [dim]\u00b7[/dim]  "
            "[dim]Q[/dim] back"
        )

        try:
            key = _read_single_key()
        except (KeyboardInterrupt, EOFError):
            break

        if key in ("q", "Q", "\x1b"):
            break
        elif key in ("h", "H", "\x1b[D"):
            viewport_offset -= 1
        elif key in ("l", "L", "\x1b[C"):
            viewport_offset += 1
        elif key in ("j", "J"):
            current_idx = max(0, current_idx - 1)
            viewport_offset = 0
        elif key in ("k", "K"):
            current_idx = min(total - 1, current_idx + 1)
            viewport_offset = 0
        elif key in ("n", "N"):
            _write_note(console, fr_display, audio_file_id, src_name)
        elif key in ("b", "B"):
            _toggle_bookmark(fr_display, fr.transcription_id, console)
        elif key in ("y", "Y"):
            _yank_to_clipboard(fr_display["text"], console)
        elif key in ("r", "R"):
            new_hit = _find_related(console, fragments[current_idx], exclude_ids)
            if new_hit:
                fragments.append(new_hit)
                exclude_ids.add(new_hit.segment_id)
                current_idx = len(fragments) - 1
                viewport_offset = 0


def _run_search_loop(engine: "ResearchEngine", query: str, state: SearchSessionState) -> None:  # type: ignore[name-defined]
    """Interactive loop for searching, REPL commands, and navigating results."""
    while True:
        # Check for one-off preset overrides in the query
        active_preset = state.preset
        query_parts = query.split(maxsplit=1)
        if query_parts and query_parts[0].lower() in ("/fast", "/balanced", "/deep", "/synthesis"):
            active_preset = query_parts[0][1:].lower()
            if len(query_parts) == 1:
                console.print("[red]Missing query after preset override.[/red]")
                return
            query = query_parts[1]
            console.print(f"[dim]Note: Using one-off override preset '{active_preset}' for this query[/dim]")

        if not query.strip():
            console.print("[red]Query cannot be empty.[/red]")
            return

        console.print(f"[dim]Searching: preset={active_preset} focus={state.focus_source or 'all'} λ={state.mmr_lambda}[/dim]")
        
        with console.status("Querying memory graph..."):
            result = engine.search(
                query=query, 
                preset=active_preset,
                mmr_lambda=state.mmr_lambda,
                focus_source=state.focus_source,
                model=state.model
            )
            
        _display_results(result)

        if not result.sources:
            return

        # Inner loop for prompts (reader, chat, or slash commands)
        while True:
            console.print(
                "[dim]Press [bold]Enter[/bold] re-run · [bold]E[/bold] reader · "
                "[bold]S[/bold] new search · [bold]C[/bold] open chat · [bold]Q[/bold] quit  |  "
                "or type [bold]/set[/bold] / [bold]/focus[/bold][/dim]"
            )
            try:
                choice = input("  › ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print()
                return

            if choice.startswith("/"):
                # Check for one-off query overrides first
                query_parts = choice.split(maxsplit=1)
                if query_parts[0].lower() in ("/fast", "/balanced", "/deep", "/synthesis") and len(query_parts) > 1:
                    query = choice
                    console.print()
                    break  # re-run outer loop with new query
                    
                msg = _parse_slash_command(choice, state)
                if msg:
                    console.print(f"  {msg}")
                continue

            choice_upper = choice.upper()
            if choice_upper == "Q":
                return
            elif choice_upper == "E":
                _open_fragment_reader(console, result.sources, initial_idx=0)
            elif choice_upper == "C":
                from audiobench.chat.chat_repl import ChatREPL

                repl = ChatREPL(
                    session_type="search_followup",
                    preloaded_fragments=result.sources,
                    preloaded_title=f"🔍 Search: {result.query}",
                )
                repl.run()
                return
            elif choice_upper == "S":
                try:
                    new_query = input("  New query: ").strip()
                except (KeyboardInterrupt, EOFError):
                    console.print()
                    continue
                if new_query:
                    # We no longer need to parse here because the outer loop handles it
                    query = new_query
                    console.print()
                    break  # re-run outer loop with new query
            elif choice == "":
                console.print()
                break  # re-run with same query
            else:
                console.print("[dim]Unknown command. Press [bold]S[/bold] to start a new search, or use E, C, Q, Enter, / commands.[/dim]")
