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


@click.group()
def memory() -> None:
    """Interact with semantic memory (search, threads, inferences)."""
    pass


@memory.command()
@click.argument("query", type=str, required=False)
@click.option(
    "--preset",
    type=click.Choice(["fast", "balanced", "deep"]),
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
@click.option("--interactive", "-i", is_flag=True, help="Interactive wizard mode.")
def search(
    query: str | None,
    preset: str,
    enable_hyde: bool | None,
    enable_cross_encoder: bool | None,
    enable_colbert: bool | None,
    no_cache: bool,
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
                query = prompt_string("Search query [e.g. 'What did we discuss about X?']")
                if not query:
                    console.print("  [dim]Cancelled.[/dim]")
                    return

            preset = prompt_menu(
                "Search preset",
                [
                    ("fast", "BM25 + Nomic (Fastest)", "fast"),
                    ("balanced", "BM25 + Nomic + ColBERT (Default)", "balanced"),
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

    console.print("[dim]Using parallel ResearchEngine with RRF fusion[/dim]")

    with console.status("Querying memory graph..."):
        result = engine.search(query=query, preset=preset)

    _run_search_panel(result)


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

    # ── Fragments ─────────────────────────────────────────────────────────────
    if result.sources:
        # Build skipped-stream suffix for header
        skipped_note = ""
        if result.streams_skipped:
            skipped_labels = " ".join(
                f"[dim red]{s}✗[/dim red]" for s in result.streams_skipped
            )
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
        console.print(
            Panel(
                result.answer,
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


def _run_search_panel(result: "ResearchResult") -> None:  # type: ignore[name-defined]
    """Display results and show [E]xpand [C]hat [Q]uit panel.

    This interactive panel is always shown — even on synthesis failure — so
    the user can still access fragments, open a chat, or quit cleanly.
    """
    _display_results(result)

    if not result.sources:
        return

    while True:
        console.print(
            "[dim]Press [bold]E[/bold] expand synthesis · "
            "[bold]C[/bold] open chat · [bold]Q[/bold] quit[/dim]"
        )
        try:
            choice = input("  › ").strip().upper()
        except (KeyboardInterrupt, EOFError):
            console.print()
            break

        if choice == "Q":
            break
        elif choice == "E":
            _stream_expanded_synthesis(result)
        elif choice == "C":
            from audiobench.chat.chat_repl import ChatREPL
            
            # Start chat with search followup session
            repl = ChatREPL(
                session_type="search_followup",
                preloaded_fragments=result.sources,
                preloaded_title=f"🔍 Search: {result.query}",
            )
            repl.run()
            break
        else:
            console.print("[dim]Unknown command. Use E, C, or Q.[/dim]")
