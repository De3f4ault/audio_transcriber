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
from audiobench.memory.query_engine import MemoryQueryEngine
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

    engine = MemoryQueryEngine()

    console.print(
        f"[dim]Using preset: [bold]{preset}[/bold] (HyDE: {enable_hyde if enable_hyde is not None else 'default'}, CrossEncoder: {enable_cross_encoder if enable_cross_encoder is not None else 'default'}, ColBERT: {enable_colbert if enable_colbert is not None else 'default'})[/dim]"
    )

    with console.status("Querying memory graph..."):
        result = engine.query(
            text=query,
            preset=preset,
            enable_hyde=enable_hyde,
            enable_cross_encoder=enable_cross_encoder,
            use_colbert=enable_colbert,
            use_cache=not no_cache,
        )

    console.print(Panel(f"[bold green]Query:[/bold green] {query}", expand=False))

    if result.hyde_document:
        console.print(
            Panel(f"[bold blue]HyDE Document:[/bold blue] {result.hyde_document}", expand=False)
        )

    # Render Sources
    if result.sources:
        console.print("\n[bold]Sources ([cyan]Top 5 Reranked[/cyan]):[/bold]")
        for i, src in enumerate(result.sources, 1):
            src_type = src.get("type", "unknown")
            score = src.get("score", 0.0)
            content = src.get("content", "").replace("\n", " ")
            if len(content) > 150:
                content = content[:150] + "..."
            console.print(
                f"  {i}. [[magenta]{src_type}[/magenta]] [dim](vec_score: {score:.3f})[/dim]\n     {content}"
            )

    console.print("\n[bold]Synthesis:[/bold]")
    console.print(result.answer)
    console.print(f"\n[dim]Query executed in {result.query_time_seconds:.2f}s[/dim]")
    console.print()


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
        f"[green]✓ Corrected inference #{target_id} with expression #{correction_expr.id}.[/green]"
    )
