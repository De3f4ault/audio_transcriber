"""Management CLI commands for the semantic memory daemon."""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import click
from rich.console import Console

from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.daemon.client import DaemonClient

logger = get_logger("cli.daemon")
console = Console()


@click.group(name="daemon")
def daemon_group() -> None:
    """Manage the background semantic memory server."""
    pass


@daemon_group.command(name="start")
def start_daemon() -> None:
    """Start the daemon in the foreground (usually fork-started by factory)."""
    from audiobench.daemon.server import run

    console.print("[dim]Starting audiobench memory daemon...[/dim]")
    try:
        run()
    except Exception as e:
        logger.exception("Daemon crashed")
        console.print(f"[bold red]Fatal error:[/bold red] {e}")


@daemon_group.command(name="stop")
def stop_daemon() -> None:
    """Stop the running daemon gracefully."""
    settings = get_settings()
    pid_path = Path(settings.daemon_pid_path)

    if not pid_path.exists():
        console.print("[yellow]Daemon is not running (no PID file found).[/yellow]")
        return

    try:
        pid = int(pid_path.read_text().strip())
    except Exception:
        console.print("[red]Failed to read PID file. It may be corrupted.[/red]")
        pid_path.unlink(missing_ok=True)
        return

    console.print(f"Sending SIGTERM to daemon (PID {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        console.print("[yellow]Process not found. Cleaning up stale PID file.[/yellow]")
        pid_path.unlink(missing_ok=True)
        return
    except PermissionError:
        console.print(
            "[bold red]Permission denied. Are you the owner of the daemon process?[/bold red]"
        )
        return

    # Wait for graceful exit
    for _ in range(50):
        if not pid_path.exists():
            console.print("[bold green]Daemon stopped successfully.[/bold green]")
            return
        time.sleep(0.1)

    console.print(
        "[yellow]Daemon did not clean up PID file in time. It may still be shutting down.[/yellow]"
    )


@daemon_group.command(name="status")
def daemon_status() -> None:
    """Check if daemon is running and view its statistics."""
    settings = get_settings()
    client = DaemonClient()

    with console.status("Pinging daemon..."):
        if not client.ping():
            console.print("[yellow]Daemon is not running or socket is unreachable.[/yellow]")
            if Path(settings.daemon_pid_path).exists():
                console.print("[dim]Note: PID file exists but daemon is dead (stale).[/dim]")
            return

    try:
        stats = client.status()
        console.print("\n[bold green]Daemon is ONLINE[/bold green]")
        console.print(f"[bold]Socket:[/bold] {settings.daemon_socket_path}")
        console.print(f"[bold]Uptime:[/bold] {stats.get('uptime_seconds', 0)}s")
        console.print(f"[bold]Model:[/bold] {stats.get('embedding_model_version', 'unknown')}")
        console.print(f"[bold]Nodes Stored:[/bold] {stats.get('total_nodes', 0)}")
    except Exception as e:
        console.print(f"[bold red]Failed to retrieve status:[/bold red] {e}")


@daemon_group.command(name="reindex")
def reindex_daemon() -> None:
    """Force the daemon to re-embed all nodes with the current model.

    Works with or without the daemon running — if the daemon is unavailable,
    models are loaded in-process for the duration of the reindex.
    """
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

    from audiobench.core.db_session import get_session
    from audiobench.daemon.factory import get_daemon_client
    from audiobench.memory.enums import SourceType
    from audiobench.storage.models import ExpressionRecord

    client = get_daemon_client()

    with get_session() as session:
        expressions = session.query(ExpressionRecord).all()

    if not expressions:
        console.print("[yellow]No expressions found in the database to reindex.[/yellow]")
        return

    console.print(f"[bold]Reindexing {len(expressions)} expression(s) into LanceDB…[/bold]")
    errors: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding", total=len(expressions))

        for expr in expressions:
            try:
                source_type = SourceType(expr.source_type)
                client.embed(
                    expression_id=expr.id,
                    content=expr.content,
                    source_type=source_type,
                    speaker=expr.speaker,
                )
            except Exception as e:
                errors.append(f"Expression {expr.id}: {e}")
                logger.warning("Failed to embed expression %d: %s", expr.id, e)
            finally:
                progress.advance(task)

    if errors:
        console.print(f"[yellow]Completed with {len(errors)} error(s):[/yellow]")
        for err in errors[:5]:
            console.print(f"  [dim red]{err}[/dim red]")
        if len(errors) > 5:
            console.print(f"  [dim]… and {len(errors) - 5} more[/dim]")
    else:
        console.print(
            f"[bold green]✓ Reindexed {len(expressions)} expression(s) successfully.[/bold green]"
        )


@daemon_group.command(name="index")
@click.option("--tx-id", type=int, default=None, help="Transcription ID to index. Omit for all unindexed.")
def index_transcripts(tx_id: int | None) -> None:
    """Register and embed unindexed transcripts into the semantic memory store.

    Runs the consistency sweep in-process — works even if the daemon is busy
    or not running. Use --tx-id to target a single transcript.
    """
    from audiobench.core.db_session import get_session
    from audiobench.memory.chunking import content_aware_router
    from audiobench.memory.enums import SourceType
    from audiobench.memory.memory_store import MemoryStore
    from audiobench.memory.singletons import pre_warm_retrieval_pipeline
    from audiobench.storage.expression_repository import ExpressionRepository
    from audiobench.storage.models import TranscriptionRecord

    with get_session() as session:
        if tx_id is not None:
            records = session.query(TranscriptionRecord).filter_by(id=tx_id).all()
            if not records:
                console.print(f"[red]Transcription #{tx_id} not found.[/red]")
                return
        else:
            records = (
                session.query(TranscriptionRecord)
                .filter(TranscriptionRecord.is_indexed == 0)
                .all()
            )

        if not records:
            console.print("[green]All transcripts are already indexed.[/green]")
            return

        console.print(f"[bold]Loading embedding model…[/bold]")
        pre_warm_retrieval_pipeline()
        store = MemoryStore()
        expr_repo = ExpressionRepository()

        console.print(f"[bold]Indexing {len(records)} transcript(s)…[/bold]")
        from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Indexing", total=len(records))
            success = 0
            for rec in records:
                try:
                    text = rec.full_text or ""
                    if not text.strip():
                        rec.is_indexed = 1
                        progress.advance(task)
                        continue
                    progress.update(task, description=f"[dim]{rec.file_name[:40]}[/dim]")
                    chunks = content_aware_router(text)
                    for chunk in chunks:
                        expr = expr_repo.register(
                            content=chunk.content,
                            source_type=SourceType.AUDIO_TRANSCRIPT.value,
                            source_id=rec.id,
                            speaker=chunk.speaker,
                        )
                        store.write_node(
                            expression_id=expr.id,
                            content=expr.content,
                            source_type=SourceType.AUDIO_TRANSCRIPT.value,
                            speaker=chunk.speaker,
                        )
                    rec.is_indexed = 1
                    success += 1
                except Exception as e:
                    logger.error("Failed to index transcript %d: %s", rec.id, e)
                    console.print(f"  [red]✗ #{rec.id} {rec.file_name}: {e}[/red]")
                finally:
                    progress.advance(task)
            session.commit()

        console.print(f"[bold green]✓ Indexed {success}/{len(records)} transcript(s).[/bold green]")
