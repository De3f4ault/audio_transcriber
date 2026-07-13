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
    """Start the daemon in the foreground (supervised)."""
    import audiobench.supervisor as supervisor

    console.print("[dim]Starting supervised audiobench memory daemon...[/dim]")
    try:
        from audiobench.observatory.db import init_journal_db
        from audiobench.observatory.subscriber import get_subscriber
        from audiobench.events import get_bus

        init_journal_db()
        get_bus().on("*", get_subscriber().record)

        supervisor.start("daemon")
        from audiobench.supervisor.commands import wait_all
        wait_all()
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted. Shutting down...[/yellow]")
        from audiobench.supervisor.watcher import _SHUTDOWN_EVENT
        _SHUTDOWN_EVENT.set()
        supervisor.stop("daemon")
    except Exception as e:
        logger.exception("Daemon crashed")
        console.print(f"[bold red]Fatal error:[/bold red] {e}")


@daemon_group.command(name="stop")
def stop_daemon() -> None:
    """Stop the running daemon gracefully via supervisor."""
    import audiobench.supervisor as supervisor
    console.print("Sending SIGTERM to daemon via supervisor...")
    supervisor.stop("daemon")
    console.print("[bold green]Stop requested.[/bold green]")


@daemon_group.command(name="status")
def daemon_status() -> None:
    """Check if daemon is running and view its statistics."""
    settings = get_settings()
    client = DaemonClient()

    with console.status("Pinging daemon..."):
        if not client.ping():
            console.print("[yellow]Daemon is not running or socket is unreachable.[/yellow]")
            try:
                from audiobench.supervisor.registry import get_process
                import psutil
                p = get_process("daemon")
                if p:
                    state = p["state"].lower()
                    if state == "fatal":
                        console.print(f"[bold red]Supervisor reports daemon is in FATAL state (restarts exhausted).[/bold red]")
                    elif state == "backoff":
                        console.print(f"[bold yellow]Supervisor reports daemon is in BACKOFF (restarting shortly).[/bold yellow]")
                    elif state == "stopped":
                        console.print(f"[dim]Supervisor reports daemon is gracefully STOPPED.[/dim]")
                    elif state == "running":
                        if p["pid"] and not psutil.pid_exists(p["pid"]):
                            console.print(f"[bold red]Supervisor reports RUNNING but PID {p['pid']} is dead (STALE state).[/bold red]")
                        else:
                            console.print(f"[bold red]Supervisor reports RUNNING but socket ping failed.[/bold red]")
            except ImportError:
                pass
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


@daemon_group.command(name="optimize")
def optimize_lancedb() -> None:
    """Optimize LanceDB tables: compact fragments and remove old versions.

    Sends the request to the running daemon if available (safe, coordinated).
    Falls back to in-process optimization if the daemon is not running.
    """
    client = DaemonClient()
    
    def _format_bytes(size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"

    def _print_result(result: dict) -> None:
        for table in result.get("tables_optimized", []):
            console.print(f"  [green]✓[/green] Optimized {table}")
            
        bytes_freed = result.get("bytes_freed", 0)
        freed_str = f" | Freed {_format_bytes(bytes_freed)}" if bytes_freed > 0 else ""
        
        cleared_writes = result.get("cleared_writes", 0)
        writes_str = f"Cleared {cleared_writes} fragmented writes" if cleared_writes > 0 else "No fragmented writes"
        
        console.print(f"[bold green]✓ Optimization complete in {result.get('duration_seconds', 0.0):.2f}s ({writes_str}{freed_str}).[/bold green]")

    if client.ping():
        console.print("Daemon is ONLINE — sending optimize request...")
        try:
            result = client.optimize()
            _print_result(result)
        except Exception as e:
            console.print(f"[bold red]Failed to optimize via daemon:[/bold red] {e}")
    else:
        console.print("Daemon is OFFLINE — running in-process optimization...")
        try:
            from audiobench.daemon.lancedb_optimizer import _do_optimize_all_tables
            result = _do_optimize_all_tables(triggered_by="cli_fallback")
            _print_result(result)
        except Exception as e:
            console.print(f"[bold red]Failed to optimize:[/bold red] {e}")
