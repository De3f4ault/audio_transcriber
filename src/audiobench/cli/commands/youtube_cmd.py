"""YouTube CLI commands."""

import click
from rich.console import Console

from audiobench.core.db_session import get_session
from audiobench.cli.display.theme import console, SUCCESS, WARNING, ERROR, ACCENT, DIM, error_panel
from audiobench.youtube.fetcher import extract_video_id, fetch_and_register
from audiobench.youtube.search import (
    resolve_channel,
    search_videos,
    write_last_search,
    load_search_result,
    SearchStateExpiredError,
    YouTubeAPIError
)
from audiobench.storage.models import YouTubeChannel

@click.group(name="youtube", invoke_without_command=True)
@click.option("--job-id", type=int, hidden=True)
@click.pass_context
def youtube_group(ctx, job_id):
    """YouTube integration: search and fetch videos."""
    ctx.ensure_object(dict)
    ctx.obj["job_id"] = job_id
    
    if ctx.invoked_subcommand is None:
        from audiobench.youtube.repl import run_youtube_repl
        run_youtube_repl()

def _resolve_fetch_target(arg: str) -> str:
    """Return a canonical YouTube URL or ID from either a number (search result) or a URL."""
    if arg.isdigit():
        try:
            return load_search_result(int(arg))
        except SearchStateExpiredError as e:
            raise click.BadParameter(str(e))
    
    if "youtube.com" in arg or "youtu.be" in arg:
        return arg
        
    # Assume it's already an 11 char ID
    if len(arg) == 11:
        return arg
        
    raise click.BadParameter(f"'{arg}' is not a valid YouTube URL or video ID.")


@youtube_group.command("_fetch_internal", hidden=True)
@click.argument("video_id", type=str)
@click.pass_context
def fetch_internal_cmd(ctx, video_id: str):
    job_id = ctx.obj.get("job_id")
    with get_session() as session:
        try:
            audio_record, queue_record = fetch_and_register(video_id, session)
            if audio_record:
                console.print(f"[{SUCCESS}]Saved[/]  {audio_record.file_path}")
                console.print(f"[{DIM}]Audio file #{audio_record.id} · Queue Job #{queue_record.id} queued for transcription[/]")
            else:
                console.print(f"[{DIM}]Already in library, skipping.[/]")
        except Exception as e:
            console.print(error_panel("Fetch failed", str(e)))
            import sys
            sys.exit(1)


@youtube_group.command("fetch")
@click.argument("target", type=str)
def fetch_cmd(target: str):
    """Fetch a YouTube video and queue it for transcription."""
    from audiobench.jobs.runner import submit_job
    from audiobench.storage.models import AudioFileRecord
    
    try:
        resolved_url = _resolve_fetch_target(target)
        video_id = extract_video_id(resolved_url)
    except Exception as e:
        console.print(error_panel("Error", str(e)))
        return

    # Uniqueness check before dispatching
    with get_session() as session:
        existing = session.query(AudioFileRecord).filter_by(youtube_video_id=video_id).first()
        if existing:
            console.print(f"[{DIM}]Already in library:[/] #{existing.id} — {existing.file_name}")
            return

    job_id = submit_job(["youtube", "_fetch_internal", video_id])
    console.print(f"[{SUCCESS}]Download queued[/] · Job #{job_id}")
    console.print(f"\nRun [bold]audiobench jobs fg {job_id}[/bold] to follow progress")


@youtube_group.command("search")
@click.argument("query", type=str)
@click.option("--channel", type=str, help="Restrict to channel name or URL")
@click.option("--limit", type=int, default=15, help="Max results (default 15)")
@click.option("--sort", type=click.Choice(["relevance", "date", "viewCount", "rating", "title"]), default="relevance", help="Sort order")
@click.option("--after", type=str, help="Published after date (YYYY-MM-DD)")
@click.option("--before", type=str, help="Published before date (YYYY-MM-DD)")
@click.option("--no-cache", is_flag=True, help="Bypass channel resolution cache")
def search_cmd(
    query: str, 
    channel: str | None, 
    limit: int,
    sort: str,
    after: str | None,
    before: str | None,
    no_cache: bool
):
    """Search YouTube for videos."""
    from rich.table import Table
    
    channel_id = None
    with get_session() as session:
        if channel:
            try:
                if "youtube.com" in channel or channel.startswith("UC"):
                    channel_id = _resolve_fetch_target(channel)
                    channel_title = channel
                else:
                    channel_id, channel_title = resolve_channel(channel, session, no_cache=no_cache)
                console.print(f"[{DIM}]Resolving channel...[/]  {channel_title}  ({channel_id})")
            except Exception as e:
                console.print(error_panel("Error resolving channel", str(e)))
                return
                
        try:
            with console.status(f"[{DIM}]Searching YouTube...[/]") as status:
                def progress(msg):
                    status.update(f"[{DIM}]{msg}[/]")
                results = search_videos(
                    query, 
                    channel_id, 
                    limit, 
                    progress_callback=progress,
                    sort=sort,
                    after=after,
                    before=before
                )
                
            if not results:
                console.print(f"[{DIM}]No results found for '{query}'[/]")
                return
                
            write_last_search(results)
            
            import shutil
            term_width = shutil.get_terminal_size().columns
            title_max = max(20, term_width - 40)
            
            table = Table(show_header=True, header_style=f"bold {ACCENT}", box=None)
            table.add_column("#", style="dim", width=4, no_wrap=True)
            table.add_column("Title")
            table.add_column("Duration", justify="right", no_wrap=True)
            table.add_column("Published", justify="right", no_wrap=True)
            
            for r in results:
                title = r.title if len(r.title) <= title_max else r.title[:title_max-1] + "…"
                table.add_row(str(r.n), title, r.duration_str, r.published_at)
                
            console.print()
            console.print(table)
            console.print(f"\n[{DIM}]fetch a result:[/]  audiobench youtube fetch <number>")
            console.print(f"[{DIM}]results expire in 1 hour[/]")
            
        except YouTubeAPIError as e:
            console.print(error_panel("Search failed", str(e)))
        except ValueError as e:
            console.print(error_panel("Error", str(e)))


@youtube_group.command("channels")
@click.option("--refresh", type=str, help="Force re-resolve a specific channel query.")
def channels_cmd(refresh: str | None):
    """List cached channel resolutions."""
    with get_session() as session:
        if refresh:
            normalized = refresh.strip().lower()
            deleted = session.query(YouTubeChannel).filter_by(query=normalized).delete()
            session.commit()
            if deleted:
                console.print(f"[{SUCCESS}]Cleared cache[/] for channel query '{refresh}'")
            else:
                console.print(f"[{DIM}]No cached entry found[/] for '{refresh}'")
            return
            
        channels = session.query(YouTubeChannel).order_by(YouTubeChannel.query).all()
        if not channels:
            console.print(f"[{DIM}]No cached channel resolutions yet.[/]")
            return
            
        from rich.table import Table
        table = Table(title="Cached channel resolutions", title_style=f"bold {ACCENT}", show_header=True, header_style=f"bold {ACCENT}", box=None)
        table.add_column("Query", style="cyan")
        table.add_column("Title")
        table.add_column("Channel ID", style="dim")
        
        for c in channels:
            table.add_row(c.query, c.title, c.channel_id)
            
        console.print()
        console.print(table)
        console.print(f"\n[{DIM}]--refresh <name>  to force re-resolve[/]")

