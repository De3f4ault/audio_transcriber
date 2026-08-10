"""Interactive REPL loop for YouTube integration."""

import dataclasses
import shlex
from typing import Any

from rich.console import Console
from rich.table import Table

from audiobench.cli.display.theme import console, SUCCESS, WARNING, ERROR, ACCENT, DIM, error_panel
from audiobench.core.db_session import get_session
from audiobench.youtube.search import search_videos, resolve_channel, VideoResult

@dataclasses.dataclass
class YouTubeSessionState:
    channel_id: str | None = None
    channel_title: str | None = None
    sort: str = "relevance"
    after: str | None = None
    before: str | None = None
    last_results: list[VideoResult] = dataclasses.field(default_factory=list)


def _handle_slash_command(user_input: str, state: YouTubeSessionState) -> bool:
    """Handle /slash commands. Returns True if REPL should exit."""
    parts = shlex.split(user_input)
    cmd = parts[0].lower()

    if cmd in ("/exit", "/quit"):
        return True

    if cmd == "/clear":
        state.channel_id = None
        state.channel_title = None
        state.sort = "relevance"
        state.after = None
        state.before = None
        console.print(f"[{SUCCESS}]Filters cleared.[/]")
        return False

    if cmd == "/channel":
        if len(parts) < 2:
            console.print(f"[{WARNING}]Usage:[/] /channel <name or url>")
            return False
            
        channel_arg = parts[1]
        with get_session() as session:
            try:
                from audiobench.cli.commands.youtube_cmd import _resolve_fetch_target
                if "youtube.com" in channel_arg or channel_arg.startswith("UC"):
                    channel_id = _resolve_fetch_target(channel_arg)
                    channel_title = channel_arg
                else:
                    channel_id, channel_title = resolve_channel(channel_arg, session)
                state.channel_id = channel_id
                state.channel_title = channel_title
                console.print(f"[{SUCCESS}]Channel filter set:[/] {channel_title} ({channel_id})")
            except Exception as e:
                console.print(error_panel("Channel resolution failed", str(e)))
        return False

    if cmd == "/sort":
        if len(parts) < 2 or parts[1] not in ["relevance", "date", "viewCount", "rating", "title"]:
            console.print(f"[{WARNING}]Usage:[/] /sort [relevance|date|viewCount|rating|title]")
            return False
        state.sort = parts[1]
        console.print(f"[{SUCCESS}]Sort order set to:[/] {state.sort}")
        return False

    if cmd == "/after":
        if len(parts) < 2:
            console.print(f"[{WARNING}]Usage:[/] /after YYYY-MM-DD")
            return False
        state.after = parts[1]
        console.print(f"[{SUCCESS}]Date filter set:[/] published after {state.after}")
        return False

    if cmd == "/before":
        if len(parts) < 2:
            console.print(f"[{WARNING}]Usage:[/] /before YYYY-MM-DD")
            return False
        state.before = parts[1]
        console.print(f"[{SUCCESS}]Date filter set:[/] published before {state.before}")
        return False

    if cmd in ("/help", "/commands"):
        console.print(f"\n[{ACCENT}]Commands:[/]")
        console.print(f"  [cyan]<query>[/]        Search YouTube")
        console.print(f"  [cyan]fetch <N>[/]      Download result #N")
        console.print(f"  [cyan]info <N>[/]       Show metadata for result #N\n")
        console.print(f"[{ACCENT}]Filters:[/]")
        console.print(f"  [cyan]/channel <name>[/] Restrict to channel")
        console.print(f"  [cyan]/sort <order>[/]   Set sort order")
        console.print(f"  [cyan]/after <date>[/]   Published after YYYY-MM-DD")
        console.print(f"  [cyan]/before <date>[/]  Published before YYYY-MM-DD")
        console.print(f"  [cyan]/clear[/]          Clear all filters")
        console.print(f"  [cyan]/exit[/]           Leave YouTube REPL\n")
        return False

    console.print(f"[{WARNING}]Unknown command:[/] {cmd}")
    return False


def _handle_action(parts: list[str], state: YouTubeSessionState) -> None:
    action = parts[0].lower()
    
    if len(parts) < 2 or not parts[1].isdigit():
        console.print(f"[{WARNING}]Usage:[/] {action} <number>")
        return
        
    idx = int(parts[1])
    target_result = None
    for r in state.last_results:
        if r.n == idx:
            target_result = r
            break
            
    if not target_result:
        console.print(f"[{WARNING}]Invalid result number:[/] {idx}")
        return

    if action in ("fetch", "download", "dl"):
        from audiobench.jobs.runner import submit_job
        from audiobench.storage.models import AudioFileRecord
        
        with get_session() as session:
            existing = session.query(AudioFileRecord).filter_by(youtube_video_id=target_result.video_id).first()
            if existing:
                console.print(f"[{DIM}]Already in library:[/] #{existing.id} — {existing.file_name}")
                return

        job_id = submit_job(["youtube", "_fetch_internal", target_result.video_id])
        console.print(f"[{SUCCESS}]Download queued[/] · Job #{job_id}")
        console.print(f"\nRun [bold]audiobench jobs fg {job_id}[/bold] to follow progress")

    elif action == "info":
        console.print(f"\n[{ACCENT}]Title:[/] {target_result.title}")
        console.print(f"[{ACCENT}]Video ID:[/] {target_result.video_id}")
        console.print(f"[{ACCENT}]Duration:[/] {target_result.duration_str}")
        console.print(f"[{ACCENT}]Published:[/] {target_result.published_at}")
        console.print(f"\n[{DIM}]Description:[/]")
        console.print(target_result.description)
        console.print()


def _run_interactive_search(query: str, state: YouTubeSessionState) -> None:
    from audiobench.youtube.search import YouTubeAPIError
    import shutil
    
    try:
        with console.status(f"[{DIM}]Searching YouTube...[/]") as status:
            def progress(msg):
                status.update(f"[{DIM}]{msg}[/]")
                
            results = search_videos(
                query, 
                state.channel_id, 
                15, 
                progress_callback=progress,
                sort=state.sort,
                after=state.after,
                before=state.before
            )
            
        if not results:
            console.print(f"[{DIM}]No results found for '{query}'[/]")
            return
            
        state.last_results = results
        
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
        console.print(f"\n[{DIM}]fetch <number>  |  info <number>[/]")
        
    except YouTubeAPIError as e:
        console.print(error_panel("Search failed", str(e)))
    except Exception as e:
        console.print(error_panel("Error", str(e)))


def run_youtube_repl():
    """Main entrypoint for the YouTube interactive loop."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.styles import Style
    
    state = YouTubeSessionState()
    
    style = Style.from_dict({
        "prompt": "ansicyan bold",
    })
    session = PromptSession(style=style)
    
    console.print(f"[{ACCENT}]YouTube Interactive Mode[/]")
    console.print(f"[{DIM}]Type a search query, or use /commands for help.[/]\n")
    
    while True:
        try:
            # Show active channel in prompt if set
            prefix = f"[{state.channel_title}] " if state.channel_title else ""
            prompt_text = [("class:prompt", f"{prefix}youtube> ")]
            user_input = session.prompt(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
            
        if not user_input:
            continue
            
        if user_input.startswith("/"):
            if _handle_slash_command(user_input, state):
                break
            continue
            
        parts = user_input.split()
        if len(parts) >= 2 and parts[0].lower() in ("fetch", "download", "dl", "info"):
            _handle_action(parts, state)
            continue
            
        _run_interactive_search(user_input, state)
