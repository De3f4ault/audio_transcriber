"""REPL dot-commands — quick actions on the current transcript context.

Provides:
    - DOT_COMMANDS: Registry of available dot-commands with descriptions
    - handle_dot_command(): Router for all dot-command input
    - Individual implementations: _dot_stats, _dot_show, _dot_play, etc.
"""

from __future__ import annotations

import contextlib
import difflib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from audiobench.cli.display.theme import (
    ACCENT,
    BOLD,
    DIM,
    SUCCESS,
    WARNING,
    console,
    format_duration,
    make_table,
)
from audiobench.cli.repl.dispatch import dispatch_command, print_context_summary
from audiobench.cli.repl.session import ReplSession

# ── Dot Commands ────────────────────────────────────────────


DOT_COMMANDS = {
    ".stats": "Word count, duration, language, model",
    ".show": "Display full transcript with timestamps",
    ".segments": "Timestamped segment breakdown",
    ".vocab": "Word frequency analysis (top 20)",
    ".info": "Full metadata for current transcript",
    ".find": 'Search within transcript: .find "keyword"',
    ".export": "Re-export: .export srt  |  .export json",
    ".ask": 'AI question: .ask "What was decided?"',
    ".chat": "Start AI chat with this transcript",
    ".summarize": "AI summary of this transcript",
    ".play": "Play audio: .play  |  .play 01:25",
    ".edit": "Edit transcript text in $EDITOR",
    ".path": "Show source audio file path",
    ".open": "Open source audio in default player",
    ".use": "Switch context: .use 42",
    ".clear": "Clear context (return to bare prompt)",
    ".next": "Jump to next transcript in history",
    ".search": 'Search all transcripts: .search "keyword"',
    ".diarize": "Interactive speaker diarization wizard",
    ".clean": "Interactive transcript cleaner",
    ".config": "Interactive configuration wizard",
    ".bookmark": "Interactive bookmark manager",
    ".chapters": "List all chapters for the focused audio file",
    ".fc": "Focus on a specific chapter: .fc 1",
    ".ufc": "Clear chapter focus",
    ".help": "Show this dot-command list",
}

from audiobench.core.platform import SUPPORTS_BACKGROUND_JOBS
if SUPPORTS_BACKGROUND_JOBS:
    DOT_COMMANDS.update({
        ".jobs": "List background jobs: .jobs",
        ".watch": "Tail a background job log: .watch <id>",
    })


# ── Typo Correction ─────────────────────────────────────────


def _suggest_dot_command(typed: str) -> str | None:
    """Find closest dot-command match using difflib."""
    all_cmds = list(DOT_COMMANDS.keys())
    # Try prefix match first
    prefix_matches = [c for c in all_cmds if c.startswith(typed[:3])]
    if prefix_matches:
        return prefix_matches[0]
    # Fuzzy match
    close = difflib.get_close_matches(typed, all_cmds, n=1, cutoff=0.5)
    return close[0] if close else None


# ── Main Router ─────────────────────────────────────────────


def handle_dot_command(cmd: str, session: ReplSession) -> None:
    """Handle a dot-command that operates on the current context."""
    parts = cmd.strip().split(None, 1)
    command = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    # Strip surrounding quotes from arg (fixes double-quote bug)
    if arg and len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in ('"', "'"):
        arg = arg[1:-1]

    # ── Context-free dot-commands ──

    if command == ".use":
        if not arg or not arg.isdigit():
            console.print(f"  [{DIM}]Usage: .use <transcript_id>[/]")
            return
        session.set_context(int(arg))
        if session.focus:
            print_context_summary(session)
        return

    if command == ".clear":
        session.focus = None
        session._history_cursor = -1
        console.print(f"  [{DIM}]Context cleared.[/]")
        return

    if command == ".chapters":
        _dot_chapters(session)
        return
        
    if command in (".fc", ".focus-chapter"):
        _dot_focus_chapter(session, arg)
        return
        
    if command in (".ufc", ".unfocus-chapter"):
        session.clear_chapter_focus()
        console.print(f"  [{DIM}]Chapter focus cleared.[/]")
        return

    if command == ".config":
        dispatch_command(session, ["config", "--interactive"])
        return

    if command == ".bookmark":
        dispatch_command(session, ["bookmark", "--interactive"])
        return

    if command == ".next":
        _navigate_context(session, direction=1)
        return

    if command == ".prev":
        _navigate_context(session, direction=-1)
        return

    if command == ".recent":
        dispatch_command(session, ["history", "--tail", "5"])
        return

    if command == ".search":
        if not arg:
            console.print(f'  [{DIM}]Usage: .search "keyword"[/]')
            return
        dispatch_command(session, ["search", arg])
        return

    if command in (".jobs", ".watch"):
        if not SUPPORTS_BACKGROUND_JOBS:
            console.print(f"  [{WARNING}]Background jobs are only supported on Linux/macOS.[/]")
            return
            
        if command == ".jobs":
            _dot_jobs()
        else:
            if not arg or not arg.isdigit():
                console.print(f"  [{DIM}]Usage: .watch <job_id>[/]")
                return
            _dot_watch(int(arg))
        return

    if command in (".help", ".commands", ".?"):
        print_dot_help()
        return

    if command in (".exit", ".quit"):
        # Redirect to /exit — user naturally types .exit
        console.print(f"  [{DIM}]Tip: Use /exit or exit to quit[/]")
        return

    # ── Context-required dot-commands ──

    if not session.focus:
        console.print(
            f"\n  [{WARNING}]No active focus.[/] Work on a file or transcript first:\n"
            f"    [{ACCENT}]work file.mp3[/]          Focus on an audio file\n"
            f"    [{ACCENT}].use <ID>[/]              Switch to a transcript\n"
            f"    [{ACCENT}].recent[/]                See recent transcriptions\n"
        )
        return

    # Commands that only need a File focus
    if command == ".info":
        _dot_info(session)
        return
    elif command == ".play":
        _dot_play(session, arg)
        return
    elif command == ".stop":
        _dot_stop(session)
        return
    elif command == ".path":
        _dot_path(session)
        return
    elif command == ".open":
        _dot_open(session)
        return
    elif command == ".chapters":
        _dot_chapters(session)
        return
    elif command in (".fc", ".focus-chapter"):
        _dot_focus_chapter(session, arg)
        return
    elif command in (".ufc", ".unfocus-chapter"):
        _dot_unfocus_chapter(session)
        return

    # Commands that strictly require a Transcript
    tx_id = session.last_id
    if not tx_id:
        console.print(f"  [{WARNING}]No transcript available for this file yet.[/] Run `transcribe` to create one.")
        return

    repo = session._get_repo()
    rec = repo.get_by_id(tx_id)

    if command == ".stats":
        _dot_stats(rec)
    elif command == ".show":
        _dot_show(rec)
    elif command == ".segments":
        _dot_segments(rec)
    elif command == ".edit":
        _dot_edit(session, rec)
    elif command == ".find":
        _dot_find(rec, arg)
    elif command == ".vocab":
        extra = arg.split() if arg else []
        dispatch_command(session, ["vocab", str(tx_id)] + extra)
    elif command == ".export":
        if not arg:
            dispatch_command(session, ["export", str(tx_id), "--interactive"])
            return
        dispatch_command(session, ["export", str(tx_id), "-f", arg])
    elif command == ".ask":
        if not arg:
            dispatch_command(session, ["ask", str(tx_id), "--interactive"])
            return
        dispatch_command(session, ["ask", str(tx_id), arg])
    elif command == ".chat":
        if not arg:
            dispatch_command(session, ["chat", str(tx_id), "--interactive"])
            return
        dispatch_command(session, ["chat", str(tx_id)])
    elif command == ".summarize":
        if not arg:
            dispatch_command(session, ["summarize", str(tx_id), "--interactive"])
            return
        dispatch_command(session, ["summarize", str(tx_id)])
    elif command == ".clean":
        dispatch_command(session, ["clean", str(tx_id), "--interactive"])
    elif command == ".diarize":
        _dot_diarize(session, rec)
    else:
        suggestion = _suggest_dot_command(command)
        if suggestion:
            console.print(
                f"  [{WARNING}]Unknown: {command}[/]\n"
                f"  [{DIM}]Did you mean [{ACCENT}]{suggestion}[/]?[/]"
            )
        else:
            console.print(
                f"  [{DIM}]Unknown: {command}  —  Type .help for available dot-commands[/]"
            )


# ── Dot-Command Implementations ─────────────────────────────


def _dot_stats(rec: dict) -> None:
    duration = rec.get("duration", 0) or 0
    mins = int(duration // 60)
    secs = int(duration % 60)
    console.print(f"\n  [{BOLD}][{ACCENT}]#{rec['id']}[/] — {rec.get('file_name', '?')}[/]")
    console.print(f"  [{DIM}]{'─' * 44}[/]")
    console.print(f"    Words:    [{ACCENT}]{rec.get('word_count', 0):,}[/]")
    console.print(f"    Duration: [{ACCENT}]{mins}m {secs}s[/]")
    console.print(f"    Segments: {rec.get('segment_count', 0)}")
    console.print(f"    Language: {rec.get('language') or 'auto-detected'}")
    console.print(f"    Model:    {rec.get('model') or '?'}")
    console.print(f"    Engine:   {rec.get('engine') or '?'}")
    console.print(f"    Created:  {rec.get('created_at', '?')}")
    console.print()


def _dot_show(rec: dict) -> None:
    console.print(f"\n  [{BOLD}][{ACCENT}]#{rec['id']}[/] — {rec.get('file_name', '?')}[/]\n")
    segments = rec.get("segments", [])
    if segments:
        for seg in segments:
            start = seg.get("start", 0) or 0
            ts = f"[{int(start // 60):02d}:{int(start % 60):02d}]"
            speaker = f" ({seg['speaker']})" if seg.get("speaker") else ""
            console.print(f"  [{DIM}]{ts}{speaker}[/] {seg.get('text', '').strip()}")
    else:
        text = rec.get("full_text", "")
        if text:
            console.print(f"  {text}")
        else:
            console.print(f"  [{DIM}]No text available[/]")
    console.print()


def _dot_segments(rec: dict) -> None:
    segments = rec.get("segments", [])
    if not segments:
        console.print(f"  [{DIM}]No segments available[/]")
        return
    table = make_table(
        f"Segments — #{rec['id']} ({len(segments)} total)",
        [
            ("Time", {"style": DIM, "width": 14}),
            ("Speaker", {"width": 10}),
            ("Text", {}),
        ],
    )
    for seg in segments:
        start = seg.get("start", 0) or 0
        end = seg.get("end", 0) or 0
        ts = (
            f"{int(start // 60):02d}:{int(start % 60):02d}→{int(end // 60):02d}:{int(end % 60):02d}"
        )
        speaker = seg.get("speaker") or "—"
        text = (seg.get("text", "") or "").strip()
        if len(text) > 80:
            text = text[:77] + "..."
        table.add_row(ts, speaker, text)
    console.print(table)


def _dot_info(session: ReplSession) -> None:
    if not session.focus:
        return
        
    repo = session._get_repo()
    
    # Always show File Metadata first if we have a file focus
    if session.focus.type == "file":
        audio_file = repo.get_audio_file(session.focus.id)
        if audio_file:
            table = make_table(
                f"File — {audio_file['file_name']}",
                [("Field", {"style": BOLD}), ("Value", {})],
            )
            for key in [
                "id", "file_path", "format", "duration_seconds", 
                "file_size_bytes", "transcript_count", "created_at"
            ]:
                val = audio_file.get(key, "—")
                if key == "duration_seconds" and val:
                    val = format_duration(val)
                elif key == "file_size_bytes" and val:
                    val = f"{val / (1024*1024):.1f} MB"
                table.add_row(key, str(val))
            console.print(table)
            
    # Then show Transcript Metadata if one exists
    tx_id = session.last_id
    if tx_id:
        rec = repo.get_by_id(tx_id)
        if rec:
            if session.focus.type == "file":
                console.print()
                
            tx_table = make_table(
                f"Transcript — #{rec['id']}",
                [("Field", {"style": BOLD}), ("Value", {})],
            )
            for key in [
                "id", "source", "language", "language_probability",
                "engine", "model", "duration", "word_count",
                "segment_count", "status", "created_at"
            ]:
                val = rec.get(key, "—")
                if key == "duration" and val:
                    val = format_duration(val)
                tx_table.add_row(key, str(val))
            console.print(tx_table)


def _dot_find(rec: dict, query: str) -> None:
    """Search within the current transcript's text."""
    if not query:
        console.print(f'  [{DIM}]Usage: .find "keyword"[/]')
        return

    segments = rec.get("segments", [])
    full_text = rec.get("full_text", "")
    query_lower = query.lower()

    if not full_text or query_lower not in full_text.lower():
        console.print(f'  [{DIM}]No matches for "{query}" in #{rec["id"]}[/]')
        return

    # Count occurrences
    count = full_text.lower().count(query_lower)
    console.print(
        f'\n  [{ACCENT}]{count}[/] match(es) for "[{ACCENT}]{query}[/]" in #{rec["id"]}:\n'
    )

    if segments:
        for seg in segments:
            text = (seg.get("text", "") or "").strip()
            if query_lower in text.lower():
                start = seg.get("start", 0) or 0
                ts = f"[{int(start // 60):02d}:{int(start % 60):02d}]"
                # Highlight the match
                highlighted = re.sub(
                    re.escape(query),
                    f"[bold {ACCENT}]{query}[/]",
                    text,
                    flags=re.IGNORECASE,
                )
                console.print(f"  [{DIM}]{ts}[/] {highlighted}")
    else:
        # No segments: highlight in full text
        highlighted = re.sub(
            re.escape(query),
            f"[bold {ACCENT}]{query}[/]",
            full_text,
            flags=re.IGNORECASE,
        )
        console.print(f"  {highlighted}")
    console.print()


def _dot_play(session: ReplSession, arg: str) -> None:
    """Play the source audio file using ffplay."""
    if not session.focus:
        return
        
    repo = session._get_repo()
    audio_file = None
    rec = None
    
    if session.focus.type == "file":
        audio_file = repo.get_audio_file(session.focus.id)
    if session.last_id:
        rec = repo.get_by_id(session.last_id)
        if rec and rec.get("audio_file_id"):
            audio_file = repo.get_audio_file(rec["audio_file_id"])
            
    file_path = audio_file.get("file_path") if audio_file else None
    
    if not file_path or not Path(file_path).exists():
        console.print(
            f"  [{WARNING}]Source audio file not found[/]\n"
            f"  [{DIM}]Path: {file_path or 'unknown'}[/]"
        )
        return

    # Parse optional start time
    start_seconds = 0.0
    if arg:
        # Handle "segment N" syntax
        seg_match = re.match(r"segment\s+(\d+)", arg, re.IGNORECASE)
        if seg_match:
            if not rec:
                console.print(f"  [{WARNING}]No transcript available to play a segment.[/]")
                return
            seg_idx = int(seg_match.group(1))
            segments = rec.get("segments", [])
            matched = [s for s in segments if s.get("index") == seg_idx]
            if matched:
                start_seconds = matched[0].get("start", 0) or 0
                end_seconds = matched[0].get("end", 0) or 0
                console.print(
                    f"  [{DIM}]Playing segment {seg_idx}: "
                    f"{int(start_seconds // 60):02d}:"
                    f"{int(start_seconds % 60):02d} → "
                    f"{int(end_seconds // 60):02d}:"
                    f"{int(end_seconds % 60):02d}[/]"
                )
            else:
                console.print(
                    f"  [{WARNING}]Segment {seg_idx} not found. Use .segments to see available.[/]"
                )
                return
        else:
            # Parse MM:SS or HH:MM:SS
            time_match = re.match(r"(?:(\d+):)?(\d+):(\d+)", arg)
            if time_match:
                hours = int(time_match.group(1) or 0)
                mins = int(time_match.group(2))
                secs = int(time_match.group(3))
                start_seconds = hours * 3600 + mins * 60 + secs
            else:
                console.print(f"  [{DIM}]Usage: .play  |  .play 01:25  |  .play segment 3[/]")
                return

    # Build ffplay command
    cmd = ["ffplay", "-nodisp", "-autoexit", str(file_path)]
    if start_seconds > 0:
        cmd.extend(["-ss", str(start_seconds)])

    # Duration limit for segment playback
    if arg and arg.lower().startswith("segment"):
        segments = rec.get("segments", [])
        seg_idx = int(re.match(r"segment\s+(\d+)", arg).group(1))
        matched = [s for s in segments if s.get("index") == seg_idx]
        if matched:
            duration = (matched[0].get("end", 0) or 0) - start_seconds
            if duration > 0:
                cmd.extend(["-t", str(duration)])

    file_name = audio_file.get("file_name", "?") if audio_file else "?"
    console.print(
        f"  [{ACCENT}]▶[/] Playing: {file_name}"
        + (
            f" from {int(start_seconds // 60):02d}:{int(start_seconds % 60):02d}"
            if start_seconds > 0
            else ""
        )
    )

    if hasattr(session, "_playback_proc") and session._playback_proc is not None:
        if session._playback_proc.poll() is None:
            session._playback_proc.terminate()
            session._playback_proc = None

    try:
        session._playback_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        console.print(f"  [{DIM}]Playing in background. Use .stop to halt playback.[/]")
    except FileNotFoundError:
        console.print(f"  [{WARNING}]ffplay not found. Install ffmpeg to use playback.[/]")

def _dot_stop(session: ReplSession) -> None:
    """Stop background audio playback."""
    if hasattr(session, "_playback_proc") and session._playback_proc is not None:
        if session._playback_proc.poll() is None:
            session._playback_proc.terminate()
            console.print(f"  [{SUCCESS}]Playback stopped.[/]")
        else:
            console.print(f"  [{DIM}]No audio currently playing.[/]")
        session._playback_proc = None
    else:
        console.print(f"  [{DIM}]No audio currently playing.[/]")


def _dot_edit(session: ReplSession, rec: dict) -> None:
    """Open transcript text in $EDITOR, save changes back to DB."""
    full_text = rec.get("full_text", "")

    editor = os.environ.get("EDITOR", "nano")

    # Write current text to a temp file
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_transcript_{rec['id']}.txt",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(full_text)
        tmp_path = tmp.name

    try:
        # Open in editor
        console.print(f"  [{DIM}]Opening #{rec['id']} in {editor}...[/]")
        result = subprocess.run([editor, tmp_path])

        if result.returncode != 0:
            console.print(f"  [{WARNING}]Editor exited with code {result.returncode}[/]")
            return

        # Read back edited text
        with open(tmp_path, encoding="utf-8") as f:
            new_text = f.read()

        # Compare
        if new_text == full_text:
            console.print(f"  [{DIM}]No changes made[/]")
            return

        # Word count diff
        old_wc = len(full_text.split())
        new_wc = len(new_text.split())
        diff = new_wc - old_wc

        # Save to DB
        repo = session._get_repo()
        ok = repo.update_text(rec["id"], new_text)
        if ok:
            console.print(
                f"  [{SUCCESS}]✓[/] Transcript #{rec['id']} updated ({new_wc} words, {diff:+d})"
            )
            session.refresh_context()
        else:
            console.print(f"  [{WARNING}]Failed to save changes[/]")
    finally:
        # Clean up temp file
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def _dot_path(session: ReplSession) -> None:
    """Show the source audio file path."""
    if not session.focus:
        return
        
    repo = session._get_repo()
    audio_file = None
    
    if session.focus.type == "file":
        audio_file = repo.get_audio_file(session.focus.id)
    elif session.last_id:
        rec = repo.get_by_id(session.last_id)
        if rec and rec.get("audio_file_id"):
            audio_file = repo.get_audio_file(rec["audio_file_id"])
            
    file_path = audio_file.get("file_path") if audio_file else None

    if file_path:
        exists = Path(file_path).exists()
        status = f"[{SUCCESS}]exists[/]" if exists else f"[{WARNING}]not found[/]"
        console.print(f"  [{ACCENT}]{file_path}[/]  ({status})")
    else:
        console.print(f"  [{DIM}]No source file path available[/]")


def _dot_unfocus_chapter(session: ReplSession) -> None:
    if not session.focus or session.focus.type != "file":
        console.print(f"  [{WARNING}]Must focus on a file first.[/]")
        return
        
    session.clear_chapter_focus()
    console.print(f"  [{SUCCESS}]Cleared chapter focus. Now focusing on entire file: {session.focus.label}[/]")


# _dot_focus_chapter and _dot_chapters defined below (lines ~755+)
def _dot_open(session: ReplSession) -> None:
    """Open the source audio file in the default system player."""
    if not session.focus:
        return
        
    repo = session._get_repo()
    audio_file = None
    
    if session.focus.type == "file":
        audio_file = repo.get_audio_file(session.focus.id)
    elif session.last_id:
        rec = repo.get_by_id(session.last_id)
        if rec and rec.get("audio_file_id"):
            audio_file = repo.get_audio_file(rec["audio_file_id"])
            
    file_path = audio_file.get("file_path") if audio_file else None

    if not file_path or not Path(file_path).exists():
        console.print(f"  [{WARNING}]Source file not found.[/]")
        return

    console.print(f"  [{ACCENT}]Opening:[/] {file_path}")
    try:
        # Linux: xdg-open, macOS: open, Windows: start
        if sys.platform == "darwin":
            subprocess.Popen(["open", file_path])
        elif sys.platform == "win32":
            os.startfile(file_path)  # noqa: S606
        else:
            subprocess.Popen(
                ["xdg-open", file_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception as e:
        console.print(f"  [{WARNING}]Failed to open file: {e}[/]")


def _dot_chapters(session: ReplSession) -> None:
    """List all chapters for the currently focused audio file."""
    if not session.focus or session.focus.type != "file":
        console.print(f"  [{WARNING}]You must focus on an audio file first using 'use <file_name>'.[/]")
        return

    from audiobench.storage.chapter_repository import get_chapter_repo
    chapters = get_chapter_repo().get_chapters(session.focus.id)

    if not chapters:
        console.print(f"  [{DIM}]No chapters found for this audio file.[/]")
        return

    table = make_table(
        f"Chapters for {session.focus.label}",
        [
            ("Index", {"style": BOLD, "justify": "right"}),
            ("Title", {}),
            ("Duration", {"justify": "right"}),
            ("Status", {}),
            ("Tags", {"style": DIM}),
        ],
    )

    for chap in chapters:
        status_icon = "⚡ Skipped" if chap.is_ghost else "completed"  # ChapterInfo has no status field; ghost is the flag
        tags = ", ".join(chap.to_dict().get("tags_list", [])) if hasattr(chap, "to_dict") else ""
        table.add_row(
            str(chap.index),
            chap.title,
            format_duration(chap.duration_seconds),
            status_icon,
            tags,
        )

    console.print(table)


def _dot_focus_chapter(session: ReplSession, arg: str) -> None:
    """Focus on a specific chapter."""
    if not session.focus or session.focus.type != "file":
        console.print(f"  [{WARNING}]You must focus on an audio file first.[/]")
        return

    if not arg.isdigit():
        console.print(f"  [{WARNING}]Usage: .fc <index>[/]")
        return

    index = int(arg)
    from audiobench.storage.chapter_repository import get_chapter_repo
    chapter = get_chapter_repo().get_chapter_by_index(session.focus.id, index)

    if not chapter:
        console.print(f"  [{WARNING}]Chapter {index} not found.[/]")
        return

    if chapter.is_ghost:
        console.print(f"  [{WARNING}]Cannot focus on skipped ghost chapter {index}.[/]")
        return

    session.focus_chapter(index, chapter.title)
    console.print(f"  [{SUCCESS}]Focused on Chapter {index}: {chapter.title}[/]")


def _navigate_context(session: ReplSession, direction: int) -> None:
    """Navigate to .next or .prev transcript in history."""
    if not session._history_ids:
        session._load_history_ids()

    if not session._history_ids:
        console.print(f"  [{DIM}]No transcription history[/]")
        return

    if session._history_cursor < 0:
        # Not in navigation yet — start from the newest or oldest
        new_cursor = 0 if direction > 0 else len(session._history_ids) - 1
    else:
        new_cursor = session._history_cursor + direction

    if new_cursor < 0 or new_cursor >= len(session._history_ids):
        label = "newest" if direction > 0 else "oldest"
        console.print(f"  [{DIM}]Already at the {label} transcript[/]")
        return

    session._history_cursor = new_cursor
    new_id = session._history_ids[new_cursor]
    session.set_context(new_id)
    if session.focus:
        pos = f"{new_cursor + 1}/{len(session._history_ids)}"
        console.print(
            f"  [{SUCCESS}]✓[/] [{ACCENT}]#{new_id}[/] — "
            f"{session.focus.label} "
            f"[{DIM}]({pos})[/]"
        )


def print_dot_help() -> None:
    """Print dot-command reference, grouped by function."""
    groups = {
        "View & Analyze": [
            ".stats",
            ".show",
            ".segments",
            ".vocab",
            ".info",
            ".find",
        ],
        "Audio": [".play", ".stop", ".open", ".path"],
        "AI": [".ask", ".chat", ".summarize"],
        "Actions": [".export", ".edit", ".diarize"],
        "Navigation": [
            ".use",
            ".clear",
            ".next",
            ".prev",
            ".recent",
        ],
        "Search": [".search"],
        "Meta": [".help"],
    }
    
    if SUPPORTS_BACKGROUND_JOBS:
        groups["Background Jobs"] = [".jobs", ".watch"]

    console.print(f"\n  [{BOLD}][{ACCENT}]Dot Commands[/][/]")
    console.print(f"  [{DIM}]Operate on the current context transcript[/]\n")
    for group_name, group_cmds in groups.items():
        console.print(f"  [{BOLD}]{group_name}[/]")
        for cmd_name in group_cmds:
            desc = DOT_COMMANDS.get(cmd_name, "")
            console.print(f"    [{ACCENT}]{cmd_name:<14}[/] {desc}")
        console.print()


def _dot_jobs() -> None:
    """Show background jobs table inline inside the REPL."""
    try:
        from audiobench.jobs.repository import JobRepository
        from audiobench.jobs.runner import get_job_phase, startup_recovery
    except ImportError:
        console.print(f"  [{WARNING}]Jobs module not available[/]")
        return

    startup_recovery()
    repo = JobRepository()
    all_jobs = repo.get_all_jobs(limit=15)

    if not all_jobs:
        console.print(f"  [{DIM}]No background jobs[/]")
        return

    table = make_table(
        "Background Jobs",
        [
            ("ID",      {"width": 4, "justify": "right"}),
            ("Status",  {"width": 10}),
            ("Command", {}),
            ("Phase",   {"width": 18}),
            ("Started", {"style": DIM}),
        ],
    )

    STATUS_COLOR = {
        "running":   ACCENT,
        "done":      SUCCESS,
        "failed":    WARNING,
        "cancelled": WARNING,
    }

    for job in all_jobs:
        job_id = job["id"]
        status = job.get("status", "unknown")
        color  = STATUS_COLOR.get(status, DIM)
        cmd    = (job.get("command") or "").removeprefix("audiobench ")
        if len(cmd) > 38:
            cmd = cmd[:35] + "..."
        started = str(job.get("started_at", ""))[:16]
        phase   = get_job_phase(job_id) if status == "running" else ""

        table.add_row(
            f"#{job_id}",
            f"[{color}]{status}[/]",
            cmd,
            phase,
            started,
        )

    console.print(table)
    console.print(
        f"  [{DIM}]Tip: .watch <id> to tail logs  ·  jobs cancel <id> to stop[/]"
    )


def _dot_diarize(session: ReplSession, rec: dict) -> None:
    """Interactive wizard to run/re-run speaker diarization on the current transcript."""
    from audiobench.cli.wizard import prompt_bool, prompt_menu, prompt_string

    file_name = rec.get("file_name", "?")
    tx_id = rec["id"]

    console.print(
        f"\n  [{BOLD}][{ACCENT}]Speaker Diarization Wizard[/][/]\n"
        f"  [{DIM}]Transcript #{tx_id} — {file_name}[/]\n"
    )

    # Check if already diarized
    segments = rec.get("segments", [])
    already_diarized = any(seg.get("speaker") for seg in segments)
    if already_diarized:
        console.print(f"  [{WARNING}]This transcript already has speaker labels.[/]")
        try:
            rerun = prompt_bool("Re-run diarization anyway?", default=False)
        except KeyboardInterrupt:
            console.print()
            return
        if not rerun:
            return

    # Step 1: Diarization model
    try:
        diarize_model = prompt_menu(
            "Diarization model",
            [
                ("3.1 (legacy)", "slower, broader support", "speaker-diarization-3.1"),
                ("3.0 (stable)", "stable general-purpose",  "speaker-diarization-3.0"),
                ("segmentation-3.0", "fast, segment-only",  "segmentation-3.0"),
            ],
            default_idx=0,
        )

        # Step 2: GPU acceleration
        use_gpu = prompt_bool("Use GPU acceleration (CUDA)?", default=False)

        # Step 3: Speaker count
        know_speakers = prompt_bool("Do you know the exact number of speakers?", default=False)
        num_speakers: int | None = None
        if know_speakers:
            raw = prompt_string(
                "Number of speakers",
                default="2",
                validator=lambda s: s.isdigit() and int(s) > 0,
                validation_msg="Please enter a positive integer.",
            )
            num_speakers = int(raw)

        # Step 4: Name mapping
        map_speakers: str | None = None
        if num_speakers:
            want_names = prompt_bool("Map speakers to real names?", default=False)
            if want_names:
                console.print(f"  [{DIM}]Format: Speaker 1=Alice, Speaker 2=Bob[/]")
                mapping = prompt_string("Speaker names", default="")
                if mapping.strip():
                    map_speakers = mapping.strip()

    except KeyboardInterrupt:
        console.print()
        return

    # Build and show the command
    repo = session._get_repo()
    audio_file = None
    if rec.get("audio_file_id"):
        audio_file = repo.get_audio_file(rec["audio_file_id"])

    if not audio_file:
        console.print(f"  [{WARNING}]Cannot find source audio file for transcript #{tx_id}.[/]")
        return

    file_path = audio_file.get("file_path", "")
    if not file_path:
        console.print(f"  [{WARNING}]Source file path is missing.[/]")
        return

    # Assemble CLI args
    cmd = ["transcribe", file_path, "--diarize", "--no-cache"]
    if use_gpu:
        cmd.append("--gpu")
    if num_speakers:
        cmd += ["--speakers", str(num_speakers)]
    if map_speakers:
        cmd += ["--map-speakers", map_speakers]

    # Pre-run summary
    console.print(f"\n  [{BOLD}]Ready to run:[/]")
    console.print(f"    [{DIM}]Model:[/]    {diarize_model}")
    console.print(f"    [{DIM}]GPU:[/]      {'yes' if use_gpu else 'no'}")
    if num_speakers:
        console.print(f"    [{DIM}]Speakers:[/] {num_speakers}")
    if map_speakers:
        console.print(f"    [{DIM}]Names:[/]    {map_speakers}")
    console.print()

    try:
        confirm = prompt_bool("Start diarization?", default=True)
    except KeyboardInterrupt:
        console.print()
        return

    if confirm:
        dispatch_command(session, cmd)
    else:
        console.print(f"  [{DIM}]Cancelled.[/]")


def _dot_watch(job_id: int) -> None:
    """Tail a background job's log from inside the REPL."""
    from audiobench.jobs.runner import watch_job
    watch_job(job_id)
