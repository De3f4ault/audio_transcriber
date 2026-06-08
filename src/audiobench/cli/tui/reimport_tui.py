"""Reverse Import (Transcript-first) TUI."""

import curses
import time
import sys
import threading
from pathlib import Path
from difflib import SequenceMatcher

from audiobench.cli.tui.directory_navigator import DirectoryNavigator
from audiobench.transcribe.transcript_parser import (
    parse_transcript_file,
    build_audio_metadata,
    HashMismatchError,
    UnsupportedSchemaVersion,
)
from audiobench.core.db_engine import init_db
from audiobench.storage.repository import TranscriptionRepository
from audiobench.storage.models import AudioFileRecord, TranscriptionRecord


class ReimportTUI:
    def __init__(self, start_path="~/Downloads", audio_start_path="~/Music/Audiobooks/Audios", batch=False):
        self.start_path = Path(start_path).expanduser()
        self.audio_start_path = Path(audio_start_path).expanduser()
        self.cancelled = False
        
        self.is_batch = batch
        self.screen = 1 if not batch else 11
        
        # Single mode state
        self.transcript_path = None
        self.audio_path = None
        self.parsed_transcript = None
        self.raw_audio = None
        self.audio_metadata = None
        
        self.db_check_result = None # None, "new", "exists"
        self.conflict_action = None # "O", "A", "S"
        self.existing_transcriptions_count = 0
        
        self.hash_error = None
        self.hash_computed = False
        
        # Batch mode state
        self.transcript_dir = None
        self.audio_dir = None
        self.batch_pairs = [] # list of dict: {"tx": Path, "audio": Path|None, "status": str}
        
        self.tx_nav = DirectoryNavigator(start_path=str(self.start_path), allowed_extensions={"json", "srt", "txt"})
        self.audio_nav = DirectoryNavigator(start_path=str(self.audio_start_path))
        self.audio_nav.allowed_extensions = None # all supported formats

    def state_export(self) -> dict:
        return {}
        
    def state_import(self, state: dict) -> None:
        pass
        
    def run(self):
        curses.wrapper(self._run_loop)
        return not self.cancelled

    def _run_loop(self, stdscr):
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.nodelay(0)
        stdscr.timeout(100)
        stdscr.clear()

        try:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, -1)  # Header/Footer
            curses.init_pair(2, curses.COLOR_WHITE, -1)  # Normal
            curses.init_pair(3, curses.COLOR_GREEN, -1)  # Success/Selected
            curses.init_pair(4, curses.COLOR_YELLOW, -1)  # Warning/Hidden
            curses.init_pair(5, curses.COLOR_MAGENTA, -1)  # Status
            curses.init_pair(6, curses.COLOR_CYAN, -1)  # Separator
            curses.init_pair(7, curses.COLOR_RED, -1)  # Error
        except:
            pass

        while not self.cancelled:
            stdscr.clear()
            
            if self.screen == 1:
                self._draw_screen_1(stdscr)
            elif self.screen == 2:
                self._draw_screen_2(stdscr)
            elif self.screen == 3:
                self._draw_screen_3(stdscr)
                if self.hash_computed and self.db_check_result == "new":
                    self.screen = 4
            elif self.screen == 4:
                self._draw_screen_4(stdscr)
            elif self.screen == 5:
                self._draw_screen_5(stdscr)
                break
            elif self.screen == 11:
                self._draw_screen_1b(stdscr)
            elif self.screen == 12:
                self._draw_screen_2b(stdscr)
            elif self.screen == 13:
                self._draw_screen_3b(stdscr)
            elif self.screen == 14:
                self._draw_screen_4b(stdscr)
                break
            elif self.screen == 15:
                self._draw_screen_15(stdscr)
                
            key = stdscr.getch()
            self._handle_input(key)
            
    def _handle_input(self, key):
        if key == -1:
            return
            
        if self.screen == 1:
            self._nav_input(key, self.tx_nav, on_enter=self._on_tx_selected)
            if key in (ord('b'), ord('B')):
                self.is_batch = True
                self.screen = 11
        elif self.screen == 2:
            self._nav_input(key, self.audio_nav, on_enter=self._on_audio_selected)
        elif self.screen == 3:
            if self.db_check_result == "exists":
                if key in (ord('o'), ord('O')):
                    self.conflict_action = "O"
                    self.screen = 4
                elif key in (ord('a'), ord('A')):
                    self.conflict_action = "A"
                    self.screen = 4
                elif key in (ord('s'), ord('S')):
                    self.screen = 1
            elif self.hash_error:
                if key in (ord('r'), ord('R')):
                    self.screen = 2
                    self.hash_computed = False
                    self.hash_error = None
        elif self.screen == 4:
            if key in (10, 13): # ENTER
                self.screen = 5
            elif key in (ord('q'), ord('Q')):
                self.cancelled = True
        elif self.screen == 11:
            self._nav_input(key, self.tx_nav, on_enter=self._on_tx_dir_selected)
            if key in (10, 13):
                self.transcript_dir = self.tx_nav.current_path
                self.screen = 12
        elif self.screen == 12:
            self._nav_input(key, self.audio_nav, on_enter=lambda p: None)
            if key in (10, 13):
                self.audio_dir = self.audio_nav.current_path
                self._match_batch()
                
                existing = self._get_existing_basenames()
                self.conflict_pairs = [p for p in self.batch_pairs if p["tx"].stem in existing]
                
                if self.conflict_pairs:
                    self.screen = 15
                else:
                    self.batch_overwrite = False
                    self.screen = 13
        elif self.screen == 13:
            if key in (ord('s'), ord('S')):
                self.screen = 14
            elif key in (ord('q'), ord('Q')):
                self.cancelled = True
        elif self.screen == 15:
            if key in (ord('s'), ord('S')):
                existing = self._get_existing_basenames()
                self.batch_pairs = [p for p in self.batch_pairs if p["tx"].stem not in existing]
                self.batch_overwrite = False
                self.screen = 13
            elif key in (ord('r'), ord('R'), ord('o'), ord('O'), ord('0')):
                self.batch_overwrite = True
                self.screen = 13
            elif key in (ord('q'), ord('Q')):
                self.cancelled = True
                
        if key in (ord('q'), ord('Q')) and self.screen < 5:
            self.cancelled = True
        elif key in (ord('q'), ord('Q')) and self.screen in (11, 12, 15):
            self.cancelled = True

    def _get_existing_basenames(self):
        if hasattr(self, "_existing_basenames"):
            return self._existing_basenames
            
        from audiobench.core.db_engine import init_db
        from audiobench.storage.repository import TranscriptionRepository
        init_db()
        repo = TranscriptionRepository()
        from audiobench.core.db_session import get_session
        self._existing_basenames = set()
        with get_session() as session:
            from audiobench.storage.models import TranscriptionRecord
            from pathlib import Path
            for (fname,) in session.query(TranscriptionRecord.file_name).all():
                if fname:
                    self._existing_basenames.add(Path(fname).stem)
        return self._existing_basenames

    def _match_batch(self):
        self.batch_pairs = []
        import json
        import re
        
        from audiobench.transcribe.audio_converter import ALL_SUPPORTED_FORMATS
        
        # Pre-index all supported audio files in the audio directory recursively
        # to avoid scanning the disk thousands of times
        available_audio_files = []
        try:
            for ext in ALL_SUPPORTED_FORMATS:
                available_audio_files.extend(list(self.audio_dir.rglob(f"*.{ext}")))
        except Exception:
            pass

        def normalize_name(name: str) -> str:
            # Lowercase, remove non-alphanumeric (except spaces), collapse spaces
            s = re.sub(r'[^a-z0-9\s]', '', name.lower())
            return re.sub(r'\s+', ' ', s).strip()

        # If the user selected specific files with spacebar, use those. 
        # Otherwise, default to all JSON files in the confirmed directory.
        if self.tx_nav.selected_files:
            tx_files = list(self.tx_nav.selected_files)
        else:
            tx_files = list(self.transcript_dir.glob("*.json"))
            
        for tx_file in tx_files:
            if not tx_file.name.endswith(".json"):
                continue
            
            matched_audio = None
            
            # 1. Try to read JSON to find exact original audio file name
            try:
                data = json.loads(tx_file.read_text(encoding="utf-8"))
                audio_name = data.get("audio", {}).get("file_name")
                if audio_name:
                    # Check recursively
                    for audio_cand in available_audio_files:
                        if audio_cand.name == audio_name:
                            matched_audio = audio_cand
                            break
            except Exception:
                pass
                
            # 2. If not found via JSON metadata, try matching by base filename (fuzzy & recursive)
            if not matched_audio:
                tx_stem_norm = normalize_name(tx_file.stem)
                best_match = None
                best_ratio = 0.0
                
                for audio_cand in available_audio_files:
                    audio_stem_norm = normalize_name(audio_cand.stem)
                    
                    if tx_stem_norm == audio_stem_norm:
                        best_match = audio_cand
                        best_ratio = 1.0
                        break
                    else:
                        ratio = SequenceMatcher(None, tx_stem_norm, audio_stem_norm).ratio()
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_match = audio_cand
                
                # Threshold for fuzzy matching, e.g., 0.85
                if best_ratio > 0.85:
                    matched_audio = best_match
                        
            # 3. Fallback: check if the audio file is already in the database library
            if not matched_audio:
                try:
                    from audiobench.core.db_session import get_session
                    from audiobench.storage.models import AudioFileRecord
                    with get_session() as session:
                        stem = tx_file.stem
                        for ext in ALL_SUPPORTED_FORMATS:
                            query_name = f"{stem}.{ext}"
                            rec = session.query(AudioFileRecord).filter_by(file_name=query_name).first()
                            if rec and Path(rec.file_path).exists():
                                matched_audio = Path(rec.file_path)
                                break
                except Exception:
                    pass
                        
            # Append pair
            self.batch_pairs.append({"tx": tx_file, "audio": matched_audio})

            
    def _nav_input(self, key, nav, on_enter):
        if key in (curses.KEY_RIGHT, 10, 13, ord("l")):
            cur = nav.get_current_item_path()
            if cur and cur.is_file():
                on_enter(cur)
            else:
                nav.enter()
        elif key in (curses.KEY_LEFT, ord("h")):
            nav.go_up()
        elif key in (curses.KEY_UP, ord("k")):
            nav.select_prev()
        elif key in (curses.KEY_DOWN, ord("j")):
            nav.select_next()
        elif key in (ord("t"), ord("H")):
            nav.toggle_hidden_visibility()
        elif key == ord(" "):
            nav.toggle_selection()
            
    def _on_tx_selected(self, path):
        self.transcript_path = path
        try:
            self.parsed_transcript, self.raw_audio = parse_transcript_file(path)
            self.screen = 2
        except Exception as e:
            pass # ignore for now or show error

    def _on_audio_selected(self, path):
        self.audio_path = path
        self.screen = 3
        self.hash_computed = False
        self.hash_error = None
        self.db_check_result = None
        
        # Start hash verification in background
        threading.Thread(target=self._verify_hash, daemon=True).start()
        
    def _on_tx_dir_selected(self, path):
        pass # TODO batch

    def _verify_hash(self):
        try:
            self.audio_metadata = build_audio_metadata(self.audio_path, self.raw_audio)
            
            init_db()
            repo = TranscriptionRepository()
            from audiobench.core.db_session import get_session
            with get_session() as session:
                # Check DB for audio
                from sqlalchemy.orm import Session
                rec = session.query(AudioFileRecord).filter_by(file_hash=self.audio_metadata.file_hash).first()
                if rec:
                    self.db_check_result = "exists"
                    self.existing_transcriptions_count = session.query(TranscriptionRecord).filter_by(audio_file_id=rec.id).count()
                else:
                    self.db_check_result = "new"
        except HashMismatchError as e:
            self.hash_error = str(e)
        except Exception as e:
            self.hash_error = f"Error computing hash: {e}"
        finally:
            self.hash_computed = True

    def _draw_screen_1(self, stdscr):
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, " 📄 Reimport: Select Transcript File ".ljust(w), curses.color_pair(1) | curses.A_BOLD)
        
        # Draw navigator left
        items = self.tx_nav.list_items()
        for i, item in enumerate(items[:h-4]):
            y = i + 2
            cur = self.tx_nav.get_current_item_path()
            is_sel = (i == self.tx_nav.selected)
            attr = curses.color_pair(3) | curses.A_BOLD if is_sel else curses.color_pair(2)
            icon = "📁" if (self.tx_nav.current_path / item).is_dir() else "📄"
            stdscr.addstr(y, 1, f"{'>' if is_sel else ' '} {icon} {item}"[:w//2], attr)
            
        try:
            stdscr.addstr(h-1, 0, " ENTER: Select | b: Batch Mode | q: Quit ".ljust(w-1), curses.color_pair(1))
        except curses.error:
            pass

    def _draw_screen_2(self, stdscr):
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, " 🎵 Reimport: Link Local Audio File ".ljust(w), curses.color_pair(1) | curses.A_BOLD)
        
        items = self.audio_nav.list_items()
        for i, item in enumerate(items[:h-4]):
            y = i + 2
            is_sel = (i == self.audio_nav.selected)
            attr = curses.color_pair(3) | curses.A_BOLD if is_sel else curses.color_pair(2)
            icon = "📁" if (self.audio_nav.current_path / item).is_dir() else "🎵"
            stdscr.addstr(y, 1, f"{'>' if is_sel else ' '} {icon} {item}"[:w//2], attr)
            
        try:
            stdscr.addstr(h-1, 0, " ENTER: Select | q: Quit ".ljust(w-1), curses.color_pair(1))
        except curses.error:
            pass

    def _draw_screen_3(self, stdscr):
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, " 🔍 Reimport: Verify Identity ".ljust(w), curses.color_pair(1) | curses.A_BOLD)
        
        if not self.hash_computed:
            stdscr.addstr(3, 3, "Computing SHA-256...", curses.color_pair(5))
        elif self.hash_error:
            stdscr.addstr(3, 3, "❌ Hash mismatch — this is not the correct audio file", curses.color_pair(7) | curses.A_BOLD)
            stdscr.addstr(5, 3, self.hash_error, curses.color_pair(7))
            stdscr.addstr(8, 3, "[R] Re-select Audio File", curses.color_pair(2) | curses.A_BOLD)
        elif self.db_check_result == "exists":
            stdscr.addstr(3, 3, "✅ Hash confirmed. File identity verified.", curses.color_pair(3))
            stdscr.addstr(5, 3, f"Existing audio record found with {self.existing_transcriptions_count} transcriptions.")
            stdscr.addstr(7, 3, "[O]verwrite | [A]dd new transcription | [S]kip", curses.color_pair(2) | curses.A_BOLD)
            
        try:
            stdscr.addstr(h-1, 0, " q: Quit ".ljust(w-1), curses.color_pair(1))
        except curses.error:
            pass

    def _draw_screen_4(self, stdscr):
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, " 🚀 Reimport: Confirm Pipeline ".ljust(w), curses.color_pair(1) | curses.A_BOLD)
        
        stdscr.addstr(2, 2, f"Transcript: {self.transcript_path.name}")
        stdscr.addstr(3, 2, f"Audio:      {self.audio_path.name}")
        
        stdscr.addstr(5, 2, "Pipeline:")
        stdscr.addstr(6, 2, " [1] Copy .mp3 and sidecars")
        stdscr.addstr(7, 2, " [2] Save AudioFileRecord")
        stdscr.addstr(8, 2, " [3] Auto-detect chapters")
        stdscr.addstr(9, 2, " [4] Save TranscriptionRecord & Segments")
        stdscr.addstr(10, 2," [5] Semantic chunk & embed")
        
        try:
            stdscr.addstr(h-1, 0, " ENTER: Start Import | q: Cancel ".ljust(w-1), curses.color_pair(1))
        except curses.error:
            pass

    def _draw_screen_5(self, stdscr):
        stdscr.clear()
        stdscr.addstr(2, 2, "Importing...", curses.color_pair(5))
        stdscr.refresh()
        
        # Actually do the import
        try:
            init_db()
            repo = TranscriptionRepository()
            
            def _phase_cb(phase: str, pct: float):
                pass
                
            tx_id = repo.save_transcription(
                self.parsed_transcript,
                self.audio_metadata,
                on_phase=_phase_cb
            )
            
            # update source to reimport
            from audiobench.core.db_session import get_session
            with get_session() as session:
                rec = session.query(TranscriptionRecord).filter_by(id=tx_id).first()
                if rec:
                    rec.source = "reimport"
                    session.commit()
            
            stdscr.clear()
            stdscr.addstr(2, 2, f"✅ Transcription #{tx_id} imported successfully!", curses.color_pair(3))
        except Exception as e:
            stdscr.clear()
            stdscr.addstr(2, 2, f"❌ Import failed: {e}", curses.color_pair(7))
            
        stdscr.addstr(4, 2, "Press any key to exit.")
        stdscr.refresh()
        stdscr.timeout(-1)
        stdscr.getch()

    def _draw_screen_1b(self, stdscr):
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, " 📂 Batch Reimport: Select Transcript Folder ".ljust(w), curses.color_pair(1) | curses.A_BOLD)
        
        existing = self._get_existing_basenames()
        items = self.tx_nav.list_items()
        for i, item in enumerate(items[:h-4]):
            y = i + 2
            is_sel = (i == self.tx_nav.selected)
            item_path = self.tx_nav.current_path / item
            is_checked = item_path in self.tx_nav.selected_files
            attr = curses.color_pair(3) | curses.A_BOLD if is_sel else curses.color_pair(2)
            
            if item_path.is_dir():
                icon = "📁"
                prefix = f"{'>' if is_sel else ' '} "
                suffix = ""
            else:
                icon = "📄"
                prefix = f"{'>' if is_sel else ' '} {'[x]' if is_checked else '[ ]'}"
                suffix = " [Transcribed]" if item_path.stem in existing else ""
                if suffix:
                    attr = curses.color_pair(5) | curses.A_BOLD if is_sel else curses.color_pair(5)
                
            stdscr.addstr(y, 1, f"{prefix} {icon} {item}{suffix}"[:w-2], attr)
            
        stdscr.addstr(h-2, 0, f" Current Folder: {self.tx_nav.current_path} ({len(self.tx_nav.selected_files)} selected)".ljust(w), curses.color_pair(5))
        try:
            stdscr.addstr(h-1, 0, " SPACE: Select File | ENTER: Confirm Folder/Files | q: Quit ".ljust(w-1), curses.color_pair(1))
        except curses.error:
            pass

    def _on_tx_dir_selected(self, path):
        # path is actually the selected item, but we want the current directory for batch
        pass # we handle it directly in handle_input for 1b

    def _draw_screen_2b(self, stdscr):
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, " 📂 Batch Reimport: Select Audio Folder ".ljust(w), curses.color_pair(1) | curses.A_BOLD)
        
        items = self.audio_nav.list_items()
        for i, item in enumerate(items[:h-4]):
            y = i + 2
            is_sel = (i == self.audio_nav.selected)
            attr = curses.color_pair(3) | curses.A_BOLD if is_sel else curses.color_pair(2)
            icon = "📁" if (self.audio_nav.current_path / item).is_dir() else "🎵"
            stdscr.addstr(y, 1, f"{'>' if is_sel else ' '} {icon} {item}"[:w-2], attr)
            
        stdscr.addstr(h-2, 0, f" Current Folder: {self.audio_nav.current_path}".ljust(w), curses.color_pair(5))
        try:
            stdscr.addstr(h-1, 0, " ENTER: Confirm Folder | q: Quit ".ljust(w-1), curses.color_pair(1))
        except curses.error:
            pass

    def _draw_screen_15(self, stdscr):
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, f" ⚠️ Conflict: {len(self.conflict_pairs)} files already exist ".ljust(w), curses.color_pair(7) | curses.A_BOLD)
        
        for i, pair in enumerate(self.conflict_pairs[:h-4]):
            y = i + 2
            tx = pair["tx"].name
            stdscr.addstr(y, 2, f"- {tx}"[:w-4], curses.color_pair(5))
            
        try:
            stdscr.addstr(h-1, 0, " [S]kip Existing | [R]eplace Existing | q: Cancel ".ljust(w-1), curses.color_pair(1))
        except curses.error:
            pass

    def _draw_screen_3b(self, stdscr):
        h, w = stdscr.getmaxyx()
        msg = f" 🚀 Batch Ready: {len(self.batch_pairs)} pairs matched "
        if getattr(self, "batch_overwrite", False):
            msg += "(OVERWRITE ON) "
        stdscr.addstr(0, 0, msg.ljust(w), curses.color_pair(1) | curses.A_BOLD)
        
        for i, pair in enumerate(self.batch_pairs[:h-4]):
            y = i + 2
            tx = pair["tx"].name
            au = pair["audio"].name if pair["audio"] else "???"
            stdscr.addstr(y, 2, f"{tx} <--> {au}"[:w-4])
            
        try:
            stdscr.addstr(h-1, 0, " S: Start Batch | q: Quit ".ljust(w-1), curses.color_pair(1))
        except curses.error:
            pass
        
    def _draw_screen_4b(self, stdscr):
        stdscr.clear()
        stdscr.addstr(2, 2, "Batch Importing...", curses.color_pair(5))
        stdscr.refresh()
        
        success = 0
        failed = 0
        
        for idx, pair in enumerate(self.batch_pairs):
            stdscr.addstr(4, 2, f"Processing {idx+1}/{len(self.batch_pairs)}: {pair['tx'].name}                  ")
            stdscr.refresh()
            
            if not pair["audio"]:
                failed += 1
                continue
                
            try:
                parsed_tx, raw_audio = parse_transcript_file(pair["tx"])
                audio_meta = build_audio_metadata(pair["audio"], raw_audio)
                
                init_db()
                repo = TranscriptionRepository()
                
                tx_id = repo.save_transcription(
                    parsed_tx, 
                    audio_meta, 
                    on_phase=lambda *args, **kwargs: None,
                    overwrite=getattr(self, "batch_overwrite", False)
                )
                from audiobench.core.db_session import get_session
                with get_session() as session:
                    rec = session.query(TranscriptionRecord).filter_by(id=tx_id).first()
                    if rec:
                        rec.source = "reimport"
                        session.commit()
                success += 1
            except Exception:
                failed += 1

        stdscr.clear()
        stdscr.addstr(2, 2, f"✅ Batch Complete! {success} succeeded, {failed} failed.", curses.color_pair(3))
        stdscr.addstr(4, 2, "Press any key to exit.")
        stdscr.refresh()
        stdscr.timeout(-1)
        stdscr.getch()
