import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from audiobench.core.settings import get_settings


class BackgroundImporter:
    def __init__(self, max_workers: int = 4):
        self.settings = get_settings()
        self.library_dir = self.settings.data_dir / "library"
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self._lock = threading.Lock()

        self.progress_callback: Callable[[int, int, str], None] | None = None
        self.file_done_callback: Callable[[Path, Path, str], None] | None = None

    def set_callbacks(
        self,
        progress: Callable[[int, int, str], None] | None = None,
        file_done: Callable[[Path, Path, str], None] | None = None,
    ) -> None:
        """Set callbacks for progress tracking.

        progress(total_bytes, copied_bytes, filename)
        file_done(original_path, new_path, engine)
        """
        self.progress_callback = progress
        self.file_done_callback = file_done

    def _copy_file_streaming(self, source: Path, destination: Path, filename: str) -> bool:
        """Atomic copy with progress tracking."""
        tmp_destination = destination.with_suffix(destination.suffix + ".tmp")
        total_size = source.stat().st_size
        copied = 0

        try:
            with open(source, "rb") as fsrc, open(tmp_destination, "wb") as fdst:
                while True:
                    chunk = fsrc.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    fdst.write(chunk)
                    copied += len(chunk)

                    if self.progress_callback:
                        self.progress_callback(total_size, copied, filename)

            # Atomic rename after successful copy
            tmp_destination.replace(destination)
            return True
        except Exception:
            if tmp_destination.exists():
                tmp_destination.unlink()
            return False

    def _process_single_import(self, allocation: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single file import (runs in thread pool)."""
        source_path: Path = allocation["file"]
        engine: str = allocation["engine"]
        # Generate True Fast Content Hash
        import hashlib

        from audiobench.core.db_session import get_session
        from audiobench.storage.models import AudioFileRecord

        def compute_fast_hash(fp: Path) -> str:
            hasher = hashlib.md5()
            hasher.update(str(fp.stat().st_size).encode())
            with open(fp, "rb") as f:
                hasher.update(f.read(1024 * 1024))
            return hasher.hexdigest()

        file_hash = compute_fast_hash(source_path)

        # Deduplication Check
        with get_session() as session:
            existing = session.query(AudioFileRecord).filter_by(file_hash=file_hash).first()
            if existing:
                # If duplicate, immediately skip the heavy copy process
                if self.progress_callback:
                    self.progress_callback(
                        source_path.stat().st_size, source_path.stat().st_size, source_path.name
                    )
                return {
                    "source": source_path,
                    "destination": Path(existing.file_path),
                    "engine": engine,
                    "is_duplicate": True,
                    "file_hash": file_hash,
                    "existing_id": existing.id,
                }

        filename = source_path.name
        destination = self.library_dir / filename

        # Handle filename collisions
        counter = 1
        while destination.exists():
            destination = self.library_dir / f"{source_path.stem}_{counter}{source_path.suffix}"
            filename = destination.name
            counter += 1

        success = self._copy_file_streaming(source_path, destination, filename)

        if success:
            if self.file_done_callback:
                self.file_done_callback(source_path, destination, engine)
            return {
                "source": source_path,
                "destination": destination,
                "engine": engine,
                "is_duplicate": False,
                "file_hash": file_hash,
            }
        return None

    def run_import_jobs(self, allocations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run all imports in parallel using ThreadPoolExecutor."""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_allocation = {
                executor.submit(self._process_single_import, alloc): alloc for alloc in allocations
            }

            for future in as_completed(future_to_allocation):
                try:
                    result = future.result()
                    if result:
                        with self._lock:
                            results.append(result)
                except Exception:
                    pass

        return results
