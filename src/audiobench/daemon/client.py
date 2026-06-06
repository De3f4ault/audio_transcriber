"""Daemon client — sends JSON commands to the running daemon over Unix socket."""

from __future__ import annotations

import json
import socket
import uuid
from pathlib import Path

from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.daemon.protocol import ChunkResult, SearchResult
from audiobench.memory.enums import SourceType

logger = get_logger("daemon.client")

# Timeout for normal requests (seconds)
_DEFAULT_TIMEOUT = 30.0


class DaemonClient:
    """Synchronous client for the AudioBench daemon."""

    def __init__(self, socket_path: Path | None = None) -> None:
        settings = get_settings()
        self._socket_path = socket_path or Path(settings.daemon_socket_path)

    # ------------------------------------------------------------------
    # Low-level transport
    # ------------------------------------------------------------------

    def _ensure_daemon_running(self) -> None:
        """Silently spawn the daemon if it's not running, and wait for it."""
        import subprocess
        import sys
        import time

        logger.info("Spawning background daemon...")
        log_file = open(
            "/home/de3f4ault/Desktop/Projects/audiobench/data/logs/daemon_autostart.log", "w"
        )
        subprocess.Popen(
            [sys.executable, "-m", "audiobench.cli.main", "daemon", "start"],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )

        # Poll until responsive
        for _ in range(120):  # Wait up to 60 seconds (120 * 0.5s)
            time.sleep(0.5)
            # Use raw ping check to avoid infinite recursion
            if self._raw_ping():
                logger.info("Background daemon is now responsive.")
                return

        raise RuntimeError("Failed to auto-start daemon: timed out waiting for ping.")

    def _raw_ping(self) -> bool:
        """Internal ping that doesn't trigger auto-start."""
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            try:
                sock.connect(str(self._socket_path))
                request_id = str(uuid.uuid4())
                payload = json.dumps({"cmd": "ping", "args": {}, "request_id": request_id}) + "\n"
                sock.sendall(payload.encode("utf-8"))
                raw = sock.recv(4096)
                response = json.loads(raw)
                return bool(response.get("success"))
            finally:
                sock.close()
        except Exception:
            return False

    def _send(self, cmd: str, args: dict, _is_retry: bool = False) -> dict:
        """Send one JSON request and read one JSON response.

        Raises:
            ConnectionRefusedError: if socket is absent or daemon is not running (after retry).
            TimeoutError: if the daemon doesn't respond within the timeout.
            RuntimeError: if the daemon returns an error response.
        """
        request_id = str(uuid.uuid4())
        payload = json.dumps({"cmd": cmd, "args": args, "request_id": request_id}) + "\n"

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(_DEFAULT_TIMEOUT)
        try:
            sock.connect(str(self._socket_path))
            sock.sendall(payload.encode("utf-8"))

            # Read until newline
            chunks: list[bytes] = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
                if b"\n" in chunk:
                    break
            raw = b"".join(chunks).strip()
        except (ConnectionRefusedError, FileNotFoundError):
            sock.close()
            if not _is_retry:
                logger.debug("Daemon not reachable, attempting auto-start...")
                self._ensure_daemon_running()
                return self._send(cmd, args, _is_retry=True)
            raise
        finally:
            # We already closed the socket if there was an exception, but close it here on success
            sock.close()

        response: dict = json.loads(raw)
        if not response.get("success"):
            raise RuntimeError(f"Daemon error [{cmd}]: {response.get('error', 'unknown')}")
        return dict(response.get("data", {}))

    # ------------------------------------------------------------------
    # RetrievalClient interface
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Return True if the daemon is alive and responsive."""
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            try:
                sock.connect(str(self._socket_path))
                request_id = str(uuid.uuid4())
                payload = json.dumps({"cmd": "ping", "args": {}, "request_id": request_id}) + "\n"
                sock.sendall(payload.encode("utf-8"))
                raw = sock.recv(4096)
                response = json.loads(raw)
                return bool(response.get("success"))
            finally:
                sock.close()
        except Exception:
            return False

    def search(
        self,
        query: str,
        top_k: int = 5,
        speaker_filter: str | None = None,
        hyde_document: str | None = None,
        use_bm25: bool = True,
        use_dense: bool = True,
        use_colbert: bool = True,
    ) -> list[SearchResult]:
        """Hybrid search over memory store via daemon."""
        args = {
            "query": query,
            "top_k": top_k,
            "speaker_filter": speaker_filter,
            "hyde_document": hyde_document,
            "use_bm25": use_bm25,
            "use_dense": use_dense,
            "use_colbert": use_colbert,
        }
        data = self._send("search", args)
        return list(data.get("results", []))

    def embed(
        self,
        expression_id: int,
        content: str,
        source_type: SourceType,
        speaker: str | None = None,
    ) -> None:
        """Embed and persist an expression to LanceDB."""
        args: dict = {
            "expression_id": expression_id,
            "content": content,
            "source_type": source_type.value,
        }
        if speaker is not None:
            args["speaker"] = speaker
        self._send("embed", args)

    def delete(self, expression_id: int) -> None:
        """Remove an expression from LanceDB."""
        self._send("delete", {"expression_id": expression_id})

    def status(self) -> dict:
        """Return daemon status dict."""
        return self._send("status", {})

    def chunk(
        self, text: str, audio_file_id: int, diarized: bool, segments: list[dict] | None = None
    ) -> list[ChunkResult]:
        """Chunk text via the daemon's chunking pipeline (Phase 4)."""
        args = {"text": text, "audio_file_id": audio_file_id, "diarized": diarized}
        if segments is not None:
            args["segments"] = segments
        data = self._send("chunk", args)
        return list(data.get("chunks", []))

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        """Rerank a list of documents against a query using the daemon's ML model."""
        data = self._send("rerank", {"query": query, "docs": docs})
        return list(data.get("scores", []))

    def check_cache(self, query: str, distance_threshold: float = 0.05) -> dict | None:
        """Check semantic cache via daemon."""
        data = self._send("check_cache", {"query": query, "distance_threshold": distance_threshold})
        return data.get("result")

    def write_cache(self, query: str, answer: str, hyde_document: str | None = None) -> None:
        """Write to semantic cache via daemon."""
        args = {"query": query, "answer": answer}
        if hyde_document:
            args["hyde_document"] = hyde_document
        self._send("write_cache", args)
