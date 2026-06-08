"""Daemon server — asyncio Unix socket server for the memory layer.

Holds all ML models resident in memory and serves requests from CLI commands
via newline-delimited JSON over a Unix domain socket.

Protocol: {"cmd": str, "args": {...}, "request_id": str}  →  {"success": bool, "data": {...}, "request_id": str}
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.daemon.protocol import DaemonRequest, DaemonResponse
from audiobench.memory.chunking import content_aware_router
from audiobench.memory.memory_store import MemoryStore
from audiobench.memory.singletons import get_reranker, pre_warm_retrieval_pipeline
from audiobench.storage.models import TranscriptionRecord

logger = get_logger("daemon.server")

# Module-level state (lives for the lifetime of the daemon process)
_start_time: float = 0.0
_memory_store: MemoryStore | None = None


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _get_store() -> MemoryStore:
    """Get the module-level MemoryStore instance (must have been initialised)."""
    if _memory_store is None:
        raise RuntimeError("MemoryStore not initialised — was pre_warm called?")
    return _memory_store


def _handle_ping(args: dict[str, Any]) -> dict[str, Any]:
    """Return daemon liveness info."""
    settings = get_settings()
    return {
        "alive": True,
        "uptime_seconds": round(time.time() - _start_time, 2),
        "embedding_model_version": _get_store().model_version,
    }


def _handle_embed(args: dict[str, Any]) -> dict[str, Any]:
    """Embed and write a single expression to LanceDB."""
    _get_store().write_node(
        expression_id=int(args["expression_id"]),
        content=str(args["content"]),
        source_type=str(args["source_type"]),
        speaker=args.get("speaker"),
    )
    return {"embedded": True}


def _handle_search(args: dict[str, Any]) -> dict[str, Any]:
    """Hybrid search over memory store."""
    query = str(args["query"])
    top_k = int(args.get("top_k", 5))
    speaker_filter = args.get("speaker_filter")
    hyde_document = args.get("hyde_document")
    use_bm25 = args.get("use_bm25", True)
    use_dense = args.get("use_dense", True)
    use_colbert = args.get("use_colbert", True)

    results = _get_store().search(
        query=query,
        top_k=top_k,
        speaker_filter=speaker_filter,
        hyde_document=hyde_document,
        use_bm25=use_bm25,
        use_dense=use_dense,
        use_colbert=use_colbert,
    )

    return {"results": results}


def _handle_delete(args: dict[str, Any]) -> dict[str, Any]:
    """Delete an expression node from LanceDB."""
    _get_store().delete_node(int(args["expression_id"]))
    return {"deleted": True}


def _handle_status(args: dict[str, Any]) -> dict[str, Any]:
    """Return daemon and store statistics."""
    store = _get_store()
    node_count = store.count_nodes()
    return {
        "uptime_seconds": round(time.time() - _start_time, 2),
        "embedding_model_version": store.model_version,
        "total_nodes": node_count,
    }


def _handle_chunk(args: dict[str, Any]) -> dict[str, Any]:
    """Run text through the chunking pipeline and return chunks."""
    text = str(args.get("text", ""))
    # audio_file_id is available but chunking is purely semantic/text-based right now
    diarized = bool(args.get("diarized", False))

    # We don't have the diarized segments here. The protocol currently sends text and a diarized bool.
    # To fully support diarization in the daemon, we should accept a segments list in args.
    # For now, we pass None to segments since the protocol doesn't send them yet.
    # We will update protocol later if needed.
    segments = args.get("segments")

    chunks = content_aware_router(text, diarized_segments=segments if diarized else None)

    results = []
    for c in chunks:
        res = {"content": c.content, "uuid": c.uuid, "tier": c.tier}
        if c.speaker:
            res["speaker"] = c.speaker
        results.append(res)

    return {"chunks": results}


def _handle_rerank(args: dict[str, Any]) -> dict[str, Any]:
    """Rerank search results using the CrossEncoder."""
    query = str(args["query"])
    docs = args.get("docs", [])

    if not docs:
        return {"scores": []}

    reranker = get_reranker()
    pairs = [(query, doc) for doc in docs]

    # predict returns a numpy array, must convert to list
    scores = reranker.predict(pairs).tolist()

    return {"scores": scores}


def _handle_check_cache(args: dict[str, Any]) -> dict[str, Any]:
    """Check semantic cache."""
    from audiobench.memory.memory_store import QueryCacheStore

    global _query_cache
    if "_query_cache" not in globals() or _query_cache is None:
        _query_cache = QueryCacheStore()

    query = str(args["query"])
    dist = float(args.get("distance_threshold", 0.05))
    result = _query_cache.check_cache(query, dist)
    return {"result": result}


def _handle_write_cache(args: dict[str, Any]) -> dict[str, Any]:
    """Write to semantic cache."""
    from audiobench.memory.memory_store import QueryCacheStore

    global _query_cache
    if "_query_cache" not in globals() or _query_cache is None:
        _query_cache = QueryCacheStore()

    query = str(args["query"])
    answer = str(args["answer"])
    hyde_document = args.get("hyde_document")
    _query_cache.write_cache(query, answer, hyde_document)
    return {"cached": True}


_HANDLERS: dict[str, Any] = {
    "ping": _handle_ping,
    "embed": _handle_embed,
    "search": _handle_search,
    "delete": _handle_delete,
    "status": _handle_status,
    "chunk": _handle_chunk,
    "rerank": _handle_rerank,
    "check_cache": _handle_check_cache,
    "write_cache": _handle_write_cache,
}


# ---------------------------------------------------------------------------
# Request dispatch
# ---------------------------------------------------------------------------


def _dispatch(raw: str) -> str:
    """Parse a JSON request line, call handler, return JSON response line."""
    request_id = "unknown"
    try:
        req: DaemonRequest = json.loads(raw)
        request_id = req.get("request_id", "unknown")  # type: ignore[assignment]
        cmd = req.get("cmd", "")
        args = req.get("args", {})

        handler = _HANDLERS.get(cmd)
        if handler is None:
            raise ValueError(f"Unknown command: {cmd!r}")

        data = handler(args)
        response: DaemonResponse = {"success": True, "data": data, "request_id": request_id}
    except Exception as exc:
        logger.exception("Error handling request %s", request_id)
        response = {"success": False, "error": str(exc), "request_id": request_id}

    return json.dumps(response)


# ---------------------------------------------------------------------------
# asyncio connection handler
# ---------------------------------------------------------------------------


async def _handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Handle a single client connection — reads one request, writes one response."""
    peer = writer.get_extra_info("peername", "<unknown>")
    try:
        raw = await asyncio.wait_for(reader.readline(), timeout=10.0)
        if not raw:
            return
        raw_str = raw.decode("utf-8").strip()
        logger.debug("Received from %s: %s", peer, raw_str[:120])

        # Run the (potentially blocking) handler in the default thread executor
        # to keep the event loop free for new connections
        loop = asyncio.get_running_loop()
        response_str = await loop.run_in_executor(None, _dispatch, raw_str)

        writer.write((response_str + "\n").encode("utf-8"))
        await writer.drain()
    except TimeoutError:
        logger.warning("Connection from %s timed out", peer)
    except (ConnectionResetError, BrokenPipeError):
        logger.warning("Client %s disconnected before response could be sent", peer)
    except Exception:
        logger.exception("Unexpected error for connection from %s", peer)
    finally:
        writer.close()
        await writer.wait_closed()


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def _write_pid_file(pid_path: Path) -> None:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))
    logger.info("PID file written: %s", pid_path)


def _cleanup(socket_path: Path, pid_path: Path) -> None:
    for p in (socket_path, pid_path):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
    logger.info("Cleanup complete")


def _rag_consistency_sweep_sync():
    """Background task to sync transcripts that were not indexed to the vector db."""
    try:
        # Wait a bit before starting the sweep to allow the daemon to fully initialize
        # and respond to the client's 'status' pings.
        time.sleep(10)

        logger.info("Starting background RAG consistency sweep...")
        with get_session() as session:
            unindexed = (
                session.query(TranscriptionRecord).filter(TranscriptionRecord.is_indexed == 0).all()
            )

            if not unindexed:
                logger.info("RAG consistency sweep: All transcripts are indexed.")
                return

            logger.info(
                f"RAG consistency sweep: Found {len(unindexed)} unindexed transcripts. Indexing..."
            )

            from audiobench.memory.enums import SourceType
            from audiobench.storage.expression_repository import ExpressionRepository

            expr_repo = ExpressionRepository()

            success_count = 0
            for transcript in unindexed:
                try:
                    text = transcript.full_text or ""
                    if not text.strip():
                        transcript.is_indexed = 1
                        continue

                    # Semantic chunking
                    chunks = content_aware_router(text)
                    for chunk in chunks:
                        # Register in SQLite (which deduplicates via content_hash)
                        expr = expr_repo.register(
                            content=chunk.content,
                            source_type=SourceType.AUDIO_TRANSCRIPT.value,
                            source_id=transcript.id,
                            speaker=chunk.speaker,
                        )
                        # Embed in LanceDB
                        _get_store().write_node(
                            expression_id=expr.id,
                            content=expr.content,
                            source_type=SourceType.AUDIO_TRANSCRIPT.value,
                            speaker=chunk.speaker,
                        )

                    transcript.is_indexed = 1
                    success_count += 1

                    # Prevent locking LanceDB entirely
                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"Failed to index transcript {transcript.id}: {e}")

            session.commit()
            logger.info(
                f"RAG consistency sweep: Successfully indexed {success_count}/{len(unindexed)} transcripts."
            )
    except Exception as e:
        logger.error(f"RAG consistency sweep failed: {e}")


async def _serve(socket_path: Path, pid_path: Path) -> None:
    """Start the Unix socket server and serve until SIGTERM/SIGINT."""
    global _start_time, _memory_store

    # Remove stale socket from previous crash
    socket_path.unlink(missing_ok=True)

    # --- warm models BEFORE writing PID file so the client waits properly ---
    logger.info("Pre-warming retrieval pipeline...")
    await asyncio.get_running_loop().run_in_executor(None, pre_warm_retrieval_pipeline)
    logger.info("Models warm. Initialising MemoryStore...")
    _memory_store = await asyncio.get_running_loop().run_in_executor(None, MemoryStore)

    _start_time = time.time()
    _write_pid_file(pid_path)

    server = await asyncio.start_unix_server(
        _handle_connection, 
        path=str(socket_path), 
        limit=104857600  # 100MB limit for massive transcript JSON payloads
    )
    logger.info("Daemon listening on %s (pid=%d)", socket_path, os.getpid())

    # Start background RAG sync task
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _rag_consistency_sweep_sync)

    # Set permissions so any user can connect (adjust to 0o600 for single-user)
    socket_path.chmod(0o666)

    stop_event = asyncio.Event()

    def _on_signal() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, _on_signal)
    loop.add_signal_handler(signal.SIGINT, _on_signal)

    async with server:
        await stop_event.wait()
        server.close()
        await server.wait_closed()

    _cleanup(socket_path, pid_path)


def run() -> None:
    """Entry point — start the daemon (blocks until SIGTERM/SIGINT)."""
    settings = get_settings()
    socket_path = Path(settings.daemon_socket_path)
    pid_path = Path(settings.daemon_pid_path)

    asyncio.run(_serve(socket_path, pid_path))
