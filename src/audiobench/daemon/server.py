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
from audiobench.memory.memory_store import MemoryStore, SegmentVectorStore
from audiobench.memory.singletons import get_primary_embedder, get_reranker, pre_warm_retrieval_pipeline
from audiobench.storage.models import TranscriptionRecord

logger = get_logger("daemon.server")

# Module-level state (lives for the lifetime of the daemon process)
_start_time: float = 0.0
_memory_store: MemoryStore | None = None
_segment_store: SegmentVectorStore | None = None
_sweep_state = None  # audiobench.daemon.sweep_state.SweepState — loaded lazily


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _get_store() -> MemoryStore:
    """Get the module-level MemoryStore instance (must have been initialised)."""
    if _memory_store is None:
        raise RuntimeError("MemoryStore not initialised — was pre_warm called?")
    return _memory_store


def _get_segment_store() -> SegmentVectorStore:
    """Get the module-level SegmentVectorStore instance (must have been initialised)."""
    if _segment_store is None:
        raise RuntimeError("SegmentVectorStore not initialised — was pre_warm called?")
    return _segment_store


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
    """Rerank search results using the CrossEncoder.

    Inputs are truncated before forming pairs to prevent tensor size mismatches.
    CrossEncoders (ms-marco-MiniLM series) have a 512-token limit for the
    concatenated [CLS] query [SEP] doc [SEP] input. Long HyDE documents or
    long segment texts can exceed this, causing a RuntimeError.

    Conservative limits: query ≤ 512 chars, doc ≤ 1024 chars — well within
    the 512-token window even at average token density of ~4 chars/token.
    """
    query = str(args["query"])[:512]
    docs = args.get("docs", [])

    if not docs:
        return {"scores": []}

    # Truncate each document to prevent tensor mismatch
    truncated_docs = [str(d)[:1024] for d in docs]

    reranker = get_reranker()
    pairs = [(query, doc) for doc in truncated_docs]

    # predict returns a numpy array, must convert to list
    scores = reranker.predict(pairs).tolist()

    return {"scores": scores}


def _handle_embed_query(args: dict[str, Any]) -> dict[str, Any]:
    """Embed a query string using the daemon's warm Nomic model.

    Returns a 768-dim vector. The caller (DenseStream/ColBERTStream) uses this
    vector for ANN search rather than cold-loading the model in the CLI process.
    """
    from audiobench.memory.singletons import get_primary_inference_lock

    text = str(args["text"])[:12_000]  # bound token count
    prefixed = f"search_query: {text}"
    model = get_primary_embedder()
    with get_primary_inference_lock():
        vector = model.encode(prefixed).tolist()
    return {"vector": vector}


def _handle_embed_segment(args: dict[str, Any]) -> dict[str, Any]:
    """Embed a single audio segment and write it to the segment_vectors table.

    Generates the vector using the warm Nomic model, then delegates the
    upsert to SegmentVectorStore.upsert_segment_with_vector so the table
    write and the embedding happen in the same daemon process.
    """
    from audiobench.memory.singletons import get_primary_inference_lock

    segment_id = int(args["segment_id"])
    text = str(args["text"])[:12_000]  # bound token count
    start_time = float(args["start_time"])
    end_time = float(args["end_time"])
    source_file = str(args["source_file"])

    prefixed = f"search_document: {text}"
    model = get_primary_embedder()
    with get_primary_inference_lock():
        vector = model.encode(prefixed).tolist()

    _get_segment_store().upsert_segment_with_vector(
        segment_id=segment_id,
        text=text,
        start_time=start_time,
        end_time=end_time,
        source_file=source_file,
        vector=vector,
    )
    return {"embedded": True}


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
    "embed_query": _handle_embed_query,
    "embed_segment": _handle_embed_segment,
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


def _do_sweep_once() -> None:
    """Perform one pass of the RAG consistency sweep (called from the loop)."""
    from audiobench.daemon.sweep_state import get_sweep_state
    from audiobench.memory.enums import SourceType
    from audiobench.storage.expression_repository import ExpressionRepository
    from sqlalchemy import text as sql_text

    state = get_sweep_state()

    # ── Pick up any new work added since last tick ─────────────────────────
    _refresh_sweep_state_incremental(state)

    # ── Expression sweep (transcript deque → expression chunks → LanceDB) ──
    tx_ids = state.pop_transcript_batch(size=20)
    if not tx_ids:
        logger.info("RAG sweep: all transcripts indexed (deque empty).")
    else:
        logger.info("RAG sweep: indexing %d transcripts from deque...", len(tx_ids))
        expr_repo = ExpressionRepository()
        success_ids: list[int] = []
        failed_ids: list[int] = []

        with get_session() as session:
            transcripts = (
                session.query(TranscriptionRecord)
                .filter(TranscriptionRecord.id.in_(tx_ids))
                .all()
            )
            for transcript in transcripts:
                try:
                    text = transcript.full_text or ""
                    if not text.strip():
                        transcript.is_indexed = 1
                        success_ids.append(transcript.id)
                        continue

                    chunks = content_aware_router(text)
                    if chunks:
                        chunk_items = [
                            {
                                "content": chunk.content,
                                "source_type": SourceType.AUDIO_TRANSCRIPT.value,
                                "source_id": transcript.id,
                                "speaker": chunk.speaker,
                            }
                            for chunk in chunks
                        ]
                        # O(1) dedup — no SQL IN() query for known hashes
                        registered = expr_repo.register_batch(
                            chunk_items,
                            known_hashes=state.known_hashes,
                        )
                        # Batch embed + write — one model forward-pass per transcript
                        nodes = [
                            {
                                "expression_id": expr.id,
                                "content": expr.content,
                                "source_type": SourceType.AUDIO_TRANSCRIPT.value,
                                "speaker": expr.speaker,
                            }
                            for expr in registered
                        ]
                        _get_store().batch_write_nodes(
                            nodes,
                            indexed_ids=state.indexed_expression_ids,
                        )

                    transcript.is_indexed = 1
                    success_ids.append(transcript.id)
                except Exception as e:
                    logger.error(
                        "RAG sweep: failed to index transcript %d: %s",
                        transcript.id, e,
                    )
                    failed_ids.append(transcript.id)

            session.commit()

        logger.info(
            "RAG sweep: indexed %d transcripts (%d failed).",
            len(success_ids), len(failed_ids),
        )
        if failed_ids:
            state.requeue_transcripts(failed_ids)

    # ── Segment sweep (segment deque → batch embed → LanceDB + SQLite) ─────
    seg_ids = state.pop_segment_batch(size=256)
    if not seg_ids:
        logger.info("RAG sweep: all segments vectorized (deque empty).")
    else:
        logger.info("RAG sweep: vectorizing %d segments from deque...", len(seg_ids))
        with get_session() as session:
            placeholders = ", ".join(str(i) for i in seg_ids)
            rows = session.execute(
                sql_text(
                    f"SELECT s.id, s.text, s.start_time, s.end_time, af.file_path "
                    f"FROM segments s "
                    f"JOIN transcriptions t ON s.transcription_id = t.id "
                    f"JOIN audio_files af ON t.audio_file_id = af.id "
                    f"WHERE s.id IN ({placeholders})"
                )
            ).mappings().all()
            rows = list(rows)

        if rows:
            from audiobench.memory.singletons import get_primary_inference_lock
            seg_store = _get_segment_store()
            model = get_primary_embedder()
            try:
                texts = [f"search_document: {r['text'][:12_000]}" for r in rows]
                with get_primary_inference_lock():
                    vectors = model.encode(
                        texts, batch_size=64, show_progress_bar=False,
                        sort_by_length=True,
                    ).tolist()
                seg_store.batch_upsert_segments(
                    rows=[dict(r) for r in rows], vectors=vectors
                )
                done_ids = [int(r["id"]) for r in rows]
                with get_session() as upd:
                    id_list = ", ".join(str(i) for i in done_ids)
                    upd.execute(
                        sql_text(
                            f"UPDATE segments SET vector_indexed=1 WHERE id IN ({id_list})"
                        )
                    )
                    upd.commit()
                logger.info(
                    "RAG sweep: vectorized %d segments in one batch.", len(rows)
                )
            except Exception as seg_exc:
                logger.error(
                    "RAG sweep: failed to batch vectorize segments: %s", seg_exc
                )
                state.requeue_segments(seg_ids)


def _refresh_sweep_state_incremental(state) -> None:
    """Pick up any transcripts or segments added since the last sweep tick.

    Only queries for IDs **not already in the deques** by comparing the
    current DB un-indexed set against the tail of the deque.  This avoids
    re-queuing duplicates after restarts while still catching new work.
    """
    from sqlalchemy import text as sql_text
    from audiobench.core.db_session import get_session as _gs

    try:
        with _gs() as session:
            # New un-indexed transcripts
            new_tx = session.execute(
                sql_text("SELECT id FROM transcriptions WHERE is_indexed=0 ORDER BY id")
            ).scalars().all()
            queued_tx = set(state.unindexed_transcript_ids)
            fresh_tx = [i for i in new_tx if i not in queued_tx]
            if fresh_tx:
                state.push_transcripts(fresh_tx)
                logger.debug(
                    "SweepState refresh: +%d new transcript IDs", len(fresh_tx)
                )

            # New un-vectorized segments
            new_seg = session.execute(
                sql_text(
                    "SELECT id FROM segments WHERE vector_indexed=0 ORDER BY id"
                )
            ).scalars().all()
            queued_seg = set(state.pending_segment_ids)
            fresh_seg = [i for i in new_seg if i not in queued_seg]
            if fresh_seg:
                state.push_segments(fresh_seg)
                logger.debug(
                    "SweepState refresh: +%d new segment IDs", len(fresh_seg)
                )
    except Exception as exc:
        logger.warning("SweepState incremental refresh failed: %s", exc)


def _rag_consistency_sweep_sync() -> None:
    """Periodic background worker — runs forever, sweeping every 5 minutes.

    The first pass starts after a 10-second warm-up delay so the daemon can
    finish initialisation and start accepting client connections before any
    heavy embedding work begins.
    """
    SWEEP_INTERVAL_SECONDS = 300  # 5 minutes

    # Initial delay — let the daemon become ready first.
    time.sleep(10)

    while True:
        try:
            _do_sweep_once()
        except Exception as exc:
            logger.error("RAG sweep loop: unexpected error: %s", exc)
        logger.info("RAG sweep: sleeping %ds until next pass.", SWEEP_INTERVAL_SECONDS)
        time.sleep(SWEEP_INTERVAL_SECONDS)


async def _serve(socket_path: Path, pid_path: Path) -> None:
    """Start the Unix socket server and serve until SIGTERM/SIGINT."""
    global _start_time, _memory_store, _segment_store, _sweep_state

    # Remove stale socket from previous crash
    socket_path.unlink(missing_ok=True)

    # --- warm models BEFORE writing PID file so the client waits properly ---
    logger.info("Pre-warming retrieval pipeline...")
    await asyncio.get_running_loop().run_in_executor(None, pre_warm_retrieval_pipeline)
    logger.info("Models warm. Initialising stores...")
    _memory_store = await asyncio.get_running_loop().run_in_executor(None, MemoryStore)
    _segment_store = await asyncio.get_running_loop().run_in_executor(None, SegmentVectorStore)
    logger.info("SegmentVectorStore ready (%d segments embedded).", _segment_store.count_embedded())

    # --- Initialise the in-memory O(1) sweep state -------------------------
    from audiobench.daemon.sweep_state import init_sweep_state

    _sweep_state = init_sweep_state()
    await asyncio.get_running_loop().run_in_executor(None, _sweep_state.load_from_db)
    await asyncio.get_running_loop().run_in_executor(
        None, _sweep_state.sync_indexed_expression_ids
    )
    logger.info(
        "SweepState ready — %d known hashes | %d LanceDB expr IDs | "
        "%d pending segments | %d unindexed transcripts",
        len(_sweep_state.known_hashes),
        len(_sweep_state.indexed_expression_ids),
        _sweep_state.pending_segment_count(),
        _sweep_state.unindexed_transcript_count(),
    )

    _start_time = time.time()
    _write_pid_file(pid_path)


    server = await asyncio.start_unix_server(
        _handle_connection, 
        path=str(socket_path), 
        limit=104857600  # 100MB limit for massive transcript JSON payloads
    )
    logger.info("Daemon listening on %s (pid=%d)", socket_path, os.getpid())

    # Start background RAG sync loop — runs in a daemon thread, sweeping every 5 min.
    import threading
    sweep_thread = threading.Thread(
        target=_rag_consistency_sweep_sync,
        name="rag-sweep",
        daemon=True,  # killed automatically when the main process exits
    )
    sweep_thread.start()
    logger.info("RAG sweep thread started (interval=300s).")

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
    from audiobench.core.logger_factory import setup_logging
    setup_logging("DEBUG")
    
    import os
    os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
    os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 4))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    settings = get_settings()
    socket_path = Path(settings.daemon_socket_path)
    pid_path = Path(settings.daemon_pid_path)

    from audiobench.observatory.db import init_journal_db
    from audiobench.observatory.subscriber import get_subscriber
    from audiobench.events import get_bus

    init_journal_db()
    get_bus().on("*", get_subscriber().record)

    asyncio.run(_serve(socket_path, pid_path))
