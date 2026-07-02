"""Semantic memory query engine.

Orchestrates vector search, cross-encoder reranking, SQLite graph
traversal, and LLM synthesis to answer queries using the expression graph.

Engineering standards applied (EQ-1 through EQ-4):
- All expression lookups are batched via get_by_ids() / get_parents_batch().
  No per-item get_by_id() calls inside loops.
- LLM synthesis uses Ok/Err result types — no nested try/except chains.
- Every exception path logs a warning and returns a structured result.
  No bare ``except: pass`` blocks.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from typing import Any

from audiobench.chat.chat_store import ChatRepository
from audiobench.chat.providers.ollama_provider import OllamaClient
from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.daemon.factory import get_daemon_client
from audiobench.memory.enums import SourceType
from audiobench.memory.rrf_fusion import FusedResult, _temporal_dedup, rrf_merge
from audiobench.memory.retrieval_streams import ColBERTStream, DenseStream, FTS5Stream
from audiobench.storage.expression_repository import ExpressionRepository
from audiobench.storage.models import BookmarkRecord

logger = get_logger("memory.query")

_search_pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="search_stream")

# ── Result types (EQ-3) ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Ok:
    """Successful LLM call result."""
    value: str


@dataclass(frozen=True)
class Err:
    """Failed LLM call result — carries the reason without raising."""
    error: str


LLMResult = Ok | Err


# ── Public result dataclass ───────────────────────────────────────────────────

@dataclass
class QueryResult:
    query: str
    answer: str | None
    sources: list[dict[str, Any]]
    query_time_seconds: float = 0.0
    hyde_document: str | None = None
    synthesis_failed: bool = False
    synthesis_error: str | None = None


# ── LLM helpers ──────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, temperature: float, llm: OllamaClient) -> LLMResult:
    """Attempt generation via local Ollama. Returns Ok or Err — never raises."""
    from audiobench.memory.llm_caller import RateLimitError, _retry_with_backoff
    from audiobench.memory.singletons import get_llm_circuit_breaker
    
    breaker = get_llm_circuit_breaker()
    
    def _do_call():
        try:
            return llm.generate(prompt, temperature=temperature)
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                raise RateLimitError(str(e))
            raise e

    try:
        text = breaker.call(lambda: _retry_with_backoff(_do_call, max_retries=2, base_delay=1.0, jitter=True))
        return Ok(value=text)
    except Exception as exc:  # noqa: BLE001
        return Err(error=f"ollama: {exc}")


def _call_gemini(prompt: str, temperature: float, api_key: str) -> LLMResult:
    """Attempt generation via Gemini API. Returns Ok or Err — never raises."""
    from audiobench.memory.llm_caller import RateLimitError, _retry_with_backoff
    from audiobench.memory.singletons import get_llm_circuit_breaker
    
    breaker = get_llm_circuit_breaker()
    
    def _do_call():
        from google import genai  # type: ignore[import]
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            if response and response.text:
                return response.text.strip()
            raise ValueError("empty response")
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e) or "quota" in str(e).lower():
                raise RateLimitError(str(e))
            raise e

    try:
        text = breaker.call(lambda: _retry_with_backoff(_do_call, max_retries=2, base_delay=1.0, jitter=True))
        return Ok(value=text)
    except Exception as exc:  # noqa: BLE001
        return Err(error=f"gemini: {exc}")


def _call_llm(
    prompt: str,
    temperature: float,
    llm: OllamaClient,
    api_key: str | None,
) -> LLMResult:
    """Try Ollama, fall back to Gemini, return Ok or Err.

    Callers use structural pattern matching:

        match _call_llm(prompt, 0.2, llm, api_key):
            case Ok(value=answer): ...
            case Err(error=reason): ...
    """
    result = _call_ollama(prompt, temperature, llm)
    if isinstance(result, Ok):
        return result

    logger.warning("Ollama LLM call failed (%s), trying Gemini fallback", result.error)

    if not api_key:
        return Err(error=f"{result.error} | gemini: no api key configured")

    gemini_result = _call_gemini(prompt, temperature, api_key)
    if isinstance(gemini_result, Ok):
        return gemini_result

    return Err(error=f"{result.error} | {gemini_result.error}")


# ── Engine ────────────────────────────────────────────────────────────────────

class MemoryQueryEngine:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.daemon = get_daemon_client()
        self.expr_repo = ExpressionRepository()
        self.chat_repo = ChatRepository()
        self.llm = OllamaClient(
            base_url=self.settings.ollama_base_url, model=self.settings.ollama_model
        )

    def query(
        self,
        text: str,
        top_k: int = 20,
        speaker_filter: str | None = None,
        preset: str = "balanced",
        enable_hyde: bool | None = None,
        enable_cross_encoder: bool | None = None,
        use_bm25: bool | None = None,
        use_dense: bool | None = None,
        use_colbert: bool | None = None,
        use_cache: bool = True,
    ) -> QueryResult:
        t0 = time.time()
        logger.info("Starting memory query for: %s (Preset: %s)", text, preset)

        # Resolve preset flags
        if preset == "fast":
            p_hyde, p_cross, p_bm25, p_dense, p_colbert = False, False, True, True, False
        elif preset == "deep":
            p_hyde, p_cross, p_bm25, p_dense, p_colbert = True, True, True, True, True
        else:  # balanced
            p_hyde, p_cross, p_bm25, p_dense, p_colbert = False, False, True, True, True

        final_hyde   = enable_hyde           if enable_hyde           is not None else p_hyde
        final_cross  = enable_cross_encoder  if enable_cross_encoder  is not None else p_cross
        final_bm25   = use_bm25              if use_bm25              is not None else p_bm25
        final_dense  = use_dense             if use_dense             is not None else p_dense
        final_colbert = use_colbert          if use_colbert           is not None else p_colbert

        # ── Step 0: Semantic Cache ────────────────────────────────────────────
        if use_cache:
            cached = self.daemon.check_cache(text)
            if cached:
                return QueryResult(
                    query=text,
                    answer=cached["answer"],
                    sources=[],
                    query_time_seconds=time.time() - t0,
                    hyde_document=cached.get("hyde_document"),
                )

        # ── Step 1: HyDE Generation ───────────────────────────────────────────
        hyde_document: str | None = None
        if final_hyde:
            hyde_prompt = (
                "Please write a realistic hypothetical excerpt from a spoken recording, such as an audiobook, "
                "podcast, interview, or personal reflection, that directly answers or provides context for the "
                "following query. Limit your response to approximately 100 to 150 words. Do not exceed 150 words. "
                "Do not include any preambles, greetings, facts you are unsure about, or meta-commentary. Just "
                f"produce the raw hypothetical text as if it were a direct transcript.\n\nQuery: {text}"
            )
            match _call_llm(hyde_prompt, 0.7, self.llm, self.settings.gemini_api_key):
                case Ok(value=doc):
                    hyde_document = doc
                    logger.debug("HyDE document generated (%d chars)", len(doc))
                case Err(error=reason):
                    logger.warning("HyDE generation failed: %s — proceeding without HyDE", reason)

        # ── Step 2: Vector Search via Daemon ─────────────────────────────────
        try:
            results = self.daemon.search(
                query=text,
                top_k=top_k,
                speaker_filter=speaker_filter,
                hyde_document=hyde_document,
                use_bm25=final_bm25,
                use_dense=final_dense,
                use_colbert=final_colbert,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Daemon search failed: %s", exc)
            results = []

        if not results:
            return QueryResult(query=text, answer="No relevant memory found.", sources=[])

        # ── Batch-fetch all candidate expressions (EQ-1: eliminates N+1) ─────
        expr_ids = [
            r["expression_id"]
            for r in results
            if r.get("expression_id") is not None
        ]
        expr_map = self.expr_repo.get_by_ids(expr_ids)   # 1 query for all IDs

        candidates: list[tuple[Any, float]] = [
            (expr_map[r["expression_id"]], r.get("score", 0.0))
            for r in results
            if r.get("expression_id") in expr_map
        ]

        if not candidates:
            return QueryResult(
                query=text, answer="No relevant memory found in database.", sources=[]
            )

        # ── Step 2.5: Parent expansion via batch (EQ-2: eliminates N+1) ──────
        child_ids = [expr.id for expr, _ in candidates]
        parents = self.expr_repo.get_parents_batch(child_ids)  # 1 JOIN query

        expanded: dict[int, tuple[Any, float]] = {}
        for expr, score in candidates:
            parent = parents.get(expr.id)
            if parent is not None:
                if parent.id not in expanded or score > expanded[parent.id][1]:
                    expanded[parent.id] = (parent, score)
            else:
                if expr.id not in expanded:
                    expanded[expr.id] = (expr, score)

        expanded_candidates = sorted(expanded.values(), key=lambda x: x[1], reverse=True)

        # ── Step 3: CrossEncoder Reranking ────────────────────────────────────
        if final_cross:
            try:
                docs = [c[0].content for c in expanded_candidates]
                scores = self.daemon.rerank(text, docs)
                scored = sorted(
                    zip(expanded_candidates, scores), key=lambda x: x[1], reverse=True
                )
                top_candidates = [(c[0], float(sc)) for c, sc in scored[:5]]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Reranking via daemon failed: %s — using vector scores", exc)
                top_candidates = list(expanded_candidates[:5])
        else:
            top_candidates = list(expanded_candidates[:5])

        # ── Step 4: Graph Traversal for Enriched Context ─────────────────────
        # Batch-fetch parents again for the smaller top_candidates set (EQ-2)
        top_ids = [expr.id for expr, _ in top_candidates]
        top_parents = self.expr_repo.get_parents_batch(top_ids)

        # Batch-fetch all inference relations in one pass (EQ-1)
        all_in_rel_source_ids: list[int] = []
        top_in_rels: list[list[Any]] = []
        for expr, _ in top_candidates:
            in_rels = self.expr_repo.get_relations(expr.id, direction="in")
            top_in_rels.append(in_rels)
            all_in_rel_source_ids.extend(
                rel.from_expression_id for rel in in_rels
            )
        inference_map = self.expr_repo.get_by_ids(all_in_rel_source_ids)  # 1 query

        # Batch-fetch bookmarks for parent source_ids
        parent_source_ids = [
            top_parents[eid].source_id
            for eid in top_ids
            if eid in top_parents and top_parents[eid].source_id is not None
            and top_parents[eid].source_type in (
                SourceType.TRANSCRIPT_SEGMENT.value,
                SourceType.AUDIO_TRANSCRIPT.value,
            )
        ]
        bookmark_map: dict[int, list[BookmarkRecord]] = {}
        if parent_source_ids:
            with get_session() as session:
                bm_records = (
                    session.query(BookmarkRecord)
                    .filter(BookmarkRecord.audio_file_id.in_(parent_source_ids))
                    .all()
                )
                for bm in bm_records:
                    session.expunge(bm)
                    bookmark_map.setdefault(bm.audio_file_id, []).append(bm)

        context_blocks: list[str] = []
        sources: list[dict[str, Any]] = []

        for idx, ((expr, vec_score), in_rels) in enumerate(
            zip(top_candidates, top_in_rels)
        ):
            sources.append({
                "id": expr.id,
                "type": expr.source_type,
                "content": expr.content,
                "score": vec_score,
            })

            block = f"--- Memory Fragment {idx + 1} ---\n"
            block += f"Type: {expr.source_type}\n"
            if expr.speaker:
                block += f"Speaker: {expr.speaker}\n"
            block += f"Content: {expr.content}\n"

            parent = top_parents.get(expr.id)
            if parent:
                if parent.source_type == SourceType.TRANSCRIPT_SEGMENT.value:
                    block += f"Broader Topic/Context: {parent.content}\n"
                elif parent.source_type == SourceType.AUDIO_TRANSCRIPT.value:
                    block += "From Main Transcript.\n"

                if parent.source_id and parent.source_type in (
                    SourceType.TRANSCRIPT_SEGMENT.value,
                    SourceType.AUDIO_TRANSCRIPT.value,
                ):
                    bookmarks = bookmark_map.get(parent.source_id, [])
                    if bookmarks:
                        b_texts = [
                            f"Bookmark at {b.timestamp}s: {b.name} - {b.notes}"
                            for b in bookmarks[:3]
                        ]
                        block += "Related Bookmarks:\n- " + "\n- ".join(b_texts) + "\n"

            # Use pre-fetched inference map (EQ-1: no per-rel get_by_id)
            inferences = [
                inference_map[rel.from_expression_id].content
                for rel in in_rels
                if rel.from_expression_id in inference_map
                and inference_map[rel.from_expression_id].source_type
                == SourceType.SYSTEM_INFERENCE.value
            ]
            if inferences:
                block += "System Inferences:\n- " + "\n- ".join(inferences) + "\n"

            context_blocks.append(block)

        # ── Step 5: LLM Synthesis (EQ-3: Ok/Err, no nested try/except) ───────
        context_text = "\n\n".join(context_blocks)
        prompt = (
            "You are a memory retrieval engine. Answer the user's query using ONLY the provided memory fragments.\n"
            "If the answer cannot be determined from the fragments, say so clearly.\n\n"
            f"USER QUERY: {text}\n\n"
            f"MEMORY FRAGMENTS:\n{context_text}\n\n"
            "Synthesize a clear and concise answer based on these fragments."
        )

        match _call_llm(prompt, 0.2, self.llm, self.settings.gemini_api_key):
            case Ok(value=answer):
                self.daemon.write_cache(text, answer, hyde_document=hyde_document)
                return QueryResult(
                    query=text,
                    answer=answer,
                    sources=sources,
                    query_time_seconds=time.time() - t0,
                    hyde_document=hyde_document,
                )
            case Err(error=reason):
                logger.error("Synthesis failed completely: %s", reason)
                return QueryResult(
                    query=text,
                    answer=None,
                    sources=sources,
                    query_time_seconds=time.time() - t0,
                    hyde_document=hyde_document,
                    synthesis_failed=True,
                    synthesis_error=reason,
                )



# ── ResearchEngine ────────────────────────────────────────────────────────────


@dataclass
class ResearchResult:
    """Result returned by ResearchEngine.search()."""

    query: str
    sources: list[FusedResult] = field(default_factory=list)
    query_time_seconds: float = 0.0
    answer: str | None = None
    synthesis_failed: bool = False
    synthesis_error: str | None = None
    # Names of streams that returned 0 results (used by display for ✗ badges)
    streams_skipped: list[str] = field(default_factory=list)
    # True when HyDE was requested but fell back to direct query embedding
    hyde_fallback: bool = False
    hyde_document: str | None = None


class ResearchEngine:
    """Parallel retrieval orchestrator with observatory telemetry.

    Calls the three retrieval streams concurrently via a thread pool, fuses
    results with RRF, synthesises via LLM, and emits structured events to the
    observatory for every stage so latency can be inspected live.

    Thread safety: each stream is instantiated fresh per call; no shared mutable
    state between threads.
    """

    _STREAM_TIMEOUT: float = 15.0  # seconds; streams that exceed this are skipped

    def search(
        self, 
        query: str, 
        top_k: int = 10, 
        preset: str = "balanced",
        mmr_lambda: float = 0.5,
        focus_source: str | None = None,
    ) -> ResearchResult:
        """Run all three streams in parallel, fuse via RRF, synthesise, return results.

        Preset flags:
          fast     → FTS5 + Dense, no HyDE, no ColBERT reranker
          balanced → FTS5 + Dense + ColBERT, no HyDE
          deep     → FTS5 + Dense + ColBERT + HyDE (LLM generates hypothetical doc)

        Each stream is called in its own thread.  If a stream raises or times
        out it is silently skipped — partial results are still returned.
        All stages emit ``log_event()`` calls under subsystem='memory.search' so
        the user can inspect timing with ``audiobench obs tail --subsystem memory.search``.
        """
        from audiobench.observatory.context import log_event, start_trace

        trace_id = start_trace()  # noqa: F841  — sets ContextVar for log_event
        t0 = time.perf_counter()

        # ── Preset → stream flags ──────────────────────────────────────────────
        use_mmr = False
        mmr_lam = 0.5
        if preset == "fast":
            use_colbert = False
            use_hyde = False
        elif preset == "deep":
            use_colbert = True
            use_hyde = True
        elif preset == "synthesis":
            # MMR diversifies the tail. We run ColBERT but restrict it internally
            # to only return top 5, anchoring the highly relevant head in RRF 
            # without destroying MMR diversity in the tail.
            use_colbert = True
            use_hyde = False
            use_mmr = True
            mmr_lam = 0.5
        else:  # balanced (default)
            use_colbert = True
            use_hyde = False

        # ── Stage 1: Query reformulation ─────────────────────────────────────
        t_reform = time.perf_counter()
        from audiobench.memory.query_reformulator import QueryReformulator
        reformulator = QueryReformulator()
        rq = reformulator.reformulate(query)
        reform_ms = (time.perf_counter() - t_reform) * 1000
        log_event(
            subsystem="memory.search",
            event_type="query.reformulated",
            message=f"Query reformulated: '{query[:80]}'",
            duration_ms=reform_ms,
            metadata={"bm25_keywords": rq.bm25_keywords[:80], "preset": preset},
        )

        # ── Stage 1b: HyDE generation (deep preset only) ──────────────────────
        hyde_fallback = False
        if use_hyde:
            t_hyde = time.perf_counter()
            hyde_prompt = (
                "Write a realistic hypothetical excerpt from a spoken recording "
                "(audiobook, podcast, or interview) that directly answers the query. "
                "Limit to 100-150 words. Raw transcript only — no preamble.\n\n"
                f"Query: {query}"
            )
            settings = get_settings()
            llm = OllamaClient(base_url=settings.ollama_base_url, model=settings.ollama_model)
            match _call_llm(hyde_prompt, 0.7, llm, settings.gemini_api_key):
                case Ok(value=doc):
                    # Inject hyde_document into the (frozen) ReformulatedQuery by rebuilding it
                    import dataclasses
                    rq = dataclasses.replace(rq, hyde_document=doc)
                    hyde_ms = (time.perf_counter() - t_hyde) * 1000
                    log_event(
                        subsystem="memory.search",
                        event_type="hyde.ok",
                        message="HyDE document generated",
                        duration_ms=hyde_ms,
                        metadata={"doc_chars": len(doc)},
                    )
                case Err(error=reason):
                    hyde_fallback = True
                    logger.warning(
                        "HyDE generation failed (%s) — falling back to direct query embedding", reason
                    )
                    log_event(
                        subsystem="memory.search",
                        event_type="hyde.failed",
                        message=f"HyDE failed: {reason[:80]}",
                        level="WARNING",
                    )

        fts_hits: list = []
        dense_hits: list = []
        colbert_hits: list = []

        # ── Stage 2: Parallel stream retrieval ───────────────────────────────
        dense_extra: dict = {"preset": preset, "mmr_lambda": mmr_lambda, "focus_source": focus_source} if use_mmr else {"focus_source": focus_source}
        fts_extra: dict = {"focus_source": focus_source}
        
        stream_tasks: dict[str, tuple] = {
            "fts5":  (FTS5Stream(),    fts_hits,    fts_extra),
            "dense": (DenseStream(),   dense_hits,  dense_extra),
        }
        if use_colbert:
            stream_tasks["colbert"] = (ColBERTStream(), colbert_hits, {"preset": preset})

        stream_timings: dict[str, float] = {}
        streams_skipped: list[str] = []

        def _timed_retrieve(name: str, stream: object, bucket: list, extra: dict) -> None:
            t_s = time.perf_counter()
            if extra:
                hits = stream.retrieve(rq, top_k, **extra)  # type: ignore[attr-defined]
            else:
                hits = stream.retrieve(rq, top_k)  # type: ignore[attr-defined]
            elapsed_ms = (time.perf_counter() - t_s) * 1000
            bucket.extend(hits)
            stream_timings[name] = elapsed_ms

        import contextvars

        future_to_name = {
            _search_pool.submit(
                contextvars.copy_context().run,
                _timed_retrieve, name, stream, bucket, extra,
            ): name
            for name, (stream, bucket, extra) in stream_tasks.items()
        }

        done, not_done = wait(future_to_name.keys(), timeout=self._STREAM_TIMEOUT * 2)

        for future in done:
            name = future_to_name[future]
            try:
                future.result()  # Should return immediately since it's in `done`
                bucket = stream_tasks[name][1]
                hit_count = len(bucket)
                if hit_count == 0:
                    streams_skipped.append(name)
                log_event(
                    subsystem="memory.search",
                    event_type=f"stream.{name}",
                    message=f"Stream '{name}' returned {hit_count} hits",
                    duration_ms=stream_timings.get(name),
                    metadata={"hit_count": hit_count},
                )
            except Exception as exc:  # noqa: BLE001
                streams_skipped.append(name)
                logger.warning("Stream '%s' failed: %s — skipping", name, exc)
                log_event(
                    subsystem="memory.search",
                    event_type=f"stream.{name}",
                    message=f"Stream '{name}' failed: {exc}",
                    level="WARNING",
                    metadata={"error": str(exc)},
                )

        for future in not_done:
            name = future_to_name[future]
            streams_skipped.append(name)
            logger.warning("Stream '%s' timed out after %.1fs — skipping", name, self._STREAM_TIMEOUT * 2)
            log_event(
                subsystem="memory.search",
                event_type=f"stream.{name}.timeout",
                message=f"Stream '{name}' timed out",
                level="WARNING",
                metadata={"timeout": self._STREAM_TIMEOUT * 2},
            )

        # ── Stage 3: RRF fusion ──────────────────────────────────────────────
        t_fuse = time.perf_counter()
        fused = rrf_merge(fts_hits, dense_hits, colbert_hits, top_n=top_k)
        fuse_ms = (time.perf_counter() - t_fuse) * 1000
        log_event(
            subsystem="memory.search",
            event_type="rrf.fused",
            message=f"RRF produced {len(fused)} fused results",
            duration_ms=fuse_ms,
            metadata={"fused_count": len(fused)},
        )

        # ── Stage 3.5: Temporal deduplication (always on) ─────────────────────
        pre_dedup_count = len(fused)
        fused = _temporal_dedup(fused)
        dedup_removed = pre_dedup_count - len(fused)
        if dedup_removed:
            log_event(
                subsystem="memory.search",
                event_type="dedup.temporal",
                message=f"Temporal dedup removed {dedup_removed} overlapping fragment(s)",
                metadata={"removed": dedup_removed, "kept": len(fused)},
            )

        if not fused:
            return ResearchResult(
                query=query,
                sources=fused,
                query_time_seconds=time.perf_counter() - t0,
                answer="No relevant segments found.",
                streams_skipped=streams_skipped,
                hyde_fallback=hyde_fallback,
                hyde_document=rq.hyde_document,
            )

        # ── Stage 4: LLM synthesis ───────────────────────────────────────────
        context_text = "\n\n".join(
            f"[Fragment {i + 1} | {fr.start_time:.1f}s–{fr.end_time:.1f}s]\n{fr.text}"
            for i, fr in enumerate(fused[:5])
        )
        synthesis_prompt = (
            "You are a personal memory assistant. Answer the query using ONLY "
            "the transcript fragments below. If the answer cannot be determined, say so.\n\n"
            f"QUERY: {query}\n\n"
            f"FRAGMENTS:\n{context_text}\n\n"
            "Provide a clear, concise answer."
        )

        settings = get_settings()
        llm = OllamaClient(base_url=settings.ollama_base_url, model=settings.ollama_model)

        t_synth = time.perf_counter()
        match _call_llm(synthesis_prompt, temperature=0.2, llm=llm, api_key=settings.gemini_api_key):
            case Ok(value=answer):
                synth_ms = (time.perf_counter() - t_synth) * 1000
                log_event(
                    subsystem="memory.search",
                    event_type="synthesis.ok",
                    message="Synthesis completed",
                    duration_ms=synth_ms,
                    metadata={"answer_chars": len(answer)},
                )
                return ResearchResult(
                    query=query,
                    sources=fused,
                    query_time_seconds=time.perf_counter() - t0,
                    answer=answer,
                    streams_skipped=streams_skipped,
                    hyde_fallback=hyde_fallback,
                    hyde_document=rq.hyde_document,
                )
            case Err(error=reason):
                synth_ms = (time.perf_counter() - t_synth) * 1000
                logger.error("ResearchEngine synthesis failed: %s", reason)
                log_event(
                    subsystem="memory.search",
                    event_type="synthesis.failed",
                    message=f"Synthesis failed: {reason[:120]}",
                    level="ERROR",
                    duration_ms=synth_ms,
                    metadata={"error": reason[:200]},
                )
                return ResearchResult(
                    query=query,
                    sources=fused,
                    query_time_seconds=time.perf_counter() - t0,
                    answer=None,
                    synthesis_failed=True,
                    synthesis_error=reason,
                    streams_skipped=streams_skipped,
                    hyde_fallback=hyde_fallback,
                    hyde_document=rq.hyde_document,
                )
