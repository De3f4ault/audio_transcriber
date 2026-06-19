"""Retrieval streams for AudioBench memory."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sqlalchemy import text

from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.memory.query_types import ReformulatedQuery

logger = get_logger("memory.retrieval_streams")


@dataclass
class SegmentHit:
    """A segment retrieved from a stream."""

    segment_id: int
    start_time: float
    end_time: float
    text: str
    bm25_score: float = 0.0
    dense_score: float = 0.0
    colbert_score: float = 0.0
    source_file: str = ""


@runtime_checkable
class RetrievalStream(Protocol):
    """Protocol for all retrieval streams."""

    def retrieve(self, query: ReformulatedQuery, top_k: int = 5) -> list[SegmentHit]:
        """Retrieve segments for a query."""
        ...


class FTS5Stream:
    """FTS5-based keyword retrieval stream."""

    def retrieve(self, query: ReformulatedQuery, top_k: int = 5) -> list[SegmentHit]:
        hits = []
        try:
            if not query.bm25_keywords.strip():
                return []
                
            # Create FTS MATCH query (e.g., OR between keywords)
            keywords = query.bm25_keywords.split()
            match_query = " OR ".join(f'"{k}"' for k in keywords)

            with get_session() as session:
                sql = text("""
                    SELECT
                        s.id as segment_id,
                        s.start_time,
                        s.end_time,
                        fts.rank as bm25_score,
                        s.text,
                        af.file_path as source_file
                    FROM segments_fts fts
                    JOIN segments s ON fts.rowid = s.id
                    JOIN transcriptions t ON s.transcription_id = t.id
                    JOIN audio_files af ON t.audio_file_id = af.id
                    WHERE segments_fts MATCH :match_query
                    ORDER BY bm25_score ASC
                    LIMIT :limit
                """)

                rows = session.execute(sql, {"match_query": match_query, "limit": top_k}).mappings().all()
                for r in rows:
                    hits.append(SegmentHit(
                        segment_id=r["segment_id"],
                        start_time=r["start_time"],
                        end_time=r["end_time"],
                        bm25_score=r["bm25_score"],
                        text=r["text"],
                        source_file=r["source_file"] or "",
                    ))
        except Exception as e:
            logger.warning("FTS5Stream retrieval failed: %s", e)
        return hits


class DenseStream:
    """Dense embedding retrieval stream.

    Embeds the query via the daemon's warm Nomic model (no cold load in CLI
    process), then performs ANN search against the segment_vectors LanceDB
    table. Uses hyde_document as the query vector when preset=deep and HyDE
    succeeded; otherwise uses the original query (rq.dense_query).
    """

    def retrieve(self, query: ReformulatedQuery, top_k: int = 5) -> list[SegmentHit]:
        try:
            from audiobench.daemon.factory import get_daemon_client
            from audiobench.memory.memory_store import SegmentVectorStore

            daemon = get_daemon_client()

            # Use HyDE document if available (preset=deep), otherwise original query.
            # dense_query is always the original — never the BM25-reformulated version.
            embed_text = query.hyde_document if query.hyde_document else query.dense_query
            if not embed_text:
                embed_text = query.original

            query_vector = daemon.embed_query(embed_text)
            store = SegmentVectorStore()

            if store.count_embedded() == 0:
                logger.debug("DenseStream: segment_vectors table is empty — skipping")
                return []

            rows = store.search(query_vector, top_k)
            hits = []
            for r in rows:
                hits.append(
                    SegmentHit(
                        segment_id=int(r["segment_id"]),
                        start_time=float(r["start_time"]),
                        end_time=float(r["end_time"]),
                        text=r["text"],
                        dense_score=float(r.get("_distance", 0.0)),
                        source_file=r.get("source_file", ""),
                    )
                )
            return hits

        except Exception as e:
            logger.warning("DenseStream retrieval failed: %s", e)
        return []


class ColBERTStream:
    """Late-interaction ColBERT retrieval stream.

    Strategy:
      1. Fetch top_k * 3 candidates from segment_vectors via ANN (same as DenseStream).
      2. Send all candidate texts + query to the daemon's CrossEncoder reranker.
      3. Return the top_k candidates sorted by rerank score.

    A 5-second timeout is applied to the reranker call. If it times out or
    fails, an empty list is returned so the caller can show a colbert✗ badge
    and RRF continues with the FTS5 + Dense results.

    Uses hyde_document as query vector when available, otherwise dense_query.
    """

    _RERANK_TIMEOUT: float = 5.0

    def retrieve(self, query: ReformulatedQuery, top_k: int = 5) -> list[SegmentHit]:
        try:
            from audiobench.daemon.factory import get_daemon_client
            from audiobench.memory.memory_store import SegmentVectorStore

            daemon = get_daemon_client()

            embed_text = query.hyde_document if query.hyde_document else query.dense_query
            if not embed_text:
                embed_text = query.original

            query_vector = daemon.embed_query(embed_text)
            store = SegmentVectorStore()

            if store.count_embedded() == 0:
                logger.debug("ColBERTStream: segment_vectors table is empty — skipping")
                return []

            # Retrieve 3× candidates for reranking headroom
            candidates = store.search(query_vector, top_k * 3)
            if not candidates:
                return []

            # CrossEncoder rerank via daemon (warm model, 5s timeout)
            import concurrent.futures
            texts = [r["text"] for r in candidates]
            rerank_query = query.dense_query if query.dense_query else query.original

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(daemon.rerank, rerank_query, texts)
                try:
                    scores = future.result(timeout=self._RERANK_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    logger.warning(
                        "ColBERTStream: reranker timed out after %.1fs — skipping", self._RERANK_TIMEOUT
                    )
                    return []

            # Sort by descending rerank score, return top_k
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            hits = []
            for r, score in ranked[:top_k]:
                hits.append(
                    SegmentHit(
                        segment_id=int(r["segment_id"]),
                        start_time=float(r["start_time"]),
                        end_time=float(r["end_time"]),
                        text=r["text"],
                        colbert_score=float(score),
                        source_file=r.get("source_file", ""),
                    )
                )
            return hits

        except Exception as e:
            logger.warning("ColBERTStream retrieval failed: %s", e)
        return []
