"""MemoryStore — LanceDB vector storage for Semantic Expressions.

Provides direct access to LanceDB for writing, deleting, and hybrid searching
of semantic memory nodes.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import lancedb
from lancedb.pydantic import LanceModel, Vector

from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.daemon.protocol import SearchResult
from audiobench.memory.embedding_engine import EmbeddingEngine

logger = get_logger("memory.memory_store")


class ExpressionNode(LanceModel):
    """LanceDB schema for stored semantic expressions."""

    expression_id: int
    vector: Vector(768)  # type: ignore[valid-type]
    content: str
    embedding_model_version: str
    embedded_at: str
    source_type: str
    speaker: str | None = None


class QueryCacheNode(LanceModel):
    """LanceDB schema for cached semantic queries."""

    query: str
    vector: Vector(768)  # type: ignore[valid-type]
    answer: str
    hyde_document: str | None = None
    created_at: str


class SpeakerProfileNode(LanceModel):
    """LanceDB schema for persistent speaker voice prints (ECAPA-TDNN)."""

    profile_id: str
    name: str
    vector: Vector(192)  # type: ignore[valid-type]
    created_at: str


class MemoryStore:
    """LanceDB adapter for audiobench expressions."""

    def __init__(self) -> None:
        """Initialize the connection to the LanceDB instance."""
        settings = get_settings()
        lancedb_dir = Path(settings.data_dir) / "lancedb"
        lancedb_dir.mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(str(lancedb_dir))
        self.table_name = "expressions"
        self._engine = EmbeddingEngine()

        # We assume the primary embedder model version is standard
        self.model_version = "nomic-embed-text-v1.5"

        if self.table_name not in self.db.table_names():
            logger.info("Creating LanceDB table '%s'", self.table_name)
            self.table = self.db.create_table(self.table_name, schema=ExpressionNode)
            # Create full-text search index
            self.table.create_fts_index("content")
        else:
            self.table = self.db.open_table(self.table_name)

    def write_node(
        self,
        expression_id: int,
        content: str,
        source_type: str,
        speaker: str | None = None,
        indexed_ids: set[int] | None = None,
    ) -> None:
        """Embed and write a single expression to LanceDB.

        If *indexed_ids* is provided (the in-memory O(1) set from SweepState),
        the speculative delete is skipped for brand-new expressions, saving one
        LanceDB round-trip. The set is updated in-place on success.
        """
        already_indexed = indexed_ids is not None and expression_id in indexed_ids
        if indexed_ids is None or already_indexed:
            self.delete_node(expression_id)

        vector = self._engine.embed_for_storage(content).tolist()
        now_str = datetime.datetime.now(datetime.UTC).isoformat()

        record = ExpressionNode(
            expression_id=expression_id,
            vector=vector,
            content=content,
            embedding_model_version=self.model_version,
            embedded_at=now_str,
            source_type=source_type,
            speaker=speaker,
        )
        self.table.add([record])

        if indexed_ids is not None:
            indexed_ids.add(expression_id)

    def batch_write_nodes(
        self,
        nodes: list[dict],
        indexed_ids: set[int] | None = None,
        batch_size: int = 64,
    ) -> None:
        """Embed and write a batch of expressions in one model forward-pass.

        Args:
            nodes:       List of dicts with keys: expression_id, content,
                         source_type, speaker (optional).
            indexed_ids: The in-memory set from SweepState.  Entries already in
                         the set are deleted before re-insertion; genuinely new
                         entries skip the delete entirely.  Updated in-place.
            batch_size:  sentence-transformers sub-batch size (64 is safe).
        """
        if not nodes:
            return

        to_delete: list[int] = []
        for n in nodes:
            eid = int(n["expression_id"])
            if indexed_ids is None or eid in indexed_ids:
                to_delete.append(eid)

        if to_delete:
            id_list = ", ".join(str(i) for i in to_delete)
            try:
                self.table.delete(f"expression_id IN ({id_list})")
            except Exception as exc:
                logger.debug("batch_write_nodes: delete failed: %s", exc)

        texts = [n["content"] for n in nodes]
        vectors = self._engine.embed_batch_for_storage(texts, batch_size=batch_size)

        now_str = datetime.datetime.now(datetime.UTC).isoformat()
        records = [
            ExpressionNode(
                expression_id=int(n["expression_id"]),
                vector=v,
                content=n["content"],
                embedding_model_version=self.model_version,
                embedded_at=now_str,
                source_type=n["source_type"],
                speaker=n.get("speaker"),
            )
            for n, v in zip(nodes, vectors)
        ]
        self.table.add(records)

        if indexed_ids is not None:
            for n in nodes:
                indexed_ids.add(int(n["expression_id"]))

        logger.info("batch_write_nodes: wrote %d expressions to LanceDB", len(records))

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
        """Perform a dynamic search over expressions based on enabled strategies."""
        if not use_bm25 and not use_dense:
            logger.warning("Both BM25 and Dense search disabled. Defaulting to Dense.")
            use_dense = True

        query_vector = self._engine.embed_for_query(query).tolist()

        if hyde_document and use_dense:
            hyde_vector = self._engine.embed_for_query(hyde_document).tolist()
            import numpy as np

            query_vector = ((np.array(query_vector) + np.array(hyde_vector)) / 2.0).tolist()

        # Determine base search strategy
        if use_bm25 and use_dense:
            # LanceDB native RRF handles the hybrid merge
            search_query = (
                self.table.search(query_type="hybrid", fts_columns="content")
                .vector(query_vector)
                .text(query)
            )
        elif use_bm25:
            search_query = self.table.search(query, query_type="fts")
        else:
            search_query = self.table.search(query_vector, query_type="vector")

        search_query = search_query.limit(top_k * 3)

        if speaker_filter:
            search_query = search_query.where(f"speaker = '{speaker_filter}'")

        # Apply ColBERT reranking if enabled
        if use_colbert:
            from audiobench.memory.singletons import get_colbert_reranker

            reranker = get_colbert_reranker()
            search_query = search_query.rerank(reranker=reranker)
        elif use_bm25 and use_dense:
            # Explicitly use RRF if ColBERT is off but Hybrid is on
            from lancedb.rerankers import RRFReranker

            search_query = search_query.rerank(reranker=RRFReranker())

        results = search_query.to_list()

        final_results = []
        for r in results:
            # In LanceDB, the score key differs based on the reranker or query type.
            # _distance (vector), score (fts), or _relevance_score (rerankers)
            score = r.get("_relevance_score", r.get("score", r.get("_distance", 0.0)))

            final_results.append(
                {
                    "expression_id": r["expression_id"],
                    "content": r["content"],
                    "source_type": r["source_type"],
                    "speaker": r.get("speaker"),
                    "score": float(score),
                }
            )

            if len(final_results) >= top_k:
                break

        return final_results

    def delete_node(self, expression_id: int) -> None:
        """Delete a node by expression_id."""
        # LanceDB delete
        self.table.delete(f"expression_id = {expression_id}")

    def count_nodes(self) -> int:
        """Get the total number of nodes in the store."""
        return self.table.count_rows()


class QueryCacheStore:
    """LanceDB adapter for caching semantic queries."""

    def __init__(self) -> None:
        settings = get_settings()
        lancedb_dir = Path(settings.data_dir) / "lancedb"
        lancedb_dir.mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(str(lancedb_dir))
        self.table_name = "query_cache"
        self._engine = EmbeddingEngine()

        if self.table_name not in self.db.table_names():
            logger.info("Creating LanceDB cache table '%s'", self.table_name)
            self.table = self.db.create_table(self.table_name, schema=QueryCacheNode)
        else:
            self.table = self.db.open_table(self.table_name)

    def check_cache(self, query: str, distance_threshold: float = 0.05) -> dict | None:
        """Check if a semantically identical query exists in the cache."""
        query_vector = self._engine.embed_for_query(query).tolist()

        # We only need the top 1 result
        results = self.table.search(query_vector).limit(1).to_list()

        if results:
            best_match = results[0]
            distance = float(best_match.get("_distance", 1.0))
            if distance <= distance_threshold:
                logger.info("Cache hit for query '%s' (distance: %.4f)", query, distance)
                return {
                    "answer": best_match["answer"],
                    "hyde_document": best_match.get("hyde_document"),
                    "distance": distance,
                }

        return None

    def write_cache(self, query: str, answer: str, hyde_document: str | None = None) -> None:
        """Write a synthesized answer to the cache."""
        query_vector = self._engine.embed_for_query(query).tolist()
        now_str = datetime.datetime.now(datetime.UTC).isoformat()

        record = QueryCacheNode(
            query=query,
            vector=query_vector,
            answer=answer,
            hyde_document=hyde_document,
            created_at=now_str,
        )
        self.table.add([record])
        logger.info("Cached answer for query '%s'", query)


class SpeakerProfileStore:
    """LanceDB adapter for persistent speaker voice prints (ECAPA-TDNN)."""

    def __init__(self) -> None:
        settings = get_settings()
        lancedb_dir = Path(settings.data_dir) / "lancedb"
        lancedb_dir.mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(str(lancedb_dir))
        self.table_name = "speaker_profiles"

        if self.table_name not in self.db.table_names():
            logger.info("Creating LanceDB speaker profiles table '%s'", self.table_name)
            self.table = self.db.create_table(self.table_name, schema=SpeakerProfileNode)
        else:
            self.table = self.db.open_table(self.table_name)

    def identify_speaker(self, voice_print: list[float], threshold: float = 0.82) -> str | None:
        """Find the closest known speaker for a given voice print.
        
        Args:
            voice_print: 192-D list of floats from SpeechBrain.
            threshold: Minimum cosine similarity required to confirm match.
                       (LanceDB uses distance, so distance <= 1 - threshold)
        """
        if self.table.count_rows() == 0:
            return None
            
        results = self.table.search(voice_print).limit(1).to_list()
        
        if results:
            best_match = results[0]
            # LanceDB distance is typically 1 - cosine_similarity for vectors
            # So a cosine similarity of 0.82 means distance of 0.18
            distance = float(best_match.get("_distance", 1.0))
            max_distance = 1.0 - threshold
            
            if distance <= max_distance:
                logger.info(
                    "Voice matched! '%s' (dist: %.4f < %.4f)",
                    best_match["name"], distance, max_distance
                )
                return best_match["name"]
            else:
                logger.debug(
                    "Voice NOT matched. Closest was '%s' (dist: %.4f > %.4f)",
                    best_match["name"], distance, max_distance
                )
                
        return None

    def save_speaker(self, profile_id: str, name: str, voice_print: list[float]) -> None:
        """Save or update a speaker's voice print in the database."""
        # Delete if it exists to allow updates.
        # LanceDB raises when the filter matches zero rows on some versions — log, never swallow.
        try:
            self.table.delete(f"profile_id = '{profile_id}'")
        except Exception as exc:
            logger.debug("LanceDB delete failed during upsert (expected if new profile): %s", exc)

        now_str = datetime.datetime.now(datetime.UTC).isoformat()
        record = SpeakerProfileNode(
            profile_id=profile_id,
            name=name,
            vector=voice_print,
            created_at=now_str,
        )
        self.table.add([record])
        logger.info("Saved speaker profile for '%s' (ID: %s)", name, profile_id)


# ── Segment Vector Store ──────────────────────────────────────────────────────


class SegmentVectorNode(LanceModel):
    """LanceDB schema for audio segment embeddings.

    One record per segment row in SQLite. Carries timestamps and source file
    path so the display layer can show provenance without additional DB lookups.
    """

    segment_id: int           # FK → segments.id in SQLite
    vector: Vector(768)       # type: ignore[valid-type]  # Nomic nomic-embed-text-v1.5
    text: str                 # raw transcript text of the segment
    start_time: float         # seconds from audio start
    end_time: float           # seconds from audio end
    source_file: str          # absolute path to the audio file
    embedded_at: str          # ISO-8601 timestamp of when this was embedded


class SegmentVectorStore:
    """LanceDB adapter for audio segment vector embeddings.

    Mirrors the structure of MemoryStore but targets the 'segment_vectors'
    table and is keyed by segment_id (not expression_id).

    The daemon's _rag_consistency_sweep_sync populates this table
    automatically for every new segment. The CLI backfill command
    'audiobench db embed-segments' handles segments that existed before
    this feature was introduced.
    """

    table_name: str = "segment_vectors"

    def __init__(self) -> None:
        settings = get_settings()
        lancedb_dir = Path(settings.data_dir) / "lancedb"
        lancedb_dir.mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(str(lancedb_dir))
        self._engine = EmbeddingEngine()
        self.model_version = "nomic-embed-text-v1.5"

        if self.table_name not in self.db.table_names():
            logger.info("Creating LanceDB segment vectors table '%s'", self.table_name)
            self.table = self.db.create_table(self.table_name, schema=SegmentVectorNode)
        else:
            self.table = self.db.open_table(self.table_name)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_segment(
        self,
        segment_id: int,
        text: str,
        start_time: float,
        end_time: float,
        source_file: str,
    ) -> None:
        """Embed text and write (or overwrite) a segment into LanceDB.

        Idempotent: safe to call multiple times for the same segment_id.
        Deletes the old record first to allow content updates.
        """
        # Delete stale record if it exists
        try:
            self.table.delete(f"segment_id = {segment_id}")
        except Exception as exc:
            logger.debug("Segment delete (upsert) for id=%d: %s", segment_id, exc)

        vector = self._engine.embed_for_storage(text).tolist()
        now_str = datetime.datetime.now(datetime.UTC).isoformat()

        record = SegmentVectorNode(
            segment_id=segment_id,
            vector=vector,
            text=text,
            start_time=start_time,
            end_time=end_time,
            source_file=source_file,
            embedded_at=now_str,
        )
        self.table.add([record])
        logger.debug("Upserted segment_id=%d into segment_vectors", segment_id)

    def upsert_segment_with_vector(
        self,
        segment_id: int,
        text: str,
        start_time: float,
        end_time: float,
        source_file: str,
        vector: list[float],
    ) -> None:
        """Write a segment using a pre-computed vector (from the daemon's warm model).

        Used by the daemon handlers so the warm Nomic model in the daemon
        process is reused rather than cold-loading it in the CLI process.
        """
        try:
            self.table.delete(f"segment_id = {segment_id}")
        except Exception as exc:
            logger.debug("Segment delete (upsert_with_vector) for id=%d: %s", segment_id, exc)

        now_str = datetime.datetime.now(datetime.UTC).isoformat()
        record = SegmentVectorNode(
            segment_id=segment_id,
            vector=vector,
            text=text,
            start_time=start_time,
            end_time=end_time,
            source_file=source_file,
            embedded_at=now_str,
        )
        self.table.add([record])
        logger.debug("Upserted segment_id=%d (pre-computed vector) into segment_vectors", segment_id)

    def batch_upsert_segments(
        self,
        rows: list[dict],
        vectors: list[list[float]],
    ) -> None:
        """Write a batch of pre-computed segment vectors to LanceDB in one operation."""
        if not rows:
            return
        ids = [r["segment_id"] for r in rows]
        id_list = ", ".join(str(i) for i in ids)
        try:
            self.table.delete(f"segment_id IN ({id_list})")
        except Exception as exc:
            logger.debug("Batch segment delete: %s", exc)

        now_str = datetime.datetime.now(datetime.UTC).isoformat()
        records = [
            SegmentVectorNode(
                segment_id=int(r["segment_id"]),
                vector=v,
                text=r["text"],
                start_time=float(r["start_time"]),
                end_time=float(r["end_time"]),
                source_file=r["source_file"] or "",
                embedded_at=now_str,
            )
            for r, v in zip(rows, vectors)
        ]
        self.table.add(records)
        logger.info("Batch upserted %d segments into segment_vectors", len(records))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        return_vectors: bool = False,
    ) -> list[dict]:
        """ANN search over segment embeddings.

        Returns a list of plain dicts with keys:
            segment_id, text, start_time, end_time, source_file, _distance
        When ``return_vectors=True``, each dict also contains a ``vector``
        key (list[float]) for downstream MMR cosine computation.
        """
        results = (
            self.table.search(query_vector, query_type="vector")
            .limit(top_k)
            .to_list()
        )
        out: list[dict] = []
        for r in results:
            row = {k: v for k, v in r.items() if k != "vector"}
            if return_vectors:
                raw_vec = r.get("vector")
                if raw_vec is not None:
                    row["vector"] = raw_vec.tolist() if hasattr(raw_vec, "tolist") else list(raw_vec)
            out.append(row)
        return out


    def count_embedded(self) -> int:
        """Total number of segments currently embedded."""
        return self.table.count_rows()

    def get_embedded_ids(self) -> set[int]:
        """Return the set of segment_ids already in this table.

        Used by the daemon sweep and backfill command to find which
        segments still need embedding without redundant work.
        """
        rows = self.table.search().select(["segment_id"]).limit(100_000).to_list()
        return {int(r["segment_id"]) for r in rows}


    def identify_speaker(self, voice_print: list[float], threshold: float = 0.82) -> str | None:
        """Find the closest known speaker for a given voice print.
        
        Args:
            voice_print: 192-D list of floats from SpeechBrain.
            threshold: Minimum cosine similarity required to confirm match.
                       (LanceDB uses distance, so distance <= 1 - threshold)
        """
        if self.table.count_rows() == 0:
            return None
            
        results = self.table.search(voice_print).limit(1).to_list()
        
        if results:
            best_match = results[0]
            # LanceDB distance is typically 1 - cosine_similarity for vectors
            # So a cosine similarity of 0.82 means distance of 0.18
            distance = float(best_match.get("_distance", 1.0))
            max_distance = 1.0 - threshold
            
            if distance <= max_distance:
                logger.info(
                    "Voice matched! '%s' (dist: %.4f < %.4f)",
                    best_match["name"], distance, max_distance
                )
                return best_match["name"]
            else:
                logger.debug(
                    "Voice NOT matched. Closest was '%s' (dist: %.4f > %.4f)",
                    best_match["name"], distance, max_distance
                )
                
        return None

    def save_speaker(self, profile_id: str, name: str, voice_print: list[float]) -> None:
        """Save or update a speaker's voice print in the database."""
        # Delete if it exists to allow updates.
        # LanceDB raises when the filter matches zero rows on some versions — log, never swallow.
        try:
            self.table.delete(f"profile_id = '{profile_id}'")
        except Exception as exc:
            logger.debug("LanceDB delete failed during upsert (expected if new profile): %s", exc)

        now_str = datetime.datetime.now(datetime.UTC).isoformat()
        record = SpeakerProfileNode(
            profile_id=profile_id,
            name=name,
            vector=voice_print,
            created_at=now_str,
        )
        self.table.add([record])
        logger.info("Saved speaker profile for '%s' (ID: %s)", name, profile_id)

