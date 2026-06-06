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
    ) -> None:
        """Embed and write an expression to LanceDB.

        If the expression_id already exists, it will be updated (deleted and recreated).
        """
        # Delete if it exists to allow updates
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

        # Optimize fts index occasionally, but for simplicity here we just ensure
        # it gets picked up. LanceDB auto-updates FTS indices in newer versions,
        # but we might need to recreate it if it gets out of sync.

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
        # Delete if it exists to allow updates
        try:
            self.table.delete(f"profile_id = '{profile_id}'")
        except Exception:
            pass

        now_str = datetime.datetime.now(datetime.UTC).isoformat()
        record = SpeakerProfileNode(
            profile_id=profile_id,
            name=name,
            vector=voice_print,
            created_at=now_str,
        )
        self.table.add([record])
        logger.info("Saved speaker profile for '%s' (ID: %s)", name, profile_id)

