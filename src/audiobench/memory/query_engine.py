"""Semantic memory query engine.

Orchestrates vector search, cross-encoder reranking, SQLite graph
traversal, and LLM synthesis to answer queries using the expression graph.
"""

from dataclasses import dataclass
from typing import Any

from audiobench.chat.chat_store import ChatRepository
from audiobench.chat.providers.ollama_provider import OllamaClient
from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.daemon.factory import get_daemon_client
from audiobench.memory.enums import SourceType
from audiobench.storage.expression_repository import ExpressionRepository
from audiobench.storage.models import BookmarkRecord

logger = get_logger("memory.query")


@dataclass
class QueryResult:
    query: str
    answer: str
    sources: list[dict[str, Any]]
    query_time_seconds: float = 0.0
    hyde_document: str | None = None


class MemoryQueryEngine:
    def __init__(self):
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
        top_k: int = 20,  # Increased to ensure enough Tier 3 candidates to fill 5 unique parents
        speaker_filter: str | None = None,
        preset: str = "balanced",
        enable_hyde: bool | None = None,
        enable_cross_encoder: bool | None = None,
        use_bm25: bool | None = None,
        use_dense: bool | None = None,
        use_colbert: bool | None = None,
        use_cache: bool = True,
    ) -> QueryResult:
        import time

        t0 = time.time()
        logger.info("Starting memory query for: %s (Preset: %s)", text, preset)

        # Resolve preset logic
        if preset == "fast":
            p_hyde, p_cross, p_bm25, p_dense, p_colbert = False, False, True, True, False
        elif preset == "deep":
            p_hyde, p_cross, p_bm25, p_dense, p_colbert = True, True, True, True, True
        else:  # balanced
            p_hyde, p_cross, p_bm25, p_dense, p_colbert = False, False, True, True, True

        final_hyde = enable_hyde if enable_hyde is not None else p_hyde
        final_cross = enable_cross_encoder if enable_cross_encoder is not None else p_cross
        final_bm25 = use_bm25 if use_bm25 is not None else p_bm25
        final_dense = use_dense if use_dense is not None else p_dense
        final_colbert = use_colbert if use_colbert is not None else p_colbert

        # Step 0: Semantic Cache Check
        if use_cache:
            cached = self.daemon.check_cache(text)
            if cached:
                query_time = time.time() - t0
                return QueryResult(
                    query=text,
                    answer=cached["answer"],
                    sources=[],
                    query_time_seconds=query_time,
                    hyde_document=cached.get("hyde_document"),
                )

        # Step 1: HyDE Generation
        hyde_document = None
        if final_hyde:
            hyde_prompt = (
                f"Please write a realistic hypothetical excerpt from a spoken recording, such as an audiobook, "
                f"podcast, interview, or personal reflection, that directly answers or provides context for the "
                f"following query. Limit your response to approximately 100 to 150 words. Do not exceed 150 words. "
                f"Do not include any preambles, greetings, facts you are unsure about, or meta-commentary. Just "
                f"produce the raw hypothetical text as if it were a direct transcript.\n\nQuery: {text}"
            )
            try:
                logger.info("Generating HyDE document via Ollama...")
                hyde_document = self.llm.generate(hyde_prompt, temperature=0.7)
                logger.debug("HyDE Document: %s", hyde_document)
            except Exception as e:
                logger.warning("HyDE generation failed: %s", e)

        # Step 2: Vector search via Daemon
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
        except Exception as e:
            logger.error("Daemon search failed: %s", e)
            results = []

        if not results:
            return QueryResult(query=text, answer="No relevant memory found.", sources=[])

        # SQLite fetch candidate expressions
        candidates = []
        for r in results:
            expr_id = r.get("expression_id")
            if expr_id is not None:
                expr = self.expr_repo.get_by_id(expr_id)
                if expr:
                    candidates.append((expr, r.get("score", 0.0)))

        if not candidates:
            return QueryResult(
                query=text, answer="No relevant memory found in database.", sources=[]
            )

        # Step 2.5: Parent-Child Expansion + Deduplication
        # Walk each Tier 3 sentence hit up to its Tier 2 parent paragraph in SQLite.
        # Collapse multiple Tier 3 siblings that share the same Tier 2 parent.
        # This ensures we rerank contextually complete paragraphs, not fragments.
        expanded: dict[int, tuple[object, float]] = {}  # parent_id -> (parent_expr, best_score)
        for expr, score in candidates:
            parent = self.expr_repo.walk_to_parent(expr.id)
            if parent is not None:
                # Use the Tier 2 parent as the retrieval unit
                if parent.id not in expanded or score > expanded[parent.id][1]:
                    expanded[parent.id] = (parent, score)
            else:
                # expr has no parent (e.g. non-transcript, or already a top-level node)
                if expr.id not in expanded:
                    expanded[expr.id] = (expr, score)

        # Sort collapsed parents by best child score descending
        expanded_candidates = sorted(expanded.values(), key=lambda x: x[1], reverse=True)

        # Step 3: CrossEncoder Reranking over the deduplicated parent paragraphs
        if final_cross:
            try:
                docs = [c[0].content for c in expanded_candidates]
                scores = self.daemon.rerank(text, docs)

                # Sort by reranker score descending
                scored_candidates = sorted(
                    zip(expanded_candidates, scores), key=lambda x: x[1], reverse=True
                )
                top_candidates = [(c[0], float(score)) for c, score in scored_candidates[:5]]
            except Exception as e:
                logger.warning("Reranking via daemon failed: %s. Using vector scores.", e)
                top_candidates = list(expanded_candidates[:5])
        else:
            top_candidates = list(expanded_candidates[:5])

        # Step 4: Graph Traversal for Enriched Context
        context_blocks = []
        sources = []

        for idx, (expr, vec_score) in enumerate(top_candidates):
            # Enriched payload for this expression
            source_info = {
                "id": expr.id,
                "type": expr.source_type,
                "content": expr.content,
                "score": vec_score,
            }
            sources.append(source_info)

            block = f"--- Memory Fragment {idx + 1} ---\n"
            block += f"Type: {expr.source_type}\n"
            if expr.speaker:
                block += f"Speaker: {expr.speaker}\n"
            block += f"Content: {expr.content}\n"

            # Walk up to find parent and transcript if applicable
            parent = self.expr_repo.walk_to_parent(expr.id)
            if parent:
                if parent.source_type == SourceType.TRANSCRIPT_SEGMENT.value:
                    block += f"Broader Topic/Context: {parent.content}\n"
                elif parent.source_type == SourceType.AUDIO_TRANSCRIPT.value:
                    block += "From Main Transcript.\n"

                # Fetch bookmarks or AskEntries related to parent?
                # We can do this via source_id if it's an audio file
                if parent.source_id and parent.source_type in [
                    SourceType.TRANSCRIPT_SEGMENT.value,
                    SourceType.AUDIO_TRANSCRIPT.value,
                ]:
                    with get_session() as session:
                        # fetch bookmarks
                        bookmarks = (
                            session.query(BookmarkRecord)
                            .filter_by(audio_file_id=parent.source_id)
                            .all()
                        )
                        if bookmarks:
                            b_texts = [
                                f"Bookmark at {b.timestamp}s: {b.name} - {b.notes}"
                                for b in bookmarks[:3]
                            ]
                            block += "Related Bookmarks:\n- " + "\n- ".join(b_texts) + "\n"

            # Fetch inferences linked to this expr
            in_rels = self.expr_repo.get_relations(expr.id, direction="in")
            inferences = []
            for rel in in_rels:
                src_expr = self.expr_repo.get_by_id(rel.from_expression_id)
                if src_expr and src_expr.source_type == SourceType.SYSTEM_INFERENCE.value:
                    inferences.append(src_expr.content)

            if inferences:
                block += "System Inferences:\n- " + "\n- ".join(inferences) + "\n"

            context_blocks.append(block)

        # Step 5: LLM Synthesis
        context_text = "\n\n".join(context_blocks)
        prompt = (
            f"You are a memory retrieval engine. Answer the user's query using ONLY the provided memory fragments.\n"
            f"If the answer cannot be determined from the fragments, say so clearly.\n\n"
            f"USER QUERY: {text}\n\n"
            f"MEMORY FRAGMENTS:\n{context_text}\n\n"
            f"Synthesize a clear and concise answer based on these fragments."
        )

        try:
            answer = self.llm.generate(prompt, temperature=0.2)
            # Write to Cache on successful synthesis
            self.daemon.write_cache(text, answer, hyde_document=hyde_document)
        except Exception as e:
            logger.error("Synthesis failed: %s", e)
            answer = "Failed to synthesize answer."

        query_time = time.time() - t0
        return QueryResult(
            query=text,
            answer=answer,
            sources=sources,
            query_time_seconds=query_time,
            hyde_document=hyde_document,
        )
