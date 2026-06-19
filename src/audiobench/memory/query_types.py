"""Dataclasses for reformulated queries — kept in their own module to avoid circular imports.

query_reformulator.py imports from query_engine.py (for _call_llm).
retrieval_streams.py needs ReformulatedQuery as a type annotation.
Placing this dataclass here lets both modules import it without forming a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReformulatedQuery:
    """A user query expanded into multiple targeted retrieval representations.

    Each field feeds a different retrieval channel:
      - bm25_keywords  → FTS5Stream  (keyword search, reformulated)
      - dense_query    → DenseStream  (original query, no reformulation — Nomic handles context)
      - hyde_document  → DenseStream/ColBERTStream when preset=deep (hypothetical answer doc)
    """

    original: str
    bm25_keywords: str
    semantic_query: str
    hyde_anchor: str
    # Independent channel for dense/colbert — always the raw original query,
    # never reformulated. Populated by QueryReformulator.
    dense_query: str = ""
    # Set by ResearchEngine.search() when preset=deep and HyDE succeeds.
    # None means HyDE was not requested or failed gracefully.
    hyde_document: str | None = None
