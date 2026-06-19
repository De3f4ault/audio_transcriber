"""Model singletons for memory embedding and reranking.

Handles thread-safe lazy-loading of ML models for semantic boundary detection,
primary embedding generation, and post-retrieval reranking.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError
from requests.exceptions import ConnectionError  # type: ignore[import-untyped]

from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.exceptions import AudioBenchError

logger = get_logger("memory.singletons")

# Singleton references
_boundary_embedder: Any = None
_boundary_lock = threading.Lock()

_primary_embedder: Any = None
_primary_lock = threading.Lock()

_reranker: Any = None
_reranker_lock = threading.Lock()

_llm_breaker: Any = None
_llm_breaker_lock = threading.Lock()

_hf_configured = False


def _configure_hf() -> bool:
    """Configure HuggingFace environment variables.

    Returns True if offline mode is active.
    """
    global _hf_configured
    settings = get_settings()

    # 1. Check offline mode
    is_offline = settings.offline_mode or os.environ.get("HF_HUB_OFFLINE") == "1"
    if is_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    _hf_configured = True
    return is_offline


def _handle_load_error(e: Exception, model_name: str) -> None:
    """Gracefully handle HuggingFace load errors for new users."""
    if isinstance(e, (LocalEntryNotFoundError, ConnectionError, HfHubHTTPError)):
        raise AudioBenchError(
            f"Could not load model '{model_name}'. "
            f"If this is your first run, ensure you are connected to the internet "
            f"and not forcing offline_mode."
        ) from e
    raise


def get_boundary_embedder() -> Any:
    """Get or load the boundary detection embedder.

    Used for chunking audio transcripts into semantic segments.
    """
    global _boundary_embedder
    if _boundary_embedder is None:
        with _boundary_lock:
            if _boundary_embedder is None:
                is_offline = _configure_hf()
                logger.info(
                    "Loading boundary embedder (sentence-transformers/all-MiniLM-L6-v2) [offline=%s]...",
                    is_offline,
                )
                t0 = time.time()
                from sentence_transformers import SentenceTransformer

                try:
                    # Prefer local cache to avoid unauthenticated/slow HF pings
                    _boundary_embedder = SentenceTransformer(
                        "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True
                    )
                except Exception:
                    if is_offline:
                        raise
                    try:
                        _boundary_embedder = SentenceTransformer(
                            "sentence-transformers/all-MiniLM-L6-v2", local_files_only=False
                        )
                    except Exception as e:
                        _handle_load_error(e, "sentence-transformers/all-MiniLM-L6-v2")
                logger.info("Boundary embedder loaded in %.2fs", time.time() - t0)
    return _boundary_embedder


def get_primary_embedder() -> Any:
    """Get or load the primary storage embedder.

    Used for vectorizing text for storage in LanceDB.
    """
    global _primary_embedder
    if _primary_embedder is None:
        with _primary_lock:
            if _primary_embedder is None:
                is_offline = _configure_hf()
                logger.info(
                    "Loading primary embedder (nomic-ai/nomic-embed-text-v1.5) [offline=%s]...",
                    is_offline,
                )
                t0 = time.time()
                from sentence_transformers import SentenceTransformer

                try:
                    # Prefer local cache
                    _primary_embedder = SentenceTransformer(
                        "nomic-ai/nomic-embed-text-v1.5",
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                except Exception:
                    if is_offline:
                        raise
                    try:
                        _primary_embedder = SentenceTransformer(
                            "nomic-ai/nomic-embed-text-v1.5",
                            trust_remote_code=True,
                            local_files_only=False,
                        )
                    except Exception as e:
                        _handle_load_error(e, "nomic-ai/nomic-embed-text-v1.5")
                logger.info("Primary embedder loaded in %.2fs", time.time() - t0)
    return _primary_embedder


def get_reranker() -> Any:
    """Get or load the cross-encoder reranker.

    Used to re-score and filter the top-K retrieval results.
    """
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                is_offline = _configure_hf()
                logger.info(
                    "Loading cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2) [offline=%s]...",
                    is_offline,
                )
                t0 = time.time()
                from sentence_transformers import CrossEncoder

                try:
                    # Prefer local cache
                    _reranker = CrossEncoder(
                        "cross-encoder/ms-marco-MiniLM-L-6-v2", local_files_only=True
                    )
                except Exception:
                    if is_offline:
                        raise
                    try:
                        _reranker = CrossEncoder(
                            "cross-encoder/ms-marco-MiniLM-L-6-v2", local_files_only=False
                        )
                    except Exception as e:
                        _handle_load_error(e, "cross-encoder/ms-marco-MiniLM-L-6-v2")
                logger.info("Reranker loaded in %.2fs", time.time() - t0)
    return _reranker


_colbert_reranker: Any = None
_colbert_lock = threading.Lock()


def get_colbert_reranker() -> Any:
    """Get or load the AnswerAI ColBERT late-interaction reranker."""
    global _colbert_reranker
    if _colbert_reranker is None:
        with _colbert_lock:
            if _colbert_reranker is None:
                is_offline = _configure_hf()
                logger.info(
                    "Loading ColBERT reranker (answerdotai/answerai-colbert-small-v1) [offline=%s]...",
                    is_offline,
                )
                t0 = time.time()
                try:
                    from lancedb.rerankers import ColbertReranker
                    from rerankers.models.colbert_ranker import ColBERTModel

                    # Monkey-patch for transformers v5.x compatibility
                    original_init = ColBERTModel.__init__

                    def patched_init(self, config, verbose: int):
                        original_init(self, config, verbose)
                        if hasattr(self, "post_init"):
                            self.post_init()
                        elif not hasattr(self, "all_tied_weights_keys"):
                            self.all_tied_weights_keys = {}

                    ColBERTModel.__init__ = patched_init

                    # Temporarily force HF_HUB_OFFLINE to prevent network ping if cached
                    original_hf_offline = os.environ.get("HF_HUB_OFFLINE")
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    try:
                        _colbert_reranker = ColbertReranker(
                            model_name="answerdotai/answerai-colbert-small-v1", column="content"
                        )
                    except Exception:
                        if original_hf_offline is not None:
                            os.environ["HF_HUB_OFFLINE"] = original_hf_offline
                        else:
                            del os.environ["HF_HUB_OFFLINE"]
                        
                        if is_offline:
                            raise
                            
                        # Retry online
                        _colbert_reranker = ColbertReranker(
                            model_name="answerdotai/answerai-colbert-small-v1", column="content"
                        )
                    finally:
                        # Restore original environment if we had to retry
                        if original_hf_offline is not None:
                            os.environ["HF_HUB_OFFLINE"] = original_hf_offline
                        elif "HF_HUB_OFFLINE" in os.environ and original_hf_offline is None and not is_offline:
                            del os.environ["HF_HUB_OFFLINE"]
                            
                    logger.info("ColBERT reranker loaded in %.2fs", time.time() - t0)
                except (ImportError, ModuleNotFoundError) as e:
                    # rerankers package not installed — ColBERT is optional
                    logger.warning(
                        "ColBERT reranker unavailable (%s). "
                        "Install with: pip install rerankers[transformers]. "
                        "Falling back to cross-encoder only.",
                        e,
                    )
                    _colbert_reranker = None  # sentinel: skip ColBERT in search
                except Exception as e:
                    _handle_load_error(e, "answerdotai/answerai-colbert-small-v1")
    return _colbert_reranker


def get_llm_circuit_breaker() -> Any:
    """Get the global circuit breaker for LLM API calls."""
    global _llm_breaker
    if _llm_breaker is None:
        with _llm_breaker_lock:
            if _llm_breaker is None:
                from audiobench.memory.llm_caller import CircuitBreaker
                # Trip after 3 failures, allow probe after 60s
                _llm_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
    return _llm_breaker


def pre_warm_retrieval_pipeline() -> None:
    """Warm up all models sequentially to avoid load penalties during requests.

    Called at daemon startup and FastAPI lifespan.
    """
    logger.info("Pre-warming retrieval pipeline models...")
    t0 = time.time()

    get_boundary_embedder()
    get_primary_embedder()
    get_reranker()
    get_colbert_reranker()  # optional — won't crash if rerankers pkg is absent

    logger.info("Retrieval pipeline fully warmed in %.2fs", time.time() - t0)
