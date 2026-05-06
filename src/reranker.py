"""
reranker.py
───────────
Fix #5 — Cross-Encoder Reranking Layer

After hybrid BM25 + dense retrieval, retrieved chunks are reranked using
a lightweight cross-encoder (or fast cosine fallback) to improve precision
of the top-K chunks sent to the LLM.

Two modes:
    CrossEncoderReranker  — uses sentence-transformers CrossEncoder
                            (ms-marco-MiniLM-L-6-v2, ~80MB, fast CPU inference)
    CosineReranker        — pure-numpy fallback, reranks by semantic overlap
                            between query embedding and chunk embeddings.
                            No extra dependencies.

The public API is identical for both:
    reranker.rerank(query, results, top_k) → list[RetrievalResult]

Usage
─────
    from reranker import get_reranker
    reranker = get_reranker()
    results = reranker.rerank(query, retrieved_chunks, top_k=5)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from retriever import RetrievalResult

logger = logging.getLogger(__name__)

_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ── Cross-encoder reranker (preferred) ───────────────────────────────────────
class CrossEncoderReranker:
    """
    Uses a cross-encoder to score (query, chunk) pairs.
    More accurate than bi-encoder reranking since the model sees both jointly.
    """

    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder  # type: ignore
        logger.info("Loading CrossEncoder: %s", _CROSS_ENCODER_MODEL)
        self._model = CrossEncoder(_CROSS_ENCODER_MODEL, max_length=512)
        logger.info("CrossEncoder ready")

    def rerank(
        self,
        query: str,
        results: list["RetrievalResult"],
        top_k: int = 5,
    ) -> list["RetrievalResult"]:
        if not results:
            return results

        pairs = [(query, r.text[:500]) for r in results]
        scores = self._model.predict(pairs)  # (n,) float array

        ranked = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        reranked = []
        for result, score in ranked[:top_k]:
            result.score = float(score)
            reranked.append(result)

        logger.info(
            "CrossEncoder reranked %d → %d  (top score=%.4f)",
            len(results), len(reranked),
            reranked[0].score if reranked else 0.0,
        )
        return reranked

    @property
    def name(self) -> str:
        return "cross-encoder"


# ── Cosine fallback reranker (no extra deps) ──────────────────────────────────
class CosineReranker:
    """
    Fallback reranker: reranks chunks by cosine similarity between the
    query embedding and each chunk's embedding, computed on-the-fly.

    Slightly less accurate than cross-encoder but zero extra dependencies
    and works with the already-loaded SentenceTransformer model.
    """

    def __init__(self, embedder) -> None:
        self._embedder = embedder

    def rerank(
        self,
        query: str,
        results: list["RetrievalResult"],
        top_k: int = 5,
    ) -> list["RetrievalResult"]:
        if not results:
            return results

        # Encode query
        query_vec = self._embedder.encode_query(query)  # (1, d)

        # Encode all chunk texts
        texts = [r.text[:500] for r in results]
        chunk_vecs = self._embedder.encode_texts(texts)  # (n, d)

        # Cosine similarity (vectors already L2-normalised → inner product = cosine)
        scores = (query_vec @ chunk_vecs.T)[0]  # (n,)

        ranked = sorted(
            zip(results, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        reranked = []
        for result, score in ranked[:top_k]:
            result.score = float(score)
            reranked.append(result)

        logger.info(
            "CosineReranker reranked %d → %d  (top score=%.4f)",
            len(results), len(reranked),
            reranked[0].score if reranked else 0.0,
        )
        return reranked

    @property
    def name(self) -> str:
        return "cosine-fallback"


# ── Factory ───────────────────────────────────────────────────────────────────
def get_reranker(embedder=None) -> CrossEncoderReranker | CosineReranker:
    """
    Return the best available reranker.

    Tries CrossEncoderReranker first; falls back to CosineReranker.
    CosineReranker requires an Embedder instance.
    """
    try:
        return CrossEncoderReranker()
    except (ImportError, Exception) as e:
        logger.warning(
            "CrossEncoder unavailable (%s) — using CosineReranker fallback", e
        )
        if embedder is None:
            raise RuntimeError(
                "CosineReranker requires an Embedder instance. "
                "Pass embedder= to get_reranker()."
            ) from e
        return CosineReranker(embedder)