"""
pipeline.py
-----------
Single shared implementation of the RAKA query pipeline:

    normalize/classify -> retrieve -> rerank -> generate -> verify

Both the CLI (chat.py) and the web API (web_server.py) call `run_pipeline()`
so there is exactly one place that decides how a question turns into a
verified answer, and exactly one place that applies the faithfulness
safety gate. Neither caller re-implements or bypasses verification.

Faithfulness feature calculations, thresholds, and the trained model are
untouched here — this module only decides what to DO with the verifier's
output (safe-substitute on "unfaithful", warn on "uncertain", pass through
on "faithful"), reusing FaithfulnessVerifier.predict() from faithfulness.py
unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from query_normalizer import (
    normalize_query,
    classify_query_category,
    get_category_sources,
)
from faithfulness import FaithfulnessVerifier

RETRIEVE_K = 10
RERANK_TOP_K = 3

# Safety-gate responses (single source of truth for both CLI and API).
SAFE_ANSWER = "Unable to verify the answer from the available sources."
UNCERTAIN_WARNING = "The answer may not be fully supported by the retrieved sources."


@dataclass
class PipelineResult:
    """Everything a caller (CLI or API) needs to present one answer."""
    question: str
    normalized_query: str
    category: str
    answer: str                        # final, safety-gated answer to show the user
    raw_answer: str                     # what the LLM actually generated, pre-gate
    context_used: str                   # exact text block checked against the answer
    sources: list[dict[str, Any]]
    model: str
    backend: str
    response_time_ms: int
    mock: bool
    faithfulness_status: str            # "faithful" | "uncertain" | "unfaithful"
    faithfulness_score: float
    faithfulness_features: dict[str, float]
    warning: Optional[str]


def run_pipeline(
    question: str,
    retriever: Any,
    reranker: Any,
    generator: Any,
    verifier: FaithfulnessVerifier,
    k: int = RETRIEVE_K,
    rerank_top_k: int = RERANK_TOP_K,
) -> PipelineResult:
    """
    Run the full pipeline for one question and apply the faithfulness gate.

    `context` used for verification is always `response.context_used` — the
    exact text block the generator actually sent to the LLM — falling back
    to the reranked chunk text only if that is unexpectedly empty, so the
    verifier never checks against anything other than what really produced
    the answer.
    """
    normalized_query = normalize_query(question)
    category = classify_query_category(normalized_query)
    allowed_sources = get_category_sources(category)

    results = retriever.retrieve(normalized_query, k=k, allowed_sources=allowed_sources)
    results = reranker.rerank(normalized_query, results, top_k=rerank_top_k)

    response = generator.generate(normalized_query, results)

    context = (
        response.context_used
        if response.context_used
        else "\n\n".join(r.text for r in results)
    )

    if context.strip() and response.answer.strip():
        verification = verifier.predict(response.answer, context)
    else:
        # Nothing was retrieved/generated to check the answer against.
        # FaithfulnessVerifier.predict() rejects empty input by design
        # (PRD §26 - never silently default to faithful) rather than
        # crash the pipeline, treat "nothing to verify" the same as a
        # failed verification.
        verification = {
            "score": 0.0,
            "status": "unfaithful",
            "features": {
                "semantic_similarity": 0.0,
                "keyword_jaccard": 0.0,
                "numeric_consistency": 0.0,
            },
        }

    status = verification["status"]
    warning: Optional[str] = None
    if status == "unfaithful":
        final_answer = SAFE_ANSWER
    elif status == "uncertain":
        final_answer = response.answer
        warning = UNCERTAIN_WARNING
    else:
        final_answer = response.answer

    return PipelineResult(
        question=question,
        normalized_query=normalized_query,
        category=category,
        answer=final_answer,
        raw_answer=response.answer,
        context_used=context,
        sources=response.sources,
        model=response.model,
        backend=response.backend,
        response_time_ms=response.response_time_ms,
        mock=response.mock,
        faithfulness_status=status,
        faithfulness_score=verification["score"],
        faithfulness_features=verification["features"],
        warning=warning,
    )
