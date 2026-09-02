"""
faithfulness.py
────────────────
Faithfulness Verification subsystem for the RAKA RAG pipeline.

WHAT this module does
─────────────────────
Given a (answer, context) pair — where `answer` is the LLM's generated
response and `context` is the exact retrieved-chunk text that was fed to
the LLM to produce that answer — this module decides whether the answer
is actually *supported* by the context, independently of the LLM that
generated it.

WHY it exists
─────────────
LLMs hallucinate even when given correct source material. This module is
a second, independent opinion: a small, transparent, auditable classifier
that sits between generation and the user, so that unsupported claims can
be caught and downgraded/blocked before they reach a student or faculty
member (PRD §4.7, "zero hallucination tolerance").

Design
──────
Three cheap, complementary signals are extracted from (answer, context):

    1. semantic_similarity — cosine similarity between SBERT embeddings
       of the answer and the context. Captures overall topical/semantic
       alignment. Cheap to compute, but can be fooled by paraphrases that
       change specifics (e.g. swapped numbers) while staying "on topic".

    2. keyword_jaccard — Jaccard overlap of lowercased, stopword-filtered
       content words. Captures lexical grounding — did the answer actually
       reuse the vocabulary of the source, or introduce new entities/terms?

    3. numeric_consistency — fraction of numeric tokens in the answer
       (percentages, years, decimals, integers, date fragments) that also
       appear in the context. This is the feature specifically designed to
       catch the most dangerous and common RAG hallucination in an academic
       setting: a confidently stated WRONG number (attendance %, fee amount,
       credit count, CGPA cutoff, etc.).

These three features are fed into a pre-trained Logistic Regression
classifier (see train_faithfulness.py) which outputs a single faithfulness
probability. See response_formatter.py for how that probability is turned
into a final user-facing response.

Reused vs. re-implemented
──────────────────────────
This module deliberately does NOT load its own SentenceTransformer model.
It reuses the project's existing `embedder.Embedder` (the same instance
used for retrieval, when available) so that:
  • no duplicate ~90MB model is loaded into memory,
  • encoding behaviour (normalisation, backend choice) stays consistent
    across retrieval and verification,
  • it works identically whether the deployment is running the SBERT
    backend or the pure-sklearn LSA fallback.

Path handling
─────────────
All defaults are computed relative to the project root
(`Path(__file__).resolve().parent.parent`), never via fragile "../data"
strings that only work when launched from a specific working directory.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Project paths (robust — not dependent on current working directory) ──────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "faithfulness_lr.joblib"

# ── Feature order — MUST stay identical between training and prediction ──────
FEATURE_NAMES: tuple[str, str, str] = (
    "semantic_similarity",
    "keyword_jaccard",
    "numeric_consistency",
)

# ── Decision thresholds (PRD §4.7.3) ──────────────────────────────────────────
THRESHOLD_FAITHFUL = 0.75
THRESHOLD_UNCERTAIN = 0.50

# ── Lightweight stopword list (no NLTK dependency) ────────────────────────────
# Deliberately small and generic-English; good enough to strip function words
# without pulling in a large NLP dependency, per the PRD's "avoid unnecessary
# dependencies" instruction.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to",
    "in", "on", "at", "for", "with", "by", "from", "as", "is", "are", "was",
    "were", "be", "been", "being", "this", "that", "these", "those", "it",
    "its", "into", "than", "such", "not", "no", "nor", "so", "do", "does",
    "did", "has", "have", "had", "will", "shall", "may", "must", "can",
    "could", "would", "should", "their", "there", "which", "who", "whom",
    "what", "when", "where", "how", "also", "any", "all", "each", "per",
    "under", "over", "up", "out", "about", "after", "before", "between",
    "within", "shall be", "student", "students",  # domain-generic, low signal
})

# Numeric token pattern, most-specific alternative first:
#   decimal-with-percent (3.5%) | integer-with-percent (75%)
#   | year-range (2024-25 / 2024-2025) | decimal (3.5) | plain integer (75)
_NUMERIC_RE = re.compile(
    r"\d+\.\d+%|\d+%|\d{4}-\d{2,4}|\d+\.\d+|\d+"
)
_TOKEN_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")


# ── Helper: tokenisation ──────────────────────────────────────────────────────
def _content_words(text: str) -> set[str]:
    """
    Extract lowercase, stopword-filtered, alphabetic content words.

    Preprocessing (documented per PRD §14):
        1. lowercase
        2. tokenize on alphabetic runs only (numbers handled separately by
           numeric_consistency — mixing them into the keyword set would
           double-count the same signal)
        3. drop tokens of length <= 2 (articles, initials, noise)
        4. drop stopwords (function words + a couple of domain-generic,
           near-universal terms like "student"/"students" that appear in
           almost every institutional sentence and would inflate overlap
           without indicating real grounding)
    """
    if not text:
        return set()
    tokens = _TOKEN_RE.findall(text.lower())
    return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}


def _normalize_text_numbers(text: str) -> str:
    """
    Normalize numbers and symbols in text:
    1. Normalizes Unicode spaces (NBSP, narrow NBSP) and Unicode hyphens/dashes.
    2. Removes commas within digits (e.g. 5,000 -> 5000, 10,000 -> 10000).
    3. Normalizes percentage whitespace (e.g. "50 %" -> "50%", "< 50 %" -> "<50%").
    4. Expands range percentages (e.g. "50-65%" or "50 % – 65 %" -> "50%-65%").
    5. Ignores citation references (Pages 7-8, pp. 14-17, Section 3.1, Source 1)
       so page/section indices are not treated as factual numeric claims.
    6. Normalizes common abbreviations before clause splitting (e.g. Rs. -> Rs).
    """
    if not text:
        return ""
    # 1. Unicode whitespace & dashes
    t = (
        text.replace("\u00a0", " ")
        .replace("\u202f", " ")
        .replace("\u200b", "")
        .replace("\u2011", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    # 2. Strip bracketed citation blocks: 【Source ...】 or [Source ...]
    t = re.sub(r"[【\[]\s*Source\b[^】\]]*[】\]]", "", t, flags=re.I)
    # 3. Comma in numbers: 5,000 -> 5000
    t = re.sub(r"(?<=\d),(?=\d)", "", t)
    # 4. Space in thousands: 8 000 -> 8000
    t = re.sub(r"(?<=\d)[\s](?=\d{3}\b)", "", t)
    # 5. Percentage whitespace: "50 %" -> "50%"
    t = re.sub(r"(\d+)\s+%", r"\1%", t)
    t = re.sub(r"<\s*(\d+)%", r"<\1%", t)
    t = re.sub(r">\s*(\d+)%", r">\1%", t)
    # 6. Expand range percentages: "50-65%" or "50% - 65%" -> "50%-65%"
    t = re.sub(r"(\d+)%?\s*[-–]\s*(\d+)%", r"\1%-\2%", t)
    t = re.sub(r"\b(\d+)%?\s+to\s+(\d+)%", r"\1% to \2%", t)
    # 7. Ignore citation references (page ranges, sections, source tags)
    t = re.sub(r"\b(?:pages?|pp\.?|p\.)\s*\d+(?:\s*[-–to]+\s*\d+)?\b", "", t, flags=re.I)
    t = re.sub(r"\b(?:section|sec\.?)\s*\d+(?:\.\d+)?\b", "", t, flags=re.I)
    t = re.sub(r"\[?\bSource\s+\d+\]?", "", t, flags=re.I)
    # 8. Guard abbreviations
    t = re.sub(r"\bRs\.\s*", "Rs ", t)
    t = re.sub(r"\bpp\.\s*", "pp ", t)
    t = re.sub(r"\bsec\.\s*", "sec ", t)
    t = re.sub(r"\bno\.\s*", "no ", t)
    return t


def _numeric_tokens(text: str) -> set[str]:
    """
    Extract normalised numeric tokens from text.

    Handles: integers (with/without commas, e.g. 5,000 -> 5000), decimals,
    percentages (50%, 65%), 4-digit years, and 2-digit academic-year ranges.
    Preserves percentage markers to distinguish rates from bare counts.
    """
    if not text:
        return set()
    cleaned = _normalize_text_numbers(text)
    raw = _NUMERIC_RE.findall(cleaned)
    return set(raw)


def _extract_condition_rules(text: str) -> dict[str, dict[str, set[str]]]:
    """
    Extract condition -> {fees: set(), exams: set()} mappings from text clauses.
    Covers attendance thresholds (<50%, 50-65%, 65-75%, >=75%) and their
    associated fees and examination types.
    """
    cleaned = _normalize_text_numbers(text)
    # Join line wraps that occur mid-sentence within paragraphs
    flat = re.sub(r"(?<![.;●\n])\n(?![●\n])", " ", cleaned)
    clauses = [c.strip() for c in re.split(r"[.;●]|\n{2,}|\band\s+(?=students|if|detained)", flat, flags=re.I) if c.strip()]

    cond_map: dict[str, dict[str, set[str]]] = {}
    for clause in clauses:
        cond = None
        if re.search(r"<\s*50%|below\s+50%|less\s+than\s+50%|under\s+50%", clause, re.I):
            cond = "<50%"
        elif re.search(r"50%?-65%|50%\s*to\s*65%|between\s+50%?\s*and\s*65%", clause, re.I):
            cond = "50-65%"
        elif re.search(r"65%?-75%|65%\s*to\s*75%|between\s+65%?\s*and\s*75%", clause, re.I):
            cond = "65-75%"
        elif re.search(r">=?\s*75%|at\s+least\s+75%|75%\s+attendance", clause, re.I):
            cond = ">=75%"

        if not cond:
            continue

        if cond not in cond_map:
            cond_map[cond] = {"fees": set(), "exams": set()}

        fee_matches = re.findall(r"(?:rs|₹|inr|fees?[:\s]*)\s*(\d+)", clause, re.I)
        if not fee_matches and re.search(r"\bfee\b|\bcost\b|\bcharges?\b", clause, re.I):
            fee_matches = [m for m in _NUMERIC_RE.findall(clause) if not m.endswith("%") and len(m) >= 3]

        for f in fee_matches:
            cond_map[cond]["fees"].add(f)

        if re.search(r"special\s+exam(?:ination)?", clause, re.I):
            cond_map[cond]["exams"].add("special_exam")
        elif re.search(r"re-?exam(?:ination)?", clause, re.I):
            cond_map[cond]["exams"].add("re_exam")

    return cond_map


def _check_condition_value_contradiction(answer: str, context: str) -> bool:
    """
    Lightweight rule-based check for condition/value cross-attribution contradictions.
    Returns True if the answer binds a condition (e.g. <50% attendance) to a fee or
    exam type that contradicts the rules for that condition specified in the context.
    """
    ans_rules = _extract_condition_rules(answer)
    if not ans_rules:
        return False
    ctx_rules = _extract_condition_rules(context)
    if not ctx_rules:
        return False

    for cond, ans_data in ans_rules.items():
        if cond in ctx_rules:
            ctx_data = ctx_rules[cond]
            # Check fee conflict: if answer claims fees for this condition, do they match context for this condition?
            if ans_data["fees"] and ctx_data["fees"]:
                unmatched_fees = ans_data["fees"] - ctx_data["fees"]
                if unmatched_fees:
                    logger.warning(
                        "Condition/fee contradiction detected: condition %s claimed fees %s, context allows %s",
                        cond, ans_data["fees"], ctx_data["fees"],
                    )
                    return True
            # Check exam type conflict
            if ans_data["exams"] and ctx_data["exams"]:
                unmatched_exams = ans_data["exams"] - ctx_data["exams"]
                if unmatched_exams:
                    logger.warning(
                        "Condition/exam contradiction detected: condition %s claimed exams %s, context allows %s",
                        cond, ans_data["exams"], ctx_data["exams"],
                    )
                    return True
    return False


# ── Embedder loading (lazy, cached, reused where possible) ───────────────────
def _load_default_embedder(index_dir: Path):
    """
    Load the project's existing Embedder from a pre-built index directory.

    This is used only when no embedder instance is explicitly passed to
    FaithfulnessVerifier — e.g. when running create_faithfulness_dataset.py
    or train_faithfulness.py as standalone scripts. In the live web/chat
    server, the already-loaded `retriever._embedder` should be passed in
    instead so the model is loaded exactly once for the whole process.
    """
    from embedder import Embedder  # local import: keeps this module importable
                                    # even before embedder.py's deps are ready
    if not index_dir.exists():
        raise FileNotFoundError(
            f"Cannot auto-load embedder: index directory not found at "
            f"{index_dir}. Build the retrieval index first (main.py build), "
            f"or pass an already-loaded Embedder instance explicitly."
        )
    return Embedder.load(index_dir)


@dataclass
class FeatureResult:
    """Container for one (answer, context) pair's extracted features."""
    semantic_similarity: float
    keyword_jaccard: float
    numeric_consistency: float

    def as_list(self) -> list[float]:
        """Fixed-order feature vector — MUST match FEATURE_NAMES order."""
        return [self.semantic_similarity, self.keyword_jaccard, self.numeric_consistency]

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class FaithfulnessVerifier:
    """
    Independent post-generation faithfulness checker.

    Usage (production — reusing an already-loaded embedder)
    ─────────────────────────────────────────────────────────
        verifier = FaithfulnessVerifier(embedder=retriever._embedder)
        result = verifier.predict(answer, context)
        # result = {"score": 0.91, "status": "faithful", "features": {...}}

    Usage (feature extraction only, e.g. for dataset labelling / training —
    no trained classifier required yet)
    ─────────────────────────────────────────────────────────────────────────
        verifier = FaithfulnessVerifier(model_path=None)
        features = verifier.extract_features(answer, context)  # dict
    """

    def __init__(
        self,
        model_path: str | Path | None = DEFAULT_MODEL_PATH,
        embedder: Optional[Any] = None,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
    ) -> None:
        """
        Parameters
        ──────────
        model_path : path to the joblib-serialised trained classifier.
                     Pass None to build a verifier that can only do feature
                     extraction (no predict()) — used during dataset
                     creation and before training exists.
        embedder   : an already-constructed embedder exposing
                     `.encode_texts(list[str]) -> np.ndarray`. If omitted,
                     one is lazily loaded from `index_dir` on first use.
        index_dir  : directory containing the pre-built embedder state
                     (embedder_meta.pkl / lsa_state.pkl). Only used if
                     `embedder` is not provided.
        """
        self._embedder = embedder
        self._index_dir = Path(index_dir)
        self._model_path = Path(model_path) if model_path is not None else None
        self._model: Any = None
        self._model_meta: dict[str, Any] = {}

        if self._model_path is not None:
            self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────
    def _load_model(self) -> None:
        """
        Load the trained Logistic Regression classifier from disk.

        Raises explicitly (per PRD §26 — no silent failure) if the model
        file is missing or malformed, rather than defaulting predict() to
        a fake "faithful" result.
        """
        import joblib

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Faithfulness model not found at {self._model_path}. "
                f"Run: python src/create_faithfulness_dataset.py "
                f"&& python src/train_faithfulness.py"
            )
        try:
            payload = joblib.load(self._model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load faithfulness model from {self._model_path}: {exc}"
            ) from exc

        if isinstance(payload, dict) and "model" in payload:
            self._model = payload["model"]
            self._model_meta = {k: v for k, v in payload.items() if k != "model"}
        else:
            # Backwards-compatible: allow a bare estimator to have been saved.
            self._model = payload
            self._model_meta = {}

        if not hasattr(self._model, "predict_proba"):
            raise RuntimeError(
                f"Loaded object from {self._model_path} has no predict_proba() "
                f"— expected a scikit-learn classifier."
            )
        logger.info(
            "FaithfulnessVerifier: model loaded from %s (meta=%s)",
            self._model_path, self._model_meta,
        )

    # ── Embedder access ───────────────────────────────────────────────────
    def _get_embedder(self):
        if self._embedder is None:
            logger.info(
                "FaithfulnessVerifier: no embedder supplied — lazily loading "
                "from %s (loaded once, then cached).", self._index_dir,
            )
            self._embedder = _load_default_embedder(self._index_dir)
        return self._embedder

    # ── Feature 1: semantic similarity ───────────────────────────────────
    def semantic_similarity(self, answer: str, context: str) -> float:
        """
        Cosine similarity between the SBERT embedding of `answer` and of
        `context`, using the project's shared Embedder (loaded once).

        Empty-input handling: an empty answer or empty context has no
        defined semantic relationship — returns 0.0 (treated as
        "no support"), which is the safe default given this feeds a
        hallucination-detection gate (PRD §26 — never silently default to
        faithful).
        """
        answer = (answer or "").strip()
        context = (context or "").strip()
        if not answer or not context:
            return 0.0

        embedder = self._get_embedder()
        # Single batched call for both texts — cheaper than two calls, and
        # the vectors are already L2-normalised by Embedder, so cosine
        # similarity reduces to a plain dot product.
        vectors = embedder.encode_texts([answer, context])
        sim = float(np.dot(vectors[0], vectors[1]))
        # Numerical safety: clip tiny FP overshoot outside [-1, 1]
        return max(-1.0, min(1.0, sim))

    # ── Feature 2: keyword Jaccard ────────────────────────────────────────
    def keyword_jaccard(self, answer: str, context: str) -> float:
        """
        Jaccard similarity between content-word sets of answer and context.
        See `_content_words()` for the exact preprocessing steps.

        Empty-set handling (documented per PRD §14):
            • both sets empty  -> 1.0 (nothing to contradict; vacuously
              consistent — this only happens for degenerate near-empty
              inputs, which semantic_similarity will separately score 0.0
              on empty text, so it does not inflate a real prediction)
            • exactly one empty -> 0.0 (no lexical overlap possible)
        """
        a_words = _content_words(answer)
        c_words = _content_words(context)

        if not a_words and not c_words:
            return 1.0
        if not a_words or not c_words:
            return 0.0

        intersection = a_words & c_words
        union = a_words | c_words
        return len(intersection) / len(union)

    # ── Feature 3: numeric consistency ───────────────────────────────────
    def numeric_consistency(self, answer: str, context: str) -> float:
        """
        Fraction of the answer's numeric tokens that also appear among the
        context's numeric tokens. See `_numeric_tokens()` for extraction
        and normalisation rules.

        Empty-input handling (documented per PRD §15):
            • answer has NO numeric tokens -> 1.0. The answer makes no
              numeric claim, so there is nothing for this feature to
              contradict; treating "no claim" as "fully consistent" avoids
              penalising perfectly good non-numeric answers (e.g. "Yes,
              students must submit the form before the deadline.").
            • answer HAS numeric tokens but context has none -> 0.0 for
              each such token (none can possibly be verified), so the
              overall fraction is 0.0.
        """
        answer_nums = _numeric_tokens(answer)
        if not answer_nums:
            return 1.0

        context_nums = _numeric_tokens(context)
        if not context_nums:
            return 0.0

        # Condition/value contradiction check (e.g. attaching 50-65% fee to <50% attendance)
        if _check_condition_value_contradiction(answer, context):
            return 0.0

        matched = {n for n in answer_nums if n in context_nums}
        unmatched = answer_nums - context_nums

        # Direct contradiction on fee / quantitative amounts:
        # A 3+ digit number in the answer not present in context is an ungrounded quantity
        fee_like_unmatched = [n for n in unmatched if re.match(r"^\d{3,}$", n)]
        if fee_like_unmatched:
            return 0.0

        if unmatched:
            # Heavily penalize any unmatched numbers so prompt numbers can't mask an incorrect number
            return max(0.0, (len(matched) - 2 * len(unmatched)) / len(answer_nums))

        return 1.0

    # ── Combined feature extraction ──────────────────────────────────────
    def extract_features(self, answer: str, context: str) -> FeatureResult:
        """
        Compute all three features for a (answer, context) pair.

        WHAT: runs semantic_similarity, keyword_jaccard, numeric_consistency.
        WHY: single entry point so training and prediction can never drift
             out of sync on feature order (FEATURE_NAMES is the single
             source of truth for ordering).
        INPUT: answer (str), context (str) — context should be the exact
               text block that was passed to the LLM, not the whole corpus.
        OUTPUT: FeatureResult — use .as_list() for the model, .as_dict()
                for a human-readable / API-friendly view.
        """
        sem = self.semantic_similarity(answer, context)
        jac = self.keyword_jaccard(answer, context)
        num = self.numeric_consistency(answer, context)
        return FeatureResult(
            semantic_similarity=round(sem, 6),
            keyword_jaccard=round(jac, 6),
            numeric_consistency=round(num, 6),
        )

    # ── Prediction ────────────────────────────────────────────────────────
    def predict(self, answer: str, context: str) -> dict[str, Any]:
        """
        Full faithfulness verification for a generated answer.

        WHAT: extracts features, runs the trained Logistic Regression
              classifier, applies the PRD §4.7.3 decision thresholds.
        WHY: single call site used by chat.py / web_server.py / main.py
             right after Generator.generate().
        INPUT: answer (str) — the LLM's generated text.
               context (str) — the exact assembled context block used to
               generate that answer (e.g. GeneratorResponse.context_used).
        OUTPUT: {
            "score": float in [0, 1]        — P(faithful) from the classifier
            "status": "faithful" | "uncertain" | "unfaithful"
            "features": {feature_name: value, ...}
        }

        Raises
        ──────
        RuntimeError  if no model was loaded (model_path=None was passed
                      at construction, or the model failed validation).
                      This module never silently treats a missing model as
                      "faithful" — that would defeat the entire purpose of
                      the gate (PRD §26).
        ValueError    if answer or context is empty — an empty answer/
                      context cannot be meaningfully verified.
        """
        if self._model is None:
            raise RuntimeError(
                "FaithfulnessVerifier.predict() called but no trained model "
                "is loaded. Construct with a valid model_path, or call "
                "extract_features() directly if you only need features."
            )
        if not answer or not answer.strip():
            raise ValueError("predict: answer is empty")
        if not context or not context.strip():
            raise ValueError("predict: context is empty")

        features = self.extract_features(answer, context)
        vector = np.array([features.as_list()], dtype=np.float64)  # shape (1, 3)

        proba = self._model.predict_proba(vector)[0]
        # Class 1 = "faithful" by dataset construction convention (see
        # create_faithfulness_dataset.py) — classes_ lets us look this up
        # robustly instead of assuming column order.
        classes = list(getattr(self._model, "classes_", [0, 1]))
        faithful_idx = classes.index(1) if 1 in classes else 1
        score = float(proba[faithful_idx])

        if score >= THRESHOLD_FAITHFUL:
            status = "faithful"
        elif score >= THRESHOLD_UNCERTAIN:
            status = "uncertain"
        else:
            status = "unfaithful"

        return {
            "score": round(score, 6),
            "status": status,
            "features": features.as_dict(),
        }

    @property
    def is_ready(self) -> bool:
        """True if a trained classifier is loaded and predict() can be called."""
        return self._model is not None