"""
create_faithfulness_dataset.py
────────────────────────────────
Builds the labelled (answer, context, label) dataset used to train the
faithfulness classifier consumed by faithfulness.py / train_faithfulness.py.

WHAT this script does
──────────────────────
Reads the REAL retrieval chunks (data/processed/chunks.json — the exact
same chunks that are served to the LLM at query time) and derives
domain-specific, labelled (answer, context) pairs:

    label = 1  (faithful)    → the answer IS supported by the context
    label = 0  (unfaithful)  → the answer is NOT supported by the context

No external LLM calls are used to generate examples — everything is
derived deterministically (seed=42) from the chunk text itself, so the
dataset:
  • requires no API key / network access,
  • is fully reproducible,
  • reflects the actual vocabulary, numbers, and phrasing style of the
    institutional documents this classifier will be evaluated against
    in production (PRD §4.7.2 — "curated domain-specific examples").

Positive (label=1) example strategies
──────────────────────────────────────
  1. single_sentence      — one real sentence lifted from the chunk.
  2. two_sentence         — two adjacent real sentences from the chunk.
  3. paraphrase           — a real sentence with a light, meaning-preserving
                             lexical substitution (numbers/entities kept
                             intact), simulating how an LLM would restate
                             (not just copy) supported content.

Negative (label=0) example strategies
────────────────────────────────────────
  1. context_mismatch     — a real sentence from chunk A is paired with the
                             FULL TEXT of a different, unrelated chunk B as
                             its "context". The claim is topically real but
                             not grounded in the paired context — the most
                             common real-world RAG failure (retrieval/LLM
                             attaching an answer to the wrong source).
  2. numeric_hallucination — a real sentence containing a number (%, fee
                             amount, credit count, year, etc.) is paired
                             with its OWN correct context, but the number in
                             the ANSWER is mutated to a different, wrong
                             value. This is the single most dangerous and
                             most common academic RAG hallucination
                             (PRD §4.7.1 — numeric consistency feature),
                             and deliberately keeps semantic_similarity and
                             keyword_jaccard HIGH while only the number is
                             wrong, forcing the classifier to actually rely
                             on the numeric_consistency feature rather than
                             shortcut on lexical overlap alone.
  3. off_topic             — a generic, plausible-sounding but entirely
                             unrelated campus statement (not derived from
                             any institutional document) is used as the
                             "answer" for a real context, simulating a
                             fully off-topic / out-of-scope generation.

Balancing & dedup
──────────────────
  • Exact-duplicate (answer, context) pairs are removed (by content hash).
  • Classes are capped to within ~15% of each other (reasonably balanced,
    per spec) by randomly downsampling the majority class with the fixed
    seed — never by inventing new rows.

Output
──────
    data/faithfulness/dataset.csv with columns:
        answer, context, label, pair_type, source, category,
        doc_id, chunk_id, group_key

    `group_key` = the context chunk's chunk_id (globally unique in
    chunks.json). train_faithfulness.py uses this column to perform a
    GROUP-aware train/test split, so that the several rows generated from
    the SAME underlying context chunk (single_sentence / two_sentence /
    paraphrase / numeric_hallucination / off_topic all reuse one chunk's
    full text as context) can never be split across train and test — the
    most direct source of leakage in this dataset. Grouping at chunk level
    (195 groups) also gives a far more representative, better-balanced
    split than grouping at document level (only 4 source documents).

Usage
─────
    python src/create_faithfulness_dataset.py
    python src/create_faithfulness_dataset.py --chunks data/processed/chunks.json \\
        --out data/faithfulness/dataset.csv --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("raka.faithfulness_dataset")

# ── Project paths (robust — independent of current working directory) ───────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "faithfulness" / "dataset.csv"

SEED = 42
MIN_SENTENCE_WORDS = 6           # discard fragments too short to be a claim
MIN_DATASET_SIZE = 500           # PRD §4.7.2 minimum labelled pairs
MAX_CLASS_IMBALANCE_RATIO = 1.15 # majority class capped to this × minority

# ── Sentence splitting (self-contained — dataset generation only) ───────────
# Deliberately simple: we only need reasonably clean, complete-looking
# sentences to build realistic answer text, not production-grade chunking.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

_ABBREV_GUARD = [
    (re.compile(r"\bRs\.\s"), "Rs<<DOT>> "),
    (re.compile(r"\bB\.Tech\.\s"), "B<<DOT>>Tech<<DOT>> "),
    (re.compile(r"\bM\.Tech\.\s"), "M<<DOT>>Tech<<DOT>> "),
    (re.compile(r"\be\.g\.\s"), "e<<DOT>>g<<DOT>> "),
    (re.compile(r"\bi\.e\.\s"), "i<<DOT>>e<<DOT>> "),
    (re.compile(r"\betc\.\s"), "etc<<DOT>> "),
    (re.compile(r"\bw\.e\.f\.\s"), "w<<DOT>>e<<DOT>>f<<DOT>> "),
    (re.compile(r"\bDr\.\s"), "Dr<<DOT>> "),
    (re.compile(r"\bProf\.\s"), "Prof<<DOT>> "),
    (re.compile(r"\bNo\.\s"), "No<<DOT>> "),
]


def _split_sentences(text: str) -> list[str]:
    """Split chunk text into clean, reasonably complete sentences."""
    text = text.replace("\n", " ")
    text = re.sub(r"\[TABLE\].*?\[/TABLE\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"\s{2,}", " ", text).strip()

    protected = text
    for pat, repl in _ABBREV_GUARD:
        protected = pat.sub(repl, protected)

    parts = _SENT_SPLIT_RE.split(protected)

    sentences: list[str] = []
    for part in parts:
        restored = part.replace("<<DOT>>", ".")
        restored = restored.strip(" \t|")
        if not restored:
            continue
        word_count = len(restored.split())
        if word_count < MIN_SENTENCE_WORDS:
            continue
        # Drop fragments that are mostly non-alphabetic (stray table remnants)
        alpha_words = sum(1 for w in restored.split() if re.search(r"[a-zA-Z]", w))
        if alpha_words / max(word_count, 1) < 0.5:
            continue
        sentences.append(restored)
    return sentences


# ── Numeric mutation (for the numeric_hallucination negative class) ─────────
_NUM_RE = re.compile(r"\d+\.\d+%|\d+%|\d{4}-\d{2,4}|\d+\.\d+|\d+")


def _mutate_number(token: str, rng: random.Random) -> str:
    """Return a plausible but DIFFERENT numeric token, preserving format."""
    has_percent = token.endswith("%")
    core = token[:-1] if has_percent else token

    if re.fullmatch(r"\d{4}-\d{2,4}", core):
        # Academic-year range, e.g. "2024-25" or "2024-2025"
        start_str, end_str = core.split("-")
        shift = rng.choice([-2, -1, 1, 2])
        new_start = int(start_str) + shift
        if len(end_str) == 2:
            new_end = f"{(new_start + 1) % 100:02d}"
        else:
            new_end = str(new_start + 1)
        return f"{new_start}-{new_end}"

    if "." in core:
        try:
            val = float(core)
            delta = rng.choice([-2.5, -1.5, -1.0, 1.0, 1.5, 2.5])
            new_val = max(round(val + delta, 1), 0.1)
            if new_val == val:
                new_val = round(val + 1.0, 1)
            new_core = f"{new_val:g}"
            return new_core + ("%" if has_percent else "")
        except ValueError:
            return token

    try:
        val = int(core)
        delta = rng.choice([-25, -20, -15, -10, -5, 5, 10, 15, 20, 25])
        new_val = max(val + delta, 0)
        if new_val == val:
            new_val += 5
        return str(new_val) + ("%" if has_percent else "")
    except ValueError:
        return token


def _hallucinate_sentence_number(sentence: str, rng: random.Random) -> str | None:
    """
    Replace exactly one numeric token in `sentence` with a wrong value.
    Returns None if no numeric token is present or mutation is a no-op.
    """
    matches = list(_NUM_RE.finditer(sentence))
    if not matches:
        return None
    match = rng.choice(matches)
    original = match.group(0)
    mutated = _mutate_number(original, rng)
    if mutated == original:
        return None
    return sentence[: match.start()] + mutated + sentence[match.end():]


# ── Light paraphrase (for the paraphrase positive class) ────────────────────
# Small, conservative substitution map: meaning-preserving, numbers/entities
# untouched. Only ONE substitution is applied per sentence (first match) so
# the result still reads naturally.
_PARAPHRASE_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bis mandatory\b", re.I), "is compulsory"),
    (re.compile(r"\bmust\b", re.I), "is required to"),
    (re.compile(r"\bshall\b", re.I), "will"),
    (re.compile(r"\bminimum\b", re.I), "least"),
    (re.compile(r"\brequirement\b", re.I), "criterion"),
    (re.compile(r"\bstudents\b", re.I), "candidates"),
    (re.compile(r"\bpermitted\b", re.I), "allowed"),
    (re.compile(r"\bapplicable\b", re.I), "applicable in this case"),
    (re.compile(r"\bin order to\b", re.I), "so as to"),
    (re.compile(r"\bregarding\b", re.I), "concerning"),
]


def _light_paraphrase(sentence: str) -> str:
    for pattern, replacement in _PARAPHRASE_MAP:
        if pattern.search(sentence):
            return pattern.sub(replacement, sentence, count=1)
    return sentence


# ── Off-topic filler pool (for the off_topic negative class) ────────────────
# Deliberately generic campus-life statements that do NOT originate from,
# and are not supported by, any institutional document in this corpus.
_OFF_TOPIC_FILLERS = [
    "The nearest railway station is about a fifteen minute auto ride from campus.",
    "The canteen introduces a new special dish every Friday afternoon.",
    "Visitors must register their vehicle number at the security gate before entering.",
    "The college football team reached the semi-finals of the inter-collegiate tournament.",
    "Wi-fi access for guest devices is limited to the seminar hall and auditorium.",
    "The library plans to extend its reading room hours during the winter break.",
    "Parking stickers for two-wheelers are issued from the admin block every August.",
    "The cultural committee is organising a photography exhibition next month.",
    "Lost identity cards can be reissued from the accounts office after paying a nominal fee.",
    "The campus shuttle service runs every thirty minutes between the two hostels.",
]


# ── Data loading ──────────────────────────────────────────────────────────────
def _load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.json not found: {chunks_path}")
    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"chunks.json must be a non-empty JSON array: {chunks_path}")
    logger.info("Loaded %d chunks from %s", len(data), chunks_path)
    return data


# ── Row builder ───────────────────────────────────────────────────────────────
def _make_row(
    answer: str,
    context_chunk: dict[str, Any],
    label: int,
    pair_type: str,
) -> dict[str, Any]:
    return {
        "answer": answer.strip(),
        "context": context_chunk["text"].strip(),
        "label": label,
        "pair_type": pair_type,
        "source": context_chunk.get("source", "unknown"),
        "category": context_chunk.get("category", "general"),
        "doc_id": context_chunk.get("doc_id", ""),
        "chunk_id": context_chunk.get("chunk_id", -1),
        # group_key drives the leakage-aware split in train_faithfulness.py:
        # all rows built from this SAME context chunk share a group, so
        # they can never be split across train and test.
        "group_key": context_chunk.get("chunk_id", context_chunk.get("doc_id", "unknown")),
    }


# ── Pair generation ────────────────────────────────────────────────────────────
def generate_pairs(
    chunks: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # Pre-split sentences once per chunk (expensive-ish regex work)
    sentence_cache: dict[int, list[str]] = {}
    for chunk in chunks:
        sentence_cache[chunk["chunk_id"]] = _split_sentences(chunk["text"])

    for chunk in chunks:
        cid = chunk["chunk_id"]
        sentences = sentence_cache[cid]
        if len(sentences) < 2:
            continue  # not enough material for reliable pairs

        numeric_sentences = [s for s in sentences if _NUM_RE.search(s)]

        # ── POSITIVE 1: single real sentence, its own context ──────────────
        single = rng.choice(sentences)
        rows.append(_make_row(single, chunk, 1, "single_sentence"))

        # ── POSITIVE 2: two adjacent real sentences, own context ───────────
        i = rng.randrange(0, len(sentences) - 1)
        combined = sentences[i] + " " + sentences[i + 1]
        rows.append(_make_row(combined, chunk, 1, "two_sentence"))

        # ── POSITIVE 3: light paraphrase of a numeric sentence ──────────────
        if numeric_sentences:
            base = rng.choice(numeric_sentences)
            paraphrased = _light_paraphrase(base)
            rows.append(_make_row(paraphrased, chunk, 1, "paraphrase"))

        # ── NEGATIVE 1: cross-chunk context mismatch ────────────────────────
        other_sources = [c for c in chunks if c.get("source") != chunk.get("source")]
        candidates = other_sources or [c for c in chunks if c["chunk_id"] != cid]
        if candidates:
            wrong_context_chunk = rng.choice(candidates)
            mismatched_answer = rng.choice(sentences)
            rows.append(
                _make_row(mismatched_answer, wrong_context_chunk, 0, "context_mismatch")
            )

        # ── NEGATIVE 2: numeric hallucination, own (correct) context ────────
        if numeric_sentences:
            base = rng.choice(numeric_sentences)
            mutated = _hallucinate_sentence_number(base, rng)
            if mutated:
                rows.append(_make_row(mutated, chunk, 0, "numeric_hallucination"))

        # ── NEGATIVE 3: off-topic filler answer, real context ───────────────
        filler = rng.choice(_OFF_TOPIC_FILLERS)
        rows.append(_make_row(filler, chunk, 0, "off_topic"))

    return rows


# ── Dedup + balance ────────────────────────────────────────────────────────────
def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = hashlib.md5(
            (row["answer"].lower().strip() + "|||" + row["context"].lower().strip()).encode()
        ).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    dropped = len(rows) - len(deduped)
    if dropped:
        logger.info("Dropped %d exact-duplicate (answer, context) pairs", dropped)
    return deduped


def _balance(rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    minority = min(len(pos), len(neg))
    cap = int(minority * MAX_CLASS_IMBALANCE_RATIO)

    pos = pos[:cap] if len(pos) > cap else pos
    neg = neg[:cap] if len(neg) > cap else neg

    balanced = pos + neg
    rng.shuffle(balanced)
    logger.info(
        "Balanced dataset: %d faithful (1) / %d unfaithful (0) — ratio %.2f",
        len(pos), len(neg), len(pos) / max(len(neg), 1),
    )
    return balanced


# ── CSV writer ────────────────────────────────────────────────────────────────
_CSV_FIELDS = [
    "answer", "context", "label", "pair_type",
    "source", "category", "doc_id", "chunk_id", "group_key",
]


def _write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info("Saved %d rows → %s", len(rows), out_path)


# ── Report ─────────────────────────────────────────────────────────────────────
def _print_report(rows: list[dict[str, Any]]) -> None:
    n_pos = sum(1 for r in rows if r["label"] == 1)
    n_neg = sum(1 for r in rows if r["label"] == 0)

    type_counts: dict[str, int] = {}
    for r in rows:
        type_counts[r["pair_type"]] = type_counts.get(r["pair_type"], 0) + 1

    print(f"\n{'═'*62}")
    print("  RAKA — Faithfulness Dataset Report")
    print(f"{'═'*62}")
    print(f"  Total examples     : {len(rows)}")
    print(f"  Faithful   (1)     : {n_pos}")
    print(f"  Unfaithful (0)     : {n_neg}")
    print(f"  Class ratio (1:0)  : {n_pos / max(n_neg, 1):.2f}")
    print(f"\n  By pair_type:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    ├─ {t:<22}: {c}")
    print(f"{'═'*62}\n")


# ── CLI / main ──────────────────────────────────────────────────────────────────
def build_dataset(chunks_path: Path, out_path: Path, seed: int = SEED) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    chunks = _load_chunks(chunks_path)

    rows = generate_pairs(chunks, rng)
    logger.info("Generated %d raw (answer, context) pairs", len(rows))

    rows = _dedupe(rows)
    rows = _balance(rows, rng)

    if len(rows) < MIN_DATASET_SIZE:
        logger.warning(
            "Dataset has only %d examples (< %d target). Consider adding more "
            "source PDFs / chunks to data/processed/chunks.json.",
            len(rows), MIN_DATASET_SIZE,
        )

    _write_csv(rows, out_path)
    _print_report(rows)
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the labelled faithfulness training dataset from chunks.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS_PATH))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_dataset(
        chunks_path=Path(args.chunks),
        out_path=Path(args.out),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()