"""
reverify_faithfulness.py
--------------------------
LOCAL RE-VERIFICATION ONLY.

Re-runs the UPDATED FaithfulnessVerifier (see faithfulness.py's condition/
fee-attribution and percentage/"out of 100" bugfixes) against an EXISTING
saved validation run — data/faithfulness/real_validation*.csv — using the
already-saved raw_answer + context_used for each row. This never calls the
generator, Groq, or OpenAI, and never produces a new answer: it only asks
"if we score this exact same saved (answer, context) pair with today's
verifier code, what does it say now?" and compares that against what was
saved at the time of the original run.

WHAT gets updated (only the 40 status="success" rows):
    faithfulness_score, semantic_similarity, keyword_jaccard,
    numeric_consistency, verifier_decision

WHAT is preserved exactly as-is, for every row:
    id, category, question, answer, raw_answer, context_used, sources_json,
    human_faithfulness, human_answerability, and (for status="api_error"
    rows) every column, untouched — those 5 rows are never re-verified.

Output: data/faithfulness/real_validation_reverified.csv (same schema/
column order as the input file).
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from faithfulness import FaithfulnessVerifier  # noqa: E402

FAITHFULNESS_DIR = PROJECT_ROOT / "data" / "faithfulness"
OUTPUT_CSV = FAITHFULNESS_DIR / "real_validation_reverified.csv"

# The path named in the request may not match what's actually on disk (e.g.
# a browser-download-style "(3)" suffix) — check likely candidates first,
# then fall back to locating any real_validation*.csv rather than creating
# a new dataset, per requirement 1.
_CANDIDATE_NAMES = [
    "real_validation(3).csv",
    "real_validation.csv",
]

REQUIRED_COLUMNS = {
    "id", "status", "verifier_decision", "raw_answer", "context_used",
    "faithfulness_score", "semantic_similarity", "keyword_jaccard",
    "numeric_consistency", "human_faithfulness", "human_answerability",
}


def _locate_validation_csv() -> Path:
    for name in _CANDIDATE_NAMES:
        candidate = FAITHFULNESS_DIR / name
        if candidate.exists():
            return candidate
    matches = sorted(
        (p for p in FAITHFULNESS_DIR.glob("real_validation*.csv") if p.name != OUTPUT_CSV.name),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No real_validation*.csv found under {FAITHFULNESS_DIR}. "
        f"This script re-verifies an EXISTING validation run — it does not "
        f"generate one. Run src/evaluate_faithfulness.py first if none exists yet."
    )


def _is_binary_label(value: str | None) -> bool:
    return (value or "").strip() in ("0", "1")


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def _binary_metrics(pairs: list[tuple[int, int]]) -> dict[str, float]:
    """pairs: list of (predicted, human), both 0/1."""
    tp = fp = tn = fn = 0
    for predicted, human in pairs:
        if predicted == 1 and human == 1:
            tp += 1
        elif predicted == 1 and human == 0:
            fp += 1
        elif predicted == 0 and human == 0:
            tn += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = (
        _safe_div(2 * precision * recall, precision + recall)
        if not (math.isnan(precision) or math.isnan(recall))
        else float("nan")
    )
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": _safe_div(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": _safe_div(fp, fp + tn),
        "fnr": _safe_div(fn, fn + tp),
    }


def _print_metrics_block(label: str, m: dict[str, float]) -> None:
    print(f"--- {label} ---")
    print(f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")
    print(f"Accuracy:  {m['accuracy']:.3f}")
    print(f"Precision: {m['precision']:.3f}")
    print(f"Recall:    {m['recall']:.3f}")
    print(f"F1:        {m['f1']:.3f}")
    print(f"FPR:       {m['fpr']:.3f}")
    print(f"FNR:       {m['fnr']:.3f}")
    print()


def main() -> None:
    input_path = _locate_validation_csv()
    print(f"Located existing validation CSV: {input_path}")

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    missing = REQUIRED_COLUMNS - set(fieldnames or [])
    if missing:
        raise ValueError(f"{input_path} is missing expected columns: {sorted(missing)}")

    print(f"Loaded {len(rows)} rows.\n")
    print("Loading the UPDATED FaithfulnessVerifier (local model + embedder only — "
          "no generator, no Groq/OpenAI, no network calls of any kind)...")
    verifier = FaithfulnessVerifier()
    print("Verifier loaded. Re-scoring saved (raw_answer, context_used) pairs...\n")

    out_rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    old_pairs: list[tuple[int, int]] = []
    new_pairs: list[tuple[int, int]] = []

    n_success = 0
    n_api_error = 0
    n_other_status = 0
    n_changed = 0

    for row in rows:
        row = dict(row)
        status = row.get("status", "")

        if status == "api_error":
            n_api_error += 1
            out_rows.append(row)
            continue

        if status != "success":
            # Any other/unexpected status — preserve unchanged; this script
            # only ever re-scores rows that actually succeeded.
            n_other_status += 1
            out_rows.append(row)
            continue

        n_success += 1
        old_decision = row.get("verifier_decision", "")
        old_score = row.get("faithfulness_score", "")
        old_numeric = row.get("numeric_consistency", "")

        raw_answer = row.get("raw_answer", "") or ""
        context = row.get("context_used", "") or ""

        result = verifier.predict(raw_answer, context)
        features = result["features"]

        row["faithfulness_score"] = f"{result['score']:.6f}"
        row["semantic_similarity"] = features["semantic_similarity"]
        row["keyword_jaccard"] = features["keyword_jaccard"]
        row["numeric_consistency"] = features["numeric_consistency"]
        row["verifier_decision"] = result["status"]

        out_rows.append(row)

        human_raw = row.get("human_faithfulness", "")
        if _is_binary_label(human_raw):
            human = int(human_raw.strip())
            old_pairs.append((1 if old_decision == "faithful" else 0, human))
            new_pairs.append((1 if result["status"] == "faithful" else 0, human))

        if result["status"] != old_decision:
            n_changed += 1
            changes.append({
                "id": row.get("id", ""),
                "old_decision": old_decision,
                "new_decision": result["status"],
                "old_score": old_score,
                "new_score": row["faithfulness_score"],
                "old_numeric": old_numeric,
                "new_numeric": row["numeric_consistency"],
            })

    FAITHFULNESS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote re-verified results to: {OUTPUT_CSV}\n")

    # ── Per-row change report ────────────────────────────────────────────
    print("=" * 70)
    print(f"VERIFIER DECISION CHANGES ({len(changes)} of {n_success} successful rows)")
    print("=" * 70)
    if not changes:
        print("No verifier decisions changed.")
    else:
        for c in changes:
            print(f"[{c['id']}]")
            print(f"  decision:            {c['old_decision']} -> {c['new_decision']}")
            print(f"  faithfulness_score:  {c['old_score']} -> {c['new_score']}")
            print(f"  numeric_consistency: {c['old_numeric']} -> {c['new_numeric']}")
            print()

    # ── Counts ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("COUNTS")
    print("=" * 70)
    print(f"Successful rows (re-verified):        {n_success}")
    print(f"API-error rows (untouched):            {n_api_error}")
    if n_other_status:
        print(f"Other-status rows (untouched):         {n_other_status}")
    print(f"Human-faithfulness labels available:   {len(new_pairs)} (of {n_success} successful rows)")
    print(f"Verifier decisions changed:            {n_changed}")
    print()

    # ── Faithfulness metrics: old vs new ────────────────────────────────
    print("=" * 70)
    print("FAITHFULNESS METRICS — OLD vs NEW (against existing human_faithfulness labels)")
    print("=" * 70)
    print("predicted faithful = verifier_decision == 'faithful'; uncertain and")
    print("unfaithful both count as predicted non-faithful.\n")

    if not new_pairs:
        print("No labeled successful rows — nothing to compute metrics from.")
    else:
        old_m = _binary_metrics(old_pairs)
        new_m = _binary_metrics(new_pairs)
        _print_metrics_block("OLD (verifier_decision as originally saved)", old_m)
        _print_metrics_block("NEW (re-verified with the updated FaithfulnessVerifier)", new_m)

        print("--- Old vs New Confusion Matrix ---")
        print(f"{'':22}{'human=1 (TP/FN)':>18}{'human=0 (FP/TN)':>18}")
        print(f"{'OLD predicted=1':22}{old_m['tp']:>18}{old_m['fp']:>18}")
        print(f"{'OLD predicted=0':22}{old_m['fn']:>18}{old_m['tn']:>18}")
        print(f"{'NEW predicted=1':22}{new_m['tp']:>18}{new_m['fp']:>18}")
        print(f"{'NEW predicted=0':22}{new_m['fn']:>18}{new_m['tn']:>18}")
        print()
        print(f"Old F1: {old_m['f1']:.3f}   New F1: {new_m['f1']:.3f}   (delta: {new_m['f1'] - old_m['f1']:+.3f})")
        print(f"Verifier decisions changed (successful rows): {n_changed}")


if __name__ == "__main__":
    main()
