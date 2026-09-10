"""
evaluate_adversarial_faithfulness.py
--------------------------------------
STAGE 2 + STAGE 3 of the adversarial faithfulness validation.

STAGE 2: runs the EXISTING, unmodified FaithfulnessVerifier locally against
data/faithfulness/adversarial_faithfulness.csv (built by
create_adversarial_faithfulness.py). No generator, no Groq/OpenAI, no
network calls of any kind — only the local embedder + the already-trained
classifier score each (answer, context_used) pair. The verifier itself is
never touched here.

STAGE 3: analyzes the results — confusion matrix, overall and per-category
metrics, every false positive/negative with its mutation, a feature-level
diagnosis of which signal seems responsible for each error, a comparison
against the real_validation findings, and a technical recommendation for
what to do next (this script does NOT retrain or change thresholds itself
— it only recommends).

Output: data/faithfulness/adversarial_faithfulness_results.csv
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
INPUT_CSV = FAITHFULNESS_DIR / "adversarial_faithfulness.csv"
OUTPUT_CSV = FAITHFULNESS_DIR / "adversarial_faithfulness_results.csv"

OUTPUT_FIELDNAMES = [
    "id", "parent_id", "category", "question", "answer",
    "human_faithfulness", "mutation_description",
    "verifier_decision", "faithfulness_score",
    "semantic_similarity", "keyword_jaccard", "numeric_consistency",
]

FEATURES = ("semantic_similarity", "keyword_jaccard", "numeric_consistency")

# Real-validation findings this run is compared against in Stage 3 (see
# data/faithfulness/real_validation_reverified.csv and its --metrics output).
REAL_VALIDATION_CONTEXT = {
    "successful_rows": 40,
    "human_faithfulness_positive": 40,
    "human_faithfulness_negative": 0,   # <- the real gap this adversarial set fills
    "old_recall": 18 / 40,               # pre-bugfix (E02/N04 unfixed)
    "new_recall": 20 / 40,               # post-bugfix (E02/N04 fixed)
    "fixed_rows": ["E02", "N04"],
}


# ── Shared metric helpers ──────────────────────────────────────────────────
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


# ── Stage 2: run the verifier ───────────────────────────────────────────────
def run_evaluation() -> list[dict[str, Any]]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV} not found. Run src/create_adversarial_faithfulness.py first."
        )
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} adversarial rows from {INPUT_CSV}\n")

    print("Loading the EXISTING FaithfulnessVerifier (local model + embedder only — "
          "no generator, no Groq/OpenAI, no network calls)...")
    verifier = FaithfulnessVerifier()
    print("Verifier loaded. Scoring (answer, context_used) pairs...\n")

    results: list[dict[str, Any]] = []
    for row in rows:
        answer = row["answer"]
        context = row["context_used"]
        prediction = verifier.predict(answer, context)
        features = prediction["features"]
        results.append({
            "id": row["id"],
            "parent_id": row["parent_id"],
            "category": row["category"],
            "question": row["question"],
            "answer": answer,
            "human_faithfulness": int(row["human_faithfulness"]),
            "mutation_description": row["mutation_description"],
            "verifier_decision": prediction["status"],
            "faithfulness_score": prediction["score"],
            "semantic_similarity": features["semantic_similarity"],
            "keyword_jaccard": features["keyword_jaccard"],
            "numeric_consistency": features["numeric_consistency"],
        })

    FAITHFULNESS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote verifier results to: {OUTPUT_CSV}\n")

    return results


# ── Stage 3: analysis ────────────────────────────────────────────────────────
def _predicted(row: dict[str, Any]) -> int:
    return 1 if row["verifier_decision"] == "faithful" else 0


def _diagnose_feature(row: dict[str, Any]) -> str:
    """
    Heuristic per-row diagnosis of which of the 3 features looks most
    responsible for this misclassification:
      - FN (should be faithful, wasn't): the feature with the LOWEST value
        is the "weakest link" dragging the composite score down.
      - FP (should be unfaithful, wasn't): the feature that FAILED to drop
        despite the mutation — i.e. whichever of the three still looks
        "healthy" (high) even though the answer is actually wrong. For
        numeric-type mutations (wrong_number/wrong_condition/wrong_fee)
        that's diagnostic of numeric_consistency specifically; for
        unsupported_claim it's diagnostic of semantic_similarity/
        keyword_jaccard, since no number changed at all.
    """
    human = row["human_faithfulness"]
    predicted = _predicted(row)
    vals = {feat: row[feat] for feat in FEATURES}

    if human == 1 and predicted == 0:  # false negative
        weakest = min(vals, key=lambda k: vals[k])
        return f"weakest feature: {weakest}={vals[weakest]:.3f}"

    if human == 0 and predicted == 1:  # false positive
        numeric_type = row["category"] in ("wrong_number", "wrong_condition", "wrong_fee")
        if numeric_type:
            return f"numeric_consistency did not drop (={vals['numeric_consistency']:.3f}); other features: semantic_similarity={vals['semantic_similarity']:.3f}, keyword_jaccard={vals['keyword_jaccard']:.3f}"
        else:
            return f"semantic_similarity/keyword_jaccard stayed high (sem={vals['semantic_similarity']:.3f}, jac={vals['keyword_jaccard']:.3f}); numeric_consistency={vals['numeric_consistency']:.3f} (mutation added no new number, so this feature can't see it)"

    return "correctly classified"


def analyze(results: list[dict[str, Any]]) -> None:
    all_pairs = [(_predicted(r), r["human_faithfulness"]) for r in results]

    print("=" * 70)
    print("STAGE 3 — OVERALL CONFUSION MATRIX & METRICS")
    print("=" * 70)
    overall = _binary_metrics(all_pairs)
    print("predicted faithful = verifier_decision == 'faithful'; uncertain and "
          "unfaithful both count as predicted non-faithful.\n")
    _print_metrics_block("Overall (all 50 adversarial rows)", overall)

    print("=" * 70)
    print("METRICS BY CATEGORY")
    print("=" * 70)
    categories = sorted({r["category"] for r in results})
    for cat in categories:
        cat_pairs = [(_predicted(r), r["human_faithfulness"]) for r in results if r["category"] == cat]
        m = _binary_metrics(cat_pairs)
        n = len(cat_pairs)
        print(f"--- {cat} (n={n}) ---")
        print(f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}   "
              f"Accuracy={m['accuracy']:.3f} Precision={m['precision']:.3f} "
              f"Recall={m['recall']:.3f} F1={m['f1']:.3f}")
        print()

    false_positives = [r for r in results if r["human_faithfulness"] == 0 and _predicted(r) == 1]
    false_negatives = [r for r in results if r["human_faithfulness"] == 1 and _predicted(r) == 0]

    print("=" * 70)
    print(f"FALSE POSITIVES ({len(false_positives)}) — verifier said faithful, but it was NOT")
    print("=" * 70)
    if not false_positives:
        print("None.")
    else:
        for r in false_positives:
            print(f"[{r['id']}] category={r['category']}  score={r['faithfulness_score']:.3f}  decision={r['verifier_decision']}")
            print(f"  answer: {r['answer'][:200]}")
            print(f"  mutation: {r['mutation_description']}")
            print(f"  feature diagnosis: {_diagnose_feature(r)}")
            print()

    print("=" * 70)
    print(f"FALSE NEGATIVES ({len(false_negatives)}) — verifier said non-faithful, but it WAS faithful")
    print("=" * 70)
    if not false_negatives:
        print("None.")
    else:
        for r in false_negatives:
            print(f"[{r['id']}] category={r['category']}  score={r['faithfulness_score']:.3f}  decision={r['verifier_decision']}")
            print(f"  answer: {r['answer'][:200]}")
            print(f"  mutation: {r['mutation_description'] or '(none — unmodified faithful example)'}")
            print(f"  feature diagnosis: {_diagnose_feature(r)}")
            print()

    print("=" * 70)
    print("FEATURE RESPONSIBILITY SUMMARY")
    print("=" * 70)
    fn_weakest_tally: dict[str, int] = {}
    for r in false_negatives:
        vals = {feat: r[feat] for feat in FEATURES}
        weakest = min(vals, key=lambda k: vals[k])
        fn_weakest_tally[weakest] = fn_weakest_tally.get(weakest, 0) + 1
    print(f"False negatives ({len(false_negatives)}): weakest feature tally: {fn_weakest_tally or '{}'}")

    fp_numeric_masked = sum(
        1 for r in false_positives
        if r["category"] in ("wrong_number", "wrong_condition", "wrong_fee") and r["numeric_consistency"] >= 0.5
    )
    fp_nonnumeric = sum(
        1 for r in false_positives if r["category"] in ("unsupported_claim",)
    )
    print(f"False positives from numeric-type mutations where numeric_consistency stayed >= 0.5: {fp_numeric_masked}")
    print(f"False positives from unsupported_claim mutations (numeric_consistency structurally blind): {fp_nonnumeric}")
    print()

    print("=" * 70)
    print("COMPARISON AGAINST REAL_VALIDATION FINDINGS")
    print("=" * 70)
    print(f"real_validation_reverified.csv: {REAL_VALIDATION_CONTEXT['successful_rows']} successful rows, "
          f"ALL {REAL_VALIDATION_CONTEXT['human_faithfulness_positive']} labeled human_faithfulness=1, "
          f"ZERO labeled human_faithfulness=0.")
    print(f"  -> that run could only ever measure recall/FNR (predicted-faithful vs. truly-faithful);")
    print(f"     precision/FPR were mathematically undefined (nan) there — no negative examples existed.")
    print(f"  -> E02, N04 (the two condition/fee and percent-equivalence bugs) were fixed: recall went "
          f"{REAL_VALIDATION_CONTEXT['old_recall']:.3f} -> {REAL_VALIDATION_CONTEXT['new_recall']:.3f} on that set.")
    print(f"  -> the other {REAL_VALIDATION_CONTEXT['successful_rows'] - 20} real false negatives (paraphrased/")
    print(f"     abstention answers with low keyword_jaccard) were NOT touched by those fixes.")
    print()
    print(f"This adversarial run is the FIRST time this project has measured precision/FPR at all — "
          f"it deliberately manufactures {sum(1 for r in results if r['human_faithfulness']==0)} real negative "
          f"examples across 5 distinct hallucination patterns, which the real_validation set structurally "
          f"could not provide. Its precision/FPR/TP/FP numbers above are the first real signal on whether "
          f"the gate actually catches a wrong answer, as opposed to only how often it needlessly blocks a "
          f"correct one.")
    print()

    _recommend(overall, results, false_positives, false_negatives)


def _recommend(overall: dict[str, float], results: list[dict[str, Any]],
                false_positives: list[dict[str, Any]], false_negatives: list[dict[str, Any]]) -> None:
    print("=" * 70)
    print("RECOMMENDATION FOR NEXT STEP (not applied — analysis only)")
    print("=" * 70)

    fp_count = len(false_positives)
    fn_count = len(false_negatives)
    fp_rate = overall["fpr"]
    fn_rate = overall["fnr"]

    print(f"Observed: FP={fp_count} (FPR={fp_rate if not math.isnan(fp_rate) else float('nan'):.3f}), "
          f"FN={fn_count} (FNR={fn_rate if not math.isnan(fn_rate) else float('nan'):.3f}) "
          f"on {len(results)} adversarial rows.\n")

    fp_unsupported = [r for r in false_positives if r["category"] == "unsupported_claim"]
    fp_numeric = [r for r in false_positives if r["category"] in ("wrong_number", "wrong_condition", "wrong_fee")]
    fp_mixed = [r for r in false_positives if r["category"] == "mixed_hallucination"]
    fp_wrong_fee = [r for r in false_positives if r["category"] == "wrong_fee"]
    fn_abstention = [r for r in false_negatives if r["category"] == "faithful_abstention"]
    fn_original = [r for r in false_negatives if r["category"] == "faithful_original"]
    n_unsupported = sum(1 for r in results if r["category"] == "unsupported_claim")
    n_numeric = sum(1 for r in results if r["category"] in ("wrong_number", "wrong_condition", "wrong_fee"))
    n_mixed = sum(1 for r in results if r["category"] == "mixed_hallucination")
    n_wrong_fee = sum(1 for r in results if r["category"] == "wrong_fee")
    n_abstention = sum(1 for r in results if r["category"] == "faithful_abstention")
    n_original = sum(1 for r in results if r["category"] == "faithful_original")

    def _rate_line(label: str, misses: list, n: int, good_note: str, bad_note: str) -> None:
        """Report a category's miss count, choosing the accurate note based
        on what actually happened — never assumes the outcome in advance."""
        k = len(misses)
        note = good_note if k == 0 else bad_note
        print(f"  - {label}: {k}/{n} — {note}")

    print("Basis for the recommendation (from the actual FP/FN breakdown above, not assumptions):")

    _rate_line(
        "wrong_fee false positives", fp_wrong_fee, n_wrong_fee,
        good_note="every wrong currency amount was caught, including the specific 50-65%-attendance "
                   "fee case (E02's mutated variant) that an earlier condition/fee logic bug used to miss.",
        bad_note="some wrong currency amounts slipped through undetected.",
    )
    _rate_line(
        "mixed_hallucination false positives", fp_mixed, n_mixed,
        good_note="every single wrong fact hidden among several correct ones was caught. Notable because "
                   f"this is the hardest case for whole-answer aggregate features by construction (sample "
                   f"is small, n={n_mixed}, so this doesn't prove full generalisation, but there's no "
                   f"observed problem here).",
        bad_note="a wrong fact hidden among several correct ones was missed at least once — the hardest "
                  "case for whole-answer aggregate features by construction.",
    )
    _rate_line(
        "unsupported_claim false positives", fp_unsupported, n_unsupported,
        good_note="every added fabricated claim was caught.",
        bad_note="a real, sizeable miss rate. An added fabricated claim with no new number is invisible to "
                  "numeric_consistency by design, and a short added sentence barely moves "
                  "semantic_similarity/keyword_jaccard on an otherwise-long, mostly-matching answer.",
    )
    numeric_only_fp = [r for r in fp_numeric if r not in fp_wrong_fee]
    _rate_line(
        "numeric-type (wrong_number/wrong_condition) false positives", numeric_only_fp, n_numeric - n_wrong_fee,
        good_note="every mutated number/condition was caught.",
        bad_note="numeric_consistency stayed at 1.000 (did NOT drop) despite the wrong value in these rows, "
                  "because the mutated number's own token still happened to appear ELSEWHERE verbatim in "
                  "the same context, or the mutation targeted a count the feature doesn't treat as fee-like. "
                  "This is a feature-extraction gap for this specific sub-pattern.",
    )
    _rate_line(
        "false negatives, faithful_abstention", fn_abstention, n_abstention,
        good_note="every genuine abstention was correctly recognised as faithful — the blind spot found in "
                   "the previous (3-feature) adversarial run and the abstention-only retrain experiment "
                   "appears resolved on this set.",
        bad_note="some genuine abstentions were still misclassified as uncertain/unfaithful — the known "
                  "blind spot persists at least partially.",
    )
    _rate_line(
        "false negatives, faithful_original", fn_original, n_original,
        good_note="every genuinely faithful, non-abstention answer was correctly recognised.",
        bad_note="the same low-keyword_jaccard/paraphrase pattern already found in real_validation persists "
                  "on fresh adversarial-set rows.",
    )
    print()

    # ── Verdict: derived from what actually happened, not a fixed script ──
    verdict_parts: list[str] = []

    if fn_abstention or fn_original:
        verdict_parts.append(
            f"A (retrain) is still indicated — {len(fn_abstention) + len(fn_original)} genuinely faithful "
            f"row(s) are still misclassified. Add more/better positive examples of that specific pattern to "
            f"the synthetic training data and retrain."
        )
    else:
        verdict_parts.append(
            "A (retrain) is NOT indicated right now on recall grounds — FN=0 across the whole 50-row set, "
            "so there is no remaining known false-negative pattern to target with more training data."
        )

    if fp_unsupported or numeric_only_fp:
        verdict_parts.append(
            f"C (features) still has room: {len(fp_unsupported)}/{n_unsupported} unsupported_claim and "
            f"{len(numeric_only_fp)}/{n_numeric - n_wrong_fee} numeric-type false positives remain. These "
            f"are the cases where claim_support_ratio's per-claim check didn't push the score low enough — "
            f"either the fabricated claim's embedding was too similar to a genuine nearby context sentence "
            f"(a paraphrase-vs-hallucination ambiguity claim-level similarity alone can't always resolve), "
            f"or numeric_consistency itself didn't extract the contradiction (see the FP list above). "
            f"Worth inspecting these specific rows before adding a 3rd claim-level signal."
        )
    else:
        verdict_parts.append(
            "C (features) — no further feature gap observed in this run's false positives (there were none)."
        )

    if overall["fpr"] > 0 and not math.isnan(overall["fpr"]):
        verdict_parts.append(
            f"B (tune thresholds) is explicitly NOT recommended here despite FPR={overall['fpr']:.3f} > 0 — "
            f"per this task's instructions, and because the right fix for the remaining false positives above "
            f"is targeted (feature/data), not a blanket threshold shift that would trade this precision loss "
            f"for recall elsewhere."
        )

    print("VERDICT (derived from the counts above, not a preset script):")
    for part in verdict_parts:
        print(f"  - {part}")


def main() -> None:
    results = run_evaluation()
    analyze(results)


if __name__ == "__main__":
    main()
