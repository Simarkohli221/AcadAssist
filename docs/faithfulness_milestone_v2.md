# Claim-Level Faithfulness Verifier v2

**Status: frozen milestone.** This document records the state of the faithfulness
verifier as of this checkpoint. No further verifier changes (features, thresholds,
retraining) were made after this point without a new, separate experiment.

## Motivation

The original verifier (`semantic_similarity`, `keyword_jaccard`, `numeric_consistency`)
was evaluated for the first time against a hand-built adversarial set with real
negative examples (`data/faithfulness/adversarial_faithfulness.csv` — the project's
40-row real-world validation set had zero negative labels, so it could never measure
precision/FPR at all). That first adversarial run exposed three specific problems, all
traceable to the same root cause:

- **Faithful abstentions were never recognized.** 7/7 genuine refusals ("I could not
  find this information...") paired with irrelevant retrieved context were
  misclassified as uncertain/unfaithful. The synthetic training data
  (`create_faithfulness_dataset.py`) had no example of this pattern at all.
- **Unsupported claims tacked onto otherwise-correct answers went undetected.** 3/6
  adversarial examples that added one fabricated sentence to a real, correct answer
  were missed.
- **All three original features are whole-answer aggregates.** A single wrong or
  fabricated sentence, hidden inside a longer answer that is otherwise entirely
  correct, barely moves a whole-answer cosine similarity or a whole-answer keyword
  Jaccard score. Aggregate features cannot see a problem localized to one sentence
  out of several.

A first attempt to fix the abstention blind spot purely with more training data
(V1.1 below) tested whether the problem was a data-quantity issue. It was not — it
degraded results slightly. That controlled, negative result motivated V2's structural
change: a feature that checks the answer's claims individually rather than as one
aggregate block.

## Versions evaluated

All three versions were evaluated on the **exact same held-out 50-row adversarial set**
(`data/faithfulness/adversarial_faithfulness.csv`, 20 human-faithful / 30
human-unfaithful, spanning 7 categories). That set was never used for training or for
calibrating any feature threshold.

### V1 — Original 3-feature verifier
`models/faithfulness_lr_before_abstention_retrain.joblib`
Trained on the original synthetic dataset (no abstention examples).

| Accuracy | Precision | Recall | F1 | FPR | FNR |
|---|---|---|---|---|---|
| 0.700 | 0.647 | 0.550 | 0.595 | 0.200 | 0.450 |

### V1.1 — Abstention-augmented training (same 3 features)
`models/faithfulness_lr_3feature_with_abstention_data.joblib`
159 synthetic "faithful abstention + irrelevant context" examples added to the training
data, then retrained on the **same 3 features**. Isolated the question "does more
training data alone fix this?"

| Accuracy | Precision | Recall | F1 | FPR | FNR |
|---|---|---|---|---|---|
| 0.660 | 0.579 | 0.550 | 0.564 | 0.267 | 0.450 |

**Result: no improvement — slightly worse.** Faithful-abstention detection remained
0/7 (scores moved up marginally but stayed below `THRESHOLD_UNCERTAIN`/`THRESHOLD_FAITHFUL`),
and 2 previously-correct hallucination catches flipped to false positives. This
established that the problem was not data quantity but feature expressiveness — the
3 aggregate features cannot separate a short, generic abstention from a short,
generic hallucination, no matter how the decision boundary is redrawn through the
same 3-dimensional space.

### V2 — Claim-level support (5 features)
`models/faithfulness_lr.joblib` (current/live) — also preserved as
`models/faithfulness_lr_v2_claim_level.joblib`
Added `claim_support_ratio` and `minimum_claim_support`, trained on the abstention-
augmented dataset with all 5 features.

| Accuracy | Precision | Recall | F1 | FPR | FNR |
|---|---|---|---|---|---|
| 0.840 | 0.714 | 1.000 | 0.833 | 0.267 | 0.000 |

Confusion matrix: **TP=20 FP=8 FN=0 TN=22** (V1 was TP=11 FP=6 FN=9 TN=24).

## Claim-level features

### `claim_support_ratio` and `minimum_claim_support`
Two features, not one, because they catch different failure shapes: a ratio of 0.8
(4 of 5 claims supported) looks fine on average but hides the one fabricated claim;
`minimum_claim_support` flags that claim directly. Conversely `minimum` alone can't
tell "1 of 5 claims wrong" from "the only claim is wrong" the way the ratio can.
Reporting both lets the classifier weigh each independently rather than forcing one
lossy combination.

### How claims are extracted
`_split_claim_units()` (in `src/faithfulness.py`) splits text on markdown
bullet/numbered-list lines and sentence boundaries — deliberately simple, regex-based,
no NLP library, reused identically for both the answer (to get claims) and the context
(to get the comparison pool), so both sides are segmented the same way. `_content_units()`
then strips markdown emphasis/citation markup, drops citation-only lines
("**Source:** ..."), and discards fragments under 4 words (`_MIN_CLAIM_WORDS`,
`_MIN_CONTEXT_UNIT_WORDS`) — usually stray headers or citation remnants, not claims.

### How support similarity is calculated
For each substantive claim, the existing shared `Embedder` encodes all claims and all
context units in one batched call; each claim's score is the maximum cosine similarity
against any context unit (vectors are already L2-normalized, so this is a dot product).
A claim counts as "supported" if its best score is ≥ `_CLAIM_SUPPORT_THRESHOLD` (0.50).
`claim_support_ratio` = supported / total claims; `minimum_claim_support` = the single
lowest best-score among the claims.

### Abstention special handling
`_is_abstention_answer()` pattern-matches (not a single literal string) against the
family of safe-refusal phrasings — `generator.NO_ANSWER_RESPONSE` plus the paraphrases
`create_faithfulness_dataset.py` trains on — gated to answers ≤40 words so a much
longer substantive answer that happens to share a similar phrase is still evaluated
normally. A recognized abstention **skips claim decomposition entirely** and returns
`(1.0, 1.0)` directly: a refusal asserts nothing, so there is nothing to individually
verify, and scoring it against context via ordinary similarity would otherwise produce
a near-zero score purely because short generic refusal text shares no vocabulary with
any context, faithful or not.

### Calibration methodology
`_CLAIM_SUPPORT_THRESHOLD = 0.50` was calibrated against the **training corpus**
(`data/faithfulness/dataset.csv`), never the held-out adversarial set. 80 known-supported
sentence pairs (`pair_type ∈ {single_sentence, two_sentence}`, label=1) and 80
known-unsupported pairs (`pair_type ∈ {context_mismatch, off_topic}`, label=0) were
scored with this exact code path:

| | median | mean | notable percentile |
|---|---|---|---|
| Known-supported | 0.880 | 0.855 | p10 = 0.658 |
| Known-unsupported | 0.246 | 0.249 | p95 = 0.459 |

The two distributions are widely and cleanly separated (positive p10 is still above
negative p95). 0.50 sits in the middle of that gap: 100% of the sampled supported
claims score above it, and only 3% (2/80) of unsupported ones incorrectly do. It was
not chosen to force a particular outcome on any specific example.

## Category-level results (V1 → V2, same 50-row set)

| Category | V1 | V2 |
|---|---|---|
| faithful_abstention | 0/7 | **7/7** |
| faithful_original | 11/13 | **13/13** |
| wrong_number | 4/6 | 4/6 |
| wrong_condition | 5/6 | 4/6 |
| wrong_fee | 6/6 | **6/6** |
| unsupported_claim | 3/6 | 3/6 |
| mixed_hallucination | 6/6 | 5/6 |

11 of 50 rows changed decision between V1 and V2: 9 were genuine fixes (all 7
abstentions plus 2 previously-missed faithful_original rows), and 2 were new false
positives (`WC-N02`, `MH-P06`) — both traced to the same cause: the mutated answer is a
near-duplicate of a true context sentence with one element swapped (an "8-week" course
duration vs. the true "12-week", and "formal sneakers" vs. the true "formal shoes and a
belt"), so their embeddings stay highly similar to genuine context despite being
factually wrong. Cosine similarity captures topical/lexical closeness, not
fine-grained factual precision on a single substituted word or number — this is a
structural limit of the current approach, not a bug.

## Remaining limitations

Documented honestly, not glossed over:

- **FPR = 0.267** — worse than V1's 0.200. 8 of 30 genuinely unfaithful adversarial
  answers are still classified as faithful.
- **Near-duplicate factual substitutions still pass** (`WC-N02`, `MH-P06` — see above).
  Embedding similarity alone cannot always distinguish "same sentence, one fact
  changed" from "the same true fact."
- **Unsupported claims still pass at the same rate as V1** — 3/6 misses, unchanged.
  The added fabricated sentences in these specific rows are similar enough to nearby
  genuine content that they still clear the 0.50 threshold.
- **The 50-row adversarial set is synthetic** — every negative example is a
  hand-specified, deterministic mutation of a real answer, not a naturally occurring
  LLM hallucination. It is a useful, real, and previously-nonexistent source of
  negative-class signal, but it is not a substitute for labeled real-world hallucinations.
- **50 examples is a small evaluation set.** Category-level counts (n=6 or n=7 per
  category) mean a single flipped row moves a category's rate by ~15-17 percentage
  points. The results above are directionally meaningful, not statistically precise.

## Important conclusion

The claim-level model (V2) substantially improved held-out performance relative to the
original verifier on this evaluation set — accuracy 0.700→0.840, F1 0.595→0.833,
recall 0.550→1.000 — and completely resolved the faithful-abstention failure that
motivated this work (0/7→7/7), which a pure training-data intervention (V1.1) had
failed to fix. This came at a real, measured cost: FPR rose from 0.200 to 0.267 (2
additional false positives, both a distinct and specific "near-duplicate substitution"
failure mode).

**The verifier is not perfect.** unsupported_claim detection is unchanged at 3/6, and
the two new false positives point at a specific, identified limitation of pure
cosine-similarity claim checking. This is a real, measured improvement on a small,
synthetic, held-out benchmark — not a claim that the verifier reliably catches all
hallucinations in production.
