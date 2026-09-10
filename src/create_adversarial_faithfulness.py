"""
create_adversarial_faithfulness.py
------------------------------------
STAGE 1 of the adversarial faithfulness validation.

Builds a hand-curated adversarial evaluation set from the EXISTING,
already-reviewed real-world validation run
(data/faithfulness/real_validation_reverified.csv, status == "success"
rows only), so it has real negative (human_faithfulness=0) examples — the
40-row real_validation set has none at all, which is why it could never
measure the faithfulness gate's precision/false-positive rate.

Every adversarial row is derived from a real parent row:
  - faithful_original / faithful_abstention: the parent's own raw_answer,
    completely unmodified, label=1.
  - the five negative categories: the parent's raw_answer with EXACTLY ONE
    targeted, hand-specified substring mutation (a wrong number/fee/
    condition, or one added/altered unsupported claim) applied to it.
    context_used is NEVER touched — only answer changes.

No LLM is used to generate or rewrite anything here: every mutation is a
literal, deterministic string substitution/append specified below, applied
via a whitespace/hyphen-tolerant matcher (the real PDF-extracted text uses
narrow no-break spaces and non-breaking hyphens around numbers, e.g.
"75 % attendance" — the matcher treats a plain " " / "-" written in a
spec as tolerant of those variants, but still requires an EXACT and UNIQUE
match, so a typo or drift in the source data fails loudly instead of
silently producing an unmutated "adversarial" row).
"""
from __future__ import annotations

import csv
import random
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

FAITHFULNESS_DIR = PROJECT_ROOT / "data" / "faithfulness"
INPUT_CSV = FAITHFULNESS_DIR / "real_validation_reverified.csv"
OUTPUT_CSV = FAITHFULNESS_DIR / "adversarial_faithfulness.csv"

SEED = 42

FIELDNAMES = [
    "id", "parent_id", "category", "question", "answer", "context_used",
    "human_faithfulness", "mutation_description", "sources_json",
]


# ── Whitespace/hyphen-tolerant exact-substring mutation ──────────────────
def _flexible_sub(text: str, old: str, new: str, parent_id: str) -> str:
    """
    Replace `old` with `new` in `text`. `old` is written with plain spaces
    and hyphens for readability; matching tolerates the narrow no-break
    space ( ), regular no-break space ( ), non-breaking hyphen
    (‑), and en dash (–) the real extracted text actually uses in
    those positions. Requires EXACTLY ONE match — raises loudly otherwise,
    so a mismatched anchor is caught at generation time, not silently
    produces a no-op "mutation".
    """
    escaped = re.escape(old)
    escaped = escaped.replace(r"\ ", r"[   ]")
    escaped = escaped.replace(r"\-", r"[\-‑–]")
    pattern = re.compile(escaped)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(
            f"[{parent_id}] expected exactly 1 match for mutation anchor {old!r}, "
            f"found {len(matches)} in parent raw_answer"
        )
    return pattern.sub(lambda m: new.replace("\\", "\\\\"), text, count=1)


# ── Category 1 & 2: unmodified, label = 1 ─────────────────────────────────
POSITIVE_ORIGINAL_PARENTS = [
    "E01", "E03", "E05", "N01", "N03", "P02", "P04",
    "S02", "S04", "T01", "T02", "A01", "A03",
]  # 13 — faithful_original

POSITIVE_ABSTENTION_PARENTS = [
    "E04", "N06", "A04", "X01", "X02", "X03", "M02",
]  # 7 — faithful_abstention (known verifier blind spot: faithful refusal + irrelevant context)


# ── Categories 3-7: exactly one targeted mutation per row, label = 0 ─────
# mode "replace": (old, new) applied via _flexible_sub to the parent's raw_answer.
# mode "append":  a fabricated sentence appended to the parent's raw_answer.
NEGATIVE_SPECS: list[dict[str, Any]] = [
    # -- wrong_number (6): an important standalone figure changed --------
    {"parent_id": "N01", "category": "wrong_number", "mode": "replace",
     "old": "86 % or higher", "new": "96 % or higher",
     "desc": "Changed the NPTEL AA-grade cutoff from 86% to 96%."},
    {"parent_id": "N03", "category": "wrong_number", "mode": "replace",
     "old": "within 10 days", "new": "within 20 days",
     "desc": "Changed the NPTEL re-registration deadline from 10 days to 20 days."},
    {"parent_id": "S04", "category": "wrong_number", "mode": "replace",
     "old": "12 credits", "new": "16 credits",
     "desc": "Changed the Semester IV Program Core Courses credit count from 12 to 16."},
    {"parent_id": "T05", "category": "wrong_number", "mode": "replace",
     "old": "five", "new": "three",
     "desc": "Changed the company-debarment count for 6 demerit points from five to three."},
    {"parent_id": "P04", "category": "wrong_number", "mode": "replace",
     "old": "8 demerit points", "new": "10 demerit points",
     "desc": "Changed the demerit-point threshold for placement disqualification from 8 to 10."},
    {"parent_id": "E07", "category": "wrong_number", "mode": "replace",
     "old": "at least 160 credits", "new": "at least 180 credits",
     "desc": "Changed the minimum B.Tech credit requirement from 160 to 180."},

    # -- wrong_condition (6): an eligibility/requirement threshold changed --
    {"parent_id": "E01", "category": "wrong_condition", "mode": "replace",
     "old": "at least 75 % attendance", "new": "at least 60 % attendance",
     "desc": "Changed the minimum attendance eligibility requirement for the ESE from 75% to 60%."},
    {"parent_id": "T01", "category": "wrong_condition", "mode": "replace",
     "old": "65 % - 75 % attendance", "new": "55 % - 65 % attendance",
     "desc": "Changed the mandatory-training attendance range from 65-75% to 55-65%."},
    {"parent_id": "S01", "category": "wrong_condition", "mode": "replace",
     "old": "**8.25**.", "new": "**7.00**.",
     "desc": "Changed the Honors by Research minimum CGPA requirement from 8.25 to 7.00 "
             "(the quoted source citation later in the answer is left unchanged)."},
    {"parent_id": "N02", "category": "wrong_condition", "mode": "replace",
     "old": "A 12-week SWAYAM-NPTEL course", "new": "An 8-week SWAYAM-NPTEL course",
     "desc": "Changed the referenced course duration from 12-week to 8-week while keeping the "
             "3-credit figure, creating a mismatched condition-to-value pairing "
             "(8-week courses are worth 2 credits per the syllabus, not 3)."},
    {"parent_id": "N04", "category": "wrong_condition", "mode": "replace",
     "old": "**40 %**", "new": "**50 %**",
     "desc": "Changed the NPTEL assignment-score passing threshold from 40% to 50%."},
    {"parent_id": "M01", "category": "wrong_condition", "mode": "replace",
     "old": "**75 % attendance**", "new": "**60 % attendance**",
     "desc": "Changed the NPTEL re-registration summer-term attendance requirement from 75% to "
             "60% (the re-examination passing-score claim is left unchanged)."},

    # -- wrong_fee (6): a currency amount changed to another plausible fee --
    {"parent_id": "E02", "category": "wrong_fee", "mode": "replace",
     "old": "**₹ 5,000**", "new": "**₹ 3,000**",
     "desc": "Changed the 50-65% attendance re-examination fee from Rs.5,000 to Rs.3,000 "
             "(the Rs.500 special-exam caveat is left unchanged)."},
    {"parent_id": "E03", "category": "wrong_fee", "mode": "replace",
     "old": "**₹ 8,000**", "new": "**₹ 6,000**",
     "desc": "Changed the below-50% special examination fee from Rs.8,000 to Rs.6,000."},
    {"parent_id": "E05", "category": "wrong_fee", "mode": "replace",
     "old": "Rs. 1,000 per course", "new": "Rs. 2,500 per course",
     "desc": "Changed the grade-improvement examination fee from Rs.1,000 to Rs.2,500."},
    {"parent_id": "E06", "category": "wrong_fee", "mode": "replace",
     "old": "fined Rs 1,000 per paper", "new": "fined Rs 3,000 per paper",
     "desc": "Changed the malpractice fine from Rs.1,000 to Rs.3,000 per paper."},
    {"parent_id": "P05", "category": "wrong_fee", "mode": "replace",
     "old": "**at least ₹3 lakh higher**", "new": "**at least ₹5 lakh higher**",
     "desc": "Changed the minimum CTC increase required to jump placement categories from "
             "Rs.3 lakh to Rs.5 lakh (the quoted source citations are left unchanged)."},
    {"parent_id": "M04", "category": "wrong_fee", "mode": "replace",
     "old": "Rs. 8 000 per course", "new": "Rs. 4 000 per course",
     "desc": "Changed the malpractice special-examination fee from Rs.8,000 to Rs.4,000 "
             "(the regular re-examination fee of Rs.500 is left unchanged)."},

    # -- unsupported_claim (6): original kept, one new fabricated claim added --
    {"parent_id": "P01", "category": "unsupported_claim", "mode": "append",
     "suffix": " Additionally, students placed in the Elite category receive a signing "
               "bonus of Rs 50,000, as per the policy.",
     "desc": "Added an unsupported claim about a Rs.50,000 Elite-category signing bonus, "
             "which is not stated anywhere in the placement policy context."},
    {"parent_id": "S02", "category": "unsupported_claim", "mode": "append",
     "suffix": " This course carries 4 credits and is taught by the Computer Networks division.",
     "desc": "Added unsupported claims about the course's credit count and teaching division, "
             "neither of which appear in the retrieved context."},
    {"parent_id": "T02", "category": "unsupported_claim", "mode": "append",
     "suffix": " The student is also barred from appearing in that semester's practical examinations.",
     "desc": "Added an unsupported claim about a practical-exam ban that is not stated in the "
             "malpractice/punishment context."},
    {"parent_id": "A02", "category": "unsupported_claim", "mode": "append",
     "suffix": " The E-Cell committee currently has 25 active student members.",
     "desc": "Added an unsupported membership-count claim not present in the annual activity "
             "report context."},
    {"parent_id": "N03", "category": "unsupported_claim", "mode": "append",
     "suffix": " This rule was introduced starting the 2024-25 academic year.",
     "desc": "Added an unsupported claim about when the rule was introduced, which the context "
             "does not state."},
    {"parent_id": "P04", "category": "unsupported_claim", "mode": "append",
     "suffix": " Such students also forfeit any Pre-Placement Offer (PPO) they may have already received.",
     "desc": "Added an unsupported claim about PPO forfeiture that is not stated in the "
             "demerit-consequence context."},

    # -- mixed_hallucination (6): several correct claims, one corrupted -----
    {"parent_id": "M04", "category": "mixed_hallucination", "mode": "replace",
     "old": "Rs. 500 per course", "new": "Rs. 1,500 per course",
     "desc": "Kept the correct malpractice fee (Rs.8,000) but changed the regular "
             "re-examination fee from Rs.500 to Rs.1,500."},
    {"parent_id": "M01", "category": "mixed_hallucination", "mode": "replace",
     "old": "**at least 40 marks out of 100**", "new": "**at least 60 marks out of 100**",
     "desc": "Kept the correct 75% attendance requirement but changed the re-examination "
             "passing score from 40 marks to 60 marks out of 100."},
    {"parent_id": "P01", "category": "mixed_hallucination", "mode": "replace",
     "old": "6 LPA", "new": "4 LPA",
     "desc": "Kept the correct Elite (>18 LPA) and Super Dream (>9 LPA) thresholds but changed "
             "the Dream category threshold from >6 LPA to >4 LPA."},
    {"parent_id": "P06", "category": "mixed_hallucination", "mode": "replace",
     "old": "**Formal shoes and a belt**", "new": "**Formal sneakers**",
     "desc": "Kept the correct shirt and trousers requirements but changed 'formal shoes and a "
             "belt' to 'formal sneakers', contradicting the dress-code context."},
    {"parent_id": "E07", "category": "mixed_hallucination", "mode": "replace",
     "old": "CGPA of 4.00 or higher", "new": "CGPA of 5.00 or higher",
     "desc": "Kept the correct 160-credit requirement but changed the minimum CGPA for degree "
             "award from 4.00 to 5.00."},
    {"parent_id": "E06", "category": "mixed_hallucination", "mode": "replace",
     "old": "Rs 1,000 per paper", "new": "Rs 5,000 per paper",
     "desc": "Kept the correct FF-grade penalty but changed the malpractice fine from "
             "Rs.1,000 to Rs.5,000 per paper."},
]


def _load_success_rows() -> dict[str, dict[str, str]]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV} not found. This script builds adversarial examples from an "
            f"EXISTING re-verified validation run — run reverify_faithfulness.py first."
        )
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        rows = {r["id"]: r for r in csv.DictReader(f) if r.get("status") == "success"}
    return rows


def _build_rows(parents: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for pid in POSITIVE_ORIGINAL_PARENTS:
        parent = parents[pid]
        out.append({
            "id": f"FO-{pid}",
            "parent_id": pid,
            "category": "faithful_original",
            "question": parent["question"],
            "answer": parent["raw_answer"],
            "context_used": parent["context_used"],
            "human_faithfulness": 1,
            "mutation_description": "",
            "sources_json": parent.get("sources_json", ""),
        })

    for pid in POSITIVE_ABSTENTION_PARENTS:
        parent = parents[pid]
        out.append({
            "id": f"FA-{pid}",
            "parent_id": pid,
            "category": "faithful_abstention",
            "question": parent["question"],
            "answer": parent["raw_answer"],
            "context_used": parent["context_used"],
            "human_faithfulness": 1,
            "mutation_description": "",
            "sources_json": parent.get("sources_json", ""),
        })

    category_prefix = {
        "wrong_number": "WN", "wrong_condition": "WC", "wrong_fee": "WF",
        "unsupported_claim": "UC", "mixed_hallucination": "MH",
    }
    for spec in NEGATIVE_SPECS:
        pid = spec["parent_id"]
        parent = parents[pid]
        original_answer = parent["raw_answer"]

        if spec["mode"] == "replace":
            mutated = _flexible_sub(original_answer, spec["old"], spec["new"], pid)
        elif spec["mode"] == "append":
            mutated = original_answer.rstrip() + spec["suffix"]
        else:
            raise ValueError(f"unknown mutation mode: {spec['mode']!r}")

        if mutated == original_answer:
            raise ValueError(f"[{pid}/{spec['category']}] mutation produced no change")

        prefix = category_prefix[spec["category"]]
        out.append({
            "id": f"{prefix}-{pid}",
            "parent_id": pid,
            "category": spec["category"],
            "question": parent["question"],
            "answer": mutated,
            "context_used": parent["context_used"],
            "human_faithfulness": 0,
            "mutation_description": spec["desc"],
            "sources_json": parent.get("sources_json", ""),
        })

    return out


def _validate(rows: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    parents_by_id = _load_success_rows()

    for row in rows:
        rid = row["id"]
        if not (row.get("question") or "").strip():
            problems.append(f"{rid}: empty question")
        if not (row.get("answer") or "").strip():
            problems.append(f"{rid}: empty answer")
        if not (row.get("context_used") or "").strip():
            problems.append(f"{rid}: empty context_used")
        if row.get("human_faithfulness") not in (0, 1):
            problems.append(f"{rid}: human_faithfulness is not 0/1 ({row.get('human_faithfulness')!r})")
        if row["human_faithfulness"] == 0 and not (row.get("mutation_description") or "").strip():
            problems.append(f"{rid}: negative example missing mutation_description")

        parent = parents_by_id.get(row["parent_id"])
        if parent is None:
            problems.append(f"{rid}: parent_id {row['parent_id']!r} not found among status=success rows")
        elif row["context_used"] != parent["context_used"]:
            problems.append(f"{rid}: context_used does not exactly match parent {row['parent_id']}")

    return problems


def main() -> None:
    random.seed(SEED)

    parents = _load_success_rows()
    print(f"Loaded {len(parents)} status=success rows from {INPUT_CSV}")

    rows = _build_rows(parents)
    random.shuffle(rows)  # deterministic (seed=42) presentation order only — content is unaffected

    print("Validating before saving...")
    problems = _validate(rows)
    if problems:
        print(f"\n{len(problems)} VALIDATION FAILURE(S) — refusing to save:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("All validation checks passed (no empty fields, labels are 0/1, every negative "
          "example has a mutation_description, every context_used exactly matches its parent).")

    FAITHFULNESS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter
    cat_counts = Counter(r["category"] for r in rows)
    label_counts = Counter(r["human_faithfulness"] for r in rows)
    parent_ids = sorted({r["parent_id"] for r in rows})

    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total rows: {len(rows)}")
    print("\nCategory counts:")
    for cat, n in sorted(cat_counts.items()):
        print(f"  {cat:<22} {n}")
    print(f"\nhuman_faithfulness=1 (faithful):   {label_counts.get(1, 0)}")
    print(f"human_faithfulness=0 (unfaithful): {label_counts.get(0, 0)}")
    print(f"\nDistinct parent IDs used ({len(parent_ids)}): {', '.join(parent_ids)}")
    print("\nValidation result: PASSED")


if __name__ == "__main__":
    main()
