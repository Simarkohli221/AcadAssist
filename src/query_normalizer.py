"""
query_normalizer.py
───────────────────
Fix #2 — Query Normalization

Expands abbreviations and normalises short/abbreviated queries before
embedding and BM25 scoring. This makes short queries like "sem 3 cg"
equivalent to "semester 3 cgpa" in embedding space.

Also implements Fix #3 — Query Category Router
Classifies queries into one of: examination | syllabus | placement | activity | general
so the retriever can filter documents by category (metadata routing).

Usage
─────
    from query_normalizer import normalize_query, classify_query_category

    clean_q, category = normalize_query("what is cg criteria for elite spit?")
    # → ("what is cgpa criteria for elite sardar patel institute of technology?", "placement")
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# ── Abbreviation expansion map ────────────────────────────────────────────────
# Order matters: longer patterns should come first where ambiguous.
_ABBREV_MAP: list[tuple[re.Pattern, str]] = [
    # Institution
    (re.compile(r"\bspit\b", re.I), "sardar patel institute of technology"),
    # Academic terms
    (re.compile(r"\bsems?\b", re.I), "semester"),
    (re.compile(r"\bcg\b", re.I), "cgpa"),
    (re.compile(r"\bsg\b", re.I), "sgpa"),
    (re.compile(r"\bcgpa\b", re.I), "cgpa"),
    (re.compile(r"\bsgpa\b", re.I), "sgpa"),
    (re.compile(r"\bai\b", re.I), "artificial intelligence"),
    (re.compile(r"\bml\b", re.I), "machine learning"),
    (re.compile(r"\bds\b", re.I), "data science"),
    (re.compile(r"\bor\b", re.I), "or"),   # keep 'or' unchanged
    (re.compile(r"\bsubjects?\b", re.I), "courses"),
    (re.compile(r"\bktb?\b", re.I), "KT"),
    (re.compile(r"\batkt\b", re.I), "ATKT"),
    (re.compile(r"\bff\b", re.I), "FF grade"),
    (re.compile(r"\bise\b", re.I), "in-semester evaluation"),
    (re.compile(r"\bmse\b", re.I), "mid-semester examination"),
    (re.compile(r"\bese\b", re.I), "end semester examination"),
    (re.compile(r"\bcoe\b", re.I), "controller of examinations"),
    (re.compile(r"\bhod\b", re.I), "head of department"),
    (re.compile(r"\btpo\b", re.I), "training and placement officer"),
    (re.compile(r"\bnptel\b", re.I), "NPTEL"),
    (re.compile(r"\bmooc\b", re.I), "MOOC online course"),
    (re.compile(r"\bug\b", re.I), "undergraduate"),
    (re.compile(r"\bpg\b", re.I), "postgraduate"),
    (re.compile(r"\bbtech\b", re.I), "bachelor of technology"),
    (re.compile(r"\bb\.tech\b", re.I), "bachelor of technology"),
    (re.compile(r"\bmtech\b", re.I), "master of technology"),
    (re.compile(r"\bcse\b", re.I), "computer science engineering"),
    (re.compile(r"\bextc\b", re.I), "electronics telecommunication engineering"),
    (re.compile(r"\baiml\b", re.I), "artificial intelligence machine learning"),
    (re.compile(r"\bctc\b", re.I), "cost to company salary package"),
    (re.compile(r"\blpa\b", re.I), "lakhs per annum"),
    (re.compile(r"\bppo\b", re.I), "pre-placement offer"),
    (re.compile(r"\bnoc\b", re.I), "no objection certificate"),
    (re.compile(r"\bloc\b", re.I), "letter of commitment"),
    (re.compile(r"\blor\b", re.I), "letter of recommendation"),
    # Attendance shorthand
    (re.compile(r"\battd\b", re.I), "attendance"),
    (re.compile(r"\batt\b", re.I), "attendance"),
    # Grades
    (re.compile(r"\bgpa\b", re.I), "grade point average"),
    (re.compile(r"\bgp\b", re.I), "grade point"),
]


def normalize_query(query: str) -> str:
    """
    Expand abbreviations in the query string.

    Parameters
    ──────────
    query : raw user input

    Returns
    ───────
    Expanded query string (lowercased, abbreviations expanded).
    """
    if not query or not query.strip():
        return query

    normalized = query.strip()
    for pattern, replacement in _ABBREV_MAP:
        normalized = pattern.sub(replacement, normalized)

    # Collapse multiple spaces
    normalized = re.sub(r"\s{2,}", " ", normalized).strip()

    if normalized.lower() != query.lower():
        logger.debug("Query normalized: '%s' → '%s'", query, normalized)

    return normalized


# ── Category routing ──────────────────────────────────────────────────────────
# Maps category name → (source doc stems that belong to it, keyword signals)
_CATEGORY_RULES: list[tuple[str, list[str], list[str]]] = [
    # (category, source_stems, keyword_signals)
    (
        "placement",
        ["placement-policy", "placement_policy", "placement"],
        [
            "placement", "internship", "recruit", "company", "ctc", "lpa",
            "elite", "dream", "demerit", "tpo", "offer", "campus", "offcamp",
            "package", "salary", "ppo", "debarred", "ban", "misbehav",
            "canvass", "noc", "startup", "deferred", "attire", "formal",
            "higher studies", "lor",
        ],
    ),
    (
        "examination",
        ["examination", "exam"],
        [
            "attendance", "cgpa", "sgpa", "grade", "ff", "kt", "atkt",
            "exam", "pass", "fail", "detain", "fee", "fine", "malpractice",
            "unfair means", "promotion", "credit", "degree", "ise", "mse",
            "ese", "reexam", "re-exam", "special exam", "median", "marks",
            "percentage", "backlog", "semester result", "open house",
            "re-registration", "registration", "grading", "relative grading",
        ],
    ),
    (
        "syllabus",
        ["cse-syllabus", "cse_syllabus", "syllabus", "curriculum"],
        [
            "syllabus", "course", "subject", "semester", "module", "unit",
            "credit hour", "teaching scheme", "elective", "minor", "nptel",
            "lecture", "tutorial", "lab", "practical", "course code",
            "curriculum", "textbook", "reference", "outcomes", "prerequisite",
            "multidisciplinary", "open elective", "hss", "llc",
        ],
    ),
    (
        "activity",
        ["annual-report", "annual_report"],
        [
            "club", "event", "activity", "fest", "cultural", "technical",
            "sports", "committee", "nss", "hackathon", "workshop", "seminar",
            "alumni", "annual report", "achievement", "award",
        ],
    ),
]


def classify_query_category(query: str) -> str:
    """
    Classify a query into one of: examination | syllabus | placement | activity | general

    Uses keyword matching — no ML model needed, fast and transparent.

    Parameters
    ──────────
    query : (optionally already normalized) query string

    Returns
    ───────
    Category string. "general" means search all documents.
    """
    lower = query.lower()

    # Score each category by keyword hits
    scores: dict[str, int] = {}
    for cat, _stems, keywords in _CATEGORY_RULES:
        score = sum(1 for kw in keywords if kw in lower)
        if score > 0:
            scores[cat] = score

    if not scores:
        return "general"

    best_cat = max(scores, key=lambda c: scores[c])
    logger.debug(
        "Query category: '%s' → %s (scores: %s)",
        query[:60], best_cat, scores,
    )
    return best_cat


def get_category_sources(category: str) -> list[str] | None:
    """
    Return the list of source document stems that belong to a category.
    Returns None for "general" (search all).
    """
    if category == "general":
        return None
    for cat, stems, _ in _CATEGORY_RULES:
        if cat == category:
            return stems
    return None