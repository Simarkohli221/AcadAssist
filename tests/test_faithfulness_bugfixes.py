"""
tests/test_faithfulness_bugfixes.py
------------------------------------
Focused regression tests for two faithfulness-verifier bugs found during the
real-world validation run (data/faithfulness/real_validation.csv, rows
E02/N04 — see the run's --metrics diagnosis):

  BUG 1 — condition/fee exam-stage confusion: a correct answer citing two
          legitimate fees for the same attendance bracket (a primary
          re-examination fee, plus a later special-examination fee that
          only applies if the student subsequently gets FF) was flagged as
          self-contradictory. Root cause: _extract_condition_rules() split
          text into "clauses" and picked ONE condition per clause via
          if/elif, so a clause spanning a whole flattened table block
          (common in the real PDF-extracted text) collapsed every fee in it
          onto whichever condition matched first — silently dropping or
          misattributing fees that legitimately belonged elsewhere. Fixed
          by anchoring extraction to each individual occurrence of a
          condition phrase with a bounded local window, instead of to a
          pre-split clause.

  BUG 2 — percentage vs. "out of 100" equivalence: "40%" and a context
          expressing the identical value as "40 out of 100" or
          "(out of 100) >= 40" were treated as unrelated numeric tokens, so
          a correct percentage answer scored as an unmatched (apparently
          hallucinated) number. Fixed by additively deriving a "%" token
          for any "<N> out of 100" / "<N>/100" phrasing found.

These tests call the pure helper functions directly (FaithfulnessVerifier is
constructed with model_path=None, which skips loading the trained
classifier entirely — numeric_consistency() and
_check_condition_value_contradiction() never touch the embedder or the
model either), so this file runs fast, offline, with no Groq/API calls and
no model/embedder loading.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import faithfulness as fm  # noqa: E402  (import after sys.path setup)


# A condensed version of the real examination.pdf Table 1 text (same
# structure/wording pattern that produced the real E02 false positive):
# multiple attendance brackets and their fees run together with no
# sentence-ending punctuation between table rows.
EXAM_FEE_CONTEXT = (
    "e) Detained students (50-65% attendance) Re-examination fee: Rs. 5000/-. "
    "If attendance is short (50-65% attendance) for medical reasons, "
    "Re-examination fee: Rs. 1000/-.\n"
    "Table 1: Examination Fee structure per course\n"
    "Detained due to Attendance in semester for any course is 50-65% "
    "Rs. 5000/- Rs. 500/-\n(if FF in Re-exam)\n"
    "Detained due to Attendance in semester for any course is 50-65% "
    "& Medical Reasons Rs. 1000/- Rs. 500/-\n(if FF in Re-exam)\n"
    "g) Detained Students (<50% attendance) Special exam fees: Rs. 8000/-. "
    "If attendance is short (<50% attendance) for medical reasons, "
    "special examination fee: Rs. 2000/- (course is re-taught)."
)


class ConditionFeeExamStageTests(unittest.TestCase):
    """BUG 1 — condition/fee exam-stage confusion."""

    def test_correct_5000_fee_not_flagged_despite_nearby_500(self):
        """Requirement 1: a correct 50-65% re-exam answer citing Rs 5,000
        must not be penalized merely because Rs 500 also appears nearby for
        the later special-exam stage."""
        answer = (
            "The re-examination fee for a student detained due to 50-65% "
            "attendance is Rs. 5,000 (with an additional Rs. 500 only if the "
            "student has an FF grade in the re-exam)."
        )
        self.assertFalse(
            fm._check_condition_value_contradiction(answer, EXAM_FEE_CONTEXT),
            "a correct answer citing both legitimate 50-65% fees must not be flagged as contradictory",
        )

    def test_genuinely_wrong_fee_for_50_65_is_still_detected(self):
        """Requirement 2: a fee that appears nowhere in context for this
        condition is still a genuine contradiction."""
        answer = "The re-examination fee for 50-65% attendance is Rs. 9000."
        self.assertTrue(
            fm._check_condition_value_contradiction(answer, EXAM_FEE_CONTEXT),
            "a fee not present anywhere in context for this condition must still be flagged",
        )

    def test_fee_borrowed_from_a_different_bracket_is_still_detected(self):
        """Requirement 6: a genuine cross-condition contradiction (citing
        the below-50% fee as if it applied to 50-65%) is still caught."""
        answer = "For attendance below 50% (non-medical), the special exam fee is Rs. 5000."
        self.assertTrue(
            fm._check_condition_value_contradiction(answer, EXAM_FEE_CONTEXT),
            "a fee borrowed from a different attendance bracket must still be flagged",
        )

    def test_correct_below_50_fee_not_flagged(self):
        answer = "For attendance below 50% (non-medical), the special exam fee is Rs. 8000."
        self.assertFalse(
            fm._check_condition_value_contradiction(answer, EXAM_FEE_CONTEXT)
        )

    def test_no_condition_language_never_triggers_the_rule(self):
        """Answers/contexts with no attendance-bracket language at all
        should never engage this check (pre-existing behavior)."""
        self.assertFalse(
            fm._check_condition_value_contradiction(
                "The course code is CS206.", "Operating Systems is CS206."
            )
        )


class PercentOutOf100EquivalenceTests(unittest.TestCase):
    """BUG 2 — percentage vs. "out of 100" equivalence."""

    def test_percent_matches_out_of_100_before(self):
        """Requirement 3 (word order: number before the marker)."""
        context = "A learner must score at least 40 out of 100 in the assignment."
        self.assertIn("40%", fm._numeric_tokens(context))

    def test_percent_matches_out_of_100_after_comparator(self):
        """Requirement 3 (real N04 phrasing: number after the marker, via a comparator)."""
        context = "Average assignment score (out of 100) >= 40 AND Final exam score (out of 100) >= 40."
        self.assertIn("40%", fm._numeric_tokens(context))

    def test_different_percentage_does_not_match(self):
        """Requirement 4: 50% must NOT be treated as equivalent to a
        completely different "40 out of 100" value."""
        context = "A learner must score at least 40 out of 100 in the assignment."
        tokens = fm._numeric_tokens(context)
        self.assertNotIn("50%", tokens)

    def test_matching_percent_answer_scores_fully_consistent(self):
        verifier = fm.FaithfulnessVerifier(model_path=None)
        context = "Average assignment score (out of 100) >= 40 AND Final exam score (out of 100) >= 40."
        answer = "Assignment: at least 40%."
        self.assertEqual(verifier.numeric_consistency(answer, context), 1.0)

    def test_wrong_percent_answer_is_not_masked_by_the_equivalence(self):
        """Requirement 4, via the full feature: a genuinely different
        percentage must still be penalized, not silently matched."""
        verifier = fm.FaithfulnessVerifier(model_path=None)
        context = "Average assignment score (out of 100) >= 40 AND Final exam score (out of 100) >= 40."
        answer = "Assignment: at least 50%."
        score = verifier.numeric_consistency(answer, context)
        self.assertLess(score, 1.0, "a genuinely wrong percentage must not be masked by the out-of-100 equivalence")

    def test_out_of_100_does_not_promote_unrelated_bare_numbers(self):
        """An "out of 100" mention with no adjacent number at all must not
        fabricate a spurious percentage token from something unrelated."""
        context = "Roughly 12 out of 100 students opted for the elective this year."
        tokens = fm._numeric_tokens(context)
        # "12" is legitimately associated (word order: number before marker).
        self.assertIn("12%", tokens)
        # But an unrelated bare number elsewhere must never turn into a percent.
        self.assertNotIn("100%", tokens)


class ExistingNumericConsistencyBehaviorTests(unittest.TestCase):
    """Requirement 5: pre-existing behavior for ordinary numbers is unchanged."""

    def test_no_numeric_claim_in_answer_is_fully_consistent(self):
        verifier = fm.FaithfulnessVerifier(model_path=None)
        self.assertEqual(
            verifier.numeric_consistency(
                "Students must submit the form before the deadline.", "Some context."
            ),
            1.0,
        )

    def test_matching_plain_number_is_consistent(self):
        verifier = fm.FaithfulnessVerifier(model_path=None)
        context = "The minimum CGPA required is 8.25 for Honors by Research."
        answer = "The minimum CGPA required is 8.25."
        self.assertEqual(verifier.numeric_consistency(answer, context), 1.0)

    def test_unmatched_three_digit_number_is_ungrounded(self):
        verifier = fm.FaithfulnessVerifier(model_path=None)
        context = "The malpractice special examination fee is Rs. 8000/-."
        answer = "The malpractice special examination fee is Rs. 1234/-."
        self.assertEqual(verifier.numeric_consistency(answer, context), 0.0)

    def test_context_with_no_numbers_at_all_is_zero(self):
        verifier = fm.FaithfulnessVerifier(model_path=None)
        self.assertEqual(
            verifier.numeric_consistency("The fee is Rs. 5000.", "No numbers appear here at all."),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
