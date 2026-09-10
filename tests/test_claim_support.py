"""
tests/test_claim_support.py
-----------------------------
Focused regression tests for the new claim-level support feature
(claim_support_ratio / minimum_claim_support) added to faithfulness.py as
Feature 4/5, alongside the existing semantic_similarity / keyword_jaccard /
numeric_consistency.

Also re-checks the E02/N04/wrong-fee regressions from
tests/test_faithfulness_bugfixes.py to confirm adding two new features did
not disturb the existing three (they are computed by separate methods, but
this is cheap insurance).

Runs offline: FaithfulnessVerifier(model_path=None) skips loading the
trained classifier — only claim_support()/numeric_consistency()/
_check_condition_value_contradiction() are exercised, and only those need
the local embedder (no Groq/API calls, no network).
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import faithfulness as fm  # noqa: E402


class ClaimSupportRatioTests(unittest.TestCase):
    """Stage 4, cases 1-3, 5: claim_support_ratio behavior."""

    EXAM_CONTEXT = (
        "The minimum attendance required for the End-Semester Examination "
        "is 75%. Students who fail to meet this requirement are detained "
        "for that course. The re-examination fee for students with 50-65% "
        "attendance is Rs 5000 per course."
    )

    def setUp(self):
        self.verifier = fm.FaithfulnessVerifier(model_path=None)

    def test_fully_supported_multi_sentence_answer_scores_high(self):
        """Case 1: every claim is directly grounded in context."""
        answer = (
            "The minimum attendance required for the ESE is 75%. "
            "Students with 50-65% attendance must pay a re-examination "
            "fee of Rs 5000."
        )
        ratio, minimum = self.verifier.claim_support(answer, self.EXAM_CONTEXT)
        self.assertEqual(ratio, 1.0, f"expected full support, got ratio={ratio}")
        self.assertGreaterEqual(minimum, fm._CLAIM_SUPPORT_THRESHOLD)

    def test_one_unsupported_sentence_drops_ratio_substantially(self):
        """Case 2: two correct claims + one fabricated claim added."""
        fully_supported = (
            "The minimum attendance required for the ESE is 75%. "
            "Students with 50-65% attendance must pay a re-examination "
            "fee of Rs 5000."
        )
        with_fabrication = fully_supported + (
            " Students who miss the ESE for medical reasons are also "
            "entitled to a full merit scholarship for that semester."
        )
        ratio_clean, _ = self.verifier.claim_support(fully_supported, self.EXAM_CONTEXT)
        ratio_fabricated, min_fabricated = self.verifier.claim_support(with_fabrication, self.EXAM_CONTEXT)

        self.assertLess(
            ratio_fabricated, ratio_clean,
            "adding one unsupported claim must reduce claim_support_ratio",
        )
        self.assertLessEqual(
            ratio_fabricated, 0.75,
            "with 1 of 3 claims fabricated, ratio should drop substantially (<=0.75)",
        )
        self.assertLess(
            min_fabricated, fm._CLAIM_SUPPORT_THRESHOLD,
            "minimum_claim_support must flag the fabricated claim directly",
        )

    def test_completely_unsupported_answer_scores_low(self):
        """Case 3: answer entirely off-topic relative to context."""
        answer = (
            "The campus cafeteria serves a special dosa every Tuesday "
            "afternoon and the hostel curfew is set at ten in the evening."
        )
        ratio, minimum = self.verifier.claim_support(answer, self.EXAM_CONTEXT)
        self.assertEqual(ratio, 0.0, f"expected zero support, got ratio={ratio}")
        self.assertLess(minimum, fm._CLAIM_SUPPORT_THRESHOLD)

    def test_paraphrased_correct_answer_not_unfairly_penalized(self):
        """Case 5: same meaning, different wording, must still score as supported."""
        context = (
            "A student must maintain at least 75% attendance to be "
            "eligible to appear for the End-Semester Examination."
        )
        paraphrased_answer = (
            "Students are required to have a minimum of 75 percent "
            "attendance in order to qualify for the End-Semester Exam."
        )
        ratio, minimum = self.verifier.claim_support(paraphrased_answer, context)
        self.assertEqual(ratio, 1.0, f"paraphrase should still be recognised as supported, got ratio={ratio}")
        self.assertGreaterEqual(minimum, fm._CLAIM_SUPPORT_THRESHOLD)


class FaithfulAbstentionSpecialCaseTests(unittest.TestCase):
    """Stage 4, case 4: abstention must be a special signal, not ordinary similarity."""

    def setUp(self):
        self.verifier = fm.FaithfulnessVerifier(model_path=None)

    IRRELEVANT_CONTEXT = (
        "The Sports Committee is chaired by a student representative and "
        "organises inter-departmental tournaments each semester."
    )

    def test_production_abstention_string_is_recognized(self):
        # generator.NO_ANSWER_RESPONSE, verbatim.
        answer = "I could not find this information in the provided documents."
        self.assertTrue(fm._is_abstention_answer(answer))
        ratio, minimum = self.verifier.claim_support(answer, self.IRRELEVANT_CONTEXT)
        self.assertEqual((ratio, minimum), (1.0, 1.0))

    def test_abstention_paraphrase_is_recognized(self):
        answer = "The provided context does not contain enough information to answer this."
        self.assertTrue(fm._is_abstention_answer(answer))
        ratio, minimum = self.verifier.claim_support(answer, self.IRRELEVANT_CONTEXT)
        self.assertEqual((ratio, minimum), (1.0, 1.0))

    def test_long_substantive_answer_is_not_misdetected_as_abstention(self):
        """A long, real answer that happens to share no vocabulary with an
        abstention phrase must go through ordinary claim evaluation, not be
        waved through as a false abstention."""
        answer = (
            "The minimum attendance required for the End-Semester "
            "Examination is 75%, and students detained for low attendance "
            "must pay a re-examination fee of Rs 5000 for courses in the "
            "50-65% attendance bracket, or Rs 8000 for attendance below "
            "50%, as specified in the examination regulations for this "
            "academic year."
        )
        self.assertFalse(fm._is_abstention_answer(answer))


class RegressionOnExistingBugfixesTests(unittest.TestCase):
    """Stage 4, cases 6-8: E02/N04/wrong-fee behavior must be unaffected by
    adding the new features (separate methods, but confirmed directly)."""

    EXAM_FEE_CONTEXT = (
        "e) Detained students (50-65% attendance) Re-examination fee: Rs. 5000/-. "
        "If attendance is short (50-65% attendance) for medical reasons, "
        "Re-examination fee: Rs. 1000/-.\n"
        "Table 1: Examination Fee structure per course\n"
        "Detained due to Attendance in semester for any course is 50-65% "
        "Rs. 5000/- Rs. 500/-\n(if FF in Re-exam)\n"
        "g) Detained Students (<50% attendance) Special exam fees: Rs. 8000/-."
    )

    def setUp(self):
        self.verifier = fm.FaithfulnessVerifier(model_path=None)

    def test_e02_correct_fee_not_flagged_contradictory(self):
        answer = (
            "The re-examination fee for a student detained due to 50-65% "
            "attendance is Rs. 5,000 (with an additional Rs. 500 only if "
            "the student has an FF grade in the re-exam)."
        )
        self.assertFalse(fm._check_condition_value_contradiction(answer, self.EXAM_FEE_CONTEXT))

    def test_n04_percent_matches_out_of_100_phrasing(self):
        context = "Average assignment score (out of 100) >= 40 AND Final exam score (out of 100) >= 40."
        answer = "Assignment: at least 40%."
        self.assertEqual(self.verifier.numeric_consistency(answer, context), 1.0)

    def test_wrong_fee_still_detected(self):
        answer = "The re-examination fee for 50-65% attendance is Rs. 9000."
        self.assertTrue(fm._check_condition_value_contradiction(answer, self.EXAM_FEE_CONTEXT))
        self.assertEqual(self.verifier.numeric_consistency(answer, self.EXAM_FEE_CONTEXT), 0.0)


class FeatureShapeTests(unittest.TestCase):
    """Sanity checks on the overall feature vector shape/order."""

    def test_feature_names_has_five_entries_in_expected_order(self):
        self.assertEqual(
            fm.FEATURE_NAMES,
            (
                "semantic_similarity", "keyword_jaccard", "numeric_consistency",
                "claim_support_ratio", "minimum_claim_support",
            ),
        )

    def test_extract_features_returns_five_values_matching_feature_names(self):
        verifier = fm.FaithfulnessVerifier(model_path=None)
        result = verifier.extract_features(
            "The minimum attendance required is 75%.",
            "A student must maintain at least 75% attendance to be eligible.",
        )
        as_list = result.as_list()
        self.assertEqual(len(as_list), len(fm.FEATURE_NAMES))
        as_dict = result.as_dict()
        self.assertEqual(set(as_dict.keys()), set(fm.FEATURE_NAMES))


if __name__ == "__main__":
    unittest.main()
