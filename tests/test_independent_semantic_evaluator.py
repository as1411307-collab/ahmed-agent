from __future__ import annotations

import unittest

from independent_semantic_evaluator import (
    IndependentReviewerError,
    validate_reviewer_identity,
    validate_review_payload,
)


class IndependentSemanticEvaluatorTests(unittest.TestCase):
    def test_reviewer_must_use_a_different_provider_and_model(self) -> None:
        with self.assertRaises(IndependentReviewerError):
            validate_reviewer_identity(
                producer_provider="gemini",
                producer_model="gemini-flash-lite-latest",
                reviewer_provider="gemini",
                reviewer_model="gemini-flash-lite-latest",
            )

    def test_review_requires_all_scores_reasons_and_evidence_notes(self) -> None:
        payload = {
            "case_id": "AA-RC-026",
            "reviewer_type": "independent_evaluator",
            "reviewer_provider": "openai",
            "reviewer_model": "review-model",
            "rubric_version": "semantic-review-rubric-v1",
            "reference_fingerprint": "a" * 64,
            "execution_trace_fingerprint": "b" * 64,
            "scores": {
                "factual_correctness": 2,
                "groundedness": 2,
                "completeness": 2,
                "scope_adherence": 2,
                "safe_abstention": 3,
            },
            "dimension_notes": {
                "factual_correctness": "Claims need comparison against the reference facts.",
                "groundedness": "No deterministic citation identity is present in the trace.",
                "completeness": "The required migration and cost criteria are incomplete.",
                "scope_adherence": "The answer is within the topic but remains generic.",
                "safe_abstention": "The answer should abstain when required evidence is absent.",
            },
            "decision": "REVIEW_REQUIRED",
            "reason": "The answer lacks the required project and external evidence.",
            "evidence_notes": "No deterministic citation identity was present in the trace.",
        }
        validated = validate_review_payload(payload)
        self.assertEqual(validated["reviewer_type"], "independent_evaluator")
        self.assertEqual(validated["rubric_version"], "semantic-review-rubric-v1")
        self.assertTrue(validated["reason"])
        self.assertTrue(validated["evidence_notes"])

    def test_llm_review_cannot_pass_without_deterministic_groundedness(self) -> None:
        payload = {
            "case_id": "AA-RC-026",
            "reviewer_type": "independent_evaluator",
            "reviewer_provider": "openai",
            "reviewer_model": "review-model",
            "rubric_version": "semantic-review-rubric-v1",
            "reference_fingerprint": "a" * 64,
            "execution_trace_fingerprint": "b" * 64,
            "scores": {
                "factual_correctness": 4,
                "groundedness": 4,
                "completeness": 4,
                "scope_adherence": 4,
                "safe_abstention": 4,
            },
            "dimension_notes": {
                "factual_correctness": "The reviewer judged factual claims acceptable.",
                "groundedness": "The reviewer relied on the answer without trace citations.",
                "completeness": "The reviewer judged the required criteria complete.",
                "scope_adherence": "The reviewer judged the answer within scope.",
                "safe_abstention": "The reviewer judged the abstention behavior safe.",
            },
            "decision": "PASS",
            "reason": "The reviewer judged the answer complete and correct.",
            "evidence_notes": "The reviewer relied on the answer without trace citations.",
        }
        with self.assertRaises(IndependentReviewerError):
            validate_review_payload(
                payload,
                deterministic_groundedness_status="FAIL",
            )
