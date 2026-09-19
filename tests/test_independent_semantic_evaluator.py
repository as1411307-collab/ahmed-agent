from __future__ import annotations

import unittest

from independent_semantic_evaluator import (
    REVIEW_DIMENSIONS,
    IndependentReviewerError,
    build_independent_review_document,
    validate_reviewer_identity,
    validate_review_payload,
)


def _full_score_completion(*, decision: str, groundedness: int) -> dict:
    scores = {dimension: 3 for dimension in REVIEW_DIMENSIONS}
    scores["groundedness"] = groundedness
    return {
        "scores": scores,
        "dimension_notes": {
            dimension: f"Fixture note explaining the {dimension} judgement in detail."
            for dimension in REVIEW_DIMENSIONS
        },
        "decision": decision,
        "reason": "Fixture reason with enough detail to satisfy the minimum length check.",
        "evidence_notes": "Fixture evidence notes with enough detail to satisfy validation.",
    }


def _packet_with_one_case(*, expected_sources: list[str], trace_extra: dict) -> dict:
    return {
        "packet_sha256": "p" * 64,
        "cases": [
            {
                "case_id": "TEST-EXTERNAL-001",
                "producer_provider": "gemini",
                "producer_model": "gemini-test",
                "reference_fingerprint": "a" * 64,
                "execution_trace_fingerprint": "b" * 64,
                "reference_assertions": {
                    "groundedness": {"expected_sources": expected_sources},
                },
                "trace": {
                    "citations": ["[1]"],
                    "sources": ["[1]"],
                    **trace_extra,
                },
            }
        ],
    }


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

    def test_deterministic_groundedness_matches_external_source_identity(self) -> None:
        # A web-search-derived case: citations/sources hold only opaque markers
        # ('[1]'), and the only trustworthy signal is the controlled
        # source_identity label our own code assigns (e.g. by classifying the
        # URL's domain). Groundedness must be derivable from that field too,
        # not just citations/sources.
        packet = _packet_with_one_case(
            expected_sources=["external_openai_official"],
            trace_extra={
                "external_evidence_provenance": [
                    {
                        "url": "https://platform.openai.com/docs/guides/agents-sdk",
                        "title": "Agents SDK",
                        "source_identity": "external_openai_official",
                    }
                ],
            },
        )
        document = build_independent_review_document(
            packet=packet,
            baseline={"execution_config": {"provider": "gemini"}},
            reviewer_provider="anthropic",
            reviewer_model="claude-test",
            completion=lambda prompt: _full_score_completion(
                decision="PASS", groundedness=3
            ),
        )
        review = document["reviews"][0]
        self.assertEqual(review["scores"]["groundedness"], 3)
        self.assertEqual(review["decision"], "PASS")
        self.assertNotIn("groundedness_guard_applied", review)

    def test_deterministic_groundedness_ignores_unrelated_external_evidence(self) -> None:
        # Same shape, but the external source's identity does not match
        # expected_sources: groundedness must stay capped below the PASS threshold.
        packet = _packet_with_one_case(
            expected_sources=["external_openai_official"],
            trace_extra={
                "external_evidence_provenance": [
                    {
                        "url": "https://blog.example.test/post",
                        "title": "An unrelated blog post",
                        "source_identity": "external_web_search",
                    }
                ],
            },
        )
        document = build_independent_review_document(
            packet=packet,
            baseline={"execution_config": {"provider": "gemini"}},
            reviewer_provider="anthropic",
            reviewer_model="claude-test",
            completion=lambda prompt: _full_score_completion(
                decision="PASS", groundedness=4
            ),
        )
        review = document["reviews"][0]
        self.assertLess(review["scores"]["groundedness"], 3)

    def test_deterministic_groundedness_does_not_trust_freeform_title_or_url(self) -> None:
        # Security regression: an untrusted external result whose title/url
        # happens to contain an expected_sources phrase must NOT upgrade
        # groundedness. Only the controlled source_identity label counts --
        # title and url are freeform text from the fetched page itself and
        # must not be able to forge a deterministic PASS.
        packet = _packet_with_one_case(
            expected_sources=["Official Example Docs"],
            trace_extra={
                "external_evidence_provenance": [
                    {
                        "url": "https://docs.example.test/official-example-docs",
                        "title": "Official Example Docs",
                        "source_identity": "external_web_search",
                    }
                ],
            },
        )
        document = build_independent_review_document(
            packet=packet,
            baseline={"execution_config": {"provider": "gemini"}},
            reviewer_provider="anthropic",
            reviewer_model="claude-test",
            completion=lambda prompt: _full_score_completion(
                decision="PASS", groundedness=4
            ),
        )
        review = document["reviews"][0]
        self.assertLess(review["scores"]["groundedness"], 3)
        self.assertTrue(review["groundedness_guard_applied"])
        self.assertNotEqual(review["decision"], "PASS")
        self.assertTrue(review["groundedness_guard_applied"])
