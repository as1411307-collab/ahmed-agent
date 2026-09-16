from __future__ import annotations

import unittest

from evaluation_aa_rc_026 import (
    AUDIT_DIMENSIONS,
    AuditabilityError,
    build_audit_ready_answer,
    build_claim_level_evidence_map,
    build_comparison_matrix,
    build_external_evidence_reconciliation,
    build_review_audit_trace,
    build_review_dimension_notes,
)


def _architecture() -> dict:
    groups = {}
    for group_name in (
        "runtime",
        "persistence_state_recovery",
        "policy_security",
        "evidence_provenance",
        "evaluation",
        "actions_integrations",
    ):
        groups[group_name] = {
            "status": "VERIFIED",
            "citations": [
                {
                    "relative_source_path": f"{group_name}.py",
                    "line_start": 10,
                    "line_end": 20,
                    "extracted_evidence": f"Verified {group_name} evidence",
                    "file_sha256": "b" * 64,
                    "trust_classification": "VERIFIED_PROJECT_EVIDENCE",
                    "verification_status": "VERIFIED",
                }
            ],
        }
    return {
        "architecture_fingerprint": "a" * 64,
        "groups": groups,
        "citations": [citation for group in groups.values() for citation in group["citations"]],
    }


def _external() -> list[dict]:
    keys = (
        "agents_sdk_evolution",
        "responses_api_tools",
        "models_catalog",
        "model_comparison",
        "api_platform_pricing",
        "agentkit",
        "workspace_agents",
    )
    return [
        {
            "key": key,
            "url": f"https://openai.example.test/{key}",
            "title": f"Official OpenAI {key}",
            "date_accessed": "2026-09-14T18:00:00+00:00",
            "retrieved_at": "2026-09-14T18:00:00+00:00",
            "content_sha256": "c" * 64,
            "source_identity": "external_openai_official",
            "verification_status": "UNVERIFIED_EXTERNAL",
            "snippets": [f"Official evidence for {key}."],
        }
        for key in keys
    ]


class AaRc026AuditabilityTests(unittest.TestCase):
    def test_material_claims_require_direct_project_or_external_sources(self) -> None:
        matrix = build_comparison_matrix()
        matrix[0]["evidence_refs"] = []
        with self.assertRaises(AuditabilityError):
            build_claim_level_evidence_map(
                architecture=_architecture(),
                external=_external(),
                comparison_matrix=matrix,
            )

    def test_external_reconciliation_records_approved_and_excluded_results(self) -> None:
        external = _external()
        reconciliation = build_external_evidence_reconciliation(
            retrieved_results=[
                {
                    "url": external[0]["url"],
                    "title": external[0]["title"],
                    "retrieved_at": external[0]["retrieved_at"],
                },
                {
                    "url": "https://example.test/unapproved",
                    "title": "Unapproved result",
                    "retrieved_at": external[0]["retrieved_at"],
                },
            ],
            approved_sources=external,
        )
        self.assertEqual(reconciliation["retrieved_result_count"], 2)
        self.assertEqual(reconciliation["approved_source_count"], len(external))
        self.assertEqual(len(reconciliation["excluded_results"]), 1)
        self.assertEqual(
            reconciliation["excluded_results"][0]["reason"],
            "not_in_approved_official_source_bundle",
        )
        for claim in reconciliation["external_claims"]:
            self.assertTrue(claim["url"])
            self.assertTrue(claim["title"])
            self.assertTrue(claim["accessed_at"])

    def test_answer_contains_direct_citations_and_explicit_unresolved_section(self) -> None:
        claims = build_claim_level_evidence_map(
            architecture=_architecture(),
            external=_external(),
            comparison_matrix=build_comparison_matrix(),
        )
        answer = build_audit_ready_answer(
            original_answer="Continue incrementally, but do not rebuild solely because the alternative is newer.",
            claims=claims,
        )
        self.assertIn("[[claim:", answer)
        self.assertIn("Unresolved claims", answer)
        self.assertIn("abstain", answer.casefold())

    def test_review_trace_preserves_answer_raw_trace_and_contexts(self) -> None:
        claims = build_claim_level_evidence_map(
            architecture=_architecture(),
            external=_external(),
            comparison_matrix=build_comparison_matrix(),
        )
        reconciliation = build_external_evidence_reconciliation(
            retrieved_results=[],
            approved_sources=_external(),
        )
        audited = build_review_audit_trace(
            answer="Audited answer",
            raw_trace={"run_id": "run-1", "output": {"answer": "Audited answer"}},
            architecture=_architecture(),
            external=_external(),
            claims=claims,
            reconciliation=reconciliation,
        )
        self.assertTrue(audited["answer_saved"])
        self.assertTrue(audited["raw_trace_saved"])
        self.assertEqual(audited["answer"], "Audited answer")
        self.assertIn("project_evidence_context", audited)
        self.assertIn("external_evidence_context", audited)
        self.assertIn("claim_level_evidence_map", audited)
        self.assertIn("external_evidence_reconciliation", audited)

    def test_review_notes_cover_every_dimension_with_evidence_refs(self) -> None:
        claims = build_claim_level_evidence_map(
            architecture=_architecture(),
            external=_external(),
            comparison_matrix=build_comparison_matrix(),
        )
        notes = build_review_dimension_notes(claims=claims)
        self.assertEqual(set(notes), set(AUDIT_DIMENSIONS))
        for dimension in AUDIT_DIMENSIONS:
            self.assertGreaterEqual(len(notes[dimension]), 40)
            self.assertIn("claim:", notes[dimension])