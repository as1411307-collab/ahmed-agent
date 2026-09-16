from __future__ import annotations

from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import evaluation_aa_rc_026 as aa_rc_026


def _architecture_result() -> dict:
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
            "architecture_unchanged": True,
            "evidence_file_hashes": {"source.py": "a" * 64},
            "implementation_evidence": [
                {
                    "relative_source_path": f"{group_name}.py",
                    "line_start": 10,
                    "line_end": 20,
                    "extracted_evidence": f"{group_name} implementation",
                    "file_sha256": "b" * 64,
                    "trust_classification": "VERIFIED_PROJECT_EVIDENCE",
                    "verification_status": "VERIFIED",
                }
            ],
            "wiring_evidence": [],
            "test_evidence": [],
        }
    return {
        "status": "VERIFIED",
        "architecture_fingerprint": "c" * 64,
        "groups": groups,
        "limitations": [],
    }


def _external_evidence() -> list[dict]:
    return [
        {
            "key": document["key"],
            "url": document["url"],
            "title": f"Official OpenAI {document['key']}",
            "source_identity": "external_openai_official",
            "evidence_classification": "EXTERNAL_EVIDENCE",
            "verification_status": "UNVERIFIED_EXTERNAL",
            "retrieved_at": "2026-09-14T18:30:00+00:00",
            "date_accessed": "2026-09-14T18:30:00+00:00",
            "content_sha256": "d" * 64,
            "snippets": [f"Official evidence for {document['key']}."],
        }
        for document in aa_rc_026.OPENAI_DOCUMENTS
    ]


class AaRc026IntegrationTests(IsolatedAsyncioTestCase):
    async def test_execute_path_sends_auditable_packet_and_report(self) -> None:
        captured: dict = {}

        async def fake_execute_real_case(*args, **kwargs) -> dict:
            return {
                "case_id": "AA-RC-026",
                "run_id": "integration-run",
                "provider": "gemini",
                "model": "gemini-test",
                "execution_status": "EXECUTED",
                "execution_classification": "EXECUTION_OK",
                "final_output": "Continue Ahmed Agent incrementally.",
                "tool_calls": [
                    {"name": "inspect_architecture_evidence", "status": "success"},
                    {"name": "web_search", "status": "success"},
                ],
                "citations": ["project:runtime", aa_rc_026.OPENAI_DOCUMENTS[0]["url"]],
                "sources": ["project:runtime", aa_rc_026.OPENAI_DOCUMENTS[0]["url"]],
                "evidence_provenance": [
                    {
                        "source_identity": "Ahmed Agent current architecture",
                        "verification_status": "VERIFIED",
                    }
                ],
                "external_evidence_provenance": [
                    {
                        "url": aa_rc_026.OPENAI_DOCUMENTS[0]["url"],
                        "title": "Search result",
                        "source_identity": "external_openai_official",
                        "verification_status": "UNVERIFIED_EXTERNAL",
                    }
                ],
                "evidence_preconditions": {
                    "status": "READY",
                    "missing": [],
                    "observed_tool_names": [
                        "inspect_architecture_evidence",
                        "web_search",
                    ],
                },
                "runtime_cleanup": {},
            }

        def fake_review_document(*, packet, baseline, reviewer_provider):
            captured["packet"] = packet
            return {
                "reviews": [
                    {
                        "case_id": "AA-RC-026",
                        "decision": "REVIEW_REQUIRED",
                        "scores": {dimension: 2 for dimension in aa_rc_026.AUDIT_DIMENSIONS},
                        "dimension_notes": {
                            dimension: (
                                "Evidence note with claim references and source "
                                "locators for independent review."
                            )
                            for dimension in aa_rc_026.AUDIT_DIMENSIONS
                        },
                        "reason": "The packet remains review-required for unresolved claims.",
                        "evidence_notes": "The packet includes claim-level evidence and audit fields.",
                    }
                ]
            }

        def fake_fetch_external() -> list[dict]:
            return _external_evidence()

        with (
            patch.object(aa_rc_026, "inspect_architecture_evidence", return_value=_architecture_result()),
            patch.object(aa_rc_026, "fetch_openai_external_evidence", new=fake_fetch_external),
            patch.object(aa_rc_026, "execute_real_case", new=fake_execute_real_case),
            patch.object(
                aa_rc_026,
                "evaluate_case",
                return_value={
                    "semantic_status": "REVIEW_REQUIRED",
                    "execution_status": "EXECUTED",
                    "execution_classification": "EXECUTION_OK",
                    "dimensions": {},
                },
            ),
            patch.object(
                aa_rc_026,
                "build_independent_review_document",
                side_effect=fake_review_document,
            ),
        ):
            report = await aa_rc_026.execute_aa_rc_026(
                base_url="http://test.invalid",
                owner_token="owner-token-for-test-only",
            )

        review_trace = captured["packet"]["cases"][0]["trace"]
        self.assertIn("claim_level_evidence_map", review_trace)
        self.assertIn("external_evidence_reconciliation", review_trace)
        self.assertIn("answer_for_review_redacted", review_trace)
        self.assertIn("Unresolved claims and abstention", review_trace["answer_for_review_redacted"])
        self.assertIn("auditability", report)
        self.assertTrue(report["auditability"]["answer_saved"])
        self.assertTrue(report["auditability"]["raw_trace_saved"])
        self.assertIn("project_evidence_context", report["auditability"])
        self.assertIn("external_evidence_context", report["auditability"])
        self.assertIn("claim_level_evidence_map", report["auditability"])
        self.assertIn("external_evidence_reconciliation", report["auditability"])
        self.assertEqual(
            set(report["independent_review"]["reviews"][0]["dimension_notes"]),
            set(aa_rc_026.AUDIT_DIMENSIONS),
        )