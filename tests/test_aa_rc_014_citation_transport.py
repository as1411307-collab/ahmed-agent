from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_core import prepare_evidence_first_context
from evaluation_baseline import (
    build_evaluation_trace,
    deterministic_grade,
    load_case_document,
)
from evidence_citations import (
    render_evidence_citation,
    render_evidence_report,
    remove_model_source_markers,
    verify_evidence_citation,
)


def _evidence_items(
    *,
    relative_source_path: str,
    file_sha256: str,
    count: int,
) -> list[dict[str, object]]:
    return [
        {
            "relative_source_path": relative_source_path,
            "file_sha256": file_sha256,
            "line_start": line_number,
            "line_end": line_number,
            "verification_status": "VERIFIED",
            "trust_classification": "PROJECT_SOURCE",
        }
        for line_number in range(1, count + 1)
    ]


class AARC014CitationTransportTests(unittest.IsolatedAsyncioTestCase):
    def test_model_output_removes_numeric_markers_but_preserves_canonical_citations(
        self,
    ) -> None:
        evidence_item = {
            "relative_source_path": "agent_core.py",
            "file_sha256": "c" * 64,
            "line_start": 1571,
            "line_end": 1571,
            "verification_status": "VERIFIED",
            "trust_classification": "PROJECT_SOURCE",
        }
        canonical_citation = render_evidence_citation(evidence_item)
        self.assertIsNotNone(canonical_citation)

        cleaned_output = remove_model_source_markers("Model answer with [1]")
        evidence_report = render_evidence_report(
            [
                {
                    "target": "source_status",
                    "evidence_status": "VERIFIED",
                    "evidence_items": [evidence_item],
                }
            ]
        )
        final_output = cleaned_output + evidence_report

        self.assertNotIn("[1]", cleaned_output)
        self.assertNotIn("[1]", final_output)
        self.assertIn(canonical_citation, final_output)

    async def test_live_citation_transport_separates_model_and_audit_lists(self) -> None:
        search_items = _evidence_items(
            relative_source_path="search_fabric.py",
            file_sha256="a" * 64,
            count=20,
        )
        fetch_items = _evidence_items(
            relative_source_path="skill_tools.py",
            file_sha256="b" * 64,
            count=11,
        )

        def inspect_status(component: str) -> dict[str, object]:
            return {
                "evidence_status": "VERIFIED",
                "evidence_items": search_items
                if component == "search_provider"
                else fetch_items,
            }

        recorded_events: list[tuple[object, ...]] = []

        async def record_event(*args: object) -> None:
            recorded_events.append(args)

        with patch(
            "agent_core.inspect_existing_source_status",
            side_effect=inspect_status,
        ):
            context = await prepare_evidence_first_context(
                "افحص WEB/search بعد المراحل السابقة وحدد الجاهز وغير الجاهز.",
                tool_event_recorder=record_event,
            )

        payloads = json.loads(context.model_context.split("evidence_payload=", 1)[1])
        self.assertEqual([len(payload["citations"]) for payload in payloads], [8, 8])
        self.assertTrue(all(payload["citations"] for payload in payloads))
        self.assertNotIn("extracted_evidence", context.model_context)

        audit_citations = [
            citation
            for event in recorded_events
            for citation in (event[3] or {}).get("evidence_citations", [])
        ]
        self.assertEqual(len(audit_citations), 31)

        case = next(
            case
            for case in load_case_document(
                Path("tests/fixtures/evaluation_baseline/real_cases.json"),
                require_baseline_size=True,
            )[1]
            if case["id"] == "AA-RC-014"
        )
        persisted = {
            "run": {"status": "succeeded", "model_name": "test-model"},
            "tool_events": [
                {
                    "tool_name": event[0],
                    "status": event[1],
                    "duration_ms": event[2],
                    "safe_metadata": event[3],
                }
                for event in recorded_events
            ],
            "pending_actions": [],
        }
        trace = build_evaluation_trace(
            case=case,
            run_id="aa-rc-014-regression",
            scope="WEB",
            provider="gemini",
            response_status=200,
            response_payload={
                "reply": render_evidence_report(context.evidence_envelopes),
            },
            persisted=persisted,
            latency_ms=1,
        )
        grade = deterministic_grade(case, trace)

        self.assertEqual(len(trace["citations"]), 31)
        self.assertEqual(len(trace["available_evidence_items"]), 31)
        self.assertEqual(len(trace["available_evidence_citations"]), 31)
        self.assertTrue(
            all(
                verify_evidence_citation(
                    citation,
                    trace["available_evidence_items"],
                    expected_claim="Ahmed Agent project files",
                )
                for citation in trace["citations"]
            )
        )
        self.assertTrue(grade["check_results"]["evidence_citation_contract"])