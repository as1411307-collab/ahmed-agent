from __future__ import annotations

import unittest

from agent_core import MAX_TOOL_CALLS
from evaluation_aa_rc_002 import (
    _compare_inspected_sources,
    _compare_source_manifest_versions,
)


def _verified(filename: str, content: str, source_id: str) -> dict[str, object]:
    return {
        "evidence_status": "VERIFIED",
        "extracted_facts": {
            "source_id": source_id,
            "source_sha256": filename,
            "extracted_sections": [{"content": content}],
        },
    }


class AaRc002EvaluationTests(unittest.TestCase):
    def test_live_manifest_records_real_identity_and_hash_conflict(self) -> None:
        result = _compare_source_manifest_versions(
            {
                "fixture_version": "2026-09-14.v1",
                "source_pack_sha256": "a" * 64,
                "logical_sources": [
                    {
                        "members": [
                            {"name": "old.md", "sha256": "b" * 64},
                        ]
                    }
                ],
            },
            {
                "fixture_version": "2026-09-14.v2",
                "source_pack_sha256": "c" * 64,
                "logical_sources": [
                    {
                        "source_id": "source-1",
                        "source_version": 1,
                        "members": [
                            {"name": "new.pdf", "sha256": "d" * 64},
                        ],
                    }
                ],
            },
        )

        self.assertEqual(result["comparison_status"], "SOURCE_VERSION_CONFLICT")
        self.assertTrue(result["conflicts"][0]["identity_changed"])
        self.assertEqual(result["source_precedence"], "MY_FILES_ORIGINAL_SOURCE")
        self.assertEqual(result["memory_or_summary_override"], "BLOCKED")

    def test_tool_budget_covers_all_source_inspections(self) -> None:
        self.assertGreaterEqual(MAX_TOOL_CALLS, 5)

    def test_missing_incident_evidence_fails_closed(self) -> None:
        result = _compare_inspected_sources(
            {
                "master-id": _verified(
                    "master-hash",
                    "Recorded incident count: 2\nKnown incidents:\n- INC-001",
                    "master-id",
                )
            },
            {"master-hash": "master-id"},
        )

        self.assertEqual(result["comparison_status"], "MISSING_EVIDENCE")
        self.assertIsNone(result["original_incident_count"])
        self.assertFalse(result["discrepancy_detected"])
        self.assertEqual(result["confirmed_incident_ids"], [])

    def test_three_confirmed_incidents_produce_source_discrepancy(self) -> None:
        inspected = {
            "master-id": _verified(
                "master-hash",
                "Recorded incident count: 2",
                "master-id",
            ),
            "incident-1": _verified(
                "incident-1-hash",
                "Incident ID: INC-001\nStatus: confirmed",
                "incident-1",
            ),
            "incident-2": _verified(
                "incident-2-hash",
                "Incident ID: INC-002\nStatus: confirmed",
                "incident-2",
            ),
            "incident-3": _verified(
                "incident-3-hash",
                "Incident ID: INC-003\nStatus: confirmed",
                "incident-3",
            ),
        }

        result = _compare_inspected_sources(
            inspected,
            {
                "master-hash": "master-id",
                "incident-1-hash": "incident-1",
                "incident-2-hash": "incident-2",
                "incident-3-hash": "incident-3",
            },
        )

        self.assertEqual(result["comparison_status"], "SOURCE_DISCREPANCY")
        self.assertTrue(result["discrepancy_detected"])
        self.assertEqual(result["original_incident_count"], 3)
        self.assertEqual(
            result["confirmed_incident_ids"],
            ["INC-001", "INC-002", "INC-003"],
        )
