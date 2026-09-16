from __future__ import annotations

import unittest

from evidence_citations import (
    parse_evidence_citation,
    render_evidence_citation,
    render_evidence_report,
    verify_evidence_citation,
)


def _item(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "relative_source_path": "search_fabric.py",
        "file_sha256": "a" * 64,
        "line_start": 10,
        "line_end": 12,
        "extracted_evidence": "redacted source",
        "trust_classification": "PROJECT_SOURCE",
        "verification_status": "VERIFIED",
    }
    value.update(overrides)
    return value


class EvidenceCitationTests(unittest.TestCase):
    def test_render_and_verify_preserve_identity_metadata(self) -> None:
        item = _item()
        citation = render_evidence_citation(item)
        self.assertIsNotNone(citation)
        assert citation is not None
        self.assertEqual(parse_evidence_citation(citation)["line_start"], 10)
        self.assertTrue(verify_evidence_citation(citation, [item]))

    def test_all_supported_statuses_render_without_inventing_paths(self) -> None:
        for status in ("VERIFIED", "DISCREPANCY", "NOT_FOUND", "DENIED"):
            citation = render_evidence_citation(_item(verification_status=status))
            self.assertIsNotNone(citation)
            assert citation is not None
            self.assertIn(f"status: {status}", citation)
        self.assertIsNone(render_evidence_citation(_item(relative_source_path="../secret")))

    def test_mismatch_and_unsupported_claim_are_rejected(self) -> None:
        item = _item()
        citation = render_evidence_citation(item)
        assert citation is not None
        self.assertFalse(verify_evidence_citation(citation.replace("a" * 64, "b" * 64), [item]))
        self.assertFalse(
            verify_evidence_citation(
                citation,
                [item],
                expected_claim="an unrelated claim",
            )
        )
        self.assertFalse(verify_evidence_citation("[source: invented]", [item]))

    def test_report_states_limitations_without_fabricated_citation(self) -> None:
        report = render_evidence_report(
            [
                {
                    "target": "page_fetcher",
                    "evidence_status": "DENIED",
                    "evidence_items": [],
                }
            ]
        )
        self.assertIn("DENIED", report)
        self.assertNotIn("[source:", report)

    def test_hash_and_line_tampering_is_rejected(self) -> None:
        item = _item()
        citation = render_evidence_citation(item)
        assert citation is not None
        self.assertFalse(
            verify_evidence_citation(
                citation.replace("lines: 10-12", "lines: 10-13"),
                [item],
            )
        )
        self.assertFalse(
            verify_evidence_citation(
                citation.replace("status: VERIFIED", "status: DISCREPANCY"),
                [item],
            )
        )


if __name__ == "__main__":
    unittest.main()