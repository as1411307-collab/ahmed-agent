from __future__ import annotations

import io
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import evaluation_aa_rc_026 as aa_rc_026


class AaRc026ExternalSnapshotTests(unittest.TestCase):
    def test_403_uses_documented_orchestrator_external_snapshot(self) -> None:
        http_error = HTTPError(
            url=aa_rc_026.OPENAI_DOCUMENTS[0]["url"],
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=io.BytesIO(),
        )
        with patch.object(
            aa_rc_026.urllib.request,
            "urlopen",
            side_effect=http_error,
        ):
            evidence = aa_rc_026.fetch_openai_external_evidence()

        self.assertTrue(evidence)
        self.assertTrue(
            all(
                item["evidence_origin"]
                == "ORCHESTRATOR_SUPPLIED_EXTERNAL_SNAPSHOT"
                for item in evidence
            )
        )
        self.assertTrue(
            all(item["source_identity"] == "external_openai_official" for item in evidence)
        )
        self.assertTrue(all(item["verification_status"] == "UNVERIFIED_EXTERNAL" for item in evidence))
        self.assertTrue(all(item["url"] and item["title"] for item in evidence))
        self.assertTrue(
            all(
                item["accessed_at"] == aa_rc_026.SNAPSHOT_ACCESSED_AT
                for item in evidence
            )
        )
        self.assertTrue(all(len(item["content_sha256"]) == 64 for item in evidence))
        self.assertTrue(all(item["content_claims"] for item in evidence))