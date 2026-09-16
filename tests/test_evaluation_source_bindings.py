from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from evaluation_source_bindings import (
    EvaluationSourcePackError,
    canonical_source_pack_bytes,
    load_aa_rc_002_my_files_manifest,
    load_aa_rc_002_source_pack,
    resolve_aa_rc_002_sources,
    source_pack_sha256,
)


PACK_ROOT = Path("tests/fixtures/evaluation_baseline/aa_rc_002")


class EvaluationSourceBindingTests(unittest.TestCase):
    def test_my_files_manifest_is_a_new_version_and_preserves_historical_pack(self) -> None:
        historical = load_aa_rc_002_source_pack()
        live = load_aa_rc_002_my_files_manifest()

        self.assertEqual(
            historical["manifest"]["fixture_version"],
            "2026-09-14.v1",
        )
        self.assertEqual(live["manifest"]["fixture_version"], "2026-09-14.v2")
        self.assertEqual(
            live["manifest"]["supersedes_source_pack_sha256"],
            historical["source_pack_sha256"],
        )
        self.assertEqual(len(live["manifest"]["logical_sources"]), 2)
        self.assertEqual(
            [
                source["source_id"]
                for source in live["manifest"]["logical_sources"]
            ],
            [
                "68c95924-16fd-4cde-bd45-fd263141a3da",
                "82084c28-b7ea-4f3b-9d08-979671fec10a",
            ],
        )
        self.assertEqual(
            [
                source["members"][0]["sha256"]
                for source in live["manifest"]["logical_sources"]
            ],
            [
                "33de413a950e0685612775d3c495fb200d84e15fbe23d8de5ede52446b1efbac",
                "9a397abf8dbb5337e7aed2211d449f7c2a05bc7c2789369d7b97993b8f6b5f0b",
            ],
        )
        self.assertEqual(
            live["manifest"]["success_criteria"],
            [
                "source_of_truth",
                "canonical_provenance",
                "source_identity",
                "version_hash_integrity",
                "conflict_handling",
                "primary_source_over_memory_or_summary",
            ],
        )

    def test_pack_is_deterministic_and_resolves_four_original_members(self) -> None:
        pack = load_aa_rc_002_source_pack()
        members = resolve_aa_rc_002_sources()

        self.assertEqual(pack["manifest"]["case_id"], "AA-RC-002")
        self.assertEqual(len(members), 4)
        self.assertEqual(
            [member["logical_name"] for member in members],
            [
                "ahmed_agent_master_v3_ar(1).md",
                "Ahmed Agent - Incident Evidence",
                "Ahmed Agent - Incident Evidence",
                "Ahmed Agent - Incident Evidence",
            ],
        )
        self.assertEqual(
            pack["source_pack_sha256"],
            "e9785be9ec1f133704bd63cec8c3ee3fb976437020e24f1d69fc9d9e12311ede",
        )
        self.assertEqual(
            source_pack_sha256(pack["manifest"]),
            hashlib.sha256(canonical_source_pack_bytes(pack["manifest"])).hexdigest(),
        )
        serialized = json.dumps(
            [{key: value for key, value in member.items() if key != "bytes"} for member in members],
            ensure_ascii=False,
        )
        self.assertNotIn(str(PACK_ROOT), serialized)
        self.assertNotIn("storage_object_key", serialized)

    def test_tampered_member_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for source in PACK_ROOT.rglob("*"):
                if source.is_file():
                    target = root / source.relative_to(PACK_ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source.read_bytes())
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["logical_sources"][0]["members"][0]["sha256"] = "a" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(EvaluationSourcePackError):
                load_aa_rc_002_source_pack(root)

    def test_manifest_traversal_and_missing_marker_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = {
                "schema_version": "aa-rc-002-source-pack.v1",
                "case_id": "AA-RC-002",
                "fixture_version": "test",
                "hash_algorithm": "SHA-256",
                "logical_sources": [
                    {
                        "logical_name": "bad",
                        "kind": "single_source",
                        "members": [{"name": "../outside.md", "sha256": "a" * 64}],
                    }
                ],
            }
            manifest["source_pack_sha256"] = source_pack_sha256(manifest)
            (root / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            with self.assertRaises(EvaluationSourcePackError):
                load_aa_rc_002_source_pack(root)
