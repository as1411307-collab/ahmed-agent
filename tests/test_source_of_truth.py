from __future__ import annotations

import hashlib
import unittest

from my_files import build_chunks, extract_document
from source_of_truth import inspect_source_of_truth
import source_of_truth


SOURCE_ID = "11111111-1111-4111-8111-111111111111"


class FakeOriginalSourceStore:
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result
        self.requested: list[tuple[str, str | None]] = []

    async def read(
        self,
        *,
        source_id: str,
        owner_principal_id: str | None,
    ) -> dict[str, object]:
        self.requested.append((source_id, owner_principal_id))
        return self.result


def _authorized_record(data: bytes) -> dict[str, object]:
    return {
        "status": "AUTHORIZED",
        "source": {
            "source_id": SOURCE_ID,
            "source_version": 1,
            "original_filename": "master.md",
            "mime_type": "text/markdown",
            "byte_size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "original_available": True,
            "extraction_status": "ready",
        },
        "data": data,
    }


class SourceOfTruthTests(unittest.IsolatedAsyncioTestCase):
    async def test_verified_original_returns_bounded_provenance_and_redacts_secret(self) -> None:
        data = b"# Master\nTruth from original.\nAPI_KEY = \"must-not-return\"\n"
        fake = FakeOriginalSourceStore(_authorized_record(data))
        original = source_of_truth.original_source_store
        source_of_truth.original_source_store = fake
        try:
            result = await inspect_source_of_truth(
                SOURCE_ID,
                owner_principal_id="owner",
            )
        finally:
            source_of_truth.original_source_store = original

        self.assertEqual(result["evidence_status"], "VERIFIED")
        facts = result["extracted_facts"]
        self.assertEqual(facts["source_id"], SOURCE_ID)
        self.assertEqual(facts["original_integrity_status"], "VERIFIED")
        self.assertTrue(result["evidence_citations"])
        self.assertIn("Ahmed Agent authorized original sources", result["evidence_citations"][0])
        serialized = str(result)
        self.assertNotIn("must-not-return", serialized)
        self.assertNotIn("storage_object_key", serialized)

    async def test_wrong_owner_is_denied_without_source_metadata(self) -> None:
        fake = FakeOriginalSourceStore({"status": "DENIED"})
        original = source_of_truth.original_source_store
        source_of_truth.original_source_store = fake
        try:
            result = await inspect_source_of_truth(
                SOURCE_ID,
                owner_principal_id="wrong-owner",
            )
        finally:
            source_of_truth.original_source_store = original

        self.assertEqual(result["evidence_status"], "DENIED")
        self.assertEqual(result["evidence_items"], [])
        self.assertNotIn("original_filename", result["extracted_facts"])

    async def test_hash_mismatch_is_discrepancy_and_has_no_verified_evidence(self) -> None:
        data = b"original"
        record = _authorized_record(data)
        record["source"] = {
            **record["source"],
            "sha256": "a" * 64,
        }
        fake = FakeOriginalSourceStore(record)
        original = source_of_truth.original_source_store
        source_of_truth.original_source_store = fake
        try:
            result = await inspect_source_of_truth(
                SOURCE_ID,
                owner_principal_id="owner",
            )
        finally:
            source_of_truth.original_source_store = original

        self.assertEqual(result["evidence_status"], "DISCREPANCY")
        self.assertEqual(result["evidence_items"], [])
        self.assertEqual(
            result["extracted_facts"]["original_integrity_status"],
            "INTEGRITY_DISCREPANCY",
        )

    async def test_unknown_and_invalid_ids_do_not_read_files(self) -> None:
        fake = FakeOriginalSourceStore({"status": "NOT_FOUND"})
        original = source_of_truth.original_source_store
        source_of_truth.original_source_store = fake
        try:
            unknown = await inspect_source_of_truth(
                SOURCE_ID,
                owner_principal_id="owner",
            )
            invalid = await inspect_source_of_truth(
                "../secrets.txt",
                owner_principal_id="owner",
            )
        finally:
            source_of_truth.original_source_store = original

        self.assertEqual(unknown["evidence_status"], "NOT_FOUND")
        self.assertEqual(invalid["evidence_status"], "NOT_FOUND")
        self.assertEqual(fake.requested, [(SOURCE_ID, "owner")])

    def test_chunks_preserve_source_identity_and_hash(self) -> None:
        data = b"source content"
        pages = extract_document("source.md", "text/markdown", data)
        chunks = build_chunks(
            document_id="doc",
            filename="source.md",
            mime_type="text/markdown",
            file_hash=hashlib.sha256(data).hexdigest(),
            source_type="upload",
            pages=pages,
            source_id=SOURCE_ID,
        )
        self.assertEqual(chunks[0].metadata["source_id"], SOURCE_ID)
        self.assertEqual(
            chunks[0].metadata["file_hash"],
            hashlib.sha256(data).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()