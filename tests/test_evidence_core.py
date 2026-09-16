from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from evidence_core import (
    EvidenceCoreError,
    EvidenceStatus,
    EvidenceTrust,
    ProjectEvidenceCore,
)
from source_status import inspect_source_status


class EvidenceCoreSecurityTests(unittest.TestCase):
    def _root(self) -> Path:
        root = Path(tempfile.mkdtemp())
        (root / "search_fabric.py").write_text(
            "class SearchProvider: pass\n"
            'API_KEY = "do-not-return"\n'
            'OWNER_TOKEN = os.environ.get("OWNER_TOKEN")\n',
            encoding="utf-8",
        )
        (root / "config.py").write_text(
            "SEARCH_PROVIDER_TIMEOUT_SECONDS = 12\n",
            encoding="utf-8",
        )
        (root / "pyproject.toml").write_text(
            '[project]\nrequires-python = ">=3.13,<3.14"\n',
            encoding="utf-8",
        )
        return root

    def test_allowlisted_read_has_hash_relative_path_and_line_evidence(self) -> None:
        root = self._root()
        try:
            core = ProjectEvidenceCore(
                project_root=root,
                capability_name="test",
                allowlisted_paths=("search_fabric.py",),
            )
            source = core.read_file("search_fabric.py")
            item = core.line_evidence(
                source,
                line_start=1,
                trust_classification=EvidenceTrust.PROJECT_SOURCE,
            )
            self.assertEqual(source.relative_path, "search_fabric.py")
            self.assertEqual(
                source.sha256,
                hashlib.sha256((root / "search_fabric.py").read_bytes()).hexdigest(),
            )
            self.assertEqual(item.line_start, 1)
            self.assertEqual(item.file_sha256, source.sha256)
            self.assertIn("SearchProvider", item.extracted_evidence)
        finally:
            for path in root.iterdir():
                path.unlink()
            root.rmdir()

    def test_rejects_arbitrary_absolute_and_traversal_paths(self) -> None:
        root = self._root()
        try:
            core = ProjectEvidenceCore(
                project_root=root,
                capability_name="test",
                allowlisted_paths=("search_fabric.py",),
            )
            for path in ("/etc/passwd", "../search_fabric.py", "x/../search_fabric.py"):
                with self.assertRaises(EvidenceCoreError):
                    core.read_file(path)
        finally:
            for path in root.iterdir():
                path.unlink()
            root.rmdir()

    def test_denies_secret_names_and_external_symlinks(self) -> None:
        root = self._root()
        outside = root.parent / "evidence-core-outside.txt"
        outside.write_text("outside", encoding="utf-8")
        try:
            with self.assertRaises(EvidenceCoreError):
                ProjectEvidenceCore(
                    project_root=root,
                    capability_name="test",
                    allowlisted_paths=(".env",),
                )
            (root / "search_fabric.py").unlink()
            (root / "search_fabric.py").symlink_to(outside)
            core = ProjectEvidenceCore(
                project_root=root,
                capability_name="test",
                allowlisted_paths=("search_fabric.py",),
            )
            with self.assertRaises(EvidenceCoreError):
                core.read_file("search_fabric.py")
        finally:
            (root / "search_fabric.py").unlink(missing_ok=True)
            for path in root.iterdir():
                path.unlink(missing_ok=True)
            root.rmdir()
            outside.unlink(missing_ok=True)

    def test_read_is_read_only_and_redacts_secret_values(self) -> None:
        root = self._root()
        before = (root / "search_fabric.py").read_bytes()
        try:
            core = ProjectEvidenceCore(
                project_root=root,
                capability_name="test",
                allowlisted_paths=("search_fabric.py",),
            )
            source = core.read_file("search_fabric.py")
            item = core.line_evidence(source, line_start=2)
            self.assertEqual(before, (root / "search_fabric.py").read_bytes())
            self.assertNotIn("do-not-return", item.extracted_evidence)
        finally:
            for path in root.iterdir():
                path.unlink()
            root.rmdir()


class SourceStatusTests(unittest.TestCase):
    def test_real_components_return_structured_provenance_without_operational_claim(self) -> None:
        search = inspect_source_status("search_provider", inspection_id="search-test")
        page = inspect_source_status("page_fetcher", inspection_id="page-test")

        self.assertEqual(search["schema_version"], "project-evidence.v1")
        self.assertEqual(search["evidence_status"], "VERIFIED")
        self.assertEqual(search["inspection_id"], "search-test")
        self.assertGreater(search["extracted_facts"]["evidence_item_count"], 0)
        self.assertTrue(search["evidence_items"][0]["relative_source_path"])
        self.assertTrue(search["evidence_items"][0]["file_sha256"])
        self.assertFalse(search["extracted_facts"]["configuration_declaration"]["facts"]["secret_values_read"])
        self.assertEqual(page["evidence_status"], "VERIFIED")
        self.assertTrue(page["extracted_facts"]["contradictions"])
        self.assertEqual(
            page["extracted_facts"]["operational_status"]["claim"],
            "not_asserted",
        )
        serialized = str(search) + str(page)
        self.assertNotIn("do-not-return", serialized)

    def test_active_content_extractor_uses_tavily_path_not_local_helper(self) -> None:
        page = inspect_source_status("page_fetcher", inspection_id="active-extractor-test")
        wiring = page["extracted_facts"]["registration_wiring"]

        self.assertEqual(page["evidence_status"], "VERIFIED")
        self.assertEqual(wiring["status"], "VERIFIED")
        self.assertEqual(wiring["facts"]["active_call_sites"], [])
        self.assertEqual(wiring["facts"]["active_extraction_alternative"], "Tavily extract")
        self.assertTrue(page["extracted_facts"]["contradictions"])
        self.assertIn(
            "local _fetch_page implementation exists",
            page["extracted_facts"]["contradictions"][0],
        )

    def test_only_fixed_component_identifiers_are_accepted(self) -> None:
        with self.assertRaises(ValueError):
            inspect_source_status("../search_fabric.py")
        with self.assertRaises(ValueError):
            inspect_source_status("repository")


if __name__ == "__main__":
    unittest.main()