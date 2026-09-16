from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from runtime_evidence import (
    RUNTIME_EVIDENCE_FILES,
    RuntimeEvidenceError,
    inspect_runtime_evidence,
)


def _write_fixture(
    root: Path,
    *,
    server: str = "",
    pyproject: str = "",
    replit: str = "",
) -> None:
    (root / "server.py").write_text(server, encoding="utf-8")
    (root / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    (root / ".replit").write_text(replit, encoding="utf-8")


class RuntimeEvidenceTests(unittest.TestCase):
    def test_runtime_evidence_references_are_citation_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(
                root,
                server="from starlette.applications import Starlette\n",
                pyproject='[project]\nrequires-python = ">=3.13,<3.14"\n',
                replit='run = ["uv", "run", "server.py"]\n',
            )
            result = inspect_runtime_evidence(project_root=root)

        references = result["evidence_references"]
        self.assertTrue(references)
        for reference in references:
            self.assertRegex(reference["relative_source_path"], r"^(server\.py|pyproject\.toml|\.replit)$")
            self.assertEqual(len(reference["file_sha256"]), 64)
            self.assertGreaterEqual(reference["line_start"], 1)
            self.assertGreaterEqual(reference["line_end"], reference["line_start"])
            self.assertTrue(reference["extracted_evidence"])
            self.assertEqual(reference["verification_status"], "VERIFIED")
            self.assertTrue(reference["trust_classification"])

    def test_discovers_runtime_from_fixture_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(
                root,
                server=(
                    "from mcp.server.mcpserver import MCPServer\n"
                    "from starlette.applications import Starlette\n"
                    "import os\n"
                    "import uvicorn\n"
                    'uvicorn.run(app, port=int(os.environ.get("PORT", "9100")))\n'
                ),
                pyproject='[project]\nrequires-python = ">=3.12,<3.13"\n',
                replit=(
                    'run = ["legacy", "legacy.py"]\n'
                    "[deployment]\n"
                    'run = ["python", "server.py"]\n'
                    "[[ports]]\nlocalPort = 9100\nexternalPort = 80\n"
                ),
            )
            result = inspect_runtime_evidence(project_root=root)

        self.assertEqual(result["status"], "verified")
        claims = result["runtime_evidence"]
        self.assertEqual(claims["language"]["value"], "Python")
        self.assertEqual(
            claims["language_version"]["value"],
            "Python (>=3.12,<3.13)",
        )
        self.assertEqual(
            claims["server_runtime_command"]["value"],
            "python server.py",
        )
        self.assertEqual(claims["active_entrypoint"]["value"], "server.py")
        self.assertEqual(result["selected_runtime_command_source"], "deployment")
        self.assertEqual(
            {candidate["source"] for candidate in result["runtime_command_candidates"]},
            {"root", "deployment"},
        )
        self.assertEqual(
            claims["http_framework"]["value"],
            "Starlette + Uvicorn",
        )
        self.assertTrue(claims["mcp_presence"]["value"])
        self.assertEqual(claims["configured_port"]["value"], 9100)
        self.assertTrue(
            all(
                reference["file"] in RUNTIME_EVIDENCE_FILES
                for reference in result["evidence_references"]
            )
        )

    def test_runtime_evidence_is_read_only_and_does_not_return_file_contents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(
                root,
                server=(
                    "from starlette.applications import Starlette\n"
                    'UNRELATED_FULL_FILE_CONTENT = "not evidence"\n'
                ),
                pyproject='[project]\nrequires-python = ">=3.13,<3.14"\n',
                replit='run = ["uv", "run", "server.py"]\n',
            )
            before = {
                name: (root / name).read_bytes()
                for name in RUNTIME_EVIDENCE_FILES
            }
            result = inspect_runtime_evidence(project_root=root)
            after = {
                name: (root / name).read_bytes()
                for name in RUNTIME_EVIDENCE_FILES
            }

        self.assertEqual(before, after)
        self.assertIn("from starlette", str(result))
        self.assertNotIn("UNRELATED_FULL_FILE_CONTENT", str(result))

    def test_rejects_arbitrary_paths_and_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            (root / "secret.txt").write_text("do not read", encoding="utf-8")
            with self.assertRaises(RuntimeEvidenceError):
                from runtime_evidence import _safe_evidence_path

                _safe_evidence_path(root, "../secret.txt")
            with self.assertRaises(RuntimeEvidenceError):
                _safe_evidence_path(root, "secret.txt")

    def test_rejects_symlinked_allowlisted_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            outside = root.parent / "runtime-evidence-secret.txt"
            outside.write_text("secret", encoding="utf-8")
            (root / "server.py").unlink()
            (root / "server.py").symlink_to(outside)
            try:
                with self.assertRaises(RuntimeEvidenceError):
                    inspect_runtime_evidence(project_root=root)
            finally:
                outside.unlink(missing_ok=True)

    def test_missing_runtime_evidence_is_partial_or_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(
                root,
                server="",
                pyproject='[project]\nrequires-python = ">=3.13,<3.14"\n',
                replit="[deployment]\n",
            )
            (root / "server.py").unlink()
            result = inspect_runtime_evidence(project_root=root)

        self.assertIn(result["status"], {"partial", "unverified"})
        self.assertEqual(
            result["runtime_evidence"]["http_framework"]["status"],
            "unverified",
        )

    def test_does_not_return_secret_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(
                root,
                server=(
                    "import os\n"
                    'OWNER_TOKEN = os.environ.get("OWNER_TOKEN")\n'
                    'uvicorn.run(app, port=int(os.environ.get("PORT", "8000")))\n'
                ),
                pyproject='[project]\nrequires-python = ">=3.13,<3.14"\n',
                replit='run = ["uv", "run", "server.py"]\n',
            )
            result = inspect_runtime_evidence(project_root=root)

        self.assertNotIn("OWNER_TOKEN", str(result))
        self.assertNotIn("os.environ.get(", str(result))

    def test_conflicting_runtime_sources_do_not_guess_an_unallowlisted_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(
                root,
                server="from starlette.applications import Starlette\n",
                pyproject='[project]\nrequires-python = ">=3.13,<3.14"\n',
                replit=(
                    'run = ["uv", "run", "server.py"]\n'
                    "[deployment]\n"
                    'run = ["python", "other.py"]\n'
                ),
            )
            result = inspect_runtime_evidence(project_root=root)

        self.assertEqual(result["selected_runtime_command_source"], "deployment")
        self.assertEqual(
            result["runtime_evidence"]["server_runtime_command"]["value"],
            "python other.py",
        )
        self.assertEqual(
            result["runtime_evidence"]["active_entrypoint"]["value"],
            "other.py",
        )
        self.assertEqual(
            result["runtime_evidence"]["active_entrypoint"]["status"],
            "unverified",
        )
        self.assertEqual(result["status"], "partial")


if __name__ == "__main__":
    unittest.main()