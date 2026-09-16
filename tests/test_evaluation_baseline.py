from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from evaluation_baseline import (
    build_scoreboard,
    build_quality_scoreboard,
    case_execution_capability,
    compare_scoreboards,
    deterministic_grade,
    freeze_baseline_manifest,
    build_evaluation_trace,
    authorized_source_identity_preconditions,
    import_real_cases,
    load_case_document,
    load_evaluation_cases,
    resolve_tool_expectation,
    resolve_case_tool_expectation,
    redact_evaluation_text,
    validate_evaluation_trace,
    validate_evaluation_cases,
    execute_real_case,
    validate_case_evidence_preconditions,
)


class EvaluationBaselineTests(unittest.TestCase):
    def test_aa_rc_002_does_not_accept_unrelated_authorized_uploads(self) -> None:
        case = next(
            case
            for case in load_case_document(
                Path("tests/fixtures/evaluation_baseline/real_cases.json")
            )[1]
            if case["id"] == "AA-RC-002"
        )
        unrelated = {
            "tool_calls": [
                {
                    "name": "inspect_source_of_truth",
                    "metadata": {
                        "source_filename": "JobRequest-(1)_(1)_1789406011025.pdf",
                        "evidence_items": [],
                    },
                },
                {
                    "name": "inspect_source_of_truth",
                    "metadata": {
                        "source_filename": "JobRequest-A_(2)_(3)_1789406011025.pdf",
                        "evidence_items": [],
                    },
                },
            ]
        }
        result = authorized_source_identity_preconditions(
            case=case,
            trace=unrelated,
        )
        self.assertEqual(result["status"], "NOT_DETERMINED")
        self.assertEqual(len(result["missing_sources"]), 2)

    def test_aa_rc_018_runs_upload_matrix_before_separate_my_files_chat(self) -> None:
        case = next(
            case
            for case in load_case_document(
                Path("real-cases-validated.json"),
                require_baseline_size=True,
            )[1]
            if case["id"] == "AA-RC-018"
        )
        upload_calls: list[dict[str, object]] = []
        cleanup_calls: list[str] = []

        def fake_upload(**kwargs: object) -> tuple[int, dict[str, object], dict[str, str]]:
            upload_calls.append(kwargs)
            files = kwargs["files"]
            names = [str(item["filename"]) for item in files]  # type: ignore[index]
            if any("unsafe" in name or ".." in name for name in names):
                return 415, {"error": "UNSAFE_FILENAME"}, {}
            if any("invalid" in name for name in names):
                return 415, {"files": [{"status": "invalid_file_content"}]}, {}
            return (
                201,
                {
                    "files": [
                        {
                            "filename": name,
                            "status": "ready",
                            "source_id": f"source-{index}",
                        }
                        for index, name in enumerate(names)
                    ]
                },
                {},
            )

        def fake_chat(**_: object) -> tuple[int, dict[str, object], dict[str, str]]:
            return (
                200,
                {
                    "reply": (
                        "تم فحص الملفات المرفوعة وعرض حالة كل ملف في نطاق MY_FILES."
                    )
                },
                {},
            )

        async def fake_load(_: str) -> dict[str, object]:
            return {
                "run": {"status": "succeeded", "model_name": "test-model"},
                "tool_events": [],
                "pending_actions": [],
            }

        async def fake_cleanup(*, source_id: str) -> None:
            cleanup_calls.append(source_id)

        with (
            patch(
                "evaluation_baseline._post_file_upload",
                side_effect=fake_upload,
                create=True,
            ),
            patch("evaluation_baseline._post_chat_message", side_effect=fake_chat),
            patch("evaluation_baseline.load_run_evaluation_data", new=fake_load),
            patch(
                "evaluation_baseline.delete_original_source",
                new=fake_cleanup,
                create=True,
            ),
            patch(
                "evaluation_baseline.cleanup_evaluation_run",
                new=AsyncMock(return_value={"audit_events_preserved": 1}),
            ),
        ):
            result = asyncio.run(
                execute_real_case(
                    case,
                    base_url="https://example.test",
                    owner_token="owner-secret",
                    provider="gemini",
                    timeout_seconds=1,
                )
            )

        self.assertEqual(result["upload_e2e"]["status"], "PASS")
        self.assertEqual(result["upload_e2e"]["supported_type_count"], 4)
        self.assertTrue(result["upload_e2e"]["unsafe_filename_rejected"])
        self.assertTrue(result["upload_e2e"]["invalid_content_rejected"])
        self.assertEqual(len(upload_calls), 3)
        self.assertTrue(cleanup_calls)

    def test_aa_rc_026_requires_compound_architecture_and_external_evidence(self) -> None:
        result = validate_case_evidence_preconditions(
            case={
                "id": "AA-RC-026",
                "required_tools": ["web_search"],
            },
            trace={
                "tool_calls": [{"name": "web_search", "status": "success"}],
                "citations": [],
                "sources": [],
                "output": {"answer": "Generic build versus buy advice."},
            },
        )
        self.assertEqual(result["status"], "NOT_DETERMINED")
        self.assertIn("architecture_evidence", result["missing"])
        self.assertIn("verified_external_documentation", result["missing"])
        self.assertIn("answer_citations", result["missing"])
        self.assertIn("tradeoff_criteria", result["missing"])

    def test_aa_rc_026_requires_all_requested_comparison_dimensions_and_openai_sources(
        self,
    ) -> None:
        result = validate_case_evidence_preconditions(
            case={
                "id": "AA-RC-026",
                "required_tools": ["web_search"],
            },
            trace={
                "tool_calls": [
                    {
                        "name": "inspect_architecture_evidence",
                        "status": "success",
                        "metadata": {
                            "evidence_provenance": [
                                {"source_identity": "Ahmed Agent project files"}
                            ]
                        },
                    },
                    {
                        "name": "web_search",
                        "status": "success",
                        "metadata": {
                            "external_source_urls": [
                                "https://example.invalid/not-openai"
                            ],
                            "external_evidence_provenance": [
                                {
                                    "url": "https://example.invalid/not-openai",
                                    "source_identity": "external_web_search",
                                    "verification_status": "UNVERIFIED_EXTERNAL",
                                }
                            ],
                        },
                    },
                ],
                "citations": ["project citation"],
                "sources": ["project citation"],
                "output": {
                    "answer": (
                        "current project state, migration cost, maintenance, quality, "
                        "features, recommendation"
                    )
                },
            },
        )

        self.assertEqual(result["status"], "NOT_DETERMINED")
        self.assertIn("official_openai_documentation", result["missing"])
        self.assertIn("operating_cost", result["missing"])
        self.assertIn("provider_independence", result["missing"])
        self.assertIn("local_self_hosted_fallback", result["missing"])
        self.assertIn("provenance_governance", result["missing"])

    def test_dataset_has_a_valid_seed_shape(self) -> None:
        cases = load_evaluation_cases()
        self.assertGreaterEqual(len(cases), 20)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        self.assertTrue(all(case["provenance"] == "contract_seed" for case in cases))

    def test_imported_real_dataset_has_qualified_provenance(self) -> None:
        version, cases = load_case_document(
            Path("tests/fixtures/evaluation_baseline/real_cases.json"),
            require_baseline_size=True,
        )
        self.assertEqual(version, "ahmed-agent-real-cases-v1")
        self.assertEqual(len(cases), 26)
        self.assertTrue(all(case["provenance"] == "real_case" for case in cases))
        self.assertTrue(all(case["source_reference"] for case in cases))

    def test_semantic_tool_mapping_preserves_case_meaning(self) -> None:
        self.assertEqual(resolve_tool_expectation("my_files")["actual"], "search_my_files")
        self.assertEqual(resolve_tool_expectation("project_file_access")["status"], "unavailable")
        self.assertEqual(
            resolve_case_tool_expectation(
                "AA-RC-016",
                "project_file_access",
            )["actual"],
            "inspect_runtime_evidence",
        )
        self.assertEqual(
            resolve_case_tool_expectation(
                "AA-RC-002",
                "file_access",
            )["actual"],
            "inspect_source_of_truth",
        )
        self.assertEqual(
            resolve_case_tool_expectation(
                "AA-RC-011",
                "project_file_access",
            )["actual"],
            "inspect_runtime_evidence",
        )
        self.assertEqual(
            resolve_case_tool_expectation(
                "AA-RC-007",
                "project_file_access",
            )["actual"],
            "inspect_architecture_evidence",
        )
        self.assertEqual(
            resolve_case_tool_expectation(
                "AA-RC-014",
                "project_file_access",
            )["actual"],
            "inspect_source_status",
        )
        real_cases = load_case_document(
            Path("tests/fixtures/evaluation_baseline/real_cases.json"),
            require_baseline_size=True,
        )[1]
        gap_case_ids = {
            case["id"]
            for case in real_cases
            if not case_execution_capability(case)[
                "executable_by_current_agent_tools"
            ]
        }
        self.assertNotIn("AA-RC-016", gap_case_ids)
        self.assertEqual(
            gap_case_ids,
            set(),
        )
        case = next(
            case for case in load_evaluation_cases()
            if case["id"] == "web-current-official"
        )
        self.assertTrue(case_execution_capability(case)["executable_by_current_agent_tools"])

    def test_deterministic_grade_uses_case_specific_tool_mapping(self) -> None:
        case = next(
            case
            for case in load_case_document(
                Path("tests/fixtures/evaluation_baseline/real_cases.json"),
                require_baseline_size=True,
            )[1]
            if case["id"] == "AA-RC-014"
        )
        trace = {
            "case_id": "AA-RC-014",
            "tool_calls": [{"name": "inspect_source_status"}],
            "sources": ["[source: search_fabric.py, line 215]"],
            "citations": ["[source: search_fabric.py, line 215]"],
            "output": {"answer": "Source evidence was inspected."},
            "approval_requested": False,
            "executed_without_approval": False,
            "abstained": False,
        }
        result = deterministic_grade(case, trace)
        self.assertTrue(result["check_results"]["required_tools"])

    def test_aa_rc_014_citation_contract_checks_tool_evidence_identity(self) -> None:
        case = next(
            case
            for case in load_case_document(
                Path("tests/fixtures/evaluation_baseline/real_cases.json"),
                require_baseline_size=True,
            )[1]
            if case["id"] == "AA-RC-014"
        )
        item = {
            "relative_source_path": "search_fabric.py",
            "file_sha256": "a" * 64,
            "line_start": 10,
            "line_end": 12,
            "verification_status": "VERIFIED",
            "trust_classification": "PROJECT_SOURCE",
        }
        citation = (
            "[source: Ahmed Agent project files; file: search_fabric.py; "
            "lines: 10-12; sha256: " + "a" * 64
            + "; status: VERIFIED; trust: PROJECT_SOURCE]"
        )
        passing = deterministic_grade(
            case,
            {
                "case_id": "AA-RC-014",
                "tool_calls": [{"name": "inspect_source_status"}],
                "sources": [citation],
                "citations": [citation],
                "available_evidence_citations": [citation],
                "available_evidence_items": [item],
                "output": {"answer": "Inspected project evidence."},
                "approval_requested": False,
                "executed_without_approval": False,
                "abstained": False,
            },
        )
        self.assertTrue(passing["check_results"]["evidence_citation_contract"])
        tampered = dict(item, file_sha256="b" * 64)
        failing = deterministic_grade(
            case,
            {
                "case_id": "AA-RC-014",
                "tool_calls": [{"name": "inspect_source_status"}],
                "sources": [citation],
                "citations": [citation],
                "available_evidence_citations": [citation],
                "available_evidence_items": [tampered],
                "output": {"answer": "Inspected project evidence."},
                "approval_requested": False,
                "executed_without_approval": False,
                "abstained": False,
            },
        )
        self.assertFalse(failing["check_results"]["evidence_citation_contract"])

    def test_trace_redacts_credentials_and_captures_hitl_boundary(self) -> None:
        case = next(case for case in load_evaluation_cases() if case["id"] == "web-current-official")
        self.assertNotIn("secret-value", redact_evaluation_text("Bearer secret-value"))
        trace = build_evaluation_trace(
            case=case,
            run_id="run-1",
            scope="WEB",
            provider="gemini",
            response_status=200,
            response_payload={"reply": "Answer [1]"},
            persisted={
                "run": {
                    "status": "succeeded",
                    "model_name": "test-model",
                    "created_at": "2026-09-13T00:00:00+00:00",
                    "finished_at": "2026-09-13T00:00:01+00:00",
                },
                "tool_events": [
                    {
                        "tool_name": "web_search",
                        "status": "success",
                        "duration_ms": 10,
                        "safe_metadata": {"candidate_count": 1},
                    }
                ],
                "pending_actions": [],
            },
            latency_ms=1000,
        )
        validate_evaluation_trace(trace)
        self.assertEqual(trace["execution_status"], "EXECUTED")
        self.assertEqual(trace["tool_calls"][0]["name"], "web_search")

    def test_trace_projects_raw_evidence_items_when_event_citations_are_empty(self) -> None:
        case = next(case for case in load_evaluation_cases() if case["id"] == "web-current-official")
        item = {
            "relative_source_path": "search_fabric.py",
            "file_sha256": "a" * 64,
            "line_start": 10,
            "line_end": 12,
            "extracted_evidence": "verified source",
            "verification_status": "VERIFIED",
            "trust_classification": "PROJECT_SOURCE",
        }
        trace = build_evaluation_trace(
            case=case,
            run_id="run-raw-evidence",
            scope="WEB",
            provider="gemini",
            response_status=200,
            response_payload={"reply": "Answer"},
            persisted={
                "run": {"status": "succeeded", "model_name": "test-model"},
                "tool_events": [
                    {
                        "tool_name": "inspect_runtime_evidence",
                        "status": "success",
                        "duration_ms": 1,
                        "safe_metadata": {
                            "evidence_status": "VERIFIED",
                            "evidence_items": [item],
                            "evidence_citations": [],
                        },
                    }
                ],
                "pending_actions": [],
            },
            latency_ms=1,
        )
        self.assertEqual(len(trace["evidence_provenance"]), 1)
        self.assertEqual(len(trace["available_evidence_citations"]), 1)
        self.assertEqual(trace["evidence_provenance"][0]["file_sha256"], "a" * 64)
        self.assertIn("file: search_fabric.py", trace["evidence_provenance"][0]["citation"])

    def test_malformed_trace_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_evaluation_trace({"case_id": "broken"})

    def test_deterministic_grader_checks_tools_sources_and_schema(self) -> None:
        case = next(case for case in load_evaluation_cases() if case["id"] == "web-current-official")
        result = deterministic_grade(
            case,
            {
                "tool_calls": [{"name": "web_search"}],
                "sources": [{"domain": "python.org"}],
                "citations": ["[1]"],
                "output": {"answer": "The current official result is cited."},
            },
        )
        self.assertEqual(result["deterministic_status"], "PASS")
        self.assertEqual(result["semantic_status"], "NOT_RUN")

    def test_scoreboard_marks_missing_traces_incomplete(self) -> None:
        cases = load_evaluation_cases()
        report = build_scoreboard(cases, {})
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["traced_case_count"], 0)
        self.assertEqual(len(report["missing_case_ids"]), len(cases))

    def test_manifest_contains_no_secret_values(self) -> None:
        manifest = freeze_baseline_manifest()
        serialized = str(manifest)
        self.assertEqual(manifest["dataset_case_count"], 22)
        self.assertEqual(manifest["real_case_count"], 0)
        self.assertIn("category_counts", manifest)
        self.assertNotIn("API_KEY", serialized)
        self.assertNotIn("TOKEN", serialized)
        self.assertFalse(manifest["live_provider_run"])

    def test_quality_scoreboard_requires_real_cases(self) -> None:
        report = build_quality_scoreboard(load_evaluation_cases(), {})
        self.assertEqual(report["status"], "REAL_CASE_DATA_REQUIRED")
        self.assertFalse(report["quality_eligible"])

    def test_real_case_requires_provenance_and_conflicting_tools_fail(self) -> None:
        _, cases = load_case_document(
            Path("tests/fixtures/evaluation_baseline/cases.json"),
            require_baseline_size=True,
        )
        invalid = dict(cases[0])
        invalid["provenance"] = "real_case"
        with self.assertRaises(ValueError):
            validate_evaluation_cases([invalid])

        conflicting = dict(cases[0])
        conflicting["required_tools"] = ["web_search"]
        conflicting["forbidden_tools"] = ["web_search"]
        with self.assertRaises(ValueError):
            validate_evaluation_cases([conflicting])

    def test_real_case_import_requires_documented_source(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "incoming.json"
            output = Path(directory) / "real-cases.json"
            source.write_text(
                '{"dataset_version":"incoming-v1","cases":[{"case_type":"real_case",'
                '"case_id":"real-1","category":"retrieval","input":"Question",'
                '"expected_behavior":"Use the file","source_reference":"chat-export-1",'
                '"required_tools":["search_my_files"],"forbidden_tools":[]}]}\n',
                encoding="utf-8",
            )
            result = import_real_cases(source, output)
            self.assertEqual(result["real_case_count"], 1)
            _, imported = load_case_document(output)
            self.assertEqual(imported[0]["provenance"], "real_case")

    def test_regression_comparison_reports_case_and_category_deltas(self) -> None:
        before = {
            "results": [
                {
                    "case_id": "a",
                    "category": "retrieval",
                    "deterministic_status": "FAIL",
                }
            ]
        }
        after = {
            "quality_eligible": True,
            "results": [
                {
                    "case_id": "a",
                    "category": "retrieval",
                    "deterministic_status": "PASS",
                }
            ]
        }
        before["quality_eligible"] = True
        comparison = compare_scoreboards(before, after)
        self.assertEqual(comparison["status"], "COMPARABLE")
        self.assertEqual(comparison["per_case"][0]["delta"], 1)
        self.assertEqual(comparison["per_category"]["retrieval"]["delta"], 1.0)


if __name__ == "__main__":
    unittest.main()