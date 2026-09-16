from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import semantic_evaluation
from semantic_evaluation import (
    DIMENSIONS,
    SemanticEvaluationError,
    apply_reviews,
    build_contract,
    build_review_packet,
    build_review_input_template,
    evaluate_baseline,
    evaluate_case,
)


ROOT = Path(__file__).parent.parent
CONTRACT_PATH = ROOT / "semantic-evaluation-contract-v1.json"
BASELINE_PATH = ROOT / "baseline-real-v6-phase0-complete.json"


class SemanticEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = build_contract(
            dataset_path=ROOT / "real-cases-validated.json",
            baseline_path=BASELINE_PATH,
        )
        cls.baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        cls.packet = build_review_packet(
            contract=cls.contract,
            baseline=cls.baseline,
        )

    def test_contract_covers_all_real_cases_and_dimensions(self) -> None:
        self.assertEqual(self.contract["case_count"], 26)
        self.assertEqual(len(self.contract["cases"]), 26)
        for case in self.contract["cases"]:
            self.assertEqual(
                set(case["reference_assertions"]),
                {"case_id", "category", "source_reference", *DIMENSIONS},
            )
            self.assertEqual(len(case["reference_fingerprint"]), 64)

    def test_baseline_evaluation_never_passes_without_independent_review(self) -> None:
        report = evaluate_baseline(
            contract=self.contract,
            baseline_path=BASELINE_PATH,
        )
        self.assertEqual(report["case_count"], 26)
        self.assertGreater(report["counts"]["REVIEW_REQUIRED"], 0)
        self.assertEqual(report["review_applied"], False)
        self.assertNotEqual(report["overall_status"], "PASS")
        self.assertTrue(
            all(
                result["independent_review"] is None
                for result in report["results"]
            )
        )

    def test_trace_and_reference_fingerprints_are_bound_to_each_result(self) -> None:
        report = evaluate_baseline(
            contract=self.contract,
            baseline_path=BASELINE_PATH,
        )
        for result in report["results"]:
            self.assertEqual(len(result["reference_fingerprint"]), 64)
            self.assertEqual(len(result["execution_trace_fingerprint"]), 64)
            self.assertTrue(result["case_id"])
            self.assertTrue(result["run_id"])

    def test_deterministic_citation_failure_is_explainable(self) -> None:
        contract_case = next(
            case for case in self.contract["cases"] if case["case_id"] == "AA-RC-002"
        )
        trace = next(
            trace
            for trace in self.baseline["cases"]
            if trace["case_id"] == "AA-RC-002"
        )
        result = evaluate_case(contract_case=contract_case, trace=trace)
        self.assertEqual(result["dimensions"]["groundedness"]["status"], "FAIL")
        self.assertIn("no citation", result["dimensions"]["groundedness"]["reason"])

    def test_missing_authorized_source_documents_are_not_determinable(self) -> None:
        contract_case = next(
            case for case in self.contract["cases"] if case["case_id"] == "AA-RC-002"
        )
        result = evaluate_case(
            contract_case=contract_case,
            trace={
                "case_id": "AA-RC-002",
                "run_id": "missing-source-documents",
                "execution_status": "EXECUTED",
                "external_input_blocker": {
                    "status": "NOT_DETERMINED",
                    "code": "AUTHORIZED_SOURCE_DOCUMENTS_MISSING",
                },
                "tool_calls": [{"name": "search_my_files", "status": "success"}],
                "citations": [],
                "sources": [],
                "output": {"answer": "The authorized source documents were not available."},
            },
        )
        self.assertEqual(result["execution_classification"], "EXTERNAL_INPUT_BLOCKER")
        self.assertEqual(result["semantic_status"], "NOT_DETERMINED")

    def test_provider_failure_is_not_scored_as_semantic_answer_failure(self) -> None:
        contract_case = self.contract["cases"][0]
        result = evaluate_case(
            contract_case=contract_case,
            trace={
                "case_id": contract_case["case_id"],
                "run_id": "provider-failure-run",
                "execution_status": "EXECUTION_FAILED",
                "execution_error": "PROVIDER_RATE_LIMITED",
                "tool_calls": [],
                "citations": [],
                "sources": [],
                "output": {"answer": ""},
            },
        )
        self.assertEqual(result["execution_classification"], "PROVIDER_FAILURE")
        self.assertEqual(result["semantic_status"], "NOT_DETERMINED")
        self.assertTrue(
            all(
                dimension["status"] == "NOT_DETERMINED"
                for dimension in result["dimensions"].values()
            )
        )
        self.assertEqual(
            result["claim_evidence_support"]["status"],
            "NOT_DETERMINED",
        )
        self.assertEqual(
            result["structured_abstention"]["status"],
            "NOT_DETERMINED",
        )

    def test_structured_abstention_requires_review_and_never_auto_passes(self) -> None:
        contract_case = self.contract["cases"][0]
        result = evaluate_case(
            contract_case=contract_case,
            trace={
                "case_id": contract_case["case_id"],
                "run_id": "abstention-run",
                "execution_status": "EXECUTED",
                "tool_calls": [],
                "citations": [],
                "sources": [],
                "output": {"answer": "لا أستطيع التحقق من ذلك من الأدلة المتاحة."},
            },
        )
        self.assertEqual(
            result["structured_abstention"]["status"],
            "ABSTAINED_WITHOUT_EVIDENCE",
        )
        self.assertEqual(result["semantic_status"], "REVIEW_REQUIRED")
        self.assertNotEqual(result["semantic_status"], "PASS")

    def test_claim_support_is_trace_level_only(self) -> None:
        contract_case = self.contract["cases"][0]
        source = contract_case["reference_assertions"]["groundedness"][
            "expected_sources"
        ][0]
        result = evaluate_case(
            contract_case=contract_case,
            trace={
                "case_id": contract_case["case_id"],
                "run_id": "supported-trace-run",
                "execution_status": "EXECUTED",
                "tool_calls": [],
                "citations": [source],
                "sources": [source],
                "evidence_provenance": [{"citation": source}],
                "output": {"answer": "هذه إجابة مقيدة بالمصدر."},
            },
        )
        self.assertEqual(
            result["claim_evidence_support"]["status"],
            "SUPPORTED_TRACE_LEVEL",
        )
        self.assertEqual(
            result["claim_evidence_support"]["claim_level_entailment"],
            "INDEPENDENT_REVIEW_REQUIRED",
        )
        self.assertNotEqual(result["semantic_status"], "PASS")

    def test_baseline_report_separates_execution_failures_from_semantic_failures(self) -> None:
        altered = copy.deepcopy(self.baseline)
        altered["cases"][0]["execution_status"] = "EXECUTION_FAILED"
        altered["cases"][0]["execution_error"] = "PROVIDER_RATE_LIMITED"
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.json"
            baseline_path.write_text(
                json.dumps(altered, ensure_ascii=False),
                encoding="utf-8",
            )
            report = evaluate_baseline(
                contract=self.contract,
                baseline_path=baseline_path,
            )
        self.assertEqual(report["counts"]["NOT_DETERMINED"], 1)
        self.assertTrue(report["deterministic_failure_analysis"]["execution_failures"])
        self.assertEqual(
            report["deterministic_failure_analysis"]["semantic_failures"],
            report["deterministic_failure_analysis"]["failures"],
        )

    def test_packet_redacts_sensitive_answer_material(self) -> None:
        serialized = json.dumps(self.packet, ensure_ascii=False)
        self.assertNotIn("owner-secret", serialized)
        self.assertNotIn("AHMED_OWNER_TOKEN", serialized)
        for case in self.packet["cases"]:
            self.assertTrue(case["observations"]["sensitive_content_redacted"])

    def test_packet_preserves_bounded_external_provenance_without_verifying_it(self) -> None:
        trace = {
            "case_id": "AA-RC-026",
            "run_id": "external-provenance-run",
            "execution_status": "EXECUTED",
            "provider": "gemini",
            "model": "gemini-test",
            "output": {"answer": "A cited answer."},
            "tool_calls": [{"name": "web_search", "status": "success"}],
            "citations": ["https://docs.example.test/current?tracking=secret"],
            "sources": ["https://docs.example.test/current?tracking=secret"],
            "external_evidence_provenance": [
                {
                    "url": "https://docs.example.test/current?tracking=secret",
                    "title": "Official documentation",
                    "snippet": "Current official guidance.",
                    "source_identity": "external_web_search",
                    "verification_status": "UNVERIFIED_EXTERNAL",
                }
            ],
        }
        packet_trace = semantic_evaluation._packet_trace_for_independent_reviewer(trace)
        self.assertEqual(
            packet_trace["external_evidence_provenance"][0]["url"],
            "https://docs.example.test/current",
        )
        self.assertEqual(
            packet_trace["external_evidence_provenance"][0]["verification_status"],
            "UNVERIFIED_EXTERNAL",
        )

    def test_aa_rc_026_groundedness_accepts_bound_project_and_openai_provenance(self) -> None:
        contract_case = next(
            case for case in self.contract["cases"] if case["case_id"] == "AA-RC-026"
        )
        trace = {
            "case_id": "AA-RC-026",
            "run_id": "aa-rc-026-grounded-run",
            "execution_status": "EXECUTED",
            "provider": "gemini",
            "model": "gemini-test",
            "evidence_preconditions": {"status": "READY"},
            "tool_calls": [
                {"name": "inspect_architecture_evidence", "status": "success"},
                {"name": "web_search", "status": "success"},
            ],
            "citations": [
                "project:architecture_fingerprint",
                "https://platform.openai.com/docs/guides/tools",
            ],
            "sources": [
                "project:architecture_fingerprint",
                "https://platform.openai.com/docs/guides/tools",
            ],
            "evidence_provenance": [
                {
                    "source_identity": "Ahmed Agent current architecture",
                    "verification_status": "VERIFIED",
                }
            ],
            "external_evidence_provenance": [
                {
                    "url": "https://platform.openai.com/docs/guides/tools",
                    "source_identity": "external_openai_official",
                    "verification_status": "UNVERIFIED_EXTERNAL",
                }
            ],
            "output": {"answer": "Bounded comparison with project and external evidence."},
        }
        result = evaluate_case(contract_case=contract_case, trace=trace)
        self.assertEqual(result["dimensions"]["groundedness"]["status"], "PASS")
        self.assertIn(
            "external_openai_official",
            result["dimensions"]["groundedness"]["evidence"],
        )

    def test_review_input_template_is_independent_and_covers_all_cases(self) -> None:
        template = build_review_input_template(self.packet)
        self.assertEqual(template["reviewer_type"], "human")
        self.assertFalse(template["independence_declaration"])
        self.assertFalse(template["runtime_model_self_grading"])
        self.assertEqual(len(template["reviews"]), 26)
        self.assertEqual(
            set(template["reviews"][0]["scores"]),
            set(DIMENSIONS),
        )

    def test_runtime_model_self_grading_is_rejected(self) -> None:
        expected = self.packet["cases"][0]
        review_document = {
            "packet_sha256": self.packet["packet_sha256"],
            "reviews": [
                {
                    "case_id": expected["case_id"],
                    "reference_fingerprint": expected["reference_fingerprint"],
                    "execution_trace_fingerprint": expected[
                        "execution_trace_fingerprint"
                    ],
                    "reviewer_type": "runtime_model",
                    "reviewer_id": "runtime",
                    "independence_declaration": False,
                    "runtime_model_self_grading": True,
                    "scores": {dimension: 4 for dimension in DIMENSIONS},
                    "decision": "PASS",
                    "reason": "This is not an independent review.",
                }
            ],
        }
        report = evaluate_baseline(
            contract=self.contract,
            baseline_path=BASELINE_PATH,
        )
        with self.assertRaises(SemanticEvaluationError):
            apply_reviews(
                evaluations=report["results"],
                packet=self.packet,
                review_document=review_document,
            )

    def test_partial_independent_review_does_not_pass_unreviewed_cases(self) -> None:
        expected = self.packet["cases"][0]
        review_document = {
            "packet_sha256": self.packet["packet_sha256"],
            "reviewer_type": "human",
            "reviewer_id": "reviewer-1",
            "independence_declaration": True,
            "runtime_model_self_grading": False,
            "reviews": [
                {
                    "case_id": expected["case_id"],
                    "reference_fingerprint": expected["reference_fingerprint"],
                    "execution_trace_fingerprint": expected[
                        "execution_trace_fingerprint"
                    ],
                    "scores": {dimension: 3 for dimension in DIMENSIONS},
                    "decision": "PASS",
                    "reason": "Independent review found the response acceptable.",
                    "evidence_notes": "Compared with the reference assertions.",
                }
            ],
        }
        report = evaluate_baseline(
            contract=self.contract,
            baseline_path=BASELINE_PATH,
        )
        merged = apply_reviews(
            evaluations=report["results"],
            packet=self.packet,
            review_document=review_document,
        )
        first = next(item for item in merged if item["case_id"] == expected["case_id"])
        self.assertEqual(first["semantic_status"], "FAIL")
        self.assertEqual(
            first["dimensions"]["groundedness"]["status"],
            "FAIL",
        )
        self.assertGreater(
            sum(item["semantic_status"] == "REVIEW_REQUIRED" for item in merged),
            0,
        )

    def test_review_fingerprint_tampering_fails_closed(self) -> None:
        expected = self.packet["cases"][0]
        review_document = {
            "packet_sha256": self.packet["packet_sha256"],
            "reviews": [
                {
                    "case_id": expected["case_id"],
                    "reference_fingerprint": "0" * 64,
                    "execution_trace_fingerprint": expected[
                        "execution_trace_fingerprint"
                    ],
                    "reviewer_type": "human",
                    "reviewer_id": "reviewer-1",
                    "independence_declaration": True,
                    "scores": {dimension: 3 for dimension in DIMENSIONS},
                    "decision": "PASS",
                    "reason": "Independent review found the response acceptable.",
                }
            ],
        }
        report = evaluate_baseline(
            contract=self.contract,
            baseline_path=BASELINE_PATH,
        )
        with self.assertRaises(SemanticEvaluationError):
            apply_reviews(
                evaluations=report["results"],
                packet=self.packet,
                review_document=review_document,
            )


if __name__ == "__main__":
    unittest.main()