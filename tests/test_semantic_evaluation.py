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

# Run artifacts from Replit-era evaluation runs are not committed to the repo.
# Skip this suite until they are regenerated (tracked in the project issues).
_MISSING_ARTIFACTS = [
    name
    for name in ("baseline-real-v6-phase0-complete.json",)
    if not (ROOT / name).exists()
]
if _MISSING_ARTIFACTS:
    raise unittest.SkipTest(
        "Run artifacts not committed (regenerate per project issues): "
        + ", ".join(_MISSING_ARTIFACTS)
    )


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
        # Grading-rule test with a synthetic trace: whether the live model
        # happens to cite sources in a given run is model behavior, not a
        # deterministic rule. The rule itself — a required-source case whose
        # trace carries no citation fails with an explainable reason — is what
        # must be pinned (consistent with the neighboring synthetic tests).
        contract_case = next(
            case for case in self.contract["cases"] if case["case_id"] == "AA-RC-002"
        )
        result = evaluate_case(
            contract_case=contract_case,
            trace={
                "case_id": "AA-RC-002",
                "run_id": "no-citation-run",
                "execution_status": "EXECUTED",
                "tool_calls": [{"name": "search_my_files", "status": "success"}],
                "citations": [],
                "sources": [],
                "output": {"answer": "الملفان متطابقان في البنية العامة ويختلفان في التفاصيل."},
            },
        )
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

    def test_review_driven_failure_appears_in_failure_analysis(self) -> None:
        # apply_reviews() can push semantic_status to FAIL purely from an
        # independent reviewer's scores, with no deterministic dimension
        # ever reaching FAIL on its own. The failure-analysis section is
        # built from deterministic dimension statuses only, so without this
        # case it silently drops every review-driven failure even though
        # counts/overall_status already report it.
        expected = self.packet["cases"][0]
        # evaluate_baseline() injects _artifact_file_sha256 before rebuilding
        # the packet internally, so its packet_sha256 differs from
        # self.packet's (built in setUpClass without that injection). Read
        # the real one back from an unreviewed run instead of assuming it
        # matches self.packet.
        baseline_packet_sha256 = evaluate_baseline(
            contract=self.contract,
            baseline_path=BASELINE_PATH,
        )["review_packet_sha256"]
        review_document = {
            "packet_sha256": baseline_packet_sha256,
            "reviewer_type": "independent_evaluator",
            "reviewer_id": "claude-sonnet-5 (anthropic)",
            "independence_declaration": True,
            "runtime_model_self_grading": False,
            "reviews": [
                {
                    "case_id": expected["case_id"],
                    "reference_fingerprint": expected["reference_fingerprint"],
                    "execution_trace_fingerprint": expected[
                        "execution_trace_fingerprint"
                    ],
                    "scores": {
                        **{dimension: 2 for dimension in DIMENSIONS},
                        "factual_correctness": 1,
                    },
                    "decision": "FAIL",
                    "reason": "The answer never engages the question the case is about.",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            review_path = Path(directory) / "review.json"
            review_path.write_text(
                json.dumps(review_document, ensure_ascii=False),
                encoding="utf-8",
            )
            report = evaluate_baseline(
                contract=self.contract,
                baseline_path=BASELINE_PATH,
                review_path=review_path,
            )
        result = next(
            item for item in report["results"] if item["case_id"] == expected["case_id"]
        )
        self.assertEqual(result["semantic_status"], "FAIL")
        analysis_case_ids = {
            item["case_id"]
            for item in report["deterministic_failure_analysis"]["failures"]
        }
        self.assertIn(expected["case_id"], analysis_case_ids)
        entry = next(
            item
            for item in report["deterministic_failure_analysis"]["failures"]
            if item["case_id"] == expected["case_id"]
        )
        self.assertEqual(entry["classification"], "independent_review_failure")
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

    def test_redacts_bare_unlabeled_api_keys(self) -> None:
        # A credential with no "api_key:"/"token:" label (e.g. leaked inline
        # in a code snippet the model quoted back) must still be caught by
        # its own shape -- otherwise raising final_output_redacted's length
        # limit risks shipping a live secret to an independent reviewer.
        # Fixtures are built by concatenating a vendor prefix with a filler
        # body at runtime, rather than as one literal, so this test file
        # never contains a contiguous credential-shaped string on disk.
        # Mixed-case+digit filler is a valid body for every pattern below
        # except AKIA (digits/uppercase only, checked separately).
        mixed_filler = "aB3dE6fG9hI2jK5lM8nO1pQ4rS7tU0vW"
        upper_filler = "AB3DE6FG9HI2JK5LM8NO1PQ4RS7TU0VW"
        vendor_prefixes = [
            "sk-" + "proj-",
            "sk-" + "ant-api03-",
            "gh" + "p_",
            "github_pat_" + "11",
            "AIza" + "SyD-",
            "xoxb" + "-",
        ]
        for prefix in vendor_prefixes:
            sample = f"here is the key: {prefix}{mixed_filler}"
            redacted = semantic_evaluation._redact_text(sample)
            self.assertIn("[REDACTED]", redacted, sample)
        akia_sample = f"here is the key: {'AKIA'}{upper_filler}"
        self.assertIn(
            "[REDACTED]", semantic_evaluation._redact_text(akia_sample), akia_sample
        )

    def test_redacts_pem_private_keys_and_bare_jwts(self) -> None:
        # Raising final_output_redacted's limit to 16000 means a PEM private
        # key or a bare JWT past character 600 is no longer hidden by
        # truncation alone -- both must be caught by shape. Fixtures are
        # assembled at runtime from markers + filler so no complete
        # credential-shaped literal exists on disk.
        begin_marker = "-----BEGIN " + "PRIVATE KEY-----"
        end_marker = "-----END " + "PRIVATE KEY-----"
        key_body = "A" * 64 + "\n" + "B" * 64
        complete_pem = f"prefix {begin_marker}\n{key_body}\n{end_marker} suffix"
        redacted = semantic_evaluation._redact_text(complete_pem, limit=16000)
        self.assertNotIn("PRIVATE KEY-----\nA", redacted)
        self.assertIn("[REDACTED PRIVATE KEY]", redacted)

        truncated_pem = "x" * 700 + begin_marker + "\n" + "C" * 500
        redacted_truncated = semantic_evaluation._redact_text(truncated_pem, limit=16000)
        self.assertNotIn(begin_marker, redacted_truncated)
        self.assertNotIn("C" * 20, redacted_truncated)

        jwt_header = "eyJ" + "hbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        jwt_payload = "e" * 40
        jwt_signature = "s" * 40
        bare_jwt = f"{jwt_header}.{jwt_payload}.{jwt_signature}"
        redacted_jwt = semantic_evaluation._redact_text(
            f"here is a token {bare_jwt} end", limit=16000
        )
        self.assertNotIn(bare_jwt, redacted_jwt)
        self.assertIn("[REDACTED]", redacted_jwt)

    def test_failure_analysis_merges_coexisting_deterministic_and_review_failures(
        self,
    ) -> None:
        # A case can have BOTH a deterministic dimension FAIL and an
        # independent reviewer's own failing score in a different dimension.
        # The old two-list construction dropped the review's failing
        # dimension/reason entirely whenever a deterministic FAIL already
        # existed for that case; both must now appear in one merged entry.
        evaluation = {
            "case_id": "SYNTHETIC-COEXIST-001",
            "execution_classification": "EXECUTION_OK",
            "semantic_status": "FAIL",
            "dimensions": {
                "groundedness": {
                    "status": "FAIL",
                    "score": 0,
                    "reason": "No citation or source in the trace.",
                    "evidence": [],
                },
                "factual_correctness": {
                    "status": "REVIEW_REQUIRED",
                    "score": None,
                    "reason": "Needs independent review.",
                    "evidence": [],
                },
            },
            "independent_review": {
                "scores": {"completeness": 1},
                "reason": "The answer omits a required success criterion.",
            },
        }
        entries = semantic_evaluation.build_failure_analysis_entries([evaluation])
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertIn("groundedness", entry["dimensions"])
        self.assertIn("completeness", entry["dimensions"])
        self.assertEqual(entry["dimensions"]["completeness"]["score"], 1)
        self.assertEqual(
            entry["classification"],
            "semantic_output_or_evidence_gap_and_independent_review_failure",
        )
        self.assertIn("required success criterion", entry["reason"])

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

    def test_packet_preserves_cited_identity_for_direct_url_citations(self) -> None:
        # A direct-URL citation is redacted to the literal "[URL]" in the
        # packet's own citations/sources fields (see the URL-redaction rule
        # in _redact_text), so an independent reviewer could never rebind
        # a cited identity from that field alone. The packet must carry the
        # identity precomputed from the raw, pre-redaction trace instead.
        trace = {
            "case_id": "AA-RC-TEST-URL",
            "execution_status": "EXECUTED",
            "provider": "gemini",
            "model": "gemini-test",
            "citations": ["https://platform.openai.com/docs/guides/tools-web-search"],
            "sources": ["https://platform.openai.com/docs/guides/tools-web-search"],
            "external_evidence_provenance": [
                {
                    "url": "https://platform.openai.com/docs/guides/tools-web-search",
                    "source_identity": "external_openai_official",
                    "verification_status": "UNVERIFIED_EXTERNAL",
                }
            ],
            "output": {"answer": "Cited directly by URL."},
        }
        packet_trace = semantic_evaluation._packet_trace_for_independent_reviewer(trace)
        self.assertEqual(packet_trace["citations"], ["[URL]"])
        self.assertEqual(
            packet_trace["cited_external_identities"], ["external_openai_official"]
        )

    def test_final_output_redacted_keeps_a_full_answer_for_the_reviewer(self) -> None:
        # final_output_redacted is what an independent reviewer judges the answer
        # against, unlike answer_excerpt_redacted (an intentionally short
        # excerpt). It must not be capped down to excerpt length, or a
        # multi-part answer's conclusion/recommendation is cut before the
        # reviewer ever sees it.
        long_answer = "س" * 4000 + " التوصية النهائية هنا." + "ص" * 100
        trace = {
            "case_id": "AA-RC-TEST",
            "execution_status": "EXECUTED",
            "provider": "gemini",
            "model": "gemini-test",
            "output": {"answer": long_answer},
        }
        packet_trace = semantic_evaluation._packet_trace_for_independent_reviewer(trace)
        self.assertGreater(len(packet_trace["final_output_redacted"]), 600)
        self.assertIn("التوصية النهائية هنا", packet_trace["final_output_redacted"])

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

    def test_aa_rc_015_groundedness_accepts_classified_openai_identity(self) -> None:
        # AA-RC-015's expected_sources are human-readable categories
        # ("official OpenAI documentation") that never appear verbatim in a
        # citation/source string. Before the source-category mapping, a
        # correctly classified external_openai_official citation could never
        # deterministically PASS this case -- it was stuck at
        # REVIEW_REQUIRED even when fully correct.
        contract_case = next(
            case for case in self.contract["cases"] if case["case_id"] == "AA-RC-015"
        )
        trace = {
            "case_id": "AA-RC-015",
            "run_id": "aa-rc-015-grounded-run",
            "execution_status": "EXECUTED",
            "provider": "gemini",
            "model": "gemini-test",
            "tool_calls": [{"name": "web_search", "status": "success"}],
            "citations": ["https://platform.openai.com/docs/guides/tools-web-search"],
            "sources": ["https://platform.openai.com/docs/guides/tools-web-search"],
            "external_evidence_provenance": [
                {
                    "url": "https://platform.openai.com/docs/guides/tools-web-search",
                    "source_identity": "external_openai_official",
                    "verification_status": "UNVERIFIED_EXTERNAL",
                }
            ],
            "output": {"answer": "Materiality-assessed, sourced from official OpenAI docs."},
        }
        result = evaluate_case(contract_case=contract_case, trace=trace)
        self.assertEqual(result["dimensions"]["groundedness"]["status"], "PASS")

    def test_aa_rc_015_groundedness_rejects_unused_official_provenance(self) -> None:
        # A search can retrieve an official OpenAI page without the answer
        # ever citing it -- external_evidence_provenance records everything
        # retrieved, not everything used. Groundedness must not PASS just
        # because an unrelated citation coexists with an unused official
        # record in the same trace.
        contract_case = next(
            case for case in self.contract["cases"] if case["case_id"] == "AA-RC-015"
        )
        trace = {
            "case_id": "AA-RC-015",
            "run_id": "aa-rc-015-unbound-run",
            "execution_status": "EXECUTED",
            "provider": "gemini",
            "model": "gemini-test",
            "tool_calls": [{"name": "web_search", "status": "success"}],
            "citations": ["https://example.test/unrelated-blog-post"],
            "sources": ["https://example.test/unrelated-blog-post"],
            "external_evidence_provenance": [
                {
                    "url": "https://platform.openai.com/docs/guides/tools-web-search",
                    "source_identity": "external_openai_official",
                    "verification_status": "UNVERIFIED_EXTERNAL",
                }
            ],
            "output": {"answer": "Some unrelated claim with a citation."},
        }
        result = evaluate_case(contract_case=contract_case, trace=trace)
        self.assertEqual(result["dimensions"]["groundedness"]["status"], "REVIEW_REQUIRED")

    def test_cited_external_identities_binds_by_index_marker_and_url(self) -> None:
        trace = {
            "citations": ["[2]"],
            "sources": ["https://official.example/openai-doc"],
            "external_evidence_provenance": [
                {"url": "https://unused.example/one", "source_identity": "external_web_search"},
                {"url": "https://official.example/openai-doc", "source_identity": "external_openai_official"},
            ],
        }
        identities = semantic_evaluation.cited_external_identities(trace)
        self.assertEqual(identities, {"external_openai_official"})

    def test_generic_web_expected_source_accepts_any_external_identity(self) -> None:
        # A generic expected-source phrase like "current web sources" should
        # not require the vendor-specific OpenAI identity -- any classified
        # external identity satisfies it.
        self.assertTrue(
            semantic_evaluation.expected_source_matches_identities(
                "current web sources", {"external_web_search"}
            )
        )
        self.assertFalse(
            semantic_evaluation.expected_source_matches_identities(
                "official OpenAI documentation", {"external_web_search"}
            )
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
        # Rule under test: a PASS review cannot override a deterministic
        # groundedness FAIL. Whether a given live baseline case happens to fail
        # deterministically is model behavior, so the failing evaluation is
        # built from a synthetic no-citation trace (as the neighboring tests
        # do); the review itself is bound to the real packet fingerprints.
        expected = self.packet["cases"][0]
        contract_case = next(
            case
            for case in self.contract["cases"]
            if case["case_id"] == expected["case_id"]
        )
        failing_evaluation = evaluate_case(
            contract_case=contract_case,
            trace={
                "case_id": expected["case_id"],
                "run_id": "deterministic-failure-run",
                "execution_status": "EXECUTED",
                "tool_calls": [{"name": "search_my_files", "status": "success"}],
                "citations": [],
                "sources": [],
                "output": {"answer": "إجابة بلا أي استشهاد بالمصادر المطلوبة."},
            },
        )
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
        others = [
            item
            for item in report["results"]
            if item["case_id"] != expected["case_id"]
        ]
        merged = apply_reviews(
            evaluations=[failing_evaluation, *others],
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