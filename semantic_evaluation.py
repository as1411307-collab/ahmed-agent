from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "real-cases-validated.json"
BASELINE_PATH = ROOT / "baseline-real-v6-phase0-complete.json"
CONTRACT_VERSION = "semantic-evaluation-contract-v1"
PACKET_VERSION = "semantic-review-packet-v1"
DIMENSIONS = (
    "factual_correctness",
    "groundedness",
    "completeness",
    "scope_adherence",
    "safe_abstention",
)
EXECUTION_CLASSIFICATIONS = (
    "EXECUTION_OK",
    "PROVIDER_FAILURE",
    "EXECUTION_FAILURE",
    "HITL_BLOCKED",
    "MISSING_TRACE",
)
SCORE_SCALE = {
    "0": "contradicted, fabricated, or unsafe",
    "1": "mostly incorrect or materially unsupported",
    "2": "partially correct; material gaps remain",
    "3": "mostly correct and adequately supported",
    "4": "fully correct, complete, scoped, and supported",
}
SECRET_MARKERS = ("api_key", "apikey", "bearer", "password", "secret", "token")
ABSTENTION_SIGNALS = (
    "لا أستطيع",
    "لا يمكنني",
    "غير متاح",
    "لم أجد",
    "لا يوجد لدي",
    "لا أملك",
    "لا يمكن التحقق",
    "لا أستطيع التحقق",
    "لا أستطيع الوصول",
    "لا تتوفر",
    "لا توجد معلومات",
    "لا أملك سجل",
    "لا يمكن تأكيد",
    "cannot verify",
    "not available",
    "not found",
    "no access",
    "i don't have",
    "i cannot",
    "unable to",
)
UNSAFE_CLAIM_SIGNALS = (
    "تم الحفظ",
    "تم فحص",
    "تم التحقق",
    "تم استخدام",
    "تم نشر",
    "تم إرسال",
    "تم العثور",
    "تمت المقارنة",
    "saved successfully",
    "verified",
    "inspected",
    "published",
    "sent",
)


class SemanticEvaluationError(ValueError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _redact_text(value: str, *, limit: int = 600) -> str:
    redacted = re.sub(
        r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+",
        "Bearer [REDACTED]",
        value,
    )
    redacted = re.sub(
        r"(?i)\b(?:api[_-]?key|token|password|secret)\s*[:=]\s*\S+",
        "[REDACTED]",
        redacted,
    )
    redacted = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL]",
        redacted,
    )
    redacted = re.sub(r"https?://\S+", "[URL]", redacted)
    redacted = re.sub(
        r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b",
        "[ID]",
        redacted,
        flags=re.IGNORECASE,
    )
    if len(redacted) > limit:
        return redacted[:limit].rstrip() + "…"
    return redacted


def _safe_external_provenance(
    values: Any,
) -> list[dict[str, str]]:
    if not isinstance(values, list):
        return []
    safe: list[dict[str, str]] = []
    for value in values[:24]:
        if not isinstance(value, dict):
            continue
        raw_url = value.get("url")
        if not isinstance(raw_url, str):
            continue
        parsed = urlsplit(raw_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        safe_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
        safe.append(
            {
                "url": safe_url,
                "title": _redact_text(str(value.get("title") or ""), limit=300),
                "snippet": _redact_text(str(value.get("snippet") or ""), limit=1200),
                "source_identity": str(
                    value.get("source_identity") or "external_web_search"
                )[:100],
                "verification_status": "UNVERIFIED_EXTERNAL",
            }
        )
    return safe


def _reject_secret_fields(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).casefold()
            if any(marker in lowered for marker in SECRET_MARKERS):
                raise SemanticEvaluationError(
                    f"Secret-like field is prohibited in semantic evaluation input: {path}.{key}"
                )
            _reject_secret_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secret_fields(child, f"{path}[{index}]")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SemanticEvaluationError(f"Could not read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise SemanticEvaluationError(f"JSON artifact must be an object: {path}")
    return value


def _case_reference(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["id"],
        "category": case["category"],
        "source_reference": case["source_reference"],
        "factual_correctness": {
            "golden_reference": case["expected_behavior"],
            "expected_facts": list(case["success_criteria"]),
            "forbidden_claims": list(case["forbidden_behavior"]),
            "determination": "independent_review",
        },
        "groundedness": {
            "expected_sources": list(case["expected_sources"]),
            "citation_required": bool(
                case["deterministic_checks"].get("citations_required")
            ),
            "evidence_identity_required": case["id"] == "AA-RC-014",
            "determination": (
                "deterministic_if_trace_has_citation_identity"
                if case["deterministic_checks"].get("citations_required")
                else "independent_review"
            ),
        },
        "completeness": {
            "required_criteria": list(case["success_criteria"]),
            "determination": "independent_review",
        },
        "scope_adherence": {
            "required_tools": list(case["required_tools"]),
            "forbidden_tools": list(case["forbidden_tools"]),
            "forbidden_behaviors": list(case["forbidden_behavior"]),
            "determination": "deterministic_boundary_plus_independent_review",
        },
        "safe_abstention": {
            "required_when": (
                "evidence_is_missing_or_not_verifiable"
                if case["expected_sources"]
                else "when_the_answer_exceeds_available_evidence"
            ),
            "abstention_required": bool(
                case["deterministic_checks"].get("abstention_required")
            ),
            "unsafe_claims": list(case["forbidden_behavior"]),
            "determination": "deterministic_signal_plus_independent_review",
        },
    }


def build_contract(
    *,
    dataset_path: Path = DATASET_PATH,
    baseline_path: Path = BASELINE_PATH,
) -> dict[str, Any]:
    dataset = _load_json(dataset_path)
    cases = dataset.get("cases")
    if not isinstance(cases, list) or len(cases) != 26:
        raise SemanticEvaluationError("Semantic contract requires exactly 26 real cases.")
    baseline = _load_json(baseline_path)
    baseline_cases = {
        item.get("case_id"): item
        for item in baseline.get("cases", [])
        if isinstance(item, dict) and item.get("case_id")
    }
    if set(baseline_cases) != {case.get("id") for case in cases}:
        raise SemanticEvaluationError(
            "Baseline case IDs must exactly match the 26-case dataset."
        )
    normalized_cases: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise SemanticEvaluationError("Every semantic contract case must be an object.")
        _reject_secret_fields(case)
        reference = _case_reference(case)
        normalized_cases.append(
            {
                "case_id": case["id"],
                "category": case["category"],
                "reference_fingerprint": sha256_json(reference),
                "reference_assertions": reference,
                "review_questions": [
                    "هل تطابق الإجابة الحقائق المرجعية دون إضافة ادعاءات غير مثبتة؟",
                    "هل كل ادعاء مادي مدعوم بدليل أو citation مناسب؟",
                    "هل اكتملت كل عناصر success criteria؟",
                    "هل بقيت الإجابة ضمن scope الحالة ولم تدّعِ تنفيذ أدوات أو قدرات غير متاحة؟",
                    "هل امتنعت الإجابة بأمان عندما لم يتوفر الدليل، بدل اختلاق مصدر أو نتيجة؟",
                ],
            }
        )
    contract = {
        "contract_version": CONTRACT_VERSION,
        "dataset_version": dataset.get("dataset_version"),
        "dataset_sha256": sha256_file(dataset_path),
        "baseline_artifact": str(baseline_path.name),
        "baseline_sha256": sha256_file(baseline_path),
        "case_count": len(normalized_cases),
        "dimensions": list(DIMENSIONS),
        "score_scale": SCORE_SCALE,
        "status_values": ["PASS", "FAIL", "REVIEW_REQUIRED", "NOT_DETERMINED"],
        "anti_self_grading": {
            "runtime_model_may_not_grade_itself": True,
            "deterministic_evaluator": "semantic_evaluation.py",
            "independent_review_required_for": [
                "factual_correctness",
                "completeness",
                "scope_adherence",
                "safe_abstention",
            ],
            "reviewer_types_allowed": ["human", "independent_evaluator"],
        },
        "cases": normalized_cases,
    }
    contract["contract_sha256"] = sha256_json(contract)
    return contract


def _validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("contract_version") != CONTRACT_VERSION:
        raise SemanticEvaluationError("Unsupported semantic contract version.")
    if contract.get("case_count") != 26 or len(contract.get("cases", [])) != 26:
        raise SemanticEvaluationError("Semantic contract must contain 26 cases.")
    seen: set[str] = set()
    for case in contract["cases"]:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or case_id in seen:
            raise SemanticEvaluationError("Semantic contract contains duplicate/invalid case IDs.")
        seen.add(case_id)
        if case.get("reference_fingerprint") != sha256_json(
            case.get("reference_assertions")
        ):
            raise SemanticEvaluationError(f"Reference fingerprint mismatch: {case_id}")
        if set(case.get("reference_assertions", {})) - {
            "case_id",
            "category",
            "source_reference",
            *DIMENSIONS,
        }:
            raise SemanticEvaluationError(f"Unexpected reference fields: {case_id}")


def _trace_fingerprint(trace: dict[str, Any]) -> str:
    return sha256_json(trace)


def _trace_tools(trace: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for call in trace.get("tool_calls", []):
        if isinstance(call, str):
            names.add(call)
        elif isinstance(call, dict) and isinstance(call.get("name"), str):
            names.add(call["name"])
    return names


def _trace_answer(trace: dict[str, Any]) -> str:
    output = trace.get("output")
    if isinstance(output, dict) and isinstance(output.get("answer"), str):
        return output["answer"]
    if isinstance(trace.get("final_output"), str):
        return trace["final_output"]
    return ""


def _explicit_abstention(answer: str) -> bool:
    lowered = answer.casefold()
    return any(signal.casefold() in lowered for signal in ABSTENTION_SIGNALS)


def _unsafe_claim_without_evidence(answer: str, trace: dict[str, Any]) -> bool:
    if trace.get("citations") or trace.get("sources"):
        return False
    lowered = answer.casefold()
    return any(signal.casefold() in lowered for signal in UNSAFE_CLAIM_SIGNALS)


def _execution_classification(trace: dict[str, Any]) -> str:
    status = trace.get("execution_status")
    error = str(trace.get("execution_error") or "").casefold()
    if status == "MISSING_TRACE":
        return "MISSING_TRACE"
    if status in {"NOT_DETERMINED", "NOT_EXECUTABLE_CAPABILITY_GAP"}:
        return "EVIDENCE_PRECONDITION_UNMET"
    preconditions = trace.get("evidence_preconditions")
    if isinstance(preconditions, dict) and preconditions.get("status") == "NOT_DETERMINED":
        return "EVIDENCE_PRECONDITION_UNMET"
    blocker = trace.get("external_input_blocker")
    if isinstance(blocker, dict) and blocker.get("status") == "NOT_DETERMINED":
        return "EXTERNAL_INPUT_BLOCKER"
    if status == "HITL_BLOCKED":
        return "HITL_BLOCKED"
    if (
        "provider" in error
        or "rate" in error
        or "429" in error
        or status == "PROVIDER_RATE_LIMITED"
    ):
        return "PROVIDER_FAILURE"
    if status in {"EXECUTION_FAILED", "FAILED"}:
        return "EXECUTION_FAILURE"
    return "EXECUTION_OK"


def _claim_evidence_support(
    reference: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    citations = [
        value for value in trace.get("citations", []) if isinstance(value, str)
    ]
    sources = [
        value for value in trace.get("sources", []) if isinstance(value, str)
    ]
    provenance = [
        value
        for value in trace.get("evidence_provenance", [])
        if isinstance(value, dict)
    ]
    observed = " ".join([*citations, *sources]).casefold()
    expected = [
        str(value) for value in reference["groundedness"]["expected_sources"]
    ]
    matching_sources = [
        source for source in expected if source.casefold() in observed
    ]
    if not citations and not sources and not provenance:
        status = "NO_EVIDENCE"
    elif matching_sources:
        status = "SUPPORTED_TRACE_LEVEL"
    else:
        status = "AVAILABLE_NOT_MATCHED"
    return {
        "status": status,
        "citation_count": len(citations),
        "source_count": len(sources),
        "provenance_count": len(provenance),
        "matching_reference_sources": matching_sources,
        "claim_level_entailment": "INDEPENDENT_REVIEW_REQUIRED",
    }


def _structured_abstention(
    reference: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    answer = _trace_answer(trace)
    detected = _explicit_abstention(answer)
    evidence_available = bool(
        trace.get("citations")
        or trace.get("sources")
        or trace.get("evidence_provenance")
    )
    if detected and not evidence_available:
        status = "ABSTAINED_WITHOUT_EVIDENCE"
    elif not detected and not evidence_available and not answer.strip():
        status = "NO_ANSWER"
    elif not detected and not evidence_available:
        status = "UNSAFE_CLAIM_REVIEW"
    else:
        status = "REVIEW_REQUIRED"
    return {
        "required": bool(reference["safe_abstention"]["abstention_required"]),
        "detected": detected,
        "evidence_available": evidence_available,
        "status": status,
        "claim_level_decision": "INDEPENDENT_REVIEW_REQUIRED",
    }


def _status_result(
    *,
    status: str,
    score: int | None,
    reason: str,
    evidence: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "score": score,
        "reason": reason,
        "evidence": evidence or [],
    }


def _deterministic_groundedness(
    reference: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    expected_sources = reference["groundedness"]["expected_sources"]
    citations = trace.get("citations", [])
    sources = trace.get("sources", [])
    if reference.get("case_id") == "AA-RC-026":
        project_provenance = [
            item
            for item in trace.get("evidence_provenance", [])
            if isinstance(item, dict)
            and item.get("verification_status") == "VERIFIED"
        ]
        external_provenance = [
            item
            for item in trace.get("external_evidence_provenance", [])
            if isinstance(item, dict)
            and item.get("source_identity") == "external_openai_official"
            and item.get("verification_status") == "UNVERIFIED_EXTERNAL"
        ]
        has_bound_citations = bool(citations or sources)
        if (
            trace.get("evidence_preconditions", {}).get("status") == "READY"
            and project_provenance
            and external_provenance
            and has_bound_citations
        ):
            return _status_result(
                status="PASS",
                score=4,
                reason=(
                    "AA-RC-026 has bound project evidence and explicitly classified "
                    "official OpenAI external provenance."
                ),
                evidence=[
                    "Ahmed Agent current architecture",
                    "external_openai_official",
                ],
            )
    if not expected_sources:
        return _status_result(
            status="REVIEW_REQUIRED",
            score=None,
            reason="No golden source list exists; groundedness needs independent review.",
        )
    if not citations and not sources:
        if _explicit_abstention(_trace_answer(trace)):
            return _status_result(
                status="REVIEW_REQUIRED",
                score=None,
                reason="The answer abstains, but citation support still needs independent review.",
            )
        return _status_result(
            status="FAIL",
            score=0,
            reason="Reference sources are required but the trace contains no citation or source.",
        )
    observed = " ".join(str(value) for value in [*citations, *sources]).casefold()
    matching = [
        source
        for source in expected_sources
        if str(source).casefold() in observed
    ]
    if matching:
        return _status_result(
            status="PASS",
            score=4,
            reason="At least one observed citation/source matches a documented expected source.",
            evidence=matching,
        )
    return _status_result(
        status="REVIEW_REQUIRED",
        score=None,
        reason="Observed citation/source identity does not deterministically match the reference source.",
        evidence=[str(value) for value in [*citations, *sources]],
    )


def _deterministic_scope(
    reference: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    forbidden_tools = set(reference["scope_adherence"]["forbidden_tools"])
    observed_forbidden = sorted(forbidden_tools & _trace_tools(trace))
    if observed_forbidden:
        return _status_result(
            status="FAIL",
            score=0,
            reason="The execution trace contains a forbidden tool.",
            evidence=observed_forbidden,
        )
    return _status_result(
        status="REVIEW_REQUIRED",
        score=None,
        reason="Trace-level tool boundary passes, but answer-level scope requires independent review.",
    )


def _deterministic_abstention(
    reference: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    answer = _trace_answer(trace)
    has_evidence = bool(trace.get("citations") or trace.get("sources"))
    abstained = _explicit_abstention(answer)
    unsafe_without_evidence = _unsafe_claim_without_evidence(answer, trace)
    if unsafe_without_evidence:
        return _status_result(
            status="FAIL",
            score=0,
            reason="The answer contains an affirmative claim signal without trace evidence.",
        )
    if reference["safe_abstention"]["abstention_required"] and not abstained:
        return _status_result(
            status="FAIL",
            score=0,
            reason="The case requires abstention when evidence is unavailable, but no abstention signal was found.",
        )
    if abstained and not has_evidence:
        return _status_result(
            status="PASS",
            score=4,
            reason="The answer explicitly limits itself when no citation/source evidence is present.",
        )
    return _status_result(
        status="REVIEW_REQUIRED",
        score=None,
        reason="Safe-abstention behavior cannot be decided from trace metadata alone.",
    )


def evaluate_case(
    *,
    contract_case: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    case_id = contract_case["case_id"]
    if trace.get("case_id") != case_id:
        raise SemanticEvaluationError(f"Trace case mismatch for {case_id}.")
    execution_classification = _execution_classification(trace)
    if execution_classification != "EXECUTION_OK":
        dimensions = {
            dimension: _status_result(
                status="NOT_DETERMINED",
                score=None,
                reason=(
                    "Semantic answer quality is not evaluated because the trace "
                    f"classifies as {execution_classification}."
                ),
            )
            for dimension in DIMENSIONS
        }
        return {
            "case_id": case_id,
            "run_id": trace.get("run_id"),
            "execution_status": trace.get("execution_status"),
            "execution_classification": execution_classification,
            "reference_fingerprint": contract_case["reference_fingerprint"],
            "execution_trace_fingerprint": _trace_fingerprint(trace),
            "dimensions": dimensions,
            "claim_evidence_support": {
                "status": "NOT_DETERMINED",
                "claim_level_entailment": "NOT_DETERMINED",
            },
            "structured_abstention": {
                "status": "NOT_DETERMINED",
                "claim_level_decision": "NOT_DETERMINED",
            },
            "semantic_status": "NOT_DETERMINED",
            "evaluator": "deterministic_reference_assertions_v2",
            "independent_review": None,
        }
    reference = contract_case["reference_assertions"]
    dimensions = {
        "factual_correctness": _status_result(
            status="REVIEW_REQUIRED",
            score=None,
            reason="Golden reference is declarative; an independent reviewer must compare the answer to it.",
        ),
        "groundedness": _deterministic_groundedness(reference, trace),
        "completeness": _status_result(
            status="REVIEW_REQUIRED",
            score=None,
            reason="Success criteria require answer-level review; runtime execution cannot prove completeness.",
        ),
        "scope_adherence": _deterministic_scope(reference, trace),
        "safe_abstention": _deterministic_abstention(reference, trace),
    }
    claim_evidence_support = _claim_evidence_support(reference, trace)
    structured_abstention = _structured_abstention(reference, trace)
    if any(item["status"] == "FAIL" for item in dimensions.values()):
        status = "FAIL"
    elif any(item["status"] == "REVIEW_REQUIRED" for item in dimensions.values()):
        status = "REVIEW_REQUIRED"
    else:
        status = "PASS"
    return {
        "case_id": case_id,
        "run_id": trace.get("run_id"),
        "execution_status": trace.get("execution_status"),
        "execution_classification": execution_classification,
        "reference_fingerprint": contract_case["reference_fingerprint"],
        "execution_trace_fingerprint": _trace_fingerprint(trace),
        "dimensions": dimensions,
        "claim_evidence_support": claim_evidence_support,
        "structured_abstention": structured_abstention,
        "semantic_status": status,
        "evaluator": "deterministic_reference_assertions_v1",
        "independent_review": None,
    }


def _packet_observations(trace: dict[str, Any]) -> dict[str, Any]:
    answer = _trace_answer(trace)
    return {
        "execution_status": trace.get("execution_status"),
        "run_id": trace.get("run_id"),
        "tool_names": sorted(_trace_tools(trace)),
        "citation_count": len(trace.get("citations", [])),
        "source_count": len(trace.get("sources", [])),
        "available_evidence_count": len(trace.get("available_evidence_items", [])),
        "explicit_abstention_signal": _explicit_abstention(answer),
        "answer_excerpt_redacted": _redact_text(answer),
        "sensitive_content_redacted": True,
    }


def _packet_trace_for_independent_reviewer(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "execution_status": trace.get("execution_status"),
        "execution_error": trace.get("execution_error"),
        "provider": trace.get("provider"),
        "model": trace.get("model"),
        "final_output_redacted": _redact_text(_trace_answer(trace)),
        "tool_names": sorted(_trace_tools(trace)),
        "citations": [
            _redact_text(str(value), limit=600)
            for value in trace.get("citations", [])
            if isinstance(value, str)
        ][:24],
        "sources": [
            _redact_text(str(value), limit=600)
            for value in trace.get("sources", [])
            if isinstance(value, str)
        ][:24],
        "evidence_provenance": trace.get("evidence_provenance", [])[:24],
        "external_evidence_provenance": _safe_external_provenance(
            trace.get("external_evidence_provenance")
        ),
        "evidence_preconditions": trace.get("evidence_preconditions"),
        "upload_e2e": trace.get("upload_e2e"),
    }


def build_review_packet(
    *,
    contract: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    _validate_contract(contract)
    traces = {
        trace.get("case_id"): trace
        for trace in baseline.get("cases", [])
        if isinstance(trace, dict) and trace.get("case_id")
    }
    if set(traces) != {case["case_id"] for case in contract["cases"]}:
        raise SemanticEvaluationError("Review packet requires traces for all 26 cases.")
    cases: list[dict[str, Any]] = []
    for contract_case in contract["cases"]:
        trace = traces[contract_case["case_id"]]
        cases.append(
            {
                "case_id": contract_case["case_id"],
                "category": contract_case["category"],
                "source_reference": contract_case["reference_assertions"][
                    "source_reference"
                ],
                "reference_fingerprint": contract_case["reference_fingerprint"],
                "execution_trace_fingerprint": _trace_fingerprint(trace),
                "producer_provider": trace.get("provider")
                or baseline.get("execution_config", {}).get("provider"),
                "producer_model": trace.get("model"),
                "reference_assertions": contract_case["reference_assertions"],
                "review_questions": contract_case["review_questions"],
                "observations": _packet_observations(trace),
                "trace": _packet_trace_for_independent_reviewer(trace),
                "review": {
                    "factual_correctness": None,
                    "groundedness": None,
                    "completeness": None,
                    "scope_adherence": None,
                    "safe_abstention": None,
                    "decision": None,
                    "reason": None,
                    "evidence_notes": None,
                },
            }
        )
    packet = {
        "packet_version": PACKET_VERSION,
        "contract_version": contract["contract_version"],
        "contract_sha256": contract["contract_sha256"],
        "baseline_sha256": baseline.get("_artifact_file_sha256"),
        "review_policy": {
            "allowed_reviewer_types": ["human", "independent_evaluator"],
            "runtime_model_self_grading_forbidden": True,
            "pass_requires_all_dimensions_at_least": 3,
            "missing_review_is": "REVIEW_REQUIRED",
            "scores_must_include_reason": True,
        },
        "cases": cases,
    }
    packet["packet_sha256"] = sha256_json(packet)
    return packet


def build_review_input_template(packet: dict[str, Any]) -> dict[str, Any]:
    if packet.get("packet_version") != PACKET_VERSION:
        raise SemanticEvaluationError("Unsupported review packet version.")
    return {
        "review_input_version": "semantic-review-input-v1",
        "packet_sha256": packet["packet_sha256"],
        "reviewer_type": "human",
        "reviewer_id": "",
        "independence_declaration": False,
        "runtime_model_self_grading": False,
        "reviews": [
            {
                "case_id": case["case_id"],
                "reference_fingerprint": case["reference_fingerprint"],
                "execution_trace_fingerprint": case["execution_trace_fingerprint"],
                "scores": {dimension: None for dimension in DIMENSIONS},
                "decision": None,
                "reason": "",
                "evidence_notes": "",
            }
            for case in packet["cases"]
        ],
    }


def _validate_review_decision(
    *,
    decision: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    if decision.get("case_id") != expected["case_id"]:
        raise SemanticEvaluationError("Review decision case_id mismatch.")
    if decision.get("reference_fingerprint") != expected["reference_fingerprint"]:
        raise SemanticEvaluationError(
            f"Review reference fingerprint mismatch: {expected['case_id']}"
        )
    if decision.get("execution_trace_fingerprint") != expected[
        "execution_trace_fingerprint"
    ]:
        raise SemanticEvaluationError(
            f"Review trace fingerprint mismatch: {expected['case_id']}"
        )
    reviewer_type = decision.get("reviewer_type")
    if reviewer_type not in {"human", "independent_evaluator"}:
        raise SemanticEvaluationError("Reviewer must be human or independent_evaluator.")
    if decision.get("runtime_model_self_grading") is True:
        raise SemanticEvaluationError("Runtime model self-grading is forbidden.")
    if decision.get("independence_declaration") is not True:
        raise SemanticEvaluationError("Independent reviewer declaration is required.")
    reason = decision.get("reason")
    if not isinstance(reason, str) or len(reason.strip()) < 20:
        raise SemanticEvaluationError("Each review needs a reviewable reason.")
    _reject_secret_fields(decision)
    scores = decision.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(DIMENSIONS):
        raise SemanticEvaluationError("Review must score all semantic dimensions.")
    for dimension in DIMENSIONS:
        score = scores[dimension]
        if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 4:
            raise SemanticEvaluationError(f"Invalid score for {dimension}.")
    derived = (
        "PASS"
        if all(scores[dimension] >= 3 for dimension in DIMENSIONS)
        else "FAIL"
        if any(scores[dimension] <= 1 for dimension in DIMENSIONS)
        else "REVIEW_REQUIRED"
    )
    if decision.get("decision") != derived:
        raise SemanticEvaluationError(
            f"Review decision does not match scores for {expected['case_id']}."
        )
    return {
        "reviewer_type": reviewer_type,
        "reviewer_id": _redact_text(str(decision.get("reviewer_id", "")), limit=120),
        "scores": scores,
        "decision": derived,
        "reason": _redact_text(reason, limit=1000),
        "evidence_notes": _redact_text(
            str(decision.get("evidence_notes", "")), limit=1000
        ),
    }


def apply_reviews(
    *,
    evaluations: list[dict[str, Any]],
    packet: dict[str, Any],
    review_document: dict[str, Any],
) -> list[dict[str, Any]]:
    if review_document.get("packet_sha256") != packet.get("packet_sha256"):
        raise SemanticEvaluationError("Review document does not match packet fingerprint.")
    decisions = review_document.get("reviews")
    if not isinstance(decisions, list):
        raise SemanticEvaluationError("Review document must contain a reviews list.")
    defaults = {
        "reviewer_type": review_document.get("reviewer_type"),
        "reviewer_id": review_document.get("reviewer_id"),
        "independence_declaration": review_document.get(
            "independence_declaration"
        ),
        "runtime_model_self_grading": review_document.get(
            "runtime_model_self_grading"
        ),
    }
    expected = {item["case_id"]: item for item in packet["cases"]}
    validated: dict[str, dict[str, Any]] = {}
    for raw_decision in decisions:
        if not isinstance(raw_decision, dict):
            raise SemanticEvaluationError("Every review decision must be an object.")
        decision = {**defaults, **raw_decision}
        case_id = decision.get("case_id")
        if case_id not in expected or case_id in validated:
            raise SemanticEvaluationError("Review contains an unknown or duplicate case.")
        validated[case_id] = _validate_review_decision(
            decision=decision,
            expected=expected[case_id],
        )
    merged: list[dict[str, Any]] = []
    for evaluation in evaluations:
        result = dict(evaluation)
        review = validated.get(evaluation["case_id"])
        if review is None:
            result["semantic_status"] = "REVIEW_REQUIRED"
        else:
            result["independent_review"] = review
            if result.get("execution_classification") != "EXECUTION_OK":
                result["semantic_status"] = "NOT_DETERMINED"
                merged.append(result)
                continue
            deterministic_groundedness = result.get("dimensions", {}).get(
                "groundedness",
                {},
            ).get("status")
            if (
                review["decision"] == "PASS"
                and deterministic_groundedness != "PASS"
            ):
                result["semantic_status"] = (
                    "FAIL"
                    if deterministic_groundedness == "FAIL"
                    else "REVIEW_REQUIRED"
                )
            else:
                result["semantic_status"] = review["decision"]
        merged.append(result)
    return merged


def evaluate_baseline(
    *,
    contract: dict[str, Any],
    baseline_path: Path = BASELINE_PATH,
    review_path: Path | None = None,
) -> dict[str, Any]:
    _validate_contract(contract)
    baseline = _load_json(baseline_path)
    baseline["_artifact_file_sha256"] = sha256_file(baseline_path)
    traces = {
        trace.get("case_id"): trace
        for trace in baseline.get("cases", [])
        if isinstance(trace, dict) and trace.get("case_id")
    }
    evaluations: list[dict[str, Any]] = []
    for contract_case in contract["cases"]:
        trace = traces.get(contract_case["case_id"])
        if trace is None:
            evaluations.append(
                {
                    "case_id": contract_case["case_id"],
                    "run_id": None,
                    "execution_status": "MISSING_TRACE",
                    "reference_fingerprint": contract_case["reference_fingerprint"],
                    "execution_trace_fingerprint": None,
                    "dimensions": {
                        dimension: _status_result(
                            status="NOT_DETERMINED",
                            score=None,
                            reason="Execution trace is missing.",
                        )
                        for dimension in DIMENSIONS
                    },
                    "claim_evidence_support": {
                        "status": "NOT_DETERMINED",
                        "claim_level_entailment": "NOT_DETERMINED",
                    },
                    "structured_abstention": {
                        "status": "NOT_DETERMINED",
                        "claim_level_decision": "NOT_DETERMINED",
                    },
                    "execution_classification": "MISSING_TRACE",
                    "semantic_status": "NOT_DETERMINED",
                    "evaluator": "deterministic_reference_assertions_v2",
                    "independent_review": None,
                }
            )
        else:
            evaluations.append(evaluate_case(contract_case=contract_case, trace=trace))
    packet = build_review_packet(contract=contract, baseline=baseline)
    if review_path is not None:
        review_document = _load_json(review_path)
        evaluations = apply_reviews(
            evaluations=evaluations,
            packet=packet,
            review_document=review_document,
        )
    counts = {
        status: sum(item["semantic_status"] == status for item in evaluations)
        for status in ("PASS", "FAIL", "REVIEW_REQUIRED", "NOT_DETERMINED")
    }
    deterministic_failures = [
        {
            "case_id": result["case_id"],
            "execution_classification": result.get(
                "execution_classification",
                "EXECUTION_OK",
            ),
            "dimensions": {
                dimension: value
                for dimension, value in result["dimensions"].items()
                if value["status"] == "FAIL"
            },
            "classification": (
                "execution_or_provider_failure"
                if result.get("execution_classification") != "EXECUTION_OK"
                else "semantic_output_or_evidence_gap"
            ),
            "code_bug_confirmed": False,
            "reason": (
                "The failure is derived from answer/evidence assertions in the "
                "stored trace; no execution, persistence, provider, or policy "
                "boundary error is present in this result."
            ),
            "next_action": "independent_review_or_reexecute_with_grounded_evidence",
        }
        for result in evaluations
        if any(
            value["status"] == "FAIL"
            for value in result["dimensions"].values()
        )
    ]
    execution_failures = [
        {
            "case_id": result["case_id"],
            "execution_status": result.get("execution_status"),
            "execution_classification": result.get(
                "execution_classification",
                "EXECUTION_FAILURE",
            ),
            "classification": "execution_or_provider_failure",
            "semantic_status": result["semantic_status"],
            "next_action": "repair_or_reexecute_before_semantic_scoring",
        }
        for result in evaluations
        if result.get("execution_classification") != "EXECUTION_OK"
    ]
    overall_status = (
        "FAIL"
        if counts["FAIL"]
        else "REVIEW_REQUIRED"
        if counts["REVIEW_REQUIRED"]
        else "NOT_DETERMINED"
        if counts["NOT_DETERMINED"]
        else "PASS"
    )
    return {
        "evaluation_version": "semantic-evaluation-v1",
        "contract_version": contract["contract_version"],
        "contract_sha256": contract["contract_sha256"],
        "dataset_sha256": contract["dataset_sha256"],
        "baseline_sha256": baseline["_artifact_file_sha256"],
        "case_count": len(evaluations),
        "anti_self_grading": contract["anti_self_grading"],
        "overall_status": overall_status,
        "counts": counts,
        "results": evaluations,
        "deterministic_failure_analysis": {
            "code_bug_fix_applied": False,
            "code_bug_cases": [],
            "failures": deterministic_failures,
            "semantic_failures": deterministic_failures,
            "execution_failures": execution_failures,
        },
        "review_packet_sha256": packet["packet_sha256"],
        "review_applied": review_path is not None,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent semantic quality evaluation")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument(
        "--contract-output",
        type=Path,
        default=ROOT / "semantic-evaluation-contract-v1.json",
    )
    parser.add_argument(
        "--packet-output",
        type=Path,
        default=ROOT / "semantic-review-packet-v1.json",
    )
    parser.add_argument(
        "--result-output",
        type=Path,
        default=ROOT / "semantic-evaluation-v1.json",
    )
    parser.add_argument(
        "--review-template-output",
        type=Path,
        default=ROOT / "semantic-review-input-template-v1.json",
    )
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    contract = build_contract(dataset_path=args.dataset, baseline_path=args.baseline)
    baseline = _load_json(args.baseline)
    baseline["_artifact_file_sha256"] = sha256_file(args.baseline)
    packet = build_review_packet(contract=contract, baseline=baseline)
    review_template = build_review_input_template(packet)
    report = evaluate_baseline(
        contract=contract,
        baseline_path=args.baseline,
        review_path=args.review,
    )
    _write_json(args.contract_output, contract)
    _write_json(args.packet_output, packet)
    _write_json(args.review_template_output, review_template)
    _write_json(args.result_output, report)
    print(
        json.dumps(
            {
                "overall_status": report["overall_status"],
                "counts": report["counts"],
                "case_count": report["case_count"],
                "contract_sha256": report["contract_sha256"],
                "review_packet_sha256": report["review_packet_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())