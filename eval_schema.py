from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DATASET_PATH = (
    Path(__file__).parent
    / "tests"
    / "fixtures"
    / "evaluation_baseline"
    / "cases.json"
)
DATASET_VERSION = "contract-seed-v1"
REQUIRED_CASE_KEYS = {
    "id",
    "category",
    "provenance",
    "input",
    "expected_behavior",
    "expected_sources",
    "forbidden_behavior",
    "required_tools",
    "forbidden_tools",
    "success_criteria",
    "deterministic_checks",
}
SECRET_FIELD_MARKERS = ("secret", "token", "password", "api_key", "apikey")
SEMANTIC_RUBRIC = {
    "grounding": "Every material claim is supported by the cited evidence.",
    "relevance": "The answer directly addresses the case input.",
    "completeness": "All required parts of the task are addressed.",
    "source_correctness": "The cited source actually supports the claim.",
}
TOOL_NAME_MAPPING = {
    "web_search": {
        "actual": "web_search",
        "status": "available",
        "note": "AgentCore tool name matches.",
    },
    "my_files": {
        "actual": "search_my_files",
        "status": "available",
        "note": "Semantic MY_FILES capability maps to the scoped retrieval tool.",
    },
    "file_access": {
        "actual": None,
        "status": "unavailable",
        "note": "No general file-access AgentCore tool exists.",
    },
    "project_file_access": {
        "actual": None,
        "status": "unavailable",
        "note": "Project workspace inspection is not an AgentCore tool.",
    },
    "runtime_evidence": {
        "actual": "inspect_runtime_evidence",
        "status": "available",
        "note": "Bounded runtime evidence inspection is available only for the current-runtime case.",
    },
    "source_status": {
        "actual": "inspect_source_status",
        "status": "available",
        "note": "Bounded source/config evidence inspection is available only for approved source-status components.",
    },
    "deploy": {
        "actual": None,
        "status": "boundary_only",
        "note": "Deployment is a platform action, not an AgentCore tool.",
    },
    "publish": {
        "actual": None,
        "status": "boundary_only",
        "note": "Publishing is a platform action, not an AgentCore tool.",
    },
}
CASE_CAPABILITY_OVERRIDES = {
    "AA-RC-002": {
        "file_access": {
            "actual": "inspect_source_of_truth",
            "status": "available",
            "note": "AA-RC-002 uses authenticated immutable MY_FILES source inspection, not generic file access.",
        }
    },
    "AA-RC-011": {
        "project_file_access": {
            "actual": "inspect_runtime_evidence",
            "status": "available",
            "note": "AA-RC-011 uses bounded deployment/runtime evidence, not general project file access.",
        }
    },
    "AA-RC-007": {
        "project_file_access": {
            "actual": "inspect_architecture_evidence",
            "status": "available",
            "note": "AA-RC-007 uses fixed architecture-group evidence, not generic project file access.",
        }
    },
    "AA-RC-016": {
        "project_file_access": {
            "actual": "inspect_runtime_evidence",
            "status": "available",
            "note": "AA-RC-016 uses bounded runtime evidence, not general project access.",
        }
    },
    "AA-RC-014": {
        "project_file_access": {
            "actual": "inspect_source_status",
            "status": "available",
            "note": "AA-RC-014 uses fixed source-status evidence for search components, not general project access.",
        }
    },
}
REQUIRED_CHECK_KEYS = {
    "citations_required",
    "schema_fields",
    "approval_required",
    "abstention_required",
}


def _read_dataset_document(path: Path) -> tuple[str, list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        version = raw.get("dataset_version")
        raw_cases = raw.get("cases")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("Dataset document is missing dataset_version.")
    else:
        version = DATASET_VERSION
        raw_cases = raw
    if not isinstance(raw_cases, list):
        raise ValueError("Evaluation dataset must contain a cases list.")
    return version, raw_cases


def validate_evaluation_cases(
    raw_cases: list[dict[str, Any]],
    *,
    require_baseline_size: bool = False,
) -> list[dict[str, Any]]:
    if require_baseline_size and not 20 <= len(raw_cases) <= 30:
        raise ValueError("Evaluation baseline must contain 20-30 cases.")
    seen_ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for case in raw_cases:
        if not isinstance(case, dict) or not REQUIRED_CASE_KEYS <= case.keys():
            raise ValueError("Evaluation case is missing required fields.")
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Evaluation case id must be a non-empty string.")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate evaluation case id: {case['id']}")
        seen_ids.add(case_id)
        if not isinstance(case["input"], str) or not case["input"].strip():
            raise ValueError(f"Evaluation case has empty input: {case_id}")
        checks = case["deterministic_checks"]
        if not isinstance(checks, dict) or not REQUIRED_CHECK_KEYS <= checks.keys():
            raise ValueError(f"Invalid deterministic checks: {case_id}")
        if case["provenance"] not in {"contract_seed", "real_case"}:
            raise ValueError(f"Invalid case provenance: {case_id}")
        if case["provenance"] == "real_case":
            source_reference = case.get("source_reference")
            if not isinstance(source_reference, str) or not source_reference.strip():
                raise ValueError(
                    f"real_case requires a documented source_reference: {case_id}"
                )
        required_tools = set(case["required_tools"])
        forbidden_tools = set(case["forbidden_tools"])
        overlap = required_tools & forbidden_tools
        if overlap:
            raise ValueError(
                f"Evaluation case has conflicting tool expectations: "
                f"{case_id}: {sorted(overlap)}"
            )
        if not isinstance(case["expected_sources"], list):
            raise ValueError(f"expected_sources must be a list: {case_id}")
        if (
            checks.get("expected_sources_required", False)
            and not case["expected_sources"]
        ):
            raise ValueError(f"Expected sources are required: {case_id}")
        validated.append(case)
    return validated


def load_evaluation_cases(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    _, raw_cases = _read_dataset_document(path)
    return validate_evaluation_cases(raw_cases, require_baseline_size=True)


def load_case_document(
    path: Path,
    *,
    require_baseline_size: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    version, raw_cases = _read_dataset_document(path)
    if any(
        isinstance(case, dict)
        and case.get("case_type") == "real_case"
        and "id" not in case
        for case in raw_cases
    ):
        raw_cases = [_normalize_real_case(case) for case in raw_cases]
    return version, validate_evaluation_cases(
        raw_cases,
        require_baseline_size=require_baseline_size,
    )


def _reject_secret_fields(case: dict[str, Any]) -> None:
    for key in case:
        lowered = str(key).casefold()
        if any(marker in lowered for marker in SECRET_FIELD_MARKERS):
            raise ValueError(f"real_case contains a prohibited field: {key}")


def _normalize_real_case(raw_case: dict[str, Any]) -> dict[str, Any]:
    _reject_secret_fields(raw_case)
    if raw_case.get("case_type", "real_case") != "real_case":
        raise ValueError("Real-case import requires case_type=real_case.")
    case_id = raw_case.get("case_id", raw_case.get("id"))
    source_reference = raw_case.get("source_reference")
    if source_reference is None:
        provenance_value = raw_case.get("provenance")
        if provenance_value not in {"contract_seed", "real_case"}:
            source_reference = provenance_value
    normalized = {
        "id": case_id,
        "category": raw_case.get("category"),
        "provenance": "real_case",
        "case_type": "real_case",
        "source_reference": source_reference,
        "input": raw_case.get("input"),
        "expected_behavior": raw_case.get("expected_behavior"),
        "expected_sources": raw_case.get("expected_sources", []),
        "forbidden_behavior": raw_case.get("forbidden_behavior", []),
        "required_tools": raw_case.get("required_tools", []),
        "forbidden_tools": raw_case.get("forbidden_tools", []),
        "success_criteria": raw_case.get("success_criteria", []),
        "deterministic_checks": raw_case.get(
            "deterministic_checks",
            {
                "citations_required": bool(raw_case.get("expected_sources")),
                "schema_fields": ["answer"],
                "approval_required": False,
                "abstention_required": False,
            },
        ),
    }
    return normalized


def import_real_cases(source_path: Path, output_path: Path) -> dict[str, Any]:
    dataset_version, raw_cases = _read_dataset_document(source_path)
    normalized = [_normalize_real_case(case) for case in raw_cases]
    validate_evaluation_cases(normalized)
    output = {
        "dataset_version": dataset_version,
        "cases": normalized,
    }
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "output": str(output_path),
        "dataset_version": output["dataset_version"],
        "real_case_count": len(normalized),
    }


