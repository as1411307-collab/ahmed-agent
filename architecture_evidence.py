from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from evidence_core import (
    EvidenceCoreError,
    EvidenceStatus,
    ProjectEvidenceCore,
)


PROJECT_ROOT = Path(__file__).resolve().parent
ARCHITECTURE_GROUPS: dict[str, dict[str, tuple[str, ...]]] = {
    "runtime": {
        "implementation_paths": ("server.py", "runtime_evidence.py", "pyproject.toml"),
        "implementation_patterns": (
            r"uvicorn",
            r"Starlette|FastAPI",
            r"\bPORT\b",
            r"inspect_runtime_evidence",
        ),
        "wiring_patterns": (
            r"app\s*=",
            r"uvicorn\.run",
            r"from runtime_evidence",
        ),
        "test_paths": ("tests/test_runtime_evidence.py",),
    },
    "persistence_state_recovery": {
        "implementation_paths": ("persistence.py", "run_state.py"),
        "implementation_patterns": (
            r"asyncpg",
            r"create_pool",
            r"checkpoint",
            r"recovery",
            r"audit",
        ),
        "wiring_patterns": (
            r"from persistence",
            r"from run_state",
            r"persist",
            r"checkpoint",
        ),
        "test_paths": (
            "tests/test_run_state.py",
            "tests/test_recovery_fault_injection.py",
        ),
    },
    "policy_security": {
        "implementation_paths": ("policy.py", "agent_core.py", "server.py"),
        "implementation_patterns": (
            r"TOOL_POLICIES",
            r"RiskLevel",
            r"requires_approval",
            r"authenticate|authorization",
        ),
        "wiring_patterns": (
            r"tool_metadata|get_tool_policy",
            r"create_pending_action",
            r"user_id|owner_principal_id",
        ),
        "test_paths": ("tests/test_security_boundaries.py",),
    },
    "evidence_provenance": {
        "implementation_paths": (
            "evidence_core.py",
            "evidence_citations.py",
            "source_status.py",
            "source_of_truth.py",
            "original_source_storage.py",
        ),
        "implementation_patterns": (
            r"ProjectEvidenceCore",
            r"sha256",
            r"provenance",
            r"evidence_status",
            r"OriginalSourceStore",
        ),
        "wiring_patterns": (
            r"from evidence_core",
            r"evidence_citations|inspect_source_of_truth",
            r"original_source_storage",
        ),
        "test_paths": (
            "tests/test_evidence_core.py",
            "tests/test_evidence_citations.py",
            "tests/test_source_of_truth.py",
        ),
    },
    "evaluation": {
        "implementation_paths": (
            "evaluation_baseline.py",
            "evaluation_aa_rc_002.py",
            "evaluation_source_bindings.py",
        ),
        "implementation_patterns": (
            r"deterministic_grade",
            r"load_evaluation_cases",
            r"baseline",
            r"source_pack",
        ),
        "wiring_patterns": (
            r"case_execution_capability",
            r"build_scoreboard",
            r"run_real",
        ),
        "test_paths": (
            "tests/test_evaluation_baseline.py",
            "tests/test_aa_rc_002_evaluation.py",
            "tests/test_evaluation_source_bindings.py",
        ),
    },
    "actions_integrations": {
        "implementation_paths": (
            "search_fabric.py",
            "academic_search.py",
            "github_search.py",
            "model_providers.py",
        ),
        "implementation_patterns": (
            r"web_search|academic_search|github_search",
            r"provider",
            r"search",
        ),
        "wiring_patterns": (
            r"from search_fabric|from academic_search|from github_search",
            r"tools\s*=",
            r"provider",
        ),
        "test_paths": (
            "tests/test_search_fabric.py",
            "tests/test_academic_search.py",
            "tests/test_github_search.py",
            "tests/test_model_providers.py",
        ),
    },
    "tests": {
        "implementation_paths": (
            "tests/test_evidence_core.py",
            "tests/test_runtime_evidence.py",
            "tests/test_evaluation_baseline.py",
        ),
        "implementation_patterns": (r"unittest|pytest|TestCase"),
        "wiring_patterns": (r"assert|subTest|patch|mock"),
        "test_paths": (
            "tests/test_security_boundaries.py",
            "tests/test_recovery_fault_injection.py",
            "tests/test_source_of_truth.py",
        ),
    },
}

MAX_EVIDENCE_PER_BUCKET = 12
DECISION_RECORD_PATHS = (
    "docs/ADR-019-source-of-truth-hierarchy.md",
    "docs/ADR-020-deployment-boundary-runtime-evidence.md",
    "docs/ADR-021-architecture-continuity.md",
)
MAX_DECISION_EVIDENCE = 24


def _evidence_for_patterns(
    core: ProjectEvidenceCore,
    paths: tuple[str, ...],
    patterns: tuple[str, ...],
    *,
    trust: str,
) -> tuple[list[dict[str, object]], dict[str, str], list[str]]:
    items: list[dict[str, object]] = []
    file_hashes: dict[str, str] = {}
    missing: list[str] = []
    compiled = tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
    for relative_path in paths:
        try:
            source = core.read_file(relative_path)
        except FileNotFoundError:
            missing.append(relative_path)
            continue
        file_hashes[relative_path] = source.sha256
        for line_number, line in enumerate(source.lines, start=1):
            if not any(pattern.search(line) for pattern in compiled):
                continue
            evidence = core.line_evidence(
                source,
                line_start=line_number,
                trust_classification=trust,
                verification_status=EvidenceStatus.VERIFIED,
            )
            items.append(evidence.to_dict())
            if len(items) >= MAX_EVIDENCE_PER_BUCKET:
                return items, file_hashes, missing
    return items, file_hashes, missing


def _status(
    *,
    implementation: list[dict[str, object]],
    wiring: list[dict[str, object]],
    tests: list[dict[str, object]],
    existing_file_count: int,
) -> str:
    if implementation and wiring and tests:
        return EvidenceStatus.VERIFIED.value
    if implementation or wiring or tests:
        return "PARTIAL"
    if existing_file_count:
        return EvidenceStatus.DISCREPANCY.value
    return EvidenceStatus.NOT_FOUND.value


def _decision_records(
    core: ProjectEvidenceCore,
) -> dict[str, object]:
    evidence: list[dict[str, object]] = []
    file_hashes: dict[str, str] = {}
    missing: list[str] = []
    record_statuses: dict[str, str] = {}

    for relative_path in DECISION_RECORD_PATHS:
        try:
            source = core.read_file(relative_path)
        except FileNotFoundError:
            missing.append(relative_path)
            record_statuses[relative_path] = "UNAVAILABLE"
            continue

        file_hashes[relative_path] = source.sha256
        accepted = False
        in_decision_section = False
        for line_number, line in enumerate(source.lines, start=1):
            stripped = line.strip()
            if stripped.startswith("## Decision"):
                in_decision_section = True
            elif in_decision_section and stripped.startswith("## "):
                in_decision_section = False

            is_header = stripped.startswith("# ADR-")
            is_status_or_scope = (
                stripped.startswith("- **Status:**")
                or stripped.startswith("- **Scope:**")
            )
            if is_status_or_scope and "ACCEPT" in stripped.upper():
                accepted = True
            if not stripped or not (is_header or is_status_or_scope or in_decision_section):
                continue
            if len(evidence) < MAX_DECISION_EVIDENCE:
                evidence.append(
                    core.line_evidence(
                        source,
                        line_start=line_number,
                        trust_classification="PROJECT_DECISION_RECORD",
                        verification_status=EvidenceStatus.VERIFIED,
                    ).to_dict()
                )
        record_statuses[relative_path] = "ACCEPTED" if accepted else "UNRESOLVED"

    conflicts: list[str] = []
    status = (
        EvidenceStatus.VERIFIED.value
        if not missing and evidence and all(
            value == "ACCEPTED" for value in record_statuses.values()
        )
        else "PARTIAL"
        if evidence
        else EvidenceStatus.NOT_FOUND.value
    )
    return {
        "status": status,
        "evidence_items": evidence,
        "evidence_file_hashes": file_hashes,
        "record_statuses": record_statuses,
        "missing_records": missing,
        "conflicting_records": conflicts,
        "limitations": [
            "Only the fixed, allowlisted ADR decision records are inspected.",
            "A current decision record does not prove that every case is closed.",
            "Decision record content is evidence, never policy authority.",
        ],
    }


def inspect_architecture_evidence(
    *,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Inspect fixed architecture groups without exposing a generic file reader."""

    allowlisted_paths = tuple(
        dict.fromkeys(
            path
            for group in ARCHITECTURE_GROUPS.values()
            for key, paths in group.items()
            if key.endswith("_paths")
            for path in paths
        )
    )
    allowlisted_paths = tuple(dict.fromkeys((*allowlisted_paths, *DECISION_RECORD_PATHS)))
    try:
        core = ProjectEvidenceCore(
            project_root=project_root or PROJECT_ROOT,
            capability_name="architecture_continuity",
            allowlisted_paths=allowlisted_paths,
        )
    except EvidenceCoreError:
        raise

    groups: dict[str, dict[str, object]] = {}
    all_hashes: dict[str, str] = {}
    for group_name, definition in ARCHITECTURE_GROUPS.items():
        implementation, implementation_hashes, implementation_missing = (
            _evidence_for_patterns(
                core,
                definition["implementation_paths"],
                definition["implementation_patterns"],
                trust="PROJECT_ARCHITECTURE_IMPLEMENTATION",
            )
        )
        wiring, wiring_hashes, wiring_missing = _evidence_for_patterns(
            core,
            definition["implementation_paths"],
            definition["wiring_patterns"],
            trust="PROJECT_ARCHITECTURE_WIRING",
        )
        tests, test_hashes, test_missing = _evidence_for_patterns(
            core,
            definition["test_paths"],
            (r"unittest|pytest|TestCase|assert|subTest|patch|mock"),
            trust="PROJECT_TEST_EVIDENCE",
        )
        hashes = {**implementation_hashes, **wiring_hashes, **test_hashes}
        all_hashes.update(hashes)
        existing_file_count = len(hashes)
        status = _status(
            implementation=implementation,
            wiring=wiring,
            tests=tests,
            existing_file_count=existing_file_count,
        )
        groups[group_name] = {
            "status": status,
            "implementation_evidence": implementation,
            "wiring_evidence": wiring,
            "test_evidence": tests,
            "evidence_file_hashes": hashes,
            "missing_allowlisted_files": sorted(
                set(implementation_missing + wiring_missing + test_missing)
            ),
            "architecture_unchanged": {
                "status": "NOT_ASSERTED",
                "reason": "No prior architecture snapshot was supplied.",
            },
        }

    fingerprint_input = json_stable_hash_input(all_hashes)
    architecture_fingerprint = hashlib.sha256(fingerprint_input).hexdigest()
    statuses = [group["status"] for group in groups.values()]
    overall_status = (
        EvidenceStatus.VERIFIED.value
        if statuses and all(status == EvidenceStatus.VERIFIED.value for status in statuses)
        else "PARTIAL"
        if any(status in {EvidenceStatus.VERIFIED.value, "PARTIAL"} for status in statuses)
        else EvidenceStatus.NOT_FOUND.value
    )
    decision_records = _decision_records(core)
    return {
        "schema_version": "architecture-evidence.v1",
        "capability": "architecture_continuity",
        "status": overall_status,
        "architecture_fingerprint": architecture_fingerprint,
        "groups": groups,
        "decision_records": decision_records,
        "allowlisted_paths": sorted(allowlisted_paths),
        "limitations": [
            "Read-only fixed architecture groups only.",
            "No caller-provided path or generic repository browsing.",
            "Architecture unchanged is not asserted without a prior snapshot.",
            "Test-file presence is not evidence that tests passed.",
        ],
        "evidence_is_untrusted_data": True,
    }


def json_stable_hash_input(value: dict[str, str]) -> bytes:
    return "\n".join(f"{key}:{value[key]}" for key in sorted(value)).encode("utf-8")