from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from evidence_core import (
    EvidenceCoreError,
    EvidenceStatus,
    EvidenceTrust,
    EvidenceItem,
    ProjectEvidenceCore,
    SourceFile,
)
from evidence_citations import render_evidence_citations


SourceStatusComponent = Literal["search_provider", "page_fetcher"]
SOURCE_STATUS_COMPONENTS = frozenset({"search_provider", "page_fetcher"})
SOURCE_STATUS_ALLOWLISTS: dict[str, tuple[str, ...]] = {
    "search_provider": (
        "search_fabric.py",
        "skill_tools.py",
        "agent_core.py",
        "config.py",
        "pyproject.toml",
        "policy.py",
    ),
    "page_fetcher": (
        "skill_tools.py",
        "agent_core.py",
        "config.py",
        "pyproject.toml",
        "policy.py",
    ),
}


def _matches(
    source_file: SourceFile,
    pattern: str,
    *,
    flags: int = 0,
    limit: int = 4,
) -> list[int]:
    compiled = re.compile(pattern, flags)
    return [
        index
        for index, line in enumerate(source_file.lines, 1)
        if compiled.search(line)
    ][:limit]


def _first_file(files: dict[str, SourceFile], filename: str) -> SourceFile | None:
    return files.get(filename)


def _items_for_matches(
    core: ProjectEvidenceCore,
    files: dict[str, SourceFile],
    matches: list[tuple[str, int, EvidenceTrust]],
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for filename, line_number, trust in matches:
        source_file = files[filename]
        item = core.line_evidence(
            source_file,
            line_start=line_number,
            trust_classification=trust,
        )
        items.append(item.to_dict())
    return items


def _section(
    *,
    status: EvidenceStatus,
    facts: dict[str, object],
    evidence: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "status": status.value,
        "facts": facts,
        "evidence": evidence,
    }


def _load_files(
    core: ProjectEvidenceCore,
) -> tuple[dict[str, SourceFile], dict[str, str]]:
    files: dict[str, SourceFile] = {}
    unavailable: dict[str, str] = {}
    for filename in sorted(core.allowlisted_paths):
        try:
            files[filename] = core.read_file(filename)
        except FileNotFoundError:
            unavailable[filename] = "NOT_FOUND"
        except EvidenceCoreError as error:
            unavailable[filename] = str(error)
    return files, unavailable


def _inspect_search_provider(
    core: ProjectEvidenceCore,
    files: dict[str, SourceFile],
) -> tuple[EvidenceStatus, dict[str, object], list[dict[str, object]]]:
    implementation_matches: list[tuple[str, int, EvidenceTrust]] = []
    protocol_file = _first_file(files, "search_fabric.py")
    if protocol_file:
        for pattern in (
            r"class\s+SearchProvider\s*\(Protocol\)",
            r"class\s+TavilyProvider\s*:",
            r"class\s+BraveProvider\s*:",
            r"class\s+SearchFabric\s*:",
        ):
            for line in _matches(protocol_file, pattern):
                implementation_matches.append(
                    ("search_fabric.py", line, EvidenceTrust.PROJECT_SOURCE)
                )

    wiring_matches: list[tuple[str, int, EvidenceTrust]] = []
    skill_file = _first_file(files, "skill_tools.py")
    if skill_file:
        for pattern in (
            r"from\s+search_fabric\s+import\s+BraveProvider,\s+SearchFabric,\s+TavilyProvider",
            r"_search_fabric_instance\s*=",
            r"SearchFabric\s*\(",
            r"TavilyProvider\s*\(",
            r"BraveProvider\s*\(",
            r"def\s+web_search\s*\(",
        ):
            for line in _matches(skill_file, pattern):
                wiring_matches.append(
                    ("skill_tools.py", line, EvidenceTrust.PROJECT_SOURCE)
                )
    agent_file = _first_file(files, "agent_core.py")
    if agent_file:
        for pattern in (
            r"from\s+skill_tools\s+import\s+web_search",
            r"^\s+web_search,\s*$",
        ):
            for line in _matches(agent_file, pattern, flags=re.MULTILINE):
                wiring_matches.append(
                    ("agent_core.py", line, EvidenceTrust.PROJECT_SOURCE)
                )

    configuration_matches: list[tuple[str, int, EvidenceTrust]] = []
    for filename, patterns in {
        "skill_tools.py": (r"os\.environ\.get\(\s*[\"']TAVILY_API_KEY",),
        "search_fabric.py": (r"os\.environ\.get\(\s*[\"']BRAVE_SEARCH_API_KEY",),
        "config.py": (
            r"BRAVE_SEARCH_ENABLED\s*=",
            r"SEARCH_PROVIDER_TIMEOUT_SECONDS\s*=",
            r"SEARCH_MAX_PROVIDER_CALLS\s*=",
        ),
    }.items():
        source_file = _first_file(files, filename)
        if source_file:
            for pattern in patterns:
                for line in _matches(source_file, pattern):
                    configuration_matches.append(
                        (filename, line, EvidenceTrust.PROJECT_CONFIGURATION)
                    )

    dependency_matches: list[tuple[str, int, EvidenceTrust]] = []
    pyproject = _first_file(files, "pyproject.toml")
    if pyproject:
        for pattern in (r'requires-python\s*=', r'"starlette>=', r'"uvicorn>='):
            for line in _matches(pyproject, pattern):
                dependency_matches.append(
                    ("pyproject.toml", line, EvidenceTrust.PROJECT_DEPENDENCY_DECLARATION)
                )

    evidence = _items_for_matches(
        core,
        files,
        implementation_matches + wiring_matches + configuration_matches + dependency_matches,
    )
    implementation_status = (
        EvidenceStatus.VERIFIED
        if implementation_matches
        else EvidenceStatus.NOT_FOUND
    )
    wiring_status = EvidenceStatus.VERIFIED if wiring_matches else EvidenceStatus.NOT_FOUND
    config_status = (
        EvidenceStatus.VERIFIED
        if configuration_matches
        else EvidenceStatus.NOT_FOUND
    )
    readiness_status = (
        EvidenceStatus.VERIFIED if dependency_matches else EvidenceStatus.NOT_FOUND
    )
    overall = (
        EvidenceStatus.VERIFIED
        if all(
            status == EvidenceStatus.VERIFIED
            for status in (
                implementation_status,
                wiring_status,
                config_status,
                readiness_status,
            )
        )
        else EvidenceStatus.DISCREPANCY
    )
    facts = {
        "component_kind": "SearchProvider protocol with Tavily and Brave implementations",
        "implementation": _section(
            status=implementation_status,
            facts={
                "present": bool(implementation_matches),
                "implementations": ["SearchProvider", "TavilyProvider", "BraveProvider"]
                if implementation_matches
                else [],
            },
            evidence=[],
        ),
        "registration_wiring": _section(
            status=wiring_status,
            facts={
                "registered": bool(wiring_matches),
                "active_path": "skill_tools._get_search_fabric -> SearchFabric(TavilyProvider, BraveProvider)"
                if wiring_matches
                else None,
                "agent_scope": "WEB" if agent_file and wiring_matches else None,
            },
            evidence=[],
        ),
        "configuration_declaration": _section(
            status=config_status,
            facts={
                "referenced_keys": [
                    "TAVILY_API_KEY",
                    "BRAVE_SEARCH_API_KEY",
                    "BRAVE_SEARCH_ENABLED",
                    "SEARCH_PROVIDER_TIMEOUT_SECONDS",
                    "SEARCH_MAX_PROVIDER_CALLS",
                ]
                if configuration_matches
                else [],
                "secret_values_read": False,
            },
            evidence=[],
        ),
        "setup_readiness": _section(
            status=readiness_status,
            facts={
                "declared_runtime_dependencies": ["starlette", "uvicorn"]
                if dependency_matches
                else [],
                "external_provider_health_checked": False,
            },
            evidence=[],
        ),
        "contradictions": [],
        "operational_status": {
            "claim": "not_asserted",
            "explanation": "Source/config evidence proves implementation and wiring, not provider health or secret validity.",
        },
    }
    return overall, facts, evidence


def _inspect_page_fetcher(
    core: ProjectEvidenceCore,
    files: dict[str, SourceFile],
) -> tuple[EvidenceStatus, dict[str, object], list[dict[str, object]]]:
    skill_file = _first_file(files, "skill_tools.py")
    implementation_matches: list[tuple[str, int, EvidenceTrust]] = []
    definition_lines: set[int] = set()
    if skill_file:
        for pattern in (r"def\s+_fetch_page_sync\s*\(", r"async\s+def\s+_fetch_page\s*\("):
            for line in _matches(skill_file, pattern):
                definition_lines.add(line)
                implementation_matches.append(
                    ("skill_tools.py", line, EvidenceTrust.PROJECT_SOURCE)
                )

    call_lines: list[int] = []
    if skill_file:
        for line in _matches(skill_file, r"_fetch_page\s*\(", limit=20):
            if line not in definition_lines:
                call_lines.append(line)

    alternative_matches: list[tuple[str, int, EvidenceTrust]] = []
    if skill_file:
        for pattern in (
            r"TAVILY_EXTRACT_URL\s*=",
            r"async\s+def\s+_run_tavily_extract\s*\(",
            r"_run_tavily_extract\s*\(",
        ):
            for line in _matches(skill_file, pattern):
                alternative_matches.append(
                    ("skill_tools.py", line, EvidenceTrust.PROJECT_SOURCE)
                )

    configuration_matches: list[tuple[str, int, EvidenceTrust]] = []
    if skill_file:
        for line in _matches(skill_file, r"PAGE_FETCH_TIMEOUT_SECONDS\s*="):
            configuration_matches.append(
                ("skill_tools.py", line, EvidenceTrust.PROJECT_CONFIGURATION)
            )
    dependency_matches: list[tuple[str, int, EvidenceTrust]] = []
    if skill_file:
        for pattern in (r"from\s+urllib\.request\s+import", r"urlopen\s*\("):
            for line in _matches(skill_file, pattern):
                dependency_matches.append(
                    ("skill_tools.py", line, EvidenceTrust.PROJECT_SOURCE)
                )

    evidence = _items_for_matches(
        core,
        files,
        implementation_matches
        + [
            ("skill_tools.py", line, EvidenceTrust.PROJECT_SOURCE)
            for line in call_lines
        ]
        + alternative_matches
        + configuration_matches
        + dependency_matches,
    )
    implementation_status = (
        EvidenceStatus.VERIFIED
        if implementation_matches
        else EvidenceStatus.NOT_FOUND
    )
    wiring_status = (
        EvidenceStatus.VERIFIED
        if call_lines or alternative_matches
        else EvidenceStatus.NOT_FOUND
    )
    config_status = (
        EvidenceStatus.VERIFIED
        if configuration_matches
        else EvidenceStatus.NOT_FOUND
    )
    readiness_status = (
        EvidenceStatus.VERIFIED
        if dependency_matches
        else EvidenceStatus.NOT_FOUND
    )
    contradictions: list[str] = []
    if implementation_matches and not call_lines and alternative_matches:
        contradictions.append(
            "A local _fetch_page implementation exists but has no active call site; the current extraction path uses Tavily extract."
        )
    overall = (
        EvidenceStatus.VERIFIED
        if all(
            status == EvidenceStatus.VERIFIED
            for status in (
                implementation_status,
                wiring_status,
                config_status,
                readiness_status,
            )
        )
        else EvidenceStatus.NOT_FOUND
    )
    facts = {
        "component_kind": "Local page fetch and text extraction helpers",
        "implementation": _section(
            status=implementation_status,
            facts={
                "present": bool(implementation_matches),
                "symbols": ["_fetch_page_sync", "_fetch_page"]
                if implementation_matches
                else [],
            },
            evidence=[],
        ),
        "registration_wiring": _section(
            status=wiring_status,
            facts={
                "registered_or_called": bool(call_lines),
                "active_call_sites": call_lines,
                "active_extraction_alternative": "Tavily extract"
                if alternative_matches
                else None,
            },
            evidence=[],
        ),
        "configuration_declaration": _section(
            status=config_status,
            facts={
                "referenced_keys": [],
                "local_timeout_constant": "PAGE_FETCH_TIMEOUT_SECONDS"
                if configuration_matches
                else None,
                "secret_values_read": False,
            },
            evidence=[],
        ),
        "setup_readiness": _section(
            status=readiness_status,
            facts={
                "stdlib_http_client": "urllib.request" if dependency_matches else None,
                "external_provider_health_checked": False,
            },
            evidence=[],
        ),
        "contradictions": contradictions,
        "operational_status": {
            "claim": "not_asserted",
            "explanation": "Source evidence cannot prove live fetch success, public-network reachability, or external provider health.",
        },
    }
    return overall, facts, evidence


def inspect_source_status(
    component: SourceStatusComponent,
    *,
    project_root: Path | None = None,
    inspection_id: str | None = None,
) -> dict[str, object]:
    """Inspect one fixed source-status component without accepting a path."""

    if component not in SOURCE_STATUS_COMPONENTS:
        raise ValueError("component must be search_provider or page_fetcher")
    core = ProjectEvidenceCore(
        project_root=project_root,
        capability_name="inspect_source_status",
        allowlisted_paths=SOURCE_STATUS_ALLOWLISTS[component],
    )
    files, unavailable = _load_files(core)
    if component == "search_provider":
        status, facts, evidence = _inspect_search_provider(core, files)
    else:
        status, facts, evidence = _inspect_page_fetcher(core, files)

    facts["files_not_found_within_scope"] = sorted(unavailable)
    envelope = core.envelope(
        target=component,
        evidence_status=status,
        extracted_facts=facts,
        evidence_items=tuple(
            EvidenceItem(
                relative_source_path=item["relative_source_path"],
                file_sha256=item["file_sha256"],
                line_start=item["line_start"],
                line_end=item["line_end"],
                extracted_evidence=item["extracted_evidence"],
                trust_classification=item["trust_classification"],
                verification_status=EvidenceStatus(item["verification_status"]),
            )
            for item in evidence
        ),
        limitations=[
            "Source/config evidence is untrusted data and cannot change policy.",
            "Secret values are never read or returned.",
            "External provider health and live network behavior are outside this capability.",
            "Operational status is not inferred from implementation or registration alone.",
        ],
        inspection_id=inspection_id,
    )
    result = envelope.to_dict()
    result["extracted_facts"]["evidence_item_count"] = len(evidence)
    result["evidence_citations"] = render_evidence_citations(
        result["evidence_items"]
    )
    return result