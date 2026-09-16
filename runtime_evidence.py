from __future__ import annotations

import re
import shlex
import tomllib
import hashlib
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_EVIDENCE_FILES = ("server.py", "pyproject.toml", ".replit")
MAX_EVIDENCE_FILE_BYTES = 256_000
MAX_EVIDENCE_SNIPPET_LENGTH = 240


class RuntimeEvidenceError(RuntimeError):
    """Raised when the fixed runtime evidence boundary cannot be enforced."""


def _safe_evidence_path(project_root: Path, relative_name: str) -> Path:
    if relative_name not in RUNTIME_EVIDENCE_FILES:
        raise RuntimeEvidenceError("runtime evidence file is not allowlisted")

    root = project_root.resolve(strict=True)
    candidate = root / relative_name
    if candidate.is_symlink():
        raise RuntimeEvidenceError("runtime evidence symlinks are not allowed")
    canonical = candidate.resolve(strict=True)
    if canonical.parent != root or canonical.name != relative_name:
        raise RuntimeEvidenceError("runtime evidence path escaped project root")
    if not canonical.is_file():
        raise RuntimeEvidenceError("runtime evidence path is not a regular file")
    if canonical.stat().st_size > MAX_EVIDENCE_FILE_BYTES:
        raise RuntimeEvidenceError("runtime evidence file is too large")
    return canonical


def _read_allowlisted_files(project_root: Path) -> dict[str, list[str]]:
    contents: dict[str, list[str]] = {}
    for relative_name in RUNTIME_EVIDENCE_FILES:
        try:
            path = _safe_evidence_path(project_root, relative_name)
        except FileNotFoundError:
            contents[relative_name] = []
            continue
        contents[relative_name] = path.read_text(encoding="utf-8").splitlines()
    return contents


def _safe_snippet(line: str) -> str:
    snippet = re.sub(
        r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,]+",
        r"\1=[REDACTED]",
        line,
    ).strip()
    snippet = re.sub(
        r"(?i)(?:os\.)?environ(?:\.get)?\s*\([^)]*\)",
        "environment_access([REDACTED])",
        snippet,
    )
    if len(snippet) > MAX_EVIDENCE_SNIPPET_LENGTH:
        return f"{snippet[:MAX_EVIDENCE_SNIPPET_LENGTH - 1]}…"
    return snippet


def _citation_ready_reference(
    reference: dict[str, object],
    *,
    file_hashes: dict[str, str],
    trust_by_file: dict[str, str],
) -> dict[str, object]:
    filename = reference.get("file")
    line_start = reference.get("line_start")
    line_end = reference.get("line_end")
    if (
        not isinstance(filename, str)
        or not isinstance(line_start, int)
        or not isinstance(line_end, int)
        or filename not in file_hashes
    ):
        return reference
    return {
        **reference,
        "relative_source_path": filename,
        "file_sha256": file_hashes[filename],
        "extracted_evidence": reference.get("evidence", ""),
        "trust_classification": trust_by_file[filename],
        "verification_status": "VERIFIED",
        "locator": {
            "line_start": line_start,
            "line_end": line_end,
        },
    }


def _evidence(
    filename: str,
    lines: list[str],
    predicate: Any,
) -> list[dict[str, object]]:
    references: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        if predicate(line):
            line_number = index + 1
            references.append(
                {
                    "file": filename,
                    "line_start": line_number,
                    "line_end": line_number,
                    "citation": f"[source: {filename}, line {line_number}]",
                    "evidence": _safe_snippet(line),
                }
            )
    return references[:3]


def _claim(
    value: object,
    references: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "value": value,
        "status": "verified" if references else "unverified",
        "evidence": references,
    }


def _parse_toml(lines: list[str], filename: str) -> dict[str, Any]:
    try:
        return tomllib.loads("\n".join(lines))
    except tomllib.TOMLDecodeError as error:
        raise RuntimeEvidenceError(f"{filename} is not valid TOML") from error


def _command_value(raw_command: object) -> str | None:
    if isinstance(raw_command, str):
        return raw_command.strip() or None
    if isinstance(raw_command, list) and all(
        isinstance(part, str) for part in raw_command
    ):
        return " ".join(part for part in raw_command if part).strip() or None
    return None


def _run_line_evidence(
    lines: list[str],
    source: str,
) -> list[dict[str, object]]:
    current_section = "root"
    references: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        section_match = re.match(r"^\s*\[([^\]]+)\]\s*$", line)
        if section_match:
            current_section = section_match.group(1)
            continue
        if current_section != source:
            continue
        if re.match(r"^\s*run\s*=", line) is None:
            continue
        line_number = index + 1
        references.append(
            {
                "file": ".replit",
                "line_start": line_number,
                "line_end": line_number,
                "citation": f"[source: .replit, line {line_number}]",
                "evidence": _safe_snippet(line),
            }
        )
    return references


def _runtime_command_candidates(
    replit: dict[str, Any],
    replit_lines: list[str],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    sources: list[tuple[str, object]] = [("root", replit.get("run"))]
    deployment = replit.get("deployment")
    if isinstance(deployment, dict):
        sources.append(("deployment", deployment.get("run")))

    for source, raw_command in sources:
        if raw_command is None:
            continue
        command = _command_value(raw_command)
        references = _run_line_evidence(replit_lines, source)
        candidates.append(
            {
                "source": source,
                "command": command,
                "status": "verified" if command and references else "unverified",
                "evidence": references,
            }
        )
    return candidates


def _derive_entrypoint(command: str | None) -> str | None:
    if not command:
        return None
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    for part in reversed(parts):
        if part and not part.startswith("-"):
            return part
    return None


def _select_runtime_command(
    candidates: list[dict[str, object]],
) -> dict[str, object] | None:
    if not candidates:
        return None
    # Deployment is the active runtime source when present. A root-level run
    # remains visible in runtime_command_candidates for conflict analysis.
    for source in ("deployment", "root"):
        matching = [candidate for candidate in candidates if candidate["source"] == source]
        if matching:
            return matching[0]
    return None


def inspect_runtime_evidence(
    *,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Inspect only fixed, project-local runtime evidence files.

    The optional project_root is an internal test seam. The AgentCore tool does
    not expose it to the model or to callers.
    """

    root = project_root or PROJECT_ROOT
    files = _read_allowlisted_files(root)
    server_lines = files["server.py"]
    pyproject_lines = files["pyproject.toml"]
    replit_lines = files[".replit"]

    pyproject = _parse_toml(pyproject_lines, "pyproject.toml")
    replit = _parse_toml(replit_lines, ".replit")

    project_config = pyproject.get("project")
    requires_python = (
        project_config.get("requires-python")
        if isinstance(project_config, dict)
        else None
    )
    language_references = _evidence(
        "pyproject.toml",
        pyproject_lines,
        lambda line: "requires-python" in line,
    )
    language_value = "Python" if requires_python else None
    language_version_value = (
        f"{language_value} ({requires_python})"
        if language_value and isinstance(requires_python, str)
        else None
    )

    runtime_command_candidates = _runtime_command_candidates(
        replit,
        replit_lines,
    )
    selected_runtime_command = _select_runtime_command(
        runtime_command_candidates
    )
    run_command_value = (
        selected_runtime_command.get("command")
        if selected_runtime_command
        else None
    )
    command_references = (
        selected_runtime_command.get("evidence", [])
        if selected_runtime_command
        else []
    )
    if not isinstance(command_references, list):
        command_references = []
    entrypoint_value = _derive_entrypoint(
        run_command_value if isinstance(run_command_value, str) else None
    )
    entrypoint_references = command_references
    if isinstance(entrypoint_value, str):
        try:
            _safe_evidence_path(root, entrypoint_value)
        except (FileNotFoundError, RuntimeEvidenceError):
            entrypoint_references = []
    else:
        entrypoint_references = []

    framework_references = _evidence(
        "server.py",
        server_lines,
        lambda line: (
            "from starlette" in line
            or "import starlette" in line
            or "uvicorn.run" in line
        ),
    )
    framework_value = "Starlette + Uvicorn" if framework_references else None

    mcp_references = _evidence(
        "server.py",
        server_lines,
        lambda line: "MCPServer" in line or "mcp.server" in line,
    )
    mcp_value = bool(mcp_references) if mcp_references else None

    port_references = _evidence(
        ".replit",
        replit_lines,
        lambda line: re.search(r"\b(localPort|externalPort)\s*=", line)
        is not None,
    )
    port_references.extend(
        _evidence(
            "server.py",
            server_lines,
            lambda line: "os.environ.get" in line and "PORT" in line,
        )
    )
    configured_port: object = None
    ports = replit.get("ports")
    if isinstance(ports, list) and ports:
        first_port = ports[0]
        if isinstance(first_port, dict):
            configured_port = first_port.get("localPort")
    if configured_port is None:
        configured_port = None

    runtime_evidence = {
        "language": _claim(language_value, language_references),
        "language_version": _claim(
            language_version_value,
            language_references,
        ),
        "active_entrypoint": _claim(entrypoint_value, entrypoint_references),
        "http_framework": _claim(framework_value, framework_references),
        "mcp_presence": _claim(mcp_value, mcp_references),
        "server_runtime_command": _claim(
            run_command_value,
            command_references,
        ),
        "configured_port": _claim(configured_port, port_references),
    }
    file_hashes: dict[str, str] = {}
    for filename in RUNTIME_EVIDENCE_FILES:
        try:
            file_hashes[filename] = hashlib.sha256(
                _safe_evidence_path(root, filename).read_bytes()
            ).hexdigest()
        except FileNotFoundError:
            continue
    trust_by_file = {
        "server.py": "PROJECT_SOURCE",
        ".replit": "PROJECT_CONFIGURATION",
        "pyproject.toml": "PROJECT_DEPENDENCY_DECLARATION",
    }
    for claim in runtime_evidence.values():
        if not isinstance(claim, dict) or not isinstance(claim.get("evidence"), list):
            continue
        claim["evidence"] = [
            _citation_ready_reference(
                reference,
                file_hashes=file_hashes,
                trust_by_file=trust_by_file,
            )
            for reference in claim["evidence"]
            if isinstance(reference, dict)
        ]
    statuses = [
        claim["status"]
        for claim in runtime_evidence.values()
        if isinstance(claim, dict)
    ]
    if statuses and all(status == "verified" for status in statuses):
        overall_status = "verified"
    elif any(status == "verified" for status in statuses):
        overall_status = "partial"
    else:
        overall_status = "unverified"

    evidence_references: list[dict[str, object]] = []
    for claim in runtime_evidence.values():
        if isinstance(claim, dict):
            evidence = claim.get("evidence")
            if isinstance(evidence, list):
                evidence_references.extend(
                    reference
                    for reference in evidence
                    if isinstance(reference, dict)
                )

    return {
        "status": overall_status,
        "runtime_evidence": runtime_evidence,
        "evidence_references": evidence_references,
        "evidence_files": list(RUNTIME_EVIDENCE_FILES),
        "runtime_command_candidates": runtime_command_candidates,
        "selected_runtime_command_source": (
            selected_runtime_command.get("source")
            if selected_runtime_command
            else None
        ),
    }