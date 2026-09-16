from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent
MAX_EVIDENCE_FILE_BYTES = 512_000


class EvidenceCoreError(RuntimeError):
    """Raised when the fixed project-evidence boundary cannot be enforced."""


class EvidenceStatus(StrEnum):
    VERIFIED = "VERIFIED"
    NOT_FOUND = "NOT_FOUND"
    DISCREPANCY = "DISCREPANCY"
    DENIED = "DENIED"


class EvidenceTrust(StrEnum):
    PROJECT_SOURCE = "PROJECT_SOURCE"
    PROJECT_CONFIGURATION = "PROJECT_CONFIGURATION"
    PROJECT_DEPENDENCY_DECLARATION = "PROJECT_DEPENDENCY_DECLARATION"


@dataclass(frozen=True)
class SourceFile:
    relative_path: str
    sha256: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceItem:
    relative_source_path: str
    file_sha256: str
    line_start: int
    line_end: int
    extracted_evidence: str
    trust_classification: str
    verification_status: EvidenceStatus

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_source_path": self.relative_source_path,
            "file_sha256": self.file_sha256,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "extracted_evidence": self.extracted_evidence,
            "trust_classification": self.trust_classification,
            "verification_status": self.verification_status.value,
        }


@dataclass(frozen=True)
class ProjectEvidenceEnvelope:
    schema_version: str
    capability_name: str
    inspection_id: str
    target: str
    evidence_status: EvidenceStatus
    extracted_facts: dict[str, object]
    provenance_metadata: dict[str, object]
    inspection_timestamp: str
    evidence_items: tuple[EvidenceItem, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "capability_name": self.capability_name,
            "inspection_id": self.inspection_id,
            "target": self.target,
            "evidence_status": self.evidence_status.value,
            "extracted_facts": self.extracted_facts,
            "provenance_metadata": self.provenance_metadata,
            "inspection_timestamp": self.inspection_timestamp,
            "evidence_items": [item.to_dict() for item in self.evidence_items],
        }


_DENIED_PARTS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
    ".cache",
    ".pytest_cache",
}
_DENIED_NAMES = {".env", ".env.local", ".env.production", ".env.development"}
_DENIED_SUFFIXES = (
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".jks",
    ".der",
    ".crt",
)
_SECRET_NAME_RE = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret|"
    r"credential|private[_-]?key)"
)
_ENV_ACCESS_RE = re.compile(
    r"(?P<prefix>(?:os\.)?environ(?:\.get)?\s*\(\s*)"
    r"(?P<quote>['\"])(?P<name>[A-Z][A-Z0-9_]+)(?P=quote)"
)
_ASSIGNMENT_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_-]*)\s*=\s*(?P<value>[^#\n]+)"
)


def _redact_source_line(line: str) -> str:
    """Keep source context while excluding credential values."""

    def replace_assignment(match: re.Match[str]) -> str:
        name = match.group("name")
        if not _SECRET_NAME_RE.search(name):
            return match.group(0)
        return f"{name}=[REDACTED]"

    redacted = _ASSIGNMENT_RE.sub(replace_assignment, line)
    redacted = _ENV_ACCESS_RE.sub(
        lambda match: (
            f"{match.group('prefix')}{match.group('quote')}"
            f"{match.group('name')}{match.group('quote')})"
        ),
        redacted,
    )
    return redacted.strip()[:360]


def _contains_encoded_traversal(value: str) -> bool:
    decoded = value
    for _ in range(2):
        decoded = unquote(decoded)
    return decoded != value and (
        ".." in decoded or "/" in decoded or "\\" in decoded
    )


def _is_denied_relative_path(relative_path: str) -> bool:
    path = Path(relative_path)
    parts = {part.casefold() for part in path.parts}
    name = path.name.casefold()
    return (
        bool(parts & _DENIED_PARTS)
        or name in _DENIED_NAMES
        or name.startswith(".env.")
        or name.endswith(_DENIED_SUFFIXES)
        or bool(_SECRET_NAME_RE.search(name))
    )


def _is_globally_allowlisted(relative_path: str) -> bool:
    path = Path(relative_path)
    if _is_denied_relative_path(relative_path):
        return False
    if relative_path in {
        ".replit",
        "package.json",
        "pyproject.toml",
        "requirements.txt",
    }:
        return True
    if path.suffix == ".py":
        return True
    return any(
        path == directory or directory in path.parents
        for directory in (Path("src"), Path("config"), Path("docs"), Path("ADR"))
    )


class ProjectEvidenceCore:
    """Bounded read-only evidence access for internal capabilities.

    The model never receives this class or its path-taking methods. Capabilities
    provide a fixed exact allowlist and expose only structured evidence.
    """

    schema_version = "project-evidence.v1"

    def __init__(
        self,
        *,
        project_root: Path | None = None,
        capability_name: str,
        allowlisted_paths: Iterable[str],
    ) -> None:
        root = (project_root or PROJECT_ROOT).resolve(strict=True)
        if not root.is_dir():
            raise EvidenceCoreError("project root is not a directory")
        paths = tuple(dict.fromkeys(allowlisted_paths))
        if not paths or any(
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or _contains_encoded_traversal(path)
            or ".." in Path(path).parts
            or not _is_globally_allowlisted(path)
            for path in paths
        ):
            raise EvidenceCoreError("capability allowlist is outside the project policy")
        self._project_root = root
        self.capability_name = capability_name
        self.allowlisted_paths = frozenset(paths)

    def _safe_path(self, relative_path: str) -> Path:
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or _contains_encoded_traversal(relative_path)
            or ".." in Path(relative_path).parts
        ):
            raise EvidenceCoreError("project evidence path is invalid")
        if relative_path not in self.allowlisted_paths:
            raise EvidenceCoreError("project evidence path is not capability-allowlisted")
        if _is_denied_relative_path(relative_path):
            raise EvidenceCoreError("project evidence path is denied")

        candidate = self._project_root / relative_path
        if candidate.is_symlink():
            raise EvidenceCoreError("project evidence symlinks are not allowed")
        try:
            canonical = candidate.resolve(strict=True)
        except FileNotFoundError:
            raise
        try:
            canonical.relative_to(self._project_root)
        except ValueError as error:
            raise EvidenceCoreError("project evidence path escaped project root") from error
        if not canonical.is_file():
            raise EvidenceCoreError("project evidence path is not a regular file")
        if canonical.stat().st_size > MAX_EVIDENCE_FILE_BYTES:
            raise EvidenceCoreError("project evidence file is too large")
        return canonical

    def read_file(self, relative_path: str) -> SourceFile:
        path = self._safe_path(relative_path)
        content = path.read_bytes()
        return SourceFile(
            relative_path=relative_path,
            sha256=hashlib.sha256(content).hexdigest(),
            lines=tuple(content.decode("utf-8", errors="replace").splitlines()),
        )

    @staticmethod
    def line_evidence(
        source_file: SourceFile,
        *,
        line_start: int,
        line_end: int | None = None,
        trust_classification: EvidenceTrust | str = EvidenceTrust.PROJECT_SOURCE,
        verification_status: EvidenceStatus = EvidenceStatus.VERIFIED,
    ) -> EvidenceItem:
        if line_start < 1 or line_start > max(len(source_file.lines), 1):
            raise EvidenceCoreError("evidence line is outside the source file")
        end = line_end or line_start
        if end < line_start or end > len(source_file.lines):
            raise EvidenceCoreError("evidence line range is invalid")
        extracted = " ".join(
            _redact_source_line(line)
            for line in source_file.lines[line_start - 1 : end]
        ).strip()
        return EvidenceItem(
            relative_source_path=source_file.relative_path,
            file_sha256=source_file.sha256,
            line_start=line_start,
            line_end=end,
            extracted_evidence=extracted,
            trust_classification=str(trust_classification),
            verification_status=verification_status,
        )

    def envelope(
        self,
        *,
        target: str,
        evidence_status: EvidenceStatus,
        extracted_facts: dict[str, object],
        evidence_items: Iterable[EvidenceItem],
        limitations: list[str],
        inspection_id: str | None = None,
    ) -> ProjectEvidenceEnvelope:
        return ProjectEvidenceEnvelope(
            schema_version=self.schema_version,
            capability_name=self.capability_name,
            inspection_id=inspection_id or str(uuid4()),
            target=target,
            evidence_status=evidence_status,
            extracted_facts=extracted_facts,
            provenance_metadata={
                "project_root": ".",
                "allowlisted_paths": sorted(self.allowlisted_paths),
                "limitations": limitations,
                "evidence_is_untrusted_data": True,
            },
            inspection_timestamp=datetime.now(timezone.utc).isoformat(),
            evidence_items=tuple(evidence_items),
        )