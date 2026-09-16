from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CITATION_RE = re.compile(
    r"\[source: (?P<source>[^;\]]+); "
    r"file: (?P<path>[^;\]]+); "
    r"lines: (?P<line_start>\d+)-(?P<line_end>\d+); "
    r"sha256: (?P<sha256>[0-9a-f]{64}); "
    r"status: (?P<status>VERIFIED|DISCREPANCY|NOT_FOUND|DENIED); "
    r"trust: (?P<trust>[A-Z_]+)\]"
)
_ALLOWED_STATUSES = {"VERIFIED", "DISCREPANCY", "NOT_FOUND", "DENIED"}
_MAX_ITEMS = 24
_MAX_CITATION_BYTES = 600
_MODEL_SOURCE_MARKER_RE = re.compile(r"\s*\[source:[^\]]+\]")
_MODEL_NUMERIC_MARKER_RE = re.compile(r"\s*\[\d+\]")


def _item_value(item: object, key: str) -> object:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _validated_item(item: object) -> dict[str, object] | None:
    path = _item_value(item, "relative_source_path")
    sha256 = _item_value(item, "file_sha256")
    line_start = _item_value(item, "line_start")
    line_end = _item_value(item, "line_end")
    status = _item_value(item, "verification_status")
    trust = _item_value(item, "trust_classification")
    if not all(isinstance(value, str) and value for value in (path, sha256, status, trust)):
        return None
    if not isinstance(line_start, int) or not isinstance(line_end, int):
        return None
    if line_start < 1 or line_end < line_start:
        return None
    if not _SHA256_RE.fullmatch(sha256.lower()) or status not in _ALLOWED_STATUSES:
        return None
    normalized_path = path.replace("\\", "/")
    if (
        ";" in path
        or "]" in path
        or ";" in trust
        or "]" in trust
        or normalized_path.startswith("/")
        or ".." in normalized_path.split("/")
    ):
        return None
    return {
        "relative_source_path": path,
        "file_sha256": sha256.lower(),
        "line_start": line_start,
        "line_end": line_end,
        "verification_status": status,
        "trust_classification": trust,
    }


def render_evidence_citation(
    item: object,
    *,
    source_label: str = "Ahmed Agent project files",
) -> str | None:
    """Render one citation using only a validated evidence item."""

    validated = _validated_item(item)
    if validated is None or not source_label or ";" in source_label or "]" in source_label:
        return None
    citation = (
        f"[source: {source_label}; "
        f"file: {validated['relative_source_path']}; "
        f"lines: {validated['line_start']}-{validated['line_end']}; "
        f"sha256: {validated['file_sha256']}; "
        f"status: {validated['verification_status']}; "
        f"trust: {validated['trust_classification']}]"
    )
    if len(citation.encode("utf-8")) > _MAX_CITATION_BYTES:
        return None
    return citation


def render_evidence_citations(
    items: Iterable[object],
    *,
    source_label: str = "Ahmed Agent project files",
) -> list[str]:
    citations: list[str] = []
    for item in list(items)[:_MAX_ITEMS]:
        citation = render_evidence_citation(item, source_label=source_label)
        if citation and citation not in citations:
            citations.append(citation)
    return citations


def build_evidence_provenance(
    items: Iterable[object],
    *,
    source_label: str = "Ahmed Agent project files",
) -> list[dict[str, object]]:
    """Return bounded provenance records only for citation-ready evidence."""

    provenance: list[dict[str, object]] = []
    for item in list(items)[:_MAX_ITEMS]:
        validated = _validated_item(item)
        citation = render_evidence_citation(item, source_label=source_label)
        if validated is None or citation is None:
            continue
        provenance.append(
            {
                "citation": citation,
                "source_identity": source_label,
                "relative_source_path": validated["relative_source_path"],
                "file_sha256": validated["file_sha256"],
                "locator": {
                    "line_start": validated["line_start"],
                    "line_end": validated["line_end"],
                },
                "verification_status": validated["verification_status"],
                "trust_classification": validated["trust_classification"],
            }
        )
    return provenance


def parse_evidence_citation(value: str) -> dict[str, object] | None:
    match = _CITATION_RE.fullmatch(value.strip())
    if not match:
        return None
    parsed = match.groupdict()
    parsed["line_start"] = int(parsed["line_start"])
    parsed["line_end"] = int(parsed["line_end"])
    return parsed


def verify_evidence_citation(
    citation: str,
    items: Iterable[object],
    *,
    expected_claim: str | None = None,
) -> bool:
    """Verify citation identity against the tool's evidence_items collection."""

    parsed = parse_evidence_citation(citation)
    if parsed is None:
        return False
    if expected_claim is not None and parsed["source"] != expected_claim:
        return False
    for item in items:
        validated = _validated_item(item)
        if validated is None:
            continue
        if (
            parsed["path"] == validated["relative_source_path"]
            and parsed["sha256"] == validated["file_sha256"]
            and parsed["line_start"] == validated["line_start"]
            and parsed["line_end"] == validated["line_end"]
            and parsed["status"] == validated["verification_status"]
            and parsed["trust"] == validated["trust_classification"]
        ):
            return True
    return False


def render_evidence_report(envelopes: Iterable[dict[str, Any]]) -> str:
    """Return bounded, deterministic citations and explicit no-claim limitations."""

    sections: list[str] = []
    for envelope in list(envelopes)[:_MAX_ITEMS]:
        target = envelope.get("target")
        status = envelope.get("evidence_status")
        if not isinstance(target, str) or not isinstance(status, str):
            continue
        source_label = envelope.get("source_label")
        if not isinstance(source_label, str):
            source_label = "Ahmed Agent project files"
        citations = render_evidence_citations(
            envelope.get("evidence_items", []),
            source_label=source_label,
        )
        if citations:
            sections.append(f"{target}: " + " ".join(citations))
        else:
            sections.append(
                f"{target}: evidence status {status}; no evidence item is available, "
                "so no source claim is asserted."
            )
    if not sections:
        return ""
    return "\n\nEvidence references:\n" + "\n".join(sections)


def remove_model_source_markers(value: str) -> str:
    """Remove model-authored source markers before canonical citations are added."""

    without_source_markers = _MODEL_SOURCE_MARKER_RE.sub("", value)
    return _MODEL_NUMERIC_MARKER_RE.sub("", without_source_markers).strip()