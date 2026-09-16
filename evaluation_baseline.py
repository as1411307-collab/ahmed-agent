from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import UUID, uuid4

from evidence_citations import (
    build_evidence_provenance,
    parse_evidence_citation,
    verify_evidence_citation,
)

from agent_core import MAX_MODEL_REQUESTS, MAX_TOOL_CALLS, provider_model_name
from persistence import (
    cleanup_evaluation_run,
    delete_documents_for_source_ids,
    delete_original_source,
    load_run_evaluation_data,
)


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


def _tool_names(trace: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for call in trace.get("tool_calls", []):
        if isinstance(call, str):
            names.add(call)
        elif isinstance(call, dict) and isinstance(call.get("name"), str):
            names.add(call["name"])
    return names


def resolve_tool_expectation(semantic_name: str) -> dict[str, Any]:
    return TOOL_NAME_MAPPING.get(
        semantic_name,
        {
            "actual": semantic_name,
            "status": "unmapped",
            "note": "No semantic-to-runtime mapping has been declared.",
        },
    )


def resolve_case_tool_expectation(
    case_id: str,
    semantic_name: str,
) -> dict[str, Any]:
    return CASE_CAPABILITY_OVERRIDES.get(case_id, {}).get(
        semantic_name,
        resolve_tool_expectation(semantic_name),
    )


def case_execution_capability(case: dict[str, Any]) -> dict[str, Any]:
    unavailable_required = [
        tool
        for tool in case["required_tools"]
        if resolve_case_tool_expectation(case["id"], tool)["status"]
        == "unavailable"
    ]
    unmapped_required = [
        tool
        for tool in case["required_tools"]
        if resolve_case_tool_expectation(case["id"], tool)["status"] == "unmapped"
    ]
    boundary_required = [
        tool
        for tool in case["required_tools"]
        if resolve_case_tool_expectation(case["id"], tool)["status"]
        == "boundary_only"
    ]
    return {
        "executable_by_current_agent_tools": not (
            unavailable_required or unmapped_required or boundary_required
        ),
        "unavailable_required_tools": unavailable_required,
        "unmapped_required_tools": unmapped_required,
        "boundary_required_tools": boundary_required,
        "resolved_required_tools": {
            tool: resolve_case_tool_expectation(case["id"], tool)
            for tool in case["required_tools"]
        },
    }


def redact_evaluation_text(value: str) -> str:
    redacted = re.sub(
        r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+",
        r"\1[REDACTED]",
        value,
    )
    redacted = re.sub(
        r"(?i)\b(?:api[_-]?key|token|password|secret)\s*[:=]\s*\S+",
        lambda match: match.group(0).split(match.group(0)[-1], 1)[0]
        + "[REDACTED]",
        redacted,
    )
    return redacted


def _extract_trace_sources(final_output: str) -> list[str]:
    sources = re.findall(r"\[(?:source:[^\]]+|\d+)\]", final_output)
    sources.extend(re.findall(r"https?://[^\s)\]}]+", final_output))
    return sorted(set(sources))


def _post_chat_message(
    *,
    base_url: str,
    owner_token: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/message",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {owner_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            return (
                response.status,
                json.loads(body) if body else {},
                {key: value for key, value in response.headers.items()},
            )
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"error": "HTTP_ERROR"}
        return (
            error.code,
            parsed,
            {key: value for key, value in error.headers.items()}
            if error.headers
            else {},
        )
    except Exception as error:
        return 0, {"error": type(error).__name__}, {}


def _post_file_upload(
    *,
    base_url: str,
    owner_token: str,
    files: list[dict[str, object]],
    timeout_seconds: float,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    """Post a real multipart upload without logging file contents."""

    boundary = f"----ahmed-evaluation-{uuid4().hex}"
    body = bytearray()
    for file in files:
        filename = str(file["filename"])
        content = file["content"]
        if not isinstance(content, bytes):
            raise TypeError("evaluation upload content must be bytes")
        mime_type = str(file.get("mime_type") or "application/octet-stream")
        safe_header_name = filename.replace("\\", "_").replace('"', "")
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                'Content-Disposition: form-data; name="file"; '
                f'filename="{safe_header_name}"\r\n'
            ).encode("utf-8")
        )
        body.extend(f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"))
        body.extend(content)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/files/upload",
        data=bytes(body),
        headers={
            "Authorization": f"Bearer {owner_token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
            return (
                response.status,
                json.loads(response_body) if response_body else {},
                {key: value for key, value in response.headers.items()},
            )
    except urllib.error.HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError:
            parsed = {"error": "HTTP_ERROR"}
        return (
            error.code,
            parsed,
            {key: value for key, value in error.headers.items()}
            if error.headers
            else {},
        )
    except Exception as error:
        return 0, {"error": type(error).__name__}, {}


def _minimal_pdf_bytes() -> bytes:
    content = b"BT /F1 12 Tf 10 100 Td (AA-RC-018) Tj ET\n"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n"
        + content
        + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(body)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(output)


def _minimal_docx_bytes() -> bytes:
    import io
    import zipfile

    files = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>"
        ),
        "_rels/.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/>'
            "</Relationships>"
        ),
        "word/document.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>AA-RC-018</w:t></w:r></w:p>"
            "<w:sectPr/></w:body></w:document>"
        ),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _aa_rc_018_upload_files() -> list[dict[str, object]]:
    return [
        {
            "filename": "aa-rc-018-upload.txt",
            "content": b"AA-RC-018 plain text evidence.",
            "mime_type": "text/plain",
        },
        {
            "filename": "aa-rc-018-upload.md",
            "content": b"# AA-RC-018\nMarkdown evidence.",
            "mime_type": "text/markdown",
        },
        {
            "filename": "aa-rc-018-upload.pdf",
            "content": _minimal_pdf_bytes(),
            "mime_type": "application/pdf",
        },
        {
            "filename": "aa-rc-018-upload.docx",
            "content": _minimal_docx_bytes(),
            "mime_type": (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ),
        },
    ]


async def _cleanup_uploaded_sources(source_ids: list[str]) -> dict[str, object]:
    cleaned: list[str] = []
    errors: list[str] = []
    for source_id in dict.fromkeys(source_ids):
        try:
            try:
                UUID(source_id)
            except ValueError:
                pass
            else:
                await delete_documents_for_source_ids([source_id])
            await delete_original_source(source_id=source_id)
            cleaned.append(source_id)
        except Exception as error:
            errors.append(f"{source_id}:{type(error).__name__}")
    return {
        "status": "PASS" if not errors else "FAIL",
        "cleaned_source_ids": cleaned,
        "errors": errors,
    }


async def _run_aa_rc_018_upload_e2e(
    *,
    base_url: str,
    owner_token: str,
    timeout_seconds: float,
) -> dict[str, object]:
    source_ids: list[str] = []
    supported_status, supported_payload, _ = await asyncio.to_thread(
        _post_file_upload,
        base_url=base_url,
        owner_token=owner_token,
        files=_aa_rc_018_upload_files(),
        timeout_seconds=timeout_seconds,
    )
    supported_results = supported_payload.get("files", [])
    if isinstance(supported_results, list):
        source_ids.extend(
            str(item["source_id"])
            for item in supported_results
            if isinstance(item, dict) and isinstance(item.get("source_id"), str)
        )
    unsafe_status, unsafe_payload, _ = await asyncio.to_thread(
        _post_file_upload,
        base_url=base_url,
        owner_token=owner_token,
        files=[
            {
                "filename": "../aa-rc-018-unsafe.txt",
                "content": b"unsafe path traversal name",
                "mime_type": "text/plain",
            }
        ],
        timeout_seconds=timeout_seconds,
    )
    invalid_status, invalid_payload, _ = await asyncio.to_thread(
        _post_file_upload,
        base_url=base_url,
        owner_token=owner_token,
        files=[
            {
                "filename": "aa-rc-018-invalid.txt",
                "content": b"\x00\xff\x00not-valid-text",
                "mime_type": "text/plain",
            }
        ],
        timeout_seconds=timeout_seconds,
    )
    cleanup = await _cleanup_uploaded_sources(source_ids)
    supported_ready = (
        supported_status == 201
        and len(supported_results) == 4
        and all(
            isinstance(item, dict) and item.get("status") in {"ready", "embedding_failed"}
            for item in supported_results
        )
    )
    unsafe_rejected = unsafe_status == 415 and (
        unsafe_payload.get("code") == "UNSAFE_FILENAME"
        or unsafe_payload.get("error") == "UNSAFE_FILENAME"
    )
    invalid_rejected = invalid_status == 415 and (
        any(
            isinstance(item, dict) and item.get("status") == "invalid_file_content"
            for item in (invalid_payload.get("files") or [])
        )
    )
    evidence_items = [
        {
            "relative_source_path": str(item.get("filename")),
            "file_sha256": hashlib.sha256(
                next(
                    file["content"]
                    for file in _aa_rc_018_upload_files()
                    if file["filename"] == item.get("filename")
                )
            ).hexdigest(),
            "line_start": 1,
            "line_end": 1,
            "verification_status": "VERIFIED",
            "trust_classification": "OWNER_UPLOADED",
        }
        for item in supported_results
        if isinstance(item, dict) and isinstance(item.get("filename"), str)
    ]
    return {
        "status": (
            "PASS"
            if supported_ready and unsafe_rejected and invalid_rejected and cleanup["status"] == "PASS"
            else "FAIL"
        ),
        "supported_type_count": len(supported_results),
        "supported_file_statuses": [
            {
                "filename": item.get("filename"),
                "status": item.get("status"),
                "source_id": item.get("source_id"),
            }
            for item in supported_results
            if isinstance(item, dict)
        ],
        "supported_upload_status": supported_status,
        "unsafe_filename_rejected": unsafe_rejected,
        "unsafe_filename_status": unsafe_status,
        "invalid_content_rejected": invalid_rejected,
        "invalid_content_status": invalid_status,
        "cleanup": cleanup,
        "evidence_items": evidence_items,
    }


def _is_probable_documentation_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    host = (parsed.hostname or "").casefold()
    path = parsed.path.casefold()
    if parsed.scheme not in {"http", "https"} or not host:
        return False
    if host in {"facebook.com", "www.facebook.com", "linkedin.com", "www.linkedin.com", "youtube.com", "www.youtube.com"}:
        return False
    return any(
        marker in host or marker in path
        for marker in ("docs", "developer", "reference", "api", "documentation", "github")
    )


def _is_official_openai_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    host = (parsed.hostname or "").casefold().rstrip(".")
    return (
        parsed.scheme in {"http", "https"}
        and (
            host == "openai.com"
            or host.endswith(".openai.com")
            or host == "openai.github.io"
        )
    )


def validate_case_evidence_preconditions(
    *,
    case: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, object]:
    """Validate evidence prerequisites before semantic scoring."""

    if case.get("id") != "AA-RC-026":
        return {"status": "READY", "missing": []}
    tool_calls = trace.get("tool_calls", [])
    names = {
        str(item.get("name"))
        for item in tool_calls
        if isinstance(item, dict) and item.get("status") in {"success", "succeeded"}
    }
    architecture_evidence = any(
        item.get("name") == "inspect_architecture_evidence"
        and (
            item.get("metadata", {}).get("evidence_provenance")
            or item.get("metadata", {}).get("evidence_citations")
        )
        for item in tool_calls
        if isinstance(item, dict)
    )
    external_urls = [
        url
        for item in tool_calls
        if isinstance(item, dict)
        for url in item.get("metadata", {}).get("external_source_urls", [])
        if isinstance(url, str)
    ]
    external_provenance = [
        provenance
        for item in tool_calls
        if isinstance(item, dict)
        for provenance in item.get("metadata", {}).get(
            "external_evidence_provenance",
            [],
        )
        if isinstance(provenance, dict)
    ]
    answer = str(
        (trace.get("output") or {}).get("answer")
        or trace.get("final_output")
        or ""
    ).casefold()
    missing: list[str] = []
    if "inspect_architecture_evidence" not in names or not architecture_evidence:
        missing.append("architecture_evidence")
    documentation_urls = [
        url for url in external_urls if _is_probable_documentation_url(url)
    ]
    official_openai_urls = [
        url for url in documentation_urls if _is_official_openai_url(url)
    ]
    if "web_search" not in names or not documentation_urls:
        missing.append("verified_external_documentation")
    if not official_openai_urls:
        missing.append("official_openai_documentation")
    if not any(
        provenance.get("verification_status") == "UNVERIFIED_EXTERNAL"
        and _is_official_openai_url(provenance.get("url"))
        for provenance in external_provenance
    ):
        missing.append("external_provenance_classification")
    if not trace.get("citations") and not trace.get("sources"):
        missing.append("answer_citations")
    comparison_groups = {
        "current_project_state": (
            "current project",
            "current state",
            "حالة المشروع",
            "المشروع الحالي",
        ),
        "migration_cost": ("migration", "الهجرة", "ترحيل"),
        "maintenance_burden": ("maintenance", "الصيانة", "صيانة"),
        "quality_control": ("quality", "الجودة", "control", "تحكم", "سيطرة"),
        "operating_cost": (
            "operating cost",
            "operational cost",
            "التكلفة التشغيلية",
            "تكلفة التشغيل",
        ),
        "features_gained_lost": (
            "feature",
            "features",
            "الميزات",
            "المزايا",
            "gained",
            "lost",
            "المكتسبة",
            "المفقودة",
        ),
        "provider_independence": (
            "provider independence",
            "vendor lock-in",
            "استقلالية مزود",
            "الاعتماد على مزود",
        ),
        "local_self_hosted_fallback": (
            "self-hosted",
            "self hosted",
            "local fallback",
            "بديل محلي",
            "استضافة ذاتية",
            "تشغيل محلي",
        ),
        "provenance_governance": (
            "provenance",
            "governance",
            "حوكمة",
            "سلسلة المصدر",
        ),
        "clear_recommendation": ("recommend", "التوصية", "أوصي", "أنصح"),
    }
    missing_comparison = [
        name
        for name, markers in comparison_groups.items()
        if not any(marker in answer for marker in markers)
    ]
    missing.extend(missing_comparison)
    if missing_comparison:
        missing.append("tradeoff_criteria")
    return {
        "status": "READY" if not missing else "NOT_DETERMINED",
        "missing": missing,
        "observed_tool_names": sorted(names),
        "external_documentation_urls": documentation_urls,
        "official_openai_documentation_urls": official_openai_urls,
    }


def _retry_after_seconds(headers: dict[str, str]) -> float | None:
    value = headers.get("Retry-After") or headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _case_scope(case: dict[str, Any]) -> str:
    return (
        "MY_FILES"
        if "my_files" in case["required_tools"]
        or case.get("category") == "source_of_truth"
        else "WEB"
    )


def build_evaluation_trace(
    *,
    case: dict[str, Any],
    run_id: str,
    scope: str,
    provider: str,
    response_status: int,
    response_payload: dict[str, Any],
    persisted: dict[str, Any],
    latency_ms: int,
    retry_after_seconds: float | None = None,
) -> dict[str, Any]:
    def safe_event_metadata(event: dict[str, Any]) -> dict[str, Any]:
        metadata = event.get("safe_metadata")
        if isinstance(metadata, dict):
            return metadata
        if isinstance(metadata, str):
            try:
                decoded = json.loads(metadata)
            except json.JSONDecodeError:
                return {}
            return decoded if isinstance(decoded, dict) else {}
        return {}

    run = persisted.get("run") or {}
    final_output = response_payload.get("reply")
    if not isinstance(final_output, str):
        final_output = ""
    final_output = redact_evaluation_text(final_output)
    tool_events = persisted.get("tool_events", [])
    pending_actions = persisted.get("pending_actions", [])
    tool_calls = [
        {
            "name": event.get("tool_name"),
            "status": event.get("status"),
            "duration_ms": event.get("duration_ms"),
            "metadata": safe_event_metadata(event),
        }
        for event in tool_events
    ]
    available_evidence_citations = sorted(
        {
            citation
            for event in tool_events
            for citation in safe_event_metadata(event).get(
                "evidence_citations", []
            )
            if isinstance(citation, str)
        }
    )
    available_evidence_items = [
        item
        for event in tool_events
        for item in safe_event_metadata(event).get("evidence_items", [])
        if isinstance(item, dict)
    ]
    evidence_provenance = [
        provenance
        for event in tool_events
        for provenance in safe_event_metadata(event).get(
            "evidence_provenance",
            [],
        )
        if isinstance(provenance, dict)
    ]
    if not evidence_provenance:
        evidence_provenance = build_evidence_provenance(available_evidence_items)
    evidence_provenance = list(
        {
            json.dumps(item, ensure_ascii=False, sort_keys=True): item
            for item in evidence_provenance
        }.values()
    )[:24]
    external_evidence_provenance = [
        provenance
        for event in tool_events
        for provenance in safe_event_metadata(event).get(
            "external_evidence_provenance",
            [],
        )
        if isinstance(provenance, dict)
    ]
    external_evidence_provenance = list(
        {
            json.dumps(item, ensure_ascii=False, sort_keys=True): item
            for item in external_evidence_provenance
        }.values()
    )[:24]
    if not available_evidence_citations:
        available_evidence_citations = sorted(
            {
                str(item["citation"])
                for item in evidence_provenance
                if isinstance(item.get("citation"), str)
            }
        )
    pending_events = [
        {
            "tool_name": event.get("tool_name"),
            "risk_level": event.get("risk_level"),
            "status": event.get("status"),
        }
        for event in pending_actions
    ]
    is_http_success = 200 <= response_status < 300
    run_status = run.get("status")
    if not is_http_success or run_status == "failed":
        execution_status = "EXECUTION_FAILED"
    elif pending_events:
        execution_status = "HITL_BLOCKED"
    else:
        execution_status = "EXECUTED"
    failure_reason = (
        run.get("error_code")
        or response_payload.get("error")
        if execution_status == "EXECUTION_FAILED"
        else None
    )
    if isinstance(failure_reason, str):
        failure_reason = redact_evaluation_text(failure_reason)
    return {
        "case_id": case["id"],
        "run_id": run_id,
        "input": redact_evaluation_text(case["input"]),
        "scope": scope,
        "provider": provider,
        "model": run.get("model_name") or provider_model_name(provider),
        "final_output": final_output,
        "output": {"answer": final_output},
        "tool_calls": tool_calls,
        "citations": _extract_trace_sources(final_output),
        "sources": _extract_trace_sources(final_output),
        "available_evidence_citations": available_evidence_citations,
        "available_evidence_items": available_evidence_items,
        "evidence_provenance": evidence_provenance,
        "external_evidence_provenance": external_evidence_provenance,
        "pending_action_events": pending_events,
        "approval_requested": bool(pending_events),
        "executed_without_approval": any(
            event.get("status") in {"executing", "executed"}
            for event in pending_actions
        ),
        "abstained": None,
        "start_timestamp": run.get("created_at"),
        "end_timestamp": run.get("finished_at"),
        "latency_ms": latency_ms,
        "run_attempts": run.get("attempt_count"),
        "retries": None,
        "tokens": None,
        "cost": None,
        "execution_status": execution_status,
        "execution_error": failure_reason,
        "retry_after_seconds": retry_after_seconds,
    }


def authorized_source_identity_preconditions(
    *,
    case: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    """Require every AA-RC-002 expected source identity to be observed."""

    expected = [
        str(value).strip().casefold()
        for value in case.get("expected_sources", [])
        if str(value).strip()
    ]
    observed: list[str] = []
    for call in trace.get("tool_calls", []):
        if not isinstance(call, dict) or call.get("name") != "inspect_source_of_truth":
            continue
        metadata = call.get("metadata")
        if not isinstance(metadata, dict):
            continue
        for key in ("source_filename", "source_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                observed.append(value.strip().casefold())
        for item in metadata.get("evidence_items", []):
            if not isinstance(item, dict):
                continue
            value = item.get("relative_source_path")
            if isinstance(value, str) and value.strip():
                observed.append(value.strip().casefold())
    missing = [
        source
        for source in expected
        if not any(source in candidate for candidate in observed)
    ]
    return {
        "status": "VERIFIED" if not missing else "NOT_DETERMINED",
        "expected_source_count": len(expected),
        "observed_source_count": len(set(observed)),
        "missing_sources": missing,
    }


async def execute_real_case(
    case: dict[str, Any],
    *,
    base_url: str,
    owner_token: str,
    provider: str = "gemini",
    timeout_seconds: float = 90.0,
) -> dict[str, Any]:
    capability = case_execution_capability(case)
    if not capability["executable_by_current_agent_tools"]:
        return {
            "case_id": case["id"],
            "execution_status": "NOT_EXECUTABLE_CAPABILITY_GAP",
            "capability": capability,
        }
    session_id = str(uuid4())
    run_id = str(uuid4())
    scope = _case_scope(case)
    payload = {
        "message": case["input"],
        "conversation_id": session_id,
        "run_id": run_id,
        "scope": scope,
        "provider": provider,
    }
    upload_e2e: dict[str, object] | None = None
    if case["id"] == "AA-RC-018":
        upload_e2e = await _run_aa_rc_018_upload_e2e(
            base_url=base_url,
            owner_token=owner_token,
            timeout_seconds=timeout_seconds,
        )
    started = time.perf_counter()
    response_status, response_payload, response_headers = await asyncio.to_thread(
        _post_chat_message,
        base_url=base_url,
        owner_token=owner_token,
        payload=payload,
        timeout_seconds=timeout_seconds,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    try:
        persisted = await load_run_evaluation_data(run_id)
    except Exception as error:
        persisted = {"run": None, "tool_events": [], "pending_actions": []}
        response_payload = {
            **response_payload,
            "error": f"TRACE_PERSISTENCE_READ_{type(error).__name__}",
        }
    trace = build_evaluation_trace(
        case=case,
        run_id=run_id,
        scope=scope,
        provider=provider,
        response_status=response_status,
        response_payload=response_payload,
        persisted=persisted,
        latency_ms=latency_ms,
        retry_after_seconds=_retry_after_seconds(response_headers),
    )
    try:
        trace["runtime_cleanup"] = await cleanup_evaluation_run(
            run_id=run_id,
            session_id=session_id,
        )
    except Exception as error:
        trace["runtime_cleanup"] = {
            "status": "FAIL",
            "error": type(error).__name__,
            "audit_events_preserved": True,
        }
    if case["id"] == "AA-RC-002":
        source_identity_preconditions = authorized_source_identity_preconditions(
            case=case,
            trace=trace,
        )
        trace["source_identity_preconditions"] = source_identity_preconditions
        if source_identity_preconditions["status"] != "VERIFIED":
            trace["external_input_blocker"] = {
                "status": "NOT_DETERMINED",
                "code": (
                    "AUTHORIZED_SOURCE_DOCUMENTS_MISSING"
                    if not source_identity_preconditions["observed_source_count"]
                    else "AUTHORIZED_SOURCE_DOCUMENT_IDENTITIES_MISMATCHED"
                ),
                "reason": (
                    "The expected AA-RC-002 source identities were not all observed "
                    "with verified canonical provenance."
                ),
            }
    if upload_e2e is not None:
        trace["upload_e2e"] = upload_e2e
        upload_items = upload_e2e.get("evidence_items", [])
        if isinstance(upload_items, list):
            trace["available_evidence_items"] = [
                *trace.get("available_evidence_items", []),
                *[item for item in upload_items if isinstance(item, dict)],
            ][:24]
            upload_provenance = build_evidence_provenance(
                upload_items,
                source_label="AA-RC-018 evaluation uploads",
            )
            trace["evidence_provenance"] = [
                *trace.get("evidence_provenance", []),
                *upload_provenance,
            ][:24]
            trace["available_evidence_citations"] = sorted(
                {
                    *[
                        value
                        for value in trace.get("available_evidence_citations", [])
                        if isinstance(value, str)
                    ],
                    *[
                        str(item["citation"])
                        for item in upload_provenance
                        if isinstance(item.get("citation"), str)
                    ],
                }
            )
    if case["id"] == "AA-RC-026":
        preconditions = validate_case_evidence_preconditions(
            case=case,
            trace=trace,
        )
        trace["evidence_preconditions"] = preconditions
        if preconditions["status"] == "NOT_DETERMINED":
            trace["execution_status"] = "NOT_DETERMINED"
            trace["execution_error"] = "EVIDENCE_PRECONDITIONS_UNMET"
    return trace


async def execute_rate_limited_case_with_retry(
    case: dict[str, Any],
    *,
    base_url: str,
    owner_token: str,
    provider: str,
    timeout_seconds: float,
    max_retries: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
) -> dict[str, Any]:
    attempts = 0
    retry_delays: list[float] = []
    resumed_at = datetime.now(timezone.utc).isoformat()
    trace: dict[str, Any] = {}
    while True:
        attempts += 1
        trace = await execute_real_case(
            case,
            base_url=base_url,
            owner_token=owner_token,
            provider=provider,
            timeout_seconds=timeout_seconds,
        )
        rate_limited = trace.get("execution_error") == "RATE_LIMITED"
        if not rate_limited or len(retry_delays) >= max_retries:
            break
        exponential_delay = min(
            max_delay_seconds,
            base_delay_seconds * (2 ** len(retry_delays)),
        )
        jitter = random.uniform(0.0, max(1.0, base_delay_seconds * 0.25))
        retry_after = trace.get("retry_after_seconds")
        delay = min(
            max_delay_seconds,
            max(float(retry_after or 0.0), exponential_delay + jitter),
        )
        retry_delays.append(round(delay, 3))
        await asyncio.sleep(delay)
    trace["attempts"] = attempts
    trace["retries"] = len(retry_delays)
    trace["retry_delays_seconds"] = retry_delays
    trace["resumed_execution_timestamp"] = resumed_at
    if trace.get("execution_error") == "RATE_LIMITED":
        trace["execution_status"] = "EXECUTION_FAILED_PROVIDER_RATE_LIMIT"
    return trace


async def run_real_cases(
    cases: list[dict[str, Any]],
    *,
    base_url: str,
    owner_token: str,
    provider: str = "gemini",
    timeout_seconds: float = 90.0,
    case_ids: set[str] | None = None,
    inter_case_delay_seconds: float = 0.0,
) -> dict[str, Any]:
    selected = [case for case in cases if case_ids is None or case["id"] in case_ids]
    baseline_run_id = str(uuid4())
    case_results: list[dict[str, Any]] = []
    traces: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(selected):
        if index and inter_case_delay_seconds > 0:
            await asyncio.sleep(inter_case_delay_seconds)
        result = await execute_real_case(
            case,
            base_url=base_url,
            owner_token=owner_token,
            provider=provider,
            timeout_seconds=timeout_seconds,
        )
        case_results.append(result)
        if result.get("execution_status") in {
            "EXECUTED",
            "HITL_BLOCKED",
            "NOT_DETERMINED",
        }:
            traces[case["id"]] = result
    executable_cases = [
        case
        for case in selected
        if case_execution_capability(case)["executable_by_current_agent_tools"]
    ]
    scoreboard = build_scoreboard(executable_cases, traces)
    coverage = Counter(result["execution_status"] for result in case_results)
    rate_limited = sum(
        result.get("execution_error") == "RATE_LIMITED"
        for result in case_results
    )
    return {
        "baseline_run_id": baseline_run_id,
        "execution_config": {
            "base_url": base_url,
            "provider": provider,
            "timeout_seconds": timeout_seconds,
            "tool_call_limit": MAX_TOOL_CALLS,
            "model_request_limit": MAX_MODEL_REQUESTS,
        },
        "coverage": {
            "total_selected": len(selected),
            "executed": coverage.get("EXECUTED", 0),
            "execution_failed": coverage.get("EXECUTION_FAILED", 0),
            "capability_gap": coverage.get("NOT_EXECUTABLE_CAPABILITY_GAP", 0),
            "hitl_blocked": coverage.get("HITL_BLOCKED", 0),
            "not_determined": coverage.get("NOT_DETERMINED", 0),
            "provider_rate_limited": rate_limited,
        },
        "cases": case_results,
        "deterministic_scoreboard": scoreboard,
        "semantic_grading": "NOT_RUN",
    }


def _code_build_identifier() -> str:
    digest = hashlib.sha256()
    for filename in ("evaluation_baseline.py", "agent_core.py", "server.py", "persistence.py"):
        path = Path(__file__).parent / filename
        digest.update(filename.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _metric_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "task_completion",
        "tool_selection",
        "forbidden_tool_violations",
        "citation_correctness",
        "retrieval_source_correctness",
        "hitl_boundary",
        "abstention",
        "instruction_adherence_deterministic",
    )
    summary: dict[str, Any] = {}
    for metric in metric_names:
        counts = Counter(
            result.get("metrics", {}).get(metric, "NOT_AVAILABLE")
            for result in results
        )
        summary[metric] = {
            "pass": counts.get("PASS", 0),
            "fail": counts.get("FAIL", 0),
            "not_determined": counts.get("NOT_DETERMINED", 0),
            "not_configured": counts.get("NOT_CONFIGURED", 0),
            "not_run": counts.get("NOT_RUN", 0),
            "case_count": len(results),
        }
    return summary


def _numeric_summary(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "average": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "average": round(sum(values) / len(values), 2),
    }


def _build_resume_analysis(
    *,
    dataset_cases: list[dict[str, Any]],
    final_cases: list[dict[str, Any]],
    scoreboard: dict[str, Any],
) -> dict[str, Any]:
    by_id = {case["id"]: case for case in dataset_cases}
    quality_results = scoreboard.get("results", [])
    quality_failures = [
        result["case_id"]
        for result in quality_results
        if result.get("deterministic_status") == "FAIL"
    ]
    provider_failures = [
        result["case_id"]
        for result in final_cases
        if result.get("execution_status")
        in {"EXECUTION_FAILED", "EXECUTION_FAILED_PROVIDER_RATE_LIMIT"}
    ]
    capability_gaps = [
        result["case_id"]
        for result in final_cases
        if result.get("execution_status") == "NOT_EXECUTABLE_CAPABILITY_GAP"
    ]
    failures_by_category = {
        "quality": {
            case_id: by_id[case_id]["category"]
            for case_id in quality_failures
        },
        "provider": {
            result["case_id"]: by_id[result["case_id"]]["category"]
            for result in final_cases
            if result["case_id"] in provider_failures
        },
        "capability_gap": {
            result["case_id"]: by_id[result["case_id"]]["category"]
            for result in final_cases
            if result["case_id"] in capability_gaps
        },
    }
    executable_results = [
        result
        for result in final_cases
        if result.get("execution_status") in {"EXECUTED", "HITL_BLOCKED"}
    ]
    latencies = [
        result["latency_ms"]
        for result in executable_results
        if isinstance(result.get("latency_ms"), (int, float))
    ]
    retries = [
        result["retries"]
        for result in final_cases
        if isinstance(result.get("retries"), int)
    ]
    tool_call_counts = [
        len(result.get("tool_calls", []))
        for result in executable_results
    ]
    tokens_available = sum(result.get("tokens") is not None for result in executable_results)
    costs_available = sum(result.get("cost") is not None for result in executable_results)
    weaknesses: list[dict[str, Any]] = []
    if provider_failures:
        weaknesses.append(
            {
                "area": "Provider",
                "reason": "Some cases remained unexecuted because the provider was rate limited.",
                "case_count": len(provider_failures),
            }
        )
    tool_selection_failures = sum(
        result.get("metrics", {}).get("tool_selection") == "FAIL"
        for result in quality_results
    )
    if tool_selection_failures:
        weaknesses.append(
            {
                "area": "Tool Selection",
                "reason": "The model did not select all required tools for some executed cases.",
                "case_count": tool_selection_failures,
            }
        )
    source_failures = sum(
        result.get("metrics", {}).get("retrieval_source_correctness") == "FAIL"
        for result in quality_results
    )
    if source_failures:
        weaknesses.append(
            {
                "area": "Grounding",
                "reason": "Executed answers did not match the expected source evidence.",
                "case_count": source_failures,
            }
        )
    weaknesses.extend(
        [
            {
                "area": "Retrieval",
                "reason": "Capability gaps prevent evaluation of workspace/project file access.",
                "case_count": len(capability_gaps),
            }
        ]
        if capability_gaps
        else []
    )
    return {
        "quality_failures": quality_failures,
        "provider_failures": provider_failures,
        "capability_gaps": capability_gaps,
        "failures_by_category": failures_by_category,
        "deterministic_metrics": _metric_summary(quality_results),
        "latency_ms": _numeric_summary(latencies),
        "retries": _numeric_summary(retries),
        "tool_call_count": _numeric_summary(tool_call_counts),
        "tokens": {
            "available_case_count": tokens_available,
            "status": "AVAILABLE" if tokens_available else "NOT_AVAILABLE",
        },
        "cost": {
            "available_case_count": costs_available,
            "status": "AVAILABLE" if costs_available else "NOT_AVAILABLE",
        },
        "strongest_observed_areas": weaknesses[:3],
        "semantic_grading": "NOT_RUN",
    }


def _get_owner_json(
    *,
    base_url: str,
    owner_token: str,
    path: str,
    timeout_seconds: float = 15.0,
) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        headers={"Authorization": f"Bearer {owner_token}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as error:
        return error.code, {}
    except Exception:
        return 0, {}


def _runtime_health_report(base_url: str, owner_token: str) -> dict[str, Any]:
    health_status, health = _get_owner_json(
        base_url=base_url,
        owner_token=owner_token,
        path="/health",
    )
    alert_status, alerts = _get_owner_json(
        base_url=base_url,
        owner_token=owner_token,
        path="/alerts/runtime",
    )
    return {
        "health": {
            "http_status": health_status,
            "ok": health.get("ok") is True,
        },
        "alerts": {
            "http_status": alert_status,
            "available": alert_status == 200,
            "delivery": alerts.get("delivery"),
            "audit": alerts.get("audit"),
            "states": alerts.get("alerts", []),
        },
    }


async def resume_rate_limited_baseline(
    baseline: dict[str, Any],
    dataset_cases: list[dict[str, Any]],
    *,
    owner_token: str,
    max_retries: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
    inter_case_delay_seconds: float,
) -> dict[str, Any]:
    original_cases = json.loads(json.dumps(baseline["cases"]))
    dataset_by_id = {case["id"]: case for case in dataset_cases}
    rate_limited_ids = [
        result["case_id"]
        for result in original_cases
        if result.get("execution_status") == "EXECUTION_FAILED"
        and result.get("execution_error") == "RATE_LIMITED"
    ]
    missing_dataset_ids = sorted(set(rate_limited_ids) - set(dataset_by_id))
    if missing_dataset_ids:
        raise ValueError(f"Baseline cases are missing from dataset: {missing_dataset_ids}")
    config = dict(baseline["execution_config"])
    provider = str(config["provider"])
    baseline_models = {
        result.get("model")
        for result in original_cases
        if result.get("model")
    }
    current_model = provider_model_name(provider)
    if baseline_models and current_model not in baseline_models:
        raise ValueError("Current provider model does not match the original baseline.")
    resume_run_id = str(uuid4())
    final_cases = list(original_cases)
    metadata: dict[str, Any] = {}
    resumed_case_ids: list[str] = []
    for index, original in enumerate(original_cases):
        case_id = original["case_id"]
        if case_id not in rate_limited_ids:
            metadata[case_id] = {
                "source": "original",
                "final_execution_status": original.get("execution_status"),
                "attempts": 0
                if original.get("execution_status") == "NOT_EXECUTABLE_CAPABILITY_GAP"
                else 1,
                "retries": 0,
                "original_execution_timestamp": original.get("start_timestamp"),
                "resumed_execution_timestamp": None,
            }
            continue
        if resumed_case_ids and inter_case_delay_seconds > 0:
            await asyncio.sleep(inter_case_delay_seconds)
        resumed = await execute_rate_limited_case_with_retry(
            dataset_by_id[case_id],
            base_url=str(config["base_url"]),
            owner_token=owner_token,
            provider=provider,
            timeout_seconds=float(config["timeout_seconds"]),
            max_retries=max_retries,
            base_delay_seconds=base_delay_seconds,
            max_delay_seconds=max_delay_seconds,
        )
        resumed["original_execution_status"] = original.get("execution_status")
        resumed["original_run_id"] = original.get("run_id")
        resumed["original_execution_timestamp"] = original.get("start_timestamp")
        resumed["resume_run_id"] = resume_run_id
        final_cases[index] = resumed
        resumed_case_ids.append(case_id)
        metadata[case_id] = {
            "source": "resume",
            "final_execution_status": resumed.get("execution_status"),
            "attempts": resumed.get("attempts"),
            "retries": resumed.get("retries"),
            "original_execution_timestamp": original.get("start_timestamp"),
            "resumed_execution_timestamp": resumed.get(
                "resumed_execution_timestamp"
            ),
            "original_run_id": original.get("run_id"),
            "resumed_run_id": resumed.get("run_id"),
            "retry_delays_seconds": resumed.get("retry_delays_seconds", []),
        }
    traces = {
        result["case_id"]: result
        for result in final_cases
        if result.get("execution_status") in {"EXECUTED", "HITL_BLOCKED"}
    }
    executable_cases = [
        case
        for case in dataset_cases
        if case_execution_capability(case)["executable_by_current_agent_tools"]
    ]
    scoreboard = build_scoreboard(executable_cases, traces)
    analysis = _build_resume_analysis(
        dataset_cases=dataset_cases,
        final_cases=final_cases,
        scoreboard=scoreboard,
    )
    coverage = Counter(result.get("execution_status") for result in final_cases)
    artifact = {
        "artifact_version": "baseline-real-v1-complete",
        "original_baseline_run_id": baseline["baseline_run_id"],
        "resume_run_id": resume_run_id,
        "dataset_version": baseline["dataset_version"],
        "dataset_sha256": baseline["dataset_sha256"],
        "dataset_case_count": len(dataset_cases),
        "execution_config": config,
        "model_config": {
            "provider": provider,
            "model": current_model,
            "tool_call_limit": config.get("tool_call_limit"),
            "model_request_limit": config.get("model_request_limit"),
        },
        "code_build_identifier": _code_build_identifier(),
        "provenance": {
            "mode": "resume_same_baseline_config",
            "original_artifact": "baseline-real-v1.json",
            "resumed_case_ids": resumed_case_ids,
            "original_case_results_preserved": True,
        },
        "resume_policy": {
            "max_retries_per_case": max_retries,
            "base_delay_seconds": base_delay_seconds,
            "max_delay_seconds": max_delay_seconds,
            "inter_case_delay_seconds": inter_case_delay_seconds,
            "retry_after_respected": True,
            "jitter": True,
        },
        "coverage": {
            "total_cases": len(final_cases),
            "executed": coverage.get("EXECUTED", 0),
            "hitl_blocked": coverage.get("HITL_BLOCKED", 0),
            "provider_failures": coverage.get("EXECUTION_FAILED", 0)
            + coverage.get("EXECUTION_FAILED_PROVIDER_RATE_LIMIT", 0),
            "capability_gaps": coverage.get("NOT_EXECUTABLE_CAPABILITY_GAP", 0),
        },
        "case_execution_metadata": metadata,
        "cases": final_cases,
        "deterministic_scoreboard": scoreboard,
        "analysis": analysis,
        "runtime_health": _runtime_health_report(
            str(config["base_url"]),
            owner_token,
        ),
        "semantic_grading": "NOT_RUN",
    }
    canonical = json.dumps(
        artifact,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    artifact["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
    return artifact
def _observed_sources(trace: dict[str, Any]) -> list[str]:
    sources: list[str] = []
    for source in trace.get("sources", []):
        if isinstance(source, str):
            sources.append(source)
        elif isinstance(source, dict):
            for key in ("domain", "url", "citation", "filename"):
                value = source.get(key)
                if isinstance(value, str):
                    sources.append(value)
                    break
    return sources


def validate_evaluation_trace(trace: dict[str, Any]) -> None:
    required_keys = {
        "case_id",
        "run_id",
        "output",
        "tool_calls",
        "citations",
        "sources",
        "execution_status",
    }
    if not isinstance(trace, dict) or not required_keys <= trace.keys():
        raise ValueError("Evaluation trace is missing required fields.")
    if not isinstance(trace["output"], dict) or not isinstance(
        trace["output"].get("answer"), str
    ):
        raise ValueError("Evaluation trace output must contain an answer string.")
    if not isinstance(trace["tool_calls"], list):
        raise ValueError("Evaluation trace tool_calls must be a list.")
    if not isinstance(trace["citations"], list) or not isinstance(
        trace["sources"], list
    ):
        raise ValueError("Evaluation trace citations and sources must be lists.")
    if trace["execution_status"] not in {
        "EXECUTED",
        "EXECUTION_FAILED",
        "HITL_BLOCKED",
        "NOT_DETERMINED",
        "NOT_EXECUTABLE_CAPABILITY_GAP",
        "INVALID_CASE",
    }:
        raise ValueError("Evaluation trace has an invalid execution status.")


def deterministic_grade(
    case: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    if trace.get("execution_status") == "NOT_DETERMINED":
        return {
            "case_id": case["id"],
            "category": case["category"],
            "provenance": case["provenance"],
            "deterministic_status": "NOT_DETERMINED",
            "check_results": {"evidence_preconditions": "NOT_AVAILABLE"},
            "metrics": {
                "task_completion": "NOT_DETERMINED",
                "tool_selection": "NOT_DETERMINED",
                "forbidden_tool_violations": "NOT_DETERMINED",
                "retrieval_source_correctness": "NOT_DETERMINED",
                "citation_correctness": "NOT_DETERMINED",
                "grounding": "NOT_RUN",
                "instruction_adherence": "NOT_CONFIGURED",
                "instruction_adherence_deterministic": "NOT_CONFIGURED",
                "abstention": "NOT_DETERMINED",
                "hitl_boundary": "NOT_DETERMINED",
            },
            "semantic_status": "NOT_DETERMINED",
            "latency_ms": trace.get("latency_ms"),
            "tool_calls": len(trace.get("tool_calls", [])),
            "retries": trace.get("retries"),
            "tokens": trace.get("tokens"),
            "cost": trace.get("cost"),
        }
    checks = case["deterministic_checks"]
    tools = _tool_names(trace)
    sources = _observed_sources(trace)
    available_evidence_citations = trace.get("available_evidence_citations", [])
    available_evidence_items = trace.get("available_evidence_items", [])
    evidence_citation_check: bool | str = True
    if case["id"] == "AA-RC-014" and available_evidence_items:
        observed_citations = [
            citation
            for citation in trace.get("citations", [])
            if isinstance(citation, str)
        ]
        evidence_citation_check = bool(
            observed_citations
            and all(
                citation in available_evidence_citations
                and verify_evidence_citation(
                    citation,
                    available_evidence_items,
                    expected_claim="Ahmed Agent project files",
                )
                for citation in observed_citations
            )
        )
    expected_sources = [str(value) for value in case["expected_sources"]]
    required_tools = {
        str(
            resolve_case_tool_expectation(case["id"], semantic_name).get(
                "actual"
            )
            or semantic_name
        )
        for semantic_name in case["required_tools"]
    }
    forbidden_tools = set(case["forbidden_tools"])
    schema_fields = set(checks["schema_fields"])
    output = trace.get("output")
    output_fields = set(output) if isinstance(output, dict) else set()

    abstention_check: bool | str
    if checks["abstention_required"]:
        abstention_check = (
            True
            if trace.get("abstained") is True
            else "NOT_AVAILABLE"
            if trace.get("abstained") is None
            else False
        )
    else:
        abstention_check = True
    check_results: dict[str, bool | str] = {
        "required_tools": required_tools <= tools,
        "forbidden_tools": not (forbidden_tools & tools),
        "citations": (
            bool(trace.get("citations") or sources)
            if checks["citations_required"]
            else True
        ),
        "evidence_citation_contract": evidence_citation_check,
        "expected_sources": (
            not expected_sources
            or any(
                expected.casefold() in observed.casefold()
                for expected in expected_sources
                for observed in sources
            )
        ),
        "schema": schema_fields <= output_fields,
        "approval_boundary": (
            (
                bool(trace.get("approval_requested"))
                and not bool(trace.get("executed_without_approval"))
            )
            if checks["approval_required"]
            else not bool(trace.get("executed_without_approval"))
        ),
        "abstention": abstention_check,
    }
    passed = all(value is True for value in check_results.values())
    metric_status = lambda value: "PASS" if value is True else (
        "NOT_DETERMINED" if value == "NOT_AVAILABLE" else "FAIL"
    )
    return {
        "case_id": case["id"],
        "category": case["category"],
        "provenance": case["provenance"],
        "deterministic_status": "PASS" if passed else "FAIL",
        "check_results": check_results,
        "metrics": {
            "task_completion": "PASS" if passed else "FAIL",
            "tool_selection": (
                "PASS"
                if check_results["required_tools"] and check_results["forbidden_tools"]
                else "FAIL"
            ),
            "forbidden_tool_violations": (
                "PASS" if check_results["forbidden_tools"] else "FAIL"
            ),
            "retrieval_source_correctness": (
                metric_status(check_results["expected_sources"])
            ),
            "citation_correctness": (
                "PASS"
            if check_results["citations"]
            and check_results["expected_sources"]
            and check_results["evidence_citation_contract"] is True
                else "NOT_DETERMINED"
            ),
            "grounding": "NOT_RUN",
            "instruction_adherence": "NOT_CONFIGURED",
            "instruction_adherence_deterministic": "NOT_CONFIGURED",
            "abstention": metric_status(check_results["abstention"]),
            "hitl_boundary": metric_status(check_results["approval_boundary"]),
        },
        "semantic_status": "NOT_RUN",
        "latency_ms": trace.get("latency_ms"),
        "tool_calls": len(trace.get("tool_calls", [])),
        "retries": trace.get("retries"),
        "tokens": trace.get("tokens"),
        "cost": trace.get("cost"),
    }


def build_scoreboard(
    cases: list[dict[str, Any]],
    traces: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    results = [
        (
            validate_evaluation_trace(traces[case["id"]]),
            deterministic_grade(case, traces[case["id"]]),
        )[1]
        for case in cases
        if case["id"] in traces
    ]
    missing = [case["id"] for case in cases if case["id"] not in traces]
    passed = sum(result["deterministic_status"] == "PASS" for result in results)
    category_counts = Counter(case["category"] for case in cases)
    provenance_counts = Counter(case["provenance"] for case in cases)
    return {
        "status": "PASS" if results and not missing and passed == len(results) else "INCOMPLETE",
        "execution_status": "COMPLETE" if not missing else "INCOMPLETE",
        "quality_status": (
            "PASS"
            if results and not missing and passed == len(results)
            else "NOT_DETERMINED"
            if any(result["deterministic_status"] == "NOT_DETERMINED" for result in results)
            else "FAIL"
            if results and not missing
            else "NOT_DETERMINED"
        ),
        "quality_eligible": provenance_counts.get("real_case", 0) > 0,
        "case_count": len(cases),
        "traced_case_count": len(results),
        "category_counts": dict(sorted(category_counts.items())),
        "provenance_counts": dict(sorted(provenance_counts.items())),
        "missing_case_ids": missing,
        "deterministic_pass_rate": round(passed / len(results), 4) if results else None,
        "semantic_grading": "NOT_RUN",
        "semantic_rubric": SEMANTIC_RUBRIC,
        "results": results,
    }


def build_quality_scoreboard(
    cases: list[dict[str, Any]],
    traces: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    real_cases = [case for case in cases if case["provenance"] == "real_case"]
    if not real_cases:
        return {
            "status": "REAL_CASE_DATA_REQUIRED",
            "quality_eligible": False,
            "real_case_count": 0,
            "semantic_grading": "NOT_RUN",
        }
    report = build_scoreboard(real_cases, traces)
    report["quality_eligible"] = True
    return report


def compare_scoreboards(
    baseline_a: dict[str, Any],
    baseline_b: dict[str, Any],
) -> dict[str, Any]:
    if not baseline_a.get("quality_eligible") or not baseline_b.get("quality_eligible"):
        return {
            "status": "REAL_CASE_DATA_REQUIRED",
            "per_case": [],
            "per_category": {},
            "semantic_grading": "NOT_RUN",
        }
    results_a = {result["case_id"]: result for result in baseline_a.get("results", [])}
    results_b = {result["case_id"]: result for result in baseline_b.get("results", [])}
    case_ids = sorted(set(results_a) | set(results_b))
    per_case: list[dict[str, Any]] = []
    for case_id in case_ids:
        before = results_a.get(case_id)
        after = results_b.get(case_id)
        before_score = (
            1 if before and before["deterministic_status"] == "PASS" else 0
        )
        after_score = 1 if after and after["deterministic_status"] == "PASS" else 0
        per_case.append(
            {
                "case_id": case_id,
                "category": (after or before).get("category"),
                "baseline_a": before_score,
                "baseline_b": after_score,
                "delta": after_score - before_score,
            }
        )
    categories: dict[str, dict[str, Any]] = {}
    for item in per_case:
        category = item["category"] or "uncategorized"
        bucket = categories.setdefault(category, {"baseline_a": [], "baseline_b": []})
        bucket["baseline_a"].append(item["baseline_a"])
        bucket["baseline_b"].append(item["baseline_b"])
    category_comparison = {
        category: {
            "baseline_a_pass_rate": round(sum(values["baseline_a"]) / len(values["baseline_a"]), 4),
            "baseline_b_pass_rate": round(sum(values["baseline_b"]) / len(values["baseline_b"]), 4),
            "delta": round(
                sum(values["baseline_b"]) / len(values["baseline_b"])
                - sum(values["baseline_a"]) / len(values["baseline_a"]),
                4,
            ),
        }
        for category, values in sorted(categories.items())
    }
    return {
        "status": "COMPARABLE" if per_case else "INCOMPLETE",
        "per_case": per_case,
        "per_category": category_comparison,
        "semantic_grading": "NOT_RUN",
    }


def freeze_baseline_manifest(
    path: Path = DATASET_PATH,
) -> dict[str, Any]:
    dataset_bytes = path.read_bytes()
    dataset_version, cases = load_case_document(path, require_baseline_size=True)
    provenance_counts = Counter(case["provenance"] for case in cases)
    return {
        "baseline_version": "evaluation-baseline-v1",
        "run_id": str(uuid4()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_version,
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "dataset_case_count": len(cases),
        "real_case_count": provenance_counts.get("real_case", 0),
        "contract_seed_count": provenance_counts.get("contract_seed", 0),
        "category_counts": dict(
            sorted(Counter(case["category"] for case in cases).items())
        ),
        "tool_name_mapping": TOOL_NAME_MAPPING,
        "case_capability_overrides": CASE_CAPABILITY_OVERRIDES,
        "dataset_provenance": (
            "real_case"
            if provenance_counts.get("real_case", 0)
            else "contract_seed_until_real_cases_are_added"
        ),
        "providers": {
            "gemini": {"model": provider_model_name("gemini")},
            "openai": {"model": provider_model_name("openai")},
        },
        "limits": {
            "max_tool_calls": MAX_TOOL_CALLS,
            "max_model_requests": MAX_MODEL_REQUESTS,
        },
        "semantic_grader": "not_configured",
        "deterministic_grading": "READY_FOR_TRACES",
        "live_provider_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ahmed Agent evaluation baseline")
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--validate-real", type=Path)
    parser.add_argument("--import-real", type=Path)
    parser.add_argument("--probe-real", type=Path)
    parser.add_argument("--run-real", type=Path)
    parser.add_argument("--resume-baseline", type=Path)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("real-cases-validated.json"),
    )
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("AHMED_EVAL_BASE_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument(
        "--provider",
        default="gemini",
        choices=("gemini", "openai"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--inter-case-delay-seconds", type=float, default=0.0)
    parser.add_argument("--resume-max-retries", type=int, default=1)
    parser.add_argument("--resume-base-delay-seconds", type=float, default=20.0)
    parser.add_argument("--resume-max-delay-seconds", type=float, default=60.0)
    parser.add_argument("--resume-inter-case-delay-seconds", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    live_commands = [args.probe_real, args.run_real, args.resume_baseline]
    if sum(value is not None for value in live_commands) > 1:
        parser.error(
            "use only one of --probe-real, --run-real, or --resume-baseline"
        )
    if args.resume_max_retries < 0:
        parser.error("--resume-max-retries must be non-negative")
    if args.resume_base_delay_seconds < 0 or args.resume_max_delay_seconds < 0:
        parser.error("resume delays must be non-negative")
    if args.validate_real:
        version, cases = load_case_document(args.validate_real)
        print(json.dumps({
            "status": "VALID",
            "dataset_version": version,
            "real_case_count": sum(case["provenance"] == "real_case" for case in cases),
        }, ensure_ascii=False, indent=2))
        return
    if args.import_real:
        if args.output is None:
            parser.error("--import-real requires --output")
        print(json.dumps(
            import_real_cases(args.import_real, args.output),
            ensure_ascii=False,
            indent=2,
        ))
        return
    live_path = args.probe_real or args.run_real
    if live_path:
        owner_token = os.environ.get("AHMED_OWNER_TOKEN")
        if not owner_token:
            parser.error("AHMED_OWNER_TOKEN is required for live evaluation.")
        version, cases = load_case_document(live_path, require_baseline_size=True)
        selected_ids = set(args.case_ids or [])
        if args.probe_real and not selected_ids:
            selected_ids = {
                case["id"]
                for case in cases
                if case_execution_capability(case)[
                    "executable_by_current_agent_tools"
                ]
            }
            selected_ids = set(sorted(selected_ids)[:3])
        result = asyncio.run(
            run_real_cases(
                cases,
                base_url=args.base_url,
                owner_token=owner_token,
                provider=args.provider,
                timeout_seconds=args.timeout_seconds,
                case_ids=selected_ids or None,
                inter_case_delay_seconds=args.inter_case_delay_seconds,
            )
        )
        result.update(
            {
                "dataset_version": version,
                "dataset_sha256": hashlib.sha256(live_path.read_bytes()).hexdigest(),
                "real_case_count": len(cases),
                "semantic_grading": "NOT_RUN",
                "live_provider_run": True,
            }
        )
        output_path = args.output or Path(
            "baseline-probe-v1.json" if args.probe_real else "baseline-real-v1.json"
        )
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": "WRITTEN",
                    "output": str(output_path),
                    "baseline_run_id": result["baseline_run_id"],
                    "coverage": result["coverage"],
                    "semantic_grading": "NOT_RUN",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.resume_baseline:
        owner_token = os.environ.get("AHMED_OWNER_TOKEN")
        if not owner_token:
            parser.error("AHMED_OWNER_TOKEN is required for baseline resume.")
        baseline = json.loads(
            args.resume_baseline.read_text(encoding="utf-8")
        )
        version, cases = load_case_document(
            args.dataset,
            require_baseline_size=True,
        )
        dataset_sha256 = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
        if dataset_sha256 != baseline.get("dataset_sha256"):
            parser.error("Dataset SHA-256 does not match the original baseline.")
        if version != baseline.get("dataset_version"):
            parser.error("Dataset version does not match the original baseline.")
        result = asyncio.run(
            resume_rate_limited_baseline(
                baseline,
                cases,
                owner_token=owner_token,
                max_retries=args.resume_max_retries,
                base_delay_seconds=args.resume_base_delay_seconds,
                max_delay_seconds=args.resume_max_delay_seconds,
                inter_case_delay_seconds=args.resume_inter_case_delay_seconds,
            )
        )
        output_path = args.output or Path("baseline-real-v1-complete.json")
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": "WRITTEN",
                    "output": str(output_path),
                    "original_baseline_run_id": result["original_baseline_run_id"],
                    "resume_run_id": result["resume_run_id"],
                    "coverage": result["coverage"],
                    "quality_failures": result["analysis"]["quality_failures"],
                    "provider_failures": result["analysis"]["provider_failures"],
                    "capability_gaps": result["analysis"]["capability_gaps"],
                    "semantic_grading": "NOT_RUN",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    cases = load_evaluation_cases()
    value = freeze_baseline_manifest() if args.manifest else {
        "status": "READY_FOR_TRACES",
        "case_count": len(cases),
        "real_case_count": sum(case["provenance"] == "real_case" for case in cases),
        "contract_seed_count": sum(
            case["provenance"] == "contract_seed" for case in cases
        ),
        "semantic_grading": "NOT_RUN",
    }
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()