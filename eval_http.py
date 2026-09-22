from __future__ import annotations

import asyncio
import hashlib
import json
import re
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse
from uuid import UUID, uuid4

from persistence import delete_documents_for_source_ids, delete_original_source
from server import _UPLOAD_DEGRADED_STATUSES, _UPLOAD_OK_STATUS


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
        supported_status in {201, 202}
        and len(supported_results) == 4
        and all(
            isinstance(item, dict)
            and item.get("status") in ({_UPLOAD_OK_STATUS} | _UPLOAD_DEGRADED_STATUSES)
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


