from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import re
import time
import zipfile
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from docx import Document
from pypdf import PdfReader

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_TOP_K,
    MY_FILES_CANDIDATE_K,
    MY_FILES_EMBEDDING_MODEL,
    MY_FILES_EMBEDDING_VERSION,
    MY_FILES_RRF_K,
    MAX_QUERY_LENGTH,
    MAX_TOP_K,
)
from embeddings import EmbeddingError, get_embedding_provider, vector_literal
from persistence import (
    PersistenceError,
    create_original_source,
    delete_original_source,
    search_fts_document_chunks,
    search_vector_document_chunks,
    store_document,
    store_document_embeddings,
    update_original_source_extraction_status,
    update_document_embedding_status,
)


logger = logging.getLogger("ahmed_agent.my_files")
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
MAX_DOCX_ZIP_ENTRIES = 1000
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024


class FileProcessingError(RuntimeError):
    def __init__(self, message: str, *, status: str = "failed") -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int | None
    content: str


@dataclass(frozen=True)
class DocumentChunk:
    chunk_index: int
    page_number: int | None
    content: str
    metadata: dict[str, Any]


def extension_for(filename: str) -> str:
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    dot_index = name.rfind(".")
    return name[dot_index:].lower() if dot_index > 0 else ""


def sanitize_filename(value: object) -> str:
    """Return a safe display/storage name without treating it as a filesystem path."""
    if not isinstance(value, str):
        return ""
    normalized = value.replace("\\", "/")
    filename = normalized.rsplit("/", 1)[-1]
    if (
        not filename
        or filename in {".", ".."}
        or "\x00" in filename
        or len(filename) > 255
    ):
        return ""
    return filename


def _normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\x00", "").splitlines()).strip()


def _extract_text_or_markdown(data: bytes) -> list[ExtractedPage]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise FileProcessingError(
            "The text file must use UTF-8 encoding.",
            status="invalid_file_content",
        ) from error
    return [ExtractedPage(page_number=None, content=_normalize_text(text))]


def _validate_text_payload(data: bytes) -> None:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise FileProcessingError(
            "The text file must use UTF-8 encoding.",
            status="invalid_file_content",
        ) from error
    if any(
        ord(character) < 32 and character not in "\n\r\t"
        for character in text
    ):
        raise FileProcessingError(
            "The text file contains binary control data.",
            status="invalid_file_content",
        )


def _validate_docx_archive(data: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_DOCX_ZIP_ENTRIES:
                raise FileProcessingError(
                    "The DOCX archive contains too many entries.",
                    status="invalid_file_content",
                )

            names: set[str] = set()
            uncompressed_bytes = 0
            for entry in entries:
                name = entry.filename.replace("\\", "/")
                path_parts = name.split("/")
                if (
                    not name
                    or name.startswith("/")
                    or any(part == ".." for part in path_parts)
                    or name in names
                ):
                    raise FileProcessingError(
                        "The DOCX archive contains an unsafe entry.",
                        status="invalid_file_content",
                    )
                names.add(name)
                uncompressed_bytes += entry.file_size
                if uncompressed_bytes > MAX_DOCX_UNCOMPRESSED_BYTES:
                    raise FileProcessingError(
                        "The DOCX archive is too large when unpacked.",
                        status="invalid_file_content",
                    )

            required_entries = {"[Content_Types].xml", "word/document.xml"}
            if not required_entries.issubset(names):
                raise FileProcessingError(
                    "The DOCX archive is missing required parts.",
                    status="invalid_file_content",
                )
    except (zipfile.BadZipFile, OSError, ValueError) as error:
        raise FileProcessingError(
            "The DOCX file is not a valid ZIP archive.",
            status="invalid_file_content",
        ) from error


def validate_file_content(filename: str, data: bytes) -> None:
    if sanitize_filename(filename) != filename:
        raise FileProcessingError(
            "The filename is invalid.",
            status="invalid_filename",
        )
    extension = extension_for(filename)
    if extension in {".txt", ".md"}:
        _validate_text_payload(data)
        return
    if extension == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise FileProcessingError(
                "The file content is not a PDF.",
                status="invalid_file_content",
            )
        return
    if extension == ".docx":
        _validate_docx_archive(data)
        return
    raise FileProcessingError(
        "This file type is not supported.",
        status="unsupported_extension",
    )


def _extract_pdf(data: bytes) -> list[ExtractedPage]:
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as error:
        raise FileProcessingError("The PDF could not be read.", status="invalid_pdf") from error
    if reader.is_encrypted:
        raise FileProcessingError("Encrypted PDFs are not supported.", status="unsupported_encryption")

    pages: list[ExtractedPage] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = _normalize_text(page.extract_text() or "")
        except Exception as error:
            raise FileProcessingError("The PDF text layer could not be read.", status="invalid_pdf") from error
        if text:
            pages.append(ExtractedPage(page_number=page_number, content=text))
    if not pages:
        raise FileProcessingError(
            "This PDF has no text layer; OCR is required.",
            status="unsupported_ocr_required",
        )
    return pages


def _extract_docx(data: bytes) -> list[ExtractedPage]:
    try:
        document = Document(io.BytesIO(data))
    except Exception as error:
        raise FileProcessingError("The DOCX file could not be read.", status="invalid_docx") from error
    text = _normalize_text("\n".join(paragraph.text for paragraph in document.paragraphs))
    return [ExtractedPage(page_number=None, content=text)]


def extract_document(filename: str, mime_type: str | None, data: bytes) -> list[ExtractedPage]:
    del mime_type
    validate_file_content(filename, data)
    extension = extension_for(filename)
    if extension in {".txt", ".md"}:
        return _extract_text_or_markdown(data)
    if extension == ".pdf":
        return _extract_pdf(data)
    if extension == ".docx":
        return _extract_docx(data)
    raise FileProcessingError("This file type is not supported.", status="unsupported_extension")


def _chunk_content(content: str) -> list[str]:
    if not content:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(content):
        end = min(len(content), start + CHUNK_SIZE)
        if end < len(content):
            boundary = content.rfind(" ", start, end)
            if boundary > start + (CHUNK_SIZE // 2):
                end = boundary
        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(content):
            break
        next_start = max(end - CHUNK_OVERLAP, start + 1)
        start = next_start
    return chunks


def build_chunks(
    *,
    document_id: str,
    filename: str,
    mime_type: str | None,
    file_hash: str,
    source_type: str,
    pages: list[ExtractedPage],
    source_id: str | None = None,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for page in pages:
        for content in _chunk_content(page.content):
            chunks.append(
                DocumentChunk(
                    chunk_index=len(chunks),
                    page_number=page.page_number,
                    content=content,
                    metadata={
                        "document_id": document_id,
                        "filename": filename,
                        "mime_type": mime_type,
                        "page_number": page.page_number,
                        "source_type": source_type,
                        "file_hash": file_hash,
                        "source_id": source_id,
                    },
                )
            )
    return chunks


async def ingest_document(
    *,
    document_id: str,
    filename: str,
    mime_type: str | None,
    data: bytes,
    owner_principal_id: str = "owner",
) -> dict[str, Any]:
    filename = sanitize_filename(filename)
    if not filename:
        return {
            "duplicate": False,
            "document_id": document_id,
            "filename": None,
            "status": "invalid_filename",
            "original_available": False,
        }
    file_hash = hashlib.sha256(data).hexdigest()
    try:
        validate_file_content(filename, data)
    except FileProcessingError as error:
        return {
            "duplicate": False,
            "document_id": document_id,
            "filename": filename,
            "status": error.status,
            "chunk_count": 0,
            "file_hash": file_hash,
            "original_available": False,
        }

    source_id = str(uuid4())
    await create_original_source(
        source_id=source_id,
        owner_principal_id=owner_principal_id,
        original_filename=filename,
        mime_type=mime_type,
        data=data,
        authorization_scope="MY_FILES_OWNER",
    )

    try:
        pages = extract_document(filename, mime_type, data)
        chunks = build_chunks(
            document_id=document_id,
            filename=filename,
            mime_type=mime_type,
            file_hash=file_hash,
            source_type="upload",
            pages=pages,
            source_id=source_id,
        )
        if not chunks:
            raise FileProcessingError(
                "The file contains no searchable text.",
                status="empty",
            )
    except FileProcessingError as error:
        try:
            await store_document(
                document_id=document_id,
                filename=filename,
                mime_type=mime_type,
                file_hash=file_hash,
                status=error.status,
                chunks=[],
                source_id=source_id,
                source_sha256=file_hash,
            )
            await update_original_source_extraction_status(
                source_id=source_id,
                status=error.status,
            )
        except PersistenceError:
            await delete_original_source(source_id=source_id)
            raise
        return {
            "duplicate": False,
            "document_id": document_id,
            "filename": filename,
            "status": error.status,
            "chunk_count": 0,
            "file_hash": file_hash,
            "source_id": source_id,
            "original_available": True,
        }

    try:
        await store_document(
            document_id=document_id,
            filename=filename,
            mime_type=mime_type,
            file_hash=file_hash,
            status="fts_ready",
            chunks=chunks,
            source_id=source_id,
            source_sha256=file_hash,
        )
        await update_original_source_extraction_status(
            source_id=source_id,
            status="fts_ready",
        )
    except PersistenceError:
        await delete_original_source(source_id=source_id)
        raise

    embedding_status = "embedding_failed"
    embedding_metrics: dict[str, int | float | str] = {}
    try:
        provider = get_embedding_provider()
        embeddings = await asyncio.to_thread(
            provider.embed_documents,
            [chunk.content for chunk in chunks],
        )
        if len(embeddings) != len(chunks) or provider.dimension <= 0:
            raise EmbeddingError("The embedding count did not match the chunks.")
        embedding_rows = [
            (chunk.chunk_index, vector_literal(vector))
            for chunk, vector in zip(chunks, embeddings, strict=True)
        ]
        await store_document_embeddings(
            document_id=document_id,
            embeddings=embedding_rows,
            embedding_model=provider.model_name,
            embedding_dimension=provider.dimension,
            embedding_version=MY_FILES_EMBEDDING_VERSION,
        )
        embedding_status = "ready"
        embedding_metrics = provider.last_metrics
        await update_original_source_extraction_status(
            source_id=source_id,
            status="ready",
        )
    except EmbeddingError as error:
        await update_document_embedding_status(
            document_id=document_id,
            status="embedding_failed",
            embedding_model=MY_FILES_EMBEDDING_MODEL,
            embedding_dimension=None,
            embedding_version=MY_FILES_EMBEDDING_VERSION,
        )
        await update_original_source_extraction_status(
            source_id=source_id,
            status="embedding_failed",
        )
        embedding_metrics = {"embedding_status": error.status}
    except PersistenceError:
        raise
    except Exception:
        await update_document_embedding_status(
            document_id=document_id,
            status="embedding_failed",
            embedding_model=MY_FILES_EMBEDDING_MODEL,
            embedding_dimension=None,
            embedding_version=MY_FILES_EMBEDDING_VERSION,
        )
        await update_original_source_extraction_status(
            source_id=source_id,
            status="embedding_failed",
        )
        embedding_metrics = {"embedding_status": "ERROR"}
    logger.info(
        json.dumps(
            {
                "document_id": document_id,
                "filename": filename,
                "mime_type": mime_type,
                "extraction_status": "ready",
                "embedding_status": embedding_status,
                "chunk_count": len(chunks),
                **embedding_metrics,
            },
            separators=(",", ":"),
        )
    )
    return {
        "duplicate": False,
        "document_id": document_id,
        "filename": filename,
        "status": embedding_status,
        "chunk_count": len(chunks),
        "file_hash": file_hash,
        "source_id": source_id,
        "original_available": True,
        "embedding": embedding_metrics,
    }


_RERANK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "how",
    "is",
    "of",
    "on",
    "the",
    "this",
    "to",
    "what",
    "which",
    "with",
    "عن",
    "في",
    "ما",
    "هو",
    "هي",
    "من",
    "هذا",
    "هذه",
    "اسم",
    "الاسم",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
        if token not in _RERANK_STOPWORDS
    }


def _rerank_score(query_terms: set[str], row: dict[str, Any], rrf_score: float) -> float:
    content_terms = _tokens(row["content"])
    filename_terms = _tokens(row["filename"])
    exact_term_match = len(query_terms & content_terms)
    filename_relevance = len(query_terms & filename_terms)
    heading_terms = _tokens(
        "\n".join(
            line for line in row["content"].splitlines() if line.lstrip().startswith("#")
        )
    )
    heading_match = len(query_terms & heading_terms)
    same_line_proximity = 0
    for line in row["content"].splitlines():
        if query_terms.issubset(_tokens(line)):
            same_line_proximity = 1
            break
    return (
        rrf_score
        + 0.01 * exact_term_match
        + 0.02 * filename_relevance
        + 0.02 * heading_match
        + 0.01 * same_line_proximity
    )


def _hybrid_results(
    *,
    query: str,
    fts_rows: list[dict[str, Any]],
    vector_rows: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    fused: dict[tuple[str, int], dict[str, Any]] = {}
    for source_rows in (fts_rows, vector_rows):
        for rank, row in enumerate(source_rows, start=1):
            key = (str(row["document_id"]), int(row["chunk_index"]))
            item = fused.setdefault(key, dict(row))
            item["rrf_score"] = item.get("rrf_score", 0.0) + 1.0 / (MY_FILES_RRF_K + rank)
    query_terms = _tokens(query)
    for item in fused.values():
        item["rerank_score"] = _rerank_score(query_terms, item, item["rrf_score"])
    ranked = sorted(
        fused.values(),
        key=lambda row: (
            -row["rerank_score"],
            str(row["document_id"]),
            int(row["chunk_index"]),
        ),
    )
    return ranked[:top_k]


async def search_my_files(query: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty")
    if len(normalized_query) > MAX_QUERY_LENGTH:
        raise ValueError("query is too long")
    if not 1 <= top_k <= min(MAX_TOP_K, 20):
        raise ValueError("top_k is out of range")

    started_at = time.perf_counter()
    candidate_k = max(top_k, min(MY_FILES_CANDIDATE_K, max(top_k * 4, 10)))
    fts_rows = await search_fts_document_chunks(normalized_query, candidate_k)
    vector_rows: list[dict[str, Any]] = []
    embedding_status = "NOT_ATTEMPTED"
    provider = get_embedding_provider()
    try:
        query_embedding = await asyncio.to_thread(
            provider.embed_query,
            normalized_query,
        )
        vector_rows = await search_vector_document_chunks(
            vector=vector_literal(query_embedding),
            top_k=candidate_k,
            embedding_model=provider.model_name,
            embedding_dimension=provider.dimension,
            embedding_version=MY_FILES_EMBEDDING_VERSION,
        )
        embedding_status = provider.status
    except EmbeddingError as error:
        embedding_status = error.status
    except PersistenceError:
        raise
    except Exception:
        embedding_status = "ERROR"
    rows = _hybrid_results(
        query=normalized_query,
        fts_rows=fts_rows,
        vector_rows=vector_rows,
        top_k=top_k,
    )
    retrieval_mode = "HYBRID_RRF" if vector_rows else "FTS_FALLBACK"
    results: list[dict[str, Any]] = []
    for row in rows:
        page_number = row["page_number"]
        citation = f"[source: {row['filename']}"
        if page_number is not None:
            citation += f", page {page_number}"
        citation += f", chunk {row['chunk_index']}]"
        results.append(
            {
                "content": row["content"],
                "document_id": str(row["document_id"]),
                "filename": row["filename"],
                "chunk_index": row["chunk_index"],
                "page_number": page_number,
                "mime_type": row["mime_type"],
                "source_type": row["source_type"],
                "file_hash": row["file_hash"],
                "source_id": (
                    str(row["source_id"]) if row.get("source_id") else None
                ),
                "source_sha256": row.get("source_sha256"),
                "original_available": bool(row.get("original_available")),
                "citation": citation,
            }
        )
    logger.info(
        json.dumps(
            {
                "retrieval_latency_ms": int((time.perf_counter() - started_at) * 1000),
                "result_count": len(results),
                "retrieval_mode": retrieval_mode,
                "fts_candidate_count": len(fts_rows),
                "vector_candidate_count": len(vector_rows),
                "embedding_status": embedding_status,
            },
            separators=(",", ":"),
        )
    )
    return {
        "ok": True,
        "scope": "MY_FILES",
        "privacy_mode": "PRIVATE_STANDARD",
        "retrieval_mode": retrieval_mode,
        "results": results,
        "message": (
            "No matching information was found in the uploaded files."
            if not results
            else "Use the returned file excerpts and citations only."
        ),
    }