from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from persistence_core import PersistenceError, _get_pool


async def find_document_by_hash(file_hash: str) -> dict[str, Any] | None:
    pool = await _get_pool()
    try:
        row = await pool.fetchrow(
            """
            SELECT d.document_id, d.filename, d.status, COUNT(c.chunk_id)::int AS chunk_count
            FROM documents d
            LEFT JOIN document_chunks c ON c.document_id = d.document_id
            WHERE d.file_hash = $1
            GROUP BY d.document_id, d.filename, d.status
            LIMIT 1
            """,
            file_hash,
        )
    except Exception as error:
        raise PersistenceError("Could not check for a duplicate document.") from error
    return dict(row) if row is not None else None


async def create_original_source(
    *,
    source_id: str,
    owner_principal_id: str,
    original_filename: str,
    mime_type: str | None,
    data: bytes,
    authorization_scope: str,
) -> dict[str, Any]:
    if not owner_principal_id:
        raise ValueError("owner principal is required")
    if not data:
        raise ValueError("original source bytes must not be empty")
    file_hash = hashlib.sha256(data).hexdigest()
    storage_object_key = f"original-source/{file_hash}"
    pool = await _get_pool()
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO original_source_blobs (
                        storage_object_key, sha256, byte_size, content
                    )
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (storage_object_key) DO NOTHING
                    """,
                    storage_object_key,
                    file_hash,
                    len(data),
                    data,
                )
                await connection.execute(
                    """
                    INSERT INTO original_sources (
                        source_id,
                        owner_principal_id,
                        source_version,
                        original_filename,
                        mime_type,
                        byte_size,
                        sha256,
                        storage_backend,
                        storage_object_key,
                        original_available,
                        authorization_scope,
                        extraction_status
                    )
                    VALUES (
                        $1::uuid, $2, 1, $3, $4, $5, $6, 'postgres_bytea',
                        $7, TRUE, $8, 'pending'
                    )
                    """,
                    source_id,
                    owner_principal_id,
                    original_filename,
                    mime_type,
                    len(data),
                    file_hash,
                    storage_object_key,
                    authorization_scope,
                )
    except Exception as error:
        raise PersistenceError("Could not store the original source.") from error
    return {
        "source_id": source_id,
        "source_version": 1,
        "owner_principal_id": owner_principal_id,
        "original_filename": original_filename,
        "mime_type": mime_type,
        "byte_size": len(data),
        "sha256": file_hash,
        "storage_backend": "postgres_bytea",
        "original_available": True,
        "authorization_scope": authorization_scope,
        "extraction_status": "pending",
    }


async def read_authorized_original_source(
    *,
    source_id: str,
    owner_principal_id: str | None,
) -> dict[str, Any]:
    pool = await _get_pool()
    try:
        row = await pool.fetchrow(
            """
            SELECT
                source_id,
                owner_principal_id,
                source_version,
                original_filename,
                mime_type,
                byte_size,
                sha256,
                storage_backend,
                original_available,
                authorization_scope,
                extraction_status,
                storage_object_key
            FROM original_sources
            WHERE source_id = $1::uuid
            """,
            source_id,
        )
    except Exception as error:
        raise PersistenceError("Could not load the original source.") from error
    if row is None:
        return {"status": "NOT_FOUND"}
    if not owner_principal_id or row["owner_principal_id"] != owner_principal_id:
        return {"status": "DENIED"}
    source = {
        key: row[key]
        for key in (
            "source_id",
            "owner_principal_id",
            "source_version",
            "original_filename",
            "mime_type",
            "byte_size",
            "sha256",
            "storage_backend",
            "original_available",
            "authorization_scope",
            "extraction_status",
        )
    }
    if not row["original_available"]:
        return {"status": "NOT_FOUND", "source": source}
    try:
        blob = await pool.fetchrow(
            """
            SELECT sha256, byte_size, content
            FROM original_source_blobs
            WHERE storage_object_key = $1
            """,
            row["storage_object_key"],
        )
    except Exception as error:
        raise PersistenceError("Could not load the original source blob.") from error
    if blob is None:
        return {"status": "AUTHORIZED", "source": source, "data": None}
    return {
        "status": "AUTHORIZED",
        "source": source,
        "data": bytes(blob["content"]),
        "blob_sha256": blob["sha256"],
        "blob_byte_size": int(blob["byte_size"]),
    }


async def update_original_source_extraction_status(
    *,
    source_id: str,
    status: str,
) -> None:
    if not status or len(status) > 64:
        raise ValueError("invalid original source extraction status")
    pool = await _get_pool()
    try:
        await pool.execute(
            """
            UPDATE original_sources
            SET extraction_status = $2
            WHERE source_id = $1::uuid
            """,
            source_id,
            status,
        )
    except Exception as error:
        raise PersistenceError(
            "Could not update original source extraction status."
        ) from error


async def delete_original_source(*, source_id: str) -> None:
    pool = await _get_pool()
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """
                    DELETE FROM original_sources
                    WHERE source_id = $1::uuid
                    RETURNING storage_object_key
                    """,
                    source_id,
                )
                if row is not None:
                    await connection.execute(
                        """
                        DELETE FROM original_source_blobs blob
                        WHERE blob.storage_object_key = $1
                          AND NOT EXISTS (
                              SELECT 1
                              FROM original_sources source
                              WHERE source.storage_object_key = blob.storage_object_key
                          )
                        """,
                        row["storage_object_key"],
                    )
    except Exception as error:
        raise PersistenceError("Could not remove the original source.") from error


async def delete_documents_for_source_ids(source_ids: Sequence[str]) -> int:
    """Remove only documents/chunks linked to the explicitly supplied sources."""

    normalized_ids: list[str] = []
    for source_id in source_ids:
        try:
            normalized_ids.append(str(UUID(str(source_id))))
        except (TypeError, ValueError) as error:
            raise ValueError("source_ids must contain UUIDs") from error
    if not normalized_ids:
        return 0
    pool = await _get_pool()
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    DELETE FROM document_chunks
                    WHERE source_id = ANY($1::uuid[])
                    """,
                    normalized_ids,
                )
                result = await connection.execute(
                    """
                    DELETE FROM documents
                    WHERE source_id = ANY($1::uuid[])
                    """,
                    normalized_ids,
                )
        return int(result.rsplit(" ", 1)[-1])
    except Exception as error:
        raise PersistenceError(
            "Could not remove documents for the supplied source IDs."
        ) from error


async def cleanup_evaluation_run(*, run_id: str, session_id: str) -> dict[str, int]:
    """Remove one evaluation run while preserving its audit history."""

    pool = await _get_pool()
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                deleted_messages = await connection.fetchval(
                    """
                    DELETE FROM agent_messages
                    WHERE run_id = $1::uuid
                    RETURNING 1
                    """,
                    run_id,
                )
                deleted_checkpoints = await connection.fetchval(
                    """
                    DELETE FROM agent_run_checkpoints
                    WHERE run_id = $1::uuid
                    RETURNING 1
                    """,
                    run_id,
                )
                deleted_tool_events = await connection.fetchval(
                    """
                    DELETE FROM tool_events
                    WHERE run_id = $1::uuid
                    RETURNING 1
                    """,
                    run_id,
                )
                deleted_pending_actions = await connection.fetchval(
                    """
                    DELETE FROM pending_actions
                    WHERE run_id = $1::uuid
                    RETURNING 1
                    """,
                    run_id,
                )
                deleted_runs = await connection.fetchval(
                    """
                    DELETE FROM agent_runs
                    WHERE run_id = $1::uuid
                    RETURNING 1
                    """,
                    run_id,
                )
                deleted_sessions = await connection.fetchval(
                    """
                    DELETE FROM agent_sessions
                    WHERE session_id = $1::uuid
                      AND NOT EXISTS (
                          SELECT 1 FROM agent_runs
                          WHERE session_id = $1::uuid
                      )
                    RETURNING 1
                    """,
                    session_id,
                )
        return {
            "messages": int(deleted_messages or 0),
            "checkpoints": int(deleted_checkpoints or 0),
            "tool_events": int(deleted_tool_events or 0),
            "pending_actions": int(deleted_pending_actions or 0),
            "runs": int(deleted_runs or 0),
            "sessions": int(deleted_sessions or 0),
            "audit_events_preserved": 1,
        }
    except Exception as error:
        raise PersistenceError("Could not clean up the evaluation run.") from error


async def store_document(
    *,
    document_id: str,
    filename: str,
    mime_type: str | None,
    file_hash: str,
    status: str,
    chunks: Sequence[Any],
    source_id: str | None = None,
    source_sha256: str | None = None,
) -> bool:
    """Store a document and its chunks.

    Returns True when the row was inserted. Returns False when a row with the
    same file_hash already existed (unique index documents_file_hash_key);
    in that case nothing is written and the caller must treat the upload as a
    duplicate of the existing document.
    """
    pool = await _get_pool()
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                # ON CONFLICT needs the unique arbiter index; on legacy
                # databases where it could not be created (pre-existing
                # duplicate file_hash rows) fall back to a plain INSERT and
                # let the application-level hash check remain the guard.
                has_hash_index = await connection.fetchval(
                    "SELECT to_regclass('documents_file_hash_key') IS NOT NULL"
                )
                conflict_clause = (
                    "ON CONFLICT (file_hash) DO NOTHING" if has_hash_index else ""
                )
                inserted_id = await connection.fetchval(
                    f"""
                    INSERT INTO documents (
                        document_id,
                        filename,
                        mime_type,
                        source_type,
                        file_hash,
                        status,
                        source_id,
                        source_sha256,
                        original_available
                    )
                    VALUES (
                        $1::uuid, $2, $3, 'upload', $4, $5, $6::uuid, $7,
                        $8
                    )
                    {conflict_clause}
                    RETURNING document_id
                    """,
                    document_id,
                    filename,
                    mime_type,
                    file_hash,
                    status,
                    source_id,
                    source_sha256,
                    source_id is not None,
                )
                if inserted_id is None:
                    return False
                if chunks:
                    await connection.executemany(
                        """
                        INSERT INTO document_chunks (
                            document_id,
                            chunk_index,
                            page_number,
                            content,
                            metadata,
                            source_id,
                            source_sha256
                        )
                        VALUES ($1::uuid, $2, $3, $4, $5::jsonb, $6::uuid, $7)
                        """,
                        [
                            (
                                document_id,
                                chunk.chunk_index,
                                chunk.page_number,
                                chunk.content,
                                json.dumps(chunk.metadata, separators=(",", ":")),
                                source_id,
                                source_sha256,
                            )
                            for chunk in chunks
                        ],
                    )
                return True
    except Exception as error:
        raise PersistenceError("Could not store the uploaded document.") from error


async def update_document_embedding_status(
    *,
    document_id: str,
    status: str,
    embedding_model: str | None,
    embedding_dimension: int | None,
    embedding_version: str | None,
) -> None:
    pool = await _get_pool()
    try:
        await pool.execute(
            """
            UPDATE documents
            SET status = $2,
                embedding_model = $3,
                embedding_dimension = $4,
                embedding_version = $5,
                updated_at = NOW()
            WHERE document_id = $1::uuid
            """,
            document_id,
            status,
            embedding_model,
            embedding_dimension,
            embedding_version,
        )
    except Exception as error:
        raise PersistenceError("Could not update document embedding status.") from error


async def store_document_embeddings(
    *,
    document_id: str,
    embeddings: Sequence[tuple[int, str]],
    embedding_model: str,
    embedding_dimension: int,
    embedding_version: str,
) -> None:
    pool = await _get_pool()
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.executemany(
                    """
                    UPDATE document_chunks
                    SET embedding = $2::vector
                    WHERE document_id = $1::uuid AND chunk_index = $3
                    """,
                    [
                        (document_id, vector, chunk_index)
                        for chunk_index, vector in embeddings
                    ],
                )
                await connection.execute(
                    """
                    UPDATE documents
                    SET status = 'ready',
                        embedding_model = $2,
                        embedding_dimension = $3,
                        embedding_version = $4,
                        updated_at = NOW()
                    WHERE document_id = $1::uuid
                    """,
                    document_id,
                    embedding_model,
                    embedding_dimension,
                    embedding_version,
                )
    except Exception as error:
        raise PersistenceError("Could not store document embeddings.") from error


async def search_fts_document_chunks(query: str, top_k: int) -> list[dict[str, Any]]:
    pool = await _get_pool()
    try:
        rows = await pool.fetch(
            """
            SELECT
                c.document_id,
                d.filename,
                d.mime_type,
                d.source_type,
                d.file_hash,
                c.source_id,
                c.source_sha256,
                d.original_available,
                c.chunk_index,
                c.page_number,
                c.content,
                ts_rank_cd(
                    to_tsvector('simple', c.content),
                    plainto_tsquery('simple', $1)
                ) AS rank
            FROM document_chunks c
            JOIN documents d ON d.document_id = c.document_id
            WHERE d.status IN ('ready', 'fts_ready', 'embedding_failed')
              AND to_tsvector('simple', c.content)
                  @@ plainto_tsquery('simple', $1)
            ORDER BY rank DESC, c.document_id, c.chunk_index
            LIMIT $2
            """,
            query,
            top_k,
        )
    except Exception as error:
        raise PersistenceError("Could not search uploaded documents.") from error
    return [dict(row) for row in rows]


async def search_vector_document_chunks(
    *,
    vector: str,
    top_k: int,
    embedding_model: str,
    embedding_dimension: int,
    embedding_version: str,
) -> list[dict[str, Any]]:
    pool = await _get_pool()
    try:
        rows = await pool.fetch(
            """
            SELECT
                c.document_id,
                d.filename,
                d.mime_type,
                d.source_type,
                d.file_hash,
                c.source_id,
                c.source_sha256,
                d.original_available,
                c.chunk_index,
                c.page_number,
                c.content,
                c.embedding <=> $1::vector AS distance
            FROM document_chunks c
            JOIN documents d ON d.document_id = c.document_id
            WHERE d.status = 'ready'
              AND d.embedding_model = $2
              AND d.embedding_dimension = $3
              AND d.embedding_version = $4
              AND c.embedding IS NOT NULL
            ORDER BY c.embedding <=> $1::vector ASC, c.document_id, c.chunk_index
            LIMIT $5
            """,
            vector,
            embedding_model,
            embedding_dimension,
            embedding_version,
            top_k,
        )
    except Exception as error:
        raise PersistenceError("Could not search document vectors.") from error
    return [dict(row) for row in rows]


async def search_document_chunks(query: str, top_k: int) -> list[dict[str, Any]]:
    return await search_fts_document_chunks(query, top_k)


