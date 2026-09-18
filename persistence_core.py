from __future__ import annotations

import asyncio
import hashlib
import logging
import json
import os
import socket
from collections.abc import Sequence
from typing import Any
from uuid import UUID

import asyncpg

logger = logging.getLogger(__name__)


class PersistenceError(RuntimeError):
    pass


_pool: asyncpg.Pool | None = None
_schema_ready = False
_schema_lock = asyncio.Lock()
TEST_RETENTION_SCOPES = ("fault_injection", "probe")
RETENTION_POLICY = {
    "completed_checkpoint_days": 30,
    "failed_orphaned_checkpoint_days": 90,
    "audit_events": "preserved",
    "test_scopes": TEST_RETENTION_SCOPES,
}


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


async def _ensure_policy_schema(pool: asyncpg.Pool) -> None:
    global _schema_ready
    if _schema_ready:
        return
    async with _schema_lock:
        if _schema_ready:
            return
        try:
            await pool.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_actions (
                    action_id UUID PRIMARY KEY,
                    session_id UUID NOT NULL,
                    run_id UUID NOT NULL,
                    user_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    arguments_json JSONB NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '15 minutes'),
                    approved_at TIMESTAMPTZ,
                    rejected_at TIMESTAMPTZ,
                    executed_at TIMESTAMPTZ
                );

                CREATE INDEX IF NOT EXISTS pending_actions_user_status_idx
                    ON pending_actions (user_id, status, created_at DESC);

                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id BIGSERIAL PRIMARY KEY,
                    session_id UUID,
                    run_id UUID,
                    action_id UUID,
                    event_type TEXT NOT NULL,
                    tool_name TEXT,
                    status TEXT NOT NULL,
                    safe_metadata JSONB,
                    previous_hash TEXT,
                    event_hash TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS audit_events_created_idx
                    ON audit_events (created_at, event_id);

                CREATE TABLE IF NOT EXISTS runtime_alerts (
                    rule TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'healthy',
                    severity TEXT NOT NULL DEFAULT 'healthy',
                    last_triggered_at TIMESTAMPTZ,
                    recovered_at TIMESTAMPTZ,
                    suppressed_count INTEGER NOT NULL DEFAULT 0,
                    last_evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
                    last_value DOUBLE PRECISION,
                    sample_size INTEGER NOT NULL DEFAULT 0,
                    last_evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS agent_run_checkpoints (
                    checkpoint_id BIGSERIAL PRIMARY KEY,
                    session_id UUID NOT NULL,
                    run_id UUID NOT NULL,
                    stage TEXT NOT NULL,
                    state_json JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS agent_run_checkpoints_run_idx
                    ON agent_run_checkpoints (run_id, checkpoint_id);

                CREATE TABLE IF NOT EXISTS original_source_blobs (
                    storage_object_key TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL,
                    byte_size BIGINT NOT NULL,
                    content BYTEA NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS original_sources (
                    source_id UUID PRIMARY KEY,
                    owner_principal_id TEXT NOT NULL,
                    source_version INTEGER NOT NULL DEFAULT 1,
                    original_filename TEXT NOT NULL,
                    mime_type TEXT,
                    byte_size BIGINT NOT NULL,
                    sha256 TEXT NOT NULL,
                    storage_backend TEXT NOT NULL,
                    storage_object_key TEXT NOT NULL
                        REFERENCES original_source_blobs(storage_object_key),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    original_available BOOLEAN NOT NULL DEFAULT TRUE,
                    authorization_scope TEXT NOT NULL,
                    extraction_status TEXT NOT NULL DEFAULT 'pending',
                    UNIQUE (source_id, source_version)
                );

                CREATE INDEX IF NOT EXISTS original_sources_owner_idx
                    ON original_sources (owner_principal_id, created_at DESC);

                ALTER TABLE documents
                    ADD COLUMN IF NOT EXISTS source_id UUID;
                ALTER TABLE documents
                    ADD COLUMN IF NOT EXISTS source_sha256 TEXT;
                ALTER TABLE documents
                    ADD COLUMN IF NOT EXISTS original_available BOOLEAN
                    NOT NULL DEFAULT FALSE;
                ALTER TABLE document_chunks
                    ADD COLUMN IF NOT EXISTS source_id UUID;
                ALTER TABLE document_chunks
                    ADD COLUMN IF NOT EXISTS source_sha256 TEXT;

                CREATE INDEX IF NOT EXISTS documents_source_idx
                    ON documents (source_id);
                CREATE INDEX IF NOT EXISTS document_chunks_source_idx
                    ON document_chunks (source_id);

                ALTER TABLE agent_runs
                    ADD COLUMN IF NOT EXISTS stage TEXT NOT NULL DEFAULT 'created';
                ALTER TABLE agent_runs
                    ADD COLUMN IF NOT EXISTS worker_id TEXT;
                ALTER TABLE agent_runs
                    ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ;
                ALTER TABLE agent_runs
                    ADD COLUMN IF NOT EXISTS lease_version BIGINT NOT NULL DEFAULT 0;
                ALTER TABLE agent_runs
                    ADD COLUMN IF NOT EXISTS attempt_count INTEGER NOT NULL DEFAULT 0;
                ALTER TABLE agent_runs
                    ADD COLUMN IF NOT EXISTS last_checkpoint_at TIMESTAMPTZ;
                ALTER TABLE agent_runs
                    ADD COLUMN IF NOT EXISTS recovery_status TEXT NOT NULL DEFAULT 'active';
                UPDATE agent_runs
                SET status = 'running', recovery_status = 'orphaned'
                WHERE status = 'orphaned';

                ALTER TABLE pending_actions
                    ADD COLUMN IF NOT EXISTS idempotency_key TEXT;
                CREATE UNIQUE INDEX IF NOT EXISTS pending_actions_run_idempotency_idx
                    ON pending_actions (run_id, idempotency_key)
                    WHERE idempotency_key IS NOT NULL;
                """
            )
        except Exception as error:
            raise PersistenceError("Could not initialize policy persistence.") from error

        # Hardening: enforce one row per file_hash so a duplicate upload racing
        # past the application-level check cannot insert twice. Kept outside the
        # block above and failure-tolerant on purpose: a pre-dedup-era database
        # may still hold duplicate file_hash rows, and failing here would brick
        # startup for existing installs. When the index cannot be built, the
        # ON CONFLICT path in store_document degrades to a plain INSERT and the
        # application-level find_document_by_hash check remains the guard.
        try:
            await pool.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS documents_file_hash_key
                    ON documents (file_hash);
                """
            )
        except Exception as error:
            logger.warning(
                "documents_file_hash_key unique index not created "
                "(existing duplicate file_hash rows?): %s",
                error,
            )
        _schema_ready = True


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise PersistenceError("DATABASE_URL is not configured.")
        try:
            _pool = await asyncpg.create_pool(
                dsn=database_url,
                min_size=1,
                max_size=5,
                command_timeout=30,
            )
        except Exception as error:
            raise PersistenceError("PostgreSQL is unavailable.") from error
    await _ensure_policy_schema(_pool)
    return _pool



async def doctor_storage_health() -> dict[str, Any]:
    pool = await _get_pool()
    try:
        row = await pool.fetchrow(
            """
            SELECT
                current_setting('server_version') AS postgres_version,
                EXISTS (
                    SELECT 1 FROM pg_extension WHERE extname = 'vector'
                ) AS pgvector_available,
                (SELECT COUNT(*)::int FROM pending_actions) AS pending_actions,
                (SELECT COUNT(*)::int FROM audit_events) AS audit_events,
                (SELECT COUNT(*)::int FROM agent_run_checkpoints) AS run_checkpoints,
                to_tsvector('simple', 'health check') @@
                    plainto_tsquery('simple', 'health') AS fts_healthy,
                EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'document_chunks'
                      AND column_name = 'embedding'
                ) AS vector_column_available
            """
        )
    except Exception as error:
        raise PersistenceError("Could not check storage health.") from error
    return dict(row)



async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None