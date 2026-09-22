from __future__ import annotations

import asyncio
import logging
import json
import os
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)


class PersistenceError(RuntimeError):
    pass


_pool: asyncpg.Pool | None = None
_schema_ready = False
_schema_lock = asyncio.Lock()
_vector_storage_available = False
_fts_index_task: asyncio.Task[None] | None = None
TEST_RETENTION_SCOPES = ("fault_injection", "probe")
RETENTION_POLICY = {
    "completed_checkpoint_days": 30,
    "failed_orphaned_checkpoint_days": 90,
    "audit_events": "preserved",
    "test_scopes": TEST_RETENTION_SCOPES,
}


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


async def _build_fts_index(pool: asyncpg.Pool) -> None:
    """Build document_chunks' FTS index in the background (see the call site
    at the end of _ensure_policy_schema for why this must never be awaited
    from there, and why it's only fired after all schema DDL completes).

    CONCURRENTLY and its own (non-multi-statement) execute() calls, on
    purpose: a plain CREATE INDEX would hold a lock that blocks writes for
    the whole build. Failure here is logged and tolerated -- FTS just uses
    a full scan until it eventually succeeds.

    A CONCURRENTLY build that gets cancelled (a timeout, or the process
    dying) leaves an INVALID index cataloged under this name -- the
    planner never uses it, and CREATE INDEX CONCURRENTLY IF NOT EXISTS
    matches on the name regardless of validity, so it would silently skip
    forever without the explicit check below.
    """
    try:
        # Resolve 'document_chunks' via ::regclass (the same search_path
        # resolution the unqualified "ON document_chunks" below uses)
        # instead of matching the index name alone: a bare relname match
        # would also catch a same-named index belonging to an unrelated
        # schema's document_chunks table on a Postgres instance shared by
        # multiple apps/schemas, wrongly dropping and rebuilding our own
        # valid index on every startup while never touching the actual
        # invalid one.
        invalid_index_exists = await pool.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM pg_index i
                JOIN pg_class c ON c.oid = i.indexrelid
                WHERE i.indrelid = 'document_chunks'::regclass
                  AND c.relname = 'document_chunks_content_fts_idx'
                  AND NOT i.indisvalid
            )
            """
        )
        if invalid_index_exists:
            logger.warning(
                "document_chunks_content_fts_idx exists but is invalid "
                "(a previous CONCURRENTLY build was likely cancelled); "
                "dropping it so it can be rebuilt."
            )
            await pool.execute(
                "DROP INDEX CONCURRENTLY IF EXISTS document_chunks_content_fts_idx;"
            )
    except Exception as error:
        logger.warning(
            "Could not check/drop an invalid document_chunks_content_fts_idx: %s",
            error,
        )
        return

    try:
        # An explicit timeout, well above the pool's 30s command_timeout:
        # this now runs in the background rather than gating any request,
        # so there's no reason to cut off a legitimately slow build on a
        # large existing corpus.
        await pool.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS document_chunks_content_fts_idx
                ON document_chunks USING GIN (to_tsvector('simple', content));
            """,
            timeout=300,
        )
    except Exception as error:
        logger.warning(
            "Could not build document_chunks_content_fts_idx; FTS queries "
            "will use a full scan until this succeeds: %s",
            error,
        )


async def _ensure_base_schema(pool: asyncpg.Pool) -> None:
    """Create the tables every other table here is ALTERed or joined against.

    These six tables (agent_sessions, agent_runs, agent_messages, tool_events,
    documents, document_chunks) were never defined anywhere in this repo --
    they only existed because the original deployment's Postgres instance had
    them provisioned out-of-band. A fresh database (e.g. a new VPS) had no way
    to create them. This intentionally matches the exact shape every query in
    persistence_runs.py/persistence_docs.py/persistence_events.py already
    assumes (see those files for the column-by-column usage), including no
    foreign keys between the operational tables -- the same looseness the
    rest of this schema already uses (pending_actions/agent_run_checkpoints
    also carry a bare run_id with no REFERENCES), which is what lets
    cleanup_retention/cleanup_evaluation_run delete across these tables in a
    specific manual order without fighting FK constraints.

    pgvector is optional here, matching the existing degraded-mode contract
    (doctor.py reports DEGRADED, not down, when pgvector is missing): the
    relational tables and the document_chunks FTS index are always created,
    and only the document_chunks.embedding column is skipped when the
    'vector' extension can't be installed, so sessions/runs/messages/audit
    and full-text MY_FILES search keep working without it.
    """

    pgvector_available = True
    try:
        await pool.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    except Exception as error:
        pgvector_available = False
        logger.warning(
            "Postgres 'vector' extension (pgvector) is not available; "
            "continuing without it. Vector search stays disabled and "
            "MY_FILES search falls back to full-text search only: %s",
            error,
        )

    try:
        await pool.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_sessions (
                session_id UUID PRIMARY KEY,
                scope TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS agent_runs (
                run_id UUID PRIMARY KEY,
                session_id UUID NOT NULL,
                status TEXT NOT NULL,
                user_prompt TEXT NOT NULL,
                provider_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                error_code TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                finished_at TIMESTAMPTZ
            );

            CREATE INDEX IF NOT EXISTS agent_runs_session_idx
                ON agent_runs (session_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS agent_messages (
                message_id BIGSERIAL PRIMARY KEY,
                session_id UUID NOT NULL,
                run_id UUID NOT NULL,
                sequence_number INTEGER NOT NULL,
                message_json JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (session_id, sequence_number)
            );

            CREATE INDEX IF NOT EXISTS agent_messages_run_idx
                ON agent_messages (run_id);

            CREATE TABLE IF NOT EXISTS tool_events (
                id BIGSERIAL PRIMARY KEY,
                run_id UUID NOT NULL,
                tool_name TEXT NOT NULL,
                status TEXT NOT NULL,
                duration_ms INTEGER,
                safe_metadata JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS tool_events_run_idx
                ON tool_events (run_id, id);

            CREATE TABLE IF NOT EXISTS documents (
                document_id UUID PRIMARY KEY,
                filename TEXT NOT NULL,
                mime_type TEXT,
                source_type TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                embedding_model TEXT,
                embedding_dimension INTEGER,
                embedding_version TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id BIGSERIAL PRIMARY KEY,
                document_id UUID NOT NULL REFERENCES documents(document_id)
                    ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                page_number INTEGER,
                content TEXT NOT NULL,
                metadata JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS document_chunks_document_idx
                ON document_chunks (document_id, chunk_index);
            """
        )
    except Exception as error:
        raise PersistenceError("Could not initialize the base persistence schema.") from error

    global _vector_storage_available
    vector_storage_ready = False
    if pgvector_available:
        try:
            await pool.execute(
                "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding vector;"
            )
            vector_storage_ready = True
        except Exception as error:
            logger.warning(
                "Could not add the pgvector 'embedding' column to "
                "document_chunks; vector search stays disabled: %s",
                error,
            )
    _vector_storage_available = vector_storage_ready


async def _ensure_policy_schema(pool: asyncpg.Pool) -> None:
    global _schema_ready
    if _schema_ready:
        return
    async with _schema_lock:
        if _schema_ready:
            return
        await _ensure_base_schema(pool)
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

        # Fire-and-forget, and only now that every ALTER TABLE on
        # document_chunks (the embedding column above, source_id/
        # source_sha256 here) has completed: CREATE INDEX CONCURRENTLY
        # takes a lock that conflicts with ALTER TABLE's ACCESS EXCLUSIVE
        # lock on the same table, so starting the build any earlier could
        # have it contend with -- and stall behind, or block -- schema
        # DDL that's still in flight on document_chunks. Awaiting it here
        # would defeat the point (blocking _schema_lock on a one-time
        # migration), so it still just runs in the background; FTS uses a
        # full scan until it finishes.
        global _fts_index_task
        if _fts_index_task is None or _fts_index_task.done():
            _fts_index_task = asyncio.create_task(_build_fts_index(pool))

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


async def vector_storage_available() -> bool:
    """Whether document_chunks.embedding actually exists on this database.

    Callers that would otherwise send SQL referencing that column (storing
    or searching embeddings) must check this first and skip straight to
    FTS-only behavior when it's False, instead of letting the resulting
    PersistenceError (missing column) propagate and defeat the FTS
    fallback that's the whole point of pgvector being optional.
    """
    await _get_pool()
    return _vector_storage_available



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