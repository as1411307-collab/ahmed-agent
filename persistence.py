from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
from collections.abc import Sequence
from typing import Any
from uuid import UUID

import asyncpg


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
            _schema_ready = True
        except Exception as error:
            raise PersistenceError("Could not initialize policy persistence.") from error


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


async def ensure_session(session_id: str, *, scope: str = "chat") -> None:
    pool = await _get_pool()
    try:
        await pool.execute(
            """
            INSERT INTO agent_sessions (session_id, scope, status)
            VALUES ($1::uuid, $2, 'active')
            ON CONFLICT (session_id) DO UPDATE
            SET status = 'active', updated_at = NOW()
            """,
            session_id,
            scope,
        )
    except Exception as error:
        raise PersistenceError("Could not create or update the agent session.") from error


async def load_message_history(
    session_id: str,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    pool = await _get_pool()
    try:
        rows = await pool.fetch(
            """
            SELECT message_json
            FROM (
                SELECT sequence_number, message_json
                FROM agent_messages
                WHERE session_id = $1::uuid
                ORDER BY sequence_number DESC
                LIMIT $2
            ) recent
            ORDER BY sequence_number ASC
            """,
            session_id,
            limit,
        )
    except Exception as error:
        raise PersistenceError("Could not load the agent session history.") from error
    history: list[dict[str, Any]] = []
    for row in rows:
        message_json = row["message_json"]
        if isinstance(message_json, str):
            message_json = json.loads(message_json)
        if not isinstance(message_json, dict):
            raise PersistenceError("Stored agent history has an invalid message.")
        history.append(message_json)
    return history


async def create_run(
    *,
    run_id: str,
    session_id: str,
    user_prompt: str,
    provider_name: str,
    model_name: str,
    worker_id: str | None = None,
    lease_seconds: int = 120,
) -> None:
    if lease_seconds < 30 or lease_seconds > 900:
        raise ValueError("invalid run lease")
    pool = await _get_pool()
    try:
        await pool.execute(
            """
            INSERT INTO agent_runs (
                run_id,
                session_id,
                status,
                user_prompt,
                provider_name,
                model_name,
                stage,
                worker_id,
                lease_expires_at,
                lease_version,
                attempt_count
            )
            VALUES (
                $1::uuid,
                $2::uuid,
                'running',
                $3,
                $4,
                $5,
                'created',
                $6,
                NOW() + ($7::integer * INTERVAL '1 second'),
                1,
                1
            )
            """,
            run_id,
            session_id,
            user_prompt,
            provider_name,
            model_name,
            worker_id,
            lease_seconds,
        )
    except Exception as error:
        raise PersistenceError("Could not create the agent run.") from error


async def finish_run(
    *,
    run_id: str,
    status: str,
    error_code: str | None = None,
) -> None:
    if status not in {"succeeded", "failed"}:
        raise ValueError("invalid run status")
    pool = await _get_pool()
    try:
        await pool.execute(
            """
            UPDATE agent_runs
            SET status = $2,
                error_code = $3,
                finished_at = NOW(),
                worker_id = NULL,
                lease_expires_at = NULL,
                recovery_status = 'resolved'
            WHERE run_id = $1::uuid
            """,
            run_id,
            status,
            error_code,
        )
    except Exception as error:
        raise PersistenceError("Could not finish the agent run.") from error


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def normalize_metrics_window(value: object) -> int:
    if value is None:
        return 24
    try:
        hours = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid metrics window") from error
    if hours < 1 or hours > 720:
        raise ValueError("invalid metrics window")
    return hours


async def runtime_metrics(*, window_hours: int = 24) -> dict[str, Any]:
    window_hours = normalize_metrics_window(window_hours)
    pool = await _get_pool()
    try:
        row = await pool.fetchrow(
            """
            WITH run_window AS (
                SELECT *
                FROM agent_runs
                WHERE created_at >= NOW() - ($1::integer * INTERVAL '1 hour')
            ),
            stage_summary AS (
                SELECT stage, COUNT(*)::integer AS count
                FROM run_window
                GROUP BY stage
            )
            SELECT
                (SELECT COUNT(*)::integer FROM run_window) AS total_runs,
                (SELECT COUNT(*)::integer FROM run_window
                 WHERE status = 'succeeded') AS completed_runs,
                (SELECT COUNT(*)::integer FROM run_window
                 WHERE status = 'failed') AS failed_runs,
                (SELECT COUNT(*)::integer FROM run_window
                 WHERE status = 'running') AS active_runs,
                (SELECT COUNT(*)::integer FROM run_window
                 WHERE recovery_status = 'orphaned') AS orphaned_runs,
                (SELECT COUNT(*)::integer FROM run_window
                 WHERE error_code = 'WORKER_LOST') AS lease_expirations,
                (SELECT COALESCE(SUM(GREATEST(attempt_count - 1, 0)), 0)::integer
                 FROM run_window) AS recovery_attempts,
                (SELECT COALESCE(
                    AVG(EXTRACT(EPOCH FROM (finished_at - created_at)) * 1000)
                    FILTER (WHERE finished_at IS NOT NULL),
                    0
                ) FROM run_window) AS average_duration_ms,
                (SELECT COUNT(*)::integer
                 FROM audit_events
                 WHERE event_type = 'pending_action.idempotency_hit'
                   AND created_at >= NOW() - ($1::integer * INTERVAL '1 hour')
                ) AS idempotency_hits,
                (SELECT COUNT(*)::integer
                 FROM audit_events
                 WHERE event_type = 'recovery.failed'
                   AND created_at >= NOW() - ($1::integer * INTERVAL '1 hour')
                ) AS recovery_failures,
                (SELECT COUNT(DISTINCT run_id)::integer
                 FROM audit_events
                 WHERE event_type = 'recovery.failed'
                   AND run_id IS NOT NULL
                   AND created_at >= NOW() - ($1::integer * INTERVAL '1 hour')
                ) AS recovery_failed_runs,
                COALESCE(
                    (SELECT jsonb_object_agg(stage, count) FROM stage_summary),
                    '{}'::jsonb
                ) AS stage_counts
            """,
            window_hours,
        )
    except Exception as error:
        raise PersistenceError("Could not load runtime metrics.") from error
    stage_counts = row["stage_counts"] or {}
    if isinstance(stage_counts, str):
        try:
            stage_counts = json.loads(stage_counts)
        except json.JSONDecodeError:
            stage_counts = {}
    if not isinstance(stage_counts, dict):
        stage_counts = {}
    return {
        "window_hours": window_hours,
        "total_runs": int(row["total_runs"] or 0),
        "completed_runs": int(row["completed_runs"] or 0),
        "failed_runs": int(row["failed_runs"] or 0),
        "active_runs": int(row["active_runs"] or 0),
        "orphaned_runs": int(row["orphaned_runs"] or 0),
        "lease_expirations": int(row["lease_expirations"] or 0),
        "recovery_attempts": int(row["recovery_attempts"] or 0),
        "average_duration_ms": round(float(row["average_duration_ms"] or 0), 2),
        "idempotency_hits": int(row["idempotency_hits"] or 0),
        "recovery_failures": int(row["recovery_failures"] or 0),
        "recovery_failed_runs": int(row["recovery_failed_runs"] or 0),
        "stage_counts": stage_counts,
    }


async def persist_runtime_alert_evaluations(
    evaluations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from alerting import transition_alert

    pool = await _get_pool()
    persisted: list[dict[str, Any]] = []
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                for evaluation in evaluations:
                    row = await connection.fetchrow(
                        """
                        SELECT rule, status, severity, last_triggered_at,
                               recovered_at, suppressed_count, last_evidence,
                               last_value, sample_size, last_evaluated_at
                        FROM runtime_alerts
                        WHERE rule = $1
                        FOR UPDATE
                        """,
                        evaluation["rule"],
                    )
                    current = dict(row) if row else None
                    transition = transition_alert(
                        evaluation=evaluation,
                        current=current,
                    )
                    await connection.execute(
                        """
                        INSERT INTO runtime_alerts (
                            rule, status, severity, last_triggered_at,
                            recovered_at, suppressed_count, last_evidence,
                            last_value, sample_size, last_evaluated_at
                        )
                        VALUES (
                            $1, $2, $3, $4, $5, $6, $7::jsonb,
                            $8, $9, $10
                        )
                        ON CONFLICT (rule) DO UPDATE SET
                            status = EXCLUDED.status,
                            severity = EXCLUDED.severity,
                            last_triggered_at = EXCLUDED.last_triggered_at,
                            recovered_at = EXCLUDED.recovered_at,
                            suppressed_count = EXCLUDED.suppressed_count,
                            last_evidence = EXCLUDED.last_evidence,
                            last_value = EXCLUDED.last_value,
                            sample_size = EXCLUDED.sample_size,
                            last_evaluated_at = EXCLUDED.last_evaluated_at
                        """,
                        transition["rule"],
                        transition["status"],
                        transition["severity"],
                        transition["last_triggered_at"],
                        transition["recovered_at"],
                        transition["suppressed_count"],
                        json.dumps(
                            transition["last_evidence"],
                            separators=(",", ":"),
                        ),
                        float(transition["last_value"]),
                        int(transition["sample_size"]),
                        transition["evaluated_at"],
                    )
                    if transition["event_type"]:
                        await _append_audit_event(
                            connection,
                            session_id=None,
                            run_id=None,
                            action_id=None,
                            event_type=str(transition["event_type"]),
                            tool_name=None,
                            status=str(transition["status"]),
                            safe_metadata={
                                "rule": transition["rule"],
                                "severity": transition["severity"],
                                "suppressed_count": transition["suppressed_count"],
                                "evidence": transition["last_evidence"],
                            },
                        )
                    persisted.append(transition)
    except PersistenceError:
        raise
    except Exception as error:
        raise PersistenceError("Could not persist runtime alert state.") from error
    return persisted


async def retention_preview() -> dict[str, Any]:
    pool = await _get_pool()
    try:
        row = await pool.fetchrow(
            """
            SELECT
                (
                    SELECT COUNT(*)::integer
                    FROM agent_run_checkpoints c
                    JOIN agent_runs r ON r.run_id = c.run_id
                    WHERE (
                        r.status = 'succeeded'
                        AND c.created_at < NOW() - INTERVAL '30 days'
                    )
                    OR (
                        (r.status = 'failed' OR r.recovery_status = 'orphaned')
                        AND c.created_at < NOW() - INTERVAL '90 days'
                    )
                ) AS checkpoint_candidates,
                (
                    SELECT COUNT(*)::integer
                    FROM agent_runs r
                    JOIN agent_sessions s ON s.session_id = r.session_id
                    WHERE s.scope = ANY($1::text[])
                ) AS test_run_candidates,
                (
                    SELECT COUNT(*)::integer
                    FROM audit_events
                ) AS audit_events_preserved
            """,
            list(TEST_RETENTION_SCOPES),
        )
    except Exception as error:
        raise PersistenceError("Could not load retention preview.") from error
    return {
        "policy": RETENTION_POLICY,
        "checkpoint_candidates": int(row["checkpoint_candidates"] or 0),
        "test_run_candidates": int(row["test_run_candidates"] or 0),
        "audit_events_preserved": int(row["audit_events_preserved"] or 0),
        "deletion_mode": "preview_only",
        "supported_cleanup_modes": ["CHECKPOINTS_ONLY", "TEST_DATA_ONLY"],
    }


async def cleanup_retention(*, include_test_data: bool = False) -> dict[str, Any]:
    pool = await _get_pool()
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                deleted_checkpoint_rows = await connection.fetchval(
                    """
                    WITH deleted AS (
                        DELETE FROM agent_run_checkpoints c
                        USING agent_runs r
                        WHERE c.run_id = r.run_id
                          AND (
                              (
                                  r.status = 'succeeded'
                                  AND c.created_at < NOW() - INTERVAL '30 days'
                              )
                              OR (
                                  (r.status = 'failed' OR r.recovery_status = 'orphaned')
                                  AND c.created_at < NOW() - INTERVAL '90 days'
                              )
                          )
                        RETURNING c.checkpoint_id
                    )
                    SELECT COUNT(*)::integer FROM deleted
                    """
                )
                run_id_values: list[str] = []
                deleted_messages = 0
                deleted_test_checkpoint_rows = 0
                deleted_tool_events = 0
                deleted_pending_actions = 0
                deleted_runs = 0
                deleted_sessions = 0

                if include_test_data:
                    run_ids = await connection.fetch(
                        """
                        SELECT r.run_id
                        FROM agent_runs r
                        JOIN agent_sessions s ON s.session_id = r.session_id
                        WHERE s.scope = ANY($1::text[])
                        FOR UPDATE OF r
                        """,
                        list(TEST_RETENTION_SCOPES),
                    )
                    run_id_values = [str(row["run_id"]) for row in run_ids]

                if run_id_values:
                    deleted_messages = await connection.fetchval(
                        """
                        WITH deleted AS (
                            DELETE FROM agent_messages
                            WHERE run_id = ANY($1::uuid[])
                            RETURNING 1
                        )
                        SELECT COUNT(*)::integer FROM deleted
                        """,
                        run_id_values,
                    )
                    deleted_test_checkpoint_rows = await connection.fetchval(
                        """
                        WITH deleted AS (
                            DELETE FROM agent_run_checkpoints
                            WHERE run_id = ANY($1::uuid[])
                            RETURNING 1
                        )
                        SELECT COUNT(*)::integer FROM deleted
                        """,
                        run_id_values,
                    )
                    deleted_tool_events = await connection.fetchval(
                        """
                        WITH deleted AS (
                            DELETE FROM tool_events
                            WHERE run_id = ANY($1::uuid[])
                            RETURNING 1
                        )
                        SELECT COUNT(*)::integer FROM deleted
                        """,
                        run_id_values,
                    )
                    deleted_pending_actions = await connection.fetchval(
                        """
                        WITH deleted AS (
                            DELETE FROM pending_actions
                            WHERE run_id = ANY($1::uuid[])
                            RETURNING 1
                        )
                        SELECT COUNT(*)::integer FROM deleted
                        """,
                        run_id_values,
                    )
                    deleted_runs = await connection.fetchval(
                        """
                        WITH deleted AS (
                            DELETE FROM agent_runs
                            WHERE run_id = ANY($1::uuid[])
                            RETURNING session_id
                        )
                        SELECT COUNT(*)::integer FROM deleted
                        """,
                        run_id_values,
                    )
                    deleted_sessions = await connection.fetchval(
                        """
                        WITH deleted AS (
                            DELETE FROM agent_sessions s
                            WHERE s.scope = ANY($1::text[])
                              AND NOT EXISTS (
                                  SELECT 1
                                  FROM agent_runs r
                                  WHERE r.session_id = s.session_id
                              )
                            RETURNING 1
                        )
                        SELECT COUNT(*)::integer FROM deleted
                        """,
                        list(TEST_RETENTION_SCOPES),
                    )

                deleted_checkpoints = int(deleted_checkpoint_rows or 0) + int(
                    deleted_test_checkpoint_rows or 0
                )
                event_type = (
                    "retention.test_data_cleanup"
                    if include_test_data
                    else "retention.checkpoint_cleanup"
                )
                await _append_audit_event(
                    connection,
                    session_id=None,
                    run_id=None,
                    action_id=None,
                    event_type=event_type,
                    tool_name=None,
                    status="completed",
                    safe_metadata={
                        "scopes": list(TEST_RETENTION_SCOPES)
                        if include_test_data
                        else [],
                        "deleted_runs": int(deleted_runs or 0),
                        "deleted_sessions": int(deleted_sessions or 0),
                        "deleted_messages": int(deleted_messages or 0),
                        "deleted_checkpoints": deleted_checkpoints,
                        "deleted_tool_events": int(deleted_tool_events or 0),
                        "deleted_pending_actions": int(
                            deleted_pending_actions or 0
                        ),
                        "audit_events_deleted": 0,
                    },
                )
                audit_events_preserved = await connection.fetchval(
                    """
                    SELECT COUNT(*)::integer FROM audit_events
                    """
                )
    except PersistenceError:
        raise
    except Exception as error:
        raise PersistenceError("Could not apply retention cleanup.") from error
    return {
        "deleted_runs": int(deleted_runs or 0),
        "deleted_sessions": int(deleted_sessions or 0),
        "deleted_messages": int(deleted_messages or 0),
        "deleted_checkpoints": deleted_checkpoints,
        "deleted_tool_events": int(deleted_tool_events or 0),
        "deleted_pending_actions": int(deleted_pending_actions or 0),
        "audit_events_preserved": int(audit_events_preserved or 0),
        "audit_events_deleted": 0,
        "scopes": list(TEST_RETENTION_SCOPES) if include_test_data else [],
    }


async def renew_run_lease(
    *,
    run_id: str,
    worker_id: str,
    lease_seconds: int = 120,
) -> bool:
    if not worker_id or lease_seconds < 30 or lease_seconds > 900:
        raise ValueError("invalid run lease")
    pool = await _get_pool()
    try:
        updated = await pool.execute(
            """
            UPDATE agent_runs
            SET lease_expires_at = NOW() + ($3::integer * INTERVAL '1 second'),
                lease_version = lease_version + 1
            WHERE run_id = $1::uuid
              AND status = 'running'
              AND worker_id = $2
              AND lease_expires_at > NOW()
            """,
            run_id,
            worker_id,
            lease_seconds,
        )
    except Exception as error:
        raise PersistenceError("Could not renew the agent run lease.") from error
    return updated.endswith("1")


async def mark_orphaned_runs() -> list[str]:
    pool = await _get_pool()
    try:
        rows = await pool.fetch(
            """
            UPDATE agent_runs
            SET recovery_status = 'orphaned',
                error_code = 'WORKER_LOST',
                worker_id = NULL,
                lease_expires_at = NULL,
                finished_at = NULL
            WHERE status = 'running'
              AND recovery_status = 'active'
              AND lease_expires_at IS NOT NULL
              AND lease_expires_at <= NOW()
            RETURNING run_id
            """
        )
    except Exception as error:
        raise PersistenceError("Could not mark orphaned agent runs.") from error
    return [str(row["run_id"]) for row in rows]


async def load_run_recovery(run_id: str) -> dict[str, Any] | None:
    pool = await _get_pool()
    try:
        row = await pool.fetchrow(
            """
            SELECT run_id, session_id, status, recovery_status, stage,
                   provider_name, model_name, error_code, worker_id,
                   lease_expires_at, attempt_count,
                   created_at, finished_at
            FROM agent_runs
            WHERE run_id = $1::uuid
            """,
            run_id,
        )
    except Exception as error:
        raise PersistenceError("Could not load run recovery state.") from error
    if row is None:
        return None
    return {
        "run_id": str(row["run_id"]),
        "session_id": str(row["session_id"]),
        "status": row["status"],
        "recovery_status": row["recovery_status"],
        "stage": row["stage"],
        "provider": row["provider_name"],
        "model": row["model_name"],
        "error_code": row["error_code"],
        "worker_id": row["worker_id"],
        "lease_expires_at": (
            row["lease_expires_at"].isoformat() if row["lease_expires_at"] else None
        ),
        "attempt_count": row["attempt_count"],
        "created_at": row["created_at"].isoformat(),
        "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
    }


async def claim_orphaned_run(
    *,
    run_id: str,
    worker_id: str,
    lease_seconds: int = 120,
) -> dict[str, Any] | None:
    if not worker_id or lease_seconds < 30 or lease_seconds > 900:
        raise ValueError("invalid run lease")
    pool = await _get_pool()
    try:
        row = await pool.fetchrow(
            """
            UPDATE agent_runs
            SET status = 'running',
                recovery_status = 'active',
                worker_id = $2,
                lease_expires_at = NOW() + ($3::integer * INTERVAL '1 second'),
                lease_version = lease_version + 1,
                attempt_count = attempt_count + 1
            WHERE run_id = $1::uuid
              AND status = 'running'
              AND recovery_status = 'orphaned'
              AND (lease_expires_at IS NULL OR lease_expires_at <= NOW())
            RETURNING run_id, session_id, status, recovery_status, stage,
                      provider_name, model_name, error_code, attempt_count
            """,
            run_id,
            worker_id,
            lease_seconds,
        )
    except Exception as error:
        raise PersistenceError("Could not claim the orphaned run.") from error
    return dict(row) if row else None


async def release_orphaned_run(*, run_id: str, worker_id: str) -> bool:
    pool = await _get_pool()
    try:
        updated = await pool.execute(
            """
            UPDATE agent_runs
            SET status = 'running',
                recovery_status = 'orphaned',
                error_code = 'RECOVERY_REQUIRES_NEW_RUN',
                worker_id = NULL,
                lease_expires_at = NULL
            WHERE run_id = $1::uuid
              AND status = 'running'
              AND worker_id = $2
            """,
            run_id,
            worker_id,
        )
    except Exception as error:
        raise PersistenceError("Could not release the recovered agent run.") from error
    return updated.endswith("1")


async def run_message_count(run_id: str) -> int:
    pool = await _get_pool()
    try:
        count = await pool.fetchval(
            "SELECT COUNT(*) FROM agent_messages WHERE run_id = $1::uuid",
            run_id,
        )
    except Exception as error:
        raise PersistenceError("Could not inspect persisted run messages.") from error
    return int(count or 0)


async def append_new_messages(
    *,
    session_id: str,
    run_id: str,
    new_messages_json: bytes | str,
) -> int:
    try:
        raw_messages = json.loads(
            new_messages_json.decode("utf-8")
            if isinstance(new_messages_json, bytes)
            else new_messages_json
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PersistenceError("The agent returned invalid message history.") from error

    if not isinstance(raw_messages, list) or not all(
        isinstance(message, dict) for message in raw_messages
    ):
        raise PersistenceError("The agent returned invalid message history.")

    pool = await _get_pool()
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                next_sequence = await connection.fetchval(
                    """
                    SELECT COALESCE(MAX(sequence_number), 0) + 1
                    FROM agent_messages
                    WHERE session_id = $1::uuid
                    """,
                    session_id,
                )
                for offset, message in enumerate(raw_messages):
                    await connection.execute(
                        """
                        INSERT INTO agent_messages (
                            session_id,
                            run_id,
                            sequence_number,
                            message_json
                        )
                        VALUES ($1::uuid, $2::uuid, $3, $4::jsonb)
                        """,
                        session_id,
                        run_id,
                        next_sequence + offset,
                        json.dumps(message, separators=(",", ":")),
                    )
                await connection.execute(
                    """
                    UPDATE agent_sessions
                    SET updated_at = NOW()
                    WHERE session_id = $1::uuid
                    """,
                    session_id,
                )
    except PersistenceError:
        raise
    except Exception as error:
        raise PersistenceError("Could not persist the agent messages.") from error
    return len(raw_messages)


async def record_tool_event(
    *,
    run_id: str,
    tool_name: str,
    status: str,
    duration_ms: int | None,
    safe_metadata: dict[str, Any] | None,
) -> None:
    if status not in {"success", "failed"}:
        raise ValueError("invalid tool event status")
    pool = await _get_pool()
    try:
        await pool.execute(
            """
            INSERT INTO tool_events (
                run_id,
                tool_name,
                status,
                duration_ms,
                safe_metadata
            )
            VALUES ($1::uuid, $2, $3, $4, $5::jsonb)
            """,
            run_id,
            tool_name,
            status,
            duration_ms,
            json.dumps(safe_metadata or {}, separators=(",", ":")),
        )
    except Exception as error:
        raise PersistenceError("Could not persist the tool event.") from error


async def load_run_evaluation_data(run_id: str) -> dict[str, Any]:
    """Load only redaction-safe metadata needed to build an evaluation trace."""
    pool = await _get_pool()
    try:
        async with pool.acquire() as connection:
            run = await connection.fetchrow(
                """
                SELECT run_id, session_id, status, stage, recovery_status,
                       provider_name, model_name, error_code, attempt_count,
                       created_at, finished_at
                FROM agent_runs
                WHERE run_id = $1::uuid
                """,
                run_id,
            )
            tool_events = await connection.fetch(
                """
                SELECT tool_name, status, duration_ms, safe_metadata
                FROM tool_events
                WHERE run_id = $1::uuid
                ORDER BY id ASC
                """,
                run_id,
            )
            pending_actions = await connection.fetch(
                """
                SELECT action_id, tool_name, risk_level, status, created_at,
                       approved_at, rejected_at, executed_at
                FROM pending_actions
                WHERE run_id = $1::uuid
                ORDER BY created_at ASC
                """,
                run_id,
            )
    except Exception as error:
        raise PersistenceError("Could not load evaluation trace metadata.") from error

    def row_dict(row: Any) -> dict[str, Any]:
        value = dict(row)
        for key, item in list(value.items()):
            if hasattr(item, "isoformat"):
                value[key] = item.isoformat()
        return value

    return {
        "run": row_dict(run) if run is not None else None,
        "tool_events": [row_dict(row) for row in tool_events],
        "pending_actions": [row_dict(row) for row in pending_actions],
    }


async def record_run_checkpoint(
    *,
    session_id: str,
    run_id: str,
    stage: str,
    state: dict[str, Any],
) -> None:
    if not stage or len(stage) > 64:
        raise ValueError("invalid checkpoint stage")
    try:
        state_json = json.dumps(state, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint state must be JSON serializable") from error

    pool = await _get_pool()
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO agent_run_checkpoints (
                        session_id,
                        run_id,
                        stage,
                        state_json
                    )
                    VALUES ($1::uuid, $2::uuid, $3, $4::jsonb)
                    """,
                    session_id,
                    run_id,
                    stage,
                    state_json,
                )
                await connection.execute(
                    """
                    UPDATE agent_runs
                    SET stage = $2,
                        last_checkpoint_at = NOW()
                    WHERE run_id = $1::uuid
                    """,
                    run_id,
                    stage,
                )
                await _append_audit_event(
                    connection,
                    session_id=session_id,
                    run_id=run_id,
                    action_id=None,
                    event_type="run.checkpoint",
                    tool_name=None,
                    status=stage,
                    safe_metadata={"stage": stage},
                )
    except PersistenceError:
        raise
    except Exception as error:
        raise PersistenceError("Could not persist the run checkpoint.") from error


async def load_run_checkpoints(
    *,
    run_id: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    if limit < 1 or limit > 100:
        raise ValueError("invalid checkpoint limit")
    pool = await _get_pool()
    try:
        rows = await pool.fetch(
            """
            SELECT checkpoint_id, session_id, run_id, stage, state_json, created_at
            FROM agent_run_checkpoints
            WHERE run_id = $1::uuid
            ORDER BY checkpoint_id ASC
            LIMIT $2
            """,
            run_id,
            limit,
        )
    except Exception as error:
        raise PersistenceError("Could not load run checkpoints.") from error
    checkpoints: list[dict[str, Any]] = []
    for row in rows:
        state = row["state_json"]
        if isinstance(state, str):
            state = json.loads(state)
        checkpoints.append(
            {
                "checkpoint_id": row["checkpoint_id"],
                "session_id": str(row["session_id"]),
                "run_id": str(row["run_id"]),
                "stage": row["stage"],
                "state": state,
                "created_at": row["created_at"].isoformat(),
            }
        )
    return checkpoints


async def record_auth_event(
    *,
    principal: str,
    authenticated: bool,
    endpoint: str,
    action_id: str | None,
    result: str,
) -> None:
    pool = await _get_pool()
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                await _append_audit_event(
                    connection,
                    session_id=None,
                    run_id=None,
                    action_id=action_id,
                    event_type="auth.authorization",
                    tool_name=None,
                    status=result,
                    safe_metadata={
                        "principal": principal,
                        "authenticated": authenticated,
                        "endpoint": endpoint,
                        "action_id": action_id,
                        "result": result,
                    },
                )
    except Exception as error:
        raise PersistenceError("Could not persist the authorization event.") from error


async def record_recovery_event(
    *,
    session_id: str,
    run_id: str,
    event_type: str,
    status: str,
    safe_metadata: dict[str, Any] | None = None,
) -> None:
    if event_type not in {"recovery.failed", "recovery.completed"}:
        raise ValueError("invalid recovery event type")
    pool = await _get_pool()
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                await _append_audit_event(
                    connection,
                    session_id=session_id,
                    run_id=run_id,
                    action_id=None,
                    event_type=event_type,
                    tool_name=None,
                    status=status,
                    safe_metadata=safe_metadata or {},
                )
    except Exception as error:
        raise PersistenceError("Could not persist the recovery event.") from error


async def _append_audit_event(
    connection: asyncpg.Connection,
    *,
    session_id: str | None,
    run_id: str | None,
    action_id: str | None,
    event_type: str,
    tool_name: str | None,
    status: str,
    safe_metadata: dict[str, Any] | None,
) -> int:
    previous_hash = await connection.fetchval(
        """
        SELECT event_hash
        FROM audit_events
        ORDER BY event_id DESC
        LIMIT 1
        FOR UPDATE
        """
    )
    payload = {
        "session_id": session_id,
        "run_id": run_id,
        "action_id": action_id,
        "event_type": event_type,
        "tool_name": tool_name,
        "status": status,
        "safe_metadata": safe_metadata or {},
        "previous_hash": previous_hash,
    }
    event_hash = hashlib.sha256(
        ((previous_hash or "") + _canonical_json(payload)).encode("utf-8")
    ).hexdigest()
    return await connection.fetchval(
        """
        INSERT INTO audit_events (
            session_id,
            run_id,
            action_id,
            event_type,
            tool_name,
            status,
            safe_metadata,
            previous_hash,
            event_hash
        )
        VALUES ($1::uuid, $2::uuid, $3::uuid, $4, $5, $6, $7::jsonb, $8, $9)
        RETURNING event_id
        """,
        session_id,
        run_id,
        action_id,
        event_type,
        tool_name,
        status,
        json.dumps(safe_metadata or {}, separators=(",", ":")),
        previous_hash,
        event_hash,
    )


async def create_pending_action(
    *,
    action_id: str,
    session_id: str,
    run_id: str,
    user_id: str,
    tool_name: str,
    risk_level: str,
    arguments: dict[str, Any],
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    pool = await _get_pool()
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO pending_actions (
                        action_id,
                        session_id,
                        run_id,
                        user_id,
                        tool_name,
                        risk_level,
                        arguments_json,
                        idempotency_key
                    )
                    VALUES ($1::uuid, $2::uuid, $3::uuid, $4, $5, $6, $7::jsonb, $8)
                    ON CONFLICT (run_id, idempotency_key)
                    WHERE idempotency_key IS NOT NULL
                    DO NOTHING
                    """,
                    action_id,
                    session_id,
                    run_id,
                    user_id,
                    tool_name,
                    risk_level,
                    json.dumps(arguments, separators=(",", ":")),
                    idempotency_key,
                )
                existing = await connection.fetchrow(
                    """
                    SELECT action_id, session_id, run_id, user_id, tool_name,
                           risk_level, status, created_at, expires_at
                    FROM pending_actions
                    WHERE run_id = $1::uuid
                      AND idempotency_key = $2
                    """,
                    run_id,
                    idempotency_key,
                ) if idempotency_key else None
                if existing is not None:
                    await _append_audit_event(
                        connection,
                        session_id=session_id,
                        run_id=run_id,
                        action_id=str(existing["action_id"]),
                        event_type="pending_action.idempotency_hit",
                        tool_name=tool_name,
                        status="deduplicated",
                        safe_metadata={"risk_level": risk_level},
                    )
                    return dict(existing)
                await _append_audit_event(
                    connection,
                    session_id=session_id,
                    run_id=run_id,
                    action_id=action_id,
                    event_type="pending_action.created",
                    tool_name=tool_name,
                    status="pending",
                    safe_metadata={"risk_level": risk_level},
                )
                row = await connection.fetchrow(
                    """
                    SELECT action_id, session_id, run_id, user_id, tool_name,
                           risk_level, status, created_at, expires_at
                    FROM pending_actions
                    WHERE action_id = $1::uuid
                    """,
                    action_id,
                )
    except Exception as error:
        raise PersistenceError("Could not create the pending action.") from error
    return dict(row)


async def approve_pending_action(
    *,
    action_id: str,
    user_id: str,
) -> dict[str, Any]:
    pool = await _get_pool()
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """
                    SELECT action_id, session_id, run_id, user_id, tool_name,
                           risk_level, status, expires_at
                    FROM pending_actions
                    WHERE action_id = $1::uuid
                    FOR UPDATE
                    """,
                    action_id,
                )
                if row is None:
                    return {"status": "not_found", "should_execute": False}
                if row["user_id"] != user_id:
                    return {"status": "forbidden", "should_execute": False}
                status = row["status"]
                if status == "pending" and row["expires_at"] <= await connection.fetchval(
                    "SELECT NOW()"
                ):
                    await connection.execute(
                        """
                        UPDATE pending_actions
                        SET status = 'expired'
                        WHERE action_id = $1::uuid
                        """,
                        action_id,
                    )
                    await _append_audit_event(
                        connection,
                        session_id=str(row["session_id"]),
                        run_id=str(row["run_id"]),
                        action_id=action_id,
                        event_type="pending_action.expired",
                        tool_name=row["tool_name"],
                        status="expired",
                        safe_metadata={},
                    )
                    return {"status": "expired", "should_execute": False}
                if status == "pending":
                    await connection.execute(
                        """
                        UPDATE pending_actions
                        SET status = 'approved', approved_at = NOW()
                        WHERE action_id = $1::uuid
                        """,
                        action_id,
                    )
                    await _append_audit_event(
                        connection,
                        session_id=str(row["session_id"]),
                        run_id=str(row["run_id"]),
                        action_id=action_id,
                        event_type="pending_action.approved",
                        tool_name=row["tool_name"],
                        status="approved",
                        safe_metadata={},
                    )
                    status = "approved"
                return {
                    "status": status,
                    "should_execute": status == "approved",
                    "tool_name": row["tool_name"],
                    "run_id": str(row["run_id"]),
                    "session_id": str(row["session_id"]),
                }
    except Exception as error:
        raise PersistenceError("Could not approve the pending action.") from error


async def claim_pending_action_execution(
    *,
    action_id: str,
    user_id: str,
) -> dict[str, Any]:
    pool = await _get_pool()
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """
                    SELECT action_id, session_id, run_id, user_id, tool_name,
                           risk_level, status, arguments_json
                    FROM pending_actions
                    WHERE action_id = $1::uuid
                    FOR UPDATE
                    """,
                    action_id,
                )
                if row is None:
                    return {"status": "not_found", "should_execute": False}
                if row["user_id"] != user_id:
                    return {"status": "forbidden", "should_execute": False}
                if row["status"] == "executed":
                    return {"status": "executed", "should_execute": False}
                if row["status"] != "approved":
                    return {"status": row["status"], "should_execute": False}
                await connection.execute(
                    """
                    UPDATE pending_actions
                    SET status = 'executing'
                    WHERE action_id = $1::uuid
                    """,
                    action_id,
                )
                await _append_audit_event(
                    connection,
                    session_id=str(row["session_id"]),
                    run_id=str(row["run_id"]),
                    action_id=action_id,
                    event_type="pending_action.execution_started",
                    tool_name=row["tool_name"],
                    status="executing",
                    safe_metadata={},
                )
                arguments = row["arguments_json"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                return {
                    "status": "executing",
                    "should_execute": True,
                    "tool_name": row["tool_name"],
                    "session_id": str(row["session_id"]),
                    "run_id": str(row["run_id"]),
                    "arguments": arguments if isinstance(arguments, dict) else {},
                }
    except Exception as error:
        raise PersistenceError("Could not claim the pending action.") from error


async def complete_pending_action(
    *,
    action_id: str,
    user_id: str,
) -> dict[str, Any]:
    pool = await _get_pool()
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """
                    SELECT session_id, run_id, user_id, tool_name, status
                    FROM pending_actions
                    WHERE action_id = $1::uuid
                    FOR UPDATE
                    """,
                    action_id,
                )
                if row is None:
                    return {"status": "not_found"}
                if row["user_id"] != user_id:
                    return {"status": "forbidden"}
                if row["status"] == "executed":
                    return {"status": "executed"}
                if row["status"] != "executing":
                    return {"status": row["status"]}
                await connection.execute(
                    """
                    UPDATE pending_actions
                    SET status = 'executed', executed_at = NOW()
                    WHERE action_id = $1::uuid
                    """,
                    action_id,
                )
                await _append_audit_event(
                    connection,
                    session_id=str(row["session_id"]),
                    run_id=str(row["run_id"]),
                    action_id=action_id,
                    event_type="pending_action.executed",
                    tool_name=row["tool_name"],
                    status="executed",
                    safe_metadata={"execution_count": 1},
                )
                return {"status": "executed"}
    except Exception as error:
        raise PersistenceError("Could not complete the pending action.") from error


async def reject_pending_action(
    *,
    action_id: str,
    user_id: str,
) -> dict[str, Any]:
    pool = await _get_pool()
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """
                    SELECT session_id, run_id, user_id, tool_name, status
                    FROM pending_actions
                    WHERE action_id = $1::uuid
                    FOR UPDATE
                    """,
                    action_id,
                )
                if row is None:
                    return {"status": "not_found"}
                if row["user_id"] != user_id:
                    return {"status": "forbidden"}
                if row["status"] == "pending":
                    await connection.execute(
                        """
                        UPDATE pending_actions
                        SET status = 'rejected', rejected_at = NOW()
                        WHERE action_id = $1::uuid
                        """,
                        action_id,
                    )
                    await _append_audit_event(
                        connection,
                        session_id=str(row["session_id"]),
                        run_id=str(row["run_id"]),
                        action_id=action_id,
                        event_type="pending_action.rejected",
                        tool_name=row["tool_name"],
                        status="rejected",
                        safe_metadata={},
                    )
                    return {"status": "rejected"}
                return {"status": row["status"]}
    except Exception as error:
        raise PersistenceError("Could not reject the pending action.") from error


async def verify_audit_chain() -> dict[str, Any]:
    pool = await _get_pool()
    try:
        rows = await pool.fetch(
            """
            SELECT event_id, session_id, run_id, action_id, event_type,
                   tool_name, status, safe_metadata, previous_hash, event_hash
            FROM audit_events
            ORDER BY event_id ASC
            """
        )
    except Exception as error:
        raise PersistenceError("Could not read the audit chain.") from error
    previous_hash: str | None = None
    for row in rows:
        safe_metadata = row["safe_metadata"]
        if isinstance(safe_metadata, str):
            safe_metadata = json.loads(safe_metadata)
        payload = {
            "session_id": str(row["session_id"]) if row["session_id"] else None,
            "run_id": str(row["run_id"]) if row["run_id"] else None,
            "action_id": str(row["action_id"]) if row["action_id"] else None,
            "event_type": row["event_type"],
            "tool_name": row["tool_name"],
            "status": row["status"],
            "safe_metadata": safe_metadata or {},
            "previous_hash": previous_hash,
        }
        expected = hashlib.sha256(
            ((
                previous_hash or ""
            ) + _canonical_json(payload)).encode("utf-8")
        ).hexdigest()
        if row["previous_hash"] != previous_hash or row["event_hash"] != expected:
            return {
                "status": "ERROR",
                "verified": False,
                "event_count": len(rows),
                "safe_error_code": "AUDIT_CHAIN_MISMATCH",
            }
        previous_hash = row["event_hash"]
    return {
        "status": "READY",
        "verified": True,
        "event_count": len(rows),
        "safe_error_code": None,
    }


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
) -> None:
    pool = await _get_pool()
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
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


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None