from __future__ import annotations

import json
import os
import socket
from typing import Any

from persistence_audit import _append_audit_event
from persistence_core import (
    RETENTION_POLICY,
    TEST_RETENTION_SCOPES,
    PersistenceError,
    _get_pool,
)


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
                await connection.execute(
                    """
                    SELECT session_id FROM agent_sessions
                    WHERE session_id = $1::uuid
                    FOR UPDATE
                    """,
                    session_id,
                )
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


