from __future__ import annotations

import json

from persistence_audit import _append_audit_event
from persistence_core import PersistenceError, _get_pool


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


