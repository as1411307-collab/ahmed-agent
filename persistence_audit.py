from __future__ import annotations

import hashlib
import json

import asyncpg

from persistence_core import PersistenceError, _canonical_json, _get_pool


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


async def verify_audit_chain(limit: int = 5000) -> dict[str, Any]:
    """Verify the audit hash chain over a bounded, most-recent window.

    Full-table scans do not scale (every audit event in one fetch). We verify
    the newest ``limit`` events, anchored at the stored previous_hash of the
    first event in the window, so tampering inside the window is still
    detected and the result stays honest about its coverage.
    """
    pool = await _get_pool()
    try:
        rows = await pool.fetch(
            """
            SELECT event_id, session_id, run_id, action_id, event_type,
                   tool_name, status, safe_metadata, previous_hash, event_hash
            FROM (
                SELECT event_id
                FROM audit_events
                ORDER BY event_id DESC
                LIMIT $1
            ) recent
            JOIN audit_events a USING (event_id)
            ORDER BY a.event_id ASC
            """,
            limit,
        )
    except Exception as error:
        raise PersistenceError("Could not read the audit chain.") from error
    if rows:
        # The window anchor: trust the first event's stored previous_hash and
        # verify forward linkage from there.
        previous_hash: str | None = rows[0]["previous_hash"]
    else:
        previous_hash = None
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
        "window_limited": len(rows) == limit,
        "safe_error_code": None,
    }


