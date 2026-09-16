from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from agent_core import all_provider_health, provider_health
from academic_search import academic_health
from auth import auth_health
from embeddings import embedding_health
from github_search import github_health
from persistence import PersistenceError, doctor_storage_health, verify_audit_chain
from policy import TOOL_POLICIES
from skill_tools import search_fabric_health


_CACHE_TTL_SECONDS = 15.0
_cache: tuple[float, dict[str, Any]] | None = None
_cache_lock = asyncio.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _component(
    *,
    status: str,
    last_success: str | None = None,
    last_failure: str | None = None,
    latency_ms: int | None = None,
    safe_error_code: str | None = None,
    **details: Any,
) -> dict[str, Any]:
    return {
        "status": status,
        "last_success": last_success,
        "last_failure": last_failure,
        "latency_ms": latency_ms,
        "safe_error_code": safe_error_code,
        **details,
    }


def _status_from_storage(storage: dict[str, Any]) -> str:
    if not storage.get("pgvector_available") or not storage.get("vector_column_available"):
        return "DEGRADED"
    if not storage.get("fts_healthy"):
        return "DEGRADED"
    return "READY"


async def _build_report(*, probe_search: bool) -> dict[str, Any]:
    started_at = time.perf_counter()
    checked_at = _now()
    storage: dict[str, Any] | None = None
    storage_error: str | None = None
    try:
        storage = await doctor_storage_health()
    except PersistenceError:
        storage_error = "POSTGRES_UNAVAILABLE"

    provider = provider_health()
    provider_status = str(provider.get("status", "ERROR"))
    core_status = "READY" if provider_status not in {"ERROR", "UNAUTHORIZED"} else provider_status
    if storage_error:
        core_status = "DEGRADED"

    if storage is None:
        storage_status = "ERROR"
        storage_details: dict[str, Any] = {}
    else:
        storage_status = _status_from_storage(storage)
        storage_details = {
            "postgres_version": storage.get("postgres_version"),
            "pgvector_available": storage.get("pgvector_available"),
            "pending_actions": storage.get("pending_actions"),
            "audit_events": storage.get("audit_events"),
            "run_checkpoints": storage.get("run_checkpoints"),
        }

    try:
        audit = await verify_audit_chain()
    except PersistenceError:
        audit = {
            "status": "ERROR",
            "verified": False,
            "safe_error_code": "AUDIT_STORAGE_UNAVAILABLE",
        }

    mcp_status = "READY" if TOOL_POLICIES else "ERROR"
    search = search_fabric_health()
    academic = academic_health()
    github = github_health()
    github_overall_status = (
        "READY"
        if github.get("status") == "READY_UNAUTHENTICATED"
        else github.get("status")
    )
    if probe_search and search["status"] == "READY":
        # A live search probe is intentionally opt-in because it can spend credits.
        search["fast_probe"] = "not_run"

    report = {
        "status": (
            "ERROR"
            if any(
                component.get("status") == "ERROR"
                for component in (
                    {"status": core_status},
                    {"status": storage_status},
                    audit,
                    academic,
                    {"status": github_overall_status},
                )
            )
            else "DEGRADED"
            if any(
                component.get("status") in {"DEGRADED", "RATE_LIMITED", "UNAUTHORIZED"}
                for component in (
                    {"status": core_status},
                    {"status": storage_status},
                    provider,
                    audit,
                    academic,
                    {"status": github_overall_status},
                )
            )
            else "READY"
        ),
        "checked_at": checked_at,
        "core": {
            "status": core_status,
            "pydantic_ai": "READY",
            "postgresql": storage_status,
            "mcp": mcp_status,
            "last_success": provider.get("last_success"),
            "last_failure": provider.get("last_failure"),
            "latency_ms": provider.get("latency_ms"),
            "safe_error_code": storage_error,
        },
        "models": {
            "status": provider_status,
            "provider": provider.get("provider"),
            "model": provider.get("model"),
            "available_providers": all_provider_health(),
            "circuit_breaker": provider_status,
            "last_success": provider.get("last_success"),
            "last_failure": provider.get("last_failure"),
            "latency_ms": provider.get("latency_ms"),
            "safe_error_code": provider.get("last_failure_code"),
        },
        "search": search,
        "academic": academic,
        "github": github,
        "my_files": _component(
            status=storage_status if storage is not None else "ERROR",
            safe_error_code=storage_error,
            pgvector_available=storage.get("pgvector_available") if storage else False,
            fts_health=bool(storage and storage.get("fts_healthy")),
            vector_health=bool(storage and storage.get("vector_column_available")),
            embedding=embedding_health(),
        ),
        "security": _component(
            status="READY" if audit.get("verified", False) else "DEGRADED",
            auth=auth_health(),
            policy_engine="READY" if TOOL_POLICIES else "ERROR",
            pending_action_storage=storage_status,
            audit_chain=audit,
            safe_error_code=audit.get("safe_error_code"),
        ),
        "latency_ms": int((time.perf_counter() - started_at) * 1000),
    }
    return report


async def get_doctor_report(*, probe_search: bool = False) -> dict[str, Any]:
    global _cache
    now = time.monotonic()
    if not probe_search and _cache is not None and now - _cache[0] < _CACHE_TTL_SECONDS:
        return _cache[1]
    async with _cache_lock:
        now = time.monotonic()
        if not probe_search and _cache is not None and now - _cache[0] < _CACHE_TTL_SECONDS:
            return _cache[1]
        report = await _build_report(probe_search=probe_search)
        _cache = (time.monotonic(), report)
        return report