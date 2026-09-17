from __future__ import annotations

import asyncio
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from uuid import uuid4

from agent_core import MAX_MODEL_REQUESTS, MAX_TOOL_CALLS, provider_model_name
from eval_http import (
    _extract_trace_sources,
    _post_chat_message,
    _run_aa_rc_018_upload_e2e,
    redact_evaluation_text,
    validate_case_evidence_preconditions,
)
from eval_tools import case_execution_capability
from evidence_citations import build_evidence_provenance
from persistence import cleanup_evaluation_run, load_run_evaluation_data


def _retry_after_seconds(headers: dict[str, str]) -> float | None:
    value = headers.get("Retry-After") or headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _case_scope(case: dict[str, Any]) -> str:
    return (
        "MY_FILES"
        if "my_files" in case["required_tools"]
        or case.get("category") == "source_of_truth"
        else "WEB"
    )


def build_evaluation_trace(
    *,
    case: dict[str, Any],
    run_id: str,
    scope: str,
    provider: str,
    response_status: int,
    response_payload: dict[str, Any],
    persisted: dict[str, Any],
    latency_ms: int,
    retry_after_seconds: float | None = None,
) -> dict[str, Any]:
    def safe_event_metadata(event: dict[str, Any]) -> dict[str, Any]:
        metadata = event.get("safe_metadata")
        if isinstance(metadata, dict):
            return metadata
        if isinstance(metadata, str):
            try:
                decoded = json.loads(metadata)
            except json.JSONDecodeError:
                return {}
            return decoded if isinstance(decoded, dict) else {}
        return {}

    run = persisted.get("run") or {}
    final_output = response_payload.get("reply")
    if not isinstance(final_output, str):
        final_output = ""
    final_output = redact_evaluation_text(final_output)
    tool_events = persisted.get("tool_events", [])
    pending_actions = persisted.get("pending_actions", [])
    tool_calls = [
        {
            "name": event.get("tool_name"),
            "status": event.get("status"),
            "duration_ms": event.get("duration_ms"),
            "metadata": safe_event_metadata(event),
        }
        for event in tool_events
    ]
    available_evidence_citations = sorted(
        {
            citation
            for event in tool_events
            for citation in safe_event_metadata(event).get(
                "evidence_citations", []
            )
            if isinstance(citation, str)
        }
    )
    available_evidence_items = [
        item
        for event in tool_events
        for item in safe_event_metadata(event).get("evidence_items", [])
        if isinstance(item, dict)
    ]
    evidence_provenance = [
        provenance
        for event in tool_events
        for provenance in safe_event_metadata(event).get(
            "evidence_provenance",
            [],
        )
        if isinstance(provenance, dict)
    ]
    if not evidence_provenance:
        evidence_provenance = build_evidence_provenance(available_evidence_items)
    evidence_provenance = list(
        {
            json.dumps(item, ensure_ascii=False, sort_keys=True): item
            for item in evidence_provenance
        }.values()
    )[:24]
    external_evidence_provenance = [
        provenance
        for event in tool_events
        for provenance in safe_event_metadata(event).get(
            "external_evidence_provenance",
            [],
        )
        if isinstance(provenance, dict)
    ]
    external_evidence_provenance = list(
        {
            json.dumps(item, ensure_ascii=False, sort_keys=True): item
            for item in external_evidence_provenance
        }.values()
    )[:24]
    if not available_evidence_citations:
        available_evidence_citations = sorted(
            {
                str(item["citation"])
                for item in evidence_provenance
                if isinstance(item.get("citation"), str)
            }
        )
    pending_events = [
        {
            "tool_name": event.get("tool_name"),
            "risk_level": event.get("risk_level"),
            "status": event.get("status"),
        }
        for event in pending_actions
    ]
    is_http_success = 200 <= response_status < 300
    run_status = run.get("status")
    if not is_http_success or run_status == "failed":
        execution_status = "EXECUTION_FAILED"
    elif pending_events:
        execution_status = "HITL_BLOCKED"
    else:
        execution_status = "EXECUTED"
    failure_reason = (
        run.get("error_code")
        or response_payload.get("error")
        if execution_status == "EXECUTION_FAILED"
        else None
    )
    if isinstance(failure_reason, str):
        failure_reason = redact_evaluation_text(failure_reason)
    return {
        "case_id": case["id"],
        "run_id": run_id,
        "input": redact_evaluation_text(case["input"]),
        "scope": scope,
        "provider": provider,
        "model": run.get("model_name") or provider_model_name(provider),
        "final_output": final_output,
        "output": {"answer": final_output},
        "tool_calls": tool_calls,
        "citations": _extract_trace_sources(final_output),
        "sources": _extract_trace_sources(final_output),
        "available_evidence_citations": available_evidence_citations,
        "available_evidence_items": available_evidence_items,
        "evidence_provenance": evidence_provenance,
        "external_evidence_provenance": external_evidence_provenance,
        "pending_action_events": pending_events,
        "approval_requested": bool(pending_events),
        "executed_without_approval": any(
            event.get("status") in {"executing", "executed"}
            for event in pending_actions
        ),
        "abstained": None,
        "start_timestamp": run.get("created_at"),
        "end_timestamp": run.get("finished_at"),
        "latency_ms": latency_ms,
        "run_attempts": run.get("attempt_count"),
        "retries": None,
        "tokens": None,
        "cost": None,
        "execution_status": execution_status,
        "execution_error": failure_reason,
        "retry_after_seconds": retry_after_seconds,
    }


def authorized_source_identity_preconditions(
    *,
    case: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    """Require every AA-RC-002 expected source identity to be observed."""

    expected = [
        str(value).strip().casefold()
        for value in case.get("expected_sources", [])
        if str(value).strip()
    ]
    observed: list[str] = []
    for call in trace.get("tool_calls", []):
        if not isinstance(call, dict) or call.get("name") != "inspect_source_of_truth":
            continue
        metadata = call.get("metadata")
        if not isinstance(metadata, dict):
            continue
        for key in ("source_filename", "source_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                observed.append(value.strip().casefold())
        for item in metadata.get("evidence_items", []):
            if not isinstance(item, dict):
                continue
            value = item.get("relative_source_path")
            if isinstance(value, str) and value.strip():
                observed.append(value.strip().casefold())
    missing = [
        source
        for source in expected
        if not any(source in candidate for candidate in observed)
    ]
    return {
        "status": "VERIFIED" if not missing else "NOT_DETERMINED",
        "expected_source_count": len(expected),
        "observed_source_count": len(set(observed)),
        "missing_sources": missing,
    }


async def execute_real_case(
    case: dict[str, Any],
    *,
    base_url: str,
    owner_token: str,
    provider: str = "gemini",
    timeout_seconds: float = 90.0,
) -> dict[str, Any]:
    capability = case_execution_capability(case)
    if not capability["executable_by_current_agent_tools"]:
        return {
            "case_id": case["id"],
            "execution_status": "NOT_EXECUTABLE_CAPABILITY_GAP",
            "capability": capability,
        }
    session_id = str(uuid4())
    run_id = str(uuid4())
    scope = _case_scope(case)
    payload = {
        "message": case["input"],
        "conversation_id": session_id,
        "run_id": run_id,
        "scope": scope,
        "provider": provider,
    }
    upload_e2e: dict[str, object] | None = None
    if case["id"] == "AA-RC-018":
        upload_e2e = await _run_aa_rc_018_upload_e2e(
            base_url=base_url,
            owner_token=owner_token,
            timeout_seconds=timeout_seconds,
        )
    started = time.perf_counter()
    response_status, response_payload, response_headers = await asyncio.to_thread(
        _post_chat_message,
        base_url=base_url,
        owner_token=owner_token,
        payload=payload,
        timeout_seconds=timeout_seconds,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    try:
        persisted = await load_run_evaluation_data(run_id)
    except Exception as error:
        persisted = {"run": None, "tool_events": [], "pending_actions": []}
        response_payload = {
            **response_payload,
            "error": f"TRACE_PERSISTENCE_READ_{type(error).__name__}",
        }
    trace = build_evaluation_trace(
        case=case,
        run_id=run_id,
        scope=scope,
        provider=provider,
        response_status=response_status,
        response_payload=response_payload,
        persisted=persisted,
        latency_ms=latency_ms,
        retry_after_seconds=_retry_after_seconds(response_headers),
    )
    try:
        trace["runtime_cleanup"] = await cleanup_evaluation_run(
            run_id=run_id,
            session_id=session_id,
        )
    except Exception as error:
        trace["runtime_cleanup"] = {
            "status": "FAIL",
            "error": type(error).__name__,
            "audit_events_preserved": True,
        }
    if case["id"] == "AA-RC-002":
        source_identity_preconditions = authorized_source_identity_preconditions(
            case=case,
            trace=trace,
        )
        trace["source_identity_preconditions"] = source_identity_preconditions
        if source_identity_preconditions["status"] != "VERIFIED":
            trace["external_input_blocker"] = {
                "status": "NOT_DETERMINED",
                "code": (
                    "AUTHORIZED_SOURCE_DOCUMENTS_MISSING"
                    if not source_identity_preconditions["observed_source_count"]
                    else "AUTHORIZED_SOURCE_DOCUMENT_IDENTITIES_MISMATCHED"
                ),
                "reason": (
                    "The expected AA-RC-002 source identities were not all observed "
                    "with verified canonical provenance."
                ),
            }
    if upload_e2e is not None:
        trace["upload_e2e"] = upload_e2e
        upload_items = upload_e2e.get("evidence_items", [])
        if isinstance(upload_items, list):
            trace["available_evidence_items"] = [
                *trace.get("available_evidence_items", []),
                *[item for item in upload_items if isinstance(item, dict)],
            ][:24]
            upload_provenance = build_evidence_provenance(
                upload_items,
                source_label="AA-RC-018 evaluation uploads",
            )
            trace["evidence_provenance"] = [
                *trace.get("evidence_provenance", []),
                *upload_provenance,
            ][:24]
            trace["available_evidence_citations"] = sorted(
                {
                    *[
                        value
                        for value in trace.get("available_evidence_citations", [])
                        if isinstance(value, str)
                    ],
                    *[
                        str(item["citation"])
                        for item in upload_provenance
                        if isinstance(item.get("citation"), str)
                    ],
                }
            )
    if case["id"] == "AA-RC-026":
        preconditions = validate_case_evidence_preconditions(
            case=case,
            trace=trace,
        )
        trace["evidence_preconditions"] = preconditions
        if preconditions["status"] == "NOT_DETERMINED":
            trace["execution_status"] = "NOT_DETERMINED"
            trace["execution_error"] = "EVIDENCE_PRECONDITIONS_UNMET"
    return trace


async def execute_rate_limited_case_with_retry(
    case: dict[str, Any],
    *,
    base_url: str,
    owner_token: str,
    provider: str,
    timeout_seconds: float,
    max_retries: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
) -> dict[str, Any]:
    attempts = 0
    retry_delays: list[float] = []
    resumed_at = datetime.now(timezone.utc).isoformat()
    trace: dict[str, Any] = {}
    while True:
        attempts += 1
        trace = await execute_real_case(
            case,
            base_url=base_url,
            owner_token=owner_token,
            provider=provider,
            timeout_seconds=timeout_seconds,
        )
        rate_limited = trace.get("execution_error") == "RATE_LIMITED"
        if not rate_limited or len(retry_delays) >= max_retries:
            break
        exponential_delay = min(
            max_delay_seconds,
            base_delay_seconds * (2 ** len(retry_delays)),
        )
        jitter = random.uniform(0.0, max(1.0, base_delay_seconds * 0.25))
        retry_after = trace.get("retry_after_seconds")
        delay = min(
            max_delay_seconds,
            max(float(retry_after or 0.0), exponential_delay + jitter),
        )
        retry_delays.append(round(delay, 3))
        await asyncio.sleep(delay)
    trace["attempts"] = attempts
    trace["retries"] = len(retry_delays)
    trace["retry_delays_seconds"] = retry_delays
    trace["resumed_execution_timestamp"] = resumed_at
    if trace.get("execution_error") == "RATE_LIMITED":
        trace["execution_status"] = "EXECUTION_FAILED_PROVIDER_RATE_LIMIT"
    return trace


async def run_real_cases(
    cases: list[dict[str, Any]],
    *,
    base_url: str,
    owner_token: str,
    provider: str = "gemini",
    timeout_seconds: float = 90.0,
    case_ids: set[str] | None = None,
    inter_case_delay_seconds: float = 0.0,
) -> dict[str, Any]:
    selected = [case for case in cases if case_ids is None or case["id"] in case_ids]
    baseline_run_id = str(uuid4())
    case_results: list[dict[str, Any]] = []
    traces: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(selected):
        if index and inter_case_delay_seconds > 0:
            await asyncio.sleep(inter_case_delay_seconds)
        result = await execute_real_case(
            case,
            base_url=base_url,
            owner_token=owner_token,
            provider=provider,
            timeout_seconds=timeout_seconds,
        )
        case_results.append(result)
        if result.get("execution_status") in {
            "EXECUTED",
            "HITL_BLOCKED",
            "NOT_DETERMINED",
        }:
            traces[case["id"]] = result
    executable_cases = [
        case
        for case in selected
        if case_execution_capability(case)["executable_by_current_agent_tools"]
    ]
    scoreboard = build_scoreboard(executable_cases, traces)
    coverage = Counter(result["execution_status"] for result in case_results)
    rate_limited = sum(
        result.get("execution_error") == "RATE_LIMITED"
        for result in case_results
    )
    return {
        "baseline_run_id": baseline_run_id,
        "execution_config": {
            "base_url": base_url,
            "provider": provider,
            "timeout_seconds": timeout_seconds,
            "tool_call_limit": MAX_TOOL_CALLS,
            "model_request_limit": MAX_MODEL_REQUESTS,
        },
        "coverage": {
            "total_selected": len(selected),
            "executed": coverage.get("EXECUTED", 0),
            "execution_failed": coverage.get("EXECUTION_FAILED", 0),
            "capability_gap": coverage.get("NOT_EXECUTABLE_CAPABILITY_GAP", 0),
            "hitl_blocked": coverage.get("HITL_BLOCKED", 0),
            "not_determined": coverage.get("NOT_DETERMINED", 0),
            "provider_rate_limited": rate_limited,
        },
        "cases": case_results,
        "deterministic_scoreboard": scoreboard,
        "semantic_grading": "NOT_RUN",
    }


