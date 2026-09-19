from __future__ import annotations

import asyncio
import hashlib
import json
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_core import provider_model_name
from eval_exec import execute_rate_limited_case_with_retry
from eval_schema import SEMANTIC_RUBRIC
from eval_tools import case_execution_capability


def _code_build_identifier() -> str:
    digest = hashlib.sha256()
    for filename in ("evaluation_baseline.py", "agent_core.py", "server.py", "persistence.py"):
        path = Path(__file__).parent / filename
        digest.update(filename.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _metric_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "task_completion",
        "tool_selection",
        "forbidden_tool_violations",
        "citation_correctness",
        "retrieval_source_correctness",
        "hitl_boundary",
        "abstention",
        "instruction_adherence_deterministic",
    )
    summary: dict[str, Any] = {}
    for metric in metric_names:
        counts = Counter(
            result.get("metrics", {}).get(metric, "NOT_AVAILABLE")
            for result in results
        )
        summary[metric] = {
            "pass": counts.get("PASS", 0),
            "fail": counts.get("FAIL", 0),
            "not_determined": counts.get("NOT_DETERMINED", 0),
            "not_configured": counts.get("NOT_CONFIGURED", 0),
            "not_run": counts.get("NOT_RUN", 0),
            "case_count": len(results),
        }
    return summary


def _numeric_summary(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "average": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "average": round(sum(values) / len(values), 2),
    }


def _build_resume_analysis(
    *,
    dataset_cases: list[dict[str, Any]],
    final_cases: list[dict[str, Any]],
    scoreboard: dict[str, Any],
) -> dict[str, Any]:
    by_id = {case["id"]: case for case in dataset_cases}
    quality_results = scoreboard.get("results", [])
    quality_failures = [
        result["case_id"]
        for result in quality_results
        if result.get("deterministic_status") == "FAIL"
    ]
    provider_failures = [
        result["case_id"]
        for result in final_cases
        if result.get("execution_status")
        in {"EXECUTION_FAILED", "EXECUTION_FAILED_PROVIDER_RATE_LIMIT"}
    ]
    capability_gaps = [
        result["case_id"]
        for result in final_cases
        if result.get("execution_status") == "NOT_EXECUTABLE_CAPABILITY_GAP"
    ]
    failures_by_category = {
        "quality": {
            case_id: by_id[case_id]["category"]
            for case_id in quality_failures
        },
        "provider": {
            result["case_id"]: by_id[result["case_id"]]["category"]
            for result in final_cases
            if result["case_id"] in provider_failures
        },
        "capability_gap": {
            result["case_id"]: by_id[result["case_id"]]["category"]
            for result in final_cases
            if result["case_id"] in capability_gaps
        },
    }
    executable_results = [
        result
        for result in final_cases
        if result.get("execution_status") in {"EXECUTED", "HITL_BLOCKED"}
    ]
    latencies = [
        result["latency_ms"]
        for result in executable_results
        if isinstance(result.get("latency_ms"), (int, float))
    ]
    retries = [
        result["retries"]
        for result in final_cases
        if isinstance(result.get("retries"), int)
    ]
    tool_call_counts = [
        len(result.get("tool_calls", []))
        for result in executable_results
    ]
    tokens_available = sum(result.get("tokens") is not None for result in executable_results)
    costs_available = sum(result.get("cost") is not None for result in executable_results)
    weaknesses: list[dict[str, Any]] = []
    if provider_failures:
        weaknesses.append(
            {
                "area": "Provider",
                "reason": "Some cases remained unexecuted because the provider was rate limited.",
                "case_count": len(provider_failures),
            }
        )
    tool_selection_failures = sum(
        result.get("metrics", {}).get("tool_selection") == "FAIL"
        for result in quality_results
    )
    if tool_selection_failures:
        weaknesses.append(
            {
                "area": "Tool Selection",
                "reason": "The model did not select all required tools for some executed cases.",
                "case_count": tool_selection_failures,
            }
        )
    source_failures = sum(
        result.get("metrics", {}).get("retrieval_source_correctness") == "FAIL"
        for result in quality_results
    )
    if source_failures:
        weaknesses.append(
            {
                "area": "Grounding",
                "reason": "Executed answers did not match the expected source evidence.",
                "case_count": source_failures,
            }
        )
    weaknesses.extend(
        [
            {
                "area": "Retrieval",
                "reason": "Capability gaps prevent evaluation of workspace/project file access.",
                "case_count": len(capability_gaps),
            }
        ]
        if capability_gaps
        else []
    )
    return {
        "quality_failures": quality_failures,
        "provider_failures": provider_failures,
        "capability_gaps": capability_gaps,
        "failures_by_category": failures_by_category,
        "deterministic_metrics": _metric_summary(quality_results),
        "latency_ms": _numeric_summary(latencies),
        "retries": _numeric_summary(retries),
        "tool_call_count": _numeric_summary(tool_call_counts),
        "tokens": {
            "available_case_count": tokens_available,
            "status": "AVAILABLE" if tokens_available else "NOT_AVAILABLE",
        },
        "cost": {
            "available_case_count": costs_available,
            "status": "AVAILABLE" if costs_available else "NOT_AVAILABLE",
        },
        "strongest_observed_areas": weaknesses[:3],
        "semantic_grading": "NOT_RUN",
    }


def _get_owner_json(
    *,
    base_url: str,
    owner_token: str,
    path: str,
    timeout_seconds: float = 15.0,
) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        headers={"Authorization": f"Bearer {owner_token}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as error:
        return error.code, {}
    except Exception:
        return 0, {}


def _runtime_health_report(base_url: str, owner_token: str) -> dict[str, Any]:
    health_status, health = _get_owner_json(
        base_url=base_url,
        owner_token=owner_token,
        path="/health",
    )
    alert_status, alerts = _get_owner_json(
        base_url=base_url,
        owner_token=owner_token,
        path="/alerts/runtime",
    )
    return {
        "health": {
            "http_status": health_status,
            "ok": health.get("ok") is True,
        },
        "alerts": {
            "http_status": alert_status,
            "available": alert_status == 200,
            "delivery": alerts.get("delivery"),
            "audit": alerts.get("audit"),
            "states": alerts.get("alerts", []),
        },
    }


async def resume_rate_limited_baseline(
    baseline: dict[str, Any],
    dataset_cases: list[dict[str, Any]],
    *,
    owner_token: str,
    max_retries: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
    inter_case_delay_seconds: float,
) -> dict[str, Any]:
    original_cases = json.loads(json.dumps(baseline["cases"]))
    dataset_by_id = {case["id"]: case for case in dataset_cases}
    rate_limited_ids = [
        result["case_id"]
        for result in original_cases
        if result.get("execution_status") == "EXECUTION_FAILED"
        and result.get("execution_error") == "RATE_LIMITED"
    ]
    missing_dataset_ids = sorted(set(rate_limited_ids) - set(dataset_by_id))
    if missing_dataset_ids:
        raise ValueError(f"Baseline cases are missing from dataset: {missing_dataset_ids}")
    config = dict(baseline["execution_config"])
    provider = str(config["provider"])
    baseline_models = {
        result.get("model")
        for result in original_cases
        if result.get("model")
    }
    current_model = provider_model_name(provider)
    if baseline_models and current_model not in baseline_models:
        raise ValueError("Current provider model does not match the original baseline.")
    resume_run_id = str(uuid4())
    final_cases = list(original_cases)
    metadata: dict[str, Any] = {}
    resumed_case_ids: list[str] = []
    for index, original in enumerate(original_cases):
        case_id = original["case_id"]
        if case_id not in rate_limited_ids:
            metadata[case_id] = {
                "source": "original",
                "final_execution_status": original.get("execution_status"),
                "attempts": 0
                if original.get("execution_status") == "NOT_EXECUTABLE_CAPABILITY_GAP"
                else 1,
                "retries": 0,
                "original_execution_timestamp": original.get("start_timestamp"),
                "resumed_execution_timestamp": None,
            }
            continue
        if resumed_case_ids and inter_case_delay_seconds > 0:
            await asyncio.sleep(inter_case_delay_seconds)
        resumed = await execute_rate_limited_case_with_retry(
            dataset_by_id[case_id],
            base_url=str(config["base_url"]),
            owner_token=owner_token,
            provider=provider,
            timeout_seconds=float(config["timeout_seconds"]),
            max_retries=max_retries,
            base_delay_seconds=base_delay_seconds,
            max_delay_seconds=max_delay_seconds,
        )
        resumed["original_execution_status"] = original.get("execution_status")
        resumed["original_run_id"] = original.get("run_id")
        resumed["original_execution_timestamp"] = original.get("start_timestamp")
        resumed["resume_run_id"] = resume_run_id
        final_cases[index] = resumed
        resumed_case_ids.append(case_id)
        metadata[case_id] = {
            "source": "resume",
            "final_execution_status": resumed.get("execution_status"),
            "attempts": resumed.get("attempts"),
            "retries": resumed.get("retries"),
            "original_execution_timestamp": original.get("start_timestamp"),
            "resumed_execution_timestamp": resumed.get(
                "resumed_execution_timestamp"
            ),
            "original_run_id": original.get("run_id"),
            "resumed_run_id": resumed.get("run_id"),
            "retry_delays_seconds": resumed.get("retry_delays_seconds", []),
        }
    traces = {
        result["case_id"]: result
        for result in final_cases
        if result.get("execution_status") in {"EXECUTED", "HITL_BLOCKED"}
    }
    executable_cases = [
        case
        for case in dataset_cases
        if case_execution_capability(case)["executable_by_current_agent_tools"]
    ]
    scoreboard = build_scoreboard(executable_cases, traces)
    analysis = _build_resume_analysis(
        dataset_cases=dataset_cases,
        final_cases=final_cases,
        scoreboard=scoreboard,
    )
    coverage = Counter(result.get("execution_status") for result in final_cases)
    artifact = {
        "artifact_version": "baseline-real-v1-complete",
        "original_baseline_run_id": baseline["baseline_run_id"],
        "resume_run_id": resume_run_id,
        "dataset_version": baseline["dataset_version"],
        "dataset_sha256": baseline["dataset_sha256"],
        "dataset_case_count": len(dataset_cases),
        "execution_config": config,
        "model_config": {
            "provider": provider,
            "model": current_model,
            "tool_call_limit": config.get("tool_call_limit"),
            "model_request_limit": config.get("model_request_limit"),
        },
        "code_build_identifier": _code_build_identifier(),
        "provenance": {
            "mode": "resume_same_baseline_config",
            "original_artifact": "baseline-real-v1.json",
            "resumed_case_ids": resumed_case_ids,
            "original_case_results_preserved": True,
        },
        "resume_policy": {
            "max_retries_per_case": max_retries,
            "base_delay_seconds": base_delay_seconds,
            "max_delay_seconds": max_delay_seconds,
            "inter_case_delay_seconds": inter_case_delay_seconds,
            "retry_after_respected": True,
            "jitter": True,
        },
        "coverage": {
            "total_cases": len(final_cases),
            "executed": coverage.get("EXECUTED", 0),
            "hitl_blocked": coverage.get("HITL_BLOCKED", 0),
            "provider_failures": coverage.get("EXECUTION_FAILED", 0)
            + coverage.get("EXECUTION_FAILED_PROVIDER_RATE_LIMIT", 0),
            "capability_gaps": coverage.get("NOT_EXECUTABLE_CAPABILITY_GAP", 0),
        },
        "case_execution_metadata": metadata,
        "cases": final_cases,
        "deterministic_scoreboard": scoreboard,
        "analysis": analysis,
        "runtime_health": _runtime_health_report(
            str(config["base_url"]),
            owner_token,
        ),
        "semantic_grading": "NOT_RUN",
    }
    canonical = json.dumps(
        artifact,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    artifact["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
    return artifact


def build_scoreboard(
    cases: list[dict[str, Any]],
    traces: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    # Lazy import: eval_grading imports from eval_metrics, so a top-level
    # import here would close an import cycle (regression guard for Wave-3).
    from eval_grading import deterministic_grade, validate_evaluation_trace

    results = [
        (
            validate_evaluation_trace(traces[case["id"]]),
            deterministic_grade(case, traces[case["id"]]),
        )[1]
        for case in cases
        if case["id"] in traces
    ]
    missing = [case["id"] for case in cases if case["id"] not in traces]
    passed = sum(result["deterministic_status"] == "PASS" for result in results)
    category_counts = Counter(case["category"] for case in cases)
    provenance_counts = Counter(case["provenance"] for case in cases)
    return {
        "status": "PASS" if results and not missing and passed == len(results) else "INCOMPLETE",
        "execution_status": "COMPLETE" if not missing else "INCOMPLETE",
        "quality_status": (
            "PASS"
            if results and not missing and passed == len(results)
            else "NOT_DETERMINED"
            if any(result["deterministic_status"] == "NOT_DETERMINED" for result in results)
            else "FAIL"
            if results and not missing
            else "NOT_DETERMINED"
        ),
        "quality_eligible": provenance_counts.get("real_case", 0) > 0,
        "case_count": len(cases),
        "traced_case_count": len(results),
        "category_counts": dict(sorted(category_counts.items())),
        "provenance_counts": dict(sorted(provenance_counts.items())),
        "missing_case_ids": missing,
        "deterministic_pass_rate": round(passed / len(results), 4) if results else None,
        "semantic_grading": "NOT_RUN",
        "semantic_rubric": SEMANTIC_RUBRIC,
        "results": results,
    }


def build_quality_scoreboard(
    cases: list[dict[str, Any]],
    traces: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    real_cases = [case for case in cases if case["provenance"] == "real_case"]
    if not real_cases:
        return {
            "status": "REAL_CASE_DATA_REQUIRED",
            "quality_eligible": False,
            "real_case_count": 0,
            "semantic_grading": "NOT_RUN",
        }
    report = build_scoreboard(real_cases, traces)
    report["quality_eligible"] = True
    return report


def compare_scoreboards(
    baseline_a: dict[str, Any],
    baseline_b: dict[str, Any],
) -> dict[str, Any]:
    if not baseline_a.get("quality_eligible") or not baseline_b.get("quality_eligible"):
        return {
            "status": "REAL_CASE_DATA_REQUIRED",
            "per_case": [],
            "per_category": {},
            "semantic_grading": "NOT_RUN",
        }
    results_a = {result["case_id"]: result for result in baseline_a.get("results", [])}
    results_b = {result["case_id"]: result for result in baseline_b.get("results", [])}
    case_ids = sorted(set(results_a) | set(results_b))
    per_case: list[dict[str, Any]] = []
    for case_id in case_ids:
        before = results_a.get(case_id)
        after = results_b.get(case_id)
        before_score = (
            1 if before and before["deterministic_status"] == "PASS" else 0
        )
        after_score = 1 if after and after["deterministic_status"] == "PASS" else 0
        per_case.append(
            {
                "case_id": case_id,
                "category": (after or before).get("category"),
                "baseline_a": before_score,
                "baseline_b": after_score,
                "delta": after_score - before_score,
            }
        )
    categories: dict[str, dict[str, Any]] = {}
    for item in per_case:
        category = item["category"] or "uncategorized"
        bucket = categories.setdefault(category, {"baseline_a": [], "baseline_b": []})
        bucket["baseline_a"].append(item["baseline_a"])
        bucket["baseline_b"].append(item["baseline_b"])
    category_comparison = {
        category: {
            "baseline_a_pass_rate": round(sum(values["baseline_a"]) / len(values["baseline_a"]), 4),
            "baseline_b_pass_rate": round(sum(values["baseline_b"]) / len(values["baseline_b"]), 4),
            "delta": round(
                sum(values["baseline_b"]) / len(values["baseline_b"])
                - sum(values["baseline_a"]) / len(values["baseline_a"]),
                4,
            ),
        }
        for category, values in sorted(categories.items())
    }
    return {
        "status": "COMPARABLE" if per_case else "INCOMPLETE",
        "per_case": per_case,
        "per_category": category_comparison,
        "semantic_grading": "NOT_RUN",
    }


