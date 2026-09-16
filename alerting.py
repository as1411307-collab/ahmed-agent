from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


ALERT_POLICY: dict[str, dict[str, Any]] = {
    "service_health": {
        "metric": "service_health",
        "minimum_sample_size": 1,
        "evaluation_window_hours": 24,
        "warning_threshold": 1,
        "critical_threshold": 1,
        "cooldown_seconds": 900,
    },
    "audit_integrity": {
        "metric": "audit_integrity",
        "minimum_sample_size": 1,
        "evaluation_window_hours": 24,
        "warning_threshold": 1,
        "critical_threshold": 1,
        "cooldown_seconds": 900,
    },
    "orphan_lease_anomaly": {
        "metric": "orphaned_runs_or_lease_expirations",
        "minimum_sample_size": 1,
        "evaluation_window_hours": 24,
        "warning_threshold": 1,
        "critical_threshold": 3,
        "cooldown_seconds": 900,
    },
    "recovery_failure": {
        "metric": "recovery_failures",
        "minimum_sample_size": 1,
        "evaluation_window_hours": 24,
        "warning_threshold": 1,
        "critical_threshold": 2,
        "cooldown_seconds": 900,
    },
    "failure_rate": {
        "metric": "failed_runs_rate",
        "minimum_sample_size": 5,
        "evaluation_window_hours": 24,
        "warning_threshold": 0.40,
        "critical_threshold": 0.75,
        "cooldown_seconds": 1800,
    },
}

_SEVERITY_ORDER = {"healthy": 0, "warning": 1, "critical": 2}


def _severity_for_value(
    *,
    value: float,
    minimum_sample_size: int,
    sample_size: int,
    warning_threshold: float,
    critical_threshold: float,
) -> str:
    if sample_size < minimum_sample_size or value < warning_threshold:
        return "healthy"
    if value >= critical_threshold:
        return "critical"
    return "warning"


def evaluate_runtime_rules(
    *,
    metrics: dict[str, Any],
    service_healthy: bool,
    audit_healthy: bool,
) -> list[dict[str, Any]]:
    total_runs = int(metrics.get("total_runs", 0) or 0)
    failed_runs = int(metrics.get("failed_runs", 0) or 0)
    failure_rate = failed_runs / total_runs if total_runs else 0.0
    orphan_lease_count = max(
        int(metrics.get("orphaned_runs", 0) or 0),
        int(metrics.get("lease_expirations", 0) or 0),
    )
    recovery_failures = int(metrics.get("recovery_failures", 0) or 0)

    values = {
        "service_health": (
            0 if service_healthy else 1,
            1,
            {"service_healthy": service_healthy},
        ),
        "audit_integrity": (
            0 if audit_healthy else 1,
            1,
            {"audit_healthy": audit_healthy},
        ),
        "orphan_lease_anomaly": (
            orphan_lease_count,
            orphan_lease_count,
            {
                "orphaned_runs": int(metrics.get("orphaned_runs", 0) or 0),
                "lease_expirations": int(metrics.get("lease_expirations", 0) or 0),
            },
        ),
        "recovery_failure": (
            recovery_failures,
            recovery_failures,
            {
                "recovery_failures": recovery_failures,
                "recovery_failed_runs": int(
                    metrics.get("recovery_failed_runs", 0) or 0
                ),
            },
        ),
        "failure_rate": (
            failure_rate,
            total_runs,
            {
                "total_runs": total_runs,
                "failed_runs": failed_runs,
                "failure_rate": round(failure_rate, 4),
            },
        ),
    }

    evaluations: list[dict[str, Any]] = []
    for rule, policy in ALERT_POLICY.items():
        value, sample_size, evidence = values[rule]
        evaluations.append(
            {
                "rule": rule,
                "severity": _severity_for_value(
                    value=float(value),
                    sample_size=sample_size,
                    minimum_sample_size=int(policy["minimum_sample_size"]),
                    warning_threshold=float(policy["warning_threshold"]),
                    critical_threshold=float(policy["critical_threshold"]),
                ),
                "value": value,
                "sample_size": sample_size,
                "evidence": evidence,
                "policy": policy,
            }
        )
    return evaluations


def _as_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def transition_alert(
    *,
    evaluation: dict[str, Any],
    current: dict[str, Any] | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    current = current or {
        "status": "healthy",
        "severity": "healthy",
        "last_triggered_at": None,
        "recovered_at": None,
        "suppressed_count": 0,
    }
    current_status = str(current.get("status", "healthy"))
    current_severity = str(current.get("severity", "healthy"))
    next_severity = str(evaluation["severity"])
    event_type: str | None = None
    status = next_severity
    last_triggered_at = _as_datetime(current.get("last_triggered_at"))
    recovered_at = _as_datetime(current.get("recovered_at"))
    suppressed_count = int(current.get("suppressed_count", 0) or 0)

    if next_severity == "healthy":
        if current_status in {"warning", "critical"}:
            event_type = "alert_recovered"
            recovered_at = now
        status = "healthy"
    elif current_status in {"healthy", "recovered"}:
        event_type = "alert_opened"
        last_triggered_at = now
        recovered_at = None
    elif _SEVERITY_ORDER.get(next_severity, 0) > _SEVERITY_ORDER.get(
        current_severity, 0
    ):
        event_type = "alert_escalated"
        last_triggered_at = now
        recovered_at = None
    else:
        cooldown = timedelta(seconds=int(evaluation["policy"]["cooldown_seconds"]))
        if last_triggered_at is None or now - last_triggered_at >= cooldown:
            event_type = "alert_suppressed"
            last_triggered_at = now
            suppressed_count += 1

    return {
        "rule": evaluation["rule"],
        "status": status,
        "severity": next_severity if next_severity != "healthy" else "healthy",
        "event_type": event_type,
        "last_triggered_at": last_triggered_at,
        "recovered_at": recovered_at,
        "suppressed_count": suppressed_count,
        "last_evidence": evaluation["evidence"],
        "last_value": evaluation["value"],
        "sample_size": evaluation["sample_size"],
        "evaluated_at": now,
    }