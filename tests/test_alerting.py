from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from alerting import evaluate_runtime_rules, transition_alert


class AlertingTests(unittest.TestCase):
    def test_clean_baseline_has_no_active_alerts(self) -> None:
        evaluations = evaluate_runtime_rules(
            metrics={
                "total_runs": 11,
                "failed_runs": 0,
                "orphaned_runs": 0,
                "lease_expirations": 0,
                "recovery_failures": 0,
            },
            service_healthy=True,
            audit_healthy=True,
        )
        self.assertTrue(all(item["severity"] == "healthy" for item in evaluations))

    def test_failure_rate_requires_minimum_sample_size(self) -> None:
        evaluations = evaluate_runtime_rules(
            metrics={
                "total_runs": 2,
                "failed_runs": 2,
                "orphaned_runs": 0,
                "lease_expirations": 0,
                "recovery_failures": 0,
            },
            service_healthy=True,
            audit_healthy=True,
        )
        failure_rate = next(item for item in evaluations if item["rule"] == "failure_rate")
        self.assertEqual(failure_rate["severity"], "healthy")

    def test_alert_opens_suppresses_and_recovers(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        evaluation = {
            "rule": "orphan_lease_anomaly",
            "severity": "warning",
            "value": 1,
            "sample_size": 1,
            "evidence": {"orphaned_runs": 1},
            "policy": {"cooldown_seconds": 900},
        }
        opened = transition_alert(evaluation=evaluation, current=None, now=now)
        self.assertEqual(opened["event_type"], "alert_opened")
        suppressed = transition_alert(
            evaluation=evaluation,
            current=opened,
            now=now + timedelta(seconds=60),
        )
        self.assertIsNone(suppressed["event_type"])
        recovered = transition_alert(
            evaluation={**evaluation, "severity": "healthy"},
            current=suppressed,
            now=now + timedelta(seconds=120),
        )
        self.assertEqual(recovered["event_type"], "alert_recovered")
        self.assertEqual(recovered["status"], "healthy")
        self.assertEqual(recovered["severity"], "healthy")


if __name__ == "__main__":
    unittest.main()