from __future__ import annotations

import asyncio
import unittest

from academic_benchmark import load_contract_fixtures, run_contract_benchmark


class AcademicBenchmarkTests(unittest.TestCase):
    def test_contract_dataset_has_fixed_shape_without_raw_payloads(self) -> None:
        fixtures = load_contract_fixtures()
        self.assertEqual(len(fixtures), 16)
        for fixture in fixtures:
            serialized = str(fixture).casefold()
            self.assertNotIn("abstract_inverted_index", serialized)
            self.assertNotIn('"message"', serialized)
            self.assertNotIn('"raw"', serialized)

    def test_deterministic_contract_benchmark_passes(self) -> None:
        report = asyncio.run(run_contract_benchmark())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["fixture_count"], 16)
        self.assertEqual(report["deterministic_pass_rate"], 1.0)
        self.assertEqual(report["routing_accuracy"], 1.0)
        self.assertEqual(report["no_hallucination_contract"], "PASS")
        self.assertEqual(report["privacy"], "PASS")
        self.assertEqual(report["average_calls_per_fixture"], 1.44)


if __name__ == "__main__":
    unittest.main()