from __future__ import annotations

import unittest

from pydantic_ai import ModelRetry

from agent_consts import AgentDeps
from agent_runtime import _reflect_on_tool_failure


class ReflectOnToolFailureTests(unittest.TestCase):
    def test_first_failure_asks_for_a_reason_before_retrying(self) -> None:
        deps = AgentDeps(tool_failure_notes=[])
        retry = _reflect_on_tool_failure(deps, "web_search", ValueError("boom"))
        self.assertIsInstance(retry, ModelRetry)
        self.assertIn("web_search", retry.message)
        self.assertIn("ValueError: boom", retry.message)
        self.assertIn("briefly state", retry.message)
        self.assertNotIn("again", retry.message)

    def test_records_a_note_for_the_first_failure(self) -> None:
        deps = AgentDeps(tool_failure_notes=[])
        _reflect_on_tool_failure(deps, "web_search", ValueError("boom"))
        self.assertEqual(len(deps.tool_failure_notes), 1)
        self.assertEqual(deps.tool_failure_notes[0]["tool"], "web_search")
        self.assertEqual(deps.tool_failure_notes[0]["attempt"], 1)

    def test_second_failure_of_the_same_tool_escalates_the_message(self) -> None:
        deps = AgentDeps(tool_failure_notes=[])
        _reflect_on_tool_failure(deps, "web_search", ValueError("first"))
        retry = _reflect_on_tool_failure(deps, "web_search", ValueError("second"))
        self.assertIn("failed again (2 times this turn)", retry.message)
        self.assertIn("Do not repeat the same call unchanged", retry.message)
        self.assertEqual(len(deps.tool_failure_notes), 2)
        self.assertEqual(deps.tool_failure_notes[1]["attempt"], 2)

    def test_different_tools_are_tracked_independently(self) -> None:
        deps = AgentDeps(tool_failure_notes=[])
        _reflect_on_tool_failure(deps, "web_search", ValueError("first"))
        retry = _reflect_on_tool_failure(deps, "github_search", ValueError("first"))
        # github_search's first failure should read as a first failure, not an
        # escalation borrowed from web_search's count.
        self.assertNotIn("again", retry.message)
        self.assertEqual(
            [note["tool"] for note in deps.tool_failure_notes],
            ["web_search", "github_search"],
        )

    def test_exception_with_no_message_still_produces_a_readable_reason(self) -> None:
        deps = AgentDeps(tool_failure_notes=[])
        retry = _reflect_on_tool_failure(deps, "search_my_files", RuntimeError())
        self.assertIn("RuntimeError", retry.message)

    def test_tolerates_a_deps_with_no_failure_memory(self) -> None:
        # AgentDeps.tool_failure_notes defaults to None outside run_ahmed's own
        # construction (e.g. a caller that never wired it up); this must not
        # crash, and simply cannot escalate since nothing is remembered.
        deps = AgentDeps()
        retry = _reflect_on_tool_failure(deps, "web_search", ValueError("boom"))
        self.assertIsInstance(retry, ModelRetry)
        self.assertNotIn("again", retry.message)


if __name__ == "__main__":
    unittest.main()
