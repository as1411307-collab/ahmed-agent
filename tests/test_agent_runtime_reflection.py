from __future__ import annotations

import unittest

from pydantic_ai import ModelRetry

from agent_consts import AgentDeps
from agent_runtime import (
    _academic_search_should_reflect,
    _clear_tool_failure_note,
    _reflect_on_tool_failure,
    _tool_call_signature,
)


class ReflectOnToolFailureTests(unittest.TestCase):
    def test_first_failure_asks_the_model_to_act_not_just_narrate(self) -> None:
        deps = AgentDeps(tool_failure_notes=[])
        sig = _tool_call_signature(query="q")
        retry = _reflect_on_tool_failure(deps, "web_search", ValueError("boom"), sig)
        self.assertIsInstance(retry, ModelRetry)
        self.assertIn("web_search", retry.message)
        self.assertIn("ValueError", retry.message)
        # A bare one-sentence diagnosis would become the run's final `str`
        # output and end the turn -- the message must push the model toward
        # another tool call, not toward a standalone text reply.
        self.assertIn("Do not reply with text yet", retry.message)
        self.assertIn("call a tool again", retry.message)
        self.assertNotIn("in a row", retry.message)

    def test_never_includes_the_raw_exception_message(self) -> None:
        # Only the exception's type name may cross the model-provider
        # boundary -- str(error) can carry a credential-bearing URL, a
        # provider response body, a database detail, or a local path.
        deps = AgentDeps(tool_failure_notes=[])
        sig = _tool_call_signature(query="q")
        secret = "postgres://user:hunter2@db.internal:5432/prod?token=abc123"
        retry = _reflect_on_tool_failure(
            deps, "search_my_files", RuntimeError(secret), sig
        )
        self.assertNotIn(secret, retry.message)
        self.assertNotIn("hunter2", retry.message)
        self.assertIn("RuntimeError", retry.message)
        self.assertNotIn(secret, str(deps.tool_failure_notes))

    def test_records_a_note_for_the_first_failure(self) -> None:
        deps = AgentDeps(tool_failure_notes=[])
        sig = _tool_call_signature(query="q")
        _reflect_on_tool_failure(deps, "web_search", ValueError("boom"), sig)
        self.assertEqual(len(deps.tool_failure_notes), 1)
        self.assertEqual(deps.tool_failure_notes[0]["tool"], "web_search")
        self.assertEqual(deps.tool_failure_notes[0]["attempt"], 1)
        self.assertEqual(deps.tool_failure_notes[0]["reason"], "ValueError")

    def test_second_failure_of_the_identical_call_escalates(self) -> None:
        deps = AgentDeps(tool_failure_notes=[])
        sig = _tool_call_signature(query="same query", mode="FAST")
        _reflect_on_tool_failure(deps, "web_search", ValueError("first"), sig)
        retry = _reflect_on_tool_failure(deps, "web_search", ValueError("second"), sig)
        self.assertIn("failed 2 times in a row", retry.message)
        self.assertIn("Do not repeat it unchanged again", retry.message)
        self.assertEqual(len(deps.tool_failure_notes), 2)
        self.assertEqual(deps.tool_failure_notes[1]["attempt"], 2)

    def test_a_different_call_to_the_same_tool_does_not_escalate(self) -> None:
        # Two distinct web_search queries failing must not be misread as the
        # same call retried twice -- each is its own first failure.
        deps = AgentDeps(tool_failure_notes=[])
        sig_a = _tool_call_signature(query="first query", mode="FAST")
        sig_b = _tool_call_signature(query="second query", mode="FAST")
        _reflect_on_tool_failure(deps, "web_search", ValueError("x"), sig_a)
        retry = _reflect_on_tool_failure(deps, "web_search", ValueError("y"), sig_b)
        self.assertIn("Do not reply with text yet", retry.message)
        self.assertNotIn("in a row", retry.message)

    def test_success_clears_the_note_so_a_later_failure_starts_fresh(self) -> None:
        # Fails once, "succeeds" (cleared), then fails again with the exact
        # same arguments much later -- this must read as a first failure
        # again, not a false escalation carried across an intervening
        # success.
        deps = AgentDeps(tool_failure_notes=[])
        sig = _tool_call_signature(query="q", mode="FAST")
        _reflect_on_tool_failure(deps, "web_search", ValueError("first"), sig)
        _clear_tool_failure_note(deps, "web_search", sig)
        retry = _reflect_on_tool_failure(deps, "web_search", ValueError("later"), sig)
        self.assertNotIn("in a row", retry.message)
        self.assertEqual(len(deps.tool_failure_notes), 1)

    def test_different_tools_are_tracked_independently(self) -> None:
        deps = AgentDeps(tool_failure_notes=[])
        sig = _tool_call_signature(query="q")
        _reflect_on_tool_failure(deps, "web_search", ValueError("first"), sig)
        retry = _reflect_on_tool_failure(deps, "github_search", ValueError("first"), sig)
        self.assertNotIn("in a row", retry.message)
        self.assertEqual(
            [note["tool"] for note in deps.tool_failure_notes],
            ["web_search", "github_search"],
        )

    def test_exception_with_no_message_still_produces_a_readable_reason(self) -> None:
        deps = AgentDeps(tool_failure_notes=[])
        sig = _tool_call_signature(query="q")
        retry = _reflect_on_tool_failure(deps, "search_my_files", RuntimeError(), sig)
        self.assertIn("RuntimeError", retry.message)

    def test_tolerates_a_deps_with_no_failure_memory(self) -> None:
        # AgentDeps.tool_failure_notes defaults to None outside run_ahmed's own
        # construction (e.g. a caller that never wired it up); this must not
        # crash, and simply cannot escalate since nothing is remembered.
        deps = AgentDeps()
        sig = _tool_call_signature(query="q")
        retry = _reflect_on_tool_failure(deps, "web_search", ValueError("boom"), sig)
        self.assertIsInstance(retry, ModelRetry)
        self.assertNotIn("in a row", retry.message)
        _clear_tool_failure_note(deps, "web_search", sig)  # must not crash either


class AcademicSearchShouldReflectTests(unittest.TestCase):
    def test_ok_result_never_reflects(self) -> None:
        self.assertFalse(_academic_search_should_reflect({"ok": True, "results": []}))

    def test_argument_validation_errors_are_retryable(self) -> None:
        for code in (
            "query_required",
            "query_too_long",
            "invalid_intent",
            "invalid_max_results",
        ):
            with self.subTest(code=code):
                self.assertTrue(
                    _academic_search_should_reflect(
                        {"ok": False, "error": code, "results": []}
                    )
                )

    def test_invalid_doi_is_a_terminal_answer_not_a_retryable_failure(self) -> None:
        # Real shape from academic_search.py's DOIResolver.resolve().
        result = {
            "ok": False,
            "intent": "doi",
            "error": "invalid_doi",
            "original_doi": "not-a-doi",
            "results": [],
        }
        self.assertFalse(_academic_search_should_reflect(result))

    def test_doi_not_found_with_a_landing_url_is_terminal_not_retryable(self) -> None:
        # Real shape: a valid, registered DOI with no metadata record --
        # discarding this for a retry prompt would lose the landing_url and
        # make the model wrongly claim the capability is broken.
        result = {
            "ok": False,
            "intent": "doi",
            "registration_agency": "crossref",
            "landing_url": "https://doi.org/10.1234/example",
            "metadata_status": "NOT_FOUND",
            "results": [],
        }
        self.assertFalse(_academic_search_should_reflect(result))


class ToolCallSignatureTests(unittest.TestCase):
    def test_same_arguments_in_any_order_produce_the_same_signature(self) -> None:
        self.assertEqual(
            _tool_call_signature(query="q", mode="FAST"),
            _tool_call_signature(mode="FAST", query="q"),
        )

    def test_different_argument_values_produce_different_signatures(self) -> None:
        self.assertNotEqual(
            _tool_call_signature(query="a"),
            _tool_call_signature(query="b"),
        )


if __name__ == "__main__":
    unittest.main()
