from __future__ import annotations

import unittest

from pydantic_ai import ModelRetry

from agent_consts import AgentDeps
from agent_runtime import (
    _academic_search_should_reflect,
    _clear_tool_failure_streak,
    _reflect_on_tool_failure,
    _tool_call_signature,
)


class ReflectOnToolFailureTests(unittest.TestCase):
    def test_first_failure_asks_the_model_to_act_not_just_narrate(self) -> None:
        deps = AgentDeps(tool_failure_notes=[])
        sig = _tool_call_signature(query="q")
        retry = _reflect_on_tool_failure(deps, "web_search", "ValueError", sig, 1)
        self.assertIsInstance(retry, ModelRetry)
        self.assertIn("web_search", retry.message)
        self.assertIn("ValueError", retry.message)
        # A bare one-sentence diagnosis would become the run's final `str`
        # output and end the turn -- the message must push the model toward
        # another tool call, not toward a standalone text reply.
        self.assertIn("Do not reply with text yet", retry.message)
        self.assertIn("call a tool again", retry.message)
        self.assertNotIn("in a row", retry.message)

    def test_reason_is_used_verbatim_since_callers_must_pre_sanitize_it(self) -> None:
        # _reflect_on_tool_failure trusts its caller: an exception call site
        # must pass type(error).__name__ (never str(error), which can carry
        # a credential-bearing URL, provider response body, database
        # detail, or local path), and an in-band {"ok": False} call site
        # must pass only the tool's own bounded, author-controlled code.
        # This test documents that contract at the boundary the helper
        # actually controls: whatever safe string comes in appears, and
        # nothing else is invented or stripped.
        deps = AgentDeps(tool_failure_notes=[])
        sig = _tool_call_signature(query="q")
        retry = _reflect_on_tool_failure(
            deps, "search_my_files", "TAVILY_NOT_CONFIGURED", sig, 1
        )
        self.assertIn("TAVILY_NOT_CONFIGURED", retry.message)

    def test_records_a_note_for_the_first_failure(self) -> None:
        deps = AgentDeps(tool_failure_notes=[])
        sig = _tool_call_signature(query="q")
        _reflect_on_tool_failure(deps, "web_search", "ValueError", sig, 1)
        self.assertEqual(len(deps.tool_failure_notes), 1)
        self.assertEqual(deps.tool_failure_notes[0]["tool"], "web_search")
        self.assertEqual(deps.tool_failure_notes[0]["attempt"], 1)
        self.assertEqual(deps.tool_failure_notes[0]["reason"], "ValueError")
        self.assertEqual(deps.tool_failure_notes[0]["run_step"], 1)

    def test_second_failure_of_the_identical_call_escalates(self) -> None:
        deps = AgentDeps(tool_failure_notes=[])
        sig = _tool_call_signature(query="same query", mode="FAST")
        _reflect_on_tool_failure(deps, "web_search", "ValueError", sig, 1)
        retry = _reflect_on_tool_failure(deps, "web_search", "ValueError", sig, 2)
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
        _reflect_on_tool_failure(deps, "web_search", "ValueError", sig_a, 1)
        retry = _reflect_on_tool_failure(deps, "web_search", "ValueError", sig_b, 1)
        self.assertIn("Do not reply with text yet", retry.message)
        self.assertNotIn("in a row", retry.message)

    def test_a_later_rounds_success_clears_an_earlier_rounds_failure(self) -> None:
        # Sequential case: A fails in round 1, something succeeds in round
        # 2, and A is retried (still round 2 or later) and fails again --
        # this must read as a fresh first failure, not a continuation of
        # round 1's streak.
        deps = AgentDeps(tool_failure_notes=[])
        sig_a = _tool_call_signature(query="a")
        _reflect_on_tool_failure(deps, "web_search", "ValueError", sig_a, run_step=1)
        _clear_tool_failure_streak(deps, run_step=2)  # something else succeeded in round 2
        self.assertEqual(deps.tool_failure_notes, [])
        retry = _reflect_on_tool_failure(deps, "web_search", "ValueError", sig_a, run_step=2)
        self.assertNotIn("in a row", retry.message)

    def test_a_concurrent_siblings_success_does_not_erase_a_same_round_failure(
        self,
    ) -> None:
        # PydanticAI's default 'graceful' end strategy runs function tools
        # from the *same* model-response round concurrently. If call A
        # fails and sibling call B succeeds in that same round (run_step),
        # B's success must NOT erase A's failure -- A was never resolved,
        # it just happened to share a round with something that worked.
        deps = AgentDeps(tool_failure_notes=[])
        sig_a = _tool_call_signature(query="a")
        _reflect_on_tool_failure(deps, "web_search", "ValueError", sig_a, run_step=1)
        _clear_tool_failure_streak(deps, run_step=1)  # sibling B succeeded, same round
        self.assertEqual(len(deps.tool_failure_notes), 1)
        # A is retried, still failing, in the same or a later round: this is
        # genuinely its second consecutive failure and must escalate.
        retry = _reflect_on_tool_failure(deps, "web_search", "ValueError", sig_a, run_step=1)
        self.assertIn("failed 2 times in a row", retry.message)

    def test_different_tools_are_tracked_independently(self) -> None:
        deps = AgentDeps(tool_failure_notes=[])
        sig = _tool_call_signature(query="q")
        _reflect_on_tool_failure(deps, "web_search", "ValueError", sig, 1)
        retry = _reflect_on_tool_failure(deps, "github_search", "ValueError", sig, 1)
        self.assertNotIn("in a row", retry.message)
        self.assertEqual(
            [note["tool"] for note in deps.tool_failure_notes],
            ["web_search", "github_search"],
        )

    def test_tolerates_a_deps_with_no_failure_memory(self) -> None:
        # AgentDeps.tool_failure_notes defaults to None outside run_ahmed's own
        # construction (e.g. a caller that never wired it up); this must not
        # crash, and simply cannot escalate since nothing is remembered.
        deps = AgentDeps()
        sig = _tool_call_signature(query="q")
        retry = _reflect_on_tool_failure(deps, "web_search", "ValueError", sig, 1)
        self.assertIsInstance(retry, ModelRetry)
        self.assertNotIn("in a row", retry.message)
        _clear_tool_failure_streak(deps, 1)  # must not crash either

    def test_guidance_can_be_overridden_for_an_idempotent_action(self) -> None:
        # test_sensitive_action's idempotency key is derived from its
        # free-form `reason` argument, so the generic "retry with different
        # arguments" advice would defeat the dedup guarantee. It overrides
        # both messages to say the opposite: keep the arguments unchanged.
        deps = AgentDeps(tool_failure_notes=[])
        sig = _tool_call_signature(reason="because")
        retry = _reflect_on_tool_failure(
            deps,
            "test_sensitive_action",
            "RuntimeError",
            sig,
            1,
            guidance="retry with the reason text UNCHANGED",
            escalated_guidance="do not change the reason text",
        )
        self.assertIn("retry with the reason text UNCHANGED", retry.message)
        self.assertNotIn("clearly different arguments", retry.message)
        escalated = _reflect_on_tool_failure(
            deps,
            "test_sensitive_action",
            "RuntimeError",
            sig,
            1,
            guidance="retry with the reason text UNCHANGED",
            escalated_guidance="do not change the reason text",
        )
        self.assertIn("Do not change the reason text", escalated.message)


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
