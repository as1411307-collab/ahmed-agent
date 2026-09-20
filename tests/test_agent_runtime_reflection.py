from __future__ import annotations

import asyncio
import unittest

from pydantic_ai import ModelRetry

from agent_consts import AgentDeps
from agent_runtime import (
    _InFlightCall,
    _academic_search_should_reflect,
    _clear_tool_failure_note,
    _mark_call_active,
    _mark_call_done,
    _reflect_on_tool_failure,
    _tool_call_signature,
)


class ReflectOnToolFailureTests(unittest.TestCase):
    def test_first_failure_asks_the_model_to_act_not_just_narrate(self) -> None:
        deps = AgentDeps(tool_failure_notes=[])
        sig = _tool_call_signature(query="q")
        retry = _reflect_on_tool_failure(deps, "web_search", "ValueError", sig)
        self.assertIsInstance(retry, ModelRetry)
        self.assertIn("web_search", retry.message)
        self.assertIn("ValueError", retry.message)
        # A bare one-sentence diagnosis would become the run's final `str`
        # output and end the turn -- the message must push the model toward
        # another tool call, not toward a standalone text reply.
        self.assertIn("Do not reply with text yet", retry.message)
        self.assertIn("call a tool again", retry.message)

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
            deps, "search_my_files", "TAVILY_NOT_CONFIGURED", sig
        )
        self.assertIn("TAVILY_NOT_CONFIGURED", retry.message)

    def test_records_a_note_for_the_first_failure(self) -> None:
        deps = AgentDeps(tool_failure_notes=[])
        sig = _tool_call_signature(query="q")
        _reflect_on_tool_failure(deps, "web_search", "ValueError", sig)
        self.assertEqual(len(deps.tool_failure_notes), 1)
        self.assertEqual(deps.tool_failure_notes[0]["tool"], "web_search")
        self.assertEqual(deps.tool_failure_notes[0]["attempt"], 1)
        self.assertEqual(deps.tool_failure_notes[0]["reason"], "ValueError")

    def test_second_failure_of_the_identical_call_escalates(self) -> None:
        deps = AgentDeps(tool_failure_notes=[])
        sig = _tool_call_signature(query="same query", mode="FAST")
        _reflect_on_tool_failure(deps, "web_search", "ValueError", sig)
        retry = _reflect_on_tool_failure(deps, "web_search", "ValueError", sig)
        self.assertIn("failed 2 times", retry.message)
        self.assertIn("Do not repeat it unchanged again", retry.message)
        self.assertEqual(len(deps.tool_failure_notes), 2)
        self.assertEqual(deps.tool_failure_notes[1]["attempt"], 2)

    def test_a_different_call_to_the_same_tool_does_not_escalate(self) -> None:
        # Two distinct web_search queries failing must not be misread as the
        # same call retried twice -- each is its own first failure.
        deps = AgentDeps(tool_failure_notes=[])
        sig_a = _tool_call_signature(query="first query", mode="FAST")
        sig_b = _tool_call_signature(query="second query", mode="FAST")
        _reflect_on_tool_failure(deps, "web_search", "ValueError", sig_a)
        retry = _reflect_on_tool_failure(deps, "web_search", "ValueError", sig_b)
        self.assertIn("Do not reply with text yet", retry.message)

    def test_that_same_calls_own_success_clears_its_own_note(self) -> None:
        deps = AgentDeps(tool_failure_notes=[])
        sig = _tool_call_signature(query="a")
        _reflect_on_tool_failure(deps, "web_search", "ValueError", sig)
        _clear_tool_failure_note(deps, "web_search", sig)
        self.assertEqual(deps.tool_failure_notes, [])
        # A later failure of the exact same call reads as fresh, not a
        # continuation of the earlier resolved failure.
        retry = _reflect_on_tool_failure(deps, "web_search", "ValueError", sig)
        self.assertIn("Do not reply with text yet", retry.message)

    def test_an_unrelated_calls_success_never_touches_a_different_keys_note(
        self,
    ) -> None:
        # This is the race-free property the whole design rests on:
        # PydanticAI's default 'graceful' end strategy runs function tools
        # from the same model-response round concurrently, so a different
        # call's success and this call's failure can be dispatched together
        # with no guaranteed completion order. Because each key only ever
        # clears its own notes, that race is structurally impossible --
        # clearing key B can never race with, or accidentally erase,
        # key A's standing failure, regardless of which finishes first.
        deps = AgentDeps(tool_failure_notes=[])
        sig_a = _tool_call_signature(query="a")
        sig_b = _tool_call_signature(query="b")
        _reflect_on_tool_failure(deps, "web_search", "ValueError", sig_a)
        _clear_tool_failure_note(deps, "web_search", sig_b)  # an unrelated call succeeded
        self.assertEqual(len(deps.tool_failure_notes), 1)
        retry = _reflect_on_tool_failure(deps, "web_search", "ValueError", sig_a)
        self.assertIn("failed 2 times", retry.message)

    def test_different_tools_are_tracked_independently(self) -> None:
        deps = AgentDeps(tool_failure_notes=[])
        sig = _tool_call_signature(query="q")
        _reflect_on_tool_failure(deps, "web_search", "ValueError", sig)
        retry = _reflect_on_tool_failure(deps, "github_search", "ValueError", sig)
        self.assertIn("Do not reply with text yet", retry.message)
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
        retry = _reflect_on_tool_failure(deps, "web_search", "ValueError", sig)
        self.assertIsInstance(retry, ModelRetry)
        _clear_tool_failure_note(deps, "web_search", sig)  # must not crash either

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
            guidance="retry with the reason text UNCHANGED",
            escalated_guidance="do not change the reason text",
        )
        self.assertIn("Do not change the reason text", escalated.message)

    def test_concurrent_failure_never_touches_shared_history(self) -> None:
        # concurrent=True is how a still-active identical sibling call
        # (_mark_call_done observing it) tells this call not to race it: the
        # shared list must be left completely untouched, and the message
        # must still read as a plain first failure regardless of what a
        # sibling is doing.
        deps = AgentDeps(tool_failure_notes=[{"tool": "web_search", "key": "x", "reason": "prior", "attempt": 1}])
        sig = _tool_call_signature(query="q")
        retry = _reflect_on_tool_failure(
            deps, "web_search", "ValueError", sig, concurrent=True
        )
        self.assertIn("Do not reply with text yet", retry.message)
        self.assertNotIn("failed 2 times", retry.message)
        self.assertEqual(len(deps.tool_failure_notes), 1)
        self.assertEqual(deps.tool_failure_notes[0]["reason"], "prior")

    def test_concurrent_clear_never_touches_shared_history(self) -> None:
        deps = AgentDeps(tool_failure_notes=[])
        sig = _tool_call_signature(query="q")
        _reflect_on_tool_failure(deps, "web_search", "ValueError", sig)
        _clear_tool_failure_note(deps, "web_search", sig, concurrent=True)
        # The clear was a no-op because it was told a sibling was still
        # in flight -- the earlier failure note must survive untouched.
        self.assertEqual(len(deps.tool_failure_notes), 1)


class InFlightCallTrackingTests(unittest.TestCase):
    def test_a_solo_call_is_never_reported_as_concurrent(self) -> None:
        deps = AgentDeps(tool_calls_in_flight={})
        sig = _tool_call_signature(query="q")
        _mark_call_active(deps, "web_search", sig)
        had_sibling = _mark_call_done(deps, "web_search", sig)
        self.assertFalse(had_sibling)
        self.assertEqual(deps.tool_calls_in_flight, {})

    def test_sequential_retries_are_never_reported_as_concurrent(self) -> None:
        # The mainline case this whole feature targets: a call fails, fully
        # completes, and only then does the model retry it. There must be
        # no window where both attempts are counted as in flight together.
        deps = AgentDeps(tool_calls_in_flight={})
        sig = _tool_call_signature(query="q")
        _mark_call_active(deps, "web_search", sig)
        self.assertFalse(_mark_call_done(deps, "web_search", sig))
        _mark_call_active(deps, "web_search", sig)
        self.assertFalse(_mark_call_done(deps, "web_search", sig))

    def test_whichever_of_two_identical_calls_finishes_last_sees_no_sibling(
        self,
    ) -> None:
        # This is the exact scenario Codex flagged: two calls dispatched
        # with the identical (tool, call_signature) key, running
        # concurrently in the same model-response round. Whichever finishes
        # first must be told a sibling is active (so it skips shared
        # history); whichever finishes last must see the coast is clear --
        # and this must hold regardless of which one that happens to be.
        deps = AgentDeps(tool_calls_in_flight={})
        sig = _tool_call_signature(query="dup")
        _mark_call_active(deps, "web_search", sig)
        _mark_call_active(deps, "web_search", sig)
        # Order A finishes, then B.
        self.assertTrue(_mark_call_done(deps, "web_search", sig))
        self.assertFalse(_mark_call_done(deps, "web_search", sig))

        deps2 = AgentDeps(tool_calls_in_flight={})
        _mark_call_active(deps2, "web_search", sig)
        _mark_call_active(deps2, "web_search", sig)
        # Reverse completion order -- the outcome pattern is identical.
        self.assertTrue(_mark_call_done(deps2, "web_search", sig))
        self.assertFalse(_mark_call_done(deps2, "web_search", sig))

    def test_duplicate_concurrent_calls_never_corrupt_the_note_list_either_order(
        self,
    ) -> None:
        # End-to-end proof for the exact scenario Codex flagged: two calls
        # sharing the identical (tool, call_signature) key, one failing and
        # one succeeding, dispatched concurrently in the same round. The
        # fix does not force a single "right" answer -- whichever outcome
        # resolves last (no sibling left in flight) is authoritative, which
        # is an ordinary, sanctioned last-write-wins rule. What it
        # eliminates is the pre-fix bug: an EARLIER-resolving call's own
        # mutation getting silently corrupted or double-read by a sibling
        # that hadn't resolved yet. So in both orders the result must be
        # well-formed -- exactly zero or one note, matching whichever
        # operation actually ran last -- never two notes, never a note left
        # over from the call that lost the race.
        def run(fail_first: bool) -> list[dict[str, object]]:
            deps = AgentDeps(tool_failure_notes=[], tool_calls_in_flight={})
            sig = _tool_call_signature(query="dup")
            _mark_call_active(deps, "web_search", sig)
            _mark_call_active(deps, "web_search", sig)
            if fail_first:
                concurrent = _mark_call_done(deps, "web_search", sig)
                _reflect_on_tool_failure(
                    deps, "web_search", "ValueError", sig, concurrent=concurrent
                )
                concurrent = _mark_call_done(deps, "web_search", sig)
                _clear_tool_failure_note(
                    deps, "web_search", sig, concurrent=concurrent
                )
            else:
                concurrent = _mark_call_done(deps, "web_search", sig)
                _clear_tool_failure_note(
                    deps, "web_search", sig, concurrent=concurrent
                )
                concurrent = _mark_call_done(deps, "web_search", sig)
                _reflect_on_tool_failure(
                    deps, "web_search", "ValueError", sig, concurrent=concurrent
                )
            return deps.tool_failure_notes

        # fail_first=True: the failure resolves while its sibling is still
        # active, so it skips the shared list untouched; the success then
        # resolves last (alone) and its clear -- a no-op on an already-empty
        # list -- leaves zero notes: a clean, well-formed "resolved" state.
        self.assertEqual(run(fail_first=True), [])
        # fail_first=False: the success resolves while its sibling is still
        # active, so it skips clearing; the failure then resolves last
        # (alone) and is the one that actually records itself -- exactly
        # one note, never two, never a stale leftover from either call.
        failed_last = run(fail_first=False)
        self.assertEqual(len(failed_last), 1)
        self.assertEqual(failed_last[0]["attempt"], 1)


class InFlightCallClassTests(unittest.IsolatedAsyncioTestCase):
    def test_done_is_idempotent_and_matches_the_module_functions(self) -> None:
        deps = AgentDeps(tool_calls_in_flight={})
        sig = _tool_call_signature(query="q")
        call = _InFlightCall(deps, "web_search", sig)
        self.assertFalse(call.done())
        # A second call to .done() must be a harmless no-op, never a second
        # real decrement -- exactly what lets normal-path code call .done()
        # once for its concurrent flag while ensure_retired() in a finally
        # block stays safe to call unconditionally afterwards.
        self.assertFalse(call.done())
        self.assertEqual(deps.tool_calls_in_flight, {})

    def test_ensure_retired_is_a_no_op_once_done_already_ran(self) -> None:
        deps = AgentDeps(tool_calls_in_flight={})
        sig = _tool_call_signature(query="q")
        other_sig = _tool_call_signature(query="other")
        call = _InFlightCall(deps, "web_search", sig)
        call.done()
        # A concurrent, unrelated call must still see no sibling after the
        # first call's normal completion + ensure_retired() sequence.
        call.ensure_retired()
        sibling = _InFlightCall(deps, "web_search", other_sig)
        self.assertFalse(sibling.done())

    async def test_cancellation_before_done_still_retires_via_ensure_retired(
        self,
    ) -> None:
        # This is the exact bug Codex flagged: PydanticAI's tool_timeout
        # cancels a tool coroutine via CancelledError (a BaseException),
        # which bypasses `except Exception` and would skip a bare
        # _mark_call_done() call sited only in normal try/except branches --
        # permanently leaking an in-flight slot. The fix wraps the call body
        # in `finally: call.ensure_retired()`, which must fire even when the
        # coroutine is cancelled before reaching its own `.done()` call.
        deps = AgentDeps(tool_calls_in_flight={})
        sig = _tool_call_signature(query="q")

        async def cancellable_call() -> None:
            call = _InFlightCall(deps, "web_search", sig)
            try:
                await asyncio.Event().wait()  # never resolves on its own
                call.done()  # pragma: no cover -- unreachable once cancelled
            finally:
                call.ensure_retired()

        task = asyncio.ensure_future(cancellable_call())
        await asyncio.sleep(0)  # let it reach the await and register active
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        # The slot must not have leaked: a fresh sequential call for the
        # identical key must see no sibling, exactly as if the timed-out
        # call had never happened.
        retry = _InFlightCall(deps, "web_search", sig)
        self.assertFalse(retry.done())
        self.assertEqual(deps.tool_calls_in_flight, {})


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
