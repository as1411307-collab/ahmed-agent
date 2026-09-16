from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from pydantic_ai.exceptions import ModelHTTPError

from pydantic_ai.models.openai import OpenAIChatModel

import agent_core
from agent_core import (
    AgentCoreError,
    EvidenceFirstContext,
    configured_provider_names,
    provider_health,
    run_ahmed,
)


class ModelProviderTests(unittest.TestCase):
    @staticmethod
    def _empty_preflight() -> EvidenceFirstContext:
        return EvidenceFirstContext(
            route=SimpleNamespace(
                required_capabilities=(),
                abstain_if_evidence_missing=False,
            ),
            model_context="",
            evidence_envelopes=(),
            external_sources=(),
        )

    @staticmethod
    def _candidates() -> tuple[SimpleNamespace, SimpleNamespace]:
        return (
            SimpleNamespace(name="gemini", model_name="gemini-test", model=object()),
            SimpleNamespace(name="openai", model_name="openai-test", model=object()),
        )

    def test_transient_gemini_failure_falls_back_to_ready_openai(self) -> None:
        gemini, openai = self._candidates()
        fallback_events: list[tuple[object, ...]] = []

        class FailingGemini:
            async def run(self, *_: object, **__: object) -> None:
                raise ModelHTTPError(
                    429,
                    "gemini-test",
                    {"error": {"code": "RATE_LIMITED", "detail": "do not persist"}},
                )

        class WorkingOpenAI:
            async def run(self, *_: object, **__: object) -> SimpleNamespace:
                return SimpleNamespace(
                    output="fallback response",
                    new_messages=lambda: [],
                )

        async def record_event(*args: object) -> None:
            fallback_events.append(args)

        with (
            patch("agent_core._configured_providers", return_value=[gemini, openai]),
            patch(
                "agent_core._build_agent",
                side_effect=lambda model, _scope: (
                    FailingGemini() if model is gemini.model else WorkingOpenAI()
                ),
            ) as build_agent,
            patch(
                "agent_core.provider_health",
                side_effect=lambda provider: {"provider": provider, "status": "READY"},
            ),
            patch(
                "agent_core.prepare_evidence_first_context",
                new=__import__("unittest").mock.AsyncMock(
                    return_value=self._empty_preflight()
                ),
            ),
        ):
            result = asyncio.run(
                run_ahmed(
                    "answer with a temporary provider failure",
                    provider="gemini",
                    tool_event_recorder=record_event,
                )
            )

        self.assertEqual(result.output, "fallback response")
        self.assertEqual(build_agent.call_count, 2)
        self.assertEqual(len(fallback_events), 1)
        tool_name, status, duration_ms, metadata = fallback_events[0]
        self.assertEqual(tool_name, "model_provider_fallback")
        self.assertEqual(status, "success")
        self.assertEqual(duration_ms, 0)
        self.assertEqual(
            metadata,
            {
                "requested_provider": "gemini",
                "actual_provider": "openai",
                "switch_reason": "RATE_LIMITED",
            },
        )

    def test_rate_limited_requested_provider_is_skipped_for_ready_fallback(self) -> None:
        gemini, openai = self._candidates()
        fallback_events: list[tuple[object, ...]] = []

        class WorkingOpenAI:
            async def run(self, *_: object, **__: object) -> SimpleNamespace:
                return SimpleNamespace(output="ready", new_messages=lambda: [])

        async def record_event(*args: object) -> None:
            fallback_events.append(args)

        with (
            patch("agent_core._configured_providers", return_value=[gemini, openai]),
            patch("agent_core._build_agent", return_value=WorkingOpenAI()) as build_agent,
            patch(
                "agent_core.provider_health",
                side_effect=lambda provider: {
                    "provider": provider,
                    "status": "RATE_LIMITED" if provider == "gemini" else "READY",
                },
            ),
            patch(
                "agent_core.prepare_evidence_first_context",
                new=__import__("unittest").mock.AsyncMock(
                    return_value=self._empty_preflight()
                ),
            ),
        ):
            result = asyncio.run(
                run_ahmed(
                    "use the available model",
                    provider="gemini",
                    tool_event_recorder=record_event,
                )
            )

        self.assertEqual(result.output, "ready")
        build_agent.assert_called_once_with(openai.model, "WEB")
        self.assertEqual(fallback_events[0][3]["switch_reason"], "RATE_LIMITED_PRECHECK")
        self.assertEqual(fallback_events[0][3]["actual_provider"], "openai")

    def test_no_fallback_to_unavailable_provider_and_retry_budget_is_bounded(self) -> None:
        gemini, openai = self._candidates()
        run_count = 0

        class FailingGemini:
            async def run(self, *_: object, **__: object) -> None:
                nonlocal run_count
                run_count += 1
                raise ModelHTTPError(429, "gemini-test", {"error": {"code": "RATE_LIMITED"}})

        async def no_sleep(_: float) -> None:
            return None

        with (
            patch("agent_core._configured_providers", return_value=[gemini, openai]),
            patch("agent_core._build_agent", return_value=FailingGemini()),
            patch(
                "agent_core.provider_health",
                side_effect=lambda provider: {
                    "provider": provider,
                    "status": "READY" if provider == "gemini" else "NOT_CONFIGURED",
                },
            ),
            patch(
                "agent_core.prepare_evidence_first_context",
                new=__import__("unittest").mock.AsyncMock(
                    return_value=self._empty_preflight()
                ),
            ),
            patch("agent_core.asyncio.sleep", new=no_sleep),
            patch("agent_core.GEMINI_429_MAX_RETRIES", 2),
        ):
            with self.assertRaises(AgentCoreError) as context:
                asyncio.run(run_ahmed("temporary failure", provider="gemini"))

        self.assertEqual(context.exception.provider_status, "RATE_LIMITED")
        self.assertEqual(run_count, 3)

    def test_configured_provider_names_include_gemini_and_openai(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "test-gemini-key",
                "AI_INTEGRATIONS_OPENAI_API_KEY": "test-openai-key",
                "AI_INTEGRATIONS_OPENAI_BASE_URL": "https://openai.example.test/v1",
            },
        ):
            self.assertEqual(configured_provider_names(), ["gemini", "openai"])

    def test_openai_health_requires_both_integration_variables(self) -> None:
        with patch.dict(
            os.environ,
            {"AI_INTEGRATIONS_OPENAI_API_KEY": "test-openai-key"},
            clear=False,
        ):
            with patch.dict(os.environ, {"AI_INTEGRATIONS_OPENAI_BASE_URL": ""}):
                self.assertEqual(provider_health("openai")["status"], "NOT_CONFIGURED")

    def test_openai_candidate_uses_chat_model_without_network_call(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AI_INTEGRATIONS_OPENAI_API_KEY": "test-openai-key",
                "AI_INTEGRATIONS_OPENAI_BASE_URL": "https://openai.example.test/v1",
            },
        ):
            from agent_core import _configured_providers

            openai_candidate = next(
                candidate
                for candidate in _configured_providers()
                if candidate.name == "openai"
            )
            self.assertIsInstance(openai_candidate.model, OpenAIChatModel)

    def test_invalid_provider_is_rejected_before_model_call(self) -> None:
        with self.assertRaises(ValueError):
            asyncio.run(run_ahmed("test", provider="invalid"))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()