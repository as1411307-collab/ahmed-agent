from __future__ import annotations

import asyncio
import unittest
from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ThinkingPart,
)
from pydantic_ai.models.function import FunctionModel

from agent_core import EvidenceFirstContext, parse_message_history, run_ahmed


_EVIDENCE_ENVELOPE = {
    "target": "runtime",
    "evidence_status": "VERIFIED",
    "evidence_items": [
        {
            "relative_source_path": "docs/runtime.md",
            "file_sha256": "a" * 64,
            "line_start": 1,
            "line_end": 3,
            "verification_status": "VERIFIED",
            "trust_classification": "PROJECT_DECISION_RECORD",
        }
    ],
}


def _run(
    parts: Sequence[ModelResponsePart],
    envelopes: Sequence[dict[str, object]],
):
    """Run the real agent loop with a scripted model response and preflight."""

    def respond(_messages: list[ModelMessage], _info: object) -> ModelResponse:
        return ModelResponse(parts=list(parts))

    candidate = SimpleNamespace(
        name="gemini",
        model_name="scripted",
        model=FunctionModel(respond),
    )
    preflight = EvidenceFirstContext(
        route=SimpleNamespace(
            required_capabilities=(),
            abstain_if_evidence_missing=False,
        ),
        model_context="",
        evidence_envelopes=tuple(envelopes),
        external_sources=(),
    )
    with (
        patch("agent_runtime._configured_providers", return_value=[candidate]),
        patch(
            "agent_runtime.provider_health",
            side_effect=lambda provider: {"provider": provider, "status": "READY"},
        ),
        patch("agent_runtime._mark_provider_ready"),
        patch(
            "agent_runtime.prepare_evidence_first_context",
            new=AsyncMock(return_value=preflight),
        ),
    ):
        return asyncio.run(run_ahmed("question", provider="gemini"))


def _saved_final_response(result: object) -> ModelResponse:
    # Same bytes the server saves, parsed the same way it reloads history.
    history = parse_message_history(result.new_messages_json())
    return next(
        message for message in reversed(history) if isinstance(message, ModelResponse)
    )


def _saved_reply(result: object) -> str:
    response = _saved_final_response(result)
    return "".join(part.content for part in response.parts if isinstance(part, TextPart))


def _displayed_reply(result: object) -> str:
    # Mirrors the reply returned by /chat/message.
    return str(result.output).strip()


class ReplyPersistenceConsistencyTests(unittest.TestCase):
    def test_saved_reply_is_exactly_the_displayed_reply(self) -> None:
        cases = {
            "model markers with evidence report": (
                [TextPart("Answer [1] per [source: model guess].")],
                [_EVIDENCE_ENVELOPE],
            ),
            "trailing whitespace with evidence report": (
                [TextPart("Answer.\n")],
                [_EVIDENCE_ENVELOPE],
            ),
            "reply split across text parts with evidence report": (
                [TextPart("Part one. "), TextPart("Part two.")],
                [_EVIDENCE_ENVELOPE],
            ),
            "clean reply with evidence report": (
                [TextPart("Answer.")],
                [_EVIDENCE_ENVELOPE],
            ),
            "trailing whitespace without evidence": (
                [TextPart("Answer.\n")],
                [],
            ),
            "model markers without evidence": (
                [TextPart("Answer [1].")],
                [],
            ),
        }
        for name, (parts, envelopes) in cases.items():
            with self.subTest(name):
                result = _run(parts, envelopes)
                self.assertEqual(_saved_reply(result), _displayed_reply(result))

    def test_saved_history_carries_canonical_references_not_model_markers(self) -> None:
        result = _run(
            [TextPart("Answer [1] per [source: model guess].")],
            [_EVIDENCE_ENVELOPE],
        )

        saved = _saved_reply(result)
        self.assertNotIn("[source: model guess]", saved)
        self.assertNotIn("[1]", saved)
        self.assertIn("\n\nEvidence references:\nruntime: [source: Ahmed Agent", saved)
        self.assertIn("file: docs/runtime.md; lines: 1-3", saved)

    def test_rewrite_keeps_non_text_parts_and_leaves_one_text_part(self) -> None:
        result = _run(
            [
                ThinkingPart("private reasoning"),
                TextPart("Part one [1]. "),
                TextPart("Part two."),
            ],
            [_EVIDENCE_ENVELOPE],
        )

        parts = _saved_final_response(result).parts
        self.assertEqual(
            [type(part) for part in parts],
            [ThinkingPart, TextPart],
        )
        self.assertEqual(parts[0].content, "private reasoning")
        self.assertEqual(parts[1].content, _displayed_reply(result))

    def test_reply_that_already_matches_is_saved_untouched(self) -> None:
        result = _run([TextPart("Part one. "), TextPart("Part two.")], [])

        parts = _saved_final_response(result).parts
        self.assertEqual(
            [part.content for part in parts],
            ["Part one. ", "Part two."],
        )


if __name__ == "__main__":
    unittest.main()
