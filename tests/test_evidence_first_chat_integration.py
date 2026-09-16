from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent_core import (
    EvidenceFirstContext,
    prepare_evidence_first_context,
    run_ahmed,
)


class EvidenceFirstChatIntegrationTests(unittest.TestCase):
    def test_runtime_chat_preflights_evidence_before_model(self) -> None:
        runtime_result = {
            "status": "verified",
            "runtime_evidence": {
                "language": {
                    "value": "Python",
                    "status": "verified",
                    "evidence": [
                        {
                            "relative_source_path": "pyproject.toml",
                            "file_sha256": "a" * 64,
                            "line_start": 1,
                            "line_end": 2,
                            "verification_status": "VERIFIED",
                            "trust_classification": "PROJECT_SOURCE",
                        }
                    ],
                }
            },
        }
        architecture_result = {
            "status": "PARTIAL",
            "groups": {"foundation": {"status": "PARTIAL"}},
        }
        with (
            patch("agent_core.inspect_existing_runtime_evidence", return_value=runtime_result),
            patch(
                "agent_core.inspect_existing_architecture_evidence",
                return_value=architecture_result,
            ),
        ):
            context = asyncio.run(
                prepare_evidence_first_context(
                    "ما هو runtime الفعلي للمشروع؟",
                    scope="WEB",
                )
            )

        self.assertIsInstance(context, EvidenceFirstContext)
        self.assertIn("inspect_runtime_evidence", context.route.required_capabilities)
        self.assertIn("Python", context.model_context)
        self.assertIn("evidence", context.model_context.casefold())

    def test_web_chat_preflights_real_web_search_and_exposes_sources(self) -> None:
        web_result = {
            "ok": True,
            "results": [
                {
                    "title": "Official documentation",
                    "url": "https://docs.example.test/current",
                    "snippet": "Current official guidance.",
                }
            ],
        }
        with patch("agent_core.existing_web_search", new=AsyncMock(return_value=web_result)) as search:
            context = asyncio.run(
                prepare_evidence_first_context(
                    "ابحث على الويب عن أحدث التوثيق الرسمي",
                    scope="WEB",
                )
            )

        search.assert_awaited_once()
        self.assertIn("web_search", context.route.required_capabilities)
        self.assertIn("https://docs.example.test/current", context.model_context)
        self.assertIn("https://docs.example.test/current", context.external_sources)

    def test_web_preflight_persists_structured_unverified_external_provenance(self) -> None:
        web_result = {
            "ok": True,
            "results": [
                {
                    "title": "Official documentation",
                    "url": "https://docs.example.test/current?tracking=secret",
                    "snippet": "Current official guidance.",
                }
            ],
        }
        events: list[tuple[object, ...]] = []

        async def record(*args: object) -> None:
            events.append(args)

        with patch(
            "agent_core.existing_web_search",
            new=AsyncMock(return_value=web_result),
        ):
            asyncio.run(
                prepare_evidence_first_context(
                    "ابحث على الويب عن أحدث التوثيق الرسمي",
                    scope="WEB",
                    tool_event_recorder=record,
                )
            )

        metadata = events[0][3]
        self.assertIsInstance(metadata, dict)
        external = metadata["external_evidence_provenance"]
        self.assertEqual(external[0]["url"], "https://docs.example.test/current")
        self.assertEqual(external[0]["verification_status"], "UNVERIFIED_EXTERNAL")
        self.assertEqual(external[0]["source_identity"], "external_web_search")

    def test_my_files_preflight_persists_authorized_source_identity(self) -> None:
        source_result = {
            "evidence_status": "VERIFIED",
            "extracted_facts": {
                "source_id": "source-1",
                "original_filename": "ahmed_agent_master_v3_ar(1).md",
                "original_integrity_status": "VERIFIED",
            },
            "evidence_items": [
                {
                    "relative_source_path": "ahmed_agent_master_v3_ar(1).md",
                    "file_sha256": "a" * 64,
                    "line_start": 1,
                    "line_end": 2,
                    "verification_status": "VERIFIED",
                    "trust_classification": "AUTHORIZED_ORIGINAL",
                }
            ],
            "evidence_citations": ["master citation"],
            "source_label": "Ahmed Agent authorized original sources",
        }
        events: list[tuple[object, ...]] = []

        async def record(*args: object) -> None:
            events.append(args)

        with (
            patch(
                "agent_core.existing_my_files_search",
                new=AsyncMock(
                    return_value={
                        "results": [
                            {
                                "source_id": "source-1",
                                "original_available": True,
                                "citation": "master citation",
                            }
                        ]
                    }
                ),
            ),
            patch(
                "agent_core.inspect_existing_source_of_truth",
                new=AsyncMock(return_value=source_result),
            ),
        ):
            asyncio.run(
                prepare_evidence_first_context(
                    "راجع الملف الفعلي وقارنه بمصادر الحوادث",
                    scope="MY_FILES",
                    user_id="owner",
                    tool_event_recorder=record,
                )
            )

        inspect_event = next(
            event for event in events if event[0] == "inspect_source_of_truth"
        )
        self.assertEqual(inspect_event[3]["source_id"], "source-1")
        self.assertEqual(
            inspect_event[3]["source_filename"],
            "ahmed_agent_master_v3_ar(1).md",
        )
        self.assertEqual(
            inspect_event[3]["original_integrity_status"],
            "VERIFIED",
        )

    def test_build_vs_buy_preflight_propagates_architecture_provenance(self) -> None:
        architecture_result = {
            "status": "VERIFIED",
            "groups": {
                "runtime": {
                    "implementation_evidence": [
                        {
                            "relative_source_path": "server.py",
                            "file_sha256": "a" * 64,
                            "line_start": 1,
                            "line_end": 1,
                            "verification_status": "VERIFIED",
                            "trust_classification": "PROJECT_ARCHITECTURE_IMPLEMENTATION",
                        }
                    ]
                }
            },
        }
        web_result = {
            "ok": True,
            "results": [
                {
                    "title": "Official documentation",
                    "url": "https://docs.example.test/current",
                    "snippet": "Current official guidance.",
                }
            ],
        }
        with (
            patch(
                "agent_core.inspect_existing_architecture_evidence",
                return_value=architecture_result,
            ),
            patch(
                "agent_core.existing_web_search",
                new=AsyncMock(return_value=web_result),
            ),
        ):
            context = asyncio.run(
                prepare_evidence_first_context(
                    "قارن build-vs-buy مع حل Template جاهز.",
                    scope="WEB",
                )
            )

        architecture_envelope = next(
            envelope
            for envelope in context.evidence_envelopes
            if envelope["target"] == "project_architecture"
        )
        self.assertEqual(architecture_envelope["evidence_status"], "VERIFIED")
        self.assertTrue(architecture_envelope["evidence_provenance"])
        self.assertIn("server.py", context.model_context)

    def test_project_decision_question_includes_allowlisted_decision_evidence(self) -> None:
        architecture_result = {
            "status": "VERIFIED",
            "groups": {},
            "decision_records": {
                "status": "VERIFIED",
                "evidence_items": [
                    {
                        "relative_source_path": "docs/ADR-019-source-of-truth-hierarchy.md",
                        "file_sha256": "a" * 64,
                        "line_start": 1,
                        "line_end": 1,
                        "verification_status": "VERIFIED",
                        "trust_classification": "PROJECT_DECISION_RECORD",
                    }
                ],
                "record_statuses": {
                    "docs/ADR-019-source-of-truth-hierarchy.md": "ACCEPTED"
                },
                "missing_records": [],
                "conflicting_records": [],
            },
        }
        runtime_result = {
            "status": "VERIFIED",
            "runtime_evidence": {},
        }
        with (
            patch(
                "agent_core.inspect_existing_runtime_evidence",
                return_value=runtime_result,
            ),
            patch(
                "agent_core.inspect_existing_architecture_evidence",
                return_value=architecture_result,
            ),
        ):
            context = asyncio.run(
                prepare_evidence_first_context(
                    "ما هي مصادر المشروع والقرار المعتمد؟",
                    scope="WEB",
                )
            )

        self.assertIn(
            "inspect_architecture_evidence",
            context.route.required_capabilities,
        )
        self.assertIn("ADR-019-source-of-truth-hierarchy.md", context.model_context)
        self.assertIn("PROJECT_DECISION_RECORD", context.model_context)

    def test_general_chat_does_not_receive_unnecessary_evidence_preflight(self) -> None:
        context = asyncio.run(
            prepare_evidence_first_context(
                "اشرح لي مفهوم recursion باختصار",
                scope="WEB",
            )
        )
        self.assertEqual(context.route.required_capabilities, ())
        self.assertEqual(context.model_context, "")
        self.assertEqual(context.external_sources, ())

    def test_run_ahmed_injects_preflight_context_before_model(self) -> None:
        captured: dict[str, object] = {}

        class FakeAgent:
            async def run(self, message: str, **_: object) -> SimpleNamespace:
                captured["message"] = message
                return SimpleNamespace(
                    output="إجابة مدعومة.",
                    new_messages=lambda: [],
                )

        preflight = EvidenceFirstContext(
            route=SimpleNamespace(
                required_capabilities=("inspect_runtime_evidence",),
                abstain_if_evidence_missing=True,
            ),
            model_context="Evidence payload: runtime is verified.",
            evidence_envelopes=(),
            external_sources=(),
        )
        with (
            patch.dict("agent_core.os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False),
            patch("agent_core._configured_providers", return_value=[]),
            patch("agent_core._build_agent", return_value=FakeAgent()),
            patch("agent_core.prepare_evidence_first_context", new=AsyncMock(return_value=preflight)) as preflight_call,
        ):
            # The provider candidate is supplied directly so the test never reaches a network.
            candidate = SimpleNamespace(name="gemini", model_name="test-model", model=object())
            with patch("agent_core._configured_providers", return_value=[candidate]):
                result = asyncio.run(
                    run_ahmed(
                        "ما هو runtime الفعلي؟",
                        provider="gemini",
                    )
                )

        preflight_call.assert_awaited_once()
        self.assertIn("Evidence payload", str(captured["message"]))
        self.assertEqual(result.output, "إجابة مدعومة.")


if __name__ == "__main__":
    unittest.main()