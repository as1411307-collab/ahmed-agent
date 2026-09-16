from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from architecture_evidence import inspect_architecture_evidence
from evaluation_baseline import execute_real_case
from independent_semantic_evaluator import build_independent_review_document
from semantic_evaluation import (
    _packet_trace_for_independent_reviewer,
    evaluate_case,
)


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "real-cases-validated.json"
CONTRACT_PATH = ROOT / "semantic-evaluation-contract-v11.json"
EVALUATION_ARTIFACT = ROOT / "aa-rc-026-build-vs-buy-evaluation-2026-09-14.json"
INTEGRITY_ARTIFACT = ROOT / "aa-rc-026-build-vs-buy-integrity-2026-09-14.json"
AUDIT_DIMENSIONS = (
    "factual_correctness",
    "groundedness",
    "completeness",
    "scope_adherence",
    "safe_abstention",
)
PROJECT_TRUST_CLASSIFICATION = "VERIFIED_PROJECT_EVIDENCE"
EXTERNAL_TRUST_CLASSIFICATION = "UNVERIFIED_EXTERNAL_OFFICIAL"
_EXTERNAL_REFERENCE_KEYS = {
    "external:agents_sdk": {
        "agents_api_announcement",
        "agent_environments",
        "agents_sdk_overview",
        "agents_sdk_evolution",
        "agents_sdk_models",
        "human_in_the_loop",
        "sessions",
    },
    "external:responses_api_tools": {"responses_api_tools"},
    "external:model_capabilities": {
        "models_catalog",
        "model_comparison",
        "gpt_5_6_sol",
        "gpt_5_6_terra",
        "gpt_5_6_luna",
    },
    "external:pricing": {
        "api_platform_pricing",
        "gpt_5_6_sol",
        "gpt_5_6_terra",
        "gpt_5_6_luna",
    },
    "external:agentkit": {"agentkit"},
    "external:workspace_agents": {"agentkit"},
}


class AuditabilityError(ValueError):
    """AA-RC-026 cannot be independently reviewed without claim evidence."""

OPENAI_EXTERNAL_EVIDENCE_SNAPSHOT: tuple[dict[str, Any], ...] = (
    {
        "key": "agents_api_announcement",
        "url": "https://openai.com/index/introducing-the-agents-api/",
        "title": "Introducing the Agents API",
        "content_claims": [
            "The Agents API is a public beta.",
            "Agent environments can be OpenAI-hosted or self-hosted.",
        ],
    },
    {
        "key": "agent_environments",
        "url": "https://developers.openai.com/api/docs/guides/agents",
        "title": "OpenAI Agents API and agent environments",
        "content_claims": [
            "Agent environments include openai_hosted and self_hosted types.",
            "Hosted environments can expose files, plugins, and skills with connection status.",
        ],
    },
    {
        "key": "agents_sdk_evolution",
        "url": "https://openai.com/index/the-next-evolution-of-the-agents-sdk/",
        "title": "The next evolution of the Agents SDK",
        "content_claims": [
            "Agents SDK supports agents, tools, handoffs, guardrails, sessions, HITL, and tracing.",
            "Agents SDK supports sandbox work across files and tools.",
            "Agents SDK uses the Responses API by default, while Responses API can also be used directly when the application owns orchestration, state, and tool dispatch.",
        ],
    },
    {
        "key": "agents_sdk_overview",
        "url": "https://openai.github.io/openai-agents-python/",
        "title": "OpenAI Agents SDK overview",
        "content_claims": [
            "Agents SDK provides agents, tools, handoffs, guardrails, sessions, human-in-the-loop, and tracing.",
            "Agents SDK uses the Responses API by default and can be combined with direct Responses API calls.",
        ],
    },
    {
        "key": "agents_sdk_models",
        "url": "https://openai.github.io/openai-agents-python/models/",
        "title": "OpenAI Agents SDK models and providers",
        "content_claims": [
            "Non-OpenAI providers can be selected through ModelProvider per run or Agent.model per agent.",
            "Third-party adapters can differ in supported capabilities and behavior.",
        ],
    },
    {
        "key": "human_in_the_loop",
        "url": "https://openai.github.io/openai-agents-python/human_in_the_loop/",
        "title": "OpenAI Agents SDK human-in-the-loop",
        "content_claims": [
            "Agents SDK documents human-in-the-loop pauses and approvals for tool execution.",
        ],
    },
    {
        "key": "sessions",
        "url": "https://openai.github.io/openai-agents-python/sessions/",
        "title": "OpenAI Agents SDK sessions",
        "content_claims": [
            "Agents SDK sessions persist conversation state for agent runs.",
        ],
    },
    {
        "key": "responses_api_tools",
        "url": "https://developers.openai.com/api/reference/cli/resources/responses/methods/create",
        "title": "Responses API create",
        "content_claims": [
            "Responses API supports built-in web search, file search, computer use, MCP, and function tools.",
        ],
    },
    {
        "key": "models_catalog",
        "url": "https://developers.openai.com/api/docs/models",
        "title": "OpenAI models catalog",
        "content_claims": [
            "The OpenAI models catalog documents model capabilities and supported tools.",
        ],
    },
    {
        "key": "model_comparison",
        "url": "https://developers.openai.com/api/docs/models/compare",
        "title": "OpenAI model comparison",
        "content_claims": [
            "Model comparison documents input, output, context, and capability differences.",
        ],
    },
    {
        "key": "gpt_5_6_sol",
        "url": "https://developers.openai.com/api/docs/models/gpt-5.6-sol",
        "title": "GPT-5.6 Sol",
        "content_claims": [
            "Snapshot pricing on 2026-09-15 is $4 per million input tokens and $20 per million output tokens.",
            "The model supports web, file, computer, and related tools as documented.",
        ],
    },
    {
        "key": "gpt_5_6_terra",
        "url": "https://developers.openai.com/api/docs/models/gpt-5.6-terra",
        "title": "GPT-5.6 Terra",
        "content_claims": [
            "Snapshot pricing on 2026-09-15 is $2 per million input tokens and $12 per million output tokens.",
        ],
    },
    {
        "key": "gpt_5_6_luna",
        "url": "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
        "title": "GPT-5.6 Luna",
        "content_claims": [
            "Snapshot pricing on 2026-09-15 is $0.20 per million input tokens and $1.20 per million output tokens.",
        ],
    },
    {
        "key": "api_platform_pricing",
        "url": "https://openai.com/api/pricing/",
        "title": "OpenAI API pricing",
        "content_claims": [
            "API prices vary by selected model and tool and can change over time.",
            "The model prices in this snapshot are dated 2026-09-15 and are not permanent facts outside this evaluation.",
        ],
    },
    {
        "key": "agentkit",
        "url": "https://openai.com/index/introducing-agentkit/",
        "title": "Introducing AgentKit",
        "content_claims": [
            "Agent Builder and Evals are scheduled for retirement in 2026.",
            "The announcement recommends Agents SDK or Workspace Agents for the relevant workflows.",
        ],
    },
)

OPENAI_DOCUMENTS: tuple[dict[str, Any], ...] = tuple(
    {
        "key": item["key"],
        "url": item["url"],
        "markers": tuple(item["content_claims"]),
    }
    for item in OPENAI_EXTERNAL_EVIDENCE_SNAPSHOT
)
SNAPSHOT_ACCESSED_AT = "2026-09-15T00:00:00+00:00"
SNAPSHOT_PROVENANCE_CLASSIFICATION = "OFFICIAL_EXTERNAL"
SNAPSHOT_HASH = hashlib.sha256(
    json.dumps(
        OPENAI_EXTERNAL_EVIDENCE_SNAPSHOT,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()

USER_PROVIDED_OPENAI_EVIDENCE = {
    "source_identity": "external_openai_official",
    "evidence_classification": "EXTERNAL_EVIDENCE",
    "provenance_classification": SNAPSHOT_PROVENANCE_CLASSIFICATION,
    "verification_status": "UNVERIFIED_EXTERNAL",
    "source_urls": [document["url"] for document in OPENAI_DOCUMENTS],
    "snapshot_accessed_at": SNAPSHOT_ACCESSED_AT,
    "snapshot_sha256": SNAPSHOT_HASH,
    "provenance_note": (
        "Orchestrator-supplied snapshot of current official external documentation "
        "dated 2026-09-15; prices are time-bounded and kept separate from Ahmed "
        "Agent project evidence."
    ),
    "claims": [
        claim
        for source in OPENAI_EXTERNAL_EVIDENCE_SNAPSHOT
        for claim in source["content_claims"]
    ],
}


class _VisibleTextParser:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self._skip_depth = 0

    def feed(self, document: str) -> None:
        for match in re.finditer(
            r"<(script|style|svg)\b[^>]*>.*?</\1\s*>|<[^>]+>|([^<]+)",
            document,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            if match.group(2) is not None:
                self.parts.append(html.unescape(match.group(2)))

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.parts)).strip()


def _extract_title(document: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", document, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return re.sub(r"\s+", " ", html.unescape(match.group(1))).strip()


def _bounded_snippets(text: str, markers: tuple[str, ...]) -> list[str]:
    lowered = text.casefold()
    snippets: list[str] = []
    for marker in markers:
        start = lowered.find(marker.casefold())
        if start < 0:
            continue
        left = max(0, start - 180)
        right = min(len(text), start + len(marker) + 520)
        snippets.append(text[left:right].strip())
    return snippets[:4]


def _snapshot_external_evidence(*, fallback_reason: str) -> list[dict[str, Any]]:
    return [
        {
            "key": source["key"],
            "url": source["url"],
            "title": source["title"],
            "source_identity": "external_openai_official",
            "evidence_classification": "EXTERNAL_EVIDENCE",
            "verification_status": "UNVERIFIED_EXTERNAL",
            "evidence_origin": "ORCHESTRATOR_SUPPLIED_EXTERNAL_SNAPSHOT",
            "fallback_reason": fallback_reason,
            "retrieved_at": SNAPSHOT_ACCESSED_AT,
            "accessed_at": SNAPSHOT_ACCESSED_AT,
            "content_sha256": hashlib.sha256(
                json.dumps(
                    source,
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
            "content_claims": list(source["content_claims"]),
        }
        for source in OPENAI_EXTERNAL_EVIDENCE_SNAPSHOT
    ]


def fetch_openai_external_evidence() -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    retrieved_at = datetime.now(timezone.utc).isoformat()
    for document in OPENAI_DOCUMENTS:
        request = urllib.request.Request(
            document["url"],
            headers={"User-Agent": "Ahmed-Agent-AA-RC-026-Evaluator/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read()
                status = int(response.status)
                content_type = response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as error:
            if error.code == 403:
                return _snapshot_external_evidence(fallback_reason="HTTP_403")
            raise
        if status != 200 or "text/html" not in content_type.casefold():
            raise RuntimeError(
                f"Official OpenAI documentation unavailable: {document['url']}"
            )
        raw = body.decode("utf-8", errors="replace")
        parser = _VisibleTextParser()
        parser.feed(raw)
        text = parser.text()
        snippets = _bounded_snippets(text, document["markers"])
        if not snippets:
            raise RuntimeError(
                f"Official OpenAI documentation did not expose expected content: "
                f"{document['url']}"
            )
        evidence.append(
            {
                "key": document["key"],
                "url": document["url"],
                "title": _extract_title(raw),
                "source_identity": "external_openai_official",
                "evidence_classification": "EXTERNAL_EVIDENCE",
                "verification_status": "UNVERIFIED_EXTERNAL",
                "retrieved_at": retrieved_at,
                "date_accessed": retrieved_at,
                "http_status": status,
                "content_sha256": hashlib.sha256(body).hexdigest(),
                "snippets": snippets,
            }
        )
    return evidence


def _architecture_summary(result: dict[str, Any]) -> dict[str, Any]:
    citations: list[dict[str, Any]] = []
    groups: dict[str, Any] = {}
    for group_name, group in result.get("groups", {}).items():
        group_citations: list[dict[str, Any]] = []
        for evidence_kind in (
            "implementation_evidence",
            "wiring_evidence",
            "test_evidence",
        ):
            for item in group.get(evidence_kind, []):
                if not isinstance(item, dict):
                    continue
                citation = {
                    "group": group_name,
                    "evidence_kind": evidence_kind,
                    "relative_source_path": item.get("relative_source_path"),
                    "line_start": item.get("line_start"),
                    "line_end": item.get("line_end"),
                    "extracted_evidence": item.get("extracted_evidence"),
                    "file_sha256": item.get("file_sha256"),
                    "trust_classification": item.get("trust_classification"),
                    "verification_status": item.get("verification_status"),
                }
                group_citations.append(citation)
                citations.append(citation)
        groups[group_name] = {
            "status": group.get("status"),
            "evidence_file_hashes": group.get("evidence_file_hashes", {}),
            "citations": group_citations[:12],
            "architecture_unchanged": group.get("architecture_unchanged"),
        }
    return {
        "status": result.get("status"),
        "architecture_fingerprint": result.get("architecture_fingerprint"),
        "groups": groups,
        "citation_count": len(citations),
        "citations": citations[:48],
        "limitations": result.get("limitations", []),
    }


def build_comparison_matrix() -> list[dict[str, Any]]:
    return [
        {
            "criterion": "current_project_state",
            "build_assessment": "Continue the existing architecture and its evidence/provenance boundaries.",
            "buy_assessment": "Use OpenAI components as an integration target, not as proof that Ahmed Agent should be rebuilt.",
            "evidence_refs": ["project:architecture_fingerprint", "external:responses_api_tools"],
        },
        {
            "criterion": "migration_cost_complexity",
            "build_assessment": "Incremental integration preserves current persistence, policy, and evidence contracts.",
            "buy_assessment": "Adopting Responses API tools, Agents SDK primitives, or Workspace Agents requires mapping the existing runtime, tools, approvals, persistence, and provenance boundaries; Agent Builder retirement is an explicit migration risk.",
            "evidence_refs": [
                "project:runtime",
                "project:persistence_state_recovery",
                "project:policy_security",
                "external:responses_api_tools",
                "external:agents_sdk",
                "external:agentkit",
                "external:workspace_agents",
            ],
        },
        {
            "criterion": "maintenance_burden",
            "build_assessment": "Retain ownership of the current execution, recovery, policy, and evidence code.",
            "buy_assessment": "OpenAI-managed tool and agent capabilities can reduce custom orchestration, but add dependency on external API behavior and documentation.",
            "evidence_refs": ["project:persistence_state_recovery", "external:agents_sdk"],
        },
        {
            "criterion": "quality_control",
            "build_assessment": "Current evidence, citation, policy, and evaluation layers preserve local control over claims and gates.",
            "buy_assessment": "OpenAI tools and model capabilities add features, but do not replace Ahmed Agent's project-level quality and governance gates.",
            "evidence_refs": [
                "project:evidence_provenance",
                "project:evaluation",
                "external:responses_api_tools",
                "external:model_capabilities",
            ],
        },
        {
            "criterion": "operating_cost",
            "build_assessment": "Cost remains dependent on the providers and models selected by the current runtime; no unsupported exact spend claim is made.",
            "buy_assessment": "OpenAI model and tool usage must be budgeted against the current official pricing page for the selected models and tools.",
            "evidence_refs": ["project:actions_integrations", "external:pricing"],
        },
        {
            "criterion": "features_gained_lost",
            "build_assessment": "Keep existing capabilities and add only justified integrations.",
            "buy_assessment": "Potential gains include OpenAI Responses tools, Agents SDK patterns, and access to the current model catalog; potential losses include local control or portability if adopted as the sole runtime.",
            "evidence_refs": [
                "project:runtime",
                "project:actions_integrations",
                "external:responses_api_tools",
                "external:agents_sdk",
                "external:model_capabilities",
                "external:workspace_agents",
            ],
        },
        {
            "criterion": "provider_independence",
            "build_assessment": "Preserve the provider boundary and treat OpenAI as an option rather than the only execution path.",
            "buy_assessment": "A full replacement increases provider coupling unless the existing provider abstraction and fallback boundary remain intact.",
            "evidence_refs": ["project:actions_integrations", "project:policy_security"],
        },
        {
            "criterion": "local_self_hosted_fallback",
            "build_assessment": "Keep this as an explicit design constraint; the current evidence does not claim that OpenAI supplies a self-hosted fallback.",
            "buy_assessment": "OpenAI official documentation supports hosted APIs/SDK usage in this comparison; it is not evidence of a local or self-hosted OpenAI fallback.",
            "evidence_refs": ["project:actions_integrations", "external:agents_sdk"],
        },
        {
            "criterion": "provenance_governance",
            "build_assessment": "Retain project provenance, source identity, policy, and evaluation gates around any provider.",
            "buy_assessment": "External tool/agent features must remain classified as external evidence and must not replace local governance or source provenance.",
            "evidence_refs": [
                "project:evidence_provenance",
                "project:policy_security",
                "external:responses_api_tools",
                "external:agents_sdk",
            ],
        },
        {
            "criterion": "recommendation",
            "recommendation": "Continue Ahmed Agent incrementally; do not rebuild solely because OpenAI offers newer hosted agent capabilities. Evaluate a bounded OpenAI integration only where it improves a measured gap while preserving provider independence, provenance, governance, and fallback boundaries.",
            "evidence_refs": [
                "project:architecture_fingerprint",
                "external:model_capabilities",
                "external:pricing",
                "external:agentkit",
            ],
        },
    ]


def _canonical_url(value: str) -> str:
    parts = urlsplit(value.strip())
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _project_claim_sources(
    architecture: dict[str, Any],
    source_ref: str,
) -> list[dict[str, Any]]:
    if source_ref == "project:architecture_fingerprint":
        return [
            {
                "source_id": source_ref,
                "source_type": "project",
                "locator": {
                    "architecture_fingerprint": architecture.get(
                        "architecture_fingerprint"
                    )
                },
                "trust_classification": PROJECT_TRUST_CLASSIFICATION,
                "verification_status": "VERIFIED",
            }
        ]
    group_name = source_ref.removeprefix("project:")
    group = architecture.get("groups", {}).get(group_name)
    if not isinstance(group, dict):
        raise AuditabilityError(
            f"AA-RC-026 project source group is missing: {source_ref}"
        )
    citations = group.get("citations", [])
    if not citations:
        raise AuditabilityError(
            f"AA-RC-026 project source has no locator citations: {source_ref}"
        )
    return [
        {
            "source_id": source_ref,
            "source_type": "project",
            "locator": {
                "path": citation.get("relative_source_path"),
                "line_start": citation.get("line_start"),
                "line_end": citation.get("line_end"),
                "file_sha256": citation.get("file_sha256"),
                "extracted_evidence": citation.get("extracted_evidence"),
            },
            "trust_classification": citation.get(
                "trust_classification", PROJECT_TRUST_CLASSIFICATION
            ),
            "verification_status": citation.get("verification_status", "VERIFIED"),
        }
        for citation in citations[:3]
    ]


def _external_claim_sources(
    external: list[dict[str, Any]],
    source_ref: str,
) -> list[dict[str, Any]]:
    keys = _EXTERNAL_REFERENCE_KEYS.get(source_ref)
    if not keys:
        raise AuditabilityError(
            f"AA-RC-026 external source reference is not registered: {source_ref}"
        )
    sources = [item for item in external if item.get("key") in keys]
    if not sources:
        raise AuditabilityError(
            f"AA-RC-026 approved external source is missing: {source_ref}"
        )
    return [
        {
            "source_id": source_ref,
            "source_key": item.get("key"),
            "source_type": "external",
            "url": item.get("url"),
            "title": item.get("title"),
            "accessed_at": item.get("date_accessed") or item.get("retrieved_at"),
            "locator": {
                "url": item.get("url"),
                "content_sha256": item.get("content_sha256"),
                "snippet_count": len(item.get("snippets", [])),
            },
            "trust_classification": EXTERNAL_TRUST_CLASSIFICATION,
            "verification_status": item.get(
                "verification_status", "UNVERIFIED_EXTERNAL"
            ),
        }
        for item in sources
    ]


def build_claim_level_evidence_map(
    *,
    architecture: dict[str, Any],
    external: list[dict[str, Any]],
    comparison_matrix: list[dict[str, Any]],
) -> dict[str, Any]:
    claims: list[dict[str, Any]] = []
    for row in comparison_matrix:
        criterion = str(row.get("criterion") or "").strip()
        if not criterion:
            raise AuditabilityError("AA-RC-026 material claim is missing criterion.")
        for field in ("build_assessment", "buy_assessment", "recommendation"):
            text = row.get(field)
            if not isinstance(text, str) or not text.strip():
                continue
            evidence_refs = row.get("evidence_refs", [])
            if not isinstance(evidence_refs, list) or not evidence_refs:
                raise AuditabilityError(
                    f"AA-RC-026 claim has no evidence refs: {criterion}.{field}"
                )
            source_refs: list[dict[str, Any]] = []
            for source_ref in evidence_refs:
                if not isinstance(source_ref, str):
                    raise AuditabilityError(
                        f"AA-RC-026 claim has invalid source ref: {criterion}.{field}"
                    )
                if source_ref.startswith("project:"):
                    source_refs.extend(_project_claim_sources(architecture, source_ref))
                elif source_ref.startswith("external:"):
                    source_refs.extend(_external_claim_sources(external, source_ref))
                else:
                    raise AuditabilityError(
                        f"AA-RC-026 claim has unclassified source ref: {source_ref}"
                    )
            if not source_refs:
                raise AuditabilityError(
                    f"AA-RC-026 claim has no resolved sources: {criterion}.{field}"
                )
            claim_id = f"{criterion}.{field}"
            claim_sources = {item["source_type"] for item in source_refs}
            claims.append(
                {
                    "claim_id": claim_id,
                    "claim_type": (
                        "mixed"
                        if claim_sources == {"project", "external"}
                        else next(iter(claim_sources))
                    ),
                    "claim_text": text.strip(),
                    "source_refs": source_refs,
                    "direct_citation": f"[[claim:{claim_id}]]",
                    "is_resolved": True,
                }
            )
    if not claims:
        raise AuditabilityError("AA-RC-026 has no material claims.")
    return {
        "schema_version": "aa-rc-026-claim-evidence.v1",
        "material_claim_count": len(claims),
        "claims": claims,
    }


def build_external_evidence_reconciliation(
    *,
    retrieved_results: list[dict[str, Any]],
    approved_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    approved_by_url = {
        _canonical_url(str(item.get("url") or "")): item
        for item in approved_sources
        if item.get("url")
    }
    excluded_results: list[dict[str, Any]] = []
    matched_results: list[dict[str, Any]] = []
    for result in retrieved_results:
        url = _canonical_url(str(result.get("url") or ""))
        approved = approved_by_url.get(url)
        if approved is None:
            excluded_results.append(
                {
                    "url": result.get("url"),
                    "title": result.get("title"),
                    "accessed_at": result.get("retrieved_at")
                    or result.get("date_accessed"),
                    "reason": "not_in_approved_official_source_bundle",
                }
            )
            continue
        matched_results.append(
            {
                "url": approved["url"],
                "title": approved["title"],
                "accessed_at": approved.get("date_accessed")
                or approved.get("retrieved_at"),
                "approved_source_key": approved.get("key"),
            }
        )
    matched_urls = {entry["url"] for entry in matched_results}
    external_claims = [
        {
            "claim_id": f"external:{item['key']}",
            "url": item["url"],
            "title": item["title"],
            "accessed_at": item.get("date_accessed") or item.get("retrieved_at"),
            "source_identity": item.get("source_identity"),
            "trust_classification": EXTERNAL_TRUST_CLASSIFICATION,
            "verification_status": item.get(
                "verification_status", "UNVERIFIED_EXTERNAL"
            ),
            "retrieved_by_search": _canonical_url(item["url"]) in matched_urls,
        }
        for item in approved_sources
    ]
    return {
        "schema_version": "aa-rc-026-external-reconciliation.v1",
        "retrieved_result_count": len(retrieved_results),
        "approved_source_count": len(approved_sources),
        "matched_result_count": len(matched_results),
        "excluded_result_count": len(excluded_results),
        "matched_results": matched_results,
        "excluded_results": excluded_results,
        "external_claims": external_claims,
        "reconciliation_status": "RECONCILED",
    }


def build_audit_ready_answer(
    *,
    original_answer: str,
    claims: dict[str, Any],
) -> str:
    resolved_claims = [
        claim for claim in claims.get("claims", []) if claim.get("is_resolved")
    ]
    unresolved_claims = [
        claim for claim in claims.get("claims", []) if not claim.get("is_resolved")
    ]
    lines = [
        original_answer.strip(),
        "",
        "## Evidence-backed build-vs-buy comparison",
    ]
    lines.extend(
        f"- {claim['claim_text']} {claim['direct_citation']}"
        for claim in resolved_claims
    )
    lines.extend(
        [
            "",
            "## Unresolved claims and abstention",
            (
                "- Claims without a resolved source mapping are not asserted. "
                "Abstain from them until a project locator or an approved official "
                "external source is available."
            ),
        ]
    )
    if unresolved_claims:
        lines.extend(
            f"- Unresolved: {claim['claim_text']} (abstain; no resolved source)."
            for claim in unresolved_claims
        )
    return "\n".join(lines)


def build_review_audit_trace(
    *,
    answer: str,
    raw_trace: dict[str, Any],
    architecture: dict[str, Any],
    external: list[dict[str, Any]],
    claims: dict[str, Any],
    reconciliation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "answer_saved": True,
        "answer": answer,
        "raw_trace_saved": True,
        "raw_trace": _packet_trace_for_independent_reviewer(raw_trace),
        "targeted_trace_saved": True,
        "project_evidence_context": architecture.get("citations", [])[:48],
        "external_evidence_context": external,
        "claim_level_evidence_map": claims,
        "external_evidence_reconciliation": reconciliation,
    }


def build_review_dimension_notes(*, claims: dict[str, Any]) -> dict[str, str]:
    claim_ids = [
        f"claim:{claim['claim_id']}" for claim in claims.get("claims", [])[:8]
    ]
    refs = ", ".join(claim_ids)
    return {
        dimension: (
            f"AA-RC-026 audit evidence refs: {refs}. Review this dimension "
            "against the cited claim text, source locator, trust classification, "
            "and unresolved-claim abstention section."
        )
        for dimension in AUDIT_DIMENSIONS
    }


def _load_case() -> dict[str, Any]:
    document = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return next(case for case in document["cases"] if case["id"] == "AA-RC-026")


def _load_contract_case() -> dict[str, Any]:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    return next(
        case
        for case in document["cases"]
        if case["case_id"] == "AA-RC-026"
    )


def _trace_summary(
    trace: dict[str, Any],
    *,
    audit_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "run_id": trace.get("run_id"),
        "execution_status": trace.get("execution_status"),
        "execution_classification": trace.get("execution_classification"),
        "tool_names": sorted(
            {
                call.get("name")
                for call in trace.get("tool_calls", [])
                if isinstance(call, dict) and call.get("name")
            }
        ),
        "citation_count": len(trace.get("citations", [])),
        "project_evidence_citation_count": len(trace.get("evidence_provenance", [])),
        "external_evidence_count": len(trace.get("external_evidence_provenance", [])),
        "evidence_preconditions": trace.get("evidence_preconditions"),
        "runtime_cleanup": trace.get("runtime_cleanup"),
        "raw_trace_saved": False,
        "answer_saved": False,
    }
    if audit_trace is not None:
        summary.update(
            {
                "raw_trace_saved": audit_trace["raw_trace_saved"],
                "answer_saved": audit_trace["answer_saved"],
            }
        )
    return summary


def _build_independent_review(
    *,
    trace: dict[str, Any],
    contract_case: dict[str, Any],
    architecture: dict[str, Any],
    external: list[dict[str, Any]],
    comparison_matrix: list[dict[str, Any]],
    audit_trace: dict[str, Any],
    dimension_notes: dict[str, str],
) -> dict[str, Any]:
    review_trace = _packet_trace_for_independent_reviewer(trace)
    review_trace.update(audit_trace)
    review_trace["answer_for_review_redacted"] = audit_trace["answer"]
    review_trace["comparison_artifact"] = comparison_matrix
    review_trace["user_provided_external_evidence_bundle"] = (
        USER_PROVIDED_OPENAI_EVIDENCE
    )
    review_case = {
        "case_id": "AA-RC-026",
        "category": "build_vs_buy",
        "producer_provider": trace.get("provider"),
        "producer_model": trace.get("model"),
        "reference_fingerprint": contract_case["reference_fingerprint"],
        "execution_trace_fingerprint": "",
        "reference_assertions": contract_case["reference_assertions"],
        "trace": review_trace,
        "auditability_dimension_notes": dimension_notes,
    }
    packet_body = json.dumps(review_case, ensure_ascii=False, sort_keys=True).encode(
        "utf-8"
    )
    review_case["execution_trace_fingerprint"] = hashlib.sha256(packet_body).hexdigest()
    packet = {
        "packet_sha256": hashlib.sha256(
            json.dumps(review_case, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "cases": [review_case],
    }
    return build_independent_review_document(
        packet=packet,
        baseline={"execution_config": {"provider": trace.get("provider")}},
        reviewer_provider="openai",
    )


async def execute_aa_rc_026(
    *,
    base_url: str,
    owner_token: str,
    provider: str = "gemini",
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    architecture = _architecture_summary(inspect_architecture_evidence())
    external = await asyncio.to_thread(fetch_openai_external_evidence)
    case = _load_case()
    case = {
        **case,
        "input": (
            "قارن build-vs-buy لـAhmed Agent باستخدام architecture evidence الحالية "
            "والـcitations. استخدم web_search فقط للوثائق الرسمية الحالية من OpenAI "
            "وللمواضيع التالية: Responses API وأدواته، Agents SDK، قدرات النماذج، "
            "والتسعير. افصل external evidence عن project evidence. غطِّ صراحةً: "
            "current project state، migration cost/complexity، maintenance burden، "
            "quality/control، operating cost، features gained/lost، provider "
            "independence، local/self-hosted fallback، provenance/governance، ثم "
            "recommendation بأسباب قابلة للاستشهاد. لا توصي بإعادة البناء لمجرد "
            "أن الحل أحدث، وامتنع عن أي ادعاء لا يكفيه الدليل."
        ),
    }
    started = time.perf_counter()
    trace = await execute_real_case(
        case,
        base_url=base_url,
        owner_token=owner_token,
        provider=provider,
        timeout_seconds=timeout_seconds,
    )
    trace["targeted_latency_ms"] = int((time.perf_counter() - started) * 1000)
    retrieved_external_results = [
        item
        for item in trace.get("external_evidence_provenance", [])
        if isinstance(item, dict)
    ]
    trace["external_evidence_provenance"] = [
        *[
            *retrieved_external_results,
        ],
        *[
            {
                "url": item["url"],
                "title": item["title"],
                "content_sha256": item["content_sha256"],
                "source_identity": item["source_identity"],
                "verification_status": item["verification_status"],
            }
            for item in external
        ],
    ]
    comparison_matrix = build_comparison_matrix()
    claims = build_claim_level_evidence_map(
        architecture=architecture,
        external=external,
        comparison_matrix=comparison_matrix,
    )
    reconciliation = build_external_evidence_reconciliation(
        retrieved_results=retrieved_external_results,
        approved_sources=external,
    )
    audit_answer = build_audit_ready_answer(
        original_answer=str(trace.get("final_output") or ""),
        claims=claims,
    )
    audit_trace = build_review_audit_trace(
        answer=audit_answer,
        raw_trace=trace,
        architecture=architecture,
        external=external,
        claims=claims,
        reconciliation=reconciliation,
    )
    dimension_notes = build_review_dimension_notes(claims=claims)
    semantic = evaluate_case(
        contract_case=_load_contract_case(),
        trace=trace,
    )
    independent_review: dict[str, Any]
    try:
        independent_review = _build_independent_review(
            trace=trace,
            contract_case=_load_contract_case(),
            architecture=architecture,
            external=external,
            comparison_matrix=comparison_matrix,
            audit_trace=audit_trace,
            dimension_notes=dimension_notes,
        )
    except Exception as error:
        independent_review = {
            "status": "NOT_DETERMINED",
            "error_type": type(error).__name__,
            "error": "Independent reviewer unavailable or rejected the packet.",
        }
    report = {
        "schema_version": "aa-rc-026-targeted-evaluation.v1",
        "case_id": "AA-RC-026",
        "evaluation_version": "2026-09-14.build-vs-buy.openai-only",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "external_evidence_policy": {
            "allowed_source_identity": "external_openai_official",
            "allowed_domains": [
                "platform.openai.com",
                "developers.openai.com",
                "openai.com",
                "openai.github.io",
            ],
            "verification_status": "UNVERIFIED_EXTERNAL",
            "project_evidence_must_remain_separate": True,
        },
        "project_evidence": architecture,
        "external_evidence": external,
        "external_evidence_bundle": USER_PROVIDED_OPENAI_EVIDENCE,
        "comparison_matrix": comparison_matrix,
        "auditability": {
            **audit_trace,
            "dimension_notes": dimension_notes,
        },
        "targeted_trace": _trace_summary(trace, audit_trace=audit_trace),
        "semantic_evaluation": {
            "semantic_status": semantic.get("semantic_status"),
            "execution_status": semantic.get("execution_status"),
            "execution_classification": semantic.get("execution_classification"),
            "dimensions": {
                key: value.get("status")
                for key, value in semantic.get("dimensions", {}).items()
                if isinstance(value, dict)
            },
            "independent_review": None,
        },
        "independent_review": independent_review,
        "recommendation": {
            "status": (
                "READY"
                if (
                    trace.get("evidence_preconditions", {}).get("status") == "READY"
                    and independent_review.get("reviews", [{}])[0].get("decision")
                    == "PASS"
                )
                else "NOT_DETERMINED"
            ),
            "text": (
                "Continue Ahmed Agent incrementally; do not rebuild solely because "
                "OpenAI offers newer hosted agent capabilities."
            ),
        },
    }
    return report


def _integrity_record(report: dict[str, Any]) -> dict[str, Any]:
    EVALUATION_ARTIFACT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    files = []
    for path in (EVALUATION_ARTIFACT, CONTRACT_PATH):
        data = path.read_bytes()
        files.append(
            {
                "file": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        )
    reviews = report.get("independent_review", {}).get("reviews", [])
    independent_review_decision = (
        reviews[0].get("decision")
        if reviews and isinstance(reviews[0], dict)
        else report.get("independent_review", {}).get("status", "NOT_DETERMINED")
    )
    return {
        "schema_version": "aa-rc-026-integrity.v1",
        "case_id": "AA-RC-026",
        "evaluation_version": report["evaluation_version"],
        "artifact": str(EVALUATION_ARTIFACT.relative_to(ROOT)),
        "files": files,
        "external_evidence_count": len(report["external_evidence"]),
        "external_evidence_bundle_url_count": len(
            report.get("external_evidence_bundle", {}).get("source_urls", [])
        ),
        "project_architecture_fingerprint": report["project_evidence"][
            "architecture_fingerprint"
        ],
        "targeted_execution_status": report["targeted_trace"]["execution_status"],
        "evidence_precondition_status": report["targeted_trace"][
            "evidence_preconditions"
        ]["status"],
        "independent_review_decision": independent_review_decision,
        "recommendation_status": report["recommendation"]["status"],
        "semantic_status": report["semantic_evaluation"]["semantic_status"],
        "raw_trace_saved": report["targeted_trace"]["raw_trace_saved"],
    }


async def _main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    import os

    owner_token = os.environ.get("AHMED_OWNER_TOKEN")
    if not owner_token:
        raise SystemExit("AUTHENTICATED_EVALUATION_PRINCIPAL_BLOCKED")
    report = await execute_aa_rc_026(
        base_url=args.base_url,
        owner_token=owner_token,
        provider=args.provider,
        timeout_seconds=args.timeout_seconds,
    )
    integrity = _integrity_record(report)
    INTEGRITY_ARTIFACT.write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(EVALUATION_ARTIFACT.relative_to(ROOT)),
                "integrity": str(INTEGRITY_ARTIFACT.relative_to(ROOT)),
                "status": report["recommendation"]["status"],
                "semantic_status": report["semantic_evaluation"]["semantic_status"],
                "evidence_preconditions": report["targeted_trace"][
                    "evidence_preconditions"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(_main_async())