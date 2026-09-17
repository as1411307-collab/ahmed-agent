from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import Field
from pydantic_ai import Agent, RunContext, UsageLimits
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.models import Model

from academic_search import academic_search as existing_academic_search
from agent_consts import (
    MAX_MESSAGE_HISTORY_BYTES,
    MAX_MESSAGE_HISTORY_ITEMS,
    MAX_MODEL_REQUESTS,
    MAX_TOOL_CALLS,
    MY_FILES_INSTRUCTIONS,
    SUPPORTED_PROVIDER_NAMES,
    WEB_INSTRUCTIONS,
    AgentCoreError,
    AgentDeps,
    logger,
)
from agent_evidence import prepare_evidence_first_context
from agent_providers import (
    ProviderCandidate,
    _configured_providers,
    _mark_provider_429,
    _mark_provider_ready,
    _open_rate_limit,
    _provider_error_code,
    _provider_state_for,
    _record_provider_fallback,
    _transient_provider_failure_reason,
    provider_health,
)


def _ready_fallback(
    candidates: Sequence[ProviderCandidate],
    *,
    excluded: set[str],
) -> ProviderCandidate | None:
    """Local fallback decision using THIS module's provider_health lookup, so tests
    patching agent_runtime.provider_health observe one consistent health view."""
    for candidate in candidates:
        if candidate.name in excluded:
            continue
        if provider_health(candidate.name)["status"] == "READY":  # type: ignore[arg-type]
            return candidate
    return None
from architecture_evidence import (
    inspect_architecture_evidence as inspect_existing_architecture_evidence,
)
from config import (
    DEFAULT_TOP_K,
    GEMINI_429_BACKOFF_BASE_SECONDS,
    GEMINI_429_BACKOFF_MAX_SECONDS,
    GEMINI_429_MAX_RETRIES,
    MAX_QUERY_LENGTH,
    MAX_TOP_K,
)
from evidence_citations import remove_model_source_markers, render_evidence_report
from github_search import github_search as existing_github_search
from my_files import search_my_files as existing_my_files_search
from persistence import create_pending_action
from policy import data_only_boundary, get_tool_policy, tool_metadata
from runtime_evidence import (
    inspect_runtime_evidence as inspect_existing_runtime_evidence,
)
from skill_tools import web_search as existing_web_search
from source_of_truth import (
    inspect_source_of_truth as inspect_existing_source_of_truth,
)
from source_status import inspect_source_status as inspect_existing_source_status


def _build_agent(model: Model, scope: Literal["WEB", "MY_FILES"]) -> Agent[AgentDeps, str]:
    async def web_search(
        ctx: RunContext[AgentDeps],
        query: Annotated[str, Field(min_length=1, max_length=2000)],
        mode: Literal["FAST", "DEEP"],
        max_results: Annotated[int, Field(ge=1, le=5)] = 5,
    ) -> dict[str, object]:
        started_at = time.perf_counter()
        try:
            result = await existing_web_search(
                query=query.strip(),
                mode=mode,
                max_results=max_results,
            )
        except Exception:
            if ctx.deps.tool_event_recorder is not None:
                await ctx.deps.tool_event_recorder(
                    "web_search",
                    "failed",
                    int((time.perf_counter() - started_at) * 1000),
                    {"mode": mode},
                )
            raise

        safe_metadata: dict[str, Any] = {"mode": mode}
        if isinstance(result, dict):
            for key in (
                "search_calls",
                "extract_calls",
                "credits_used",
                "urls_extracted",
                "candidate_count",
                "deduplicated_count",
                "failed_calls",
            ):
                value = result.get(key)
                if isinstance(value, (int, float)):
                    safe_metadata[key] = value
        safe_metadata["policy"] = tool_metadata("web_search")
        if ctx.deps.tool_event_recorder is not None:
            await ctx.deps.tool_event_recorder(
                "web_search",
                "success" if result.get("ok", True) else "failed",
                int((time.perf_counter() - started_at) * 1000),
                safe_metadata,
            )
        if isinstance(result, dict):
            return {
                **result,
                "data_boundary": data_only_boundary("web_search"),
            }
        return result

    async def academic_search(
        ctx: RunContext[AgentDeps],
        query: Annotated[str, Field(min_length=1, max_length=2000)],
        intent: Literal[
            "auto",
            "doi",
            "exact_title",
            "author",
            "topic",
            "citations",
            "latest_research",
        ] = "auto",
        max_results: Annotated[int, Field(ge=1, le=5)] = 5,
    ) -> dict[str, object]:
        started_at = time.perf_counter()
        try:
            result = await existing_academic_search(
                query=query.strip(),
                intent=intent,
                max_results=max_results,
            )
        except Exception:
            if ctx.deps.tool_event_recorder is not None:
                await ctx.deps.tool_event_recorder(
                    "academic_search",
                    "failed",
                    int((time.perf_counter() - started_at) * 1000),
                    {"intent": intent, "scope": "WEB"},
                )
            raise

        if ctx.deps.tool_event_recorder is not None:
            await ctx.deps.tool_event_recorder(
                "academic_search",
                "success" if result.get("ok", True) else "failed",
                int((time.perf_counter() - started_at) * 1000),
                {
                    "intent": intent,
                    "scope": "WEB",
                    "result_count": len(result.get("results", [])),
                    "providers_used": result.get("providers_used", []),
                },
            )
        return {
            **result,
            "data_boundary": data_only_boundary("academic_external"),
        }

    async def github_search(
        ctx: RunContext[AgentDeps],
        query: Annotated[str, Field(min_length=1, max_length=256)],
        intent: Literal[
            "auto",
            "repository",
            "repository_lookup",
            "issue",
            "issue_lookup",
            "release",
            "releases",
            "latest_release",
        ] = "auto",
        owner: Annotated[str | None, Field(max_length=100)] = None,
        repo: Annotated[str | None, Field(max_length=100)] = None,
        issue_number: Annotated[int | None, Field(ge=1)] = None,
        max_results: Annotated[int, Field(ge=1, le=5)] = 5,
    ) -> dict[str, object]:
        started_at = time.perf_counter()
        try:
            result = await existing_github_search(
                query=query.strip(),
                intent=intent,
                owner=owner.strip() if owner else None,
                repo=repo.strip() if repo else None,
                issue_number=issue_number,
                max_results=max_results,
            )
        except Exception:
            if ctx.deps.tool_event_recorder is not None:
                await ctx.deps.tool_event_recorder(
                    "github_search",
                    "failed",
                    int((time.perf_counter() - started_at) * 1000),
                    {"intent": intent, "scope": "WEB"},
                )
            raise

        if ctx.deps.tool_event_recorder is not None:
            await ctx.deps.tool_event_recorder(
                "github_search",
                "success" if result.get("ok", True) else "failed",
                int((time.perf_counter() - started_at) * 1000),
                {
                    "intent": intent,
                    "scope": "WEB",
                    "result_count": len(result.get("results", [])),
                    "providers_used": result.get("providers_used", []),
                },
            )
        return {
            **result,
            "data_boundary": data_only_boundary("github_public_api"),
        }

    async def search_my_files(
        ctx: RunContext[AgentDeps],
        query: Annotated[str, Field(min_length=1, max_length=MAX_QUERY_LENGTH)],
        top_k: Annotated[int, Field(ge=1, le=MAX_TOP_K)] = DEFAULT_TOP_K,
    ) -> dict[str, object]:
        started_at = time.perf_counter()
        try:
            result = await existing_my_files_search(query=query, top_k=top_k)
        except Exception:
            if ctx.deps.tool_event_recorder is not None:
                await ctx.deps.tool_event_recorder(
                    "search_my_files",
                    "failed",
                    int((time.perf_counter() - started_at) * 1000),
                    {"scope": "MY_FILES"},
                )
            raise
        if ctx.deps.tool_event_recorder is not None:
            await ctx.deps.tool_event_recorder(
                "search_my_files",
                "success" if result.get("ok", True) else "failed",
                int((time.perf_counter() - started_at) * 1000),
                {
                    "scope": "MY_FILES",
                    "result_count": len(result.get("results", [])),
                    "policy": tool_metadata("search_my_files"),
                },
            )
        return {
            **result,
            "data_boundary": data_only_boundary("uploaded_files"),
        }

    async def test_sensitive_action(
        ctx: RunContext[AgentDeps],
        reason: Annotated[str, Field(min_length=1, max_length=500)],
    ) -> dict[str, object]:
        policy = get_tool_policy("test_sensitive_action")
        if not ctx.deps.conversation_id or not ctx.deps.run_id:
            raise RuntimeError("Sensitive actions require a persisted session and run.")
        action_id = str(uuid4())
        idempotency_key = hashlib.sha256(
            f"{ctx.deps.run_id}:test_sensitive_action:{reason.strip()}".encode("utf-8")
        ).hexdigest()
        pending_action = await create_pending_action(
            action_id=action_id,
            session_id=ctx.deps.conversation_id,
            run_id=ctx.deps.run_id,
            user_id=ctx.deps.user_id or "unauthenticated",
            tool_name=policy.tool_name,
            risk_level=policy.risk_level.value,
            arguments={"reason": reason.strip()},
            idempotency_key=idempotency_key,
        )
        action_id = str(pending_action["action_id"])
        if ctx.deps.tool_event_recorder is not None:
            await ctx.deps.tool_event_recorder(
                policy.tool_name,
                "success",
                0,
                {
                    "policy": tool_metadata(policy.tool_name),
                    "action_id": action_id,
                    "status": "pending_approval",
                },
            )
        return {
            "ok": False,
            "requires_approval": True,
            "action_id": action_id,
            "risk_level": policy.risk_level.value,
            "message": "Approval is required before this action can execute.",
        }

    async def inspect_runtime_evidence(
        ctx: RunContext[AgentDeps],
    ) -> dict[str, object]:
        """Read fixed project files to identify the active runtime.

        This is a bounded, read-only evidence lookup. It never accepts a path,
        executes commands, reads secrets, or changes project files.
        """

        started_at = time.perf_counter()
        try:
            result = await asyncio.to_thread(inspect_existing_runtime_evidence)
        except Exception:
            if ctx.deps.tool_event_recorder is not None:
                await ctx.deps.tool_event_recorder(
                    "inspect_runtime_evidence",
                    "failed",
                    int((time.perf_counter() - started_at) * 1000),
                    {"scope": "PROJECT_RUNTIME_EVIDENCE"},
                )
            raise

        policy = tool_metadata("inspect_runtime_evidence")
        safe_metadata = {
            "scope": "PROJECT_RUNTIME_EVIDENCE",
            "status": result.get("status"),
            "evidence_files": result.get("evidence_files", []),
            "claim_statuses": {
                name: claim.get("status")
                for name, claim in result.get("runtime_evidence", {}).items()
                if isinstance(claim, dict)
            },
            "policy": policy,
        }
        if ctx.deps.tool_event_recorder is not None:
            await ctx.deps.tool_event_recorder(
                "inspect_runtime_evidence",
                "success",
                int((time.perf_counter() - started_at) * 1000),
                safe_metadata,
            )
        return {
            **result,
            "policy": policy,
            "data_boundary": data_only_boundary("project_runtime_evidence"),
        }

    async def inspect_architecture_evidence(
        ctx: RunContext[AgentDeps],
    ) -> dict[str, object]:
        """Read fixed architecture groups and return structured evidence."""

        started_at = time.perf_counter()
        try:
            result = await asyncio.to_thread(
                inspect_existing_architecture_evidence
            )
        except Exception:
            if ctx.deps.tool_event_recorder is not None:
                await ctx.deps.tool_event_recorder(
                    "inspect_architecture_evidence",
                    "failed",
                    int((time.perf_counter() - started_at) * 1000),
                    {"scope": "PROJECT_ARCHITECTURE_EVIDENCE"},
                )
            raise

        policy = tool_metadata("inspect_architecture_evidence")
        safe_metadata = {
            "scope": "PROJECT_ARCHITECTURE_EVIDENCE",
            "status": result.get("status"),
            "architecture_fingerprint": result.get("architecture_fingerprint"),
            "group_statuses": {
                name: group.get("status")
                for name, group in result.get("groups", {}).items()
                if isinstance(group, dict)
            },
            "policy": policy,
        }
        if ctx.deps.tool_event_recorder is not None:
            await ctx.deps.tool_event_recorder(
                "inspect_architecture_evidence",
                "success",
                int((time.perf_counter() - started_at) * 1000),
                safe_metadata,
            )
        return {
            **result,
            "policy": policy,
            "data_boundary": data_only_boundary(
                "project_architecture_evidence"
            ),
        }

    async def inspect_source_status(
        ctx: RunContext[AgentDeps],
        component: Annotated[
            Literal["search_provider", "page_fetcher"],
            Field(description="Fixed source-status component identifier."),
        ],
    ) -> dict[str, object]:
        """Inspect fixed source/config evidence for one approved component."""

        started_at = time.perf_counter()
        try:
            result = await asyncio.to_thread(
                inspect_existing_source_status,
                component,
            )
        except Exception:
            if ctx.deps.tool_event_recorder is not None:
                await ctx.deps.tool_event_recorder(
                    "inspect_source_status",
                    "failed",
                    int((time.perf_counter() - started_at) * 1000),
                    {
                        "scope": "PROJECT_SOURCE_STATUS_EVIDENCE",
                        "component": component,
                    },
                )
            raise

        policy = tool_metadata("inspect_source_status")
        safe_metadata = {
            "scope": "PROJECT_SOURCE_STATUS_EVIDENCE",
            "component": component,
            "evidence_status": result.get("evidence_status"),
            "evidence_item_count": result.get("extracted_facts", {}).get(
                "evidence_item_count", 0
            ),
            "policy": policy,
            "evidence_citations": result.get("evidence_citations", [])[:24],
            "evidence_items": [
                {
                    key: item.get(key)
                    for key in (
                        "relative_source_path",
                        "file_sha256",
                        "line_start",
                        "line_end",
                        "verification_status",
                        "trust_classification",
                    )
                    if key in item
                }
                for item in result.get("evidence_items", [])[:24]
                if isinstance(item, dict)
            ],
        }
        if ctx.deps.evidence_envelopes is not None:
            ctx.deps.evidence_envelopes.append(
                {
                    "target": result.get("target"),
                    "evidence_status": result.get("evidence_status"),
                    "evidence_items": result.get("evidence_items", [])[:24],
                }
            )
        if ctx.deps.tool_event_recorder is not None:
            await ctx.deps.tool_event_recorder(
                "inspect_source_status",
                "success",
                int((time.perf_counter() - started_at) * 1000),
                safe_metadata,
            )
        return {
            **result,
            "policy": policy,
            "data_boundary": data_only_boundary("project_source_status_evidence"),
        }

    async def inspect_source_of_truth(
        ctx: RunContext[AgentDeps],
        source_id: Annotated[
            str,
            Field(description="Authorized MY_FILES source identifier; never a path."),
        ],
    ) -> dict[str, object]:
        """Inspect one authorized immutable original source by source ID."""

        started_at = time.perf_counter()
        try:
            result = await inspect_existing_source_of_truth(
                source_id,
                owner_principal_id=ctx.deps.user_id,
            )
        except Exception:
            if ctx.deps.tool_event_recorder is not None:
                await ctx.deps.tool_event_recorder(
                    "inspect_source_of_truth",
                    "failed",
                    int((time.perf_counter() - started_at) * 1000),
                    {"scope": "AUTHORIZED_MY_FILES_SOURCE"},
                )
            raise

        policy = tool_metadata("inspect_source_of_truth")
        evidence_items = [
            item
            for item in result.get("evidence_items", [])[:24]
            if isinstance(item, dict)
        ]
        safe_metadata = {
            "scope": "AUTHORIZED_MY_FILES_SOURCE",
            "source_id": result.get("extracted_facts", {}).get("source_id"),
            "source_filename": result.get("extracted_facts", {}).get(
                "original_filename"
            ),
            "evidence_status": result.get("evidence_status"),
            "original_integrity_status": result.get("extracted_facts", {}).get(
                "original_integrity_status"
            ),
            "evidence_citations": result.get("evidence_citations", [])[:24],
            "evidence_items": [
                {
                    key: item.get(key)
                    for key in (
                        "relative_source_path",
                        "file_sha256",
                        "line_start",
                        "line_end",
                        "verification_status",
                        "trust_classification",
                    )
                    if key in item
                }
                for item in evidence_items
            ],
            "policy": policy,
        }
        if ctx.deps.evidence_envelopes is not None:
            ctx.deps.evidence_envelopes.append(
                {
                    "target": result.get("target"),
                    "evidence_status": result.get("evidence_status"),
                    "source_label": result.get("source_label"),
                    "evidence_items": evidence_items,
                }
            )
        if ctx.deps.tool_event_recorder is not None:
            await ctx.deps.tool_event_recorder(
                "inspect_source_of_truth",
                "success",
                int((time.perf_counter() - started_at) * 1000),
                safe_metadata,
            )
        return {
            **result,
            "policy": policy,
            "data_boundary": data_only_boundary("authorized_my_files_source"),
        }

    if scope == "WEB":
        tools = [
            web_search,
            academic_search,
            github_search,
            inspect_runtime_evidence,
            inspect_architecture_evidence,
            inspect_source_status,
            inspect_source_of_truth,
            test_sensitive_action,
        ]
        instructions = WEB_INSTRUCTIONS
    else:
        tools = [search_my_files, inspect_source_of_truth, test_sensitive_action]
        instructions = MY_FILES_INSTRUCTIONS

    return Agent(
        model=model,
        deps_type=AgentDeps,
        instructions=instructions,
        retries=2,
        tools=tools,
        tool_timeout=45,
    )


def _retry_after_seconds(error: ModelHTTPError, attempt: int) -> float:
    headers = getattr(error, "headers", None)
    if hasattr(headers, "get"):
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(0.0, float(retry_after))
            except (TypeError, ValueError):
                pass
    return min(
        GEMINI_429_BACKOFF_MAX_SECONDS,
        GEMINI_429_BACKOFF_BASE_SECONDS * (2.0**attempt),
    )


def parse_message_history(raw_history: object) -> Sequence[ModelMessage] | None:
    if raw_history is None:
        return None

    if isinstance(raw_history, bytes):
        if len(raw_history) > MAX_MESSAGE_HISTORY_BYTES:
            raise ValueError("history is too large")
        try:
            history = ModelMessagesTypeAdapter.validate_json(raw_history)
        except ValueError as error:
            raise ValueError("history is invalid") from error
    elif isinstance(raw_history, str):
        if len(raw_history.encode("utf-8")) > MAX_MESSAGE_HISTORY_BYTES:
            raise ValueError("history is too large")
        try:
            history = ModelMessagesTypeAdapter.validate_json(raw_history)
        except ValueError as error:
            raise ValueError("history is invalid") from error
    elif isinstance(raw_history, list):
        if len(raw_history) > MAX_MESSAGE_HISTORY_ITEMS:
            raise ValueError("history has too many messages")
        try:
            history = ModelMessagesTypeAdapter.validate_python(raw_history)
        except ValueError as error:
            raise ValueError("history is invalid") from error
    else:
        raise ValueError("history must be a JSON string, bytes, or list")

    if len(history) > MAX_MESSAGE_HISTORY_ITEMS:
        raise ValueError("history has too many messages")
    return history


async def run_ahmed(
    user_message: str,
    *,
    message_history: Sequence[ModelMessage] | None = None,
    conversation_id: str | None = None,
    run_id: str | None = None,
    user_id: str | None = None,
    scope: Literal["WEB", "MY_FILES"] = "WEB",
    provider: ProviderName = "gemini",
    tool_event_recorder: (
        Callable[[str, str, int, dict[str, Any] | None], Awaitable[None]] | None
    ) = None,
):
    if provider not in SUPPORTED_PROVIDER_NAMES:
        raise ValueError("invalid model provider")

    providers = {
        candidate.name: candidate for candidate in _configured_providers()
    }
    candidate = providers.get(provider)
    if candidate is None:
        raise AgentCoreError(
            "No authorized model provider is configured.",
            provider=provider,
            provider_status="NOT_CONFIGURED",
        )

    if scope not in {"WEB", "MY_FILES"}:
        raise ValueError("invalid agent scope")

    candidates = [candidate] + [
        configured_candidate
        for configured_candidate in providers.values()
        if configured_candidate.name != candidate.name
    ]
    attempted_providers: set[str] = set()
    active_candidate = candidate
    health = provider_health(active_candidate.name)  # type: ignore[arg-type]
    if health["status"] == "RATE_LIMITED":
        fallback = _ready_fallback(candidates, excluded={active_candidate.name})
        if fallback is None:
            raise AgentCoreError(
                f"{provider} is temporarily rate limited.",
                provider=active_candidate.name,
                status_code=429,
                provider_code="RATE_LIMITED",
                provider_status="RATE_LIMITED",
            )
        await _record_provider_fallback(
            tool_event_recorder,
            requested_provider=provider,
            actual_provider=fallback.name,
            switch_reason="RATE_LIMITED_PRECHECK",
        )
        active_candidate = fallback

    preflight_context = await prepare_evidence_first_context(
        user_message,
        scope=scope,
        user_id=user_id,
        tool_event_recorder=tool_event_recorder,
    )
    model_message = user_message
    if preflight_context.model_context:
        model_message = f"{user_message}\n\n{preflight_context.model_context}"
    last_error: Exception | None = None
    for _ in range(GEMINI_429_MAX_RETRIES + 1):
        attempted_providers.add(active_candidate.name)
        agent = _build_agent(active_candidate.model, scope)
        attempt_started = time.perf_counter()
        try:
            evidence_envelopes: list[dict[str, Any]] = list(
                preflight_context.evidence_envelopes
            )
            result = await agent.run(
                model_message,
                message_history=message_history,
                deps=AgentDeps(
                    conversation_id=conversation_id,
                    run_id=run_id,
                    user_id=user_id,
                    scope=scope,
                    tool_event_recorder=tool_event_recorder,
                    evidence_envelopes=evidence_envelopes,
                ),
                conversation_id=conversation_id,
                run_id=run_id,
                usage_limits=UsageLimits(
                    request_limit=MAX_MODEL_REQUESTS,
                    tool_calls_limit=MAX_TOOL_CALLS,
                ),
            )
            evidence_report = render_evidence_report(evidence_envelopes)
            if evidence_report:
                original_output = remove_model_source_markers(str(result.output).strip())
                final_output = original_output + evidence_report
                result.output = final_output
                for message in result.new_messages():
                    for part in message.parts:
                        if getattr(part, "content", None) == original_output:
                            part.content = final_output
            _mark_provider_ready(
                active_candidate.name,  # type: ignore[arg-type]
                int((time.perf_counter() - attempt_started) * 1000)
            )
            return result
        except ModelHTTPError as error:
            last_error = error
            reason = _transient_provider_failure_reason(error)
            if reason is not None:
                if error.status_code == 429:
                    _mark_provider_429(active_candidate.name)  # type: ignore[arg-type]
                fallback = _ready_fallback(
                    candidates,
                    excluded=attempted_providers,
                )
                if fallback is not None:
                    await _record_provider_fallback(
                        tool_event_recorder,
                        requested_provider=provider,
                        actual_provider=fallback.name,
                        switch_reason=reason,
                    )
                    active_candidate = fallback
                    continue
                if len(attempted_providers) <= GEMINI_429_MAX_RETRIES:
                    await asyncio.sleep(
                        _retry_after_seconds(error, len(attempted_providers) - 1)
                    )
                    continue
            break
        except Exception as error:
            last_error = error
            break

    logger.warning(
        "Agent provider failed provider=%s error_type=%s",
        active_candidate.name,
        type(last_error).__name__ if last_error else "UnknownError",
    )

    provider_code = _provider_error_code(last_error) if last_error else None
    status_code = (
        last_error.status_code
        if isinstance(last_error, ModelHTTPError)
        else None
    )
    if status_code == 429:
        _open_rate_limit(active_candidate.name)  # type: ignore[arg-type]
        provider_status = "RATE_LIMITED"
        provider_code = "RATE_LIMITED"
    elif status_code in {401, 403} or provider_code == "oauth.v2.ApiKeyNotApproved":
        _provider_state_for(active_candidate.name).status = "UNAUTHORIZED"  # type: ignore[arg-type]
        provider_status = "UNAUTHORIZED"
    else:
        _provider_state_for(active_candidate.name).status = "ERROR"  # type: ignore[arg-type]
        provider_status = "ERROR"
    state = _provider_state_for(active_candidate.name)  # type: ignore[arg-type]
    state.last_failure = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    state.last_failure_code = provider_code or provider_status
    state.last_latency_ms = None
    raise AgentCoreError(
        "All configured model providers failed.",
        provider=active_candidate.name,
        status_code=status_code,
        provider_code=provider_code,
        provider_status=provider_status,
    ) from last_error