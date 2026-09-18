from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider

from agent_consts import (
    GEMINI_MODEL,
    OPENAI_MODEL,
    TRANSIENT_PROVIDER_STATUS_CODES,
    logger,
)
from config import GEMINI_429_CIRCUIT_THRESHOLD, GEMINI_429_COOLDOWN_SECONDS


@dataclass(frozen=True)
class ProviderCandidate:
    name: str
    model: Model
    model_name: str


@dataclass
class _ProviderState:
    consecutive_429: int = 0
    rate_limited_until: float = 0.0
    status: str = "READY"
    last_success: str | None = None
    last_failure: str | None = None
    last_failure_code: str | None = None
    last_latency_ms: int | None = None


_provider_states: dict[str, _ProviderState] = {
    "gemini": _ProviderState(),
    "openai": _ProviderState(),
}


def _provider_state_for(provider: ProviderName) -> _ProviderState:
    return _provider_states[provider]


def _provider_is_configured(provider: ProviderName) -> bool:
    if provider == "gemini":
        return bool(os.environ.get("GEMINI_API_KEY"))
    return bool(
        os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
        and os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
    )


def provider_model_name(provider: ProviderName) -> str:
    return GEMINI_MODEL if provider == "gemini" else OPENAI_MODEL


def provider_health(provider: ProviderName = "gemini") -> dict[str, object]:
    state = _provider_state_for(provider)
    model_name = provider_model_name(provider)
    if not _provider_is_configured(provider):
        return {
            "provider": provider,
            "model": model_name,
            "status": "NOT_CONFIGURED",
            "last_success": state.last_success,
            "last_failure": state.last_failure,
            "last_failure_code": state.last_failure_code,
            "latency_ms": state.last_latency_ms,
        }
    if state.status == "RATE_LIMITED":
        remaining = max(0.0, state.rate_limited_until - time.monotonic())
        if remaining > 0:
            return {
                "provider": provider,
                "model": model_name,
                "status": "RATE_LIMITED",
                "cooldown_remaining_seconds": round(remaining, 3),
                "last_success": state.last_success,
                "last_failure": state.last_failure,
                "last_failure_code": state.last_failure_code,
                "latency_ms": state.last_latency_ms,
            }
        state.status = "READY"
        state.consecutive_429 = 0
    return {
        "provider": provider,
        "model": model_name,
        "status": state.status,
        "last_success": state.last_success,
        "last_failure": state.last_failure,
        "last_failure_code": state.last_failure_code,
        "latency_ms": state.last_latency_ms,
    }


def all_provider_health() -> dict[str, dict[str, object]]:
    return {
        provider: provider_health(provider)
        for provider in ("gemini", "openai")
    }


def _mark_provider_ready(provider: ProviderName, latency_ms: int) -> None:
    state = _provider_state_for(provider)
    state.consecutive_429 = 0
    state.rate_limited_until = 0.0
    state.status = "READY"
    state.last_success = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    state.last_failure = None
    state.last_failure_code = None
    state.last_latency_ms = latency_ms


def _mark_provider_429(provider: ProviderName) -> None:
    state = _provider_state_for(provider)
    state.consecutive_429 += 1
    if state.consecutive_429 >= GEMINI_429_CIRCUIT_THRESHOLD:
        _open_rate_limit(provider)


def _open_rate_limit(provider: ProviderName) -> None:
    state = _provider_state_for(provider)
    state.status = "RATE_LIMITED"
    state.rate_limited_until = (
        time.monotonic() + GEMINI_429_COOLDOWN_SECONDS
    )


# Lazily-built provider model instances, keyed by (provider, credentials).
# Each model owns an httpx client; building one per request churns TCP/TLS
# connections for zero benefit, so instances are shared for the process
# lifetime. A credential change (key rotation) yields a new cache key, and the
# stale entry for that provider is evicted so old keys are not kept alive.
_model_cache: dict[tuple[str, str, str | None], Model] = {}


def _clear_model_cache() -> None:
    _model_cache.clear()


_ModelT = TypeVar("_ModelT", bound=Model)


def _cached_model(
    cache_key: tuple[str, str, str | None],
    build: Callable[[], _ModelT],
) -> _ModelT:
    model = _model_cache.get(cache_key)
    if model is not None:
        return model
    model = build()
    for stale_key in [
        key for key in _model_cache if key[0] == cache_key[0] and key != cache_key
    ]:
        del _model_cache[stale_key]
    _model_cache[cache_key] = model
    return model


def _gemini_model(api_key: str) -> GoogleModel:
    return _cached_model(
        ("gemini", api_key, None),
        lambda: GoogleModel(
            GEMINI_MODEL,
            provider=GoogleProvider(api_key=api_key),
        ),
    )


def _openai_model(api_key: str, base_url: str) -> OpenAIChatModel:
    return _cached_model(
        ("openai", api_key, base_url),
        lambda: OpenAIChatModel(
            OPENAI_MODEL,
            provider=OpenAIProvider(api_key=api_key, base_url=base_url),
        ),
    )


def _configured_providers() -> list[ProviderCandidate]:
    providers: list[ProviderCandidate] = []
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        providers.append(
            ProviderCandidate(
                name="gemini",
                model=_gemini_model(gemini_key),
                model_name=GEMINI_MODEL,
            )
        )
    openai_key = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
    openai_base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
    if openai_key and openai_base_url:
        providers.append(
            ProviderCandidate(
                name="openai",
                model=_openai_model(openai_key, openai_base_url),
                model_name=OPENAI_MODEL,
            )
        )
    return providers


def configured_provider_names() -> list[str]:
    return [candidate.name for candidate in _configured_providers()]


def _provider_error_code(error: Exception) -> str | None:
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        error_body = body.get("error")
        if isinstance(error_body, dict) and isinstance(error_body.get("code"), str):
            return error_body["code"]
        fault = body.get("fault")
        if isinstance(fault, dict):
            detail = fault.get("detail")
            if isinstance(detail, dict) and isinstance(detail.get("errorcode"), str):
                return detail["errorcode"]
    raw_error = str(error)
    if "ApiKeyNotApproved" in raw_error:
        return "oauth.v2.ApiKeyNotApproved"
    return None


def _transient_provider_failure_reason(error: Exception) -> str | None:
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return "RATE_LIMITED"
    if status_code in TRANSIENT_PROVIDER_STATUS_CODES:
        return f"HTTP_{status_code}"
    return None


def _ready_fallback(
    candidates: Sequence[ProviderCandidate],
    *,
    excluded: set[str],
) -> ProviderCandidate | None:
    for candidate in candidates:
        if candidate.name in excluded:
            continue
        if provider_health(candidate.name)["status"] == "READY":  # type: ignore[arg-type]
            return candidate
    return None


async def _record_provider_fallback(
    tool_event_recorder: (
        Callable[[str, str, int, dict[str, Any] | None], Awaitable[None]] | None
    ),
    *,
    requested_provider: str,
    actual_provider: str,
    switch_reason: str,
) -> None:
    if tool_event_recorder is None:
        return
    try:
        await tool_event_recorder(
            "model_provider_fallback",
            "success",
            0,
            {
                "requested_provider": requested_provider,
                "actual_provider": actual_provider,
                "switch_reason": switch_reason,
            },
        )
    except Exception:
        logger.warning("Provider fallback trace persistence failed")


