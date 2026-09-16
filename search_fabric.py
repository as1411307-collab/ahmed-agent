from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from typing import Any, Awaitable, Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from config import (
    BRAVE_SEARCH_ENABLED,
    SEARCH_MAX_CONCURRENCY,
    SEARCH_MAX_PROVIDER_CALLS,
    SEARCH_MAX_RESULTS_PER_PROVIDER,
    SEARCH_PROVIDER_TIMEOUT_SECONDS,
    SEARCH_RRF_K,
)


logger = logging.getLogger("ahmed_agent.search_fabric")
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
TRACKING_PARAMETER_PREFIXES = ("utm_",)
TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "referrer",
}
OFFICIAL_DOCUMENTATION_HOSTS = {
    "docs.python.org",
    "python.org",
    "modelcontextprotocol.io",
    "openai.com",
    "platform.openai.com",
    "docs.anthropic.com",
    "docs.github.com",
}
COMMUNITY_HOSTS = {"reddit.com", "x.com", "stackoverflow.com"}
RECENCY_MARKERS = (
    "latest",
    "current",
    "today",
    "recent",
    "this week",
    "هذا الأسبوع",
    "اليوم",
    "آخر",
    "الحالي",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_date(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    if not re.search(r"\b(?:19|20)\d{2}\b", value):
        return None
    return value


def normalize_url(url: str) -> str:
    """Normalize identity-changing URL parts conservatively."""
    parsed = urlparse(url.strip())
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        return ""
    hostname = parsed.hostname.casefold()
    try:
        port = parsed.port
    except ValueError:
        return ""
    default_port = (parsed.scheme.casefold() == "http" and port == 80) or (
        parsed.scheme.casefold() == "https" and port == 443
    )
    netloc = hostname
    if port and not default_port:
        netloc = f"{netloc}:{port}"
    path = parsed.path.rstrip("/") or "/"
    query_items = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        lowered = key.casefold()
        if lowered in TRACKING_PARAMETERS or any(
            lowered.startswith(prefix) for prefix in TRACKING_PARAMETER_PREFIXES
        ):
            continue
        query_items.append((key, value))
    query = urlencode(sorted(query_items))
    return urlunparse(
        (parsed.scheme.casefold(), netloc, path, "", query, "")
    )


def domain_for_url(url: str) -> str:
    hostname = (urlparse(url).hostname or "").casefold()
    return hostname.removeprefix("www.")


def _is_safe_result_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme.casefold() in {"http", "https"}
        and bool(parsed.hostname)
        and not parsed.username
        and not parsed.password
    )


def classify_source(url: str, title: str, query_type: str = "general_web") -> tuple[str, str]:
    domain = domain_for_url(url)
    lowered_title = title.casefold()
    if domain in OFFICIAL_DOCUMENTATION_HOSTS or domain.endswith(".gov"):
        return "official_documentation", "primary"
    if domain.endswith(".gov") or ".gov." in domain or domain.endswith(".int"):
        return "government", "primary"
    if domain == "github.com" or domain.endswith(".github.io"):
        official_markers = ("official", "python", "modelcontextprotocol", "openai")
        return (
            ("official_repository", "primary")
            if any(marker in url.casefold() or marker in lowered_title for marker in official_markers)
            else ("repository", "secondary")
        )
    if domain in COMMUNITY_HOSTS or any(
        marker in domain for marker in ("forum", "community", "reddit")
    ):
        return "social_community", "community"
    if any(marker in domain for marker in ("news", "reuters", "apnews", "bbc")):
        return "news", "secondary"
    if query_type == "academic" and any(
        marker in domain
        for marker in ("arxiv", "nature", "springer", "sciencedirect", "pubmed")
    ):
        return "academic", "secondary"
    return "general_web", "unknown"


@dataclass(frozen=True)
class SearchOptions:
    mode: str = "FAST"
    max_results: int = SEARCH_MAX_RESULTS_PER_PROVIDER
    query_type: str = "general_web"
    timeout_seconds: float = SEARCH_PROVIDER_TIMEOUT_SECONDS
    provider_call_budget: int = SEARCH_MAX_PROVIDER_CALLS


@dataclass
class NormalizedSearchResult:
    provider: str
    title: str
    url: str
    canonical_url: str
    domain: str
    snippet: str
    published_at: str | None
    language: str | None
    provider_rank: int
    provider_score: float | None
    retrieved_at: str
    source_type: str
    source_class: str
    found_by: list[str] = field(default_factory=list)
    source_family: str | None = None
    content: str = ""
    provider_ranks: dict[str, int] = field(default_factory=dict)
    fusion_score: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "title": self.title,
            "url": self.url,
            "original_url": self.url,
            "canonical_url": self.canonical_url,
            "domain": self.domain,
            "snippet": self.snippet,
            "content": self.content or self.snippet,
            "published_at": self.published_at,
            "date": self.published_at,
            "language": self.language,
            "provider_rank": self.provider_rank,
            "provider_score": self.provider_score,
            "provider_ranks": dict(self.provider_ranks or {self.provider: self.provider_rank}),
            "retrieved_at": self.retrieved_at,
            "source_type": self.source_type,
            "source_class": self.source_class,
            "classification": self.source_class,
            "primary_or_secondary": self.source_class,
            "found_by": list(self.found_by or [self.provider]),
            "source_family": self.source_family or self.domain,
            "fusion_score": self.fusion_score,
            "tavily_score": self.provider_score if self.provider == "tavily" else None,
        }


@dataclass
class ProviderSearchOutput:
    results: list[NormalizedSearchResult]
    metrics: dict[str, object]
    legacy: dict[str, object] | None = None


class SearchProvider(Protocol):
    name: str

    @property
    def status(self) -> str: ...

    def capabilities(self) -> dict[str, object]: ...

    async def healthcheck(self) -> dict[str, object]: ...

    async def search(self, query: str, options: SearchOptions) -> ProviderSearchOutput: ...

    def usage(self) -> dict[str, object]: ...


TavilySearch = Callable[..., Awaitable[dict[str, object]]]
TavilyHealth = Callable[[], dict[str, object]]


class TavilyProvider:
    name = "tavily"

    def __init__(self, search_callable: TavilySearch, health_callable: TavilyHealth) -> None:
        self._search_callable = search_callable
        self._health_callable = health_callable
        self._last_usage: dict[str, object] = {}

    @property
    def status(self) -> str:
        return str(self._health_callable().get("status", "ERROR"))

    def capabilities(self) -> dict[str, object]:
        return {
            "search": True,
            "independent_index": False,
            "page_extraction": True,
            "modes": ["FAST", "DEEP"],
        }

    async def healthcheck(self) -> dict[str, object]:
        return dict(self._health_callable())

    def usage(self) -> dict[str, object]:
        return dict(self._last_usage)

    async def search(self, query: str, options: SearchOptions) -> ProviderSearchOutput:
        started_at = time.perf_counter()
        legacy = await self._search_callable(
            query=query,
            mode=options.mode,
            max_results=options.max_results,
        )
        raw_results = legacy.get("results", []) if isinstance(legacy, dict) else []
        results: list[NormalizedSearchResult] = []
        if isinstance(raw_results, list):
            for rank, item in enumerate(raw_results, 1):
                if not isinstance(item, dict):
                    continue
                url = str(item.get("original_url") or item.get("url") or "")
                canonical_url = normalize_url(url)
                if not canonical_url or not _is_safe_result_url(url):
                    continue
                title = str(item.get("title") or url)
                source_type, source_class = classify_source(
                    url,
                    title,
                    options.query_type,
                )
                declared_class = str(item.get("source_class") or item.get("classification") or "")
                if declared_class in {"primary", "secondary", "community", "unknown"}:
                    source_class = declared_class
                score = item.get("provider_score", item.get("tavily_score"))
                try:
                    numeric_score = float(score) if score is not None else None
                except (TypeError, ValueError):
                    numeric_score = None
                results.append(
                    NormalizedSearchResult(
                        provider=self.name,
                        title=title,
                        url=url,
                        canonical_url=canonical_url,
                        domain=domain_for_url(url),
                        snippet=str(item.get("snippet") or item.get("content") or ""),
                        content=str(item.get("content") or item.get("snippet") or ""),
                        published_at=_safe_date(item.get("published_at") or item.get("date")),
                        language=str(item.get("language")) if item.get("language") else None,
                        provider_rank=rank,
                        provider_score=numeric_score,
                        retrieved_at=_utc_now(),
                        source_type=str(item.get("source_type") or source_type),
                        source_class=source_class,
                        found_by=[self.name],
                        source_family=domain_for_url(url),
                    )
                )
        self._last_usage = {
            "provider": self.name,
            "request_count": int(legacy.get("search_calls", 0)) if isinstance(legacy, dict) else 0,
            "latency_ms": round((time.perf_counter() - started_at) * 1000),
            "result_count": len(results),
            "status": "success" if legacy.get("ok") else "error",
            "credits_used": legacy.get("credits_used") if isinstance(legacy, dict) else None,
        }
        return ProviderSearchOutput(results=results, metrics=self._last_usage, legacy=legacy)


class BraveProvider:
    name = "brave"

    def __init__(self) -> None:
        self._state: dict[str, object] = {
            "last_success": None,
            "last_failure": None,
            "last_failure_code": None,
            "latency_ms": None,
            "request_count": 0,
            "result_count": 0,
            "status": None,
        }

    @property
    def api_key(self) -> str:
        return os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()

    @property
    def status(self) -> str:
        if not self.api_key:
            return "NOT_CONFIGURED"
        return str(self._state.get("status") or "READY")

    def capabilities(self) -> dict[str, object]:
        return {
            "search": True,
            "independent_index": True,
            "page_extraction": False,
            "modes": ["DEEP", "BENCHMARK"],
        }

    async def healthcheck(self) -> dict[str, object]:
        return {
            "status": self.status,
            "last_success": self._state["last_success"],
            "last_failure": self._state["last_failure"],
            "latency_ms": self._state["latency_ms"],
            "request_count": self._state["request_count"],
            "result_count": self._state["result_count"],
            "safe_error_code": (
                self._state["last_failure_code"] if self.status != "READY" else None
            ),
        }

    def usage(self) -> dict[str, object]:
        return {
            "provider": self.name,
            "request_count": self._state["request_count"],
            "result_count": self._state["result_count"],
            "latency_ms": self._state["latency_ms"],
            "status": self.status,
        }

    def _search_sync(self, query: str, options: SearchOptions) -> list[dict[str, object]]:
        request_url = f"{BRAVE_SEARCH_URL}?{urlencode({'q': query, 'count': options.max_results})}"
        request = UrlRequest(
            request_url,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
                "User-Agent": "Ahmed-Agent/1.0",
            },
            method="GET",
        )
        with urlopen(request, timeout=options.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
        if not isinstance(payload, dict):
            raise ValueError("Brave returned a non-object response.")
        web = payload.get("web")
        raw_results = web.get("results", []) if isinstance(web, dict) else []
        return [item for item in raw_results if isinstance(item, dict)] if isinstance(raw_results, list) else []

    async def search(self, query: str, options: SearchOptions) -> ProviderSearchOutput:
        if self.status == "NOT_CONFIGURED":
            return ProviderSearchOutput(
                results=[],
                metrics={"provider": self.name, "status": "NOT_CONFIGURED", "request_count": 0},
            )
        started_at = time.perf_counter()
        try:
            raw_results = await asyncio.to_thread(self._search_sync, query, options)
        except HTTPError as error:
            code = (
                "BRAVE_UNAUTHORIZED"
                if error.code in {401, 403}
                else "BRAVE_RATE_LIMITED"
                if error.code == 429
                else f"BRAVE_HTTP_{error.code}"
            )
            self._mark_failure(code, started_at, "RATE_LIMITED" if error.code == 429 else "UNAUTHORIZED" if error.code in {401, 403} else "ERROR")
            return ProviderSearchOutput(results=[], metrics=self.usage())
        except (OSError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
            del error
            self._mark_failure("BRAVE_REQUEST_ERROR", started_at, "ERROR")
            return ProviderSearchOutput(results=[], metrics=self.usage())

        results: list[NormalizedSearchResult] = []
        for rank, item in enumerate(raw_results[: options.max_results], 1):
            url = str(item.get("url") or "")
            canonical_url = normalize_url(url)
            if not canonical_url or not _is_safe_result_url(url):
                continue
            title = str(item.get("title") or url)
            source_type, source_class = classify_source(url, title, options.query_type)
            published_at = _safe_date(
                item.get("published") or item.get("published_date") or item.get("page_age")
            )
            snippet = str(item.get("description") or item.get("snippet") or "")
            results.append(
                NormalizedSearchResult(
                    provider=self.name,
                    title=title,
                    url=url,
                    canonical_url=canonical_url,
                    domain=domain_for_url(url),
                    snippet=snippet,
                    content=snippet,
                    published_at=published_at,
                    language=str(item.get("language")) if item.get("language") else None,
                    provider_rank=rank,
                    provider_score=None,
                    retrieved_at=_utc_now(),
                    source_type=source_type,
                    source_class=source_class,
                    found_by=[self.name],
                    source_family=domain_for_url(url),
                )
            )
        self._state.update(
            {
                "status": "READY",
                "last_success": _utc_now(),
                "last_failure_code": None,
                "latency_ms": round((time.perf_counter() - started_at) * 1000),
                "request_count": int(self._state["request_count"]) + 1,
                "result_count": len(results),
            }
        )
        return ProviderSearchOutput(results=results, metrics=self.usage())

    def _mark_failure(self, code: str, started_at: float, status: str) -> None:
        self._state.update(
            {
                "status": status,
                "last_failure": _utc_now(),
                "last_failure_code": code,
                "latency_ms": round((time.perf_counter() - started_at) * 1000),
                "request_count": int(self._state["request_count"]) + 1,
                "result_count": 0,
            }
        )


def _query_type_from_results(query: str) -> str:
    lowered = query.casefold()
    if any(marker in lowered for marker in ("official", "documentation", "رسمي", "توثيق")):
        return "official"
    if any(marker in lowered for marker in ("latest", "current", "today", "recent", "اليوم", "آخر")):
        return "current_news"
    if any(marker in lowered for marker in ("research", "paper", "study", "دراسة", "بحث")):
        return "academic"
    return "general_web"


def _recent_bonus(published_at: str | None, query: str) -> float:
    if not published_at or not any(marker in query.casefold() for marker in RECENCY_MARKERS):
        return 0.0
    try:
        parsed = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(published_at)
        except (TypeError, ValueError, IndexError):
            return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds() / 86400)
    return 0.04 if age_days <= 30 else 0.01 if age_days <= 365 else 0.0


def _title_similarity(left: str, right: str) -> float:
    left_tokens = " ".join(re.findall(r"\w+", left.casefold()))
    right_tokens = " ".join(re.findall(r"\w+", right.casefold()))
    return SequenceMatcher(None, left_tokens, right_tokens).ratio()


def _merge_results(
    results: list[NormalizedSearchResult],
    query: str,
    max_results: int,
) -> tuple[list[NormalizedSearchResult], int]:
    groups: list[list[NormalizedSearchResult]] = []
    by_url: dict[str, list[NormalizedSearchResult]] = {}
    for result in results:
        exact_group = by_url.get(result.canonical_url)
        if exact_group is not None:
            exact_group.append(result)
            continue
        near_group = next(
            (
                group
                for group in groups
                if group
                and group[0].domain == result.domain
                and _title_similarity(group[0].title, result.title) >= 0.92
            ),
            None,
        )
        if near_group is None:
            near_group = []
            groups.append(near_group)
        near_group.append(result)
        by_url.setdefault(result.canonical_url, near_group)

    fused: list[NormalizedSearchResult] = []
    query_type = _query_type_from_results(query)
    for group in groups:
        representative = max(
            group,
            key=lambda item: (
                len(item.content),
                item.source_class == "primary",
                -item.provider_rank,
            ),
        )
        provider_ranks: dict[str, int] = {}
        for item in group:
            provider_ranks[item.provider] = min(
                item.provider_rank,
                provider_ranks.get(item.provider, item.provider_rank),
            )
        rrf_score = sum(1.0 / (SEARCH_RRF_K + rank) for rank in provider_ranks.values())
        quality_bonus = 0.03 if representative.source_class == "primary" else 0.0
        if query_type == "official" and representative.source_class == "primary":
            quality_bonus += 0.12
        quality_bonus += _recent_bonus(representative.published_at, query)
        found_by = list(provider_ranks)
        fused.append(
            NormalizedSearchResult(
                provider=representative.provider,
                title=representative.title,
                url=representative.url,
                canonical_url=representative.canonical_url,
                domain=representative.domain,
                snippet=representative.snippet,
                content=representative.content,
                published_at=representative.published_at,
                language=representative.language,
                provider_rank=min(provider_ranks.values()),
                provider_score=representative.provider_score,
                retrieved_at=representative.retrieved_at,
                source_type=representative.source_type,
                source_class=representative.source_class,
                found_by=found_by,
                source_family=representative.source_family,
                provider_ranks=provider_ranks,
                fusion_score=rrf_score + quality_bonus,
            )
        )
    fused.sort(
        key=lambda item: (
            -(item.fusion_score or 0.0),
            item.source_class != "primary",
            item.provider_rank,
            item.canonical_url,
        )
    )
    return fused[:max_results], len(groups)


class SearchFabric:
    def __init__(self, tavily: SearchProvider, brave: SearchProvider) -> None:
        self.tavily = tavily
        self.brave = brave

    def health(self) -> dict[str, object]:
        tavily = self.tavily._health_callable() if isinstance(self.tavily, TavilyProvider) else {"status": self.tavily.status}
        brave = {
            "status": self.brave.status,
            **self.brave.usage(),
        }
        return {
            "status": "READY" if tavily.get("status") == "READY" else str(tavily.get("status", "ERROR")),
            "providers": {
                "tavily": tavily,
                "brave": brave,
            },
            "budgets": {
                "provider_timeout_seconds": SEARCH_PROVIDER_TIMEOUT_SECONDS,
                "max_provider_calls": SEARCH_MAX_PROVIDER_CALLS,
                "max_concurrency": SEARCH_MAX_CONCURRENCY,
                "max_results_per_provider": SEARCH_MAX_RESULTS_PER_PROVIDER,
                "rrf_k": SEARCH_RRF_K,
            },
            "brave_enabled_by_default": BRAVE_SEARCH_ENABLED,
        }

    def _should_route_brave(self, query: str, mode: str) -> bool:
        if mode != "DEEP" or not BRAVE_SEARCH_ENABLED or self.brave.status != "READY":
            return False
        lowered = query.casefold()
        return (
            any("\u0600" <= character <= "\u06ff" for character in query)
            or any(
                marker in lowered
                for marker in ("independent", "coverage", "regional", "niche", "compare", "مقارنة")
            )
        )

    async def search(
        self,
        query: str,
        *,
        mode: str,
        max_results: int,
        tavily_legacy_search: TavilySearch,
    ) -> dict[str, object]:
        query_type = _query_type_from_results(query)
        options = SearchOptions(
            mode=mode,
            max_results=min(max_results, SEARCH_MAX_RESULTS_PER_PROVIDER),
            query_type=query_type,
        )
        use_brave = self._should_route_brave(query, mode)
        if mode == "FAST":
            tavily_output = await self.tavily.search(query, options)
            return self._legacy_response(tavily_output, mode=mode, providers=["tavily"])

        if use_brave:
            tavily_output, brave_output = await asyncio.gather(
                self.tavily.search(query, options),
                self.brave.search(query, options),
            )
        else:
            tavily_output = await self.tavily.search(query, options)
            brave_output = ProviderSearchOutput([], {"provider": "brave", "status": "not_selected"})

        tavily_ok = bool(tavily_output.legacy and tavily_output.legacy.get("ok"))
        brave_ok = bool(brave_output.results)
        if not tavily_ok and not use_brave and self.brave.status == "READY":
            brave_output = await self.brave.search(query, options)
            brave_ok = bool(brave_output.results)
            use_brave = True
        if use_brave and brave_ok:
            merged, unique_count = _merge_results(
                tavily_output.results + brave_output.results,
                query,
                min(max_results, SEARCH_MAX_RESULTS_PER_PROVIDER),
            )
            if merged:
                response = dict(tavily_output.legacy or {})
                response.update(
                    {
                        "ok": True,
                        "query": query,
                        "mode": mode,
                        "query_type": query_type,
                        "providers_used": ["tavily"] + (["brave"] if brave_ok else []),
                        "brave_used": True,
                        "fusion": "rrf",
                        "results": [item.to_dict() for item in merged],
                        "candidate_count": len(tavily_output.results) + len(brave_output.results),
                        "deduplicated_count": unique_count,
                        "provider_request_count": 1 + int(brave_output.metrics.get("request_count", 0) or 0),
                        "provider_metrics": {
                            "tavily": tavily_output.metrics,
                            "brave": brave_output.metrics,
                        },
                    }
                )
                return response
        if tavily_output.legacy is not None:
            response = dict(tavily_output.legacy)
            response.update(
                {
                    "providers_used": ["tavily"],
                    "brave_used": False,
                    "fusion": "single_provider",
                    "provider_metrics": {"tavily": tavily_output.metrics},
                }
            )
            return response
        if brave_ok:
            return {
                "ok": True,
                "query": query,
                "mode": mode,
                "query_type": query_type,
                "providers_used": ["brave"],
                "brave_used": True,
                "fusion": "single_provider",
                "results": [item.to_dict() for item in brave_output.results[:max_results]],
                "provider_metrics": {"brave": brave_output.metrics},
            }
        return {
            "ok": False,
            "query": query,
            "mode": mode,
            "results": [],
            "providers_used": [],
            "brave_used": False,
            "error": "تعذر تنفيذ البحث الآن. حاول مرة أخرى لاحقًا.",
        }

    @staticmethod
    def _legacy_response(
        output: ProviderSearchOutput,
        *,
        mode: str,
        providers: list[str],
    ) -> dict[str, object]:
        response = dict(output.legacy or {})
        response.update(
            {
                "mode": mode,
                "providers_used": providers,
                "brave_used": False,
                "fusion": "single_provider",
                "results": [item.to_dict() for item in output.results],
                "provider_metrics": {"tavily": output.metrics},
            }
        )
        return response


async def run_benchmark(
    fabric: SearchFabric,
    fixtures: list[dict[str, object]],
) -> dict[str, object]:
    if fabric.brave.status != "READY":
        return {
            "status": "BLOCKED_BY_CONFIGURATION",
            "reason": "BRAVE_SEARCH_API_KEY is not configured.",
            "queries": len(fixtures),
        }
    rows: list[dict[str, object]] = []
    for fixture in fixtures:
        query = str(fixture.get("query") or "")
        options = SearchOptions(mode="DEEP", max_results=SEARCH_MAX_RESULTS_PER_PROVIDER, query_type=_query_type_from_results(query))
        started_at = time.perf_counter()
        tavily = await fabric.tavily.search(query, options)
        tavily_latency = round((time.perf_counter() - started_at) * 1000)
        started_at = time.perf_counter()
        brave = await fabric.brave.search(query, options)
        brave_latency = round((time.perf_counter() - started_at) * 1000)
        fused, _ = _merge_results(tavily.results + brave.results, query, options.max_results)
        tavily_metrics = _benchmark_metrics(
            tavily.results,
            tavily_latency,
            tavily.metrics,
        )
        brave_metrics = _benchmark_metrics(
            brave.results,
            brave_latency,
            brave.metrics,
        )
        fused_metrics = _benchmark_metrics(
            fused,
            tavily_latency + brave_latency,
            {
                "request_count": (
                    int(tavily.metrics.get("request_count", 0) or 0)
                    + int(brave.metrics.get("request_count", 0) or 0)
                )
            },
        )
        rows.append(
            {
                "query": query,
                "tavily_only": tavily_metrics,
                "brave_only": brave_metrics,
                "fused": fused_metrics,
                "expected_domains": fixture.get("expected_domains", []),
            }
        )
    additional_unique = sum(
        max(0, int(row["fused"]["useful_unique_urls"]) - int(row["tavily_only"]["useful_unique_urls"]))
        for row in rows
    )
    additional_primary = sum(
        max(0, int(row["fused"]["primary_sources"]) - int(row["tavily_only"]["primary_sources"]))
        for row in rows
    )
    return {
        "status": "PASS",
        "queries": rows,
        "additional_useful_unique_sources": additional_unique,
        "additional_primary_sources": additional_primary,
        "recommendation": (
            "ENABLE_BRAVE_FOR_DEEP"
            if additional_unique or additional_primary
            else "KEEP_BRAVE_DISABLED_BY_DEFAULT"
        ),
    }


def _benchmark_metrics(
    results: list[NormalizedSearchResult],
    latency_ms: int,
    usage: dict[str, object] | None = None,
) -> dict[str, object]:
    unique_urls = {item.canonical_url for item in results}
    return {
        "useful_unique_urls": len(unique_urls),
        "primary_sources": sum(item.source_class == "primary" for item in results),
        "duplicate_rate": round(1 - len(unique_urls) / max(1, len(results)), 4),
        "source_diversity": len({item.source_family or item.domain for item in results}),
        "latency_ms": latency_ms,
        "result_count": len(results),
        "request_count": int((usage or {}).get("request_count", 0) or 0),
        "credits_used": (usage or {}).get("credits_used"),
    }