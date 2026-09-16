from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from config import (
    GITHUB_API_TIMEOUT_SECONDS,
    GITHUB_MAX_RETRIES,
    GITHUB_TOKEN,
)


logger = logging.getLogger("ahmed_agent.github_search")

GITHUB_API_URL = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"
MAX_QUERY_LENGTH = 256
MAX_RESULTS = 10
GITHUB_INTENTS = {
    "auto",
    "repository",
    "repository_lookup",
    "issue",
    "issue_lookup",
    "release",
    "releases",
    "latest_release",
}
REPO_PATH_PATTERN = re.compile(
    r"(?<![\w.-])([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)(?![\w.-])"
)
OWNER_REPO_PART = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")


class GitHubProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider_code: str | None = None,
        retry_after: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider_code = provider_code
        self.retry_after = retry_after
        self.headers = headers or {}


@dataclass(frozen=True)
class _GitHubResponse:
    payload: Any
    status_code: int
    headers: dict[str, str]
    retries: int
    latency_ms: int
    bucket: str


@dataclass
class _RateLimit:
    limit: int | None = None
    remaining: int | None = None
    reset_at: int | None = None

    def update(self, headers: dict[str, str]) -> None:
        self.limit = _int_header(headers, "x-ratelimit-limit", self.limit)
        self.remaining = _int_header(
            headers,
            "x-ratelimit-remaining",
            self.remaining,
        )
        self.reset_at = _int_header(headers, "x-ratelimit-reset", self.reset_at)

    def to_dict(self) -> dict[str, object]:
        return {
            "limit": self.limit,
            "remaining": self.remaining,
            "reset_at": self.reset_at,
        }


@dataclass
class _ProviderState:
    status: str = "READY"
    last_success: str | None = None
    last_failure: str | None = None
    last_failure_code: str | None = None
    latency_ms: int | None = None
    request_count: int = 0
    result_count: int = 0
    retry_count: int = 0
    core: _RateLimit = field(default_factory=_RateLimit)
    search: _RateLimit = field(default_factory=_RateLimit)

    def success(self, response: _GitHubResponse, result_count: int) -> None:
        self.status = "READY"
        self.last_success = _utc_now()
        self.last_failure = None
        self.last_failure_code = None
        self.latency_ms = response.latency_ms
        self.request_count += 1
        self.result_count = result_count
        self.retry_count += response.retries
        self._rate_limit(response.bucket).update(response.headers)

    def failure(
        self,
        error: GitHubProviderError,
        *,
        bucket: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        if error.status_code == 401:
            self.status = "UNAUTHORIZED"
        elif error.status_code in {403, 429} and (
            error.provider_code == "RATE_LIMITED"
            or error.status_code == 429
        ):
            self.status = "RATE_LIMITED"
        else:
            self.status = "ERROR"
        self.last_failure = _utc_now()
        self.last_failure_code = error.provider_code or "REQUEST_ERROR"
        self.latency_ms = None
        self.request_count += 1
        self._rate_limit(bucket).update(headers or {})

    def _rate_limit(self, bucket: str) -> _RateLimit:
        return self.search if bucket == "search" else self.core

    def health(self, *, authenticated: bool) -> dict[str, object]:
        status = self.status
        if status == "READY" and not authenticated:
            status = "READY_UNAUTHENTICATED"
        return {
            "status": status,
            "auth_mode": "authenticated" if authenticated else "unauthenticated",
            "last_success": self.last_success,
            "last_failure": self.last_failure,
            "latency_ms": self.latency_ms,
            "request_count": self.request_count,
            "result_count": self.result_count,
            "retry_count": self.retry_count,
            "core_rate_limit": self.core.to_dict(),
            "search_rate_limit": self.search.to_dict(),
            "safe_error_code": self.last_failure_code
            if status not in {"READY", "READY_UNAUTHENTICATED"}
            else None,
        }


@dataclass
class GitHubResult:
    result_type: str
    owner: str | None
    repo: str | None
    full_name: str | None
    description: str | None
    html_url: str | None
    default_branch: str | None
    stars: int | None
    forks: int | None
    language: str | None
    topics: list[str]
    archived: bool | None
    updated_at: str | None
    pushed_at: str | None
    license: dict[str, object] | None
    visibility: str | None
    issue_number: int | None
    title: str | None
    state: str | None
    labels: list[dict[str, object]]
    author: dict[str, object] | None
    created_at: str | None
    comments_count: int | None
    body_text: str | None
    tag_name: str | None
    release_name: str | None
    published_at: str | None
    prerelease: bool | None
    draft: bool | None
    assets: list[dict[str, object]]
    provider: str
    retrieved_at: str

    @property
    def identity(self) -> str | None:
        if not self.owner or not self.repo:
            return None
        full_name = f"{self.owner}/{self.repo}"
        if self.result_type == "REPOSITORY":
            return full_name
        if self.result_type in {"ISSUE", "PULL_REQUEST"} and self.issue_number is not None:
            return f"{full_name}#{self.issue_number}"
        if self.result_type == "RELEASE" and self.tag_name:
            return f"{full_name}@{self.tag_name}"
        return full_name

    def to_dict(self) -> dict[str, object]:
        return {
            "result_type": self.result_type,
            "identity": self.identity,
            "owner": self.owner,
            "repo": self.repo,
            "full_name": self.full_name,
            "description": self.description,
            "html_url": self.html_url,
            "default_branch": self.default_branch,
            "stars": self.stars,
            "forks": self.forks,
            "language": self.language,
            "topics": self.topics,
            "archived": self.archived,
            "updated_at": self.updated_at,
            "pushed_at": self.pushed_at,
            "license": self.license,
            "visibility": self.visibility,
            "issue_number": self.issue_number,
            "title": self.title,
            "state": self.state,
            "labels": self.labels,
            "author": self.author,
            "created_at": self.created_at,
            "comments_count": self.comments_count,
            "body_text": self.body_text,
            "tag_name": self.tag_name,
            "release_name": self.release_name,
            "published_at": self.published_at,
            "prerelease": self.prerelease,
            "draft": self.draft,
            "assets": self.assets,
            "provider": self.provider,
            "retrieved_at": self.retrieved_at,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _int_header(
    headers: dict[str, str],
    name: str,
    fallback: int | None,
) -> int | None:
    value = headers.get(name) or headers.get(name.title())
    try:
        return int(value) if value is not None else fallback
    except (TypeError, ValueError):
        return fallback


def _retry_after(headers: dict[str, str]) -> float | None:
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _reset_delay(headers: dict[str, str]) -> float | None:
    reset_at = _int_header(headers, "x-ratelimit-reset", None)
    if reset_at is None:
        return None
    return max(0.0, reset_at - time.time())


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def _http_json_sync(
    url: str,
    headers: dict[str, str],
    timeout: float,
) -> tuple[Any, int, dict[str, str]]:
    request = Request(
        url,
        headers={"User-Agent": "Ahmed-Agent/1.0", **headers},
        method="GET",
    )
    with urlopen(request, timeout=timeout) as response:
        body = response.read(4 * 1024 * 1024)
        payload = json.loads(body.decode("utf-8", errors="replace"))
        return (
            payload,
            int(response.status),
            {str(key).lower(): str(value) for key, value in response.headers.items()},
        )


async def _http_json(
    url: str,
    *,
    bucket: str,
    timeout: float = GITHUB_API_TIMEOUT_SECONDS,
) -> _GitHubResponse:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "api.github.com":
        raise GitHubProviderError(
            "GitHub provider URL is not safe.",
            provider_code="UNSAFE_PROVIDER_URL",
        )
    started_at = time.perf_counter()
    for attempt in range(GITHUB_MAX_RETRIES + 1):
        try:
            payload, status_code, response_headers = await asyncio.to_thread(
                _http_json_sync,
                url,
                _headers(),
                timeout,
            )
            return _GitHubResponse(
                payload=payload,
                status_code=status_code,
                headers=response_headers,
                retries=attempt,
                latency_ms=round((time.perf_counter() - started_at) * 1000),
                bucket=bucket,
            )
        except HTTPError as error:
            response_headers = {
                str(key).lower(): str(value)
                for key, value in error.headers.items()
            }
            rate_limited = error.code == 429 or (
                error.code == 403
                and _int_header(response_headers, "x-ratelimit-remaining", 1) == 0
            )
            retryable = rate_limited or error.code >= 500
            if retryable and attempt < GITHUB_MAX_RETRIES:
                delay = _retry_after(response_headers)
                if delay is None and rate_limited:
                    delay = _reset_delay(response_headers)
                await asyncio.sleep(min(delay if delay is not None else 0.25 * (2**attempt), 8.0))
                continue
            raise GitHubProviderError(
                "GitHub API request failed.",
                status_code=error.code,
                provider_code=(
                    "RATE_LIMITED"
                    if rate_limited
                    else f"HTTP_{error.code}"
                ),
                retry_after=_retry_after(response_headers),
                headers=response_headers,
            ) from error
        except (OSError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
            if attempt < GITHUB_MAX_RETRIES:
                await asyncio.sleep(min(0.25 * (2**attempt), 8.0))
                continue
            raise GitHubProviderError(
                "GitHub API request failed.",
                provider_code="REQUEST_ERROR",
            ) from error
    raise GitHubProviderError("GitHub API request failed.", provider_code="REQUEST_ERROR")


def _validate_part(value: str | None) -> str:
    if not isinstance(value, str) or not OWNER_REPO_PART.fullmatch(value):
        raise ValueError("Invalid GitHub owner or repository name.")
    return value


def _repo_path(owner: str, repo: str) -> str:
    return f"/repos/{quote(_validate_part(owner), safe='')}/{quote(_validate_part(repo), safe='')}"


def _repo_from_url(value: object) -> tuple[str | None, str | None]:
    if not isinstance(value, str):
        return None, None
    parsed = urlparse(value)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None, None
    owner, repo = parts[-2], parts[-1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not OWNER_REPO_PART.fullmatch(owner) or not OWNER_REPO_PART.fullmatch(repo):
        return None, None
    return owner, repo


def _extract_repo(query: str) -> tuple[str | None, str | None]:
    match = REPO_PATH_PATTERN.search(query)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def _normalize_text(value: object, max_length: int = 4000) -> str | None:
    if not isinstance(value, str):
        return None
    text = html.unescape(re.sub(r"<[^>]*>", " ", value))
    text = " ".join(text.split()).strip()
    return text[:max_length] if text else None


def _author(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    result = {
        "login": value.get("login"),
        "id": value.get("id"),
        "type": value.get("type"),
    }
    return {key: item for key, item in result.items() if item is not None}


def _labels(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        label = {
            "name": item.get("name"),
            "color": item.get("color"),
            "description": item.get("description"),
        }
        result.append({key: value for key, value in label.items() if value is not None})
    return result


def _repository_result(payload: dict[str, object]) -> GitHubResult:
    owner, repo = _repo_from_url(payload.get("html_url"))
    full_name = str(payload.get("full_name")) if payload.get("full_name") else (
        f"{owner}/{repo}" if owner and repo else None
    )
    license_value = payload.get("license")
    license_data = (
        {
            "key": license_value.get("key"),
            "name": license_value.get("name"),
            "spdx_id": license_value.get("spdx_id"),
        }
        if isinstance(license_value, dict)
        else None
    )
    return GitHubResult(
        result_type="REPOSITORY",
        owner=owner,
        repo=repo,
        full_name=full_name,
        description=_normalize_text(payload.get("description")),
        html_url=str(payload.get("html_url")) if payload.get("html_url") else None,
        default_branch=str(payload.get("default_branch")) if payload.get("default_branch") else None,
        stars=payload.get("stargazers_count") if isinstance(payload.get("stargazers_count"), int) else None,
        forks=payload.get("forks_count") if isinstance(payload.get("forks_count"), int) else None,
        language=str(payload.get("language")) if payload.get("language") else None,
        topics=[str(item) for item in payload.get("topics", []) if isinstance(item, str)]
        if isinstance(payload.get("topics"), list)
        else [],
        archived=payload.get("archived") if isinstance(payload.get("archived"), bool) else None,
        updated_at=str(payload.get("updated_at")) if payload.get("updated_at") else None,
        pushed_at=str(payload.get("pushed_at")) if payload.get("pushed_at") else None,
        license=license_data,
        visibility=str(payload.get("visibility")) if payload.get("visibility") else (
            "public" if payload.get("private") is False else None
        ),
        issue_number=None,
        title=None,
        state=None,
        labels=[],
        author=None,
        created_at=None,
        comments_count=None,
        body_text=None,
        tag_name=None,
        release_name=None,
        published_at=None,
        prerelease=None,
        draft=None,
        assets=[],
        provider="github",
        retrieved_at=_utc_now(),
    )


def _issue_result(payload: dict[str, object]) -> GitHubResult:
    owner, repo = _repo_from_url(payload.get("repository_url"))
    if not owner or not repo:
        owner, repo = _repo_from_url(payload.get("html_url"))
    is_pull_request = isinstance(payload.get("pull_request"), dict)
    return GitHubResult(
        result_type="PULL_REQUEST" if is_pull_request else "ISSUE",
        owner=owner,
        repo=repo,
        full_name=f"{owner}/{repo}" if owner and repo else None,
        description=None,
        html_url=str(payload.get("html_url")) if payload.get("html_url") else None,
        default_branch=None,
        stars=None,
        forks=None,
        language=None,
        topics=[],
        archived=None,
        updated_at=str(payload.get("updated_at")) if payload.get("updated_at") else None,
        pushed_at=None,
        license=None,
        visibility=None,
        issue_number=payload.get("number") if isinstance(payload.get("number"), int) else None,
        title=str(payload.get("title")) if payload.get("title") else None,
        state=str(payload.get("state")) if payload.get("state") else None,
        labels=_labels(payload.get("labels")),
        author=_author(payload.get("user")),
        created_at=str(payload.get("created_at")) if payload.get("created_at") else None,
        comments_count=payload.get("comments") if isinstance(payload.get("comments"), int) else None,
        body_text=_normalize_text(payload.get("body")),
        tag_name=None,
        release_name=None,
        published_at=None,
        prerelease=None,
        draft=None,
        assets=[],
        provider="github",
        retrieved_at=_utc_now(),
    )


def _release_result(payload: dict[str, object], owner: str, repo: str) -> GitHubResult:
    raw_assets = payload.get("assets")
    assets: list[dict[str, object]] = []
    if isinstance(raw_assets, list):
        for item in raw_assets:
            if not isinstance(item, dict):
                continue
            asset = {
                "name": item.get("name"),
                "size": item.get("size"),
                "content_type": item.get("content_type"),
                "download_count": item.get("download_count"),
                "updated_at": item.get("updated_at"),
            }
            assets.append({key: value for key, value in asset.items() if value is not None})
    return GitHubResult(
        result_type="RELEASE",
        owner=owner,
        repo=repo,
        full_name=f"{owner}/{repo}",
        description=None,
        html_url=str(payload.get("html_url")) if payload.get("html_url") else None,
        default_branch=None,
        stars=None,
        forks=None,
        language=None,
        topics=[],
        archived=None,
        updated_at=None,
        pushed_at=None,
        license=None,
        visibility=None,
        issue_number=None,
        title=None,
        state=None,
        labels=[],
        author=_author(payload.get("author")),
        created_at=None,
        comments_count=None,
        body_text=_normalize_text(payload.get("body")),
        tag_name=str(payload.get("tag_name")) if payload.get("tag_name") else None,
        release_name=str(payload.get("name")) if payload.get("name") else None,
        published_at=str(payload.get("published_at")) if payload.get("published_at") else None,
        prerelease=payload.get("prerelease") if isinstance(payload.get("prerelease"), bool) else None,
        draft=payload.get("draft") if isinstance(payload.get("draft"), bool) else None,
        assets=assets,
        provider="github",
        retrieved_at=_utc_now(),
    )


class GitHubProvider:
    name = "github"

    def __init__(self) -> None:
        self._state = _ProviderState()

    def health(self) -> dict[str, object]:
        return self._state.health(authenticated=bool(GITHUB_TOKEN))

    async def _get(self, path: str, *, bucket: str) -> _GitHubResponse | None:
        try:
            response = await _http_json(f"{GITHUB_API_URL}{path}", bucket=bucket)
        except GitHubProviderError as error:
            self._state.failure(error, bucket=bucket, headers=error.headers)
            return None
        return response

    async def lookup_repository(
        self,
        owner: str,
        repo: str,
    ) -> GitHubResult | None:
        path = _repo_path(owner, repo)
        response = await self._get(path, bucket="core")
        if response is None:
            return None
        if not isinstance(response.payload, dict):
            self._state.failure(
                GitHubProviderError(
                    "GitHub repository response has an invalid shape.",
                    provider_code="INVALID_RESPONSE",
                ),
                bucket="core",
                headers=response.headers,
            )
            return None
        result = _repository_result(response.payload)
        self._state.success(response, 1)
        return result

    async def search_repositories(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[GitHubResult]:
        params = urlencode({"q": query, "per_page": min(max_results, MAX_RESULTS)})
        response = await self._get(f"/search/repositories?{params}", bucket="search")
        if response is None:
            return []
        items = response.payload.get("items") if isinstance(response.payload, dict) else None
        if not isinstance(items, list):
            self._state.failure(
                GitHubProviderError(
                    "GitHub repository search response has an invalid shape.",
                    provider_code="INVALID_RESPONSE",
                ),
                bucket="search",
                headers=response.headers,
            )
            return []
        results = [
            _repository_result(item)
            for item in items[:max_results]
            if isinstance(item, dict)
        ]
        self._state.success(response, len(results))
        return results

    async def search_issues(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[GitHubResult]:
        params = urlencode({"q": query, "per_page": min(max_results, MAX_RESULTS)})
        response = await self._get(f"/search/issues?{params}", bucket="search")
        if response is None:
            return []
        items = response.payload.get("items") if isinstance(response.payload, dict) else None
        if not isinstance(items, list):
            self._state.failure(
                GitHubProviderError(
                    "GitHub issue search response has an invalid shape.",
                    provider_code="INVALID_RESPONSE",
                ),
                bucket="search",
                headers=response.headers,
            )
            return []
        results = [
            _issue_result(item)
            for item in items[:max_results]
            if isinstance(item, dict)
        ]
        self._state.success(response, len(results))
        return results

    async def list_issues(
        self,
        owner: str,
        repo: str,
        max_results: int = 5,
    ) -> list[GitHubResult]:
        params = urlencode({"state": "all", "per_page": min(max_results, MAX_RESULTS)})
        response = await self._get(
            f"{_repo_path(owner, repo)}/issues?{params}",
            bucket="core",
        )
        if response is None:
            return []
        if not isinstance(response.payload, list):
            self._state.failure(
                GitHubProviderError(
                    "GitHub issue list response has an invalid shape.",
                    provider_code="INVALID_RESPONSE",
                ),
                bucket="core",
                headers=response.headers,
            )
            return []
        results = [
            _issue_result(item)
            for item in response.payload[:max_results]
            if isinstance(item, dict)
        ]
        self._state.success(response, len(results))
        return results

    async def lookup_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int,
    ) -> GitHubResult | None:
        if not isinstance(issue_number, int) or issue_number < 1:
            raise ValueError("Invalid GitHub issue number.")
        response = await self._get(
            f"{_repo_path(owner, repo)}/issues/{issue_number}",
            bucket="core",
        )
        if response is None:
            return None
        if not isinstance(response.payload, dict):
            self._state.failure(
                GitHubProviderError(
                    "GitHub issue response has an invalid shape.",
                    provider_code="INVALID_RESPONSE",
                ),
                bucket="core",
                headers=response.headers,
            )
            return None
        result = _issue_result(response.payload)
        self._state.success(response, 1)
        return result

    async def latest_release(self, owner: str, repo: str) -> GitHubResult | None:
        response = await self._get(
            f"{_repo_path(owner, repo)}/releases/latest",
            bucket="core",
        )
        if response is None:
            return None
        if not isinstance(response.payload, dict):
            self._state.failure(
                GitHubProviderError(
                    "GitHub release response has an invalid shape.",
                    provider_code="INVALID_RESPONSE",
                ),
                bucket="core",
                headers=response.headers,
            )
            return None
        result = _release_result(response.payload, owner, repo)
        self._state.success(response, 1)
        return result

    async def list_releases(
        self,
        owner: str,
        repo: str,
        max_results: int = 5,
    ) -> list[GitHubResult]:
        params = urlencode({"per_page": min(max_results, MAX_RESULTS)})
        response = await self._get(
            f"{_repo_path(owner, repo)}/releases?{params}",
            bucket="core",
        )
        if response is None:
            return []
        if not isinstance(response.payload, list):
            self._state.failure(
                GitHubProviderError(
                    "GitHub release list response has an invalid shape.",
                    provider_code="INVALID_RESPONSE",
                ),
                bucket="core",
                headers=response.headers,
            )
            return []
        results = [
            _release_result(item, owner, repo)
            for item in response.payload[:max_results]
            if isinstance(item, dict)
        ]
        self._state.success(response, len(results))
        return results


def _intent_from_query(query: str) -> str:
    lowered = query.casefold()
    owner, repo = _extract_repo(query)
    if owner and repo and any(marker in lowered for marker in ("release", "version", "إصدار")):
        return "latest_release"
    if any(marker in lowered for marker in ("issue", "issues", "bug", "مشكلة", "قضايا")):
        return "issue"
    if any(marker in lowered for marker in ("repository", "repo", "github", "مستودع")):
        return "repository"
    return "auto"


class GitHubSearchService:
    def __init__(self, provider: GitHubProvider | None = None) -> None:
        self.provider = provider or GitHubProvider()

    def health(self) -> dict[str, object]:
        health = self.provider.health()
        health["privacy_scope"] = "WEB"
        health["write_capability"] = False
        health["code_search"] = "DEFERRED"
        return health

    async def search(
        self,
        query: str,
        *,
        intent: str = "auto",
        owner: str | None = None,
        repo: str | None = None,
        issue_number: int | None = None,
        max_results: int = 5,
    ) -> dict[str, object]:
        if not isinstance(query, str) or len(query.strip()) > MAX_QUERY_LENGTH:
            return {"ok": False, "error": "query_too_long", "results": []}
        query = query.strip()
        if not query:
            return {"ok": False, "error": "query_required", "results": []}
        if intent not in GITHUB_INTENTS:
            return {"ok": False, "error": "invalid_intent", "results": []}
        if not isinstance(max_results, int) or not 1 <= max_results <= MAX_RESULTS:
            return {"ok": False, "error": "invalid_max_results", "results": []}

        resolved_intent = _intent_from_query(query) if intent == "auto" else intent
        if resolved_intent == "release":
            resolved_intent = "latest_release"
        extracted_owner, extracted_repo = _extract_repo(query)
        owner = owner or extracted_owner
        repo = repo or extracted_repo
        if owner is not None:
            _validate_part(owner)
        if repo is not None:
            _validate_part(repo)

        if intent == "auto" and resolved_intent == "auto":
            return {
                "ok": True,
                "routed": False,
                "recommended_provider": "tavily",
                "reason": "generic_technical_query",
                "results": [],
            }

        results: list[GitHubResult] = []
        if resolved_intent == "repository_lookup":
            if not owner or not repo:
                return {"ok": False, "error": "owner_and_repo_required", "results": []}
            result = await self.provider.lookup_repository(owner, repo)
            results = [result] if result else []
        elif resolved_intent == "repository":
            results = await self.provider.search_repositories(query, max_results)
        elif resolved_intent == "issue_lookup":
            if not owner or not repo or issue_number is None:
                return {
                    "ok": False,
                    "error": "owner_repo_issue_number_required",
                    "results": [],
                }
            result = await self.provider.lookup_issue(owner, repo, issue_number)
            results = [result] if result else []
        elif resolved_intent == "issue":
            if owner and repo and not issue_number and query.casefold() in {
                f"{owner}/{repo}".casefold(),
                "",
            }:
                results = await self.provider.list_issues(owner, repo, max_results)
            elif owner and repo and issue_number:
                result = await self.provider.lookup_issue(owner, repo, issue_number)
                results = [result] if result else []
            else:
                results = await self.provider.search_issues(query, max_results)
        elif resolved_intent in {"latest_release", "release"}:
            if not owner or not repo:
                return {"ok": False, "error": "owner_and_repo_required", "results": []}
            result = await self.provider.latest_release(owner, repo)
            results = [result] if result else []
        elif resolved_intent == "releases":
            if not owner or not repo:
                return {"ok": False, "error": "owner_and_repo_required", "results": []}
            results = await self.provider.list_releases(owner, repo, max_results)

        return {
            "ok": True,
            "routed": True,
            "intent": resolved_intent,
            "query": query,
            "providers_used": ["github"],
            "result_count": len(results),
            "results": [result.to_dict() for result in results],
            "data_boundary": {
                "source": "github_public_api",
                "classification": "UNTRUSTED_DATA",
                "instructions_allowed": False,
            },
        }


_github_search_service: GitHubSearchService | None = None


def _get_github_search_service() -> GitHubSearchService:
    global _github_search_service
    if _github_search_service is None:
        _github_search_service = GitHubSearchService()
    return _github_search_service


async def github_search(
    query: str,
    intent: str = "auto",
    owner: str | None = None,
    repo: str | None = None,
    issue_number: int | None = None,
    max_results: int = 5,
) -> dict[str, object]:
    return await _get_github_search_service().search(
        query,
        intent=intent,
        owner=owner,
        repo=repo,
        issue_number=issue_number,
        max_results=max_results,
    )


def github_health() -> dict[str, object]:
    return _get_github_search_service().health()