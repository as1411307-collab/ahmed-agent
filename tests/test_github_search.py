from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from urllib.error import HTTPError

import github_search
from github_search import (
    GitHubProvider,
    GitHubProviderError,
    GitHubSearchService,
    _GitHubResponse,
    github_health,
)


FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "github" / "contracts.json").read_text(
        encoding="utf-8"
    )
)


def _response(payload: object, *, bucket: str = "core", headers: dict[str, str] | None = None) -> _GitHubResponse:
    return _GitHubResponse(
        payload=payload,
        status_code=200,
        headers=headers or {"x-ratelimit-limit": "60", "x-ratelimit-remaining": "59"},
        retries=0,
        latency_ms=1,
        bucket=bucket,
    )


class GitHubSearchTests(unittest.TestCase):
    def fixture(self, fixture_id: str) -> dict[str, object]:
        return next(item for item in FIXTURES if item["id"] == fixture_id)

    def test_fixture_contracts_have_fourteen_cases(self) -> None:
        self.assertEqual(len(FIXTURES), 14)
        self.assertEqual(
            {item["id"] for item in FIXTURES},
            {
                "repository_lookup", "repository_search", "latest_release",
                "release_list", "issue_lookup", "issue_search",
                "pull_request_distinction", "archived_repository",
                "no_result", "rate_limit", "malformed_response", "timeout",
                "arabic_query", "privacy",
            },
        )

    def test_repository_lookup_normalizes_identity(self) -> None:
        provider = GitHubProvider()
        fixture = self.fixture("repository_lookup")
        with patch("github_search._http_json", new=AsyncMock(
            return_value=_response(fixture["payload"])
        )):
            result = asyncio.run(provider.lookup_repository("modelcontextprotocol", "python-sdk"))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.identity, "modelcontextprotocol/python-sdk")
        self.assertEqual(result.result_type, "REPOSITORY")
        self.assertFalse(result.archived)

    def test_repository_search_and_no_result(self) -> None:
        provider = GitHubProvider()
        search_fixture = self.fixture("repository_search")
        no_result_fixture = self.fixture("no_result")
        responses = [
            _response(search_fixture["payload"], bucket="search"),
            _response(no_result_fixture["payload"], bucket="search"),
        ]
        with patch("github_search._http_json", new=AsyncMock(side_effect=responses)):
            results = asyncio.run(provider.search_repositories("modelcontextprotocol", 5))
            empty = asyncio.run(provider.search_repositories("does-not-exist", 5))
        self.assertEqual(results[0].identity, "modelcontextprotocol/python-sdk")
        self.assertEqual(empty, [])
        self.assertEqual(provider.health()["search_rate_limit"]["remaining"], 59)

    def test_release_lookup_and_release_alias(self) -> None:
        fixture = self.fixture("latest_release")
        service = GitHubSearchService(provider=GitHubProvider())
        with patch("github_search._http_json", new=AsyncMock(
            return_value=_response(fixture["payload"])
        )):
            latest = asyncio.run(service.search(
                "modelcontextprotocol/python-sdk", intent="release"
            ))
        self.assertTrue(latest["ok"])
        self.assertEqual(latest["intent"], "latest_release")
        self.assertEqual(latest["results"][0]["identity"], "modelcontextprotocol/python-sdk@v1.2.3")

    def test_release_list(self) -> None:
        fixture = self.fixture("release_list")
        provider = GitHubProvider()
        with patch("github_search._http_json", new=AsyncMock(
            return_value=_response(fixture["payload"])
        )):
            results = asyncio.run(provider.list_releases("modelcontextprotocol", "python-sdk", 5))
        self.assertEqual(results[0].result_type, "RELEASE")

    def test_issue_lookup_search_and_pr_distinction(self) -> None:
        issue_fixture = self.fixture("issue_lookup")
        search_fixture = self.fixture("issue_search")
        pr_fixture = self.fixture("pull_request_distinction")
        provider = GitHubProvider()
        with patch("github_search._http_json", new=AsyncMock(side_effect=[
            _response(issue_fixture["payload"]),
            _response(search_fixture["payload"], bucket="search"),
            _response(pr_fixture["payload"], bucket="search"),
        ])):
            issue = asyncio.run(provider.lookup_issue("modelcontextprotocol", "python-sdk", 17))
            issues = asyncio.run(provider.search_issues("transport", 5))
            prs = asyncio.run(provider.search_issues("stream", 5))
        self.assertEqual(issue.identity, "modelcontextprotocol/python-sdk#17")
        self.assertEqual(issues[0].result_type, "ISSUE")
        self.assertEqual(prs[0].result_type, "PULL_REQUEST")

    def test_issue_list_preserves_issue_and_pr_types(self) -> None:
        payload = [
            {
                "number": 20,
                "title": "An issue",
                "state": "open",
                "html_url": "https://github.com/modelcontextprotocol/python-sdk/issues/20",
                "repository_url": "https://api.github.com/repos/modelcontextprotocol/python-sdk",
            },
            {
                "number": 21,
                "title": "A pull request",
                "state": "open",
                "html_url": "https://github.com/modelcontextprotocol/python-sdk/pull/21",
                "repository_url": "https://api.github.com/repos/modelcontextprotocol/python-sdk",
                "pull_request": {"url": "https://api.github.com/repos/modelcontextprotocol/python-sdk/pulls/21"},
            },
        ]
        provider = GitHubProvider()
        with patch("github_search._http_json", new=AsyncMock(
            return_value=_response(payload)
        )):
            results = asyncio.run(provider.list_issues("modelcontextprotocol", "python-sdk", 5))
        self.assertEqual([result.result_type for result in results], ["ISSUE", "PULL_REQUEST"])

    def test_archived_repository_is_preserved(self) -> None:
        fixture = self.fixture("archived_repository")
        provider = GitHubProvider()
        with patch("github_search._http_json", new=AsyncMock(
            return_value=_response(fixture["payload"])
        )):
            result = asyncio.run(provider.lookup_repository("example", "archived"))
        self.assertTrue(result.archived)

    def test_invalid_shape_sets_safe_error(self) -> None:
        fixture = self.fixture("malformed_response")
        provider = GitHubProvider()
        with patch("github_search._http_json", new=AsyncMock(
            return_value=_response(fixture["payload"], bucket="search")
        )):
            self.assertEqual(asyncio.run(provider.search_repositories("bad", 5)), [])
        health = provider.health()
        self.assertEqual(health["status"], "ERROR")
        self.assertEqual(health["safe_error_code"], "INVALID_RESPONSE")

    def test_rate_limit_preserves_headers_and_state(self) -> None:
        fixture = self.fixture("rate_limit")
        error = GitHubProviderError(
            "rate limited",
            status_code=429,
            provider_code="RATE_LIMITED",
            headers=fixture["headers"],
        )
        provider = GitHubProvider()
        with patch("github_search._http_json", new=AsyncMock(side_effect=error)):
            self.assertEqual(asyncio.run(provider.search_repositories("busy", 5)), [])
        health = provider.health()
        self.assertEqual(health["status"], "RATE_LIMITED")
        self.assertEqual(health["search_rate_limit"]["remaining"], 0)

    def test_timeout_sets_request_error(self) -> None:
        provider = GitHubProvider()
        error = GitHubProviderError("timeout", provider_code="REQUEST_ERROR")
        with patch("github_search._http_json", new=AsyncMock(side_effect=error)):
            self.assertIsNone(asyncio.run(provider.lookup_repository("example", "timeout")))
        self.assertEqual(provider.health()["safe_error_code"], "REQUEST_ERROR")

    def test_arabic_query_is_encoded_without_leaking_data(self) -> None:
        fixture = self.fixture("arabic_query")
        provider = GitHubProvider()
        async def fake_http(url: str, **_: object) -> _GitHubResponse:
            self.assertIn("%D9%85%D8%B3%D8%AA%D9%88%D8%AF%D8%B9", url)
            return _response(fixture["payload"], bucket="search")
        with patch("github_search._http_json", new=fake_http):
            results = asyncio.run(provider.search_repositories(fixture["query"], 5))
        self.assertEqual(results, [])

    def test_privacy_boundary_and_no_write_capability(self) -> None:
        fixture = self.fixture("privacy")
        service = GitHubSearchService(provider=GitHubProvider())
        result = service.health()
        self.assertEqual(result["privacy_scope"], fixture["expected_scope"])
        self.assertFalse(result["write_capability"])
        self.assertEqual(result["code_search"], "DEFERRED")
        self.assertNotIn("MY_FILES", json.dumps(result))

    def test_generic_technical_query_routes_to_tavily(self) -> None:
        service = GitHubSearchService(provider=GitHubProvider())
        result = asyncio.run(service.search("How do I configure MCP transport?"))
        self.assertFalse(result["routed"])
        self.assertEqual(result["recommended_provider"], "tavily")

    def test_unauthenticated_health_is_explicit(self) -> None:
        with patch("github_search.GITHUB_TOKEN", ""):
            provider = GitHubProvider()
            health = provider.health()
        self.assertEqual(health["status"], "READY_UNAUTHENTICATED")
        self.assertEqual(health["auth_mode"], "unauthenticated")

    def test_headers_include_api_version_and_optional_auth(self) -> None:
        with patch("github_search.GITHUB_TOKEN", ""):
            headers = github_search._headers()
        self.assertEqual(headers["X-GitHub-Api-Version"], "2022-11-28")
        self.assertNotIn("Authorization", headers)
        with patch("github_search.GITHUB_TOKEN", "test-token"):
            headers = github_search._headers()
        self.assertEqual(headers["Authorization"], "Bearer test-token")


if __name__ == "__main__":
    unittest.main()