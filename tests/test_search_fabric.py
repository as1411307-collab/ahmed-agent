from __future__ import annotations

import os
import unittest
from dataclasses import dataclass
from unittest.mock import patch

from search_fabric import (
    BraveProvider,
    NormalizedSearchResult,
    ProviderSearchOutput,
    SearchFabric,
    SearchOptions,
    _merge_results,
    classify_source,
    normalize_url,
)


@dataclass
class _FakeProvider:
    name: str
    status: str
    output: ProviderSearchOutput

    def capabilities(self) -> dict[str, object]:
        return {"search": True}

    async def healthcheck(self) -> dict[str, object]:
        return {"status": self.status}

    async def search(self, query: str, options: SearchOptions) -> ProviderSearchOutput:
        del query, options
        return self.output

    def usage(self) -> dict[str, object]:
        return self.output.metrics


def _fake_result(provider: str, url: str, rank: int = 1) -> NormalizedSearchResult:
    return NormalizedSearchResult(
        provider=provider,
        title="Official source",
        url=url,
        canonical_url=normalize_url(url),
        domain="docs.python.org",
        snippet="source",
        content="source",
        published_at=None,
        language=None,
        provider_rank=rank,
        provider_score=None,
        retrieved_at="2026-09-12T00:00:00+00:00",
        source_type="official_documentation",
        source_class="primary",
        found_by=[provider],
        source_family="docs.python.org",
    )


class SearchFabricTests(unittest.TestCase):
    def test_tracking_parameters_are_removed_but_meaningful_query_is_kept(self) -> None:
        self.assertEqual(
            normalize_url(
                "HTTPS://WWW.Example.com:443/page/?utm_source=x&item=1#section"
            ),
            "https://www.example.com/page?item=1",
        )
        self.assertNotEqual(
            normalize_url("https://example.com/page?id=1"),
            normalize_url("https://example.com/page?id=2"),
        )

    def test_same_page_from_two_providers_has_shared_provenance(self) -> None:
        common = {
            "title": "Official Python documentation",
            "canonical_url": "https://docs.python.org/3/",
            "domain": "docs.python.org",
            "snippet": "Python docs",
            "published_at": None,
            "language": None,
            "provider_score": None,
            "retrieved_at": "2026-09-12T00:00:00+00:00",
            "source_type": "official_documentation",
            "source_class": "primary",
            "source_family": "docs.python.org",
        }
        tavily = NormalizedSearchResult(
            provider="tavily",
            url="https://docs.python.org/3/",
            provider_rank=1,
            found_by=["tavily"],
            content="Python docs",
            **common,
        )
        brave = NormalizedSearchResult(
            provider="brave",
            url="https://docs.python.org/3/?utm_campaign=test",
            provider_rank=1,
            found_by=["brave"],
            content="Python docs",
            **{**common, "canonical_url": "https://docs.python.org/3"},
        )
        results, group_count = _merge_results(
            [tavily, brave],
            "official Python documentation",
            5,
        )
        self.assertEqual(group_count, 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].found_by, ["tavily", "brave"])
        self.assertEqual(results[0].provider_ranks, {"tavily": 1, "brave": 1})

    def test_official_source_classification_is_not_rank_based(self) -> None:
        self.assertEqual(
            classify_source("https://docs.python.org/3/", "Python docs"),
            ("official_documentation", "primary"),
        )
        self.assertEqual(
            classify_source("https://www.reddit.com/r/python", "Discussion"),
            ("social_community", "community"),
        )

    def test_brave_is_not_configured_without_a_key(self) -> None:
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": ""}, clear=False):
            self.assertEqual(BraveProvider().status, "NOT_CONFIGURED")

    def test_provider_failure_isolation_and_fusion(self) -> None:
        tavily = _FakeProvider(
            "tavily",
            "READY",
            ProviderSearchOutput(
                [_fake_result("tavily", "https://docs.python.org/3/")],
                {"provider": "tavily", "request_count": 1},
                {"ok": True, "results": [{"url": "https://docs.python.org/3/"}]},
            ),
        )
        brave = _FakeProvider(
            "brave",
            "READY",
            ProviderSearchOutput(
                [_fake_result("brave", "https://docs.python.org/3/?utm_source=test")],
                {"provider": "brave", "request_count": 1},
            ),
        )
        fabric = SearchFabric(tavily, brave)
        with patch("search_fabric.BRAVE_SEARCH_ENABLED", True):
            import asyncio

            response = asyncio.run(
                fabric.search(
                    "ما هي وثائق Python الرسمية؟",
                    mode="DEEP",
                    max_results=5,
                    tavily_legacy_search=lambda **_: None,
                )
            )
        self.assertEqual(response["fusion"], "rrf")
        self.assertEqual(response["results"][0]["found_by"], ["tavily", "brave"])

    def test_tavily_continues_when_brave_fails(self) -> None:
        tavily = _FakeProvider(
            "tavily",
            "READY",
            ProviderSearchOutput(
                [_fake_result("tavily", "https://docs.python.org/3/")],
                {"provider": "tavily", "request_count": 1},
                {"ok": True, "results": [{"url": "https://docs.python.org/3/"}]},
            ),
        )
        brave = _FakeProvider(
            "brave",
            "ERROR",
            ProviderSearchOutput([], {"provider": "brave", "status": "ERROR"}),
        )
        fabric = SearchFabric(tavily, brave)
        with patch("search_fabric.BRAVE_SEARCH_ENABLED", True):
            import asyncio

            response = asyncio.run(
                fabric.search(
                    "مقارنة مصادر Python",
                    mode="DEEP",
                    max_results=5,
                    tavily_legacy_search=lambda **_: None,
                )
            )
        self.assertTrue(response["ok"])
        self.assertEqual(response["providers_used"], ["tavily"])


if __name__ == "__main__":
    unittest.main()