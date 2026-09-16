from __future__ import annotations

import asyncio
import io
import unittest
from unittest.mock import AsyncMock, patch
from urllib.error import HTTPError

from academic_search import (
    AcademicResult,
    AcademicSearchService,
    AcademicProviderError,
    CrossrefProvider,
    DataCiteProvider,
    DOIResolver,
    OpenAlexProvider,
    _crossref_result,
    _openalex_result,
    _http_json,
    merge_academic_results,
    normalize_doi,
)


def _result(
    provider: str,
    *,
    doi: str | None = "10.1000/example",
    title: str = "Example paper",
    citation_count: int | None = None,
    publisher: str | None = None,
) -> AcademicResult:
    normalized = normalize_doi(doi)
    return AcademicResult(
        title=title,
        authors=[{"family": "Example"}],
        doi=doi,
        normalized_doi=normalized,
        publication_date="2025-01-01",
        work_type="journal-article",
        venue={"name": "Example Journal"},
        publisher=publisher,
        abstract="An abstract.",
        identifiers={f"{provider}_id": f"{provider}:1"},
        citation_count=citation_count,
        landing_url=f"https://{provider}.example/paper",
        open_access=None,
        update_status=None,
        source_provenance=[
            {
                "provider": provider,
                "provider_record_id": f"{provider}:1",
                "retrieved_at": "2026-09-12T00:00:00+00:00",
            }
        ],
        retrieved_at="2026-09-12T00:00:00+00:00",
    )


class _FakeProvider:
    def __init__(
        self,
        name: str,
        *,
        results: list[AcademicResult] | None = None,
        authors: list[dict[str, object]] | None = None,
    ) -> None:
        self.name = name
        self.results = results or []
        self.authors = authors or []
        self.queries: list[str] = []

    async def lookup_doi(self, normalized_doi: str) -> AcademicResult | None:
        return next(
            (
                result
                for result in self.results
                if result.normalized_doi == normalized_doi
            ),
            None,
        )

    async def search_title(self, title: str, max_results: int) -> list[AcademicResult]:
        self.queries.append(title)
        return self.results[:max_results]

    async def search_topic(
        self,
        query: str,
        max_results: int,
        *,
        latest: bool = False,
    ) -> list[AcademicResult]:
        del latest
        self.queries.append(query)
        return self.results[:max_results]

    async def search_author(
        self,
        query: str,
        max_results: int,
    ) -> dict[str, object]:
        self.queries.append(query)
        return {"authors": self.authors[:max_results], "results": []}

    def health(self) -> dict[str, object]:
        return {"status": "READY", "request_count": 0}


class AcademicSearchTests(unittest.TestCase):
    def test_doi_normalization_variants(self) -> None:
        expected = "10.1000/xyz123"
        for value in (
            "10.1000/XYZ123",
            "doi:10.1000/XYZ123",
            "https://doi.org/10.1000/XYZ123",
            "http://dx.doi.org/10.1000/XYZ123",
        ):
            self.assertEqual(normalize_doi(value), expected)
        self.assertIsNone(normalize_doi("not-a-doi"))
        self.assertIsNone(normalize_doi("10.1000/has whitespace"))

    def test_crossref_exact_doi_lookup(self) -> None:
        async def fake_http(url: str, **_: object) -> object:
            self.assertIn("/works/10.1000%2Fexample", url)
            return type(
                "Response",
                (),
                {
                    "payload": {
                        "message": {
                            "DOI": "10.1000/example",
                            "title": ["Example paper"],
                            "author": [{"given": "Ada", "family": "Example"}],
                            "publisher": "Example Press",
                            "type": "journal-article",
                            "URL": "https://doi.org/10.1000/example",
                            "is-referenced-by-count": 4,
                            "published-print": {"date-parts": [[2025, 1, 1]]},
                        }
                    },
                    "status_code": 200,
                    "headers": {},
                    "retries": 0,
                    "latency_ms": 1,
                },
            )()

        with patch("academic_search._http_json", new=fake_http):
            result = asyncio.run(CrossrefProvider().lookup_doi("10.1000/example"))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.normalized_doi, "10.1000/example")
        self.assertEqual(result.citation_count, 4)

    def test_datacite_route_is_not_false_crossref_failure(self) -> None:
        async def fake_http(url: str, **_: object) -> object:
            if url.endswith("/agency"):
                payload = {"message": {"agency": {"id": "DataCite"}}}
            else:
                payload = {
                    "data": {
                        "attributes": {
                            "doi": "10.9999/data.1",
                            "titles": [{"title": "Data paper"}],
                            "creators": [{"name": "Data Author"}],
                            "publisher": "Data Repository",
                            "types": {"resourceTypeGeneral": "Dataset"},
                            "url": "https://doi.org/10.9999/data.1",
                        }
                    }
                }
            return type(
                "Response",
                (),
                {
                    "payload": payload,
                    "status_code": 200,
                    "headers": {},
                    "retries": 0,
                    "latency_ms": 1,
                },
            )()

        with patch("academic_search._http_json", new=fake_http):
            response = asyncio.run(
                DOIResolver(CrossrefProvider(), DataCiteProvider()).resolve(
                    "https://doi.org/10.9999/data.1"
                )
            )
        self.assertTrue(response["ok"])
        self.assertEqual(response["registration_agency"], "datacite")
        self.assertEqual(response["results"][0]["source_provenance"][0]["provider"], "datacite")

    def test_invalid_doi_returns_clean_state(self) -> None:
        response = asyncio.run(
            DOIResolver(CrossrefProvider(), DataCiteProvider()).resolve("not a DOI")
        )
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"], "invalid_doi")
        self.assertEqual(response["results"], [])

    def test_unknown_agency_uses_datacite_exact_lookup_fallback(self) -> None:
        async def fake_http(url: str, **_: object) -> object:
            if url.endswith("/agency"):
                payload = {"message": {"agency": {"id": "UnknownAgency"}}}
            elif "api.datacite.org" in url:
                payload = {
                    "data": {
                        "attributes": {
                            "doi": "10.9999/fallback.1",
                            "titles": [{"title": "Fallback paper"}],
                        }
                    }
                }
            else:
                raise AcademicProviderError("not found", status_code=404)
            return type(
                "Response",
                (),
                {
                    "payload": payload,
                    "status_code": 200,
                    "headers": {},
                    "retries": 0,
                    "latency_ms": 1,
                },
            )()

        with patch("academic_search._http_json", new=fake_http):
            response = asyncio.run(
                DOIResolver(CrossrefProvider(), DataCiteProvider()).resolve(
                    "10.9999/fallback.1"
                )
            )
        self.assertTrue(response["ok"])
        self.assertEqual(response["registration_agency"], "datacite")

    def test_exact_title_routes_crossref_then_merges_openalex_when_ambiguous(self) -> None:
        crossref = _FakeProvider(
            "crossref",
            results=[
                _result("crossref", title="Example paper", publisher="Crossref Press"),
                _result("crossref", doi="10.1000/other", title="Other paper"),
            ],
        )
        openalex = _FakeProvider(
            "openalex",
            results=[_result("openalex", title="Example paper", citation_count=8)],
        )
        service = AcademicSearchService(crossref, _FakeProvider("datacite"), openalex)
        response = asyncio.run(
            service.search("Example paper", intent="exact_title", max_results=5)
        )
        self.assertEqual(response["providers_used"], ["crossref", "openalex"])
        self.assertEqual(response["result_count"], 2)
        self.assertEqual(response["results"][0]["normalized_doi"], "10.1000/example")
        self.assertTrue(response["results"][0]["metadata_conflicts"])

    def test_author_and_topic_route_to_openalex(self) -> None:
        openalex = _FakeProvider(
            "openalex",
            results=[_result("openalex")],
            authors=[{"openalex_id": "https://openalex.org/A1", "display_name": "Ada"}],
        )
        service = AcademicSearchService(
            _FakeProvider("crossref"),
            _FakeProvider("datacite"),
            openalex,
        )
        author_response = asyncio.run(
            service.search("Ada Lovelace", intent="author", max_results=5)
        )
        topic_response = asyncio.run(
            service.search("machine learning", intent="topic", max_results=5)
        )
        self.assertEqual(author_response["providers_used"], ["openalex"])
        self.assertEqual(author_response["authors"][0]["display_name"], "Ada")
        self.assertEqual(topic_response["providers_used"], ["openalex"])
        self.assertEqual(topic_response["result_count"], 1)

    def test_duplicate_merge_preserves_provenance_and_conflicts(self) -> None:
        merged = merge_academic_results(
            [
                _result("crossref", publisher="Publisher A", citation_count=4),
                _result("openalex", publisher="Publisher B", citation_count=9),
            ]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].publisher, "Publisher A")
        self.assertEqual(merged[0].citation_count, 9)
        self.assertEqual(
            {item["provider"] for item in merged[0].source_provenance},
            {"crossref", "openalex"},
        )
        self.assertTrue(merged[0].metadata_conflicts)

    def test_provider_timeout_is_isolated(self) -> None:
        provider = CrossrefProvider()
        with patch(
            "academic_search._http_json",
            new=AsyncMock(
                side_effect=AcademicProviderError(
                    "timeout",
                    provider_code="REQUEST_ERROR",
                )
            ),
        ):
            result = asyncio.run(provider.lookup_doi("10.1000/example"))
        self.assertIsNone(result)
        self.assertEqual(provider.health()["status"], "ERROR")

    def test_datacite_timeout_is_isolated(self) -> None:
        provider = DataCiteProvider()
        with patch(
            "academic_search._http_json",
            new=AsyncMock(
                side_effect=AcademicProviderError(
                    "timeout",
                    provider_code="REQUEST_ERROR",
                )
            ),
        ):
            result = asyncio.run(provider.lookup_doi("10.5438/0012"))
        self.assertIsNone(result)
        self.assertEqual(provider.health()["status"], "ERROR")

    def test_openalex_timeout_is_isolated(self) -> None:
        provider = OpenAlexProvider()
        with patch(
            "academic_search._http_json",
            new=AsyncMock(
                side_effect=AcademicProviderError(
                    "timeout",
                    provider_code="REQUEST_ERROR",
                )
            ),
        ):
            result = asyncio.run(provider.search_topic("machine learning", 2))
        self.assertEqual(result, [])
        self.assertEqual(provider.health()["status"], "ERROR")

    def test_malformed_provider_payload_is_not_false_success(self) -> None:
        provider = CrossrefProvider()
        with patch(
            "academic_search._http_json",
            new=AsyncMock(
                return_value=type(
                    "Response",
                    (),
                    {
                        "payload": {"unexpected": "shape"},
                        "status_code": 200,
                        "headers": {},
                        "retries": 0,
                        "latency_ms": 1,
                    },
                )(),
            ),
        ):
            result = asyncio.run(provider.lookup_doi("10.1000/example"))
        self.assertIsNone(result)
        self.assertEqual(provider.health()["status"], "ERROR")

    def test_incomplete_metadata_does_not_invent_identifiers_or_status(self) -> None:
        result = _crossref_result({"title": ["Only a title"]})
        self.assertEqual(result.title, "Only a title")
        self.assertIsNone(result.normalized_doi)
        self.assertIsNone(result.citation_count)
        self.assertIsNone(result.landing_url)
        self.assertIsNone(result.update_status)

    def test_retraction_status_is_unknown_until_provider_confirms_it(self) -> None:
        unknown = _openalex_result(
            {
                "id": "https://openalex.org/W1",
                "title": "Unconfirmed paper",
            }
        )
        confirmed = _openalex_result(
            {
                "id": "https://openalex.org/W2",
                "title": "Retracted paper",
                "is_retracted": True,
            }
        )
        self.assertIsNone(unknown.update_status)
        self.assertEqual(confirmed.update_status, {"retracted": True})
        self.assertTrue(confirmed.source_provenance)

    def test_rate_limit_backoff_is_bounded(self) -> None:
        error = HTTPError(
            "https://api.example.test",
            429,
            "rate limited",
            {"Retry-After": "0"},
            io.BytesIO(),
        )
        with (
            patch("academic_search._http_json_sync", side_effect=[error, ({}, 200, {})]),
            patch("academic_search.asyncio.sleep", new=AsyncMock()) as sleep,
        ):
            response = asyncio.run(_http_json("https://api.example.test"))
        self.assertEqual(response.retries, 1)
        sleep.assert_awaited_once()

    def test_arabic_query_does_not_include_private_content(self) -> None:
        openalex = _FakeProvider("openalex", results=[_result("openalex")])
        service = AcademicSearchService(
            _FakeProvider("crossref"),
            _FakeProvider("datacite"),
            openalex,
        )
        query = "أبحاث حديثة عن معالجة اللغة العربية"
        response = asyncio.run(service.search(query, intent="topic"))
        self.assertTrue(response["ok"])
        self.assertEqual(openalex.queries, [query])
        self.assertNotIn("private", openalex.queries[0].casefold())


if __name__ == "__main__":
    unittest.main()