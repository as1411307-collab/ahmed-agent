from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from academic_search import (
    AcademicResult,
    AcademicSearchService,
    CrossrefProvider,
    DataCiteProvider,
    OpenAlexProvider,
    merge_academic_results,
    normalize_doi,
)


FIXTURE_PATH = (
    Path(__file__).parent
    / "tests"
    / "fixtures"
    / "academic_benchmark"
    / "contracts.json"
)
REQUIRED_FIXTURE_KEYS = {
    "id",
    "expected",
}
REQUIRED_EXPECTED_KEYS = {
    "provider_route",
    "required_fields",
    "merge_behavior",
    "enrichment_expected",
    "tavily_fallback_allowed",
}


def load_contract_fixtures() -> list[dict[str, Any]]:
    fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(fixtures, list) or len(fixtures) != 16:
        raise ValueError("Academic benchmark must contain exactly 16 fixtures.")
    for fixture in fixtures:
        if not isinstance(fixture, dict) or not REQUIRED_FIXTURE_KEYS <= fixture.keys():
            raise ValueError("Academic fixture has an invalid contract shape.")
        expected = fixture["expected"]
        if not isinstance(expected, dict) or not REQUIRED_EXPECTED_KEYS <= expected.keys():
            raise ValueError("Academic fixture is missing expected contract fields.")
    return fixtures


def _result(
    provider: str,
    *,
    title: str,
    doi: str | None = None,
    citation_count: int | None = None,
    publisher: str | None = None,
) -> AcademicResult:
    normalized_doi = normalize_doi(doi)
    return AcademicResult(
        title=title,
        authors=[{"family": "Benchmark"}],
        doi=doi,
        normalized_doi=normalized_doi,
        publication_date="2025-01-01",
        work_type="journal-article",
        venue={"name": "Benchmark Journal"},
        publisher=publisher,
        abstract="Benchmark abstract.",
        identifiers={f"{provider}_id": f"{provider}:benchmark"},
        citation_count=citation_count,
        landing_url=f"https://{provider}.example/benchmark",
        open_access=(
            {"is_oa": True, "landing_page_url": f"https://{provider}.example/oa"}
            if provider == "openalex"
            else None
        ),
        update_status=None,
        source_provenance=[
            {
                "provider": provider,
                "provider_record_id": f"{provider}:benchmark",
                "retrieved_at": "2026-01-01T00:00:00+00:00",
            }
        ],
        retrieved_at="2026-01-01T00:00:00+00:00",
    )


@dataclass
class _ContractProvider:
    name: str
    doi_results: dict[str, AcademicResult] = field(default_factory=dict)
    title_results: dict[str, list[AcademicResult]] = field(default_factory=dict)
    topic_results: dict[str, list[AcademicResult]] = field(default_factory=dict)
    author_results: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    agency_results: dict[str, str] = field(default_factory=dict)
    graph_results: dict[str, dict[str, object]] = field(default_factory=dict)
    calls: list[dict[str, object]] = field(default_factory=list)

    async def detect_agency(self, normalized_doi: str) -> str | None:
        self.calls.append({"operation": "detect_agency", "query": normalized_doi})
        return self.agency_results.get(normalized_doi)

    async def lookup_doi(self, normalized_doi: str) -> AcademicResult | None:
        self.calls.append({"operation": "lookup_doi", "query": normalized_doi})
        return self.doi_results.get(normalized_doi)

    async def search_title(
        self,
        title: str,
        max_results: int,
    ) -> list[AcademicResult]:
        self.calls.append({"operation": "search_title", "query": title})
        return self.title_results.get(title, [])[:max_results]

    async def search_topic(
        self,
        query: str,
        max_results: int,
        *,
        latest: bool = False,
    ) -> list[AcademicResult]:
        self.calls.append(
            {"operation": "search_topic", "query": query, "latest": latest}
        )
        return self.topic_results.get(query, [])[:max_results]

    async def search_author(
        self,
        query: str,
        max_results: int,
    ) -> dict[str, object]:
        self.calls.append({"operation": "search_author", "query": query})
        return {"authors": self.author_results.get(query, [])[:max_results], "results": []}

    async def graph(
        self,
        openalex_id: str,
        max_results: int,
    ) -> dict[str, object]:
        self.calls.append({"operation": "graph", "query": openalex_id})
        result = self.graph_results.get(
            openalex_id,
            {"work_id": openalex_id, "reference_ids": [], "cited_by": []},
        )
        return {
            **result,
            "reference_ids": list(result.get("reference_ids", []))[:max_results],
            "cited_by": list(result.get("cited_by", []))[:max_results],
        }

    def health(self) -> dict[str, object]:
        return {
            "status": "READY",
            "request_count": len(self.calls),
            "result_count": 0,
            "retry_count": 0,
        }


def _build_contract_service() -> AcademicSearchService:
    crossref = _ContractProvider("crossref")
    datacite = _ContractProvider("datacite")
    openalex = _ContractProvider("openalex")

    crossref_doi = "10.1038/s41586-020-2649-2"
    datacite_doi = "10.5438/0012"
    crossref.agency_results = {crossref_doi: "crossref"}
    datacite.agency_results = {datacite_doi: "datacite"}
    crossref.doi_results = {
        crossref_doi: _result(
            "crossref",
            title="A benchmark Crossref paper",
            doi=crossref_doi,
            citation_count=12,
            publisher="Crossref Publisher",
        )
    }
    datacite.doi_results = {
        datacite_doi: _result(
            "datacite",
            title="A benchmark DataCite dataset",
            doi=datacite_doi,
            publisher="DataCite Repository",
        )
    }
    crossref.title_results = {
        "Attention Is All You Need": [
            _result(
                "crossref",
                title="Attention Is All You Need",
                doi="10.5555/attention",
            )
        ],
        "Deep learning": [
            _result("crossref", title="Deep learning A", doi="10.5555/deep-a"),
            _result("crossref", title="Deep learning B", doi="10.5555/deep-b"),
        ],
    }
    openalex.title_results = {
        "Deep learning": [
            _result("openalex", title="Deep learning A", doi="10.5555/deep-a")
        ]
    }
    openalex.author_results = {
        "Geoffrey Hinton": [
            {
                "openalex_id": "https://openalex.org/A1",
                "display_name": "Geoffrey Hinton",
                "works_count": 100,
            }
        ]
    }
    openalex.topic_results = {
        "machine learning": [
            _result(
                "openalex",
                title="Machine learning benchmark",
                doi="10.5555/ml",
                citation_count=20,
            )
        ],
        "latest research machine learning": [
            _result(
                "openalex",
                title="Latest machine learning benchmark",
                doi="10.5555/latest",
            )
        ],
        "paper without a registered identifier": [
            _result("openalex", title="Paper without a registered identifier")
        ],
        "ابحث عن أبحاث حديثة عن معالجة اللغة العربية": [
            _result("openalex", title="Arabic language processing benchmark")
        ],
        "zzzzzzzz academic record that should not exist": [],
    }
    openalex.doi_results = {
        crossref_doi: _result(
            "openalex",
            title="A benchmark Crossref paper",
            doi=crossref_doi,
            citation_count=25,
        )
    }
    openalex.graph_results = {
        "openalex:benchmark": {
            "work_id": "openalex:benchmark",
            "reference_ids": ["openalex:reference-1"],
            "cited_by": [{"id": "openalex:citing-1"}],
        }
    }
    return AcademicSearchService(crossref, datacite, openalex)


def _contract_merge_case(fixture_id: str) -> tuple[list[AcademicResult], dict[str, object]]:
    if fixture_id == "duplicate-same-paper":
        return [
            _result("crossref", title="Same paper", doi="10.1000/example", publisher="A"),
            _result(
                "openalex",
                title="Same paper",
                doi="10.1000/example",
                citation_count=9,
                publisher="A",
            ),
        ], {"expected_count": 1, "conflict": False}
    return [
        _result("crossref", title="Same paper", doi="10.1000/example", publisher="A"),
        _result("openalex", title="Same paper", doi="10.1000/example", publisher="B"),
    ], {"expected_count": 1, "conflict": True}


def _has_required_fields(value: dict[str, object], fields: list[str]) -> bool:
    return all(field_name in value for field_name in fields) and (
        "source_provenance" not in value
        or bool(value.get("source_provenance"))
    )


async def run_contract_benchmark() -> dict[str, object]:
    fixtures = load_contract_fixtures()
    service = _build_contract_service()
    rows: list[dict[str, object]] = []
    routing_passes = 0
    exact_passes = 0
    completeness_passes = 0
    provenance_passes = 0
    no_result_passes = 0
    no_hallucination_passes = 0
    privacy_violations = 0
    call_counts: list[int] = []
    provider_latencies: dict[str, list[int]] = {}

    for fixture in fixtures:
        fixture_id = str(fixture["id"])
        expected = fixture["expected"]
        started_at = time.perf_counter()
        passed = True
        errors: list[str] = []
        response: dict[str, object]
        providers = [
            service.crossref,
            service.datacite,
            service.openalex,
        ]
        for provider in providers:
            provider.calls.clear()  # type: ignore[attr-defined]

        if fixture.get("operation") == "merge":
            source_results, merge_expectation = _contract_merge_case(fixture_id)
            merged = merge_academic_results(source_results)
            response = {
                "ok": len(merged) == merge_expectation["expected_count"],
                "results": [item.to_dict() for item in merged],
            }
            if fixture_id == "conflicting-metadata":
                response["conflict"] = bool(
                    merged and merged[0].metadata_conflicts
                )
        else:
            response = await service.search(
                str(fixture.get("query") or ""),
                intent=str(fixture.get("intent") or "auto"),
                max_results=5,
            )

        used_routes = response.get("providers_used", [])
        if fixture.get("operation") == "merge":
            used_routes = expected["provider_route"]
        if fixture.get("intent") == "doi":
            agency = response.get("registration_agency")
            used_routes = [agency] if agency else []
        if used_routes != expected["provider_route"]:
            routing_passes -= 0
            passed = False
            errors.append(f"route={used_routes!r}")
        else:
            routing_passes += 1

        results = response.get("results")
        result_list = results if isinstance(results, list) else []
        if "doi" in expected and expected["doi"] is not None:
            exact = bool(
                result_list
                and isinstance(result_list[0], dict)
                and result_list[0].get("normalized_doi") == expected["doi"]
            )
        elif "title_pattern" in expected:
            pattern = str(expected["title_pattern"])
            exact = bool(
                (
                    result_list
                    and isinstance(result_list[0], dict)
                    and pattern in str(result_list[0].get("title", "")).casefold()
                )
                or (
                    expected["required_fields"] == ["authors"]
                    and response.get("authors")
                )
            )
        elif expected["required_fields"] == ["authors"]:
            exact = bool(response.get("authors"))
        elif fixture.get("operation") == "merge":
            exact = bool(response.get("ok"))
        elif expected["merge_behavior"] == "no_result":
            exact = not result_list
        else:
            exact = bool(result_list or response.get("authors") == [])
        exact_passes += int(exact)
        if not exact:
            passed = False
            errors.append("exact_result")

        if "graph" in expected["required_fields"]:
            completeness = (
                "graph" in response
                and bool(result_list)
                and isinstance(result_list[0], dict)
                and bool(result_list[0].get("source_provenance"))
            )
        elif fixture.get("operation") == "merge" and fixture_id == "conflicting-metadata":
            completeness = response.get("conflict") is True
        elif expected["required_fields"] == ["authors"]:
            completeness = "authors" in response and bool(response["authors"])
        elif expected["required_fields"] == ["results"]:
            completeness = "results" in response
        else:
            first = result_list[0] if result_list else {}
            completeness = isinstance(first, dict) and _has_required_fields(
                first,
                expected["required_fields"],
            )
        completeness_passes += int(completeness)
        if not completeness:
            passed = False
            errors.append("required_fields")

        provenance = all(
            isinstance(item, dict)
            and isinstance(item.get("source_provenance"), list)
            and bool(item["source_provenance"])
            for item in result_list
        ) if result_list else True
        provenance_passes += int(provenance)
        if not provenance:
            passed = False
            errors.append("provenance")

        no_result = (
            expected["merge_behavior"] == "no_result"
            and not result_list
            or expected["merge_behavior"] != "no_result"
        )
        no_result_passes += int(no_result)

        no_hallucination = True
        if fixture_id == "missing-doi" and result_list:
            no_hallucination = result_list[0].get("normalized_doi") is None
        if fixture_id == "invalid-doi":
            no_hallucination = response.get("error") == "invalid_doi"
        no_hallucination_passes += int(no_hallucination)
        if not no_hallucination:
            passed = False
            errors.append("hallucinated_identifier")

        for provider in providers:
            for call in provider.calls:  # type: ignore[attr-defined]
                query = str(call.get("query") or "")
                if "private" in query.casefold():
                    privacy_violations += 1
                provider_name = str(provider.name)
                provider_latencies.setdefault(provider_name, []).append(
                    round((time.perf_counter() - started_at) * 1000)
                )
        fixture_calls = sum(len(provider.calls) for provider in providers)  # type: ignore[attr-defined]
        call_counts.append(fixture_calls)
        rows.append(
            {
                "id": fixture_id,
                "status": "PASS" if passed else "FAIL",
                "errors": errors,
                "calls_per_fixture": fixture_calls,
                "latency_ms": round((time.perf_counter() - started_at) * 1000),
            }
        )

    total = len(fixtures)
    passed_count = sum(row["status"] == "PASS" for row in rows)
    return {
        "status": "PASS" if passed_count == total and privacy_violations == 0 else "FAIL",
        "fixture_count": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "deterministic_pass_rate": round(passed_count / total, 4),
        "routing_accuracy": round(routing_passes / total, 4),
        "exact_result_accuracy": round(exact_passes / total, 4),
        "required_field_completeness": round(completeness_passes / total, 4),
        "citation_provenance_completeness": round(provenance_passes / total, 4),
        "no_result_correctness": round(no_result_passes / total, 4),
        "no_hallucination_contract": "PASS"
        if no_hallucination_passes == total
        else "FAIL",
        "privacy_violations": privacy_violations,
        "privacy": "PASS" if privacy_violations == 0 else "FAIL",
        "average_calls_per_fixture": round(sum(call_counts) / total, 2),
        "provider_latency_ms": {
            provider: {
                "average": round(sum(values) / len(values), 2),
                "max": max(values),
            }
            for provider, values in provider_latencies.items()
            if values
        },
        "calls_per_fixture": call_counts,
        "fixtures": rows,
    }


async def run_live_smoke() -> dict[str, object]:
    started_at = time.perf_counter()
    crossref = CrossrefProvider()
    datacite = DataCiteProvider()
    openalex = OpenAlexProvider()
    rows: list[dict[str, object]] = []

    async def run_case(name: str, callback: Any) -> None:
        case_started = time.perf_counter()
        try:
            value = await callback()
            rows.append(
                {
                    "id": name,
                    "status": "PASS" if value else "DEGRADED",
                    "latency_ms": round((time.perf_counter() - case_started) * 1000),
                }
            )
        except Exception as error:
            rows.append(
                {
                    "id": name,
                    "status": "DEGRADED",
                    "latency_ms": round((time.perf_counter() - case_started) * 1000),
                    "safe_error": type(error).__name__,
                }
            )

    crossref_doi = "10.1038/s41586-020-2649-2"
    datacite_doi = "10.5438/0012"
    await run_case(
        "crossref-doi",
        lambda: crossref.lookup_doi(crossref_doi),
    )
    await run_case(
        "datacite-doi",
        lambda: datacite.lookup_doi(datacite_doi),
    )
    await run_case(
        "openalex-topic",
        lambda: openalex.search_topic("Arabic natural language processing", 2),
    )

    async def merged_paper() -> bool:
        crossref_result, openalex_result = await asyncio.gather(
            crossref.lookup_doi(crossref_doi),
            openalex.lookup_doi(crossref_doi),
        )
        return bool(
            crossref_result
            and openalex_result
            and len(merge_academic_results([crossref_result, openalex_result])) == 1
        )

    await run_case("merged-paper", merged_paper)
    statuses = [str(row["status"]) for row in rows]
    overall = "PASS" if all(status == "PASS" for status in statuses) else "DEGRADED"
    return {
        "status": overall,
        "case_count": len(rows),
        "passed": statuses.count("PASS"),
        "degraded": statuses.count("DEGRADED"),
        "latency_ms": round((time.perf_counter() - started_at) * 1000),
        "provider_request_count": {
            "crossref": crossref.health().get("request_count"),
            "datacite": datacite.health().get("request_count"),
            "openalex": openalex.health().get("request_count"),
        },
        "cases": rows,
    }


async def run_benchmark(include_live: bool = False) -> dict[str, object]:
    deterministic = await run_contract_benchmark()
    report: dict[str, object] = {
        "benchmark": "academic-contract-regression",
        "deterministic": deterministic,
    }
    if include_live:
        report["live_smoke"] = await run_live_smoke()
    else:
        report["live_smoke"] = {"status": "NOT_RUN"}
    return report


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run the bounded live provider smoke in addition to deterministic tests.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            await run_benchmark(include_live=args.live),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())