from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Awaitable, Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from config import (
    ACADEMIC_MAX_RETRIES,
    ACADEMIC_PROVIDER_TIMEOUT_SECONDS,
    CROSSREF_MAILTO,
    OPENALEX_API_KEY,
)


logger = logging.getLogger("ahmed_agent.academic_search")

CROSSREF_API_URL = "https://api.crossref.org"
DATACITE_API_URL = "https://api.datacite.org"
OPENALEX_API_URL = "https://api.openalex.org"
DOI_URL = "https://doi.org/"
MAX_QUERY_LENGTH = 2000
MAX_RESULTS = 10
ACADEMIC_INTENTS = {
    "auto",
    "doi",
    "exact_title",
    "author",
    "topic",
    "citations",
    "latest_research",
}
DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
DOI_IN_TEXT_PATTERN = re.compile(
    r"(?:(?:https?://)?(?:dx\.)?doi\.org/|doi:\s*)?(10\.\d{4,9}/[^\s<>\"']+)",
    re.IGNORECASE,
)


class AcademicProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider_code: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider_code = provider_code
        self.retry_after = retry_after


@dataclass(frozen=True)
class _HTTPResponse:
    payload: dict[str, Any]
    status_code: int
    headers: dict[str, str]
    retries: int
    latency_ms: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_date(value: object) -> str | None:
    if isinstance(value, str) and re.search(r"\b(?:19|20)\d{2}\b", value):
        return value.strip()
    return None


def _strip_html(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = re.sub(r"<[^>]*>", " ", html.unescape(value))
    text = " ".join(text.split()).strip()
    return text or None


def normalize_doi(value: str | None) -> str | None:
    """Return a canonical DOI without changing its meaningful suffix."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None

    lowered = candidate.casefold()
    if lowered.startswith("https://doi.org/") or lowered.startswith("http://doi.org/"):
        candidate = candidate.split("://", 1)[1].split("/", 1)[1]
    elif lowered.startswith("https://dx.doi.org/") or lowered.startswith(
        "http://dx.doi.org/"
    ):
        candidate = candidate.split("://", 1)[1].split("/", 1)[1]
    elif lowered.startswith("doi:"):
        candidate = candidate[4:].strip()

    candidate = candidate.strip()
    candidate = re.sub(r"[.,;]+$", "", candidate)
    if any(character.isspace() for character in candidate):
        return None
    if not DOI_PATTERN.fullmatch(candidate):
        return None
    return candidate.casefold()


def extract_doi(value: str) -> str | None:
    if not isinstance(value, str):
        return None
    match = DOI_IN_TEXT_PATTERN.search(value.strip())
    return normalize_doi(match.group(1)) if match else normalize_doi(value)


def _safe_provider_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.hostname)
        and not parsed.username
        and not parsed.password
    )


def _safe_openalex_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.hostname == "api.openalex.org"
        and not parsed.username
        and not parsed.password
    )


def _parse_retry_after(headers: dict[str, str]) -> float | None:
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _http_json_sync(
    url: str,
    headers: dict[str, str],
    timeout: float,
) -> tuple[dict[str, Any], int, dict[str, str]]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Ahmed-Agent/1.0",
            **headers,
        },
        method="GET",
    )
    with urlopen(request, timeout=timeout) as response:
        body = response.read(4 * 1024 * 1024)
        payload = json.loads(body.decode("utf-8", errors="replace"))
        if not isinstance(payload, dict):
            raise ValueError("Provider response must be a JSON object.")
        return (
            payload,
            int(response.status),
            {str(key): str(value) for key, value in response.headers.items()},
        )


async def _http_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = ACADEMIC_PROVIDER_TIMEOUT_SECONDS,
) -> _HTTPResponse:
    if not _safe_provider_url(url):
        raise AcademicProviderError(
            "Provider URL is not safe.",
            provider_code="UNSAFE_PROVIDER_URL",
        )

    started_at = time.perf_counter()
    request_headers = headers or {}
    for attempt in range(ACADEMIC_MAX_RETRIES + 1):
        try:
            payload, status_code, response_headers = await asyncio.to_thread(
                _http_json_sync,
                url,
                request_headers,
                timeout,
            )
            return _HTTPResponse(
                payload=payload,
                status_code=status_code,
                headers=response_headers,
                retries=attempt,
                latency_ms=round((time.perf_counter() - started_at) * 1000),
            )
        except HTTPError as error:
            response_headers = {
                str(key): str(value) for key, value in error.headers.items()
            }
            retry_after = _parse_retry_after(response_headers)
            retryable = error.code == 429 or error.code >= 500
            if retryable and attempt < ACADEMIC_MAX_RETRIES:
                delay = min(
                    retry_after
                    if retry_after is not None
                    else 0.25 * (2**attempt),
                    8.0,
                )
                await asyncio.sleep(delay)
                continue
            raise AcademicProviderError(
                "Academic provider request failed.",
                status_code=error.code,
                provider_code=(
                    "RATE_LIMITED"
                    if error.code == 429
                    else f"HTTP_{error.code}"
                ),
                retry_after=retry_after,
            ) from error
        except (OSError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
            if attempt < ACADEMIC_MAX_RETRIES:
                await asyncio.sleep(min(0.25 * (2**attempt), 8.0))
                continue
            raise AcademicProviderError(
                "Academic provider request failed.",
                provider_code="REQUEST_ERROR",
            ) from error

    raise AcademicProviderError(
        "Academic provider request failed.",
        provider_code="REQUEST_ERROR",
    )


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

    def success(self, response: _HTTPResponse, result_count: int) -> None:
        self.status = "READY"
        self.last_success = _utc_now()
        self.last_failure = None
        self.last_failure_code = None
        self.latency_ms = response.latency_ms
        self.request_count += 1
        self.result_count = result_count
        self.retry_count += response.retries

    def failure(self, error: AcademicProviderError) -> None:
        self.status = "RATE_LIMITED" if error.status_code == 429 else "ERROR"
        self.last_failure = _utc_now()
        self.last_failure_code = error.provider_code or "REQUEST_ERROR"
        self.latency_ms = None
        self.request_count += 1

    def health(self, *, configured: bool = True) -> dict[str, object]:
        return {
            "status": self.status if configured else "NOT_CONFIGURED",
            "last_success": self.last_success,
            "last_failure": self.last_failure,
            "latency_ms": self.latency_ms,
            "request_count": self.request_count,
            "result_count": self.result_count,
            "retry_count": self.retry_count,
            "safe_error_code": self.last_failure_code
            if self.status != "READY"
            else None,
        }


@dataclass
class AcademicResult:
    title: str
    authors: list[dict[str, object]]
    doi: str | None
    normalized_doi: str | None
    publication_date: str | None
    work_type: str | None
    venue: dict[str, object] | None
    publisher: str | None
    abstract: str | None
    identifiers: dict[str, object]
    citation_count: int | None
    landing_url: str | None
    open_access: dict[str, object] | None
    update_status: dict[str, object] | None
    source_provenance: list[dict[str, object]]
    retrieved_at: str
    metadata_conflicts: list[dict[str, object]] = field(default_factory=list)
    original_doi: str | None = None

    def identity_key(self) -> tuple[str, str]:
        if self.normalized_doi:
            return "doi", self.normalized_doi
        for key in ("crossref_id", "datacite_id", "openalex_id"):
            value = self.identifiers.get(key)
            if value:
                return key, str(value)
        year = (self.publication_date or "")[:4]
        authors = "|".join(
            str(author.get("family") or author.get("name") or "").casefold()
            for author in self.authors[:3]
        )
        return "fallback", f"{self.title.casefold()}|{authors}|{year}"

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "authors": self.authors,
            "doi": self.doi,
            "normalized_doi": self.normalized_doi,
            "publication_date": self.publication_date,
            "work_type": self.work_type,
            "venue": self.venue,
            "publisher": self.publisher,
            "abstract": self.abstract,
            "identifiers": self.identifiers,
            "citation_count": self.citation_count,
            "landing_url": self.landing_url,
            "open_access": self.open_access,
            "update_status": self.update_status,
            "source_provenance": self.source_provenance,
            "retrieved_at": self.retrieved_at,
            "metadata_conflicts": self.metadata_conflicts,
            "original_doi": self.original_doi,
        }


class AcademicProvider(Protocol):
    name: str

    async def lookup_doi(self, normalized_doi: str) -> AcademicResult | None: ...

    async def search_title(
        self,
        title: str,
        max_results: int,
    ) -> list[AcademicResult]: ...

    async def search_topic(
        self,
        query: str,
        max_results: int,
        *,
        latest: bool = False,
    ) -> list[AcademicResult]: ...

    async def search_author(
        self,
        query: str,
        max_results: int,
    ) -> dict[str, object]: ...

    def health(self) -> dict[str, object]: ...


def _authors_from_crossref(items: object) -> list[dict[str, object]]:
    if not isinstance(items, list):
        return []
    authors: list[dict[str, object]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        author: dict[str, object] = {}
        for key in ("given", "family", "sequence", "authenticated-orcid"):
            if item.get(key):
                author[key.replace("-", "_")] = item[key]
        if item.get("ORCID"):
            author["orcid"] = item["ORCID"]
        if author:
            authors.append(author)
    return authors


def _authors_from_datacite(items: object) -> list[dict[str, object]]:
    if not isinstance(items, list):
        return []
    authors: list[dict[str, object]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or " ".join(
            str(value)
            for value in (item.get("givenName"), item.get("familyName"))
            if value
        )
        if not name:
            continue
        author = {"name": str(name)}
        name_identifiers = item.get("nameIdentifiers")
        if isinstance(name_identifiers, list):
            for identifier in name_identifiers:
                if isinstance(identifier, dict) and identifier.get("nameIdentifier"):
                    author["orcid"] = identifier["nameIdentifier"]
                    break
        authors.append(author)
    return authors


def _date_from_parts(value: object) -> str | None:
    if isinstance(value, list) and value and isinstance(value[0], list):
        parts = value[0]
    elif isinstance(value, list):
        parts = value
    else:
        return None
    numbers = [int(item) for item in parts if isinstance(item, int)]
    if not numbers:
        return None
    try:
        if len(numbers) >= 3:
            return date(numbers[0], numbers[1], numbers[2]).isoformat()
        if len(numbers) == 2:
            return f"{numbers[0]:04d}-{numbers[1]:02d}"
        return f"{numbers[0]:04d}"
    except ValueError:
        return None


def _source_provenance(
    provider: str,
    native_id: str | None,
    *,
    retrieved_at: str,
) -> list[dict[str, object]]:
    return [
        {
            "provider": provider,
            "provider_record_id": native_id,
            "retrieved_at": retrieved_at,
        }
    ]


def _crossref_result(message: dict[str, object]) -> AcademicResult:
    doi = str(message.get("DOI") or "") or None
    normalized = normalize_doi(doi)
    retrieved_at = _utc_now()
    title_items = message.get("title")
    title = (
        str(title_items[0])
        if isinstance(title_items, list) and title_items
        else str(message.get("title") or doi or "Untitled work")
    )
    container = message.get("container-title")
    venue = {
        "name": str(container[0]),
        "issn": message.get("ISSN", []),
    } if isinstance(container, list) and container else None
    identifiers: dict[str, object] = {
        "crossref_id": normalized or doi,
    }
    if message.get("URL"):
        identifiers["crossref_url"] = message["URL"]
    if message.get("author"):
        identifiers["author_count"] = len(message["author"])
    update_to = message.get("update-to")
    update_status = (
        {"updates": update_to}
        if isinstance(update_to, list) and update_to
        else None
    )
    links = message.get("link")
    full_text_url = None
    if isinstance(links, list):
        for link in links:
            if isinstance(link, dict) and _safe_provider_url(str(link.get("URL") or "")):
                full_text_url = str(link["URL"])
                break
    return AcademicResult(
        title=title,
        authors=_authors_from_crossref(message.get("author")),
        doi=doi,
        normalized_doi=normalized,
        publication_date=(
            _date_from_parts(message.get("published-print"))
            or _date_from_parts(message.get("published-online"))
            or _date_from_parts(message.get("issued"))
            or _date_from_parts(message.get("created"))
        ),
        work_type=str(message.get("type")) if message.get("type") else None,
        venue=venue,
        publisher=str(message.get("publisher")) if message.get("publisher") else None,
        abstract=_strip_html(message.get("abstract")),
        identifiers=identifiers,
        citation_count=(
            int(message["is-referenced-by-count"])
            if isinstance(message.get("is-referenced-by-count"), int)
            else None
        ),
        landing_url=str(message.get("URL")) if message.get("URL") else (
            DOI_URL + normalized if normalized else None
        ),
        open_access=(
            {"full_text_url": full_text_url}
            if full_text_url
            else None
        ),
        update_status=update_status,
        source_provenance=_source_provenance(
            "crossref",
            normalized or doi,
            retrieved_at=retrieved_at,
        ),
        retrieved_at=retrieved_at,
    )


def _datacite_result(attributes: dict[str, object], doi: str) -> AcademicResult:
    normalized = normalize_doi(str(attributes.get("doi") or doi))
    retrieved_at = _utc_now()
    titles = attributes.get("titles")
    title = (
        str(titles[0].get("title"))
        if isinstance(titles, list)
        and titles
        and isinstance(titles[0], dict)
        and titles[0].get("title")
        else str(attributes.get("title") or normalized or "Untitled work")
    )
    container = attributes.get("container")
    venue = (
        {"name": container.get("title")}
        if isinstance(container, dict) and container.get("title")
        else None
    )
    descriptions = attributes.get("descriptions")
    abstract = None
    if isinstance(descriptions, list):
        for description in descriptions:
            if isinstance(description, dict) and description.get("description"):
                abstract = str(description["description"])
                break
    identifiers: dict[str, object] = {"datacite_id": normalized or doi}
    for item in attributes.get("identifiers", []) if isinstance(attributes.get("identifiers"), list) else []:
        if isinstance(item, dict) and item.get("value") and item.get("identifierType"):
            identifiers[str(item["identifierType"]).casefold()] = item["value"]
    return AcademicResult(
        title=title,
        authors=_authors_from_datacite(attributes.get("creators")),
        doi=str(attributes.get("doi") or doi),
        normalized_doi=normalized,
        publication_date=(
            next(
                (
                    str(item.get("date"))
                    for item in attributes.get("dates", [])
                    if isinstance(item, dict)
                    and item.get("date")
                    and item.get("dateType") in {"Issued", "Created", "Published"}
                ),
                None,
            )
            if isinstance(attributes.get("dates"), list)
            else None
        ),
        work_type=(
            str(attributes.get("types", {}).get("resourceTypeGeneral"))
            if isinstance(attributes.get("types"), dict)
            and attributes.get("types", {}).get("resourceTypeGeneral")
            else None
        ),
        venue=venue,
        publisher=str(attributes.get("publisher")) if attributes.get("publisher") else None,
        abstract=abstract,
        identifiers=identifiers,
        citation_count=None,
        landing_url=str(attributes.get("url")) if attributes.get("url") else (
            DOI_URL + normalized if normalized else None
        ),
        open_access=None,
        update_status=None,
        source_provenance=_source_provenance(
            "datacite",
            normalized or doi,
            retrieved_at=retrieved_at,
        ),
        retrieved_at=retrieved_at,
    )


def _abstract_from_inverted_index(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    words: list[tuple[int, str]] = []
    for token, positions in value.items():
        if not isinstance(token, str) or not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                words.append((position, token))
    if not words:
        return None
    return " ".join(token for _, token in sorted(words))


def _openalex_result(work: dict[str, object]) -> AcademicResult:
    ids = work.get("ids") if isinstance(work.get("ids"), dict) else {}
    doi = normalize_doi(ids.get("doi") if isinstance(ids, dict) else None)
    openalex_id = work.get("id")
    if isinstance(ids, dict) and ids.get("openalex"):
        openalex_id = ids["openalex"]
    authors: list[dict[str, object]] = []
    authorships = work.get("authorships")
    if isinstance(authorships, list):
        for authorship in authorships:
            if not isinstance(authorship, dict):
                continue
            author = authorship.get("author")
            if not isinstance(author, dict):
                continue
            item: dict[str, object] = {
                "name": author.get("display_name"),
                "openalex_id": author.get("id"),
            }
            if author.get("orcid"):
                item["orcid"] = author["orcid"]
            authors.append({key: value for key, value in item.items() if value})
    primary_location = work.get("primary_location")
    source = (
        primary_location.get("source")
        if isinstance(primary_location, dict)
        else None
    )
    venue = (
        {
            "name": source.get("display_name"),
            "openalex_id": source.get("id"),
        }
        if isinstance(source, dict) and source.get("display_name")
        else None
    )
    best_oa = work.get("best_oa_location")
    open_access = work.get("open_access")
    oa: dict[str, object] | None = (
        dict(open_access)
        if isinstance(open_access, dict)
        else None
    )
    if isinstance(best_oa, dict):
        oa = oa or {}
        if best_oa.get("landing_page_url"):
            oa["landing_page_url"] = best_oa["landing_page_url"]
        if best_oa.get("pdf_url"):
            oa["pdf_url"] = best_oa["pdf_url"]
    identifiers = {
        "openalex_id": openalex_id,
        **(
            {key: value for key, value in ids.items() if value}
            if isinstance(ids, dict)
            else {}
        ),
    }
    retrieved_at = _utc_now()
    landing_url = (
        str(best_oa.get("landing_page_url"))
        if isinstance(best_oa, dict) and best_oa.get("landing_page_url")
        else str(work.get("doi") or work.get("id"))
        if work.get("doi") or work.get("id")
        else None
    )
    return AcademicResult(
        title=str(work.get("title") or "Untitled work"),
        authors=authors,
        doi=str(work.get("doi") or "") or None,
        normalized_doi=doi,
        publication_date=_safe_date(work.get("publication_date")),
        work_type=str(work.get("type")) if work.get("type") else None,
        venue=venue,
        publisher=(
            str(source.get("host_organization_name"))
            if isinstance(source, dict) and source.get("host_organization_name")
            else None
        ),
        abstract=_abstract_from_inverted_index(work.get("abstract_inverted_index")),
        identifiers=identifiers,
        citation_count=(
            int(work["cited_by_count"])
            if isinstance(work.get("cited_by_count"), int)
            else None
        ),
        landing_url=landing_url,
        open_access=oa,
        update_status=(
            {"retracted": True}
            if work.get("is_retracted") is True
            else None
        ),
        source_provenance=_source_provenance(
            "openalex",
            str(openalex_id) if openalex_id else None,
            retrieved_at=retrieved_at,
        ),
        retrieved_at=retrieved_at,
    )


class _HTTPAcademicProvider:
    name = "academic"

    def __init__(self) -> None:
        self._state = _ProviderState()

    def health(self) -> dict[str, object]:
        return self._state.health()

    async def _get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> _HTTPResponse | None:
        try:
            response = await _http_json(url, headers=headers)
        except AcademicProviderError as error:
            if error.status_code == 404:
                return None
            self._state.failure(error)
            return None
        return response


class CrossrefProvider(_HTTPAcademicProvider):
    name = "crossref"

    def _headers(self) -> dict[str, str]:
        if not CROSSREF_MAILTO:
            return {}
        return {"User-Agent": f"Ahmed-Agent/1.0 (mailto:{CROSSREF_MAILTO})"}

    async def detect_agency(self, normalized_doi: str) -> str | None:
        url = f"{CROSSREF_API_URL}/works/{quote(normalized_doi, safe='')}/agency"
        response = await self._get(url, headers=self._headers())
        if response is None:
            return None
        self._state.success(response, 0)
        message = response.payload.get("message", response.payload)
        if isinstance(message, dict):
            agency = message.get("agency")
            if isinstance(agency, dict):
                value = agency.get("id") or agency.get("name")
            else:
                value = agency
            if isinstance(value, str):
                lowered = value.casefold()
                if "datacite" in lowered:
                    return "datacite"
                if "crossref" in lowered:
                    return "crossref"
                return lowered
        return None

    async def lookup_doi(self, normalized_doi: str) -> AcademicResult | None:
        url = f"{CROSSREF_API_URL}/works/{quote(normalized_doi, safe='')}"
        response = await self._get(url, headers=self._headers())
        if response is None:
            return None
        message = response.payload.get("message")
        if not isinstance(message, dict):
            self._state.failure(
                AcademicProviderError(
                    "Crossref response missing message.",
                    provider_code="INVALID_RESPONSE",
                )
            )
            return None
        result = _crossref_result(message)
        self._state.success(response, 1)
        return result

    async def search_title(
        self,
        title: str,
        max_results: int,
    ) -> list[AcademicResult]:
        params = urlencode({"query.title": title, "rows": min(max_results, MAX_RESULTS)})
        response = await self._get(
            f"{CROSSREF_API_URL}/works?{params}",
            headers=self._headers(),
        )
        if response is None:
            return []
        message = response.payload.get("message")
        items = message.get("items") if isinstance(message, dict) else []
        results = [
            _crossref_result(item)
            for item in items
            if isinstance(item, dict)
        ] if isinstance(items, list) else []
        self._state.success(response, len(results))
        return results

    async def search_topic(
        self,
        query: str,
        max_results: int,
        *,
        latest: bool = False,
    ) -> list[AcademicResult]:
        del latest
        return await self.search_title(query, max_results)

    async def search_author(
        self,
        query: str,
        max_results: int,
    ) -> dict[str, object]:
        results = await self.search_title(query, max_results)
        return {"authors": [], "results": [result.to_dict() for result in results]}


class DataCiteProvider(_HTTPAcademicProvider):
    name = "datacite"

    async def lookup_doi(self, normalized_doi: str) -> AcademicResult | None:
        url = f"{DATACITE_API_URL}/dois/{quote(normalized_doi, safe='')}"
        response = await self._get(url)
        if response is None:
            return None
        data = response.payload.get("data")
        attributes = data.get("attributes") if isinstance(data, dict) else None
        if not isinstance(attributes, dict):
            self._state.failure(
                AcademicProviderError(
                    "DataCite response missing attributes.",
                    provider_code="INVALID_RESPONSE",
                )
            )
            return None
        result = _datacite_result(attributes, normalized_doi)
        self._state.success(response, 1)
        return result

    async def search_title(
        self,
        title: str,
        max_results: int,
    ) -> list[AcademicResult]:
        del title, max_results
        return []

    async def search_topic(
        self,
        query: str,
        max_results: int,
        *,
        latest: bool = False,
    ) -> list[AcademicResult]:
        del query, max_results, latest
        return []

    async def search_author(
        self,
        query: str,
        max_results: int,
    ) -> dict[str, object]:
        del query, max_results
        return {"authors": [], "results": []}


class OpenAlexProvider(_HTTPAcademicProvider):
    name = "openalex"

    def _params(self, params: dict[str, object]) -> str:
        values = dict(params)
        if OPENALEX_API_KEY:
            values["api_key"] = OPENALEX_API_KEY
        return urlencode(values)

    async def lookup_doi(self, normalized_doi: str) -> AcademicResult | None:
        params = self._params(
            {
                "filter": f"doi:{DOI_URL}{normalized_doi}",
                "per-page": 1,
            }
        )
        response = await self._get(f"{OPENALEX_API_URL}/works?{params}")
        if response is None:
            return None
        results = response.payload.get("results")
        work = results[0] if isinstance(results, list) and results else None
        if not isinstance(work, dict):
            self._state.success(response, 0)
            return None
        result = _openalex_result(work)
        self._state.success(response, 1)
        return result

    async def search_title(
        self,
        title: str,
        max_results: int,
    ) -> list[AcademicResult]:
        return await self.search_topic(title, max_results)

    async def search_topic(
        self,
        query: str,
        max_results: int,
        *,
        latest: bool = False,
    ) -> list[AcademicResult]:
        params: dict[str, object] = {
            "search": query,
            "per-page": min(max_results, MAX_RESULTS),
        }
        if latest:
            params["filter"] = f"from_publication_date:{date.today().year}-01-01"
            params["sort"] = "publication_date:desc"
        response = await self._get(
            f"{OPENALEX_API_URL}/works?{self._params(params)}"
        )
        if response is None:
            return []
        raw_results = response.payload.get("results")
        results = [
            _openalex_result(item)
            for item in raw_results
            if isinstance(item, dict)
        ] if isinstance(raw_results, list) else []
        self._state.success(response, len(results))
        return results

    async def search_author(
        self,
        query: str,
        max_results: int,
    ) -> dict[str, object]:
        params = self._params({"search": query, "per-page": min(max_results, MAX_RESULTS)})
        response = await self._get(f"{OPENALEX_API_URL}/authors?{params}")
        if response is None:
            return {"authors": [], "results": []}
        raw_results = response.payload.get("results")
        authors: list[dict[str, object]] = []
        if isinstance(raw_results, list):
            for item in raw_results[:max_results]:
                if not isinstance(item, dict):
                    continue
                authors.append(
                    {
                        "openalex_id": item.get("id"),
                        "display_name": item.get("display_name"),
                        "orcid": item.get("orcid"),
                        "works_count": item.get("works_count"),
                        "cited_by_count": item.get("cited_by_count"),
                    }
                )
        self._state.success(response, len(authors))
        return {"authors": authors, "results": []}

    async def graph(
        self,
        openalex_id: str,
        max_results: int,
    ) -> dict[str, object]:
        identifier = quote(openalex_id, safe="")
        response = await self._get(f"{OPENALEX_API_URL}/works/{identifier}")
        if response is None:
            return {"references": [], "cited_by": []}
        work = response.payload
        references = work.get("referenced_works")
        cited_by_url = work.get("cited_by_api_url")
        cited_by: list[dict[str, object]] = []
        if isinstance(cited_by_url, str) and _safe_openalex_url(cited_by_url):
            params = self._params({"per-page": min(max_results, MAX_RESULTS)})
            cited_response = await self._get(f"{cited_by_url}?{params}")
            if cited_response is not None:
                raw = cited_response.payload.get("results")
                if isinstance(raw, list):
                    cited_by = [
                        {
                            "id": item.get("id"),
                            "title": item.get("title"),
                            "doi": item.get("doi"),
                        }
                        for item in raw[:max_results]
                        if isinstance(item, dict)
                    ]
        return {
            "work_id": openalex_id,
            "reference_ids": references[:max_results]
            if isinstance(references, list)
            else [],
            "cited_by": cited_by,
        }


class DOIResolver:
    def __init__(
        self,
        crossref: CrossrefProvider,
        datacite: DataCiteProvider,
        openalex: OpenAlexProvider | None = None,
    ) -> None:
        self.crossref = crossref
        self.datacite = datacite
        self.openalex = openalex

    async def resolve(self, value: str) -> dict[str, object]:
        original_doi = value
        normalized_doi = normalize_doi(value)
        if not normalized_doi:
            return {
                "ok": False,
                "intent": "doi",
                "error": "invalid_doi",
                "original_doi": original_doi,
                "results": [],
            }

        agency = await self.crossref.detect_agency(normalized_doi)
        provider: AcademicProvider | None = None
        if agency == "crossref":
            provider = self.crossref
        elif agency == "datacite":
            provider = self.datacite

        result = await provider.lookup_doi(normalized_doi) if provider else None
        if result is None and agency not in {"crossref", "datacite"}:
            # DataCite is a fallback for an unavailable/unknown agency. The
            # exact lookup proves ownership before structured metadata is used.
            result = await self.datacite.lookup_doi(normalized_doi)
            if result is not None:
                agency = "datacite"
            else:
                result = await self.crossref.lookup_doi(normalized_doi)
                if result is not None:
                    agency = "crossref"
        if result is None and agency not in {"crossref", "datacite"}:
            # A safe resolver URL is useful even when the registration agency
            # endpoint is unavailable; no structured metadata is invented.
            return {
                "ok": True,
                "intent": "doi",
                "registration_agency": None,
                "original_doi": original_doi,
                "normalized_doi": normalized_doi,
                "landing_url": DOI_URL + normalized_doi,
                "results": [],
                "metadata_status": "NOT_AVAILABLE",
            }
        if result is not None:
            result.original_doi = original_doi
        return {
            "ok": result is not None,
            "intent": "doi",
            "registration_agency": agency,
            "original_doi": original_doi,
            "normalized_doi": normalized_doi,
            "landing_url": DOI_URL + normalized_doi,
            "results": [result.to_dict()] if result else [],
            "metadata_status": "READY" if result else "NOT_FOUND",
        }


def _field_value(result: AcademicResult, field_name: str) -> object:
    return getattr(result, field_name)


def merge_academic_results(results: list[AcademicResult]) -> list[AcademicResult]:
    groups: dict[tuple[str, str], list[AcademicResult]] = {}
    for result in results:
        groups.setdefault(result.identity_key(), []).append(result)

    merged: list[AcademicResult] = []
    preferred_registration = {"crossref", "datacite"}
    for group in groups.values():
        ordered = sorted(
            group,
            key=lambda item: (
                0
                if any(
                    record.get("provider") in preferred_registration
                    for record in item.source_provenance
                )
                else 1,
                item.title.casefold(),
            ),
        )
        base = ordered[0]
        conflicts: list[dict[str, object]] = []
        for field_name in (
            "title",
            "publication_date",
            "work_type",
            "venue",
            "publisher",
            "abstract",
            "landing_url",
            "open_access",
            "update_status",
        ):
            values: list[tuple[str, object]] = []
            for item in group:
                value = _field_value(item, field_name)
                if value not in (None, "", [], {}):
                    provider = str(item.source_provenance[0].get("provider"))
                    if (provider, value) not in values:
                        values.append((provider, value))
            if len(values) > 1 and len({json.dumps(value, sort_keys=True, default=str) for _, value in values}) > 1:
                conflicts.append(
                    {
                        "field": field_name,
                        "values": [
                            {"provider": provider, "value": value}
                            for provider, value in values
                        ],
                    }
                )

        def choose(field_name: str) -> object:
            values = [
                (str(item.source_provenance[0].get("provider")), _field_value(item, field_name))
                for item in group
                if _field_value(item, field_name) not in (None, "", [], {})
            ]
            if not values:
                return None
            for provider, value in values:
                if provider in preferred_registration:
                    return value
            return values[0][1]

        citation_counts = [
            item.citation_count for item in group if item.citation_count is not None
        ]
        identifiers: dict[str, object] = {}
        for item in group:
            identifiers.update(item.identifiers)
        merged.append(
            AcademicResult(
                title=str(choose("title") or base.title),
                authors=next((item.authors for item in group if item.authors), []),
                doi=next((item.doi for item in group if item.doi), None),
                normalized_doi=next(
                    (item.normalized_doi for item in group if item.normalized_doi),
                    None,
                ),
                publication_date=choose("publication_date"),
                work_type=choose("work_type"),
                venue=choose("venue"),
                publisher=choose("publisher"),
                abstract=max(
                    (item.abstract for item in group if item.abstract),
                    key=len,
                    default=None,
                ),
                identifiers=identifiers,
                citation_count=max(citation_counts) if citation_counts else None,
                landing_url=choose("landing_url"),
                open_access=next(
                    (item.open_access for item in group if item.open_access),
                    None,
                ),
                update_status=next(
                    (item.update_status for item in group if item.update_status),
                    None,
                ),
                source_provenance=[
                    provenance
                    for item in group
                    for provenance in item.source_provenance
                ],
                retrieved_at=max(item.retrieved_at for item in group),
                metadata_conflicts=conflicts,
            )
        )
    return merged


def _looks_like_author_query(query: str) -> bool:
    lowered = query.casefold()
    return any(
        marker in lowered
        for marker in ("author:", "author ", "by ", "مؤلف", "باحث", "للكاتب")
    )


def _looks_like_latest_query(query: str) -> bool:
    lowered = query.casefold()
    return any(
        marker in lowered
        for marker in (
            "latest research",
            "recent papers",
            "new papers",
            "أحدث الأبحاث",
            "أحدث الدراسات",
            "آخر الأبحاث",
            "بحث حديث",
        )
    )


def _detect_intent(query: str) -> str:
    if extract_doi(query):
        return "doi"
    if _looks_like_author_query(query):
        return "author"
    if any(
        marker in query.casefold()
        for marker in ("citation", "reference", "استشهاد", "مراجع", "اقتباس")
    ):
        return "citations"
    if _looks_like_latest_query(query):
        return "latest_research"
    if query.strip().startswith(("\"", "“", "'")) and query.strip().endswith(
        ("\"", "”", "'")
    ):
        return "exact_title"
    return "topic"


class AcademicSearchService:
    def __init__(
        self,
        crossref: CrossrefProvider | None = None,
        datacite: DataCiteProvider | None = None,
        openalex: OpenAlexProvider | None = None,
    ) -> None:
        self.crossref = crossref or CrossrefProvider()
        self.datacite = datacite or DataCiteProvider()
        self.openalex = openalex or OpenAlexProvider()
        self.resolver = DOIResolver(self.crossref, self.datacite, self.openalex)

    def health(self) -> dict[str, object]:
        openalex_health = self.openalex.health()
        openalex_health["api_key_configured"] = bool(OPENALEX_API_KEY)
        openalex_health["budget_mode"] = "key" if OPENALEX_API_KEY else "keyless"
        provider_health = {
            "crossref": self.crossref.health(),
            "datacite": self.datacite.health(),
            "openalex": openalex_health,
        }
        statuses = [str(item.get("status")) for item in provider_health.values()]
        overall_status = (
            "ERROR"
            if all(status in {"ERROR", "NOT_CONFIGURED"} for status in statuses)
            else "DEGRADED"
            if any(status in {"ERROR", "RATE_LIMITED"} for status in statuses)
            else "READY"
        )
        return {
            "status": overall_status,
            "providers": provider_health,
            "privacy_scope": "WEB",
            "raw_payload_retention": False,
        }

    async def search(
        self,
        query: str,
        *,
        intent: str = "auto",
        max_results: int = 5,
    ) -> dict[str, object]:
        if not isinstance(query, str) or not query.strip():
            return {"ok": False, "error": "query_required", "results": []}
        query = query.strip()
        if len(query) > MAX_QUERY_LENGTH:
            return {"ok": False, "error": "query_too_long", "results": []}
        if intent not in ACADEMIC_INTENTS:
            return {"ok": False, "error": "invalid_intent", "results": []}
        if not isinstance(max_results, int) or not 1 <= max_results <= MAX_RESULTS:
            return {"ok": False, "error": "invalid_max_results", "results": []}

        resolved_intent = _detect_intent(query) if intent == "auto" else intent
        if resolved_intent == "doi":
            return await self.resolver.resolve(extract_doi(query) or query)

        if resolved_intent == "citations" and extract_doi(query):
            doi_response = await self.resolver.resolve(extract_doi(query) or query)
            normalized_doi = str(doi_response.get("normalized_doi") or "")
            openalex_result = (
                await self.openalex.lookup_doi(normalized_doi)
                if normalized_doi
                else None
            )
            graph = (
                await self.openalex.graph(
                    str(openalex_result.identifiers["openalex_id"]),
                    max_results,
                )
                if openalex_result
                and openalex_result.identifiers.get("openalex_id")
                else {"references": [], "cited_by": []}
            )
            return {
                **doi_response,
                "ok": bool(doi_response.get("ok")),
                "intent": "citations",
                "providers_used": [
                    *(
                        ["crossref"]
                        if doi_response.get("registration_agency") == "crossref"
                        else ["datacite"]
                        if doi_response.get("registration_agency") == "datacite"
                        else []
                    ),
                    "openalex",
                ],
                "graph": graph,
                "graph_edges": "loaded_on_demand",
            }

        results: list[AcademicResult] = []
        response: dict[str, object] = {
            "ok": True,
            "intent": resolved_intent,
            "query": query,
            "providers_used": [],
        }
        if resolved_intent == "exact_title":
            results = await self.crossref.search_title(query, max_results)
            response["providers_used"] = ["crossref"]
            if len(results) != 1:
                openalex_results = await self.openalex.search_title(query, max_results)
                results.extend(openalex_results)
                response["providers_used"] = ["crossref", "openalex"]
        elif resolved_intent == "author":
            response.update(await self.openalex.search_author(query, max_results))
            response["providers_used"] = ["openalex"]
            return response
        elif resolved_intent in {"topic", "latest_research", "citations"}:
            results = await self.openalex.search_topic(
                query,
                max_results,
                latest=resolved_intent == "latest_research",
            )
            response["providers_used"] = ["openalex"]
        merged = merge_academic_results(results)
        response["results"] = [result.to_dict() for result in merged[:max_results]]
        response["result_count"] = len(merged[:max_results])
        response["deduplicated_count"] = len(merged)
        if resolved_intent == "citations":
            response["graph_edges"] = "on_demand"
        return response


_academic_search_service: AcademicSearchService | None = None


def _get_academic_search_service() -> AcademicSearchService:
    global _academic_search_service
    if _academic_search_service is None:
        _academic_search_service = AcademicSearchService()
    return _academic_search_service


async def academic_search(
    query: str,
    intent: str = "auto",
    max_results: int = 5,
) -> dict[str, object]:
    """Search structured academic sources without touching private file data."""
    return await _get_academic_search_service().search(
        query,
        intent=intent,
        max_results=max_results,
    )


def academic_health() -> dict[str, object]:
    return _get_academic_search_service().health()