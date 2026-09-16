import asyncio
import ipaddress
import json
import logging
import os
import re
import socket
import time
import uuid
from collections import defaultdict
from html.parser import HTMLParser
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request as UrlRequest
from urllib.request import urlopen


logger = logging.getLogger("ahmed_agent.web_search")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"
PAGE_FETCH_TIMEOUT_SECONDS = 15
MAX_QUERY_LENGTH = 1000
MAX_RESULTS = 10
MAX_RESEARCH_QUERIES = 3
MAX_RESEARCH_CANDIDATES = 12
RESEARCH_MODES = {"FAST", "DEEP"}
MAX_DEEP_SOURCES = 6
DEEP_CREDIT_BUDGET = 12.0
MAX_CONCURRENT_SEARCHES = 2
MAX_TAVILY_RETRIES = 2
TEMPORARY_TAVILY_STATUSES = {429, 500, 502, 503, 504}
_tavily_health_state: dict[str, object] = {
    "last_success": None,
    "last_failure": None,
    "last_failure_code": None,
    "latency_ms": None,
}
TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "referrer",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
}


class _PageTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self._title_parts: list[str] = []
        self._text_parts: list[str] = []
        self._published_dates: list[str] = []
        self._other_dates: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key.lower(): value or "" for key, value in attrs}
        lowered_tag = tag.lower()
        if lowered_tag in {"script", "style", "noscript", "template", "svg"}:
            self._skip_depth += 1
        if lowered_tag == "title":
            self._in_title = True
        if lowered_tag == "meta":
            key = (
                attributes.get("property")
                or attributes.get("name")
                or attributes.get("itemprop")
                or ""
            ).lower()
            if any(marker in key for marker in ("date", "published", "modified", "created")):
                content = attributes.get("content", "").strip()
                if content:
                    if "published" in key or "datepublished" in key:
                        self._published_dates.append(content)
                    else:
                        self._other_dates.append(content)
        if lowered_tag == "time":
            date_value = attributes.get("datetime", "").strip()
            if date_value:
                self._published_dates.append(date_value)

    def handle_endtag(self, tag: str) -> None:
        lowered_tag = tag.lower()
        if lowered_tag == "title":
            self._in_title = False
        if lowered_tag in {"script", "style", "noscript", "template", "svg"}:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title:
            self._title_parts.append(text)
        self._text_parts.append(text)

    @property
    def title(self) -> str:
        return " ".join(self._title_parts).strip()

    @property
    def text(self) -> str:
        return " ".join(self._text_parts).strip()

    @property
    def published_at(self) -> str | None:
        candidate = (
            self._published_dates[0]
            if self._published_dates
            else (self._other_dates[0] if self._other_dates else None)
        )
        return _valid_date_value(candidate)


def _valid_date_value(value: str | None) -> str | None:
    if not value or not re.search(r"\b(?:19|20)\d{2}\b", value):
        return None
    return value


def _is_public_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return address.is_global


async def _is_safe_public_url(url: str) -> bool:
    parsed = urlparse(url)
    hostname = parsed.hostname
    if (
        parsed.scheme not in {"http", "https"}
        or not hostname
        or parsed.username
        or parsed.password
        or hostname.lower() in {"localhost", "localhost.localdomain"}
        or hostname.lower().endswith(".local")
    ):
        return False
    if _is_public_ip(hostname):
        return True
    try:
        addresses = await asyncio.to_thread(
            socket.getaddrinfo,
            hostname,
            443 if parsed.scheme == "https" else 80,
            type=socket.SOCK_STREAM,
        )
    except OSError:
        return False
    return bool(addresses) and all(_is_public_ip(item[4][0]) for item in addresses)


def _fetch_page_sync(url: str) -> tuple[str, str]:
    request = UrlRequest(
        url,
        headers={
            "User-Agent": "Ahmed-Agent/1.0",
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
        },
    )
    with urlopen(request, timeout=PAGE_FETCH_TIMEOUT_SECONDS) as response:
        content_type = response.headers.get_content_type()
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace"), content_type


async def _fetch_page(
    url: str,
    fallback_published_at: str | None = None,
) -> dict[str, str | None] | None:
    if not await _is_safe_public_url(url):
        logger.error("Skipped unsafe or non-public URL: %s", url)
        return None
    try:
        html, content_type = await asyncio.to_thread(_fetch_page_sync, url)
        parser = _PageTextParser()
        if content_type in {"text/html", "application/xhtml+xml"}:
            parser.feed(html)
            title = parser.title
            text = parser.text
            published_at = parser.published_at
        else:
            title = ""
            text = " ".join(html.split())
            published_at = None
        words = text.split()
        if not words:
            logger.error("Opened page but extracted no text: %s", url)
            return None
        if len(words) < 200:
            logger.error(
                "Skipped page with insufficient extracted text url=%s words=%d",
                url,
                len(words),
            )
            return None
        snippet = " ".join(words[:350])
        logger.info(
            "Opened page url=%s content_type=%s words=%d",
            url,
            content_type,
            len(words),
        )
        return {
            "title": title or urlparse(url).netloc,
            "url": url,
            "date": _valid_date_value(published_at)
            or _valid_date_value(fallback_published_at),
            "published_at": _valid_date_value(published_at)
            or _valid_date_value(fallback_published_at),
            "snippet": snippet,
        }
    except Exception as error:
        logger.exception("Failed to fetch page url=%s error=%s", url, error)
        return None


def _tavily_post_sync(
    endpoint: str,
    payload: dict[str, object],
    api_key: str,
) -> tuple[int, dict[str, object]]:
    request_payload = {**payload, "api_key": api_key}
    request = UrlRequest(
        endpoint,
        data=json.dumps(request_payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Ahmed-Agent/1.0",
        },
        method="POST",
    )
    with urlopen(request, timeout=PAGE_FETCH_TIMEOUT_SECONDS) as response:
        response_status = response.status
        response_body = response.read().decode("utf-8", errors="replace")
    parsed_response = json.loads(response_body)
    if not isinstance(parsed_response, dict):
        raise ValueError("Tavily returned a non-object response.")
    return response_status, parsed_response


def _credits_from_response(response: dict[str, object]) -> float | None:
    usage = response.get("usage")
    candidates: list[object] = [usage, response]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key in ("credits_used", "credits", "credit"):
            value = candidate.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


async def _tavily_post(
    endpoint: str,
    payload: dict[str, object],
    api_key: str,
    mode: str,
) -> tuple[dict[str, object] | None, float | None]:
    query_id = uuid.uuid4().hex[:12]
    for attempt in range(MAX_TAVILY_RETRIES + 1):
        started_at = time.perf_counter()
        response_status: int | str = "error"
        try:
            response_status, response = await asyncio.to_thread(
                _tavily_post_sync,
                endpoint,
                payload,
                api_key,
            )
            elapsed_ms = round((time.perf_counter() - started_at) * 1000)
            credits = _credits_from_response(response)
            logger.info(
                "Tavily request endpoint=%s mode=%s query_id=%s status=%s "
                "elapsed_ms=%d credits_used=%s",
                endpoint,
                mode,
                query_id,
                response_status,
                elapsed_ms,
                credits if credits is not None else "unknown",
            )
            return response, credits
        except HTTPError as error:
            response_status = error.code
            elapsed_ms = round((time.perf_counter() - started_at) * 1000)
            logger.info(
                "Tavily request endpoint=%s mode=%s query_id=%s status=%s "
                "elapsed_ms=%d credits_used=unknown",
                endpoint,
                mode,
                query_id,
                response_status,
                elapsed_ms,
            )
            if response_status not in TEMPORARY_TAVILY_STATUSES or attempt >= MAX_TAVILY_RETRIES:
                logger.exception("Tavily request failed after retry policy.")
                return None, None
        except (OSError, TimeoutError, ValueError, json.JSONDecodeError):
            elapsed_ms = round((time.perf_counter() - started_at) * 1000)
            logger.info(
                "Tavily request endpoint=%s mode=%s query_id=%s status=error "
                "elapsed_ms=%d credits_used=unknown",
                endpoint,
                mode,
                query_id,
                elapsed_ms,
            )
            if attempt >= MAX_TAVILY_RETRIES:
                logger.exception("Tavily request failed after retry policy.")
                return None, None
        await asyncio.sleep(0.25 * (2**attempt))
    return None, None


def _valid_query(query: str) -> str:
    return query.strip() if isinstance(query, str) else ""


def _domain_for_url(url: str) -> str:
    hostname = (urlparse(url).hostname or "").lower()
    return hostname.removeprefix("www.")


def _normalized_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.casefold() not in TRACKING_QUERY_KEYS
    ]
    query = urlencode(sorted(query_items))
    return urlunparse(
        (
            parsed.scheme.lower(),
            (parsed.hostname or "").lower(),
            path,
            "",
            query,
            "",
        )
    )


def _classify_query(query: str) -> str:
    lowered = query.casefold()
    if any(
        marker in lowered
        for marker in (
            "official",
            "primary source",
            "رسمي",
            "المصدر الأصلي",
            "توثيق رسمي",
        )
    ):
        return "official"
    technical_markers = (
        "github",
        "documentation",
        "docs",
        "api",
        "sdk",
        "python",
        "javascript",
        "technical",
        "توثيق",
        "برمجة",
        "تقني",
    )
    if any(
        marker in lowered
        for marker in (
            "breaking news",
            "current news",
            "news",
            "أخبار",
            "خبر",
            "آخر المستجدات",
            "اليوم",
        )
    ) or ("latest" in lowered and not any(marker in lowered for marker in technical_markers)):
        return "current_news"
    if any(
        marker in lowered
        for marker in (
            "finance",
            "financial",
            "stock",
            "stocks",
            "market",
            "سهم",
            "أسهم",
            "مالي",
            "أسواق",
        )
    ):
        return "finance"
    if any(
        marker in lowered
        for marker in (
            "academic",
            "scientific",
            "research paper",
            "study",
            "arxiv",
            "evidence",
            "consensus",
            "disputed",
            "controversial",
            "بحث علمي",
            "دراسة",
            "أكاديمي",
        )
    ):
        return "academic"
    if any(
        marker in lowered
        for marker in (
            "law",
            "legal",
            "regulation",
            "government",
            "قانون",
            "قانوني",
            "حكومي",
            "لائحة",
        )
    ):
        return "legal_government"
    if any(
        marker in lowered
        for marker in (
            *technical_markers,
        )
    ):
        return "technical"
    if any(
        marker in lowered
        for marker in ("reddit", "forum", "community", "social", "مجتمع", "منتدى")
    ):
        return "social_community"
    return "general_web"


def _research_queries(query: str, query_type: str, mode: str) -> list[str]:
    queries = [query]
    if mode == "FAST":
        return queries
    if query_type == "general_web" and not any(
        marker in query.casefold() for marker in (" and ", " or ", "compare", "مقارنة", "تحقيق")
    ):
        return queries

    suffixes = {
        "official": (
            "official primary documentation",
            "official repository or organization source",
        ),
        "current_news": (
            "latest official statement news",
            "international English coverage independent sources",
        ),
        "finance": (
            "market data primary financial source",
            "independent financial analysis",
        ),
        "academic": (
            "research paper study primary academic source",
            "site:edu OR site:ac.uk academic evidence",
        ),
        "technical": (
            "official documentation repository",
            "implementation guide independent technical source",
        ),
        "legal_government": (
            "official government primary source",
            "regulation text government or court source",
        ),
        "social_community": (
            "community discussion independent sources",
            "English and international community sources",
        ),
        "general_web": (
            "English international sources",
            "independent analysis and primary sources",
        ),
    }[query_type]
    for suffix in suffixes:
        candidate = f"{query} {suffix}"
        if candidate not in queries:
            queries.append(candidate)
    return queries[:MAX_RESEARCH_QUERIES]


def _temporal_filters(query: str, query_type: str) -> dict[str, str]:
    lowered = query.casefold()
    if query_type == "current_news" or any(
        marker in lowered
        for marker in (
            "breaking news",
            "current news",
            "news today",
            "today's news",
            "أخبار اليوم",
            "خبر عاجل",
            "هذا الأسبوع",
        )
    ):
        return {"time_range": "week" if any(
            marker in lowered
            for marker in ("today", "today's", "this week", "هذا الأسبوع", "اليوم")
        ) else "month"}
    return {}


def _source_type(
    domain: str,
    url: str,
    title: str,
    query_type: str,
) -> tuple[str, str, float]:
    lowered_domain = domain.casefold()
    lowered_url = url.casefold()
    official_documentation_domains = (
        "modelcontextprotocol.io",
        "python.org",
        "openai.com",
        "docs.anthropic.com",
        "docs.github.com",
    )
    academic_domains = (
        "arxiv.org",
        "frontiersin.org",
        "nature.com",
        "sciencedirect.com",
        "springer.com",
        "pmc.ncbi.nlm.nih.gov",
        "pubmed.ncbi.nlm.nih.gov",
    )
    if any(
        lowered_domain == host or lowered_domain.endswith(f".{host}")
        for host in official_documentation_domains
    ):
        return "official_documentation", "primary", 0.98
    if lowered_domain.endswith("europa.eu") or lowered_domain.endswith(".int"):
        return "government", "primary", 0.95
    if any(
        lowered_domain == host or lowered_domain.endswith(f".{host}")
        for host in academic_domains
    ):
        return "academic", "secondary", 0.85
    if lowered_domain.endswith(".gov") or ".gov." in lowered_domain:
        return "government", "primary", 0.95
    if lowered_domain.endswith(".edu") or ".ac." in lowered_domain:
        return "academic", "secondary", 0.8
    if lowered_domain == "github.com" or lowered_domain.endswith(".github.io"):
        if any(
            marker in lowered_url
            for marker in ("modelcontextprotocol", "openai", "python", "official")
        ):
            return "official_repository", "primary", 0.98
        return "repository", "secondary", 0.75
    if any(marker in lowered_domain for marker in ("wikipedia.org", "reddit.com", "x.com")):
        return "social_community", "community", 0.7
    if any(
        marker in lowered_domain
        for marker in ("news", "reuters.com", "apnews.com", "bbc.com")
    ):
        return "news", "secondary", 0.8
    if query_type == "social_community":
        return "social_community", "community", 0.65
    return "general_web", "unknown", 0.45


def _enrich_source(
    page: dict[str, str | None],
    tavily_item: dict[str, object],
    query: str,
    query_type: str,
) -> dict[str, object]:
    url = str(page.get("url") or "")
    title = str(page.get("title") or "")
    domain = _domain_for_url(url)
    source_type, classification, confidence = _source_type(
        domain,
        url,
        title,
        query_type,
    )
    published_at = _valid_date_value(
        str(page.get("published_at") or tavily_item.get("published_date") or "")
    )
    score_value = tavily_item.get("score")
    try:
        tavily_score = float(score_value) if score_value is not None else 0.0
    except (TypeError, ValueError):
        tavily_score = 0.0
    priority = 100 if classification == "primary" else 0
    if source_type in {"official_documentation", "official_repository", "government", "academic"}:
        priority += 25
    priority += round(tavily_score * 20, 4)
    if published_at:
        priority += 2
    return {
        "title": title,
        "url": url,
        "domain": domain,
        "published_at": str(published_at) if published_at else None,
        "date": str(published_at) if published_at else None,
        "source_type": source_type,
        "classification": classification,
        "confidence": confidence,
        "primary_or_secondary": classification,
        "content": page.get("content") or page.get("snippet") or "",
        "snippet": page.get("content") or page.get("snippet") or "",
        "query": query,
        "tavily_score": tavily_score,
        "_priority": priority,
    }


async def _run_tavily_query(
    query: str,
    max_results: int,
    api_key: str,
    mode: str,
    query_type: str,
    include_domains: list[str] | None = None,
) -> tuple[list[dict[str, object]], float | None]:
    topic = "news" if query_type == "current_news" else "general"
    if query_type == "finance":
        topic = "finance"
    payload: dict[str, object] = {
        "query": query,
        "max_results": max_results,
        "search_depth": "basic" if mode == "FAST" else "advanced",
        "topic": topic,
        "include_answer": False,
        "include_usage": True,
    }
    payload.update(_temporal_filters(query, query_type))
    if mode == "DEEP" and include_domains:
        payload["include_domains"] = include_domains
    tavily_response, credits = await _tavily_post(
        TAVILY_SEARCH_URL,
        payload,
        api_key,
        mode,
    )
    if tavily_response is None:
        return [], credits

    raw_results = tavily_response.get("results", [])
    if not isinstance(raw_results, list):
        logger.error("Tavily response has invalid results field.")
        return [], credits

    candidates: list[dict[str, object]] = []
    for item in raw_results[:max_results]:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not isinstance(url, str) or not url:
            logger.error("Tavily result had no usable URL.")
            continue
        candidates.append(
            {
                "url": url,
                "title": str(item.get("title") or url),
                "content": str(item.get("content") or ""),
                "published_at": _valid_date_value(
                    str(item.get("published_date") or "")
                ),
                "tavily_score": item.get("score"),
                "_tavily_item": item,
                "_query": query,
            }
        )
    return candidates, credits


def _rank_and_deduplicate(
    candidates: list[dict[str, object]],
    query_type: str,
    mode: str,
) -> tuple[list[dict[str, object]], int]:
    best_by_url: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        url = str(candidate.get("url") or "")
        normalized = _normalized_url(url)
        if not normalized:
            continue
        existing = best_by_url.get(normalized)
        if existing is None or candidate.get("_priority", 0) > existing.get("_priority", 0):
            best_by_url[normalized] = candidate

    ordered = sorted(
        best_by_url.values(),
        key=lambda item: float(item.get("_priority", 0)),
        reverse=True,
    )
    selected: list[dict[str, object]] = []
    domains_used: dict[str, int] = defaultdict(int)
    limit = 5 if mode == "FAST" else MAX_DEEP_SOURCES
    for item in ordered:
        domain = str(item.get("domain") or "")
        if domains_used[domain] >= 2:
            continue
        domains_used[domain] += 1
        item = {
            key: value
            for key, value in item.items()
            if not key.startswith("_")
        }
        selected.append(item)
        if len(selected) >= limit:
            break
    logger.info(
        "Research ranking completed mode=%s query_type=%s candidates=%d selected=%d "
        "independent_domains=%d",
        mode,
        query_type,
        len(candidates),
        len(selected),
        len({item.get("domain") for item in selected}),
    )
    return selected, len(candidates)


async def _run_tavily_extract(
    urls: list[str],
    mode: str,
    api_key: str,
    extract_depth: str,
) -> tuple[dict[str, dict[str, object]], float | None]:
    if not urls:
        return {}, None
    response, credits = await _tavily_post(
        TAVILY_EXTRACT_URL,
        {
            "urls": urls,
            "extract_depth": extract_depth,
            "include_images": False,
            "include_usage": True,
        },
        api_key,
        mode,
    )
    if response is None:
        return {}, credits

    extracted: dict[str, dict[str, object]] = {}
    raw_results = response.get("results", [])
    if isinstance(raw_results, list):
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if not isinstance(url, str):
                continue
            content = item.get("raw_content") or item.get("content") or ""
            extracted[_normalized_url(url)] = {
                "content": str(content),
                "status": "extracted" if str(content).strip() else "partial",
            }

    failed_results = response.get("failed_results", [])
    if isinstance(failed_results, list):
        for item in failed_results:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if isinstance(url, str):
                extracted.setdefault(
                    _normalized_url(url),
                    {"content": "", "status": "failed"},
                )
    for url in urls:
        extracted.setdefault(
            _normalized_url(url),
            {"content": "", "status": "failed"},
        )
    return extracted, credits


def _official_domains_for_query(query: str, query_type: str) -> list[str]:
    lowered = query.casefold()
    if query_type != "official":
        return []
    domains: list[str] = []
    if "mcp" in lowered or "model context protocol" in lowered:
        domains.extend(
            [
                "modelcontextprotocol.io",
                "github.com/modelcontextprotocol",
                "py.sdk.modelcontextprotocol.io",
            ]
        )
    if "python" in lowered:
        domains.extend(["docs.python.org", "python.org", "github.com/python"])
    if "openai" in lowered:
        domains.extend(["platform.openai.com", "openai.com"])
    return list(dict.fromkeys(domains))


async def _deep_extract(
    selected: list[dict[str, object]],
    mode: str,
    api_key: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    urls = [str(item.get("url")) for item in selected if item.get("url")]
    basic, basic_credits = await _run_tavily_extract(
        urls,
        mode,
        api_key,
        "basic",
    )
    advanced_urls = [
        url
        for url in urls
        if basic.get(_normalized_url(url), {}).get("status") in {"failed", "partial"}
    ]
    advanced: dict[str, dict[str, object]] = {}
    advanced_credits: float | None = None
    if advanced_urls:
        advanced, advanced_credits = await _run_tavily_extract(
            advanced_urls,
            mode,
            api_key,
            "advanced",
        )

    final_results: list[dict[str, object]] = []
    for item in selected:
        url = str(item.get("url") or "")
        extraction = advanced.get(_normalized_url(url)) or basic.get(
            _normalized_url(url),
            {"content": "", "status": "failed"},
        )
        content = str(extraction.get("content") or "")
        updated = dict(item)
        if content.strip():
            updated["content"] = content
            updated["snippet"] = content[:2500]
        updated["extraction_status"] = extraction.get("status", "failed")
        final_results.append(updated)

    credits = [value for value in (basic_credits, advanced_credits) if value is not None]
    return final_results, {
        "extract_calls": 1 + (1 if advanced_urls else 0),
        "urls_extracted": sum(
            1
            for item in final_results
            if item.get("extraction_status") == "extracted"
        ),
        "failed_extractions": sum(
            1
            for item in final_results
            if item.get("extraction_status") == "failed"
        ),
        "extract_credits": sum(credits) if len(credits) == 2 else (
            credits[0] if credits else None
        ),
    }


async def _tavily_web_search(
    query: str,
    max_results: int = 5,
    mode: str = "FAST",
) -> dict[str, object]:
    query = _valid_query(query)
    mode = mode.upper().strip() if isinstance(mode, str) else ""
    if not query or len(query) > MAX_QUERY_LENGTH:
        return {
            "ok": False,
            "error": "يجب أن يكون البحث بين 1 و1000 حرف.",
            "results": [],
        }
    if mode not in RESEARCH_MODES:
        return {
            "ok": False,
            "error": "وضع البحث يجب أن يكون FAST أو DEEP.",
            "results": [],
        }
    if not isinstance(max_results, int) or not 1 <= max_results <= MAX_RESULTS:
        return {
            "ok": False,
            "error": f"عدد النتائج يجب أن يكون بين 1 و{MAX_RESULTS}.",
            "results": [],
        }

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        logger.error("TAVILY_API_KEY is not configured")
        _tavily_health_state["last_failure"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        _tavily_health_state["last_failure_code"] = "TAVILY_NOT_CONFIGURED"
        return {
            "ok": False,
            "error": "البحث غير متاح حاليًا لأن إعداد البحث غير مكتمل.",
            "results": [],
        }

    query_type = _classify_query(query)
    started_at = time.perf_counter()
    if mode == "FAST":
        candidates, credits = await _run_tavily_query(
            query,
            5,
            api_key,
            mode,
            query_type,
        )
        if not candidates:
            _tavily_health_state["last_failure"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            _tavily_health_state["last_failure_code"] = "TAVILY_NO_RESULTS"
            return {
                "ok": False,
                "error": "تعذر تنفيذ البحث الآن. حاول مرة أخرى لاحقًا.",
                "results": [],
                "mode": mode,
                "query_type": query_type,
                "search_calls": 1,
                "extract_calls": 0,
                "credits_used": credits,
            }
        enriched = [
            _enrich_source(
                candidate,
                candidate.get("_tavily_item")
                if isinstance(candidate.get("_tavily_item"), dict)
                else {},
                query,
                query_type,
            )
            for candidate in candidates
        ]
        results, candidate_count = _rank_and_deduplicate(
            enriched,
            query_type,
            mode,
        )
        _tavily_health_state["last_success"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        _tavily_health_state["last_failure"] = None
        _tavily_health_state["last_failure_code"] = None
        _tavily_health_state["latency_ms"] = int(
            (time.perf_counter() - started_at) * 1000
        )
        return {
            "ok": True,
            "query": query,
            "mode": mode,
            "query_type": query_type,
            "queries": [query],
            "search_calls": 1,
            "extract_calls": 0,
            "credits_used": credits,
            "candidate_count": candidate_count,
            "deduplicated_count": len(results),
            "failed_calls": 0,
            "results": results,
        }

    queries = _research_queries(query, query_type, mode)
    official_domains = _official_domains_for_query(query, query_type)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)

    async def run_query(index: int, related_query: str) -> tuple[list[dict[str, object]], float | None]:
        async with semaphore:
            domains = official_domains if query_type == "official" and index > 0 else None
            return await _run_tavily_query(
                related_query,
                7,
                api_key,
                mode,
                query_type,
                domains,
            )

    query_results: list[tuple[list[dict[str, object]], float | None]] = []
    known_search_credits = 0.0
    for batch_start in range(0, len(queries), MAX_CONCURRENT_SEARCHES):
        batch = queries[batch_start : batch_start + MAX_CONCURRENT_SEARCHES]
        batch_results = await asyncio.gather(
            *(
                run_query(batch_start + index, related_query)
                for index, related_query in enumerate(batch)
            )
        )
        query_results.extend(batch_results)
        known_search_credits += sum(
            value for _, value in batch_results if value is not None
        )
        if known_search_credits >= DEEP_CREDIT_BUDGET:
            break
    candidates: list[dict[str, object]] = []
    credits: list[float] = []
    failed_calls = 0
    for query_candidates, query_credits in query_results:
        if not query_candidates:
            failed_calls += 1
        candidates.extend(
            _enrich_source(
                candidate,
                candidate.get("_tavily_item")
                if isinstance(candidate.get("_tavily_item"), dict)
                else {},
                str(candidate.get("_query") or query),
                query_type,
            )
            for candidate in query_candidates
        )
        if query_credits is not None:
            credits.append(query_credits)

    ranked, candidate_count = _rank_and_deduplicate(candidates, query_type, mode)
    ranked = ranked[:MAX_DEEP_SOURCES]
    if not ranked:
        _tavily_health_state["last_failure"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        _tavily_health_state["last_failure_code"] = "TAVILY_NO_RESULTS"
        return {
            "ok": False,
            "error": "تعذر تنفيذ البحث الآن. حاول مرة أخرى لاحقًا.",
            "results": [],
            "mode": mode,
            "query_type": query_type,
            "search_calls": len(query_results),
            "extract_calls": 0,
            "credits_used": sum(credits) if len(credits) == len(query_results) else None,
            "credit_budget": DEEP_CREDIT_BUDGET,
            "budget_exhausted": known_search_credits >= DEEP_CREDIT_BUDGET,
            "failed_calls": failed_calls,
        }

    if known_search_credits >= DEEP_CREDIT_BUDGET:
        extracted_results = [
            {**item, "extraction_status": "not_run_budget"} for item in ranked
        ]
        extraction_metrics = {
            "extract_calls": 0,
            "urls_extracted": 0,
            "failed_extractions": 0,
            "extract_credits": 0.0,
        }
    else:
        extracted_results, extraction_metrics = await _deep_extract(
            ranked,
            mode,
            api_key,
        )
    total_credits = (
        sum(credits) + extraction_metrics["extract_credits"]
        if extraction_metrics["extract_credits"] is not None
        and len(credits) == len(query_results)
        else None
    )
    _tavily_health_state["last_success"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    _tavily_health_state["last_failure"] = None
    _tavily_health_state["last_failure_code"] = None
    _tavily_health_state["latency_ms"] = int(
        (time.perf_counter() - started_at) * 1000
    )
    return {
        "ok": True,
        "query": query,
        "mode": mode,
        "query_type": query_type,
        "queries": queries,
        "search_calls": len(query_results),
        "extract_calls": extraction_metrics["extract_calls"],
        "credits_used": total_credits,
        "credit_budget": DEEP_CREDIT_BUDGET,
        "budget_exhausted": known_search_credits >= DEEP_CREDIT_BUDGET,
        "candidate_count": candidate_count,
        "deduplicated_count": len(ranked),
        "urls_extracted": extraction_metrics["urls_extracted"],
        "failed_calls": failed_calls + extraction_metrics["failed_extractions"],
        "independent_domains": len({item.get("domain") for item in ranked}),
        "results": extracted_results,
    }


def tavily_health() -> dict[str, object]:
    if not os.environ.get("TAVILY_API_KEY"):
        status = "NOT_CONFIGURED"
    else:
        status = "READY"
    return {
        "status": status,
        "last_success": _tavily_health_state["last_success"],
        "last_failure": _tavily_health_state["last_failure"],
        "latency_ms": _tavily_health_state["latency_ms"],
        "safe_error_code": (
            _tavily_health_state["last_failure_code"]
            if status != "READY"
            else None
        ),
        "fast_probe": "not_run",
    }


_search_fabric_instance: object | None = None


def _get_search_fabric() -> object:
    global _search_fabric_instance
    if _search_fabric_instance is None:
        from search_fabric import BraveProvider, SearchFabric, TavilyProvider

        _search_fabric_instance = SearchFabric(
            TavilyProvider(_tavily_web_search, tavily_health),
            BraveProvider(),
        )
    return _search_fabric_instance


async def web_search(
    query: str,
    max_results: int = 5,
    mode: str = "FAST",
) -> dict[str, object]:
    fabric = _get_search_fabric()
    return await fabric.search(  # type: ignore[union-attr]
        query=query,
        mode=mode,
        max_results=max_results,
        tavily_legacy_search=_tavily_web_search,
    )


def search_fabric_health() -> dict[str, object]:
    fabric = _get_search_fabric()
    return fabric.health()  # type: ignore[union-attr]


def register_skill_tools(server: MCPServer) -> None:
    server.tool(
        description=(
            "Searches the public web through the provider-neutral Search Fabric. "
            "FAST uses Tavily only; DEEP uses selective provider routing, "
            "normalization, deduplication, and rank-based fusion."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
        structured_output=True,
    )(web_search)