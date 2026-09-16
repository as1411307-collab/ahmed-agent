from __future__ import annotations

import os


AHMED_PRIMARY_MODEL = (
    os.environ.get("AHMED_PRIMARY_MODEL", "gemini-flash-lite-latest").strip()
    or "gemini-flash-lite-latest"
)
AHMED_OPENAI_MODEL = (
    os.environ.get("AHMED_OPENAI_MODEL", "gpt-5.6-terra").strip()
    or "gpt-5.6-terra"
)
AHMED_OWNER_TOKEN = os.environ.get("AHMED_OWNER_TOKEN", "").strip()
MY_FILES_EMBEDDING_MODEL = os.environ.get(
    "MY_FILES_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
).strip()
MY_FILES_EMBEDDING_VERSION = os.environ.get(
    "MY_FILES_EMBEDDING_VERSION",
    "v1",
).strip() or "v1"


def _bounded_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def _bounded_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def _bounded_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


CHUNK_SIZE = _bounded_int("CHUNK_SIZE", 1200, minimum=200, maximum=10000)
CHUNK_OVERLAP = _bounded_int("CHUNK_OVERLAP", 200, minimum=0, maximum=2000)
if CHUNK_OVERLAP >= CHUNK_SIZE:
    CHUNK_OVERLAP = max(0, CHUNK_SIZE // 5)

MAX_UPLOAD_BYTES = _bounded_int(
    "MAX_UPLOAD_BYTES",
    10 * 1024 * 1024,
    minimum=1024,
    maximum=50 * 1024 * 1024,
)
MAX_QUERY_LENGTH = _bounded_int("MY_FILES_MAX_QUERY_LENGTH", 500, minimum=1, maximum=2000)
DEFAULT_TOP_K = _bounded_int("MY_FILES_TOP_K", 5, minimum=1, maximum=20)
MAX_TOP_K = _bounded_int("MY_FILES_MAX_TOP_K", 10, minimum=1, maximum=20)
MY_FILES_RRF_K = _bounded_int("MY_FILES_RRF_K", 60, minimum=1, maximum=1000)
MY_FILES_CANDIDATE_K = _bounded_int(
    "MY_FILES_CANDIDATE_K",
    20,
    minimum=5,
    maximum=100,
)
GEMINI_429_MAX_RETRIES = _bounded_int(
    "GEMINI_429_MAX_RETRIES",
    2,
    minimum=0,
    maximum=3,
)
GEMINI_429_CIRCUIT_THRESHOLD = _bounded_int(
    "GEMINI_429_CIRCUIT_THRESHOLD",
    3,
    minimum=1,
    maximum=10,
)
GEMINI_429_COOLDOWN_SECONDS = _bounded_int(
    "GEMINI_429_COOLDOWN_SECONDS",
    30,
    minimum=1,
    maximum=3600,
)
GEMINI_429_BACKOFF_BASE_SECONDS = _bounded_float(
    "GEMINI_429_BACKOFF_BASE_SECONDS",
    1.0,
    minimum=0.1,
    maximum=30.0,
)
GEMINI_429_BACKOFF_MAX_SECONDS = _bounded_float(
    "GEMINI_429_BACKOFF_MAX_SECONDS",
    30.0,
    minimum=0.1,
    maximum=300.0,
)

SEARCH_PROVIDER_TIMEOUT_SECONDS = _bounded_float(
    "SEARCH_PROVIDER_TIMEOUT_SECONDS",
    12.0,
    minimum=2.0,
    maximum=60.0,
)
SEARCH_MAX_PROVIDER_CALLS = _bounded_int(
    "SEARCH_MAX_PROVIDER_CALLS",
    2,
    minimum=1,
    maximum=6,
)
SEARCH_MAX_CONCURRENCY = _bounded_int(
    "SEARCH_MAX_CONCURRENCY",
    2,
    minimum=1,
    maximum=4,
)
SEARCH_MAX_RESULTS_PER_PROVIDER = _bounded_int(
    "SEARCH_MAX_RESULTS_PER_PROVIDER",
    5,
    minimum=1,
    maximum=10,
)
SEARCH_RRF_K = _bounded_int(
    "SEARCH_RRF_K",
    60,
    minimum=1,
    maximum=1000,
)
BRAVE_SEARCH_ENABLED = _bounded_bool("BRAVE_SEARCH_ENABLED", False)

ACADEMIC_PROVIDER_TIMEOUT_SECONDS = _bounded_float(
    "ACADEMIC_PROVIDER_TIMEOUT_SECONDS",
    12.0,
    minimum=2.0,
    maximum=60.0,
)
ACADEMIC_MAX_RETRIES = _bounded_int(
    "ACADEMIC_MAX_RETRIES",
    2,
    minimum=0,
    maximum=3,
)
CROSSREF_MAILTO = os.environ.get("CROSSREF_MAILTO", "").strip()
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "").strip()
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
GITHUB_API_TIMEOUT_SECONDS = _bounded_float(
    "GITHUB_API_TIMEOUT_SECONDS",
    12.0,
    minimum=2.0,
    maximum=60.0,
)
GITHUB_MAX_RETRIES = _bounded_int(
    "GITHUB_MAX_RETRIES",
    2,
    minimum=0,
    maximum=3,
)