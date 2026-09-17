from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass

from starlette.requests import Request

from config import AHMED_OWNER_TOKEN
from persistence import record_auth_event

# Failed-attempt throttling (Issue: auth backoff). In-memory is sufficient for the
# single-process, single-owner deployment; it resets on restart by design.
_AUTH_BACKOFF_BASE_SECONDS = 1.0
_AUTH_BACKOFF_MAX_SECONDS = 60.0
_AUTH_FAILURE_RESET_SECONDS = 15 * 60
_failed_attempts: dict[str, tuple[int, float]] = {}


def _client_key(request: Request) -> str:
    client = getattr(request, "client", None)
    return getattr(client, "host", None) or "unknown"


def _throttle_delay(key: str) -> float:
    """Seconds the client must still wait before another attempt is processed."""
    entry = _failed_attempts.get(key)
    if not entry:
        return 0.0
    failures, last_failure = entry
    if time.monotonic() - last_failure > _AUTH_FAILURE_RESET_SECONDS:
        _failed_attempts.pop(key, None)
        return 0.0
    delay = min(_AUTH_BACKOFF_BASE_SECONDS * (2 ** (failures - 1)), _AUTH_BACKOFF_MAX_SECONDS)
    remaining = delay - (time.monotonic() - last_failure)
    return max(0.0, remaining)


def _record_failure(key: str) -> None:
    failures, _ = _failed_attempts.get(key, (0, 0.0))
    _failed_attempts[key] = (failures + 1, time.monotonic())


def _clear_failures(key: str) -> None:
    _failed_attempts.pop(key, None)


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    display_name: str | None = None


_OWNER_USER = AuthenticatedUser(user_id="owner", display_name="owner")


async def get_authenticated_user(request: Request) -> AuthenticatedUser | None:
    """Authenticate the single owner using only the configured bearer secret."""

    authorization = request.headers.get("authorization", "")
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer":
        return None
    token = token.strip()
    if not AHMED_OWNER_TOKEN or not token:
        return None
    if secrets.compare_digest(token, AHMED_OWNER_TOKEN):
        return _OWNER_USER
    return None


async def authorize_owner(
    request: Request,
    *,
    endpoint: str,
    action_id: str | None = None,
) -> tuple[AuthenticatedUser | None, int, str | None]:
    key = _client_key(request)
    delay = _throttle_delay(key)
    if delay > 0:
        await _record_auth_attempt(
            endpoint=endpoint,
            action_id=action_id,
            authenticated=False,
            result="throttled",
        )
        # Uniform response: do not reveal throttle state details to the caller.
        await asyncio.sleep(min(delay, _AUTH_BACKOFF_BASE_SECONDS))
        return None, 429, "AUTH_THROTTLED"
    user = await get_authenticated_user(request)
    if user is None:
        _record_failure(key)
        await _record_auth_attempt(
            endpoint=endpoint,
            action_id=action_id,
            authenticated=False,
            result="unauthenticated",
        )
        return None, 401, "AUTHENTICATION_REQUIRED"
    _clear_failures(key)
    await _record_auth_attempt(
        endpoint=endpoint,
        action_id=action_id,
        authenticated=True,
        result="authenticated",
    )
    return user, 200, None


async def _record_auth_attempt(
    *,
    endpoint: str,
    action_id: str | None,
    authenticated: bool,
    result: str,
) -> None:
    try:
        await record_auth_event(
            principal="owner",
            authenticated=authenticated,
            endpoint=endpoint,
            action_id=action_id,
            result=result,
        )
    except Exception:
        # Authorization must never become dependent on audit availability.
        return


def auth_health() -> dict[str, object]:
    return {
        "status": "READY" if AHMED_OWNER_TOKEN else "NOT_CONFIGURED",
        "mode": "single_owner_bearer_token",
        "owner_token_configured": bool(AHMED_OWNER_TOKEN),
        "safe_error_code": None if AHMED_OWNER_TOKEN else "OWNER_TOKEN_NOT_CONFIGURED",
    }
