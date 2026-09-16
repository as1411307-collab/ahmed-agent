from __future__ import annotations

import secrets
from dataclasses import dataclass

from starlette.requests import Request

from config import AHMED_OWNER_TOKEN
from persistence import record_auth_event


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
    user = await get_authenticated_user(request)
    if user is None:
        await _record_auth_attempt(
            endpoint=endpoint,
            action_id=action_id,
            authenticated=False,
            result="unauthenticated",
        )
        return None, 401, "AUTHENTICATION_REQUIRED"
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