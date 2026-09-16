from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import auth
import server as server_module
from my_files import FileProcessingError, sanitize_filename, validate_file_content
from server import OwnerMCPAuthMiddleware
from starlette.testclient import TestClient


async def _fake_app(scope, receive, send) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": 204,
            "headers": [],
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": b"",
        }
    )


def _run_middleware(
    middleware: OwnerMCPAuthMiddleware,
    *,
    path: str,
    authorization: str = "",
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "path": path,
        "method": "POST",
        "headers": [
            (b"authorization", authorization.encode("utf-8"))
        ]
        if authorization
        else [],
    }
    asyncio.run(middleware(scope, receive, send))
    return messages


class SecurityBoundaryTests(unittest.TestCase):
    def test_mcp_rejects_missing_owner_token(self) -> None:
        middleware = OwnerMCPAuthMiddleware(_fake_app)
        with patch.object(auth, "AHMED_OWNER_TOKEN", "owner-secret"), patch.object(
            auth, "record_auth_event", new=AsyncMock()
        ):
            messages = _run_middleware(middleware, path="/mcp")
        self.assertEqual(messages[0]["status"], 401)

    def test_mcp_accepts_valid_owner_token(self) -> None:
        middleware = OwnerMCPAuthMiddleware(_fake_app)
        with patch.object(auth, "AHMED_OWNER_TOKEN", "owner-secret"), patch.object(
            auth, "record_auth_event", new=AsyncMock()
        ):
            messages = _run_middleware(
                middleware,
                path="/mcp",
                authorization="Bearer owner-secret",
            )
        self.assertEqual(messages[0]["status"], 204)

    def test_mcp_initialize_succeeds_after_lifespan_startup(self) -> None:
        with patch.object(auth, "AHMED_OWNER_TOKEN", "owner-secret"), patch.object(
            auth, "record_auth_event", new=AsyncMock()
        ), patch.object(
            server_module,
            "mark_orphaned_runs",
            new=AsyncMock(return_value=[]),
        ):
            app = OwnerMCPAuthMiddleware(
                server_module.server.streamable_http_app(
                    streamable_http_path="/mcp",
                    host="127.0.0.1",
                ),
                path="/mcp",
            )
            with TestClient(app, base_url="http://127.0.0.1") as client:
                response = client.post(
                    "/mcp",
                    headers={
                        "Authorization": "Bearer owner-secret",
                        "Host": "127.0.0.1:8000",
                        "Accept": "application/json, text/event-stream",
                        "Content-Type": "application/json",
                        "MCP-Protocol-Version": "2025-03-26",
                    },
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2025-03-26",
                            "capabilities": {},
                            "clientInfo": {
                                "name": "security-boundary-test",
                                "version": "1",
                            },
                        },
                    },
                )
        self.assertEqual(response.status_code, 200)
        self.assertIn("result", response.text)

    def test_filename_canonicalization_rejects_control_and_empty_names(self) -> None:
        self.assertEqual(sanitize_filename("../private.txt"), "private.txt")
        self.assertEqual(sanitize_filename(r"..\private.txt"), "private.txt")
        self.assertEqual(sanitize_filename(""), "")
        self.assertEqual(sanitize_filename("bad\x00.txt"), "")

    def test_ingestion_rejects_noncanonical_internal_filename(self) -> None:
        with self.assertRaises(FileProcessingError) as context:
            validate_file_content("../private.txt", b"safe text")
        self.assertEqual(context.exception.status, "invalid_filename")


if __name__ == "__main__":
    unittest.main()