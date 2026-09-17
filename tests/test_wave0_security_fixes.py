from __future__ import annotations

import asyncio
import hashlib
import json
import unittest
from io import BytesIO
from unittest.mock import AsyncMock, patch
from urllib.error import HTTPError
from urllib.request import Request as UrlRequest


class SkillToolsSSRFTests(unittest.TestCase):
    """Issue #3: redirects must be re-validated and page bodies bounded."""

    def test_redirect_to_private_ip_is_blocked(self) -> None:
        import skill_tools

        handler = skill_tools._SSRFBlockRedirectHandler()
        req = UrlRequest("https://example.com/page")
        with self.assertRaises(HTTPError):
            handler.redirect_request(req, None, 302, "Found", {}, "http://169.254.169.254/latest/meta-data")

    def test_redirect_to_localhost_is_blocked(self) -> None:
        import skill_tools

        handler = skill_tools._SSRFBlockRedirectHandler()
        req = UrlRequest("https://example.com/page")
        with self.assertRaises(HTTPError):
            handler.redirect_request(req, None, 302, "Found", {}, "http://localhost:8080/admin")

    def test_page_body_is_capped(self) -> None:
        import skill_tools

        big = b"x" * (skill_tools.MAX_PAGE_BYTES + 100)

        class _FakeResponse:
            def __init__(self, body: bytes) -> None:
                self.headers = _FakeHeaders()
                self._body = body

            def read(self, _n: int = -1) -> bytes:
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class _FakeHeaders:
            def get_content_type(self) -> str:
                return "text/plain"

            def get_content_charset(self) -> str:
                return "utf-8"

        with patch.object(skill_tools._SAFE_PAGE_OPENER, "open", return_value=_FakeResponse(big)):
            body, _ = skill_tools._fetch_page_sync("https://example.com/huge")
        self.assertEqual(len(body.encode("utf-8")), skill_tools.MAX_PAGE_BYTES)


class AuthBackoffTests(unittest.IsolatedAsyncioTestCase):
    """Issue #5: failed auth attempts are throttled with exponential backoff."""

    def setUp(self) -> None:
        import auth

        auth._failed_attempts.clear()

    def _request(self, token: str = "wrong-token"):
        import auth  # noqa: F401

        class _Client:
            host = "10.0.0.9"

        class _Request:
            headers = {"authorization": f"Bearer {token}"}
            client = _Client()

        return _Request()

    async def test_repeated_failures_trigger_throttle(self) -> None:
        import auth
        import time

        request = self._request()
        user, status, code = await auth.authorize_owner(request, endpoint="/mcp")
        self.assertIsNone(user)
        self.assertEqual(status, 401)
        self.assertEqual(code, "AUTHENTICATION_REQUIRED")
        # Simulate accumulated failures deterministically (avoids sub-second timing races):
        auth._failed_attempts["10.0.0.9"] = (5, time.monotonic())
        user, status, code = await auth.authorize_owner(request, endpoint="/mcp")
        self.assertIsNone(user)
        self.assertEqual(status, 429)
        self.assertEqual(code, "AUTH_THROTTLED")

    async def test_successful_auth_clears_failures(self) -> None:
        import auth
        import config

        token = config.AHMED_OWNER_TOKEN
        if not token:
            self.skipTest("AHMED_OWNER_TOKEN not configured")
        bad = self._request("wrong-token")
        await auth.authorize_owner(bad, endpoint="/mcp")
        good = self._request(token)
        user, status, _ = await auth.authorize_owner(good, endpoint="/mcp")
        self.assertIsNotNone(user)
        self.assertEqual(status, 200)
        self.assertNotIn("10.0.0.9", auth._failed_attempts)


class MyFilesDedupTests(unittest.IsolatedAsyncioTestCase):
    """Issue #4: uploading identical content returns duplicate without re-ingesting."""

    async def test_duplicate_upload_short_circuits(self) -> None:
        import my_files

        data = b"hello duplicate world"
        file_hash = hashlib.sha256(data).hexdigest()
        existing = {
            "document_id": "11111111-1111-1111-1111-111111111111",
            "filename": "notes.md",
            "status": "fts_ready",
            "chunk_count": 3,
        }
        with patch.object(
            my_files, "find_document_by_hash", new=AsyncMock(return_value=existing)
        ) as find_mock, patch.object(
            my_files, "create_original_source", new=AsyncMock()
        ) as create_mock:
            result = await my_files.ingest_document(
                document_id="22222222-2222-2222-2222-222222222222",
                filename="copy-of-notes.md",
                mime_type="text/markdown",
                data=data,
            )
        find_mock.assert_awaited_once_with(file_hash)
        create_mock.assert_not_awaited()
        self.assertTrue(result["duplicate"])
        self.assertEqual(result["status"], "duplicate")
        self.assertEqual(result["document_id"], existing["document_id"])


class AuditChainBoundedTests(unittest.IsolatedAsyncioTestCase):
    """Issue #6: verify_audit_chain accepts a window limit parameter."""

    async def test_limit_parameter_exists(self) -> None:
        import inspect

        import persistence

        signature = inspect.signature(persistence.verify_audit_chain)
        self.assertIn("limit", signature.parameters)
        self.assertEqual(signature.parameters["limit"].default, 5000)


if __name__ == "__main__":
    unittest.main()
