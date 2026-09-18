from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import unittest
from types import SimpleNamespace
from io import BytesIO
from unittest.mock import AsyncMock, patch
from urllib.error import HTTPError
from urllib.request import Request as UrlRequest


class SkillToolsSSRFTests(unittest.TestCase):
    """Issues #3/#13: DNS-pinned transport — one validated resolution per hop."""

    def test_connection_uses_the_validated_ip(self) -> None:
        """DNS rebinding: the second answer differs, but the connection must
        target the IP from the single validated resolution (no re-resolve)."""
        import skill_tools

        resolutions = [
            [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))],
            [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
        ]
        resolve_calls = []
        connected_targets = []

        def fake_getaddrinfo(host, port, *args, **kwargs):
            resolve_calls.append((host, port))
            return resolutions[min(len(resolve_calls) - 1, 1)]

        def fake_connect(addresses, timeout):
            connected_targets.append(addresses[0][4])
            raise OSError("test stops before any real socket work")

        with patch.object(
            skill_tools.socket, "getaddrinfo", side_effect=fake_getaddrinfo
        ), patch.object(
            skill_tools, "_connect_pinned_socket", side_effect=fake_connect
        ):
            with self.assertRaises(OSError):
                skill_tools._fetch_page_sync("https://example.com/page")

        self.assertEqual(len(resolve_calls), 1)
        self.assertEqual(connected_targets, [("8.8.8.8", 443)])

    def test_redirect_to_private_ip_is_blocked(self) -> None:
        import skill_tools

        real_pinned_request = skill_tools._pinned_request
        calls = {"count": 0}

        def first_hop_then_real(url: str):
            calls["count"] += 1
            if calls["count"] == 1:
                return 302, _FakeHeaders("http://169.254.169.254/latest/meta-data"), b""
            return real_pinned_request(url)

        with patch.object(
            skill_tools, "_pinned_request", side_effect=first_hop_then_real
        ):
            with self.assertRaises(skill_tools._UnsafeTargetError):
                skill_tools._fetch_page_sync("http://example.com/start")

    def test_redirect_to_localhost_is_blocked(self) -> None:
        import skill_tools

        real_pinned_request = skill_tools._pinned_request
        calls = {"count": 0}

        def first_hop_then_real(url: str):
            calls["count"] += 1
            if calls["count"] == 1:
                return 302, _FakeHeaders("http://localhost:8080/admin"), b""
            return real_pinned_request(url)

        with patch.object(
            skill_tools, "_pinned_request", side_effect=first_hop_then_real
        ):
            with self.assertRaises(skill_tools._UnsafeTargetError):
                skill_tools._fetch_page_sync("https://example.com/start")

    def test_redirect_loop_is_capped(self) -> None:
        import skill_tools

        def always_redirect(url: str):
            return 302, _FakeHeaders("https://example.com/next"), b""

        with patch.object(
            skill_tools, "_pinned_request", side_effect=always_redirect
        ) as request_mock:
            with self.assertRaises(skill_tools._UnsafeTargetError):
                skill_tools._fetch_page_sync("https://example.com/start")
        self.assertEqual(request_mock.call_count, skill_tools._MAX_REDIRECTS + 1)

    def test_page_body_is_capped(self) -> None:
        import skill_tools

        big = b"x" * (skill_tools.MAX_PAGE_BYTES + 100)

        class _FakeResponse:
            status = 200
            headers = _FakeHeaders()

            def read(self, n: int = -1) -> bytes:
                return big if n < 0 else big[:n]

        class _FakeConnection:
            def request(self, *args, **kwargs) -> None:
                return None

            def getresponse(self) -> _FakeResponse:
                return _FakeResponse()

            def close(self) -> None:
                return None

        with patch.object(
            skill_tools,
            "_resolve_public_addresses",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))],
        ), patch.object(
            skill_tools, "_pinned_connection", return_value=_FakeConnection()
        ):
            body, _content_type = skill_tools._fetch_page_sync("https://example.com/huge")
        self.assertEqual(len(body.encode("utf-8")), skill_tools.MAX_PAGE_BYTES)


class _FakeHeaders:
    def __init__(self, location: str | None = None) -> None:
        self._location = location

    def get(self, name: str, default=None):
        if name.lower() == "location":
            return self._location if self._location is not None else default
        return default

    def get_content_type(self) -> str:
        return "text/plain"

    def get_content_charset(self) -> str:
        return "utf-8"


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

    async def test_store_conflict_returns_duplicate_and_cleans_up(self) -> None:
        """Issue #14: losing the dedup race at the unique constraint returns
        the duplicate shape and removes the just-created original source."""
        import my_files

        data = b"hello duplicate world"
        existing = {
            "document_id": "11111111-1111-1111-1111-111111111111",
            "filename": "notes.md",
            "status": "fts_ready",
            "chunk_count": 3,
        }
        chunk = SimpleNamespace(
            chunk_index=0, page_number=None, content="hello duplicate world", metadata={}
        )
        with patch.object(
            my_files, "find_document_by_hash", new=AsyncMock(side_effect=[None, existing])
        ), patch.object(
            my_files, "create_original_source", new=AsyncMock()
        ), patch.object(
            my_files, "store_document", new=AsyncMock(return_value=False)
        ), patch.object(
            my_files, "delete_original_source", new=AsyncMock()
        ) as delete_mock, patch.object(
            my_files, "extract_document", return_value=["hello duplicate world"]
        ), patch.object(
            my_files, "build_chunks", return_value=[chunk]
        ):
            result = await my_files.ingest_document(
                document_id="22222222-2222-2222-2222-222222222222",
                filename="copy-of-notes.md",
                mime_type="text/markdown",
                data=data,
            )
        self.assertTrue(result["duplicate"])
        self.assertEqual(result["status"], "duplicate")
        self.assertEqual(result["document_id"], existing["document_id"])
        delete_mock.assert_awaited_once()


class StoreDocumentUniqueTests(unittest.IsolatedAsyncioTestCase):
    """Issue #14: store_document inserts with ON CONFLICT and reports races."""

    async def _run_store(self, *, has_index: bool, insert_returns):
        import persistence_docs

        class _Transaction:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        class _Connection:
            def __init__(self) -> None:
                self.statements: list[str] = []

            def transaction(self) -> _Transaction:
                return _Transaction()

            async def fetchval(self, sql, *args):
                self.statements.append(sql)
                if "to_regclass" in sql:
                    return has_index
                return insert_returns

            async def executemany(self, sql, rows) -> None:
                return None

        class _Acquire:
            def __init__(self, connection) -> None:
                self._connection = connection

            async def __aenter__(self):
                return self._connection

            async def __aexit__(self, *args):
                return False

        class _Pool:
            def __init__(self, connection) -> None:
                self._connection = connection

            def acquire(self) -> _Acquire:
                return _Acquire(self._connection)

        connection = _Connection()
        with patch.object(
            persistence_docs, "_get_pool", new=AsyncMock(return_value=_Pool(connection))
        ):
            result = await persistence_docs.store_document(
                document_id="22222222-2222-2222-2222-222222222222",
                filename="a.md",
                mime_type="text/markdown",
                file_hash="abc123",
                status="fts_ready",
                chunks=[],
            )
        return result, connection

    async def test_insert_uses_on_conflict_when_index_exists(self) -> None:
        result, connection = await self._run_store(has_index=True, insert_returns="id")
        self.assertTrue(result)
        insert_sql = next(s for s in connection.statements if "INSERT INTO documents" in s)
        self.assertIn("ON CONFLICT (file_hash) DO NOTHING", insert_sql)

    async def test_conflict_returns_false(self) -> None:
        result, _connection = await self._run_store(has_index=True, insert_returns=None)
        self.assertFalse(result)

    async def test_plain_insert_when_index_missing(self) -> None:
        result, connection = await self._run_store(has_index=False, insert_returns="id")
        self.assertTrue(result)
        insert_sql = next(s for s in connection.statements if "INSERT INTO documents" in s)
        self.assertNotIn("ON CONFLICT", insert_sql)

    async def test_unique_index_migration_is_tolerant(self) -> None:
        """A pre-existing duplicate file_hash must not brick schema init."""
        import persistence_core

        class _Pool:
            def __init__(self) -> None:
                self.statements: list[str] = []

            async def execute(self, sql) -> None:
                self.statements.append(sql)
                if "documents_file_hash_key" in sql:
                    raise Exception("could not create unique index: duplicates exist")

        pool = _Pool()
        persistence_core._schema_ready = False
        try:
            await persistence_core._ensure_policy_schema(pool)
        finally:
            persistence_core._schema_ready = False
        self.assertTrue(
            any("documents_file_hash_key" in s for s in pool.statements)
        )


class ProviderModelCacheTests(unittest.TestCase):
    """Issue #15: provider model instances are shared per credential set."""

    def setUp(self) -> None:
        import agent_providers

        agent_providers._clear_model_cache()

    def tearDown(self) -> None:
        import agent_providers

        agent_providers._clear_model_cache()

    def test_same_credentials_share_one_instance(self) -> None:
        import agent_providers

        first = agent_providers._gemini_model("key-a")
        second = agent_providers._gemini_model("key-a")
        self.assertIs(first, second)

    def test_new_credentials_evict_stale_instance(self) -> None:
        import agent_providers

        first = agent_providers._gemini_model("key-a")
        second = agent_providers._gemini_model("key-b")
        self.assertIsNot(first, second)
        self.assertIs(agent_providers._gemini_model("key-b"), second)
        self.assertEqual(
            len([k for k in agent_providers._model_cache if k[0] == "gemini"]), 1
        )

    def test_openai_instance_shared_and_reevicted_on_change(self) -> None:
        import agent_providers

        first = agent_providers._openai_model("key", "https://api-a.example/v1")
        self.assertIs(
            agent_providers._openai_model("key", "https://api-a.example/v1"), first
        )
        second = agent_providers._openai_model("key", "https://api-b.example/v1")
        self.assertIsNot(first, second)
        self.assertEqual(
            len([k for k in agent_providers._model_cache if k[0] == "openai"]), 1
        )


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
