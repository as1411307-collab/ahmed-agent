from __future__ import annotations

import hashlib
import json
import socket
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


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
        import time

        token = config.AHMED_OWNER_TOKEN
        if not token:
            self.skipTest("AHMED_OWNER_TOKEN not configured")
        bad = self._request("wrong-token")
        await auth.authorize_owner(bad, endpoint="/mcp")
        # Expire the throttle window deterministically (avoids sleeping in tests):
        # the recorded failure is older than the reset window, so the next request
        # is processed instead of throttled.
        failures, _ = auth._failed_attempts["10.0.0.9"]
        auth._failed_attempts["10.0.0.9"] = (
            failures,
            time.monotonic() - auth._AUTH_FAILURE_RESET_SECONDS - 1,
        )
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

    async def test_duplicate_result_is_json_serializable_with_uuid(self) -> None:
        """Regression: asyncpg returns UUID objects; the duplicate response
        must stay JSON-serializable (found via live E2E 500)."""
        import uuid
        import my_files

        data = b"hello duplicate world"
        existing = {
            "document_id": uuid.UUID("11111111-1111-1111-1111-111111111111"),
            "filename": "notes.md",
            "status": "fts_ready",
            "chunk_count": 3,
        }
        with patch.object(
            my_files, "find_document_by_hash", new=AsyncMock(return_value=existing)
        ):
            result = await my_files.ingest_document(
                document_id="22222222-2222-2222-2222-222222222222",
                filename="copy-of-notes.md",
                mime_type="text/markdown",
                data=data,
            )
        json.dumps(result)
        self.assertEqual(result["document_id"], str(existing["document_id"]))


class VectorUnavailableFallbackTests(unittest.IsolatedAsyncioTestCase):
    """Codex finding on PR #19: when pgvector is unavailable, document_chunks
    has no embedding column, so store_document_embeddings /
    search_vector_document_chunks raise PersistenceError on that missing
    column -- and that error wasn't caught, defeating the FTS fallback that
    pgvector being optional is supposed to provide. ingest_document and
    search_my_files must check vector_storage_available() first and skip
    the vector path entirely rather than let that PersistenceError escape.
    """

    async def test_ingest_document_skips_embedding_when_vector_unavailable(self) -> None:
        import my_files

        chunk = SimpleNamespace(
            chunk_index=0, page_number=None, content="hello world", metadata={}
        )
        with patch.object(
            my_files, "find_document_by_hash", new=AsyncMock(return_value=None)
        ), patch.object(
            my_files, "create_original_source", new=AsyncMock()
        ), patch.object(
            my_files, "extract_document", return_value=["hello world"]
        ), patch.object(
            my_files, "build_chunks", return_value=[chunk]
        ), patch.object(
            my_files, "store_document", new=AsyncMock(return_value=True)
        ), patch.object(
            my_files, "update_original_source_extraction_status", new=AsyncMock()
        ), patch.object(
            my_files, "vector_storage_available", new=AsyncMock(return_value=False)
        ), patch.object(
            my_files, "get_embedding_provider"
        ) as provider_mock, patch.object(
            my_files, "store_document_embeddings", new=AsyncMock()
        ) as store_embeddings_mock:
            result = await my_files.ingest_document(
                document_id="55555555-5555-5555-5555-555555555555",
                filename="notes.md",
                mime_type="text/markdown",
                data=b"hello world",
            )
        provider_mock.assert_not_called()
        store_embeddings_mock.assert_not_awaited()
        self.assertEqual(result["status"], "VECTOR_UNAVAILABLE")

    async def test_search_my_files_falls_back_to_fts_when_vector_unavailable(self) -> None:
        import my_files

        fts_row = {
            "document_id": "11111111-1111-1111-1111-111111111111",
            "filename": "notes.md",
            "mime_type": "text/markdown",
            "source_type": "upload",
            "file_hash": "abc123",
            "source_id": None,
            "source_sha256": None,
            "original_available": False,
            "chunk_index": 0,
            "page_number": None,
            "content": "hello world",
            "rank": 1.0,
        }
        with patch.object(
            my_files,
            "search_fts_document_chunks",
            new=AsyncMock(return_value=[fts_row]),
        ), patch.object(
            my_files, "vector_storage_available", new=AsyncMock(return_value=False)
        ), patch.object(
            my_files, "get_embedding_provider"
        ) as provider_mock, patch.object(
            my_files, "search_vector_document_chunks", new=AsyncMock()
        ) as search_vector_mock:
            result = await my_files.search_my_files("hello")
        provider_mock.assert_not_called()
        search_vector_mock.assert_not_awaited()
        self.assertEqual(result["retrieval_mode"], "FTS_FALLBACK")
        self.assertEqual(len(result["results"]), 1)


class UploadStatusCodeTests(unittest.TestCase):
    """Codex finding on PR #19: /files/upload's status-code branching only
    accepted {"ready", "embedding_failed"}, so the new VECTOR_UNAVAILABLE
    status (a successful, FTS-searchable upload with no embedding attempted
    because pgvector isn't installed) was rejected as a 422 client error.
    """

    def test_ready_is_201(self) -> None:
        import server

        self.assertEqual(server._upload_response_status_code([{"status": "ready"}]), 201)

    def test_embedding_failed_is_202(self) -> None:
        import server

        self.assertEqual(
            server._upload_response_status_code([{"status": "embedding_failed"}]), 202
        )

    def test_vector_unavailable_is_202_not_422(self) -> None:
        import server

        self.assertEqual(
            server._upload_response_status_code([{"status": "VECTOR_UNAVAILABLE"}]), 202
        )

    def test_unrecognized_status_is_422(self) -> None:
        import server

        self.assertEqual(server._upload_response_status_code([{"status": "empty"}]), 422)


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

    async def test_fts_query_is_or_matched_and_escaped(self) -> None:
        """Long natural-language queries must still retrieve partial matches.

        Regression: plainto_tsquery ANDs every term, so one absent word
        (common in Arabic questions) returned zero rows even when documents
        contained the answer.
        """
        import persistence_docs

        tsquery = persistence_docs._fts_match_query(
            "راجع الملف الفعلي لـ Ahmed Agent ولا تعتمد على الذاكرة"
        )
        self.assertIn("|", tsquery)
        self.assertIn("'راجع'", tsquery)
        # single quotes inside a token are escaped (lexeme quoting)
        escaped = persistence_docs._fts_match_query("it's a test")
        self.assertIn("'it''s'", escaped)
        # empty / whitespace-only query yields an empty tsquery (matches nothing)
        self.assertEqual(persistence_docs._fts_match_query("   "), "")

    async def test_find_document_by_hash_returns_str_uuid(self) -> None:
        import uuid
        import persistence_docs

        row = {
            "document_id": uuid.UUID("11111111-1111-1111-1111-111111111111"),
            "filename": "notes.md",
            "status": "fts_ready",
            "chunk_count": 3,
        }

        class _Pool:
            async def fetchval(self, *args):
                raise AssertionError("not used")

            async def fetchrow(self, *args):
                return row

        with patch.object(
            persistence_docs, "_get_pool", new=AsyncMock(return_value=_Pool())
        ):
            result = await persistence_docs.find_document_by_hash("abc123")
        self.assertEqual(result["document_id"], "11111111-1111-1111-1111-111111111111")
        json.dumps(result)

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

    async def test_base_schema_tolerates_missing_pgvector(self) -> None:
        """A Postgres without pgvector must still get the core tables and the
        FTS index (Codex P1 finding: startup used to hard-fail on any
        persistence operation when the 'vector' extension wasn't installed).
        """
        import persistence_core

        class _Pool:
            def __init__(self) -> None:
                self.statements: list[str] = []

            async def execute(self, sql, **kwargs) -> None:
                self.statements.append(sql)
                if "CREATE EXTENSION IF NOT EXISTS vector" in sql:
                    raise Exception('extension "vector" is not available')
                if "ADD COLUMN IF NOT EXISTS embedding vector" in sql:
                    raise AssertionError(
                        "embedding column must not be attempted when pgvector is unavailable"
                    )

            async def fetchval(self, sql, *args):
                self.statements.append(sql)
                return False

        pool = _Pool()
        persistence_core._schema_ready = False
        try:
            await persistence_core._ensure_policy_schema(pool)
        finally:
            persistence_core._schema_ready = False
        assert persistence_core._fts_index_task is not None
        await persistence_core._fts_index_task
        self.assertTrue(
            any("CREATE TABLE IF NOT EXISTS agent_sessions" in s for s in pool.statements)
        )
        self.assertTrue(
            any("document_chunks_content_fts_idx" in s for s in pool.statements)
        )

    async def test_base_schema_recovers_invalid_fts_index(self) -> None:
        """Codex P2 finding: CREATE INDEX CONCURRENTLY IF NOT EXISTS matches
        an existing index by name regardless of validity, so a build
        cancelled by the command timeout leaves a permanently-ignored
        invalid index unless it's explicitly detected and dropped first.
        """
        import persistence_core

        class _Pool:
            def __init__(self) -> None:
                self.statements: list[str] = []

            async def execute(self, sql, **kwargs) -> None:
                self.statements.append(sql)

            async def fetchval(self, sql, *args):
                self.statements.append(sql)
                if "indisvalid" in sql:
                    return True
                return None

        pool = _Pool()
        persistence_core._schema_ready = False
        try:
            await persistence_core._ensure_policy_schema(pool)
        finally:
            persistence_core._schema_ready = False
        assert persistence_core._fts_index_task is not None
        await persistence_core._fts_index_task
        drop_index = next(
            i for i, s in enumerate(pool.statements) if "DROP INDEX CONCURRENTLY" in s
        )
        create_index = next(
            i
            for i, s in enumerate(pool.statements)
            if "CREATE INDEX CONCURRENTLY IF NOT EXISTS document_chunks_content_fts_idx" in s
        )
        self.assertLess(drop_index, create_index)


class AppendMessagesSequenceLockTests(unittest.IsolatedAsyncioTestCase):
    """Codex P2 finding: sequence allocation must be serialized per session,
    or two concurrent runs can compute the same MAX(sequence_number) and one
    loses its response to the new UNIQUE (session_id, sequence_number)
    constraint.
    """

    async def test_locks_session_row_before_computing_next_sequence(self) -> None:
        import persistence_runs

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

            async def execute(self, sql, *args) -> None:
                self.statements.append(sql)

            async def fetchval(self, sql, *args):
                self.statements.append(sql)
                return 1

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
            persistence_runs, "_get_pool", new=AsyncMock(return_value=_Pool(connection))
        ):
            await persistence_runs.append_new_messages(
                session_id="11111111-1111-1111-1111-111111111111",
                run_id="22222222-2222-2222-2222-222222222222",
                new_messages_json=b'[{"role": "user", "content": "hi"}]',
            )

        lock_index = next(
            i for i, s in enumerate(connection.statements) if "FOR UPDATE" in s
        )
        max_index = next(
            i for i, s in enumerate(connection.statements) if "MAX(sequence_number)" in s
        )
        self.assertLess(lock_index, max_index)
        self.assertIn("agent_sessions", connection.statements[lock_index])


class AgentDepsAnnotationsTests(unittest.TestCase):
    """Codex finding on PR #19: AgentDeps.tool_event_recorder's annotation
    references Callable/Awaitable, so anything that resolves this exported
    dataclass's type hints (typing.get_type_hints, framework/schema
    introspection) needs those names importable from agent_consts, even
    though `from __future__ import annotations` means nothing evaluates
    them at class-definition time.
    """

    def test_get_type_hints_resolves_without_nameerror(self) -> None:
        import typing

        import agent_consts

        hints = typing.get_type_hints(agent_consts.AgentDeps)
        self.assertIn("tool_event_recorder", hints)


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
