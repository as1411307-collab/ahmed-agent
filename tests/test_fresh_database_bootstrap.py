from __future__ import annotations

import os
import unittest
from urllib.parse import urlparse

import asyncpg

CORE_TABLES = (
    "agent_sessions",
    "agent_runs",
    "agent_messages",
    "tool_events",
    "documents",
    "document_chunks",
)


def _test_database_url() -> str | None:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        return None
    db_name = urlparse(url).path.lstrip("/")
    if "test" not in db_name:
        return None
    return url


@unittest.skipUnless(
    _test_database_url() is not None,
    "set TEST_DATABASE_URL (database name must contain 'test') for this "
    "destructive, real-Postgres bootstrap test",
)
class FreshDatabaseBootstrapTests(unittest.IsolatedAsyncioTestCase):
    """PR #19 added the base schema so a genuinely empty database can bootstrap
    itself on first use, but that was only ever verified by hand. This runs
    the real bootstrap path against a truly empty `public` schema and asserts
    every core table it's supposed to create actually exists.
    """

    async def asyncTearDown(self) -> None:
        import persistence_core

        if persistence_core._fts_index_task is not None:
            await persistence_core._fts_index_task
        await persistence_core.close_pool()

    async def test_get_pool_bootstraps_all_core_tables_from_empty_schema(self) -> None:
        database_url = _test_database_url()
        connection = await asyncpg.connect(database_url)
        try:
            await connection.execute("DROP SCHEMA public CASCADE")
            await connection.execute("CREATE SCHEMA public")
        finally:
            await connection.close()

        os.environ["DATABASE_URL"] = database_url
        import persistence_core

        if persistence_core._pool is not None:
            await persistence_core.close_pool()
        persistence_core._schema_ready = False
        persistence_core._vector_storage_available = False
        persistence_core._fts_index_task = None

        pool = await persistence_core._get_pool()

        rows = await pool.fetch(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            """
        )
        existing = {row["table_name"] for row in rows}
        missing = [table for table in CORE_TABLES if table not in existing]
        self.assertEqual(missing, [], f"tables missing after bootstrap: {missing}")


if __name__ == "__main__":
    unittest.main()
