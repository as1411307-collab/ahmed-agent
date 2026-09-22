from __future__ import annotations

import asyncio
import os
import unittest
from urllib.parse import urlparse


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
    "destructive, real-Postgres concurrency test",
)
class AuditChainConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    """Regression test for a real, reproduced race in the audit hash chain.

    Reproduced against a real PostgreSQL 16 instance: 10 bursts of 5
    concurrent record_auth_event calls forked the chain (two rows sharing
    the same previous_hash), and verify_audit_chain() returned
    AUDIT_CHAIN_MISMATCH. Root cause: _append_audit_event's
    "SELECT ... ORDER BY event_id DESC LIMIT 1 FOR UPDATE" doesn't
    serialize concurrent appends, because the row each transaction is
    about to insert doesn't exist yet -- there's nothing for FOR UPDATE to
    block on, so two transactions can both read the same head before
    either commits.
    """

    async def asyncSetUp(self) -> None:
        os.environ["DATABASE_URL"] = _test_database_url()
        import persistence_core

        if persistence_core._pool is not None:
            await persistence_core.close_pool()
        persistence_core._schema_ready = False
        persistence_core._vector_storage_available = False
        persistence_core._fts_index_task = None

    async def asyncTearDown(self) -> None:
        import persistence_core

        if persistence_core._fts_index_task is not None:
            await persistence_core._fts_index_task
        await persistence_core.close_pool()

    async def test_concurrent_bursts_do_not_fork_the_chain(self) -> None:
        import persistence_core
        from persistence import record_auth_event, verify_audit_chain

        pool = await persistence_core._get_pool()
        if persistence_core._fts_index_task is not None:
            await persistence_core._fts_index_task
        async with pool.acquire() as connection:
            await connection.execute("TRUNCATE audit_events")

        for i in range(3):
            await record_auth_event(
                principal=f"seq-{i}",
                authenticated=True,
                endpoint="/test",
                action_id=None,
                result="ok",
            )

        for burst in range(10):
            await asyncio.gather(
                *[
                    record_auth_event(
                        principal=f"burst-{burst}-{slot}",
                        authenticated=True,
                        endpoint="/test",
                        action_id=None,
                        result="ok",
                    )
                    for slot in range(5)
                ]
            )

        result = await verify_audit_chain()
        self.assertTrue(result["verified"], result)
        self.assertEqual(result["event_count"], 53)

        rows = await pool.fetch("SELECT previous_hash FROM audit_events")
        previous_hashes = [row["previous_hash"] for row in rows]
        self.assertEqual(
            len(previous_hashes),
            len(set(previous_hashes)),
            "two rows share the same previous_hash: the chain forked",
        )


if __name__ == "__main__":
    unittest.main()
