from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from urllib.request import urlopen
from uuid import uuid4

import asyncpg

from persistence import (
    append_new_messages,
    claim_orphaned_run,
    create_pending_action,
    create_run,
    ensure_session,
    finish_run,
    load_run_recovery,
    mark_orphaned_runs,
    record_run_checkpoint,
    run_message_count,
)
from run_state import RunStage, RecoveryAction, recovery_action


_WORKER_CODE = r"""
import asyncio
import json
import sys
from persistence import (
    append_new_messages,
    create_run,
    ensure_session,
    record_run_checkpoint,
)


async def main():
    mode, run_id, session_id = sys.argv[1:4]
    await ensure_session(session_id, scope="fault_injection")
    await create_run(
        run_id=run_id,
        session_id=session_id,
        user_prompt="fault injection",
        provider_name="test",
        model_name="test",
        worker_id="fault-injection-worker",
        lease_seconds=30,
    )
    await record_run_checkpoint(
        session_id=session_id,
        run_id=run_id,
        stage="context_loaded",
        state={"test": True},
    )
    if mode in {"model_running", "response_ready", "response_persisted"}:
        await record_run_checkpoint(
            session_id=session_id,
            run_id=run_id,
            stage="model_running",
            state={"test": True},
        )
    if mode in {"response_ready", "response_persisted"}:
        await record_run_checkpoint(
            session_id=session_id,
            run_id=run_id,
            stage="response_ready",
            state={"response_bytes": 42, "test": True},
        )
        await append_new_messages(
            session_id=session_id,
            run_id=run_id,
            new_messages_json=json.dumps(
                [{"role": "assistant", "content": "persisted test response"}]
            ),
        )
    if mode == "response_persisted":
        await record_run_checkpoint(
            session_id=session_id,
            run_id=run_id,
            stage="response_persisted",
            state={"message_count": 1, "test": True},
        )
    await asyncio.sleep(120)


asyncio.run(main())
"""


@unittest.skipUnless(
    os.environ.get("AHMED_RUN_RECOVERY_E2E") == "1"
    and bool(os.environ.get("DATABASE_URL")),
    "set AHMED_RUN_RECOVERY_E2E=1 with DATABASE_URL for the destructive process test",
)
class RecoveryFaultInjectionE2ETests(unittest.TestCase):
    def _run_worker_until_killed(self, mode: str, run_id: str, session_id: str) -> None:
        root = Path(__file__).resolve().parents[1]
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _WORKER_CODE,
                mode,
                run_id,
                session_id,
            ],
            cwd=root,
            env=os.environ.copy(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        time.sleep(1)
        process.kill()
        _, stderr = process.communicate(timeout=10)
        self.assertNotEqual(process.returncode, 0)
        self.assertIsNotNone(stderr)

    async def _expire_lease(self, run_ids: list[str]) -> None:
        connection = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            await connection.execute(
                """
                UPDATE agent_runs
                SET lease_expires_at = NOW() - INTERVAL '1 second'
                WHERE run_id = ANY($1::uuid[])
                """,
                run_ids,
            )
        finally:
            await connection.close()

    def _restart_server_for_startup_scan(self) -> None:
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PORT"] = "18927"
        process = subprocess.Popen(
            [sys.executable, "server.py"],
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            ready = False
            for _ in range(80):
                if process.poll() is not None:
                    break
                try:
                    with urlopen("http://127.0.0.1:18927/health", timeout=0.5) as response:
                        ready = response.status == 200
                        if ready:
                            break
                except OSError:
                    time.sleep(0.25)
            self.assertTrue(ready, "server did not become ready during startup recovery scan")
            self.assertIsNone(process.poll(), "server failed during startup recovery scan")
        finally:
            process.terminate()
            output, _ = process.communicate(timeout=10)
        self.assertIn("Marked orphaned runs count=", output)

    async def _recover_persisted_tail(self, run_id: str, session_id: str) -> None:
        claimed = await claim_orphaned_run(
            run_id=run_id,
            worker_id="fault-injection-recovery",
            lease_seconds=30,
        )
        self.assertIsNotNone(claimed)
        assert claimed is not None
        recovery = await load_run_recovery(run_id)
        assert recovery is not None
        stage = RunStage(str(recovery["stage"]))
        count = await run_message_count(run_id)
        if stage is RunStage.RESPONSE_READY:
            self.assertEqual(count, 1)
            await record_run_checkpoint(
                session_id=session_id,
                run_id=run_id,
                stage=RunStage.RESPONSE_PERSISTED.value,
                state={"message_count": count, "recovered": True},
            )
        await finish_run(run_id=run_id, status="succeeded")
        await record_run_checkpoint(
            session_id=session_id,
            run_id=run_id,
            stage=RunStage.COMPLETED.value,
            state={"message_count": count, "recovered": True},
        )

    def test_killed_workers_are_recovered_without_replaying_work(self) -> None:
        modes = ["model_running", "response_ready", "response_persisted"]
        session_id = str(uuid4())
        runs = {mode: str(uuid4()) for mode in modes}
        for mode, run_id in runs.items():
            self._run_worker_until_killed(mode, run_id, session_id)

        asyncio.run(self._expire_lease(list(runs.values())))
        self._restart_server_for_startup_scan()

        async def verify_and_recover() -> None:
            for mode, run_id in runs.items():
                recovery = await load_run_recovery(run_id)
                self.assertIsNotNone(recovery)
                assert recovery is not None
                self.assertEqual(recovery["status"], "running")
                self.assertEqual(recovery["recovery_status"], "orphaned")
                action = recovery_action(RunStage(str(recovery["stage"])))
                if mode == "model_running":
                    self.assertEqual(action, RecoveryAction.MANUAL_REVIEW)
                else:
                    self.assertEqual(action, RecoveryAction.COMPLETE_PERSISTED_TAIL)
                    await self._recover_persisted_tail(run_id, session_id)

            for mode, run_id in runs.items():
                recovery = await load_run_recovery(run_id)
                assert recovery is not None
                if mode == "model_running":
                    self.assertEqual(recovery["recovery_status"], "orphaned")
                else:
                    self.assertEqual(recovery["status"], "succeeded")
                    self.assertEqual(recovery["stage"], RunStage.COMPLETED.value)
                    self.assertEqual(recovery["recovery_status"], "resolved")

            first = await create_pending_action(
                action_id=str(uuid4()),
                session_id=session_id,
                run_id=runs["model_running"],
                user_id="fault-test-owner",
                tool_name="test_sensitive_action",
                risk_level="high",
                arguments={"reason": "idempotency test"},
                idempotency_key="fault-injection-idempotency-key",
            )
            second = await create_pending_action(
                action_id=str(uuid4()),
                session_id=session_id,
                run_id=runs["model_running"],
                user_id="fault-test-owner",
                tool_name="test_sensitive_action",
                risk_level="high",
                arguments={"reason": "idempotency test"},
                idempotency_key="fault-injection-idempotency-key",
            )
            self.assertEqual(first["action_id"], second["action_id"])

        asyncio.run(verify_and_recover())


if __name__ == "__main__":
    unittest.main()