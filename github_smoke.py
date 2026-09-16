from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from github_search import GitHubProvider


async def run_live_smoke() -> dict[str, object]:
    provider = GitHubProvider()
    owner = "modelcontextprotocol"
    repo = "python-sdk"
    started_at = time.perf_counter()
    cases: list[dict[str, object]] = []

    async def case(name: str, callback: Any) -> None:
        case_started = time.perf_counter()
        try:
            result = await callback()
            cases.append(
                {
                    "id": name,
                    "status": "PASS" if result else "DEGRADED",
                    "latency_ms": round((time.perf_counter() - case_started) * 1000),
                }
            )
        except Exception as error:
            cases.append(
                {
                    "id": name,
                    "status": "DEGRADED",
                    "safe_error": type(error).__name__,
                    "latency_ms": round((time.perf_counter() - case_started) * 1000),
                }
            )

    await case(
        "repository_metadata",
        lambda: provider.lookup_repository(owner, repo),
    )
    await case(
        "latest_release",
        lambda: provider.latest_release(owner, repo),
    )
    await case(
        "issue_list",
        lambda: provider.list_issues(owner, repo, 1),
    )
    statuses = [str(item["status"]) for item in cases]
    return {
        "status": "PASS" if all(status == "PASS" for status in statuses) else "DEGRADED",
        "auth_mode": provider.health()["auth_mode"],
        "case_count": len(cases),
        "passed": statuses.count("PASS"),
        "degraded": statuses.count("DEGRADED"),
        "latency_ms": round((time.perf_counter() - started_at) * 1000),
        "provider_request_count": provider.health()["request_count"],
        "cases": cases,
        "health": provider.health(),
    }


async def main() -> None:
    print(json.dumps(await run_live_smoke(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())