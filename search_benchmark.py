from __future__ import annotations

import asyncio
import json
from pathlib import Path

from search_fabric import run_benchmark
from skill_tools import _get_search_fabric


async def main() -> None:
    fixture_path = Path(__file__).parent / "tests" / "fixtures" / "search_benchmark.json"
    fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))
    result = await run_benchmark(_get_search_fabric(), fixtures)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())