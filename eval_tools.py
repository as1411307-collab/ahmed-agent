from __future__ import annotations

from typing import Any

from eval_schema import CASE_CAPABILITY_OVERRIDES, TOOL_NAME_MAPPING


def _tool_names(trace: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for call in trace.get("tool_calls", []):
        if isinstance(call, str):
            names.add(call)
        elif isinstance(call, dict) and isinstance(call.get("name"), str):
            names.add(call["name"])
    return names


def resolve_tool_expectation(semantic_name: str) -> dict[str, Any]:
    return TOOL_NAME_MAPPING.get(
        semantic_name,
        {
            "actual": semantic_name,
            "status": "unmapped",
            "note": "No semantic-to-runtime mapping has been declared.",
        },
    )


def resolve_case_tool_expectation(
    case_id: str,
    semantic_name: str,
) -> dict[str, Any]:
    return CASE_CAPABILITY_OVERRIDES.get(case_id, {}).get(
        semantic_name,
        resolve_tool_expectation(semantic_name),
    )


def case_execution_capability(case: dict[str, Any]) -> dict[str, Any]:
    unavailable_required = [
        tool
        for tool in case["required_tools"]
        if resolve_case_tool_expectation(case["id"], tool)["status"]
        == "unavailable"
    ]
    unmapped_required = [
        tool
        for tool in case["required_tools"]
        if resolve_case_tool_expectation(case["id"], tool)["status"] == "unmapped"
    ]
    boundary_required = [
        tool
        for tool in case["required_tools"]
        if resolve_case_tool_expectation(case["id"], tool)["status"]
        == "boundary_only"
    ]
    return {
        "executable_by_current_agent_tools": not (
            unavailable_required or unmapped_required or boundary_required
        ),
        "unavailable_required_tools": unavailable_required,
        "unmapped_required_tools": unmapped_required,
        "boundary_required_tools": boundary_required,
        "resolved_required_tools": {
            tool: resolve_case_tool_expectation(case["id"], tool)
            for tool in case["required_tools"]
        },
    }


