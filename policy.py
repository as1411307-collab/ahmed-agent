from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class RiskLevel(StrEnum):
    READ_SAFE = "READ_SAFE"
    CREATE_REVERSIBLE = "CREATE_REVERSIBLE"
    MODIFY_EXISTING = "MODIFY_EXISTING"
    SENSITIVE_SIDE_EFFECT = "SENSITIVE_SIDE_EFFECT"
    HIGH_RISK = "HIGH_RISK"


@dataclass(frozen=True)
class ToolPolicy:
    tool_name: str
    risk_level: RiskLevel
    requires_approval: bool
    reversible: bool
    external_side_effect: bool
    data_scope: str


TOOL_POLICIES: dict[str, ToolPolicy] = {
    "ping": ToolPolicy(
        tool_name="ping",
        risk_level=RiskLevel.READ_SAFE,
        requires_approval=False,
        reversible=True,
        external_side_effect=False,
        data_scope="INTERNAL",
    ),
    "web_search": ToolPolicy(
        tool_name="web_search",
        risk_level=RiskLevel.READ_SAFE,
        requires_approval=False,
        reversible=True,
        external_side_effect=False,
        data_scope="WEB",
    ),
    "search_my_files": ToolPolicy(
        tool_name="search_my_files",
        risk_level=RiskLevel.READ_SAFE,
        requires_approval=False,
        reversible=True,
        external_side_effect=False,
        data_scope="MY_FILES",
    ),
    "inspect_runtime_evidence": ToolPolicy(
        tool_name="inspect_runtime_evidence",
        risk_level=RiskLevel.READ_SAFE,
        requires_approval=False,
        reversible=True,
        external_side_effect=False,
        data_scope="PROJECT_RUNTIME_EVIDENCE",
    ),
    "inspect_architecture_evidence": ToolPolicy(
        tool_name="inspect_architecture_evidence",
        risk_level=RiskLevel.READ_SAFE,
        requires_approval=False,
        reversible=True,
        external_side_effect=False,
        data_scope="PROJECT_ARCHITECTURE_EVIDENCE",
    ),
    "inspect_source_status": ToolPolicy(
        tool_name="inspect_source_status",
        risk_level=RiskLevel.READ_SAFE,
        requires_approval=False,
        reversible=True,
        external_side_effect=False,
        data_scope="PROJECT_SOURCE_STATUS_EVIDENCE",
    ),
    "inspect_source_of_truth": ToolPolicy(
        tool_name="inspect_source_of_truth",
        risk_level=RiskLevel.READ_SAFE,
        requires_approval=False,
        reversible=True,
        external_side_effect=False,
        data_scope="AUTHORIZED_MY_FILES_SOURCE",
    ),
    "test_sensitive_action": ToolPolicy(
        tool_name="test_sensitive_action",
        risk_level=RiskLevel.SENSITIVE_SIDE_EFFECT,
        requires_approval=True,
        reversible=False,
        external_side_effect=True,
        data_scope="INTERNAL_TEST",
    ),
}


def get_tool_policy(tool_name: str) -> ToolPolicy:
    try:
        return TOOL_POLICIES[tool_name]
    except KeyError as error:
        raise ValueError(f"Unknown tool policy: {tool_name}") from error


def tool_metadata(tool_name: str) -> dict[str, Any]:
    policy = get_tool_policy(tool_name)
    return {
        "tool_name": policy.tool_name,
        "risk_level": policy.risk_level.value,
        "requires_approval": policy.requires_approval,
        "reversible": policy.reversible,
        "external_side_effect": policy.external_side_effect,
        "data_scope": policy.data_scope,
    }


def data_only_boundary(source: str) -> dict[str, str | bool]:
    return {
        "source": source,
        "classification": "UNTRUSTED_DATA",
        "instructions_allowed": False,
    }