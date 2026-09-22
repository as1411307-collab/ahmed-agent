from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_core import MAX_MODEL_REQUESTS, MAX_TOOL_CALLS, provider_model_name
from eval_exec import run_real_cases
from eval_metrics import resume_rate_limited_baseline
from eval_schema import (
    CASE_CAPABILITY_OVERRIDES,
    DATASET_PATH,
    TOOL_NAME_MAPPING,
    import_real_cases,
    load_case_document,
    load_evaluation_cases,
)
from eval_tools import (
    _tool_names,
    case_execution_capability,
    resolve_case_tool_expectation,
)
from evidence_citations import verify_evidence_citation


def _observed_sources(trace: dict[str, Any]) -> list[str]:
    sources: list[str] = []
    for source in trace.get("sources", []):
        if isinstance(source, str):
            sources.append(source)
        elif isinstance(source, dict):
            for key in ("domain", "url", "citation", "filename"):
                value = source.get(key)
                if isinstance(value, str):
                    sources.append(value)
                    break
    return sources


def validate_evaluation_trace(trace: dict[str, Any]) -> None:
    required_keys = {
        "case_id",
        "run_id",
        "output",
        "tool_calls",
        "citations",
        "sources",
        "execution_status",
    }
    if not isinstance(trace, dict) or not required_keys <= trace.keys():
        raise ValueError("Evaluation trace is missing required fields.")
    if not isinstance(trace["output"], dict) or not isinstance(
        trace["output"].get("answer"), str
    ):
        raise ValueError("Evaluation trace output must contain an answer string.")
    if not isinstance(trace["tool_calls"], list):
        raise ValueError("Evaluation trace tool_calls must be a list.")
    if not isinstance(trace["citations"], list) or not isinstance(
        trace["sources"], list
    ):
        raise ValueError("Evaluation trace citations and sources must be lists.")
    if trace["execution_status"] not in {
        "EXECUTED",
        "EXECUTION_FAILED",
        "HITL_BLOCKED",
        "NOT_DETERMINED",
        "NOT_EXECUTABLE_CAPABILITY_GAP",
        "INVALID_CASE",
    }:
        raise ValueError("Evaluation trace has an invalid execution status.")


def deterministic_grade(
    case: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    if trace.get("execution_status") == "NOT_DETERMINED":
        return {
            "case_id": case["id"],
            "category": case["category"],
            "provenance": case["provenance"],
            "deterministic_status": "NOT_DETERMINED",
            "check_results": {"evidence_preconditions": "NOT_AVAILABLE"},
            "metrics": {
                "task_completion": "NOT_DETERMINED",
                "tool_selection": "NOT_DETERMINED",
                "forbidden_tool_violations": "NOT_DETERMINED",
                "retrieval_source_correctness": "NOT_DETERMINED",
                "citation_correctness": "NOT_DETERMINED",
                "grounding": "NOT_RUN",
                "instruction_adherence": "NOT_CONFIGURED",
                "instruction_adherence_deterministic": "NOT_CONFIGURED",
                "abstention": "NOT_DETERMINED",
                "hitl_boundary": "NOT_DETERMINED",
            },
            "semantic_status": "NOT_DETERMINED",
            "latency_ms": trace.get("latency_ms"),
            "tool_calls": len(trace.get("tool_calls", [])),
            "retries": trace.get("retries"),
            "tokens": trace.get("tokens"),
            "cost": trace.get("cost"),
        }
    checks = case["deterministic_checks"]
    tools = _tool_names(trace)
    sources = _observed_sources(trace)
    available_evidence_citations = trace.get("available_evidence_citations", [])
    available_evidence_items = trace.get("available_evidence_items", [])
    evidence_citation_check: bool | str = True
    if case["id"] == "AA-RC-014" and available_evidence_items:
        observed_citations = [
            citation
            for citation in trace.get("citations", [])
            if isinstance(citation, str)
        ]
        evidence_citation_check = bool(
            observed_citations
            and all(
                citation in available_evidence_citations
                and verify_evidence_citation(
                    citation,
                    available_evidence_items,
                    expected_claim="Ahmed Agent project files",
                )
                for citation in observed_citations
            )
        )
    expected_sources = [str(value) for value in case["expected_sources"]]
    required_tools = {
        str(
            resolve_case_tool_expectation(case["id"], semantic_name).get(
                "actual"
            )
            or semantic_name
        )
        for semantic_name in case["required_tools"]
    }
    forbidden_tools = set(case["forbidden_tools"])
    schema_fields = set(checks["schema_fields"])
    output = trace.get("output")
    output_fields = set(output) if isinstance(output, dict) else set()

    abstention_check: bool | str
    if checks["abstention_required"]:
        abstention_check = (
            True
            if trace.get("abstained") is True
            else "NOT_AVAILABLE"
            if trace.get("abstained") is None
            else False
        )
    else:
        abstention_check = True
    check_results: dict[str, bool | str] = {
        "required_tools": required_tools <= tools,
        "forbidden_tools": not (forbidden_tools & tools),
        "citations": (
            bool(trace.get("citations") or sources)
            if checks["citations_required"]
            else True
        ),
        "evidence_citation_contract": evidence_citation_check,
        "expected_sources": (
            not expected_sources
            or any(
                expected.casefold() in observed.casefold()
                for expected in expected_sources
                for observed in sources
            )
        ),
        "schema": schema_fields <= output_fields,
        "approval_boundary": (
            (
                bool(trace.get("approval_requested"))
                and not bool(trace.get("executed_without_approval"))
            )
            if checks["approval_required"]
            else not bool(trace.get("executed_without_approval"))
        ),
        "abstention": abstention_check,
    }
    passed = all(value is True for value in check_results.values())
    metric_status = lambda value: "PASS" if value is True else (
        "NOT_DETERMINED" if value == "NOT_AVAILABLE" else "FAIL"
    )
    return {
        "case_id": case["id"],
        "category": case["category"],
        "provenance": case["provenance"],
        "deterministic_status": "PASS" if passed else "FAIL",
        "check_results": check_results,
        "metrics": {
            "task_completion": "PASS" if passed else "FAIL",
            "tool_selection": (
                "PASS"
                if check_results["required_tools"] and check_results["forbidden_tools"]
                else "FAIL"
            ),
            "forbidden_tool_violations": (
                "PASS" if check_results["forbidden_tools"] else "FAIL"
            ),
            "retrieval_source_correctness": (
                metric_status(check_results["expected_sources"])
            ),
            "citation_correctness": (
                "PASS"
            if check_results["citations"]
            and check_results["expected_sources"]
            and check_results["evidence_citation_contract"] is True
                else "NOT_DETERMINED"
            ),
            "grounding": "NOT_RUN",
            "instruction_adherence": "NOT_CONFIGURED",
            "instruction_adherence_deterministic": "NOT_CONFIGURED",
            "abstention": metric_status(check_results["abstention"]),
            "hitl_boundary": metric_status(check_results["approval_boundary"]),
        },
        "semantic_status": "NOT_RUN",
        "latency_ms": trace.get("latency_ms"),
        "tool_calls": len(trace.get("tool_calls", [])),
        "retries": trace.get("retries"),
        "tokens": trace.get("tokens"),
        "cost": trace.get("cost"),
    }




def freeze_baseline_manifest(
    path: Path = DATASET_PATH,
) -> dict[str, Any]:
    dataset_bytes = path.read_bytes()
    dataset_version, cases = load_case_document(path, require_baseline_size=True)
    provenance_counts = Counter(case["provenance"] for case in cases)
    return {
        "baseline_version": "evaluation-baseline-v1",
        "run_id": str(uuid4()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_version,
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "dataset_case_count": len(cases),
        "real_case_count": provenance_counts.get("real_case", 0),
        "contract_seed_count": provenance_counts.get("contract_seed", 0),
        "category_counts": dict(
            sorted(Counter(case["category"] for case in cases).items())
        ),
        "tool_name_mapping": TOOL_NAME_MAPPING,
        "case_capability_overrides": CASE_CAPABILITY_OVERRIDES,
        "dataset_provenance": (
            "real_case"
            if provenance_counts.get("real_case", 0)
            else "contract_seed_until_real_cases_are_added"
        ),
        "providers": {
            "gemini": {"model": provider_model_name("gemini")},
            "openai": {"model": provider_model_name("openai")},
        },
        "limits": {
            "max_tool_calls": MAX_TOOL_CALLS,
            "max_model_requests": MAX_MODEL_REQUESTS,
        },
        "semantic_grader": "not_configured",
        "deterministic_grading": "READY_FOR_TRACES",
        "live_provider_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ahmed Agent evaluation baseline")
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--validate-real", type=Path)
    parser.add_argument("--import-real", type=Path)
    parser.add_argument("--probe-real", type=Path)
    parser.add_argument("--run-real", type=Path)
    parser.add_argument("--resume-baseline", type=Path)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("real-cases-validated.json"),
    )
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("AHMED_EVAL_BASE_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument(
        "--provider",
        default="gemini",
        choices=("gemini", "openai"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--inter-case-delay-seconds", type=float, default=0.0)
    parser.add_argument("--resume-max-retries", type=int, default=1)
    parser.add_argument("--resume-base-delay-seconds", type=float, default=20.0)
    parser.add_argument("--resume-max-delay-seconds", type=float, default=60.0)
    parser.add_argument("--resume-inter-case-delay-seconds", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    live_commands = [args.probe_real, args.run_real, args.resume_baseline]
    if sum(value is not None for value in live_commands) > 1:
        parser.error(
            "use only one of --probe-real, --run-real, or --resume-baseline"
        )
    if args.resume_max_retries < 0:
        parser.error("--resume-max-retries must be non-negative")
    if args.resume_base_delay_seconds < 0 or args.resume_max_delay_seconds < 0:
        parser.error("resume delays must be non-negative")
    if args.validate_real:
        version, cases = load_case_document(args.validate_real)
        print(json.dumps({
            "status": "VALID",
            "dataset_version": version,
            "real_case_count": sum(case["provenance"] == "real_case" for case in cases),
        }, ensure_ascii=False, indent=2))
        return
    if args.import_real:
        if args.output is None:
            parser.error("--import-real requires --output")
        print(json.dumps(
            import_real_cases(args.import_real, args.output),
            ensure_ascii=False,
            indent=2,
        ))
        return
    live_path = args.probe_real or args.run_real
    if live_path:
        owner_token = os.environ.get("AHMED_OWNER_TOKEN")
        if not owner_token:
            parser.error("AHMED_OWNER_TOKEN is required for live evaluation.")
        version, cases = load_case_document(live_path, require_baseline_size=True)
        selected_ids = set(args.case_ids or [])
        if args.probe_real and not selected_ids:
            selected_ids = {
                case["id"]
                for case in cases
                if case_execution_capability(case)[
                    "executable_by_current_agent_tools"
                ]
            }
            selected_ids = set(sorted(selected_ids)[:3])
        result = asyncio.run(
            run_real_cases(
                cases,
                base_url=args.base_url,
                owner_token=owner_token,
                provider=args.provider,
                timeout_seconds=args.timeout_seconds,
                case_ids=selected_ids or None,
                inter_case_delay_seconds=args.inter_case_delay_seconds,
            )
        )
        result.update(
            {
                "dataset_version": version,
                "dataset_sha256": hashlib.sha256(live_path.read_bytes()).hexdigest(),
                "real_case_count": len(cases),
                "semantic_grading": "NOT_RUN",
                "live_provider_run": True,
            }
        )
        output_path = args.output or Path(
            "baseline-probe-v1.json" if args.probe_real else "baseline-real-v1.json"
        )
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": "WRITTEN",
                    "output": str(output_path),
                    "baseline_run_id": result["baseline_run_id"],
                    "coverage": result["coverage"],
                    "semantic_grading": "NOT_RUN",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.resume_baseline:
        owner_token = os.environ.get("AHMED_OWNER_TOKEN")
        if not owner_token:
            parser.error("AHMED_OWNER_TOKEN is required for baseline resume.")
        baseline = json.loads(
            args.resume_baseline.read_text(encoding="utf-8")
        )
        version, cases = load_case_document(
            args.dataset,
            require_baseline_size=True,
        )
        dataset_sha256 = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
        if dataset_sha256 != baseline.get("dataset_sha256"):
            parser.error("Dataset SHA-256 does not match the original baseline.")
        if version != baseline.get("dataset_version"):
            parser.error("Dataset version does not match the original baseline.")
        result = asyncio.run(
            resume_rate_limited_baseline(
                baseline,
                cases,
                owner_token=owner_token,
                max_retries=args.resume_max_retries,
                base_delay_seconds=args.resume_base_delay_seconds,
                max_delay_seconds=args.resume_max_delay_seconds,
                inter_case_delay_seconds=args.resume_inter_case_delay_seconds,
            )
        )
        output_path = args.output or Path("baseline-real-v1-complete.json")
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": "WRITTEN",
                    "output": str(output_path),
                    "original_baseline_run_id": result["original_baseline_run_id"],
                    "resume_run_id": result["resume_run_id"],
                    "coverage": result["coverage"],
                    "quality_failures": result["analysis"]["quality_failures"],
                    "provider_failures": result["analysis"]["provider_failures"],
                    "capability_gaps": result["analysis"]["capability_gaps"],
                    "semantic_grading": "NOT_RUN",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    cases = load_evaluation_cases()
    value = freeze_baseline_manifest() if args.manifest else {
        "status": "READY_FOR_TRACES",
        "case_count": len(cases),
        "real_case_count": sum(case["provenance"] == "real_case" for case in cases),
        "contract_seed_count": sum(
            case["provenance"] == "contract_seed" for case in cases
        ),
        "semantic_grading": "NOT_RUN",
    }
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()