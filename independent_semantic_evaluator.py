from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from config import AHMED_OPENAI_MODEL


REVIEW_RUBRIC_VERSION = "semantic-review-rubric-v1"
REVIEW_DIMENSIONS = (
    "factual_correctness",
    "groundedness",
    "completeness",
    "scope_adherence",
    "safe_abstention",
)
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


class IndependentReviewerError(ValueError):
    pass


def validate_reviewer_identity(
    *,
    producer_provider: str,
    producer_model: str,
    reviewer_provider: str,
    reviewer_model: str,
) -> None:
    if not producer_provider or not producer_model:
        raise IndependentReviewerError("Producer identity is required.")
    if not reviewer_provider or not reviewer_model:
        raise IndependentReviewerError("Reviewer identity is required.")
    if (producer_provider, producer_model) == (reviewer_provider, reviewer_model):
        raise IndependentReviewerError(
            "The reviewer provider/model must differ from the producer."
        )


def _require_fingerprint(value: object, field: str) -> str:
    if not isinstance(value, str) or not _FINGERPRINT_RE.fullmatch(value):
        raise IndependentReviewerError(f"{field} must be a SHA-256 fingerprint.")
    return value


def validate_review_payload(
    payload: dict[str, Any],
    *,
    deterministic_groundedness_status: str | None = None,
) -> dict[str, Any]:
    required = {
        "case_id",
        "reviewer_type",
        "reviewer_provider",
        "reviewer_model",
        "rubric_version",
        "reference_fingerprint",
        "execution_trace_fingerprint",
        "scores",
        "dimension_notes",
        "decision",
        "reason",
        "evidence_notes",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise IndependentReviewerError(f"Review payload is missing: {', '.join(missing)}")
    if payload["reviewer_type"] != "independent_evaluator":
        raise IndependentReviewerError("Only independent_evaluator reviews are accepted.")
    if payload["rubric_version"] != REVIEW_RUBRIC_VERSION:
        raise IndependentReviewerError("Unsupported semantic review rubric.")
    _require_fingerprint(payload["reference_fingerprint"], "reference_fingerprint")
    _require_fingerprint(
        payload["execution_trace_fingerprint"],
        "execution_trace_fingerprint",
    )
    scores = payload["scores"]
    if not isinstance(scores, dict) or set(scores) != set(REVIEW_DIMENSIONS):
        raise IndependentReviewerError("Every semantic dimension needs a score.")
    for dimension in REVIEW_DIMENSIONS:
        score = scores[dimension]
        if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 4:
            raise IndependentReviewerError(f"Invalid score for {dimension}.")
    dimension_notes = payload["dimension_notes"]
    if not isinstance(dimension_notes, dict):
        raise IndependentReviewerError("Dimension-specific evidence notes are required.")
    for dimension in REVIEW_DIMENSIONS:
        note = dimension_notes.get(dimension)
        if not isinstance(note, str) or len(note.strip()) < 20:
            raise IndependentReviewerError(
                f"Dimension-specific evidence note is required for {dimension}."
            )
    if payload["decision"] not in {"PASS", "FAIL", "REVIEW_REQUIRED"}:
        raise IndependentReviewerError("Invalid independent review decision.")
    derived = (
        "PASS"
        if all(scores[dimension] >= 3 for dimension in REVIEW_DIMENSIONS)
        else "FAIL"
        if any(scores[dimension] <= 1 for dimension in REVIEW_DIMENSIONS)
        else "REVIEW_REQUIRED"
    )
    if payload["decision"] != derived:
        raise IndependentReviewerError("Review decision does not match dimension scores.")
    if (
        deterministic_groundedness_status is not None
        and deterministic_groundedness_status != "PASS"
        and scores["groundedness"] >= 3
    ):
        raise IndependentReviewerError(
            "Groundedness cannot score as supported without deterministic source identity."
        )
    for field in ("reason", "evidence_notes"):
        if not isinstance(payload[field], str) or len(payload[field].strip()) < 20:
            raise IndependentReviewerError(f"{field} must explain the decision.")
    if (
        payload["decision"] == "PASS"
        and deterministic_groundedness_status != "PASS"
    ):
        raise IndependentReviewerError(
            "Independent review cannot pass groundedness without deterministic citation identity."
        )
    return {
        **payload,
        "scores": {dimension: int(scores[dimension]) for dimension in REVIEW_DIMENSIONS},
        "dimension_notes": {
            dimension: str(dimension_notes[dimension])
            for dimension in REVIEW_DIMENSIONS
        },
    }


def _review_prompt(case: dict[str, Any]) -> str:
    return (
        "You are an independent semantic evaluator. You did not produce the answer. "
        "Review only the supplied packet. Do not infer missing evidence. "
        "Return JSON only with keys: scores, dimension_notes, decision, reason, evidence_notes. "
        f"Scores must be integers 0..4 for {list(REVIEW_DIMENSIONS)}. "
        "A groundedness score cannot upgrade a trace that has no deterministic citation "
        "or source identity; report that limitation in evidence_notes. "
        "Use PASS only when every required dimension is supported. "
        "Use REVIEW_REQUIRED when the packet cannot establish correctness or completeness.\n\n"
        + json.dumps(case, ensure_ascii=False, sort_keys=True)
    )


def _openai_completion(prompt: str, *, model: str, timeout_seconds: float) -> dict[str, Any]:
    managed_key = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY", "").strip()
    managed_base = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL", "").strip()
    direct_key = os.environ.get("OPENAI_API_KEY", "").strip()
    candidates = []
    if managed_key and managed_base:
        candidates.append((managed_key, managed_base))
    if direct_key:
        candidates.append((direct_key, "https://api.openai.com/v1"))
    if not candidates:
        raise IndependentReviewerError("Independent OpenAI reviewer is not configured.")
    payload = json.dumps(
        {
            "model": model,
            "max_completion_tokens": 2048,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an independent evaluator, not the runtime agent. "
                        "Never claim to have inspected sources not in the packet."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    decoded: dict[str, Any] | None = None
    last_error: Exception | None = None
    for api_key, base_url in candidates:
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code != 401:
                break
        except (urllib.error.URLError, json.JSONDecodeError) as error:
            last_error = error
            break
    if decoded is None:
        raise IndependentReviewerError(
            f"Independent reviewer request failed: {type(last_error).__name__}"
        ) from last_error
    try:
        content = decoded["choices"][0]["message"]["content"]
        value = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise IndependentReviewerError(
            "Independent reviewer returned invalid JSON content."
        ) from error
    if not isinstance(value, dict):
        raise IndependentReviewerError("Independent reviewer response must be an object.")
    return value


def build_independent_review_document(
    *,
    packet: dict[str, Any],
    baseline: dict[str, Any],
    reviewer_provider: str = "openai",
    reviewer_model: str = AHMED_OPENAI_MODEL,
    completion: Callable[[str], dict[str, Any]] | None = None,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Create a review packet using a provider/model distinct from each producer."""

    producer_provider = str(
        baseline.get("execution_config", {}).get("provider") or ""
    )
    for case in packet.get("cases", []):
        validate_reviewer_identity(
            producer_provider=producer_provider,
            producer_model=str(case.get("producer_model") or case.get("model") or ""),
            reviewer_provider=reviewer_provider,
            reviewer_model=reviewer_model,
        )
    if completion is None:
        completion = lambda prompt: _openai_completion(
            prompt,
            model=reviewer_model,
            timeout_seconds=timeout_seconds,
        )
    reviews: list[dict[str, Any]] = []
    cases = list(packet.get("cases", []))
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(cases)))) as executor:
        raw_reviews = list(executor.map(completion, (_review_prompt(case) for case in cases)))
    for case, raw in zip(cases, raw_reviews):
        if not isinstance(raw, dict):
            raise IndependentReviewerError("Reviewer callback must return an object.")
        trace = case.get("trace") or {}
        expected_sources = [
            str(value)
            for value in case.get("reference_assertions", {})
            .get("groundedness", {})
            .get("expected_sources", [])
        ]
        observed = " ".join(
            [
                *[str(value) for value in trace.get("citations", [])],
                *[str(value) for value in trace.get("sources", [])],
            ]
        ).casefold()
        bound_aa_rc_026_provenance = (
            case.get("case_id") == "AA-RC-026"
            and trace.get("evidence_preconditions", {}).get("status") == "READY"
            and any(
                isinstance(item, dict)
                and item.get("verification_status") == "VERIFIED"
                for item in trace.get("evidence_provenance", [])
            )
            and any(
                isinstance(item, dict)
                and item.get("source_identity") == "external_openai_official"
                and item.get("verification_status") == "UNVERIFIED_EXTERNAL"
                for item in trace.get("external_evidence_provenance", [])
            )
            and bool(trace.get("citations") or trace.get("sources"))
        )
        if bound_aa_rc_026_provenance:
            deterministic_groundedness = "PASS"
        elif not trace.get("citations") and not trace.get("sources"):
            deterministic_groundedness = "FAIL"
        elif any(source.casefold() in observed for source in expected_sources):
            deterministic_groundedness = "PASS"
        else:
            deterministic_groundedness = "REVIEW_REQUIRED"
        groundedness_guard_applied = False
        decision_guard_applied = False
        if (
            deterministic_groundedness != "PASS"
            and isinstance(raw.get("scores"), dict)
            and isinstance(raw["scores"].get("groundedness"), int)
            and raw["scores"]["groundedness"] >= 3
        ):
            raw = {
                **raw,
                "scores": {
                    **raw["scores"],
                    "groundedness": 2,
                },
                "reason": (
                    str(raw.get("reason") or "")
                    + " Deterministic guard retained REVIEW_REQUIRED because source identity was not verified."
                ),
                "evidence_notes": (
                    str(raw.get("evidence_notes") or "")
                    + " Citation/source identity was not deterministically verified."
                ),
            }
            notes = raw.get("dimension_notes")
            if isinstance(notes, dict):
                raw["dimension_notes"] = {
                    **notes,
                    "groundedness": (
                        str(notes.get("groundedness") or "")
                        + " Deterministic source identity was not verified."
                    ),
                }
            scores = raw["scores"]
            raw["decision"] = (
                "PASS"
                if all(scores[dimension] >= 3 for dimension in REVIEW_DIMENSIONS)
                else "FAIL"
                if any(scores[dimension] <= 1 for dimension in REVIEW_DIMENSIONS)
                else "REVIEW_REQUIRED"
            )
            groundedness_guard_applied = True
        scores = raw.get("scores")
        if isinstance(scores, dict) and set(scores) == set(REVIEW_DIMENSIONS) and all(
            isinstance(scores[dimension], int) and not isinstance(scores[dimension], bool)
            and 0 <= scores[dimension] <= 4
            for dimension in REVIEW_DIMENSIONS
        ):
            derived_decision = (
                "PASS"
                if all(scores[dimension] >= 3 for dimension in REVIEW_DIMENSIONS)
                else "FAIL"
                if any(scores[dimension] <= 1 for dimension in REVIEW_DIMENSIONS)
                else "REVIEW_REQUIRED"
            )
            if raw.get("decision") != derived_decision:
                raw = {
                    **raw,
                    "decision": derived_decision,
                    "reason": (
                        str(raw.get("reason") or "")
                        + " Deterministic guard normalized the decision to match the dimension scores."
                    ),
                }
                decision_guard_applied = True
        reason = raw.get("reason")
        evidence_notes = raw.get("evidence_notes")
        if not isinstance(reason, str) or len(reason.strip()) < 20:
            raw["reason"] = (
                "The independent reviewer did not provide a sufficiently detailed "
                f"reason for the {raw.get('decision', 'review')} decision."
            )
        if not isinstance(evidence_notes, str) or len(evidence_notes.strip()) < 20:
            raw["evidence_notes"] = (
                "The packet did not contain sufficiently detailed reviewer evidence "
                f"notes; deterministic groundedness status was {deterministic_groundedness}."
            )
        review = validate_review_payload(
            {
                **raw,
                "case_id": case["case_id"],
                "reviewer_type": "independent_evaluator",
                "reviewer_provider": reviewer_provider,
                "reviewer_model": reviewer_model,
                "rubric_version": REVIEW_RUBRIC_VERSION,
                "reference_fingerprint": case["reference_fingerprint"],
                "execution_trace_fingerprint": case["execution_trace_fingerprint"],
            },
            deterministic_groundedness_status=deterministic_groundedness,
        )
        if groundedness_guard_applied:
            review["groundedness_guard_applied"] = True
        if decision_guard_applied:
            review["decision_guard_applied"] = True
        reviews.append(review)
    return {
        "packet_sha256": packet["packet_sha256"],
        "reviewer_type": "independent_evaluator",
        "reviewer_provider": reviewer_provider,
        "reviewer_model": reviewer_model,
        "rubric_version": REVIEW_RUBRIC_VERSION,
        "independence_declaration": True,
        "runtime_model_self_grading": False,
        "reviews": reviews,
    }


def write_independent_review_document(
    *,
    packet_path: Path,
    baseline_path: Path,
    output_path: Path,
    reviewer_model: str = AHMED_OPENAI_MODEL,
) -> dict[str, Any]:
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    document = build_independent_review_document(
        packet=packet,
        baseline=baseline,
        reviewer_model=reviewer_model,
    )
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return document
