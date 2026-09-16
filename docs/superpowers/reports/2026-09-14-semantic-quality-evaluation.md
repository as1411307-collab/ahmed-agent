# Semantic Quality Evaluation Report

- **Date:** 2026-09-14
- **Branch:** `agent/blueprint-docs-2026-09-14`
- **Scope:** the 26 imported real cases from `ahmed-agent-real-cases-v1`
- **Contract:** `semantic-evaluation-contract-v1`
- **Evaluator:** deterministic reference assertions only
- **Runtime model self-grading:** forbidden
- **External sends:** none
- **Deployment:** not performed

## Status

**REVIEW_REQUIRED. Semantic quality is not closed.**

The evaluation deliberately does not turn execution success into semantic
quality. The stored baseline has valid execution traces for all 26 cases, but
answer-level factual correctness, completeness, and scope require an
independent reviewer unless a golden assertion is strong enough to decide them.

Current deterministic result:

```text
26 cases
0 PASS
21 FAIL
5 REVIEW_REQUIRED
```

This is a quality result, not an execution result. It must not be promoted to a
quality baseline.

## Contract

The contract is stored in `semantic-evaluation-contract-v1.json`. Every case
binds:

- `case_id`
- dataset/source reference
- `reference_fingerprint`
- golden/reference behavior
- expected facts represented by the documented success criteria
- expected evidence sources
- forbidden claims and tools
- completeness criteria
- scope boundary
- safe-abstention rule
- review questions

The dimensions are scored on a 0–4 scale:

| Score | Meaning |
|---:|---|
| 0 | Contradicted, fabricated, or unsafe |
| 1 | Mostly incorrect or materially unsupported |
| 2 | Partially correct with material gaps |
| 3 | Mostly correct and adequately supported |
| 4 | Fully correct, complete, scoped, and supported |

## Independent review boundary

`semantic-review-packet-v1.json` is a redacted review packet. It contains no
owner token, credential, URL, or raw case prompt. It retains the minimum
non-secret run ID and fingerprints required for trace binding. It binds every
case to:

- the reference fingerprint
- the execution trace fingerprint
- the run ID
- safe trace observations
- reference assertions
- five review questions

`semantic-review-input-template-v1.json` is the independent review interface.
It requires:

- reviewer type `human` or `independent_evaluator`
- an independence declaration
- scores for all five dimensions
- a reason and evidence notes
- matching case/reference/trace fingerprints

Missing reviews remain `REVIEW_REQUIRED`. A runtime model, or a review marked
as self-grading, is rejected. A review can only produce `PASS` when every
dimension is at least 3.

## Deterministic results

| Dimension | PASS | FAIL | REVIEW_REQUIRED |
|---|---:|---:|---:|
| Factual correctness | 0 | 0 | 26 |
| Groundedness/citation support | 2 | 21 | 3 |
| Completeness | 0 | 0 | 26 |
| Scope adherence | 0 | 0 | 26 |
| Safe abstention | 1 | 3 | 22 |

The 21 deterministic failures are primarily missing citation/evidence support
in the stored answer traces. The three safe-abstention failures also contain
affirmative claim signals without trace evidence. They are classified as
`semantic_output_or_evidence_gap`, not as confirmed runtime code defects.

## Code-failure handling

No deterministic failure was attributable to a confirmed code error in the
runtime, persistence, provider, or policy boundary. Therefore:

- no runtime code change was made to force semantic scores;
- no model-generated self-correction was accepted as evidence;
- no real-case model execution was repeated;
- the deterministic evaluator and its fail-closed review interface were tested
  instead.

If an independent review later identifies a reproducible runtime defect, that
defect must be fixed with TDD and the affected cases re-executed before this
stage can close.

## Artifacts

- `semantic_evaluation.py`
- `semantic-evaluation-contract-v1.json`
- `semantic-review-packet-v1.json`
- `semantic-review-input-template-v1.json`
- `semantic-evaluation-v1.json`
- `tests/test_semantic_evaluation.py`

The evaluation result records dataset, contract, baseline, reference, and
execution-trace fingerprints for auditability.

## Gating decision

Model Gateway, Search Fabric, and Multimodal/Voice remain gated. This stage
requires explicit independent review of the unresolved cases before it can be
marked complete or promoted.
