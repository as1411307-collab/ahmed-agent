# Core Operational Acceptance Report

- **Date:** 2026-09-14
- **Branch:** `agent/blueprint-docs-2026-09-14`
- **Scope:** authenticated operational paths already present in Ahmed Agent
- **External sends:** provider-backed Gemini execution was invoked for acceptance and the real-case baseline; no user-facing external side effect was performed
- **Deployment:** not performed
- **Secrets:** used internally by the environment, never printed or persisted

## Result

**PASS for the authenticated operational boundary.** Authenticated HTTP,
PostgreSQL persistence, MCP lifecycle, file validation, recovery controls,
idempotency, audit integrity, internal runtime alerting, real Gemini chat, and
provider fallback fault injection were exercised with bounded test data.

The real chat acceptance returned `200` through Gemini. The fallback check used
an in-test transient provider fault and a ready alternate provider; it did not
alter production provider configuration or send a synthetic fault to Gemini.

## Paths accepted

### Authentication

- Unauthenticated protected routes returned `401 AUTHENTICATION_REQUIRED`.
- Authenticated `GET /health/provider?provider=gemini` returned `200` with
  provider status `READY`.
- Authenticated `/doctor`, `/metrics/runtime`, `/alerts/runtime`, and
  `/retention/preview` returned `200`.

### Chat and persistence

- Authenticated `/chat/message` with the real Gemini provider returned `200`.
- The provider health route reported Gemini `READY` before the request.
- The reply was persisted as a successful run.
- Persisted run state was:
  `succeeded / completed / resolved`.
- Five checkpoints were observed:
  `context_loaded`, `model_running`, `response_ready`,
  `response_persisted`, `completed`.
- One assistant message was persisted.

### MY_FILES

- Safe UTF-8 text upload returned `201`.
- The document was `ready`, had one searchable chunk, and
  `original_available=true`.
- Unsupported `.exe` upload returned `415`.
- Invalid binary content with a `.txt` name returned `415` with
  `invalid_file_content`.

### MCP

- Authenticated Streamable HTTP `initialize` returned `200`.
- An MCP session ID was issued.
- The server returned protocol version `2025-03-26`.

### Evidence and semantic evaluation

- Runtime evidence references now carry canonical source identity, SHA-256,
  locator, verification status, and trust classification.
- Raw persisted evidence items are projected into `evidence_provenance` and
  citation lists only when the citation contract validates; absent evidence
  does not produce a fabricated citation.
- New baseline: 26/26 executed, 0 execution failures, 0 provider rate limits,
  0 capability gaps, and 0 missing traces.
- Baseline contained 263 output citations and 279 available evidence citations.
- Semantic evaluation: 0 PASS, 2 deterministic semantic FAIL, 24
  `REVIEW_REQUIRED`, with independent review still required. The two FAIL
  cases are answer/evidence failures, not provider or execution failures.
- Provider/execution failures are classified as `NOT_DETERMINED` and reported
  separately from semantic answer failures.

### PostgreSQL, leases, recovery, and idempotency

- Session, run, and checkpoints were created in PostgreSQL.
- Lease renewal returned `true`.
- An expired lease was detected as orphaned.
- Recovery action for a run at `model_running` was `manual_review`.
- `claim_orphaned_run` succeeded.
- Repeating the same pending-action idempotency key returned the same action.
- Recovery endpoint returned `200`; resume correctly returned `409
  MODEL_EXECUTION_NOT_IDEMPOTENT` for the model stage.

### Audit and runtime alerts

- `verify_audit_chain()` returned `verified=true`.
- Runtime metrics exposed orphan, lease-expiration, recovery-failure, and
  idempotency counters.
- `/alerts/runtime` returned `200`; all current alert rules were healthy.
- Recovery health is already observable internally through runtime metrics and
  the `orphan_lease_anomaly` and `recovery_failure` rules. No extra behavior
  was needed.

## Regression fixed

The live MCP initialize request originally returned `500` with
`Task group is not initialized`. `OwnerMCPAuthMiddleware` consumed lifespan
startup without forwarding it to MCP's Streamable HTTP app.

The minimal fix forwards the lifespan messages after performing the existing
orphan scan. A regression test now proves that an authenticated initialize
request succeeds after lifecycle startup.

## Cleanup

All temporary operational runs, sessions, checkpoints, messages, pending
actions, tool events, and uploaded document records were removed by exact
identifiers. The 26 baseline runs and sessions were also removed by their
recorded run IDs after artifact generation.
Verification after cleanup found:

- marked test runs: `0`
- marked test sessions: `0`
- marked test documents: `0`
- marked test pending actions: `0`

Audit events were preserved according to the retention policy.

## Verification

- Targeted provenance/provider/semantic tests: `49 passed`
- Full suite: `138 passed, 1 skipped`
- Python compile check: passed
- `uv pip check`: passed; 119 packages compatible
- `git diff --check`: passed
- Controlled fallback fault injection: passed; fallback event was recorded with
  the actual alternate provider and bounded retry behavior.

## Not started

- Independent human semantic review for the 26 cases
- Multimodal/Voice
- Deployment or publishing
