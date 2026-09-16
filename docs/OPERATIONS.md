# Ahmed Agent operations

## Start and restart

The configured workflow is:

```text
artifacts/api-server: API Server
```

Its command is:

```bash
uv run server.py
```

The server reads `PORT` from the environment and defaults to `8000`.

## Required configuration

Use Replit Secrets for:

```text
AHMED_OWNER_TOKEN
GEMINI_API_KEY
AI_INTEGRATIONS_OPENAI_API_KEY
AI_INTEGRATIONS_OPENAI_BASE_URL
```

Use the exact names above. Do not keep typo aliases such as `GEMNI_API_KEY`;
they are ignored by the application and make incident diagnosis harder.

Optional configuration includes:

```text
TAVILY_API_KEY
GITHUB_TOKEN
BRAVE_SEARCH_API_KEY
MY_FILES_EMBEDDING_MODEL
```

## Verification sequence

Run the fast local checks first:

```bash
python -B -m unittest discover -s tests -p 'test_*.py' -v
python -B -m compileall -q server.py auth.py my_files.py embeddings.py agent_core.py tests
git diff --check
```

Then verify the public health boundary:

```bash
curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/mcp
```

Expected results:

- `/health` returns `200`.
- `/mcp` without a bearer token returns `401`.

Authenticated provider, chat, upload, doctor, and MCP protocol checks must be
run through an approved client without printing the owner token or provider
secret.

Runtime metrics are available to the owner as aggregated operational metadata:

```text
GET /metrics/runtime?hours=24
```

The response includes total, completed, failed, active, and currently orphaned
runs; average duration; lease expirations; recovery attempts; idempotency hits;
and counts by execution stage. The maximum window is 720 hours, and prompts,
provider payloads, and secrets are not returned.
These counts are windowed by `hours`; they should not be compared directly with
historical totals from storage or `/doctor` unless the time window and run
definition match.

Retention is intentionally conservative:

- succeeded-run checkpoints become eligible after 30 days;
- failed and orphaned checkpoints become eligible after 90 days;
- `audit_events` are preserved and never deleted by this policy;
- test cleanup is restricted to the explicit `fault_injection` and `probe`
  session scopes.

Use the owner-only preview before cleanup:

```text
GET /retention/preview
POST /retention/cleanup?confirm=CHECKPOINTS_ONLY
POST /retention/cleanup?confirm=TEST_DATA_ONLY
```

The first cleanup mode removes only checkpoints that crossed their retention
age. The second additionally removes operational rows belonging to the explicit
test scopes. Both modes append an auditable cleanup event and report the
preserved audit-event count.

Minimal operational alerts are evaluated separately from delivery:

```text
GET /alerts/runtime
```

The endpoint is owner-only and evaluates only service health, audit integrity,
orphan/lease anomalies, recovery failures, and sustained failure-rate increases.
It returns persisted states such as `healthy`, `warning`, and `critical`;
`recovered` is recorded as a transition and the current state returns to
`healthy`. Delivery is currently `not_configured`. Rules use minimum sample
sizes and cooldowns, so repeated polling does not create an audit event unless a
rule opens, escalates, recovers, or reaches a new suppression interval.

The Evaluation Baseline harness is deterministic by default and does not call a
live provider:

```text
python evaluation_baseline.py
python evaluation_baseline.py --manifest
python -m unittest tests.test_evaluation_baseline -v
```

The initial 22 cases are labeled `contract_seed`, so their deterministic
contract results must not be presented as a real-user quality score. Replace or
extend them with 20–30 `real_case` entries before running a live baseline.
Semantic grading remains `NOT_RUN` until a reviewed rubric and safe grader are
configured.

Historical cases are imported through the same harness rather than a parallel
evaluation system. The input must be a JSON document containing `cases`, where
each case uses `case_type: "real_case"` and a non-empty `source_reference`
(for example, a reviewed export reference). The importer accepts the structured
fields `case_id`, `category`, `input`, `expected_behavior`,
`expected_sources`, `required_tools`, `forbidden_tools`,
`success_criteria`, and `forbidden_behavior`:

```text
python evaluation_baseline.py --validate-real incoming.json
python evaluation_baseline.py --import-real incoming.json --output real-cases.json
```

The quality scoreboard refuses to produce a quality result when there are no
qualified `real_case` entries. Contract seeds and real cases are counted and
reported separately. Regression comparison is case- and category-level; it
does not reduce a mixed dataset to one unqualified average.

Case tool names are semantic expectations and are mapped without rewriting the
case. `my_files` maps to the actual `search_my_files` AgentCore tool;
`web_search` maps directly. `file_access` remains unavailable. The
`project_file_access` expectation is case-specific: AA-RC-007 uses the fixed
architecture-group evidence adapter, AA-RC-011 and AA-RC-016 use bounded
runtime evidence, and AA-RC-014 uses fixed source-status evidence. These
adapters are read-only and are not generic repository browsers. `deploy` and
`publish` remain platform boundaries rather than model tools.

For a persisted run, inspect its owner-only state trace with:

```text
GET /runs/{run_id}/checkpoints
```

The trace identifies whether a failure occurred during context loading, model
execution, response persistence, or completion without returning prompt text.

Recovery is explicit and owner-only:

```text
GET  /runs/{run_id}/recovery
POST /runs/{run_id}/resume
```

`POST /resume` only completes a response whose messages were already persisted.
Runs interrupted during model execution return a safe retry/manual-review result
instead of invoking the model again.

`orphaned` is a recovery condition, not a replacement for the execution stage:
inspect both `status`/`recovery_status` and `stage` when diagnosing a run.

## Failure diagnosis

- `401`: check the exact `AHMED_OWNER_TOKEN` secret name and bearer header.
- provider failure: check `GEMINI_API_KEY` and `/health/provider`.
- persistence failure: check PostgreSQL availability and `/doctor`.
- an incomplete run: inspect `/runs/{run_id}/checkpoints` before changing model
  or tool code.
- a recovery conflict: inspect `/runs/{run_id}/recovery`; an active lease means
  another worker still owns the run.
- a full crash/restart check: run
  `AHMED_RUN_RECOVERY_E2E=1 python -m unittest tests.test_recovery_fault_injection -v`
  against a development PostgreSQL database only.
- empty search results: distinguish provider no-results from provider failure
  in the returned provenance and doctor report.
- embedding failure: continue with FTS and inspect the embedding health field.