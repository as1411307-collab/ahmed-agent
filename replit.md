# Ahmed Agent

Ahmed Agent is a single-owner AI assistant with selectable Gemini or ChatGPT
chat, web search, academic search, GitHub structured search, and private
`MY_FILES` retrieval.

## Run & Operate

- `uv run server.py` — start the Python API and MCP server on `PORT` (default 8000)
- `python -m unittest discover -s tests -p 'test_*.py'` — run the regression suite
- `python -m compileall -q server.py my_files.py tests` — compile check
- `git diff --check` — whitespace check

The active Replit workflow is `artifacts/api-server: API Server`. The Node/TypeScript
artifact under `artifacts/api-server` is a scaffold and is not the Ahmed runtime.

## Stack

- Python 3.13
- Starlette + Uvicorn
- MCP SDK streamable HTTP transport
- PydanticAI + Google Gemini/OpenAI-compatible ChatGPT
- PostgreSQL via asyncpg
- PostgreSQL FTS with optional local FastEmbed hybrid retrieval
- Crossref, DataCite, OpenAlex, GitHub REST API, and Tavily

## Reference status and deferred blueprint

- Phase 0 capability coverage is complete:
  `26 EXECUTED / 0 Capability Gaps / 0 Provider Failures / 0 HITL_BLOCKED`.
- Semantic grading remains `NOT_RUN`.
- Multimodal Perception & Communication is a separate deferred plane covering
  Voice, Audio, Vision, Image, Video, Generation, Editing, and Composition.
- Perception is separate from Generation/Output, and future adapters must be
  provider-independent with Cloud, Local, Self-hosted, and fallback modes.
- Controlled Self-Improvement is limited to the future governed sequence
  `Sandbox → Baseline → Proposal → Approval → Deploy → Re-evaluate`.
- There is no direct Self-Modifying implementation. Policy Gates, the HITL
  Controller, and the Provenance Verifier remain outside that boundary.
- Superpowers is an optional external development aid, not an Ahmed Agent
  runtime capability.

See [`docs/BLUEPRINT.md`](docs/BLUEPRINT.md) for the canonical architecture
blueprint.

## Capabilities

- `WEB` scope: `web_search`, `academic_search`, `github_search`, and the
  test-only HITL action.
- `MY_FILES` scope: private file search and the test-only HITL action.
- MCP-public tools: `ping` and `web_search`.
- Supported uploads: TXT, Markdown, PDF text layers, and DOCX.

## Security decisions

- HTTP routes and MCP transport require the single-owner bearer token.
- Uploaded filenames are canonicalized and never treated as filesystem paths.
- File content, archive entries, request bodies, provider responses, and model
  outputs are treated as untrusted input.
- SQL uses parameterized queries.
- Search results and tool returns can be retained inside persisted agent message
  history; raw provider payloads are not stored as a separate record.
- Gemini and owner authentication fail closed when their secrets are missing or
  invalid.

## Required configuration

- `DATABASE_URL` — managed PostgreSQL connection
- `GEMINI_API_KEY` — valid Google Gemini API key
- `AI_INTEGRATIONS_OPENAI_API_KEY` — Replit-managed OpenAI-compatible key
- `AI_INTEGRATIONS_OPENAI_BASE_URL` — Replit-managed OpenAI-compatible base URL
- `AHMED_OWNER_TOKEN` — owner bearer token

Optional:

- `TAVILY_API_KEY`
- `GITHUB_TOKEN`
- `BRAVE_SEARCH_API_KEY`
- `MY_FILES_EMBEDDING_MODEL`

Never place secret values in source files, logs, browser code, or chat.

## Gotchas

- Gemini and ChatGPT are selectable model providers; web search is a separate
  Tavily/Search Fabric capability. Google Search grounding is not enabled by
  default.
- The ChatGPT provider is only available when both OpenAI integration variables
  are configured. It never silently falls back to Gemini after the user selects
  ChatGPT.
- Brave is optional and disabled unless configured and selected by routing.
- FastEmbed is lazy-loaded and falls back to FTS when unavailable.
- PostgreSQL schema tables must exist before persistence health checks can pass.
- `AHMED_OWNER_TOKEN` must be restored before authenticated route or MCP E2E tests.
- Run recovery is owner-only and lease-based. Expired leases set
  `recovery_status=orphaned` without changing the execution stage.
  `/runs/{run_id}/resume` completes only persisted response tails; it never
  replays a `model_running` call automatically.
- `/metrics/runtime?hours=24` exposes bounded, owner-only aggregate runtime
  metrics without prompts or secrets.
- Retention uses 30-day succeeded-checkpoint and 90-day failed/orphaned-
  checkpoint windows. Cleanup never deletes `audit_events`; test cleanup is
  limited to the explicit `fault_injection` and `probe` scopes.
- `/alerts/runtime` evaluates a small owner-only operational policy for health,
  audit integrity, orphaning/leases, recovery failures, and sustained failure
  rate. It persists transitions but does not deliver external notifications.
- `evaluation_baseline.py` provides a deterministic harness. Its initial 22
  cases are `contract_seed` entries, not a real-user quality score; add reviewed
  `real_case` entries before live-provider evaluation.
- Real cases enter through the harness importer with `case_type=real_case` and a
  required `source_reference`. Quality reports are gated until qualified real
  cases exist; regression comparison is per case and category.