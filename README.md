# Ahmed Agent

Ahmed Agent is a single-owner assistant with selectable Gemini and ChatGPT
providers. It combines conversation, web research, academic lookup, GitHub
structured search, and private `MY_FILES` retrieval behind a small
authenticated API and MCP transport.

## Reference status

Phase 0 is complete at the capability-coverage level:

```text
26 EXECUTED / 0 Capability Gaps / 0 Provider Failures / 0 HITL_BLOCKED
```

Semantic grading remains `NOT_RUN`. The current runtime does not implement
Multimodal/Voice or Controlled Self-Improvement; both are deferred roadmap
layers documented in [`docs/BLUEPRINT.md`](docs/BLUEPRINT.md).

## Runtime

The active runtime is Python, not the Node scaffold:

```bash
uv run server.py
```

The server listens on `PORT` (default `8000`). The Replit workflow maps the
service to the preview and deployment router.

## Product surface

- Arabic RTL owner console at `/`
- Owner-authenticated chat at `/chat/message`
- Private uploads at `/files/upload`
- Provider health at `/health/provider`
- Dependency and persistence report at `/doctor`
- Authenticated streamable HTTP MCP at `/mcp`

The console keeps the owner token in the current browser tab only. It does not
write the token into the URL, chat payload, page content, or application logs.

## Skills

| Scope | Capabilities |
| --- | --- |
| `WEB` | Tavily/Search Fabric, academic search, GitHub structured search |
| `MY_FILES` | TXT, Markdown, PDF text layers, DOCX, PostgreSQL FTS, optional FastEmbed |
| `MODELS` | Per-message choice between Gemini and ChatGPT |
| `MCP` | `ping` and `web_search`, protected by owner bearer authentication |
| `POLICY` | risk classification, approval/rejection, persistence, audit events |

## Required secrets

Configure these in Replit Secrets; never paste their values into source or
chat:

- `AHMED_OWNER_TOKEN`
- `GEMINI_API_KEY`
- `AI_INTEGRATIONS_OPENAI_API_KEY`
- `AI_INTEGRATIONS_OPENAI_BASE_URL`

Optional provider secrets are documented in `replit.md`.

## Verification

```bash
python -B -m unittest discover -s tests -p 'test_*.py' -v
python -B -m compileall -q server.py auth.py my_files.py embeddings.py agent_core.py tests
git diff --check
```

The repository uses `unittest` for the current regression suite. The Node
packages under `artifacts/` remain registered Replit artifacts; they are not
the active Ahmed runtime.

## Structure

See:

- [`docs/BLUEPRINT.md`](docs/BLUEPRINT.md)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md)
- [`replit.md`](replit.md)