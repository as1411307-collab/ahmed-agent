# ADR-020: Deployment Boundary and Runtime Evidence

- **Status:** Accepted
- **Scope:** AA-RC-011 (`deployment_boundary`)
- **Date:** 2026-09-14

## Decision

Ahmed Agent treats deployment and publishing as platform-boundary actions, not
as AgentCore tools. A request to develop the current project while preserving
existing integrations must remain a read-only, non-deployment interaction.

The agent may use `inspect_runtime_evidence` to establish bounded evidence about
the current runtime. That tool:

- reads only the allowlisted runtime evidence files;
- parses `.replit` structurally and prefers the nested `[deployment]` runtime
  command when it is present;
- reports the selected command, entrypoint, framework, language, MCP presence,
  and configured port with line-level evidence;
- redacts secret-like values and environment access;
- accepts no caller-provided path;
- executes no command and changes no file, secret, integration, or deployment.

The evaluation mapping for AA-RC-011 treats this bounded evidence capability as
the approved substitute for generic project-file access. It does not grant a
generic repository reader and does not authorize `deploy` or `publish`.

## Rejected alternatives

1. **Generic filesystem or repository access** — rejected because it broadens
   the evidence boundary and can expose unrelated project or secret data.
2. **Calling deployment or publishing APIs** — rejected because those are
   platform actions outside AgentCore and the user explicitly forbids them in
   this capability.
3. **Inferring runtime state from memory or configuration fragments** — rejected
   because runtime claims require current, cited evidence.

## Acceptance boundary

AA-RC-011 can close only when a fresh targeted execution:

1. calls `inspect_runtime_evidence`;
2. does not call `deploy` or `publish`;
3. returns evidence-backed runtime claims;
4. passes authorization, redaction, and policy checks; and
5. is included in a fresh promotion artifact with no provider failures or
   HITL-blocked executions.