# ADR-021: Architecture Continuity Evidence

- **Status:** Accepted
- **Scope:** AA-RC-007 (`architecture_continuity`)
- **Date:** 2026-09-14

## Decision

Foundation review uses one fixed, read-only architecture evidence adapter. It
inspects the repository's known component groups:

1. runtime;
2. persistence/state/recovery;
3. policy/security;
4. evidence/provenance;
5. evaluation;
6. actions/integrations;
7. tests.

For each group the adapter returns separate implementation evidence, wiring
evidence, and test-file evidence. Every evidence item includes the relative
allowlisted source path, file SHA-256, line range, redacted source context, and
verification status. Group status is one of `VERIFIED`, `PARTIAL`,
`DISCREPANCY`, or `NOT_FOUND`.

The adapter does not accept a path, expose source files wholesale, execute
tests or commands, modify architecture, or browse the repository generically.
Test-file presence is never reported as proof that tests passed. An
architecture fingerprint is returned for repeatable comparison, but
`architecture_unchanged` remains `NOT_ASSERTED` unless a prior trusted
snapshot is supplied.

## Rejected alternatives

1. **Generic repository browsing** — rejected because it exceeds the evidence
   boundary and can expose unrelated or sensitive files.
2. **Treating imports or file presence as operational proof** — rejected because
   implementation, wiring, and test evidence must remain distinct.
3. **Claiming continuity without a baseline fingerprint** — rejected because a
   current snapshot cannot prove historical equality by itself.

## Acceptance boundary

AA-RC-007 can close only when a fresh targeted execution:

1. calls `inspect_architecture_evidence`;
2. reports all fixed groups and their three evidence buckets;
3. preserves the non-assertion when no prior architecture snapshot exists;
4. passes security, policy, and provenance checks; and
5. is included in a fresh final Phase-0 promotion artifact with
   `26 EXECUTED`, `0 Capability Gaps`, `0 Provider Failures`, and
   `0 HITL_BLOCKED`.