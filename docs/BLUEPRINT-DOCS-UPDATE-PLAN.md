# Blueprint and Architecture Documentation Update Plan

- **Date:** 2026-09-14
- **Branch:** `agent/blueprint-docs-2026-09-14`
- **Scope:** documentation-only alignment with the implemented Ahmed Agent and
  accepted Phase 0 decisions.

## Execution order

1. Establish one canonical blueprint with the current Phase 0 reference state,
   the Immutable Core boundaries, and the deferred architecture roadmap.
2. Update `docs/ARCHITECTURE.md` so its current-state map and future layers
   match the canonical blueprint.
3. Update `README.md` and `replit.md` with the same status and explicit
   non-implementation statements for Multimodal/Voice and Controlled
   Self-Improvement.
4. Run documentation diff checks, the existing test suite, and compile checks.

## Required documentation decisions

- Reference Phase 0 as `26 EXECUTED / 0 Capability Gaps / 0 Provider Failures /
  0 HITL_BLOCKED`.
- Keep Multimodal Perception & Communication Plane independent and deferred.
- Cover Voice, Audio, Vision, Image, Video, Generation, Editing, and
  Composition, with Perception separated from Generation/Output.
- Describe provider-independent adapters with Cloud, Local, Self-hosted, and
  fallback options without adding providers or libraries.
- Define Controlled Self-Improvement as a governed lifecycle only:
  Sandbox → Baseline → Proposal → Approval → Deploy → Re-evaluate.
- Keep Policy Gates, HITL Controller, and Provenance Verifier outside any
  self-modifying boundary.
- Record Superpowers as an optional external development aid, not an Ahmed
  Agent runtime capability.

## Explicit non-goals

- No Python or JavaScript logic changes.
- No runtime, workflow, secret, database, dependency, or provider changes.
- No multimodal, voice, audio, vision, or self-improvement implementation.
- No deployment or publish action.