# Ahmed Agent Blueprint

## Reference state

- **Reference date:** 2026-09-14
- **Current phase:** Phase 0 complete
- **Coverage reference:** `26 EXECUTED / 0 Capability Gaps / 0 Provider Failures / 0 HITL_BLOCKED`
- **Accepted evidence decisions:** ADR-019, ADR-020, and ADR-021
- **Current runtime:** Python 3.13, Starlette/Uvicorn, MCP streamable HTTP,
  PostgreSQL via `asyncpg`
- **Semantic grading:** `NOT_RUN`; coverage closure is not a semantic quality
  acceptance claim.

The current application is a single-owner assistant with WEB and MY_FILES
scopes, Gemini as the live primary provider, optional OpenAI-compatible
ChatGPT selection, private file retrieval, web/academic/GitHub search, policy
gates, persisted run state, recovery, audit provenance, and an authenticated
MCP surface.

## Current architecture boundary

The implemented core is the Immutable Core. Its governing boundaries are:

- **Policy Gates:** classify tools and prevent unauthorized or unsafe actions.
- **HITL Controller:** persists approval-required actions and prevents execution
  before approval.
- **Provenance Verifier:** keeps evidence, hashes, source ownership, and
  citation boundaries explicit.

These boundaries are not editable by an agent's future improvement proposals.
The current runtime does not contain a multimodal plane or a self-modifying
engine.

## Deferred Multimodal Perception & Communication Plane

This is an independent architectural layer and a roadmap only. Its
implementation is intentionally deferred until the current foundation and
quality gates receive the required approval.

### Perception

Provider-independent adapters may eventually cover:

- Voice and audio input
- Vision and image input
- Video input

Perception converts supported media into bounded, reviewable representations.
It must not silently grant permissions, change policy, or bypass provenance.

### Generation and output

Generation/output is separate from perception and may eventually cover:

- Text, voice, audio, image, and video generation
- Editing and transformation
- Composition of multimodal outputs

An input adapter must not imply that a generation or output adapter exists.
Each direction requires its own capability, policy, provenance, and evaluation
boundary.

### Provider-independent adapter boundary

The future plane must use provider-independent interfaces with selectable
execution modes:

1. Cloud
2. Local
3. Self-hosted
4. Explicit fallback

This blueprint does not select or connect a provider. No audio, vision, video,
or generation library is part of the current runtime.

## Controlled Self-Improvement boundary

Only governed, proposal-based self-improvement is allowed in the blueprint.
There is no direct self-modifying behavior.

The future controlled lifecycle is:

```text
Sandbox
  → Baseline
  → Proposal
  → Approval
  → Deploy
  → Re-evaluate
```

Each transition requires explicit artifacts, provenance, rollback/revert
ability, and evaluation evidence. In particular:

- Sandbox changes are isolated from the active runtime.
- Baselines are immutable references.
- Proposals are reviewable and cannot edit the Immutable Core directly.
- Approval is explicit and policy-gated.
- Deploy means controlled promotion, not autonomous release.
- Re-evaluate is required after promotion.

Policy Gates, the HITL Controller, and the Provenance Verifier remain outside
the self-modifying boundary and cannot be rewritten by a proposal.

The current repository implements evaluation and evidence tooling only. It does
not implement this lifecycle as an autonomous self-improvement engine.

## External development aid

Superpowers may be used as an optional development aid outside Ahmed Agent's
runtime architecture. It is not a provider, skill, runtime tool, policy
component, or self-improvement capability.

## Explicitly deferred work

- Multimodal perception and communication
- Voice and audio input/output
- Vision, image, and video processing
- Generation, editing, and composition
- Controlled self-improvement execution
- New provider connections and media libraries
