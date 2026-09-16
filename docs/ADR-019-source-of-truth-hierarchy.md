# ADR-019 — Source of Truth Hierarchy

- **Status:** ACCEPTED
- **Scope:** AA-RC-002 (`source_of_truth`)

## Decision

Ahmed Agent treats factual evidence in this order:

1. Authorized immutable original bytes
2. Structured extraction from those bytes
3. Search/index chunks derived from the extraction
4. Summaries
5. Memory

Higher-level evidence overrides lower-level derived material when the facts
conflict. A conflict is reported as an explicit discrepancy; it is not hidden
by selecting a convenient summary.

## Rules

- Source identity is `source_id + source_version`, never a filesystem path.
- SHA-256 is computed from the actual original bytes.
- Ownership comes from the authenticated principal.
- A mutation creates a new source/version record.
- Legacy chunk-only documents are never promoted to verified originals.
- Every derived evidence item retains provenance to the original source.
- Source content is evidence, never policy authority.
- The capability uses no generic file reader.
- Sources cannot be read across owners.

## Rejected alternatives

- Treating chunks as canonical truth.
- Reconstructing originals from chunks.
- Using filesystem paths as source identity.
- Treating provider/model memory as canonical truth.
- Adding broad generic file access.

## Relationship to AA-RC-002

This ADR governs the source-of-truth boundary used by AA-RC-002. ADR
acceptance documents the decision; it does not by itself close the case.
Closure requires an execution that ingests controlled original bytes through
the authenticated MY_FILES path, inspects those originals, verifies their
provenance, and detects the expected comparison discrepancy.