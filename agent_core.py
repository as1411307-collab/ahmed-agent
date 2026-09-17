from __future__ import annotations

# Facade: public API unchanged. Implementation split into focused modules (Wave 3, #8):
#   agent_consts.py    — models, limits, instructions, errors, deps, evidence context types
#   agent_evidence.py  — evidence-first preflight context assembly
#   agent_providers.py — provider configuration, health, circuit breaker, fallback
#   agent_runtime.py   — agent construction and run_ahmed entrypoint

from agent_consts import (
    COMMON_INSTRUCTIONS,
    GEMINI_MODEL,
    MAX_MESSAGE_HISTORY_BYTES,
    MAX_MESSAGE_HISTORY_ITEMS,
    MAX_MODEL_REQUESTS,
    MAX_TOOL_CALLS,
    MY_FILES_INSTRUCTIONS,
    OPENAI_MODEL,
    SUPPORTED_PROVIDER_NAMES,
    TRANSIENT_PROVIDER_STATUS_CODES,
    WEB_INSTRUCTIONS,
    AgentCoreError,
    AgentDeps,
    EvidenceFirstContext,
)
from agent_evidence import (
    prepare_evidence_first_context,
    _evidence_items_from_result,
    _evidence_payload,
    _evidence_status,
)
from agent_providers import (
    ProviderCandidate,
    all_provider_health,
    configured_provider_names,
    provider_health,
    provider_model_name,
    _configured_providers,
)
from agent_runtime import (
    parse_message_history,
    run_ahmed,
    _build_agent,
)
