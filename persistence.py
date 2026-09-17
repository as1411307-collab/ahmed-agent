from __future__ import annotations

# Facade: public API unchanged. Implementation split into focused modules (Wave 3, #8):
#   persistence_core.py   — pool, schema, errors, canonical JSON, storage health
#   persistence_runs.py   — sessions, runs, messages, leases, recovery, retention, metrics
#   persistence_events.py — tool events, checkpoints, auth/recovery events
#   persistence_audit.py  — audit hash chain, pending actions, chain verification
#   persistence_docs.py   — MY_FILES documents, original sources, search

from persistence_audit import (
    _append_audit_event,
    approve_pending_action,
    claim_pending_action_execution,
    complete_pending_action,
    create_pending_action,
    reject_pending_action,
    verify_audit_chain,
)
from persistence_core import (
    RETENTION_POLICY,
    TEST_RETENTION_SCOPES,
    PersistenceError,
    _canonical_json,
    _ensure_policy_schema,
    _get_pool,
    close_pool,
    doctor_storage_health,
)
from persistence_docs import (
    cleanup_evaluation_run,
    create_original_source,
    delete_documents_for_source_ids,
    delete_original_source,
    find_document_by_hash,
    read_authorized_original_source,
    search_document_chunks,
    search_fts_document_chunks,
    search_vector_document_chunks,
    store_document,
    store_document_embeddings,
    update_document_embedding_status,
    update_original_source_extraction_status,
)
from persistence_events import (
    load_run_checkpoints,
    load_run_evaluation_data,
    record_auth_event,
    record_recovery_event,
    record_run_checkpoint,
    record_tool_event,
)
from persistence_runs import (
    append_new_messages,
    claim_orphaned_run,
    cleanup_retention,
    create_run,
    default_worker_id,
    ensure_session,
    finish_run,
    load_message_history,
    load_run_recovery,
    mark_orphaned_runs,
    normalize_metrics_window,
    release_orphaned_run,
    renew_run_lease,
    retention_preview,
    run_message_count,
    runtime_metrics,
    persist_runtime_alert_evaluations,
)
