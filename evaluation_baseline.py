from __future__ import annotations

# Facade: public API unchanged. Implementation split into focused modules (Wave 3, #8):
#   eval_schema.py  — dataset constants, loading, validation, real-case normalization
#   eval_tools.py   — semantic-to-actual tool mapping and capability resolution
#   eval_http.py    — redaction, chat/upload HTTP helpers, upload E2E, evidence preconditions
#   eval_exec.py    — trace building and real-case execution
#   eval_metrics.py — run summaries, resume analysis, scoreboards
#   eval_grading.py — deterministic grading, manifest, CLI

from eval_exec import (
    authorized_source_identity_preconditions,
    build_evaluation_trace,
    execute_rate_limited_case_with_retry,
    execute_real_case,
    run_real_cases,
    _case_scope,
    _retry_after_seconds,
)
from eval_grading import (
    deterministic_grade,
    freeze_baseline_manifest,
    main,
    validate_evaluation_trace,
    _observed_sources,
)
from eval_http import (
    redact_evaluation_text,
    validate_case_evidence_preconditions,
    _aa_rc_018_upload_files,
    _cleanup_uploaded_sources,
    _extract_trace_sources,
    _is_official_openai_url,
    _is_probable_documentation_url,
    _minimal_docx_bytes,
    _minimal_pdf_bytes,
    _post_chat_message,
    _post_file_upload,
    _run_aa_rc_018_upload_e2e,
)
from eval_metrics import (
    build_quality_scoreboard,
    build_scoreboard,
    compare_scoreboards,
    resume_rate_limited_baseline,
    _build_resume_analysis,
    _code_build_identifier,
    _get_owner_json,
    _metric_summary,
    _numeric_summary,
    _runtime_health_report,
)
from eval_schema import (
    CASE_CAPABILITY_OVERRIDES,
    DATASET_PATH,
    DATASET_VERSION,
    REQUIRED_CASE_KEYS,
    REQUIRED_CHECK_KEYS,
    SECRET_FIELD_MARKERS,
    SEMANTIC_RUBRIC,
    TOOL_NAME_MAPPING,
    import_real_cases,
    load_case_document,
    load_evaluation_cases,
    validate_evaluation_cases,
    _normalize_real_case,
    _read_dataset_document,
    _reject_secret_fields,
)
from eval_tools import (
    case_execution_capability,
    resolve_case_tool_expectation,
    resolve_tool_expectation,
    _tool_names,
)


if __name__ == "__main__":
    main()
