import os
import asyncio
import logging
import json
import socket
import time
from uuid import UUID, uuid4
from pathlib import Path
from typing import TypedDict

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from agent_core import (
    SUPPORTED_PROVIDER_NAMES,
    AgentCoreError,
    parse_message_history,
    provider_model_name,
    provider_health,
    run_ahmed,
)
from alerting import evaluate_runtime_rules
from auth import authorize_owner
from doctor import get_doctor_report
from persistence import (
    PersistenceError,
    append_new_messages,
    approve_pending_action,
    claim_pending_action_execution,
    complete_pending_action,
    create_run,
    claim_orphaned_run,
    default_worker_id,
    ensure_session,
    finish_run,
    load_run_recovery,
    load_run_checkpoints,
    load_message_history,
    mark_orphaned_runs,
    normalize_metrics_window,
    reject_pending_action,
    record_tool_event,
    record_run_checkpoint,
    cleanup_retention,
    persist_runtime_alert_evaluations,
    record_recovery_event,
    retention_preview,
    release_orphaned_run,
    renew_run_lease,
    run_message_count,
    runtime_metrics,
    verify_audit_chain,
)
from config import MAX_UPLOAD_BYTES
from my_files import SUPPORTED_EXTENSIONS, extension_for, ingest_document, sanitize_filename
from run_state import RecoveryAction, RunStage, RunState, recovery_action
from skill_tools import register_skill_tools


class PingResult(TypedDict):
    ok: bool
    message: str


server = MCPServer("Ahmed Agent")
WEB_DIR = Path(__file__).parent / "web"
logger = logging.getLogger("ahmed_agent")
WORKER_ID = default_worker_id()


async def _run_lease_heartbeat(run_id: str) -> None:
    try:
        while True:
            await asyncio.sleep(30)
            renewed = await renew_run_lease(run_id=run_id, worker_id=WORKER_ID)
            if not renewed:
                logger.warning("Run lease is no longer owned run_id=%s", run_id)
                return
    except asyncio.CancelledError:
        raise
    except (PersistenceError, ValueError) as error:
        logger.error(
            "Run lease heartbeat failed run_id=%s error_type=%s",
            run_id,
            type(error).__name__,
        )


class OwnerMCPAuthMiddleware:
    """Require Ahmed's single-owner bearer token for every MCP HTTP request."""

    def __init__(self, app: ASGIApp, *, path: str = "/mcp") -> None:
        self.app = app
        self.path = path.rstrip("/") or "/"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") == "lifespan":
            await self._handle_lifespan(scope, receive, send)
            return
        request_path = str(scope.get("path", ""))
        is_mcp_request = request_path == self.path or request_path.startswith(
            f"{self.path}/"
        )
        if scope.get("type") != "http" or not is_mcp_request:
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        user, status_code, error_code = await authorize_owner(
            request,
            endpoint="/mcp",
        )
        if user is None:
            response = JSONResponse(
                {
                    "error": "owner authentication required",
                    "code": error_code,
                },
                status_code=status_code,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)

    async def _handle_lifespan(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        first_message = await receive()
        if first_message.get("type") == "lifespan.startup":
            try:
                orphaned = await mark_orphaned_runs()
                if orphaned:
                    logger.warning("Marked orphaned runs count=%d", len(orphaned))
            except PersistenceError as error:
                logger.error(
                    "Startup run recovery scan failed error_type=%s",
                    type(error).__name__,
                )

        replay_first_message = True

        async def replay_receive() -> dict[str, object]:
            nonlocal replay_first_message
            if replay_first_message:
                replay_first_message = False
                return first_message
            return await receive()

        await self.app(scope, replay_receive, send)


def _structured_log(
    *,
    trace_id: str,
    session_id: str,
    run_id: str,
    provider: str,
    status: str,
    latency_ms: int,
    error_code: str | None = None,
) -> None:
    logger.info(
        json.dumps(
            {
                "trace_id": trace_id,
                "session_id": session_id,
                "run_id": run_id,
                "provider": provider,
                "model": provider_model_name(provider),  # type: ignore[arg-type]
                "status": status,
                "latency_ms": latency_ms,
                **({"error_code": error_code} if error_code else {}),
            },
            separators=(",", ":"),
        )
    )


def _request_uuid(value: object, field_name: str) -> str:
    if value is None:
        return str(uuid4())
    if not isinstance(value, str):
        raise ValueError(f"{field_name} غير صالح.")
    try:
        return str(UUID(value))
    except ValueError as error:
        raise ValueError(f"{field_name} غير صالح.") from error


def _action_id_for_auth_audit(request: Request) -> str | None:
    raw_action_id = request.path_params.get("action_id")
    if not isinstance(raw_action_id, str):
        return None
    try:
        return str(UUID(raw_action_id))
    except ValueError:
        return None


@server.tool(
    description="Safely echoes a message for connectivity testing.",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    structured_output=True,
)
def ping(message: str) -> PingResult:
    return {"ok": True, "message": message}


register_skill_tools(server)


@server.custom_route("/", methods=["GET"])
async def chat_page(_: Request) -> Response:
    return FileResponse(WEB_DIR / "index.html")


@server.custom_route("/health", methods=["GET"])
async def health_route(_: Request) -> Response:
    return JSONResponse({"ok": True})


@server.custom_route("/health/provider", methods=["GET"])
async def provider_health_route(request: Request) -> Response:
    user, status_code, error_code = await authorize_owner(
        request,
        endpoint="/health/provider",
    )
    if user is None:
        return JSONResponse(
            {"error": "owner authentication required", "code": error_code},
            status_code=status_code,
        )
    provider = request.query_params.get("provider", "gemini")
    if provider not in SUPPORTED_PROVIDER_NAMES:
        return JSONResponse(
            {"error": "provider غير صالح. استخدم gemini أو openai."},
            status_code=400,
        )
    return JSONResponse(provider_health(provider))  # type: ignore[arg-type]


@server.custom_route("/doctor", methods=["GET"])
async def doctor_route(request: Request) -> Response:
    user, status_code, error_code = await authorize_owner(
        request,
        endpoint="/doctor",
    )
    if user is None:
        return JSONResponse(
            {"error": "owner authentication required", "code": error_code},
            status_code=status_code,
        )
    report = await get_doctor_report(
        probe_search=request.query_params.get("probe") == "1"
    )
    return JSONResponse(report)


@server.custom_route("/metrics/runtime", methods=["GET"])
async def runtime_metrics_route(request: Request) -> Response:
    user, status_code, error_code = await authorize_owner(
        request,
        endpoint="/metrics/runtime",
    )
    if user is None:
        return JSONResponse(
            {"error": "owner authentication required", "code": error_code},
            status_code=status_code,
        )
    try:
        window_hours = normalize_metrics_window(
            request.query_params.get("hours", "24")
        )
        metrics = await runtime_metrics(window_hours=window_hours)
    except ValueError:
        return JSONResponse(
            {"error": "نافذة المقاييس يجب أن تكون بين 1 و720 ساعة."},
            status_code=400,
        )
    except PersistenceError:
        return JSONResponse(
            {"error": "تعذر تحميل مقاييس التشغيل."},
            status_code=502,
        )
    return JSONResponse(
        {
            "metrics": metrics,
            "data_boundary": "aggregated_operational_metadata",
        }
    )


def _serialize_alert_state(alert: dict[str, object]) -> dict[str, object]:
    serialized = dict(alert)
    for field in ("last_triggered_at", "recovered_at", "evaluated_at"):
        value = serialized.get(field)
        if hasattr(value, "isoformat"):
            serialized[field] = value.isoformat()
    return serialized


@server.custom_route("/alerts/runtime", methods=["GET"])
async def runtime_alerts_route(request: Request) -> Response:
    user, status_code, error_code = await authorize_owner(
        request,
        endpoint="/alerts/runtime",
    )
    if user is None:
        return JSONResponse(
            {"error": "owner authentication required", "code": error_code},
            status_code=status_code,
        )
    try:
        metrics = await runtime_metrics(window_hours=24)
        service_healthy = True
    except PersistenceError:
        metrics = {}
        service_healthy = False
    try:
        audit = await verify_audit_chain()
        audit_healthy = bool(audit.get("verified"))
    except PersistenceError:
        audit = {
            "status": "ERROR",
            "verified": False,
            "safe_error_code": "AUDIT_STORAGE_UNAVAILABLE",
        }
        audit_healthy = False
    evaluations = evaluate_runtime_rules(
        metrics=metrics,
        service_healthy=service_healthy,
        audit_healthy=audit_healthy,
    )
    try:
        states = await persist_runtime_alert_evaluations(evaluations)
    except PersistenceError:
        return JSONResponse(
            {
                "error": "تعذر حفظ حالة التنبيهات التشغيلية.",
                "service_healthy": service_healthy,
                "audit": audit,
            },
            status_code=502,
        )
    return JSONResponse(
        {
            "alerts": [_serialize_alert_state(state) for state in states],
            "delivery": "not_configured",
            "metrics_window_hours": 24,
            "audit": {
                "verified": audit.get("verified"),
                "safe_error_code": audit.get("safe_error_code"),
            },
        }
    )


@server.custom_route("/retention/preview", methods=["GET"])
async def retention_preview_route(request: Request) -> Response:
    user, status_code, error_code = await authorize_owner(
        request,
        endpoint="/retention/preview",
    )
    if user is None:
        return JSONResponse(
            {"error": "owner authentication required", "code": error_code},
            status_code=status_code,
        )
    try:
        preview = await retention_preview()
    except PersistenceError:
        return JSONResponse(
            {"error": "تعذر تحميل معاينة سياسة الاحتفاظ."},
            status_code=502,
        )
    return JSONResponse(preview)


@server.custom_route("/retention/cleanup", methods=["POST"])
async def retention_cleanup_route(request: Request) -> Response:
    user, status_code, error_code = await authorize_owner(
        request,
        endpoint="/retention/cleanup",
    )
    if user is None:
        return JSONResponse(
            {"error": "owner authentication required", "code": error_code},
            status_code=status_code,
        )
    confirmation = request.query_params.get("confirm")
    if confirmation not in {"TEST_DATA_ONLY", "CHECKPOINTS_ONLY"}:
        return JSONResponse(
            {
                "error": (
                    "يتطلب التنظيف تأكيد TEST_DATA_ONLY أو CHECKPOINTS_ONLY؛ "
                    "لا يتم حذف audit_events."
                ),
            },
            status_code=400,
        )
    try:
        result = await cleanup_retention(
            include_test_data=confirmation == "TEST_DATA_ONLY"
        )
    except PersistenceError:
        return JSONResponse(
            {"error": "تعذر تنفيذ تنظيف بيانات الاختبارات."},
            status_code=502,
        )
    return JSONResponse(result)


async def _load_recovery_state(run_id: str) -> dict[str, object] | None:
    await mark_orphaned_runs()
    return await load_run_recovery(run_id)


@server.custom_route("/runs/{run_id}/recovery", methods=["GET"])
async def run_recovery_route(request: Request) -> Response:
    user, status_code, error_code = await authorize_owner(
        request,
        endpoint="/runs/{run_id}/recovery",
    )
    if user is None:
        return JSONResponse(
            {"error": "owner authentication required", "code": error_code},
            status_code=status_code,
        )
    try:
        run_id = _request_uuid(request.path_params.get("run_id"), "run_id")
        recovery = await _load_recovery_state(run_id)
    except (ValueError, PersistenceError):
        return JSONResponse(
            {"error": "تعذر تحميل حالة استرداد التنفيذ."},
            status_code=400,
        )
    if recovery is None:
        return JSONResponse({"error": "التنفيذ غير موجود."}, status_code=404)
    try:
        action = recovery_action(RunStage(str(recovery["stage"])))
    except ValueError:
        action = RecoveryAction.MANUAL_REVIEW
    return JSONResponse(
        {
            "run": recovery,
            "recovery_action": action.value,
            "safe_to_resume": (
                recovery.get("recovery_status") == "orphaned"
                and action is RecoveryAction.COMPLETE_PERSISTED_TAIL
            ),
        }
    )


@server.custom_route("/runs/{run_id}/resume", methods=["POST"])
async def run_resume_route(request: Request) -> Response:
    user, status_code, error_code = await authorize_owner(
        request,
        endpoint="/runs/{run_id}/resume",
    )
    if user is None:
        return JSONResponse(
            {"error": "owner authentication required", "code": error_code},
            status_code=status_code,
        )
    try:
        run_id = _request_uuid(request.path_params.get("run_id"), "run_id")
        recovery = await _load_recovery_state(run_id)
    except (ValueError, PersistenceError):
        return JSONResponse(
            {"error": "تعذر تحميل حالة استرداد التنفيذ."},
            status_code=400,
        )
    if recovery is None:
        return JSONResponse({"error": "التنفيذ غير موجود."}, status_code=404)

    status = str(recovery["status"])
    if status == "succeeded":
        return JSONResponse({"status": "already_completed", "run": recovery})
    if status == "failed":
        return JSONResponse(
            {
                "status": "terminal_failure",
                "code": "NEW_RUN_REQUIRED",
                "run": recovery,
            },
            status_code=409,
        )
    if status == "running" and recovery.get("recovery_status") != "orphaned":
        return JSONResponse(
            {
                "status": "busy",
                "code": "LEASE_ACTIVE",
                "message": "التنفيذ ما زال مملوكًا لعامل نشط.",
            },
            status_code=409,
        )

    try:
        stage = RunStage(str(recovery["stage"]))
        action = recovery_action(stage)
    except ValueError:
        return JSONResponse(
            {"status": "manual_review", "code": "UNKNOWN_RUN_STAGE"},
            status_code=409,
        )

    if action is RecoveryAction.RETRY_NEW_RUN:
        return JSONResponse(
            {
                "status": "retry_new_run",
                "code": "MODEL_RETRY_REQUIRES_NEW_RUN",
                "message": "لا تتم إعادة استدعاء النموذج تلقائيًا من هذه المرحلة.",
                "run": recovery,
            },
            status_code=409,
        )
    if action is RecoveryAction.MANUAL_REVIEW:
        return JSONResponse(
            {
                "status": "manual_review",
                "code": "MODEL_EXECUTION_NOT_IDEMPOTENT",
                "message": "لا يمكن إعادة تشغيل مرحلة النموذج تلقائيًا بأمان.",
                "run": recovery,
            },
            status_code=409,
        )
    if action is RecoveryAction.NOOP:
        return JSONResponse({"status": "no_op", "run": recovery})

    worker_id = f"{WORKER_ID}:recovery"
    claimed = await claim_orphaned_run(
        run_id=run_id,
        worker_id=worker_id,
    )
    if claimed is None:
        return JSONResponse(
            {"status": "busy", "code": "RECOVERY_CLAIM_FAILED"},
            status_code=409,
        )

    try:
        message_count = await run_message_count(run_id)
        if stage is RunStage.RESPONSE_READY and message_count == 0:
            await release_orphaned_run(run_id=run_id, worker_id=worker_id)
            return JSONResponse(
                {
                    "status": "retry_new_run",
                    "code": "RESPONSE_NOT_PERSISTED",
                    "message": "لم تُحفظ الرسالة؛ يلزم تنفيذ جديد بدل تكرار استدعاء النموذج.",
                },
                status_code=409,
            )
        if stage is RunStage.RESPONSE_READY:
            await record_run_checkpoint(
                session_id=str(claimed["session_id"]),
                run_id=run_id,
                stage=RunStage.RESPONSE_PERSISTED.value,
                state={"message_count": message_count, "recovered": True},
            )
        await finish_run(run_id=run_id, status="succeeded")
        await record_run_checkpoint(
            session_id=str(claimed["session_id"]),
            run_id=run_id,
            stage=RunStage.COMPLETED.value,
            state={"message_count": message_count, "recovered": True},
        )
    except PersistenceError:
        try:
            await release_orphaned_run(run_id=run_id, worker_id=worker_id)
        except PersistenceError:
            logger.error("Failed to release run after recovery failure")
        try:
            await record_recovery_event(
                session_id=str(claimed["session_id"]),
                run_id=run_id,
                event_type="recovery.failed",
                status="failed",
                safe_metadata={"code": "RECOVERY_PERSISTENCE_ERROR"},
            )
        except PersistenceError:
            logger.error("Failed to record recovery failure event run_id=%s", run_id)
        return JSONResponse(
            {"status": "recovery_failed", "code": "RECOVERY_PERSISTENCE_ERROR"},
            status_code=502,
        )
    return JSONResponse(
        {
            "status": "recovered",
            "message_count": message_count,
            "run_id": run_id,
        }
    )


@server.custom_route("/runs/{run_id}/checkpoints", methods=["GET"])
async def run_checkpoints_route(request: Request) -> Response:
    user, status_code, error_code = await authorize_owner(
        request,
        endpoint="/runs/{run_id}/checkpoints",
    )
    if user is None:
        return JSONResponse(
            {"error": "owner authentication required", "code": error_code},
            status_code=status_code,
        )
    try:
        run_id = _request_uuid(request.path_params.get("run_id"), "run_id")
        raw_limit = request.query_params.get("limit", "50")
        limit = int(raw_limit)
        checkpoints = await load_run_checkpoints(run_id=run_id, limit=limit)
    except (ValueError, PersistenceError):
        return JSONResponse(
            {"error": "تعذر تحميل نقاط تفتيش التنفيذ."},
            status_code=400,
        )
    return JSONResponse({"run_id": run_id, "checkpoints": checkpoints})


async def _execute_approved_action(
    *,
    action_id: str,
    user_id: str,
) -> dict[str, object]:
    claim = await claim_pending_action_execution(
        action_id=action_id,
        user_id=user_id,
    )
    status = claim.get("status")
    if status in {"not_found", "forbidden"}:
        return {"status": status, "executed": False}
    if not claim.get("should_execute"):
        return {"status": status, "executed": status == "executed"}
    if claim.get("tool_name") != "test_sensitive_action":
        return {"status": "unsupported_action", "executed": False}
    # This internal test action intentionally has no external side effect.
    completed = await complete_pending_action(
        action_id=action_id,
        user_id=user_id,
    )
    return {
        "status": completed.get("status", "ERROR"),
        "executed": completed.get("status") == "executed",
        "side_effect": "none",
    }


@server.custom_route("/actions/{action_id}/approve", methods=["POST"])
async def approve_action(request: Request) -> Response:
    user, status_code, error_code = await authorize_owner(
        request,
        endpoint="/actions/approve",
        action_id=_action_id_for_auth_audit(request),
    )
    if user is None:
        return JSONResponse(
            {"error": "owner authentication required", "code": error_code},
            status_code=status_code,
        )
    try:
        action_id = str(UUID(request.path_params["action_id"]))
    except (KeyError, ValueError):
        return JSONResponse({"error": "action_id غير صالح."}, status_code=400)
    try:
        approval = await approve_pending_action(
            action_id=action_id,
            user_id=user.user_id,
        )
        if approval["status"] == "not_found":
            return JSONResponse({"error": "الإجراء غير موجود."}, status_code=404)
        if approval["status"] == "forbidden":
            return JSONResponse({"error": "لا تملك هذا الإجراء."}, status_code=403)
        if approval["status"] in {"expired", "rejected"}:
            return JSONResponse(
                {"status": approval["status"], "executed": False},
                status_code=409,
            )
        execution = await _execute_approved_action(
            action_id=action_id,
            user_id=user.user_id,
        )
    except PersistenceError:
        logger.exception("Action approval persistence failed")
        return JSONResponse(
            {"error": "تعذر معالجة الموافقة الآن."},
            status_code=503,
        )
    if execution["status"] == "unsupported_action":
        return JSONResponse(
            {"error": "الأداة غير مدعومة في مسار التنفيذ."},
            status_code=422,
        )
    return JSONResponse(
        {
            "action_id": action_id,
            "status": execution["status"],
            "executed": execution["executed"],
            **(
                {"side_effect": execution["side_effect"]}
                if "side_effect" in execution
                else {}
            ),
        }
    )


@server.custom_route("/actions/{action_id}/reject", methods=["POST"])
async def reject_action(request: Request) -> Response:
    user, status_code, error_code = await authorize_owner(
        request,
        endpoint="/actions/reject",
        action_id=_action_id_for_auth_audit(request),
    )
    if user is None:
        return JSONResponse(
            {"error": "owner authentication required", "code": error_code},
            status_code=status_code,
        )
    try:
        action_id = str(UUID(request.path_params["action_id"]))
    except (KeyError, ValueError):
        return JSONResponse({"error": "action_id غير صالح."}, status_code=400)
    try:
        rejection = await reject_pending_action(
            action_id=action_id,
            user_id=user.user_id,
        )
    except PersistenceError:
        logger.exception("Action rejection persistence failed")
        return JSONResponse(
            {"error": "تعذر معالجة الرفض الآن."},
            status_code=503,
        )
    if rejection["status"] == "not_found":
        return JSONResponse({"error": "الإجراء غير موجود."}, status_code=404)
    if rejection["status"] == "forbidden":
        return JSONResponse({"error": "لا تملك هذا الإجراء."}, status_code=403)
    return JSONResponse(
        {
            "action_id": action_id,
            "status": rejection["status"],
            "executed": False,
        }
    )


@server.custom_route("/files/upload", methods=["POST"])
async def files_upload(request: Request) -> Response:
    owner, status_code, error_code = await authorize_owner(
        request,
        endpoint="/files/upload",
    )
    if owner is None:
        return JSONResponse(
            {"error": "owner authentication required", "code": error_code},
            status_code=status_code,
        )
    try:
        form = await request.form()
    except Exception:
        return JSONResponse({"error": "صيغة رفع الملف غير صالحة."}, status_code=400)

    uploads = [item for item in form.getlist("file") if isinstance(item, UploadFile)]
    if not uploads:
        return JSONResponse({"error": "يجب إرفاق ملف واحد على الأقل باسم file."}, status_code=400)

    prepared: list[tuple[UploadFile, str, str | None, bytes]] = []
    for upload in uploads:
        raw_filename = upload.filename or ""
        filename = sanitize_filename(raw_filename)
        if not filename or filename != raw_filename:
            return JSONResponse(
                {
                    "error": "اسم الملف غير آمن.",
                    "code": "UNSAFE_FILENAME",
                    "filename": None,
                },
                status_code=415,
            )
        extension = extension_for(filename)
        if not filename or extension not in SUPPORTED_EXTENSIONS:
            return JSONResponse(
                {
                    "error": "نوع الملف غير مدعوم.",
                    "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
                    "filename": filename or None,
                },
                status_code=415,
            )
        data = await upload.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            return JSONResponse(
                {"error": "حجم الملف أكبر من الحد المسموح.", "filename": filename},
                status_code=413,
            )
        prepared.append((upload, filename, upload.content_type, data))

    results: list[dict[str, object]] = []
    try:
        for _, filename, mime_type, data in prepared:
            result = await ingest_document(
                document_id=str(uuid4()),
                filename=filename,
                mime_type=mime_type,
                data=data,
                owner_principal_id=owner.user_id,
            )
            results.append(result)
    except PersistenceError as error:
        logger.error("File persistence failed error_type=%s", type(error).__name__)
        return JSONResponse(
            {"error": "تعذر حفظ الملف الآن. حاول مرة أخرى."},
            status_code=503,
        )

    if any(result.get("status") == "invalid_file_content" for result in results):
        return JSONResponse({"files": results}, status_code=415)
    if any(result.get("duplicate") for result in results):
        return JSONResponse({"files": results}, status_code=409)
    if any(
        result.get("status") not in {"ready", "embedding_failed"}
        for result in results
    ):
        return JSONResponse({"files": results}, status_code=422)
    if any(result.get("status") == "embedding_failed" for result in results):
        return JSONResponse({"files": results}, status_code=202)
    return JSONResponse({"files": results}, status_code=201)


@server.custom_route("/chat/message", methods=["POST"])
async def chat_message(request: Request) -> Response:
    authenticated_user, status_code, error_code = await authorize_owner(
        request,
        endpoint="/chat/message",
    )
    if authenticated_user is None:
        return JSONResponse(
            {"error": "owner authentication required", "code": error_code},
            status_code=status_code,
        )
    try:
        payload = await request.json()
    except ValueError:
        return JSONResponse({"error": "صيغة الطلب غير صالحة."}, status_code=400)

    message = payload.get("message") if isinstance(payload, dict) else None
    if not isinstance(message, str):
        return JSONResponse({"error": "الرسالة مطلوبة."}, status_code=400)

    message = message.strip()
    if not message or len(message) > 2000:
        return JSONResponse(
            {"error": "يجب أن تكون الرسالة بين 1 و2000 حرف."},
            status_code=400,
        )

    raw_history = payload.get("history")
    try:
        message_history = parse_message_history(raw_history)
    except ValueError as error:
        return JSONResponse({"error": str(error)}, status_code=400)

    conversation_id = payload.get("conversation_id", payload.get("session_id"))
    run_id = payload.get("run_id")
    scope = payload.get("scope", "WEB")
    if scope not in {"WEB", "MY_FILES"}:
        return JSONResponse(
            {"error": "scope غير صالح. استخدم WEB أو MY_FILES."},
            status_code=400,
        )
    provider = payload.get("provider", "gemini")
    if not isinstance(provider, str) or provider not in SUPPORTED_PROVIDER_NAMES:
        return JSONResponse(
            {"error": "provider غير صالح. استخدم gemini أو openai."},
            status_code=400,
        )
    try:
        session_id = _request_uuid(conversation_id, "conversation_id")
        run_uuid = _request_uuid(run_id, "run_id")
    except ValueError as error:
        return JSONResponse({"error": str(error)}, status_code=400)

    trace_id = str(uuid4())
    started_at = time.perf_counter()
    run_state: RunState | None = None

    async def save_checkpoint(
        next_stage: RunStage,
        state: dict[str, object],
    ) -> None:
        if run_state is None:
            return
        try:
            run_state.transition(next_stage)
        except ValueError as error:
            logger.error("Invalid run state transition error=%s", error)
            return
        try:
            await record_run_checkpoint(
                session_id=session_id,
                run_id=run_uuid,
                stage=run_state.stage.value,
                state=state,
            )
        except (PersistenceError, ValueError) as error:
            logger.error(
                "Run checkpoint persistence failed stage=%s error_type=%s",
                run_state.stage.value,
                type(error).__name__,
            )

    try:
        await ensure_session(session_id, scope=scope)
        stored_history = await load_message_history(session_id)
        if stored_history:
            message_history = parse_message_history(stored_history)
        else:
            message_history = parse_message_history(raw_history)
        await create_run(
            run_id=run_uuid,
            session_id=session_id,
            user_prompt=message,
            provider_name=provider,
            model_name=provider_model_name(provider),  # type: ignore[arg-type]
            worker_id=WORKER_ID,
        )
        run_state = RunState()
        await save_checkpoint(
            RunStage.CONTEXT_LOADED,
            {
                "scope": scope,
                "provider": provider,
                "history_items": len(message_history or []),
            },
        )
    except (PersistenceError, ValueError) as error:
        logger.error("Agent persistence setup failed error_type=%s", type(error).__name__)
        return JSONResponse(
            {"error": "تعذر الحصول على رد من الوكيل الآن. حاول مرة أخرى."},
            status_code=502,
        )

    async def save_tool_event(
        tool_name: str,
        status: str,
        duration_ms: int,
        safe_metadata: dict[str, object] | None,
    ) -> None:
        try:
            await record_tool_event(
                run_id=run_uuid,
                tool_name=tool_name,
                status=status,
                duration_ms=duration_ms,
                safe_metadata=safe_metadata,
            )
        except PersistenceError as error:
            logger.error("Tool event persistence failed error_type=%s", type(error).__name__)

    lease_heartbeat = asyncio.create_task(_run_lease_heartbeat(run_uuid))

    async def stop_lease_heartbeat() -> None:
        lease_heartbeat.cancel()
        await asyncio.gather(lease_heartbeat, return_exceptions=True)

    try:
        await save_checkpoint(
            RunStage.MODEL_RUNNING,
            {"provider": provider, "scope": scope},
        )
        result = await run_ahmed(
            message,
            message_history=message_history,
            conversation_id=session_id,
            run_id=run_uuid,
            user_id=authenticated_user.user_id if authenticated_user else None,
            scope=scope,
            provider=provider,  # type: ignore[arg-type]
            tool_event_recorder=save_tool_event,
        )
    except AgentCoreError as error:
        await stop_lease_heartbeat()
        provider_status = error.provider_status or error.provider_code or type(error).__name__
        try:
            await finish_run(
                run_id=run_uuid,
                status="failed",
                error_code=provider_status,
            )
        except PersistenceError:
            logger.error("Failed to persist agent failure status")
        _structured_log(
            trace_id=trace_id,
            session_id=session_id,
            run_id=run_uuid,
            provider=provider,
            status="failed",
            latency_ms=int((time.perf_counter() - started_at) * 1000),
            error_code=provider_status,
        )
        await save_checkpoint(
            RunStage.FAILED,
            {"error_code": provider_status, "provider": provider},
        )
        logger.warning(
            "Agent core failed provider=%s status_code=%s provider_status=%s",
            error.provider,
            error.status_code,
            provider_status,
        )
        return JSONResponse(
            (
                {"error": "الخدمة غير متاحة مؤقتًا. حاول لاحقًا."}
                if provider_status == "RATE_LIMITED"
                else {"error": "تعذر الحصول على رد من الوكيل الآن. حاول مرة أخرى."}
            ),
            status_code=503 if provider_status == "RATE_LIMITED" else 502,
        )
    except Exception as error:
        await stop_lease_heartbeat()
        try:
            await finish_run(
                run_id=run_uuid,
                status="failed",
                error_code=type(error).__name__,
            )
        except PersistenceError:
            logger.error("Failed to persist agent failure status")
        _structured_log(
            trace_id=trace_id,
            session_id=session_id,
            run_id=run_uuid,
            provider=provider,
            status="failed",
            latency_ms=int((time.perf_counter() - started_at) * 1000),
            error_code=type(error).__name__,
        )
        await save_checkpoint(
            RunStage.FAILED,
            {"error_code": type(error).__name__, "provider": provider},
        )
        logger.error("Agent core failed exception_type=%s", type(error).__name__)
        return JSONResponse(
            {"error": "تعذر الحصول على رد من الوكيل الآن. حاول مرة أخرى."},
            status_code=502,
        )

    await stop_lease_heartbeat()
    final_text = str(result.output).strip()
    new_messages_json = result.new_messages_json()
    await save_checkpoint(
        RunStage.RESPONSE_READY,
        {"response_bytes": len(new_messages_json), "provider": provider},
    )
    try:
        message_count = await append_new_messages(
            session_id=session_id,
            run_id=run_uuid,
            new_messages_json=new_messages_json,
        )
        await save_checkpoint(
            RunStage.RESPONSE_PERSISTED,
            {"message_count": message_count},
        )
        await finish_run(run_id=run_uuid, status="succeeded")
        await save_checkpoint(
            RunStage.COMPLETED,
            {"message_count": message_count},
        )
    except PersistenceError as error:
        logger.error("Agent result persistence failed error_type=%s", type(error).__name__)
        try:
            await finish_run(
                run_id=run_uuid,
                status="failed",
                error_code=type(error).__name__,
            )
        except PersistenceError:
            logger.error("Failed to persist agent persistence failure status")
        _structured_log(
            trace_id=trace_id,
            session_id=session_id,
            run_id=run_uuid,
            provider=provider,
            status="failed",
            latency_ms=int((time.perf_counter() - started_at) * 1000),
            error_code=type(error).__name__,
        )
        await save_checkpoint(
            RunStage.FAILED,
            {"error_code": type(error).__name__, "provider": provider},
        )
        return JSONResponse(
            {"error": "تعذر الحصول على رد من الوكيل الآن. حاول مرة أخرى."},
            status_code=502,
        )

    _structured_log(
        trace_id=trace_id,
        session_id=session_id,
        run_id=run_uuid,
        provider=provider,
        status="succeeded",
        latency_ms=int((time.perf_counter() - started_at) * 1000),
    )
    logger.info(
        "Agent run completed message_bytes=%d persisted_messages=%d",
        len(new_messages_json),
        message_count,
    )
    if not final_text:
        return JSONResponse(
            {"error": "لم يُرجع النموذج ردًا نصيًا."},
            status_code=502,
        )
    return JSONResponse({"reply": final_text})


if __name__ == "__main__":
    import uvicorn

    mcp_path = "/mcp"
    app = OwnerMCPAuthMiddleware(
        server.streamable_http_app(
            streamable_http_path=mcp_path,
            host="0.0.0.0",
        ),
        path=mcp_path,
    )
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        log_level="info",
    )