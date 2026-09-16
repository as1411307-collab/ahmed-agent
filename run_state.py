from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RunStage(StrEnum):
    CREATED = "created"
    CONTEXT_LOADED = "context_loaded"
    MODEL_RUNNING = "model_running"
    RESPONSE_READY = "response_ready"
    RESPONSE_PERSISTED = "response_persisted"
    COMPLETED = "completed"
    FAILED = "failed"


class RecoveryAction(StrEnum):
    COMPLETE_PERSISTED_TAIL = "complete_persisted_tail"
    RETRY_NEW_RUN = "retry_new_run"
    MANUAL_REVIEW = "manual_review"
    NOOP = "noop"


_ALLOWED_TRANSITIONS: dict[RunStage, frozenset[RunStage]] = {
    RunStage.CREATED: frozenset({RunStage.CONTEXT_LOADED, RunStage.FAILED}),
    RunStage.CONTEXT_LOADED: frozenset({RunStage.MODEL_RUNNING, RunStage.FAILED}),
    RunStage.MODEL_RUNNING: frozenset({RunStage.RESPONSE_READY, RunStage.FAILED}),
    RunStage.RESPONSE_READY: frozenset({RunStage.RESPONSE_PERSISTED, RunStage.FAILED}),
    RunStage.RESPONSE_PERSISTED: frozenset({RunStage.COMPLETED, RunStage.FAILED}),
    RunStage.COMPLETED: frozenset(),
    RunStage.FAILED: frozenset(),
}


def recovery_action(stage: RunStage) -> RecoveryAction:
    if stage in {RunStage.RESPONSE_READY, RunStage.RESPONSE_PERSISTED}:
        return RecoveryAction.COMPLETE_PERSISTED_TAIL
    if stage in {RunStage.CREATED, RunStage.CONTEXT_LOADED}:
        return RecoveryAction.RETRY_NEW_RUN
    if stage is RunStage.MODEL_RUNNING:
        return RecoveryAction.MANUAL_REVIEW
    return RecoveryAction.NOOP


@dataclass
class RunState:
    stage: RunStage = RunStage.CREATED

    def transition(self, next_stage: RunStage) -> RunStage:
        if next_stage not in _ALLOWED_TRANSITIONS[self.stage]:
            raise ValueError(
                f"invalid run transition: {self.stage.value} -> {next_stage.value}"
            )
        self.stage = next_stage
        return self.stage