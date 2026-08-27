"""Typed recovery choices for viewpoint execution failures."""

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple

from piper_mobile_manipulation.infrastructure.failure_model import (
    as_failure,
    Failure,
    FailureCode,
    FailureTag,
)


class RecoveryContext(str, Enum):
    """Executor context in which a failure occurred."""

    RUNTIME = 'RUNTIME'
    ACQUISITION = 'ACQUISITION'
    PLANNING = 'PLANNING'
    TRAJECTORY = 'TRAJECTORY'
    CAPTURE = 'CAPTURE'


class RecoveryAction(str, Enum):
    """Explicit recovery outcome."""

    CONTINUE = 'continue'
    RETRY = 'retry'
    REACQUIRE = 'reacquire'
    REPLAN = 'replan'
    ABORT = 'abort'


@dataclass(frozen=True)
class RecoveryDecision:
    """Recovery action with typed supporting failures."""

    action: RecoveryAction
    failures: Tuple[Failure, ...] = ()


class RecoveryPolicy:
    """Choose recovery from typed failure codes/tags, never detail text."""

    @staticmethod
    def decide(
            context: RecoveryContext,
            failures: Iterable[Failure]) -> RecoveryDecision:
        """Apply the executor's existing recovery precedence."""
        typed = tuple(failures)
        if any(not isinstance(item, Failure) for item in typed):
            raise TypeError('recovery policy requires typed failures')
        if not typed:
            return RecoveryDecision(RecoveryAction.CONTINUE)
        if any(item.code is FailureCode.CANCELLED for item in typed):
            return RecoveryDecision(RecoveryAction.ABORT, typed)
        if context is RecoveryContext.RUNTIME:
            if all(item.has(FailureTag.RUNTIME_FRESHNESS_GAP)
                   for item in typed):
                return RecoveryDecision(RecoveryAction.RETRY, typed)
            return RecoveryDecision(RecoveryAction.ABORT, typed)
        if context is RecoveryContext.CAPTURE:
            if any(item.has(FailureTag.CAPTURE_RETRY_SAME_VIEW)
                   for item in typed):
                return RecoveryDecision(RecoveryAction.RETRY, typed)
            if any(item.has(FailureTag.CAPTURE_REJECT_VIEW)
                   for item in typed):
                return RecoveryDecision(RecoveryAction.REPLAN, typed)
            return RecoveryDecision(RecoveryAction.ABORT, typed)
        if context in (RecoveryContext.ACQUISITION, RecoveryContext.PLANNING):
            if any(item.has(
                    FailureTag.PLAN_APPROVAL_VISUAL_REACQUISITION)
                    or item.has(FailureTag.PLAN_REQUEST_VISUAL_REACQUISITION)
                    for item in typed):
                return RecoveryDecision(RecoveryAction.REACQUIRE, typed)
            if any(item.has(FailureTag.TARGET_DRIFT_REPLAN)
                   for item in typed):
                return RecoveryDecision(RecoveryAction.REPLAN, typed)
        return RecoveryDecision(RecoveryAction.ABORT, typed)


def runtime_gate_action(failures):
    """Compatibility wrapper returning the historical runtime action."""
    decision = RecoveryPolicy.decide(
        RecoveryContext.RUNTIME,
        tuple(as_failure(item) for item in failures),
    )
    if decision.action is RecoveryAction.RETRY:
        return 'hold_for_refresh'
    return decision.action.value


def runtime_refresh_action(reasons, elapsed_sec, timeout_sec):
    """Preserve bounded pre-motion refresh action ordering."""
    if not reasons:
        return 'start'
    if float(elapsed_sec) >= float(timeout_sec):
        return 'abort'
    return 'wait'
