"""Pure settled-capture sequencing and retry decisions."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from piper_mobile_manipulation.failure_model import (
    as_failure,
    Failure,
    FailureTag,
)


class CaptureAction(str, Enum):
    """Next action for the executor's ROS capture adapter."""

    PUBLISH_AUTHORIZATION = 'publish_authorization'
    REQUEST_CAPTURE = 'request_capture'
    WAIT_RESPONSE = 'wait_response'
    START_SETTLE_WINDOW = 'start_settle_window'
    READY = 'ready'
    ACCEPT = 'accept'
    RETRY_SAME_VIEW = 'retry_same_view'
    REPLAN_VIEW = 'replan_view'
    ABORT = 'abort'


@dataclass(frozen=True)
class CaptureDecision:
    """Typed capture decision independent of service response wording."""

    action: CaptureAction
    failure: Optional[Failure] = None
    reset_settle_window: bool = False


class CaptureCoordinator:
    """Coordinate established capture propagation and retry policy."""

    def __init__(self, maximum_readiness_retries):
        """Bind the unchanged same-view readiness retry limit."""
        self.maximum_readiness_retries = int(maximum_readiness_retries)
        if self.maximum_readiness_retries < 1:
            raise ValueError(
                'maximum capture readiness retries must be positive')

    @staticmethod
    def settle(
            state_age_sec, settled, settle_started_at, now_sec,
            settle_duration_sec, settle_timeout_sec):
        """Apply the established continuous settled-window requirement."""
        if float(state_age_sec) > float(settle_timeout_sec):
            return CaptureDecision(CaptureAction.ABORT)
        if not bool(settled):
            return CaptureDecision(
                CaptureAction.WAIT_RESPONSE,
                reset_settle_window=True,
            )
        if settle_started_at is None:
            return CaptureDecision(CaptureAction.START_SETTLE_WINDOW)
        if (
                float(now_sec) - float(settle_started_at)
                < float(settle_duration_sec)):
            return CaptureDecision(CaptureAction.WAIT_RESPONSE)
        return CaptureDecision(CaptureAction.READY)

    @staticmethod
    def handoff(request_inflight, state_age_sec, propagation_sec):
        """Sequence status delivery before the one RGB-D service request."""
        if bool(request_inflight):
            return CaptureDecision(CaptureAction.WAIT_RESPONSE)
        if float(state_age_sec) < max(0.0, float(propagation_sec)):
            return CaptureDecision(CaptureAction.PUBLISH_AUTHORIZATION)
        return CaptureDecision(CaptureAction.REQUEST_CAPTURE)

    def classify_result(self, success, failure, attempts):
        """Classify a typed service result using the existing retry bound."""
        if bool(success):
            return CaptureDecision(CaptureAction.ACCEPT)
        if not isinstance(failure, Failure):
            raise TypeError(
                'capture failure must be typed before coordination')
        if (
                failure.has(FailureTag.CAPTURE_RETRY_SAME_VIEW)
                and int(attempts) < self.maximum_readiness_retries):
            return CaptureDecision(CaptureAction.RETRY_SAME_VIEW, failure)
        if failure.has(FailureTag.CAPTURE_REJECT_VIEW):
            return CaptureDecision(CaptureAction.REPLAN_VIEW, failure)
        return CaptureDecision(CaptureAction.ABORT, failure)


def rgbd_capture_handoff_action(
        request_inflight, state_age_sec, propagation_sec):
    """Compatibility wrapper returning the historical string action."""
    return CaptureCoordinator.handoff(
        request_inflight, state_age_sec, propagation_sec).action.value


def retryable_rgbd_capture_rejection(failure):
    """Return the typed same-view retry policy."""
    return as_failure(failure).has(FailureTag.CAPTURE_RETRY_SAME_VIEW)


def visual_capture_rejection(failure):
    """Return the typed replacement-view policy."""
    return as_failure(failure).has(FailureTag.CAPTURE_REJECT_VIEW)
