"""Tk-independent presentation state for the production scan action."""

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Optional, Sequence, Tuple


class MissionUiPhase(str, Enum):
    IDLE = "IDLE"
    SUBMITTING = "SUBMITTING"
    ACTIVE = "ACTIVE"
    CANCELLING = "CANCELLING"


@dataclass(frozen=True)
class MissionRequest:
    coordinates: Tuple[float, float, float]
    target_label: str


@dataclass(frozen=True)
class MissionFeedbackView:
    phase: str
    reason: str
    accepted_captures: int
    required_captures: int


@dataclass(frozen=True)
class MissionResultView:
    task_id: str
    outcome: str
    reason: str
    failure_code: str
    retryable: bool
    safe_shutdown: bool
    capture_count: int
    dataset_path: str
    manifest_sha256: str
    mesh_job_id: str

    @property
    def reconstruction_payload(self):
        if (
                self.outcome != "SUCCEEDED"
                or not self.safe_shutdown
                or not self.mesh_job_id):
            return None
        return {
            "task_id": self.task_id,
            "outcome": self.outcome,
            "safe_shutdown": self.safe_shutdown,
            "dataset_path": self.dataset_path,
            "manifest_sha256": self.manifest_sha256,
            "mesh_job_id": self.mesh_job_id,
        }


@dataclass(frozen=True)
class MissionUiState:
    phase: MissionUiPhase = MissionUiPhase.IDLE
    status: str = "Autonomous mission idle"
    feedback_phase: str = ""
    last_result: Optional[MissionResultView] = None

    @property
    def can_start(self):
        return self.phase == MissionUiPhase.IDLE

    @property
    def can_cancel(self):
        return self.phase == MissionUiPhase.ACTIVE


def validate_mission_request(
        coordinates: Sequence[object], target_label: object) -> MissionRequest:
    try:
        values = tuple(float(value) for value in coordinates)
    except (TypeError, ValueError):
        raise ValueError("rough target XYZ must be numeric")
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("rough target XYZ must contain three finite values")
    label = str(target_label).strip() or "green cube"
    return MissionRequest(values, label)


class MissionViewModel:
    """
    Own only operator-facing action lifecycle state.

    This class never selects a mission phase, retry, plan, motion, shutdown or
    safety outcome.  It reflects events emitted by the production action.
    """

    def __init__(self):
        self._state = MissionUiState()

    @property
    def state(self):
        return self._state

    def begin_submission(self, coordinates, target_label):
        if not self._state.can_start:
            raise RuntimeError("an automatic mission is already starting or active")
        request = validate_mission_request(coordinates, target_label)
        self._state = replace(
            self._state,
            phase=MissionUiPhase.SUBMITTING,
            status="submitting complete task through /piper/run_target_scan",
            feedback_phase="",
            last_result=None,
        )
        return request

    def goal_accepted(self):
        self._state = replace(
            self._state,
            phase=MissionUiPhase.ACTIVE,
            status="mission accepted; automatic startup is beginning",
        )

    def submission_failed(self, message):
        self._state = replace(
            self._state,
            phase=MissionUiPhase.IDLE,
            status=str(message),
        )

    def cancellation_requested(self):
        if self._state.phase not in (
                MissionUiPhase.SUBMITTING, MissionUiPhase.ACTIVE):
            return False
        self._state = replace(
            self._state,
            phase=MissionUiPhase.CANCELLING,
            status="mission cancellation requested; awaiting production shutdown",
        )
        return True

    def apply_feedback(self, feedback):
        view = MissionFeedbackView(
            phase=str(feedback.phase),
            reason=str(feedback.reason),
            accepted_captures=int(feedback.accepted_captures),
            required_captures=int(feedback.required_captures),
        )
        self._state = replace(
            self._state,
            feedback_phase=view.phase,
            status="%s: %s (%d accepted; model seed floor %d)" % (
                view.phase,
                view.reason,
                view.accepted_captures,
                view.required_captures,
            ),
        )
        return view

    def apply_result(self, result):
        self._state = replace(
            self._state,
            phase=MissionUiPhase.IDLE,
            status=format_result(result),
            last_result=result,
        )


def format_result(result):
    return ("%s: %s; code=%s; retryable=%s; safe shutdown=%s; "
            "captures=%d; dataset=%s") % (
        result.outcome,
        result.reason,
        result.failure_code or "none",
        "yes" if result.retryable else "no",
        "proved" if result.safe_shutdown else "not proved",
        result.capture_count,
        result.dataset_path or "unavailable",
    )
