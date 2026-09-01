"""Backend-neutral motion-planner selection and application contracts."""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class PlannerBackend(str, Enum):
    """Supported motion-planning implementations."""

    TESSERACT = 'tesseract'
    CUROBO = 'curobo'


def parse_planner_backend(value):
    """Return a supported backend or fail closed on an unknown value."""
    normalized = str(value).strip().lower()
    try:
        return PlannerBackend(normalized)
    except ValueError as error:
        raise ValueError(
            'planner_backend must be exactly tesseract or curobo') from error


@dataclass(frozen=True)
class PlannerSelection:
    """Planner choice frozen at a mission/process-generation boundary."""

    backend: PlannerBackend
    worker_python: str = ''


@dataclass(frozen=True)
class PlannerReadiness:
    """ROS-independent planner readiness snapshot."""

    backend: PlannerBackend
    backend_version: str
    generation_id: str
    worker_ready: bool
    acquisition_ready: bool
    multiview_ready: bool
    manipulation_ready: bool
    acquisition_blockers: Tuple[str, ...] = ()
    multiview_blockers: Tuple[str, ...] = ()
    manipulation_blockers: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MotionPlanRequest:
    """Backend-neutral identity for one planning transaction."""

    session_id: str
    plan_kind: str
    force_refresh: bool = False
    home_stage: str = ''
    joint_goal_positions_rad: Tuple[float, ...] = ()


@dataclass(frozen=True)
class MotionPlanResult:
    """Backend-neutral result identity before ROS transport conversion."""

    backend: PlannerBackend
    backend_version: str
    request_id: str
    plan_id: str
    plan_kind: str
    valid: bool
    collision_model_qualified: bool
    rejection_codes: Tuple[str, ...] = ()
    reason: str = ''
