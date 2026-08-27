"""Pure authorization decisions for an already-normalized scan plan."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Tuple

import numpy as np

from piper_mobile_manipulation.infrastructure.failure_model import (
    Failure,
    FailureCode,
)
from piper_mobile_manipulation.scan_motion import approval_rejection_reason


def trajectory_count_rejection(
        plan_kind, trajectory_count, viewpoint_count, closed_loop_one_view):
    """Bind trajectory count to the established plan-kind contract."""
    trajectories = int(trajectory_count)
    viewpoints = int(viewpoint_count)
    if trajectories < 0 or viewpoints < 0:
        return 'trajectory and viewpoint counts are invalid'
    if plan_kind == 'RETURN_HOME':
        if trajectories != 1 or viewpoints != 0:
            return (
                'RETURN_HOME must contain one home trajectory and no '
                'viewpoints')
        return ''
    if plan_kind == 'MULTIVIEW_SCAN':
        expected = viewpoints if closed_loop_one_view else viewpoints + 1
        if trajectories != expected:
            return (
                'closed-loop MULTIVIEW_SCAN must contain only its one capture '
                'trajectory' if closed_loop_one_view else
                'MULTIVIEW_SCAN must include one final return-home trajectory')
        return ''
    if trajectories != viewpoints:
        return 'trajectory and viewpoint counts differ'
    return ''


def configured_home_endpoint_rejection(
        home_stage, trajectory_endpoint, declared_goal, rough_home,
        pre_home=None):
    """Validate the exact stage goal without conflating staged targets."""
    stage = str(home_stage).strip().upper()
    allowed = {
        'CONFIGURED_HOME', 'STARTUP_WRIST', 'PRE_HOME', 'ROUGH_HOME',
        'STORAGE_WRIST'}
    if stage not in allowed:
        return 'configured-home stage is invalid'
    try:
        endpoint = np.asarray(trajectory_endpoint, dtype=float)
        goal = np.asarray(declared_goal, dtype=float)
    except (TypeError, ValueError):
        return 'configured-home declared goal is invalid'
    if (
            endpoint.shape != (6,) or goal.shape != (6,)
            or not np.all(np.isfinite(endpoint))
            or not np.all(np.isfinite(goal))):
        return 'configured-home declared goal must contain six finite joints'
    if float(np.max(np.abs(endpoint - goal))) > 1e-9:
        return (
            'configured-home declared goal does not match the trajectory '
            'endpoint')
    if stage in ('CONFIGURED_HOME', 'ROUGH_HOME'):
        try:
            configured = np.asarray(rough_home, dtype=float)
        except (TypeError, ValueError):
            return 'configured return-home pose is invalid'
        if configured.shape != (6,) or not np.all(np.isfinite(configured)):
            return 'configured return-home pose must contain six finite joints'
        if float(np.max(np.abs(endpoint - configured))) > 1e-6:
            return (
                'Tesseract return-home endpoint does not match the executor '
                'configuration')
    if stage == 'PRE_HOME':
        try:
            configured = np.asarray(pre_home, dtype=float)
        except (TypeError, ValueError):
            return 'configured pre-home pose is invalid'
        if configured.shape != (6,) or not np.all(np.isfinite(configured)):
            return 'configured pre-home pose must contain six finite joints'
        if float(np.max(np.abs(endpoint - configured))) > 1e-6:
            return 'pre-home endpoint does not match the executor configuration'
    return ''


def direct_home_stage_rejection(
        home_stage, requested_goal, current_joints, rough_home, joint_limits,
        unchanged_tolerance_rad=0.025,
        start_limit_tolerance_rad=0.3, pre_home=None):
    """Validate one configured direct-home endpoint without path planning."""
    stage = str(home_stage).strip().upper()
    try:
        goal = np.asarray(requested_goal, dtype=float)
        current = np.asarray(current_joints, dtype=float)
        configured = np.asarray(rough_home, dtype=float)
        limits = np.asarray(joint_limits, dtype=float)
    except (TypeError, ValueError):
        return 'direct home request contains invalid joint values'
    endpoint_reason = configured_home_endpoint_rejection(
        stage, goal, goal, rough_home, pre_home=pre_home)
    if endpoint_reason:
        return endpoint_reason.replace('Tesseract ', '')
    if current.shape != (6,) or not np.all(np.isfinite(current)):
        return 'direct home requires six finite current joints'
    if configured.shape != (6,) or not np.all(np.isfinite(configured)):
        return 'direct home configured rough-home target is invalid'
    if limits.shape != (6, 2) or not np.all(np.isfinite(limits)):
        return 'direct home joint limits are invalid'
    if stage == 'STARTUP_WRIST':
        if float(np.max(np.abs(goal[:5] - current[:5]))) > float(
                unchanged_tolerance_rad):
            return 'STARTUP_WRIST may change only J6 from current feedback'
        if goal[5] < current[5] - 1e-6:
            return 'STARTUP_WRIST must move J6 only in the positive direction'
    if stage == 'STORAGE_WRIST':
        if float(np.max(np.abs(goal[:5] - configured[:5]))) > 1e-6:
            return 'STORAGE_WRIST must preserve configured rough-home J1-J5'
        if goal[5] > current[5] + 1e-6:
            return 'STORAGE_WRIST must move J6 only in the negative direction'
    for index, (value, bounds) in enumerate(zip(goal, limits)):
        low, high = [float(item) for item in bounds]
        if low <= value <= high:
            continue
        unchanged_powered_start = bool(
            stage == 'STARTUP_WRIST'
            and index < 5
            and abs(value - current[index]) <= float(
                unchanged_tolerance_rad)
            and value >= low - float(start_limit_tolerance_rad)
            and value <= high + float(start_limit_tolerance_rad))
        if not unchanged_powered_start:
            return 'direct home joint%d target is outside limits' % (index + 1)
    return ''


def direct_home_stage_targets(home_stage, current_joints, requested_goal):
    """Return measured-gated endpoints, including the startup wrap bridge."""
    stage = str(home_stage).strip().upper()
    current = np.asarray(current_joints, dtype=float)
    goal = np.asarray(requested_goal, dtype=float)
    targets = []
    if (
            stage == 'STARTUP_WRIST'
            and current.shape == (6,)
            and np.all(np.isfinite(current))
            and current[5] < -math.pi - 1e-6):
        bridge = current.copy()
        bridge[5] = 3.2 - 2.0 * math.pi
        targets.append(bridge)
    targets.append(goal.copy())
    return targets


class PlanAuthorizationStatus(str, Enum):
    """Machine-readable outcome of exact plan authorization."""

    AUTHORIZED = 'AUTHORIZED'
    WRONG_MISSION_AUTHORIZATION = 'WRONG_MISSION_AUTHORIZATION'
    STALE_PLAN = 'STALE_PLAN'
    INVALID_PLAN = 'INVALID_PLAN'
    TARGET_STALE = 'TARGET_STALE'
    TARGET_DRIFT = 'TARGET_DRIFT'
    DEPENDENCY_UNAVAILABLE = 'DEPENDENCY_UNAVAILABLE'
    PLANNER_FAILURE = 'PLANNER_FAILURE'
    PATH_INVALID = 'PATH_INVALID'


@dataclass(frozen=True)
class PlanAuthorizationRequest:
    """Frozen evidence needed to authorize one exact proposal."""

    state: str
    loaded_plan_id: str
    requested_plan_id: str
    confirmation: str
    expected_confirmation: str
    real_motion_enabled: bool
    plan_age_sec: float
    plan_max_age_sec: float
    loaded_trajectory_sha256: str
    requested_trajectory_sha256: str
    mission_authorization_required: bool = False
    mission_authorization_granted: bool = True
    target_required: bool = False
    target_available: bool = True
    target_drift_m: Optional[float] = None
    maximum_target_drift_m: float = 0.0
    allow_target_motion: bool = False
    unavailable_dependencies: Tuple[str, ...] = ()
    path_reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanAuthorizationDecision:
    """Authorization result with stable status and compatible explanation."""

    permitted: bool
    status: PlanAuthorizationStatus
    failure: Optional[Failure] = None

    @property
    def detail(self) -> str:
        """Return the existing public explanation at the ROS boundary."""
        return '' if self.failure is None else self.failure.detail


class PlanAuthorizer:
    """Evaluate exact plan ownership without ROS or mutable node state."""

    @staticmethod
    def planner_result(
            valid: bool, reason: str) -> PlanAuthorizationDecision:
        """Classify a command-free planner result before plan loading."""
        if bool(valid):
            return PlanAuthorizationDecision(
                True, PlanAuthorizationStatus.AUTHORIZED)
        detail = 'Tesseract proposal rejected: ' + str(reason)
        return PlanAuthorizationDecision(
            False,
            PlanAuthorizationStatus.PLANNER_FAILURE,
            Failure(FailureCode.NO_REACHABLE_PLAN, detail),
        )

    @staticmethod
    def evaluate(
            request: PlanAuthorizationRequest) -> PlanAuthorizationDecision:
        """Return the first rejection in the executor's established order."""
        if (
                request.mission_authorization_required
                and not request.mission_authorization_granted):
            detail = (
                'autonomous execution is not bound to a live mission '
                'authorization')
            return PlanAuthorizationDecision(
                False,
                PlanAuthorizationStatus.WRONG_MISSION_AUTHORIZATION,
                Failure(FailureCode.CONTROL_UNTRUSTWORTHY, detail),
            )

        detail = approval_rejection_reason(
            request.state,
            request.loaded_plan_id,
            request.requested_plan_id,
            request.confirmation,
            request.expected_confirmation,
            request.real_motion_enabled,
            request.plan_age_sec,
            request.plan_max_age_sec,
            current_trajectory_sha256=request.loaded_trajectory_sha256,
            requested_trajectory_sha256=request.requested_trajectory_sha256,
            require_trajectory_hash=True,
        )
        if detail:
            status = (
                PlanAuthorizationStatus.STALE_PLAN
                if request.plan_age_sec > request.plan_max_age_sec
                else PlanAuthorizationStatus.INVALID_PLAN)
            return PlanAuthorizationDecision(
                False, status, Failure(FailureCode.NO_REACHABLE_PLAN, detail))

        if request.target_required and not request.target_available:
            detail = 'latest target center is unavailable'
            return PlanAuthorizationDecision(
                False,
                PlanAuthorizationStatus.TARGET_STALE,
                Failure(FailureCode.TARGET_NOT_FOUND, detail),
            )
        if (
                request.target_required
                and request.target_drift_m is not None
                and not request.allow_target_motion
                and request.target_drift_m
                > request.maximum_target_drift_m):
            detail = (
                'target moved %.3fm after planning; refresh the plan'
                % float(request.target_drift_m))
            return PlanAuthorizationDecision(
                False,
                PlanAuthorizationStatus.TARGET_DRIFT,
                Failure(FailureCode.NO_REACHABLE_PLAN, detail),
            )
        if request.unavailable_dependencies:
            detail = '%s is not ready' % request.unavailable_dependencies[0]
            return PlanAuthorizationDecision(
                False,
                PlanAuthorizationStatus.DEPENDENCY_UNAVAILABLE,
                Failure(FailureCode.SENSOR_UNAVAILABLE, detail),
            )
        if request.path_reasons:
            detail = 'fresh trajectory validation failed: ' + '; '.join(
                request.path_reasons)
            return PlanAuthorizationDecision(
                False,
                PlanAuthorizationStatus.PATH_INVALID,
                Failure(FailureCode.NO_REACHABLE_PLAN, detail),
            )
        return PlanAuthorizationDecision(
            True, PlanAuthorizationStatus.AUTHORIZED)


def target_drift_before_approval_rejection(
        drift_m, maximum_m, allow_target_motion):
    """Retain the existing helper boundary for downstream imports."""
    if bool(allow_target_motion):
        return ''
    if float(drift_m) > float(maximum_m):
        return 'target moved %.3fm after planning; refresh the plan' % float(
            drift_m)
    return ''
