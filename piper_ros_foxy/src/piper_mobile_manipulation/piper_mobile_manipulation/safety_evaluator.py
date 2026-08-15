"""
Pure shadow evaluation of executor safety and readiness evidence.

The legacy executor remains authoritative in Phase 5.  This module names the
operating contexts hidden behind its boolean arguments, evaluates one immutable
telemetry snapshot, and records structured comparisons without commanding ROS
or reading node parameters.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
import json
import math
import threading
from typing import Any, Deque, Dict, Optional, Sequence, Tuple

from piper_mobile_manipulation.failure_model import (
    as_failure,
    Failure,
    FailureCode,
    FailureTag,
)
from piper_mobile_manipulation.telemetry_store import (
    TelemetryObservation,
    TelemetrySnapshot,
)


class SafetyMode(str, Enum):
    """Named executor contexts with distinct existing evidence policy."""

    PLAN_VALIDATION = 'PLAN_VALIDATION'
    ACQUISITION_APPROVAL = 'ACQUISITION_APPROVAL'
    ACQUISITION_MOTION = 'ACQUISITION_MOTION'
    SCAN_APPROVAL = 'SCAN_APPROVAL'
    SCAN_MOTION = 'SCAN_MOTION'
    SCAN_CAPTURE = 'SCAN_CAPTURE'
    RETURN_HOME = 'RETURN_HOME'
    HOLD_CURRENT = 'HOLD_CURRENT'


@dataclass(frozen=True)
class SafetyProfile:
    """Thresholds copied unchanged from executor parameters at construction."""

    data_timeout_sec: float
    motion_limits_timeout_sec: float
    max_tracking_measurement_age_sec: float
    min_tracking_speed_scale: float
    configured_speed_percent: float
    max_target_drift_before_approval_m: float
    joint_feedback_limit_tolerance_rad: float
    configured_home_feedback_limit_tolerance_rad: float
    hold_joint_feedback_timeout_sec: float


@dataclass(frozen=True)
class SafetyAuthorization:
    """Explicit mission authorization evidence supplied by the caller."""

    required: bool = False
    granted: bool = True


@dataclass(frozen=True)
class SafetyInputs:
    """Non-telemetry facts already computed by existing authoritative code."""

    planner_result_valid: bool = True
    plan_schema_valid: bool = True
    collision_model_qualified: bool = True
    path_valid: bool = True
    motion_limits_compatible: bool = True
    joints_settled: Optional[bool] = None
    target_drift_m: Optional[float] = None
    allow_target_motion: bool = False
    approved_obstacle_snapshot: bool = False
    static_obstacle_scene_authorized: bool = False
    auto_capture: bool = True
    workflow_required: bool = False
    workflow_ready: Optional[bool] = None
    plan_execution_speed_percent: float = 0.0
    configured_home_direct: bool = False
    motion_control_authorized: bool = True
    capture_services_ready: bool = True
    joint_limits: Tuple[Tuple[float, float], ...] = ()


@dataclass(frozen=True)
class SafetyEvidence:
    """One stable rule result suitable for logs and later equivalence review."""

    rule: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class SafetyDecision:
    """Immutable result from one pure shadow evaluation."""

    mode: SafetyMode
    permitted: bool
    failure: Optional[Failure]
    explanation: str
    reasons: Tuple[str, ...]
    evidence: Tuple[SafetyEvidence, ...]
    telemetry_ages: Tuple[Tuple[str, Optional[float]], ...]
    replan_required: bool = False
    reacquisition_required: bool = False

    @property
    def failure_code(self) -> Optional[FailureCode]:
        return None if self.failure is None else self.failure.code


@dataclass(frozen=True)
class SafetyComparison:
    """Structured legacy-versus-shadow observation; never a command input."""

    sequence: int
    context: str
    mode: SafetyMode
    legacy_permitted: bool
    shadow_permitted: bool
    legacy_failure_code: Optional[FailureCode]
    shadow_failure_code: Optional[FailureCode]
    permission_agreement: bool
    failure_code_agreement: bool
    reason_text_agreement: bool
    agreement: bool
    legacy_reasons: Tuple[str, ...]
    shadow_reasons: Tuple[str, ...]
    telemetry_ages: Tuple[Tuple[str, Optional[float]], ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            'sequence': self.sequence,
            'context': self.context,
            'mode': self.mode.value,
            'legacy_permitted': self.legacy_permitted,
            'shadow_permitted': self.shadow_permitted,
            'legacy_failure_code': (
                None if self.legacy_failure_code is None
                else self.legacy_failure_code.value),
            'shadow_failure_code': (
                None if self.shadow_failure_code is None
                else self.shadow_failure_code.value),
            'permission_agreement': self.permission_agreement,
            'failure_code_agreement': self.failure_code_agreement,
            'reason_text_agreement': self.reason_text_agreement,
            'agreement': self.agreement,
            'legacy_reasons': list(self.legacy_reasons),
            'shadow_reasons': list(self.shadow_reasons),
            'telemetry_ages': dict(self.telemetry_ages),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True)


class SafetyComparisonLogger:
    """Bounded thread-safe store for structured shadow comparisons."""

    def __init__(self, maximum_records: int = 256):
        if int(maximum_records) < 1:
            raise ValueError('maximum_records must be positive')
        self._records = deque(maxlen=int(maximum_records))  # type: Deque[SafetyComparison]
        self._lock = threading.RLock()
        self._sequence = 0

    def record(
            self, context: str, legacy_reasons: Sequence[str],
            shadow: SafetyDecision) -> SafetyComparison:
        legacy = tuple(str(item) for item in legacy_reasons)
        legacy_permitted = not legacy
        legacy_failure = (
            None if legacy_permitted else as_failure('; '.join(legacy)))
        legacy_code = (
            None if legacy_failure is None else legacy_failure.code)
        permission_agreement = legacy_permitted == shadow.permitted
        failure_code_agreement = legacy_code == shadow.failure_code
        reason_text_agreement = legacy == shadow.reasons
        with self._lock:
            self._sequence += 1
            comparison = SafetyComparison(
                sequence=self._sequence,
                context=str(context),
                mode=shadow.mode,
                legacy_permitted=legacy_permitted,
                shadow_permitted=shadow.permitted,
                legacy_failure_code=legacy_code,
                shadow_failure_code=shadow.failure_code,
                permission_agreement=permission_agreement,
                failure_code_agreement=failure_code_agreement,
                reason_text_agreement=reason_text_agreement,
                agreement=(permission_agreement and failure_code_agreement),
                legacy_reasons=legacy,
                shadow_reasons=shadow.reasons,
                telemetry_ages=shadow.telemetry_ages,
            )
            self._records.append(comparison)
        return comparison

    def records(self) -> Tuple[SafetyComparison, ...]:
        with self._lock:
            return tuple(self._records)

    def summary(self) -> Dict[str, int]:
        records = self.records()
        disagreements = sum(not item.agreement for item in records)
        return {
            'comparisons': len(records),
            'agreements': len(records) - disagreements,
            'disagreements': disagreements,
        }


class SafetyEvaluator:
    """Evaluate existing executor rules from explicit inputs and a snapshot."""

    _TELEMETRY_FIELDS = (
        'joints', 'arm_status', 'motion_limits', 'camera_clock',
        'tracking', 'target_status', 'obstacles', 'workflow')

    def __init__(self, profile: SafetyProfile):
        self.profile = profile

    @staticmethod
    def _observation(
            snapshot: TelemetrySnapshot,
            key: str) -> Optional[TelemetryObservation[Any]]:
        if key == 'joints':
            return snapshot.arm.joints
        if key == 'arm_status':
            return snapshot.arm.status
        if key == 'motion_limits':
            return snapshot.arm.motion_limits
        if key == 'camera_clock':
            return snapshot.perception.camera
        if key == 'tracking':
            return snapshot.perception.tracking
        if key == 'target_status':
            return snapshot.perception.target_status
        if key == 'obstacles':
            return snapshot.perception.obstacles
        if key == 'workflow':
            return snapshot.mission.workflow
        return None

    @staticmethod
    def _required_fields(mode: SafetyMode) -> Tuple[str, ...]:
        if mode == SafetyMode.PLAN_VALIDATION:
            return ('motion_limits',)
        if mode == SafetyMode.HOLD_CURRENT:
            return ('joints',)
        if mode == SafetyMode.RETURN_HOME:
            return ('joints', 'arm_status', 'motion_limits')
        if mode in (
                SafetyMode.ACQUISITION_APPROVAL,
                SafetyMode.ACQUISITION_MOTION):
            return ('joints', 'arm_status', 'camera_clock', 'motion_limits')
        if mode == SafetyMode.SCAN_APPROVAL:
            return (
                'joints', 'arm_status', 'camera_clock', 'obstacles',
                'tracking', 'target_status', 'motion_limits')
        return ('joints', 'arm_status', 'camera_clock', 'motion_limits')

    @staticmethod
    def _failure_for_rule(rule: str, detail: str) -> Failure:
        if rule.startswith(('plan.', 'path.', 'target.drift')):
            code = FailureCode.NO_REACHABLE_PLAN
        elif rule.startswith('target.') or rule.startswith('tracking.'):
            code = FailureCode.TARGET_NOT_FOUND
        elif rule.startswith(('joints.', 'arm.', 'motor.', 'limits.',
                              'authorization.')):
            code = FailureCode.CONTROL_UNTRUSTWORTHY
        elif rule.startswith(('camera.', 'obstacles.', 'workflow.')):
            code = FailureCode.SENSOR_UNAVAILABLE
        else:
            code = FailureCode.MISSION_FAILED
        tags = frozenset({FailureTag.TARGET_DRIFT_REPLAN}) \
            if rule == 'target.drift' else frozenset()
        return Failure(code=code, detail=detail, tags=tags)

    @staticmethod
    def _joint_values(observation: TelemetryObservation[Any]) -> Optional[Tuple[float, ...]]:
        positions = getattr(observation.value, 'position', ())
        if len(positions) < 6:
            return None
        try:
            values = tuple(float(value) for value in positions[:6])
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in values):
            return None
        return values

    @staticmethod
    def _motor_reasons(status: Any) -> Tuple[Tuple[str, str], ...]:
        results = []
        if not bool(getattr(status, 'motor_feedback_valid', False)):
            results.append(('motor.feedback', 'low-speed motor feedback is invalid'))
        states = tuple(bool(getattr(
            status, 'motor_%d_driver_enabled' % index, False))
            for index in range(1, 7))
        faults = tuple(str(value) for value in getattr(
            status, 'motor_faults', ()) if str(value))
        watchdog = str(getattr(status, 'motor_watchdog_reason', '')).strip()
        if any(states) and not all(states):
            results.append((
                'motor.partial_enable',
                'partial motor enable flags=%s' % (states,)))
        if faults:
            results.append(('motor.faults', 'motor faults=%s' % ','.join(faults)))
        if watchdog:
            results.append(('motor.watchdog', 'motor watchdog=%s' % watchdog))
        if not all(states):
            results.append((
                'motor.enabled', 'all six motor drivers are not enabled'))
        return tuple(results)

    @staticmethod
    def _obstacle_reasons(scene: Any) -> Tuple[Tuple[str, str], ...]:
        instances = list(getattr(scene, 'instances', []))
        if bool(getattr(scene, 'scene_blocked', True)) and not instances:
            return ((
                'obstacles.blocked', 'scene_blocked: %s' % str(
                    getattr(scene, 'blocking_reason', 'unknown reason'))),)
        invalid = [item for item in instances if not bool(
            getattr(item, 'valid', False))]
        if not invalid:
            return ()
        failures = tuple(as_failure(getattr(
            item, 'validity_reason', '')) for item in invalid)
        if failures and all(failure.has(
                FailureTag.OBSTACLE_TRANSFORM_TRANSIENT)
                for failure in failures):
            return ((
                'obstacles.fresh', 'obstacles data missing or stale'),)
        return ((
            'obstacles.geometry', 'invalid obstacle geometry is present'),)

    def evaluate(
            self, mode: SafetyMode, snapshot: TelemetrySnapshot,
            inputs: SafetyInputs = SafetyInputs(),
            authorization: SafetyAuthorization = SafetyAuthorization(),
    ) -> SafetyDecision:
        """Return a decision without ROS, parameter, clock, or mutable reads."""
        if not isinstance(mode, SafetyMode):
            mode = SafetyMode(mode)
        evidence = []
        failures = []

        def check(rule: str, passed: bool, detail: str) -> None:
            evidence.append(SafetyEvidence(rule, bool(passed), str(detail)))
            if not passed:
                failures.append((rule, str(detail)))

        check(
            'authorization.mission',
            not authorization.required or authorization.granted,
            'mission authorization is absent or expired')
        check(
            'authorization.motion_control',
            inputs.motion_control_authorized,
            'motion-control authority is unavailable')
        check(
            'plan.result', inputs.planner_result_valid,
            'planner result is invalid')
        check(
            'plan.schema', inputs.plan_schema_valid,
            'plan schema or hash binding is invalid')
        check(
            'plan.collision_model', inputs.collision_model_qualified,
            'Tesseract collision model is not qualified for hardware')
        check('path.valid', inputs.path_valid, 'approved path is invalid')
        if mode == SafetyMode.SCAN_APPROVAL and inputs.auto_capture:
            check(
                'workflow.capture_services', inputs.capture_services_ready,
                'one or more capture services are not ready')

        if inputs.target_drift_m is not None and not inputs.allow_target_motion:
            drift = float(inputs.target_drift_m)
            check(
                'target.drift',
                math.isfinite(drift) and drift <= float(
                    self.profile.max_target_drift_before_approval_m),
                'target moved %.3fm after planning; refresh the plan' % drift)

        required = list(self._required_fields(mode))
        obstacle_required = mode in (
            SafetyMode.ACQUISITION_MOTION,
            SafetyMode.SCAN_APPROVAL,
            SafetyMode.SCAN_MOTION)
        if obstacle_required and not (
                inputs.static_obstacle_scene_authorized
                or inputs.approved_obstacle_snapshot):
            if 'obstacles' not in required:
                required.append('obstacles')
        elif 'obstacles' in required and (
                inputs.static_obstacle_scene_authorized
                or inputs.approved_obstacle_snapshot):
            required.remove('obstacles')

        ages = []
        fresh = {}
        for key in self._TELEMETRY_FIELDS:
            observation = self._observation(snapshot, key)
            age = None if observation is None else observation.age_at(
                snapshot.captured_at)
            ages.append((key, age))
            timeout = (
                self.profile.motion_limits_timeout_sec
                if key == 'motion_limits' else
                self.profile.hold_joint_feedback_timeout_sec
                if mode == SafetyMode.HOLD_CURRENT and key == 'joints' else
                self.profile.data_timeout_sec)
            fresh[key] = bool(
                observation is not None
                and not observation.is_stale_at(
                    snapshot.captured_at, timeout))
            if key in required:
                legacy_key = 'camera_clock' if key == 'camera_clock' else key
                check(
                    '%s.fresh' % key,
                    fresh[key],
                    '%s data missing or stale' % legacy_key)

        limits_observation = snapshot.arm.motion_limits
        if 'motion_limits' in required and fresh.get('motion_limits'):
            limits = limits_observation.value
            check(
                'limits.valid',
                bool(getattr(limits, 'valid', False)),
                'controller motion limits changed after trajectory planning')
            check(
                'limits.compatible', inputs.motion_limits_compatible,
                'fresh controller motion limits are malformed')

        joints_observation = snapshot.arm.joints
        joint_values = None
        if fresh.get('joints'):
            joint_values = self._joint_values(joints_observation)
            check(
                'joints.shape', joint_values is not None,
                'joint feedback has fewer than six finite arm joints')
        if joint_values is not None and inputs.joint_limits:
            tolerance = (
                self.profile.configured_home_feedback_limit_tolerance_rad
                if inputs.configured_home_direct else
                self.profile.joint_feedback_limit_tolerance_rad)
            within = len(inputs.joint_limits) == 6 and all(
                float(bounds[0]) - tolerance <= value <=
                float(bounds[1]) + tolerance
                for value, bounds in zip(joint_values, inputs.joint_limits))
            check(
                'joints.limits', within,
                'joint feedback is outside configured limits')

        status_observation = snapshot.arm.status
        if 'arm_status' in required and fresh.get('arm_status'):
            status = status_observation.value
            check(
                'arm.error_code', int(getattr(status, 'err_code', 0)) == 0,
                'arm err_code=%d' % int(getattr(status, 'err_code', 0)))
            angle_limits = tuple(bool(getattr(
                status, 'joint_%d_angle_limit' % index, False))
                for index in range(1, 7))
            check(
                'arm.angle_limit', not any(angle_limits),
                'arm reports a joint angle-limit fault')
            communications = tuple(bool(getattr(
                status, 'communication_status_joint_%d' % index, False))
                for index in range(1, 7))
            check(
                'arm.communication', not any(communications),
                'arm reports a joint communication fault')
            for rule, detail in self._motor_reasons(status):
                check(rule, False, detail)

        camera_observation = snapshot.perception.camera
        if 'camera_clock' in required and fresh.get('camera_clock'):
            camera = camera_observation.value
            check(
                'camera.healthy', bool(getattr(camera, 'healthy', False)),
                'camera timestamp %s: %s' % (
                    str(getattr(camera, 'state', 'MISSING')),
                    str(getattr(camera, 'reason', 'no watchdog status'))))

        obstacles_observation = snapshot.perception.obstacles
        if (
                obstacle_required
                and obstacles_observation is None
                and inputs.approved_obstacle_snapshot
                and not inputs.static_obstacle_scene_authorized):
            check(
                'obstacles.fresh', False,
                'obstacles data missing or stale')
        if obstacle_required and obstacles_observation is not None:
            # Approved/static snapshots waive age/absence only. Newly observed
            # blocked or invalid geometry remains an immediate rejection.
            for rule, detail in self._obstacle_reasons(
                    obstacles_observation.value):
                check(rule, False, detail)

        if inputs.joints_settled is not None:
            check(
                'joints.settled', inputs.joints_settled,
                'joint feedback is not settled at the current-position hold'
                if mode == SafetyMode.HOLD_CURRENT else
                'joint feedback is not settled for acquisition'
                if mode == SafetyMode.ACQUISITION_APPROVAL else
                'joint feedback is not settled')

        if mode == SafetyMode.SCAN_APPROVAL:
            tracking_observation = snapshot.perception.tracking
            if fresh.get('tracking'):
                tracking = tracking_observation.value
                check(
                    'tracking.state',
                    str(getattr(tracking, 'lifecycle_state', '')) == 'TRACKING'
                    and bool(getattr(tracking, 'camera_settled', False)),
                    'tracking is not settled TRACKING')
                check(
                    'tracking.prediction_only',
                    not bool(getattr(tracking, 'prediction_only', True)),
                    'tracking is prediction-only')
                measurement_age = float(getattr(
                    tracking, 'measurement_age_sec', math.inf))
                check(
                    'tracking.measurement_age',
                    measurement_age <= float(
                        self.profile.max_tracking_measurement_age_sec),
                    'tracking measurement is stale')
                speed_scale = float(getattr(
                    tracking, 'recommended_speed_scale', 0.0))
                check(
                    'tracking.speed_scale',
                    speed_scale >= float(self.profile.min_tracking_speed_scale),
                    'tracking speed scale is below the motion threshold')
                if inputs.plan_execution_speed_percent > 0.0:
                    check(
                        'tracking.approved_speed',
                        self.profile.configured_speed_percent * speed_scale
                        + 1e-6 >=
                        inputs.plan_execution_speed_percent,
                        'tracking speed allowance fell below the approved '
                        'MoveJ speed; replan at the lower speed')
            target_observation = snapshot.perception.target_status
            if fresh.get('target_status'):
                target_status = str(target_observation.value)
                check(
                    'target.status',
                    target_status in ('TRACKING', 'LOCKED'),
                    'target_status=%s' % target_status)

        workflow_required = (
            inputs.workflow_required
            or mode == SafetyMode.SCAN_APPROVAL)
        if workflow_required and inputs.auto_capture:
            workflow_ready = inputs.workflow_ready
            if workflow_ready is None:
                observation = snapshot.mission.workflow
                workflow = None if observation is None else observation.value
                workflow_ready = bool(
                    isinstance(workflow, dict)
                    and str(workflow.get('state', '')) == 'SCAN_READY')
            check(
                'workflow.ready', bool(workflow_ready),
                'supervised workflow is not SCAN_READY')

        reasons = tuple(detail for _, detail in failures)
        first_failure = None
        if failures:
            first_failure = self._failure_for_rule(
                failures[0][0], failures[0][1])
        reacquisition = any(rule.startswith(('target.', 'tracking.'))
                            for rule, _ in failures)
        replan = any(rule.startswith(('plan.', 'path.', 'target.drift'))
                     for rule, _ in failures)
        return SafetyDecision(
            mode=mode,
            permitted=not failures,
            failure=first_failure,
            explanation=(
                'all shadow safety rules passed'
                if not failures else '; '.join(reasons)),
            reasons=reasons,
            evidence=tuple(evidence),
            telemetry_ages=tuple(ages),
            replan_required=replan,
            reacquisition_required=reacquisition,
        )
