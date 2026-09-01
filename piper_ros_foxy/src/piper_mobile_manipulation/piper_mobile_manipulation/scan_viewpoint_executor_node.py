#!/usr/bin/env python3
"""Operator-approved execution of collision-validated scan viewpoints."""

import json
import math
import time
from functools import wraps
from threading import RLock

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Vector3
from piper_msgs.msg import PiperMotionLimits, PiperStatusMsg
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from piper_mobile_manipulation.msg import (
    CameraTimestampHealth,
    ObstacleInstance3DArray,
    ScanExecutionPlan,
    ScanExecutionStatus,
    MotionPlan,
    TrackedTarget,
    TrackingHealth,
)
from piper_mobile_manipulation.infrastructure.failure_model import (
    as_failure,
    FailureTag,
)
from piper_mobile_manipulation.configuration import (
    configured_value,
    load_executor_configuration,
)
from piper_mobile_manipulation.execution.capture import (
    CaptureAction,
    CaptureCoordinator,
    rgbd_capture_handoff_action as _rgbd_capture_handoff_action,
    retryable_rgbd_capture_rejection as _retryable_capture_rejection,
    visual_capture_rejection as _visual_capture_rejection,
)
from piper_mobile_manipulation.execution.recovery import (
    RecoveryAction,
    RecoveryContext,
    RecoveryPolicy,
    runtime_gate_action as _runtime_gate_action,
    runtime_refresh_action,
)
from piper_mobile_manipulation.execution.authorization import (
    configured_home_endpoint_rejection,
    direct_home_stage_rejection,
    direct_home_stage_targets,
    PlanAuthorizationRequest,
    PlanAuthorizer,
    target_drift_before_approval_rejection as _target_drift_rejection,
    trajectory_count_rejection,
)
from piper_mobile_manipulation.execution.modes import (
    acquired_target_rejection,
    commanded_speed_percent,
    correlated_obstacle_scene_status,
    heavy_refresh_status_action,
    MULTIVIEW_SCAN,
    plan_count_rejection,
    planned_speed_rejection,
    RETURN_HOME,
    ROUGH_ACQUISITION,
    uses_bootstrap_static_scene,
)
from piper_mobile_manipulation.execution.motion import (
    camera_target_path_reasons,
    bootstrap_recovery_declaration_reasons,
    bootstrap_start_limit_recovery_reasons,
    CollisionBox,
    configured_home_feedback_limit_reasons,
    energized_hold_target,
    feedback_joint_limit_reasons,
    interpolate_joint_path,
    PiperScanKinematics,
    load_accepted_hand_eye,
    motor_control_reasons,
    powered_motion_calibration_rejection,
    load_conservative_joint_limits,
    validate_attached_box_external_clearance_path,
    validate_monotonic_self_clearance_escape,
    validate_joint_path,
)
from piper_mobile_manipulation.motion_limit_stability import MotionLimitStability
from piper_mobile_manipulation.planning.rays import decoded_ray_id
from piper_mobile_manipulation.execution.validation import (
    EXECUTION_TIMING_POLICY_VERSION,
    validate_sdk_movej_waypoint_path,
    validate_planner_point,
)


from piper_mobile_manipulation.safety_evaluator import (
    ObstacleAuthority,
    RuntimeGatePolicy,
    runtime_gate_policy,
    SafetyMode,
)
from piper_mobile_manipulation.srv import (
    ApproveScanExecution,
    AuthorizeMission,
    ExecuteHomeStage,
)
from piper_mobile_manipulation.infrastructure.telemetry_store import (
    TelemetryStore,
)
from piper_mobile_manipulation.perception.target_envelope import (
    classify_centered_silhouette,
    CROPPED_TOO_LARGE_DISTANCE_M,
    stamp_nanoseconds,
    validate_capture_model_seed,
    validate_shape_measurement,
    validate_shape_rejection,
)
from piper_mobile_manipulation.execution.trajectory import (
    joint_progress_error,
    TrajectoryAction,
    TrajectoryRunner,
    waypoint_motion_action,
)


# Private driver/executor handoff contract.  Ordinary configured-home stages
# retain ``home_goal_tolerance_rad``; STARTUP_WRIST alone must reach the
# driver's ready-zero window before ROUGH_HOME may send ordinary J6 commands.
STARTUP_WRIST_READY_TOLERANCE_RAD = 0.03
ACTIVE_STATES = {
    'WAITING_FOR_RUNTIME_REFRESH',
    'MOVING', 'SETTLING', 'SETTLING_HOME',
    'CAPTURING', 'CAPTURING_RGBD', 'WAIT_CAPTURE',
    'WAITING_FOR_CAPTURE_REFRESH',
    'WAITING_FOR_FRESH_FRAME', 'WAITING_FOR_GROUNDING_DINO',
    'WAITING_FOR_TRACKING_LOCK', 'WAITING_FOR_OBSTACLE_SCENE',
}
MAX_RGBD_CAPTURE_READINESS_RETRIES = 10
MAX_FINAL_CAPTURE_AIM_ERROR_DEG = 5.0
STREAM_LATE_WARNING_INTERVAL_SEC = 5.0
TARGET_SHAPE_QOS = QoSProfile(
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


def next_target_framing_standoff(
        current_distance_m,
        maximum_distance_m=CROPPED_TOO_LARGE_DISTANCE_M):
    """Return the next outward 0.10m rung, bounded by the framing maximum."""
    current = float(current_distance_m)
    maximum = float(maximum_distance_m)
    if (
            not math.isfinite(current) or current <= 0.0
            or not math.isfinite(maximum) or maximum <= current + 1e-6):
        return None
    next_tenth = (math.floor(current * 10.0 + 1e-6) + 1.0) / 10.0
    return min(maximum, max(current + 0.05, next_tenth))


def first_capture_framing_decision(
        payload, camera_target_distance_m, previous_minimum_m=None):
    """Translate a settled, aimed first silhouette into executor action."""
    state, reason = classify_centered_silhouette(
        payload, camera_target_distance_m)
    if state == 'CLEAR':
        return 'CLEAR', reason, None
    if state == 'TOO_CLOSE':
        return 'TOO_CLOSE', reason, None
    if state == 'TOO_LARGE':
        return 'TOO_LARGE', reason, None
    farther = next_target_framing_standoff(camera_target_distance_m)
    if previous_minimum_m is not None:
        advanced = next_target_framing_standoff(previous_minimum_m)
        if advanced is None:
            farther = None
        elif farther is not None:
            farther = max(farther, advanced)
    if farther is None:
        return (
            'NO_AIMED_ENDPOINT',
            reason + '; no farther target-facing endpoint remains',
            None,
        )
    return 'RETRY_FARTHER', reason, farther


def first_capture_model_seed(payload, framing_action):
    """Return model evidence only after the border gate reports clear."""
    if str(framing_action) != 'CLEAR':
        return None
    return validate_shape_measurement(payload)


def capture_result_from_response(message):
    """Validate the exact persisted RGB-D capture result from Trigger JSON."""
    try:
        payload = json.loads(str(message))
    except (TypeError, ValueError, json.JSONDecodeError):
        raise ValueError('capture response does not contain model-seed JSON')
    if (
            not isinstance(payload, dict)
            or int(payload.get('capture_result_schema_version', -1)) != 1):
        raise ValueError('capture response model-seed schema is unsupported')
    state = str(payload.get('occlusion_state', '')).upper()
    if state not in ('CLEAR', 'PARTIALLY_OCCLUDED', 'HEAVILY_OCCLUDED'):
        raise ValueError('capture response occlusion state is unqualified')
    payload = dict(payload)
    payload['occlusion_state'] = state
    payload['qualified_target_model_seed'] = validate_capture_model_seed(
        payload.get('qualified_target_model_seed'))
    return payload


def capture_model_seed_from_response(message):
    """Extract the exact persisted capture-one model seed from Trigger JSON."""
    return capture_result_from_response(message)['qualified_target_model_seed']


def qualified_scan_center_update(
        current_center, planned_center, accepted_views,
        qualified_target_shape, already_qualified=False):
    """Apply the one allowed bootstrap-to-object-centre transition."""
    proposed = np.asarray(planned_center, dtype=float)
    if proposed.shape != (3,) or not np.all(np.isfinite(proposed)):
        raise ValueError('planned target centre is malformed')
    qualified = bool(
        int(accepted_views) > 0
        and isinstance(qualified_target_shape, dict))
    if current_center is None:
        return proposed.copy(), bool(qualified or already_qualified), True
    current = np.asarray(current_center, dtype=float)
    if current.shape != (3,) or not np.all(np.isfinite(current)):
        raise ValueError('scan target centre is malformed')
    if qualified and not already_qualified:
        changed = not np.allclose(current, proposed, atol=1e-9, rtol=0.0)
        return proposed.copy(), True, changed
    return current.copy(), bool(already_qualified), False


def sdk_command_path(path, velocities, accelerations, times, execution_mode,
                     direct_home=False):
    """Collapse a fully validated straight chord to one PiPER MoveJ goal."""
    command_path = [np.asarray(item, dtype=float).copy() for item in path[1:]]
    command_velocities = [
        np.asarray(item, dtype=float).copy() for item in velocities[1:]]
    command_accelerations = [
        np.asarray(item, dtype=float).copy() for item in accelerations[1:]]
    command_times = [float(item) for item in times[1:]]
    mode = str(execution_mode).strip().upper()
    if mode == 'TESSERACT_STREAM':
        mode = 'TIMED_STREAM'
    if mode == 'DIRECT_MOVEJ' and not bool(direct_home):
        return (
            [np.asarray(path[-1], dtype=float).copy()],
            [np.zeros(6, dtype=float)],
            [np.zeros(6, dtype=float)],
            [float(times[-1])],
            False,
        )
    return (
        command_path, command_velocities, command_accelerations,
        command_times, bool(not direct_home and mode == 'TIMED_STREAM'))


# Preserve Phase 1/downstream pure-helper imports while their implementation
# lives in the focused Phase 7 application components.
rgbd_capture_handoff_action = _rgbd_capture_handoff_action
retryable_rgbd_capture_rejection = _retryable_capture_rejection
visual_capture_rejection = _visual_capture_rejection
target_drift_before_approval_rejection = _target_drift_rejection
runtime_gate_action = _runtime_gate_action


def message_source_metadata(message):
    """Read optional ROS header metadata without changing callback policy."""
    header = getattr(message, 'header', None)
    stamp = getattr(header, 'stamp', None)
    source_stamp_ns = None
    if stamp is not None:
        source_stamp_ns = (
            int(getattr(stamp, 'sec', 0)) * 1_000_000_000
            + int(getattr(stamp, 'nanosec', 0)))
    return source_stamp_ns, str(getattr(header, 'frame_id', ''))


def snapshot_observation(snapshot, key):
    """Compatibility map from established freshness keys to typed fields."""
    if key == 'joints':
        return snapshot.arm.joints
    if key == 'arm_status':
        return snapshot.arm.status
    if key == 'motion_limits':
        return snapshot.arm.motion_limits
    if key == 'camera_clock':
        return snapshot.perception.camera
    if key == 'tracked_target':
        return snapshot.perception.target
    if key == 'tracking':
        return snapshot.perception.tracking
    if key == 'target_status':
        return snapshot.perception.target_status
    if key == 'obstacles':
        return snapshot.perception.obstacles
    if key == 'scan':
        return snapshot.mission.reachable_scan
    if key == 'workflow':
        return snapshot.mission.workflow
    return None


def mark_callback_observation(owner, key):
    """Keep old one-argument test seams while atomically dating production."""
    telemetry_store = getattr(owner, 'telemetry_store', None)
    if telemetry_store is None:
        owner.mark(key)
        return None, None
    observed_at = owner.now()
    owner.updated[key] = observed_at
    return observed_at, telemetry_store


def approved_return_home_obstacle_snapshot(
        returning_home, collision_model_qualified, obstacles):
    """Keep the approval-time scene for the exact planned home segment."""
    return bool(
        returning_home
        and collision_model_qualified
        and obstacles is not None
    )


def approved_multiview_motion_obstacle_snapshot(
        plan_kind, state, collision_model_qualified, obstacles):
    """
    Keep the approval-time scene only while one exact scan target moves.

    The eye-in-hand SAM2 stream can pause or request a new seed under motion
    blur.  It is not a dynamic safety scanner.  A MULTIVIEW_SCAN target reaches
    MOVING only after the current scene was fresh, valid, collision-qualified
    and bound into the approved direct segment.  Permit that existing snapshot
    to age until the endpoint; any newly received blocked/invalid scene is still
    evaluated immediately, and the next target still requires a fresh scene.
    """
    return bool(
        plan_kind == MULTIVIEW_SCAN
        and state == 'MOVING'
        and collision_model_qualified
        and obstacles is not None
    )


def bootstrap_abort_retrace_uses_static_scene(
        plan_kind, viewpoint_index, collision_model_qualified):
    """
    Mirror the approval scene for the exact first acquisition retrace.

    The first rough-acquisition segment is deliberately planned and validated
    before perception obstacle geometry exists.  A benign operator cancel may
    reverse only endpoints already executed from that approval; requiring a
    newly-created obstacle array for the reverse would strand the arm at the
    acquisition look even though the outward segment used the qualified static
    robot scene.  This exception never applies to later acquisition looks or
    multiview motion.
    """
    return bool(
        collision_model_qualified
        and uses_bootstrap_static_scene(plan_kind, viewpoint_index)
    )


def terminal_home_hold_required(state, reason):
    """Recognize an abort that has already completed its bounded home retrace."""
    return bool(
        str(state) == 'ABORTED'
        and as_failure(reason).has(FailureTag.TERMINAL_HOME_REACHED)
    )


def home_position_sample_settled(
        current, target, previous, target_tolerance, motion_tolerance):
    """Prove a home sample from position, independent of noisy SDK speed."""
    current_values = np.asarray(current, dtype=float)
    target_values = np.asarray(target, dtype=float)
    if (
            current_values.shape != (6,)
            or target_values.shape != (6,)
            or not np.all(np.isfinite(current_values))
            or not np.all(np.isfinite(target_values))):
        return False
    if float(np.max(np.abs(current_values - target_values))) > float(
            target_tolerance):
        return False
    if previous is None:
        return False
    previous_values = np.asarray(previous, dtype=float)
    if (
            previous_values.shape != (6,)
            or not np.all(np.isfinite(previous_values))):
        return False
    return float(np.max(np.abs(
        current_values - previous_values))) <= float(motion_tolerance)


def target_position_window_sample_settled(
        current, target, anchor, target_tolerance, motion_tolerance):
    """Prove endpoint stability without trusting noisy SDK speed samples."""
    current_values = np.asarray(current, dtype=float)
    target_values = np.asarray(target, dtype=float)
    if (
            current_values.shape != (6,)
            or target_values.shape != (6,)
            or not np.all(np.isfinite(current_values))
            or not np.all(np.isfinite(target_values))
            or float(np.max(np.abs(
                current_values - target_values))) > float(target_tolerance)):
        return False, None
    if anchor is None:
        return False, current_values.copy()
    anchor_values = np.asarray(anchor, dtype=float)
    if (
            anchor_values.shape != (6,)
            or not np.all(np.isfinite(anchor_values))
            or float(np.max(np.abs(
                current_values - anchor_values))) > float(motion_tolerance)):
        return False, current_values.copy()
    return True, anchor_values.copy()


def obstacle_scene_runtime_reasons(scene):
    """Classify temporary transform gaps separately from unsafe geometry."""
    instances = list(getattr(scene, 'instances', []))
    if bool(getattr(scene, 'scene_blocked', True)) and not instances:
        return ['scene_blocked: %s' % str(
            getattr(scene, 'blocking_reason', 'unknown reason'))]
    invalid = [item for item in instances if not bool(
        getattr(item, 'valid', False))]
    if not invalid:
        return []
    validity_failures = [
        as_failure(getattr(item, 'validity_reason', ''))
        for item in invalid
    ]
    if validity_failures and all(
            failure.has(FailureTag.OBSTACLE_TRANSFORM_TRANSIENT)
            for failure in validity_failures):
        return ['obstacles data missing or stale']
    return ['invalid obstacle geometry is present']


def abort_return_home_blocker(reason):
    """Block direct home only when command/feedback authority is untrusted."""
    return as_failure(reason).blocker


def approved_retrace_validation_reasons(reasons):
    """
    Keep changing-scene checks without re-rejecting an executed path.

    Every target in the abort history is an endpoint that this executor has
    already reached from an approval-bound, collision-qualified planner
    path.  Reversing those same SDK MoveJ segments cannot introduce a new
    robot self-collision.  The generic validator can nevertheless report the
    folded-home contact again because it does not have the proposal's bounded
    recovery metadata.  Ignore only that static self-clearance duplicate;
    obstacle, floor, limit and malformed-path failures remain blockers.
    """
    return [
        str(reason) for reason in reasons
        if not as_failure(reason).has(
            FailureTag.SELF_COLLISION_CLEARANCE_DUPLICATE)
    ]


def missing_obstacles_can_wait(
        plan_kind, viewpoint_index, state,
        bootstrap_abort_retrace=False):
    """
    Let stationary phases reach the bounded pre-motion refresh.

    Heavy refinement can briefly reseed the live instance stream after an
    accepted capture. Missing geometry still blocks the next command, but it
    must not consume the exact approval while the arm is already stopped.
    """
    if bool(bootstrap_abort_retrace):
        return True
    if (
            plan_kind == ROUGH_ACQUISITION
            and (
                uses_bootstrap_static_scene(plan_kind, viewpoint_index)
                or state == 'WAITING_FOR_OBSTACLE_SCENE')):
        return True
    return (
        plan_kind == MULTIVIEW_SCAN
        and state in (
            'SETTLING', 'CAPTURING', 'CAPTURING_RGBD', 'WAIT_CAPTURE',
            'WAITING_FOR_CAPTURE_REFRESH'))


class ScanViewpointExecutorNode(Node):
    def __init__(self):
        super().__init__('scan_viewpoint_executor')
        # The 200 Hz state-machine timer needs its own scheduler lane, while
        # every state-mutating callback still shares one explicit lock. This
        # keeps approval/cancel/capture responses serviceable without allowing
        # the timer and a control callback to mutate execution state together.
        self.control_state_lock = RLock()
        self.control_callback_group = MutuallyExclusiveCallbackGroup()
        self.timer_callback_group = MutuallyExclusiveCallbackGroup()
        self.telemetry_callback_group = MutuallyExclusiveCallbackGroup()
        self.client_callback_group = MutuallyExclusiveCallbackGroup()
        self.configuration = load_executor_configuration(self)

        calibration_path = self.configuration.interfaces.hand_eye_calibration_path
        bounds_path = self.configuration.interfaces.joint_bounds_path
        if not calibration_path:
            raise RuntimeError('hand_eye_calibration_path is required')
        if not bounds_path:
            raise RuntimeError('joint_bounds_path is required')
        if self.configuration.motion.enable_real_arm_motion:
            calibration_rejection = powered_motion_calibration_rejection(
                calibration_path)
            if calibration_rejection:
                raise RuntimeError(calibration_rejection)
        self.kinematics = PiperScanKinematics(load_accepted_hand_eye(calibration_path))
        self.joint_limits, ignored_bounds = load_conservative_joint_limits(bounds_path)
        self.telemetry_store = TelemetryStore(clock=self.now)
        self.plan_authorizer = PlanAuthorizer()
        self.trajectory_runner = TrajectoryRunner()
        self.capture_coordinator = CaptureCoordinator(
            MAX_RGBD_CAPTURE_READINESS_RETRIES)
        self.recovery_policy = RecoveryPolicy()
        self.latest_scan = None
        self.latest_joint_state = None
        self.latest_arm_status = None
        self.latest_motion_limits = None
        self.motion_limit_stability = MotionLimitStability(
            self.configuration.safety.motion_limits_change_confirmation_sec,
            self.configuration.safety.motion_limits_change_minimum_samples,
        )
        self.latest_tracking_health = None
        self.latest_tracked_target = None
        self.latest_camera_timestamp_health = None
        self.latest_target_status = 'UNKNOWN'
        self.latest_target_shape = None
        self.latest_target_shape_at = -1e9
        self.latest_target_shape_stamp_ns = 0
        self.latest_obstacles = None
        self.latest_workflow = None
        self.scan_session_id = ''
        self.scan_history = []
        self.scan_rejections = []
        self.scan_qualified_target_shape = None
        self.pending_scan_qualified_target_shape = None
        self.scan_qualified_target_model_seed = None
        self.pending_scan_qualified_target_model_seed = None
        self.pending_capture_occlusion_state = 'UNKNOWN'
        self.capture_semantic_probe_pending = False
        self.latest_achieved_scan_view = None
        self.scan_coverage_target_center = None
        self.scan_target_center_qualified = False
        self.updated = {}
        self.state = 'IDLE'
        self.reason = 'waiting for a validated viewpoint proposal'
        self.plan_id = ''
        self.plan_backend = ''
        self.plan_kind = MULTIVIEW_SCAN
        self.plan_source_request_id = ''
        self.plan_created = 0.0
        self.plan_targets = []
        self.plan_paths = []
        self.plan_path_velocities = []
        self.plan_path_accelerations = []
        self.plan_path_times = []
        self.plan_bootstrap_recovery_end_points = []
        self.plan_bootstrap_recovery_joints = []
        self.plan_bootstrap_recovery_joint_sets = []
        self.plan_powered_start_recovery_end_points = []
        self.plan_powered_start_recovery_joint_sets = []
        self.plan_startup_home_static = []
        self.plan_configured_home_direct = []
        self.plan_configured_home_stages = []
        self.plan_segment_execution_modes = []
        self.plan_viewpoints = []
        self.plan_candidate_count = 0
        self.plan_capture_count = 0
        self.plan_returns_home = False
        self.plan_target_center = None
        self.plan_source_trajectory_sha256 = ''
        self.plan_trajectory_sha256 = ''
        self.plan_execution_speed_percent = 0.0
        self.plan_motion_limits_sha256 = ''
        self.runtime_motion_limits_sha256 = ''
        self.plan_collision_model_qualified = False
        self.pending_limit_refresh_plan = None
        self.pending_limit_refresh_deadline = 0.0
        self.current_view = 0
        self.current_path = []
        self.current_path_velocities = []
        self.current_path_accelerations = []
        self.current_path_times = []
        self.current_path_streaming = False
        self.current_trajectory = None
        self.path_index = 0
        self.command_target = None
        self.command_sent_at = 0.0
        self.command_samples_sent = 0
        self.max_command_interval_sec = 0.0
        self.dropped_command_samples = 0
        self.motion_started_at = None
        self.stream_wall_started_at = None
        self.stream_last_tick_at = None
        self.stream_schedule_paused_sec = 0.0
        self.stream_following_hold_started_at = None
        self.stream_schedule_completion_logged = False
        self.stream_late_event_count = 0
        self.stream_late_missed_samples = 0
        self.stream_late_max_sec = 0.0
        self.stream_late_reported_at = None
        self.last_stream_planned_duration_sec = 0.0
        self.last_stream_actual_duration_sec = 0.0
        self.last_stream_achieved_rate_hz = 0.0
        self.last_motion_status_at = 0.0
        self.waypoint_started_at = None
        self.waypoint_last_progress_at = None
        self.waypoint_best_error = math.inf
        self.current_waypoint_error = 0.0
        self.max_waypoint_error = 0.0
        self.pending_motion_reason = ''
        self.runtime_refresh_require_workflow = False
        self.runtime_refresh_allow_missing_obstacles = False
        self.runtime_refresh_resume_state = ''
        self.runtime_recovery_started_at = None
        self.settle_started = None
        self.settle_started_ros_ns = 0
        self.settle_position_anchor = None
        self.settle_last_joint_update = -1e9
        self.settle_last_sample_ok = False
        self.settle_diagnostic = 'settle proof has not sampled joint feedback'
        self.settle_reset_count = 0
        self.settle_longest_window_sec = 0.0
        self.settle_last_reset_reason = ''
        self.home_settle_previous_joints = None
        self.home_settle_last_joint_update = -1e9
        self.home_settle_last_sample_ok = False
        self.state_started = self.now()
        self.capture_future = None
        self.rgbd_capture_future = None
        self.rgbd_capture_attempts = 0
        self.capture_heavy_refresh_started = None
        self.capture_heavy_refresh_request_id = ''
        self.capture_heavy_refresh_min_image_stamp_ns = 0
        self.capture_heavy_refresh_publish_attempts = 0
        self.capture_heavy_refresh_waiting_for_worker = False
        self.capture_rejection_reason = ''
        self.pending_capture_occlusion_state = 'UNKNOWN'
        self.capture_semantic_probe_pending = False
        self.finish_scan_future = None
        self.return_home_warning = ''
        self.abort_return_in_progress = False
        self.abort_return_reason = ''
        self.abort_return_bootstrap_static_scene = False
        self.retrace_joint_targets = []
        self.capture_accepted_before = 0
        self.acquisition_refresh_started = None
        self.acquisition_request_id = ''
        self.acquisition_request_attempt = 0
        self.acquisition_min_image_stamp_ns = 0
        self.acquisition_job_image_stamp_ns = 0
        self.acquisition_job_started = None
        self.acquisition_detection_completed = None
        self.acquisition_waiting_for_worker = False
        self.pending_acquisition_heavy_status = None
        self.acquisition_scene_snapshot_validated = False
        self.mission_task_id = ''
        self.mission_sha256 = ''
        self.mission_planner_backend = ''
        self.mission_expires_at_sec = 0.0

        history_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        # A proposal can be produced before the GUI or automatic mission has
        # completed its service round-trip. Latch the exact hash-bound plan so
        # that subscriber timing cannot turn a valid proposal into a timeout.
        self.plan_pub = self.create_publisher(
            ScanExecutionPlan,
            self.configuration.interfaces.plan_topic,
            history_qos,
        )
        self.status_pub = self.create_publisher(
            ScanExecutionStatus, self.configuration.interfaces.status_topic,
            10)
        self.scan_history_pub = self.create_publisher(
            String, self.configuration.interfaces.scan_session_history_topic,
            history_qos)
        self.command_pub = None
        if self.real_motion_enabled():
            self.command_pub = self.create_publisher(
                JointState, self.configuration.interfaces.joint_command_topic,
                10)
        self.create_subscription(
            String, self.configuration.interfaces.reachable_viewpoints_topic,
            self.serialized_control_callback(self.scan_cb), 10,
            callback_group=self.control_callback_group)
        self.motion_plan_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.motion_plan_sub = self.create_subscription(
            MotionPlan, self.configuration.interfaces.motion_plan_topic,
            self.serialized_control_callback(self.motion_plan_cb),
            self.motion_plan_qos,
            callback_group=self.control_callback_group)
        self.create_subscription(
            JointState, self.configuration.interfaces.joint_states_topic,
            self.joint_cb, 10,
            callback_group=self.telemetry_callback_group)
        self.create_subscription(
            PiperStatusMsg, self.configuration.interfaces.arm_status_topic,
            self.arm_status_cb, 10,
            callback_group=self.telemetry_callback_group)
        self.create_subscription(
            PiperMotionLimits,
            self.configuration.interfaces.motion_limits_topic,
            self.serialized_control_callback(self.motion_limits_cb), 10,
            callback_group=self.control_callback_group)
        self.create_subscription(
            TrackingHealth,
            self.configuration.interfaces.tracking_health_topic,
            self.tracking_health_cb, 10,
            callback_group=self.telemetry_callback_group)
        self.create_subscription(
            TrackedTarget, self.configuration.interfaces.tracked_target_topic,
            self.tracked_target_cb, 10,
            callback_group=self.telemetry_callback_group)
        self.create_subscription(
            CameraTimestampHealth,
            self.configuration.interfaces.camera_timestamp_health_topic,
            self.camera_timestamp_health_cb, 10,
            callback_group=self.telemetry_callback_group)
        self.create_subscription(
            String, self.configuration.interfaces.target_status_topic,
            self.target_status_cb, 10,
            callback_group=self.telemetry_callback_group)
        self.create_subscription(
            String, self.configuration.interfaces.target_shape_topic,
            self.serialized_control_callback(self.target_shape_cb),
            TARGET_SHAPE_QOS,
            callback_group=self.control_callback_group)
        self.create_subscription(
            ObstacleInstance3DArray,
            self.configuration.interfaces.obstacle_topic,
            self.obstacle_cb, 10,
            callback_group=self.telemetry_callback_group)
        self.create_subscription(
            String, self.configuration.interfaces.workflow_status_topic,
            self.serialized_control_callback(self.workflow_cb), 10,
            callback_group=self.control_callback_group)
        self.create_service(
            ApproveScanExecution, '~/approve',
            self.serialized_control_callback(self.approve_cb),
            callback_group=self.control_callback_group)
        self.create_service(
            AuthorizeMission, '~/authorize_mission',
            self.serialized_control_callback(self.authorize_mission_cb),
            callback_group=self.control_callback_group)
        self.create_service(
            ExecuteHomeStage, '~/execute_home_stage',
            self.serialized_control_callback(self.execute_home_stage_cb),
            callback_group=self.control_callback_group)
        self.create_service(
            Trigger, '~/hold',
            self.serialized_control_callback(self.hold_cb),
            callback_group=self.control_callback_group)
        self.create_service(
            Trigger, '~/cancel',
            self.serialized_control_callback(self.cancel_cb),
            callback_group=self.control_callback_group)
        self.create_service(
            Trigger, '~/refresh_plan',
            self.serialized_control_callback(self.refresh_cb),
            callback_group=self.control_callback_group)
        self.create_service(
            Trigger, '~/diagnostic_state',
            self.serialized_control_callback(self.diagnostic_state_cb),
            callback_group=self.control_callback_group)
        self.capture_client = self.create_client(
            Trigger, self.configuration.interfaces.capture_service,
            callback_group=self.client_callback_group)
        self.rgbd_capture_client = self.create_client(
            Trigger, self.configuration.interfaces.rgbd_capture_service,
            callback_group=self.client_callback_group)
        self.finish_scan_client = self.create_client(
            Trigger, self.configuration.interfaces.finish_scan_service,
            callback_group=self.client_callback_group)
        self.heavy_refresh_pub = self.create_publisher(
            String,
            self.configuration.interfaces.heavy_refresh_request_topic, 10)
        self.create_subscription(
            String, self.configuration.interfaces.heavy_refresh_status_topic,
            self.serialized_control_callback(self.heavy_refresh_status_cb), 10,
            callback_group=self.control_callback_group)
        command_rate = self.configuration.motion.trajectory_command_rate_hz
        if not math.isfinite(command_rate) or command_rate <= 0.0:
            raise RuntimeError('trajectory_command_rate_hz must be finite and positive')
        tick_rate = self.configuration.motion.executor_tick_rate_hz
        if (
                not math.isfinite(tick_rate)
                or tick_rate < 2.0 * command_rate):
            raise RuntimeError(
                'executor_tick_rate_hz must be at least twice '
                'trajectory_command_rate_hz')
        self.create_timer(
            1.0 / tick_rate,
            self.serialized_control_callback(self.tick),
            callback_group=self.timer_callback_group)

        mode = 'motion opt-in available' if self.real_motion_enabled() else 'proposal-only'
        self.get_logger().warn(
            'Scan viewpoint executor started %s at %.1f%% configured speed; '
            'An exact all-six-joint planner is mandatory. '
            'Ignored saved bounds: %s'
            % (mode, self.speed_percent(),
               ','.join(ignored_bounds) or 'none'))
        self.publish_status()
        self.publish_scan_history()

    def serialized_control_callback(self, callback):
        """Serialize timer and control callbacks across scheduler groups."""
        @wraps(callback)
        def guarded(*args, **kwargs):
            with self.control_state_lock:
                return callback(*args, **kwargs)
        return guarded

    def now(self):
        return time.monotonic()

    def mark(self, key, observed_at=None):
        self.updated[key] = (
            self.now() if observed_at is None else float(observed_at))

    def fresh(self, key, timeout=None):
        maximum = float(configured_value(self, 'data_timeout_sec')) \
            if timeout is None else float(timeout)
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            return self.now() - self.updated.get(key, -1e9) <= maximum
        snapshot = telemetry_store.snapshot()
        observation = snapshot_observation(snapshot, key)
        return bool(
            observation is not None
            and not observation.is_stale_at(snapshot.captured_at, maximum))

    def scan_cb(self, msg):
        if self.state in ACTIVE_STATES:
            return
        try:
            payload = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError) as error:
            self.invalidate_plan('invalid reachable viewpoint JSON: %s' % error)
            return
        self.latest_scan = payload
        observed_at, telemetry_store = mark_callback_observation(self, 'scan')
        if telemetry_store is not None:
            telemetry_store.update_reachable_scan(
                payload, received_at=observed_at)
        if self.state == 'ABORTED':
            self.state = 'IDLE'

    def motion_plan_cb(self, msg):
        if self.state in ACTIVE_STATES:
            return
        self.plan_backend = str(
            getattr(msg, 'backend', 'tesseract')).strip().lower()
        plan_kind = str(msg.plan_kind)
        source_request_id = str(msg.source_request_id)
        if not msg.valid:
            planner_decision = self.plan_authorizer.planner_result(
                False, msg.reason)
            self.invalidate_plan(
                planner_decision.detail,
                plan_kind=plan_kind,
                source_request_id=source_request_id,
                plan_id=str(msg.plan_id),
            )
            return
        motion_limits_timeout = float(
            configured_value(self, 'motion_limits_timeout_sec'))
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            limits = self.latest_motion_limits
            limits_fresh = self.fresh(
                'motion_limits', motion_limits_timeout)
        else:
            snapshot = telemetry_store.snapshot()
            observation = snapshot.arm.motion_limits
            limits = None if observation is None else observation.value
            limits_fresh = bool(
                observation is not None
                and not observation.is_stale_at(
                    snapshot.captured_at, motion_limits_timeout))
        limits_match_plan = bool(
            limits is not None
            and bool(limits.valid)
            and len(str(msg.motion_limits_sha256)) == 64
            and str(msg.motion_limits_sha256) == str(limits.limits_sha256)
        )
        if (
                not limits_fresh
                and limits_match_plan):
            # The typed plan and the last accepted controller generation agree,
            # but a single-threaded executor can dequeue the plan just before a
            # queued 1 Hz limit sample.  Keep the command-free proposal only
            # long enough to observe a genuinely fresh matching sample.  No
            # approval or motion is possible in this state.
            self.pending_limit_refresh_plan = msg
            self.pending_limit_refresh_deadline = self.now() + \
                motion_limits_timeout
            self.set_state(
                'WAITING_FOR_PLAN_LIMITS',
                'hash-bound planner proposal is waiting for one fresh '
                'matching controller-limit sample',
            )
            return
        reasons = []
        if not msg.dry_run or msg.real_arm_motion:
            reasons.append('planner proposal flags are not command-free')
        if self.plan_backend not in ('tesseract', 'curobo'):
            reasons.append(
                'unsupported planning backend: %s' % self.plan_backend)
        if (
                getattr(self, 'mission_planner_backend', '')
                and self.plan_backend != self.mission_planner_backend):
            reasons.append(
                'planner backend does not match active mission authorization')
        if not msg.plan_id or len(msg.trajectory_sha256) != 64:
            reasons.append('plan identity or trajectory hash is invalid')
        if str(msg.timing_policy) != EXECUTION_TIMING_POLICY_VERSION:
            reasons.append('planner timing policy is unsupported')
        closed_loop_one_view = self.param_bool('closed_loop_one_view')
        returns_home = (
            plan_kind in (MULTIVIEW_SCAN, RETURN_HOME)
            and len(msg.trajectories) == len(msg.viewpoint_indices) + 1)
        count_reason = trajectory_count_rejection(
            plan_kind,
            len(msg.trajectories),
            len(msg.viewpoint_indices),
            closed_loop_one_view,
        )
        if count_reason:
            reasons.append(count_reason)
        if plan_kind == ROUGH_ACQUISITION and not source_request_id:
            reasons.append('rough acquisition source request ID is missing')
        metadata_lengths = {
            'bootstrap recovery endpoints': len(msg.bootstrap_recovery_end_points),
            'bootstrap recovery joints': len(msg.bootstrap_recovery_joints),
            'bootstrap recovery deltas': len(msg.bootstrap_recovery_delta_rad),
            'bootstrap recovery evidence': len(msg.bootstrap_recovery_evidence_json),
        }
        for label, count in metadata_lengths.items():
            if count != len(msg.trajectories):
                reasons.append('%s and trajectory counts differ' % label)
        count_rejection = plan_count_rejection(
            plan_kind,
            len(msg.viewpoint_indices),
            int(configured_value(self, 'min_execution_viewpoints')),
            int(configured_value(self, 'acquisition_max_viewpoints')),
            session_accepted_views=(
                len(self.scan_history)
                if plan_kind == MULTIVIEW_SCAN else 0),
            session_maximum_views=(
                int(configured_value(self, 'max_execution_viewpoints'))
                if plan_kind == MULTIVIEW_SCAN else None),
            closed_loop_one_view=self.param_bool('closed_loop_one_view'),
        )
        if count_rejection:
            reasons.append(count_rejection)
        paths = []
        path_velocities = []
        path_accelerations = []
        path_times = []
        targets = []
        recovery_end_points = []
        recovery_joints = []
        recovery_joint_sets = []
        powered_start_recovery_end_points = []
        powered_start_recovery_joint_sets = []
        startup_home_static_segments = []
        configured_home_direct_segments = []
        configured_home_stages = []
        segment_execution_modes = []
        maximum_step = float(
            configured_value(self, 'trajectory_joint_step_rad'))
        command_rate = float(configured_value(self, 'trajectory_command_rate_hz'))
        if abs(float(msg.command_rate_hz) - command_rate) > 1e-6:
            reasons.append('planner command rate does not match the executor')
        if not limits_fresh:
            reasons.append('controller motion limits are missing or stale')
        if limits is None or not limits.valid:
            reasons.append('controller motion limits are invalid')
        else:
            if (
                    str(msg.motion_limits_sha256) != str(limits.limits_sha256)
                    or len(str(msg.motion_limits_sha256)) != 64):
                reasons.append(
                    'planner controller-limit binding is stale or mismatched')
        if telemetry_store is None:
            tracking_health = self.latest_tracking_health
        else:
            tracking_observation = snapshot.perception.tracking
            tracking_health = (
                None if tracking_observation is None
                else tracking_observation.value)
        tracking_scale = (
            float(tracking_health.recommended_speed_scale)
            if tracking_health is not None else 1.0)
        execution_speed = float(msg.execution_speed_percent)
        speed_rejection = planned_speed_rejection(
            float(configured_value(self, 'speed_percent')),
            plan_kind,
            tracking_scale,
            execution_speed,
        )
        if speed_rejection:
            reasons.append(speed_rejection)
        expected_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        for segment_index, trajectory in enumerate(msg.trajectories):
            if list(trajectory.joint_names) != expected_names:
                reasons.append('segment %d joint order is invalid' % segment_index)
                continue
            path = []
            velocities = []
            accelerations = []
            times = []
            previous_time = -1.0
            for point_index, point in enumerate(trajectory.points):
                when = float(point.time_from_start.sec) + float(
                    point.time_from_start.nanosec) * 1e-9
                try:
                    values, point_velocities, point_accelerations, when = (
                        validate_planner_point(
                            point.positions,
                            point.velocities,
                            point.accelerations,
                            when,
                            previous_time if point_index else None,
                        ))
                except ValueError as error:
                    reasons.append(
                        'segment %d point %d is invalid: %s'
                        % (segment_index, point_index, error))
                    break
                path.append(values)
                velocities.append(point_velocities)
                accelerations.append(point_accelerations)
                times.append(when)
                previous_time = when
            if len(path) < 2:
                reasons.append('segment %d has fewer than two valid points' % segment_index)
                continue
            recovery_end = int(msg.bootstrap_recovery_end_points[segment_index]) \
                if segment_index < len(msg.bootstrap_recovery_end_points) else -1
            recovery_joint = int(msg.bootstrap_recovery_joints[segment_index]) \
                if segment_index < len(msg.bootstrap_recovery_joints) else 0
            recovery_delta = float(msg.bootstrap_recovery_delta_rad[segment_index]) \
                if segment_index < len(msg.bootstrap_recovery_delta_rad) else 0.0
            evidence_text = str(msg.bootstrap_recovery_evidence_json[segment_index]) \
                if segment_index < len(msg.bootstrap_recovery_evidence_json) else ''
            try:
                segment_evidence = json.loads(evidence_text)
            except (TypeError, json.JSONDecodeError):
                segment_evidence = None
            if not isinstance(segment_evidence, dict):
                startup_home_static = False
                reasons.append(
                    'segment %d recovery evidence is invalid' % segment_index)
            else:
                startup_home_static = segment_evidence.get(
                    'startup_home_static', False)
                if not isinstance(startup_home_static, bool):
                    startup_home_static = False
                    reasons.append(
                        'segment %d startup-home static evidence is invalid'
                        % segment_index)
            configured_home_direct = bool(
                isinstance(segment_evidence, dict)
                and segment_evidence.get(
                    'configured_home_direct_joint_move', False))
            collision_bypassed = bool(
                isinstance(segment_evidence, dict)
                and segment_evidence.get(
                    'collision_validation_bypassed', False))
            configured_home_stage = str(
                segment_evidence.get('home_stage', '')
                if isinstance(segment_evidence, dict) else '').strip().upper()
            configured_home_validation = str(
                segment_evidence.get('validation', '')
                if isinstance(segment_evidence, dict) else '')
            sdk_execution_mode = str(
                segment_evidence.get(
                    'sdk_execution_mode', 'TIMED_STREAM')
                if isinstance(segment_evidence, dict)
                else 'TIMED_STREAM').strip().upper()
            if sdk_execution_mode == 'TESSERACT_STREAM':
                sdk_execution_mode = 'TIMED_STREAM'
            if sdk_execution_mode not in ('DIRECT_MOVEJ', 'TIMED_STREAM'):
                reasons.append(
                    'segment %d SDK execution mode is invalid' % segment_index)
                sdk_execution_mode = 'TIMED_STREAM'
            if sdk_execution_mode == 'DIRECT_MOVEJ':
                if (
                        recovery_end >= 0 or configured_home_direct
                        or int(segment_evidence.get(
                            'sdk_command_anchor_count', -1)) != 1
                        or int(segment_evidence.get(
                            'direct_movej_source_points', -1)) != 2
                        or 'independently passed dense collision' not in str(
                            segment_evidence.get(
                                'direct_movej_validation', ''))):
                    reasons.append(
                        'segment %d direct MoveJ evidence is invalid'
                        % segment_index)
            if configured_home_direct:
                if not (
                        plan_kind == RETURN_HOME
                        and returns_home
                        and segment_index == 0
                        and len(msg.trajectories) == 1
                        and len(msg.viewpoint_indices) == 0
                        and len(path) == 2
                        and collision_bypassed
                        and configured_home_stage in (
                            'CONFIGURED_HOME', 'STARTUP_WRIST', 'PRE_HOME',
                            'ROUGH_HOME', 'STORAGE_WRIST')
                        and configured_home_validation ==
                        'configured_home_collision_validation_bypassed'):
                    reasons.append(
                        'segment %d configured-home collision bypass escaped '
                        'its dedicated RETURN_HOME scope' % segment_index)
                endpoint_rejection = configured_home_endpoint_rejection(
                    configured_home_stage,
                    path[-1],
                    segment_evidence.get(
                        'configured_home_goal_positions_rad', []),
                    configured_value(self, 'return_home_positions_rad'),
                )
                if endpoint_rejection:
                    reasons.append(
                        'segment %d %s' % (
                            segment_index, endpoint_rejection))
            elif collision_bypassed or configured_home_stage:
                reasons.append(
                    'segment %d has undeclared configured-home bypass evidence'
                    % segment_index)
            if startup_home_static and not (
                    plan_kind == RETURN_HOME
                    and returns_home
                    and segment_index == 0
                    and len(msg.trajectories) == 1
                    and len(msg.viewpoint_indices) == 0):
                reasons.append(
                    'segment %d startup-home static evidence escaped its scope'
                    % segment_index)
            declared_joints = []
            declared_deltas = []
            powered_start_end = -1
            powered_start_joints = []
            powered_start_deltas = []
            if recovery_end >= 0:
                terminal_home_recovery = bool(
                    plan_kind in (MULTIVIEW_SCAN, RETURN_HOME)
                    and returns_home
                    and segment_index == len(msg.trajectories) - 1)
                acquisition_recovery = bool(
                    plan_kind == ROUGH_ACQUISITION and segment_index == 0)
                if not acquisition_recovery and not terminal_home_recovery:
                    reasons.append(
                        'bounded folded-home recovery is permitted only on '
                        'rough acquisition segment 0 or the final return-home segment')
                if recovery_end < 1 or recovery_end >= len(path):
                    reasons.append(
                        'segment %d bootstrap recovery endpoint is invalid'
                        % segment_index)
                try:
                    evidence = json.loads(evidence_text)
                except (TypeError, json.JSONDecodeError):
                    evidence = None
                    reasons.append(
                        'segment %d bootstrap recovery evidence is invalid'
                        % segment_index)
                if not isinstance(evidence, dict) or not evidence.get('used', False):
                    reasons.append(
                        'segment %d bootstrap recovery evidence is missing'
                        % segment_index)
                else:
                    try:
                        declared_joints = [
                            int(value)
                            for value in evidence.get('joint_numbers', [])]
                        declared_deltas = [
                            float(value)
                            for value in evidence.get('delta_rad', [])]
                        powered_start = evidence.get('powered_start', {})
                        if not isinstance(powered_start, dict):
                            raise TypeError('powered_start is not an object')
                        if bool(powered_start.get('used', False)):
                            powered_start_end = int(
                                powered_start.get('end_point', -1))
                            powered_start_joints = [
                                int(value) for value in
                                powered_start.get('joint_numbers', [])]
                            powered_start_deltas = [
                                float(value) for value in
                                powered_start.get('delta_rad', [])]
                    except (TypeError, ValueError):
                        declared_joints = []
                        declared_deltas = []
                        powered_start_end = -2
                        powered_start_joints = []
                        powered_start_deltas = []
                if powered_start_end >= 0:
                    if not (
                            plan_kind == RETURN_HOME
                            and terminal_home_recovery
                            and 1 <= powered_start_end < recovery_end
                            and recovery_end < len(path) - 1):
                        reasons.append(
                            'segment %d powered-start home recovery targets '
                            'are invalid' % segment_index)
                    if (
                            powered_start_joints != [3]
                            or len(powered_start_deltas) != 1
                            or not math.isfinite(powered_start_deltas[0])
                            or abs(powered_start_deltas[0]) > 0.100001):
                        reasons.append(
                            'segment %d powered-start recovery declaration '
                            'is invalid' % segment_index)
                    else:
                        for detail in bootstrap_recovery_declaration_reasons(
                                path, powered_start_end,
                                powered_start_joints,
                                powered_start_deltas,
                                maximum_delta_rad=0.10):
                            reasons.append(
                                'segment %d powered-start %s'
                                % (segment_index, detail))
                elif not (1 <= recovery_end < len(path) - 1):
                    reasons.append(
                        'segment %d bootstrap recovery endpoint must be an '
                        'internal scheduled planner point'
                        % segment_index)
                if not declared_joints and 1 <= recovery_joint <= 6:
                    declared_joints = [recovery_joint]
                    declared_deltas = [recovery_delta]
                if (
                        not declared_joints
                        or len(declared_joints) > 2
                        or len(declared_joints) != len(declared_deltas)
                        or len(set(declared_joints)) != len(declared_joints)
                        or any(
                            joint < 1 or joint > 6
                            for joint in declared_joints)):
                    reasons.append(
                        'segment %d bootstrap recovery joints are invalid'
                        % segment_index)
                if (
                        any(
                            not math.isfinite(delta)
                            or abs(delta) > 0.150001
                            for delta in declared_deltas)
                        or (
                            len(declared_joints) == 1
                            and (
                                recovery_joint != declared_joints[0]
                                or abs(
                                    recovery_delta
                                    - declared_deltas[0]) > 1e-6))
                        or (
                            len(declared_joints) > 1
                            and (
                                recovery_joint != 0
                                or abs(recovery_delta) > 1e-12))):
                    reasons.append(
                        'segment %d bootstrap recovery deltas are invalid'
                        % segment_index)
                declared_path = (
                    list(reversed(path[recovery_end:]))
                    if terminal_home_recovery else path)
                declared_endpoint = (
                    len(declared_path) - 1
                    if terminal_home_recovery else recovery_end)
                for detail in bootstrap_recovery_declaration_reasons(
                        declared_path, declared_endpoint,
                        declared_joints, declared_deltas):
                    reasons.append('segment %d %s' % (segment_index, detail))
            elif (
                    recovery_joint != 0
                    or not math.isfinite(recovery_delta)
                    or abs(recovery_delta) > 1e-12):
                reasons.append(
                    'segment %d has inconsistent empty bootstrap recovery metadata'
                    % segment_index)
            try:
                validation_step = maximum_step
                if configured_home_direct:
                    validation_step = max(
                        maximum_step,
                        float(np.max(np.abs(path[-1] - path[0]))),
                    )
                emitted_path, emitted_velocities, emitted_accelerations, \
                    emitted_times = validate_sdk_movej_waypoint_path(
                        path,
                        velocities,
                        accelerations,
                        times,
                        command_rate_hz=command_rate,
                        maximum_step_rad=validation_step,
                        velocity_limits_rad_s=(
                            None if configured_home_direct
                            else limits.max_velocity_rad_s),
                        acceleration_limits_rad_s2=(
                            None if configured_home_direct
                            else limits.max_acceleration_rad_s2),
                        speed_percent=execution_speed,
                    )
            except ValueError as error:
                reasons.append(
                    'segment %d SDK MoveJ target validation failed: %s'
                    % (segment_index, error))
                continue
            paths.append([item.copy() for item in emitted_path])
            path_velocities.append([
                item.copy() for item in emitted_velocities])
            path_accelerations.append([
                item.copy() for item in emitted_accelerations])
            path_times.append([float(item) for item in emitted_times])
            targets.append(emitted_path[-1].copy())
            recovery_end_points.append(recovery_end)
            recovery_joints.append(recovery_joint)
            recovery_joint_sets.append(declared_joints)
            powered_start_recovery_end_points.append(powered_start_end)
            powered_start_recovery_joint_sets.append(powered_start_joints)
            startup_home_static_segments.append(startup_home_static)
            configured_home_direct_segments.append(configured_home_direct)
            configured_home_stages.append(configured_home_stage)
            segment_execution_modes.append(sdk_execution_mode)
        if (
                returns_home and targets
                and configured_home_direct_segments != [True]):
            configured_home = np.asarray(
                configured_value(self, 'return_home_positions_rad'),
                dtype=float)
            if configured_home.shape != (6,) or not np.all(
                    np.isfinite(configured_home)):
                reasons.append(
                    'configured return-home pose must contain six finite joints')
            elif float(np.max(np.abs(
                    targets[-1] - configured_home))) > 1e-6:
                reasons.append(
                    'planner return-home endpoint does not match the '
                    'executor configuration')
        if reasons:
            self.invalidate_plan(
                'invalid motion-planner proposal: ' + '; '.join(reasons),
                plan_kind=plan_kind,
                source_request_id=source_request_id,
                plan_id=str(msg.plan_id),
            )
            return
        self.plan_id = str(msg.plan_id)
        self.plan_kind = plan_kind
        self.plan_source_request_id = source_request_id
        self.plan_created = self.now()
        self.plan_targets = targets
        self.plan_paths = paths
        self.plan_path_velocities = path_velocities
        self.plan_path_accelerations = path_accelerations
        self.plan_path_times = path_times
        self.plan_bootstrap_recovery_end_points = recovery_end_points
        self.plan_bootstrap_recovery_joints = recovery_joints
        self.plan_bootstrap_recovery_joint_sets = recovery_joint_sets
        self.plan_powered_start_recovery_end_points = (
            powered_start_recovery_end_points)
        self.plan_powered_start_recovery_joint_sets = (
            powered_start_recovery_joint_sets)
        self.plan_startup_home_static = startup_home_static_segments
        self.plan_configured_home_direct = configured_home_direct_segments
        self.plan_configured_home_stages = configured_home_stages
        self.plan_segment_execution_modes = segment_execution_modes
        self.plan_candidate_count = len(msg.viewpoint_indices)
        self.plan_capture_count = len(msg.viewpoint_indices)
        self.plan_returns_home = bool(returns_home)
        self.plan_target_center = np.asarray([
            msg.target_center.x, msg.target_center.y, msg.target_center.z,
        ], dtype=float)
        if plan_kind == MULTIVIEW_SCAN and self.scan_session_id:
            center, qualified, changed = qualified_scan_center_update(
                getattr(self, 'scan_coverage_target_center', None),
                self.plan_target_center,
                len(getattr(self, 'scan_history', [])),
                getattr(self, 'scan_qualified_target_shape', None),
                getattr(self, 'scan_target_center_qualified', False),
            )
            self.scan_coverage_target_center = center
            self.scan_target_center_qualified = qualified
            if changed:
                self.publish_scan_history()
        self.plan_source_trajectory_sha256 = str(msg.trajectory_sha256)
        self.plan_trajectory_sha256 = str(msg.trajectory_sha256)
        self.plan_motion_limits_sha256 = str(msg.motion_limits_sha256)
        self.plan_execution_speed_percent = float(execution_speed)
        self.plan_collision_model_qualified = bool(msg.collision_model_qualified)
        self.acquisition_scene_snapshot_validated = False
        self.runtime_motion_limits_sha256 = self.plan_motion_limits_sha256
        self.plan_viewpoints = []
        for index, camera, look in zip(
                msg.viewpoint_indices, msg.camera_positions, msg.look_directions):
            ray_id = decoded_ray_id(index)
            self.plan_viewpoints.append({
                'index': int(index),
                'ray_id': ray_id,
                'desired_camera_position': {
                    'x': float(camera.x), 'y': float(camera.y), 'z': float(camera.z),
                },
                'desired_look_at_direction': {
                    'x': float(look.x), 'y': float(look.y), 'z': float(look.z),
                },
            })
        self.current_view = 0
        qualification = 'qualified' if self.plan_collision_model_qualified \
            else 'proposal-only model'
        direct_home_text = bool(
            plan_kind == RETURN_HOME
            and configured_home_direct_segments == [True])
        self.set_state(
            'PROPOSAL_READY',
            '%d six-joint %s %s viewpoints%s use exact %sSDK MoveJ '
            'position targets at %.1f%% (%s); '
            'exact trajectory hash approval required'
            % (
                self.plan_capture_count,
                self.plan_backend or 'planner',
                self.plan_kind.lower(),
                ' plus an approved return-home segment'
                if self.plan_returns_home else '',
                ('configured-home direct self-collision-exempt, external-clearance-checked '
                 if direct_home_text else 'collision-checked '),
                self.plan_execution_speed_percent,
                qualification,
            ))
        self.publish_plan(True, self.reason)

    def tesseract_plan_cb(self, msg):
        """Accept the legacy callback name as a compatibility alias."""
        return ScanViewpointExecutorNode.motion_plan_cb(self, msg)

    def joint_cb(self, msg):
        self.latest_joint_state = msg
        observed_at, telemetry_store = mark_callback_observation(
            self, 'joints')
        if telemetry_store is not None:
            source_stamp_ns, frame_id = message_source_metadata(msg)
            telemetry_store.update_joints(
                msg, received_at=observed_at,
                source_stamp_ns=source_stamp_ns, frame_id=frame_id)

    def arm_status_cb(self, msg):
        self.latest_arm_status = msg
        observed_at, telemetry_store = mark_callback_observation(
            self, 'arm_status')
        if telemetry_store is not None:
            source_stamp_ns, frame_id = message_source_metadata(msg)
            telemetry_store.update_arm_status(
                msg, received_at=observed_at,
                source_stamp_ns=source_stamp_ns, frame_id=frame_id)

    def motion_limits_cb(self, msg):
        # A controller query updates twelve CAN replies independently.  A
        # single 1 Hz publication can therefore be invalid while the next
        # query is completing even though the last fully validated limit set
        # is still inside the bounded freshness window.  Keep that last valid
        # set; if invalid samples persist, its timestamp expires after
        # motion_limits_timeout_sec and every motion gate still fails closed.
        accepted, refreshed = self.motion_limit_stability.observe(
            msg, self.now())
        if accepted is not None:
            self.latest_motion_limits = accepted
        if refreshed:
            observed_at, telemetry_store = mark_callback_observation(
                self, 'motion_limits')
            if telemetry_store is not None:
                source_stamp_ns, frame_id = message_source_metadata(accepted)
                telemetry_store.update_motion_limits(
                    accepted, received_at=observed_at,
                    source_stamp_ns=source_stamp_ns, frame_id=frame_id)
            pending = getattr(self, 'pending_limit_refresh_plan', None)
            if pending is not None:
                if self.now() <= self.pending_limit_refresh_deadline:
                    self.pending_limit_refresh_plan = None
                    self.pending_limit_refresh_deadline = 0.0
                    callback = getattr(self, 'motion_plan_cb', None)
                    if callback is None:
                        callback = self.tesseract_plan_cb
                    callback(pending)
                else:
                    plan_kind = str(pending.plan_kind)
                    source_request_id = str(pending.source_request_id)
                    plan_id = str(pending.plan_id)
                    self.pending_limit_refresh_plan = None
                    self.pending_limit_refresh_deadline = 0.0
                    self.invalidate_plan(
                        'invalid motion-planner proposal: controller motion limits '
                        'did not refresh before the bounded deadline',
                        plan_kind=plan_kind,
                        source_request_id=source_request_id,
                        plan_id=plan_id,
                    )

    def tracking_health_cb(self, msg):
        self.latest_tracking_health = msg
        observed_at, telemetry_store = mark_callback_observation(
            self, 'tracking')
        if telemetry_store is not None:
            source_stamp_ns, frame_id = message_source_metadata(msg)
            telemetry_store.update_tracking(
                msg, received_at=observed_at,
                source_stamp_ns=source_stamp_ns, frame_id=frame_id)

    def tracked_target_cb(self, msg):
        self.latest_tracked_target = msg
        observed_at, telemetry_store = mark_callback_observation(
            self, 'tracked_target')
        if telemetry_store is not None:
            source_stamp_ns, frame_id = message_source_metadata(msg)
            telemetry_store.update_target(
                msg, received_at=observed_at,
                source_stamp_ns=source_stamp_ns, frame_id=frame_id)

    def camera_timestamp_health_cb(self, msg):
        self.latest_camera_timestamp_health = msg
        telemetry_store = getattr(self, 'telemetry_store', None)
        self.mark('camera_clock')
        if telemetry_store is not None:
            observed_at = self.updated['camera_clock']
            source_stamp_ns, frame_id = message_source_metadata(msg)
            telemetry_store.update_camera(
                msg, received_at=observed_at,
                source_stamp_ns=source_stamp_ns, frame_id=frame_id)

    def target_status_cb(self, msg):
        status = str(msg.data).upper()
        self.latest_target_status = status
        observed_at, telemetry_store = mark_callback_observation(
            self, 'target_status')
        if telemetry_store is not None:
            telemetry_store.update_target_status(
                status, received_at=observed_at)

    def target_shape_cb(self, msg):
        """Keep only digest-valid silhouette evidence for the capture gate."""
        try:
            payload = json.loads(msg.data)
            if payload.get('valid') is True:
                payload = validate_shape_measurement(payload)
            else:
                payload = validate_shape_rejection(payload)
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            return
        self.latest_target_shape = payload
        self.latest_target_shape_at = self.now()
        self.latest_target_shape_stamp_ns = stamp_nanoseconds(
            payload['header']['stamp'])

    def obstacle_cb(self, msg):
        self.latest_obstacles = msg
        observed_at, telemetry_store = mark_callback_observation(
            self, 'obstacles')
        if telemetry_store is not None:
            source_stamp_ns, frame_id = message_source_metadata(msg)
            telemetry_store.update_obstacles(
                msg, received_at=observed_at,
                source_stamp_ns=source_stamp_ns, frame_id=frame_id)

    def workflow_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            return
        self.latest_workflow = payload
        observed_at, telemetry_store = mark_callback_observation(
            self, 'workflow')
        if telemetry_store is not None:
            telemetry_store.update_workflow(
                payload, received_at=observed_at)
        session_id = str(payload.get('session_id', ''))
        if session_id and session_id != self.scan_session_id:
            self.scan_session_id = session_id
            self.scan_history = []
            self.scan_rejections = []
            self.scan_qualified_target_shape = None
            self.pending_scan_qualified_target_shape = None
            self.scan_qualified_target_model_seed = None
            self.pending_scan_qualified_target_model_seed = None
            self.latest_achieved_scan_view = None
            self.scan_coverage_target_center = None
            self.scan_target_center_qualified = False
            self.publish_scan_history()

    def heavy_refresh_status_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            return
        if self.state == 'WAITING_FOR_CAPTURE_REFRESH':
            if not self.capture_heavy_refresh_request_id:
                return
            action, reason, _image_stamp_ns = heavy_refresh_status_action(
                payload,
                self.capture_heavy_refresh_request_id,
                self.capture_heavy_refresh_min_image_stamp_ns,
            )
            if action == 'idle':
                if self.capture_heavy_refresh_waiting_for_worker:
                    if self.capture_heavy_refresh_publish_attempts >= 2:
                        self.abort_motion(
                            'capture heavy worker stayed busy after one '
                            'bounded retry')
                        return
                    self.capture_heavy_refresh_waiting_for_worker = False
                    self.publish_capture_heavy_refresh()
                return
            if action in (
                    'ignore', 'waiting_for_frame', 'waiting_for_worker',
                    'queued'):
                self.capture_heavy_refresh_waiting_for_worker = False
                return
            if action == 'detected':
                self.rgbd_capture_future = None
                # The correlated refresh starts a new evidence epoch.  The
                # bounded readiness retries consumed while waiting for the
                # pre-refresh mask/depth status must not make the first
                # refreshed capture fail immediately.  Keep the refresh ID so
                # a second heavy refresh is still impossible for this view.
                self.rgbd_capture_attempts = 0
                self.set_state(
                    'CAPTURING_RGBD',
                    'matching heavy refresh found the target; validating '
                    'the same achieved viewpoint again')
                return
            if action in ('not_found', 'not_found_clear'):
                self.reject_achieved_capture_view(
                    self.capture_rejection_reason or reason)
                return
            if action == 'busy':
                if self.capture_heavy_refresh_publish_attempts >= 2:
                    self.abort_motion(
                        'capture heavy worker stayed busy after one bounded '
                        'retry')
                    return
                self.capture_heavy_refresh_waiting_for_worker = True
                self.set_state(
                    'WAITING_FOR_CAPTURE_REFRESH',
                    'heavy perception worker is busy; holding the achieved '
                    'pose for one bounded refresh retry')
                return
            if action == 'abort':
                self.abort_motion(
                    'capture heavy perception refresh failed: ' + reason)
            return

        acquisition_wait_states = (
            'WAITING_FOR_FRESH_FRAME',
            'WAITING_FOR_GROUNDING_DINO',
            'WAITING_FOR_TRACKING_LOCK',
        )
        if not self.is_acquisition():
            return
        if self.state == 'WAITING_FOR_RUNTIME_REFRESH':
            if (
                    self.runtime_refresh_resume_state
                    not in acquisition_wait_states
                    or not self.acquisition_request_id):
                return
            action, _reason, _image_stamp_ns = heavy_refresh_status_action(
                payload,
                self.acquisition_request_id,
                self.acquisition_min_image_stamp_ns,
            )
            if action not in ('ignore', 'idle'):
                # The heavy worker publishes each correlated terminal result
                # once. Preserve it while motion remains blocked by an
                # unrelated runtime freshness hold, then revalidate and replay
                # it only after that safety gate recovers.
                self.pending_acquisition_heavy_status = payload
            return
        if self.state not in acquisition_wait_states:
            return
        state = str(payload.get('state', '')).lower()
        if (
                state == 'idle'
                and self.state == 'WAITING_FOR_FRESH_FRAME'
                and self.acquisition_waiting_for_worker):
            if self.acquisition_request_attempt >= 2:
                self.abort_motion(
                    'GroundingDINO worker stayed busy after one bounded retry')
                return
            self.acquisition_waiting_for_worker = False
            self.publish_acquisition_refresh()
            return
        if not self.acquisition_request_id:
            return
        action, reason, image_stamp_ns = heavy_refresh_status_action(
            payload,
            self.acquisition_request_id,
            self.acquisition_min_image_stamp_ns,
        )
        if action in ('ignore', 'idle'):
            return
        if action == 'busy':
            self.acquisition_waiting_for_worker = True
            self.set_state(
                'WAITING_FOR_FRESH_FRAME',
                'GroundingDINO worker is busy; waiting for one bounded retry')
            return
        if action == 'waiting_for_frame':
            self.set_state(
                'WAITING_FOR_FRESH_FRAME',
                'waiting for a new camera frame captured after this settled look')
            return
        if action == 'waiting_for_worker':
            self.set_state(
                'WAITING_FOR_FRESH_FRAME',
                'GroundingDINO worker is busy; the correlated request is '
                'retained until it becomes available')
            return
        if action == 'queued':
            self.acquisition_job_image_stamp_ns = int(image_stamp_ns)
            self.acquisition_job_started = self.now()
            self.set_state(
                'WAITING_FOR_GROUNDING_DINO',
                'matching GroundingDINO job queued on a fresh post-settle frame')
            return
        if action == 'too_far':
            self.acquisition_job_image_stamp_ns = int(image_stamp_ns)
            self.acquisition_detection_completed = self.now()
            self.command_target = None
            self.current_path = []
            self.current_path_times = []
            self.motion_started_at = None
            self.publish_hold()
            self.set_state('ACQUISITION_TARGET_TOO_FAR', reason)
            return
        if action == 'detected':
            self.acquisition_job_image_stamp_ns = int(image_stamp_ns)
            self.acquisition_detection_completed = self.now()
            self.set_state(
                'WAITING_FOR_TRACKING_LOCK',
                'GroundingDINO found the target; waiting for a new measured LOCKED state')
            return
        if action in ('not_found', 'not_found_clear'):
            self.get_logger().info(
                '%s at acquisition look %d; waiting for its correlated '
                'obstacle scene before advancing'
                % (reason, self.current_view + 1))
            self.acquisition_job_image_stamp_ns = int(image_stamp_ns)
            self.acquisition_detection_completed = self.now()
            if action == 'not_found_clear':
                clear_scene = ObstacleInstance3DArray()
                clear_scene.header.stamp.sec = (
                    self.acquisition_job_image_stamp_ns // 1_000_000_000)
                clear_scene.header.stamp.nanosec = (
                    self.acquisition_job_image_stamp_ns % 1_000_000_000)
                clear_scene.header.frame_id = 'base_link'
                clear_scene.scene_blocked = False
                clear_scene.blocking_reason = (
                    'clear:correlated_post_settle_semantic_result')
                self.obstacle_cb(clear_scene)
                self.acquisition_scene_snapshot_validated = True
                self.get_logger().info(
                    'correlated post-settle result contains zero obstacles; '
                    'advancing directly to the next approved acquisition look')
                self.advance_acquisition_view()
                return
            self.set_state(
                'WAITING_FOR_OBSTACLE_SCENE',
                'target absent; waiting for the matching post-settle semantic scene')
            return
        if action == 'abort':
            self.abort_motion('GroundingDINO acquisition failed: ' + reason)

    def tick(self):
        if self.pending_limit_refresh_plan is not None:
            if self.now() > self.pending_limit_refresh_deadline:
                pending = self.pending_limit_refresh_plan
                self.pending_limit_refresh_plan = None
                self.pending_limit_refresh_deadline = 0.0
                self.invalidate_plan(
                    'invalid motion-planner proposal: controller motion limits '
                    'did not refresh before the bounded deadline',
                    plan_kind=str(pending.plan_kind),
                    source_request_id=str(pending.source_request_id),
                    plan_id=str(pending.plan_id),
                )
            return
        if self.state == 'FINISHING_WORKFLOW':
            self.finish_workflow_tick()
            return
        if self.state in ACTIVE_STATES:
            self.execution_tick()
            return
        if self.state == 'PROPOSAL_READY':
            telemetry_store = getattr(self, 'telemetry_store', None)
            if telemetry_store is None:
                camera_health = self.latest_camera_timestamp_health
                camera_fresh = self.fresh('camera_clock')
            else:
                snapshot = telemetry_store.snapshot()
                observation = snapshot.perception.camera
                camera_health = (
                    None if observation is None else observation.value)
                camera_fresh = bool(
                    observation is not None
                    and not observation.is_stale_at(
                        snapshot.captured_at,
                        float(configured_value(self, 'data_timeout_sec'))))
            if (
                    not self.is_configured_home_direct()
                    and (
                    not camera_fresh
                    or camera_health is None
                    or not camera_health.healthy)):
                # Approval repeats the fresh camera-clock gate. Preserve the
                # immutable proposal while that transient blocks approval so a
                # confirmation dialog cannot race a replacement plan/hash.
                return
            if self.now() - self.plan_created > float(
                    configured_value(self, 'plan_max_age_sec')):
                self.invalidate_plan('proposal expired; refresh viewpoints')
            return

    def approve_cb(self, request, response):
        expected = str(configured_value(self, 'approval_confirmation'))
        mission_confirmation = 'MISSION_POLICY:' + self.mission_sha256
        mission_authorization_requested = (
            str(request.confirmation) == mission_confirmation)
        mission_authorization_granted = (
            self.mission_authorization_valid()
            and str(getattr(self, 'plan_backend', '')).strip().lower() ==
            str(getattr(
                self, 'mission_planner_backend', '')).strip().lower()
            if mission_authorization_requested else True)
        if mission_authorization_requested and mission_authorization_granted:
            expected = mission_confirmation
        authorization_request = PlanAuthorizationRequest(
            state=self.state,
            loaded_plan_id=self.plan_id if self.plan_targets else '',
            requested_plan_id=str(request.plan_id),
            confirmation=str(request.confirmation),
            expected_confirmation=expected,
            real_motion_enabled=self.real_motion_enabled(),
            plan_age_sec=self.now() - self.plan_created,
            plan_max_age_sec=float(
                configured_value(self, 'plan_max_age_sec')),
            loaded_trajectory_sha256=self.plan_trajectory_sha256,
            requested_trajectory_sha256=str(request.trajectory_sha256),
            mission_authorization_required=mission_authorization_requested,
            mission_authorization_granted=mission_authorization_granted,
        )
        authorizer = getattr(self, 'plan_authorizer', PlanAuthorizer())
        authorization = authorizer.evaluate(authorization_request)
        if not authorization.permitted:
            response.accepted = False
            response.message = authorization.detail
            return response
        acquisition = self.is_acquisition()
        return_only = self.is_return_home()
        approval_mode = (
            SafetyMode.ACQUISITION_APPROVAL if acquisition else
            SafetyMode.RETURN_HOME if return_only else
            SafetyMode.SCAN_APPROVAL)
        reasons = self.runtime_reasons(runtime_gate_policy(
            approval_mode,
            require_settled=True,
            obstacle_authority=(
                ObstacleAuthority.STATIC_BOOTSTRAP
                if acquisition or return_only else
                ObstacleAuthority.LIVE),
        ))
        if reasons:
            response.accepted = False
            response.message = 'execution blocked: ' + '; '.join(reasons)
            return response
        if not self.plan_collision_model_qualified:
            response.accepted = False
            response.message = (
                'planner collision model is proposal-only and not qualified for hardware')
            return response
        target_required = not (acquisition or return_only)
        latest_target_center = None
        target_drift = None
        if target_required:
            telemetry_store = getattr(self, 'telemetry_store', None)
            if telemetry_store is None:
                latest_scan = self.latest_scan
            else:
                observation = telemetry_store.snapshot().mission.reachable_scan
                latest_scan = (
                    None if observation is None else observation.value)
            latest_target_center = self.vector(
                latest_scan.get('target_object_center')
                if isinstance(latest_scan, dict) else None)
            if latest_target_center is not None and self.plan_target_center is not None:
                target_drift = float(np.linalg.norm(
                    latest_target_center - self.plan_target_center))
        unavailable_dependencies = []
        if target_required and self.param_bool('auto_capture'):
            if not self.capture_client.service_is_ready():
                unavailable_dependencies.append('capture service')
            elif not self.rgbd_capture_client.service_is_ready():
                unavailable_dependencies.append('RGB-D capture service')
            elif not self.finish_scan_client.service_is_ready():
                unavailable_dependencies.append('workflow finish service')
        authorization = authorizer.evaluate(
            PlanAuthorizationRequest(
                **dict(
                    authorization_request.__dict__,
                    target_required=target_required,
                    target_available=bool(
                        latest_target_center is not None
                        and self.plan_target_center is not None),
                    target_drift_m=target_drift,
                    maximum_target_drift_m=float(configured_value(
                        self, 'max_target_drift_before_approval_m')),
                    allow_target_motion=bool(configured_value(
                        self, 'allow_target_motion_during_scan')),
                    unavailable_dependencies=tuple(
                        unavailable_dependencies),
                )))
        if not authorization.permitted:
            response.accepted = False
            response.message = authorization.detail
            return response
        path_reasons = self.prepare_current_view()
        if path_reasons:
            authorization = authorizer.evaluate(
                PlanAuthorizationRequest(
                    **dict(
                        authorization_request.__dict__,
                        path_reasons=tuple(path_reasons),
                    )))
            response.accepted = False
            response.message = authorization.detail
            return response
        self.abort_return_in_progress = False
        self.abort_return_reason = ''
        self.abort_return_bootstrap_static_scene = False
        start_joints = self.current_joints().copy()
        if (
                acquisition
                or not self.retrace_joint_targets
                or float(np.max(np.abs(
                    start_joints - self.retrace_joint_targets[-1]))) > float(
                        configured_value(self, 'plan_start_tolerance_rad'))):
            # An uninterrupted acquisition+scan session can safely retrace
            # every already executed, separately approved endpoint to the
            # original loaded pose. A restarted executor falls back to the
            # current scan-start pose.
            self.retrace_joint_targets = [start_joints]
        start_reason = (
            'approved rough-target acquisition motion started'
            if acquisition else (
                'approved dedicated configured-home motion started'
                if return_only else 'approved scan motion started'))
        self.begin_runtime_refresh(
            start_reason,
            require_workflow=not (acquisition or return_only),
            allow_missing_obstacles=(
                acquisition or self.is_startup_home_static()),
        )
        response.accepted = True
        response.message = (
            'approved %d %s viewpoints at no more than %.1f%% speed; '
            'waiting for fresh runtime telemetry before motion'
        ) % (
            self.plan_capture_count,
            'acquisition' if acquisition else (
                'return-home' if return_only else 'scan'),
            self.execution_speed_percent(),
        )
        return response

    def authorize_mission_cb(self, request, response):
        if request.revoke:
            if (
                    self.mission_task_id
                    and str(request.task_id) not in ('', self.mission_task_id)):
                response.accepted = False
                response.message = 'mission task ID does not match the active authorization'
                return response
            self.mission_task_id = ''
            self.mission_sha256 = ''
            self.mission_planner_backend = ''
            self.mission_expires_at_sec = 0.0
            response.accepted = True
            response.message = 'mission authorization revoked'
            return response
        if not self.param_bool('allow_mission_policy'):
            response.accepted = False
            response.message = 'MISSION_POLICY authorization is disabled'
            return response
        task_id = str(request.task_id).strip()
        digest = str(request.mission_sha256).strip()
        planner_backend = str(request.planner_backend).strip().lower()
        expires = float(request.expires_at.sec) + float(
            request.expires_at.nanosec) * 1e-9
        now = time.time()
        if not task_id or len(digest) != 64 or any(
                character not in '0123456789abcdef' for character in digest):
            response.accepted = False
            response.message = 'mission identity or SHA-256 is invalid'
            return response
        if planner_backend not in ('tesseract', 'curobo'):
            response.accepted = False
            response.message = 'mission planner backend is invalid'
            return response
        if expires <= now or expires - now > 1200.5:
            response.accepted = False
            response.message = 'mission authorization expiry is invalid'
            return response
        if self.state in ACTIVE_STATES:
            response.accepted = False
            response.message = 'cannot replace mission authorization while motion is active'
            return response
        self.mission_task_id = task_id
        self.mission_sha256 = digest
        self.mission_planner_backend = planner_backend
        self.mission_expires_at_sec = expires
        response.accepted = True
        response.message = 'mission policy authorization bound to task and deadline'
        return response

    def execute_home_stage_cb(self, request, response):
        """Execute one mission-authorized configured MoveJ stage directly."""
        response.accepted = False
        response.execution_id = ''
        if self.state in ACTIVE_STATES:
            response.message = 'executor is already running a motion transaction'
            return response
        if not self.real_motion_enabled() or self.command_pub is None:
            response.message = 'real arm motion is not enabled in the executor'
            return response
        if not self.mission_authorization_valid():
            response.message = 'mission authorization is missing or expired'
            return response
        if (str(request.task_id) != self.mission_task_id
                or str(request.mission_sha256) != self.mission_sha256):
            response.message = 'direct home mission identity does not match'
            return response
        try:
            current = self.current_joints()
        except ValueError as exc:
            response.message = 'direct home has no valid current joints: %s' % exc
            return response
        goal = np.asarray(request.joint_goal_positions_rad, dtype=float)
        rejection = direct_home_stage_rejection(
            request.home_stage,
            goal,
            current,
            self.configuration.motion.return_home_positions_rad,
            self.joint_limits,
            unchanged_tolerance_rad=(
                self.configuration.motion.plan_start_tolerance_rad),
            start_limit_tolerance_rad=(
                self.configuration.safety.
                configured_home_feedback_limit_tolerance_rad),
            pre_home=self.configuration.motion.pre_home_positions_rad,
        )
        if rejection:
            response.message = rejection
            return response
        motor_reasons = self.arm_status_reasons()
        if motor_reasons:
            response.message = 'direct home motor gate: ' + '; '.join(
                motor_reasons)
            return response
        stage = str(request.home_stage).strip().upper()
        execution_id = 'direct-home-%d' % time.time_ns()
        self.clear_plan()
        self.plan_id = execution_id
        self.plan_kind = RETURN_HOME
        self.plan_created = time.time()
        stage_targets = direct_home_stage_targets(stage, current, goal)
        direct_path = [current.copy()] + [
            item.copy() for item in stage_targets]
        external_reasons = self.validate_attached_tool_external_path(
            direct_path, [])
        if external_reasons:
            response.message = (
                'direct home attached-tool clearance gate: '
                + '; '.join(external_reasons))
            return response
        zeros = np.zeros(6, dtype=float)
        self.plan_targets = [goal.copy()]
        self.plan_paths = [[item.copy() for item in direct_path]]
        self.plan_path_velocities = [[
            zeros.copy() for _item in direct_path]]
        self.plan_path_accelerations = [[
            zeros.copy() for _item in direct_path]]
        self.plan_path_times = [[0.0 for _item in direct_path]]
        self.plan_configured_home_direct = [True]
        self.plan_configured_home_stages = [stage]
        self.plan_capture_count = 0
        self.plan_returns_home = True
        self.plan_execution_speed_percent = self.speed_percent()
        # Direct configured home uses the driver's bounded SDK MoveJ speed and
        # the exact configured joint endpoint. Planner timing/limit hashes do
        # not own this transaction; hard joint bounds, live motor authority and
        # attached-tool external clearance remain mandatory.
        self.plan_motion_limits_sha256 = ''
        self.runtime_motion_limits_sha256 = ''
        self.plan_collision_model_qualified = False
        self.current_view = 0
        self.current_path = [item.copy() for item in stage_targets]
        self.current_path_velocities = [zeros.copy() for _item in stage_targets]
        self.current_path_accelerations = [zeros.copy() for _item in stage_targets]
        self.current_path_times = [0.0 for _item in stage_targets]
        self.current_path_streaming = False
        self.path_index = 0
        self.command_target = None
        self.command_sent_at = 0.0
        self.command_samples_sent = 0
        self.max_command_interval_sec = 0.0
        self.dropped_command_samples = 0
        self.motion_started_at = None
        self.waypoint_started_at = None
        self.waypoint_last_progress_at = None
        self.waypoint_best_error = math.inf
        self.current_waypoint_error = 0.0
        self.max_waypoint_error = 0.0
        self.begin_runtime_refresh(
            'approved direct %s MoveJ endpoint started' % stage,
            require_workflow=False,
            allow_missing_obstacles=True,
        )
        response.accepted = True
        response.execution_id = execution_id
        response.message = (
            'direct %s endpoint accepted without motion planning' % stage)
        return response

    def mission_authorization_valid(self):
        return (
            self.param_bool('allow_mission_policy')
            and bool(self.mission_task_id)
            and len(self.mission_sha256) == 64
            and self.mission_planner_backend in ('tesseract', 'curobo')
            and time.time() < self.mission_expires_at_sec
        )

    def cancel_cb(self, _request, response):
        reason = 'operator cancelled scan execution'
        if terminal_home_hold_required(self.state, self.reason):
            held = self.publish_hold()
            response.success = bool(held)
            response.message = (
                'proposal cancelled; configured home reached; current joint '
                'hold requested'
                if held else
                'proposal cancelled; configured home reached but current '
                'joint hold was unavailable')
            return response
        held = self.publish_hold()
        self._terminal_abort(
            reason + '; current joint hold requested before dedicated '
            'current-state return-home replanning')
        response.success = bool(held)
        response.message = (
            'proposal cancelled; current joint hold requested before '
            'dedicated current-state return-home replanning'
            if held else
            'proposal cancelled; current joint hold was unavailable'
        )
        return response

    def hold_cb(self, _request, response):
        """Request an energized hold without cancelling or changing plan state."""
        held = self.publish_hold()
        response.success = bool(held)
        response.message = (
            'current joint hold requested'
            if held else 'current joint hold was unavailable')
        return response

    def refresh_cb(self, _request, response):
        if self.state in ACTIVE_STATES:
            response.success = False
            response.message = 'cannot refresh a plan while motion is active; cancel first'
            return response
        self.clear_plan()
        self.state = 'IDLE'
        response.success = True
        response.message = 'latest viewpoints will be replanned'
        return response

    def diagnostic_state_cb(self, _request, response):
        """Expose command-free executor state when Foxy topic echo is unreliable."""
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            camera_health = self.latest_camera_timestamp_health
            camera_fresh = self.fresh('camera_clock')
        else:
            telemetry_snapshot = telemetry_store.snapshot()
            camera_observation = telemetry_snapshot.perception.camera
            camera_health = (
                None if camera_observation is None
                else camera_observation.value)
            camera_fresh = bool(
                camera_observation is not None
                and not camera_observation.is_stale_at(
                    telemetry_snapshot.captured_at,
                    float(configured_value(self, 'data_timeout_sec'))))
        first_delta = None
        first_goal = []
        if self.plan_paths and len(self.plan_paths[0]) >= 2:
            first_delta = float(np.max(np.abs(
                self.plan_paths[0][-1] - self.plan_paths[0][0])))
            first_goal = [float(value) for value in self.plan_paths[0][-1]]
        response.success = True
        response.message = json.dumps({
            'state': self.state,
            'reason': self.reason,
            'plan_id': self.plan_id,
            'plan_kind': self.plan_kind,
            'source_trajectory_sha256': self.plan_source_trajectory_sha256,
            'trajectory_sha256': self.plan_trajectory_sha256,
            'execution_speed_percent': self.plan_execution_speed_percent,
            'trajectory_command_rate_hz': float(
                configured_value(self, 'trajectory_command_rate_hz')),
            'motion_adapter': EXECUTION_TIMING_POLICY_VERSION,
            'executor_tick_rate_hz': float(
                configured_value(self, 'executor_tick_rate_hz')),
            'command_samples_sent': self.command_samples_sent,
            'dropped_command_samples': self.dropped_command_samples,
            'max_command_interval_sec': self.max_command_interval_sec,
            'current_stream_planned_duration_sec': (
                float(self.current_path_times[-1])
                if self.current_path_streaming and self.current_path_times
                else 0.0),
            'current_stream_elapsed_sec': (
                max(0.0, self.now() - float(
                    self.stream_wall_started_at
                    if self.stream_wall_started_at is not None
                    else self.motion_started_at))
                if self.current_path_streaming
                and self.motion_started_at is not None else 0.0),
            'current_stream_late_tick_count': self.stream_late_event_count,
            'current_stream_max_lateness_sec': self.stream_late_max_sec,
            'last_stream_planned_duration_sec': (
                self.last_stream_planned_duration_sec),
            'last_stream_actual_duration_sec': (
                self.last_stream_actual_duration_sec),
            'last_stream_achieved_rate_hz': (
                self.last_stream_achieved_rate_hz),
            'current_waypoint_error_rad': self.current_waypoint_error,
            'max_waypoint_error_rad': self.max_waypoint_error,
            'planned_viewpoints': self.plan_capture_count,
            'scan_session_id': self.scan_session_id,
            'session_accepted_views': len(self.scan_history),
            'session_remaining_views': max(
                0,
                int(configured_value(self, 'max_execution_viewpoints'))
                - len(self.scan_history)),
            'returns_home': self.plan_returns_home,
            'first_view_max_joint_delta_rad': first_delta,
            'first_view_goal_rad': first_goal,
            'collision_model_qualified': self.plan_collision_model_qualified,
            'real_motion_enabled': self.real_motion_enabled(),
            'mission_policy_enabled': self.param_bool('allow_mission_policy'),
            'mission_task_id': self.mission_task_id,
            'mission_authorization_valid': self.mission_authorization_valid(),
            'camera_clock_fresh': camera_fresh,
            'camera_clock_healthy': bool(
                camera_health is not None and camera_health.healthy),
        }, sort_keys=True)
        return response

    def execution_tick(self):
        if self.state == 'WAITING_FOR_RUNTIME_REFRESH':
            self.waiting_for_runtime_refresh_tick()
            return
        telemetry_store = getattr(self, 'telemetry_store', None)
        telemetry_snapshot = (
            None if telemetry_store is None else telemetry_store.snapshot())
        if telemetry_snapshot is None:
            obstacles = self.latest_obstacles
        else:
            observation = telemetry_snapshot.perception.obstacles
            obstacles = None if observation is None else observation.value
        runtime_mode = (
            SafetyMode.RETURN_HOME if self.returning_home() else
            SafetyMode.ACQUISITION_MOTION
            if self.is_acquisition() else
            SafetyMode.SCAN_CAPTURE if self.state in (
                'SETTLING', 'CAPTURING', 'CAPTURING_RGBD',
                'WAIT_CAPTURE', 'WAITING_FOR_CAPTURE_REFRESH') else
            SafetyMode.SCAN_MOTION)
        static_scene = bool(
            missing_obstacles_can_wait(
                self.plan_kind, self.current_view, self.state,
                getattr(
                    self, 'abort_return_bootstrap_static_scene', False))
            or self.is_return_home())
        approved_scene = bool(
            (self.is_acquisition()
             and self.acquisition_scene_snapshot_validated)
            or getattr(self, 'abort_return_bootstrap_static_scene', False)
            or self.is_startup_home_static()
            or approved_multiview_motion_obstacle_snapshot(
                self.plan_kind, self.state,
                self.plan_collision_model_qualified, obstacles)
            or approved_return_home_obstacle_snapshot(
                self.returning_home(),
                self.plan_collision_model_qualified, obstacles))
        scene_authority = (
            ObstacleAuthority.STATIC_BOOTSTRAP if static_scene else
            ObstacleAuthority.APPROVED_SNAPSHOT if approved_scene else
            ObstacleAuthority.LIVE)
        reasons = self.runtime_reasons(
            runtime_gate_policy(
                runtime_mode, obstacle_authority=scene_authority),
            telemetry_snapshot=telemetry_snapshot)
        if reasons:
            recovery_policy = getattr(
                self, 'recovery_policy', RecoveryPolicy())
            recovery = recovery_policy.decide(
                RecoveryContext.RUNTIME,
                tuple(as_failure(reason) for reason in reasons),
            )
            if recovery.action is RecoveryAction.RETRY:
                self.begin_runtime_recovery(reasons)
            elif self.returning_home():
                ScanViewpointExecutorNode.handle_return_home_failure(
                    self,
                    'return-home runtime safety gate stopped motion: '
                    + '; '.join(reasons))
            else:
                self.abort_motion(
                    'runtime safety gate: ' + '; '.join(reasons))
            return
        if self.state == 'MOVING':
            self.moving_tick()
        elif self.state == 'SETTLING':
            self.settling_tick()
        elif self.state == 'SETTLING_HOME':
            self.return_home_settling_tick()
        elif self.state == 'CAPTURING':
            self.capturing_tick()
        elif self.state == 'CAPTURING_RGBD':
            self.capturing_rgbd_tick()
        elif self.state == 'WAIT_CAPTURE':
            self.wait_capture_tick()
        elif self.state == 'WAITING_FOR_CAPTURE_REFRESH':
            self.waiting_for_capture_refresh_tick()
        elif self.state == 'WAITING_FOR_FRESH_FRAME':
            self.waiting_for_fresh_frame_tick()
        elif self.state == 'WAITING_FOR_GROUNDING_DINO':
            self.waiting_for_grounding_dino_tick()
        elif self.state == 'WAITING_FOR_TRACKING_LOCK':
            self.waiting_for_tracking_lock_tick()
        elif self.state == 'WAITING_FOR_OBSTACLE_SCENE':
            self.waiting_for_obstacle_scene_tick()

    def begin_runtime_refresh(
            self, motion_reason, require_workflow,
            allow_missing_obstacles):
        self.runtime_refresh_resume_state = ''
        self.runtime_recovery_started_at = None
        self.pending_motion_reason = str(motion_reason)
        self.runtime_refresh_require_workflow = bool(require_workflow)
        self.runtime_refresh_allow_missing_obstacles = bool(
            allow_missing_obstacles)
        self.set_state(
            'WAITING_FOR_RUNTIME_REFRESH',
            'trajectory validated; waiting for fresh runtime telemetry '
            'before motion')

    def begin_runtime_recovery(self, reasons):
        """Hold an in-flight exact target through a bounded DDS freshness gap."""
        resume_state = str(self.state)
        self.publish_hold()
        # The original command_target must remain intact so it can be
        # reissued after telemetry recovers.  Reset only the settle evidence:
        # the bounded recovery gate proves the newly commanded current-pose
        # hold, not convergence to the interrupted endpoint.
        self.settle_position_anchor = None
        self.settle_last_joint_update = -1e9
        self.settle_last_sample_ok = False
        self.runtime_refresh_resume_state = resume_state
        clock = getattr(self, 'now', None)
        self.runtime_recovery_started_at = (
            clock() if callable(clock) else None)
        self.runtime_refresh_require_workflow = False
        # Preserve the exact scene authority that applied to the interrupted
        # state.  In particular, rough-acquisition look zero is deliberately
        # planned against bootstrap_static and has no perception obstacle
        # stream yet.  A transient motion-limit refresh must not manufacture
        # an obstacle dependency while the arm is held.
        self.runtime_refresh_allow_missing_obstacles = (
            missing_obstacles_can_wait(
                self.plan_kind,
                self.current_view,
                resume_state,
                getattr(
                    self, 'abort_return_bootstrap_static_scene', False),
            ) or self.is_return_home()
        )
        self.pending_motion_reason = (
            'fresh runtime telemetry restored; resuming the same approved '
            'planner target')
        self.set_state(
            'WAITING_FOR_RUNTIME_REFRESH',
            'holding current position while runtime telemetry refreshes: '
            + '; '.join(reasons))

    def waiting_for_runtime_refresh_tick(self):
        recovering = bool(self.runtime_refresh_resume_state)
        return_home = getattr(self, 'plan_kind', '') == RETURN_HOME
        telemetry_store = getattr(self, 'telemetry_store', None)
        telemetry_snapshot = (
            None if telemetry_store is None else telemetry_store.snapshot())
        if telemetry_snapshot is None:
            obstacles = getattr(self, 'latest_obstacles', None)
        else:
            observation = telemetry_snapshot.perception.obstacles
            obstacles = None if observation is None else observation.value
        refresh_mode = (
            SafetyMode.RETURN_HOME if return_home else
            SafetyMode.ACQUISITION_MOTION
            if self.is_acquisition() else
            SafetyMode.SCAN_APPROVAL
            if self.runtime_refresh_require_workflow else
            SafetyMode.SCAN_CAPTURE if str(
                self.runtime_refresh_resume_state) in (
                    'SETTLING', 'CAPTURING', 'CAPTURING_RGBD',
                    'WAIT_CAPTURE') else
            SafetyMode.SCAN_MOTION)
        static_scene = bool(
            self.runtime_refresh_allow_missing_obstacles or return_home)
        approved_scene = bool(
            (self.is_acquisition()
             and self.acquisition_scene_snapshot_validated)
            or approved_return_home_obstacle_snapshot(
                self.returning_home(),
                getattr(self, 'plan_collision_model_qualified', False),
                obstacles))
        scene_authority = (
            ObstacleAuthority.STATIC_BOOTSTRAP if static_scene else
            ObstacleAuthority.APPROVED_SNAPSHOT if approved_scene else
            ObstacleAuthority.LIVE)
        reasons = self.runtime_reasons(
            runtime_gate_policy(
                refresh_mode,
                require_settled=True,
                obstacle_authority=scene_authority,
                settle_at_current_hold=recovering),
            telemetry_snapshot=telemetry_snapshot)
        timeout = float(configured_value(
            self,
            'runtime_recovery_timeout_sec'
            if recovering else 'runtime_refresh_timeout_sec'))
        action = runtime_refresh_action(
            reasons, self.now() - self.state_started, timeout)
        if action == 'wait':
            return
        if action == 'abort':
            message = (
                'fresh runtime telemetry did not arrive within %.1f seconds '
                'after trajectory validation: %s'
                % (timeout, '; '.join(reasons)))
            if self.returning_home():
                ScanViewpointExecutorNode.handle_return_home_failure(
                    self, message)
            else:
                self.abort_motion(message)
            return
        reason = self.pending_motion_reason or (
            'fresh runtime telemetry received; approved motion started')
        self.pending_motion_reason = ''
        resume_state = self.runtime_refresh_resume_state
        self.runtime_refresh_resume_state = ''
        if resume_state:
            recovery_duration = 0.0
            recovery_started = getattr(
                self, 'runtime_recovery_started_at', None)
            if recovery_started is not None:
                recovery_duration = max(
                    0.0, self.now() - recovery_started)
            self.runtime_recovery_started_at = None
            self.set_state(resume_state, reason)
            pending_heavy_status = self.pending_acquisition_heavy_status
            self.pending_acquisition_heavy_status = None
            if (
                    pending_heavy_status is not None
                    and resume_state in (
                        'WAITING_FOR_FRESH_FRAME',
                        'WAITING_FOR_GROUNDING_DINO',
                        'WAITING_FOR_TRACKING_LOCK',
                    )):
                replay = String()
                replay.data = json.dumps(
                    pending_heavy_status, sort_keys=True)
                self.heavy_refresh_status_cb(replay)
            if resume_state == 'MOVING' and self.command_target is not None:
                now = self.now()
                if (
                        getattr(self, 'current_path_streaming', False)
                        and self.motion_started_at is not None):
                    self.motion_started_at += recovery_duration
                self.publish_joint_command(self.command_target)
                self.command_samples_sent += 1
                self.command_sent_at = now
                self.waypoint_started_at = now
                self.waypoint_last_progress_at = now
                self.waypoint_best_error = self.total_joint_error(
                    self.command_target)
            return
        self.set_state('MOVING', reason)

    def moving_tick(self):
        now = self.now()
        freshness_check = getattr(self, 'fresh', None)
        if callable(freshness_check) and not freshness_check(
                'joints', float(configured_value(self, 'home_joint_feedback_timeout_sec'))):
            self.abort_or_finish_captures(
                'joint feedback became invalid during SDK MoveJ: no fresh '
                'application-level sample')
            return
        if getattr(self, 'current_path_streaming', False):
            ScanViewpointExecutorNode.streaming_moving_tick(self, now)
            return
        ScanViewpointExecutorNode.feedback_gated_moving_tick(self, now)

    def feedback_gated_moving_tick(self, now):
        """Execute the dedicated direct-home endpoint transaction."""
        if self.command_target is None:
            if self.path_index >= len(self.current_path):
                if self.returning_home():
                    self.begin_return_home_settle()
                    return
                self.settle_started = None
                self.settle_position_anchor = None
                self.settle_last_sample_ok = False
                self.set_state(
                    'SETTLING',
                    'acquisition look reached; waiting for arm and camera to settle'
                    if self.is_acquisition()
                    else 'viewpoint reached; waiting for arm and camera to settle')
                return
            self.publish_next_waypoint(now)
            return

        error = self.max_joint_error(self.command_target)
        progress_error = self.total_joint_error(self.command_target)
        self.current_waypoint_error = error
        if math.isfinite(error):
            self.max_waypoint_error = max(self.max_waypoint_error, error)
        epsilon = float(
            configured_value(self, 'waypoint_progress_epsilon_rad'))
        if progress_error + epsilon < self.waypoint_best_error:
            self.waypoint_best_error = progress_error
            self.waypoint_last_progress_at = now
        action = waypoint_motion_action(
            error,
            float(configured_value(self, 'waypoint_reached_tolerance_rad')),
            now - float(self.waypoint_started_at),
            float(configured_value(self, 'waypoint_timeout_sec')),
            now - float(self.waypoint_last_progress_at),
            float(configured_value(self, 'waypoint_progress_timeout_sec')),
        )
        if action == 'abort_invalid':
            self.abort_or_finish_captures(
                'joint feedback became invalid during SDK MoveJ')
            return
        if action == 'abort_timeout':
            self.abort_or_finish_captures(
                'SDK MoveJ waypoint did not reach its position tolerance '
                'before timeout')
            return
        if action == 'abort_stalled':
            try:
                current = self.current_joints().tolist()
            except ValueError:
                current = []
            self.abort_or_finish_captures(
                'SDK MoveJ waypoint made no measurable joint progress before '
                'timeout: max_error=%.9f rad total_remaining=%.9f rad '
                'best_total=%.9f rad current=%s target=%s'
                % (
                    error,
                    progress_error,
                    self.waypoint_best_error,
                    current,
                    self.command_target.tolist(),
                ))
            return
        if action != 'advance':
            # PiPER JointCtrl is a MoveJ endpoint, not a streaming trajectory.
            # Re-sending the same endpoint can restart the SDK interpolation
            # and make slow 5% motion stall. Publish exactly once in
            # publish_next_waypoint(), then advance only from measured
            # feedback or abort on the bounded watchdog.
            if now - self.last_motion_status_at >= 0.10:
                self.last_motion_status_at = now
                self.publish_status()
            return
        if not self.abort_return_in_progress:
            self.record_retrace_target(self.command_target)
        if self.path_index >= len(self.current_path):
            if self.returning_home():
                self.begin_return_home_settle()
                return
            self.settle_started = None
            self.settle_position_anchor = None
            self.settle_last_sample_ok = False
            self.settle_reset_count = 0
            self.settle_longest_window_sec = 0.0
            self.settle_last_reset_reason = ''
            self.set_state(
                'SETTLING',
                'acquisition look reached; waiting for arm and camera to settle'
                if self.is_acquisition()
                else 'viewpoint reached; waiting for arm and camera to settle')
            return
        self.publish_next_waypoint(now)

    def streaming_moving_tick(self, now):
        """Follow planner timestamps; feedback gates safety and the endpoint."""
        if self.motion_started_at is None:
            self.motion_started_at = now
        if getattr(self, 'stream_wall_started_at', None) is None:
            self.stream_wall_started_at = self.motion_started_at
        last_tick = getattr(self, 'stream_last_tick_at', None)
        tick_interval = (
            max(0.0, now - float(last_tick))
            if last_tick is not None else 0.0)
        self.stream_last_tick_at = now
        elapsed = max(
            0.0,
            now - self.motion_started_at
            - float(getattr(self, 'stream_schedule_paused_sec', 0.0)),
        )

        def hold_or_abort_following(error, target, grace, limit):
            hold_started = getattr(
                self, 'stream_following_hold_started_at', None)
            if hold_started is None:
                hold_started = now
                self.stream_following_hold_started_at = now
                self.get_logger().warning(
                    'planner stream feedback corridor reached; holding '
                    'the last collision-qualified setpoint until measured '
                    'joints catch up')
            decision = TrajectoryRunner.following_decision(
                elapsed,
                error,
                grace,
                limit,
                over_limit_elapsed_sec=now - hold_started,
            )
            if decision.action is TrajectoryAction.HOLD_FOLLOWING:
                self.stream_schedule_paused_sec = float(getattr(
                    self, 'stream_schedule_paused_sec', 0.0)) + tick_interval
                if now - self.last_motion_status_at >= 0.10:
                    self.last_motion_status_at = now
                    self.publish_status()
                return True
            if decision.action is TrajectoryAction.FAILED_FOLLOWING:
                try:
                    current = self.current_joints().tolist()
                except (AttributeError, ValueError):
                    current = []
                self.abort_or_finish_captures(
                    'measured joints did not recover inside the scheduled '
                    'planner following corridor: max_error=%.9f rad '
                    'limit=%.9f rad current=%s target=%s'
                    % (error, limit, current, np.asarray(target).tolist()))
                return True
            return False

        if self.command_target is not None:
            grace = float(configured_value(
                self, 'trajectory_following_error_grace_sec'))
            limit = float(configured_value(
                self, 'trajectory_following_error_rad'))
            following_error = self.max_joint_error(self.command_target)
            self.current_waypoint_error = following_error
            if math.isfinite(following_error):
                self.max_waypoint_error = max(
                    self.max_waypoint_error, following_error)
            following_decision = TrajectoryRunner.following_decision(
                elapsed, following_error, grace, limit)
            if following_decision.action is TrajectoryAction.FAILED_FOLLOWING:
                if hold_or_abort_following(
                        following_error, self.command_target, grace, limit):
                    return

        if self.path_index < len(self.current_path):
            stream_decision = TrajectoryRunner.stream_decision(
                self.path_index, self.current_path_times, elapsed)
            if stream_decision.action is TrajectoryAction.WAIT:
                if now - self.last_motion_status_at >= 0.10:
                    self.last_motion_status_at = now
                    self.publish_status()
                return
            if stream_decision.action is TrajectoryAction.FAILED_OVERRUN:
                self.dropped_command_samples += stream_decision.missed_samples
                self.abort_or_finish_captures(
                    'scheduled planner stream overran by %d samples; '
                    'refusing to burst or shortcut the approved path'
                    % stream_decision.missed_samples)
                return
            if stream_decision.missed_samples:
                # Preserve every collision-qualified point.  A late executor
                # tick stretches the remaining schedule instead of bursting
                # multiple commands, skipping path corners, or aborting a
                # valid trajectory because Linux scheduling slipped once.
                delay = float(stream_decision.schedule_delay_sec)
                self.stream_schedule_paused_sec = float(getattr(
                    self, 'stream_schedule_paused_sec', 0.0)) + delay
                elapsed = max(0.0, elapsed - delay)
                ScanViewpointExecutorNode.record_stream_lateness(
                    self, now, delay, stream_decision.missed_samples)
            # Never burst stale goals and never shortcut collision-qualified
            # path corners. The pure runner returns exactly one due index.
            due_index = int(stream_decision.sample_index)
            target = self.current_path[due_index]
            grace = float(configured_value(
                self, 'trajectory_following_error_grace_sec'))
            limit = float(configured_value(
                self, 'trajectory_following_error_rad'))
            candidate_error = self.max_joint_error(target)
            candidate_decision = TrajectoryRunner.following_decision(
                elapsed, candidate_error, grace, limit)
            if candidate_decision.action is TrajectoryAction.FAILED_FOLLOWING:
                if hold_or_abort_following(
                        candidate_error, target, grace, limit):
                    return
            if getattr(self, 'stream_following_hold_started_at', None) is not None:
                self.get_logger().info(
                    'measured joints recovered inside the planner stream '
                    'feedback corridor; resuming the unchanged path')
                self.stream_following_hold_started_at = None
            self.path_index = due_index + 1
            self.publish_joint_command(target)
            self.command_target = np.asarray(target, dtype=float).copy()
            if self.command_sent_at > 0.0:
                self.max_command_interval_sec = max(
                    self.max_command_interval_sec,
                    now - self.command_sent_at,
                )
            self.command_sent_at = now
            self.command_samples_sent += 1
            if self.path_index >= len(self.current_path):
                self.waypoint_started_at = now
                self.waypoint_last_progress_at = now
                self.waypoint_best_error = self.total_joint_error(target)
                if not self.stream_schedule_completion_logged:
                    planned_duration = float(self.current_path_times[-1])
                    actual_duration = max(
                        0.0,
                        now - float(getattr(
                            self, 'stream_wall_started_at',
                            self.motion_started_at)),
                    )
                    interval_count = max(0, self.path_index - 1)
                    achieved_rate = (
                        float(interval_count) / actual_duration
                        if actual_duration > 0.0 else 0.0)
                    self.last_stream_planned_duration_sec = planned_duration
                    self.last_stream_actual_duration_sec = actual_duration
                    self.last_stream_achieved_rate_hz = achieved_rate
                    self.stream_schedule_completion_logged = True
                    self.get_logger().info(
                        'planner stream complete: planned=%.3fs '
                        'wall=%.3fs samples=%d achieved_rate=%.1fHz '
                        'max_interval=%.4fs max_following_error=%.4frad '
                        'late_ticks=%d max_lateness=%.4fs'
                        % (
                            planned_duration,
                            actual_duration,
                            self.path_index,
                            achieved_rate,
                            self.max_command_interval_sec,
                            self.max_waypoint_error,
                            int(getattr(
                                self, 'stream_late_event_count', 0)),
                            float(getattr(
                                self, 'stream_late_max_sec', 0.0)),
                        ))
            self.publish_status()
            return

        # All scheduled samples have been issued. Only the final endpoint is
        # feedback-gated so intermediate samples never create stop-and-go
        # motion, while convergence/stall/timeout behavior remains unchanged.
        ScanViewpointExecutorNode.feedback_gated_moving_tick(self, now)

    def record_stream_lateness(self, now, delay_sec, missed_samples):
        """Aggregate recurring timer lateness and warn at a bounded rate."""
        delay = max(0.0, float(delay_sec))
        missed = max(0, int(missed_samples))
        self.stream_late_event_count = int(getattr(
            self, 'stream_late_event_count', 0)) + 1
        self.stream_late_missed_samples = int(getattr(
            self, 'stream_late_missed_samples', 0)) + missed
        self.stream_late_max_sec = max(
            float(getattr(self, 'stream_late_max_sec', 0.0)), delay)
        last_report = getattr(self, 'stream_late_reported_at', None)
        if (
                last_report is not None
                and float(now) - float(last_report)
                < STREAM_LATE_WARNING_INTERVAL_SEC):
            return
        self.stream_late_reported_at = float(now)
        self.get_logger().warning(
            'Planner stream timer lateness detected: latest=%.4fs '
            'maximum=%.4fs events=%d later_samples_due=%d; preserving '
            'every approved point in order and stretching the schedule'
            % (
                delay,
                self.stream_late_max_sec,
                self.stream_late_event_count,
                self.stream_late_missed_samples,
            ))

    def publish_next_waypoint(self, now):
        target = np.asarray(
            self.current_path[self.path_index], dtype=float).copy()
        if self.is_startup_wrist_direct():
            # STARTUP_WRIST owns J6 only. Refresh J1-J5 at the exact command
            # boundary so powered gravity relaxation cannot turn an old
            # pre-enable snapshot into an unintended multi-joint target. The
            # tagged driver command independently replaces J1-J5 with its
            # newest coherent CAN sample before sending JointCtrl.
            measured = self.current_joints()
            target[:5] = measured[:5]
            self.current_path[self.path_index] = target.copy()
        self.path_index += 1
        self.publish_joint_command(target)
        self.command_target = np.asarray(target, dtype=float).copy()
        if self.command_sent_at > 0.0:
            self.max_command_interval_sec = max(
                self.max_command_interval_sec,
                now - self.command_sent_at,
            )
        self.command_sent_at = now
        self.command_samples_sent += 1
        if self.motion_started_at is None:
            self.motion_started_at = now
        self.waypoint_started_at = now
        self.waypoint_last_progress_at = now
        self.waypoint_best_error = self.total_joint_error(target)
        self.current_waypoint_error = self.max_joint_error(target)
        self.max_waypoint_error = max(
            self.max_waypoint_error,
            self.current_waypoint_error
            if math.isfinite(self.current_waypoint_error) else 0.0,
        )
        self.publish_status()

    def settling_tick(self):
        now = self.now()
        settled = (
            self.joints_settled()
            if self.is_acquisition() else self.capture_pose_settled())
        if (
                settled and self.plan_kind == MULTIVIEW_SCAN
                and not self.scan_history
                and (
                    self.latest_target_shape is None
                    or self.settle_started is None
                    or self.latest_target_shape_at
                    < float(self.settle_started)
                    or int(getattr(self, 'latest_target_shape_stamp_ns', 0))
                    < int(getattr(self, 'settle_started_ros_ns', 0)))
                and now - self.state_started > float(
                    configured_value(self, 'settle_timeout_sec'))):
            self.abort_motion(
                'camera target silhouette was not refreshed at the settled '
                'first scan endpoint before timeout')
            return
        coordinator = getattr(
            self, 'capture_coordinator',
            CaptureCoordinator(MAX_RGBD_CAPTURE_READINESS_RETRIES))
        settle_decision = coordinator.settle(
            now - self.state_started,
            settled,
            self.settle_started,
            now,
            configured_value(self, 'settle_duration_sec'),
            configured_value(self, 'settle_timeout_sec'),
        )
        if settle_decision.action is CaptureAction.ABORT:
            current_window = (
                0.0 if self.settle_started is None else
                max(0.0, now - float(self.settle_started)))
            self.get_logger().error(
                'Settle proof diagnostic: %s; resets=%d; '
                'longest_window=%.3fs; current_window=%.3fs; '
                'last_reset=%s'
                % (
                    str(getattr(
                        self, 'settle_diagnostic', 'unavailable')),
                    int(getattr(self, 'settle_reset_count', 0)),
                    float(getattr(self, 'settle_longest_window_sec', 0.0)),
                    current_window,
                    str(getattr(
                        self, 'settle_last_reset_reason', 'unavailable')),
                ))
            self.abort_motion(
                'arm/camera did not settle before acquisition refresh'
                if self.is_acquisition()
                else 'arm/camera did not settle before timeout')
            return
        if settle_decision.reset_settle_window:
            if self.settle_started is not None:
                self.settle_longest_window_sec = max(
                    float(self.settle_longest_window_sec),
                    max(0.0, now - float(self.settle_started)),
                )
            self.settle_reset_count += 1
            self.settle_last_reset_reason = str(self.settle_diagnostic)
            self.settle_started = None
            self.settle_started_ros_ns = 0
            return
        if settle_decision.action is CaptureAction.START_SETTLE_WINDOW:
            self.settle_started = now
            self.settle_started_ros_ns = int(
                self.get_clock().now().nanoseconds)
            return
        if settle_decision.action is not CaptureAction.READY:
            return
        if self.is_acquisition():
            self.request_acquisition_refresh()
            return
        if (
                not self.latest_achieved_matches_current_view()
                and not self.record_latest_achieved_scan_view()):
            return
        aim_rejection = self.final_capture_aim_rejection()
        if aim_rejection:
            if self.first_capture_framing_retry_active():
                aim_rejection = (
                    'TARGET_FRAMING_NO_AIMED_ENDPOINT: farther endpoint did '
                    'not retain target-facing aim; ' + aim_rejection)
            self.reject_achieved_capture_view(aim_rejection)
            return
        framing = self.settled_first_capture_framing_result()
        if framing is None:
            return
        framing_action, framing_reason, farther = framing
        if framing_action != 'CLEAR':
            marker = {
                'RETRY_FARTHER': 'TARGET_FRAMING_RETRY_FARTHER',
                'TOO_CLOSE': 'TARGET_FRAMING_TOO_CLOSE',
                'TOO_LARGE': 'TARGET_FRAMING_TOO_LARGE',
                'NO_AIMED_ENDPOINT': 'TARGET_FRAMING_NO_AIMED_ENDPOINT',
            }[framing_action]
            metadata = None
            if framing_action == 'RETRY_FARTHER':
                viewpoint = self.plan_viewpoints[self.current_view]
                if viewpoint.get('ray_id') is None:
                    self.reject_achieved_capture_view(
                        'TARGET_FRAMING_NO_AIMED_ENDPOINT: current first '
                        'view has no bounded target ray for an outward retry')
                    return
                metadata = {
                    'framing_retry_ray_id': int(viewpoint['ray_id']),
                    'framing_retry_min_standoff_m': float(farther),
                }
            self.reject_achieved_capture_view(
                '%s: %s' % (marker, framing_reason), metadata)
            return
        if not self.param_bool('auto_capture'):
            self.pending_scan_qualified_target_shape = None
            self.pending_scan_qualified_target_model_seed = None
            self.advance_view()
            return
        telemetry_store = getattr(self, 'telemetry_store', None)
        snapshot = (
            None if telemetry_store is None else telemetry_store.snapshot())
        workflow_ready = (
            self.workflow_ready() if snapshot is None else
            ScanViewpointExecutorNode.workflow_ready(self, snapshot))
        if not workflow_ready:
            self.abort_motion('supervised workflow is not SCAN_READY at capture time')
            return
        if snapshot is None:
            workflow = self.latest_workflow
        else:
            observation = snapshot.mission.workflow
            workflow = None if observation is None else observation.value
        self.capture_accepted_before = int(workflow.get('accepted_views', 0))
        self.set_state(
            'CAPTURING_RGBD',
            'settled viewpoint reached; saving synchronized RGB-D record')
        self.rgbd_capture_future = None
        self.rgbd_capture_attempts = 0
        self.capture_heavy_refresh_started = None
        self.capture_heavy_refresh_request_id = ''
        self.capture_heavy_refresh_min_image_stamp_ns = 0
        self.capture_heavy_refresh_publish_attempts = 0
        self.capture_heavy_refresh_waiting_for_worker = False
        self.capture_rejection_reason = ''
        self.finish_scan_future = None

    def first_capture_framing_retry_active(self):
        """Return whether this zero-capture plan is a farther same-ray retry."""
        if self.plan_kind != MULTIVIEW_SCAN or self.scan_history:
            return False
        if self.current_view >= len(self.plan_viewpoints):
            return False
        ray_id = self.plan_viewpoints[self.current_view].get('ray_id')
        if ray_id is None or not self.scan_rejections:
            return False
        latest = self.scan_rejections[-1]
        return bool(
            latest.get('framing_retry_min_standoff_m') is not None
            and int(latest.get('framing_retry_ray_id', -1)) == int(ray_id))

    def settled_first_capture_framing_result(self):
        """Check border contact only at the settled, aimed first scan pose."""
        if self.plan_kind != MULTIVIEW_SCAN or self.scan_history:
            return 'CLEAR', 'first capture framing already qualified', None
        if (
                self.latest_target_shape is None
                or self.settle_started is None
                or self.latest_target_shape_at < float(self.settle_started)
                or int(getattr(self, 'latest_target_shape_stamp_ns', 0))
                < int(getattr(self, 'settle_started_ros_ns', 0))):
            self.settle_diagnostic = (
                'waiting for a source-stamped post-settle target silhouette')
            return None
        achieved = self.latest_achieved_scan_view
        try:
            camera = np.asarray([
                achieved['camera_position'][axis]
                for axis in ('x', 'y', 'z')], dtype=float)
            target = np.asarray(self.plan_target_center, dtype=float)
            distance = float(np.linalg.norm(target - camera))
        except (KeyError, TypeError, ValueError):
            return (
                'NO_AIMED_ENDPOINT',
                'settled target-facing camera geometry is unavailable',
                None,
            )
        previous_minimum = None
        if self.first_capture_framing_retry_active():
            previous_minimum = self.scan_rejections[-1][
                'framing_retry_min_standoff_m']
        return first_capture_framing_decision(
            self.latest_target_shape, distance, previous_minimum)

    def request_acquisition_refresh(self):
        self.acquisition_refresh_started = self.now()
        self.acquisition_request_attempt = 0
        self.acquisition_waiting_for_worker = False
        self.pending_acquisition_heavy_status = None
        self.acquisition_job_started = None
        self.acquisition_detection_completed = None
        self.acquisition_job_image_stamp_ns = 0
        self.publish_acquisition_refresh()

    def publish_acquisition_refresh(self):
        self.acquisition_request_attempt += 1
        stamp = self.get_clock().now().to_msg()
        self.acquisition_min_image_stamp_ns = (
            int(stamp.sec) * 1000000000 + int(stamp.nanosec))
        self.acquisition_request_id = '%s-acquire-%d-attempt-%d' % (
            self.plan_id, self.current_view, self.acquisition_request_attempt)
        request = String()
        request.data = json.dumps({
            'request_id': self.acquisition_request_id,
            'reason': 'rough_acquisition_viewpoint',
            'min_image_stamp': {
                'sec': int(stamp.sec),
                'nanosec': int(stamp.nanosec),
            },
        }, sort_keys=True)
        self.heavy_refresh_pub.publish(request)
        self.set_state(
            'WAITING_FOR_FRESH_FRAME',
            'GroundingDINO request %s waiting for a fresh post-settle frame'
            % self.acquisition_request_id)

    def waiting_for_fresh_frame_tick(self):
        if self.acquisition_refresh_started is None:
            self.abort_motion('acquisition refresh timing is missing')
            return
        if self.now() - self.acquisition_refresh_started > float(
                configured_value(self, 'acquisition_fresh_frame_timeout_sec')):
            self.abort_motion(
                'fresh post-settle camera frame or idle GroundingDINO worker '
                'did not become available before timeout')

    def waiting_for_grounding_dino_tick(self):
        if self.acquisition_job_started is None:
            self.abort_motion('GroundingDINO queue acknowledgement is missing')
            return
        if self.now() - self.acquisition_job_started > float(
                configured_value(self, 'acquisition_grounding_timeout_sec')):
            self.abort_motion(
                'matching GroundingDINO job exceeded the %.1f-second timeout'
                % float(configured_value(self, 'acquisition_grounding_timeout_sec')))

    def waiting_for_tracking_lock_tick(self):
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            tracked_target = self.latest_tracked_target
            tracking_health = self.latest_tracking_health
            target_status = self.latest_target_status
            tracked_target_at = self.updated.get('tracked_target', -1e9)
            tracking_at = self.updated.get('tracking', -1e9)
            target_status_at = self.updated.get('target_status', -1e9)
            now = self.now()
        else:
            snapshot = telemetry_store.snapshot()
            target_observation = snapshot.perception.target
            tracking_observation = snapshot.perception.tracking
            status_observation = snapshot.perception.target_status
            tracked_target = (
                None if target_observation is None
                else target_observation.value)
            tracking_health = (
                None if tracking_observation is None
                else tracking_observation.value)
            target_status = (
                'UNKNOWN' if status_observation is None
                else status_observation.value)
            tracked_target_at = (
                -1e9 if target_observation is None
                else target_observation.received_at)
            tracking_at = (
                -1e9 if tracking_observation is None
                else tracking_observation.received_at)
            target_status_at = (
                -1e9 if status_observation is None
                else status_observation.received_at)
            now = snapshot.captured_at
        rejection = acquired_target_rejection(
            tracked_target,
            tracking_health,
            target_status,
            tracked_target_at,
            tracking_at,
            target_status_at,
            self.acquisition_detection_completed,
            now,
            float(configured_value(self, 'data_timeout_sec')),
            float(configured_value(self, 'max_tracking_measurement_age_sec')),
            self.acquisition_job_image_stamp_ns,
            self.plan_target_center,
            float(configured_value(self, 'acquisition_target_tolerance_m')),
        )
        if not rejection:
            self.command_target = None
            self.current_path = []
            self.current_path_times = []
            self.motion_started_at = None
            self.publish_hold()
            self.set_state(
                'ACQUIRED',
                'measured target tracking locked; remaining acquisition looks cancelled')
            return
        if self.now() - self.state_started < float(
                configured_value(self, 'acquisition_tracking_lock_timeout_sec')):
            return
        self.get_logger().warn(
            'GroundingDINO detection did not produce an acceptable measured lock: %s'
            % rejection)
        self.set_state(
            'WAITING_FOR_OBSTACLE_SCENE',
            'target lock was not established; waiting for the matching '
            'post-settle semantic scene')

    def waiting_for_obstacle_scene_tick(self):
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            obstacles = self.latest_obstacles
            obstacles_at = self.updated.get('obstacles', -1e9)
            now = self.now()
        else:
            snapshot = telemetry_store.snapshot()
            observation = snapshot.perception.obstacles
            obstacles = None if observation is None else observation.value
            obstacles_at = (
                -1e9 if observation is None else observation.received_at)
            now = snapshot.captured_at
        status, reason = correlated_obstacle_scene_status(
            obstacles,
            obstacles_at,
            now,
            float(configured_value(self, 'data_timeout_sec')),
            self.acquisition_job_image_stamp_ns,
        )
        if status == 'ready':
            self.acquisition_scene_snapshot_validated = True
            self.advance_acquisition_view()
            return
        if status == 'blocked':
            self.abort_motion(
                'post-settle semantic scene rejected: ' + reason)
            return
        if self.now() - self.state_started > float(
                configured_value(self, 'acquisition_scene_timeout_sec')):
            self.abort_motion(
                'post-settle semantic scene did not become ready before timeout: '
                + reason)

    def advance_acquisition_view(self):
        self.current_view += 1
        self.acquisition_refresh_started = None
        self.acquisition_request_id = ''
        self.acquisition_request_attempt = 0
        self.acquisition_min_image_stamp_ns = 0
        self.acquisition_job_image_stamp_ns = 0
        self.acquisition_job_started = None
        self.acquisition_detection_completed = None
        self.acquisition_waiting_for_worker = False
        self.pending_acquisition_heavy_status = None
        if self.current_view >= self.plan_capture_count:
            self.command_target = None
            self.current_path = []
            self.current_path_times = []
            self.motion_started_at = None
            self.publish_hold()
            self.set_state(
                'ACQUISITION_LOOK_COMPLETE',
                'one acquisition look completed without measured target lock; '
                'holding for fresh closed-loop replanning')
            return
        runtime = self.runtime_reasons(runtime_gate_policy(
            SafetyMode.ACQUISITION_MOTION,
            require_settled=True,
            obstacle_authority=ObstacleAuthority.LIVE,
        ))
        if runtime:
            self.abort_motion(
                'next acquisition runtime gate failed: ' + '; '.join(runtime))
            return
        reasons = self.prepare_current_view()
        if reasons:
            self.abort_motion(
                'next acquisition trajectory validation failed: ' + '; '.join(reasons))
            return
        self.begin_runtime_refresh(
            'moving slowly to the next approved acquisition look',
            require_workflow=False,
            allow_missing_obstacles=False,
        )

    def capturing_tick(self):
        if self.now() - self.state_started > float(
                configured_value(self, 'capture_timeout_sec')):
            self.abort_motion('capture service response timed out')
            return
        if self.capture_future is None or not self.capture_future.done():
            return
        try:
            result = self.capture_future.result()
        except Exception as error:
            self.abort_motion('capture service failed: %s' % error)
            return
        if result is None or not result.success:
            message = result.message if result is not None else 'empty service response'
            self.abort_motion('capture was rejected: %s' % message)
            return
        self.set_state(
            'WAIT_CAPTURE',
            'RGB-D viewpoint saved and accepted; advancing after workflow '
            'status propagation')

    def capturing_rgbd_tick(self):
        if self.now() - self.state_started > float(
                configured_value(self, 'capture_timeout_sec')):
            self.abort_motion('RGB-D capture service response timed out')
            return
        coordinator = getattr(
            self, 'capture_coordinator',
            CaptureCoordinator(MAX_RGBD_CAPTURE_READINESS_RETRIES))
        capture_decision = coordinator.handoff(
            self.rgbd_capture_future is not None,
            self.now() - self.state_started,
            configured_value(self, 'capture_status_propagation_sec'),
        )
        if capture_decision.action is CaptureAction.PUBLISH_AUTHORIZATION:
            self.publish_status()
            return
        if capture_decision.action is CaptureAction.REQUEST_CAPTURE:
            # Republish immediately before dispatch as well.  The service is
            # called exactly once because rgbd_capture_future is assigned
            # synchronously here.
            self.publish_status()
            self.rgbd_capture_attempts += 1
            self.rgbd_capture_future = self.rgbd_capture_client.call_async(
                Trigger.Request())
            return
        if not self.rgbd_capture_future.done():
            return
        try:
            result = self.rgbd_capture_future.result()
        except Exception as error:
            self.abort_motion('RGB-D capture service failed: %s' % error)
            return
        if result is None or not result.success:
            message = result.message if result is not None else 'empty service response'
            failure = as_failure(message)
            capture_decision = coordinator.classify_result(
                False, failure, self.rgbd_capture_attempts,
                bool(self.capture_heavy_refresh_request_id))
            if capture_decision.action is CaptureAction.RETRY_SAME_VIEW:
                self.rgbd_capture_future = None
                self.set_state(
                    'CAPTURING_RGBD',
                    'capture evidence is still catching up with the settled '
                    'frame; retrying the same viewpoint without moving')
                return
            if capture_decision.action is CaptureAction.REFRESH_SAME_VIEW:
                self.request_capture_heavy_refresh(message)
                return
            if capture_decision.action is CaptureAction.REPLAN_VIEW:
                self.reject_achieved_capture_view(message)
                return
            self.abort_motion('RGB-D viewpoint capture was rejected: %s' % message)
            return
        try:
            capture_result = capture_result_from_response(result.message)
        except ValueError as error:
            self.abort_motion(
                'accepted RGB-D capture result is malformed: %s' % error)
            return
        self.pending_capture_occlusion_state = str(
            capture_result['occlusion_state'])
        self.capture_semantic_probe_pending = bool(
            not getattr(self, 'scan_history', [])
            or self.pending_capture_occlusion_state != 'CLEAR')
        if (
                not getattr(self, 'scan_history', [])
                and getattr(
                    self, 'scan_qualified_target_model_seed', None) is None):
            seed = capture_result['qualified_target_model_seed']
            self.pending_scan_qualified_target_model_seed = dict(seed)
            self.pending_scan_qualified_target_shape = dict(seed['shape'])
        # Count a view only after its synchronized files exist. The workflow's
        # cloud/quality products are diagnostic and must not precede the
        # primary capture contract.
        self.capture_future = self.capture_client.call_async(Trigger.Request())
        self.set_state(
            'CAPTURING',
            'RGB-D viewpoint saved; recording viewpoint acceptance')

    def request_capture_heavy_refresh(self, reason):
        """Request one request-correlated semantic refresh without moving."""
        if self.capture_heavy_refresh_request_id:
            self.reject_achieved_capture_view(reason)
            return
        self.capture_heavy_refresh_started = self.now()
        self.capture_rejection_reason = str(reason)
        stamp = self.get_clock().now().to_msg()
        self.capture_heavy_refresh_min_image_stamp_ns = (
            int(stamp.sec) * 1000000000 + int(stamp.nanosec))
        self.capture_heavy_refresh_request_id = (
            '%s-capture-%d-refresh' % (self.plan_id, self.current_view))
        self.capture_heavy_refresh_publish_attempts = 0
        self.capture_heavy_refresh_waiting_for_worker = False
        self.publish_capture_heavy_refresh()
        self.publish_hold()
        self.set_state(
            'WAITING_FOR_CAPTURE_REFRESH',
            'fresh visual capture evidence was invalid; holding the achieved '
            'pose for one correlated heavy perception refresh')

    def publish_capture_heavy_refresh(self):
        """Publish the current correlated capture refresh request."""
        self.capture_heavy_refresh_publish_attempts += 1
        seconds, nanoseconds = divmod(
            int(self.capture_heavy_refresh_min_image_stamp_ns), 1000000000)
        request = String()
        request.data = json.dumps({
            'request_id': self.capture_heavy_refresh_request_id,
            'reason': 'settled_scan_capture_revalidation',
            'min_image_stamp': {
                'sec': int(seconds),
                'nanosec': int(nanoseconds),
            },
        }, sort_keys=True)
        self.heavy_refresh_pub.publish(request)

    def waiting_for_capture_refresh_tick(self):
        """Bound the single capture refresh by the existing capture timeout."""
        if self.capture_heavy_refresh_started is None:
            self.abort_motion('capture heavy refresh timing is missing')
            return
        if self.now() - self.capture_heavy_refresh_started > float(
                configured_value(self, 'capture_timeout_sec')):
            self.abort_motion('capture heavy perception refresh timed out')

    def reject_achieved_capture_view(self, reason, metadata=None):
        """Reject one achieved observation and hand a clean replan result up."""
        if not self.record_rejected_view(reason, metadata):
            return
        self.command_target = None
        self.current_path = []
        self.current_path_velocities = []
        self.current_path_accelerations = []
        self.current_path_times = []
        self.publish_hold()
        self.set_state(
            'VIEW_REJECTED',
            'RGB-D viewpoint rejected after settled visual validation: %s; '
            'holding for a new NBV plan' % reason)

    def wait_capture_tick(self):
        first_capture_pending = not bool(getattr(self, 'scan_history', []))
        semantic_probe_pending = bool(getattr(
            self, 'capture_semantic_probe_pending', first_capture_pending))
        acceptance_timeout = float(configured_value(
            self,
            'first_capture_acceptance_timeout_sec'
            if semantic_probe_pending else 'capture_timeout_sec'))
        if self.now() - self.state_started > acceptance_timeout:
            self.abort_motion(
                'capture semantic acceptance did not return workflow to '
                'SCAN_READY'
                if semantic_probe_pending else
                'accepted capture did not return workflow to SCAN_READY')
            return
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            workflow = self.latest_workflow
        else:
            observation = telemetry_store.snapshot().mission.workflow
            workflow = None if observation is None else observation.value
        if isinstance(workflow, dict) and str(
                workflow.get('state', '')) == 'ABORTED':
            self.abort_motion(
                'workflow rejected first capture: '
                + str(workflow.get('reason', 'workflow aborted')))
            return
        if not (
                isinstance(workflow, dict)
                and str(workflow.get('state', '')) == 'SCAN_READY'):
            return
        accepted = int(workflow.get('accepted_views', 0))
        if accepted <= self.capture_accepted_before:
            return
        if not self.record_accepted_view(accepted):
            return
        self.advance_view()

    def record_accepted_view(self, accepted_views):
        """Remember an accepted camera pose only after synchronized capture succeeds."""
        if self.plan_kind != MULTIVIEW_SCAN:
            return True
        if not self.scan_session_id:
            self.abort_motion('scan workflow session identity is missing')
            return False
        expected = len(self.scan_history) + 1
        if int(accepted_views) != expected:
            self.abort_motion(
                'workflow accepted-view count is inconsistent with scan memory')
            return False
        if self.current_view >= len(self.plan_viewpoints):
            self.abort_motion('accepted viewpoint is missing from the approved plan')
            return False
        if (
                not self.scan_history
                and self.scan_qualified_target_shape is None):
            pending_shape = getattr(
                self, 'pending_scan_qualified_target_shape', None)
            pending_seed = getattr(
                self, 'pending_scan_qualified_target_model_seed', None)
            if (
                    not isinstance(pending_shape, dict)
                    or not isinstance(pending_seed, dict)):
                self.abort_motion(
                    'accepted first view has no capture-bound model seed')
                return False
            self.scan_qualified_target_shape = dict(pending_shape)
            self.scan_qualified_target_model_seed = dict(pending_seed)
        self.pending_scan_qualified_target_shape = None
        self.pending_scan_qualified_target_model_seed = None
        viewpoint = self.plan_viewpoints[self.current_view]
        if (
                not self.latest_achieved_matches_current_view()
                and not self.record_latest_achieved_scan_view()):
            return False
        achieved = dict(self.latest_achieved_scan_view)
        joints = list(achieved['joint_positions_rad'])
        actual_camera = dict(achieved['camera_position'])
        actual_look = dict(achieved['look_direction'])
        planning_target = dict(zip(
            ('x', 'y', 'z'),
            (float(value) for value in self.plan_target_center)))
        workflow = getattr(self, 'latest_workflow', None)
        occlusion_labels = []
        if isinstance(workflow, dict):
            raw_labels = workflow.get('occlusion_labels', [])
            if isinstance(raw_labels, list):
                occlusion_labels = sorted(set(
                    str(label) for label in raw_labels if str(label)))
        self.scan_history.append({
            'accepted_view': int(accepted_views),
            'plan_id': self.plan_id,
            'viewpoint_index': int(viewpoint.get('index', self.current_view)),
            # The session planner consumes these achieved values on the next
            # one-view transaction. Keep the original proposal alongside them
            # for auditability, but never score coverage from an ideal pose.
            'desired_camera_position': actual_camera,
            'desired_look_at_direction': actual_look,
            'actual_camera_position': actual_camera,
            'actual_look_at_direction': actual_look,
            'proposed_camera_position': dict(
                viewpoint.get('desired_camera_position', {})),
            'proposed_look_at_direction': dict(
                viewpoint.get('desired_look_at_direction', {})),
            'joint_positions_rad': joints,
            'achieved_at_sec': float(achieved['achieved_at_sec']),
            'target_estimate_used': planning_target,
            'occlusion_state': str(getattr(
                self, 'pending_capture_occlusion_state', 'UNKNOWN')),
            'occluded': bool(
                occlusion_labels
                or str(getattr(
                    self, 'pending_capture_occlusion_state', 'UNKNOWN'))
                in ('PARTIALLY_OCCLUDED', 'HEAVILY_OCCLUDED')),
            'occlusion_labels': occlusion_labels,
            **({'ray_id': int(viewpoint['ray_id'])}
               if viewpoint.get('ray_id') is not None else {}),
        })
        self.pending_capture_occlusion_state = 'UNKNOWN'
        self.capture_semantic_probe_pending = False
        self.publish_scan_history()
        return True

    def record_rejected_view(self, reason, metadata=None):
        """Exclude one visually unusable pose from the next session replan."""
        # A rejected RGB-D observation cannot promote its pending silhouette
        # into the mission-frozen revolved model.
        self.pending_scan_qualified_target_shape = None
        self.pending_scan_qualified_target_model_seed = None
        if (
                self.plan_kind != MULTIVIEW_SCAN
                or self.current_view >= len(self.plan_viewpoints)):
            return False
        viewpoint = self.plan_viewpoints[self.current_view]
        if (
                not self.latest_achieved_matches_current_view()
                and not self.record_latest_achieved_scan_view()):
            return False
        achieved = dict(self.latest_achieved_scan_view)
        actual_camera = dict(achieved['camera_position'])
        actual_look = dict(achieved['look_direction'])
        planning_target = dict(zip(
            ('x', 'y', 'z'),
            (float(value) for value in self.plan_target_center)))
        entry = {
            'rejected_view': len(self.scan_rejections) + 1,
            'plan_id': self.plan_id,
            'viewpoint_index': int(
                viewpoint.get('index', self.current_view)),
            'desired_camera_position': actual_camera,
            'desired_look_at_direction': actual_look,
            'actual_camera_position': actual_camera,
            'actual_look_at_direction': actual_look,
            'proposed_camera_position': dict(
                viewpoint.get('desired_camera_position', {})),
            'proposed_look_at_direction': dict(
                viewpoint.get('desired_look_at_direction', {})),
            'achieved_at_sec': float(achieved['achieved_at_sec']),
            'target_estimate_used': planning_target,
            'reason': str(reason),
            **({'ray_id': int(viewpoint['ray_id'])}
               if viewpoint.get('ray_id') is not None else {}),
        }
        if metadata:
            entry.update(dict(metadata))
        self.scan_rejections.append(entry)
        self.publish_scan_history()
        return True

    def record_latest_achieved_scan_view(self):
        """Snapshot settled FK independently of capture acceptance."""
        if self.plan_kind != MULTIVIEW_SCAN:
            return True
        if self.current_view >= len(self.plan_viewpoints):
            self.abort_motion('achieved viewpoint is missing from the approved plan')
            return False
        try:
            joints = [float(value) for value in self.current_joints()]
            achieved_transform = self.kinematics.camera_transform(joints)
            achieved_camera = np.asarray(
                achieved_transform[:3, 3], dtype=float)
            achieved_look = np.asarray(
                achieved_transform[:3, :3], dtype=float).dot(
                    np.asarray([0.0, 0.0, 1.0], dtype=float))
            achieved_look /= np.linalg.norm(achieved_look)
            if (
                    achieved_camera.shape != (3,)
                    or achieved_look.shape != (3,)
                    or not np.all(np.isfinite(achieved_camera))
                    or not np.all(np.isfinite(achieved_look))):
                raise ValueError('achieved camera FK is non-finite')
        except (TypeError, ValueError) as error:
            self.abort_motion('cannot record achieved viewpoint: %s' % error)
            return False
        viewpoint = self.plan_viewpoints[self.current_view]
        self.latest_achieved_scan_view = {
            'plan_id': str(self.plan_id),
            'viewpoint_index': int(
                viewpoint.get('index', self.current_view)),
            'camera_position': dict(zip(
                ('x', 'y', 'z'),
                (float(value) for value in achieved_camera))),
            'look_direction': dict(zip(
                ('x', 'y', 'z'),
                (float(value) for value in achieved_look))),
            'joint_positions_rad': joints,
            'achieved_at_sec': float(self.now()),
            **({'ray_id': int(viewpoint['ray_id'])}
               if viewpoint.get('ray_id') is not None else {}),
        }
        self.publish_scan_history()
        return True

    def latest_achieved_matches_current_view(self):
        """Return whether the settled FK record belongs to this exact view."""
        achieved = self.latest_achieved_scan_view
        if (
                not isinstance(achieved, dict)
                or self.current_view >= len(self.plan_viewpoints)):
            return False
        viewpoint = self.plan_viewpoints[self.current_view]
        return bool(
            str(achieved.get('plan_id', '')) == str(self.plan_id)
            and int(achieved.get('viewpoint_index', -1)) == int(
                viewpoint.get('index', self.current_view)))

    def final_capture_aim_rejection(self):
        """Require the achieved optical axis to retain the five-degree aim."""
        if self.plan_kind != MULTIVIEW_SCAN:
            return ''
        achieved = self.latest_achieved_scan_view
        if not isinstance(achieved, dict):
            return 'FINAL_AIM_EXCEEDED: achieved camera FK is unavailable'
        try:
            camera = np.asarray([
                achieved['camera_position'][axis]
                for axis in ('x', 'y', 'z')], dtype=float)
            look = np.asarray([
                achieved['look_direction'][axis]
                for axis in ('x', 'y', 'z')], dtype=float)
            target = np.asarray(self.plan_target_center, dtype=float)
            tracked = getattr(self, 'latest_tracked_target', None)
            tracked_header = getattr(tracked, 'header', None)
            tracked_fresh = bool(
                tracked is not None
                and getattr(tracked, 'valid', False)
                and str(getattr(tracked_header, 'frame_id', '')) == 'base_link'
                and self.fresh('tracked_target'))
            viewpoint = (
                self.plan_viewpoints[self.current_view]
                if self.current_view < len(self.plan_viewpoints) else {})
            frozen_ray_target = viewpoint.get('ray_id') is not None
            if tracked_fresh and not frozen_ray_target:
                target = np.asarray([
                    tracked.position.x,
                    tracked.position.y,
                    tracked.position.z,
                ], dtype=float)
            drift = float(np.linalg.norm(
                target - np.asarray(self.plan_target_center, dtype=float)))
            expected = target - camera
            expected /= np.linalg.norm(expected)
            look /= np.linalg.norm(look)
            error_deg = math.degrees(math.acos(float(np.clip(
                np.dot(expected, look), -1.0, 1.0))))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return 'FINAL_AIM_EXCEEDED: achieved target-facing geometry is invalid'
        achieved['final_aim_error_deg'] = float(error_deg)
        achieved['target_estimate_at_capture'] = dict(zip(
            ('x', 'y', 'z'), (float(value) for value in target)))
        achieved['target_estimate_drift_m'] = float(drift)
        maximum_drift = float(configured_value(
            self, 'max_target_drift_before_approval_m'))
        if (
                tracked_fresh and not frozen_ray_target
                and drift > maximum_drift + 1e-9):
            return (
                'TARGET_DRIFT_REPLAN: target estimate changed %.4fm after '
                'planning; hold the achieved pose and request a fresh NBV'
                % drift)
        if error_deg > MAX_FINAL_CAPTURE_AIM_ERROR_DEG + 1e-6:
            return (
                'FINAL_AIM_EXCEEDED: achieved camera aim %.3fdeg exceeds '
                '%.3fdeg' % (
                    error_deg, MAX_FINAL_CAPTURE_AIM_ERROR_DEG))
        return ''

    def advance_view(self):
        self.current_view += 1
        if self.current_view >= self.plan_capture_count:
            required = (
                int(configured_value(self, 'max_execution_viewpoints'))
                if hasattr(self, 'get_parameter')
                else int(self.plan_capture_count))
            accepted_total = (
                len(self.scan_history)
                if hasattr(self, 'scan_history')
                else int(self.current_view))
            closed_loop = (
                self.param_bool('closed_loop_one_view')
                if hasattr(self, 'param_bool') else False)
            if closed_loop:
                self.command_target = None
                self.current_path = []
                self.current_path_times = []
                self.publish_hold()
                self.set_state(
                    'VIEW_COMPLETE',
                    'one accepted scan view persisted; holding for coordinator '
                    'coverage decision and direct configured-home shutdown'
                    if accepted_total >= required else
                    'one accepted scan view persisted; holding for fresh '
                    'measured-coverage replanning')
                return
            if (
                    self.plan_returns_home
                    and self.current_view < len(self.plan_targets)):
                reasons = self.prepare_current_view()
                if reasons:
                    self.finish_captures_without_home(
                        'return-home trajectory validation failed: '
                        + '; '.join(reasons))
                    return
                self.begin_runtime_refresh(
                    'returning to the approved home position after all captures',
                    require_workflow=False,
                    allow_missing_obstacles=False,
                )
                return
            self.set_state('COMPLETE', 'all approved viewpoints reached and captured')
            return
        reasons = self.prepare_current_view()
        if reasons:
            self.abort_motion('next trajectory validation failed: ' + '; '.join(reasons))
            return
        self.begin_runtime_refresh(
            'moving to the next approved viewpoint',
            require_workflow=True,
            allow_missing_obstacles=False,
        )

    def returning_home(self):
        return bool(
            getattr(self, 'abort_return_in_progress', False)
            or getattr(self, 'plan_kind', '') == RETURN_HOME
            or (
                self.plan_returns_home
                and self.current_view >= self.plan_capture_count))

    def record_retrace_target(self, target):
        if target is None:
            return
        value = np.asarray(target, dtype=float).copy()
        if (
                not self.retrace_joint_targets
                or float(np.max(np.abs(
                    value - self.retrace_joint_targets[-1]))) > 1e-7):
            self.retrace_joint_targets.append(value)

    def begin_return_home_settle(self):
        self.current_path = []
        self.current_path_velocities = []
        self.current_path_accelerations = []
        self.current_path_times = []
        self.settle_started = None
        self.home_settle_previous_joints = None
        self.home_settle_last_joint_update = -1e9
        self.home_settle_last_sample_ok = False
        # PiPER retains the last MoveJ target after it is reached. A tagged
        # hold between STARTUP_WRIST and ROUGH_HOME is redundant and would be
        # a second asynchronous J6 transaction. Settle against the actual
        # final startup command and let the mission start ROUGH_HOME next.
        if not self.is_startup_wrist_direct():
            self.publish_hold()
        self.set_state(
            'SETTLING_HOME',
            (
                'approved-path abort retrace reached the plan start; waiting '
                'for stable joint feedback'
                if getattr(self, 'abort_return_in_progress', False) else
                'approved home target reached; waiting for stable joint feedback'
            ))

    def return_home_settling_tick(self):
        if self.now() - self.state_started > float(
                configured_value(self, 'home_settle_timeout_sec')):
            ScanViewpointExecutorNode.handle_return_home_failure(
                self,
                'approved home position did not settle before timeout')
            return
        if not self.home_joints_settled():
            self.settle_started = None
            return
        if self.settle_started is None:
            self.settle_started = self.now()
            return
        if self.now() - self.settle_started < float(
                configured_value(self, 'home_settle_duration_sec')):
            return
        self.complete_return_home()

    def complete_return_home(self):
        self.command_target = None
        self.current_path = []
        self.current_path_velocities = []
        self.current_path_accelerations = []
        self.current_path_times = []
        self.publish_hold()
        if getattr(self, 'abort_return_in_progress', False):
            reason = self.abort_return_reason
            self.abort_return_in_progress = False
            self.abort_return_reason = ''
            self.abort_return_bootstrap_static_scene = False
            self.set_state(
                'ABORTED',
                reason
                + '; arm safely retraced along already executed approved '
                'targets; approved plan start reached; dedicated '
                'configured-home transaction required')
            return
        if getattr(self, 'plan_kind', '') == RETURN_HOME:
            self.set_state(
                'ABORTED',
                'dedicated collision-qualified configured home reached')
            return
        self.finish_scan_future = self.finish_scan_client.call_async(
            Trigger.Request())
        self.return_home_warning = ''
        self.set_state(
            'FINISHING_WORKFLOW',
            'all approved viewpoints captured and home reached; finalizing '
            'the workflow session')

    def abort_or_finish_captures(self, reason):
        """Keep successful captures terminal even if only return-home fails."""
        if self.returning_home():
            ScanViewpointExecutorNode.handle_return_home_failure(self, reason)
        else:
            self.abort_motion(reason)

    def handle_return_home_failure(self, reason):
        if getattr(self, 'abort_return_in_progress', False):
            original = self.abort_return_reason
            self.abort_return_in_progress = False
            self.abort_return_reason = ''
            self.abort_return_bootstrap_static_scene = False
            self._terminal_abort(
                original + '; automatic return-home retrace stopped: ' + reason)
            return
        if getattr(self, 'plan_kind', '') == RETURN_HOME:
            self._terminal_abort(
                'dedicated configured-home execution failed: ' + str(reason))
            return
        self.finish_captures_without_home(reason)

    def finish_captures_without_home(self, reason):
        """
        Hold the current pose and finalize a fully captured scan.

        Return-home is a post-capture convenience segment.  A safety or
        telemetry failure must stop that motion, but it must not erase thirteen
        synchronized captures or start Step-4/5 recovery with a consumed
        approval.
        """
        self.command_target = None
        self.current_path = []
        self.current_path_velocities = []
        self.current_path_accelerations = []
        self.current_path_times = []
        self.publish_hold()
        self.return_home_warning = str(reason)
        self.finish_scan_future = self.finish_scan_client.call_async(
            Trigger.Request())
        self.set_state(
            'FINISHING_WORKFLOW',
            'all approved viewpoints captured; return-home incomplete and '
            'the arm is held at its current position: %s; finalizing the '
            'workflow session' % self.return_home_warning)

    def finish_workflow_tick(self):
        self.publish_hold()
        if self.now() - self.state_started > float(
                configured_value(self, 'finish_scan_timeout_sec')):
            if self.current_view >= self.plan_capture_count:
                self.finish_capture_session(
                    'workflow finish service response timed out')
            else:
                self.abort_motion('workflow finish service response timed out')
            return
        if self.finish_scan_future is None or not self.finish_scan_future.done():
            return
        try:
            result = self.finish_scan_future.result()
        except Exception as error:
            if self.current_view >= self.plan_capture_count:
                self.finish_capture_session(
                    'workflow finish service failed: %s' % error)
            else:
                self.abort_motion('workflow finish service failed: %s' % error)
            return
        if result is None or not result.success:
            message = result.message if result is not None else 'empty service response'
            if self.current_view >= self.plan_capture_count:
                self.finish_capture_session(
                    'workflow finish was rejected: %s' % message)
            else:
                self.abort_motion('workflow finish was rejected: %s' % message)
            return
        self.finish_capture_session('')

    def finish_capture_session(self, workflow_warning):
        """Publish one terminal successful-capture result, with any warning."""
        self.finish_scan_future = None
        self.scan_history = []
        self.scan_rejections = []
        self.scan_qualified_target_shape = None
        self.pending_scan_qualified_target_shape = None
        self.scan_qualified_target_model_seed = None
        self.pending_scan_qualified_target_model_seed = None
        self.latest_achieved_scan_view = None
        self.scan_session_id = ''
        self.scan_coverage_target_center = None
        self.scan_target_center_qualified = False
        self.publish_scan_history()
        warnings = [
            item for item in (
                self.return_home_warning, str(workflow_warning))
            if item
        ]
        if warnings:
            reason = (
                'all approved viewpoints captured and saved; arm held at its '
                'current position; ' + '; '.join(warnings))
        else:
            reason = (
                'all approved viewpoints captured, home reached, and scan '
                'session finalized')
        self.return_home_warning = ''
        self.set_state(
            'COMPLETE',
            reason)

    def prepare_current_view(self):
        start = self.current_joints()
        if not self.plan_collision_model_qualified:
            return ['planner collision model is not qualified for hardware']
        path = [item.copy() for item in self.plan_paths[self.current_view]]
        path_times = [
            float(item) for item in self.plan_path_times[self.current_view]]
        if len(path) != len(path_times):
            return ['waypoint positions and order stamps differ in length']
        start_tolerance = float(
            configured_value(self, 'plan_start_tolerance_rad'))
        if float(np.max(np.abs(path[0] - start))) > start_tolerance:
            return ['current state changed beyond the approved plan-start tolerance']
        # A cumulative joint-distance gate rejects valid collision-aware
        # detours even though the executor never sends that cumulative
        # displacement as one command. The planner path has already been
        # hash-bound and exact-path validated; retain the per-sample step
        # limit, fresh-start match, and feedback convergence below.
        command_path = path[1:]
        command_velocities = [
            item.copy() for item in
            self.plan_path_velocities[self.current_view][1:]]
        command_accelerations = [
            item.copy() for item in
            self.plan_path_accelerations[self.current_view][1:]]
        command_times = path_times[1:]
        if not (
                len(command_path)
                == len(command_velocities)
                == len(command_accelerations)
                == len(command_times)):
            return [
                'waypoint positions, derivative placeholders and order stamps '
                'differ in length']
        powered_start_end = (
            getattr(
                self, 'plan_powered_start_recovery_end_points', []
            )[self.current_view]
            if self.current_view < len(getattr(
                self, 'plan_powered_start_recovery_end_points', []))
            else -1)
        direct_home = bool(
            self.current_view < len(getattr(
                self, 'plan_configured_home_direct', []))
            and self.plan_configured_home_direct[self.current_view])
        obstacle_boxes = (
            [] if (
                uses_bootstrap_static_scene(
                    self.plan_kind, self.current_view)
                or ScanViewpointExecutorNode.is_startup_home_static(self)
                or direct_home)
            else self.obstacle_boxes())
        recovery_end = (
            self.plan_bootstrap_recovery_end_points[self.current_view]
            if self.current_view < len(self.plan_bootstrap_recovery_end_points)
            else -1)
        if direct_home:
            configured_stages = getattr(
                self, 'plan_configured_home_stages', [''])
            configured_stage = str(configured_stages[0]).strip().upper()
            startup_bridge_path = bool(
                configured_stage == 'STARTUP_WRIST'
                and len(path) == 3
                and np.max(np.abs(path[1][:5] - path[0][:5])) <= 1e-9
                and abs(float(path[1][5]) - (3.2 - 2.0 * math.pi)) <= 1e-9)
            if not (
                    self.is_return_home()
                    and self.plan_returns_home
                    and self.plan_capture_count == 0
                    and self.current_view == 0
                    and (len(path) == 2 or startup_bridge_path)
                    and configured_stage in (
                        'CONFIGURED_HOME', 'STARTUP_WRIST', 'PRE_HOME',
                        'ROUGH_HOME', 'STORAGE_WRIST')):
                return [
                    'configured-home collision bypass escaped its approved scope']
            # The configured resting fold is intentionally not representable
            # by the conservative collision envelopes.  The worker and
            # approval contract bind this exact direct joint target; retain
            # start matching, limits, runtime health, feedback convergence,
            # hold and disable gates, but do not collision-recheck this path.
            reasons = []
        elif recovery_end >= 0:
            acquisition_recovery = bool(
                self.is_acquisition() and self.current_view == 0)
            terminal_home_recovery = bool(
                self.plan_kind in (MULTIVIEW_SCAN, RETURN_HOME)
                and self.plan_returns_home
                and self.current_view == self.plan_capture_count)
            if not acquisition_recovery and not terminal_home_recovery:
                return [
                    'bounded folded-home recovery escaped its approved scope']
            recovery_path = (
                list(reversed(path[recovery_end:]))
                if terminal_home_recovery
                else path[:recovery_end + 1])
            reasons = validate_monotonic_self_clearance_escape(
                self.kinematics,
                self.validation_path(recovery_path),
                self.joint_limits,
                obstacle_boxes=obstacle_boxes,
                joint_margin_rad=0.0,
                floor_z_m=float(configured_value(self, 'floor_z_m')),
                link_radius_m=float(configured_value(self, 'link_radius_m')),
                self_clearance_m=float(configured_value(self, 'self_clearance_m')),
                recovery_joint_number=(
                    self.plan_bootstrap_recovery_joint_sets[self.current_view]
                    if self.current_view
                    < len(self.plan_bootstrap_recovery_joint_sets) else None),
                maximum_start_limit_violation_rad=0.04,
            )
            if not reasons and powered_start_end >= 0:
                if not (
                        self.plan_kind == RETURN_HOME
                        and terminal_home_recovery
                        and powered_start_end == 1
                        and recovery_end == 2):
                    return [
                        'powered-start home recovery escaped its approved scope']
                reasons = validate_monotonic_self_clearance_escape(
                    self.kinematics,
                    self.validation_path(path[:powered_start_end + 1]),
                    self.joint_limits,
                    obstacle_boxes=obstacle_boxes,
                    joint_margin_rad=0.0,
                    floor_z_m=float(configured_value(self, 'floor_z_m')),
                    link_radius_m=float(
                        configured_value(self, 'link_radius_m')),
                    self_clearance_m=float(
                        configured_value(self, 'self_clearance_m')),
                    recovery_joint_number=(
                        getattr(
                            self, 'plan_powered_start_recovery_joint_sets', []
                        )[
                            self.current_view]
                        if self.current_view < len(
                            getattr(
                                self,
                                'plan_powered_start_recovery_joint_sets', []))
                        else None),
                    maximum_start_limit_violation_rad=0.0,
                )
            if not reasons:
                normal_path = (
                    path[
                        powered_start_end if powered_start_end >= 0 else 0:
                        recovery_end + 1]
                    if terminal_home_recovery else path[recovery_end:])
                reasons = self.validate_path(normal_path, obstacle_boxes)
        else:
            reasons = self.validate_path(path, obstacle_boxes)
        if not reasons:
            external_validator = getattr(
                self, 'validate_attached_tool_external_path', None)
            if callable(external_validator):
                reasons = external_validator(path, obstacle_boxes)
        if reasons:
            return reasons
        if (
                self.plan_kind == MULTIVIEW_SCAN
                and self.current_view < self.plan_capture_count):
            visibility_reasons = camera_target_path_reasons(
                self.kinematics,
                self.validation_path(path),
                self.plan_target_center,
                configured_value(self, 'scan_target_max_boresight_deg'),
                configured_value(self, 'scan_target_min_distance_m'),
                initial_alignment=not bool(self.scan_history),
                final_aim_deg=MAX_FINAL_CAPTURE_AIM_ERROR_DEG,
            )
            if visibility_reasons:
                return visibility_reasons
        execution_mode = (
            getattr(self, 'plan_segment_execution_modes', [])[self.current_view]
            if self.current_view < len(getattr(
                self, 'plan_segment_execution_modes', []))
            else 'TIMED_STREAM')
        command_path, command_velocities, command_accelerations, \
            command_times, streaming = sdk_command_path(
                path,
                self.plan_path_velocities[self.current_view],
                self.plan_path_accelerations[self.current_view],
                path_times, execution_mode, direct_home)
        self.current_path = command_path
        self.current_path_velocities = command_velocities
        self.current_path_accelerations = command_accelerations
        self.current_path_times = command_times
        self.current_path_streaming = streaming
        try:
            runner = getattr(self, 'trajectory_runner', TrajectoryRunner())
            self.current_trajectory = runner.begin(
                getattr(self, 'plan_id', 'compatibility-plan'),
                command_path,
                command_times,
                self.current_path_streaming,
            )
        except ValueError as error:
            return [str(error)]
        self.path_index = 0
        self.command_target = None
        self.command_sent_at = 0.0
        self.command_samples_sent = 0
        self.max_command_interval_sec = 0.0
        self.dropped_command_samples = 0
        self.motion_started_at = None
        self.stream_wall_started_at = None
        self.stream_last_tick_at = None
        self.stream_schedule_paused_sec = 0.0
        self.stream_following_hold_started_at = None
        self.stream_schedule_completion_logged = False
        self.stream_late_event_count = 0
        self.stream_late_missed_samples = 0
        self.stream_late_max_sec = 0.0
        self.stream_late_reported_at = None
        self.last_motion_status_at = 0.0
        self.waypoint_started_at = None
        self.waypoint_last_progress_at = None
        self.waypoint_best_error = math.inf
        self.current_waypoint_error = 0.0
        self.max_waypoint_error = 0.0
        self.capture_future = None
        self.rgbd_capture_future = None
        self.rgbd_capture_attempts = 0
        return []

    def validate_path(self, path, obstacle_boxes):
        return validate_joint_path(
            self.kinematics,
            self.validation_path(path),
            self.joint_limits,
            obstacle_boxes=obstacle_boxes,
            joint_margin_rad=0.0,
            floor_z_m=float(configured_value(self, 'floor_z_m')),
            link_radius_m=float(configured_value(self, 'link_radius_m')),
            self_clearance_m=float(configured_value(self, 'self_clearance_m')),
        )

    def validate_attached_tool_external_path(self, path, obstacle_boxes):
        """Retain holder/L515 floor and scene clearance for every mode."""
        return validate_attached_box_external_clearance_path(
            self.kinematics,
            self.validation_path(path),
            configured_value(self, 'camera_holder_envelope_center_link6_m'),
            configured_value(self, 'camera_holder_envelope_size_m'),
            obstacle_boxes=obstacle_boxes,
            floor_z_m=float(configured_value(self, 'floor_z_m')),
            clearance_m=float(configured_value(self, 'camera_holder_external_clearance_m')),
            label='camera holder/L515',
        )

    def validation_path(self, path):
        """Densely check SDK-interpolated segments without publishing samples."""
        if not path:
            return []
        maximum_step = float(
            configured_value(self, 'trajectory_joint_step_rad'))
        dense = [np.asarray(path[0], dtype=float).copy()]
        for endpoint in path[1:]:
            dense.extend(interpolate_joint_path(
                dense[-1], endpoint, maximum_step))
        return dense

    def runtime_reasons(self, policy, telemetry_snapshot=None):
        """Evaluate one explicit phase policy from one telemetry snapshot."""
        if not isinstance(policy, RuntimeGatePolicy):
            policy = runtime_gate_policy(policy)
        require_settled = policy.require_settled
        require_workflow = policy.require_workflow
        allow_untracked = not policy.require_tracking
        allow_missing_camera = not policy.require_camera
        allow_missing_obstacles = (
            policy.obstacle_authority
            == ObstacleAuthority.STATIC_BOOTSTRAP)
        allow_stale_obstacles = (
            policy.obstacle_authority
            == ObstacleAuthority.APPROVED_SNAPSHOT)
        settle_at_current_hold = policy.settle_at_current_hold
        telemetry_store = getattr(self, 'telemetry_store', None)
        snapshot = (
            None if telemetry_store is None else
            (telemetry_store.snapshot()
             if telemetry_snapshot is None else telemetry_snapshot))

        def snapshot_fresh(key, timeout=None):
            if snapshot is None:
                return self.fresh(key, timeout)
            maximum = float(configured_value(self, 'data_timeout_sec')) \
                if timeout is None else float(timeout)
            observation = snapshot_observation(snapshot, key)
            return bool(
                observation is not None
                and not observation.is_stale_at(
                    snapshot.captured_at, maximum))

        reasons = []
        required_keys = ['joints', 'arm_status']
        if not allow_missing_camera:
            required_keys.append('camera_clock')
        if not allow_missing_obstacles and not allow_stale_obstacles:
            required_keys.append('obstacles')
        if not allow_untracked:
            required_keys.extend(['tracking', 'target_status'])
        for key in required_keys:
            if not snapshot_fresh(key):
                reasons.append('%s data missing or stale' % key)
        if (
                policy.require_motion_limits
                and not snapshot_fresh(
                    'motion_limits',
                    float(configured_value(
                        self, 'motion_limits_timeout_sec')))):
            reasons.append('motion_limits data missing or stale')
        if reasons:
            return reasons
        if policy.require_motion_limits:
            limits = (
                self.latest_motion_limits if snapshot is None else
                snapshot.arm.motion_limits.value)
            if limits is None or not limits.valid:
                reasons.append(
                    'controller motion limits changed after trajectory planning')
            elif (
                    str(limits.limits_sha256)
                    != self.runtime_motion_limits_sha256):
                rejection = self.runtime_motion_limit_rejection(limits)
                if rejection:
                    reasons.append(rejection)
                else:
                    previous_hash = self.runtime_motion_limits_sha256
                    self.runtime_motion_limits_sha256 = str(
                        limits.limits_sha256)
                    self.get_logger().warn(
                        'Controller limits changed from %s to %s; the exact '
                        'approved waypoint path remains within the fresh limits'
                        % (previous_hash, self.runtime_motion_limits_sha256))
        try:
            joints = (
                self.current_joints() if snapshot is None else
                ScanViewpointExecutorNode.current_joints(self, snapshot))
        except ValueError as error:
            reasons.append(str(error))
            return reasons
        try:
            current_view = int(getattr(self, 'current_view', -1))
            recovery_end_points = getattr(
                self, 'plan_bootstrap_recovery_end_points', [])
            recovery_end = (
                recovery_end_points[current_view]
                if 0 <= current_view < len(recovery_end_points) else -1)
            if (
                    current_view == 0
                    and recovery_end >= 1
                    and current_view < len(self.plan_paths)
                    and recovery_end < len(self.plan_paths[current_view])
                    and self.is_acquisition()):
                recovery_path = [
                    item.copy()
                    for item in self.plan_paths[
                        current_view][:recovery_end + 1]]
                recovery_path[0] = joints.copy()
                reasons.extend(bootstrap_start_limit_recovery_reasons(
                    recovery_path,
                    self.joint_limits,
                    self.plan_bootstrap_recovery_joint_sets[current_view],
                    0.04,
                ))
            else:
                if self.is_configured_home_direct():
                    configured_stages = getattr(
                        self, 'plan_configured_home_stages', [])
                    configured_stage = (
                        str(configured_stages[current_view]).strip().upper()
                        if 0 <= current_view < len(configured_stages)
                        else '')
                    reasons.extend(configured_home_feedback_limit_reasons(
                        joints,
                        self.joint_limits,
                        float(configured_value(
                            self,
                            'configured_home_feedback_limit_tolerance_rad')),
                        home_stage=configured_stage,
                    ))
                else:
                    reasons.extend(feedback_joint_limit_reasons(
                        joints,
                        self.joint_limits,
                        float(configured_value(self, 'joint_feedback_limit_tolerance_rad')),
                    ))
        except (IndexError, ValueError) as error:
            reasons.append(str(error))
        if snapshot is None:
            reasons.extend(self.arm_status_reasons())
            camera_health = self.latest_camera_timestamp_health
            obstacles = self.latest_obstacles
            health = self.latest_tracking_health
            target_status = self.latest_target_status
        else:
            status_observation = snapshot.arm.status
            reasons.extend(ScanViewpointExecutorNode.arm_status_reasons(
                self,
                None if status_observation is None
                else status_observation.value,
                use_provided=True))
            camera_observation = snapshot.perception.camera
            camera_health = (
                None if camera_observation is None
                else camera_observation.value)
            obstacle_observation = snapshot.perception.obstacles
            obstacles = (
                None if obstacle_observation is None
                else obstacle_observation.value)
            tracking_observation = snapshot.perception.tracking
            health = (
                None if tracking_observation is None
                else tracking_observation.value)
            status_observation = snapshot.perception.target_status
            target_status = (
                'UNKNOWN' if status_observation is None
                else status_observation.value)
        if not allow_missing_camera and (
                camera_health is None or not camera_health.healthy):
            state = camera_health.state if camera_health is not None else 'MISSING'
            detail = camera_health.reason if camera_health is not None else 'no watchdog status'
            reasons.append('camera timestamp %s: %s' % (state, detail))
        if not allow_missing_obstacles:
            reasons.extend(obstacle_scene_runtime_reasons(
                obstacles))
        if allow_untracked:
            if require_settled:
                settled = self.joints_settled(
                    settle_at_current=settle_at_current_hold)
                if not settled:
                    reasons.append(
                        'joint feedback is not settled at the current-position hold'
                        if settle_at_current_hold else
                        'joint feedback is not settled for acquisition')
        else:
            if require_settled:
                if health.lifecycle_state != 'TRACKING' or not health.camera_settled:
                    reasons.append('tracking is not settled TRACKING')
                if health.prediction_only:
                    reasons.append('tracking is prediction-only')
            elif health.lifecycle_state not in ('TRACKING', 'DEGRADED'):
                reasons.append('tracking lifecycle=%s' % health.lifecycle_state)
            if float(health.measurement_age_sec) > float(
                    configured_value(self, 'max_tracking_measurement_age_sec')):
                reasons.append('tracking measurement is stale')
            if float(health.recommended_speed_scale) < float(
                    configured_value(self, 'min_tracking_speed_scale')):
                reasons.append(
                    'tracking speed scale is below the motion threshold')
            current_allowed_speed = commanded_speed_percent(
                float(configured_value(self, 'speed_percent')),
                self.plan_kind,
                float(health.recommended_speed_scale),
            )
            if (
                    self.plan_execution_speed_percent > 0.0
                    and current_allowed_speed + 1e-6
                    < self.plan_execution_speed_percent):
                reasons.append(
                    'tracking speed allowance fell below the approved MoveJ speed; '
                    'replan at the lower speed')
            if (
                    target_status not in ('TRACKING', 'LOCKED')):
                reasons.append('target_status=%s' % target_status)
        if require_workflow and self.param_bool('auto_capture'):
            workflow_ready = (
                self.workflow_ready() if snapshot is None else
                ScanViewpointExecutorNode.workflow_ready(self, snapshot))
            if not workflow_ready:
                reasons.append('supervised workflow is not SCAN_READY')
        return reasons

    def runtime_motion_limit_rejection(self, limits):
        """
        Accept a fresh valid limit generation for position-only SDK MoveJ.

        The executor sends one joint-position target plus an aggregate speed
        percentage. It never sends planner velocities, accelerations, or
        timing to the controller, so a changed valid hash does not change the
        approved geometric path.
        """
        try:
            velocities = np.asarray(limits.max_velocity_rad_s, dtype=float)
            accelerations = np.asarray(
                limits.max_acceleration_rad_s2, dtype=float)
        except (AttributeError, TypeError, ValueError):
            return 'fresh controller motion limits are malformed'
        if (
                velocities.shape != (6,)
                or accelerations.shape != (6,)
                or not np.all(np.isfinite(velocities))
                or not np.all(np.isfinite(accelerations))
                or np.any(velocities <= 0.0)
                or np.any(accelerations <= 0.0)):
            return 'fresh controller motion limits are malformed'
        return ''

    def arm_status_reasons(self, status=None, use_provided=False):
        if not use_provided:
            telemetry_store = getattr(self, 'telemetry_store', None)
            if telemetry_store is None:
                status = self.latest_arm_status
            else:
                observation = telemetry_store.snapshot().arm.status
                status = None if observation is None else observation.value
        if status is None:
            return ['arm status is missing']
        reasons = []
        if int(status.err_code) != 0:
            reasons.append('arm err_code=%d' % int(status.err_code))
        angle_limits = [
            status.joint_1_angle_limit, status.joint_2_angle_limit,
            status.joint_3_angle_limit, status.joint_4_angle_limit,
            status.joint_5_angle_limit, status.joint_6_angle_limit,
        ]
        if any(angle_limits):
            reasons.append('arm reports a joint angle-limit fault')
        communication = [
            status.communication_status_joint_1, status.communication_status_joint_2,
            status.communication_status_joint_3, status.communication_status_joint_4,
            status.communication_status_joint_5, status.communication_status_joint_6,
        ]
        if any(communication):
            reasons.append('arm reports a joint communication fault')
        reasons.extend(motor_control_reasons(
            status, require_all_enabled=True))
        return reasons

    def settled_and_tracking(self):
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            camera_health = self.latest_camera_timestamp_health
            health = self.latest_tracking_health
            target_status = self.latest_target_status
        else:
            snapshot = telemetry_store.snapshot()
            camera_observation = snapshot.perception.camera
            tracking_observation = snapshot.perception.tracking
            status_observation = snapshot.perception.target_status
            camera_health = (
                None if camera_observation is None
                else camera_observation.value)
            health = (
                None if tracking_observation is None
                else tracking_observation.value)
            target_status = (
                'UNKNOWN' if status_observation is None
                else status_observation.value)
        if (
                camera_health is None or not camera_health.healthy):
            return False
        if health is None or health.lifecycle_state != 'TRACKING' or not health.camera_settled:
            return False
        if health.prediction_only or target_status not in ('TRACKING', 'LOCKED'):
            return False
        return (
            self.joints_settled() if telemetry_store is None else
            ScanViewpointExecutorNode.joints_settled(
                self, snapshot=snapshot))

    def capture_pose_settled(self):
        """
        Gate RGB-D on a stationary arm and healthy camera clock.

        The scan target may move and SAM tracking may temporarily reacquire at
        a viewpoint.  Neither changes the approved robot pose or whether a
        synchronized RGB-D record can be saved, so they must not discard the
        remaining 13-view plan.
        """
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            camera_health = self.latest_camera_timestamp_health
            snapshot = None
        else:
            snapshot = telemetry_store.snapshot()
            observation = snapshot.perception.camera
            camera_health = None if observation is None else observation.value
        return bool(
            camera_health is not None
            and camera_health.healthy
            and (
                self.joints_settled() if telemetry_store is None else
                ScanViewpointExecutorNode.joints_settled(
                    self, snapshot=snapshot)))

    def joints_settled(self, settle_at_current=False, snapshot=None):
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            joints = self.latest_joint_state
            joints_fresh = self.fresh('joints', 1.0)
            updated_at = float(self.updated.get('joints', -1e9))
            snapshot = None
        else:
            snapshot = (
                telemetry_store.snapshot() if snapshot is None else snapshot)
            observation = snapshot.arm.joints
            joints = None if observation is None else observation.value
            joints_fresh = bool(
                observation is not None
                and not observation.is_stale_at(snapshot.captured_at, 1.0))
            updated_at = (
                -1e9 if observation is None else observation.received_at)
        if joints is None or not joints_fresh:
            self.settle_position_anchor = None
            self.settle_last_sample_ok = False
            self.settle_diagnostic = 'joint feedback is missing or stale'
            return False
        current = (
            self.current_joints() if snapshot is None else
            ScanViewpointExecutorNode.current_joints(self, snapshot))
        # Proposal approval has no active endpoint yet.  In that phase the
        # measured pose itself is the only valid stationary-hold target.  Once
        # execution publishes an endpoint, continue to bind settling to that
        # commanded target so merely stopping short cannot pass this gate.
        target = self.command_target
        if settle_at_current or target is None:
            target = current
        if updated_at <= self.settle_last_joint_update:
            return self.settle_last_sample_ok
        target_error = float(np.max(np.abs(current - target)))
        anchor_error = (
            math.nan if self.settle_position_anchor is None else
            float(np.max(np.abs(
                current - np.asarray(self.settle_position_anchor, dtype=float))))
        )
        self.settle_diagnostic = (
            'max_target_error=%.9frad (limit %.9f), '
            'max_anchor_drift=%s (limit %.9f), current=%s, target=%s'
            % (
                target_error,
                float(configured_value(self, 'joint_goal_tolerance_rad')),
                ('not-started' if not math.isfinite(anchor_error)
                 else '%.9frad' % anchor_error),
                float(configured_value(self, 'endpoint_position_settled_rad')),
                np.asarray(current, dtype=float).tolist(),
                np.asarray(target, dtype=float).tolist(),
            ))
        settled, anchor = target_position_window_sample_settled(
            current,
            target,
            self.settle_position_anchor,
            float(configured_value(self, 'joint_goal_tolerance_rad')),
            float(configured_value(self, 'endpoint_position_settled_rad')),
        )
        self.settle_position_anchor = anchor
        self.settle_last_joint_update = updated_at
        self.settle_last_sample_ok = bool(settled)
        return bool(settled)

    def home_joints_settled(self):
        """
        Use successive positions for final home proof.

        PiPER's SDK speed feedback can briefly spike while the measured joint
        positions remain stationary. The final disable gate therefore matches
        the GUI's independent safe-disable proof: fresh feedback must stay
        within the approved home tolerance and successive samples must move by
        no more than a small bounded delta.
        """
        feedback_timeout = float(configured_value(self, 'home_joint_feedback_timeout_sec'))
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            joints = self.latest_joint_state
            joints_fresh = self.fresh('joints', feedback_timeout)
            updated_at = float(self.updated.get('joints', -1e9))
            snapshot = None
        else:
            snapshot = telemetry_store.snapshot()
            observation = snapshot.arm.joints
            joints = None if observation is None else observation.value
            joints_fresh = bool(
                observation is not None
                and not observation.is_stale_at(
                    snapshot.captured_at, feedback_timeout))
            updated_at = (
                -1e9 if observation is None else observation.received_at)
        if (
                joints is None
                or self.command_target is None
                or not joints_fresh):
            self.home_settle_previous_joints = None
            self.home_settle_last_sample_ok = False
            return False
        if updated_at <= self.home_settle_last_joint_update:
            return self.home_settle_last_sample_ok
        current = np.asarray(
            self.current_joints() if snapshot is None else
            ScanViewpointExecutorNode.current_joints(self, snapshot),
            dtype=float)
        measured_current = current.copy()
        target = np.asarray(self.command_target, dtype=float)
        previous = self.home_settle_previous_joints
        if self.is_startup_wrist_direct():
            # STARTUP_WRIST is explicitly a J6-only transaction. J1-J5 may
            # relax slightly during the long powered wrist rotation; the
            # immediately following ROUGH_HOME stage owns and proves those
            # endpoints. Keep this stage's settle proof bound to fresh,
            # stationary J6 feedback only.
            selected_current = target.copy()
            selected_current[5] = current[5]
            current = selected_current
            if previous is not None:
                selected_previous = target.copy()
                selected_previous[5] = np.asarray(
                    previous, dtype=float)[5]
                previous = selected_previous
        target_tolerance = float(configured_value(
            self, 'home_goal_tolerance_rad'))
        if self.is_startup_wrist_direct():
            target_tolerance = min(
                target_tolerance, STARTUP_WRIST_READY_TOLERANCE_RAD)
        settled = home_position_sample_settled(
            current,
            target,
            previous,
            target_tolerance,
            float(configured_value(self, 'home_motion_tolerance_rad')),
        )
        # Retain the complete measured pose so a stage change can never reuse
        # a one-joint snapshot as ordinary six-joint home evidence.
        self.home_settle_previous_joints = measured_current
        self.home_settle_last_joint_update = updated_at
        self.home_settle_last_sample_ok = bool(settled)
        return bool(settled)

    def workflow_ready(self, snapshot=None):
        if snapshot is None:
            telemetry_store = getattr(self, 'telemetry_store', None)
            if telemetry_store is None:
                workflow = self.latest_workflow
            else:
                observation = telemetry_store.snapshot().mission.workflow
                workflow = None if observation is None else observation.value
        else:
            observation = snapshot.mission.workflow
            workflow = None if observation is None else observation.value
        return isinstance(workflow, dict) and \
            str(workflow.get('state', '')) == 'SCAN_READY'

    def obstacle_boxes(self):
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            obstacles = self.latest_obstacles
        else:
            observation = telemetry_store.snapshot().perception.obstacles
            obstacles = None if observation is None else observation.value
        if obstacles is None:
            return []
        boxes = []
        for item in obstacles.instances:
            if not item.valid:
                continue
            boxes.append(CollisionBox(
                '%s:%d' % (item.semantic_label, int(item.object_id)),
                np.asarray([
                    item.base_bounds_min.x, item.base_bounds_min.y, item.base_bounds_min.z,
                ], dtype=float),
                np.asarray([
                    item.base_bounds_max.x, item.base_bounds_max.y, item.base_bounds_max.z,
                ], dtype=float),
            ))
        return boxes

    def publish_joint_command(self, target, explicit_hold=False):
        if self.command_pub is None:
            raise RuntimeError('real-motion publisher does not exist in proposal-only mode')
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        stage = ''
        configured_home_check = getattr(
            self, 'is_configured_home_direct', None)
        if callable(configured_home_check) and configured_home_check():
            stages = getattr(self, 'plan_configured_home_stages', [])
            if 0 <= self.current_view < len(stages):
                stage = str(stages[self.current_view]).strip().upper()
        if bool(explicit_hold):
            msg.header.frame_id = 'piper_scan_executor_hold'
        elif stage == 'STARTUP_WRIST':
            msg.header.frame_id = 'piper_scan_executor_startup_wrist'
        else:
            msg.header.frame_id = 'piper_scan_executor_sdk_movej'
        msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        # Six positions deliberately mean "arm only" to the PiPER driver.
        # velocity[6] retains the driver's established aggregate-speed field
        # without asking it to clip or command the gripper.
        msg.position = [float(value) for value in target]
        speed = self.execution_speed_percent()
        msg.velocity = [0.0] * 6 + [speed]
        self.command_pub.publish(msg)

    def publish_hold(self):
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            joints = self.latest_joint_state
            freshness_check = getattr(self, 'fresh', None)
            joints_fresh = bool(
                not callable(freshness_check)
                or freshness_check(
                    'joints', float(configured_value(self, 'home_joint_feedback_timeout_sec'))))
            snapshot = None
        else:
            snapshot = telemetry_store.snapshot()
            observation = snapshot.arm.joints
            joints = None if observation is None else observation.value
            joints_fresh = bool(
                observation is not None
                and not observation.is_stale_at(
                    snapshot.captured_at,
                    float(configured_value(self, 'home_joint_feedback_timeout_sec'))))
        if joints is None or not self.real_motion_enabled():
            return False
        if not joints_fresh:
            return False
        try:
            self.publish_joint_command(
                energized_hold_target(
                    self.current_joints() if snapshot is None else
                    ScanViewpointExecutorNode.current_joints(
                        self, snapshot)),
                explicit_hold=True)
        except ValueError:
            return False
        return True

    def abort_motion(self, reason):
        blocker = abort_return_home_blocker(reason)
        suffix = (
            '; arm held at the current pose; safety fault forbids automatic '
            'home motion: ' + blocker
            if blocker else
            '; arm held at the current pose for a fresh dedicated '
            'return-home plan')
        self._terminal_abort(str(reason) + suffix)

    def try_start_abort_return(self, reason):
        """Legacy/manual fallback; autonomous cancellation does not call it."""
        if self.abort_return_in_progress:
            return False, ''
        blocker = abort_return_home_blocker(reason)
        if blocker:
            return False, 'abort is safety-related (%s)' % blocker
        if not self.retrace_joint_targets:
            return False, 'no executed approved target history is available'
        try:
            current = self.current_joints()
        except ValueError as error:
            return False, str(error)
        use_bootstrap_scene = bootstrap_abort_retrace_uses_static_scene(
            self.plan_kind,
            self.current_view,
            getattr(self, 'plan_collision_model_qualified', False),
        )
        runtime = self.runtime_reasons(runtime_gate_policy(
            SafetyMode.RETURN_HOME,
            require_settled=True,
        ))
        if runtime:
            return False, 'fresh return-home safety gate failed: ' + '; '.join(runtime)
        tolerance = float(
            configured_value(self, 'plan_start_tolerance_rad'))
        at_latest_endpoint = float(np.max(np.abs(
            current - self.retrace_joint_targets[-1]))) <= tolerance
        # At a settled endpoint, omit that same endpoint.  After cancellation
        # stopped an in-flight SDK MoveJ, first retrace to the preceding
        # approved endpoint, then continue through the already executed
        # approval-bound history to the original powered home pose.
        history = (
            self.retrace_joint_targets[:-1]
            if at_latest_endpoint else self.retrace_joint_targets)
        reverse_targets = [item.copy() for item in reversed(history)]
        if not reverse_targets:
            self.publish_hold()
            self._terminal_abort(
                str(reason) + '; configured home reached (arm was already at home)')
            return True, ''
        path = [current.copy()] + reverse_targets
        obstacle_boxes = [] if use_bootstrap_scene else self.obstacle_boxes()
        validation = approved_retrace_validation_reasons(
            self.validate_path(path, obstacle_boxes))
        if validation:
            return False, (
                'approved-path retrace validation failed: '
                + '; '.join(validation))

        self.abort_return_in_progress = True
        self.abort_return_reason = str(reason)
        self.abort_return_bootstrap_static_scene = use_bootstrap_scene
        self.current_view = self.plan_capture_count
        self.current_path = reverse_targets
        self.current_path_velocities = [
            np.zeros(6, dtype=float) for _ in reverse_targets]
        self.current_path_accelerations = [
            np.zeros(6, dtype=float) for _ in reverse_targets]
        self.current_path_times = [
            float(index + 1) for index in range(len(reverse_targets))]
        self.current_path_streaming = False
        self.path_index = 0
        self.command_target = None
        self.command_sent_at = 0.0
        self.command_samples_sent = 0
        self.motion_started_at = None
        self.waypoint_started_at = None
        self.waypoint_last_progress_at = None
        self.waypoint_best_error = math.inf
        self.current_waypoint_error = 0.0
        self.publish_hold()
        if use_bootstrap_scene:
            # Keep every runtime tick bound to the same static scene that
            # authorized the exact outward segment, including the interval
            # after WAITING_FOR_RUNTIME_REFRESH transitions to MOVING.
            self.acquisition_scene_snapshot_validated = True
        self.begin_runtime_refresh(
            'non-safety abort accepted; retracing already executed approved '
            'targets to the configured home',
            require_workflow=False,
            allow_missing_obstacles=use_bootstrap_scene,
        )
        return True, ''

    def _terminal_abort(self, reason):
        was_active = self.state in ACTIVE_STATES
        self.state = 'ABORTED'
        self.reason = reason
        self.state_started = self.now()
        self.command_target = None
        self.current_path = []
        self.current_path_velocities = []
        self.current_path_accelerations = []
        self.current_path_times = []
        self.current_path_streaming = False
        self.current_trajectory = None
        self.motion_started_at = None
        self.waypoint_started_at = None
        self.waypoint_last_progress_at = None
        self.waypoint_best_error = math.inf
        self.current_waypoint_error = 0.0
        self.pending_motion_reason = ''
        self.runtime_refresh_require_workflow = False
        self.runtime_refresh_allow_missing_obstacles = False
        self.runtime_refresh_resume_state = ''
        self.acquisition_refresh_started = None
        self.acquisition_request_id = ''
        self.acquisition_request_attempt = 0
        self.acquisition_min_image_stamp_ns = 0
        self.acquisition_job_image_stamp_ns = 0
        self.acquisition_job_started = None
        self.acquisition_detection_completed = None
        self.acquisition_waiting_for_worker = False
        self.pending_acquisition_heavy_status = None
        self.acquisition_scene_snapshot_validated = False
        self.abort_return_in_progress = False
        self.abort_return_reason = ''
        self.abort_return_bootstrap_static_scene = False
        if was_active:
            self.publish_hold()
        self.publish_status()
        self.get_logger().error(reason)

    def invalidate_plan(
            self, reason, candidate_count=None, plan_kind=None,
            source_request_id=None, plan_id=None):
        if self.state in ACTIVE_STATES:
            return
        preserved_plan_kind = self.plan_kind
        preserved_plan_backend = getattr(
            self, 'plan_backend', 'tesseract')
        preserved_source_request_id = self.plan_source_request_id
        preserved_candidate_count = self.plan_candidate_count
        preserved_retrace = [
            np.asarray(value, dtype=float).copy()
            for value in self.retrace_joint_targets]
        self.clear_plan()
        # A proposal rejection must not erase endpoints that were already
        # executed under a separately approved acquisition plan. They are the
        # only collision-qualified authority for a benign failure return.
        self.retrace_joint_targets = preserved_retrace
        self.plan_id = str(plan_id or '')
        self.plan_backend = preserved_plan_backend
        self.plan_kind = (
            plan_kind
            if plan_kind in (MULTIVIEW_SCAN, ROUGH_ACQUISITION, RETURN_HOME)
            else preserved_plan_kind)
        self.plan_source_request_id = (
            str(source_request_id)
            if source_request_id is not None
            else preserved_source_request_id)
        self.plan_candidate_count = max(
            0,
            int(
                preserved_candidate_count
                if candidate_count is None else candidate_count))
        self.state = 'INVALID'
        self.reason = reason
        self.state_started = self.now()
        if self.param_bool('debug'):
            self.get_logger().warn('INVALID: %s' % reason)
        self.publish_plan(False, reason)
        self.publish_status()

    def clear_plan(self):
        self.plan_id = ''
        self.plan_backend = ''
        self.plan_kind = MULTIVIEW_SCAN
        self.plan_source_request_id = ''
        self.plan_created = 0.0
        self.plan_targets = []
        self.plan_paths = []
        self.plan_path_velocities = []
        self.plan_path_accelerations = []
        self.plan_path_times = []
        self.plan_bootstrap_recovery_end_points = []
        self.plan_bootstrap_recovery_joints = []
        self.plan_bootstrap_recovery_joint_sets = []
        self.plan_powered_start_recovery_end_points = []
        self.plan_powered_start_recovery_joint_sets = []
        self.plan_startup_home_static = []
        self.plan_configured_home_direct = []
        self.plan_configured_home_stages = []
        self.plan_segment_execution_modes = []
        self.plan_viewpoints = []
        self.plan_candidate_count = 0
        self.plan_capture_count = 0
        self.plan_returns_home = False
        self.return_home_warning = ''
        self.abort_return_in_progress = False
        self.abort_return_reason = ''
        self.abort_return_bootstrap_static_scene = False
        self.retrace_joint_targets = []
        self.plan_target_center = None
        self.plan_source_trajectory_sha256 = ''
        self.plan_trajectory_sha256 = ''
        self.plan_execution_speed_percent = 0.0
        self.plan_motion_limits_sha256 = ''
        self.runtime_motion_limits_sha256 = ''
        self.plan_collision_model_qualified = False
        self.pending_limit_refresh_plan = None
        self.pending_limit_refresh_deadline = 0.0
        self.runtime_recovery_started_at = None
        self.current_view = 0
        self.current_path = []
        self.current_path_velocities = []
        self.current_path_accelerations = []
        self.current_path_times = []
        self.current_path_streaming = False
        self.current_trajectory = None
        self.path_index = 0
        self.command_target = None
        self.command_sent_at = 0.0
        self.command_samples_sent = 0
        self.dropped_command_samples = 0
        self.max_command_interval_sec = 0.0
        self.motion_started_at = None
        self.last_motion_status_at = 0.0
        self.waypoint_started_at = None
        self.waypoint_last_progress_at = None
        self.waypoint_best_error = math.inf
        self.current_waypoint_error = 0.0
        self.max_waypoint_error = 0.0
        self.acquisition_refresh_started = None
        self.acquisition_request_id = ''
        self.acquisition_request_attempt = 0
        self.acquisition_min_image_stamp_ns = 0
        self.acquisition_job_image_stamp_ns = 0
        self.acquisition_job_started = None
        self.acquisition_detection_completed = None
        self.acquisition_waiting_for_worker = False
        self.pending_acquisition_heavy_status = None
        self.acquisition_scene_snapshot_validated = False

    def set_state(self, state, reason):
        self.state = state
        self.reason = reason
        self.state_started = self.now()
        self.publish_status()
        if self.param_bool('debug'):
            self.get_logger().info('%s: %s' % (state, reason))

    def publish_plan(self, valid, reason):
        msg = ScanExecutionPlan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.plan_id = self.plan_id
        msg.plan_kind = self.plan_kind
        msg.source_request_id = self.plan_source_request_id
        msg.planner_backend = (
            self.plan_backend or self.mission_planner_backend or 'tesseract')
        msg.trajectory_sha256 = self.plan_trajectory_sha256
        msg.motion_limits_sha256 = self.plan_motion_limits_sha256
        msg.execution_speed_percent = float(
            self.plan_execution_speed_percent)
        msg.command_rate_hz = float(
            configured_value(self, 'trajectory_command_rate_hz'))
        msg.timing_policy = EXECUTION_TIMING_POLICY_VERSION
        msg.valid = bool(valid)
        msg.dry_run = True
        msg.real_arm_motion = False
        msg.collision_model_qualified = bool(self.plan_collision_model_qualified)
        msg.reason = reason
        msg.candidate_viewpoints = int(self.plan_candidate_count)
        msg.planned_viewpoints = self.plan_capture_count
        msg.trajectory_points = sum(len(path) for path in self.plan_paths)
        if self.plan_target_center is not None:
            msg.target_center = Point(
                x=float(self.plan_target_center[0]),
                y=float(self.plan_target_center[1]),
                z=float(self.plan_target_center[2]),
            )
        for viewpoint, target in zip(self.plan_viewpoints, self.plan_targets):
            msg.viewpoint_indices.append(int(viewpoint.get('index', 0)))
            msg.joint_positions.extend(float(value) for value in target)
            camera = self.vector(viewpoint.get('desired_camera_position'))
            look = self.vector(viewpoint.get('desired_look_at_direction'))
            if camera is not None:
                point = Point()
                point.x, point.y, point.z = [float(value) for value in camera]
                msg.camera_positions.append(point)
            if look is not None:
                vector = Vector3()
                vector.x, vector.y, vector.z = [float(value) for value in look]
                msg.look_directions.append(vector)
        self.plan_pub.publish(msg)

    def publish_status(self):
        msg = ScanExecutionStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.plan_id = self.plan_id
        msg.execution_mode = self.plan_kind
        msg.state = self.state
        msg.reason = self.reason
        active = self.state in ACTIVE_STATES
        command_active = (
            active and self.state != 'WAITING_FOR_RUNTIME_REFRESH')
        msg.dry_run = not command_active
        msg.real_arm_motion = command_active
        msg.approval_required = self.state == 'PROPOSAL_READY'
        msg.current_view = (
            min(self.current_view + 1, self.plan_capture_count)
            if self.plan_capture_count else 0)
        msg.total_views = self.plan_capture_count
        msg.commanded_speed_percent = self.execution_speed_percent() if active else 0.0
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            tracking_health = self.latest_tracking_health
        else:
            observation = telemetry_store.snapshot().perception.tracking
            tracking_health = (
                None if observation is None else observation.value)
        msg.tracking_speed_scale = float(
            tracking_health.recommended_speed_scale
            if tracking_health is not None else 0.0)
        msg.max_joint_error_rad = float(
            self.current_waypoint_error
            if self.command_target is not None else 0.0)
        self.status_pub.publish(msg)

    def publish_scan_history(self):
        coverage_target_center = None
        frozen_center = getattr(self, 'scan_coverage_target_center', None)
        if frozen_center is not None:
            coverage_target_center = dict(zip(
                ('x', 'y', 'z'),
                (float(value) for value in frozen_center)))
        msg = String()
        msg.data = json.dumps({
            'session_id': self.scan_session_id,
            'accepted_views': len(self.scan_history),
            'max_views': int(
                configured_value(self, 'max_execution_viewpoints')),
            'entries': list(self.scan_history),
            'rejected_entries': list(self.scan_rejections),
            'latest_achieved_camera': (
                dict(getattr(self, 'latest_achieved_scan_view', None))
                if isinstance(
                    getattr(self, 'latest_achieved_scan_view', None), dict)
                else None),
            'coverage_target_center': coverage_target_center,
            'qualified_target_shape': (
                dict(getattr(self, 'scan_qualified_target_shape', None))
                if isinstance(
                    getattr(self, 'scan_qualified_target_shape', None), dict)
                else None),
            'qualified_target_model_seed': (
                dict(getattr(
                    self, 'scan_qualified_target_model_seed', None))
                if isinstance(getattr(
                    self, 'scan_qualified_target_model_seed', None), dict)
                else None),
        }, sort_keys=True)
        self.scan_history_pub.publish(msg)

    def current_joints(self, snapshot=None):
        if snapshot is None:
            telemetry_store = getattr(self, 'telemetry_store', None)
            if telemetry_store is None:
                joints = self.latest_joint_state
            else:
                observation = telemetry_store.snapshot().arm.joints
                joints = None if observation is None else observation.value
        else:
            observation = snapshot.arm.joints
            joints = None if observation is None else observation.value
        if joints is None or len(joints.position) < 6:
            raise ValueError('joint feedback has fewer than six arm joints')
        values = np.asarray(joints.position[:6], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError('joint feedback contains non-finite values')
        return values

    def max_joint_error(self, target):
        try:
            current = self.current_joints()
            goal = np.asarray(target, dtype=float)
            if self.is_startup_wrist_direct():
                return float(abs(current[5] - goal[5]))
            return float(np.max(np.abs(current - goal)))
        except ValueError:
            return math.inf

    def total_joint_error(self, target):
        try:
            current = self.current_joints()
            goal = np.asarray(target, dtype=float)
            if self.is_startup_wrist_direct():
                return float(abs(current[5] - goal[5]))
            return joint_progress_error(current, goal)
        except ValueError:
            return math.inf

    @staticmethod
    def vector(value):
        if not isinstance(value, dict):
            return None
        try:
            result = np.asarray([value['x'], value['y'], value['z']], dtype=float)
        except (KeyError, TypeError, ValueError):
            return None
        return result if np.all(np.isfinite(result)) else None

    def speed_percent(self):
        return max(1.0, min(100.0, float(configured_value(self, 'speed_percent'))))

    def execution_speed_percent(self):
        if self.plan_execution_speed_percent > 0.0:
            return float(self.plan_execution_speed_percent)
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            tracking_health = self.latest_tracking_health
        else:
            observation = telemetry_store.snapshot().perception.tracking
            tracking_health = (
                None if observation is None else observation.value)
        scale = (
            float(tracking_health.recommended_speed_scale)
            if tracking_health is not None else 0.0)
        return commanded_speed_percent(
            float(configured_value(self, 'speed_percent')),
            self.plan_kind,
            scale,
        )

    def is_acquisition(self):
        return self.plan_kind == ROUGH_ACQUISITION

    def is_return_home(self):
        return self.plan_kind == RETURN_HOME

    def is_powered_start_home_recovery(self):
        endpoints = getattr(
            self, 'plan_powered_start_recovery_end_points', [])
        return bool(
            self.is_return_home()
            and self.current_view == 0
            and len(endpoints) == 1
            and int(endpoints[0]) == 1
            and self.plan_returns_home
            and self.plan_capture_count == 0)

    def is_startup_home_static(self):
        markers = getattr(self, 'plan_startup_home_static', [])
        return bool(
            getattr(self, 'plan_kind', '') == RETURN_HOME
            and getattr(self, 'current_view', -1) == 0
            and markers == [True]
            and bool(getattr(self, 'plan_returns_home', False))
            and int(getattr(self, 'plan_capture_count', -1)) == 0)

    def is_configured_home_direct(self):
        markers = getattr(self, 'plan_configured_home_direct', [])
        return bool(
            self.is_return_home()
            and self.current_view == 0
            and markers == [True]
            and self.plan_returns_home
            and self.plan_capture_count == 0)

    def is_startup_wrist_direct(self):
        """Return true only for the authenticated direct STARTUP_WRIST stage."""
        stages = getattr(self, 'plan_configured_home_stages', [])
        return bool(
            self.is_configured_home_direct()
            and stages == ['STARTUP_WRIST'])

    def real_motion_enabled(self):
        return self.param_bool('enable_real_arm_motion')

    def param_bool(self, name):
        value = configured_value(self, name)
        if isinstance(value, str):
            return value.lower() in ('1', 'true', 'yes', 'on')
        return bool(value)


def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = ScanViewpointExecutorNode()
    except (KeyError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print('scan viewpoint executor startup error: %s' % error)
        rclpy.shutdown()
        return
    try:
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
    finally:
        if executor is not None:
            executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
