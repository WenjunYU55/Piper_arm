#!/usr/bin/env python3
"""Operator-approved execution of collision-validated scan viewpoints."""

import json
import math
import time
import numpy as np
import rclpy
from geometry_msgs.msg import Point, Vector3
from piper_msgs.msg import PiperMotionLimits, PiperStatusMsg
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
    TesseractPlan,
    TrackedTarget,
    TrackingHealth,
)
from piper_mobile_manipulation.scan_execution_modes import (
    acquired_target_rejection,
    commanded_speed_percent,
    correlated_obstacle_scene_status,
    heavy_refresh_status_action,
    MULTIVIEW_SCAN,
    plan_count_rejection,
    planned_speed_rejection,
    ROUGH_ACQUISITION,
    uses_bootstrap_static_scene,
)
from piper_mobile_manipulation.scan_motion import (
    approval_rejection_reason,
    bootstrap_recovery_declaration_reasons,
    bootstrap_start_limit_recovery_reasons,
    CollisionBox,
    energized_hold_target,
    feedback_joint_limit_reasons,
    interpolate_joint_path,
    PiperScanKinematics,
    load_accepted_hand_eye,
    load_conservative_joint_limits,
    validate_monotonic_self_clearance_escape,
    validate_joint_path,
)
from piper_mobile_manipulation.motion_limit_stability import MotionLimitStability
from piper_mobile_manipulation.scan_trajectory import (
    TIMING_POLICY_VERSION,
    validate_sdk_movej_waypoint_path,
    validate_tesseract_point,
)
from piper_mobile_manipulation.srv import ApproveScanExecution, AuthorizeMission


ACTIVE_STATES = {
    'WAITING_FOR_RUNTIME_REFRESH',
    'MOVING', 'SETTLING', 'SETTLING_HOME',
    'CAPTURING', 'CAPTURING_RGBD', 'WAIT_CAPTURE',
    'WAITING_FOR_FRESH_FRAME', 'WAITING_FOR_GROUNDING_DINO',
    'WAITING_FOR_TRACKING_LOCK', 'WAITING_FOR_OBSTACLE_SCENE',
}
MAX_RGBD_TRANSFORM_RETRIES = 3


def retryable_rgbd_capture_rejection(message):
    """Classify the small TF publication lag seen immediately after settle."""
    text = str(message).lower()
    return (
        'timestamped camera transform is unavailable' in text
        and 'extrapolation into the future' in text
    )


def approved_return_home_obstacle_snapshot(
        returning_home, collision_model_qualified, obstacles):
    """Keep the approval-time scene for the exact planned home segment."""
    return bool(
        returning_home
        and collision_model_qualified
        and obstacles is not None
    )


def bootstrap_abort_retrace_uses_static_scene(
        plan_kind, viewpoint_index, collision_model_qualified):
    """Mirror the approval scene for the exact first acquisition retrace.

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
        and 'configured home reached' in str(reason).lower()
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


def runtime_refresh_action(reasons, elapsed_sec, timeout_sec):
    if not reasons:
        return 'start'
    if float(elapsed_sec) >= float(timeout_sec):
        return 'abort'
    return 'wait'


def runtime_gate_action(reasons):
    """Hold for transient transport freshness gaps; abort real safety faults."""
    if not reasons:
        return 'continue'
    if all(str(reason).endswith('data missing or stale') for reason in reasons):
        return 'hold_for_refresh'
    return 'abort'


def abort_return_home_blocker(reason):
    """Reject automatic retrace when the abort implies unsafe arm motion."""
    text = str(reason).strip().lower()
    blockers = (
        'emergency stop',
        'collision',
        'clearance',
        'obstacle',
        'scene_blocked',
        'invalid geometry',
        'joint feedback became invalid',
        'outside configured limits',
        'motion limits',
        'arm status',
        'arm is not enabled',
        'err_code',
        'waypoint did not reach',
        'no measurable joint progress',
        'command publisher',
    )
    return next((item for item in blockers if item in text), '')


def approved_retrace_validation_reasons(reasons):
    """Keep changing-scene checks without re-rejecting an executed path.

    Every target in the abort history is an endpoint that this executor has
    already reached from an approval-bound, collision-qualified Tesseract
    path.  Reversing those same SDK MoveJ segments cannot introduce a new
    robot self-collision.  The generic validator can nevertheless report the
    folded-home contact again because it does not have the proposal's bounded
    recovery metadata.  Ignore only that static self-clearance duplicate;
    obstacle, floor, limit and malformed-path failures remain blockers.
    """
    return [
        str(reason) for reason in reasons
        if 'self-collision clearance between link segments' not in str(reason)
    ]


def rgbd_capture_handoff_action(
        request_inflight, state_age_sec, propagation_sec):
    """Sequence the status authorization before the RGB-D service request."""
    if bool(request_inflight):
        return 'wait_response'
    if float(state_age_sec) < max(0.0, float(propagation_sec)):
        return 'publish_authorization'
    return 'request_capture'


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
            'SETTLING', 'CAPTURING', 'CAPTURING_RGBD', 'WAIT_CAPTURE'))


def waypoint_motion_action(
        error_rad,
        reached_tolerance_rad,
        waypoint_elapsed_sec,
        waypoint_timeout_sec,
        progress_elapsed_sec,
        progress_timeout_sec,
):
    """Classify feedback for one SDK MoveJ position target."""
    error = float(error_rad)
    if not math.isfinite(error):
        return 'abort_invalid'
    if error <= float(reached_tolerance_rad):
        return 'advance'
    if float(waypoint_elapsed_sec) > float(waypoint_timeout_sec):
        return 'abort_timeout'
    if float(progress_elapsed_sec) > float(progress_timeout_sec):
        return 'abort_stalled'
    return 'wait'


def joint_progress_error(current, target):
    """Measure total remaining motion so progress by any joint is visible."""
    current_values = np.asarray(current, dtype=float)
    target_values = np.asarray(target, dtype=float)
    if (
            current_values.shape != (6,)
            or target_values.shape != (6,)
            or not np.all(np.isfinite(current_values))
            or not np.all(np.isfinite(target_values))):
        return math.inf
    return float(np.sum(np.abs(current_values - target_values)))


def target_drift_before_approval_rejection(
        drift_m, maximum_m, allow_target_motion):
    if bool(allow_target_motion):
        return ''
    if float(drift_m) > float(maximum_m):
        return 'target moved %.3fm after planning; refresh the plan' % float(
            drift_m)
    return ''


class ScanViewpointExecutorNode(Node):
    def __init__(self):
        super().__init__('scan_viewpoint_executor')
        defaults = {
            'reachable_viewpoints_topic': '/piper/reachable_scan_viewpoints',
            'joint_states_topic': '/joint_states_single',
            'arm_status_topic': '/arm_status',
            'motion_limits_topic': '/piper/motion_limits',
            'tracking_health_topic': '/piper/tracking_health',
            'tracked_target_topic': '/piper/tracked_target',
            'camera_timestamp_health_topic': '/piper/camera_timestamp_health',
            'target_status_topic': '/piper/target_status',
            'obstacle_topic': '/piper/obstacle_instances_3d',
            'workflow_status_topic': '/piper/supervised_workflow_status',
            'scan_session_history_topic': '/piper/scan_session_history',
            'joint_command_topic': '/joint_ctrl_single',
            'plan_topic': '/piper/scan_execution_plan',
            'status_topic': '/piper/scan_execution_status',
            'capture_service': '/supervised_cube_workflow/capture_view',
            'finish_scan_service': '/supervised_cube_workflow/finish_scan',
            'rgbd_capture_service': '/scan_capture/capture_view',
            'heavy_refresh_request_topic': '/piper/heavy_refresh_request',
            'heavy_refresh_status_topic': '/piper/heavy_refresh_status',
            'tesseract_plan_topic': '/piper/tesseract_plan',
            'hand_eye_calibration_path': '',
            'joint_bounds_path': '',
            'enable_real_arm_motion': False,
            'auto_capture': True,
            'speed_percent': 5.0,
            'max_execution_viewpoints': 13,
            'min_execution_viewpoints': 13,
            'trajectory_joint_step_rad': 0.025,
            'trajectory_command_rate_hz': 100.0,
            'executor_tick_rate_hz': 200.0,
            'plan_start_tolerance_rad': 0.025,
            'joint_goal_tolerance_rad': 0.025,
            'waypoint_reached_tolerance_rad': 0.025,
            'waypoint_progress_epsilon_rad': 0.001,
            'waypoint_timeout_sec': 90.0,
            'waypoint_progress_timeout_sec': 20.0,
            'joint_velocity_settled': 0.20,
            'home_motion_tolerance_rad': 0.005,
            'home_joint_feedback_timeout_sec': 1.0,
            'home_settle_duration_sec': 1.0,
            'home_settle_timeout_sec': 30.0,
            'joint_feedback_limit_tolerance_rad': 0.001,
            'motion_limits_timeout_sec': 3.0,
            'motion_limits_change_confirmation_sec': 7.0,
            'motion_limits_change_minimum_samples': 3,
            'runtime_refresh_timeout_sec': 3.0,
            'runtime_recovery_timeout_sec': 10.0,
            'settle_duration_sec': 1.5,
            'settle_timeout_sec': 15.0,
            'capture_timeout_sec': 20.0,
            'finish_scan_timeout_sec': 10.0,
            # The capture node independently authorizes service-mode saves
            # from the executor status topic.  Give DDS time to deliver the
            # CAPTURING_RGBD state before issuing the service request.
            'capture_status_propagation_sec': 0.25,
            'acquisition_fresh_frame_timeout_sec': 10.0,
            'acquisition_grounding_timeout_sec': 60.0,
            'acquisition_tracking_lock_timeout_sec': 10.0,
            'acquisition_scene_timeout_sec': 15.0,
            'acquisition_target_tolerance_m': 0.30,
            'acquisition_max_viewpoints': 5,
            'data_timeout_sec': 2.0,
            'max_tracking_measurement_age_sec': 0.75,
            'min_tracking_speed_scale': 0.10,
            # Planning a collision-qualified 13-view proposal can consume most
            # of the bridge's 180-second request window.  Start-state,
            # obstacle, camera, tracking, arm and motion-limit checks are all
            # repeated at approval, so leave enough post-result time for an
            # operator to inspect and confirm the exact hash.
            'plan_max_age_sec': 300.0,
            'max_target_drift_before_approval_m': 0.015,
            'allow_target_motion_during_scan': True,
            # Nearest one-joint collision-qualified low-drop adjustment from
            # GUI feedback. It is part of the exact trajectory/approval hash.
            'return_home_positions_rad': [
                0.000366362, 0.0, 0.0, 0.0, 0.43869236, 0.0],
            'floor_z_m': 0.0,
            'link_radius_m': 0.025,
            'self_clearance_m': 0.060,
            'approval_confirmation': 'EXECUTE APPROVED SCAN',
            'allow_mission_policy': False,
            'debug': True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        calibration_path = str(self.get_parameter('hand_eye_calibration_path').value)
        bounds_path = str(self.get_parameter('joint_bounds_path').value)
        if not calibration_path:
            raise RuntimeError('hand_eye_calibration_path is required')
        if not bounds_path:
            raise RuntimeError('joint_bounds_path is required')
        self.kinematics = PiperScanKinematics(load_accepted_hand_eye(calibration_path))
        self.joint_limits, ignored_bounds = load_conservative_joint_limits(bounds_path)

        self.latest_scan = None
        self.latest_joint_state = None
        self.latest_arm_status = None
        self.latest_motion_limits = None
        self.motion_limit_stability = MotionLimitStability(
            self.get_parameter(
                'motion_limits_change_confirmation_sec').value,
            self.get_parameter(
                'motion_limits_change_minimum_samples').value,
        )
        self.latest_tracking_health = None
        self.latest_tracked_target = None
        self.latest_camera_timestamp_health = None
        self.latest_target_status = 'UNKNOWN'
        self.latest_obstacles = None
        self.latest_workflow = None
        self.scan_session_id = ''
        self.scan_history = []
        self.updated = {}
        self.state = 'IDLE'
        self.reason = 'waiting for a validated viewpoint proposal'
        self.plan_id = ''
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
        self.current_view = 0
        self.current_path = []
        self.current_path_velocities = []
        self.current_path_accelerations = []
        self.current_path_times = []
        self.path_index = 0
        self.command_target = None
        self.command_sent_at = 0.0
        self.command_samples_sent = 0
        self.max_command_interval_sec = 0.0
        self.motion_started_at = None
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
        self.settle_started = None
        self.home_settle_previous_joints = None
        self.home_settle_last_joint_update = -1e9
        self.home_settle_last_sample_ok = False
        self.state_started = self.now()
        self.capture_future = None
        self.rgbd_capture_future = None
        self.rgbd_capture_attempts = 0
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
        self.acquisition_scene_snapshot_validated = False
        self.mission_task_id = ''
        self.mission_sha256 = ''
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
            self.get_parameter('plan_topic').value,
            history_qos,
        )
        self.status_pub = self.create_publisher(
            ScanExecutionStatus, self.get_parameter('status_topic').value, 10)
        self.scan_history_pub = self.create_publisher(
            String, self.get_parameter('scan_session_history_topic').value,
            history_qos)
        self.command_pub = None
        if self.real_motion_enabled():
            self.command_pub = self.create_publisher(
                JointState, self.get_parameter('joint_command_topic').value, 10)
        self.create_subscription(
            String, self.get_parameter('reachable_viewpoints_topic').value,
            self.scan_cb, 10)
        self.tesseract_plan_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.tesseract_plan_sub = self.create_subscription(
            TesseractPlan, self.get_parameter('tesseract_plan_topic').value,
            self.tesseract_plan_cb, self.tesseract_plan_qos)
        self.create_subscription(
            JointState, self.get_parameter('joint_states_topic').value,
            self.joint_cb, 10)
        self.create_subscription(
            PiperStatusMsg, self.get_parameter('arm_status_topic').value,
            self.arm_status_cb, 10)
        self.create_subscription(
            PiperMotionLimits, self.get_parameter('motion_limits_topic').value,
            self.motion_limits_cb, 10)
        self.create_subscription(
            TrackingHealth, self.get_parameter('tracking_health_topic').value,
            self.tracking_health_cb, 10)
        self.create_subscription(
            TrackedTarget, self.get_parameter('tracked_target_topic').value,
            self.tracked_target_cb, 10)
        self.create_subscription(
            CameraTimestampHealth,
            self.get_parameter('camera_timestamp_health_topic').value,
            self.camera_timestamp_health_cb, 10)
        self.create_subscription(
            String, self.get_parameter('target_status_topic').value,
            self.target_status_cb, 10)
        self.create_subscription(
            ObstacleInstance3DArray, self.get_parameter('obstacle_topic').value,
            self.obstacle_cb, 10)
        self.create_subscription(
            String, self.get_parameter('workflow_status_topic').value,
            self.workflow_cb, 10)
        self.create_service(ApproveScanExecution, '~/approve', self.approve_cb)
        self.create_service(
            AuthorizeMission, '~/authorize_mission', self.authorize_mission_cb)
        self.create_service(Trigger, '~/cancel', self.cancel_cb)
        self.create_service(Trigger, '~/refresh_plan', self.refresh_cb)
        self.create_service(Trigger, '~/diagnostic_state', self.diagnostic_state_cb)
        self.capture_client = self.create_client(
            Trigger, self.get_parameter('capture_service').value)
        self.rgbd_capture_client = self.create_client(
            Trigger, self.get_parameter('rgbd_capture_service').value)
        self.finish_scan_client = self.create_client(
            Trigger, self.get_parameter('finish_scan_service').value)
        self.heavy_refresh_pub = self.create_publisher(
            String, self.get_parameter('heavy_refresh_request_topic').value, 10)
        self.create_subscription(
            String, self.get_parameter('heavy_refresh_status_topic').value,
            self.heavy_refresh_status_cb, 10)
        command_rate = float(self.get_parameter('trajectory_command_rate_hz').value)
        if not math.isfinite(command_rate) or command_rate <= 0.0:
            raise RuntimeError('trajectory_command_rate_hz must be finite and positive')
        tick_rate = float(self.get_parameter('executor_tick_rate_hz').value)
        if (
                not math.isfinite(tick_rate)
                or tick_rate < 2.0 * command_rate):
            raise RuntimeError(
                'executor_tick_rate_hz must be at least twice '
                'trajectory_command_rate_hz')
        self.create_timer(1.0 / tick_rate, self.tick)

        mode = 'motion opt-in available' if self.real_motion_enabled() else 'proposal-only'
        self.get_logger().warn(
            'Scan viewpoint executor started %s at %.1f%% configured speed; '
            'Tesseract all-six-joint planning is mandatory. '
            'Ignored saved bounds: %s'
            % (mode, self.speed_percent(),
               ','.join(ignored_bounds) or 'none'))
        self.publish_status()
        self.publish_scan_history()

    def now(self):
        return time.monotonic()

    def mark(self, key):
        self.updated[key] = self.now()

    def fresh(self, key, timeout=None):
        maximum = float(self.get_parameter('data_timeout_sec').value) \
            if timeout is None else float(timeout)
        return self.now() - self.updated.get(key, -1e9) <= maximum

    def scan_cb(self, msg):
        if self.state in ACTIVE_STATES:
            return
        try:
            payload = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError) as error:
            self.invalidate_plan('invalid reachable viewpoint JSON: %s' % error)
            return
        self.latest_scan = payload
        self.mark('scan')
        if self.state == 'ABORTED':
            self.state = 'IDLE'

    def tesseract_plan_cb(self, msg):
        if self.state in ACTIVE_STATES:
            return
        plan_kind = str(msg.plan_kind)
        source_request_id = str(msg.source_request_id)
        if not msg.valid:
            self.invalidate_plan(
                'Tesseract proposal rejected: ' + msg.reason,
                plan_kind=plan_kind,
                source_request_id=source_request_id,
                plan_id=str(msg.plan_id),
            )
            return
        reasons = []
        if not msg.dry_run or msg.real_arm_motion:
            reasons.append('Tesseract proposal flags are not command-free')
        if msg.backend != 'tesseract':
            reasons.append('unexpected planning backend: %s' % msg.backend)
        if not msg.plan_id or len(msg.trajectory_sha256) != 64:
            reasons.append('plan identity or trajectory hash is invalid')
        if str(msg.timing_policy) != TIMING_POLICY_VERSION:
            reasons.append('Tesseract timing policy is unsupported')
        returns_home = (
            plan_kind == MULTIVIEW_SCAN
            and len(msg.trajectories) == len(msg.viewpoint_indices) + 1)
        if (
                len(msg.trajectories) != len(msg.viewpoint_indices)
                and not returns_home):
            reasons.append('trajectory and viewpoint counts differ')
        if plan_kind == MULTIVIEW_SCAN and not returns_home:
            reasons.append(
                'MULTIVIEW_SCAN must include one final return-home trajectory')
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
            int(self.get_parameter('min_execution_viewpoints').value),
            int(self.get_parameter('acquisition_max_viewpoints').value),
            session_accepted_views=(
                len(self.scan_history)
                if plan_kind == MULTIVIEW_SCAN else 0),
            session_maximum_views=(
                int(self.get_parameter('max_execution_viewpoints').value)
                if plan_kind == MULTIVIEW_SCAN else None),
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
        maximum_step = float(
            self.get_parameter('trajectory_joint_step_rad').value)
        command_rate = float(self.get_parameter('trajectory_command_rate_hz').value)
        if abs(float(msg.command_rate_hz) - command_rate) > 1e-6:
            reasons.append('Tesseract command rate does not match the executor')
        if not self.fresh(
                'motion_limits',
                float(self.get_parameter('motion_limits_timeout_sec').value)):
            reasons.append('controller motion limits are missing or stale')
        limits = self.latest_motion_limits
        if limits is None or not limits.valid:
            reasons.append('controller motion limits are invalid')
        else:
            if (
                    str(msg.motion_limits_sha256) != str(limits.limits_sha256)
                    or len(str(msg.motion_limits_sha256)) != 64):
                reasons.append(
                    'Tesseract controller-limit binding is stale or mismatched')
        tracking_scale = (
            float(self.latest_tracking_health.recommended_speed_scale)
            if self.latest_tracking_health is not None else 1.0)
        execution_speed = float(msg.execution_speed_percent)
        speed_rejection = planned_speed_rejection(
            float(self.get_parameter('speed_percent').value),
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
                        validate_tesseract_point(
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
            declared_joints = []
            declared_deltas = []
            if recovery_end >= 0:
                terminal_home_recovery = bool(
                    plan_kind == MULTIVIEW_SCAN
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
                if len(path) != 3 or recovery_end != 1:
                    reasons.append(
                        'segment %d bootstrap recovery must be one SDK MoveJ '
                        'target before the final viewpoint target'
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
                    except (TypeError, ValueError):
                        declared_joints = []
                        declared_deltas = []
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
            elif len(path) != 2:
                reasons.append(
                    'segment %d has an undeclared intermediate SDK MoveJ target'
                    % segment_index)
            try:
                emitted_path, emitted_velocities, emitted_accelerations, \
                    emitted_times = validate_sdk_movej_waypoint_path(
                        path,
                        velocities,
                        accelerations,
                        times,
                        command_rate_hz=command_rate,
                        maximum_step_rad=maximum_step,
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
        if returns_home and targets:
            configured_home = np.asarray(
                self.get_parameter('return_home_positions_rad').value,
                dtype=float)
            if configured_home.shape != (6,) or not np.all(
                    np.isfinite(configured_home)):
                reasons.append(
                    'configured return-home pose must contain six finite joints')
            elif float(np.max(np.abs(
                    targets[-1] - configured_home))) > 1e-6:
                reasons.append(
                    'Tesseract return-home endpoint does not match the '
                    'executor configuration')
        if reasons:
            self.invalidate_plan(
                'invalid Tesseract proposal: ' + '; '.join(reasons),
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
        self.plan_candidate_count = len(msg.viewpoint_indices)
        self.plan_capture_count = len(msg.viewpoint_indices)
        self.plan_returns_home = bool(returns_home)
        self.plan_target_center = np.asarray([
            msg.target_center.x, msg.target_center.y, msg.target_center.z,
        ], dtype=float)
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
            self.plan_viewpoints.append({
                'index': int(index),
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
        self.set_state(
            'PROPOSAL_READY',
            '%d six-joint Tesseract %s viewpoints%s use exact collision-checked '
            'SDK MoveJ position targets at %.1f%% (%s); '
            'exact trajectory hash approval required'
            % (
                self.plan_capture_count,
                self.plan_kind.lower(),
                ' plus an approved return-home segment'
                if self.plan_returns_home else '',
                self.plan_execution_speed_percent,
                qualification,
            ))
        self.publish_plan(True, self.reason)

    def joint_cb(self, msg):
        self.latest_joint_state = msg
        self.mark('joints')

    def arm_status_cb(self, msg):
        self.latest_arm_status = msg
        self.mark('arm_status')

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
            self.mark('motion_limits')

    def tracking_health_cb(self, msg):
        self.latest_tracking_health = msg
        self.mark('tracking')

    def tracked_target_cb(self, msg):
        self.latest_tracked_target = msg
        self.mark('tracked_target')

    def camera_timestamp_health_cb(self, msg):
        self.latest_camera_timestamp_health = msg
        self.mark('camera_clock')

    def target_status_cb(self, msg):
        self.latest_target_status = str(msg.data).upper()
        self.mark('target_status')

    def obstacle_cb(self, msg):
        self.latest_obstacles = msg
        self.mark('obstacles')

    def workflow_cb(self, msg):
        try:
            self.latest_workflow = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            return
        session_id = str(self.latest_workflow.get('session_id', ''))
        if session_id and session_id != self.scan_session_id:
            self.scan_session_id = session_id
            self.scan_history = []
            self.publish_scan_history()
        self.mark('workflow')

    def heavy_refresh_status_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            return
        if (
                not self.is_acquisition()
                or self.state not in (
                    'WAITING_FOR_FRESH_FRAME',
                    'WAITING_FOR_GROUNDING_DINO',
                    'WAITING_FOR_TRACKING_LOCK',
                )):
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
        if action == 'queued':
            self.acquisition_job_image_stamp_ns = int(image_stamp_ns)
            self.acquisition_job_started = self.now()
            self.set_state(
                'WAITING_FOR_GROUNDING_DINO',
                'matching GroundingDINO job queued on a fresh post-settle frame')
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
        if self.state == 'FINISHING_WORKFLOW':
            self.finish_workflow_tick()
            return
        if self.state in ACTIVE_STATES:
            self.execution_tick()
            return
        if self.state == 'PROPOSAL_READY':
            camera_health = self.latest_camera_timestamp_health
            if (
                    not self.fresh('camera_clock')
                    or camera_health is None
                    or not camera_health.healthy):
                # Approval repeats the fresh camera-clock gate. Preserve the
                # immutable proposal while that transient blocks approval so a
                # confirmation dialog cannot race a replacement plan/hash.
                return
            if self.now() - self.plan_created > float(
                    self.get_parameter('plan_max_age_sec').value):
                self.invalidate_plan('proposal expired; refresh viewpoints')
            return

    def approve_cb(self, request, response):
        expected = str(self.get_parameter('approval_confirmation').value)
        mission_confirmation = 'MISSION_POLICY:' + self.mission_sha256
        if str(request.confirmation) == mission_confirmation:
            if not self.mission_authorization_valid():
                response.accepted = False
                response.message = (
                    'autonomous execution is not bound to a live mission authorization')
                return response
            expected = mission_confirmation
        rejection = approval_rejection_reason(
            self.state,
            self.plan_id if self.plan_targets else '',
            request.plan_id,
            request.confirmation,
            expected,
            self.real_motion_enabled(),
            self.now() - self.plan_created,
            float(self.get_parameter('plan_max_age_sec').value),
            current_trajectory_sha256=self.plan_trajectory_sha256,
            requested_trajectory_sha256=request.trajectory_sha256,
            require_trajectory_hash=True,
        )
        if rejection:
            response.accepted = False
            response.message = rejection
            return response
        acquisition = self.is_acquisition()
        reasons = self.runtime_reasons(
            require_settled=True,
            require_workflow=not acquisition,
            allow_untracked=acquisition,
            allow_missing_obstacles=acquisition,
        )
        if reasons:
            response.accepted = False
            response.message = 'execution blocked: ' + '; '.join(reasons)
            return response
        if not self.plan_collision_model_qualified:
            response.accepted = False
            response.message = (
                'Tesseract collision model is proposal-only and not qualified for hardware')
            return response
        if not acquisition:
            latest_target_center = self.vector(
                self.latest_scan.get('target_object_center')
                if isinstance(self.latest_scan, dict) else None)
            if latest_target_center is None or self.plan_target_center is None:
                response.accepted = False
                response.message = 'latest target center is unavailable'
                return response
            target_drift = float(np.linalg.norm(latest_target_center - self.plan_target_center))
            drift_rejection = target_drift_before_approval_rejection(
                target_drift,
                self.get_parameter(
                    'max_target_drift_before_approval_m').value,
                self.get_parameter(
                    'allow_target_motion_during_scan').value,
            )
            if drift_rejection:
                response.accepted = False
                response.message = drift_rejection
                return response
        if (
                not acquisition
                and self.param_bool('auto_capture')
                and not self.capture_client.service_is_ready()):
            response.accepted = False
            response.message = 'capture service is not ready'
            return response
        if (
                not acquisition
                and self.param_bool('auto_capture')
                and not self.rgbd_capture_client.service_is_ready()):
            response.accepted = False
            response.message = 'RGB-D capture service is not ready'
            return response
        if (
                not acquisition
                and self.param_bool('auto_capture')
                and not self.finish_scan_client.service_is_ready()):
            response.accepted = False
            response.message = 'workflow finish service is not ready'
            return response
        path_reasons = self.prepare_current_view()
        if path_reasons:
            response.accepted = False
            response.message = 'fresh trajectory validation failed: ' + '; '.join(path_reasons)
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
                        self.get_parameter('plan_start_tolerance_rad').value)):
            # An uninterrupted acquisition+scan session can safely retrace
            # every already executed, separately approved endpoint to the
            # original loaded pose. A restarted executor falls back to the
            # current scan-start pose.
            self.retrace_joint_targets = [start_joints]
        start_reason = (
            'approved rough-target acquisition motion started'
            if acquisition else 'approved scan motion started')
        self.begin_runtime_refresh(
            start_reason,
            require_workflow=not acquisition,
            allow_missing_obstacles=acquisition,
        )
        response.accepted = True
        response.message = (
            'approved %d %s viewpoints at no more than %.1f%% speed; '
            'waiting for fresh runtime telemetry before motion'
        ) % (
            self.plan_capture_count,
            'acquisition' if acquisition else 'scan',
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
        expires = float(request.expires_at.sec) + float(
            request.expires_at.nanosec) * 1e-9
        now = time.time()
        if not task_id or len(digest) != 64 or any(
                character not in '0123456789abcdef' for character in digest):
            response.accepted = False
            response.message = 'mission identity or SHA-256 is invalid'
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
        self.mission_expires_at_sec = expires
        response.accepted = True
        response.message = 'mission policy authorization bound to task and deadline'
        return response

    def mission_authorization_valid(self):
        return (
            self.param_bool('allow_mission_policy')
            and bool(self.mission_task_id)
            and len(self.mission_sha256) == 64
            and time.time() < self.mission_expires_at_sec
        )

    def cancel_cb(self, _request, response):
        if self.state in ACTIVE_STATES:
            self.abort_motion('operator cancelled scan execution')
            response.success = True
            response.message = (
                'motion cancelled; approved-path return to configured home started'
                if self.abort_return_in_progress else
                'motion cancelled; current joint hold requested; request '
                'cancel again after the hold settles to return home')
            return response
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
        started, blocker = self.try_start_abort_return(
            'operator cancelled scan execution')
        if started:
            response.success = True
            response.message = (
                'proposal cancelled; approved-path return to configured home '
                'started'
                if self.abort_return_in_progress else
                'proposal cancelled; configured home reached; current joint '
                'hold requested')
            return response
        held = self.publish_hold()
        self._terminal_abort(
            'operator cancelled scan execution; automatic return home was '
            'not started: ' + (blocker or 'no approved retrace is available'))
        response.success = True
        response.message = (
            'proposal cancelled; current joint hold requested; automatic '
            'return home unavailable: ' + (
                blocker or 'no approved retrace is available')
            if held else
            'proposal cancelled; automatic return home and current joint hold '
            'were unavailable: ' + (
                blocker or 'no approved retrace is available')
        )
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
                self.get_parameter('trajectory_command_rate_hz').value),
            'motion_adapter': TIMING_POLICY_VERSION,
            'executor_tick_rate_hz': float(
                self.get_parameter('executor_tick_rate_hz').value),
            'command_samples_sent': self.command_samples_sent,
            'max_command_interval_sec': self.max_command_interval_sec,
            'current_waypoint_error_rad': self.current_waypoint_error,
            'max_waypoint_error_rad': self.max_waypoint_error,
            'planned_viewpoints': self.plan_capture_count,
            'scan_session_id': self.scan_session_id,
            'session_accepted_views': len(self.scan_history),
            'session_remaining_views': max(
                0,
                int(self.get_parameter('max_execution_viewpoints').value)
                - len(self.scan_history)),
            'returns_home': self.plan_returns_home,
            'first_view_max_joint_delta_rad': first_delta,
            'first_view_goal_rad': first_goal,
            'collision_model_qualified': self.plan_collision_model_qualified,
            'real_motion_enabled': self.real_motion_enabled(),
            'mission_policy_enabled': self.param_bool('allow_mission_policy'),
            'mission_task_id': self.mission_task_id,
            'mission_authorization_valid': self.mission_authorization_valid(),
            'camera_clock_fresh': self.fresh('camera_clock'),
            'camera_clock_healthy': bool(
                self.latest_camera_timestamp_health is not None
                and self.latest_camera_timestamp_health.healthy),
        }, sort_keys=True)
        return response

    def execution_tick(self):
        if self.state == 'WAITING_FOR_RUNTIME_REFRESH':
            self.waiting_for_runtime_refresh_tick()
            return
        reasons = self.runtime_reasons(
            require_settled=False,
            require_workflow=False,
            allow_untracked=(
                self.is_acquisition()
                or getattr(self, 'plan_kind', '') == MULTIVIEW_SCAN),
            allow_missing_obstacles=missing_obstacles_can_wait(
                self.plan_kind, self.current_view, self.state,
                getattr(
                    self, 'abort_return_bootstrap_static_scene', False)),
            allow_stale_obstacles=(
                self.is_acquisition()
                and self.acquisition_scene_snapshot_validated
            ) or bool(getattr(
                self, 'abort_return_bootstrap_static_scene', False
            )) or approved_return_home_obstacle_snapshot(
                self.returning_home(),
                self.plan_collision_model_qualified,
                self.latest_obstacles,
            ),
            # PiPER's SDK MoveJ speed is fixed when a target is issued. Camera
            # motion can lower the tracking speed recommendation while that
            # exact target is in flight, so enforce the allowance at the
            # approval gate rather than retroactively aborting an already
            # issued command. All non-tracking live safety gates remain
            # active, and the next target still requires a fresh refresh.
            enforce_tracking_speed_allowance=False,
            # Eye-in-hand confidence can dip while the camera itself is
            # moving, and the target is explicitly allowed to move during a
            # scan.  Once the exact multiview proposal is approved, tracking
            # is diagnostic rather than a motion/capture cancellation gate.
            enforce_target_status=False,
            enforce_tracking_motion_state=False,
        )
        if reasons:
            if self.returning_home():
                ScanViewpointExecutorNode.handle_return_home_failure(
                    self,
                    'return-home runtime safety gate stopped motion: '
                    + '; '.join(reasons))
                return
            if runtime_gate_action(reasons) == 'hold_for_refresh':
                self.begin_runtime_recovery(reasons)
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
        self.runtime_refresh_resume_state = resume_state
        self.runtime_refresh_require_workflow = False
        self.runtime_refresh_allow_missing_obstacles = False
        self.pending_motion_reason = (
            'fresh runtime telemetry restored; resuming the same approved '
            'Tesseract target')
        self.set_state(
            'WAITING_FOR_RUNTIME_REFRESH',
            'holding current position while runtime telemetry refreshes: '
            + '; '.join(reasons))

    def waiting_for_runtime_refresh_tick(self):
        reasons = self.runtime_reasons(
            require_settled=True,
            require_workflow=self.runtime_refresh_require_workflow,
            allow_untracked=(
                self.is_acquisition()
                or getattr(self, 'plan_kind', '') == MULTIVIEW_SCAN),
            allow_missing_obstacles=(
                self.runtime_refresh_allow_missing_obstacles),
            allow_stale_obstacles=(
                self.is_acquisition()
                and self.acquisition_scene_snapshot_validated),
        )
        recovering = bool(self.runtime_refresh_resume_state)
        timeout = float(self.get_parameter(
            'runtime_recovery_timeout_sec'
            if recovering else 'runtime_refresh_timeout_sec').value)
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
            self.set_state(resume_state, reason)
            if resume_state == 'MOVING' and self.command_target is not None:
                now = self.now()
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
        if self.command_target is None:
            if self.path_index >= len(self.current_path):
                if self.returning_home():
                    self.begin_return_home_settle()
                    return
                self.settle_started = None
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
            self.get_parameter('waypoint_progress_epsilon_rad').value)
        if progress_error + epsilon < self.waypoint_best_error:
            self.waypoint_best_error = progress_error
            self.waypoint_last_progress_at = now
        action = waypoint_motion_action(
            error,
            float(self.get_parameter(
                'waypoint_reached_tolerance_rad').value),
            now - float(self.waypoint_started_at),
            float(self.get_parameter('waypoint_timeout_sec').value),
            now - float(self.waypoint_last_progress_at),
            float(self.get_parameter(
                'waypoint_progress_timeout_sec').value),
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
            self.set_state(
                'SETTLING',
                'acquisition look reached; waiting for arm and camera to settle'
                if self.is_acquisition()
                else 'viewpoint reached; waiting for arm and camera to settle')
            return
        self.publish_next_waypoint(now)

    def publish_next_waypoint(self, now):
        target = self.current_path[self.path_index]
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
        if self.now() - self.state_started > float(
                self.get_parameter('settle_timeout_sec').value):
            self.abort_motion(
                'arm/camera did not settle before acquisition refresh'
                if self.is_acquisition()
                else 'arm/camera did not settle before timeout')
            return
        settled = (
            self.joints_settled()
            if self.is_acquisition() else self.capture_pose_settled())
        if not settled:
            self.settle_started = None
            return
        if self.settle_started is None:
            self.settle_started = self.now()
            return
        if self.now() - self.settle_started < float(
                self.get_parameter('settle_duration_sec').value):
            return
        if self.is_acquisition():
            self.request_acquisition_refresh()
            return
        if not self.param_bool('auto_capture'):
            self.advance_view()
            return
        if not self.workflow_ready():
            self.abort_motion('supervised workflow is not SCAN_READY at capture time')
            return
        self.capture_accepted_before = int(self.latest_workflow.get('accepted_views', 0))
        self.set_state(
            'CAPTURING_RGBD',
            'settled viewpoint reached; saving synchronized RGB-D record')
        self.rgbd_capture_future = None
        self.rgbd_capture_attempts = 0
        self.finish_scan_future = None

    def request_acquisition_refresh(self):
        self.acquisition_refresh_started = self.now()
        self.acquisition_request_attempt = 0
        self.acquisition_waiting_for_worker = False
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
                self.get_parameter('acquisition_fresh_frame_timeout_sec').value):
            self.abort_motion(
                'fresh post-settle camera frame or idle GroundingDINO worker '
                'did not become available before timeout')

    def waiting_for_grounding_dino_tick(self):
        if self.acquisition_job_started is None:
            self.abort_motion('GroundingDINO queue acknowledgement is missing')
            return
        if self.now() - self.acquisition_job_started > float(
                self.get_parameter('acquisition_grounding_timeout_sec').value):
            self.abort_motion(
                'matching GroundingDINO job exceeded the %.1f-second timeout'
                % float(self.get_parameter(
                    'acquisition_grounding_timeout_sec').value))

    def waiting_for_tracking_lock_tick(self):
        rejection = acquired_target_rejection(
            self.latest_tracked_target,
            self.latest_tracking_health,
            self.latest_target_status,
            self.updated.get('tracked_target', -1e9),
            self.updated.get('tracking', -1e9),
            self.updated.get('target_status', -1e9),
            self.acquisition_detection_completed,
            self.now(),
            float(self.get_parameter('data_timeout_sec').value),
            float(self.get_parameter('max_tracking_measurement_age_sec').value),
            self.acquisition_job_image_stamp_ns,
            self.plan_target_center,
            float(self.get_parameter('acquisition_target_tolerance_m').value),
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
                self.get_parameter('acquisition_tracking_lock_timeout_sec').value):
            return
        self.get_logger().warn(
            'GroundingDINO detection did not produce an acceptable measured lock: %s'
            % rejection)
        self.set_state(
            'WAITING_FOR_OBSTACLE_SCENE',
            'target lock was not established; waiting for the matching '
            'post-settle semantic scene')

    def waiting_for_obstacle_scene_tick(self):
        status, reason = correlated_obstacle_scene_status(
            self.latest_obstacles,
            self.updated.get('obstacles', -1e9),
            self.now(),
            float(self.get_parameter('data_timeout_sec').value),
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
                self.get_parameter('acquisition_scene_timeout_sec').value):
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
        if self.current_view >= len(self.plan_targets):
            self.command_target = None
            self.current_path = []
            self.current_path_times = []
            self.motion_started_at = None
            self.publish_hold()
            self.set_state(
                'ACQUISITION_FAILED',
                'bounded acquisition sweep completed without measured target lock')
            return
        runtime = self.runtime_reasons(
            require_settled=True,
            require_workflow=False,
            allow_untracked=True,
            allow_missing_obstacles=False,
        )
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
                self.get_parameter('capture_timeout_sec').value):
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
                self.get_parameter('capture_timeout_sec').value):
            self.abort_motion('RGB-D capture service response timed out')
            return
        action = rgbd_capture_handoff_action(
            self.rgbd_capture_future is not None,
            self.now() - self.state_started,
            self.get_parameter('capture_status_propagation_sec').value,
        )
        if action == 'publish_authorization':
            self.publish_status()
            return
        if action == 'request_capture':
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
            if (
                    retryable_rgbd_capture_rejection(message)
                    and self.rgbd_capture_attempts < MAX_RGBD_TRANSFORM_RETRIES):
                self.rgbd_capture_future = None
                self.set_state(
                    'CAPTURING_RGBD',
                    'camera transform is still catching up with the settled '
                    'frame; retrying the same viewpoint capture')
                return
            self.abort_motion('RGB-D viewpoint capture was rejected: %s' % message)
            return
        # Count a view only after its synchronized files exist. The workflow's
        # cloud/quality products are diagnostic and must not precede the
        # primary capture contract.
        self.capture_future = self.capture_client.call_async(Trigger.Request())
        self.set_state(
            'CAPTURING',
            'RGB-D viewpoint saved; recording viewpoint acceptance')

    def wait_capture_tick(self):
        if self.now() - self.state_started > float(
                self.get_parameter('capture_timeout_sec').value):
            self.abort_motion('accepted capture did not return workflow to SCAN_READY')
            return
        if not self.workflow_ready():
            return
        accepted = int(self.latest_workflow.get('accepted_views', 0))
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
        viewpoint = self.plan_viewpoints[self.current_view]
        try:
            joints = [float(value) for value in self.current_joints()]
        except ValueError as error:
            self.abort_motion('cannot record accepted viewpoint: %s' % error)
            return False
        self.scan_history.append({
            'accepted_view': int(accepted_views),
            'plan_id': self.plan_id,
            'viewpoint_index': int(viewpoint.get('index', self.current_view)),
            'desired_camera_position': dict(
                viewpoint.get('desired_camera_position', {})),
            'desired_look_at_direction': dict(
                viewpoint.get('desired_look_at_direction', {})),
            'joint_positions_rad': joints,
        })
        self.publish_scan_history()
        return True

    def advance_view(self):
        self.current_view += 1
        if self.current_view >= self.plan_capture_count:
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
        self.publish_hold()
        self.set_state(
            'SETTLING_HOME',
            (
                'approved-path abort retrace reached home; waiting for stable '
                'joint feedback'
                if getattr(self, 'abort_return_in_progress', False) else
                'approved home target reached; waiting for stable joint feedback'
            ))

    def return_home_settling_tick(self):
        if self.now() - self.state_started > float(
                self.get_parameter('home_settle_timeout_sec').value):
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
                self.get_parameter('home_settle_duration_sec').value):
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
                'targets; configured home reached')
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
                self.get_parameter('finish_scan_timeout_sec').value):
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
        self.scan_session_id = ''
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
            return ['Tesseract collision model is not qualified for hardware']
        path = [item.copy() for item in self.plan_paths[self.current_view]]
        path_times = [
            float(item) for item in self.plan_path_times[self.current_view]]
        if len(path) != len(path_times):
            return ['waypoint positions and order stamps differ in length']
        start_tolerance = float(
            self.get_parameter('plan_start_tolerance_rad').value)
        if float(np.max(np.abs(path[0] - start))) > start_tolerance:
            return ['current state changed beyond the approved plan-start tolerance']
        # A cumulative joint-distance gate rejects valid collision-aware
        # detours even though the executor never sends that cumulative
        # displacement as one command. The Tesseract path has already been
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
        obstacle_boxes = (
            [] if uses_bootstrap_static_scene(
                self.plan_kind, self.current_view)
            else self.obstacle_boxes())
        recovery_end = (
            self.plan_bootstrap_recovery_end_points[self.current_view]
            if self.current_view < len(self.plan_bootstrap_recovery_end_points)
            else -1)
        if recovery_end >= 0:
            acquisition_recovery = bool(
                self.is_acquisition() and self.current_view == 0)
            terminal_home_recovery = bool(
                self.plan_kind == MULTIVIEW_SCAN
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
                floor_z_m=float(self.get_parameter('floor_z_m').value),
                link_radius_m=float(self.get_parameter('link_radius_m').value),
                self_clearance_m=float(self.get_parameter('self_clearance_m').value),
                recovery_joint_number=(
                    self.plan_bootstrap_recovery_joint_sets[self.current_view]
                    if self.current_view
                    < len(self.plan_bootstrap_recovery_joint_sets) else None),
                maximum_start_limit_violation_rad=0.04,
            )
            if not reasons:
                normal_path = (
                    path[:recovery_end + 1]
                    if terminal_home_recovery else path[recovery_end:])
                reasons = self.validate_path(normal_path, obstacle_boxes)
        else:
            reasons = self.validate_path(path, obstacle_boxes)
        if reasons:
            return reasons
        self.current_path = command_path
        self.current_path_velocities = command_velocities
        self.current_path_accelerations = command_accelerations
        self.current_path_times = command_times
        self.path_index = 0
        self.command_target = None
        self.command_sent_at = 0.0
        self.command_samples_sent = 0
        self.max_command_interval_sec = 0.0
        self.motion_started_at = None
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
            floor_z_m=float(self.get_parameter('floor_z_m').value),
            link_radius_m=float(self.get_parameter('link_radius_m').value),
            self_clearance_m=float(self.get_parameter('self_clearance_m').value),
        )

    def validation_path(self, path):
        """Densely check SDK-interpolated segments without publishing samples."""
        if not path:
            return []
        maximum_step = float(
            self.get_parameter('trajectory_joint_step_rad').value)
        dense = [np.asarray(path[0], dtype=float).copy()]
        for endpoint in path[1:]:
            dense.extend(interpolate_joint_path(
                dense[-1], endpoint, maximum_step))
        return dense

    def runtime_reasons(
            self, require_settled, require_workflow, allow_untracked=False,
            allow_missing_obstacles=False,
            allow_stale_obstacles=False,
            enforce_tracking_speed_allowance=True,
            enforce_target_status=True,
            enforce_tracking_motion_state=True):
        reasons = []
        required_keys = ['joints', 'arm_status', 'camera_clock']
        if not allow_missing_obstacles and not allow_stale_obstacles:
            required_keys.append('obstacles')
        if not allow_untracked:
            required_keys.extend(['tracking', 'target_status'])
        for key in required_keys:
            if not self.fresh(key):
                reasons.append('%s data missing or stale' % key)
        if not self.fresh(
                'motion_limits',
                float(self.get_parameter('motion_limits_timeout_sec').value)):
            reasons.append('motion_limits data missing or stale')
        if reasons:
            return reasons
        limits = self.latest_motion_limits
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
            joints = self.current_joints()
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
                reasons.extend(feedback_joint_limit_reasons(
                    joints,
                    self.joint_limits,
                    float(self.get_parameter(
                        'joint_feedback_limit_tolerance_rad').value),
                ))
        except (IndexError, ValueError) as error:
            reasons.append(str(error))
        reasons.extend(self.arm_status_reasons())
        camera_health = self.latest_camera_timestamp_health
        if camera_health is None or not camera_health.healthy:
            state = camera_health.state if camera_health is not None else 'MISSING'
            detail = camera_health.reason if camera_health is not None else 'no watchdog status'
            reasons.append('camera timestamp %s: %s' % (state, detail))
        if not allow_missing_obstacles:
            if (
                    self.latest_obstacles.scene_blocked
                    and not self.latest_obstacles.instances):
                reasons.append('scene_blocked: %s' % self.latest_obstacles.blocking_reason)
            invalid = [item for item in self.latest_obstacles.instances if not item.valid]
            if invalid:
                reasons.append('invalid obstacle geometry is present')
        if allow_untracked:
            if require_settled and not self.joints_settled():
                reasons.append('joint feedback is not settled for acquisition')
        else:
            health = self.latest_tracking_health
            if enforce_tracking_motion_state:
                if require_settled:
                    if health.lifecycle_state != 'TRACKING' or not health.camera_settled:
                        reasons.append('tracking is not settled TRACKING')
                    if health.prediction_only:
                        reasons.append('tracking is prediction-only')
                elif health.lifecycle_state not in ('TRACKING', 'DEGRADED'):
                    reasons.append('tracking lifecycle=%s' % health.lifecycle_state)
                if float(health.measurement_age_sec) > float(
                        self.get_parameter('max_tracking_measurement_age_sec').value):
                    reasons.append('tracking measurement is stale')
                if float(health.recommended_speed_scale) < float(
                        self.get_parameter('min_tracking_speed_scale').value):
                    reasons.append(
                        'tracking speed scale is below the motion threshold')
            current_allowed_speed = commanded_speed_percent(
                float(self.get_parameter('speed_percent').value),
                self.plan_kind,
                float(health.recommended_speed_scale),
            )
            if (
                    enforce_tracking_motion_state
                    and enforce_tracking_speed_allowance
                    and
                    self.plan_execution_speed_percent > 0.0
                    and current_allowed_speed + 1e-6
                    < self.plan_execution_speed_percent):
                reasons.append(
                    'tracking speed allowance fell below the approved MoveJ speed; '
                    'replan at the lower speed')
            if (
                    enforce_tracking_motion_state
                    and enforce_target_status
                    and self.latest_target_status not in ('TRACKING', 'LOCKED')):
                reasons.append('target_status=%s' % self.latest_target_status)
        if require_workflow and self.param_bool('auto_capture') and not self.workflow_ready():
            reasons.append('supervised workflow is not SCAN_READY')
        return reasons

    def runtime_motion_limit_rejection(self, limits):
        """Accept a fresh valid limit generation for position-only SDK MoveJ.

        The executor sends one joint-position target plus an aggregate speed
        percentage. It never sends Tesseract velocities, accelerations, or
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

    def arm_status_reasons(self):
        status = self.latest_arm_status
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
        return reasons

    def settled_and_tracking(self):
        if (
                self.latest_camera_timestamp_health is None
                or not self.latest_camera_timestamp_health.healthy):
            return False
        health = self.latest_tracking_health
        if health is None or health.lifecycle_state != 'TRACKING' or not health.camera_settled:
            return False
        if health.prediction_only or self.latest_target_status not in ('TRACKING', 'LOCKED'):
            return False
        return self.joints_settled()

    def capture_pose_settled(self):
        """Gate RGB-D on a stationary arm and healthy camera clock.

        The scan target may move and SAM tracking may temporarily reacquire at
        a viewpoint.  Neither changes the approved robot pose or whether a
        synchronized RGB-D record can be saved, so they must not discard the
        remaining 13-view plan.
        """
        return bool(
            self.latest_camera_timestamp_health is not None
            and self.latest_camera_timestamp_health.healthy
            and self.joints_settled()
        )

    def joints_settled(self):
        if self.latest_joint_state is None:
            return False
        if (
                self.command_target is not None
                and self.max_joint_error(self.command_target) > float(
                    self.get_parameter('joint_goal_tolerance_rad').value)):
            return False
        velocities = list(self.latest_joint_state.velocity[:6])
        if len(velocities) < 6:
            return False
        try:
            values = [float(value) for value in velocities]
        except (TypeError, ValueError):
            return False
        return all(math.isfinite(value) for value in values) and \
            max(abs(value) for value in values) <= float(
                self.get_parameter('joint_velocity_settled').value)

    def home_joints_settled(self):
        """
        Use successive positions for final home proof.

        PiPER's SDK speed feedback can briefly spike while the measured joint
        positions remain stationary. The final disable gate therefore matches
        the GUI's independent safe-disable proof: fresh feedback must stay
        within the approved home tolerance and successive samples must move by
        no more than a small bounded delta.
        """
        feedback_timeout = float(self.get_parameter(
            'home_joint_feedback_timeout_sec').value)
        if (
                self.latest_joint_state is None
                or self.command_target is None
                or not self.fresh('joints', feedback_timeout)):
            self.home_settle_previous_joints = None
            self.home_settle_last_sample_ok = False
            return False
        updated_at = float(self.updated.get('joints', -1e9))
        if updated_at <= self.home_settle_last_joint_update:
            return self.home_settle_last_sample_ok
        current = np.asarray(self.current_joints(), dtype=float)
        settled = home_position_sample_settled(
            current,
            self.command_target,
            self.home_settle_previous_joints,
            float(self.get_parameter('joint_goal_tolerance_rad').value),
            float(self.get_parameter('home_motion_tolerance_rad').value),
        )
        self.home_settle_previous_joints = current
        self.home_settle_last_joint_update = updated_at
        self.home_settle_last_sample_ok = bool(settled)
        return bool(settled)

    def workflow_ready(self):
        return isinstance(self.latest_workflow, dict) and \
            str(self.latest_workflow.get('state', '')) == 'SCAN_READY'

    def obstacle_boxes(self):
        if self.latest_obstacles is None:
            return []
        boxes = []
        for item in self.latest_obstacles.instances:
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

    def publish_joint_command(self, target):
        if self.command_pub is None:
            raise RuntimeError('real-motion publisher does not exist in proposal-only mode')
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
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
        if self.latest_joint_state is None or not self.real_motion_enabled():
            return False
        try:
            self.publish_joint_command(
                energized_hold_target(self.current_joints()))
        except ValueError:
            return False
        return True

    def abort_motion(self, reason):
        started, blocker = self.try_start_abort_return(reason)
        if started:
            return
        suffix = (
            '; arm held at the current pose; automatic return home was not '
            'started: ' + blocker
            if blocker else '')
        self._terminal_abort(str(reason) + suffix)

    def try_start_abort_return(self, reason):
        """Retrace only already executed, approved targets after benign aborts."""
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
        runtime = self.runtime_reasons(
            require_settled=True,
            require_workflow=False,
            allow_untracked=True,
            allow_missing_obstacles=use_bootstrap_scene,
            allow_stale_obstacles=use_bootstrap_scene,
        )
        if runtime:
            return False, 'fresh return-home safety gate failed: ' + '; '.join(runtime)
        tolerance = float(
            self.get_parameter('plan_start_tolerance_rad').value)
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
        self.plan_kind = (
            plan_kind
            if plan_kind in (MULTIVIEW_SCAN, ROUGH_ACQUISITION)
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
        self.current_view = 0
        self.current_path = []
        self.current_path_velocities = []
        self.current_path_accelerations = []
        self.current_path_times = []
        self.path_index = 0
        self.command_target = None
        self.command_sent_at = 0.0
        self.command_samples_sent = 0
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
        msg.planner_backend = 'tesseract'
        msg.trajectory_sha256 = self.plan_trajectory_sha256
        msg.motion_limits_sha256 = self.plan_motion_limits_sha256
        msg.execution_speed_percent = float(
            self.plan_execution_speed_percent)
        msg.command_rate_hz = float(
            self.get_parameter('trajectory_command_rate_hz').value)
        msg.timing_policy = TIMING_POLICY_VERSION
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
        msg.tracking_speed_scale = float(
            self.latest_tracking_health.recommended_speed_scale
            if self.latest_tracking_health is not None else 0.0)
        msg.max_joint_error_rad = float(
            self.current_waypoint_error
            if self.command_target is not None else 0.0)
        self.status_pub.publish(msg)

    def publish_scan_history(self):
        msg = String()
        msg.data = json.dumps({
            'session_id': self.scan_session_id,
            'accepted_views': len(self.scan_history),
            'max_views': int(
                self.get_parameter('max_execution_viewpoints').value),
            'entries': list(self.scan_history),
        }, sort_keys=True)
        self.scan_history_pub.publish(msg)

    def current_joints(self):
        if self.latest_joint_state is None or len(self.latest_joint_state.position) < 6:
            raise ValueError('joint feedback has fewer than six arm joints')
        values = np.asarray(self.latest_joint_state.position[:6], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError('joint feedback contains non-finite values')
        return values

    def max_joint_error(self, target):
        try:
            return float(np.max(np.abs(self.current_joints() - np.asarray(target, dtype=float))))
        except ValueError:
            return math.inf

    def total_joint_error(self, target):
        try:
            return joint_progress_error(self.current_joints(), target)
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
        return max(1.0, min(100.0, float(self.get_parameter('speed_percent').value)))

    def execution_speed_percent(self):
        if self.plan_execution_speed_percent > 0.0:
            return float(self.plan_execution_speed_percent)
        scale = (
            float(self.latest_tracking_health.recommended_speed_scale)
            if self.latest_tracking_health is not None else 0.0)
        return commanded_speed_percent(
            float(self.get_parameter('speed_percent').value),
            self.plan_kind,
            scale,
        )

    def is_acquisition(self):
        return self.plan_kind == ROUGH_ACQUISITION

    def real_motion_enabled(self):
        return self.param_bool('enable_real_arm_motion')

    def param_bool(self, name):
        value = self.get_parameter(name).value
        if isinstance(value, str):
            return value.lower() in ('1', 'true', 'yes', 'on')
        return bool(value)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = ScanViewpointExecutorNode()
    except (KeyError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print('scan viewpoint executor startup error: %s' % error)
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
