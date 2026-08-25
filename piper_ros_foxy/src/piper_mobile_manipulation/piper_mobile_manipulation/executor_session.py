"""Authoritative mutable application state for one executor process."""

import math

from piper_mobile_manipulation.capture_coordinator import CaptureCoordinator
from piper_mobile_manipulation.executor_recovery import RecoveryPolicy
from piper_mobile_manipulation.plan_authorizer import PlanAuthorizer
from piper_mobile_manipulation.scan_execution_modes import MULTIVIEW_SCAN
from piper_mobile_manipulation.trajectory_runner import TrajectoryRunner


EXECUTOR_SESSION_FIELDS = (
    'plan_authorizer', 'trajectory_runner', 'capture_coordinator',
    'recovery_policy', 'scan_session_id', 'scan_history', 'scan_rejections',
    'latest_achieved_scan_view', 'scan_coverage_target_center', 'state',
    'reason', 'plan_id', 'plan_kind', 'plan_source_request_id',
    'plan_created', 'plan_targets', 'plan_paths', 'plan_path_velocities',
    'plan_path_accelerations', 'plan_path_times',
    'plan_bootstrap_recovery_end_points', 'plan_bootstrap_recovery_joints',
    'plan_bootstrap_recovery_joint_sets',
    'plan_powered_start_recovery_end_points',
    'plan_powered_start_recovery_joint_sets', 'plan_startup_home_static',
    'plan_configured_home_direct', 'plan_configured_home_stages',
    'plan_segment_execution_modes', 'plan_viewpoints',
    'plan_candidate_count', 'plan_capture_count', 'plan_returns_home',
    'plan_target_center', 'plan_source_trajectory_sha256',
    'plan_trajectory_sha256', 'plan_execution_speed_percent',
    'plan_motion_limits_sha256', 'runtime_motion_limits_sha256',
    'plan_collision_model_qualified', 'pending_limit_refresh_plan',
    'pending_limit_refresh_deadline', 'current_view', 'current_path',
    'current_path_velocities', 'current_path_accelerations',
    'current_path_times', 'current_path_streaming', 'current_trajectory',
    'path_index', 'command_target', 'command_sent_at',
    'command_samples_sent', 'max_command_interval_sec',
    'dropped_command_samples', 'motion_started_at', 'stream_last_tick_at',
    'stream_schedule_paused_sec', 'stream_following_hold_started_at',
    'stream_schedule_completion_logged', 'last_stream_planned_duration_sec',
    'last_stream_actual_duration_sec', 'last_stream_achieved_rate_hz',
    'last_motion_status_at', 'waypoint_started_at',
    'waypoint_last_progress_at', 'waypoint_best_error',
    'current_waypoint_error', 'max_waypoint_error', 'pending_motion_reason',
    'runtime_refresh_require_workflow',
    'runtime_refresh_allow_missing_obstacles',
    'runtime_refresh_resume_state', 'runtime_recovery_started_at',
    'settle_started', 'settle_position_anchor', 'settle_last_joint_update',
    'settle_last_sample_ok', 'settle_diagnostic', 'settle_reset_count',
    'settle_longest_window_sec', 'settle_last_reset_reason',
    'home_settle_previous_joints', 'home_settle_last_joint_update',
    'home_settle_last_sample_ok', 'state_started', 'capture_future',
    'rgbd_capture_future', 'rgbd_capture_attempts',
    'capture_heavy_refresh_started', 'capture_heavy_refresh_request_id',
    'capture_heavy_refresh_min_image_stamp_ns',
    'capture_heavy_refresh_publish_attempts',
    'capture_heavy_refresh_waiting_for_worker', 'capture_rejection_reason',
    'finish_scan_future', 'return_home_warning', 'abort_return_in_progress',
    'abort_return_reason', 'abort_return_bootstrap_static_scene',
    'retrace_joint_targets', 'capture_accepted_before',
    'acquisition_refresh_started', 'acquisition_request_id',
    'acquisition_request_attempt', 'acquisition_min_image_stamp_ns',
    'acquisition_job_image_stamp_ns', 'acquisition_job_started',
    'acquisition_detection_completed', 'acquisition_waiting_for_worker',
    'pending_acquisition_heavy_status',
    'acquisition_scene_snapshot_validated', 'mission_task_id',
    'mission_sha256', 'mission_expires_at_sec',
)


class SessionField:
    """Compatibility descriptor backed only by ``ExecutorSession``."""

    def __init__(self, name):
        self.name = str(name)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance.executor_session, self.name)

    def __set__(self, instance, value):
        setattr(instance.executor_session, self.name, value)


class ExecutorSession:
    """Own plan, motion, acquisition, capture, recovery, and home state."""

    def __init__(self, now, maximum_capture_retries):
        self.plan_authorizer = PlanAuthorizer()
        self.trajectory_runner = TrajectoryRunner()
        self.capture_coordinator = CaptureCoordinator(
            maximum_capture_retries)
        self.recovery_policy = RecoveryPolicy()
        self.scan_session_id = ''
        self.scan_history = []
        self.scan_rejections = []
        self.latest_achieved_scan_view = None
        self.scan_coverage_target_center = None
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
        self.stream_last_tick_at = None
        self.stream_schedule_paused_sec = 0.0
        self.stream_following_hold_started_at = None
        self.stream_schedule_completion_logged = False
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
        self.settle_position_anchor = None
        self.settle_last_joint_update = -1e9
        self.settle_last_sample_ok = False
        self.settle_diagnostic = (
            'settle proof has not sampled joint feedback')
        self.settle_reset_count = 0
        self.settle_longest_window_sec = 0.0
        self.settle_last_reset_reason = ''
        self.home_settle_previous_joints = None
        self.home_settle_last_joint_update = -1e9
        self.home_settle_last_sample_ok = False
        self.state_started = float(now)
        self.capture_future = None
        self.rgbd_capture_future = None
        self.rgbd_capture_attempts = 0
        self.capture_heavy_refresh_started = None
        self.capture_heavy_refresh_request_id = ''
        self.capture_heavy_refresh_min_image_stamp_ns = 0
        self.capture_heavy_refresh_publish_attempts = 0
        self.capture_heavy_refresh_waiting_for_worker = False
        self.capture_rejection_reason = ''
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
        self.mission_expires_at_sec = 0.0
