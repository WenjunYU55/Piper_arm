"""Typed static configuration for the target-scan application boundaries."""

from dataclasses import dataclass
import math
import os
from types import MappingProxyType
from typing import Mapping, Tuple

from piper_mobile_manipulation.mission.core import (
    MAX_FEATURE_CAPTURES,
    MAX_PENDING_MISSIONS,
    MISSION_QUEUE_COALESCE_SEC,
    REQUIRED_CAPTURES,
)


class ConfigurationError(ValueError):
    """Report an invalid static configuration before runtime decisions."""


def _as_bool(value):
    if isinstance(value, str):
        return value.lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def _finite(name, value):
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError('%s must be finite' % name)
    return result


def _positive(name, value):
    result = _finite(name, value)
    if result <= 0.0:
        raise ConfigurationError('%s must be positive' % name)
    return result


def _non_negative(name, value):
    result = _finite(name, value)
    if result < 0.0:
        raise ConfigurationError('%s must be non-negative' % name)
    return result


def _positive_int(name, value):
    result = int(value)
    if result < 1:
        raise ConfigurationError('%s must be at least one' % name)
    return result


def _nonempty(name, value):
    result = str(value)
    if not result.strip():
        raise ConfigurationError('%s must not be empty' % name)
    return result


def _vector(name, value, length):
    result = tuple(_finite(name, item) for item in value)
    if len(result) != int(length):
        raise ConfigurationError(
            '%s must contain exactly %d values' % (name, length))
    return result


@dataclass(frozen=True)
class MissionConfig:
    """Coordinator admission, queue, and diagnostic configuration."""

    project_root: str
    require_gateway_heartbeat: bool
    max_pending_missions: int
    mission_queue_coalesce_sec: float
    debug: bool


@dataclass(frozen=True)
class ProcessConfig:
    """Coordinator-owned process and durable-state locations."""

    manage_processes: bool
    mission_spool_root: str
    process_log_root: str
    floor_profile: str
    floor_profile_path: str


@dataclass(frozen=True)
class MissionMotionConfig:
    """Existing mission-level motion authorization and speed policy."""

    enable_real_arm_motion: bool
    motion_speed_profile_qualified: bool
    free_motion_speed_percent: float
    contact_speed_percent: float
    home_pose_path: str
    require_staged_home_profile: bool


@dataclass(frozen=True)
class MissionCaptureConfig:
    """Adaptive mission capture bounds."""

    required_captures: int
    maximum_captures: int


@dataclass(frozen=True)
class MissionWorkflowConfig:
    """Static application timing and retry policy formerly in engine code."""

    enable_service_timeout_sec: float = 30.0
    acquisition_service_timeout_sec: float = 8.0
    startup_readiness_stable_sec: float = 2.0
    startup_readiness_timeout_sec: float = 90.0
    startup_joint_stable_sec: float = 2.0
    startup_joint_timeout_sec: float = 15.0
    acquisition_max_looks: int = 5
    plan_request_queue_timeout_sec: float = 12.0
    plan_result_timeout_sec: float = 185.0
    acquisition_execution_timeout_sec: float = 180.0
    workflow_assessment_timeout_sec: float = 75.0
    multiview_readiness_stable_sec: float = 1.0
    multiview_readiness_timeout_sec: float = 30.0
    replan_readiness_stable_sec: float = 0.5
    plan_approval_transient_timeout_sec: float = 5.0
    scan_visual_reacquisition_timeout_sec: float = 30.0
    max_scan_quality_replans: int = 8
    max_scan_target_drift_replans: int = 8
    final_history_timeout_sec: float = 5.0
    motor_disabled_proof_timeout_sec: float = 2.0


@dataclass(frozen=True)
class MissionConfigurationGroups:
    """Named coordinator groups loaded once at the ROS boundary."""

    mission: MissionConfig
    process: ProcessConfig
    motion: MissionMotionConfig
    capture: MissionCaptureConfig
    workflow: MissionWorkflowConfig
    parameter_values: Mapping

    def value(self, name):
        """Return one frozen legacy parameter value at a ROS adapter seam."""
        return self.parameter_values[name]


@dataclass(frozen=True)
class ExecutorInterfaceConfig:
    """ROS names and static file inputs for the viewpoint executor."""

    reachable_viewpoints_topic: str
    joint_states_topic: str
    arm_status_topic: str
    motion_limits_topic: str
    tracking_health_topic: str
    tracked_target_topic: str
    camera_timestamp_health_topic: str
    target_status_topic: str
    target_shape_topic: str
    obstacle_topic: str
    workflow_status_topic: str
    scan_session_history_topic: str
    joint_command_topic: str
    plan_topic: str
    status_topic: str
    capture_service: str
    finish_scan_service: str
    rgbd_capture_service: str
    heavy_refresh_request_topic: str
    heavy_refresh_status_topic: str
    tesseract_plan_topic: str
    hand_eye_calibration_path: str
    joint_bounds_path: str


@dataclass(frozen=True)
class MotionConfig:
    """Executor motion timing, tolerances, speed, and home target."""

    enable_real_arm_motion: bool
    speed_percent: float
    trajectory_joint_step_rad: float
    trajectory_command_rate_hz: float
    executor_tick_rate_hz: float
    trajectory_following_error_rad: float
    trajectory_following_error_grace_sec: float
    plan_start_tolerance_rad: float
    joint_goal_tolerance_rad: float
    waypoint_reached_tolerance_rad: float
    waypoint_progress_epsilon_rad: float
    waypoint_timeout_sec: float
    waypoint_progress_timeout_sec: float
    joint_velocity_settled: float
    endpoint_position_settled_rad: float
    home_goal_tolerance_rad: float
    home_motion_tolerance_rad: float
    home_joint_feedback_timeout_sec: float
    home_settle_duration_sec: float
    home_settle_timeout_sec: float
    pre_home_positions_rad: Tuple[float, ...]
    return_home_positions_rad: Tuple[float, ...]


@dataclass(frozen=True)
class TrackingConfig:
    """Executor target/tracking freshness and acquisition configuration."""

    data_timeout_sec: float
    max_tracking_measurement_age_sec: float
    min_tracking_speed_scale: float
    acquisition_fresh_frame_timeout_sec: float
    acquisition_grounding_timeout_sec: float
    acquisition_tracking_lock_timeout_sec: float
    acquisition_scene_timeout_sec: float
    acquisition_target_tolerance_m: float
    acquisition_max_viewpoints: int
    scan_target_max_boresight_deg: float
    scan_target_min_distance_m: float
    allow_target_motion_during_scan: bool
    max_target_drift_before_approval_m: float


@dataclass(frozen=True)
class CaptureConfig:
    """Executor capture/settling policy."""

    auto_capture: bool
    max_execution_viewpoints: int
    min_execution_viewpoints: int
    settle_duration_sec: float
    settle_timeout_sec: float
    capture_timeout_sec: float
    first_capture_acceptance_timeout_sec: float
    finish_scan_timeout_sec: float
    capture_status_propagation_sec: float


@dataclass(frozen=True)
class SafetyConfig:
    """Executor safety freshness, geometry, and feedback limits."""

    joint_feedback_limit_tolerance_rad: float
    configured_home_feedback_limit_tolerance_rad: float
    motion_limits_timeout_sec: float
    motion_limits_change_confirmation_sec: float
    motion_limits_change_minimum_samples: int
    runtime_refresh_timeout_sec: float
    runtime_recovery_timeout_sec: float
    floor_z_m: float
    link_radius_m: float
    self_clearance_m: float
    camera_holder_envelope_center_link6_m: Tuple[float, ...]
    camera_holder_envelope_size_m: Tuple[float, ...]
    camera_holder_external_clearance_m: float


@dataclass(frozen=True)
class PlanningConfig:
    """Executor plan lifetime, authorization, and execution-mode policy."""

    plan_max_age_sec: float
    approval_confirmation: str
    allow_mission_policy: bool
    closed_loop_one_view: bool


@dataclass(frozen=True)
class ExecutorConfig:
    """Executor diagnostics and mode flags not owned by another group."""

    debug: bool


@dataclass(frozen=True)
class ExecutorConfigurationGroups:
    """Named executor groups loaded once at the ROS boundary."""

    interfaces: ExecutorInterfaceConfig
    motion: MotionConfig
    tracking: TrackingConfig
    capture: CaptureConfig
    safety: SafetyConfig
    planning: PlanningConfig
    executor: ExecutorConfig
    parameter_values: Mapping

    def value(self, name):
        """Return one frozen legacy parameter value at a ROS adapter seam."""
        return self.parameter_values[name]


def mission_parameter_defaults(environ=None):
    """Return the exact pre-Phase-9 coordinator ROS parameter defaults."""
    environment = os.environ if environ is None else environ
    runtime_root = environment.get('XDG_RUNTIME_DIR', '/tmp')
    return {
        'project_root': '/home/prl/Piper_arm',
        'manage_processes': True,
        'floor_profile': 'saved',
        'floor_profile_path': '',
        'enable_real_arm_motion': False,
        'motion_speed_profile_qualified': False,
        'free_motion_speed_percent': 30.0,
        'contact_speed_percent': 10.0,
        'required_captures': REQUIRED_CAPTURES,
        'maximum_captures': MAX_FEATURE_CAPTURES,
        'home_pose_path': '',
        'require_staged_home_profile': True,
        'mission_spool_root': os.path.join(
            runtime_root, 'piper_target_scan_missions'),
        'process_log_root': os.path.join(
            runtime_root, 'piper_target_scan_logs'),
        'require_gateway_heartbeat': False,
        'max_pending_missions': MAX_PENDING_MISSIONS,
        'mission_queue_coalesce_sec': MISSION_QUEUE_COALESCE_SEC,
        'debug': True,
    }


def executor_parameter_defaults():
    """Return the exact pre-Phase-9 viewpoint-executor parameter defaults."""
    return {
        'reachable_viewpoints_topic': '/piper/reachable_scan_viewpoints',
        'joint_states_topic': '/joint_states_single',
        'arm_status_topic': '/arm_status',
        'motion_limits_topic': '/piper/motion_limits',
        'tracking_health_topic': '/piper/tracking_health',
        'tracked_target_topic': '/piper/tracked_target',
        'camera_timestamp_health_topic': '/piper/camera_timestamp_health',
        'target_status_topic': '/piper/target_status',
        'target_shape_topic': '/piper/target_shape_measurement',
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
        'trajectory_joint_step_rad': 0.05,
        'trajectory_command_rate_hz': 20.0,
        'executor_tick_rate_hz': 200.0,
        'trajectory_following_error_rad': 0.30,
        'trajectory_following_error_grace_sec': 1.0,
        'plan_start_tolerance_rad': 0.025,
        'joint_goal_tolerance_rad': 0.025,
        'waypoint_reached_tolerance_rad': 0.025,
        'waypoint_progress_epsilon_rad': 0.001,
        'waypoint_timeout_sec': 90.0,
        'waypoint_progress_timeout_sec': 20.0,
        'joint_velocity_settled': 0.20,
        'endpoint_position_settled_rad': 0.007,
        # Operator-authorized configured-home acceptance. This applies
        # independently to every arm joint and is not a Tesseract planning
        # tolerance.
        'home_goal_tolerance_rad': 0.20,
        'home_motion_tolerance_rad': 0.20,
        'home_joint_feedback_timeout_sec': 1.0,
        'home_settle_duration_sec': 1.0,
        'home_settle_timeout_sec': 30.0,
        'joint_feedback_limit_tolerance_rad': 0.005,
        'configured_home_feedback_limit_tolerance_rad': 0.3,
        'motion_limits_timeout_sec': 3.0,
        'motion_limits_change_confirmation_sec': 7.0,
        'motion_limits_change_minimum_samples': 3,
        'runtime_refresh_timeout_sec': 3.0,
        'runtime_recovery_timeout_sec': 30.0,
        'settle_duration_sec': 1.5,
        'settle_timeout_sec': 15.0,
        'capture_timeout_sec': 20.0,
        'first_capture_acceptance_timeout_sec': 75.0,
        'finish_scan_timeout_sec': 10.0,
        'capture_status_propagation_sec': 0.25,
        'acquisition_fresh_frame_timeout_sec': 10.0,
        'acquisition_grounding_timeout_sec': 60.0,
        'acquisition_tracking_lock_timeout_sec': 10.0,
        'acquisition_scene_timeout_sec': 15.0,
        'acquisition_target_tolerance_m': 0.30,
        'acquisition_max_viewpoints': 5,
        'closed_loop_one_view': False,
        'scan_target_max_boresight_deg': 20.0,
        'scan_target_min_distance_m': 0.22,
        'data_timeout_sec': 2.0,
        'max_tracking_measurement_age_sec': 0.75,
        'min_tracking_speed_scale': 0.10,
        'plan_max_age_sec': 300.0,
        'max_target_drift_before_approval_m': 0.015,
        'allow_target_motion_during_scan': False,
        'return_home_positions_rad': [
            0.000366362, 0.0, 0.0, 0.0, 0.43869236, 0.0],
        'pre_home_positions_rad': [
            0.0, 0.400357244, -0.498793736, 0.0, 0.600614364, 0.0],
        'floor_z_m': 0.005,
        'link_radius_m': 0.025,
        'self_clearance_m': 0.060,
        'camera_holder_envelope_center_link6_m': [
            -0.029750002, 0.0, 0.0375],
        'camera_holder_envelope_size_m': [0.1395, 0.10572671, 0.053],
        'camera_holder_external_clearance_m': 0.005,
        'approval_confirmation': 'EXECUTE APPROVED SCAN',
        'allow_mission_policy': False,
        'debug': True,
    }


def _declare_and_read(node, defaults):
    for name, default in defaults.items():
        node.declare_parameter(name, default)
    return {name: node.get_parameter(name).value for name in defaults}


def _freeze_parameter_values(values):
    frozen = {
        name: tuple(value) if isinstance(value, list) else value
        for name, value in values.items()
    }
    return MappingProxyType(frozen)


def load_mission_configuration(node, environ=None):
    """Declare, read once, validate, and freeze coordinator parameters."""
    values = _declare_and_read(node, mission_parameter_defaults(environ))
    mission = MissionConfig(
        project_root=_nonempty('project_root', values['project_root']),
        require_gateway_heartbeat=_as_bool(
            values['require_gateway_heartbeat']),
        max_pending_missions=_positive_int(
            'max_pending_missions', values['max_pending_missions']),
        mission_queue_coalesce_sec=_non_negative(
            'mission_queue_coalesce_sec',
            values['mission_queue_coalesce_sec']),
        debug=_as_bool(values['debug']),
    )
    process = ProcessConfig(
        manage_processes=_as_bool(values['manage_processes']),
        mission_spool_root=_nonempty(
            'mission_spool_root', values['mission_spool_root']),
        process_log_root=_nonempty(
            'process_log_root', values['process_log_root']),
        floor_profile=_nonempty(
            'floor_profile', values['floor_profile']).lower(),
        floor_profile_path=str(values['floor_profile_path']).strip(),
    )
    if process.floor_profile not in ('saved', 'tabletop', 'ground'):
        raise ConfigurationError(
            'floor_profile must be exactly saved, tabletop or ground')
    motion = MissionMotionConfig(
        enable_real_arm_motion=_as_bool(values['enable_real_arm_motion']),
        motion_speed_profile_qualified=_as_bool(
            values['motion_speed_profile_qualified']),
        free_motion_speed_percent=_positive(
            'free_motion_speed_percent', values['free_motion_speed_percent']),
        contact_speed_percent=_positive(
            'contact_speed_percent', values['contact_speed_percent']),
        home_pose_path=str(values['home_pose_path']),
        require_staged_home_profile=_as_bool(
            values['require_staged_home_profile']),
    )
    capture = MissionCaptureConfig(
        required_captures=_positive_int(
            'required_captures', values['required_captures']),
        maximum_captures=_positive_int(
            'maximum_captures', values['maximum_captures']),
    )
    if capture.maximum_captures < capture.required_captures:
        raise ConfigurationError(
            'maximum_captures must be at least required_captures')
    return MissionConfigurationGroups(
        mission=mission,
        process=process,
        motion=motion,
        capture=capture,
        workflow=MissionWorkflowConfig(),
        parameter_values=_freeze_parameter_values(values),
    )


def _interface_config(values):
    names = (
        'reachable_viewpoints_topic', 'joint_states_topic', 'arm_status_topic',
        'motion_limits_topic', 'tracking_health_topic',
        'tracked_target_topic', 'camera_timestamp_health_topic',
        'target_status_topic', 'target_shape_topic', 'obstacle_topic',
        'workflow_status_topic',
        'scan_session_history_topic', 'joint_command_topic', 'plan_topic',
        'status_topic', 'capture_service', 'finish_scan_service',
        'rgbd_capture_service', 'heavy_refresh_request_topic',
        'heavy_refresh_status_topic', 'tesseract_plan_topic',
    )
    checked = {name: _nonempty(name, values[name]) for name in names}
    checked['hand_eye_calibration_path'] = str(
        values['hand_eye_calibration_path'])
    checked['joint_bounds_path'] = str(values['joint_bounds_path'])
    return ExecutorInterfaceConfig(**checked)


def load_executor_configuration(node):
    """Declare, read once, validate, and freeze executor parameters."""
    values = _declare_and_read(node, executor_parameter_defaults())
    interfaces = _interface_config(values)
    motion = MotionConfig(
        enable_real_arm_motion=_as_bool(values['enable_real_arm_motion']),
        speed_percent=_positive('speed_percent', values['speed_percent']),
        trajectory_joint_step_rad=_positive(
            'trajectory_joint_step_rad', values['trajectory_joint_step_rad']),
        trajectory_command_rate_hz=_positive(
            'trajectory_command_rate_hz',
            values['trajectory_command_rate_hz']),
        executor_tick_rate_hz=_positive(
            'executor_tick_rate_hz', values['executor_tick_rate_hz']),
        trajectory_following_error_rad=_positive(
            'trajectory_following_error_rad',
            values['trajectory_following_error_rad']),
        trajectory_following_error_grace_sec=_non_negative(
            'trajectory_following_error_grace_sec',
            values['trajectory_following_error_grace_sec']),
        plan_start_tolerance_rad=_positive(
            'plan_start_tolerance_rad', values['plan_start_tolerance_rad']),
        joint_goal_tolerance_rad=_positive(
            'joint_goal_tolerance_rad', values['joint_goal_tolerance_rad']),
        waypoint_reached_tolerance_rad=_positive(
            'waypoint_reached_tolerance_rad',
            values['waypoint_reached_tolerance_rad']),
        waypoint_progress_epsilon_rad=_positive(
            'waypoint_progress_epsilon_rad',
            values['waypoint_progress_epsilon_rad']),
        waypoint_timeout_sec=_positive(
            'waypoint_timeout_sec', values['waypoint_timeout_sec']),
        waypoint_progress_timeout_sec=_positive(
            'waypoint_progress_timeout_sec',
            values['waypoint_progress_timeout_sec']),
        joint_velocity_settled=_non_negative(
            'joint_velocity_settled', values['joint_velocity_settled']),
        endpoint_position_settled_rad=_positive(
            'endpoint_position_settled_rad',
            values['endpoint_position_settled_rad']),
        home_goal_tolerance_rad=_positive(
            'home_goal_tolerance_rad', values['home_goal_tolerance_rad']),
        home_motion_tolerance_rad=_positive(
            'home_motion_tolerance_rad', values['home_motion_tolerance_rad']),
        home_joint_feedback_timeout_sec=_positive(
            'home_joint_feedback_timeout_sec',
            values['home_joint_feedback_timeout_sec']),
        home_settle_duration_sec=_positive(
            'home_settle_duration_sec', values['home_settle_duration_sec']),
        home_settle_timeout_sec=_positive(
            'home_settle_timeout_sec', values['home_settle_timeout_sec']),
        pre_home_positions_rad=_vector(
            'pre_home_positions_rad', values['pre_home_positions_rad'], 6),
        return_home_positions_rad=_vector(
            'return_home_positions_rad', values['return_home_positions_rad'], 6),
    )
    if motion.executor_tick_rate_hz < 2.0 * motion.trajectory_command_rate_hz:
        raise ConfigurationError(
            'executor_tick_rate_hz must be at least twice '
            'trajectory_command_rate_hz')
    tracking = TrackingConfig(
        data_timeout_sec=_positive(
            'data_timeout_sec', values['data_timeout_sec']),
        max_tracking_measurement_age_sec=_positive(
            'max_tracking_measurement_age_sec',
            values['max_tracking_measurement_age_sec']),
        min_tracking_speed_scale=_non_negative(
            'min_tracking_speed_scale', values['min_tracking_speed_scale']),
        acquisition_fresh_frame_timeout_sec=_positive(
            'acquisition_fresh_frame_timeout_sec',
            values['acquisition_fresh_frame_timeout_sec']),
        acquisition_grounding_timeout_sec=_positive(
            'acquisition_grounding_timeout_sec',
            values['acquisition_grounding_timeout_sec']),
        acquisition_tracking_lock_timeout_sec=_positive(
            'acquisition_tracking_lock_timeout_sec',
            values['acquisition_tracking_lock_timeout_sec']),
        acquisition_scene_timeout_sec=_positive(
            'acquisition_scene_timeout_sec',
            values['acquisition_scene_timeout_sec']),
        acquisition_target_tolerance_m=_positive(
            'acquisition_target_tolerance_m',
            values['acquisition_target_tolerance_m']),
        acquisition_max_viewpoints=_positive_int(
            'acquisition_max_viewpoints', values['acquisition_max_viewpoints']),
        scan_target_max_boresight_deg=_positive(
            'scan_target_max_boresight_deg',
            values['scan_target_max_boresight_deg']),
        scan_target_min_distance_m=_positive(
            'scan_target_min_distance_m',
            values['scan_target_min_distance_m']),
        allow_target_motion_during_scan=_as_bool(
            values['allow_target_motion_during_scan']),
        max_target_drift_before_approval_m=_non_negative(
            'max_target_drift_before_approval_m',
            values['max_target_drift_before_approval_m']),
    )
    capture = CaptureConfig(
        auto_capture=_as_bool(values['auto_capture']),
        max_execution_viewpoints=_positive_int(
            'max_execution_viewpoints', values['max_execution_viewpoints']),
        min_execution_viewpoints=_positive_int(
            'min_execution_viewpoints', values['min_execution_viewpoints']),
        settle_duration_sec=_positive(
            'settle_duration_sec', values['settle_duration_sec']),
        settle_timeout_sec=_positive(
            'settle_timeout_sec', values['settle_timeout_sec']),
        capture_timeout_sec=_positive(
            'capture_timeout_sec', values['capture_timeout_sec']),
        first_capture_acceptance_timeout_sec=_positive(
            'first_capture_acceptance_timeout_sec',
            values['first_capture_acceptance_timeout_sec']),
        finish_scan_timeout_sec=_positive(
            'finish_scan_timeout_sec', values['finish_scan_timeout_sec']),
        capture_status_propagation_sec=_non_negative(
            'capture_status_propagation_sec',
            values['capture_status_propagation_sec']),
    )
    if capture.max_execution_viewpoints < capture.min_execution_viewpoints:
        raise ConfigurationError(
            'max_execution_viewpoints must be at least '
            'min_execution_viewpoints')
    if capture.first_capture_acceptance_timeout_sec < capture.capture_timeout_sec:
        raise ConfigurationError(
            'first_capture_acceptance_timeout_sec must be at least '
            'capture_timeout_sec')
    safety = SafetyConfig(
        joint_feedback_limit_tolerance_rad=_non_negative(
            'joint_feedback_limit_tolerance_rad',
            values['joint_feedback_limit_tolerance_rad']),
        configured_home_feedback_limit_tolerance_rad=_non_negative(
            'configured_home_feedback_limit_tolerance_rad',
            values['configured_home_feedback_limit_tolerance_rad']),
        motion_limits_timeout_sec=_positive(
            'motion_limits_timeout_sec', values['motion_limits_timeout_sec']),
        motion_limits_change_confirmation_sec=_non_negative(
            'motion_limits_change_confirmation_sec',
            values['motion_limits_change_confirmation_sec']),
        motion_limits_change_minimum_samples=_positive_int(
            'motion_limits_change_minimum_samples',
            values['motion_limits_change_minimum_samples']),
        runtime_refresh_timeout_sec=_positive(
            'runtime_refresh_timeout_sec', values['runtime_refresh_timeout_sec']),
        runtime_recovery_timeout_sec=_positive(
            'runtime_recovery_timeout_sec',
            values['runtime_recovery_timeout_sec']),
        floor_z_m=_finite('floor_z_m', values['floor_z_m']),
        link_radius_m=_positive('link_radius_m', values['link_radius_m']),
        self_clearance_m=_positive(
            'self_clearance_m', values['self_clearance_m']),
        camera_holder_envelope_center_link6_m=_vector(
            'camera_holder_envelope_center_link6_m',
            values['camera_holder_envelope_center_link6_m'], 3),
        camera_holder_envelope_size_m=_vector(
            'camera_holder_envelope_size_m',
            values['camera_holder_envelope_size_m'], 3),
        camera_holder_external_clearance_m=_non_negative(
            'camera_holder_external_clearance_m',
            values['camera_holder_external_clearance_m']),
    )
    if any(item <= 0.0 for item in safety.camera_holder_envelope_size_m):
        raise ConfigurationError(
            'camera_holder_envelope_size_m values must be positive')
    planning = PlanningConfig(
        plan_max_age_sec=_positive(
            'plan_max_age_sec', values['plan_max_age_sec']),
        approval_confirmation=_nonempty(
            'approval_confirmation', values['approval_confirmation']),
        allow_mission_policy=_as_bool(values['allow_mission_policy']),
        closed_loop_one_view=_as_bool(values['closed_loop_one_view']),
    )
    return ExecutorConfigurationGroups(
        interfaces=interfaces,
        motion=motion,
        tracking=tracking,
        capture=capture,
        safety=safety,
        planning=planning,
        executor=ExecutorConfig(debug=_as_bool(values['debug'])),
        parameter_values=_freeze_parameter_values(values),
    )


def configured_value(owner, name):
    """
    Read frozen production config, with a no-ROS test-double fallback.

    Existing characterization tests call unbound node methods on small objects
    that expose only ``get_parameter``. Production nodes always have a typed
    ``configuration`` and never reach the fallback after startup.
    """
    configuration = getattr(owner, 'configuration', None)
    if configuration is not None:
        return configuration.value(name)
    return owner.get_parameter(name).value
