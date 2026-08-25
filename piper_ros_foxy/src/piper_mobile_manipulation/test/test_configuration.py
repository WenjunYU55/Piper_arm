"""Regression tests for the Phase 9 typed configuration boundary."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from piper_mobile_manipulation.configuration import (
    CaptureConfig,
    ConfigurationError,
    ExecutorInterfaceConfig,
    MissionCaptureConfig,
    MissionConfig,
    MissionMotionConfig,
    MissionWorkflowConfig,
    MotionConfig,
    ProcessConfig,
    SafetyConfig,
    TrackingConfig,
    configured_value,
    executor_parameter_defaults,
    load_executor_configuration,
    load_mission_configuration,
    mission_parameter_defaults,
)
from piper_mobile_manipulation.mission_engine import MissionEngine


LEGACY_MISSION_DEFAULTS = {
    'project_root': '/home/prl/Piper_arm',
    'manage_processes': True,
    'floor_profile': 'saved',
    'floor_profile_path': '',
    'enable_real_arm_motion': False,
    'motion_speed_profile_qualified': False,
    'free_motion_speed_percent': 30.0,
    'contact_speed_percent': 10.0,
    'required_captures': 8,
    'maximum_captures': 24,
    'home_pose_path': '',
    'require_staged_home_profile': True,
    'mission_spool_root': '/phase9/piper_target_scan_missions',
    'process_log_root': '/phase9/piper_target_scan_logs',
    'require_gateway_heartbeat': False,
    'max_pending_missions': 8,
    'mission_queue_coalesce_sec': 1.0,
    'debug': True,
}


LEGACY_EXECUTOR_DEFAULTS = {
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


def test_deployed_home_acceptance_matches_typed_configuration_defaults():
    path = Path(__file__).parents[1] / 'config' / 'scan_execution_params.yaml'
    with path.open(encoding='utf-8') as stream:
        deployed = yaml.safe_load(stream)['/**']['ros__parameters']

    defaults = executor_parameter_defaults()
    assert deployed['home_goal_tolerance_rad'] == \
        defaults['home_goal_tolerance_rad'] == 0.20
    assert deployed['home_motion_tolerance_rad'] == \
        defaults['home_motion_tolerance_rad'] == 0.20


def test_executor_and_tesseract_share_raised_virtual_floor():
    """Planning and execution must reject against the same support plane."""
    source_root = Path(__file__).resolve().parents[2]
    executor_path = (
        source_root / 'piper_mobile_manipulation' / 'config'
        / 'scan_execution_params.yaml')
    worker_path = (
        source_root / 'piper_tesseract_foxy' / 'model'
        / 'collision_model.yaml')
    with executor_path.open(encoding='utf-8') as stream:
        executor = yaml.safe_load(stream)['/**']['ros__parameters']
    with worker_path.open(encoding='utf-8') as stream:
        worker = yaml.safe_load(stream)['external_floor_clearance']

    assert executor['floor_z_m'] == 0.005
    assert worker['floor_z_m'] == executor['floor_z_m']
    assert worker['clearance_m'] == executor[
        'camera_holder_external_clearance_m'] == 0.005


class FakeParameterNode:
    """Minimal declare/get boundary with ROS-like override behavior."""

    def __init__(self, overrides=None):
        self.overrides = dict(overrides or {})
        self.parameters = {}
        self.read_counts = {}

    def declare_parameter(self, name, default):
        self.parameters[name] = self.overrides.get(name, default)

    def get_parameter(self, name):
        self.read_counts[name] = self.read_counts.get(name, 0) + 1
        return SimpleNamespace(value=self.parameters[name])


def test_parameter_defaults_exactly_match_the_frozen_pre_phase9_values():
    assert mission_parameter_defaults({
        'XDG_RUNTIME_DIR': '/phase9'}) == LEGACY_MISSION_DEFAULTS
    assert executor_parameter_defaults() == LEGACY_EXECUTOR_DEFAULTS


def test_mission_configuration_is_grouped_typed_and_read_once():
    node = FakeParameterNode({
        'required_captures': 9,
        'maximum_captures': 21,
        'free_motion_speed_percent': 42.0,
    })

    config = load_mission_configuration(
        node, {'XDG_RUNTIME_DIR': '/phase9'})

    assert isinstance(config.mission, MissionConfig)
    assert isinstance(config.process, ProcessConfig)
    assert isinstance(config.motion, MissionMotionConfig)
    assert isinstance(config.capture, MissionCaptureConfig)
    assert isinstance(config.workflow, MissionWorkflowConfig)
    assert config.capture.required_captures == 9
    assert config.capture.maximum_captures == 21
    assert config.motion.free_motion_speed_percent == 42.0
    assert config.process.floor_profile == 'saved'
    assert config.process.floor_profile_path == ''
    assert set(node.read_counts.values()) == {1}


def test_mission_configuration_rejects_unknown_floor_profile():
    with pytest.raises(ConfigurationError, match='saved, tabletop or ground'):
        load_mission_configuration(
            FakeParameterNode({'floor_profile': 'unknown'}),
            {'XDG_RUNTIME_DIR': '/phase9'},
        )


def test_executor_configuration_preserves_overrides_and_units():
    node = FakeParameterNode({
        'speed_percent': 12.5,
        'data_timeout_sec': 4.25,
        'plan_start_tolerance_rad': 0.04,
        'joint_states_topic': '/commissioning/joints',
    })

    config = load_executor_configuration(node)

    assert isinstance(config.interfaces, ExecutorInterfaceConfig)
    assert isinstance(config.motion, MotionConfig)
    assert isinstance(config.tracking, TrackingConfig)
    assert isinstance(config.capture, CaptureConfig)
    assert isinstance(config.safety, SafetyConfig)
    assert config.motion.speed_percent == 12.5
    assert config.tracking.data_timeout_sec == 4.25
    assert config.motion.plan_start_tolerance_rad == 0.04
    assert config.interfaces.joint_states_topic == '/commissioning/joints'
    assert set(node.read_counts.values()) == {1}


def test_configuration_snapshots_are_immutable():
    config = load_executor_configuration(FakeParameterNode())

    with pytest.raises(FrozenInstanceError):
        config.motion.speed_percent = 30.0
    with pytest.raises(TypeError):
        config.parameter_values['speed_percent'] = 30.0
    assert isinstance(
        config.parameter_values['return_home_positions_rad'], tuple)


def test_runtime_reads_the_frozen_value_not_the_ros_parameter_again():
    node = FakeParameterNode({'speed_percent': 12.5})
    config = load_executor_configuration(node)
    owner = SimpleNamespace(configuration=config)
    node.parameters['speed_percent'] = 99.0

    assert configured_value(owner, 'speed_percent') == 12.5
    assert node.read_counts['speed_percent'] == 1


def test_mission_engine_accepts_explicit_groups_without_option_lookups():
    class NoLegacyOptions:
        @staticmethod
        def boolean_option(*_args):
            raise AssertionError('legacy boolean option lookup was used')

        @staticmethod
        def numeric_option(*_args):
            raise AssertionError('legacy numeric option lookup was used')

    configuration = load_mission_configuration(FakeParameterNode())

    engine = MissionEngine(
        NoLegacyOptions(),
        motion_config=configuration.motion,
        capture_config=configuration.capture,
        workflow_config=configuration.workflow,
    )

    assert engine.motion_config is configuration.motion
    assert engine.capture_config is configuration.capture
    assert engine.workflow_config is configuration.workflow


def test_production_nodes_do_not_query_parameters_during_execution():
    package = Path(__file__).resolve().parents[1] / \
        'piper_mobile_manipulation'

    for name in (
            'target_scan_mission_node.py',
            'scan_viewpoint_executor_node.py'):
        source = (package / name).read_text(encoding='utf-8')
        assert 'self.get_parameter(' not in source


@pytest.mark.parametrize('overrides, expected', [
    ({'required_captures': 25, 'maximum_captures': 24},
     'maximum_captures'),
    ({'free_motion_speed_percent': float('nan')},
     'free_motion_speed_percent'),
    ({'max_pending_missions': 0}, 'max_pending_missions'),
])
def test_invalid_mission_configuration_fails_at_startup(overrides, expected):
    with pytest.raises(ConfigurationError, match=expected):
        load_mission_configuration(FakeParameterNode(overrides))


@pytest.mark.parametrize('overrides, expected', [
    ({'trajectory_command_rate_hz': 20.0, 'executor_tick_rate_hz': 30.0},
     'executor_tick_rate_hz'),
    ({'return_home_positions_rad': [0.0] * 5},
     'return_home_positions_rad'),
    ({'camera_holder_envelope_size_m': [0.1, 0.0, 0.1]},
     'camera_holder_envelope_size_m'),
    ({'capture_timeout_sec': -1.0}, 'capture_timeout_sec'),
])
def test_invalid_executor_configuration_fails_at_startup(overrides, expected):
    with pytest.raises(ConfigurationError, match=expected):
        load_executor_configuration(FakeParameterNode(overrides))


def test_configuration_does_not_mix_input_telemetry_or_derived_state():
    names = set()
    for group in (
            load_mission_configuration(FakeParameterNode()).mission,
            load_mission_configuration(FakeParameterNode()).capture,
            load_executor_configuration(FakeParameterNode()).tracking):
        names.update(group.__dataclass_fields__)

    assert not names.intersection({
        'target_label', 'rough_target', 'latest_joint_state',
        'tracked_target', 'plan', 'score', 'decision',
    })
