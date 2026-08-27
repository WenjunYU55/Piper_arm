import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from builtin_interfaces.msg import Time
from piper_sdk import C_PiperForwardKinematics

from piper_mobile_manipulation.scan_execution_modes import (
    acquisition_tracking_locked,
    commanded_speed_percent,
    measured_target_lock_rejection,
    MULTIVIEW_SCAN,
    plan_count_rejection,
    planned_speed_rejection,
    ROUGH_ACQUISITION,
    RETURN_HOME,
)
from piper_mobile_manipulation.scan_motion import (
    camera_target_path_reasons,
    approval_rejection_reason,
    bootstrap_start_limit_recovery_reasons,
    bootstrap_recovery_declaration_reasons,
    CollisionBox,
    configured_home_feedback_limit_reasons,
    energized_hold_target,
    startup_measured_hold_reference,
    feedback_joint_limit_reasons,
    PiperScanKinematics,
    configuration_collision_reasons,
    interpolate_joint_path,
    load_accepted_hand_eye,
    load_conservative_joint_limits,
    minimum_self_segment_clearance,
    motor_control_reasons,
    orbit_camera_view,
    powered_motion_calibration_rejection,
    segment_intersects_expanded_box,
    URDF_JOINT_LIMITS,
    validate_attached_box_external_clearance_path,
    validate_joint_path,
    validate_monotonic_self_clearance_escape,
)
from piper_mobile_manipulation.motion_limit_stability import MotionLimitStability
from piper_mobile_manipulation.scan_trajectory import (
    TIMING_POLICY_VERSION,
    validate_sdk_movej_waypoint_path,
    validate_tesseract_point,
)
from piper_mobile_manipulation.safety_evaluator import (
    ObstacleAuthority,
    runtime_gate_policy,
    SafetyMode,
)
from piper_mobile_manipulation.scan_viewpoint_executor_node import (
    MAX_RGBD_CAPTURE_READINESS_RETRIES,
    bootstrap_abort_retrace_uses_static_scene,
    configured_home_endpoint_rejection,
    abort_return_home_blocker,
    approved_retrace_validation_reasons,
    approved_multiview_motion_obstacle_snapshot,
    approved_return_home_obstacle_snapshot,
    home_position_sample_settled,
    target_position_window_sample_settled,
    joint_progress_error,
    missing_obstacles_can_wait,
    obstacle_scene_runtime_reasons,
    qualified_scan_center_update,
    rgbd_capture_handoff_action,
    retryable_rgbd_capture_rejection,
    visual_capture_rejection,
    runtime_gate_action,
    runtime_refresh_action,
    target_drift_before_approval_rejection,
    terminal_home_hold_required,
    waypoint_motion_action,
    ScanViewpointExecutorNode,
    trajectory_count_rejection,
)


def test_scan_center_upgrades_once_after_qualified_first_capture():
    bootstrap = np.asarray([0.40, 0.0, 0.10])
    model = np.asarray([0.35, -0.01, 0.14])

    center, qualified, changed = qualified_scan_center_update(
        None, bootstrap, 0, None)
    assert np.allclose(center, bootstrap)
    assert qualified is False
    assert changed is True

    center, qualified, changed = qualified_scan_center_update(
        center, model, 1, {'valid': True}, qualified)
    assert np.allclose(center, model)
    assert qualified is True
    assert changed is True

    later, still_qualified, changed = qualified_scan_center_update(
        center, [0.9, 0.9, 0.9], 2, {'valid': True}, qualified)
    assert np.allclose(later, model)
    assert still_qualified is True
    assert changed is False


def test_staged_direct_home_endpoint_uses_its_own_hash_bound_goal():
    rough = np.asarray([0.0, 0.0, 0.0, 0.0, 0.4, 0.0])
    storage = rough.copy()
    storage[5] = -math.pi

    assert configured_home_endpoint_rejection(
        'STORAGE_WRIST', storage, storage, rough) == ''
    assert configured_home_endpoint_rejection(
        'STARTUP_WRIST', rough, rough, rough) == ''
    assert 'declared goal does not match' in configured_home_endpoint_rejection(
        'STORAGE_WRIST', storage, rough, rough)
    assert 'executor configuration' in configured_home_endpoint_rejection(
        'ROUGH_HOME', storage, storage, rough)


def test_motion_requires_all_six_authoritative_motor_flags():
    status = SimpleNamespace(
        motor_feedback_valid=True,
        motor_1_driver_enabled=True,
        motor_2_driver_enabled=True,
        motor_3_driver_enabled=True,
        motor_4_driver_enabled=True,
        motor_5_driver_enabled=False,
        motor_6_driver_enabled=True,
        motor_faults=[],
        motor_watchdog_reason='',
    )
    reasons = motor_control_reasons(status, require_all_enabled=True)
    assert 'partial motor enable' in reasons[0]
    assert reasons[-1] == 'all six motor drivers are not enabled'


def test_mechanically_registered_camera_blocks_powered_motion_until_revalidated(
        tmp_path):
    calibration = {
        'status': 'accepted',
        'mechanical_registration': {
            'physical_revalidation': {'status': 'pending'},
        },
    }
    path = tmp_path / 'calibration.yaml'
    path.write_text(yaml.safe_dump(calibration), encoding='utf-8')
    assert 'blocked from powered motion' in powered_motion_calibration_rejection(path)

    calibration['mechanical_registration']['physical_revalidation']['status'] = 'passed'
    path.write_text(yaml.safe_dump(calibration), encoding='utf-8')
    assert powered_motion_calibration_rejection(path) == ''


def test_hold_request_does_not_change_executor_state():
    executor = SimpleNamespace(
        state='PROPOSAL_READY',
        reason='approved proposal is awaiting authorization',
        publish_hold=lambda: True,
    )
    response = SimpleNamespace(success=False, message='')

    result = ScanViewpointExecutorNode.hold_cb(executor, None, response)

    assert result.success
    assert result.message == 'current joint hold requested'
    assert executor.state == 'PROPOSAL_READY'
    assert executor.reason == 'approved proposal is awaiting authorization'


def test_startup_home_approval_gate_allows_missing_obstacle_telemetry():
    captured = {}
    executor = SimpleNamespace(
        mission_sha256='a' * 64,
        mission_authorization_valid=lambda: True,
        state='PROPOSAL_READY',
        plan_targets=[np.zeros(6)],
        plan_id='startup-home-plan',
        plan_created=9.0,
        plan_trajectory_sha256='b' * 64,
        real_motion_enabled=lambda: True,
        now=lambda: 10.0,
        get_parameter=lambda name: SimpleNamespace(value={
            'approval_confirmation': 'APPROVE',
            'plan_max_age_sec': 30.0,
        }[name]),
        is_acquisition=lambda: False,
        is_return_home=lambda: True,
        is_startup_home_static=lambda: True,
        runtime_reasons=lambda policy, **_kwargs: (
            captured.update({'policy': policy})
            or ['intentional test stop']),
    )
    request = SimpleNamespace(
        confirmation='MISSION_POLICY:' + executor.mission_sha256,
        plan_id=executor.plan_id,
        trajectory_sha256=executor.plan_trajectory_sha256,
    )
    response = SimpleNamespace(accepted=True, message='')

    ScanViewpointExecutorNode.approve_cb(executor, request, response)

    assert not response.accepted
    assert captured['policy'].mode == SafetyMode.RETURN_HOME
    assert captured['policy'].obstacle_authority == \
        ObstacleAuthority.STATIC_BOOTSTRAP
    assert not captured['policy'].require_motion_limits


LINK6_FROM_CAMERA = np.asarray([
    [-0.0635035764, 0.9974167728, -0.0335719700, -0.0745866291],
    [-0.9979815660, -0.0634575393, 0.0024360971, -0.0027843239],
    [0.0002994095, 0.0336589081, 0.9994333336, 0.0266401932],
    [0.0, 0.0, 0.0, 1.0],
])


def test_energized_hold_uses_the_measured_powered_pose():
    disabled_pose = [-0.017, -0.041, 0.033, -0.082, 0.290, 0.102]
    np.testing.assert_allclose(
        energized_hold_target(disabled_pose),
        [-0.017, 0.0, 0.0, -0.082, 0.290, 0.102],
    )


def test_startup_hold_reference_preserves_extended_negative_joint6():
    measured = [0.0, 0.0, 0.0, 0.0, 0.32, -3.459665]

    np.testing.assert_allclose(
        startup_measured_hold_reference(measured), measured)


def test_capture_input_liveness_gaps_are_retryable_while_held():
    assert retryable_rgbd_capture_rejection(
        'timestamped camera transform is unavailable: Lookup would require '
        'extrapolation into the future')
    assert retryable_rgbd_capture_rejection(
        'OCCLUSION_REJECTED: occlusion evidence is stale')
    assert retryable_rgbd_capture_rejection(
        'QUALITY_REJECTED: scan quality is missing')
    assert retryable_rgbd_capture_rejection('missing target_3d')
    assert retryable_rgbd_capture_rejection('missing detection mask')
    assert retryable_rgbd_capture_rejection(
        'confidence-qualified RGB-D bundle is still catching up with the '
        'exact detection-mask timestamp')
    assert retryable_rgbd_capture_rejection(
        'confidence-qualified native depth bundle is stale')
    assert retryable_rgbd_capture_rejection(
        'RGB and native depth timestamps are not synchronized')
    assert not retryable_rgbd_capture_rejection(
        'timestamped camera transform is unavailable: invalid frame')
    assert not retryable_rgbd_capture_rejection('camera files could not be saved')


def test_only_fresh_visual_rejections_exclude_a_pose():
    assert visual_capture_rejection(
        'QUALITY_REJECTED: quality POOR 0.300 is below GOOD 0.650')
    assert visual_capture_rejection(
        'OCCLUSION_REJECTED: settled target view is PARTIALLY_OCCLUDED')
    assert visual_capture_rejection('target_3d invalid')
    assert visual_capture_rejection(
        'DEPTH_QUALITY_REJECTED: ambiguous target depth layers')
    assert not retryable_rgbd_capture_rejection('target_3d invalid')
    assert not visual_capture_rejection(
        'OCCLUSION_REJECTED: occlusion evidence is stale')
    assert not visual_capture_rejection('camera files could not be saved')


def test_return_home_reuses_only_a_present_collision_qualified_scene():
    scene = SimpleNamespace(instances=[])
    assert approved_return_home_obstacle_snapshot(True, True, scene)
    assert not approved_return_home_obstacle_snapshot(False, True, scene)
    assert not approved_return_home_obstacle_snapshot(True, False, scene)
    assert not approved_return_home_obstacle_snapshot(True, True, None)


def test_only_an_inflight_qualified_multiview_segment_reuses_scene_snapshot():
    scene = SimpleNamespace(instances=[], scene_blocked=False)
    assert approved_multiview_motion_obstacle_snapshot(
        MULTIVIEW_SCAN, 'MOVING', True, scene)
    assert not approved_multiview_motion_obstacle_snapshot(
        MULTIVIEW_SCAN, 'SETTLING', True, scene)
    assert not approved_multiview_motion_obstacle_snapshot(
        MULTIVIEW_SCAN, 'MOVING', False, scene)
    assert not approved_multiview_motion_obstacle_snapshot(
        MULTIVIEW_SCAN, 'MOVING', True, None)
    assert not approved_multiview_motion_obstacle_snapshot(
        ROUGH_ACQUISITION, 'MOVING', True, scene)


def test_inflight_multiview_uses_stale_snapshot_but_not_missing_scene():
    calls = []
    scene = SimpleNamespace(instances=[], scene_blocked=False)
    executor = SimpleNamespace(
        state='MOVING',
        plan_kind=MULTIVIEW_SCAN,
        current_view=0,
        plan_collision_model_qualified=True,
        latest_obstacles=scene,
        acquisition_scene_snapshot_validated=False,
        abort_return_bootstrap_static_scene=False,
        is_acquisition=lambda: False,
        is_return_home=lambda: False,
        is_startup_home_static=lambda: False,
        returning_home=lambda: False,
        runtime_reasons=lambda policy, **_kwargs: calls.append(policy) or [],
        moving_tick=lambda: calls.append('moving'),
    )

    ScanViewpointExecutorNode.execution_tick(executor)

    assert calls[0].mode == SafetyMode.SCAN_MOTION
    assert calls[0].obstacle_authority == \
        ObstacleAuthority.APPROVED_SNAPSHOT
    assert calls[1] == 'moving'


def test_only_first_qualified_acquisition_segment_uses_static_abort_scene():
    assert bootstrap_abort_retrace_uses_static_scene(
        ROUGH_ACQUISITION, 0, True)
    assert not bootstrap_abort_retrace_uses_static_scene(
        ROUGH_ACQUISITION, 1, True)
    assert not bootstrap_abort_retrace_uses_static_scene(
        ROUGH_ACQUISITION, 0, False)
    assert not bootstrap_abort_retrace_uses_static_scene(
        MULTIVIEW_SCAN, 0, True)


def test_terminal_home_cancel_becomes_hold_instead_of_second_retrace():
    assert terminal_home_hold_required(
        'ABORTED',
        'operator cancelled; configured home reached')
    assert not terminal_home_hold_required(
        'ABORTED',
        'operator cancelled; current joint hold requested')
    assert not terminal_home_hold_required(
        'MOVING',
        'configured home reached')

    executor = SimpleNamespace(
        state='ABORTED',
        reason='operator cancelled; configured home reached',
        publish_hold=lambda: True,
    )
    response = SimpleNamespace(success=False, message='')
    ScanViewpointExecutorNode.cancel_cb(executor, None, response)
    assert response.success
    assert 'current joint hold requested' in response.message


def test_cancel_service_stops_and_holds_for_current_state_home_plan():
    calls = []
    executor = SimpleNamespace(
        state='ABORTED',
        reason='dedicated semantic occlusion assessment timed out',
        publish_hold=lambda: calls.append('hold') or True,
        _terminal_abort=lambda reason: calls.append(('abort', reason)),
    )
    response = SimpleNamespace(success=False, message='')

    ScanViewpointExecutorNode.cancel_cb(executor, None, response)

    assert response.success
    assert calls[0] == 'hold'
    assert calls[1][0] == 'abort'
    assert 'current-state return-home replanning' in calls[1][1]
    assert 'current joint hold requested' in response.message


def test_first_acquisition_cancel_retraces_without_new_obstacle_array():
    calls = []
    home = np.zeros(6)
    reached = np.full(6, 0.1)
    executor = SimpleNamespace(
        abort_return_in_progress=False,
        abort_return_reason='',
        plan_kind=ROUGH_ACQUISITION,
        plan_collision_model_qualified=True,
        plan_returns_home=False,
        state='WAITING_FOR_GROUNDING_DINO',
        retrace_joint_targets=[home, reached],
        current_joints=lambda: reached.copy(),
        get_parameter=lambda _name: SimpleNamespace(value=0.025),
        runtime_reasons=lambda policy, **_kwargs: (
            calls.append(('runtime', policy)) or []),
        validate_path=lambda path, boxes: (
            calls.append(('validate', path, boxes)) or []),
        obstacle_boxes=lambda: (_ for _ in ()).throw(
            AssertionError('bootstrap retrace must use its static scene')),
        plan_capture_count=3,
        current_view=0,
        current_path=[], current_path_velocities=[],
        current_path_accelerations=[], current_path_times=[], path_index=0,
        command_target=None, command_sent_at=0.0, command_samples_sent=0,
        motion_started_at=None, waypoint_started_at=None,
        waypoint_last_progress_at=None, waypoint_best_error=0.0,
        current_waypoint_error=0.0,
        acquisition_scene_snapshot_validated=False,
        publish_hold=lambda: calls.append(('hold',)),
        begin_runtime_refresh=lambda reason, require_workflow,
        allow_missing_obstacles: calls.append((
            'refresh', reason, require_workflow, allow_missing_obstacles)),
    )

    started, blocker = ScanViewpointExecutorNode.try_start_abort_return(
        executor, 'operator cancelled scan execution')

    assert started and blocker == ''
    runtime = next(item[1] for item in calls if item[0] == 'runtime')
    assert runtime.mode == SafetyMode.RETURN_HOME
    assert runtime.obstacle_authority == \
        ObstacleAuthority.STATIC_BOOTSTRAP
    validation = next(item for item in calls if item[0] == 'validate')
    assert validation[2] == []
    assert calls[-1][0] == 'refresh'
    assert calls[-1][3]
    assert executor.acquisition_scene_snapshot_validated


def test_invalid_new_plan_preserves_already_executed_retrace_history():
    history = [np.zeros(6), np.ones(6) * 0.2]
    executor = SimpleNamespace(
        state='IDLE',
        plan_kind=MULTIVIEW_SCAN,
        plan_source_request_id='',
        plan_candidate_count=13,
        plan_id='old-plan',
        retrace_joint_targets=[item.copy() for item in history],
        clear_plan=lambda: executor.retrace_joint_targets.clear(),
        now=lambda: 10.0,
        param_bool=lambda _name: False,
        publish_plan=lambda *_args: None,
        publish_status=lambda: None,
    )

    ScanViewpointExecutorNode.invalidate_plan(
        executor,
        'Tesseract proposal rejected',
        plan_kind=MULTIVIEW_SCAN,
        plan_id='full-request-id',
    )

    assert executor.plan_id == 'full-request-id'
    assert len(executor.retrace_joint_targets) == 2
    np.testing.assert_allclose(executor.retrace_joint_targets[0], history[0])
    np.testing.assert_allclose(executor.retrace_joint_targets[1], history[1])


def test_valid_but_rejected_tesseract_proposal_keeps_full_request_id():
    captured = {}
    executor = SimpleNamespace(
        state='IDLE',
        scan_history=[],
        latest_motion_limits=None,
        latest_tracking_health=None,
        fresh=lambda *_args: False,
        get_parameter=lambda name: SimpleNamespace(value={
            'min_execution_viewpoints': 13,
            'max_execution_viewpoints': 13,
            'acquisition_max_viewpoints': 5,
            'trajectory_joint_step_rad': 0.03,
            'trajectory_command_rate_hz': 100.0,
            'motion_limits_timeout_sec': 2.0,
            'speed_percent': 5.0,
        }[name]),
        param_bool=lambda name: {
            'closed_loop_one_view': False,
        }[name],
        invalidate_plan=lambda reason, **kwargs: captured.update(
            reason=reason, **kwargs),
    )
    proposal = SimpleNamespace(
        plan_kind=MULTIVIEW_SCAN,
        source_request_id='',
        valid=True,
        dry_run=True,
        real_arm_motion=False,
        backend='tesseract',
        plan_id='0123456789abcdef0123456789abcdef',
        trajectory_sha256='a' * 64,
        timing_policy=TIMING_POLICY_VERSION,
        trajectories=[],
        viewpoint_indices=[],
        bootstrap_recovery_end_points=[],
        bootstrap_recovery_joints=[],
        bootstrap_recovery_delta_rad=[],
        bootstrap_recovery_evidence_json=[],
        command_rate_hz=100.0,
        motion_limits_sha256='b' * 64,
        execution_speed_percent=5.0,
    )

    ScanViewpointExecutorNode.tesseract_plan_cb(executor, proposal)

    assert captured['plan_id'] == proposal.plan_id
    assert captured['plan_kind'] == MULTIVIEW_SCAN
    assert captured['reason'].startswith('invalid Tesseract proposal:')


def test_terminal_folded_home_recovery_validates_its_safe_reverse(monkeypatch):
    current = np.asarray([0.4] * 6)
    entry = np.asarray([0.0, 0.04, -0.04, 0.0, 0.439, 0.0])
    home = np.asarray([0.0, 0.0, 0.0, 0.0, 0.439, 0.0])
    checked = {}

    def monotonic(_kinematics, path, *_args, **_kwargs):
        checked['recovery'] = [np.asarray(item).copy() for item in path]
        return []

    monkeypatch.setattr(
        'piper_mobile_manipulation.scan_viewpoint_executor_node.'
        'validate_monotonic_self_clearance_escape',
        monotonic,
    )

    def parameter(name):
        return SimpleNamespace(value={
            'plan_start_tolerance_rad': 0.05,
            'floor_z_m': -1.0,
            'link_radius_m': 0.01,
            'self_clearance_m': 0.01,
            'trajectory_joint_step_rad': 0.01,
        }[name])

    def validate_normal(path, _obstacles):
        checked['normal'] = [np.asarray(item).copy() for item in path]
        return []

    executor = SimpleNamespace(
        current_joints=lambda: current.copy(),
        plan_collision_model_qualified=True,
        plan_paths=[[]] * 13 + [[current.copy(), entry.copy(), home.copy()]],
        plan_path_times=[[]] * 13 + [[0.0, 1.0, 2.0]],
        plan_path_velocities=[[]] * 13 + [[
            np.zeros(6), np.zeros(6), np.zeros(6)]],
        plan_path_accelerations=[[]] * 13 + [[
            np.zeros(6), np.zeros(6), np.zeros(6)]],
        current_view=13,
        plan_capture_count=13,
        plan_kind=MULTIVIEW_SCAN,
        plan_returns_home=True,
        plan_bootstrap_recovery_end_points=[-1] * 13 + [1],
        plan_bootstrap_recovery_joint_sets=[[]] * 13 + [[2, 3]],
        plan_startup_home_static=[False] * 14,
        kinematics=object(),
        joint_limits=[(-1.0, 1.0)] * 6,
        get_parameter=parameter,
        obstacle_boxes=lambda: [],
        validation_path=lambda path: [np.asarray(item).copy() for item in path],
        validate_path=validate_normal,
        is_acquisition=lambda: False,
    )

    reasons = ScanViewpointExecutorNode.prepare_current_view(executor)

    assert reasons == []
    np.testing.assert_allclose(checked['recovery'], [home, entry])
    np.testing.assert_allclose(checked['normal'], [current, entry])
    np.testing.assert_allclose(executor.current_path, [entry, home])


def test_powered_start_return_home_validates_both_recovery_corridors(monkeypatch):
    current = np.asarray([0.0, 0.0, -0.026, 0.031, 0.601, 0.036])
    powered_entry = current.copy()
    powered_entry[2] = -0.086
    home_entry = np.asarray([0.0, 0.0, -0.047, 0.068, 0.441, 0.013])
    home = home_entry.copy()
    home[2] = -0.017
    checked = {'recovery': []}

    def monotonic(_kinematics, path, *_args, **_kwargs):
        checked['recovery'].append([
            np.asarray(item).copy() for item in path])
        return []

    monkeypatch.setattr(
        'piper_mobile_manipulation.scan_viewpoint_executor_node.'
        'validate_monotonic_self_clearance_escape',
        monotonic,
    )

    def parameter(name):
        return SimpleNamespace(value={
            'plan_start_tolerance_rad': 0.05,
            'floor_z_m': -1.0,
            'link_radius_m': 0.01,
            'self_clearance_m': 0.01,
            'trajectory_joint_step_rad': 0.01,
        }[name])

    def validate_normal(path, _obstacles):
        checked['normal'] = [np.asarray(item).copy() for item in path]
        return []

    path = [current, powered_entry, home_entry, home]
    executor = SimpleNamespace(
        current_joints=lambda: current.copy(),
        plan_collision_model_qualified=True,
        plan_paths=[path],
        plan_path_times=[[0.0, 0.01, 0.02, 0.03]],
        plan_path_velocities=[[np.zeros(6) for _ in path]],
        plan_path_accelerations=[[np.zeros(6) for _ in path]],
        current_view=0,
        plan_capture_count=0,
        plan_kind=RETURN_HOME,
        plan_returns_home=True,
        plan_bootstrap_recovery_end_points=[2],
        plan_bootstrap_recovery_joint_sets=[[3]],
        plan_powered_start_recovery_end_points=[1],
        plan_powered_start_recovery_joint_sets=[[3]],
        plan_startup_home_static=[True],
        kinematics=object(),
        joint_limits=[(-1.0, 1.0)] * 6,
        get_parameter=parameter,
        obstacle_boxes=lambda: (_ for _ in ()).throw(
            AssertionError('startup home must not consume target obstacles')),
        validation_path=lambda values: [
            np.asarray(item).copy() for item in values],
        validate_path=validate_normal,
        is_acquisition=lambda: False,
    )

    reasons = ScanViewpointExecutorNode.prepare_current_view(executor)

    assert reasons == []
    np.testing.assert_allclose(checked['recovery'][0], [home, home_entry])
    np.testing.assert_allclose(
        checked['recovery'][1], [current, powered_entry])
    np.testing.assert_allclose(
        checked['normal'], [powered_entry, home_entry])
    np.testing.assert_allclose(
        executor.current_path, [powered_entry, home_entry, home])


def test_configured_direct_home_bypasses_self_collision_but_keeps_external_check():
    current = np.asarray([0.1, 0.2, -0.3, 0.0, 0.4, 0.5])
    home = np.asarray([0.0, 0.0, 0.0, 0.0, 0.399345492, 0.0])
    checked = {}

    def parameter(name):
        return SimpleNamespace(value={
            'plan_start_tolerance_rad': 0.05,
            'trajectory_joint_step_rad': 0.01,
        }[name])

    executor = SimpleNamespace(
        current_joints=lambda: current.copy(),
        plan_collision_model_qualified=True,
        plan_paths=[[current.copy(), home.copy()]],
        plan_path_times=[[0.0, 0.01]],
        plan_path_velocities=[[np.zeros(6), np.zeros(6)]],
        plan_path_accelerations=[[np.zeros(6), np.zeros(6)]],
        current_view=0,
        plan_capture_count=0,
        plan_kind=RETURN_HOME,
        plan_returns_home=True,
        plan_bootstrap_recovery_end_points=[-1],
        plan_bootstrap_recovery_joint_sets=[[]],
        plan_powered_start_recovery_end_points=[-1],
        plan_powered_start_recovery_joint_sets=[[]],
        plan_startup_home_static=[False],
        plan_configured_home_direct=[True],
        plan_configured_home_stages=['ROUGH_HOME'],
        get_parameter=parameter,
        obstacle_boxes=lambda: (_ for _ in ()).throw(
            AssertionError('direct home must not consume obstacle geometry')),
        validate_path=lambda *_args: (_ for _ in ()).throw(
            AssertionError('direct home must not run robot self-collision validation')),
        validate_attached_tool_external_path=lambda path, obstacles: (
            checked.update({
                'path': [np.asarray(item).copy() for item in path],
                'obstacles': list(obstacles),
            }) or []),
        is_acquisition=lambda: False,
        is_return_home=lambda: True,
    )

    reasons = ScanViewpointExecutorNode.prepare_current_view(executor)

    assert reasons == []
    np.testing.assert_allclose(checked['path'], [current, home])
    assert checked['obstacles'] == []
    np.testing.assert_allclose(executor.current_path, [home])


def test_camera_target_path_accepts_small_boresight_changes():
    class FakeKinematics:
        @staticmethod
        def camera_transform(joints):
            angle = float(joints[0])
            transform = np.eye(4)
            transform[:3, 2] = [math.sin(angle), 0.0, math.cos(angle)]
            return transform

    path = [np.asarray([math.radians(value), 0, 0, 0, 0, 0])
            for value in (0.0, 10.0, 19.5)]

    assert camera_target_path_reasons(
        FakeKinematics(), path, [0.0, 0.0, 1.0], 20.0, 0.22) == []


def test_camera_target_path_rejects_off_axis_joint_shortcut():
    class FakeKinematics:
        @staticmethod
        def camera_transform(joints):
            angle = float(joints[0])
            transform = np.eye(4)
            transform[:3, 2] = [math.sin(angle), 0.0, math.cos(angle)]
            return transform

    rejection = camera_target_path_reasons(
        FakeKinematics(),
        [np.zeros(6), np.asarray([math.radians(25.0), 0, 0, 0, 0, 0])],
        [0.0, 0.0, 1.0], 20.0, 0.22)

    assert 'leaves the 20.0-degree camera boresight cone' in rejection[0]


def test_camera_target_path_rejects_motion_through_close_target():
    class FakeKinematics:
        @staticmethod
        def camera_transform(joints):
            transform = np.eye(4)
            transform[2, 3] = float(joints[0])
            return transform

    rejection = camera_target_path_reasons(
        FakeKinematics(),
        [np.zeros(6), np.asarray([0.80, 0, 0, 0, 0, 0])],
        [0.0, 0.0, 1.0], 20.0, 0.22)

    assert 'approaches target to 0.200m' in rejection[0]


def test_home_settle_uses_target_error_and_successive_position_delta():
    target = np.asarray([0.0, 0.0, 0.0, -0.041, 0.355, 0.043])
    first = target + np.asarray([0.0, 0.021, -0.001, 0.0, 0.001, 0.0])
    second = first + np.asarray([0.0, 0.0005, 0.0, 0.0, -0.0005, 0.0])
    assert not home_position_sample_settled(
        first, target, None, 0.025, 0.005)
    assert home_position_sample_settled(
        second, target, first, 0.025, 0.005)
    assert not home_position_sample_settled(
        target + 0.026, target, target, 0.025, 0.005)
    assert not home_position_sample_settled(
        target, target, target + 0.006, 0.025, 0.005)


def test_endpoint_settle_uses_target_bounded_position_window_not_speed_spikes():
    target = np.zeros(6)
    first = np.asarray([0.020, 0.0, 0.0, 0.0, 0.0, 0.0])
    settled, anchor = target_position_window_sample_settled(
        first, target, None, 0.025, 0.005)
    assert not settled
    settled, retained = target_position_window_sample_settled(
        first + 0.002, target, anchor, 0.025, 0.005)
    assert settled
    assert np.allclose(retained, anchor)
    settled, reset = target_position_window_sample_settled(
        first + 0.006, target, anchor, 0.030, 0.005)
    assert not settled
    assert np.allclose(reset, first + 0.006)
    settled, reset = target_position_window_sample_settled(
        target + 0.031, target, anchor, 0.030, 0.005)
    assert not settled
    assert reset is None


def test_endpoint_settle_accepts_measured_piper_j4_feedback_step_only():
    target = np.asarray([0.0, 0.0, 0.0, -0.0052, 0.9, 0.0])
    first = target.copy()
    first[3] = -0.006227508
    second = first.copy()
    second[3] = 0.0

    settled, anchor = target_position_window_sample_settled(
        first, target, None, 0.025, 0.007)
    assert not settled
    settled, retained = target_position_window_sample_settled(
        second, target, anchor, 0.025, 0.007)
    assert settled
    assert np.allclose(retained, anchor)

    larger_motion = first.copy()
    larger_motion[3] += 0.0071
    settled, reset = target_position_window_sample_settled(
        larger_motion, target, anchor, 0.025, 0.007)
    assert not settled
    assert np.allclose(reset, larger_motion)


def test_plan_approval_settle_anchors_to_current_pose_before_endpoint_exists():
    current = np.asarray([0.1, 0.2, -0.3, 0.4, -0.5, 0.6])
    executor = SimpleNamespace(
        latest_joint_state=object(),
        fresh=lambda key, timeout: key == 'joints' and timeout == 1.0,
        command_target=None,
        settle_position_anchor=None,
        settle_last_sample_ok=False,
        settle_last_joint_update=-1e9,
        updated={'joints': 1.0},
        current_joints=lambda: current.copy(),
        get_parameter=lambda name: SimpleNamespace(value={
            'joint_goal_tolerance_rad': 0.025,
            'endpoint_position_settled_rad': 0.005,
        }[name]),
    )

    assert not ScanViewpointExecutorNode.joints_settled(executor)
    assert np.allclose(executor.settle_position_anchor, current)

    current += 0.002
    executor.updated['joints'] = 2.0
    assert ScanViewpointExecutorNode.joints_settled(executor)
    assert executor.command_target is None


def test_runtime_recovery_settle_proves_hold_without_losing_resume_target():
    current = np.asarray([0.1, 0.2, -0.3, 0.4, -0.5, 0.6])
    resume_target = current + 0.2
    executor = SimpleNamespace(
        latest_joint_state=object(),
        fresh=lambda key, timeout: key == 'joints' and timeout == 1.0,
        command_target=resume_target.copy(),
        settle_position_anchor=None,
        settle_last_sample_ok=False,
        settle_last_joint_update=-1e9,
        updated={'joints': 1.0},
        current_joints=lambda: current.copy(),
        get_parameter=lambda name: SimpleNamespace(value={
            'joint_goal_tolerance_rad': 0.025,
            'endpoint_position_settled_rad': 0.005,
        }[name]),
    )

    assert not ScanViewpointExecutorNode.joints_settled(
        executor, settle_at_current=True)
    current += 0.002
    executor.updated['joints'] = 2.0
    assert ScanViewpointExecutorNode.joints_settled(
        executor, settle_at_current=True)
    assert np.allclose(executor.command_target, resume_target)


def test_runtime_recovery_wait_requests_current_hold_settle_authority():
    calls = []
    executor = SimpleNamespace(
        runtime_reasons=lambda policy, **_kwargs: calls.append(policy) or [
            'obstacles data missing or stale'],
        runtime_refresh_resume_state='MOVING',
        runtime_refresh_require_workflow=False,
        runtime_refresh_allow_missing_obstacles=False,
        acquisition_scene_snapshot_validated=False,
        plan_kind=MULTIVIEW_SCAN,
        plan_collision_model_qualified=True,
        latest_obstacles=object(),
        is_acquisition=lambda: False,
        get_parameter=lambda name: SimpleNamespace(value={
            'runtime_recovery_timeout_sec': 10.0,
            'runtime_refresh_timeout_sec': 3.0,
        }[name]),
        now=lambda: 1.0,
        state_started=0.0,
        returning_home=lambda: False,
        pending_motion_reason='',
    )

    ScanViewpointExecutorNode.waiting_for_runtime_refresh_tick(executor)

    assert calls[0].settle_at_current_hold


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_transient_invalid_motion_limits_do_not_replace_fresh_valid_set():
    valid = SimpleNamespace(valid=True, limits_sha256='a' * 64)
    invalid = SimpleNamespace(valid=False, limits_sha256='0' * 64)
    marked = []
    executor = SimpleNamespace(
        latest_motion_limits=valid,
        motion_limit_stability=MotionLimitStability(),
        now=lambda: 1.0,
        mark=lambda key: marked.append(key),
    )
    executor.motion_limit_stability.observe(valid, 0.0)

    ScanViewpointExecutorNode.motion_limits_cb(executor, invalid)

    assert executor.latest_motion_limits is valid
    assert marked == []


def test_motion_limits_ignore_initial_invalid_and_accept_next_valid_sample():
    invalid = SimpleNamespace(valid=False, limits_sha256='0' * 64)
    valid = SimpleNamespace(valid=True, limits_sha256='a' * 64)
    marked = []
    executor = SimpleNamespace(
        latest_motion_limits=None,
        motion_limit_stability=MotionLimitStability(),
        now=lambda: 1.0,
        mark=lambda key: marked.append(key),
    )

    ScanViewpointExecutorNode.motion_limits_cb(executor, invalid)
    assert executor.latest_motion_limits is None
    ScanViewpointExecutorNode.motion_limits_cb(executor, valid)

    assert executor.latest_motion_limits is valid
    assert marked == ['motion_limits']


def test_hash_bound_plan_waits_for_queued_matching_limit_refresh():
    limits = SimpleNamespace(valid=True, limits_sha256='a' * 64)
    proposal = SimpleNamespace(
        plan_kind=MULTIVIEW_SCAN,
        source_request_id='',
        plan_id='plan-waiting-for-limits',
        valid=True,
        motion_limits_sha256='a' * 64,
    )
    states = []
    executor = SimpleNamespace(
        state='IDLE',
        latest_motion_limits=limits,
        pending_limit_refresh_plan=None,
        pending_limit_refresh_deadline=0.0,
        fresh=lambda *_args: False,
        now=lambda: 10.0,
        get_parameter=lambda name: SimpleNamespace(value={
            'motion_limits_timeout_sec': 3.0,
        }[name]),
        set_state=lambda state, reason: states.append((state, reason)),
    )

    ScanViewpointExecutorNode.tesseract_plan_cb(executor, proposal)

    assert executor.pending_limit_refresh_plan is proposal
    assert executor.pending_limit_refresh_deadline == 13.0
    assert states[0][0] == 'WAITING_FOR_PLAN_LIMITS'


def test_fresh_matching_limit_retries_deferred_plan_without_motion():
    valid = SimpleNamespace(valid=True, limits_sha256='a' * 64)
    proposal = SimpleNamespace(plan_id='deferred')
    retried = []
    marked = []
    executor = SimpleNamespace(
        latest_motion_limits=valid,
        motion_limit_stability=MotionLimitStability(),
        pending_limit_refresh_plan=proposal,
        pending_limit_refresh_deadline=4.0,
        now=lambda: 2.0,
        mark=lambda key: marked.append(key),
        tesseract_plan_cb=lambda msg: retried.append(msg),
    )

    ScanViewpointExecutorNode.motion_limits_cb(executor, valid)

    assert marked == ['motion_limits']
    assert retried == [proposal]
    assert executor.pending_limit_refresh_plan is None
    assert executor.pending_limit_refresh_deadline == 0.0


def test_deferred_plan_expires_fail_closed_without_fresh_limits():
    proposal = SimpleNamespace(
        plan_kind=MULTIVIEW_SCAN,
        source_request_id='',
        plan_id='expired-deferred-plan',
    )
    invalid = {}
    executor = SimpleNamespace(
        pending_limit_refresh_plan=proposal,
        pending_limit_refresh_deadline=3.0,
        now=lambda: 3.1,
        invalidate_plan=lambda reason, **kwargs: invalid.update(
            reason=reason, **kwargs),
    )

    ScanViewpointExecutorNode.tick(executor)

    assert executor.pending_limit_refresh_plan is None
    assert 'did not refresh before the bounded deadline' in invalid['reason']
    assert invalid['plan_id'] == proposal.plan_id


def test_moving_target_mode_does_not_reject_approved_camera_pose_for_drift():
    assert target_drift_before_approval_rejection(0.20, 0.015, True) == ''
    assert 'target moved 0.020m' in target_drift_before_approval_rejection(
        0.020, 0.015, False)
    assert target_drift_before_approval_rejection(
        0.010, 0.015, False) == ''


def test_runtime_gate_holds_only_for_transport_freshness_gaps():
    assert runtime_gate_action([]) == 'continue'
    assert runtime_gate_action([
        'obstacles data missing or stale',
        'camera_clock data missing or stale',
    ]) == 'hold_for_refresh'
    assert runtime_gate_action([
        'camera timestamp CLOCK_OFFSET: camera timestamp is in the future',
    ]) == 'hold_for_refresh'
    assert runtime_gate_action([
        'invalid obstacle geometry is present',
    ]) == 'abort'
    assert runtime_gate_action([
        'obstacles data missing or stale',
        'arm err_code=2',
    ]) == 'abort'


def test_obstacle_transform_gap_waits_but_unqualified_geometry_aborts():
    transform_gap = SimpleNamespace(
        scene_blocked=True,
        blocking_reason='2:transform_unavailable:future extrapolation',
        instances=[SimpleNamespace(
            valid=False,
            validity_reason='transform_unavailable:future extrapolation')],
    )
    stale_transform = SimpleNamespace(
        scene_blocked=True,
        blocking_reason='2:stale_transform',
        instances=[SimpleNamespace(
            valid=False, validity_reason='stale_transform')],
    )
    unqualified = SimpleNamespace(
        scene_blocked=True,
        blocking_reason='2:semantic_probe_3d_geometry_not_qualified',
        instances=[SimpleNamespace(
            valid=False,
            validity_reason='semantic_probe_3d_geometry_not_qualified')],
    )

    assert obstacle_scene_runtime_reasons(transform_gap) == [
        'obstacles data missing or stale']
    assert obstacle_scene_runtime_reasons(stale_transform) == [
        'obstacles data missing or stale']
    assert obstacle_scene_runtime_reasons(unqualified) == [
        'invalid obstacle geometry is present']


def test_abort_return_home_only_rejects_untrusted_control_authority():
    assert abort_return_home_blocker('capture service response timed out') == ''
    assert abort_return_home_blocker('workflow finish service failed') == ''
    assert abort_return_home_blocker(
        'runtime safety gate: invalid obstacle geometry is present') == ''
    assert abort_return_home_blocker(
        'SDK MoveJ waypoint made no measurable joint progress') == ''
    assert abort_return_home_blocker('operator cancelled scan execution') == ''
    assert abort_return_home_blocker(
        'runtime safety gate: obstacle collision is present') == ''
    assert abort_return_home_blocker(
        'joint feedback became invalid') == 'joint feedback became invalid'
    assert abort_return_home_blocker(
        'occlusion scene contains unsafe, blocked, or invalid obstacle geometry') == ''


def test_approved_retrace_ignores_only_repeated_static_self_clearance():
    reasons = approved_retrace_validation_reasons([
        'trajectory step 125: self-collision clearance between link segments 2 and 5',
        'trajectory step 12: obstacle collision with box leaf',
        'trajectory step 4: joint 3 is outside configured limits',
    ])

    assert reasons == [
        'trajectory step 12: obstacle collision with box leaf',
        'trajectory step 4: joint 3 is outside configured limits',
    ]


def test_multiview_capture_settle_does_not_require_tracking_lock():
    executor = SimpleNamespace(
        latest_camera_timestamp_health=SimpleNamespace(healthy=True),
        latest_tracking_health=SimpleNamespace(
            lifecycle_state='LOST', camera_settled=False,
            prediction_only=True),
        latest_target_status='LOST',
        joints_settled=lambda: True,
    )

    assert ScanViewpointExecutorNode.capture_pose_settled(executor)
    assert not ScanViewpointExecutorNode.settled_and_tracking(executor)


def test_multiview_capture_still_requires_stationary_arm_and_healthy_clock():
    executor = SimpleNamespace(
        latest_camera_timestamp_health=SimpleNamespace(healthy=False),
        joints_settled=lambda: True,
    )
    assert not ScanViewpointExecutorNode.capture_pose_settled(executor)

    executor.latest_camera_timestamp_health.healthy = True
    executor.joints_settled = lambda: False
    assert not ScanViewpointExecutorNode.capture_pose_settled(executor)


def test_non_safety_abort_retraces_only_reached_approved_targets():
    events = []
    home = np.zeros(6)
    first = np.full(6, 0.1)
    current = np.full(6, 0.2)
    executor = SimpleNamespace(
        abort_return_in_progress=False,
        abort_return_reason='',
        plan_kind=MULTIVIEW_SCAN,
        plan_returns_home=True,
        state='CAPTURING_RGBD',
        retrace_joint_targets=[home, first, current],
        current_joints=lambda: current.copy(),
        get_parameter=lambda name: SimpleNamespace(value={
            'plan_start_tolerance_rad': 0.025,
        }[name]),
        runtime_reasons=lambda *_args, **_kwargs: [],
        validate_path=lambda path, boxes: (
            events.append(('validate', [item.copy() for item in path], boxes))
            or []),
        obstacle_boxes=lambda: [],
        plan_capture_count=13,
        current_view=2,
        current_path=[],
        current_path_velocities=[],
        current_path_accelerations=[],
        current_path_times=[],
        path_index=0,
        command_target=current.copy(),
        command_sent_at=1.0,
        command_samples_sent=3,
        motion_started_at=1.0,
        waypoint_started_at=1.0,
        waypoint_last_progress_at=1.0,
        waypoint_best_error=0.0,
        current_waypoint_error=0.0,
        publish_hold=lambda: events.append(('hold',)),
        begin_runtime_refresh=lambda reason, require_workflow,
        allow_missing_obstacles: events.append((
            'refresh', reason, require_workflow, allow_missing_obstacles)),
    )

    started, blocker = ScanViewpointExecutorNode.try_start_abort_return(
        executor, 'capture service response timed out')

    assert started
    assert blocker == ''
    assert executor.abort_return_in_progress
    assert executor.abort_return_reason == 'capture service response timed out'
    assert executor.current_view == 13
    assert np.allclose(executor.current_path[0], first)
    assert np.allclose(executor.current_path[1], home)
    assert events[0][0] == 'validate'
    assert np.allclose(events[0][1][0], current)
    assert events[-2:] == [
        ('hold',),
        (
            'refresh',
            'non-safety abort accepted; retracing already executed approved '
            'targets to the configured home',
            False,
            False,
        ),
    ]


def test_cancelled_acquisition_retraces_to_original_powered_home():
    events = []
    home = np.zeros(6)
    reached = np.full(6, 0.1)
    executor = SimpleNamespace(
        abort_return_in_progress=False,
        plan_kind='ROUGH_ACQUISITION',
        plan_returns_home=False,
        state='ACQUIRED',
        retrace_joint_targets=[home, reached],
        current_joints=lambda: reached.copy(),
        get_parameter=lambda name: SimpleNamespace(value={
            'plan_start_tolerance_rad': 0.025,
        }[name]),
        runtime_reasons=lambda *_args, **_kwargs: [],
        validate_path=lambda path, _boxes: (
            events.append([item.copy() for item in path]) or []),
        obstacle_boxes=lambda: [],
        plan_capture_count=2,
        current_view=2,
        current_path=[],
        current_path_velocities=[],
        current_path_accelerations=[],
        current_path_times=[],
        path_index=0,
        command_target=None,
        command_sent_at=0.0,
        command_samples_sent=0,
        motion_started_at=None,
        waypoint_started_at=None,
        waypoint_last_progress_at=None,
        waypoint_best_error=0.0,
        current_waypoint_error=0.0,
        publish_hold=lambda: None,
        begin_runtime_refresh=lambda *_args, **_kwargs: None,
    )

    started, blocker = ScanViewpointExecutorNode.try_start_abort_return(
        executor, 'operator cancelled scan execution')

    assert started and not blocker
    assert np.allclose(executor.current_path, [home])
    assert np.allclose(events[0], [reached, home])


def test_cancelled_inflight_move_first_retraces_to_last_reached_endpoint():
    validated = []
    home = np.zeros(6)
    reached = np.full(6, 0.1)
    stopped = np.full(6, 0.15)
    executor = SimpleNamespace(
        abort_return_in_progress=False,
        plan_kind=MULTIVIEW_SCAN,
        plan_returns_home=True,
        state='ABORTED',
        retrace_joint_targets=[home, reached],
        current_joints=lambda: stopped.copy(),
        get_parameter=lambda _name: SimpleNamespace(value=0.025),
        runtime_reasons=lambda *_args, **_kwargs: [],
        validate_path=lambda path, _boxes: (
            validated.extend(item.copy() for item in path) or []),
        obstacle_boxes=lambda: [],
        plan_capture_count=13,
        current_view=1,
        current_path=[], current_path_velocities=[],
        current_path_accelerations=[], current_path_times=[], path_index=0,
        command_target=None, command_sent_at=0.0, command_samples_sent=0,
        motion_started_at=None, waypoint_started_at=None,
        waypoint_last_progress_at=None, waypoint_best_error=0.0,
        current_waypoint_error=0.0,
        publish_hold=lambda: None,
        begin_runtime_refresh=lambda *_args, **_kwargs: None,
    )

    started, blocker = ScanViewpointExecutorNode.try_start_abort_return(
        executor, 'operator cancelled scan execution')

    assert started and not blocker
    assert np.allclose(executor.current_path, [reached, home])
    assert np.allclose(validated, [stopped, reached, home])


def test_legacy_retrace_still_requires_executed_target_history():
    executor = SimpleNamespace(
        abort_return_in_progress=False,
        plan_kind=MULTIVIEW_SCAN,
        plan_returns_home=True,
        retrace_joint_targets=[],
    )

    started, blocker = ScanViewpointExecutorNode.try_start_abort_return(
        executor, 'invalid obstacle geometry is present')

    assert not started
    assert 'no executed approved target history' in blocker


def test_fixed_j6_planner_cannot_return_as_a_fallback():
    executor_source = (
        PACKAGE_ROOT / 'piper_mobile_manipulation' /
        'scan_viewpoint_executor_node.py'
    ).read_text()
    motion_source = (
        PACKAGE_ROOT / 'piper_mobile_manipulation' / 'scan_motion.py'
    ).read_text()
    launch_source = (
        PACKAGE_ROOT / 'launch' / 'supervised_viewpoint_execution.launch.py'
    ).read_text()
    combined = '\n'.join((executor_source, motion_source, launch_source))

    assert 'legacy_fixed_j6' not in combined
    assert 'solve_fixed_j6_viewpoint' not in combined
    assert "'planning_backend'" not in combined
    assert "msg.backend != 'tesseract'" in executor_source
    assert "msg.planner_backend = 'tesseract'" in executor_source


def test_automatic_capture_maximum_is_shared_with_session_planner():
    launch_source = (
        PACKAGE_ROOT / 'launch' / 'supervised_viewpoint_execution.launch.py'
    ).read_text()
    planner_block = launch_source.split("planner = Node(", 1)[1].split(
        "acquisition = Node(", 1)[0]

    assert "'session_max_views': ParameterValue(" in planner_block
    assert "LaunchConfiguration('max_execution_viewpoints')" in planner_block


def test_execution_plan_publisher_is_latched_for_sequential_consumers():
    executor_source = (
        PACKAGE_ROOT / 'piper_mobile_manipulation' /
        'scan_viewpoint_executor_node.py'
    ).read_text()
    plan_block = executor_source.split(
        'self.plan_pub = self.create_publisher(', 1)[1].split(
            'self.status_pub =', 1)[0]
    assert 'ScanExecutionPlan' in plan_block
    assert 'history_qos' in plan_block


def test_fk_position_matches_piper_sdk_mode_zero():
    joints = np.asarray([0.2, 1.1, -1.4, 0.25, 0.3, -0.1])
    ours = PiperScanKinematics(LINK6_FROM_CAMERA).forward(joints)
    sdk = C_PiperForwardKinematics(dh_is_offset=0).CalFK(joints.tolist())[-1]
    np.testing.assert_allclose(ours[:3, 3] * 1000.0, sdk[:3], atol=1e-6)


def test_joint_path_respects_maximum_step():
    start = np.zeros(6)
    goal = np.asarray([0.11, 0.02, -0.05, 0.0, 0.0, 0.0])
    path = interpolate_joint_path(start, goal, 0.03)
    previous = start
    for joints in path:
        assert np.max(np.abs(joints - previous)) <= 0.0300001
        previous = joints
    np.testing.assert_allclose(path[-1], goal)


def test_feedback_limit_tolerance_accepts_only_encoder_boundary_noise():
    joints = np.zeros(6)
    joints[2] = 0.0005
    assert feedback_joint_limit_reasons(
        joints, URDF_JOINT_LIMITS, tolerance_rad=0.001) == []

    joints[2] = 0.0021
    reasons = feedback_joint_limit_reasons(
        joints, URDF_JOINT_LIMITS, tolerance_rad=0.001)
    assert len(reasons) == 1
    assert 'joint3 feedback 0.002100' in reasons[0]
    assert 'encoder tolerance' in reasons[0]


def test_feedback_limit_hard_cap_accepts_observed_powered_boundary_noise():
    joints = np.zeros(6)
    joints[2] = 0.002460
    assert feedback_joint_limit_reasons(
        joints, URDF_JOINT_LIMITS, tolerance_rad=0.005) == []

    joints[2] = 0.005001
    reasons = feedback_joint_limit_reasons(
        joints, URDF_JOINT_LIMITS, tolerance_rad=0.005)
    assert len(reasons) == 1
    assert 'joint3 feedback 0.005001' in reasons[0]


def test_feedback_limit_tolerance_is_hard_capped():
    with pytest.raises(ValueError, match=r'within \[0.0, 0.0050\]'):
        feedback_joint_limit_reasons(
            np.zeros(6), URDF_JOINT_LIMITS, tolerance_rad=0.0051)


def test_configured_home_feedback_allows_only_the_operator_qualified_band():
    joints = np.zeros(6)
    joints[5] = -math.pi - 0.03168
    assert configured_home_feedback_limit_reasons(
        joints, URDF_JOINT_LIMITS, tolerance_rad=0.3) == []

    joints[5] = -math.pi - 0.300001
    reasons = configured_home_feedback_limit_reasons(
        joints, URDF_JOINT_LIMITS, tolerance_rad=0.3)
    assert len(reasons) == 1
    assert 'direct-home tolerance' in reasons[0]

    with pytest.raises(ValueError, match=r'within \[0.0, 0.3000\]'):
        configured_home_feedback_limit_reasons(
            np.zeros(6), URDF_JOINT_LIMITS, tolerance_rad=0.3001)


def test_startup_wrist_uses_its_existing_240_degree_branch_not_home_slack():
    joints = np.zeros(6)
    joints[5] = -3.459665

    assert configured_home_feedback_limit_reasons(
        joints, URDF_JOINT_LIMITS, tolerance_rad=0.3,
        home_stage='STARTUP_WRIST') == []
    assert configured_home_feedback_limit_reasons(
        joints, URDF_JOINT_LIMITS, tolerance_rad=0.3)

    joints[5] = -math.radians(240.0) - 0.000001
    reasons = configured_home_feedback_limit_reasons(
        joints, URDF_JOINT_LIMITS, tolerance_rad=0.3,
        home_stage='STARTUP_WRIST')
    assert len(reasons) == 1


def test_bootstrap_limit_recovery_accepts_only_monotonic_joint3_inward_path():
    path = [
        np.asarray([0.0, 0.0, 0.0327, 0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, -0.0073, 0.0, 0.0, 0.0]),
    ]
    assert bootstrap_start_limit_recovery_reasons(
        path, URDF_JOINT_LIMITS, 3, 0.04) == []

    outward = [
        path[0],
        np.asarray([0.0, 0.0, 0.035, 0.0, 0.0, 0.0]),
        path[1],
    ]
    assert any(
        'farther outside' in reason
        for reason in bootstrap_start_limit_recovery_reasons(
            outward, URDF_JOINT_LIMITS, 3, 0.04))

    wrong_joint = [
        path[0],
        np.asarray([2.5, 0.0, -0.0073, 0.0, 0.0, 0.0]),
    ]
    assert any(
        'joint1' in reason
        for reason in bootstrap_start_limit_recovery_reasons(
            wrong_joint, URDF_JOINT_LIMITS, 3, 0.04))


def test_bootstrap_limit_recovery_accepts_two_declared_inward_joints():
    start = np.zeros(6)
    start[1] = URDF_JOINT_LIMITS[1, 0] - 0.001
    start[2] = URDF_JOINT_LIMITS[2, 1] + 0.035
    endpoint = start.copy()
    endpoint[1] = URDF_JOINT_LIMITS[1, 0]
    endpoint[2] = URDF_JOINT_LIMITS[2, 1] - 0.03
    path = [start, 0.5 * (start + endpoint), endpoint]

    assert bootstrap_start_limit_recovery_reasons(
        path, URDF_JOINT_LIMITS, [2, 3], 0.04) == []
    assert bootstrap_recovery_declaration_reasons(
        path, 2, [2, 3],
        [endpoint[1] - start[1], endpoint[2] - start[2]]) == []

    undeclared = [item.copy() for item in path]
    undeclared[1][0] = 0.01
    assert any(
        'non-declared joint' in reason
        for reason in bootstrap_recovery_declaration_reasons(
            undeclared, 2, [2, 3],
            [endpoint[1] - start[1], endpoint[2] - start[2]]))


def test_scan_handoff_accepts_the_same_fresh_measured_lock_as_acquisition():
    target = SimpleNamespace(
        header=SimpleNamespace(frame_id='base_link'),
        valid=True,
        stable=True,
        position=SimpleNamespace(x=0.38, y=-0.12, z=0.0),
    )
    health = SimpleNamespace(
        lifecycle_state='TRACKING',
        prediction_only=False,
        camera_settled=True,
        measurement_age_sec=0.10,
    )
    assert measured_target_lock_rejection(
        target, health, 'LOCKED',
        10.0, 10.0, 10.0, 10.1, 1.0, 0.75,
    ) == ''
    assert 'prediction-only' in measured_target_lock_rejection(
        target,
        SimpleNamespace(
            lifecycle_state='TRACKING',
            prediction_only=True,
            camera_settled=True,
            measurement_age_sec=0.10),
        'LOCKED', 10.0, 10.0, 10.0, 10.1, 1.0, 0.75,
    )


def test_sdk_movej_target_path_is_accepted_without_geometry_changes():
    positions = np.asarray([
        np.zeros(6),
        [0.01, 0.0, -0.01, 0.0, 0.0, 0.01],
        [0.02, 0.0, -0.02, 0.0, 0.0, 0.02],
    ])
    velocities = np.zeros_like(positions)
    accelerations = np.zeros_like(positions)
    times = np.asarray([0.0, 0.05, 0.10])
    q, qd, qdd, validated_times = validate_sdk_movej_waypoint_path(
        positions, velocities, accelerations, times,
        command_rate_hz=20.0,
    )
    np.testing.assert_allclose(q, positions)
    np.testing.assert_allclose(qd, velocities)
    np.testing.assert_allclose(qdd, accelerations)
    np.testing.assert_allclose(validated_times, times)
    assert TIMING_POLICY_VERSION == 'tesseract_stream_v3'


def test_tesseract_point_requires_complete_finite_derivatives_and_time():
    values = np.arange(6, dtype=float)
    q, qd, qdd, when = validate_tesseract_point(
        values, values * 0.1, values * 0.01, 0.5, previous_time_s=0.4)
    np.testing.assert_allclose(q, values)
    np.testing.assert_allclose(qd, values * 0.1)
    np.testing.assert_allclose(qdd, values * 0.01)
    assert when == pytest.approx(0.5)

    with pytest.raises(ValueError, match='velocities'):
        validate_tesseract_point(values, [], values, 0.5)
    with pytest.raises(ValueError, match='accelerations'):
        validate_tesseract_point(values, values, [math.nan] * 6, 0.5)
    with pytest.raises(ValueError, match='strictly increasing'):
        validate_tesseract_point(values, values, values, 0.4, previous_time_s=0.4)


def test_tesseract_stream_rejects_rate_step_and_derivative_claims():
    positions = np.asarray([np.zeros(6), np.full(6, 0.03)])
    derivatives = np.asarray([np.zeros(6), np.zeros(6)])
    with pytest.raises(ValueError, match='faster'):
        validate_sdk_movej_waypoint_path(
            positions, derivatives, derivatives, [0.0, 0.005],
            command_rate_hz=100.0,
        )
    with pytest.raises(ValueError, match='joint step'):
        validate_sdk_movej_waypoint_path(
            np.vstack([
                positions, np.full(6, 0.04), np.full(6, 0.05),
                np.full(6, 0.06)]),
            np.zeros((5, 6)), np.zeros((5, 6)),
            [0.0, 0.05, 0.10, 0.15, 0.20],
            command_rate_hz=100.0,
            maximum_step_rad=0.02,
        )
    excessive_velocity = derivatives.copy()
    excessive_velocity[1, 5] = 1.1
    with pytest.raises(ValueError, match='derivatives must be zero'):
        validate_sdk_movej_waypoint_path(
            positions * 0.5, excessive_velocity, derivatives, [0.0, 0.05],
            command_rate_hz=100.0,
        )


def test_tesseract_stream_rejects_speed_scaled_controller_limit_violation():
    positions = np.asarray([
        np.zeros(6),
        [0.13, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    derivatives = np.zeros_like(positions)
    with pytest.raises(ValueError, match='velocity limit'):
        validate_sdk_movej_waypoint_path(
            positions,
            derivatives,
            derivatives,
            [0.0, 0.05],
            command_rate_hz=20.0,
            maximum_step_rad=0.20,
            velocity_limits_rad_s=[3.0] * 6,
            acceleration_limits_rad_s2=[5.0] * 6,
            speed_percent=50.0,
        )


def test_saved_invalid_j6_is_ignored_and_urdf_limit_remains(tmp_path):
    path = tmp_path / 'bounds.json'
    path.write_text(
        '{"joints":{"joint1":{"min":-1,"max":1,"valid":true},'
        '"joint6":{"min":-9,"max":9,"valid":false}}}'
    )
    limits, ignored = load_conservative_joint_limits(str(path))
    np.testing.assert_allclose(limits[0], [-1.0, 1.0])
    np.testing.assert_allclose(limits[5], [-math.pi, math.pi])
    assert 'joint6' in ignored


def test_planning_model_accepts_existing_validated_joint2_zero_range():
    assert URDF_JOINT_LIMITS[1, 0] == -0.044796192
    limits, ignored = load_conservative_joint_limits(
        str(Path(__file__).resolve().parents[4] / 'piper_joint_bounds.json'))
    assert 'joint2' not in ignored
    assert limits[1, 0] == -0.044796192
    assert limits[1, 0] <= -0.033632032 <= limits[1, 1]


def test_expanded_obstacle_intersection():
    box = CollisionBox('test', np.asarray([0.4, -0.05, 0.1]), np.asarray([0.5, 0.05, 0.2]))
    assert segment_intersects_expanded_box(
        np.asarray([0.0, 0.0, 0.15]), np.asarray([0.6, 0.0, 0.15]), box, 0.02
    )
    assert not segment_intersects_expanded_box(
        np.asarray([0.0, 0.2, 0.15]), np.asarray([0.6, 0.2, 0.15]), box, 0.02
    )


def test_path_rejects_obstacle_around_camera_segment():
    kinematics = PiperScanKinematics(LINK6_FROM_CAMERA)
    joints = np.asarray([0.0, 1.2, -1.4, 0.1, 0.2, 0.0])
    points = kinematics.collision_points(joints)
    center = 0.5 * (points[-2] + points[-1])
    box = CollisionBox('camera obstacle', center - 0.01, center + 0.01)
    reasons = validate_joint_path(
        kinematics,
        [joints],
        np.asarray([
            [-2.618, 2.168], [0.0, 3.14], [-2.967, 0.0],
            [-1.745, 1.745], [-1.22, 1.22], [-math.pi, math.pi],
        ]),
        obstacle_boxes=[box],
        floor_z_m=-0.20,
        self_clearance_m=0.001,
    )
    assert any('camera obstacle' in reason for reason in reasons)


def test_nominal_configuration_has_no_floor_collision():
    kinematics = PiperScanKinematics(LINK6_FROM_CAMERA)
    joints = np.asarray([0.0, 1.2, -1.4, 0.1, 0.2, 0.0])
    reasons = configuration_collision_reasons(
        kinematics,
        joints,
        floor_z_m=-0.02,
        self_clearance_m=0.001,
    )
    assert reasons == []


def test_camera_holder_envelope_rejects_floor_grazing_path():
    class TranslatingLink6:
        @staticmethod
        def chain_transforms(joints):
            transform = np.eye(4)
            transform[2, 3] = float(joints[0])
            return [transform] * 6

    path = [np.asarray([0.004, 0, 0, 0, 0, 0], dtype=float)]
    reasons = validate_attached_box_external_clearance_path(
        TranslatingLink6(),
        path,
        origin_link6_m=[0.0, 0.0, 0.0],
        size_m=[0.10, 0.10, 0.004],
        floor_z_m=0.0,
        clearance_m=0.005,
        label='camera holder/L515',
    )

    assert reasons == [
        'trajectory step 0: camera holder/L515 envelope floor clearance '
        '0.002000m is below 0.005000m']


def test_camera_holder_envelope_accepts_floor_clear_path():
    class TranslatingLink6:
        @staticmethod
        def chain_transforms(joints):
            transform = np.eye(4)
            transform[2, 3] = float(joints[0])
            return [transform] * 6

    path = [np.asarray([height, 0, 0, 0, 0, 0], dtype=float)
            for height in (0.020, 0.030, 0.040)]

    assert validate_attached_box_external_clearance_path(
        TranslatingLink6(),
        path,
        origin_link6_m=[0.0, 0.0, 0.0],
        size_m=[0.10, 0.10, 0.004],
        floor_z_m=0.0,
        clearance_m=0.005,
        label='camera holder/L515',
    ) == []


def test_august_11_incident_home_start_is_rejected_by_holder_floor_envelope():
    root = Path(__file__).resolve().parents[4]
    kinematics = PiperScanKinematics(load_accepted_hand_eye(
        str(root / 'L515_camera/calibration/hand_eye/'
            'session_20260808_straight_mount/calibration_result.yaml')))
    incident_start = np.asarray([
        -1.389240160, 2.151002196, -0.657167812,
        1.080463916, 1.230063660, -2.974289220,
    ])
    rough_home = np.asarray([0.0, 0.0, 0.0, 0.0, 0.399345492, 0.0])
    path = [incident_start]
    path.extend(interpolate_joint_path(incident_start, rough_home, 0.025))

    reasons = validate_attached_box_external_clearance_path(
        kinematics,
        path,
        origin_link6_m=[-0.029750002, 0.0, 0.0375],
        size_m=[0.1395, 0.10572671, 0.053],
        floor_z_m=0.0,
        clearance_m=0.005,
        label='camera holder/L515',
    )

    assert reasons
    assert 'trajectory step 0' in reasons[0]
    assert 'floor clearance 0.001245m is below 0.005000m' in reasons[0]


def test_deployed_home_stages_clear_raised_virtual_floor():
    """The contact buffer must not make the recorded safe home unusable."""
    root = Path(__file__).resolve().parents[4]
    kinematics = PiperScanKinematics(load_accepted_hand_eye(
        str(root / 'L515_camera/calibration/hand_eye/'
            'session_20260808_straight_mount/calibration_result.yaml')))
    with (root / 'piper_home_pose.json').open(encoding='utf-8') as stream:
        home = json.load(stream)
    pre_home = np.asarray(home['pre_home_positions_rad'], dtype=float)
    rough_home = np.asarray(home['positions_rad'], dtype=float)
    storage = rough_home.copy()
    storage[5] = float(home['storage_joint6_rad'])
    path = [pre_home]
    path.extend(interpolate_joint_path(pre_home, rough_home, 0.025))
    path.extend(interpolate_joint_path(rough_home, storage, 0.025))

    assert validate_attached_box_external_clearance_path(
        kinematics,
        path,
        origin_link6_m=[-0.029750002, 0.0, 0.0375],
        size_m=[0.1395, 0.10572671, 0.053],
        floor_z_m=0.005,
        clearance_m=0.005,
        label='camera holder/L515',
    ) == []


def test_folded_start_escape_must_monotonically_reach_normal_proxy_clearance():
    kinematics = PiperScanKinematics(LINK6_FROM_CAMERA)
    start = np.asarray([
        -0.010100076, -0.033632032, -0.014356412,
        0.04517996, 0.533315412, -0.052018008,
    ])
    path = [
        start + np.asarray([0.0, 0.0, -step, 0.0, 0.0, 0.0])
        for step in np.linspace(0.0, 0.05, 51)
    ]
    start_clearance, _ = minimum_self_segment_clearance(kinematics, path[0])
    end_clearance, _ = minimum_self_segment_clearance(kinematics, path[-1])
    assert start_clearance < 0.060
    assert end_clearance >= 0.060
    assert validate_monotonic_self_clearance_escape(
        kinematics,
        path,
        URDF_JOINT_LIMITS,
        floor_z_m=0.0,
        link_radius_m=0.025,
        self_clearance_m=0.060,
    ) == []


def test_folded_start_escape_rejects_worsening_proxy_clearance():
    kinematics = PiperScanKinematics(LINK6_FROM_CAMERA)
    start = np.asarray([
        -0.010100076, -0.033632032, -0.014356412,
        0.04517996, 0.533315412, -0.052018008,
    ])
    path = [
        start,
        start + np.asarray([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]),
    ]
    reasons = validate_monotonic_self_clearance_escape(
        kinematics,
        path,
        URDF_JOINT_LIMITS,
        floor_z_m=0.0,
        link_radius_m=0.025,
        self_clearance_m=0.060,
        monotonic_tolerance_m=0.0,
    )
    assert any('worsens proxy self-clearance' in reason for reason in reasons)


def test_bootstrap_recovery_declaration_checks_every_prefix_sample():
    start = np.zeros(6)
    valid = [
        start,
        np.asarray([0.0, 0.0, -0.01, 0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, -0.02, 0.0, 0.0, 0.0]),
    ]
    assert bootstrap_recovery_declaration_reasons(
        valid, 2, 3, -0.02) == []
    hidden_other_joint = [
        start,
        np.asarray([0.01, 0.0, -0.01, 0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, -0.02, 0.0, 0.0, 0.0]),
    ]
    assert any(
        'non-declared joint' in reason
        for reason in bootstrap_recovery_declaration_reasons(
            hidden_other_joint, 2, 3, -0.02))


def test_bootstrap_recovery_declaration_rejects_reversal_and_oversize():
    reversal = [
        np.zeros(6),
        np.asarray([0.0, 0.0, -0.02, 0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, -0.01, 0.0, 0.0, 0.0]),
    ]
    assert any(
        'reverses' in reason
        for reason in bootstrap_recovery_declaration_reasons(
            reversal, 2, 3, -0.01))
    assert any(
        'bounded range' in reason
        for reason in bootstrap_recovery_declaration_reasons(
            [np.zeros(6), np.asarray([0.0, 0.0, -0.16, 0.0, 0.0, 0.0])],
            1, 3, -0.16))


def test_negative_camera_pitch_places_camera_above_target():
    target = np.asarray([0.4, 0.0, 0.05])
    position, look = orbit_camera_view(target, 0.0, 0.45, -10.0)
    assert position[2] > target[2]
    distance = float(np.linalg.norm(position - target))
    assert math.isclose(distance, 0.45, abs_tol=1e-9)
    assert look[2] < 0.0


def test_approval_gate_requires_exact_fresh_opt_in_plan():
    args = dict(
        state='PROPOSAL_READY',
        current_plan_id='abc123',
        requested_plan_id='abc123',
        confirmation='EXECUTE APPROVED SCAN',
        expected_confirmation='EXECUTE APPROVED SCAN',
        real_motion_enabled=True,
        plan_age_sec=2.0,
        plan_max_age_sec=30.0,
    )
    assert approval_rejection_reason(**args) == ''
    assert 'plan_id' in approval_rejection_reason(
        **dict(args, requested_plan_id='wrong'))
    assert 'confirmation' in approval_rejection_reason(
        **dict(args, confirmation='yes'))
    assert 'proposal-only' in approval_rejection_reason(
        **dict(args, real_motion_enabled=False))
    assert 'expired' in approval_rejection_reason(
        **dict(args, plan_age_sec=31.0))
    hash_args = dict(
        args,
        current_trajectory_sha256='a' * 64,
        requested_trajectory_sha256='a' * 64,
        require_trajectory_hash=True,
    )
    assert approval_rejection_reason(**hash_args) == ''
    assert 'trajectory_sha256' in approval_rejection_reason(
        **dict(hash_args, requested_trajectory_sha256='b' * 64))


def test_acquisition_plan_count_is_bounded_without_scan_minimum():
    assert plan_count_rejection(ROUGH_ACQUISITION, 1, 5, 5) == ''
    assert plan_count_rejection(ROUGH_ACQUISITION, 5, 5, 5) == ''
    assert 'bounded' in plan_count_rejection(ROUGH_ACQUISITION, 6, 5, 5)
    assert plan_count_rejection(
        MULTIVIEW_SCAN, 1, 5, 5,
        closed_loop_one_view=True) == ''
    assert 'exactly one' in plan_count_rejection(
        MULTIVIEW_SCAN, 2, 5, 5,
        session_accepted_views=3, session_maximum_views=5,
        closed_loop_one_view=True)
    assert plan_count_rejection(
        MULTIVIEW_SCAN, 2, 5, 5,
        session_accepted_views=3, session_maximum_views=5) == ''
    assert 'unsupported' in plan_count_rejection('UNKNOWN', 1, 5, 5)


def test_acquisition_uses_configured_speed_with_sdk_range():
    assert commanded_speed_percent(100.0, ROUGH_ACQUISITION, 0.0) == 100.0
    assert commanded_speed_percent(120.0, ROUGH_ACQUISITION, 0.0) == 100.0
    assert commanded_speed_percent(4.0, ROUGH_ACQUISITION, 0.0) == 4.0
    assert commanded_speed_percent(100.0, MULTIVIEW_SCAN, 0.5) == 100.0


def test_planned_speed_is_bound_only_by_the_selected_sdk_percentage():
    assert planned_speed_rejection(5.0, MULTIVIEW_SCAN, 1.0, 4.0) == ''
    assert planned_speed_rejection(5.0, MULTIVIEW_SCAN, 0.1, 5.0) == ''
    assert 'configured limit' in planned_speed_rejection(
        5.0, MULTIVIEW_SCAN, 1.0, 5.1)
    assert planned_speed_rejection(
        5.0, ROUGH_ACQUISITION, 0.0, 5.0) == ''
    assert 'acquisition speed' in planned_speed_rejection(
        5.0, ROUGH_ACQUISITION, 1.0, 4.0)
    assert planned_speed_rejection(5.0, RETURN_HOME, 0.0, 5.0) == ''
    assert 'return-home speed' in planned_speed_rejection(
        5.0, RETURN_HOME, 0.0, 4.0)


def test_closed_loop_one_view_requires_no_unused_home_trajectory():
    assert trajectory_count_rejection(
        MULTIVIEW_SCAN, 1, 1, True) == ''
    assert 'only its one capture trajectory' in trajectory_count_rejection(
        MULTIVIEW_SCAN, 2, 1, True)
    assert trajectory_count_rejection(
        MULTIVIEW_SCAN, 2, 1, False) == ''
    assert 'final return-home' in trajectory_count_rejection(
        MULTIVIEW_SCAN, 1, 1, False)
    assert trajectory_count_rejection(RETURN_HOME, 1, 0, True) == ''


def test_runtime_refresh_waits_for_fresh_data_before_motion():
    reasons = ['joints data missing or stale']
    assert runtime_refresh_action(reasons, 0.1, 3.0) == 'wait'
    assert runtime_refresh_action([], 0.2, 3.0) == 'start'
    assert runtime_refresh_action(reasons, 3.0, 3.0) == 'abort'


def test_runtime_recovery_preserves_first_look_bootstrap_static_scene():
    events = []
    executor = SimpleNamespace(
        state='MOVING',
        plan_kind=ROUGH_ACQUISITION,
        current_view=0,
        abort_return_bootstrap_static_scene=False,
        publish_hold=lambda: events.append('hold'),
        settle_position_anchor=np.ones(6),
        settle_last_joint_update=1.0,
        settle_last_sample_ok=True,
        is_return_home=lambda: False,
        set_state=lambda state, reason: events.append((state, reason)),
    )

    ScanViewpointExecutorNode.begin_runtime_recovery(
        executor, ['motion_limits data missing or stale'])

    assert executor.runtime_refresh_resume_state == 'MOVING'
    assert executor.runtime_refresh_allow_missing_obstacles
    assert events[0] == 'hold'
    assert events[1][0] == 'WAITING_FOR_RUNTIME_REFRESH'


def test_dedicated_home_holds_for_transient_motion_limit_refresh():
    events = []
    executor = SimpleNamespace(
        state='MOVING',
        plan_kind=RETURN_HOME,
        current_view=0,
        plan_collision_model_qualified=True,
        latest_obstacles=None,
        acquisition_scene_snapshot_validated=False,
        abort_return_bootstrap_static_scene=False,
        is_acquisition=lambda: False,
        is_return_home=lambda: True,
        is_startup_home_static=lambda: False,
        returning_home=lambda: True,
        runtime_reasons=lambda *_args, **_kwargs: [
            'motion_limits data missing or stale'],
        begin_runtime_recovery=lambda reasons: events.append(
            ('recover', reasons)),
        _terminal_abort=lambda reason: events.append(('fail', reason)),
    )

    ScanViewpointExecutorNode.execution_tick(executor)

    assert events == [(
        'recover', ['motion_limits data missing or stale'])]


def test_dedicated_home_still_aborts_nontransient_control_fault():
    events = []
    executor = SimpleNamespace(
        state='MOVING',
        plan_kind=RETURN_HOME,
        current_view=0,
        plan_collision_model_qualified=True,
        latest_obstacles=None,
        acquisition_scene_snapshot_validated=False,
        abort_return_bootstrap_static_scene=False,
        is_acquisition=lambda: False,
        is_return_home=lambda: True,
        is_startup_home_static=lambda: False,
        returning_home=lambda: True,
        runtime_reasons=lambda *_args, **_kwargs: ['arm controller fault'],
        begin_runtime_recovery=lambda reasons: events.append(
            ('recover', reasons)),
        _terminal_abort=lambda reason: events.append(('fail', reason)),
    )

    ScanViewpointExecutorNode.execution_tick(executor)

    assert len(events) == 1
    assert events[0][0] == 'fail'
    assert 'arm controller fault' in events[0][1]


def test_runtime_recovery_does_not_hide_multiview_motion_obstacles():
    executor = SimpleNamespace(
        state='MOVING',
        plan_kind=MULTIVIEW_SCAN,
        current_view=0,
        abort_return_bootstrap_static_scene=False,
        publish_hold=lambda: True,
        settle_position_anchor=None,
        settle_last_joint_update=-1e9,
        settle_last_sample_ok=False,
        is_return_home=lambda: False,
        set_state=lambda *_args: None,
    )

    ScanViewpointExecutorNode.begin_runtime_recovery(
        executor, ['motion_limits data missing or stale'])

    assert not executor.runtime_refresh_allow_missing_obstacles


def test_rgbd_capture_handoff_publishes_authorization_before_one_request():
    assert rgbd_capture_handoff_action(False, 0.0, 0.25) == \
        'publish_authorization'
    assert rgbd_capture_handoff_action(False, 0.249, 0.25) == \
        'publish_authorization'
    assert rgbd_capture_handoff_action(False, 0.25, 0.25) == \
        'request_capture'
    assert rgbd_capture_handoff_action(True, 10.0, 0.25) == \
        'wait_response'


def test_multiview_obstacle_gap_waits_only_while_arm_is_stationary():
    for state in (
            'SETTLING', 'CAPTURING', 'CAPTURING_RGBD', 'WAIT_CAPTURE',
            'WAITING_FOR_CAPTURE_REFRESH'):
        assert missing_obstacles_can_wait('MULTIVIEW_SCAN', 2, state)
    assert not missing_obstacles_can_wait('MULTIVIEW_SCAN', 2, 'MOVING')
    assert not missing_obstacles_can_wait(
        'MULTIVIEW_SCAN', 2, 'WAITING_FOR_RUNTIME_REFRESH')
    assert missing_obstacles_can_wait(
        ROUGH_ACQUISITION, 3, 'MOVING',
        bootstrap_abort_retrace=True)


def test_capture_heavy_refresh_waits_for_idle_and_retries_only_once():
    events = []
    executor = SimpleNamespace(
        state='WAITING_FOR_CAPTURE_REFRESH',
        capture_heavy_refresh_request_id='scan-view-2-refresh',
        capture_heavy_refresh_min_image_stamp_ns=10_000_000_000,
        capture_heavy_refresh_publish_attempts=1,
        capture_heavy_refresh_waiting_for_worker=False,
        capture_rejection_reason='target depth invalid',
        set_state=lambda state, reason: events.append(
            ('state', state, reason)),
        publish_capture_heavy_refresh=lambda: (
            setattr(
                executor, 'capture_heavy_refresh_publish_attempts',
                executor.capture_heavy_refresh_publish_attempts + 1),
            events.append(('publish',))),
        abort_motion=lambda reason: events.append(('abort', reason)),
        reject_achieved_capture_view=lambda reason: events.append(
            ('reject', reason)),
    )

    busy = SimpleNamespace(data=json.dumps({
        'state': 'request_ignored_busy',
        'request_id': executor.capture_heavy_refresh_request_id,
    }))
    idle = SimpleNamespace(data=json.dumps({'state': 'idle'}))

    ScanViewpointExecutorNode.heavy_refresh_status_cb(executor, busy)

    assert executor.capture_heavy_refresh_waiting_for_worker
    assert events[0][0] == 'state'
    assert not any(event[0] == 'abort' for event in events)

    ScanViewpointExecutorNode.heavy_refresh_status_cb(executor, idle)

    assert not executor.capture_heavy_refresh_waiting_for_worker
    assert executor.capture_heavy_refresh_publish_attempts == 2
    assert events[-1] == ('publish',)

    ScanViewpointExecutorNode.heavy_refresh_status_cb(executor, busy)

    assert events[-1][0] == 'abort'
    assert 'one bounded retry' in events[-1][1]


def test_successful_capture_refresh_starts_a_new_readiness_retry_epoch():
    events = []
    executor = SimpleNamespace(
        state='WAITING_FOR_CAPTURE_REFRESH',
        capture_heavy_refresh_request_id='scan-view-3-refresh',
        capture_heavy_refresh_min_image_stamp_ns=10_000_000_000,
        capture_heavy_refresh_publish_attempts=1,
        capture_heavy_refresh_waiting_for_worker=False,
        capture_rejection_reason='target depth invalid',
        rgbd_capture_future=object(),
        rgbd_capture_attempts=MAX_RGBD_CAPTURE_READINESS_RETRIES,
        set_state=lambda state, reason: events.append(
            ('state', state, reason)),
        abort_motion=lambda reason: events.append(('abort', reason)),
        reject_achieved_capture_view=lambda reason: events.append(
            ('reject', reason)),
    )
    detected = SimpleNamespace(data=json.dumps({
        'state': 'published',
        'request_id': executor.capture_heavy_refresh_request_id,
        'image_stamp': {'sec': 11, 'nanosec': 0},
    }))

    ScanViewpointExecutorNode.heavy_refresh_status_cb(executor, detected)

    assert executor.rgbd_capture_future is None
    assert executor.rgbd_capture_attempts == 0
    assert executor.capture_heavy_refresh_request_id == 'scan-view-3-refresh'
    assert events[0][0:2] == ('state', 'CAPTURING_RGBD')
    assert not any(event[0] in ('abort', 'reject') for event in events)


def test_correlated_acquisition_scene_may_age_during_its_exact_segment():
    health = SimpleNamespace(
        lifecycle_state='TRACKING',
        camera_settled=True,
        prediction_only=False,
        measurement_age_sec=0.1,
        recommended_speed_scale=1.0,
    )
    fake = executor_runtime_fixture(health)
    fake.fresh = lambda key, *_args: key != 'obstacles'

    stale = ScanViewpointExecutorNode.runtime_reasons(
        fake, runtime_gate_policy(
            SafetyMode.ACQUISITION_MOTION,
            obstacle_authority=ObstacleAuthority.LIVE))
    bound_to_segment = ScanViewpointExecutorNode.runtime_reasons(
        fake, runtime_gate_policy(
            SafetyMode.ACQUISITION_MOTION,
            obstacle_authority=ObstacleAuthority.APPROVED_SNAPSHOT))

    assert 'obstacles data missing or stale' in stale
    assert bound_to_segment == []


def test_correlated_acquisition_scene_does_not_hide_a_blocker():
    health = SimpleNamespace(
        lifecycle_state='TRACKING',
        camera_settled=True,
        prediction_only=False,
        measurement_age_sec=0.1,
        recommended_speed_scale=1.0,
    )
    fake = executor_runtime_fixture(health)
    fake.fresh = lambda key, *_args: key != 'obstacles'
    fake.latest_obstacles = SimpleNamespace(
        scene_blocked=True,
        instances=[],
        blocking_reason='hand',
    )

    reasons = ScanViewpointExecutorNode.runtime_reasons(
        fake, runtime_gate_policy(
            SafetyMode.ACQUISITION_MOTION,
            obstacle_authority=ObstacleAuthority.APPROVED_SNAPSHOT))

    assert 'scene_blocked: hand' in reasons


def executor_runtime_fixture(health):
    return SimpleNamespace(
        fresh=lambda *args: True,
        get_parameter=lambda name: SimpleNamespace(value={
            'motion_limits_timeout_sec': 1.0,
            'joint_feedback_limit_tolerance_rad': 0.001,
            'configured_home_feedback_limit_tolerance_rad': 0.3,
            'max_tracking_measurement_age_sec': 1.0,
            'min_tracking_speed_scale': 0.2,
            'speed_percent': 5.0,
        }[name]),
        latest_motion_limits=SimpleNamespace(
            valid=True, limits_sha256='limits'),
        runtime_motion_limits_sha256='limits',
        current_joints=lambda: np.zeros(6),
        joint_limits=np.asarray([[-2.0, 2.0]] * 6),
        arm_status_reasons=lambda: [],
        latest_camera_timestamp_health=SimpleNamespace(
            healthy=True, state='HEALTHY', reason=''),
        latest_obstacles=SimpleNamespace(
            scene_blocked=False, instances=[], blocking_reason=''),
        latest_tracking_health=health,
        latest_target_status='LOCKED',
        plan_kind=MULTIVIEW_SCAN,
        is_configured_home_direct=lambda: False,
        plan_execution_speed_percent=5.0,
        param_bool=lambda name: False,
        workflow_ready=lambda: True,
    )


def test_authoritative_direct_home_gate_uses_only_live_control_evidence():
    fake = executor_runtime_fixture(SimpleNamespace(
        lifecycle_state='LOST',
        camera_settled=False,
        prediction_only=True,
        measurement_age_sec=math.inf,
        recommended_speed_scale=0.0,
    ))
    fake.fresh = lambda key, *_args: key in ('joints', 'arm_status')
    fake.plan_kind = RETURN_HOME
    fake.is_configured_home_direct = lambda: True

    reasons = ScanViewpointExecutorNode.runtime_reasons(
        fake, runtime_gate_policy(SafetyMode.RETURN_HOME))

    assert reasons == []


def test_tracking_scale_does_not_override_the_selected_sdk_speed():
    fake = executor_runtime_fixture(SimpleNamespace(
        lifecycle_state='TRACKING',
        camera_settled=False,
        prediction_only=False,
        measurement_age_sec=0.1,
        recommended_speed_scale=0.8,
    ))

    before_target = ScanViewpointExecutorNode.runtime_reasons(
        fake, runtime_gate_policy(SafetyMode.SCAN_APPROVAL))
    in_flight = ScanViewpointExecutorNode.runtime_reasons(
        fake, runtime_gate_policy(
            SafetyMode.SCAN_MOTION,
            obstacle_authority=ObstacleAuthority.APPROVED_SNAPSHOT))

    assert not any('tracking speed allowance' in reason for reason in before_target)
    assert not any('tracking speed allowance' in reason for reason in in_flight)


def test_tracking_gates_remain_at_scan_approval():
    fake = executor_runtime_fixture(SimpleNamespace(
        lifecycle_state='TRACKING',
        camera_settled=False,
        prediction_only=False,
        measurement_age_sec=2.0,
        recommended_speed_scale=0.8,
    ))

    reasons = ScanViewpointExecutorNode.runtime_reasons(
        fake, runtime_gate_policy(SafetyMode.SCAN_APPROVAL))

    assert 'tracking measurement is stale' in reasons


def test_low_confidence_target_status_waits_until_post_move_settling():
    fake = executor_runtime_fixture(SimpleNamespace(
        lifecycle_state='TRACKING',
        camera_settled=False,
        prediction_only=False,
        measurement_age_sec=0.1,
        recommended_speed_scale=1.0,
    ))
    fake.latest_target_status = 'LOW_CONFIDENCE'

    before_target = ScanViewpointExecutorNode.runtime_reasons(
        fake, runtime_gate_policy(SafetyMode.SCAN_APPROVAL))
    in_flight = ScanViewpointExecutorNode.runtime_reasons(
        fake, runtime_gate_policy(
            SafetyMode.SCAN_MOTION,
            obstacle_authority=ObstacleAuthority.APPROVED_SNAPSHOT))

    assert 'target_status=LOW_CONFIDENCE' in before_target
    assert 'target_status=LOW_CONFIDENCE' not in in_flight


def test_fresh_reacquisition_state_may_finish_only_the_issued_target():
    fake = executor_runtime_fixture(SimpleNamespace(
        lifecycle_state='WAITING_TO_REACQUIRE',
        camera_settled=False,
        prediction_only=True,
        measurement_age_sec=10.0,
        recommended_speed_scale=0.0,
    ))
    fake.latest_target_status = 'LOW_CONFIDENCE'

    before_target = ScanViewpointExecutorNode.runtime_reasons(
        fake, runtime_gate_policy(SafetyMode.SCAN_APPROVAL))
    in_flight = ScanViewpointExecutorNode.runtime_reasons(
        fake, runtime_gate_policy(
            SafetyMode.SCAN_MOTION,
            obstacle_authority=ObstacleAuthority.APPROVED_SNAPSHOT))

    assert any('tracking lifecycle' in reason for reason in before_target)
    assert any('tracking measurement is stale' in reason for reason in before_target)
    assert in_flight == []


def test_missing_tracking_telemetry_does_not_abort_an_issued_target():
    fake = executor_runtime_fixture(SimpleNamespace(
        lifecycle_state='WAITING_TO_REACQUIRE',
        camera_settled=False,
        prediction_only=True,
        measurement_age_sec=10.0,
        recommended_speed_scale=0.0,
    ))
    fake.fresh = lambda key, *args: key != 'tracking'

    reasons = ScanViewpointExecutorNode.runtime_reasons(
        fake, runtime_gate_policy(
            SafetyMode.SCAN_MOTION,
            obstacle_authority=ObstacleAuthority.APPROVED_SNAPSHOT))

    assert reasons == []


def test_sdk_movej_waypoint_progress_is_feedback_gated_and_bounded():
    assert waypoint_motion_action(
        0.011, 0.012, 1.0, 20.0, 1.0, 5.0) == 'advance'
    assert waypoint_motion_action(
        0.020, 0.012, 1.0, 20.0, 1.0, 5.0) == 'wait'
    assert waypoint_motion_action(
        0.020, 0.012, 20.1, 20.0, 1.0, 5.0) == 'abort_timeout'
    assert waypoint_motion_action(
        0.020, 0.012, 4.0, 20.0, 5.1, 5.0) == 'abort_stalled'
    assert waypoint_motion_action(
        math.nan, 0.012, 0.0, 20.0, 0.0, 5.0) == 'abort_invalid'


def test_sdk_movej_progress_counts_motion_by_any_joint():
    target = np.asarray([0.0, 0.0, -0.10, 0.0, 0.0, 0.0])
    start = np.asarray([0.0, 0.0, 0.0, 0.0, 0.20, 0.0])
    progressed = np.asarray([0.0, 0.0, -0.02, 0.0, 0.20, 0.0])

    # The maximum error remains J5's 0.20 rad in both samples, but J3 has
    # genuinely progressed 0.02 rad toward the approved endpoint.
    assert np.max(np.abs(start - target)) == pytest.approx(0.20)
    assert np.max(np.abs(progressed - target)) == pytest.approx(0.20)
    assert joint_progress_error(progressed, target) == pytest.approx(
        joint_progress_error(start, target) - 0.02)


def test_executor_sdk_movej_command_is_arm_only_and_carries_aggregate_speed():
    published = []
    stamp = Time(sec=1, nanosec=2)
    fake = SimpleNamespace(
        command_pub=SimpleNamespace(publish=published.append),
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: stamp)),
        execution_speed_percent=lambda: 5.0,
    )

    ScanViewpointExecutorNode.publish_joint_command(
        fake, np.asarray([0.1, 0.2, -0.3, 0.4, -0.5, 0.6]))

    assert len(published) == 1
    command = published[0]
    assert command.header.frame_id == 'piper_scan_executor_sdk_movej'
    assert len(command.position) == 6
    assert command.position == pytest.approx([0.1, 0.2, -0.3, 0.4, -0.5, 0.6])
    assert command.velocity == pytest.approx([0.0] * 6 + [5.0])


def test_executor_sends_each_sdk_movej_endpoint_only_once():
    published = []
    parameters = {
        'waypoint_progress_epsilon_rad': 0.001,
        'waypoint_reached_tolerance_rad': 0.012,
        'waypoint_timeout_sec': 90.0,
        'waypoint_progress_timeout_sec': 5.0,
    }
    target = np.asarray([0.1, 0.2, -0.3, 0.4, -0.5, 0.6])
    fake = SimpleNamespace(
        command_target=target,
        now=lambda: 1.11,
        max_joint_error=lambda _target: 0.5,
        total_joint_error=lambda _target: 0.5,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        waypoint_best_error=0.5,
        waypoint_last_progress_at=1.0,
        waypoint_started_at=1.0,
        current_waypoint_error=0.5,
        max_waypoint_error=0.5,
        command_sent_at=1.0,
        command_samples_sent=1,
        max_command_interval_sec=0.0,
        publish_joint_command=lambda value: published.append(
            np.asarray(value).copy()),
        last_motion_status_at=1.11,
        publish_status=lambda: None,
        abort_motion=lambda reason: pytest.fail(reason),
    )

    ScanViewpointExecutorNode.moving_tick(fake)

    assert published == []
    assert fake.command_samples_sent == 1
    assert fake.command_sent_at == pytest.approx(1.0)
    assert fake.waypoint_started_at == pytest.approx(1.0)
    assert fake.waypoint_last_progress_at == pytest.approx(1.0)


def test_executor_streams_due_tesseract_samples_without_waiting_at_each_one():
    published = []
    first = np.full(6, 0.01)
    second = np.full(6, 0.02)
    parameters = {
        'trajectory_following_error_grace_sec': 1.0,
        'trajectory_following_error_rad': 0.30,
    }
    fake = SimpleNamespace(
        motion_started_at=0.0,
        command_target=None,
        path_index=0,
        current_path=[first, second],
        current_path_times=[0.05, 0.10],
        current_waypoint_error=0.0,
        max_waypoint_error=0.0,
        dropped_command_samples=0,
        command_sent_at=0.0,
        command_samples_sent=0,
        max_command_interval_sec=0.0,
        stream_schedule_completion_logged=False,
        last_stream_planned_duration_sec=0.0,
        last_stream_actual_duration_sec=0.0,
        last_stream_achieved_rate_hz=0.0,
        last_motion_status_at=0.0,
        waypoint_started_at=None,
        waypoint_last_progress_at=None,
        waypoint_best_error=math.inf,
        max_joint_error=lambda _target: 0.20,
        total_joint_error=lambda _target: 0.30,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        publish_joint_command=lambda target: published.append(
            np.asarray(target).copy()),
        publish_status=lambda: None,
        get_logger=lambda: SimpleNamespace(info=lambda _message: None),
        abort_or_finish_captures=lambda reason: pytest.fail(reason),
    )

    ScanViewpointExecutorNode.streaming_moving_tick(fake, 0.051)
    assert len(published) == 1
    assert fake.path_index == 1

    # The first point remains 0.20 rad away, but is below the generous path-
    # following guard; the second scheduled point is sent without a stop.
    ScanViewpointExecutorNode.streaming_moving_tick(fake, 0.101)
    assert len(published) == 2
    np.testing.assert_allclose(published[-1], second)
    assert fake.path_index == 2
    assert fake.waypoint_started_at == pytest.approx(0.101)
    assert fake.stream_schedule_completion_logged
    assert fake.last_stream_planned_duration_sec == pytest.approx(0.10)
    assert fake.last_stream_actual_duration_sec == pytest.approx(0.101)
    assert fake.last_stream_achieved_rate_hz == pytest.approx(1.0 / 0.101)


def test_executor_stretches_schedule_without_bursting_or_skipping_path():
    published = []
    aborts = []
    parameters = {
        'trajectory_following_error_grace_sec': 1.0,
        'trajectory_following_error_rad': 0.30,
    }
    fake = SimpleNamespace(
        motion_started_at=0.0,
        stream_last_tick_at=0.05,
        stream_schedule_paused_sec=0.0,
        stream_following_hold_started_at=None,
        command_target=None,
        path_index=0,
        current_path=[np.full(6, 0.01), np.full(6, 0.02)],
        current_path_times=[0.05, 0.10],
        current_waypoint_error=0.0,
        max_waypoint_error=0.0,
        dropped_command_samples=0,
        command_sent_at=0.0,
        command_samples_sent=0,
        max_command_interval_sec=0.0,
        stream_schedule_completion_logged=False,
        last_stream_planned_duration_sec=0.0,
        last_stream_actual_duration_sec=0.0,
        last_stream_achieved_rate_hz=0.0,
        last_motion_status_at=0.0,
        waypoint_started_at=None,
        waypoint_last_progress_at=None,
        waypoint_best_error=math.inf,
        max_joint_error=lambda _target: 0.01,
        total_joint_error=lambda _target: 0.01,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        publish_joint_command=lambda value: published.append(
            np.asarray(value).copy()),
        publish_status=lambda: None,
        get_logger=lambda: SimpleNamespace(
            info=lambda _message: None,
            warning=lambda _message: None),
        abort_or_finish_captures=aborts.append,
    )

    ScanViewpointExecutorNode.streaming_moving_tick(fake, 0.101)

    assert aborts == []
    assert len(published) == 1
    np.testing.assert_allclose(published[0], np.full(6, 0.01))
    assert fake.path_index == 1
    assert fake.dropped_command_samples == 0
    assert fake.stream_schedule_paused_sec == pytest.approx(0.051)


def test_executor_holds_stream_until_feedback_catches_up():
    published = []
    aborts = []
    target = np.full(6, 0.2)
    errors = iter((0.31, 0.31, 0.25))
    parameters = {
        'trajectory_following_error_grace_sec': 1.0,
        'trajectory_following_error_rad': 0.30,
    }
    logger = SimpleNamespace(
        info=lambda _message: None,
        warning=lambda _message: None,
    )
    fake = SimpleNamespace(
        motion_started_at=0.0,
        stream_last_tick_at=1.0,
        stream_schedule_paused_sec=0.0,
        stream_following_hold_started_at=None,
        command_target=None,
        path_index=0,
        current_path=[target],
        current_path_times=[1.0],
        current_waypoint_error=0.0,
        max_waypoint_error=0.0,
        dropped_command_samples=0,
        command_sent_at=0.0,
        command_samples_sent=0,
        max_command_interval_sec=0.0,
        stream_schedule_completion_logged=False,
        last_stream_planned_duration_sec=0.0,
        last_stream_actual_duration_sec=0.0,
        last_stream_achieved_rate_hz=0.0,
        last_motion_status_at=0.0,
        waypoint_started_at=None,
        waypoint_last_progress_at=None,
        waypoint_best_error=math.inf,
        max_joint_error=lambda _target: next(errors),
        total_joint_error=lambda _target: 0.25,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        publish_joint_command=lambda value: published.append(
            np.asarray(value).copy()),
        publish_status=lambda: None,
        get_logger=lambda: logger,
        abort_or_finish_captures=aborts.append,
    )

    ScanViewpointExecutorNode.streaming_moving_tick(fake, 1.05)
    ScanViewpointExecutorNode.streaming_moving_tick(fake, 1.10)
    assert published == []
    assert aborts == []
    assert fake.path_index == 0
    assert fake.stream_schedule_paused_sec == pytest.approx(0.10)

    ScanViewpointExecutorNode.streaming_moving_tick(fake, 1.15)
    assert len(published) == 1
    assert fake.path_index == 1
    assert fake.stream_following_hold_started_at is None


def test_executor_aborts_movej_immediately_when_joint_delivery_is_stale():
    aborts = []
    fake = SimpleNamespace(
        now=lambda: 5.0,
        fresh=lambda key, timeout=None: False,
        get_parameter=lambda name: SimpleNamespace(value={
            'home_joint_feedback_timeout_sec': 1.0,
        }[name]),
        abort_or_finish_captures=aborts.append,
    )

    ScanViewpointExecutorNode.moving_tick(fake)

    assert aborts == [
        'joint feedback became invalid during SDK MoveJ: no fresh '
        'application-level sample']


def test_runtime_limit_change_accepts_fresh_valid_sdk_limits():
    fake = SimpleNamespace()
    changed_limits = SimpleNamespace(
        max_velocity_rad_s=[0.05] * 6,
        max_acceleration_rad_s2=[0.3] * 6,
    )
    rejection = ScanViewpointExecutorNode.runtime_motion_limit_rejection(
        fake, changed_limits)
    assert rejection == ''


def test_runtime_limit_change_rejects_malformed_sdk_limits():
    malformed = SimpleNamespace(
        max_velocity_rad_s=[0.05] * 5,
        max_acceleration_rad_s2=[0.3] * 6,
    )
    rejection = ScanViewpointExecutorNode.runtime_motion_limit_rejection(
        SimpleNamespace(), malformed)
    assert 'malformed' in rejection


def test_acquisition_lock_requires_post_refresh_measured_tracking():
    health = SimpleNamespace(
        lifecycle_state='TRACKING',
        prediction_only=False,
        camera_settled=True,
        measurement_age_sec=0.1,
    )
    args = dict(
        health=health,
        target_status='LOCKED',
        tracking_updated_at=11.0,
        target_updated_at=11.0,
        refresh_started_at=10.0,
        now=11.1,
        data_timeout_sec=1.0,
        max_measurement_age_sec=0.75,
    )
    assert acquisition_tracking_locked(**args)
    assert not acquisition_tracking_locked(
        **dict(args, tracking_updated_at=9.9))
    assert not acquisition_tracking_locked(
        **dict(args, health=SimpleNamespace(
            lifecycle_state='TRACKING',
            prediction_only=True,
            camera_settled=True,
            measurement_age_sec=0.1,
        )))
    assert not acquisition_tracking_locked(
        **dict(args, now=12.1))


def test_final_capture_starts_bound_return_home_without_an_extra_capture():
    events = []
    executor = SimpleNamespace(
        current_view=12,
        plan_capture_count=13,
        plan_returns_home=True,
        plan_targets=[object()] * 14,
        prepare_current_view=lambda: [],
        abort_motion=lambda reason: events.append(('abort', reason)),
        begin_runtime_refresh=lambda reason, require_workflow,
        allow_missing_obstacles: events.append((
            'refresh', reason, require_workflow, allow_missing_obstacles)),
        set_state=lambda state, reason: events.append(
            ('state', state, reason)),
    )

    ScanViewpointExecutorNode.advance_view(executor)

    assert executor.current_view == 13
    assert events == [(
        'refresh',
        'returning to the approved home position after all captures',
        False,
        False,
    )]


def test_closed_loop_capture_24_holds_for_direct_mission_home():
    events = []
    executor = SimpleNamespace(
        current_view=0,
        plan_capture_count=1,
        plan_returns_home=True,
        plan_targets=[object(), object()],
        scan_history=[object()] * 24,
        get_parameter=lambda name: SimpleNamespace(value={
            'max_execution_viewpoints': 24,
        }[name]),
        param_bool=lambda name: name == 'closed_loop_one_view',
        command_target=np.ones(6),
        current_path=[np.ones(6)],
        current_path_times=[0.0],
        publish_hold=lambda: events.append(('hold',)),
        prepare_current_view=lambda: (_ for _ in ()).throw(
            AssertionError('automatic capture 24 must not execute embedded home')),
        set_state=lambda state, reason: events.append(
            ('state', state, reason)),
    )

    ScanViewpointExecutorNode.advance_view(executor)

    assert executor.current_view == 1
    assert events[0] == ('hold',)
    assert events[1][0:2] == ('state', 'VIEW_COMPLETE')
    assert 'direct configured-home shutdown' in events[1][2]


def test_quality_rejected_view_is_recorded_for_replan_exclusion():
    published = []
    executor = SimpleNamespace(
        plan_kind='MULTIVIEW_SCAN',
        current_view=0,
        plan_id='plan-a',
        plan_viewpoints=[{
            'index': 7,
            'desired_camera_position': {'x': 0.3, 'y': 0.1, 'z': 0.2},
            'desired_look_at_direction': {'x': -1.0, 'y': 0.0, 'z': 0.0},
        }],
        plan_target_center=np.asarray([0.4, 0.0, 0.04]),
        scan_rejections=[],
        pending_scan_qualified_target_shape={'valid': True},
        latest_achieved_scan_view={
            'plan_id': 'plan-a',
            'viewpoint_index': 7,
            'camera_position': {'x': 0.31, 'y': 0.09, 'z': 0.21},
            'look_direction': {'x': -0.99, 'y': 0.01, 'z': 0.0},
            'joint_positions_rad': [0.0] * 6,
            'achieved_at_sec': 12.0,
            'target_estimate_at_capture': {'x': 0.4, 'y': 0.0, 'z': 0.04},
        },
        latest_achieved_matches_current_view=lambda: True,
        publish_scan_history=lambda: published.append(True),
    )

    assert ScanViewpointExecutorNode.record_rejected_view(
        executor, 'QUALITY_REJECTED: poor focus')
    assert executor.scan_rejections[0]['viewpoint_index'] == 7
    assert executor.scan_rejections[0]['actual_camera_position']['x'] == 0.31
    assert 'poor focus' in executor.scan_rejections[0]['reason']
    assert executor.pending_scan_qualified_target_shape is None
    assert published == [True]


def test_first_accepted_view_promotes_pending_model_seed():
    published = []
    seed = {'valid': True, 'measurement_sha256': 'seed'}
    capture_seed = {'model_seed_sha256': 'capture-seed'}
    executor = SimpleNamespace(
        plan_kind='MULTIVIEW_SCAN',
        scan_session_id='session-a',
        scan_history=[],
        scan_qualified_target_shape=None,
        pending_scan_qualified_target_shape=seed,
        scan_qualified_target_model_seed=None,
        pending_scan_qualified_target_model_seed=capture_seed,
        current_view=0,
        plan_id='plan-a',
        plan_target_center=np.asarray([0.4, 0.0, 0.04]),
        plan_viewpoints=[{
            'index': 7,
            'ray_id': 12,
            'desired_camera_position': {'x': 0.3, 'y': 0.1, 'z': 0.2},
            'desired_look_at_direction': {'x': -1.0, 'y': 0.0, 'z': 0.0},
        }],
        latest_achieved_scan_view={
            'plan_id': 'plan-a',
            'viewpoint_index': 7,
            'camera_position': {'x': 0.31, 'y': 0.09, 'z': 0.21},
            'look_direction': {'x': -0.99, 'y': 0.01, 'z': 0.0},
            'joint_positions_rad': [0.0] * 6,
            'achieved_at_sec': 12.0,
        },
        latest_achieved_matches_current_view=lambda: True,
        publish_scan_history=lambda: published.append(True),
        abort_motion=lambda reason: pytest.fail(reason),
    )

    assert ScanViewpointExecutorNode.record_accepted_view(executor, 1)
    assert executor.scan_qualified_target_shape == seed
    assert executor.scan_qualified_target_model_seed == capture_seed
    assert executor.pending_scan_qualified_target_shape is None
    assert executor.pending_scan_qualified_target_model_seed is None
    assert len(executor.scan_history) == 1
    assert published == [True]


def test_scan_history_publishes_frozen_measured_coverage_center():
    published = []
    executor = SimpleNamespace(
        scan_session_id='session-a',
        scan_history=[],
        scan_rejections=[],
        scan_qualified_target_shape=None,
        pending_scan_qualified_target_shape={'valid': True},
        scan_qualified_target_model_seed=None,
        pending_scan_qualified_target_model_seed={
            'model_seed_sha256': 'capture-seed'},
        scan_coverage_target_center=np.asarray([0.4, -0.08, 0.03]),
        get_parameter=lambda _name: SimpleNamespace(value=13),
        scan_history_pub=SimpleNamespace(
            publish=lambda message: published.append(json.loads(message.data))),
    )

    ScanViewpointExecutorNode.publish_scan_history(executor)

    assert published[0]['coverage_target_center'] == {
        'x': 0.4, 'y': -0.08, 'z': 0.03}
    assert published[0]['qualified_target_shape'] is None


def test_new_workflow_session_clears_frozen_coverage_center():
    published = []
    marked = []
    executor = SimpleNamespace(
        latest_workflow=None,
        scan_session_id='session-a',
        scan_history=[{'accepted_view': 1}],
        scan_rejections=[{'rejected_view': 1}],
        scan_coverage_target_center=np.asarray([0.4, -0.08, 0.03]),
        publish_scan_history=lambda: published.append(True),
        mark=lambda name: marked.append(name),
    )
    message = SimpleNamespace(data=json.dumps({'session_id': 'session-b'}))

    ScanViewpointExecutorNode.workflow_cb(executor, message)

    assert executor.scan_session_id == 'session-b'
    assert executor.scan_history == []
    assert executor.scan_rejections == []
    assert executor.scan_coverage_target_center is None
    assert published == [True]
    assert marked == ['workflow']


def test_return_home_completion_holds_and_reports_complete():
    events = []
    executor = SimpleNamespace(
        plan_returns_home=True,
        current_view=13,
        plan_capture_count=13,
        command_target=np.ones(6),
        current_path=[np.ones(6)],
        current_path_velocities=[np.zeros(6)],
        current_path_accelerations=[np.zeros(6)],
        current_path_times=[1.0],
        publish_hold=lambda: events.append(('hold',)),
        finish_scan_client=SimpleNamespace(
            call_async=lambda request: events.append(('finish', request))),
        set_state=lambda state, reason: events.append(
            ('state', state, reason)),
    )

    assert ScanViewpointExecutorNode.returning_home(executor)
    ScanViewpointExecutorNode.complete_return_home(executor)

    assert events[0] == ('hold',)
    assert events[1][0] == 'finish'
    assert events[2][0:2] == ('state', 'FINISHING_WORKFLOW')
    assert 'home reached' in events[2][2]
    assert executor.command_target is None
    assert executor.current_path == []


def test_abort_retrace_home_reports_original_abort_after_reaching_home():
    events = []
    executor = SimpleNamespace(
        abort_return_in_progress=True,
        abort_return_reason='capture service response timed out',
        command_target=np.ones(6),
        current_path=[np.ones(6)],
        current_path_velocities=[np.zeros(6)],
        current_path_accelerations=[np.zeros(6)],
        current_path_times=[1.0],
        publish_hold=lambda: events.append(('hold',)),
        set_state=lambda state, reason: events.append(
            ('state', state, reason)),
    )

    ScanViewpointExecutorNode.complete_return_home(executor)

    assert events[0] == ('hold',)
    assert events[1][0:2] == ('state', 'ABORTED')
    assert 'capture service response timed out' in events[1][2]
    assert 'safely retraced' in events[1][2]
    assert 'approved plan start reached' in events[1][2]
    assert 'configured home reached' not in events[1][2]
    assert not executor.abort_return_in_progress


def test_return_home_waits_for_stable_feedback_before_complete():
    events = []
    executor = SimpleNamespace(
        state_started=0.0,
        settle_started=0.5,
        now=lambda: 2.0,
        get_parameter=lambda name: SimpleNamespace(value={
            'home_settle_timeout_sec': 30.0,
            'home_settle_duration_sec': 1.0,
        }[name]),
        home_joints_settled=lambda: True,
        abort_motion=lambda reason: events.append(('abort', reason)),
        complete_return_home=lambda: events.append(('complete',)),
    )

    ScanViewpointExecutorNode.return_home_settling_tick(executor)

    assert events == [('complete',)]


def test_return_home_unsettled_feedback_resets_stability_window():
    executor = SimpleNamespace(
        state_started=0.0,
        settle_started=0.5,
        now=lambda: 2.0,
        get_parameter=lambda name: SimpleNamespace(value={
            'home_settle_timeout_sec': 30.0,
        }[name]),
        home_joints_settled=lambda: False,
        abort_motion=lambda _reason: None,
    )

    ScanViewpointExecutorNode.return_home_settling_tick(executor)

    assert executor.settle_started is None


def test_return_runtime_timeout_keeps_thirteen_captures_terminal():
    events = []
    executor = SimpleNamespace(
        runtime_reasons=lambda *_args, **_kwargs: [
            'obstacles data missing or stale'],
        runtime_refresh_resume_state='MOVING',
        runtime_refresh_require_workflow=False,
        runtime_refresh_allow_missing_obstacles=False,
        acquisition_scene_snapshot_validated=False,
        is_acquisition=lambda: False,
        get_parameter=lambda name: SimpleNamespace(value={
            'runtime_recovery_timeout_sec': 10.0,
            'runtime_refresh_timeout_sec': 3.0,
        }[name]),
        now=lambda: 12.0,
        state_started=0.0,
        returning_home=lambda: True,
        finish_captures_without_home=lambda reason: events.append(
            ('captured', reason)),
        abort_motion=lambda reason: events.append(('abort', reason)),
        pending_motion_reason='',
    )

    ScanViewpointExecutorNode.waiting_for_runtime_refresh_tick(executor)

    assert len(events) == 1
    assert events[0][0] == 'captured'
    assert 'obstacles data missing or stale' in events[0][1]


def test_dedicated_return_refresh_reuses_its_approved_obstacle_snapshot():
    calls = []
    scene = object()
    executor = SimpleNamespace(
        runtime_reasons=lambda policy, **_kwargs: calls.append(policy) or [],
        runtime_refresh_resume_state='',
        runtime_refresh_require_workflow=False,
        runtime_refresh_allow_missing_obstacles=False,
        acquisition_scene_snapshot_validated=False,
        plan_kind=RETURN_HOME,
        plan_collision_model_qualified=True,
        latest_obstacles=scene,
        is_acquisition=lambda: False,
        get_parameter=lambda name: SimpleNamespace(value={
            'runtime_recovery_timeout_sec': 10.0,
            'runtime_refresh_timeout_sec': 3.0,
        }[name]),
        now=lambda: 0.1,
        state_started=0.0,
        returning_home=lambda: True,
        pending_motion_reason='approved dedicated home',
        set_state=lambda state, reason: calls.append((state, reason)),
    )

    ScanViewpointExecutorNode.waiting_for_runtime_refresh_tick(executor)

    assert calls[0].mode == SafetyMode.RETURN_HOME
    assert calls[0].obstacle_authority == \
        ObstacleAuthority.STATIC_BOOTSTRAP
    assert not calls[0].require_tracking
    assert calls[1] == ('MOVING', 'approved dedicated home')


def test_correlated_heavy_result_is_preserved_during_runtime_hold():
    request_id = 'acquire-0-attempt-1'
    payload = {
        'state': 'published',
        'request_id': request_id,
        'image_stamp': {'sec': 10, 'nanosec': 5},
        'target_detected': True,
    }
    executor = SimpleNamespace(
        state='WAITING_FOR_RUNTIME_REFRESH',
        runtime_refresh_resume_state='WAITING_FOR_GROUNDING_DINO',
        acquisition_request_id=request_id,
        acquisition_min_image_stamp_ns=10_000_000_000,
        pending_acquisition_heavy_status=None,
        is_acquisition=lambda: True,
    )

    ScanViewpointExecutorNode.heavy_refresh_status_cb(
        executor, SimpleNamespace(data=json.dumps(payload)))

    assert executor.pending_acquisition_heavy_status == payload


def test_runtime_recovery_replays_preserved_correlated_heavy_result():
    payload = {
        'state': 'published',
        'request_id': 'acquire-0-attempt-1',
        'image_stamp': {'sec': 10, 'nanosec': 5},
        'target_detected': True,
    }
    events = []
    executor = SimpleNamespace(
        runtime_reasons=lambda *_args, **_kwargs: [],
        runtime_refresh_resume_state='WAITING_FOR_GROUNDING_DINO',
        runtime_refresh_require_workflow=False,
        runtime_refresh_allow_missing_obstacles=False,
        acquisition_scene_snapshot_validated=False,
        pending_acquisition_heavy_status=payload,
        is_acquisition=lambda: True,
        get_parameter=lambda name: SimpleNamespace(value={
            'runtime_recovery_timeout_sec': 10.0,
            'runtime_refresh_timeout_sec': 3.0,
        }[name]),
        now=lambda: 1.0,
        state_started=0.0,
        returning_home=lambda: False,
        pending_motion_reason='runtime telemetry restored',
        set_state=lambda state, reason: events.append(('state', state, reason)),
        heavy_refresh_status_cb=lambda msg: events.append(
            ('heavy', json.loads(msg.data))),
        command_target=None,
    )

    ScanViewpointExecutorNode.waiting_for_runtime_refresh_tick(executor)

    assert events[0][:2] == ('state', 'WAITING_FOR_GROUNDING_DINO')
    assert events[1] == ('heavy', payload)
    assert executor.pending_acquisition_heavy_status is None


def test_dedicated_return_failure_is_not_reported_as_captured_scan_success():
    events = []
    executor = SimpleNamespace(
        abort_return_in_progress=False,
        plan_kind=RETURN_HOME,
        _terminal_abort=lambda reason: events.append(('abort', reason)),
        finish_captures_without_home=lambda reason: events.append(
            ('captured', reason)),
    )

    ScanViewpointExecutorNode.handle_return_home_failure(
        executor, 'home waypoint stalled')

    assert events == [(
        'abort',
        'dedicated configured-home execution failed: home waypoint stalled',
    )]


def test_post_capture_return_failure_holds_current_pose_and_finalizes():
    events = []
    executor = SimpleNamespace(
        command_target=np.ones(6),
        current_path=[np.ones(6)],
        current_path_velocities=[np.zeros(6)],
        current_path_accelerations=[np.zeros(6)],
        current_path_times=[1.0],
        publish_hold=lambda: events.append(('hold',)),
        finish_scan_client=SimpleNamespace(
            call_async=lambda request: events.append(('finish', request))),
        set_state=lambda state, reason: events.append(
            ('state', state, reason)),
    )

    ScanViewpointExecutorNode.finish_captures_without_home(
        executor, 'obstacle telemetry expired')

    assert events[0] == ('hold',)
    assert events[1][0] == 'finish'
    assert events[2][0:2] == ('state', 'FINISHING_WORKFLOW')
    assert 'current position' in events[2][2]
    assert executor.command_target is None
    assert executor.return_home_warning == 'obstacle telemetry expired'


def test_legacy_multiview_without_return_segment_still_completes_capture():
    events = []
    executor = SimpleNamespace(
        current_view=12,
        plan_capture_count=13,
        plan_returns_home=False,
        plan_targets=[object()] * 13,
        set_state=lambda state, reason: events.append((state, reason)),
    )

    ScanViewpointExecutorNode.advance_view(executor)

    assert events[0][0] == 'COMPLETE'
    assert 'captured' in events[0][1]
