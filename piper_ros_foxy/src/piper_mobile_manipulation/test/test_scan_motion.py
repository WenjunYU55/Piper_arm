import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
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
)
from piper_mobile_manipulation.scan_motion import (
    approval_rejection_reason,
    bootstrap_start_limit_recovery_reasons,
    bootstrap_recovery_declaration_reasons,
    CollisionBox,
    feedback_joint_limit_reasons,
    PiperScanKinematics,
    configuration_collision_reasons,
    interpolate_joint_path,
    load_conservative_joint_limits,
    minimum_self_segment_clearance,
    orbit_camera_view,
    segment_intersects_expanded_box,
    URDF_JOINT_LIMITS,
    validate_joint_path,
    validate_monotonic_self_clearance_escape,
)
from piper_mobile_manipulation.motion_limit_stability import MotionLimitStability
from piper_mobile_manipulation.scan_trajectory import (
    TIMING_POLICY_VERSION,
    validate_sdk_movej_waypoint_path,
    validate_tesseract_point,
)
from piper_mobile_manipulation.scan_viewpoint_executor_node import (
    abort_return_home_blocker,
    joint_progress_error,
    missing_obstacles_can_wait,
    rgbd_capture_handoff_action,
    runtime_gate_action,
    runtime_refresh_action,
    target_drift_before_approval_rejection,
    waypoint_motion_action,
    ScanViewpointExecutorNode,
)


LINK6_FROM_CAMERA = np.asarray([
    [-0.0635035764, 0.9974167728, -0.0335719700, -0.0745866291],
    [-0.9979815660, -0.0634575393, 0.0024360971, -0.0027843239],
    [0.0002994095, 0.0336589081, 0.9994333336, 0.0266401932],
    [0.0, 0.0, 0.0, 1.0],
])

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
        'invalid obstacle geometry is present',
    ]) == 'abort'
    assert runtime_gate_action([
        'obstacles data missing or stale',
        'arm err_code=2',
    ]) == 'abort'


def test_abort_return_home_rejects_safety_faults_but_allows_capture_faults():
    assert abort_return_home_blocker('capture service response timed out') == ''
    assert abort_return_home_blocker('workflow finish service failed') == ''
    assert abort_return_home_blocker(
        'runtime safety gate: invalid obstacle geometry is present')
    assert abort_return_home_blocker(
        'SDK MoveJ waypoint made no measurable joint progress')
    assert abort_return_home_blocker('operator cancelled scan execution')


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
        runtime_reasons=lambda **_kwargs: [],
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
            'targets to the scan start',
            False,
            False,
        ),
    ]


def test_safety_abort_never_starts_return_motion():
    executor = SimpleNamespace(
        abort_return_in_progress=False,
        plan_kind=MULTIVIEW_SCAN,
        plan_returns_home=True,
    )

    started, blocker = ScanViewpointExecutorNode.try_start_abort_return(
        executor, 'invalid obstacle geometry is present')

    assert not started
    assert 'safety-related' in blocker


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


def test_feedback_limit_tolerance_is_hard_capped():
    with pytest.raises(ValueError, match=r'within \[0.0, 0.0020\]'):
        feedback_joint_limit_reasons(
            np.zeros(6), URDF_JOINT_LIMITS, tolerance_rad=0.0021)


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
    times = np.asarray([0.0, 0.01, 0.02])
    q, qd, qdd, validated_times = validate_sdk_movej_waypoint_path(
        positions, velocities, accelerations, times,
        command_rate_hz=100.0,
    )
    np.testing.assert_allclose(q, positions)
    np.testing.assert_allclose(qd, velocities)
    np.testing.assert_allclose(qdd, accelerations)
    np.testing.assert_allclose(validated_times, times)
    assert TIMING_POLICY_VERSION == 'sdk_movej_targets_v1'


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


def test_sdk_movej_target_path_rejects_rate_shape_and_derivative_claims():
    positions = np.asarray([np.zeros(6), np.full(6, 0.03)])
    derivatives = np.asarray([np.zeros(6), np.zeros(6)])
    with pytest.raises(ValueError, match='faster'):
        validate_sdk_movej_waypoint_path(
            positions, derivatives, derivatives, [0.0, 0.005],
            command_rate_hz=100.0,
        )
    with pytest.raises(ValueError, match='at most one bootstrap'):
        validate_sdk_movej_waypoint_path(
            np.vstack([positions, np.full(6, 0.04), np.full(6, 0.05)]),
            np.zeros((4, 6)), np.zeros((4, 6)),
            [0.0, 0.01, 0.02, 0.03],
            command_rate_hz=100.0,
        )
    excessive_velocity = derivatives.copy()
    excessive_velocity[1, 5] = 1.1
    with pytest.raises(ValueError, match='derivatives must be zero'):
        validate_sdk_movej_waypoint_path(
            positions * 0.5, excessive_velocity, derivatives, [0.0, 0.01],
            command_rate_hz=100.0,
        )


def test_saved_invalid_j6_is_ignored_and_urdf_limit_remains(tmp_path):
    path = tmp_path / 'bounds.json'
    path.write_text(
        '{"joints":{"joint1":{"min":-1,"max":1,"valid":true},'
        '"joint6":{"min":-9,"max":9,"valid":false}}}'
    )
    limits, ignored = load_conservative_joint_limits(str(path))
    np.testing.assert_allclose(limits[0], [-1.0, 1.0])
    np.testing.assert_allclose(limits[5], [-2.0944, 2.0944])
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
            [-1.745, 1.745], [-1.22, 1.22], [-2.0944, 2.0944],
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
    assert 'remaining' in plan_count_rejection(MULTIVIEW_SCAN, 1, 5, 5)
    assert plan_count_rejection(
        MULTIVIEW_SCAN, 2, 5, 5,
        session_accepted_views=3, session_maximum_views=5) == ''
    assert 'unsupported' in plan_count_rejection('UNKNOWN', 1, 5, 5)


def test_acquisition_uses_configured_speed_with_sdk_range():
    assert commanded_speed_percent(100.0, ROUGH_ACQUISITION, 0.0) == 100.0
    assert commanded_speed_percent(120.0, ROUGH_ACQUISITION, 0.0) == 100.0
    assert commanded_speed_percent(4.0, ROUGH_ACQUISITION, 0.0) == 4.0
    assert commanded_speed_percent(100.0, MULTIVIEW_SCAN, 0.5) == 50.0


def test_planned_speed_accepts_a_safer_older_tracking_scale_only():
    assert planned_speed_rejection(5.0, MULTIVIEW_SCAN, 1.0, 4.0) == ''
    assert 'allowance' in planned_speed_rejection(
        5.0, MULTIVIEW_SCAN, 0.8, 5.0)
    assert planned_speed_rejection(
        5.0, ROUGH_ACQUISITION, 0.0, 5.0) == ''
    assert 'acquisition speed' in planned_speed_rejection(
        5.0, ROUGH_ACQUISITION, 1.0, 4.0)


def test_runtime_refresh_waits_for_fresh_data_before_motion():
    reasons = ['joints data missing or stale']
    assert runtime_refresh_action(reasons, 0.1, 3.0) == 'wait'
    assert runtime_refresh_action([], 0.2, 3.0) == 'start'
    assert runtime_refresh_action(reasons, 3.0, 3.0) == 'abort'


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
    for state in ('SETTLING', 'CAPTURING', 'CAPTURING_RGBD', 'WAIT_CAPTURE'):
        assert missing_obstacles_can_wait('MULTIVIEW_SCAN', 2, state)
    assert not missing_obstacles_can_wait('MULTIVIEW_SCAN', 2, 'MOVING')
    assert not missing_obstacles_can_wait(
        'MULTIVIEW_SCAN', 2, 'WAITING_FOR_RUNTIME_REFRESH')


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
        fake,
        require_settled=False,
        require_workflow=False,
        allow_stale_obstacles=False,
    )
    bound_to_segment = ScanViewpointExecutorNode.runtime_reasons(
        fake,
        require_settled=False,
        require_workflow=False,
        allow_stale_obstacles=True,
    )

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
        fake,
        require_settled=False,
        require_workflow=False,
        allow_stale_obstacles=True,
    )

    assert 'scene_blocked: hand' in reasons


def executor_runtime_fixture(health):
    return SimpleNamespace(
        fresh=lambda *args: True,
        get_parameter=lambda name: SimpleNamespace(value={
            'motion_limits_timeout_sec': 1.0,
            'joint_feedback_limit_tolerance_rad': 0.001,
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
        plan_execution_speed_percent=5.0,
        param_bool=lambda name: False,
        workflow_ready=lambda: True,
    )


def test_tracking_speed_allowance_is_enforced_before_but_not_during_move():
    fake = executor_runtime_fixture(SimpleNamespace(
        lifecycle_state='TRACKING',
        camera_settled=False,
        prediction_only=False,
        measurement_age_sec=0.1,
        recommended_speed_scale=0.8,
    ))

    before_target = ScanViewpointExecutorNode.runtime_reasons(
        fake,
        require_settled=False,
        require_workflow=False,
        enforce_tracking_speed_allowance=True,
    )
    in_flight = ScanViewpointExecutorNode.runtime_reasons(
        fake,
        require_settled=False,
        require_workflow=False,
        enforce_tracking_speed_allowance=False,
    )

    assert any('tracking speed allowance' in reason for reason in before_target)
    assert not any('tracking speed allowance' in reason for reason in in_flight)


def test_in_flight_speed_exception_keeps_other_tracking_gates_active():
    fake = executor_runtime_fixture(SimpleNamespace(
        lifecycle_state='TRACKING',
        camera_settled=False,
        prediction_only=False,
        measurement_age_sec=2.0,
        recommended_speed_scale=0.8,
    ))

    reasons = ScanViewpointExecutorNode.runtime_reasons(
        fake,
        require_settled=False,
        require_workflow=False,
        enforce_tracking_speed_allowance=False,
    )

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
        fake,
        require_settled=False,
        require_workflow=False,
        enforce_target_status=True,
    )
    in_flight = ScanViewpointExecutorNode.runtime_reasons(
        fake,
        require_settled=False,
        require_workflow=False,
        enforce_target_status=False,
    )

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
        fake,
        require_settled=False,
        require_workflow=False,
        enforce_tracking_motion_state=True,
    )
    in_flight = ScanViewpointExecutorNode.runtime_reasons(
        fake,
        require_settled=False,
        require_workflow=False,
        enforce_tracking_motion_state=False,
    )

    assert any('tracking lifecycle' in reason for reason in before_target)
    assert any('tracking measurement is stale' in reason for reason in before_target)
    assert in_flight == []


def test_missing_tracking_telemetry_still_aborts_an_issued_target():
    fake = executor_runtime_fixture(SimpleNamespace(
        lifecycle_state='WAITING_TO_REACQUIRE',
        camera_settled=False,
        prediction_only=True,
        measurement_age_sec=10.0,
        recommended_speed_scale=0.0,
    ))
    fake.fresh = lambda key, *args: key != 'tracking'

    reasons = ScanViewpointExecutorNode.runtime_reasons(
        fake,
        require_settled=False,
        require_workflow=False,
        enforce_tracking_motion_state=False,
    )

    assert 'tracking data missing or stale' in reasons


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


def test_runtime_limit_change_requires_a_fresh_exact_target_plan():
    path = [
        np.zeros(6),
        np.full(6, 0.01),
    ]
    velocities = [
        np.zeros(6),
        np.full(6, 0.1),
    ]
    accelerations = [
        np.zeros(6),
        np.full(6, 0.2),
    ]
    fake = SimpleNamespace(
        plan_paths=[path],
        plan_path_velocities=[velocities],
        plan_path_accelerations=[accelerations],
        plan_path_times=[[0.0, 0.1]],
        current_view=0,
        get_parameter=lambda name: SimpleNamespace(
            value={
                'trajectory_command_rate_hz': 100.0,
                'trajectory_joint_step_rad': 0.025,
            }[name]),
    )
    changed_limits = SimpleNamespace(
        max_velocity_rad_s=[0.05] * 6,
        max_acceleration_rad_s2=[0.3] * 6,
    )
    rejection = ScanViewpointExecutorNode.runtime_motion_limit_rejection(
        fake, changed_limits)
    assert 'request a fresh' in rejection
    assert 'target plan' in rejection


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
    assert not executor.abort_return_in_progress


def test_return_home_waits_for_stable_feedback_before_complete():
    events = []
    executor = SimpleNamespace(
        state_started=0.0,
        settle_started=0.5,
        now=lambda: 2.0,
        get_parameter=lambda name: SimpleNamespace(value={
            'settle_timeout_sec': 15.0,
            'settle_duration_sec': 1.0,
        }[name]),
        joints_settled=lambda: True,
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
            'settle_timeout_sec': 15.0,
        }[name]),
        joints_settled=lambda: False,
        abort_motion=lambda _reason: None,
    )

    ScanViewpointExecutorNode.return_home_settling_tick(executor)

    assert executor.settle_started is None


def test_return_runtime_timeout_keeps_thirteen_captures_terminal():
    events = []
    executor = SimpleNamespace(
        runtime_reasons=lambda **_kwargs: ['obstacles data missing or stale'],
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
