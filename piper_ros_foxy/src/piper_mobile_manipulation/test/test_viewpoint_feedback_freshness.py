from types import SimpleNamespace

from piper_mobile_manipulation.viewpoint_reachability_filter_node import (
    bounded_ray_intersects_workspace,
    RAY_WORKSPACE_REJECTION,
    ViewpointReachabilityFilterNode,
)


def filter_fixture(status, status_at, timeout=1.0):
    return SimpleNamespace(
        arm_status=status,
        arm_status_at=status_at,
        get_parameter=lambda name: SimpleNamespace(value={
            'arm_status_timeout_sec': timeout,
        }[name]),
    )


def healthy_status():
    return SimpleNamespace(
        err_code=0,
        joint_1_angle_limit=False,
        joint_2_angle_limit=False,
        joint_3_angle_limit=False,
        joint_4_angle_limit=False,
        joint_5_angle_limit=False,
        joint_6_angle_limit=False,
        communication_status_joint_1=False,
        communication_status_joint_2=False,
        communication_status_joint_3=False,
        communication_status_joint_4=False,
        communication_status_joint_5=False,
        communication_status_joint_6=False,
        motor_feedback_valid=True,
        motor_1_driver_enabled=False,
        motor_2_driver_enabled=False,
        motor_3_driver_enabled=False,
        motor_4_driver_enabled=False,
        motor_5_driver_enabled=False,
        motor_6_driver_enabled=False,
        motor_faults=[],
        motor_watchdog_reason='',
    )


def test_missing_arm_status_fails_closed():
    node = filter_fixture(None, None)
    assert ViewpointReachabilityFilterNode.arm_status_reasons(node) == [
        'arm status is missing'
    ]


def test_stale_arm_status_fails_closed(monkeypatch):
    monkeypatch.setattr(
        'piper_mobile_manipulation.viewpoint_reachability_filter_node.time.monotonic',
        lambda: 10.0,
    )
    node = filter_fixture(healthy_status(), 8.5)
    assert ViewpointReachabilityFilterNode.arm_status_reasons(node) == [
        'arm status is stale 1.500s > 1.000s'
    ]


def test_typed_arm_faults_are_rejected(monkeypatch):
    monkeypatch.setattr(
        'piper_mobile_manipulation.viewpoint_reachability_filter_node.time.monotonic',
        lambda: 10.0,
    )
    status = healthy_status()
    status.err_code = 7
    status.joint_2_angle_limit = True
    status.communication_status_joint_4 = True
    node = filter_fixture(status, 9.9)
    assert ViewpointReachabilityFilterNode.arm_status_reasons(node) == [
        'arm err_code=7',
        'arm reports a joint angle-limit fault',
        'arm reports a joint communication fault',
    ]


def test_fully_disabled_motor_feedback_is_valid_for_preflight(monkeypatch):
    monkeypatch.setattr(
        'piper_mobile_manipulation.viewpoint_reachability_filter_node.time.monotonic',
        lambda: 10.0,
    )
    node = filter_fixture(healthy_status(), 9.9)
    assert ViewpointReachabilityFilterNode.arm_status_reasons(node) == []


def test_partial_motor_enable_is_never_planning_authority(monkeypatch):
    monkeypatch.setattr(
        'piper_mobile_manipulation.viewpoint_reachability_filter_node.time.monotonic',
        lambda: 10.0,
    )
    status = healthy_status()
    status.motor_1_driver_enabled = True
    node = filter_fixture(status, 9.9)
    assert ViewpointReachabilityFilterNode.arm_status_reasons(node) == [
        'partial motor enable flags=(True, False, False, False, False, False)'
    ]


def test_dense_batch_reuses_the_admission_snapshot_for_every_viewpoint():
    """Slow command-free lookup must not expire its own admitted evidence."""
    node = reachability_fixture(False)
    calls = []

    def arm_reasons(now=None):
        calls.append(now)
        return [] if now == 10.0 else ['arm status expired during lookup']

    node.arm_status_reasons = arm_reasons
    admission = ViewpointReachabilityFilterNode.batch_admission_reasons(
        node, 'MULTIVIEW_SCAN', now=10.0)
    viewpoint = {
        'desired_camera_position': {'x': 0.4, 'y': 0.0, 'z': 0.2},
        'target_object_center': {'x': 0.1, 'y': 0.0, 'z': 0.2},
        'camera_object_distance_m': 0.30,
    }

    for _index in range(360):
        reasons, _capability = (
            ViewpointReachabilityFilterNode.evaluate_viewpoint(
                node, viewpoint, 'MULTIVIEW_SCAN',
                admission_reasons=admission))
        assert reasons == []
    assert calls == [10.0]


def test_stale_batch_is_rejected_at_admission_and_never_refreshed_mid_loop():
    node = reachability_fixture(False)
    calls = []

    def arm_reasons(now=None):
        calls.append(now)
        return ['arm status is stale 1.500s > 1.000s']

    node.arm_status_reasons = arm_reasons
    admission = ViewpointReachabilityFilterNode.batch_admission_reasons(
        node, 'MULTIVIEW_SCAN', now=10.0)
    viewpoint = {
        'desired_camera_position': {'x': 0.4, 'y': 0.0, 'z': 0.2},
        'target_object_center': {'x': 0.1, 'y': 0.0, 'z': 0.2},
        'camera_object_distance_m': 0.30,
    }

    reasons, _capability = ViewpointReachabilityFilterNode.evaluate_viewpoint(
        node, viewpoint, 'MULTIVIEW_SCAN', admission_reasons=admission)
    assert reasons == ['arm status is stale 1.500s > 1.000s']
    assert calls == [10.0]


def reachability_fixture(enforce_static_reach_bounds):
    parameters = {
        'dry_run': True,
        'enforce_static_reach_bounds': enforce_static_reach_bounds,
        'min_reach_m': 0.20,
        'max_reach_m': 0.75,
        'min_camera_object_distance_m': 0.25,
        'max_camera_object_distance_m': 0.80,
        'max_height_change_m': 0.40,
    }
    return SimpleNamespace(
        target_status='LOCKED',
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        param_bool=lambda name: bool(parameters[name]),
        arm_status_reasons=lambda: [],
        valid_vector=ViewpointReachabilityFilterNode.valid_vector,
        vector_norm=ViewpointReachabilityFilterNode.vector_norm,
        is_finite_number=ViewpointReachabilityFilterNode.is_finite_number,
        distance=ViewpointReachabilityFilterNode.distance,
    )


def test_static_radial_workspace_is_disabled_by_default():
    viewpoint = {
        'desired_camera_position': {'x': 2.0, 'y': 0.0, 'z': 1.0},
        'target_object_center': {'x': 2.3, 'y': 0.0, 'z': 1.0},
        'camera_object_distance_m': 0.30,
    }
    dynamic = ViewpointReachabilityFilterNode.reject_reasons(
        reachability_fixture(False), viewpoint)
    legacy = ViewpointReachabilityFilterNode.reject_reasons(
        reachability_fixture(True), viewpoint)

    assert dynamic == []
    assert any('too far' in reason for reason in legacy)


def target_ray(target, direction, minimum=0.28, maximum=0.50):
    return {
        'candidate_geometry': 'target_ray',
        'desired_camera_position': {
            'x': target[0] + direction[0] * maximum,
            'y': target[1] + direction[1] * maximum,
            'z': target[2] + direction[2] * maximum,
        },
        'target_object_center': dict(zip(('x', 'y', 'z'), target)),
        # Match the planner's live JSON contract rather than relying only on
        # the older positional-array representation.
        'ray_direction': dict(zip(('x', 'y', 'z'), direction)),
        'ray_min_standoff_m': minimum,
        'ray_max_standoff_m': maximum,
    }


def ray_intersects(viewpoint, max_height=0.40):
    return bounded_ray_intersects_workspace(
        viewpoint,
        min_reach_m=0.20,
        max_reach_m=0.75,
        min_camera_distance_m=0.25,
        max_camera_distance_m=0.80,
        max_height_change_m=max_height,
    )


def test_ray_is_kept_when_any_standoff_intersects_workspace():
    viewpoint = target_ray([0.40, 0.0, 0.20], [1.0, 0.0, 0.0])

    assert ray_intersects(viewpoint)
    assert ViewpointReachabilityFilterNode.reject_reasons(
        reachability_fixture(False), viewpoint) == []


def test_recorded_list_ray_direction_remains_compatible():
    viewpoint = target_ray([0.40, 0.0, 0.20], [1.0, 0.0, 0.0])
    viewpoint['ray_direction'] = [1.0, 0.0, 0.0]

    assert ray_intersects(viewpoint)


def test_ray_outside_base_reach_is_culled_before_tesseract():
    viewpoint = target_ray([0.70, 0.0, 0.20], [1.0, 0.0, 0.0])

    assert not ray_intersects(viewpoint)
    assert RAY_WORKSPACE_REJECTION in (
        ViewpointReachabilityFilterNode.reject_reasons(
            reachability_fixture(False), viewpoint))


def test_ray_toward_base_is_kept_when_interval_enters_workspace():
    viewpoint = target_ray([0.90, 0.0, 0.20], [-1.0, 0.0, 0.0])

    assert ray_intersects(viewpoint)


def test_ray_outside_height_slab_is_culled():
    viewpoint = target_ray([0.40, 0.0, 0.20], [0.0, 0.0, 1.0])

    assert not ray_intersects(viewpoint, max_height=0.20)


def test_malformed_ray_fails_closed_without_crashing_filter():
    viewpoint = target_ray([0.40, 0.0, 0.20], [1.0, 0.0, 0.0])
    viewpoint['ray_direction'] = [1.0, 0.0]

    assert not ray_intersects(viewpoint)
    assert RAY_WORKSPACE_REJECTION in (
        ViewpointReachabilityFilterNode.reject_reasons(
            reachability_fixture(False), viewpoint))
