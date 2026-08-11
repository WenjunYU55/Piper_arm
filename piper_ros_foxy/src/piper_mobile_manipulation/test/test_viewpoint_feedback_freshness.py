from types import SimpleNamespace

from piper_mobile_manipulation.viewpoint_reachability_filter_node import (
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
