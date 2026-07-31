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
