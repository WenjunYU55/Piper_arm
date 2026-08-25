"""Compatibility and single-owner checks for core-stack modularization."""

from types import SimpleNamespace

from piper_mobile_manipulation.executor_session import (
    EXECUTOR_SESSION_FIELDS,
    ExecutorSession,
    SessionField,
)
from piper_mobile_manipulation.mission_artifacts import (
    calibration_identity_for_mission,
    discard_failed_zero_capture_dataset,
    find_failed_mission_dataset,
)
from piper_mobile_manipulation.mission_ros_operations import (
    MissionNodeOperations,
)
import piper_mobile_manipulation.scan_viewpoint_executor_node as executor_node
import piper_mobile_manipulation.target_scan_mission_node as mission_node


def test_mission_node_preserves_imports_from_focused_owners():
    assert mission_node.calibration_identity_for_mission is (
        calibration_identity_for_mission)
    assert mission_node.discard_failed_zero_capture_dataset is (
        discard_failed_zero_capture_dataset)
    assert mission_node.find_failed_mission_dataset is (
        find_failed_mission_dataset)
    assert mission_node._MissionNodeOperations is MissionNodeOperations


def test_executor_production_fields_are_session_descriptors():
    for name in EXECUTOR_SESSION_FIELDS:
        assert isinstance(
            vars(executor_node.ScanViewpointExecutorNode)[name],
            SessionField,
        )


def test_executor_session_is_the_only_descriptor_storage_owner():
    session = ExecutorSession(now=12.5, maximum_capture_retries=10)
    owner = SimpleNamespace(executor_session=session)
    plan_id = vars(executor_node.ScanViewpointExecutorNode)['plan_id']
    current_path = vars(
        executor_node.ScanViewpointExecutorNode)['current_path']

    plan_id.__set__(owner, 'plan-identity')
    current_path.__set__(owner, [[0.0] * 6])

    assert session.plan_id == 'plan-identity'
    assert session.current_path == [[0.0] * 6]
    assert plan_id.__get__(owner, type(owner)) == 'plan-identity'
    assert 'plan_id' not in vars(owner)
    assert 'current_path' not in vars(owner)
