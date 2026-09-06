"""Command-free replay of slow approval and abandoned client transactions."""

from types import SimpleNamespace

import numpy as np
import pytest

from piper_mobile_manipulation.execution.authorization import (
    bounded_mission_confirmation, parse_approval_confirmation,
    PLAN_APPROVAL_SERVICE_TIMEOUT_SEC,
)
from piper_mobile_manipulation.scan_viewpoint_executor_node import ScanViewpointExecutorNode


@pytest.mark.parametrize('validation_seconds,accepted', [(11.5, True), (54., True), (55., False), (61., False)])
def test_slow_approval_simulation_never_starts_after_deadline(monkeypatch, validation_seconds, accepted):
    clock = [1_000_000_000_000]
    monkeypatch.setattr('piper_mobile_manipulation.scan_viewpoint_executor_node.time.time_ns', lambda: clock[0])
    started = []
    params = dict(approval_confirmation='APPROVE', plan_max_age_sec=120.,
                  max_target_drift_before_approval_m=.015,
                  allow_target_motion_during_scan=False,
                  plan_start_tolerance_rad=.025)

    def validate():
        clock[0] += int(validation_seconds * 1e9)
        return []

    executor = SimpleNamespace(
        mission_sha256='a' * 64, mission_authorization_valid=lambda: True,
        mission_planner_backend='curobo', plan_backend='curobo',
        state='PROPOSAL_READY', plan_targets=[np.zeros(6)], plan_id='recorded-plan',
        plan_created=9., now=lambda: 10., plan_trajectory_sha256='b' * 64,
        real_motion_enabled=lambda: True,
        get_parameter=lambda name: SimpleNamespace(value=params[name]),
        is_acquisition=lambda: True, is_return_home=lambda: False,
        runtime_reasons=lambda *_args, **_kwargs: [],
        plan_collision_model_qualified=True,
        prepare_current_view=validate,
        current_joints=lambda: np.zeros(6), retrace_joint_targets=[],
        begin_runtime_refresh=lambda *args, **kwargs: started.append(args),
        plan_capture_count=1, execution_speed_percent=lambda: 5.)
    request = SimpleNamespace(
        confirmation=bounded_mission_confirmation(executor.mission_sha256, clock[0]),
        plan_id=executor.plan_id, trajectory_sha256=executor.plan_trajectory_sha256)
    response = SimpleNamespace(accepted=False, message='')
    ScanViewpointExecutorNode.approve_cb(executor, request, response)
    assert response.accepted is accepted
    assert bool(started) is accepted
    if accepted:
        assert validation_seconds < PLAN_APPROVAL_SERVICE_TIMEOUT_SEC
    else:
        assert 'deadline expired' in response.message


def test_queued_expired_request_is_rejected_before_reading_executor_state(monkeypatch):
    token = bounded_mission_confirmation('a' * 64, 0)
    monkeypatch.setattr('piper_mobile_manipulation.scan_viewpoint_executor_node.time.time_ns', lambda: 60_000_000_000)
    response = SimpleNamespace(accepted=True, message='')
    ScanViewpointExecutorNode.approve_cb(SimpleNamespace(), SimpleNamespace(confirmation=token), response)
    assert not response.accepted
    assert 'deadline expired' in response.message


@pytest.mark.parametrize('token', ['MISSION_POLICY:a;deadline_ns=bad', 'APPROVE;deadline_ns=4',
                                  'MISSION_POLICY:a;deadline_ns=9999999999999999999'])
def test_malformed_or_unbounded_deadline_rejected(token):
    with pytest.raises(ValueError):
        parse_approval_confirmation(token, 0)


def test_manual_and_legacy_confirmation_remain_compatible():
    assert parse_approval_confirmation('APPROVE', 0) == ('APPROVE', None)
    assert parse_approval_confirmation('MISSION_POLICY:a', 0) == ('MISSION_POLICY:a', None)


def test_complete_simulated_mission_with_incident_approval_delay(monkeypatch):
    """Run the real mission engine with simulated sensors/motion/captures."""
    from test_mission_engine import FakeMissionOperations, _execute
    operations = FakeMissionOperations()
    ordinary_approval = operations.approve_plan
    delayed = []

    def approve(context, plan):
        if plan.plan_kind == 'ROUGH_ACQUISITION':
            test_slow_approval_simulation_never_starts_after_deadline(
                monkeypatch, 11.5, True)
            delayed.append(plan.plan_kind)
        ordinary_approval(context, plan)

    operations.approve_plan = approve
    _, context, result = _execute(operations, task_id='offline-slow-approval-replay')
    assert delayed == ['ROUGH_ACQUISITION']
    assert result.succeeded
    assert context.session.accepted_captures == 8
    assert context.session.return_home_proved
    assert context.session.storage_wrist_proved
    assert context.session.disabled_proved
    assert context.session.processes_stopped
