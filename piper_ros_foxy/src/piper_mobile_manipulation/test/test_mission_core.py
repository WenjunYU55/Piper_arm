import pytest
from pathlib import Path
from types import SimpleNamespace

from piper_mobile_manipulation.mission_core import (
    MissionPhase,
    MissionRegistry,
    MissionSession,
    validate_goal_payload,
)
from piper_mobile_manipulation.target_scan_mission_node import (
    MissionFailure,
    TargetScanMissionNode,
)
from piper_mobile_manipulation.mission_spool import MissionSpool


def goal(task_id='scan-task-0001', profile='green_cube', now=1000.0):
    covariance = [0.0] * 36
    covariance[0] = covariance[7] = covariance[14] = 0.01
    return {
        'task_id': task_id,
        'task_type': 'SCAN_3D',
        'target_label': 'green cube',
        'target_profile': profile,
        'target_confidence': 0.8,
        'deadline_sec': 1200.0,
        'rough_target': {
            'frame_id': 'odom',
            'stamp_sec': now,
            'position': [0.25, -0.25, 0.0],
            'covariance': covariance,
        },
    }


def test_goal_is_bounded_hashed_and_green_cube_profiled():
    normalized = validate_goal_payload(goal(), now_sec=1001.0)
    assert normalized['mission_sha256']
    assert normalized['deadline_sec'] == 1200.0
    with pytest.raises(ValueError, match='unsupported target profile'):
        validate_goal_payload(goal(profile='unknown'), now_sec=1001.0)
    wrong_label = goal()
    wrong_label['target_label'] = 'red sphere'
    with pytest.raises(ValueError, match='target_label'):
        validate_goal_payload(wrong_label, now_sec=1001.0)
    weak = goal()
    weak['target_confidence'] = 0.59
    with pytest.raises(ValueError, match='confidence'):
        validate_goal_payload(weak, now_sec=1001.0)


def test_goal_rejects_stale_pose_and_excessive_uncertainty():
    with pytest.raises(ValueError, match='stale'):
        validate_goal_payload(goal(now=990.0), now_sec=1001.0)
    uncertain = goal()
    uncertain['rough_target']['covariance'][0] = 0.31 ** 2
    with pytest.raises(ValueError, match='uncertainty'):
        validate_goal_payload(uncertain, now_sec=1001.0)


def test_registry_rejects_busy_and_changed_duplicate_but_replays_result():
    registry = MissionRegistry()
    first = validate_goal_payload(goal(), now_sec=1001.0)
    second = validate_goal_payload(goal('scan-task-0002'), now_sec=1001.0)
    assert registry.admit(first)[0] == 'ACCEPTED'
    assert registry.admit(first)[0] == 'ACTIVE'
    assert registry.admit(second)[0] == 'BUSY'
    changed = dict(first)
    changed['mission_sha256'] = 'f' * 64
    assert registry.admit(changed)[0] == 'CONFLICT'
    result = registry.active.result_payload('FAILED', 'test')
    registry.finish(result)
    assert registry.admit(first)[0] == 'CACHED'


def test_link_loss_and_deadline_are_bounded():
    session = MissionSession(validate_goal_payload(goal(), now_sec=1001.0))
    session.started_monotonic = 10.0
    session.heartbeat_monotonic = 10.0
    assert session.heartbeat_stale(15.01)
    assert session.deadline_expired(1210.01)


def test_disable_never_counts_as_safe_without_current_feedback_hold():
    session = MissionSession(validate_goal_payload(goal(), now_sec=1001.0))
    session.disabled_proved = True
    session.processes_stopped = True
    phase, reason = session.shutdown_outcome()
    assert phase == MissionPhase.NEEDS_OPERATOR
    assert 'current-position hold' in reason
    session.current_hold_proved = True
    phase, reason = session.shutdown_outcome()
    assert phase == MissionPhase.NEEDS_OPERATOR
    assert 'home return' in reason
    session.return_home_proved = True
    result = session.result_payload('FAILED', 'planned failure')
    assert result['safe_shutdown']
    assert len(result['result_sha256']) == 64


def test_shutdown_never_calls_disable_before_home_is_proved():
    session = MissionSession(validate_goal_payload(goal(), now_sec=1001.0))
    session.arm_enabled = True
    calls = []
    harness = SimpleNamespace(
        transition=lambda *_args, **_kwargs: None,
        prove_return_home_for_shutdown=lambda _session: False,
        prove_current_hold_for_shutdown=lambda _session: True,
        call_enable=lambda enabled: calls.append(bool(enabled)),
    )

    failure = TargetScanMissionNode.safe_shutdown(
        harness, session, normal_completion=False)

    assert 'home return was not proved' in str(failure)
    assert calls == []
    assert session.arm_enabled


def test_cancel_shutdown_returns_home_before_disable_and_process_stop():
    session = MissionSession(validate_goal_payload(goal(), now_sec=1001.0))
    session.arm_enabled = True
    calls = []

    def prove_home(bound_session):
        calls.append('home')
        bound_session.return_home_proved = True
        return True

    harness = SimpleNamespace(
        transition=lambda *_args, **_kwargs: None,
        prove_return_home_for_shutdown=prove_home,
        prove_current_hold_for_shutdown=lambda _session: (
            calls.append('hold') or True),
        call_enable=lambda enabled: calls.append(('enable', bool(enabled))),
        authorize_mission=lambda *_args, **_kwargs: calls.append('revoke'),
        processes=SimpleNamespace(
            stop_all=lambda: (calls.append('stop') or True)),
    )

    failure = TargetScanMissionNode.safe_shutdown(
        harness, session, normal_completion=False,
        failure=RuntimeError('operator cancelled scan execution'))

    assert failure is None
    assert calls[:3] == ['home', 'hold', ('enable', False)]
    assert calls[-1] == 'stop'
    assert session.disabled_proved and session.processes_stopped


def test_direct_mission_launch_inherits_loopback_udp_transport():
    launch_source = (
        Path(__file__).resolve().parents[1]
        / 'launch'
        / 'target_scan_mission.launch.py'
    ).read_text(encoding='utf-8')
    assert "'FASTRTPS_DEFAULT_PROFILES_FILE'" in launch_source
    assert "'fastdds_gui_udp_only.xml'" in launch_source
    assert "SetEnvironmentVariable('RMW_FASTRTPS_USE_QOS_FROM_XML', '0')" \
        in launch_source
    assert "SetEnvironmentVariable('ROS_LOCALHOST_ONLY', '0')" \
        in launch_source


def test_motion_safety_failure_never_attempts_automatic_home():
    session = MissionSession(validate_goal_payload(goal(), now_sec=1001.0))
    session.arm_enabled = True
    calls = []
    harness = SimpleNamespace(
        prove_return_home_for_shutdown=lambda _session: calls.append('home'),
    )

    failure = TargetScanMissionNode.safe_shutdown(
        harness, session, normal_completion=False,
        failure=RuntimeError('invalid obstacle geometry is present'))

    assert 'motion-safety-related' in str(failure)
    assert calls == []
    assert session.arm_enabled


def test_multiview_planning_rejection_is_correlated_and_fails_immediately():
    request_id = '0123456789abcdef0123456789abcdef'
    harness = SimpleNamespace(
        latest_plan=SimpleNamespace(
            plan_kind='MULTIVIEW_SCAN',
            plan_id=request_id,
            source_request_id='',
            valid=False,
            reason='PLANNING_FAILED: folded-home qualification failed',
        ),
        latest_plan_at=float('inf'),
        wait_for=lambda _goal, _session, predicate, _timeout, _failure: predicate(),
    )

    with pytest.raises(MissionFailure, match='folded-home qualification failed'):
        TargetScanMissionNode.wait_for_plan(
            harness, None, None, 'MULTIVIEW_SCAN', request_id, 185.0)


def test_multiview_plan_prefix_cannot_correlate_to_full_request_id():
    request_id = '0123456789abcdef0123456789abcdef'

    def wait_once(_goal, _session, predicate, _timeout, _failure):
        if not predicate():
            raise RuntimeError('not correlated')

    harness = SimpleNamespace(
        latest_plan=SimpleNamespace(
            plan_kind='MULTIVIEW_SCAN',
            plan_id=request_id[:16],
            source_request_id='',
            valid=True,
            reason='stale prefix plan',
        ),
        latest_plan_at=float('inf'),
        wait_for=wait_once,
    )

    with pytest.raises(RuntimeError, match='not correlated'):
        TargetScanMissionNode.wait_for_plan(
            harness, None, None, 'MULTIVIEW_SCAN', request_id, 185.0)


def test_pipeline_waits_for_stable_multiview_readiness_before_request():
    source = Path(
        TargetScanMissionNode.__module__.replace('.', '/') + '.py')
    package_source = (
        Path(__file__).resolve().parents[1]
        / 'piper_mobile_manipulation'
        / source.name
    ).read_text(encoding='utf-8')
    lock_wait = package_source.index(
        "'measured target lock ready; waiting for stable multiview readiness'")
    readiness_wait = package_source.index(
        "goal_handle, session, 'multiview', 1.0, 30.0")
    plan_request = package_source.index(
        "'requesting one correlated diverse 13-view plan'")

    assert lock_wait < readiness_wait < plan_request


def test_gateway_spool_is_atomic_hashed_and_permission_bounded(tmp_path):
    spool = MissionSpool(tmp_path / 'missions')
    payload = spool.write('goals', 'scan-task-0001', {'state': 'GOAL_LATCHED'})
    assert spool.read('goals', 'scan-task-0001') == payload
    assert (spool.root.stat().st_mode & 0o777) == 0o700
    assert (spool.path('goals', 'scan-task-0001').stat().st_mode & 0o777) == 0o600
    spool.path('goals', 'scan-task-0001').write_text('{}')
    with pytest.raises(ValueError, match='SHA-256'):
        spool.read('goals', 'scan-task-0001')
