import pytest
import inspect
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import piper_mobile_manipulation.target_scan_mission_node as mission_node

from piper_mobile_manipulation.mission_core import (
    closest_pending_mission,
    MAX_FEATURE_CAPTURES,
    mission_target_distance_m,
    mission_queue_ready,
    queued_cancel_result,
    REQUIRED_CAPTURES,
    MissionPhase,
    MissionRegistry,
    MissionSession,
    validate_goal_payload,
)
from piper_mobile_manipulation.mission_spool import MissionSpool
from piper_mobile_manipulation.mission_engine import MissionEngine
from piper_mobile_manipulation.target_scan_mission_node import (
    ManagedProcessSet,
    failure_code_for_reason,
    feature_capture_decision,
    MissionFailure,
    planning_rejection_allows_current_state_home,
    retryable_plan_approval_rejection,
    runtime_freshness_plan_request_rejection,
    shutdown_uses_startup_home,
    safe_view_exhaustion_after_capture,
    target_drift_requires_replan,
    TargetScanMissionNode,
    visual_reacquisition_plan_approval_rejection,
    visual_reacquisition_plan_request_rejection,
)


def test_mission_constructor_does_not_read_parameters_before_declaration():
    source = inspect.getsource(TargetScanMissionNode.__init__)
    before_declarations = source.split(
        'for name, value in defaults.items():', 1)[0]
    assert 'self.get_parameter(' not in before_declarations


def test_startup_wrist_direction_is_proved_before_motor_enable():
    handlers = list(MissionEngine.PIPELINE_HANDLERS)
    assert handlers.index(MissionPhase.PREFLIGHT) < handlers.index(
        MissionPhase.ENABLE_AND_HOLD)
    assert 'validate_staged_wrist_direction(' in inspect.getsource(
        MissionEngine._handle_preflight)
    assert 'enable_arm(context, True)' in inspect.getsource(
        MissionEngine._handle_enable_and_hold)


def test_configured_home_acceptance_uses_operator_authorized_point_two_rad():
    node = SimpleNamespace(
        require_fresh_joint_feedback=lambda: None,
        telemetry_store=None,
        latest_joints=SimpleNamespace(position=[0.199] * 6),
    )
    session = SimpleNamespace(home_positions_rad=[0.0] * 6)

    assert TargetScanMissionNode.at_configured_home(node, session)
    node.latest_joints = SimpleNamespace(position=[0.201] + [0.0] * 5)
    assert not TargetScanMissionNode.at_configured_home(node, session)


def test_pre_home_proof_does_not_claim_rough_home_completion():
    session = SimpleNamespace(
        pre_home_completed=False,
        return_home_proved=False,
        storage_wrist_proved=False,
        home_positions_rad=[0.0] * 6,
    )
    node = SimpleNamespace(
        last_return_home_diagnostic='',
        processes=SimpleNamespace(failed=lambda: {}),
        at_configured_home=lambda _session, **_kwargs: True,
    )

    assert TargetScanMissionNode.prove_return_home_for_shutdown(
        node, session, target_positions=[0.0] * 6,
        home_stage='PRE_HOME')
    assert session.pre_home_completed
    assert not session.return_home_proved
    assert not session.storage_wrist_proved


def test_compatibility_hold_helpers_use_acknowledgement_not_noise_window():
    startup_source = inspect.getsource(TargetScanMissionNode.prove_current_hold)
    shutdown_source = inspect.getsource(
        TargetScanMissionNode.prove_current_hold_for_shutdown)

    assert 'HOLD_REQUESTED' in startup_source
    assert 'HOLD_REQUESTED' in shutdown_source
    assert 'position-window proof' in startup_source
    assert 'position-window proof' in shutdown_source
    assert 'delta <= 0.005' not in startup_source
    assert 'delta <= 0.005' not in shutdown_source

    engine_startup = inspect.getsource(MissionEngine._handle_enable_and_hold)
    engine_shutdown = inspect.getsource(MissionEngine.shutdown)
    assert 'prove_current_hold' not in engine_startup
    assert 'prove_shutdown_hold' not in engine_shutdown


def test_direct_home_waits_for_fresh_feedback_not_settle_window():
    source = inspect.getsource(
        TargetScanMissionNode.prove_return_home_for_shutdown)

    assert 'ExecuteHomeStage.Request()' in source
    assert 'require_fresh_joint_feedback()' in source
    assert "held feedback before direct home motion" not in source


def test_only_transient_live_perception_blocks_retry_exact_plan_approval():
    assert retryable_plan_approval_rejection(
        'execution blocked: tracking is not settled TRACKING; '
        'tracking is prediction-only; tracking speed scale is below the '
        'motion threshold')
    assert retryable_plan_approval_rejection(
        'execution blocked: camera timestamp health is stale')
    assert retryable_plan_approval_rejection(
        'execution blocked: joint feedback is not settled for acquisition')
    assert retryable_plan_approval_rejection(
        'execution blocked: target_status=LOW_CONFIDENCE')
    assert retryable_plan_approval_rejection(
        'execution blocked: target_status=LOST')
    assert retryable_plan_approval_rejection(
        'execution blocked: target_status=SEARCHING')
    assert not retryable_plan_approval_rejection(
        'execution blocked: collision scene is stale')
    assert not retryable_plan_approval_rejection(
        'fresh trajectory validation failed: target left the optical cone')


def test_only_target_status_approval_blocks_enter_visual_reacquisition_hold():
    assert visual_reacquisition_plan_approval_rejection(
        'execution blocked: target_status=LOW_CONFIDENCE')
    assert visual_reacquisition_plan_approval_rejection(
        'execution blocked: tracking is not settled TRACKING; '
        'target_status=LOST')
    assert not visual_reacquisition_plan_approval_rejection(
        'execution blocked: tracking is prediction-only')
    assert not visual_reacquisition_plan_approval_rejection(
        'fresh trajectory validation failed: target_status=LOST')


def test_only_visual_tracking_snapshot_blocks_retry_plan_request():
    assert visual_reacquisition_plan_request_rejection(
        'planning blocked: tracking is not settled TRACKING')
    assert visual_reacquisition_plan_request_rejection(
        'planning blocked: tracking is prediction-only; '
        'tracking measurement is stale')
    assert visual_reacquisition_plan_request_rejection(
        'planning blocked: target_status=LOW_CONFIDENCE')
    assert visual_reacquisition_plan_request_rejection(
        'planning blocked: target_status=LOST')
    assert not visual_reacquisition_plan_request_rejection(
        'planning blocked: collision scene is stale')
    assert not visual_reacquisition_plan_request_rejection(
        'Tesseract proposal rejected: PLANNING_FAILED')


def test_only_transient_snapshot_freshness_retries_plan_request():
    assert runtime_freshness_plan_request_rejection(
        'planning blocked: controller motion limits are missing or stale')
    assert runtime_freshness_plan_request_rejection(
        'planning blocked: obstacles data missing or stale')
    assert runtime_freshness_plan_request_rejection(
        'planning blocked: obstacles data is missing or stale')
    assert not runtime_freshness_plan_request_rejection(
        'planning blocked: controller motion limits are invalid: no message')
    assert not runtime_freshness_plan_request_rejection(
        'planning blocked: controller motion-limit payload is malformed')


def test_multiview_request_holds_then_snapshots_after_measured_lock_recovers(
        monkeypatch):
    responses = [
        SimpleNamespace(
            accepted=False, request_id='',
            message='planning blocked: tracking is not settled TRACKING'),
        SimpleNamespace(
            accepted=True, request_id='fresh-plan', message='queued'),
    ]
    progress = []
    fake = SimpleNamespace(
        plan_client=object(),
        call_service=lambda *_args, **_kwargs: responses.pop(0),
        startup_progress=lambda *_args: progress.append(_args[-1]),
        guard=lambda *_args: None,
    )
    monkeypatch.setattr(mission_node.time, 'sleep', lambda _seconds: None)

    request_id = TargetScanMissionNode.request_multiview_plan(
        fake, object(), object())

    assert request_id == 'fresh-plan'
    assert not responses
    assert progress == [
        'target confidence dipped before scan planning; holding without motion '
        'while perception reacquires a measured lock']


def test_multiview_request_holds_then_retries_after_runtime_refresh(
        monkeypatch):
    responses = [
        SimpleNamespace(
            accepted=False, request_id='',
            message=(
                'planning blocked: controller motion limits are missing or '
                'stale')),
        SimpleNamespace(
            accepted=True, request_id='fresh-plan', message='queued'),
    ]
    progress = []
    fake = SimpleNamespace(
        plan_client=object(),
        call_service=lambda *_args, **_kwargs: responses.pop(0),
        startup_progress=lambda *_args: progress.append(_args[-1]),
        guard=lambda *_args: None,
    )
    monkeypatch.setattr(mission_node.time, 'sleep', lambda _seconds: None)

    request_id = TargetScanMissionNode.request_multiview_plan(
        fake, object(), object())

    assert request_id == 'fresh-plan'
    assert not responses
    assert progress == [
        'runtime telemetry dipped before scan planning; holding without motion '
        'for one fresh snapshot']


def test_multiview_request_retries_live_bridge_obstacle_freshness_wording(
        monkeypatch):
    responses = [
        SimpleNamespace(
            accepted=False, request_id='',
            message='planning blocked: obstacles data is missing or stale'),
        SimpleNamespace(
            accepted=True, request_id='fresh-plan', message='queued'),
    ]
    progress = []
    fake = SimpleNamespace(
        plan_client=object(),
        call_service=lambda *_args, **_kwargs: responses.pop(0),
        startup_progress=lambda *_args: progress.append(_args[-1]),
        guard=lambda *_args: None,
    )
    monkeypatch.setattr(mission_node.time, 'sleep', lambda _seconds: None)

    request_id = TargetScanMissionNode.request_multiview_plan(
        fake, object(), object())

    assert request_id == 'fresh-plan'
    assert not responses
    assert progress == [
        'runtime telemetry dipped before scan planning; holding without motion '
        'for one fresh snapshot']


def test_multiview_approval_holds_then_accepts_after_measured_lock_recovers(
        monkeypatch):
    responses = [
        SimpleNamespace(
            accepted=False,
            message='execution blocked: target_status=LOW_CONFIDENCE'),
        SimpleNamespace(accepted=True, message='approved'),
    ]
    progress = []
    fake = SimpleNamespace(
        approve_client=object(),
        call_service=lambda *_args, **_kwargs: responses.pop(0),
        startup_progress=lambda *_args: progress.append(_args[-1]),
        guard=lambda *_args: None,
    )
    session = SimpleNamespace(
        mission_sha256='mission-sha', return_home_proved=True)
    plan = SimpleNamespace(
        plan_id='plan-1', trajectory_sha256='trajectory-sha',
        plan_kind='MULTIVIEW_SCAN')
    monkeypatch.setattr(mission_node.time, 'sleep', lambda _seconds: None)

    TargetScanMissionNode.approve_plan(fake, object(), session, plan)

    assert not responses
    assert session.return_home_proved is False
    assert progress == [
        'target confidence dipped between scan views; holding without capture '
        'while perception reacquires a measured lock']


def test_shutdown_uses_static_home_until_perception_scene_is_established():
    session = MissionSession(validate_goal_payload(goal(), now_sec=1001.0))
    assert shutdown_uses_startup_home(session)
    session.startup_home_completed = True
    assert shutdown_uses_startup_home(session)
    session.acquisition_attempt = 1
    assert shutdown_uses_startup_home(session)
    session.perception_scene_established = True
    session.pre_home_positions_rad = [0.0] * 6
    assert not shutdown_uses_startup_home(session)


def test_target_drift_requests_new_measured_plan_without_authorizing_motion():
    assert target_drift_requires_replan(
        'target moved 0.015m after planning; refresh the plan')
    assert not target_drift_requires_replan(
        'execution blocked: target tracking is stale')
    assert not target_drift_requires_replan(
        'fresh trajectory validation failed: target left the optical cone')


def test_only_proved_post_capture_safe_view_exhaustion_completes_adaptively():
    reason = (
        'MULTIVIEW_SCAN planning failed: Tesseract proposal rejected: '
        'PLANNING_FAILED: only 0 viewpoints planned; require at least 1 of 1 '
        '(view 7: no finite bounded collision-free IK goal for any roll)')
    complete_history = {
        'sufficient': True,
        'accepted_achieved_views': 7,
    }
    assert REQUIRED_CAPTURES == 8
    assert safe_view_exhaustion_after_capture(
        reason, 7, complete_history)
    assert not safe_view_exhaustion_after_capture(
        reason, 7, dict(complete_history, sufficient=False))
    assert not safe_view_exhaustion_after_capture(
        reason, 7, {'accepted_achieved_views': 6})
    assert not safe_view_exhaustion_after_capture(
        reason, 0, {'accepted_achieved_views': 0})
    assert not safe_view_exhaustion_after_capture(
        'MULTIVIEW_SCAN planning failed: worker timed out',
        7, complete_history)


def test_feature_capture_contract_uses_seed_floor_then_bounded_extras():
    assert MAX_FEATURE_CAPTURES == 24
    assert REQUIRED_CAPTURES == 8
    assert feature_capture_decision(
        7, 8, 24, {'sufficient': True}) == 'CONTINUE'
    assert feature_capture_decision(
        8, 8, 24, {'sufficient': False}) == 'CONTINUE'
    assert feature_capture_decision(
        8, 8, 24, {'sufficient': True}) == 'COMPLETE'
    assert feature_capture_decision(
        19, 8, 24, {'sufficient': True}) == 'COMPLETE'
    assert feature_capture_decision(
        24, 8, 24, {'sufficient': False}) == 'EXHAUSTED'


def test_feature_coverage_uses_persisted_pose_when_transient_history_lags(
        monkeypatch):
    persisted_entries = [
        {'actual_camera_position': {'x': 0.1, 'y': 0.0, 'z': 0.2}},
        {'actual_camera_position': {'x': 0.1, 'y': 0.1, 'z': 0.2}},
    ]
    captured = {}
    monkeypatch.setattr(mission_node, 'persisted_achieved_history', lambda _path: {
        'available': True,
        'entries': persisted_entries,
        'target_center': {'x': 0.4, 'y': 0.0, 'z': 0.0},
        'reason': 'persisted achieved history recovered',
    })
    monkeypatch.setattr(mission_node, 'measured_surface_coverage', lambda *_args, **_kwargs: {
        'sufficient': False,
    })

    def fake_coverage(entries, center, **_kwargs):
        captured['entries'] = entries
        captured['center'] = center
        return {'sufficient': False, 'accepted_achieved_views': len(entries)}

    monkeypatch.setattr(mission_node, 'achieved_feature_coverage', fake_coverage)
    fake = SimpleNamespace(
        latest_scan_history={
            'accepted_entries': [persisted_entries[0]],
            'coverage_target_center': None,
        },
        latest_scan_target_center=None,
        latest_capture={'scan_dir': '/dataset'},
        get_parameter=lambda _name: SimpleNamespace(value=8),
        last_scan_feature_coverage={},
    )

    result = TargetScanMissionNode.current_scan_feature_coverage(fake)

    assert captured['entries'] is persisted_entries
    assert captured['center'] == {'x': 0.4, 'y': 0.0, 'z': 0.0}
    assert result['accepted_achieved_views'] == 2
    assert result['achieved_history_source'] == 'persisted_capture_metadata'


def test_feature_capture_contract_rejects_invalid_bounds():
    with pytest.raises(ValueError, match='bounds'):
        feature_capture_decision(0, 13, 12, {})


def test_rejected_proposal_still_allows_separate_current_state_home_plan():
    assert planning_rejection_allows_current_state_home(
        'MULTIVIEW_SCAN planning failed: Tesseract proposal rejected: '
        'PLANNING_FAILED: only 0 viewpoints planned; require at least 1 of 1 '
        '(view 7: no finite bounded collision-free IK goal for any roll)')
    assert planning_rejection_allows_current_state_home(
        'runtime safety gate: invalid obstacle geometry is present')
    assert planning_rejection_allows_current_state_home(
        'ABORTED: runtime safety gate: invalid obstacle geometry is present; '
        'arm held at the current pose; safety fault forbids automatic home motion')
    assert planning_rejection_allows_current_state_home(
        'ABORTED: fresh runtime telemetry did not arrive within 10 seconds: '
        'obstacles data missing or stale')
    assert not planning_rejection_allows_current_state_home(
        'runtime safety gate: obstacle collision is present')


class _GenerationProcess:
    def __init__(self, returncode):
        self.returncode = returncode

    def poll(self):
        return self.returncode


class _GenerationLog:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_managed_process_generation_discards_only_stopped_history(tmp_path):
    processes = ManagedProcessSet(tmp_path)
    stopped_log = _GenerationLog()
    processes.entries['vision'] = (
        _GenerationProcess(0), stopped_log, str(tmp_path / 'vision.log'))
    processes.log_offsets['vision'] = 41

    assert processes.begin_generation() == []
    assert stopped_log.closed
    assert processes.entries == {}
    assert processes.log_offsets == {}

    live_log = _GenerationLog()
    processes.entries['driver'] = (
        _GenerationProcess(None), live_log, str(tmp_path / 'driver.log'))

    assert processes.begin_generation() == ['driver']
    assert not live_log.closed
    assert 'driver' in processes.entries


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


def test_shutdown_accepts_fresh_feedback_when_already_at_home():
    session = SimpleNamespace(
        return_home_proved=False,
        home_positions_rad=[0.0] * 6,
    )
    node = SimpleNamespace(
        cancel_client=object(),
        call_service=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError('already-home feedback must not command motion')),
        processes=SimpleNamespace(failed=lambda: {}),
        at_configured_home=lambda _session, **_kwargs: True,
    )

    assert TargetScanMissionNode.prove_return_home_for_shutdown(node, session)
    assert session.return_home_proved


def test_shutdown_holds_current_state_then_uses_dedicated_home_plan():
    session = SimpleNamespace(
        return_home_proved=False,
        home_positions_rad=[0.0] * 6,
        task_id='task-direct-home', mission_sha256='a' * 64,
    )
    cancel_client = object()
    execute_client = object()
    state = {'at_home': False, 'approved': False}
    events = []
    node = SimpleNamespace(
        cancel_client=cancel_client,
        execute_home_stage_client=execute_client,
        latest_execution=SimpleNamespace(
            state='ABORTED', reason='approved plan start reached'),
        latest_execution_at=time.monotonic() + 10.0,
        processes=SimpleNamespace(failed=lambda: {}),
        at_configured_home=lambda _session, **_kwargs: state['at_home'],
        require_fresh_joint_feedback=lambda: None,
        wait_for_stable_joint_stream=lambda *_args: None,
        authorize_mission=lambda _session, terminal_home=False: events.append(
            'terminal_authority' if terminal_home else 'mission_authority'),
    )

    def call_service(client, *_args, **_kwargs):
        if client is cancel_client:
            events.append('stop_hold')
            return SimpleNamespace(
                success=True,
                message=('proposal cancelled; current joint hold requested '
                         'before dedicated current-state return-home replanning'))
        assert client is execute_client
        events.append('direct_home')
        state['approved'] = True
        state['at_home'] = True
        node.latest_execution = SimpleNamespace(
            plan_id='direct-home',
            state='ABORTED',
            reason='dedicated collision-qualified configured home reached')
        node.latest_execution_at = time.monotonic() + 10.0
        return SimpleNamespace(
            accepted=True, execution_id='direct-home', message='accepted')
    node.call_service = call_service

    assert TargetScanMissionNode.prove_return_home_for_shutdown(node, session)
    assert state['approved']
    assert session.return_home_proved
    assert events == ['stop_hold', 'terminal_authority', 'direct_home']


def test_terminal_home_authority_outlives_expired_scan_deadline(monkeypatch):
    requests = []
    node = SimpleNamespace(
        authorize_client=object(),
        call_service=lambda _client, request, _timeout, _label: (
            requests.append(request) or SimpleNamespace(
                accepted=True, message='accepted')),
    )
    session = SimpleNamespace(
        task_id='expired-scan', mission_sha256='a' * 64,
        remaining=lambda: -1.0,
    )
    monkeypatch.setattr(mission_node.time, 'time', lambda: 1000.0)

    TargetScanMissionNode.authorize_mission(
        node, session, terminal_home=True)

    assert len(requests) == 1
    expiry = (
        float(requests[0].expires_at.sec)
        + float(requests[0].expires_at.nanosec) * 1e-9)
    assert expiry == pytest.approx(1207.0)
    assert not requests[0].revoke


def test_startup_home_uses_nonterminal_hold_and_static_plan_service():
    session = SimpleNamespace(
        return_home_proved=False,
        home_positions_rad=[0.0] * 6,
        task_id='task-startup-home', mission_sha256='b' * 64,
    )
    execute_client = object()
    state = {'at_home': False, 'approved': False}
    node = SimpleNamespace(
        execute_home_stage_client=execute_client,
        latest_execution=None,
        latest_execution_at=0.0,
        processes=SimpleNamespace(failed=lambda: {}),
        at_configured_home=lambda _session, **_kwargs: state['at_home'],
        require_fresh_joint_feedback=lambda: None,
        guard=lambda *_args: None,
    )

    def call_service(client, *_args, **_kwargs):
        assert client is execute_client
        state['approved'] = True
        state['at_home'] = True
        node.latest_execution = SimpleNamespace(
            plan_id='direct-startup-home',
            state='ABORTED', reason='configured home reached')
        node.latest_execution_at = time.monotonic() + 10.0
        return SimpleNamespace(
            accepted=True, execution_id='direct-startup-home',
            message='accepted')
    node.call_service = call_service

    assert TargetScanMissionNode.prove_return_home_for_shutdown(
        node, session, startup=True, goal_handle=object())
    assert state['approved']
    assert session.return_home_proved


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


def test_blank_profile_selects_open_vocabulary_for_arbitrary_label():
    payload = goal(profile='')
    payload['target_label'] = 'Red sphere | hand .'
    normalized = validate_goal_payload(payload, now_sec=1001.0)
    assert normalized['target_profile'] == 'generic_open_vocab'
    assert normalized['target_prompt'] == 'red sphere hand .'
    assert '|' not in normalized['target_prompt']


def test_blank_profile_keeps_strict_green_cube_baseline():
    normalized = validate_goal_payload(goal(profile=''), now_sec=1001.0)
    assert normalized['target_profile'] == 'green_cube'
    assert normalized['target_prompt'] == 'green cube .'


def test_sensor_timeout_retains_an_actionable_failure_code():
    assert failure_code_for_reason(
        'camera vision startup timed out') == 'SENSOR_UNAVAILABLE'
    assert failure_code_for_reason(
        'mission deadline expired') == 'DEADLINE_EXPIRED'
    assert failure_code_for_reason(
        'occlusion scene contains unsafe obstacle geometry') \
        == 'OCCLUSION_NOT_CLEARED'
    assert failure_code_for_reason(
        'request creation failed: no safe scan candidate lies within the '
        'closed-loop view frontier') == 'NO_REACHABLE_PLAN'
    assert failure_code_for_reason(
        'CAN bus feedback is unavailable') == 'CONTROL_UNTRUSTWORTHY'


def test_every_bounded_startup_wait_observes_mission_cancellation(tmp_path):
    def cancelled(_goal, _session):
        raise MissionFailure('tracked robot cancelled the task')

    node = SimpleNamespace(guard=cancelled)
    goal, session = object(), object()
    waits = (
        lambda: TargetScanMissionNode.wait_for_stable_joint_stream(
            node, 1.0, 1.0, 'feedback', goal, session),
        lambda: TargetScanMissionNode.wait_for_vision_boot(
            node, 0.0, 1.0, goal, session),
        lambda: TargetScanMissionNode.wait_for_hand_eye_boot(
            node, 0, 1.0, goal, session),
        lambda: TargetScanMissionNode.wait_for_worker_boot(
            node, tmp_path / 'health.json', '', 1.0, goal, session),
    )
    for wait in waits:
        with pytest.raises(MissionFailure, match='cancelled'):
            wait()


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


def test_pending_missions_choose_closest_target_then_arrival_order():
    farther = validate_goal_payload(goal('scan-task-0002'), now_sec=1001.0)
    closer = validate_goal_payload(goal('scan-task-0003'), now_sec=1001.0)
    equal = validate_goal_payload(goal('scan-task-0004'), now_sec=1001.0)
    farther['rough_target']['position'] = [0.8, 0.0, 0.0]
    closer['rough_target']['position'] = [0.3, 0.0, 0.0]
    equal['rough_target']['position'] = [0.3, 0.0, 0.0]
    records = [
        {'normalized': farther, 'sequence': 0},
        {'normalized': equal, 'sequence': 2},
        {'normalized': closer, 'sequence': 1},
    ]

    selected = closest_pending_mission(records)

    assert selected['normalized']['task_id'] == 'scan-task-0003'
    assert mission_target_distance_m(farther) == pytest.approx(0.8)


def test_pending_mission_distance_rejects_nonfinite_geometry():
    pending = validate_goal_payload(goal(), now_sec=1001.0)
    pending['rough_target']['position'] = [0.3, float('nan'), 0.0]

    with pytest.raises(ValueError, match='finite XYZ'):
        mission_target_distance_m(pending)


def test_pending_queue_waits_for_short_coalescing_window():
    pending = {'admitted_monotonic': 10.0}

    assert not mission_queue_ready([pending], 10.99, 1.0)
    assert mission_queue_ready([pending], 11.0, 1.0)


def test_queued_cancel_is_terminal_without_claiming_arm_ownership():
    normalized = validate_goal_payload(goal(), now_sec=1001.0)

    result = queued_cancel_result(normalized)

    assert result['outcome'] == 'CANCELLED'
    assert result['safe_shutdown'] is True
    assert result['action_summary']['arm_resources_started'] is False


def test_mission_dispatch_selects_nearest_without_overlapping_active_work():
    farther = validate_goal_payload(goal('scan-task-0002'), now_sec=1001.0)
    nearer = validate_goal_payload(goal('scan-task-0003'), now_sec=1001.0)
    farther['rough_target']['position'] = [0.8, 0.0, 0.0]
    nearer['rough_target']['position'] = [0.3, 0.0, 0.0]
    executed = []

    def handle(task_id):
        return SimpleNamespace(
            is_cancel_requested=False,
            execute=lambda: executed.append(task_id))

    records = {
        farther['task_id']: {
            'normalized': farther, 'sequence': 1,
            'admitted_monotonic': 0.0, 'source': 'action',
            'goal_handle': handle(farther['task_id']),
        },
        nearer['task_id']: {
            'normalized': nearer, 'sequence': 2,
            'admitted_monotonic': 0.0, 'source': 'action',
            'goal_handle': handle(nearer['task_id']),
        },
    }
    harness = SimpleNamespace(
        discover_spool_goals=lambda: None,
        _lock=threading.RLock(),
        _pending_missions=records,
        _process_shutdown_requested=False,
        _prevalidated_goals={},
        _dispatch_task_id='',
        registry=SimpleNamespace(active=None),
        spool=SimpleNamespace(read=lambda *_args: {}),
        write_queued_status=lambda _record: None,
        get_parameter=lambda name: SimpleNamespace(
            value=0.0 if name == 'mission_queue_coalesce_sec' else 8),
        finish_queued_cancel=lambda *_args: None,
    )

    TargetScanMissionNode.poll_mission_queue(harness)

    assert executed == [nearer['task_id']]
    assert harness._dispatch_task_id == nearer['task_id']
    assert list(harness._pending_missions) == [farther['task_id']]


def test_non_client_shutdown_aborts_action_instead_of_invalid_cancel_transition():
    transitions = []
    handle = SimpleNamespace(
        is_cancel_requested=False,
        canceled=lambda: transitions.append('canceled'),
        abort=lambda: transitions.append('aborted'))

    TargetScanMissionNode.finish_action_handle(handle, 'CANCELLED')

    assert transitions == ['aborted']


def test_client_cancel_uses_ros_canceled_transition():
    transitions = []
    handle = SimpleNamespace(
        is_cancel_requested=True,
        canceled=lambda: transitions.append('canceled'),
        abort=lambda: transitions.append('aborted'))

    TargetScanMissionNode.finish_action_handle(handle, 'CANCELLED')

    assert transitions == ['canceled']


def test_link_loss_and_deadline_are_bounded():
    session = MissionSession(validate_goal_payload(goal(), now_sec=1001.0))
    session.started_monotonic = 10.0
    session.heartbeat_monotonic = 10.0
    assert session.heartbeat_stale(15.01)
    assert session.deadline_expired(1210.01)


def test_disable_never_counts_as_safe_without_verified_home():
    session = MissionSession(validate_goal_payload(goal(), now_sec=1001.0))
    session.disabled_proved = True
    session.processes_stopped = True
    phase, reason = session.shutdown_outcome()
    assert phase == MissionPhase.NEEDS_OPERATOR
    assert 'home return' in reason
    session.return_home_proved = True
    session.storage_wrist_proved = True
    result = session.result_payload('FAILED', 'planned failure')
    assert result['safe_shutdown']
    assert not session.current_hold_proved
    assert len(result['result_sha256']) == 64


def test_shutdown_never_calls_disable_before_home_is_proved():
    session = MissionSession(validate_goal_payload(goal(), now_sec=1001.0))
    session.arm_enabled = True
    session.startup_home_completed = True
    session.perception_scene_established = True
    session.pre_home_positions_rad = [0.0] * 6
    session.storage_positions_rad = [0.0] * 6
    calls = []
    harness = SimpleNamespace(
        transition=lambda *_args, **_kwargs: None,
        prove_return_home_for_shutdown=lambda _session, **_kwargs: False,
        prove_current_hold_for_shutdown=lambda _session: True,
        call_enable=lambda enabled: calls.append(bool(enabled)),
    )

    failure = TargetScanMissionNode.safe_shutdown(
        harness, session, normal_completion=False)

    assert 'pre-home was not proved' in str(failure)
    assert calls == []
    assert session.arm_enabled


def test_cancel_shutdown_returns_home_before_disable_and_process_stop():
    session = MissionSession(validate_goal_payload(goal(), now_sec=1001.0))
    session.arm_enabled = True
    session.startup_home_completed = True
    session.perception_scene_established = True
    session.pre_home_positions_rad = [0.0] * 6
    session.storage_positions_rad = [0.0] * 6
    calls = []

    def prove_home(bound_session, **kwargs):
        stage = kwargs.get('home_stage', 'ROUGH_HOME')
        calls.append({
            'PRE_HOME': 'pre_home',
            'STORAGE_WRIST': 'storage',
        }.get(stage, 'home'))
        if stage == 'PRE_HOME':
            bound_session.pre_home_completed = True
        else:
            bound_session.return_home_proved = True
        if stage == 'STORAGE_WRIST':
            bound_session.storage_wrist_proved = True
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
    assert calls[:4] == [
        'pre_home', 'home', 'storage', ('enable', False)]
    assert 'hold' not in calls
    assert calls[-1] == 'stop'
    assert session.disabled_proved and session.processes_stopped


def test_pre_acquisition_shutdown_reuses_static_startup_home_authority():
    session = MissionSession(validate_goal_payload(goal(), now_sec=1001.0))
    session.arm_enabled = True
    session.pre_home_positions_rad = [0.0] * 6
    calls = []

    def prove_home(_session, startup=False, **_kwargs):
        calls.append(bool(startup))
        return False

    harness = SimpleNamespace(
        transition=lambda *_args, **_kwargs: None,
        prove_return_home_for_shutdown=prove_home,
        last_return_home_diagnostic='startup home plan was unavailable',
        current_home_profile={
            'positions_rad': [0.0] * 6,
            'pre_home_configured': True,
            'pre_home_positions_rad': [0.0] * 6,
            'mission_ready_joint6_rad': 0.0,
            'storage_joint6_rad': -3.13,
            'staged_home_configured': True,
            'startup_wrist_direction': 'increasing',
            'storage_wrist_direction': 'decreasing',
        },
        latest_joints=SimpleNamespace(
            position=[0.0, 0.0, 0.0, 0.0, 0.0, -3.13]),
    )

    failure = TargetScanMissionNode.safe_shutdown(
        harness, session, normal_completion=False,
        failure=RuntimeError('startup approval was transiently blocked'))

    assert calls == [True]
    assert 'startup home plan was unavailable' in str(failure)
    assert session.arm_enabled


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
    assert "sigterm_timeout='180'" in launch_source


def test_supported_mission_entrypoint_holds_singleton_lock_for_launch_lifetime():
    source = (
        Path(__file__).resolve().parents[4]
        / 'run_target_scan_mission.sh'
    ).read_text(encoding='utf-8')
    assert 'exec 9>"$MISSION_LOCK_FILE"' in source
    assert 'flock -n 9' in source
    assert 'exit 73' in source


def test_process_shutdown_request_is_treated_as_mission_cancellation():
    session = MissionSession(validate_goal_payload(goal(), now_sec=1001.0))
    harness = SimpleNamespace(
        latest_capture={},
        _process_shutdown_requested=True,
    )

    with pytest.raises(MissionFailure) as raised:
        TargetScanMissionNode.guard(harness, None, session)

    assert raised.value.outcome == 'CANCELLED'
    assert raised.value.failure_code == 'CANCELLED'
    assert raised.value.retryable


def test_startup_home_returns_immediately_on_non_success_abort():
    calls = []

    def call_service(client, _request, _timeout, _label):
        calls.append(client)
        return SimpleNamespace(
            accepted=True, execution_id='startup-home-request',
            message='accepted')

    harness = SimpleNamespace(
        last_return_home_diagnostic='',
        at_configured_home=lambda _session, **_kwargs: False,
        call_service=call_service,
        execute_home_stage_client='direct_home',
        require_fresh_joint_feedback=lambda: None,
        latest_execution=SimpleNamespace(
            plan_id='startup-home-request',
            state='ABORTED',
            reason='return-home runtime safety gate stopped motion'),
        latest_execution_at=float('inf'),
        processes=SimpleNamespace(failed=lambda: {}),
    )
    session = SimpleNamespace(
        return_home_proved=False, home_positions_rad=[0.0] * 6,
        task_id='task-startup-abort', mission_sha256='d' * 64)

    harness.guard = lambda *_args: None
    proved = TargetScanMissionNode.prove_return_home_for_shutdown(
        harness, session, startup=True, goal_handle=object())

    assert not proved
    assert 'home execution aborted' in harness.last_return_home_diagnostic
    assert calls == ['direct_home']


def test_transient_invalid_scene_attempts_fresh_qualified_home():
    session = MissionSession(validate_goal_payload(goal(), now_sec=1001.0))
    session.arm_enabled = True
    session.startup_home_completed = True
    session.perception_scene_established = True
    session.pre_home_positions_rad = [0.0] * 6
    session.storage_positions_rad = [0.0] * 6
    calls = []

    def prove_home(bound_session, **kwargs):
        stage = kwargs.get('home_stage', 'ROUGH_HOME')
        calls.append({
            'PRE_HOME': 'pre_home',
            'STORAGE_WRIST': 'storage',
        }.get(stage, 'home'))
        if stage == 'PRE_HOME':
            bound_session.pre_home_completed = True
        else:
            bound_session.return_home_proved = True
        if stage == 'STORAGE_WRIST':
            bound_session.storage_wrist_proved = True
        return True

    harness = SimpleNamespace(
        transition=lambda *_args, **_kwargs: None,
        prove_return_home_for_shutdown=prove_home,
        prove_current_hold_for_shutdown=lambda _session: True,
        call_enable=lambda enabled: calls.append(('enable', bool(enabled))),
        authorize_mission=lambda *_args, **_kwargs: None,
        processes=SimpleNamespace(stop_all=lambda: True),
    )

    failure = TargetScanMissionNode.safe_shutdown(
        harness, session, normal_completion=False,
        failure=RuntimeError(
            'runtime safety gate: invalid obstacle geometry is present'))

    assert failure is None
    assert calls == ['pre_home', 'home', 'storage', ('enable', False)]
    assert not session.arm_enabled


def test_collision_failure_still_attempts_direct_configured_home():
    session = MissionSession(validate_goal_payload(goal(), now_sec=1001.0))
    session.arm_enabled = True
    session.startup_home_completed = True
    session.perception_scene_established = True
    session.pre_home_positions_rad = [0.0] * 6
    session.storage_positions_rad = [0.0] * 6
    calls = []

    def prove_home(bound_session, **kwargs):
        stage = kwargs.get('home_stage', 'ROUGH_HOME')
        calls.append({
            'PRE_HOME': 'pre_home',
            'STORAGE_WRIST': 'storage',
        }.get(stage, 'home'))
        if stage == 'PRE_HOME':
            bound_session.pre_home_completed = True
        else:
            bound_session.return_home_proved = True
        if stage == 'STORAGE_WRIST':
            bound_session.storage_wrist_proved = True
        return True

    harness = SimpleNamespace(
        transition=lambda *_args, **_kwargs: None,
        prove_return_home_for_shutdown=prove_home,
        prove_current_hold_for_shutdown=lambda _session: True,
        call_enable=lambda enabled: calls.append(('enable', bool(enabled))),
        authorize_mission=lambda *_args, **_kwargs: None,
        processes=SimpleNamespace(stop_all=lambda: True),
    )

    failure = TargetScanMissionNode.safe_shutdown(
        harness, session, normal_completion=False,
        failure=RuntimeError('obstacle collision is present'))

    assert failure is None
    assert calls == ['pre_home', 'home', 'storage', ('enable', False)]
    assert not session.arm_enabled


def motor_status(states, faults=(), watchdog=''):
    payload = {
        'motor_feedback_valid': True,
        'motor_faults': list(faults),
        'motor_watchdog_reason': watchdog,
    }
    payload.update({
        'motor_%d_driver_enabled' % index: bool(states[index - 1])
        for index in range(1, 7)
    })
    return SimpleNamespace(**payload)


def test_mission_motor_guard_detects_joint5_drop_before_home(monkeypatch):
    session = MissionSession(validate_goal_payload(goal(), now_sec=1001.0))
    session.arm_enabled = True
    harness = SimpleNamespace(
        _lock=threading.RLock(),
        latest_arm_status=motor_status(
            [True, True, True, True, False, True]),
        latest_arm_status_at=10.0,
        motor_enable_guard_after=0.0,
    )
    monkeypatch.setattr(mission_node.time, 'monotonic', lambda: 10.1)

    with pytest.raises(MissionFailure, match='automatic home is forbidden'):
        TargetScanMissionNode.guard_motor_control(harness, session)

    assert 'partial motor enable' in session.motor_control_lost_reason


def test_motor_loss_shutdown_never_commands_home_and_stops_after_six_disabled(
        monkeypatch):
    session = MissionSession(validate_goal_payload(goal(), now_sec=1001.0))
    session.arm_enabled = True
    session.motor_control_lost_reason = (
        'partial motor enable flags=(True, True, True, True, False, True)')
    calls = []
    harness = SimpleNamespace(
        _lock=threading.RLock(),
        latest_arm_status=motor_status([False] * 6),
        latest_arm_status_at=10.0,
        authorize_mission=lambda *_args, **_kwargs: calls.append('revoke'),
        transition=lambda *_args, **_kwargs: calls.append('stopping'),
        processes=SimpleNamespace(
            stop_all=lambda: calls.append('stop_all') or True),
    )
    monkeypatch.setattr(mission_node.time, 'monotonic', lambda: 10.1)

    failure = TargetScanMissionNode.safe_shutdown(
        harness, session, normal_completion=False,
        failure=RuntimeError('planning was interrupted'))

    assert isinstance(failure, MissionFailure)
    assert 'no home command was attempted' in str(failure)
    assert calls == ['revoke', 'stopping', 'stop_all']
    assert session.disabled_proved
    assert session.processes_stopped
    assert not session.return_home_proved


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
    package_source = inspect.getsource(MissionEngine)
    lock_wait = package_source.index(
        "'measured target lock ready; waiting for stable multiview readiness'")
    readiness_wait = package_source.index(
        "context, 'multiview',")
    plan_request = package_source.index(
        "'requesting one correlated feature-driven view; up to %d '")

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
