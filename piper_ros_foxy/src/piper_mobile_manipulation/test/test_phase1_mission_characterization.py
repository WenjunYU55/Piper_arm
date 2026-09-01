"""Phase 1 characterization of the production mission orchestrator."""

import hashlib
import threading
from types import SimpleNamespace

import pytest

import piper_mobile_manipulation.target_scan_mission_node as mission_node
from piper_mobile_manipulation.mission_core import MissionPhase
from piper_mobile_manipulation.mission_engine import CancellationToken
from piper_mobile_manipulation.target_scan_mission_node import MissionFailure
from piper_mobile_manipulation.target_scan_mission_node import (
    _MissionNodeOperations,
    TargetScanMissionNode,
    previous_generation_cleanup_targets,
)


def _run(harness, goal_handle_factory):
    goal_handle = goal_handle_factory()
    result = harness.execute_cb(goal_handle)
    return goal_handle, result


def _cancel_failure(goal_handle, reason):
    def cancel(_harness, _session):
        goal_handle.is_cancel_requested = True
        return MissionFailure(
            reason,
            outcome='CANCELLED',
            failure_code='CANCELLED',
            retryable=True,
        )

    return cancel


def test_previous_processing_only_generation_is_safe_to_reap():
    live = ('vision', 'hand_eye', 'tesseract_worker', 'scan_stack')

    assert previous_generation_cleanup_targets(live, False) == live


def test_previous_live_driver_requires_fresh_six_disabled_proof():
    live = ('driver', 'vision', 'scan_stack')

    assert previous_generation_cleanup_targets(live, False) == ()
    assert previous_generation_cleanup_targets(live, True) == live


class _GenerationLogger:
    def warn(self, _message):
        pass

    def error(self, _message):
        pass


class _PreviousGenerationProcesses:
    def __init__(self, live, cleanup_complete=True, cleanup_error=None):
        self.live = tuple(live)
        self.cleanup_complete = bool(cleanup_complete)
        self.cleanup_error = cleanup_error
        self.shutdown_calls = []

    def begin_generation(self):
        return list(self.live)

    def shutdown(self, names):
        selected = tuple(names)
        self.shutdown_calls.append(selected)
        if self.cleanup_error is not None:
            raise self.cleanup_error
        remaining = () if self.cleanup_complete else selected
        if self.cleanup_complete:
            self.live = ()
        return SimpleNamespace(
            complete=not remaining,
            still_running=remaining)


def _generation_operations(
        live, all_disabled, cleanup_complete=True, cleanup_error=None):
    processes = _PreviousGenerationProcesses(
        live, cleanup_complete, cleanup_error)
    node = SimpleNamespace(
        processes=processes,
        fresh_all_motors_disabled=lambda: bool(all_disabled),
        get_logger=lambda: _GenerationLogger(),
    )
    operations = _MissionNodeOperations(
        node, goal_handle=None, cancellation=CancellationToken())
    return operations, processes


def test_admission_reaps_exact_previous_processing_handles():
    live = ('vision', 'hand_eye', 'tesseract_worker', 'scan_stack')
    operations, processes = _generation_operations(live, False)

    assert operations.begin_process_generation(None) == []
    assert processes.shutdown_calls == [live]


def test_admission_does_not_signal_live_unproved_driver_generation():
    live = ('driver', 'vision', 'scan_stack')
    operations, processes = _generation_operations(live, False)

    assert operations.begin_process_generation(None) == list(live)
    assert processes.shutdown_calls == []


def test_admission_reaps_disabled_driver_generation_and_reports_survivors():
    live = ('driver', 'vision', 'scan_stack')
    operations, processes = _generation_operations(
        live, True, cleanup_complete=False)

    assert operations.begin_process_generation(None) == list(live)
    assert processes.shutdown_calls == [live]


def test_admission_cleanup_exception_still_blocks_new_generation():
    live = ('vision', 'scan_stack')
    operations, processes = _generation_operations(
        live, False, cleanup_error=PermissionError('signal denied'))

    assert operations.begin_process_generation(None) == list(live)
    assert processes.shutdown_calls == [live]


def test_successful_mission_characterizes_complete_stage_and_shutdown_order(
        mission_harness, goal_handle_factory):
    goal_handle, result = _run(mission_harness, goal_handle_factory)

    assert result['outcome'] == 'SUCCEEDED'
    assert result['failure_code'] == ''
    assert result['capture_count'] == 8
    assert result['safe_shutdown'] is True
    assert result['mesh_job_id']
    assert goal_handle.transitions == ['succeeded']
    assert mission_harness.phase_trace == [
        'STARTING',
        'PREFLIGHT',
        'ENABLE_AND_HOLD',
        'RETURNING_HOME',
        'RETURNING_HOME',
        'ROUGH_ACQUISITION',
        'TARGET_LOCK',
        'OCCLUSION_PROBE',
        *['VIEW_PLANNING', 'CAPTURING'] * 8,
        'RETURNING_HOME',
        'RETURNING_HOME',
        'RETURNING_HOME',
        'HOLDING',
        'DISABLING',
        'STOPPING',
    ]

    events = mission_harness.events
    ordered = [
        'process_startup',
        'readiness:acquisition',
        'phase:PREFLIGHT',
        'authority_grant',
        'arm_enable',
        'startup_wrist',
        'startup_rough_home',
        'target_acquisition_request',
        'target_lock_measurement',
        'occlusion_probe',
        'viewpoint_planning',
        'viewpoint_execution',
        'capture',
        'pre_home',
        'return_home',
        'storage_wrist',
        'motor_disable',
        'authority_revoke',
    ]
    positions = [events.index(item) for item in ordered]
    assert positions == sorted(positions)
    assert mission_harness.processes.events == [
        'begin_generation', 'child_process_termination']
    assert [item[0] for item in mission_harness.spool.writes] == [
        'results', 'mesh_jobs']


def test_mission_engine_is_the_only_production_pipeline_authority(
        mission_harness_factory, goal_handle_factory):
    engine_harness = mission_harness_factory('phase6-engine-compare')
    _goal_handle, engine_result = _run(
        engine_harness,
        lambda: goal_handle_factory('phase6-engine-compare'))

    assert engine_result['outcome'] == 'SUCCEEDED'
    assert not hasattr(TargetScanMissionNode, '_legacy_run_pipeline')
    assert not hasattr(TargetScanMissionNode, '_legacy_safe_shutdown')


@pytest.mark.parametrize(
    'planner_backend, worker_name',
    [('tesseract', 'tesseract_worker'), ('curobo', 'curobo_worker')])
def test_process_startup_order_and_environment_are_characterized(
        tmp_path, monkeypatch, planner_backend, worker_name):
    starts = []
    waits = []

    class Processes:
        def start(self, name, command, environment):
            starts.append((name, command, dict(environment)))

        @staticmethod
        def failed():
            return {}

    values = {
        'project_root': str(tmp_path),
        'manage_processes': True,
        'enable_real_arm_motion': True,
        'free_motion_speed_percent': 30.0,
        'maximum_captures': 24,
        'required_captures': 8,
    }
    calibration = (
        tmp_path / 'L515_camera' / 'calibration' / 'hand_eye'
        / 'session_20260808_straight_mount' / 'calibration_result.yaml')
    calibration.parent.mkdir(parents=True)
    calibration.write_text('status: accepted\n', encoding='utf-8')
    floor_selection = (
        tmp_path / 'piper_ros_foxy/src/piper_mobile_manipulation/config/'
        'collision_environment.yaml')
    floor_selection.parent.mkdir(parents=True)
    floor_selection.write_text(
        'schema_version: 1\nfloor_profile: "ground"\n', encoding='utf-8')
    harness = SimpleNamespace(
        get_parameter=lambda name: SimpleNamespace(value=values[name]),
        param_bool=lambda name: bool(values[name]),
        processes=Processes(),
        configuration=SimpleNamespace(
            process=SimpleNamespace(
                floor_profile='saved', floor_profile_path=''),
            value=lambda name: values[name]),
        _lock=threading.RLock(),
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=123456789)),
        enable_client=SimpleNamespace(service_is_ready=lambda: True),
        guard=lambda *_args: None,
        startup_progress=lambda *_args: None,
        wait_for_stable_joint_stream=lambda *_args: waits.append('joints'),
        wait_for_vision_boot=lambda *_args: waits.append('vision'),
        wait_for_hand_eye_boot=lambda *_args: waits.append('hand_eye'),
        worker_generation=lambda _path: 'old-worker',
        wait_for_worker_boot=lambda *_args: waits.append('worker'),
    )
    session = SimpleNamespace(
        home_positions_rad=(0.0, 0.0, 0.0, 0.0, 0.399345492, 0.0),
        pre_home_positions_rad=(0.0, 0.4, -0.5, 0.0, 0.6, 0.0),
        task_id='phase1-scan-0001',
        mission_sha256='c' * 64,
        goal={
            'target_label': 'green cube',
            'target_profile': 'green_cube',
            'target_prompt': 'green cube .',
            'planner_backend': planner_backend,
        },
    )
    monkeypatch.setattr(mission_node.time, 'sleep', lambda _seconds: None)

    TargetScanMissionNode.start_processes(harness, object(), session)

    assert [item[0] for item in starts] == [
        'driver', 'vision', 'hand_eye', worker_name, 'scan_stack']
    assert waits == ['joints', 'vision', 'hand_eye', 'worker']
    environment = starts[0][2]
    assert environment['PIPER_AUTO_ENABLE'] == 'false'
    assert environment['PIPER_ENABLE_REAL_VIEWPOINT_MOTION'] == '1'
    assert environment['PIPER_VIEWPOINT_MISSION_POLICY'] == '1'
    assert environment['PIPER_VIEWPOINT_CLOSED_LOOP_ONE_VIEW'] == '1'
    assert environment['PIPER_VIEWPOINT_SPEED_PERCENT'] == '30.0'
    assert environment['PIPER_VIEWPOINT_MIN_VIEWS'] == '8'
    assert environment['PIPER_VIEWPOINT_MAX_VIEWS'] == '24'
    assert environment['PIPER_FLOOR_PROFILE'] == 'ground'
    assert environment['PIPER_MISSION_TASK_ID'] == session.task_id
    assert environment['PIPER_MISSION_SHA256'] == session.mission_sha256
    assert environment['PIPER_PLANNER_BACKEND'] == planner_backend
    assert starts[3][1] == [str(
        tmp_path / 'motion_planning' / planner_backend / 'run_worker.sh')]
    assert environment['PIPER_HAND_EYE_CALIBRATION'] == str(calibration)
    assert environment['PIPER_CALIBRATION_SHA256'] == hashlib.sha256(
        calibration.read_bytes()).hexdigest()


def test_occlusion_plan_ready_remains_needs_operator_and_never_contacts_scene(
        mission_harness, goal_handle_factory):
    mission_harness.workflow_state = 'PLAN_READY'

    goal_handle, result = _run(mission_harness, goal_handle_factory)

    assert result['outcome'] == 'NEEDS_OPERATOR'
    assert result['failure_code'] == 'OCCLUSION_NOT_CLEARED'
    assert result['capture_count'] == 0
    assert result['safe_shutdown'] is True
    assert 'viewpoint_execution' not in mission_harness.events
    assert 'contact' not in ' '.join(mission_harness.events).lower()
    assert goal_handle.transitions == ['aborted']


@pytest.mark.parametrize(
    'name,configure,expected_code,expected_outcome,safe_shutdown', [
        (
            'camera unavailable',
            lambda harness: harness.inject(
                'startup', MissionFailure(
                    'vision startup timed out: camera is unavailable')),
            'SENSOR_UNAVAILABLE', 'FAILED', True,
        ),
        (
            'camera stale after enable',
            lambda harness: harness.inject(
                'acquisition', MissionFailure(
                    'camera timestamp health is stale')),
            'SENSOR_UNAVAILABLE', 'FAILED', True,
        ),
        (
            'joint feedback unavailable before enable',
            lambda harness: harness.inject(
                'joint_feedback', MissionFailure(
                    'joint feedback is unavailable')),
            'CONTROL_UNTRUSTWORTHY', 'FAILED', True,
        ),
        (
            'joint feedback stale while powered',
            lambda harness: harness.inject(
                'acquisition_execution', MissionFailure(
                    'joint feedback became invalid during SDK MoveJ')),
            'CONTROL_UNTRUSTWORTHY', 'FAILED', True,
        ),
        (
            'arm status stale while powered',
            lambda harness: harness.inject(
                'acquisition_execution', MissionFailure(
                    'arm status is missing or stale')),
            'CONTROL_UNTRUSTWORTHY', 'FAILED', True,
        ),
        (
            'target lost and reacquisition failed',
            lambda harness: harness.inject(
                'planning_approval', MissionFailure(
                    'measured target lock did not recover during the bounded '
                    'between-view hold',
                    failure_code='TARGET_NOT_FOUND', retryable=True)),
            'TARGET_NOT_FOUND', 'FAILED', True,
        ),
        (
            'no reachable viewpoint',
            lambda harness: harness.inject(
                'planner', MissionFailure(
                    'no safe scan candidate lies within the view frontier')),
            'NO_REACHABLE_PLAN', 'FAILED', True,
        ),
        (
            'planner failure',
            lambda harness: harness.inject(
                'planner', MissionFailure(
                    'MULTIVIEW_SCAN planning failed: Tesseract proposal '
                    'rejected: PLANNING_FAILED')),
            'NO_REACHABLE_PLAN', 'FAILED', True,
        ),
        (
            'trajectory execution failure',
            lambda harness: harness.inject(
                'trajectory_execution', MissionFailure(
                    'ABORTED: trajectory waypoint did not reach target')),
            'CONTROL_UNTRUSTWORTHY', 'FAILED', True,
        ),
        (
            'child process crash',
            lambda harness: harness.inject(
                'startup', MissionFailure(
                    "managed process exited: {'vision': 1}")),
            'SENSOR_UNAVAILABLE', 'FAILED', True,
        ),
        (
            'mission timeout',
            lambda harness: harness.inject(
                'acquisition', MissionFailure('mission deadline expired')),
            'DEADLINE_EXPIRED', 'FAILED', True,
        ),
    ],
)
def test_failure_matrix_preserves_result_and_shutdown_policy(
        name, configure, expected_code, expected_outcome, safe_shutdown,
        mission_harness_factory, goal_handle_factory):
    del name
    harness = mission_harness_factory()
    configure(harness)

    goal_handle, result = _run(harness, goal_handle_factory)

    assert result['failure_code'] == expected_code
    assert result['outcome'] == expected_outcome
    assert result['safe_shutdown'] is safe_shutdown
    assert goal_handle.transitions == ['aborted']
    if safe_shutdown:
        assert 'child_process_termination' in harness.processes.events
    else:
        assert 'motor_disable' not in harness.events


def test_target_not_detected_uses_exactly_five_closed_loop_looks(
        mission_harness, goal_handle_factory):
    mission_harness.acquisition_states = [
        'ACQUISITION_LOOK_COMPLETE'] * 5

    _goal_handle, result = _run(mission_harness, goal_handle_factory)

    assert result['outcome'] == 'FAILED'
    assert result['failure_code'] == 'TARGET_NOT_FOUND'
    assert result['retryable'] is True
    assert result['safe_shutdown'] is True
    assert mission_harness.events.count('target_acquisition_request') == 5
    assert mission_harness.events.count('target_lock_measurement') == 5


def test_fresh_capture_rejections_consume_eight_replans_then_fail(
        mission_harness, goal_handle_factory):
    mission_harness.capture_states = ['VIEW_REJECTED'] * 9

    _goal_handle, result = _run(mission_harness, goal_handle_factory)

    assert result['outcome'] == 'FAILED'
    assert result['failure_code'] == 'MISSION_FAILED'
    assert result['capture_count'] == 0
    assert result['safe_shutdown'] is True
    assert mission_harness.events.count('viewpoint_execution') == 9


def test_capture_limit_fails_when_achieved_coverage_is_insufficient(
        mission_harness, goal_handle_factory):
    mission_harness.required_captures = 8
    mission_harness.maximum_captures = 8
    mission_harness.coverage_sufficient = False

    _goal_handle, result = _run(mission_harness, goal_handle_factory)

    assert result['outcome'] == 'FAILED'
    assert result['failure_code'] == 'INSUFFICIENT_CAPTURE_QUALITY'
    assert result['capture_count'] == 8
    assert result['safe_shutdown'] is True


def test_motion_control_authority_loss_forbids_home_and_service_disable(
        mission_harness, goal_handle_factory):
    def lose_authority(harness, session):
        session.motor_control_lost_reason = 'partial motor enable on joint 5'
        harness.latest_arm_status = harness.disabled_motor_status()
        return MissionFailure(
            'motor control became untrustworthy; automatic home is forbidden',
            needs_operator=True,
            failure_code='CONTROL_UNTRUSTWORTHY',
            retryable=False,
        )

    mission_harness.inject('acquisition', lose_authority)

    _goal_handle, result = _run(mission_harness, goal_handle_factory)

    assert result['outcome'] == 'NEEDS_OPERATOR'
    assert result['failure_code'] == 'CONTROL_UNTRUSTWORTHY'
    assert result['safe_shutdown'] is False
    post_enable = mission_harness.events[
        mission_harness.events.index('arm_enable') + 1:]
    assert 'return_home' not in post_enable
    assert 'storage_wrist' not in post_enable
    assert 'motor_disable' not in post_enable
    assert 'child_process_termination' in mission_harness.processes.events


def test_queued_cancel_never_starts_processes_or_touches_arm(
        mission_harness, goal_handle_factory):
    goal_handle = goal_handle_factory()
    goal_handle.is_cancel_requested = True

    result = mission_harness.execute_cb(goal_handle)

    assert result['outcome'] == 'CANCELLED'
    assert result['safe_shutdown'] is True
    assert result['action_summary']['arm_resources_started'] is False
    assert goal_handle.transitions == ['canceled']
    assert mission_harness.processes.events == []
    assert 'arm_enable' not in mission_harness.events


def test_ros_cancel_callback_forwards_to_application_token():
    token = CancellationToken()
    goal_handle = SimpleNamespace(
        request=SimpleNamespace(task_id='phase6-cancel-0001'))
    harness = SimpleNamespace(
        _lock=threading.RLock(),
        _cancellation_tokens={'phase6-cancel-0001': token},
    )

    TargetScanMissionNode.cancel_cb(harness, goal_handle)

    assert token.cancelled
    assert token.reason == 'tracked robot cancelled the task'


def test_production_guard_uses_token_without_polling_ros_goal_handle():
    token = CancellationToken()
    token.cancel('application cancellation')
    goal_handle = SimpleNamespace()
    session = SimpleNamespace(accepted_captures=0)
    harness = SimpleNamespace(
        telemetry_store=None,
        latest_capture={},
        _process_shutdown_requested=False,
        _active_cancellation_token=token,
    )

    with pytest.raises(MissionFailure) as raised:
        TargetScanMissionNode.guard(harness, goal_handle, session)

    assert raised.value.outcome == 'CANCELLED'
    assert str(raised.value) == 'application cancellation'


@pytest.mark.parametrize(
    'stage,expect_powered_shutdown', [
        ('startup', False),
        ('acquisition_execution', True),
        ('planning_request', True),
        ('trajectory_execution', True),
        ('capture', True),
    ],
)
def test_cancel_during_active_stage_uses_current_terminal_policy(
        stage, expect_powered_shutdown, mission_harness_factory,
        goal_handle_factory):
    harness = mission_harness_factory()
    goal_handle = goal_handle_factory()
    harness.inject(
        stage,
        _cancel_failure(
            goal_handle, 'tracked robot cancelled during ' + stage),
    )

    result = harness.execute_cb(goal_handle)

    assert result['outcome'] == 'CANCELLED'
    assert result['failure_code'] == 'CANCELLED'
    assert result['safe_shutdown'] is True
    assert goal_handle.transitions == ['canceled']
    assert 'child_process_termination' in harness.processes.events
    if expect_powered_shutdown:
        assert 'return_home' in harness.events
        assert 'storage_wrist' in harness.events
        assert 'settled_hold' in harness.events
        assert 'motor_disable' in harness.events
    else:
        assert 'return_home' not in harness.events
        assert 'motor_disable' not in harness.events


def test_cancel_arriving_during_terminal_return_home_does_not_interrupt_home(
        mission_harness, goal_handle_factory):
    goal_handle = goal_handle_factory()
    original = mission_harness.prove_return_home_for_shutdown

    def cancel_during_home(session, **kwargs):
        if (
                not kwargs.get('startup', False)
                and kwargs.get('home_stage', 'ROUGH_HOME') == 'ROUGH_HOME'):
            goal_handle.is_cancel_requested = True
        return original(session, **kwargs)

    mission_harness.prove_return_home_for_shutdown = cancel_during_home

    result = mission_harness.execute_cb(goal_handle)

    assert result['outcome'] == 'SUCCEEDED'
    assert result['safe_shutdown'] is True
    assert goal_handle.is_cancel_requested is True
    assert goal_handle.transitions == ['succeeded']
    assert 'return_home' in mission_harness.events
    assert 'motor_disable' in mission_harness.events


def test_cancel_arriving_during_process_cleanup_does_not_change_result(
        mission_harness, goal_handle_factory):
    goal_handle = goal_handle_factory()
    original = mission_harness.processes.stop_all

    def cancel_during_cleanup():
        goal_handle.is_cancel_requested = True
        return original()

    mission_harness.processes.stop_all = cancel_during_cleanup

    result = mission_harness.execute_cb(goal_handle)

    assert result['outcome'] == 'SUCCEEDED'
    assert result['safe_shutdown'] is True
    assert goal_handle.is_cancel_requested is True
    assert goal_handle.transitions == ['succeeded']


@pytest.mark.parametrize(
    'stage,expected_present,expected_absent', [
        (
            'return_home',
            ('stop_motion', 'settled_hold', 'return_home'),
            ('storage_wrist', 'motor_disable'),
        ),
        (
            'motor_disable',
            ('return_home', 'storage_wrist', 'settled_hold', 'motor_disable'),
            (),
        ),
    ],
)
def test_shutdown_failure_does_not_claim_unproved_later_steps(
        stage, expected_present, expected_absent, mission_harness_factory,
        goal_handle_factory):
    harness = mission_harness_factory()
    harness.inject(stage, MissionFailure('%s failed' % stage))

    _goal_handle, result = _run(harness, goal_handle_factory)

    assert result['outcome'] == 'NEEDS_OPERATOR'
    assert result['safe_shutdown'] is False
    for event in expected_present:
        assert event in harness.events
    for event in expected_absent:
        assert event not in harness.events
    assert 'child_process_termination' not in harness.processes.events


def test_obsolete_shutdown_hold_failure_injection_is_not_a_runtime_gate(
        mission_harness_factory, goal_handle_factory):
    harness = mission_harness_factory()
    harness.inject(
        'shutdown_hold', MissionFailure('obsolete shutdown hold failed'))

    _goal_handle, result = _run(harness, goal_handle_factory)

    assert result['outcome'] == 'SUCCEEDED'
    assert result['safe_shutdown'] is True
    assert 'motor_disable' in harness.events


def test_cleanup_failure_occurs_after_disable_but_is_not_safe_shutdown(
        mission_harness, goal_handle_factory):
    mission_harness.processes.cleanup_succeeds = False

    _goal_handle, result = _run(mission_harness, goal_handle_factory)

    assert result['outcome'] == 'NEEDS_OPERATOR'
    assert result['safe_shutdown'] is False
    assert 'motor_disable' in mission_harness.events
    assert 'child_process_termination' in mission_harness.processes.events
    assert result['action_summary']['processes'] == {
        'fake': {'running': False}}


def test_terminal_phase_is_success_only_after_disable_and_cleanup(
        mission_harness, goal_handle_factory):
    _goal_handle, result = _run(mission_harness, goal_handle_factory)
    cached = mission_harness.registry.results[result['task_id']]

    assert cached['outcome'] == 'SUCCEEDED'
    assert cached['safe_shutdown'] is True
    assert mission_harness.registry.active is None
    assert mission_harness.phase_trace[-1] == MissionPhase.STOPPING.value
