"""Pure, no-hardware characterization tests for the Phase 6 mission engine."""

from types import SimpleNamespace

import pytest

from piper_mobile_manipulation.mission_core import (
    MissionPhase,
    MissionSession,
    validate_goal_payload,
)
from piper_mobile_manipulation.mission_engine import (
    CancellationToken,
    MissionContext,
    MissionEngine,
    MissionFailure,
)


LEGACY_SUCCESS_PHASES = (
    'STARTING',
    'PREFLIGHT',
    'ENABLE_AND_HOLD',
    'RETURNING_HOME',
    'RETURNING_HOME',
    'ROUGH_ACQUISITION',
    'TARGET_LOCK',
    'OCCLUSION_PROBE',
    *('VIEW_PLANNING', 'CAPTURING') * 8,
    'RETURNING_HOME',
    'RETURNING_HOME',
    'RETURNING_HOME',
    'HOLDING',
    'DISABLING',
    'STOPPING',
)


def _session(task_id='phase6-scan-0001'):
    covariance = [0.0] * 36
    covariance[0] = covariance[7] = covariance[14] = 0.01
    goal = validate_goal_payload({
        'task_id': task_id,
        'task_type': 'SCAN_3D',
        'target_label': 'green cube',
        'target_profile': 'green_cube',
        'target_confidence': 0.8,
        'deadline_sec': 1200.0,
        'rough_target': {
            'frame_id': 'base_link',
            'stamp_sec': 1000.0,
            'position': [0.4, 0.0, 0.0],
            'covariance': covariance,
        },
    }, now_sec=1001.0)
    return MissionSession(goal, started_monotonic=0.0)


class FakeMissionOperations:
    """Deterministic application adapter with no ROS, GPU, or hardware."""

    def __init__(self):
        self.events = []
        self.phase_trace = []
        self.failure_stage = ''
        self.cancel_stage = ''
        self.terminal_cancel_stage = ''
        self._triggered = set()
        self.capture_states = []
        self.acquisition_states = ['ACQUIRED']
        self.workflow_state = 'SCAN_READY'
        self.target_drift_once = False
        self._target_drift_done = False
        self._captures = {}
        self.plan_requests = 0
        self.options = {
            'enable_real_arm_motion': True,
            'motion_speed_profile_qualified': True,
            'free_motion_speed_percent': 30.0,
            'contact_speed_percent': 10.0,
            'required_captures': 8,
            'maximum_captures': 24,
        }

    def _record(self, stage, context, cancellable=True):
        self.events.append(stage)
        if stage == self.failure_stage and stage not in self._triggered:
            self._triggered.add(stage)
            raise MissionFailure('%s failed' % stage)
        if (
                cancellable
                and stage == self.cancel_stage
                and stage not in self._triggered):
            self._triggered.add(stage)
            context.cancellation.cancel(
                'tracked robot cancelled during ' + stage)
            raise MissionFailure(
                context.cancellation.reason,
                outcome='CANCELLED', failure_code='CANCELLED',
                retryable=True)

    def _capture_count(self, context):
        return self._captures.setdefault(context.session.task_id, 0)

    def begin_process_generation(self, context):
        self._record('process_startup', context)
        self._captures[context.session.task_id] = 0
        return []

    def snapshot_target(self, context):
        self._record('target_snapshot', context)
        return [0.4, 0.0, 0.0]

    def transition(self, context, phase, _reason):
        context.session.transition(phase, str(_reason), now=0.0)
        self.phase_trace.append(MissionPhase(phase).value)
        if (
                MissionPhase(phase) is MissionPhase.HOLDING
                and self.terminal_cancel_stage == 'holding'):
            context.cancellation.cancel('cancel during holding')

    def progress(self, _context, _reason):
        pass

    def selected_home_profile(self, _context):
        return {
            'positions_rad': [0.0, 0.0, 0.0, 0.0, 0.399345492, 0.0],
            'mission_ready_joint6_rad': 0.0,
            'storage_joint6_rad': -3.139536232,
            'pre_home_configured': True,
            'pre_home_positions_rad': [
                0.0, 0.4, -0.5, 0.0, 0.6, 0.0],
            'staged_home_configured': True,
            'startup_wrist_direction': 'increasing',
            'storage_wrist_direction': 'decreasing',
        }

    def bind_home_profile(self, _context, profile):
        self.profile = profile

    def current_home_profile(self, _context):
        return self.profile

    def start_processes(self, context):
        self._record('starting', context)

    def wait_for_enable_service(self, context, _timeout):
        self._record('enable_service', context)

    def wait_for_stable_readiness(
            self, context, mode, _stable, _timeout):
        self._record('%s_readiness' % mode, context)

    def wait_for_stable_joint_stream(
            self, context, _stable, _timeout, _label):
        self._record('joint_stream', context)

    def require_fresh_joint_feedback(self, context):
        self._record('preflight', context)

    @staticmethod
    def current_joint_positions(_context):
        return [0.0] * 6

    def boolean_option(self, _context, name):
        return bool(self.options[name])

    def numeric_option(self, _context, name):
        return self.options[name]

    def authorize_mission(self, context, revoke=False):
        self._record(
            'authority_revoke' if revoke else 'authority_grant',
            context, cancellable=not revoke)

    def enable_arm(self, context, enabled):
        stage = 'disable' if not enabled else 'enable'
        self._record(stage, context, cancellable=enabled)
        if not enabled and self.terminal_cancel_stage == 'disable':
            context.cancellation.cancel('cancel during disable')

    def arm_enable_guard_started(self, _context):
        pass

    def prove_current_hold(self, context):
        self._record('enable_hold', context)
        context.session.current_hold_proved = True
        return True

    @staticmethod
    def hold_diagnostic(_context):
        return 'fake hold diagnostic'

    def prove_home(
            self, context, startup=False, target_positions=None,
            home_stage='ROUGH_HOME', interruptible=False):
        del target_positions, interruptible
        if startup and not context.session.startup_home_completed:
            stage = (
                'startup_wrist' if home_stage == 'STARTUP_WRIST'
                else 'startup_home')
            self._record(stage, context)
        else:
            stage = {
                'PRE_HOME': 'pre_home',
                'STORAGE_WRIST': 'storage_wrist',
            }.get(home_stage, 'return_home')
            self._record(stage, context, cancellable=False)
            if self.terminal_cancel_stage == stage:
                context.cancellation.cancel('cancel during ' + stage)
        if home_stage == 'STORAGE_WRIST':
            context.session.storage_wrist_proved = True
        elif home_stage == 'PRE_HOME':
            context.session.pre_home_completed = True
        else:
            context.session.return_home_proved = True
        return True

    @staticmethod
    def return_home_diagnostic(_context):
        return 'fake home diagnostic'

    def clear_plan_cache(self, _context):
        pass

    def prepare_acquisition(self, context):
        self._record('acquisition', context)
        return 'acquisition-%d' % context.session.acquisition_attempt

    def wait_for_plan(self, context, kind, request_id, _timeout):
        del request_id
        stage = 'acquisition_plan' if kind == 'ROUGH_ACQUISITION' else 'planning'
        self._record(stage, context)
        return SimpleNamespace(
            plan_kind=kind, plan_id='plan', trajectory_sha256='a' * 64)

    def approve_plan(self, context, plan):
        if (
                plan.plan_kind == 'MULTIVIEW_SCAN'
                and self.target_drift_once
                and not self._target_drift_done):
            self._target_drift_done = True
            raise MissionFailure('target moved 0.015m after planning; refresh the plan')
        context.session.return_home_proved = False

    def wait_for_execution(
            self, context, successes, _timeout, _failures):
        if 'ACQUIRED' in successes:
            self._record('target_lock', context)
            state = (
                self.acquisition_states.pop(0)
                if self.acquisition_states else 'ACQUIRED')
            return SimpleNamespace(state=state, reason='acquisition result')
        self._record('capture', context)
        state = (
            self.capture_states.pop(0)
            if self.capture_states else 'VIEW_COMPLETE')
        if state == 'VIEW_COMPLETE':
            self._captures[context.session.task_id] += 1
        return SimpleNamespace(state=state, reason='view result')

    def start_and_wait_workflow(self, context):
        self._record('occlusion', context)
        return {'state': self.workflow_state, 'measured_lock_ready': True}

    @staticmethod
    def readiness_rejection(_context, mode):
        return 'manipulation unavailable' if mode == 'manipulation' else ''

    def capture_count(self, context):
        return self._capture_count(context)

    def current_feature_coverage(self, context):
        count = self._capture_count(context)
        sufficient = count >= int(self.options['required_captures'])
        return {
            'sufficient': sufficient,
            'accepted_achieved_views': count,
            'blockers': [] if sufficient else ['coverage incomplete'],
        }

    def request_multiview_plan(self, context):
        self._record('view_planning', context)
        self.plan_requests += 1
        return 'view-%d' % self.plan_requests

    def wait_for_view_generation(self, context, accepted_views, _timeout):
        assert accepted_views == self._capture_count(context)
        self._record('view_generation', context)

    @staticmethod
    def remaining_time(context):
        return context.session.remaining(now=1.0)

    def wait_for_scan_history(self, _context, _timeout):
        pass

    @staticmethod
    def wait_for_all_motors_disabled(_context, _timeout):
        return True

    def stop_processes(self, context):
        self._record('stopping', context, cancellable=False)
        if self.terminal_cancel_stage == 'stopping':
            context.cancellation.cancel('cancel during stopping')
        return True

    def stop_processing_processes(self, context):
        self._record('processing_stop', context, cancellable=False)
        return True

    @staticmethod
    def abort_return_home_blocker(_context, _failure):
        return ''

    def prove_shutdown_hold(self, context):
        self._record('holding', context, cancellable=False)
        if self.terminal_cancel_stage == 'holding':
            context.cancellation.cancel('cancel during holding')
        context.session.current_hold_proved = True
        return True


def _execute(operations=None, task_id='phase6-scan-0001'):
    operations = operations or FakeMissionOperations()
    context = MissionContext(
        session=_session(task_id), cancellation=CancellationToken())
    return operations, context, MissionEngine(operations).execute(context)


def test_successful_engine_matches_frozen_legacy_phase_sequence():
    operations, context, result = _execute()

    assert result.succeeded
    assert result.outcome == 'SUCCEEDED'
    assert result.phase_sequence == LEGACY_SUCCESS_PHASES
    assert tuple(operations.phase_trace) == LEGACY_SUCCESS_PHASES
    assert context.session.accepted_captures == 8
    assert context.session.current_hold_proved
    assert context.session.pre_home_completed
    assert context.session.return_home_proved
    assert context.session.storage_wrist_proved
    assert context.session.disabled_proved
    assert context.session.processes_stopped
    assert operations.events.count('view_generation') == 8
    assert operations.events.index('view_generation') < \
        operations.events.index('view_planning')


@pytest.mark.parametrize('stage', [
    'starting',
    'preflight',
    'enable',
    'startup_home',
    'acquisition',
    'target_lock',
    'occlusion',
    'view_generation',
    'view_planning',
    'capture',
])
def test_major_phase_failures_use_the_existing_shutdown(stage):
    operations = FakeMissionOperations()
    operations.failure_stage = stage

    operations, context, result = _execute(operations)

    assert not result.succeeded
    assert result.outcome in ('FAILED', 'NEEDS_OPERATOR')
    assert context.session.processes_stopped
    if context.session.arm_enabled:
        assert result.outcome == 'NEEDS_OPERATOR'
    else:
        assert context.session.disabled_proved


def test_original_failure_cannot_prevent_fresh_direct_home_qualification():
    class Operations(FakeMissionOperations):
        @staticmethod
        def abort_return_home_blocker(_context, _failure):
            raise AssertionError(
                'the original scan failure must not own direct-home safety')

    operations = Operations()
    operations.failure_stage = 'capture'

    operations, context, result = _execute(operations)

    assert not result.succeeded
    assert context.session.pre_home_completed
    assert context.session.return_home_proved
    assert context.session.storage_wrist_proved
    assert context.session.disabled_proved
    assert operations.events.index('pre_home') < operations.events.index(
        'disable')


def test_autonomous_path_does_not_call_redundant_hold_services():
    class Operations(FakeMissionOperations):
        def prove_current_hold(self, context):
            raise AssertionError('startup hold service must not be called')

        def prove_shutdown_hold(self, context):
            raise AssertionError('shutdown hold service must not be called')

    operations, context, result = _execute(Operations())

    assert result.succeeded
    assert context.session.current_hold_proved
    assert context.session.return_home_proved
    assert context.session.storage_wrist_proved
    assert context.session.disabled_proved
    assert context.session.processes_stopped
    assert operations.phase_trace.index('HOLDING') < \
        operations.phase_trace.index('DISABLING')


def test_confirmed_motor_authority_loss_remains_the_only_no_home_path():
    class Operations(FakeMissionOperations):
        def wait_for_execution(
                self, context, successes, _timeout, _failures):
            if 'ACQUIRED' in successes:
                context.session.motor_control_lost_reason = 'J5 disabled'
                raise MissionFailure(
                    'motor control became untrustworthy',
                    needs_operator=True,
                    failure_code='CONTROL_UNTRUSTWORTHY',
                    retryable=False)
            return super().wait_for_execution(
                context, successes, _timeout, _failures)

    operations, context, result = _execute(Operations())

    assert not result.succeeded
    assert result.outcome == 'NEEDS_OPERATOR'
    assert not any(stage in operations.events for stage in (
        'pre_home', 'return_home', 'storage_wrist'))
    assert context.session.disabled_proved
    assert context.session.processes_stopped


@pytest.mark.parametrize('stage', [
    'starting',
    'preflight',
    'enable',
    'startup_home',
    'acquisition',
    'target_lock',
    'occlusion',
    'view_generation',
    'view_planning',
    'capture',
])
def test_cancellation_at_each_active_major_phase_is_application_level(stage):
    operations = FakeMissionOperations()
    operations.cancel_stage = stage

    operations, context, result = _execute(operations)

    assert context.cancellation.cancelled
    assert result.outcome == 'CANCELLED'
    assert result.failure.failure.code.value == 'CANCELLED'
    assert context.session.processes_stopped


@pytest.mark.parametrize('stage', [
    'pre_home', 'return_home', 'storage_wrist', 'holding', 'disable', 'stopping',
])
def test_terminal_cancellation_does_not_interrupt_committed_shutdown(stage):
    operations = FakeMissionOperations()
    operations.terminal_cancel_stage = stage

    _operations, context, result = _execute(operations)

    assert context.cancellation.cancelled
    assert result.outcome == 'SUCCEEDED'
    assert context.session.disabled_proved
    assert context.session.processes_stopped


@pytest.mark.parametrize('stage', [
    'pre_home', 'return_home', 'storage_wrist', 'disable', 'stopping',
])
def test_each_terminal_failure_retains_needs_operator_policy(stage):
    operations = FakeMissionOperations()
    operations.failure_stage = stage

    _operations, context, result = _execute(operations)

    assert result.outcome == 'NEEDS_OPERATOR'
    assert not result.succeeded
    if stage in ('pre_home', 'return_home', 'storage_wrist'):
        assert context.session.arm_enabled
        assert not context.session.disabled_proved
        assert 'processing_stop' in operations.events
        assert not context.session.processes_stopped


def test_failed_home_releases_processing_but_retains_command_owner():
    operations = FakeMissionOperations()
    operations.failure_stage = 'pre_home'

    operations, context, result = _execute(operations)

    assert result.outcome == 'NEEDS_OPERATOR'
    assert operations.events.count('processing_stop') == 1
    assert 'disable' not in operations.events
    assert context.session.arm_enabled
    assert not context.session.processes_stopped


def test_deadline_expiry_uses_failure_shutdown_path():
    operations = FakeMissionOperations()
    operations.failure_stage = 'view_planning'

    def deadline(stage, context, cancellable=True):
        if stage == 'view_planning':
            raise MissionFailure('mission deadline expired')
        return FakeMissionOperations._record(
            operations, stage, context, cancellable)

    operations._record = deadline
    _operations, context, result = _execute(operations)

    assert result.failure.failure.code.value == 'DEADLINE_EXPIRED'
    assert result.outcome == 'FAILED'
    assert context.session.disabled_proved


def test_visual_rejection_retries_with_a_new_view():
    operations = FakeMissionOperations()
    operations.capture_states = ['VIEW_REJECTED']

    operations, context, result = _execute(operations)

    assert result.succeeded
    assert operations.plan_requests == 9
    assert context.session.accepted_captures == 8
    assert operations.events.count('multiview_readiness') == 1


def test_target_drift_replans_through_command_free_request_without_capture():
    operations = FakeMissionOperations()
    operations.target_drift_once = True

    operations, context, result = _execute(operations)

    assert result.succeeded
    assert operations.plan_requests == 9
    assert context.session.accepted_captures == 8
    assert operations.events.count('multiview_readiness') == 1


def test_acquisition_replans_from_two_absent_looks_then_locks():
    operations = FakeMissionOperations()
    operations.acquisition_states = [
        'ACQUISITION_LOOK_COMPLETE',
        'ACQUISITION_LOOK_COMPLETE',
        'ACQUIRED',
    ]

    operations, context, result = _execute(operations)

    assert result.succeeded
    assert operations.events.count('acquisition') == 3
    assert context.session.acquisition_attempt == 3


def test_repeated_missions_do_not_leak_engine_retry_or_capture_state():
    operations = FakeMissionOperations()
    engine = MissionEngine(operations)
    first = MissionContext(_session('phase6-repeat-0001'), CancellationToken())
    second = MissionContext(_session('phase6-repeat-0002'), CancellationToken())

    first_result = engine.execute(first)
    first_requests = operations.plan_requests
    operations.plan_requests = 0
    second_result = engine.execute(second)

    assert first_result.succeeded and second_result.succeeded
    assert first.session.accepted_captures == 8
    assert second.session.accepted_captures == 8
    assert first_requests == 8
    assert operations.plan_requests == 8
    assert first_result.phase_sequence == second_result.phase_sequence
