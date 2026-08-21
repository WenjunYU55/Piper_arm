"""Reusable no-hardware doubles for mission characterization tests."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from piper_mobile_manipulation.mission_core import (
    MissionRegistry,
    validate_goal_payload,
)
from piper_mobile_manipulation.target_scan_mission_node import (
    MissionFailure,
    TargetScanMissionNode,
)


class ExecutorRuntimeHarness:
    """Fresh nominal executor snapshot that tests may selectively degrade."""

    def __init__(self):
        self.freshness = {
            'joints': True,
            'arm_status': True,
            'camera_clock': True,
            'obstacles': True,
            'tracking': True,
            'target_status': True,
            'motion_limits': True,
        }
        self.parameters = {
            'motion_limits_timeout_sec': 3.0,
            'joint_feedback_limit_tolerance_rad': 0.005,
            'configured_home_feedback_limit_tolerance_rad': 0.3,
            'max_tracking_measurement_age_sec': 0.75,
            'min_tracking_speed_scale': 0.10,
            'speed_percent': 30.0,
        }
        self.latest_motion_limits = SimpleNamespace(
            valid=True,
            limits_sha256='limits-a',
            max_velocity_rad_s=[1.0] * 6,
            max_acceleration_rad_s2=[1.0] * 6,
        )
        self.runtime_motion_limits_sha256 = 'limits-a'
        self.latest_camera_timestamp_health = SimpleNamespace(
            healthy=True, state='HEALTHY', reason='')
        self.latest_obstacles = SimpleNamespace(
            scene_blocked=False, instances=[], blocking_reason='')
        self.latest_tracking_health = SimpleNamespace(
            lifecycle_state='TRACKING',
            camera_settled=True,
            prediction_only=False,
            measurement_age_sec=0.1,
            recommended_speed_scale=1.0,
        )
        self.latest_target_status = 'LOCKED'
        self.latest_arm_status = None
        self.joint_limits = np.asarray([[-3.2, 3.2]] * 6)
        self.plan_kind = 'MULTIVIEW_SCAN'
        self.plan_execution_speed_percent = 30.0
        self.current_view = 0
        self.plan_paths = []
        self.plan_bootstrap_recovery_end_points = []
        self.plan_bootstrap_recovery_joint_sets = []
        self.logged = []

    def fresh(self, key, _timeout=None):
        return bool(self.freshness.get(key, False))

    def get_parameter(self, name):
        return SimpleNamespace(value=self.parameters[name])

    @staticmethod
    def current_joints():
        return np.zeros(6)

    @staticmethod
    def arm_status_reasons():
        return []

    @staticmethod
    def is_configured_home_direct():
        return False

    @staticmethod
    def is_acquisition():
        return False

    @staticmethod
    def joints_settled(settle_at_current=False):
        del settle_at_current
        return True

    @staticmethod
    def param_bool(_name):
        return False

    @staticmethod
    def workflow_ready():
        return True

    def runtime_motion_limit_rejection(self, _limits):
        return ''

    def get_logger(self):
        return SimpleNamespace(warn=self.logged.append)


def normalized_scan_goal(task_id='phase1-scan-0001'):
    """Return one deterministic valid goal without using the ROS clock."""
    covariance = [0.0] * 36
    covariance[0] = covariance[7] = covariance[14] = 0.01
    return validate_goal_payload({
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


class FakeGoalHandle:
    """Small action-goal double that records terminal transitions."""

    def __init__(self, task_id='phase1-scan-0001'):
        self.request = SimpleNamespace(
            task_id=task_id,
            rough_target=SimpleNamespace(),
        )
        self.is_cancel_requested = False
        self.transitions = []
        self.feedback = []

    def succeed(self):
        self.transitions.append('succeeded')

    def abort(self):
        self.transitions.append('aborted')

    def canceled(self):
        self.transitions.append('canceled')

    def publish_feedback(self, message):
        self.feedback.append(message)


class FakeSpool:
    """In-memory replacement for durable mission files."""

    def __init__(self):
        self.writes = []

    def write(self, section, identity, payload):
        self.writes.append((section, identity, dict(payload)))
        return payload


class FakeProcesses:
    """Generation-aware child-process double with configurable cleanup."""

    def __init__(self):
        self.events = []
        self.cleanup_succeeds = True
        self.live_generation = []

    def begin_generation(self):
        self.events.append('begin_generation')
        return list(self.live_generation)

    def failed(self):
        return {}

    def health(self):
        return {'fake': {'running': False}}

    def stop_all(self):
        self.events.append('child_process_termination')
        return bool(self.cleanup_succeeds)

    def shutdown(self, names=None):
        selected = tuple(self.live_generation if names is None else names)
        self.events.append(('selected_process_termination', selected))
        remaining = () if self.cleanup_succeeds else selected
        return SimpleNamespace(
            complete=not remaining,
            still_running=remaining)


class MissionCharacterizationHarness:
    """Run production orchestration against deterministic subsystem fakes."""

    execute_cb = TargetScanMissionNode.execute_cb
    run_pipeline = TargetScanMissionNode.run_pipeline
    safe_shutdown = TargetScanMissionNode.safe_shutdown

    def __init__(self, task_id='phase1-scan-0001'):
        normalized = normalized_scan_goal(task_id)
        self._lock = threading.RLock()
        self._prevalidated_goals = {task_id: normalized}
        self._process_shutdown_requested = False
        self._dispatch_task_id = task_id
        self.registry = MissionRegistry()
        self.spool = FakeSpool()
        self.processes = FakeProcesses()
        self.events = []
        self.phase_trace = []
        self.progress = []
        self.failure_stage = ''
        self.failure = None
        self.acquisition_states = ['ACQUIRED']
        self.capture_states = []
        self.workflow_state = 'SCAN_READY'
        self.coverage_sufficient = True
        self.required_captures = 8
        self.maximum_captures = 24
        self.latest_capture = {
            'captured_frame_count': 0,
            'scan_dir': '/tmp',
            'manifest_sha256': 'a' * 64,
        }
        self.latest_scan_history = {}
        self.last_scan_feature_coverage = {}
        self.latest_joints = SimpleNamespace(position=[0.0] * 6)
        self.latest_arm_status = self.disabled_motor_status()
        self.latest_arm_status_at = time.monotonic()
        self.motor_enable_guard_after = float('inf')
        self.last_return_home_diagnostic = ''
        self.current_home_profile = None
        self.enable_client = SimpleNamespace(service_is_ready=lambda: True)
        self.parameters = {
            'enable_real_arm_motion': True,
            'motion_speed_profile_qualified': True,
            'manage_processes': True,
            'require_gateway_heartbeat': False,
            'free_motion_speed_percent': 30.0,
            'contact_speed_percent': 10.0,
            'required_captures': self.required_captures,
            'maximum_captures': self.maximum_captures,
        }

    @staticmethod
    def disabled_motor_status():
        values = {
            'motor_feedback_valid': True,
            'motor_faults': [],
            'motor_watchdog_reason': '',
        }
        values.update({
            'motor_%d_driver_enabled' % index: False
            for index in range(1, 7)
        })
        return SimpleNamespace(**values)

    def inject(self, stage, failure):
        self.failure_stage = str(stage)
        self.failure = failure
        return self

    def maybe_fail(self, stage, session=None):
        if self.failure_stage != stage:
            return
        failure = self.failure
        if callable(failure):
            failure = failure(self, session)
        if failure is None:
            failure = MissionFailure('%s failed' % stage)
        raise failure

    def get_parameter(self, name):
        if name == 'required_captures':
            value = self.required_captures
        elif name == 'maximum_captures':
            value = self.maximum_captures
        else:
            value = self.parameters[name]
        return SimpleNamespace(value=value)

    def param_bool(self, name):
        return bool(self.get_parameter(name).value)

    def clear_runtime_caches(self):
        self.events.append('runtime_cache_clear')

    def write_status(self, session):
        self.events.append('status:%s' % session.phase.value)

    def snapshot_target(self, _rough_target):
        self.events.append('target_snapshot')
        return [0.4, 0.0, 0.0]

    def transition(self, _goal_handle, session, phase, reason):
        session.transition(phase, reason)
        self.phase_trace.append(phase.value)
        self.events.append('phase:%s' % phase.value)

    def startup_progress(self, _goal_handle, _session, reason):
        self.progress.append(str(reason))

    def selected_home_profile(self):
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

    def start_processes(self, _goal_handle, session):
        self.events.append('process_startup')
        self.maybe_fail('startup', session)

    def wait_for(self, _goal_handle, session, predicate, _timeout, label):
        self.events.append('readiness_check:%s' % label)
        self.maybe_fail('wait:%s' % label, session)
        if not predicate():
            raise MissionFailure(str(label))

    def wait_for_stable_readiness(
            self, _goal_handle, session, mode, _stable, _timeout):
        self.events.append('readiness:%s' % mode)
        self.maybe_fail('%s_readiness' % mode, session)

    def wait_for_stable_joint_stream(
            self, _stable, _timeout, label, _goal_handle=None, session=None):
        self.events.append('joint_feedback:%s' % label)
        self.maybe_fail('joint_feedback', session)

    def require_fresh_joint_feedback(self):
        self.events.append('preflight_joint_feedback')
        self.maybe_fail('preflight')

    def authorize_mission(self, session, revoke=False):
        event = 'authority_revoke' if revoke else 'authority_grant'
        self.events.append(event)
        self.maybe_fail(event, session)

    def call_enable(self, enabled):
        event = 'arm_enable' if enabled else 'motor_disable'
        self.events.append(event)
        self.maybe_fail(event)

    def prove_current_hold(self, _goal_handle, session):
        self.events.extend(('stop_motion', 'settled_hold'))
        self.maybe_fail('enable_hold', session)
        session.current_hold_proved = True
        return True

    def prove_return_home_for_shutdown(
            self, session, startup=False, goal_handle=None,
            target_positions=None, home_stage='ROUGH_HOME'):
        del goal_handle, target_positions
        stage = str(home_stage)
        if startup and not session.startup_home_completed:
            event = (
                'startup_wrist' if stage == 'STARTUP_WRIST'
                else 'startup_rough_home')
            failure_stage = event
        else:
            event = {
                'PRE_HOME': 'pre_home',
                'STORAGE_WRIST': 'storage_wrist',
            }.get(stage, 'return_home')
            failure_stage = event
            self.events.extend(('stop_motion', 'settled_hold'))
        self.events.append(event)
        self.maybe_fail(failure_stage, session)
        if stage == 'STORAGE_WRIST':
            session.storage_wrist_proved = True
        elif stage == 'PRE_HOME':
            session.pre_home_completed = True
        else:
            session.return_home_proved = True
        return True

    def prove_current_hold_for_shutdown(self, session):
        self.events.extend(('stop_motion', 'settled_hold'))
        self.maybe_fail('shutdown_hold', session)
        session.current_hold_proved = True
        return True

    def prepare_acquisition(self, session, _target):
        self.events.append('target_acquisition_request')
        self.maybe_fail('acquisition', session)
        return 'acquisition-request-%d' % session.acquisition_attempt

    def clear_plan_cache(self):
        self.events.append('plan_cache_clear')

    def wait_for_plan(
            self, _goal_handle, session, plan_kind, request_id,
            _timeout):
        self.events.append('planner_result:%s' % plan_kind)
        self.maybe_fail(
            'acquisition_planner' if plan_kind == 'ROUGH_ACQUISITION'
            else 'planner', session)
        return SimpleNamespace(
            plan_kind=plan_kind,
            plan_id=request_id,
            trajectory_sha256='b' * 64,
            planned_viewpoints=1,
        )

    def approve_plan(self, _goal_handle, session, plan):
        self.events.append('plan_approval:%s' % plan.plan_kind)
        self.maybe_fail(
            'acquisition_approval'
            if plan.plan_kind == 'ROUGH_ACQUISITION'
            else 'planning_approval', session)
        if plan.plan_kind in ('ROUGH_ACQUISITION', 'MULTIVIEW_SCAN'):
            session.return_home_proved = False

    def wait_for_execution(
            self, _goal_handle, session, successes, _timeout, _failures):
        if 'ACQUIRED' in successes:
            self.events.append('target_lock_measurement')
            self.maybe_fail('acquisition_execution', session)
            state = (
                self.acquisition_states.pop(0)
                if self.acquisition_states else 'ACQUISITION_LOOK_COMPLETE')
            return SimpleNamespace(state=state, reason='acquisition result')
        self.events.append('viewpoint_execution')
        self.maybe_fail('trajectory_execution', session)
        self.events.append('capture')
        self.maybe_fail('capture', session)
        state = (
            self.capture_states.pop(0)
            if self.capture_states else 'VIEW_COMPLETE')
        if state == 'VIEW_COMPLETE':
            self.latest_capture['captured_frame_count'] += 1
        return SimpleNamespace(state=state, reason='characterized view result')

    def start_and_wait_workflow(self, _goal_handle, session):
        self.events.append('occlusion_probe')
        self.maybe_fail('occlusion_probe', session)
        return {
            'state': self.workflow_state,
            'measured_lock_ready': True,
        }

    def readiness_rejection(self, mode):
        return (
            'manipulation model is not qualified'
            if mode == 'manipulation' else '')

    def request_multiview_plan(self, _goal_handle, session):
        self.events.append('viewpoint_planning')
        self.maybe_fail('planning_request', session)
        return 'multiview-request-%d' % (
            int(self.latest_capture['captured_frame_count']) + 1)

    def wait_for_view_generation(
            self, _goal_handle, _session, accepted_views, _timeout):
        assert int(accepted_views) == int(
            self.latest_capture['captured_frame_count'])
        self.events.append('view_generation')

    def current_scan_feature_coverage(self):
        accepted = int(self.latest_capture['captured_frame_count'])
        sufficient = bool(
            self.coverage_sufficient
            and accepted >= self.required_captures)
        value = {
            'sufficient': sufficient,
            'accepted_achieved_views': accepted,
            'blockers': [] if sufficient else ['coverage incomplete'],
        }
        self.last_scan_feature_coverage = dict(value)
        return value

    def result_message(self, result):
        return result

    def finish_queued_cancel(self, normalized, reason):
        result = TargetScanMissionNode.finish_queued_cancel(
            self, normalized, reason)
        return result

    @staticmethod
    def finish_action_handle(goal_handle, outcome):
        return TargetScanMissionNode.finish_action_handle(
            goal_handle, outcome)

    def finish_queue_dispatch(self, task_id):
        if self._dispatch_task_id == str(task_id):
            self._dispatch_task_id = ''


@pytest.fixture
def mission_harness():
    """Return a fresh successful mission harness."""
    return MissionCharacterizationHarness()


@pytest.fixture
def mission_harness_factory():
    """Create independent mission harnesses for parameterized cases."""
    return MissionCharacterizationHarness


@pytest.fixture
def goal_handle_factory():
    """Create action-goal doubles without importing test support modules."""
    return FakeGoalHandle


@pytest.fixture
def executor_runtime_harness():
    """Return a nominal fresh executor safety snapshot."""
    return ExecutorRuntimeHarness()
