#!/usr/bin/env python3
"""Headless, bounded ROS action orchestrator for one PiPER target scan."""

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from piper_msgs.srv import Enable
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from piper_mobile_manipulation.action import RunTargetScan
from piper_mobile_manipulation.home_pose import load_home_pose
from piper_mobile_manipulation.mission_core import (
    MAX_OCCLUSION_ACTIONS,
    MissionPhase,
    MissionRegistry,
    REQUIRED_CAPTURES,
    validate_goal_payload,
)
from piper_mobile_manipulation.mission_spool import MissionSpool
from piper_mobile_manipulation.msg import (
    CameraTimestampHealth,
    ScanExecutionPlan,
    ScanExecutionStatus,
    TesseractReadiness,
)
from piper_mobile_manipulation.scan_capture import rigid_transform_matrix
from piper_mobile_manipulation.scan_motion import energized_hold_target
from piper_mobile_manipulation.scan_viewpoint_executor_node import (
    abort_return_home_blocker,
)
from piper_mobile_manipulation.startup_gates import (
    joint_sample_rejection,
    joint_stability_update,
    readiness_stability_update,
    worker_health_rejection,
)
from piper_mobile_manipulation.srv import (
    ApproveScanExecution,
    AuthorizeMission,
    GetTargetScanResult,
    PrepareAcquisition,
    RequestTesseractPlan,
)


ACQUISITION_SERVICE_TIMEOUT_SEC = 8.0
WORKFLOW_ASSESSMENT_TIMEOUT_SEC = 15.0
PLAN_REQUEST_QUEUE_TIMEOUT_SEC = 12.0
PLAN_RESULT_TIMEOUT_SEC = 185.0


class ManagedProcessSet:
    """Own exact process groups and stop them in reverse dependency order."""

    def __init__(self, log_root):
        self.log_root = Path(log_root)
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.entries = {}
        self.log_offsets = {}

    def start(self, name, command, environment):
        if name in self.entries and self.entries[name][0].poll() is None:
            return
        log_path = self.log_root / (name + '.log')
        self.log_offsets[name] = (
            log_path.stat().st_size if log_path.exists() else 0)
        log = open(log_path, 'ab', buffering=0)
        process = subprocess.Popen(
            list(command), stdout=log, stderr=subprocess.STDOUT,
            env=dict(environment), start_new_session=True)
        self.entries[name] = (process, log, str(log_path))

    def log_since_start(self, name):
        entry = self.entries.get(name)
        if entry is None:
            return ''
        path = Path(entry[2])
        try:
            with open(path, 'rb') as stream:
                stream.seek(self.log_offsets.get(name, 0))
                return stream.read(256 * 1024).decode('utf-8', errors='replace')
        except OSError:
            return ''

    def failed(self):
        return {
            name: process.returncode
            for name, (process, _log, _path) in self.entries.items()
            if process.poll() is not None
        }

    def health(self):
        return {
            name: {
                'pid': int(process.pid),
                'running': process.poll() is None,
                'returncode': process.poll(),
                'log': path,
            }
            for name, (process, _log, path) in self.entries.items()
        }

    def stop_all(self):
        entries = list(self.entries.items())[::-1]
        for _name, (process, _log, _path) in entries:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and any(
                process.poll() is None for _, (process, _, _) in entries):
            time.sleep(0.05)
        for _name, (process, _log, _path) in entries:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and any(
                process.poll() is None for _, (process, _, _) in entries):
            time.sleep(0.05)
        for _name, (process, log, _path) in entries:
            if process.poll() is None:
                # Do not SIGKILL an arm command owner. Its continued liveness
                # is surfaced as NEEDS_OPERATOR instead of claiming shutdown.
                continue
            log.close()
        return not any(process.poll() is None for _, (process, _, _) in entries)


class MissionFailure(RuntimeError):
    def __init__(self, reason, needs_operator=False, outcome='FAILED'):
        super().__init__(reason)
        self.needs_operator = bool(needs_operator)
        self.outcome = str(outcome)


class TargetScanMissionNode(Node):
    def __init__(self):
        super().__init__('target_scan_mission')
        defaults = {
            'project_root': '/home/prl/Piper_arm',
            'manage_processes': True,
            'enable_real_arm_motion': False,
            'motion_speed_profile_qualified': False,
            'free_motion_speed_percent': 30.0,
            'contact_speed_percent': 10.0,
            'required_captures': REQUIRED_CAPTURES,
            'home_pose_path': '',
            'mission_spool_root': os.path.join(
                os.environ.get('XDG_RUNTIME_DIR', '/tmp'),
                'piper_target_scan_missions'),
            'process_log_root': os.path.join(
                os.environ.get('XDG_RUNTIME_DIR', '/tmp'),
                'piper_target_scan_logs'),
            'require_gateway_heartbeat': False,
            'debug': True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.callback_group = ReentrantCallbackGroup()
        self._lock = threading.RLock()
        self._spool_seen = set()
        self._spool_thread = None
        self.registry = MissionRegistry()
        self.spool = MissionSpool(
            self.get_parameter('mission_spool_root').value)
        self.load_durable_results()
        self.processes = ManagedProcessSet(
            self.get_parameter('process_log_root').value)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.latest_readiness = None
        self.latest_readiness_at = 0.0
        self.latest_plan = None
        self.latest_plan_at = 0.0
        self.latest_execution = None
        self.latest_execution_at = 0.0
        self.latest_capture = {}
        self.latest_capture_at = 0.0
        self.last_hold_diagnostic = ''
        self.latest_joints = None
        self.latest_joints_at = 0.0
        self.latest_joint_source_ns = 0
        self.joint_generation_started_ns = 0
        self.joint_stream_stable_since = 0.0
        self.joint_stream_reference = None
        self.joint_feedback_rejection = 'driver generation has not started'
        self.latest_camera_health = None
        self.latest_camera_health_at = 0.0

        latched = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            TesseractReadiness, '/piper/tesseract_readiness',
            self.readiness_cb, 10)
        self.create_subscription(
            ScanExecutionPlan, '/piper/scan_execution_plan',
            self.plan_cb, latched)
        self.create_subscription(
            ScanExecutionStatus, '/piper/scan_execution_status',
            self.execution_cb, 10)
        self.create_subscription(
            String, '/piper/scan_capture_status', self.capture_cb, 10)
        self.create_subscription(
            JointState, '/joint_states_single', self.joint_cb, 10)
        self.create_subscription(
            CameraTimestampHealth, '/piper/camera_timestamp_health',
            self.camera_health_cb, 10)

        self.enable_client = self.create_client(
            Enable, '/enable_srv', callback_group=self.callback_group)
        self.prepare_client = self.create_client(
            PrepareAcquisition, '/scan_target_acquisition/prepare',
            callback_group=self.callback_group)
        self.plan_client = self.create_client(
            RequestTesseractPlan, '/tesseract_plan_bridge/request_plan',
            callback_group=self.callback_group)
        self.approve_client = self.create_client(
            ApproveScanExecution, '/scan_viewpoint_executor/approve',
            callback_group=self.callback_group)
        self.authorize_client = self.create_client(
            AuthorizeMission, '/scan_viewpoint_executor/authorize_mission',
            callback_group=self.callback_group)
        self.cancel_client = self.create_client(
            Trigger, '/scan_viewpoint_executor/cancel',
            callback_group=self.callback_group)
        self.workflow_start_client = self.create_client(
            Trigger, '/supervised_cube_workflow/start',
            callback_group=self.callback_group)
        self.workflow_diagnostic_client = self.create_client(
            Trigger, '/supervised_cube_workflow/diagnostic_state',
            callback_group=self.callback_group)

        self.action_server = ActionServer(
            self, RunTargetScan, '/piper/run_target_scan',
            execute_callback=self.execute_cb,
            goal_callback=self.goal_cb,
            cancel_callback=self.cancel_cb,
            callback_group=self.callback_group)
        self.create_service(
            GetTargetScanResult, '/piper/get_target_scan_result',
            self.get_result_cb, callback_group=self.callback_group)
        self.create_timer(0.5, self.poll_spool_goals)

    def load_durable_results(self):
        """Restore idempotency before the first persisted goal is polled."""
        for path in sorted((self.spool.root / 'results').glob('*.json')):
            task_id = path.stem
            try:
                payload = self.spool.read('results', task_id)
                if str(payload.get('task_id', '')) != task_id:
                    raise ValueError('result task ID does not match its filename')
                self.registry.results[task_id] = payload
                self._spool_seen.add(task_id)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self.get_logger().error(
                    'ignoring invalid durable result %s: %s' % (path, exc))

    def param_bool(self, name):
        value = self.get_parameter(name).value
        return value.lower() in ('1', 'true', 'yes', 'on') \
            if isinstance(value, str) else bool(value)

    def readiness_cb(self, msg):
        with self._lock:
            self.latest_readiness, self.latest_readiness_at = msg, time.monotonic()

    def plan_cb(self, msg):
        with self._lock:
            self.latest_plan, self.latest_plan_at = msg, time.monotonic()

    def execution_cb(self, msg):
        with self._lock:
            self.latest_execution, self.latest_execution_at = msg, time.monotonic()

    def capture_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return
        if isinstance(payload, dict):
            with self._lock:
                self.latest_capture, self.latest_capture_at = payload, time.monotonic()

    def joint_cb(self, msg):
        positions = list(msg.position[:6])
        stamp_ns = (
            int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec))
        receive_ns = int(self.get_clock().now().nanoseconds)
        received_at = time.monotonic()
        with self._lock:
            previous = (
                None if self.latest_joints is None
                else list(self.latest_joints.position[:6]))
            reason = joint_sample_rejection(
                previous, self.latest_joint_source_ns, positions, stamp_ns,
                receive_ns, self.joint_generation_started_ns)
            if reason:
                self.joint_feedback_rejection = reason
                self.joint_stream_stable_since = 0.0
                self.joint_stream_reference = None
                return
            self.joint_stream_reference, self.joint_stream_stable_since = \
                joint_stability_update(
                    self.joint_stream_reference,
                    self.joint_stream_stable_since,
                    positions,
                    received_at,
                )
            self.latest_joints = msg
            self.latest_joints_at = received_at
            self.latest_joint_source_ns = stamp_ns
            self.joint_feedback_rejection = ''

    def camera_health_cb(self, msg):
        with self._lock:
            self.latest_camera_health = msg
            self.latest_camera_health_at = time.monotonic()

    @staticmethod
    def goal_payload(goal):
        stamp = goal.rough_target.header.stamp
        return {
            'task_id': goal.task_id,
            'task_type': goal.task_type,
            'target_label': goal.target_label,
            'target_profile': goal.target_profile,
            'target_confidence': float(goal.target_confidence),
            'deadline_sec': float(goal.deadline_sec),
            'rough_target': {
                'frame_id': goal.rough_target.header.frame_id,
                'stamp_sec': float(stamp.sec) + float(stamp.nanosec) * 1e-9,
                'position': [
                    goal.rough_target.pose.pose.position.x,
                    goal.rough_target.pose.pose.position.y,
                    goal.rough_target.pose.pose.position.z,
                ],
                'covariance': list(goal.rough_target.pose.covariance),
            },
        }

    def goal_cb(self, goal):
        try:
            normalized = validate_goal_payload(self.goal_payload(goal))
        except (TypeError, ValueError) as exc:
            self.get_logger().warn('rejecting target-scan goal: %s' % exc)
            return GoalResponse.REJECT
        with self._lock:
            cached = self.registry.results.get(normalized['task_id'])
            if cached is not None:
                return (
                    GoalResponse.ACCEPT
                    if cached.get('mission_sha256') == normalized['mission_sha256']
                    else GoalResponse.REJECT)
            return (
                GoalResponse.ACCEPT
                if self.registry.active is None else GoalResponse.REJECT)

    def cancel_cb(self, _goal_handle):
        return CancelResponse.ACCEPT

    def get_result_cb(self, request, response):
        with self._lock:
            result = self.registry.results.get(str(request.task_id))
        response.found = result is not None
        response.result_json = json.dumps(result, sort_keys=True) if result else ''
        response.message = 'cached result returned' if result else 'task result not found'
        return response

    def execute_cb(self, goal_handle):
        try:
            normalized = validate_goal_payload(self.goal_payload(goal_handle.request))
        except (TypeError, ValueError) as exc:
            goal_handle.abort()
            return self.action_result('FAILED', str(exc), False)
        with self._lock:
            status, record = self.registry.admit(normalized)
        if status == 'CACHED':
            if record.get('outcome') == 'SUCCEEDED':
                goal_handle.succeed()
            elif record.get('outcome') == 'CANCELLED':
                goal_handle.canceled()
            else:
                goal_handle.abort()
            return self.result_message(record)
        if status != 'ACCEPTED':
            goal_handle.abort()
            return self.action_result(status, 'task was not admitted', False)
        session = record
        self.clear_runtime_caches()
        self.write_status(session)
        transformed_target = None
        failure = None
        try:
            transformed_target = self.snapshot_target(goal_handle.request.rough_target)
            self.run_pipeline(goal_handle, session, transformed_target)
        except MissionFailure as exc:
            failure = exc
        except Exception as exc:  # Preserve shutdown even on unexpected ROS errors.
            failure = MissionFailure('mission exception: %s' % exc)

        if failure is None:
            shutdown_failure = self.safe_shutdown(
                session, normal_completion=True)
            if shutdown_failure is not None:
                failure = shutdown_failure
        else:
            shutdown_failure = self.safe_shutdown(
                session, normal_completion=False, failure=failure)
            if shutdown_failure is not None:
                failure = MissionFailure(
                    '%s; safe shutdown also failed: %s'
                    % (failure, shutdown_failure),
                    needs_operator=True)
            elif failure.outcome == 'CANCELLED':
                failure = MissionFailure(
                    'task failed: cancelled; arm returned to configured home, '
                    'disabled, and pipeline stopped; please retry',
                    outcome='CANCELLED')

        capture = dict(self.latest_capture)
        if failure is None:
            session.phase = MissionPhase.SUCCEEDED
            session.reason = '13-view target scan completed and PiPER shut down safely'
            outcome = 'SUCCEEDED'
            goal_handle.succeed()
        else:
            session.phase = (
                MissionPhase.NEEDS_OPERATOR
                if failure.needs_operator else MissionPhase.FAILED)
            session.reason = str(failure)
            outcome = (
                'NEEDS_OPERATOR' if failure.needs_operator
                else failure.outcome)
            if outcome == 'CANCELLED':
                goal_handle.canceled()
            else:
                goal_handle.abort()
        result = session.result_payload(
            outcome, session.reason,
            dataset_path=str(capture.get('scan_dir', '')),
            manifest_sha256=str(capture.get('manifest_sha256', '')),
            action_summary={
                'processes': self.processes.health(),
                'home_positions_rad': list(session.home_positions_rad),
            },
        )
        with self._lock:
            self.registry.finish(result)
        self.spool.write('results', session.task_id, result)
        return self.result_message(result)

    def run_pipeline(self, goal_handle, session, target):
        session.home_positions_rad = tuple(self.selected_home_positions())
        self.transition(goal_handle, session, MissionPhase.STARTING,
                        'starting PiPER-owned process groups')
        self.start_processes(goal_handle, session)
        self.startup_progress(
            goal_handle, session,
            'scan stack started; waiting for typed acquisition readiness')
        self.wait_for(
            goal_handle, session,
            lambda: self.enable_client.service_is_ready(), 30.0,
            'PiPER enable service did not become ready')
        self.wait_for_stable_readiness(
            goal_handle, session, 'acquisition', 2.0, 90.0)
        self.startup_progress(
            goal_handle, session,
            'acquisition ready; proving final settled joint feedback')
        self.wait_for_stable_joint_stream(
            2.0, 15.0, 'pre-enable joint feedback')

        self.transition(goal_handle, session, MissionPhase.PREFLIGHT,
                        'validating current feedback and mission authority')
        self.require_fresh_joint_feedback()
        if not self.param_bool('enable_real_arm_motion'):
            raise MissionFailure(
                'mission node is proposal-only; real arm motion was not enabled')
        if not self.param_bool('motion_speed_profile_qualified'):
            raise MissionFailure(
                'configured %.1f-percent transit and %.1f-percent contact '
                'speed profile is not physically qualified; arm remained '
                'disabled'
                % (
                    self.get_parameter('free_motion_speed_percent').value,
                    self.get_parameter('contact_speed_percent').value))
        self.authorize_mission(session)
        self.transition(goal_handle, session, MissionPhase.ENABLE_AND_HOLD,
                        'enabling arm and proving current-position hold')
        self.call_enable(True)
        session.arm_enabled = True
        # J2/J3 can settle from their unpowered gravity droop to a coherent
        # powered zero. Require a full stable window before binding the hold.
        with self._lock:
            self.joint_stream_stable_since = 0.0
            self.joint_stream_reference = None
        self.wait_for_stable_joint_stream(
            1.0, 15.0, 'post-enable powered joint feedback')
        if not self.prove_current_hold(goal_handle, session):
            raise MissionFailure(
                'current-position hold did not settle after enable: '
                + self.last_hold_diagnostic, True)

        self.transition(goal_handle, session, MissionPhase.ROUGH_ACQUISITION,
                        'requesting bounded rough-target acquisition')
        session.acquisition_attempt += 1
        self.clear_plan_cache()
        request_id = self.prepare_acquisition(session, target)
        plan = self.wait_for_plan(
            goal_handle, session, 'ROUGH_ACQUISITION', request_id,
            PLAN_RESULT_TIMEOUT_SEC)
        self.approve_plan(session, plan)
        self.transition(goal_handle, session, MissionPhase.TARGET_LOCK,
                        'waiting for a fresh measured target lock')
        self.wait_for_execution(
            goal_handle, session, ('ACQUIRED',), 180.0,
            ('ACQUISITION_FAILED', 'ABORTED', 'INVALID'))

        self.transition(goal_handle, session, MissionPhase.OCCLUSION_PROBE,
                        'assessing the measured target and occluder scene')
        workflow = self.start_and_wait_workflow(goal_handle, session)
        if workflow.get('state') == 'PLAN_READY':
            readiness = self.readiness_rejection('manipulation')
            raise MissionFailure(
                'beneficial occluder removal is required but unavailable: '
                + (readiness or 'contact planner is not implemented'),
                needs_operator=True)

        # ACQUIRED and the workflow diagnostic can arrive before the
        # independently published multiview readiness generation catches up.
        # Do not fire the one exact request into that short gap: it would be
        # rejected immediately and look like the target was never allowed to
        # lock.  Require a fresh stable generation before entering planning.
        self.startup_progress(
            goal_handle, session,
            'measured target lock ready; waiting for stable multiview readiness')
        self.wait_for_stable_readiness(
            goal_handle, session, 'multiview', 1.0, 30.0)
        self.transition(goal_handle, session, MissionPhase.VIEW_PLANNING,
                        'requesting one correlated diverse 13-view plan')
        self.clear_plan_cache()
        request_id = self.request_multiview_plan()
        plan = self.wait_for_plan(
            goal_handle, session, 'MULTIVIEW_SCAN', request_id,
            PLAN_RESULT_TIMEOUT_SEC)
        self.approve_plan(session, plan)
        self.transition(goal_handle, session, MissionPhase.CAPTURING,
                        'executing settled viewpoint captures')
        execution = self.wait_for_execution(
            goal_handle, session, ('COMPLETE',), session.remaining(),
            ('ABORTED', 'INVALID'))
        session.accepted_captures = int(
            self.latest_capture.get('captured_frame_count', 0))
        required = int(self.get_parameter('required_captures').value)
        if session.accepted_captures != required:
            raise MissionFailure(
                'executor completed with %d/%d captures'
                % (session.accepted_captures, required))
        if 'home reached' not in str(execution.reason).lower():
            raise MissionFailure(
                'captures completed but return-home was not proved: %s'
                % execution.reason)
        session.return_home_proved = True

    def start_processes(self, goal_handle, session):
        if not self.param_bool('manage_processes'):
            return
        root = Path(str(self.get_parameter('project_root').value)).resolve()
        environment = dict(os.environ)
        environment.update({
            'PIPER_ARM_ROOT': str(root),
            'PIPER_AUTO_ENABLE': 'false',
            'PIPER_ENABLE_REAL_VIEWPOINT_MOTION': (
                '1' if self.param_bool('enable_real_arm_motion') else '0'),
            'PIPER_VIEWPOINT_MISSION_POLICY': '1',
            'PIPER_VIEWPOINT_SPEED_PERCENT': str(
                self.get_parameter('free_motion_speed_percent').value),
            'PIPER_VIEWPOINT_MAX_VIEWS': str(REQUIRED_CAPTURES),
            'PIPER_VIEWPOINT_MIN_VIEWS': str(REQUIRED_CAPTURES),
            'PIPER_RETURN_HOME_POSITIONS_RAD': json.dumps(
                list(session.home_positions_rad), separators=(',', ':')),
        })
        with self._lock:
            self.joint_generation_started_ns = int(
                self.get_clock().now().nanoseconds)
            self.latest_joints = None
            self.latest_joints_at = 0.0
            self.latest_joint_source_ns = 0
            self.joint_stream_stable_since = 0.0
            self.joint_stream_reference = None
            self.joint_feedback_rejection = (
                'waiting for current driver-generation feedback')
        self.startup_progress(
            goal_handle, session,
            'starting disabled PiPER driver and waiting for its service')
        self.processes.start(
            'driver', [str(root / 'start_piper.sh')], environment)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            failed = self.processes.failed()
            if 'driver' in failed:
                raise MissionFailure(
                    'driver process exited during startup with status %s'
                    % failed['driver'])
            if self.enable_client.service_is_ready():
                break
            time.sleep(0.1)
        else:
            raise MissionFailure('driver enable service did not become ready')
        self.startup_progress(
            goal_handle, session,
            'driver service ready; proving coherent settled joint feedback')
        self.wait_for_stable_joint_stream(
            2.0, 15.0, 'driver startup joint feedback')

        with self._lock:
            self.latest_camera_health = None
            self.latest_camera_health_at = 0.0
        self.startup_progress(
            goal_handle, session,
            'driver feedback ready; starting camera and GPU perception')
        vision_started_at = time.monotonic()
        self.processes.start(
            'vision', [str(root / 'L515_camera/run_gpu_vision_pipeline.sh')],
            environment)
        self.wait_for_vision_boot(vision_started_at, 120.0)

        self.startup_progress(
            goal_handle, session,
            'camera and GPU perception ready; starting hand-eye transform')
        hand_eye_started_ns = int(self.get_clock().now().nanoseconds)
        self.processes.start(
            'hand_eye', [str(root / 'L515_camera/run_hand_eye_tf.sh')],
            environment)
        self.wait_for_hand_eye_boot(hand_eye_started_ns, 20.0)

        self.startup_progress(
            goal_handle, session,
            'fresh hand-eye transform ready; starting Tesseract worker')
        worker_health_path = Path(
            environment.get(
                'PIPER_TESSERACT_SPOOL',
                os.path.join(
                    environment.get('XDG_RUNTIME_DIR', '/tmp'),
                    'piper_tesseract_plans'))) / 'worker_health.json'
        previous_worker_generation = self.worker_generation(
            worker_health_path)
        self.processes.start(
            'tesseract_worker', [
                str(root / 'motion_planning/tesseract/run_worker.sh')],
            environment)
        self.wait_for_worker_boot(
            worker_health_path, previous_worker_generation, 45.0)

        self.startup_progress(
            goal_handle, session,
            'new Tesseract worker ready; starting supervised scan stack')
        self.processes.start(
            'scan_stack', [
                str(root / 'L515_camera/run_supervised_viewpoint_execution.sh')],
            environment)
        time.sleep(0.5)
        failed = self.processes.failed()
        if 'scan_stack' in failed:
            raise MissionFailure(
                'scan stack exited during startup with status %s'
                % failed['scan_stack'])

    def wait_for_stable_joint_stream(self, stable_sec, timeout_sec, label):
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            failed = self.processes.failed()
            if 'driver' in failed:
                raise MissionFailure(
                    'driver exited while waiting for %s with status %s'
                    % (label, failed['driver']))
            now = time.monotonic()
            with self._lock:
                fresh = (
                    self.latest_joints is not None
                    and now - self.latest_joints_at <= 0.25)
                stable_since = self.joint_stream_stable_since
                rejection = self.joint_feedback_rejection
            if (
                    fresh and stable_since > 0.0
                    and now - stable_since >= float(stable_sec)):
                return
            time.sleep(0.05)
        raise MissionFailure(
            '%s did not remain coherent and settled for %.1f seconds: %s'
            % (label, stable_sec, rejection or 'feedback is missing or moving'))

    def wait_for_vision_boot(self, started_at, timeout_sec):
        required_markers = (
            'Heavy-model worker ready:',
            'SAM2 live worker ready on',
            'GPU vision pipeline is running',
        )
        deadline = time.monotonic() + float(timeout_sec)
        last_reason = 'waiting for camera timestamp health and CUDA workers'
        while time.monotonic() < deadline:
            failed = self.processes.failed()
            if 'vision' in failed:
                raise MissionFailure(
                    'vision process exited during startup with status %s'
                    % failed['vision'])
            log = self.processes.log_since_start('vision')
            missing = [marker for marker in required_markers if marker not in log]
            with self._lock:
                health = self.latest_camera_health
                health_at = self.latest_camera_health_at
            camera_ready = (
                health is not None
                and health_at >= started_at
                and time.monotonic() - health_at <= 1.0
                and bool(health.healthy)
                and int(health.consecutive_healthy_frames) >= 5)
            if camera_ready and not missing:
                return
            details = []
            if not camera_ready:
                details.append(
                    str(health.reason) if health is not None
                    else 'camera timestamp health is missing')
            if missing:
                details.append('missing startup markers: ' + ', '.join(missing))
            last_reason = '; '.join(details)
            time.sleep(0.1)
        raise MissionFailure('vision startup timed out: ' + last_reason)

    def wait_for_hand_eye_boot(self, started_ns, timeout_sec):
        deadline = time.monotonic() + float(timeout_sec)
        last_error = 'base_link -> camera_link transform is missing'
        while time.monotonic() < deadline:
            failed = self.processes.failed()
            if 'hand_eye' in failed:
                raise MissionFailure(
                    'hand-eye process exited during startup with status %s'
                    % failed['hand_eye'])
            try:
                transform = self.tf_buffer.lookup_transform(
                    'base_link', 'camera_link', Time())
                stamp_ns = (
                    int(transform.header.stamp.sec) * 1_000_000_000
                    + int(transform.header.stamp.nanosec))
                if stamp_ns >= int(started_ns):
                    return
                last_error = 'hand-eye transform predates its process generation'
            except TransformException as exc:
                last_error = str(exc)
            time.sleep(0.1)
        raise MissionFailure('hand-eye startup timed out: ' + last_error)

    @staticmethod
    def read_worker_health(path):
        try:
            stat = path.lstat()
            if path.is_symlink() or not path.is_file() or stat.st_nlink != 1:
                return None
            if stat.st_size <= 0 or stat.st_size > 16 * 1024:
                return None
            with open(path, 'r', encoding='utf-8') as stream:
                value = json.load(stream)
            return value if isinstance(value, dict) else None
        except (FileNotFoundError, OSError, TypeError, json.JSONDecodeError):
            return None

    def worker_generation(self, path):
        health = self.read_worker_health(path)
        return str(health.get('generation_id', '')) if health else ''

    def wait_for_worker_boot(self, path, previous_generation, timeout_sec):
        deadline = time.monotonic() + float(timeout_sec)
        last_reason = 'Tesseract worker heartbeat is missing'
        while time.monotonic() < deadline:
            failed = self.processes.failed()
            if 'tesseract_worker' in failed:
                raise MissionFailure(
                    'Tesseract worker exited during startup with status %s'
                    % failed['tesseract_worker'])
            health = self.read_worker_health(path)
            last_reason = worker_health_rejection(
                health, time.time_ns(), previous_generation)
            if not last_reason:
                return
            time.sleep(0.1)
        raise MissionFailure('Tesseract worker startup timed out: ' + last_reason)

    def selected_home_positions(self):
        configured = str(self.get_parameter('home_pose_path').value).strip()
        if not configured:
            configured = str(
                Path(str(self.get_parameter('project_root').value)).resolve()
                / 'piper_home_pose.json')
        try:
            payload = load_home_pose(configured)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MissionFailure('configured home pose is invalid: %s' % exc)
        if payload is not None:
            return payload['positions_rad']
        return [0.000366362, 0.0, 0.0, 0.0, 0.43869236, 0.0]

    def snapshot_target(self, rough_target):
        source = str(rough_target.header.frame_id)
        point = np.asarray([
            rough_target.pose.pose.position.x,
            rough_target.pose.pose.position.y,
            rough_target.pose.pose.position.z,
        ], dtype=float)
        if source == 'base_link':
            return point.tolist()
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link', source, Time.from_msg(rough_target.header.stamp))
        except TransformException as exc:
            raise MissionFailure(
                'cannot snapshot odom to piper_base_link transform: %s' % exc)
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        matrix = rigid_transform_matrix(
            [translation.x, translation.y, translation.z],
            [rotation.x, rotation.y, rotation.z, rotation.w])
        transformed = matrix @ np.asarray([point[0], point[1], point[2], 1.0])
        return transformed[:3].tolist()

    def authorize_mission(self, session, revoke=False):
        request = AuthorizeMission.Request()
        request.task_id = session.task_id
        request.mission_sha256 = session.mission_sha256
        expires = time.time() + session.remaining()
        request.expires_at.sec = int(expires)
        request.expires_at.nanosec = int((expires - int(expires)) * 1e9)
        request.revoke = bool(revoke)
        result = self.call_service(
            self.authorize_client, request, 8.0,
            'mission authorization service')
        if not result.accepted:
            raise MissionFailure(str(result.message))

    def prepare_acquisition(self, session, target):
        request = PrepareAcquisition.Request()
        request.session_id = '%s-%d' % (session.task_id, session.acquisition_attempt)
        request.rough_target = PointStamped()
        request.rough_target.header.stamp = self.get_clock().now().to_msg()
        request.rough_target.header.frame_id = 'base_link'
        request.rough_target.point.x = float(target[0])
        request.rough_target.point.y = float(target[1])
        request.rough_target.point.z = float(target[2])
        result = self.call_service(
            self.prepare_client, request, ACQUISITION_SERVICE_TIMEOUT_SEC,
            'rough acquisition service')
        if not result.accepted or str(result.session_id) != request.session_id:
            raise MissionFailure(str(result.message))
        return request.session_id

    def request_multiview_plan(self):
        deadline = time.monotonic() + PLAN_REQUEST_QUEUE_TIMEOUT_SEC
        while time.monotonic() < deadline:
            request = RequestTesseractPlan.Request()
            request.force_refresh = False
            result = self.call_service(
                self.plan_client, request,
                max(0.1, deadline - time.monotonic()),
                'Tesseract plan request service')
            if result.accepted:
                return str(result.request_id)
            if result.request_id and 'already pending' in str(result.message):
                time.sleep(0.25)
                continue
            raise MissionFailure(str(result.message))
        raise MissionFailure('timed out waiting to queue one fresh multiview plan')

    def wait_for_plan(self, goal_handle, session, kind, request_id, timeout):
        generation_started = time.monotonic()

        def matching():
            plan = self.latest_plan
            if plan is None or self.latest_plan_at < generation_started:
                return False
            if str(plan.plan_kind) != kind:
                return False
            if kind == 'ROUGH_ACQUISITION':
                correlated = (
                    str(plan.source_request_id) == str(request_id))
            else:
                correlated = bool(
                    request_id
                    and str(plan.plan_id) == str(request_id))
            if not correlated:
                return False
            if not bool(plan.valid):
                raise MissionFailure(
                    '%s planning failed: %s' % (kind, str(plan.reason)))
            return True

        self.wait_for(
            goal_handle, session, matching, timeout,
            'timed out waiting for correlated %s plan' % kind)
        plan = self.latest_plan
        if kind == 'MULTIVIEW_SCAN' and int(plan.planned_viewpoints) != REQUIRED_CAPTURES:
            raise MissionFailure('multiview plan does not contain exactly 13 views')
        return plan

    def approve_plan(self, session, plan):
        request = ApproveScanExecution.Request()
        request.plan_id = str(plan.plan_id)
        request.trajectory_sha256 = str(plan.trajectory_sha256)
        request.confirmation = 'MISSION_POLICY:' + session.mission_sha256
        result = self.call_service(
            self.approve_client, request, 10.0, 'mission plan approval service')
        if not result.accepted:
            raise MissionFailure(str(result.message))

    def start_and_wait_workflow(self, goal_handle, session):
        result = self.call_service(
            self.workflow_start_client, Trigger.Request(), 8.0,
            'workflow start service')
        if not result.success and 'already active' not in str(result.message):
            raise MissionFailure(str(result.message))
        deadline = time.monotonic() + WORKFLOW_ASSESSMENT_TIMEOUT_SEC
        while time.monotonic() < deadline:
            self.guard(goal_handle, session)
            diagnostic = self.call_service(
                self.workflow_diagnostic_client, Trigger.Request(), 3.0,
                'workflow diagnostic service')
            if not diagnostic.success:
                raise MissionFailure(str(diagnostic.message))
            try:
                payload = json.loads(diagnostic.message)
            except json.JSONDecodeError as exc:
                raise MissionFailure('workflow diagnostic is invalid JSON: %s' % exc)
            state = str(payload.get('state', ''))
            if state == 'SCAN_READY' and payload.get('measured_lock_ready'):
                return payload
            if state == 'PLAN_READY':
                return payload
            if state == 'ABORTED':
                raise MissionFailure(str(payload.get('reason', 'workflow aborted')))
            if state not in ('IDLE', 'INITIALIZING'):
                raise MissionFailure('workflow is in incompatible state %s' % state)
            time.sleep(0.25)
        raise MissionFailure('workflow assessment exceeded 15 seconds')

    def wait_for_execution(self, goal_handle, session, successes, timeout, failures):
        started = time.monotonic()

        def completed():
            status = self.latest_execution
            if status is None or self.latest_execution_at < started:
                return False
            state = str(status.state)
            if (
                    state == 'SETTLING_HOME'
                    and session.phase == MissionPhase.CAPTURING):
                session.transition(
                    MissionPhase.RETURNING_HOME,
                    'all captures persisted; settling at approved home')
                self.write_status(session)
            if state in failures:
                raise MissionFailure('%s: %s' % (state, status.reason))
            return state in successes

        self.wait_for(
            goal_handle, session, completed, timeout,
            'execution did not reach %s before timeout' % '/'.join(successes))
        return self.latest_execution

    def prove_current_hold(self, goal_handle, session):
        self.require_fresh_joint_feedback()
        # The enable service can return between CAN feedback samples.  Bind
        # the hold to the first sample received after enable completes so a
        # pre-enable sample cannot make a correctly held arm look displaced.
        previous_sample_at = self.latest_joints_at
        fresh_deadline = time.monotonic() + 1.0
        while (
                self.latest_joints_at <= previous_sample_at
                and time.monotonic() < fresh_deadline):
            self.guard(goal_handle, session)
            time.sleep(0.01)
        self.require_fresh_joint_feedback()
        initial = energized_hold_target(self.latest_joints.position[:6])
        result = self.call_service(
            self.cancel_client, Trigger.Request(), 8.0,
            'executor current-position hold service')
        if not result.success or 'hold requested' not in str(result.message):
            return False
        settled_since = None
        deadline = time.monotonic() + 15.0
        last_delta = math.inf
        last_velocity = math.inf
        last_current = np.asarray(initial, dtype=float)
        while time.monotonic() < deadline:
            self.guard(goal_handle, session)
            if self.latest_joints is None or time.monotonic() - self.latest_joints_at > 1.0:
                settled_since = None
                time.sleep(0.05)
                continue
            current = np.asarray(self.latest_joints.position[:6], dtype=float)
            delta = float(np.max(np.abs(current - initial)))
            velocities = np.asarray(self.latest_joints.velocity[:6], dtype=float)
            velocity = float(np.max(np.abs(velocities))) if velocities.size == 6 else math.inf
            last_delta = delta
            last_velocity = velocity
            last_current = current
            if delta <= 0.005 and velocity <= 0.20:
                settled_since = settled_since or time.monotonic()
                if time.monotonic() - settled_since >= 1.0:
                    session.current_hold_proved = True
                    self.last_hold_diagnostic = (
                        'settled delta=%.6f rad velocity=%.6f rad/s'
                        % (delta, velocity))
                    return True
            else:
                settled_since = None
            time.sleep(0.05)
        self.last_hold_diagnostic = (
            'last delta=%.6f rad velocity=%.6f rad/s '
            '(limits 0.005000 rad and 0.200000 rad/s); '
            'initial=%s current=%s per_joint_delta=%s'
            % (
                last_delta, last_velocity,
                [round(float(value), 6) for value in initial],
                [round(float(value), 6) for value in last_current],
                [round(float(value), 6) for value in np.abs(
                    last_current - initial)]))
        return False

    def safe_shutdown(self, session, normal_completion, failure=None):
        try:
            if not session.arm_enabled:
                session.current_hold_proved = True
                session.return_home_proved = True
                session.disabled_proved = True
                try:
                    self.authorize_mission(session, revoke=True)
                except MissionFailure:
                    pass
                self.transition(None, session, MissionPhase.STOPPING,
                                'stopping never-enabled PiPER process groups')
                session.processes_stopped = self.processes.stop_all()
                if not session.processes_stopped:
                    return MissionFailure(
                        'one or more PiPER-owned processes remain alive', True)
                return None
            if not session.return_home_proved:
                blocker = abort_return_home_blocker(
                    str(failure) if failure is not None else '')
                if blocker:
                    return MissionFailure(
                        'configured home return was not attempted because the '
                        'failure is motion-safety-related (%s); arm remains '
                        'enabled in a current-position hold' % blocker, True)
                self.transition(
                    None, session, MissionPhase.RETURNING_HOME,
                    'cancellation/failure accepted; holding, then retracing '
                    'the approved path to configured home')
                if not self.prove_return_home_for_shutdown(session):
                    return MissionFailure(
                        'configured home return was not proved; arm remains '
                        'enabled in a current-position hold', True)
            self.transition(None, session, MissionPhase.HOLDING,
                            'proving final current-position hold')
            if not self.prove_current_hold_for_shutdown(session):
                return MissionFailure(
                    'final current-position hold did not settle; arm remains enabled', True)
            self.transition(None, session, MissionPhase.DISABLING,
                            'disabling arm with feedback-confirmed service')
            self.call_enable(False)
            session.disabled_proved = True
            session.arm_enabled = False
            try:
                self.authorize_mission(session, revoke=True)
            except MissionFailure:
                pass
            self.transition(None, session, MissionPhase.STOPPING,
                            'stopping PiPER-owned process groups')
            session.processes_stopped = self.processes.stop_all()
            if not session.processes_stopped:
                return MissionFailure(
                    'one or more PiPER-owned processes remain alive', True)
            return None
        except MissionFailure as exc:
            return MissionFailure(str(exc), True)

    def at_configured_home(self, session, tolerance_rad=0.025):
        self.require_fresh_joint_feedback()
        current = np.asarray(self.latest_joints.position[:6], dtype=float)
        target = np.asarray(session.home_positions_rad, dtype=float)
        return bool(
            current.shape == (6,)
            and target.shape == (6,)
            and np.all(np.isfinite(current))
            and np.all(np.isfinite(target))
            and float(np.max(np.abs(current - target))) <= float(tolerance_rad)
        )

    def prove_return_home_for_shutdown(self, session):
        """Request a bounded approved-path retrace and verify its endpoint."""
        try:
            if self.at_configured_home(session):
                session.return_home_proved = True
                return True
        except MissionFailure:
            return False

        # The first cancellation can interrupt an in-flight SDK MoveJ and
        # establish a stable current-position hold.  Once that hold settles,
        # the second request starts the freshly revalidated reverse path.
        for attempt in range(2):
            requested_at = time.monotonic()
            result = self.call_service(
                self.cancel_client, Trigger.Request(), 8.0,
                'executor cancel-and-home service')
            if not result.success:
                return False
            deadline = time.monotonic() + PLAN_RESULT_TIMEOUT_SEC
            while time.monotonic() < deadline:
                status = self.latest_execution
                status_is_fresh = (
                    status is not None
                    and self.latest_execution_at >= requested_at)
                if status_is_fresh:
                    state = str(status.state)
                    reason = str(status.reason)
                    if (
                            state == 'ABORTED'
                            and 'configured home reached' in reason.lower()):
                        if self.at_configured_home(session):
                            session.return_home_proved = True
                            return True
                        return False
                    if (
                            state == 'ABORTED'
                            and 'current joint hold' in str(result.message).lower()):
                        break
                failed = self.processes.failed()
                if 'driver' in failed or 'executor' in failed:
                    return False
                time.sleep(0.05)
            if attempt == 0:
                try:
                    self.wait_for_stable_joint_stream(
                        0.5, 15.0, 'cancelled-motion hold before home retrace')
                except MissionFailure:
                    return False
        return False

    def prove_current_hold_for_shutdown(self, session):
        try:
            self.require_fresh_joint_feedback()
            initial = energized_hold_target(self.latest_joints.position[:6])
            result = self.call_service(
                self.cancel_client, Trigger.Request(), 8.0,
                'executor shutdown hold service')
            if not result.success or 'hold requested' not in str(result.message):
                return False
            deadline, settled_since = time.monotonic() + 15.0, None
            while time.monotonic() < deadline:
                if self.latest_joints is None or time.monotonic() - self.latest_joints_at > 1.0:
                    settled_since = None
                else:
                    current = np.asarray(self.latest_joints.position[:6], dtype=float)
                    delta = float(np.max(np.abs(current - initial)))
                    velocity_values = np.asarray(
                        self.latest_joints.velocity[:6], dtype=float)
                    velocity = float(np.max(np.abs(velocity_values))) \
                        if velocity_values.size == 6 else math.inf
                    if delta <= 0.005 and velocity <= 0.20:
                        settled_since = settled_since or time.monotonic()
                        if time.monotonic() - settled_since >= 1.0:
                            session.current_hold_proved = True
                            return True
                    else:
                        settled_since = None
                time.sleep(0.05)
        except MissionFailure:
            return False
        return False

    def call_enable(self, enabled):
        request = Enable.Request()
        request.enable_request = bool(enabled)
        result = self.call_service(
            self.enable_client, request, 20.0,
            'PiPER %s service' % ('enable' if enabled else 'disable'))
        if not bool(result.enable_response):
            raise MissionFailure(
                'PiPER did not confirm %s' % ('enable' if enabled else 'disable'),
                needs_operator=not enabled)

    def call_service(self, client, request, timeout, label):
        if not client.wait_for_service(timeout_sec=min(float(timeout), 5.0)):
            raise MissionFailure('%s is unavailable' % label)
        future = client.call_async(request)
        deadline = time.monotonic() + float(timeout)
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done():
            try:
                future.cancel()
            except Exception:
                pass
            raise MissionFailure('%s timed out' % label)
        try:
            result = future.result()
        except Exception as exc:
            raise MissionFailure('%s failed: %s' % (label, exc))
        if result is None:
            raise MissionFailure('%s returned no response' % label)
        return result

    def wait_for(self, goal_handle, session, predicate, timeout, failure):
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            self.guard(goal_handle, session)
            if predicate():
                return
            time.sleep(0.05)
        reason = failure() if callable(failure) else str(failure)
        raise MissionFailure(reason or 'bounded wait timed out')

    def wait_for_stable_readiness(
            self, goal_handle, session, mode, stable_sec, timeout_sec):
        """Require one readiness generation to stay good before enable."""
        # Camera health and the worker heartbeat can briefly cross their ready
        # thresholds during startup. A single good publication is not enough
        # authority to enable the arm and immediately request acquisition.
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        stable_since = 0.0
        last_reason = '%s readiness is missing' % mode
        while time.monotonic() < deadline:
            self.guard(goal_handle, session)
            now = time.monotonic()
            last_reason = self.readiness_rejection(mode)
            stable_since = readiness_stability_update(
                stable_since, last_reason, now)
            if stable_since > 0.0 and now - stable_since >= float(stable_sec):
                return
            time.sleep(0.05)
        raise MissionFailure(
            '%s readiness did not remain healthy for %.1f seconds: %s'
            % (mode, float(stable_sec), last_reason or 'readiness flapped'))

    def guard(self, goal_handle, session):
        session.accepted_captures = int(
            self.latest_capture.get('captured_frame_count',
                                    session.accepted_captures))
        if goal_handle is not None and goal_handle.is_cancel_requested:
            raise MissionFailure(
                'tracked robot cancelled the task', outcome='CANCELLED')
        if session.deadline_expired():
            raise MissionFailure('mission deadline expired')
        if self.param_bool('require_gateway_heartbeat'):
            try:
                heartbeat = self.spool.read('heartbeat', session.task_id)
            except (FileNotFoundError, OSError, ValueError):
                heartbeat = {}
            wall_time = float(heartbeat.get('wall_time_sec', 0.0))
            if wall_time > 0.0 and time.time() - wall_time <= 2.0:
                session.heartbeat()
            if bool(heartbeat.get('cancel_requested', False)):
                raise MissionFailure(
                    'tracked robot cancelled the task', outcome='CANCELLED')
            if session.heartbeat_stale():
                raise MissionFailure(
                    'tracked-robot gateway heartbeat was stale for 5 seconds')
        failed = self.processes.failed()
        if failed:
            raise MissionFailure('managed process exited: %s' % failed)
        self.publish_feedback(goal_handle, session)

    def readiness_rejection(self, mode):
        readiness = self.latest_readiness
        if readiness is None or time.monotonic() - self.latest_readiness_at > 1.0:
            return 'Tesseract readiness is missing or stale'
        if not readiness.worker_ready:
            return 'Tesseract worker is not ready'
        if mode == 'acquisition':
            return '' if readiness.acquisition_ready else '; '.join(
                readiness.acquisition_blockers)
        if mode == 'multiview':
            return '' if readiness.multiview_ready else '; '.join(
                readiness.multiview_blockers)
        if mode == 'manipulation':
            return '' if readiness.manipulation_ready else '; '.join(
                readiness.manipulation_blockers)
        return 'unsupported Tesseract readiness mode'

    def require_fresh_joint_feedback(self):
        joints = self.latest_joints
        if joints is None or time.monotonic() - self.latest_joints_at > 1.0:
            raise MissionFailure('joint feedback is missing or stale')
        values = np.asarray(joints.position[:6], dtype=float)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise MissionFailure('joint feedback is not six finite positions')

    def clear_plan_cache(self):
        with self._lock:
            self.latest_plan = None
            self.latest_plan_at = 0.0

    def clear_runtime_caches(self):
        """Discard evidence and terminal messages from the previous mission."""
        with self._lock:
            self.latest_readiness = None
            self.latest_readiness_at = 0.0
            self.latest_plan = None
            self.latest_plan_at = 0.0
            self.latest_execution = None
            self.latest_execution_at = 0.0
            self.latest_capture = {}
            self.latest_capture_at = 0.0

    def transition(self, goal_handle, session, phase, reason):
        session.transition(phase, reason)
        self.write_status(session)
        self.publish_feedback(goal_handle, session)

    def startup_progress(self, goal_handle, session, reason):
        session.reason = str(reason)
        self.write_status(session)
        self.publish_feedback(goal_handle, session)

    def write_status(self, session):
        self.spool.write('status', session.task_id, {
            'task_id': session.task_id,
            'mission_sha256': session.mission_sha256,
            'phase': session.phase.value,
            'reason': session.reason,
            'elapsed_sec': session.elapsed(),
            'remaining_sec': session.remaining(),
            'accepted_captures': session.accepted_captures,
            'required_captures': REQUIRED_CAPTURES,
            'processes': self.processes.health(),
        })

    def publish_feedback(self, goal_handle, session):
        if goal_handle is None:
            return
        feedback = RunTargetScan.Feedback()
        feedback.phase = session.phase.value
        feedback.reason = session.reason
        feedback.elapsed_sec = float(session.elapsed())
        feedback.remaining_sec = float(session.remaining())
        feedback.acquisition_attempt = int(session.acquisition_attempt)
        feedback.occlusion_action = int(session.occlusion_action)
        feedback.occlusion_action_limit = MAX_OCCLUSION_ACTIONS
        feedback.accepted_captures = int(
            self.latest_capture.get('captured_frame_count', 0))
        feedback.required_captures = REQUIRED_CAPTURES
        feedback.process_health_json = json.dumps(
            self.processes.health(), sort_keys=True)
        feedback.shutdown_phase = (
            session.phase.value if session.phase in (
                MissionPhase.RETURNING_HOME, MissionPhase.HOLDING,
                MissionPhase.DISABLING, MissionPhase.STOPPING) else '')
        goal_handle.publish_feedback(feedback)

    def poll_spool_goals(self):
        if self._spool_thread is not None and self._spool_thread.is_alive():
            return
        goals_dir = self.spool.root / 'goals'
        for path in sorted(goals_dir.glob('*.json')):
            task_id = path.stem
            with self._lock:
                if (task_id in self._spool_seen
                        or task_id in self.registry.results):
                    continue
                self._spool_seen.add(task_id)
            self._spool_thread = threading.Thread(
                target=self.run_spool_goal, args=(task_id,), daemon=True)
            self._spool_thread.start()
            return

    def run_spool_goal(self, task_id):
        try:
            payload = self.spool.read('goals', task_id)
            request = RunTargetScan.Goal()
            request.task_id = str(payload['task_id'])
            request.task_type = str(payload['task_type'])
            request.target_label = str(payload['target_label'])
            request.target_profile = str(payload['target_profile'])
            request.target_confidence = float(payload['target_confidence'])
            request.deadline_sec = float(payload['deadline_sec'])
            target = payload['rough_target']
            request.rough_target.header.frame_id = str(target['frame_id'])
            stamp = float(target['stamp_sec'])
            request.rough_target.header.stamp.sec = int(stamp)
            request.rough_target.header.stamp.nanosec = int(
                (stamp - int(stamp)) * 1e9)
            position = target['position']
            request.rough_target.pose.pose.position.x = float(position[0])
            request.rough_target.pose.pose.position.y = float(position[1])
            request.rough_target.pose.pose.position.z = float(position[2])
            request.rough_target.pose.pose.orientation.w = 1.0
            request.rough_target.pose.covariance = list(target['covariance'])
            self.execute_cb(SpoolGoalHandle(request))
        except Exception as exc:
            self.get_logger().error(
                'spooled mission %s could not execute: %s' % (task_id, exc))

    @staticmethod
    def action_result(outcome, reason, safe_shutdown):
        result = RunTargetScan.Result()
        outcomes = {
            # Constants declared before the first separator in a ROS action
            # belong to the generated Goal class, not the Result instance.
            'SUCCEEDED': RunTargetScan.Goal.OUTCOME_SUCCEEDED,
            'FAILED': RunTargetScan.Goal.OUTCOME_FAILED,
            'CANCELLED': RunTargetScan.Goal.OUTCOME_CANCELLED,
            'BUSY': RunTargetScan.Goal.OUTCOME_BUSY,
            'UNSUPPORTED_TARGET_PROFILE': (
                RunTargetScan.Goal.OUTCOME_UNSUPPORTED_TARGET_PROFILE),
            'NEEDS_OPERATOR': RunTargetScan.Goal.OUTCOME_NEEDS_OPERATOR,
        }
        result.outcome = outcomes.get(
            str(outcome), RunTargetScan.Goal.OUTCOME_FAILED)
        result.reason = str(reason)
        result.safe_shutdown = bool(safe_shutdown)
        return result

    def result_message(self, payload):
        result = self.action_result(
            payload.get('outcome', 'FAILED'), payload.get('reason', ''),
            payload.get('safe_shutdown', False))
        result.dataset_path = str(payload.get('dataset_path', ''))
        result.manifest_sha256 = str(payload.get('manifest_sha256', ''))
        result.capture_count = int(payload.get('capture_count', 0))
        result.mesh_job_id = str(payload.get('mesh_job_id', ''))
        result.action_summary_json = json.dumps(
            payload.get('action_summary', {}), sort_keys=True)
        return result


def main(args=None):
    rclpy.init(args=args)
    node = TargetScanMissionNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.action_server.destroy()
        node.destroy_node()
        rclpy.shutdown()


class SpoolGoalHandle:
    """Small action-handle adapter for the isolated filesystem gateway."""

    def __init__(self, request):
        self.request = request
        self.is_cancel_requested = False

    def publish_feedback(self, _feedback):
        return None

    def succeed(self):
        return None

    def abort(self):
        return None

    def canceled(self):
        return None


if __name__ == '__main__':
    main()
