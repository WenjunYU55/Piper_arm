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
from piper_msgs.msg import PiperStatusMsg
from piper_msgs.srv import Enable
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from piper_mobile_manipulation.action import RunTargetScan
from piper_mobile_manipulation.home_pose import (
    load_home_pose,
    staged_home_targets,
    validate_home_profile_limits,
    validate_staged_wrist_direction,
)
from piper_mobile_manipulation.mission_core import (
    MISSION_QUEUE_COALESCE_SEC,
    MAX_FEATURE_CAPTURES,
    MAX_OCCLUSION_ACTIONS,
    MAX_PENDING_MISSIONS,
    MissionPhase,
    MissionRegistry,
    REQUIRED_CAPTURES,
    closest_pending_mission,
    mission_queue_ready,
    queued_cancel_result,
    validate_goal_payload,
)
from piper_mobile_manipulation.mission_spool import MissionSpool
from piper_mobile_manipulation.reconstruction_jobs import (
    mesh_job_id,
    waiting_job,
)
from piper_mobile_manipulation.msg import (
    CameraTimestampHealth,
    ScanExecutionPlan,
    ScanExecutionStatus,
    TesseractReadiness,
)
from piper_mobile_manipulation.scan_capture import rigid_transform_matrix
from piper_mobile_manipulation.scan_motion import (
    energized_hold_target,
    motor_control_reasons,
    motor_driver_states,
    URDF_JOINT_LIMITS,
)
from piper_mobile_manipulation.scan_session_memory import (
    achieved_feature_coverage,
    history_coverage_target_center,
    validate_history_payload,
)
from piper_mobile_manipulation.scan_viewpoint_executor_node import (
    abort_return_home_blocker,
)
from piper_mobile_manipulation.startup_gates import (
    joint_sample_rejection,
    joint_stability_update,
    readiness_stability_update,
    worker_health_rejection,
)
from piper_mobile_manipulation.surface_coverage import (
    measured_surface_coverage,
    persisted_achieved_history,
)
from piper_mobile_manipulation.srv import (
    ApproveScanExecution,
    AuthorizeMission,
    GetTargetScanResult,
    PrepareAcquisition,
    RequestTesseractPlan,
)


ACQUISITION_SERVICE_TIMEOUT_SEC = 8.0
WORKFLOW_ASSESSMENT_TIMEOUT_SEC = 75.0
PLAN_REQUEST_QUEUE_TIMEOUT_SEC = 12.0
PLAN_RESULT_TIMEOUT_SEC = 185.0
PLAN_APPROVAL_TRANSIENT_TIMEOUT_SEC = 5.0
SCAN_VISUAL_REACQUISITION_TIMEOUT_SEC = 30.0
MAX_SCAN_QUALITY_REPLANS = 8
MAX_SCAN_TARGET_DRIFT_REPLANS = 8


class ManagedProcessSet:
    """Own exact process groups and stop them in reverse dependency order."""

    def __init__(self, log_root):
        self.log_root = Path(log_root)
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.entries = {}
        self.log_offsets = {}

    def begin_generation(self):
        """Forget only fully stopped entries before admitting a new mission."""
        live = sorted(
            name for name, (process, _log, _path) in self.entries.items()
            if process.poll() is None)
        if live:
            return live
        for _process, log, _path in self.entries.values():
            if not log.closed:
                log.close()
        self.entries.clear()
        self.log_offsets.clear()
        return []

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
    def __init__(self, reason, needs_operator=False, outcome='FAILED',
                 failure_code='', retryable=None):
        super().__init__(reason)
        self.needs_operator = bool(needs_operator)
        self.outcome = str(outcome)
        self.failure_code = str(failure_code)
        self.retryable = (
            not self.needs_operator if retryable is None else bool(retryable))


def failure_code_for_reason(reason):
    """Return one stable tracked-robot failure code for legacy exceptions."""
    lowered = str(reason).lower()
    if 'cancel' in lowered:
        return 'CANCELLED'
    # Preserve the actionable root cause when a sensor startup itself times
    # out instead of misreporting it as the overall mission deadline.
    if 'camera' in lowered or 'vision' in lowered or 'sensor' in lowered:
        return 'SENSOR_UNAVAILABLE'
    if 'deadline' in lowered or 'timed out' in lowered:
        return 'DEADLINE_EXPIRED'
    if 'capture' in lowered or 'quality' in lowered:
        return 'INSUFFICIENT_CAPTURE_QUALITY'
    if (
            'occlud' in lowered
            or 'occlusion' in lowered
            or 'manipulation' in lowered):
        return 'OCCLUSION_NOT_CLEARED'
    if 'target' in lowered and ('not found' in lowered or 'lock' in lowered):
        return 'TARGET_NOT_FOUND'
    if (
            'plan' in lowered or 'reachable' in lowered or 'ik' in lowered
            or 'scan candidate' in lowered or 'view frontier' in lowered):
        return 'NO_REACHABLE_PLAN'
    if any(term in lowered for term in (
            'joint feedback', 'collision', 'hold', 'disable',
            'control', 'arm status')) or any(
                token == 'can' for token in lowered.replace(':', ' ').split()):
        return 'CONTROL_UNTRUSTWORTHY'
    return 'MISSION_FAILED'


def retryable_plan_approval_rejection(reason):
    """Retry only an unchanged plan's transient live-state gate."""
    text = str(reason).lower()
    if not text.startswith('execution blocked:'):
        return False
    return any(marker in text for marker in (
        'tracking is not settled tracking',
        'tracking is prediction-only',
        'tracking speed scale is below the motion threshold',
        'camera timestamp health is stale',
        'joint feedback is not settled',
        # TrackedTarget and target_status are separate ROS topics published
        # back-to-back by the tracker.  At a closed-loop replan boundary the
        # executor can therefore briefly have the new measured target and the
        # preceding LOW_CONFIDENCE/LOST/SEARCHING string.  No motion is
        # authorized while approval is rejected; retain the exact proposal
        # long enough for the automatic SAM2/GroundingDINO recovery to publish
        # a fresh measured lock.
        'target_status=low_confidence',
        'target_status=lost',
        'target_status=searching',
    ))


def visual_reacquisition_plan_approval_rejection(reason):
    """Recognize a no-motion scan approval wait for a fresh measured lock."""
    text = str(reason).lower()
    return (
        text.startswith('execution blocked:')
        and any(marker in text for marker in (
            'target_status=low_confidence',
            'target_status=lost',
            'target_status=searching',
        ))
    )


def visual_reacquisition_plan_request_rejection(reason):
    """Recognize a command-free planning snapshot that only lacks a lock."""
    text = str(reason).lower()
    return (
        text.startswith('planning blocked:')
        and any(marker in text for marker in (
            'tracking is not settled tracking',
            'tracking is prediction-only',
            'tracking measurement is stale',
        ))
    )


def shutdown_uses_startup_home(session):
    """Use static home authority only before perception owns a scene."""
    return bool(
        not getattr(session, 'perception_scene_established', False)
        and int(getattr(session, 'accepted_captures', 0)) == 0)


def target_drift_requires_replan(reason):
    """Recognize the executor's no-motion stale-target-plan rejection."""
    text = str(reason).lower().strip()
    return (
        text.startswith('target moved ')
        and ' after planning; refresh the plan' in text
    )


def safe_view_exhaustion_after_capture(
        reason, accepted_captures, feature_coverage=None,
        required_captures=REQUIRED_CAPTURES):
    """Recognize a proved end of the adaptive safe-view frontier.

    This is deliberately narrower than a generic planning failure.  It only
    applies after the model-seed capture floor and only when the one-view
    Tesseract transaction reports that none of the remaining distinct
    candidates has a finite, bounded, collision-free IK solution.
    """
    text = str(reason).lower().strip()
    return (
        int(accepted_captures) >= int(required_captures)
        and isinstance(feature_coverage, dict)
        and feature_coverage.get('sufficient') is True
        and 'multiview_scan planning failed:' in text
        and 'planning_failed: only 0 viewpoints planned; require at least 1 of 1'
        in text
        and 'no finite bounded collision-free ik goal for any roll' in text
    )


def feature_capture_decision(
        accepted_captures, required_captures, maximum_captures,
        feature_coverage):
    """Choose continue, complete, or exhausted from achieved feature proof."""
    accepted = int(accepted_captures)
    required = int(required_captures)
    maximum = int(maximum_captures)
    if required < 1 or maximum < required or accepted < 0:
        raise ValueError('feature capture bounds are invalid')
    sufficient = bool(
        isinstance(feature_coverage, dict)
        and feature_coverage.get('sufficient') is True)
    if accepted >= required and sufficient:
        return 'COMPLETE'
    if accepted >= maximum:
        return 'EXHAUSTED'
    return 'CONTINUE'


def planning_rejection_allows_current_state_home(reason):
    """Identify failures that may be re-qualified by a fresh home plan.

    A rejected proposal is not evidence that a new home route is unsafe.  A
    transient invalid obstacle snapshot also is not permanent evidence: the
    dedicated current-state home transaction takes a new scene snapshot and
    independently fails closed if geometry is still invalid.  All hardware,
    collision, clearance, progress, and general obstacle faults remain under
    ``abort_return_home_blocker``.
    """
    text = str(reason).lower().strip()
    return (
        (
            'planning failed: tesseract proposal rejected:' in text
            and 'planning_failed:' in text
        )
        or 'runtime safety gate: invalid obstacle geometry is present' in text
        or (
            'fresh runtime telemetry did not arrive' in text
            and 'obstacles data missing or stale' in text
        )
    )


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
            'maximum_captures': MAX_FEATURE_CAPTURES,
            'home_pose_path': '',
            'require_staged_home_profile': True,
            'mission_spool_root': os.path.join(
                os.environ.get('XDG_RUNTIME_DIR', '/tmp'),
                'piper_target_scan_missions'),
            'process_log_root': os.path.join(
                os.environ.get('XDG_RUNTIME_DIR', '/tmp'),
                'piper_target_scan_logs'),
            'require_gateway_heartbeat': False,
            'max_pending_missions': MAX_PENDING_MISSIONS,
            'mission_queue_coalesce_sec': MISSION_QUEUE_COALESCE_SEC,
            'debug': True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.callback_group = ReentrantCallbackGroup()
        self._lock = threading.RLock()
        self._spool_seen = set()
        self._pending_missions = {}
        self._action_reservations = {}
        self._prevalidated_goals = {}
        self._queue_sequence = 0
        self._dispatch_task_id = ''
        self._process_shutdown_requested = False
        self._process_shutdown_quiescent_since = 0.0
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
        self.last_return_home_diagnostic = ''
        self.latest_joints = None
        self.latest_joints_at = 0.0
        self.latest_joint_source_ns = 0
        self.joint_generation_started_ns = 0
        self.joint_stream_stable_since = 0.0
        self.joint_stream_reference = None
        self.joint_feedback_rejection = 'driver generation has not started'
        self.latest_arm_status = None
        self.latest_arm_status_at = 0.0
        self.motor_enable_guard_after = 0.0
        self.latest_camera_health = None
        self.latest_camera_health_at = 0.0
        self.latest_scan_history = None
        self.latest_scan_history_at = 0.0
        self.latest_scan_target_center = None
        self.last_scan_feature_coverage = {}
        self.current_home_profile = None

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
            String, '/piper/scan_session_history', self.scan_history_cb,
            latched)
        self.create_subscription(
            JointState, '/joint_states_single', self.joint_cb, 10)
        self.create_subscription(
            PiperStatusMsg, '/arm_status', self.arm_status_cb, 10)
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
        self.return_home_plan_client = self.create_client(
            RequestTesseractPlan,
            '/tesseract_plan_bridge/request_return_home_plan',
            callback_group=self.callback_group)
        self.startup_home_plan_client = self.create_client(
            RequestTesseractPlan,
            '/tesseract_plan_bridge/request_startup_home_plan',
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
        self.hold_client = self.create_client(
            Trigger, '/scan_viewpoint_executor/hold',
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
            handle_accepted_callback=self.handle_accepted_cb,
            cancel_callback=self.cancel_cb,
            callback_group=self.callback_group)
        self.create_service(
            GetTargetScanResult, '/piper/get_target_scan_result',
            self.get_result_cb, callback_group=self.callback_group)
        self.create_timer(0.1, self.poll_mission_queue)
        self.create_timer(0.1, self.poll_process_shutdown)

    def request_process_shutdown(self):
        """Turn coordinator SIGINT into the normal bounded cancel path."""
        with self._lock:
            first_request = not self._process_shutdown_requested
            self._process_shutdown_requested = True
            self._process_shutdown_quiescent_since = 0.0
        if first_request:
            self.get_logger().warn(
                'coordinator shutdown requested; cancelling any active '
                'mission through home, hold, disable, and child cleanup')

    def poll_process_shutdown(self):
        """Stop ROS only after the active mission and owned children finish."""
        with self._lock:
            if not self._process_shutdown_requested:
                return
            active = self.registry.active is not None
            queued = bool(
                self._pending_missions
                or self._action_reservations
                or self._dispatch_task_id)
        if active or queued:
            self._process_shutdown_quiescent_since = 0.0
            return
        live = any(
            process.poll() is None
            for process, _log, _path in self.processes.entries.values())
        if live:
            self._process_shutdown_quiescent_since = 0.0
            return
        now = time.monotonic()
        if self._process_shutdown_quiescent_since <= 0.0:
            # Give an action result one second to leave the process before its
            # DDS endpoints are destroyed.
            self._process_shutdown_quiescent_since = now
            return
        if now - self._process_shutdown_quiescent_since >= 1.0 and rclpy.ok():
            rclpy.shutdown()

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
            if (
                    str(msg.plan_kind) == 'MULTIVIEW_SCAN'
                    and bool(msg.valid)):
                self.latest_scan_target_center = {
                    'x': float(msg.target_center.x),
                    'y': float(msg.target_center.y),
                    'z': float(msg.target_center.z),
                }

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

    def scan_history_cb(self, msg):
        try:
            payload = validate_history_payload(
                json.loads(msg.data),
                int(self.get_parameter('maximum_captures').value))
        except (json.JSONDecodeError, TypeError, ValueError):
            return
        with self._lock:
            self.latest_scan_history = payload
            self.latest_scan_history_at = time.monotonic()

    def current_scan_feature_coverage(self):
        history = self.latest_scan_history or {}
        persisted = persisted_achieved_history(
            self.latest_capture.get('scan_dir', ''))
        achieved_entries = history.get('accepted_entries', [])
        if (
                persisted.get('available')
                and len(persisted.get('entries', [])) > len(achieved_entries)):
            achieved_entries = persisted['entries']
        target_center = history_coverage_target_center(
            history, self.latest_scan_target_center)
        if target_center is None:
            target_center = persisted.get('target_center')
        surface = measured_surface_coverage(
            self.latest_capture.get('scan_dir', ''),
            minimum_views=int(self.get_parameter('required_captures').value))
        coverage = achieved_feature_coverage(
            achieved_entries,
            target_center,
            minimum_views=int(self.get_parameter('required_captures').value),
            surface_coverage=surface)
        coverage['achieved_history_source'] = (
            'persisted_capture_metadata'
            if achieved_entries is persisted.get('entries')
            else 'transient_scan_history')
        coverage['persisted_history_reason'] = persisted.get('reason', '')
        self.last_scan_feature_coverage = coverage
        return coverage

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

    def arm_status_cb(self, msg):
        with self._lock:
            self.latest_arm_status = msg
            self.latest_arm_status_at = time.monotonic()

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
            task_id = normalized['task_id']
            if (
                    task_id in self._action_reservations
                    or task_id in self._pending_missions
                    or (
                        self.registry.active is not None
                        and self.registry.active.task_id == task_id)):
                return GoalResponse.REJECT
            pending = (
                len(self._action_reservations)
                + len(self._pending_missions))
            maximum = int(
                self.get_parameter('max_pending_missions').value)
            if maximum < 1 or pending >= maximum:
                self.get_logger().warn(
                    'rejecting target-scan goal: bounded mission queue is full')
                return GoalResponse.REJECT
            self._queue_sequence += 1
            self._action_reservations[task_id] = {
                'normalized': normalized,
                'sequence': self._queue_sequence,
                'admitted_monotonic': time.monotonic(),
                'source': 'action',
            }
            return GoalResponse.ACCEPT

    def handle_accepted_cb(self, goal_handle):
        """Hold accepted action goals until closest-first dispatch selects one."""
        task_id = str(goal_handle.request.task_id)
        with self._lock:
            cached = self.registry.results.get(task_id)
            if cached is not None:
                try:
                    normalized = validate_goal_payload(
                        self.goal_payload(goal_handle.request))
                except (TypeError, ValueError):
                    goal_handle.execute()
                    return
                self._prevalidated_goals[task_id] = normalized
                goal_handle.execute()
                return
            record = self._action_reservations.pop(task_id, None)
            if record is None:
                self.get_logger().error(
                    'accepted action goal %s has no queue reservation' % task_id)
                goal_handle.execute()
                return
            record['goal_handle'] = goal_handle
            self._pending_missions[task_id] = record
            self.write_queued_status(record)

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
        task_id = str(goal_handle.request.task_id)
        with self._lock:
            normalized = self._prevalidated_goals.pop(task_id, None)
        if normalized is None:
            try:
                normalized = validate_goal_payload(
                    self.goal_payload(goal_handle.request))
            except (TypeError, ValueError) as exc:
                goal_handle.abort()
                self.finish_queue_dispatch(task_id)
                return self.action_result('FAILED', str(exc), False)
        if (
                bool(goal_handle.is_cancel_requested)
                or self._process_shutdown_requested):
            reason = (
                'coordinator stopped before queued mission started'
                if self._process_shutdown_requested
                else 'tracked robot cancelled queued mission before start')
            result = self.finish_queued_cancel(normalized, reason)
            self.finish_action_handle(goal_handle, 'CANCELLED')
            self.finish_queue_dispatch(task_id)
            return self.result_message(result)
        with self._lock:
            status, record = self.registry.admit(normalized)
        if status == 'CACHED':
            if record.get('outcome') == 'SUCCEEDED':
                goal_handle.succeed()
            elif record.get('outcome') == 'CANCELLED':
                self.finish_action_handle(goal_handle, 'CANCELLED')
            else:
                goal_handle.abort()
            self.finish_queue_dispatch(task_id)
            return self.result_message(record)
        if status != 'ACCEPTED':
            goal_handle.abort()
            self.finish_queue_dispatch(task_id)
            return self.action_result(status, 'task was not admitted', False)
        session = record
        self.clear_runtime_caches()
        self.write_status(session)
        transformed_target = None
        failure = None
        owns_process_generation = False
        try:
            live_processes = self.processes.begin_generation()
            if live_processes:
                raise MissionFailure(
                    'previous mission still owns live process groups: %s'
                    % ', '.join(live_processes),
                    needs_operator=True,
                    failure_code='CONTROL_UNTRUSTWORTHY', retryable=False)
            owns_process_generation = True
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
            # A generation-admission failure means every live process group
            # belongs to the previous mission.  Never let the rejected mission
            # stop command owners that it did not start, especially while the
            # previous arm state may still be enabled and awaiting recovery.
            shutdown_failure = (
                self.safe_shutdown(
                    session, normal_completion=False, failure=failure)
                if owns_process_generation else None)
            if shutdown_failure is not None:
                failure = MissionFailure(
                    '%s; safe shutdown also failed: %s'
                    % (failure, shutdown_failure),
                    needs_operator=True,
                    failure_code='CONTROL_UNTRUSTWORTHY', retryable=False)
            elif failure.outcome == 'CANCELLED':
                failure = MissionFailure(
                    'task failed: cancelled; arm returned to configured home, '
                    'disabled, and pipeline stopped; please retry',
                    outcome='CANCELLED', failure_code='CANCELLED',
                    retryable=True)

        capture = dict(self.latest_capture)
        if failure is None:
            session.phase = MissionPhase.SUCCEEDED
            session.reason = (
                'distinctive-feature target scan completed with %d accepted '
                'diverse views and PiPER shut down safely'
                % session.accepted_captures)
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
                self.finish_action_handle(goal_handle, 'CANCELLED')
            else:
                goal_handle.abort()
        pending_mesh_job_id = ''
        if failure is None:
            pending_mesh_job_id = mesh_job_id(
                session.task_id,
                str(capture.get('manifest_sha256', '')))
        result = session.result_payload(
            outcome, session.reason,
            dataset_path=str(capture.get('scan_dir', '')),
            manifest_sha256=str(capture.get('manifest_sha256', '')),
            mesh_job_id=pending_mesh_job_id,
            failure_code=(
                '' if failure is None else (
                    failure.failure_code
                    or failure_code_for_reason(failure))),
            retryable=(False if failure is None else failure.retryable),
            action_summary={
                'processes': self.processes.health(),
                'home_positions_rad': list(session.home_positions_rad),
                'storage_positions_rad': list(
                    session.storage_positions_rad),
                'startup_wrist_completed': bool(
                    session.startup_wrist_completed),
                'storage_wrist_proved': bool(
                    session.storage_wrist_proved),
                'motor_control_lost_reason': str(
                    session.motor_control_lost_reason),
                'target_label': session.goal['target_label'],
                'target_profile': session.goal['target_profile'],
                'target_prompt': session.goal['target_prompt'],
                'scan_feature_coverage': dict(
                    self.last_scan_feature_coverage),
            },
        )
        with self._lock:
            self.registry.finish(result)
        self.spool.write('results', session.task_id, result)
        if failure is None:
            self.spool.write(
                'mesh_jobs', pending_mesh_job_id, waiting_job(result))
        self.finish_queue_dispatch(task_id)
        return self.result_message(result)

    def finish_queued_cancel(self, normalized, reason):
        result = queued_cancel_result(normalized, reason)
        task_id = str(normalized['task_id'])
        with self._lock:
            self.registry.results[task_id] = dict(result)
        self.spool.write('results', task_id, result)
        return result

    def finish_queue_dispatch(self, task_id):
        with self._lock:
            if self._dispatch_task_id == str(task_id):
                self._dispatch_task_id = ''

    @staticmethod
    def finish_action_handle(goal_handle, outcome):
        """Use CANCELED only after ROS accepted a client cancel transition."""
        if (
                str(outcome) == 'CANCELLED'
                and bool(goal_handle.is_cancel_requested)):
            goal_handle.canceled()
        else:
            goal_handle.abort()

    def run_pipeline(self, goal_handle, session, target):
        profile = self.selected_home_profile()
        self.current_home_profile = profile
        session.home_positions_rad = tuple(profile['positions_rad'])
        session.mission_ready_joint6_rad = float(
            profile['mission_ready_joint6_rad'])
        session.storage_joint6_rad = float(profile['storage_joint6_rad'])
        session.storage_positions_rad = tuple(
            list(session.home_positions_rad[:5])
            + [session.storage_joint6_rad])
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
            2.0, 15.0, 'pre-enable joint feedback', goal_handle, session)

        self.transition(goal_handle, session, MissionPhase.PREFLIGHT,
                        'validating current feedback and mission authority')
        self.require_fresh_joint_feedback()
        try:
            validate_staged_wrist_direction(
                profile, self.latest_joints.position[:6])
        except (TypeError, ValueError) as exc:
            raise MissionFailure(
                'configured startup wrist direction is unsafe: %s; arm '
                'remained disabled' % exc,
                failure_code='CONTROL_UNTRUSTWORTHY', retryable=False)
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
        # The enable service has already proved all six FOC flags. Allow one
        # short publication interval for /arm_status to replace its last
        # pre-enable sample, then continuously require all six axes enabled.
        self.motor_enable_guard_after = time.monotonic() + 0.5
        # Bind the first post-enable feedback immediately. Waiting for a
        # passive stable window before commanding hold lets gravity settling
        # continue indefinitely on some powered starts.
        if not self.prove_current_hold(goal_handle, session):
            raise MissionFailure(
                'current-position hold did not settle after enable: '
                + self.last_hold_diagnostic, True)

        startup_targets = staged_home_targets(
            profile, self.latest_joints.position[:6])
        self.transition(
            goal_handle, session, MissionPhase.RETURNING_HOME,
            'rotating J6 from the measured powered start to the configured '
            'mission-ready wrist angle')
        if not self.prove_return_home_for_shutdown(
                session, startup=True, goal_handle=goal_handle,
                target_positions=startup_targets[
                    'startup_wrist_positions_rad'],
                home_stage='STARTUP_WRIST'):
            raise MissionFailure(
                'startup wrist normalization was not proved; arm remains in '
                'a current-position hold: ' + self.last_return_home_diagnostic,
                True, failure_code='CONTROL_UNTRUSTWORTHY', retryable=False)
        session.startup_wrist_completed = True
        # The first wrist stage is not the rough-home proof used by terminal
        # shutdown. Clear it before the independently planned full home stage.
        session.return_home_proved = False
        self.transition(
            goal_handle, session, MissionPhase.RETURNING_HOME,
            'normalizing joints 1-6 to the configured rough mission home')
        if not self.prove_return_home_for_shutdown(
                session, startup=True, goal_handle=goal_handle,
                target_positions=startup_targets[
                    'rough_home_positions_rad'],
                home_stage='ROUGH_HOME'):
            raise MissionFailure(
                'startup configured-home normalization was not proved; '
                'arm remains in a current-position hold: '
                + self.last_return_home_diagnostic, True,
                failure_code='CONTROL_UNTRUSTWORTHY', retryable=False)
        session.startup_home_completed = True

        self.transition(goal_handle, session, MissionPhase.ROUGH_ACQUISITION,
                        'starting closed-loop rough-target acquisition')
        acquired = False
        for look_index in range(5):
            session.acquisition_attempt = look_index + 1
            self.clear_plan_cache()
            request_id = self.prepare_acquisition(session, target)
            plan = self.wait_for_plan(
                goal_handle, session, 'ROUGH_ACQUISITION', request_id,
                PLAN_RESULT_TIMEOUT_SEC)
            self.approve_plan(goal_handle, session, plan)
            self.transition(
                goal_handle, session, MissionPhase.TARGET_LOCK,
                'settling and measuring acquisition look %d/5'
                % (look_index + 1))
            execution = self.wait_for_execution(
                goal_handle, session,
                ('ACQUIRED', 'ACQUISITION_LOOK_COMPLETE'), 180.0,
                ('ACQUISITION_FAILED', 'ABORTED', 'INVALID'))
            # Either terminal acquisition outcome proves that a correlated
            # semantic obstacle scene now exists. Failures before this point
            # retain the same static robot/floor/cable scene authority used by
            # startup and the first acquisition segment, so a pre-inference
            # sensor fault cannot strand an enabled arm.
            session.perception_scene_established = True
            if str(execution.state) == 'ACQUIRED':
                acquired = True
                break
            self.transition(
                goal_handle, session, MissionPhase.ROUGH_ACQUISITION,
                'target absent after look %d/5; replanning once from fresh '
                'measured arm and camera state' % (look_index + 1))
        if not acquired:
            raise MissionFailure(
                'target not found after five distinct closed-loop looks',
                failure_code='TARGET_NOT_FOUND', retryable=True)

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
        quality_replans = 0
        target_drift_replans = 0
        execution = None
        required = int(self.get_parameter('required_captures').value)
        maximum = int(self.get_parameter('maximum_captures').value)
        if required < 1 or maximum < required:
            raise MissionFailure(
                'capture bounds are invalid: minimum %d maximum %d'
                % (required, maximum),
                failure_code='MISSION_FAILED', retryable=False)
        adaptive_completion = False
        while True:
            accepted = int(
                self.latest_capture.get('captured_frame_count', 0))
            coverage = self.current_scan_feature_coverage()
            history_count = int(coverage.get('accepted_achieved_views', 0))
            decision = feature_capture_decision(
                accepted, required, maximum,
                coverage if history_count >= accepted else {})
            if decision == 'COMPLETE':
                session.accepted_captures = accepted
                adaptive_completion = True
                self.startup_progress(
                    goal_handle, session,
                    'distinctive feature floors are complete after %d '
                    'accepted views; holding for a fresh current-state home '
                    'plan' % accepted)
                break
            if decision == 'EXHAUSTED':
                raise MissionFailure(
                    '%d-view bounded scan limit reached but distinctive '
                    'feature coverage remained insufficient: %s'
                    % (accepted, '; '.join(coverage.get('blockers', []))),
                    failure_code='INSUFFICIENT_CAPTURE_QUALITY',
                    retryable=True)
            remaining = maximum - accepted
            self.transition(
                goal_handle, session, MissionPhase.VIEW_PLANNING,
                'requesting one correlated feature-driven view; up to %d '
                'bounded views remain (model seed floor %d)' % (
                    remaining, required))
            self.clear_plan_cache()
            request_id = self.request_multiview_plan(goal_handle, session)
            try:
                plan = self.wait_for_plan(
                    goal_handle, session, 'MULTIVIEW_SCAN', request_id,
                    PLAN_RESULT_TIMEOUT_SEC)
            except MissionFailure as exc:
                coverage = self.current_scan_feature_coverage()
                if not safe_view_exhaustion_after_capture(
                        exc, accepted, coverage, required):
                    if (
                            'only 0 viewpoints planned; require at least 1 of 1'
                            in str(exc)
                            and coverage.get('blockers')):
                        raise MissionFailure(
                            '%s; safe-view frontier ended before distinctive '
                            'feature coverage was sufficient: %s'
                            % (exc, '; '.join(coverage['blockers'])),
                            failure_code='INSUFFICIENT_CAPTURE_QUALITY',
                            retryable=True)
                    raise
                session.accepted_captures = accepted
                adaptive_completion = True
                self.startup_progress(
                    goal_handle, session,
                    'adaptive scan complete after %d accepted diverse views; '
                    'Tesseract proved no meaningfully different collision-free '
                    'view remains' % accepted)
                break
            try:
                self.approve_plan(goal_handle, session, plan)
            except MissionFailure as exc:
                if (
                        not target_drift_requires_replan(exc)
                        or target_drift_replans
                        >= MAX_SCAN_TARGET_DRIFT_REPLANS):
                    raise
                target_drift_replans += 1
                self.startup_progress(
                    goal_handle, session,
                    'measured target changed after planning; no motion was '
                    'authorized, replanning from the fresh lock (%d/%d)'
                    % (
                        target_drift_replans,
                        MAX_SCAN_TARGET_DRIFT_REPLANS))
                self.wait_for_stable_readiness(
                    goal_handle, session, 'multiview', 0.5, 30.0)
                continue
            self.transition(goal_handle, session, MissionPhase.CAPTURING,
                            'executing one settled quality-gated viewpoint')
            execution = self.wait_for_execution(
                goal_handle, session,
                ('VIEW_COMPLETE', 'VIEW_REJECTED', 'COMPLETE'),
                session.remaining(), ('ABORTED', 'INVALID'))
            if str(execution.state) == 'COMPLETE':
                break
            session.accepted_captures = int(
                self.latest_capture.get('captured_frame_count', 0))
            if str(execution.state) == 'VIEW_REJECTED':
                if quality_replans >= MAX_SCAN_QUALITY_REPLANS:
                    raise MissionFailure(
                        'visual replacement budget exhausted: '
                        + str(execution.reason))
                quality_replans += 1
                self.startup_progress(
                    goal_handle, session,
                    'view rejected by fresh visual gates; executor is holding, '
                    'excluding that pose and replanning '
                    '(%d/%d)' % (
                        quality_replans, MAX_SCAN_QUALITY_REPLANS))
                self.wait_for_stable_readiness(
                    goal_handle, session, 'multiview', 0.5, 30.0)
                continue
            self.startup_progress(
                goal_handle, session,
                'accepted view %d (minimum %d, bounded maximum %d); '
                'replanning one next view from measured pose and achieved '
                'feature coverage'
                % (session.accepted_captures, required, maximum))
            self.wait_for_stable_readiness(
                goal_handle, session, 'multiview', 0.5, 30.0)
        session.accepted_captures = int(
            self.latest_capture.get('captured_frame_count', 0))
        if adaptive_completion:
            # Normal mission shutdown performs one fresh, correlated, direct
            # current-state-to-home joint target. It must not reuse or reverse
            # any prior scan endpoint sequence; collision validation is
            # intentionally bypassed only for this dedicated home request.
            return
        if not required <= session.accepted_captures <= maximum:
            raise MissionFailure(
                'executor completed with %d captures outside the bounded '
                '%d-%d contract'
                % (session.accepted_captures, required, maximum))
        if 'home reached' not in str(execution.reason).lower():
            raise MissionFailure(
                'captures completed but return-home was not proved: %s'
                % execution.reason)
        session.return_home_proved = True
        self.wait_for(
            goal_handle, session,
            lambda: int((self.latest_scan_history or {}).get(
                'accepted_views', 0)) >= session.accepted_captures,
            5.0,
            'scan history did not catch up with the final accepted capture')
        coverage = self.current_scan_feature_coverage()
        if not coverage.get('sufficient'):
            raise MissionFailure(
                '%d captures completed but distinctive feature coverage was '
                'insufficient: %s'
                % (
                    session.accepted_captures,
                    '; '.join(coverage.get('blockers', []))),
                failure_code='INSUFFICIENT_CAPTURE_QUALITY', retryable=True)

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
            'PIPER_VIEWPOINT_CLOSED_LOOP_ONE_VIEW': '1',
            'PIPER_VIEWPOINT_SPEED_PERCENT': str(
                self.get_parameter('free_motion_speed_percent').value),
            'PIPER_VIEWPOINT_MAX_VIEWS': str(
                self.get_parameter('maximum_captures').value),
            'PIPER_VIEWPOINT_MIN_VIEWS': str(
                self.get_parameter('required_captures').value),
            'PIPER_RETURN_HOME_POSITIONS_RAD': json.dumps(
                list(session.home_positions_rad), separators=(',', ':')),
            'PIPER_MISSION_TASK_ID': session.task_id,
            'PIPER_MISSION_SHA256': session.mission_sha256,
            'PIPER_TARGET_LABEL': session.goal['target_label'],
            'PIPER_TARGET_PROFILE': session.goal['target_profile'],
            'PIPER_TARGET_PROMPT': session.goal['target_prompt'],
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
            self.latest_arm_status = None
            self.latest_arm_status_at = 0.0
            self.motor_enable_guard_after = 0.0
        self.startup_progress(
            goal_handle, session,
            'starting disabled PiPER driver and waiting for its service')
        self.processes.start(
            'driver', [str(root / 'start_piper.sh')], environment)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            self.guard(goal_handle, session)
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
            2.0, 15.0, 'driver startup joint feedback',
            goal_handle, session)

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
        self.wait_for_vision_boot(
            vision_started_at, 120.0, goal_handle, session)

        self.startup_progress(
            goal_handle, session,
            'camera and GPU perception ready; starting hand-eye transform')
        hand_eye_started_ns = int(self.get_clock().now().nanoseconds)
        self.processes.start(
            'hand_eye', [str(root / 'L515_camera/run_hand_eye_tf.sh')],
            environment)
        self.wait_for_hand_eye_boot(
            hand_eye_started_ns, 20.0, goal_handle, session)

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
            worker_health_path, previous_worker_generation, 45.0,
            goal_handle, session)

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

    def wait_for_stable_joint_stream(
            self, stable_sec, timeout_sec, label,
            goal_handle=None, session=None):
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            if goal_handle is not None and session is not None:
                self.guard(goal_handle, session)
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

    def wait_for_vision_boot(
            self, started_at, timeout_sec, goal_handle=None, session=None):
        required_markers = (
            'Heavy-model worker ready:',
            'SAM2 live worker ready on',
            'GPU vision pipeline is running',
        )
        deadline = time.monotonic() + float(timeout_sec)
        last_reason = 'waiting for camera timestamp health and CUDA workers'
        while time.monotonic() < deadline:
            if goal_handle is not None and session is not None:
                self.guard(goal_handle, session)
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

    def wait_for_hand_eye_boot(
            self, started_ns, timeout_sec, goal_handle=None, session=None):
        deadline = time.monotonic() + float(timeout_sec)
        last_error = 'base_link -> camera_link transform is missing'
        while time.monotonic() < deadline:
            if goal_handle is not None and session is not None:
                self.guard(goal_handle, session)
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

    def wait_for_worker_boot(
            self, path, previous_generation, timeout_sec,
            goal_handle=None, session=None):
        deadline = time.monotonic() + float(timeout_sec)
        last_reason = 'Tesseract worker heartbeat is missing'
        while time.monotonic() < deadline:
            if goal_handle is not None and session is not None:
                self.guard(goal_handle, session)
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

    def selected_home_profile(self):
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
            try:
                validate_home_profile_limits(payload, URDF_JOINT_LIMITS)
            except (TypeError, ValueError) as exc:
                raise MissionFailure(
                    'configured home pose is invalid: %s' % exc,
                    failure_code='CONTROL_UNTRUSTWORTHY', retryable=False)
            if (
                    self.param_bool('require_staged_home_profile')
                    and not bool(payload.get('staged_home_configured', False))):
                raise MissionFailure(
                    'configured home pose is legacy-only; record the '
                    'mission-ready and storage J6 poses in the GUI before '
                    'enabling autonomous motion',
                    failure_code='CONTROL_UNTRUSTWORTHY', retryable=False)
            return payload
        raise MissionFailure(
            'configured home pose is missing; production missions require '
            'the hash-verified home-pose file')

    def selected_home_positions(self):
        return self.selected_home_profile()['positions_rad']

    def snapshot_target(self, rough_target):
        source = str(rough_target.header.frame_id)
        point = np.asarray([
            rough_target.pose.pose.position.x,
            rough_target.pose.pose.position.y,
            rough_target.pose.pose.position.z,
        ], dtype=float)
        if source == 'base_link':
            transformed = point
        else:
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
            transformed = (
                matrix @ np.asarray([point[0], point[1], point[2], 1.0]))[:3]
        if float(np.linalg.norm(transformed)) < 0.10:
            raise MissionFailure(
                'rough target is inside the 0.10m base-link exclusion radius; '
                'tracked robot must reposition',
                outcome='REPOSITION_REQUIRED',
                failure_code='NO_REACHABLE_PLAN', retryable=True)
        return np.asarray(transformed, dtype=float).tolist()

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
        request.look_index = max(0, int(session.acquisition_attempt) - 1)
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

    def request_multiview_plan(self, goal_handle, session):
        queue_deadline = time.monotonic() + PLAN_REQUEST_QUEUE_TIMEOUT_SEC
        visual_deadline = None
        while time.monotonic() < (
                visual_deadline if visual_deadline is not None
                else queue_deadline):
            request = RequestTesseractPlan.Request()
            request.force_refresh = False
            deadline = (
                visual_deadline if visual_deadline is not None
                else queue_deadline)
            result = self.call_service(
                self.plan_client, request,
                max(0.1, deadline - time.monotonic()),
                'Tesseract plan request service')
            if result.accepted:
                return str(result.request_id)
            if result.request_id and 'already pending' in str(result.message):
                time.sleep(0.25)
                continue
            if visual_reacquisition_plan_request_rejection(result.message):
                now = time.monotonic()
                if visual_deadline is None:
                    # No request entered the worker spool and no motion can be
                    # authorized.  Let the existing SAM2/heavy recovery restore
                    # one measured lock, then retry a completely fresh snapshot.
                    visual_deadline = (
                        now + SCAN_VISUAL_REACQUISITION_TIMEOUT_SEC)
                    self.startup_progress(
                        goal_handle, session,
                        'target confidence dipped before scan planning; '
                        'holding without motion while perception reacquires a '
                        'measured lock')
                if now < visual_deadline:
                    self.guard(goal_handle, session)
                    time.sleep(0.1)
                    continue
                raise MissionFailure(
                    'measured target lock did not recover before command-free '
                    'scan planning: %s' % result.message,
                    failure_code='TARGET_NOT_FOUND', retryable=True)
            raise MissionFailure(str(result.message))
        if visual_deadline is not None:
            raise MissionFailure(
                'measured target lock did not recover before command-free '
                'scan planning', failure_code='TARGET_NOT_FOUND', retryable=True)
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
        if kind == 'MULTIVIEW_SCAN':
            if int(plan.planned_viewpoints) != 1:
                raise MissionFailure(
                    'closed-loop multiview plan contains %d views; expected 1'
                    % int(plan.planned_viewpoints))
        return plan

    def approve_plan(self, goal_handle, session, plan):
        normal_deadline = (
            time.monotonic() + PLAN_APPROVAL_TRANSIENT_TIMEOUT_SEC)
        visual_reacquisition_deadline = None
        while True:
            request = ApproveScanExecution.Request()
            request.plan_id = str(plan.plan_id)
            request.trajectory_sha256 = str(plan.trajectory_sha256)
            request.confirmation = 'MISSION_POLICY:' + session.mission_sha256
            result = self.call_service(
                self.approve_client, request, 10.0,
                'mission plan approval service')
            if result.accepted:
                break
            now = time.monotonic()
            visual_reacquisition = bool(
                str(plan.plan_kind) == 'MULTIVIEW_SCAN'
                and visual_reacquisition_plan_approval_rejection(
                    result.message))
            if visual_reacquisition and visual_reacquisition_deadline is None:
                # The arm is already held at the last achieved view.  The
                # perception stack automatically runs its bounded SAM2/heavy
                # refresh here.  Continue revalidating the immutable proposal;
                # a changed measured center is still rejected by the executor
                # and handled by the outer fresh-plan loop.
                visual_reacquisition_deadline = (
                    now + SCAN_VISUAL_REACQUISITION_TIMEOUT_SEC)
                self.startup_progress(
                    goal_handle, session,
                    'target confidence dipped between scan views; holding '
                    'without capture while perception reacquires a measured '
                    'lock')
            deadline = (
                visual_reacquisition_deadline
                if visual_reacquisition_deadline is not None
                else normal_deadline)
            if (
                    not retryable_plan_approval_rejection(result.message)
                    or now >= deadline):
                if (
                        visual_reacquisition_deadline is not None
                        and now >= deadline):
                    raise MissionFailure(
                        'measured target lock did not recover during the '
                        'bounded between-view hold: %s' % result.message,
                        failure_code='TARGET_NOT_FOUND',
                        retryable=True)
                raise MissionFailure(str(result.message))
            self.guard(goal_handle, session)
            time.sleep(0.1)
        if str(plan.plan_kind) in ('ROUGH_ACQUISITION', 'MULTIVIEW_SCAN'):
            # Approval authorizes motion away from configured home. Clear the
            # startup proof before the first command can be emitted so every
            # later failure must prove a new return transaction.
            session.return_home_proved = False

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
        raise MissionFailure(
            'dedicated workflow occlusion assessment exceeded %.0f seconds'
            % WORKFLOW_ASSESSMENT_TIMEOUT_SEC)

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
            self.hold_client, Trigger.Request(), 8.0,
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
            if delta <= 0.005:
                settled_since = settled_since or time.monotonic()
                if time.monotonic() - settled_since >= 1.0:
                    session.current_hold_proved = True
                    self.last_hold_diagnostic = (
                        'position-settled delta=%.6f rad '
                        '(reported velocity=%.6f rad/s is diagnostic only)'
                        % (delta, velocity))
                    return True
            else:
                settled_since = None
            time.sleep(0.05)
        self.last_hold_diagnostic = (
            'last delta=%.6f rad velocity=%.6f rad/s '
            '(position limit 0.005000 rad; reported velocity is diagnostic); '
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
            if session.motor_control_lost_reason:
                # The driver watchdog owns the emergency all-axis disable.
                # Never issue hold or home motion after any axis has dropped;
                # only observe the resulting six-disabled proof and clean up
                # command/perception processes.
                deadline = time.monotonic() + 2.0
                all_disabled = False
                while time.monotonic() < deadline:
                    with self._lock:
                        status = self.latest_arm_status
                        age = time.monotonic() - self.latest_arm_status_at
                    if (
                            status is not None
                            and age <= 0.5
                            and bool(getattr(
                                status, 'motor_feedback_valid', False))
                            and not any(motor_driver_states(status))):
                        all_disabled = True
                        break
                    time.sleep(0.02)
                if not all_disabled:
                    return MissionFailure(
                        'motor control was lost and six-disabled feedback was '
                        'not proved; automatic home was forbidden and the '
                        'driver remains available for operator recovery', True)
                session.disabled_proved = True
                session.arm_enabled = False
                try:
                    self.authorize_mission(session, revoke=True)
                except MissionFailure:
                    pass
                self.transition(
                    None, session, MissionPhase.STOPPING,
                    'motor watchdog proved all six axes disabled; skipping '
                    'automatic home and stopping mission-owned processes')
                session.processes_stopped = self.processes.stop_all()
                if not session.processes_stopped:
                    return MissionFailure(
                        'motor control was lost; all axes are disabled but one '
                        'or more PiPER-owned processes remain alive', True)
                return MissionFailure(
                    'motor control was lost before configured home; no home '
                    'command was attempted, all six motors are disabled, and '
                    'mission-owned processes are stopped: '
                    + session.motor_control_lost_reason, True)
            if not session.arm_enabled:
                session.current_hold_proved = True
                session.return_home_proved = True
                session.storage_wrist_proved = True
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
                failure_reason = str(failure) if failure is not None else ''
                blocker = (
                    '' if planning_rejection_allows_current_state_home(
                        failure_reason)
                    else abort_return_home_blocker(failure_reason))
                if blocker:
                    return MissionFailure(
                        'configured home return was not attempted because the '
                        'failure is motion-safety-related (%s); arm remains '
                        'enabled in a current-position hold' % blocker, True)
                self.transition(
                    None, session, MissionPhase.RETURNING_HOME,
                    'cancellation/failure accepted; holding, then requesting '
                    'one fresh direct configured-home joint target with '
                    'only the configured folded self-collision exemption; '
                    'camera-holder floor/external clearance remains mandatory')
                startup_home = shutdown_uses_startup_home(session)
                home_proved = (
                    self.prove_return_home_for_shutdown(
                        session, startup=True)
                    if startup_home else
                    self.prove_return_home_for_shutdown(session))
                if not home_proved:
                    if session.motor_control_lost_reason:
                        return self.safe_shutdown(
                            session, normal_completion=False, failure=failure)
                    diagnostic = str(getattr(
                        self, 'last_return_home_diagnostic', '')).strip()
                    return MissionFailure(
                        'configured home return was not proved; arm remains '
                        'enabled in a current-position hold'
                        + (': ' + diagnostic if diagnostic else ''), True)
            if not session.storage_wrist_proved:
                self.transition(
                    None, session, MissionPhase.RETURNING_HOME,
                    'rough home proved; rotating J6 to the configured storage '
                    'angle before disable')
                storage_target = list(session.storage_positions_rad)
                if len(storage_target) != 6:
                    return MissionFailure(
                        'storage wrist target is missing; arm remains enabled '
                        'at rough home', True)
                startup_home = shutdown_uses_startup_home(session)
                if not self.prove_return_home_for_shutdown(
                        session, startup=startup_home,
                        target_positions=storage_target,
                        home_stage='STORAGE_WRIST'):
                    if session.motor_control_lost_reason:
                        return self.safe_shutdown(
                            session, normal_completion=False, failure=failure)
                    diagnostic = str(getattr(
                        self, 'last_return_home_diagnostic', '')).strip()
                    return MissionFailure(
                        'storage J6 rotation was not proved; arm remains '
                        'enabled in a current-position hold'
                        + (': ' + diagnostic if diagnostic else ''), True)
                session.storage_wrist_proved = True
            self.transition(None, session, MissionPhase.HOLDING,
                            'proving final current-position hold')
            if not self.prove_current_hold_for_shutdown(session):
                if session.motor_control_lost_reason:
                    return self.safe_shutdown(
                        session, normal_completion=False, failure=failure)
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
            if session.motor_control_lost_reason:
                return self.safe_shutdown(
                    session, normal_completion=False, failure=failure)
            return MissionFailure(str(exc), True)

    def at_configured_home(
            self, session, tolerance_rad=0.030, target_positions=None):
        self.require_fresh_joint_feedback()
        current = np.asarray(self.latest_joints.position[:6], dtype=float)
        target = np.asarray(
            session.home_positions_rad
            if target_positions is None else target_positions,
            dtype=float)
        return bool(
            current.shape == (6,)
            and target.shape == (6,)
            and np.all(np.isfinite(current))
            and np.all(np.isfinite(target))
            and float(np.max(np.abs(current - target))) <= float(tolerance_rad)
        )

    def prove_return_home_for_shutdown(
            self, session, startup=False, goal_handle=None,
            target_positions=None, home_stage='ROUGH_HOME'):
        """Stop at measured state, then request one direct configured home target."""
        self.last_return_home_diagnostic = ''
        target_positions = list(
            session.home_positions_rad
            if target_positions is None else target_positions)
        if len(target_positions) != 6 or not all(
                math.isfinite(float(value)) for value in target_positions):
            self.last_return_home_diagnostic = (
                'home stage %s has no finite six-joint target' % home_stage)
            return False

        def fail_home(reason):
            self.last_return_home_diagnostic = str(reason)
            return False

        def guard_active_mission():
            # Shutdown recovery deliberately has no action handle and must run
            # to completion after cancellation. Startup normalization remains
            # interruptible while it waits for planning/execution evidence.
            if goal_handle is not None:
                self.guard(goal_handle, session)
            else:
                TargetScanMissionNode.guard_motor_control(self, session)

        def critical_process_failure():
            failed = self.processes.failed()
            names = sorted(
                set(failed).intersection(
                    ('driver', 'scan_stack', 'tesseract_worker')))
            return names

        try:
            if self.at_configured_home(
                    session, target_positions=target_positions):
                session.return_home_proved = True
                self.last_return_home_diagnostic = (
                    'fresh feedback is already within %s tolerance'
                    % str(home_stage).lower())
                return True
        except MissionFailure as exc:
            return fail_home(exc)
        result = self.call_service(
            self.hold_client if startup else self.cancel_client,
            Trigger.Request(), 8.0,
            ('executor startup hold-before-home service' if startup else
             'executor stop-and-hold-before-home service'))
        message = str(result.message).lower()
        if not result.success:
            return fail_home(result.message or 'executor could not hold before home')
        if 'hold' not in message:
            return fail_home(
                'executor response did not prove a hold before home: '
                + str(result.message))
        try:
            self.wait_for_stable_joint_stream(
                0.5, 15.0, 'held feedback before dedicated home planning')
        except MissionFailure as exc:
            return fail_home(exc)

        self.clear_plan_cache()
        queued_at = time.monotonic()
        request = RequestTesseractPlan.Request()
        request.force_refresh = False
        request.home_stage = str(home_stage).upper()
        request.joint_goal_positions_rad = [
            float(value) for value in target_positions]
        planned = self.call_service(
            (self.startup_home_plan_client
             if startup else self.return_home_plan_client), request,
            PLAN_REQUEST_QUEUE_TIMEOUT_SEC,
            ('startup Tesseract configured-home plan service'
             if startup else
             'dedicated Tesseract return-home plan service'))
        if not planned.accepted or not planned.request_id:
            return fail_home(
                'home planning request was rejected: ' + str(planned.message))
        request_id = str(planned.request_id)
        deadline = time.monotonic() + PLAN_RESULT_TIMEOUT_SEC
        plan = None
        while time.monotonic() < deadline:
            guard_active_mission()
            candidate = self.latest_plan
            if (
                    candidate is not None
                    and self.latest_plan_at >= queued_at
                    and str(candidate.plan_kind) == 'RETURN_HOME'
                    and str(candidate.plan_id) == request_id):
                if not bool(candidate.valid):
                    return fail_home(
                        'home plan was invalid: ' + str(candidate.reason))
                plan = candidate
                break
            failed = critical_process_failure()
            if failed:
                return fail_home(
                    'critical process exited while planning home: '
                    + ', '.join(failed))
            time.sleep(0.05)
        if plan is None:
            return fail_home('timed out waiting for the correlated home plan')
        execution_started = time.monotonic()
        self.approve_plan(goal_handle, session, plan)
        deadline = time.monotonic() + PLAN_RESULT_TIMEOUT_SEC
        while time.monotonic() < deadline:
            guard_active_mission()
            status = self.latest_execution
            if status is not None and self.latest_execution_at >= execution_started:
                state = str(status.state)
                reason = str(status.reason)
                if state == 'ABORTED' and 'configured home reached' in reason.lower():
                    if self.at_configured_home(
                            session, target_positions=target_positions):
                        session.return_home_proved = True
                        self.last_return_home_diagnostic = (
                            '%s reached and feedback-proved'
                            % str(home_stage).lower())
                        return True
                    return fail_home(
                        'executor reported configured home but feedback was outside tolerance')
                if state == 'ABORTED':
                    return fail_home(
                        'home execution aborted: ' + (reason or 'no reason'))
                if state in ('INVALID', 'COMPLETE'):
                    return fail_home(
                        'home execution entered %s: %s' % (state, reason))
            failed = critical_process_failure()
            if failed:
                return fail_home(
                    'critical process exited during home motion: '
                    + ', '.join(failed))
            time.sleep(0.05)
        return fail_home('timed out waiting for configured-home execution proof')

    def prove_current_hold_for_shutdown(self, session):
        try:
            self.guard_motor_control(session)
            self.require_fresh_joint_feedback()
            initial = energized_hold_target(self.latest_joints.position[:6])
            result = self.call_service(
                self.hold_client, Trigger.Request(), 8.0,
                'executor shutdown hold service')
            if not result.success or 'hold requested' not in str(result.message):
                return False
            deadline, settled_since = time.monotonic() + 15.0, None
            while time.monotonic() < deadline:
                self.guard_motor_control(session)
                if self.latest_joints is None or time.monotonic() - self.latest_joints_at > 1.0:
                    settled_since = None
                else:
                    current = np.asarray(self.latest_joints.position[:6], dtype=float)
                    delta = float(np.max(np.abs(current - initial)))
                    velocity_values = np.asarray(
                        self.latest_joints.velocity[:6], dtype=float)
                    velocity = float(np.max(np.abs(velocity_values))) \
                        if velocity_values.size == 6 else math.inf
                    if delta <= 0.005:
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

    def guard_motor_control(self, session):
        """Fail immediately when a powered mission loses any motor axis."""
        if (
                bool(getattr(session, 'arm_enabled', False))
                and time.monotonic() >= self.motor_enable_guard_after):
            with self._lock:
                arm_status = self.latest_arm_status
                arm_status_age = time.monotonic() - self.latest_arm_status_at
            motor_reasons = (
                ['arm/motor status is missing or stale']
                if arm_status is None or arm_status_age > 0.5 else
                motor_control_reasons(
                    arm_status, require_all_enabled=True))
            if motor_reasons:
                session.motor_control_lost_reason = '; '.join(motor_reasons)
                raise MissionFailure(
                    'motor control became untrustworthy during the mission: '
                    + session.motor_control_lost_reason
                    + '; automatic home is forbidden',
                    needs_operator=True,
                    failure_code='CONTROL_UNTRUSTWORTHY', retryable=False)

    def guard(self, goal_handle, session):
        session.accepted_captures = int(
            self.latest_capture.get('captured_frame_count',
                                    session.accepted_captures))
        if self._process_shutdown_requested:
            raise MissionFailure(
                'coordinator shutdown requested', outcome='CANCELLED',
                failure_code='CANCELLED', retryable=True)
        if goal_handle is not None and goal_handle.is_cancel_requested:
            raise MissionFailure(
                'tracked robot cancelled the task', outcome='CANCELLED')
        if session.deadline_expired():
            raise MissionFailure('mission deadline expired')
        self.guard_motor_control(session)
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
            self.latest_scan_history = None
            self.latest_scan_history_at = 0.0
            self.latest_scan_target_center = None
            self.last_scan_feature_coverage = {}

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
            'required_captures': int(
                self.get_parameter('required_captures').value),
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
        feedback.required_captures = int(
            self.get_parameter('required_captures').value)
        feedback.process_health_json = json.dumps(
            self.processes.health(), sort_keys=True)
        feedback.shutdown_phase = (
            session.phase.value if session.phase in (
                MissionPhase.RETURNING_HOME, MissionPhase.HOLDING,
                MissionPhase.DISABLING, MissionPhase.STOPPING) else '')
        goal_handle.publish_feedback(feedback)

    def write_queued_status(self, record):
        normalized = record['normalized']
        now = time.monotonic()
        if now - float(record.get('last_status_at', 0.0)) < 0.5:
            return
        record['last_status_at'] = now
        waited = max(
            0.0, now - float(record['admitted_monotonic']))
        reason = (
            'queued closest-first; waiting for the active target to '
            'finish its validated shutdown boundary')
        self.spool.write('status', normalized['task_id'], {
            'task_id': normalized['task_id'],
            'mission_sha256': normalized['mission_sha256'],
            'phase': MissionPhase.QUEUED.value,
            'reason': reason,
            'elapsed_sec': waited,
            'remaining_sec': float(normalized['deadline_sec']),
            'accepted_captures': 0,
            'required_captures': int(
                self.get_parameter('required_captures').value),
            'processes': self.processes.health(),
        })
        if record['source'] == 'action':
            feedback = RunTargetScan.Feedback()
            feedback.phase = MissionPhase.QUEUED.value
            feedback.reason = reason
            feedback.elapsed_sec = float(waited)
            feedback.remaining_sec = float(normalized['deadline_sec'])
            feedback.accepted_captures = 0
            feedback.required_captures = int(
                self.get_parameter('required_captures').value)
            feedback.occlusion_action_limit = MAX_OCCLUSION_ACTIONS
            feedback.process_health_json = json.dumps(
                self.processes.health(), sort_keys=True)
            record['goal_handle'].publish_feedback(feedback)

    def poll_mission_queue(self):
        """Discover durable goals, cancel queued work, and dispatch one scan."""
        self.discover_spool_goals()
        action_cancellations = []
        spool_cancellations = []
        with self._lock:
            for task_id, record in list(self._pending_missions.items()):
                cancelled = self._process_shutdown_requested
                if record['source'] == 'action':
                    cancelled = cancelled or bool(
                        record['goal_handle'].is_cancel_requested)
                else:
                    try:
                        heartbeat = self.spool.read('heartbeat', task_id)
                    except (FileNotFoundError, OSError, TypeError, ValueError):
                        heartbeat = {}
                    cancelled = cancelled or bool(
                        heartbeat.get('cancel_requested', False))
                if not cancelled:
                    self.write_queued_status(record)
                    continue
                self._pending_missions.pop(task_id, None)
                if record['source'] == 'action':
                    self._prevalidated_goals[task_id] = record['normalized']
                    action_cancellations.append(record['goal_handle'])
                else:
                    spool_cancellations.append(record)

            active = self.registry.active is not None
            if (
                    self._process_shutdown_requested
                    or active
                    or self._dispatch_task_id
                    or not self._pending_missions):
                selected = None
            else:
                records = list(self._pending_missions.values())
                ready = mission_queue_ready(
                    records, time.monotonic(),
                    float(self.get_parameter(
                        'mission_queue_coalesce_sec').value))
                selected = closest_pending_mission(records) if ready else None
                if selected is not None:
                    task_id = selected['normalized']['task_id']
                    self._pending_missions.pop(task_id, None)
                    self._prevalidated_goals[task_id] = selected['normalized']
                    self._dispatch_task_id = task_id

        for goal_handle in action_cancellations:
            goal_handle.execute()
        for record in spool_cancellations:
            self.finish_queued_cancel(
                record['normalized'],
                'queued mission cancelled before arm resources started')
        if selected is None:
            return
        if selected['source'] == 'action':
            selected['goal_handle'].execute()
            return
        threading.Thread(
            target=self.run_spool_goal, args=(selected,), daemon=True).start()

    def discover_spool_goals(self):
        goals_dir = self.spool.root / 'goals'
        for path in sorted(goals_dir.glob('*.json')):
            task_id = path.stem
            with self._lock:
                if (task_id in self._spool_seen
                        or task_id in self.registry.results
                        or task_id in self._pending_missions):
                    continue
                maximum = int(
                    self.get_parameter('max_pending_missions').value)
                if (
                        len(self._pending_missions)
                        + len(self._action_reservations) >= maximum):
                    return
            try:
                payload = self.spool.read('goals', task_id)
                stamp = float(payload['rough_target']['stamp_sec'])
                normalized = validate_goal_payload(payload, now_sec=stamp)
                if str(payload.get('mission_sha256', '')) != str(
                        normalized['mission_sha256']):
                    raise ValueError('local mission identity hash mismatch')
                admitted_wall = float(
                    payload.get('queue_admitted_wall_time_sec', time.time()))
                queue_age = max(0.0, time.time() - admitted_wall)
            except (
                    KeyError, OSError, TypeError, ValueError,
                    json.JSONDecodeError) as exc:
                self.get_logger().error(
                    'ignoring invalid spooled mission %s: %s' % (task_id, exc))
                with self._lock:
                    self._spool_seen.add(task_id)
                continue
            with self._lock:
                self._queue_sequence += 1
                record = {
                    'normalized': normalized,
                    'payload': payload,
                    'sequence': self._queue_sequence,
                    'admitted_monotonic': time.monotonic() - queue_age,
                    'source': 'spool',
                }
                self._pending_missions[task_id] = record
                self._spool_seen.add(task_id)
                self.write_queued_status(record)

    def run_spool_goal(self, record):
        task_id = str(record['normalized']['task_id'])
        try:
            payload = record['payload']
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
            self.finish_queue_dispatch(task_id)
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
            'REPOSITION_REQUIRED': (
                RunTargetScan.Goal.OUTCOME_REPOSITION_REQUIRED),
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
        result.failure_code = str(payload.get('failure_code', ''))
        result.retryable = bool(payload.get('retryable', False))
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
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def request_shutdown(_signum, _frame):
        node.request_process_shutdown()

    signal.signal(signal.SIGINT, request_shutdown)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        # The explicit handler normally prevents KeyboardInterrupt.  Preserve
        # the same cancellation semantics if the runtime raises it anyway.
        node.request_process_shutdown()
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        try:
            node.action_server.destroy()
        finally:
            node.destroy_node()
        if rclpy.ok():
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
