#!/usr/bin/env python3
"""Headless, bounded ROS action orchestrator for one PiPER target scan."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import threading
import time

import numpy as np
import yaml
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
from piper_mobile_manipulation.configuration import (
    configured_value,
    load_mission_configuration,
    MissionCaptureConfig,
    MissionMotionConfig,
    MissionWorkflowConfig,
)
from piper_mobile_manipulation.home_pose import (
    load_home_pose,
    staged_home_targets as _staged_home_targets,
    validate_home_profile_limits,
    validate_staged_wrist_direction as _validate_staged_wrist_direction,
)
from piper_mobile_manipulation.failure_model import (
    as_failure,
    FailureTag,
)
from piper_mobile_manipulation.mission_core import (
    MAX_OCCLUSION_ACTIONS,
    MissionPhase,
    MissionRegistry,
    closest_pending_mission,
    mission_queue_ready,
    queued_cancel_result,
    validate_goal_payload,
)
from piper_mobile_manipulation.mission_engine import (
    ACQUISITION_SERVICE_TIMEOUT_SEC,
    CancellationToken,
    failure_code_for_reason,
    feature_capture_decision as _feature_capture_decision,
    MAX_SCAN_QUALITY_REPLANS,
    MAX_SCAN_TARGET_DRIFT_REPLANS,
    MissionContext,
    MissionEngine,
    MissionFailure,
    PLAN_APPROVAL_TRANSIENT_TIMEOUT_SEC,
    planning_rejection_allows_current_state_home as _planning_rejection_allows_current_state_home,
    PLAN_REQUEST_QUEUE_TIMEOUT_SEC,
    PLAN_RESULT_TIMEOUT_SEC,
    retryable_plan_approval_rejection,
    runtime_freshness_plan_request_rejection,
    safe_view_exhaustion_after_capture as _safe_view_exhaustion_after_capture,
    SCAN_VISUAL_REACQUISITION_TIMEOUT_SEC,
    shutdown_uses_startup_home as _shutdown_uses_startup_home,
    target_drift_requires_replan as _target_drift_requires_replan,
    visual_reacquisition_plan_approval_rejection,
    visual_reacquisition_plan_request_rejection,
    WORKFLOW_ASSESSMENT_TIMEOUT_SEC,
)
from piper_mobile_manipulation.mission_spool import MissionSpool
from piper_mobile_manipulation.process_supervisor import ProcessSupervisor
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
    startup_measured_hold_reference,
    motor_control_reasons,
    motor_driver_states,
    URDF_JOINT_LIMITS,
)
from piper_mobile_manipulation.scan_session_memory import (
    achieved_feature_coverage,
    history_coverage_target_center,
    validate_history_payload,
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
from piper_mobile_manipulation.telemetry_store import TelemetryStore
from piper_mobile_manipulation.view_generation import (
    generation_matches_expected,
)
from piper_mobile_manipulation.srv import (
    ApproveScanExecution,
    AuthorizeMission,
    ExecuteHomeStage,
    GetTargetScanResult,
    PrepareAcquisition,
    RequestTesseractPlan,
)

# Preserve established pure-helper imports at the ROS adapter boundary while
# their implementations live with their single application/domain owners.
staged_home_targets = _staged_home_targets
validate_staged_wrist_direction = _validate_staged_wrist_direction
feature_capture_decision = _feature_capture_decision
planning_rejection_allows_current_state_home = (
    _planning_rejection_allows_current_state_home)
safe_view_exhaustion_after_capture = _safe_view_exhaustion_after_capture
shutdown_uses_startup_home = _shutdown_uses_startup_home
target_drift_requires_replan = _target_drift_requires_replan


NON_COMMAND_PROCESS_GROUPS = (
    'vision',
    'hand_eye',
    'tesseract_worker',
)


def calibration_identity_for_mission(root, environment):
    """Bind capture provenance to the exact hand-eye file used by this run."""
    default_path = (
        Path(root) / 'L515_camera' / 'calibration' / 'hand_eye'
        / 'session_20260808_straight_mount' / 'calibration_result.yaml')
    path = Path(str(environment.get(
        'PIPER_HAND_EYE_CALIBRATION', default_path))).expanduser().resolve()
    if not path.is_file():
        raise MissionFailure(
            'hand-eye calibration file is missing: %s' % path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return path, digest.hexdigest()


def previous_generation_cleanup_targets(
        live_processes, all_motors_disabled):
    """Select exact stale handles that are safe to stop before admission."""
    live = tuple(dict.fromkeys(str(name) for name in live_processes))
    if 'driver' in live and not bool(all_motors_disabled):
        return ()
    return live


def discard_failed_zero_capture_dataset(
        scan_dir, dataset_root, task_id, mission_sha256):
    """Delete only an identity-matched failed dataset with no captures."""
    try:
        raw_candidate = Path(str(scan_dir))
        if raw_candidate.is_symlink():
            return False, 'scan dataset path is a symbolic link'
        root = Path(dataset_root).resolve(strict=True)
        candidate = raw_candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return False, 'scan directory or dataset root does not exist'
    if candidate.parent != root or not re.fullmatch(
            r'scan_[0-9]{8}_[0-9]{6}(?:_[A-Za-z0-9_-]+)?',
            candidate.name):
        return False, 'scan directory is outside the guarded dataset root'
    if not candidate.is_dir():
        return False, 'scan dataset is not a regular directory'
    try:
        metadata = yaml.safe_load(
            (candidate / 'metadata.yaml').read_text(encoding='utf-8'))
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        return False, 'scan metadata is unreadable: %s' % exc
    if not isinstance(metadata, dict):
        return False, 'scan metadata is not a mapping'
    if (str(metadata.get('task_id', '')) != str(task_id)
            or str(metadata.get('mission_sha256', ''))
            != str(mission_sha256)):
        return False, 'scan metadata does not match the failed mission'
    allowed_root_files = {
        'metadata.yaml', 'manifest.json', 'coverage_envelope.yaml'}
    incomplete_frame_pattern = re.compile(
        r'view_[0-9]{3}_(?:rgb|depth|mask|native_depth|confidence|'
        r'target_depth|target_support_mask)(?:\.partial)?\.(?:png|npy)$')
    for path in candidate.rglob('*'):
        if path.is_symlink():
            return False, 'scan dataset contains a symbolic link'
        if not path.is_file():
            continue
        relative = path.relative_to(candidate)
        if relative.as_posix() in allowed_root_files:
            continue
        if relative.parent.as_posix() == 'frames':
            if re.fullmatch(r'view_[0-9]{3}_metadata\.yaml', path.name):
                return False, 'scan contains one or more completed captures'
            if incomplete_frame_pattern.fullmatch(path.name):
                continue
        return False, 'scan contains unknown or derived artifacts'
    manifest_path = candidate / 'manifest.json'
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            capture_count = int(manifest.get('capture_count', -1))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return False, 'scan manifest is unreadable: %s' % exc
        if capture_count != 0:
            return False, 'scan manifest records one or more captures'
    shutil.rmtree(candidate)
    return True, 'failed zero-capture scan dataset was permanently removed'


def find_failed_mission_dataset(dataset_root, task_id, mission_sha256):
    """Find one exact identity-matched mission directory."""
    try:
        root = Path(dataset_root).resolve(strict=True)
    except (OSError, RuntimeError):
        return '', 'dataset root does not exist'
    matches = []
    for candidate in sorted(root.glob('scan_*')):
        if candidate.is_symlink() or not candidate.is_dir():
            continue
        if not re.fullmatch(
                r'scan_[0-9]{8}_[0-9]{6}(?:_[A-Za-z0-9_-]+)?',
                candidate.name):
            continue
        try:
            metadata = yaml.safe_load(
                (candidate / 'metadata.yaml').read_text(encoding='utf-8'))
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            continue
        if (isinstance(metadata, dict)
                and str(metadata.get('task_id', '')) == str(task_id)
                and str(metadata.get('mission_sha256', ''))
                == str(mission_sha256)):
            matches.append(candidate)
    if len(matches) == 1:
        return str(matches[0]), 'found exact identity-matched mission dataset'
    if not matches:
        return '', 'no identity-matched mission dataset exists'
    return '', 'multiple identity-matched mission datasets exist'


# Import compatibility for Phase 0/1 tests and downstream tooling.  Production
# ownership is implemented only by ProcessSupervisor.
ManagedProcessSet = ProcessSupervisor

# Phase 1 froze these module-level imports as a downstream compatibility seam.
__all__ = (
    'ACQUISITION_SERVICE_TIMEOUT_SEC',
    'MAX_SCAN_QUALITY_REPLANS',
    'MAX_SCAN_TARGET_DRIFT_REPLANS',
    'PLAN_APPROVAL_TRANSIENT_TIMEOUT_SEC',
    'PLAN_REQUEST_QUEUE_TIMEOUT_SEC',
    'PLAN_RESULT_TIMEOUT_SEC',
    'SCAN_VISUAL_REACQUISITION_TIMEOUT_SEC',
    'WORKFLOW_ASSESSMENT_TIMEOUT_SEC',
)


class _MissionNodeOperations:
    """Translate pure MissionEngine operations to the existing ROS node."""

    def __init__(self, node, goal_handle, cancellation, rough_target=None):
        self.node = node
        self.goal_handle = goal_handle
        self.cancellation = cancellation
        self.rough_target = rough_target

    @property
    def session(self):
        """Return the currently bound session for compatibility diagnostics."""
        return getattr(self.node, '_active_engine_session', None)

    def begin_process_generation(self, _context):
        live = self.node.processes.begin_generation()
        if not live:
            return []
        cleanup_targets = previous_generation_cleanup_targets(
            live, self.node.fresh_all_motors_disabled())
        if not cleanup_targets:
            self.node.get_logger().error(
                'retaining previous process generation because its driver is '
                'live and fresh six-disabled feedback is not proved: %s'
                % ', '.join(live))
            return live
        self.node.get_logger().warn(
            'cleaning exact previous-generation process handles before new '
            'mission admission: %s' % ', '.join(cleanup_targets))
        try:
            report = self.node.processes.shutdown(cleanup_targets)
        except Exception as error:
            self.node.get_logger().error(
                'previous-generation cleanup raised %s: %s'
                % (type(error).__name__, error))
            return live
        if not report.complete:
            self.node.get_logger().error(
                'previous-generation cleanup remains incomplete: %s'
                % ', '.join(report.still_running))
            return list(report.still_running)
        return self.node.processes.begin_generation()

    def snapshot_target(self, _context):
        return self.node.snapshot_target(self.rough_target)

    def transition(self, context, phase, reason):
        self.node.transition(self.goal_handle, context.session, phase, reason)

    def progress(self, context, reason):
        self.node.startup_progress(
            self.goal_handle, context.session, reason)

    def selected_home_profile(self, _context):
        return self.node.selected_home_profile()

    def bind_home_profile(self, _context, profile):
        self.node.current_home_profile = profile

    def current_home_profile(self, _context):
        return self.node.current_home_profile

    def start_processes(self, context):
        self.node.start_processes(self.goal_handle, context.session)

    def wait_for_enable_service(self, context, timeout):
        self.node.wait_for(
            self.goal_handle, context.session,
            lambda: self.node.enable_client.service_is_ready(), timeout,
            'PiPER enable service did not become ready')

    def wait_for_stable_readiness(
            self, context, mode, stable_sec, timeout_sec):
        self.node.wait_for_stable_readiness(
            self.goal_handle, context.session, mode, stable_sec, timeout_sec)

    def wait_for_stable_joint_stream(
            self, context, stable_sec, timeout_sec, label):
        self.node.wait_for_stable_joint_stream(
            stable_sec, timeout_sec, label,
            self.goal_handle, context.session)

    def require_fresh_joint_feedback(self, _context):
        self.node.require_fresh_joint_feedback()

    def current_joint_positions(self, _context):
        return self.node.latest_joints.position[:6]

    def boolean_option(self, _context, name):
        return self.node.param_bool(name)

    def numeric_option(self, _context, name):
        return configured_value(self.node, name)

    def authorize_mission(self, context, revoke=False):
        return self.node.authorize_mission(context.session, revoke=revoke)

    def enable_arm(self, _context, enabled):
        return self.node.call_enable(enabled)

    def arm_enable_guard_started(self, _context):
        self.node.motor_enable_guard_after = time.monotonic() + 0.5

    def prove_current_hold(self, context):
        return self.node.prove_current_hold(
            self.goal_handle, context.session)

    def hold_diagnostic(self, _context):
        return str(self.node.last_hold_diagnostic)

    def prove_home(
            self, context, startup=False, target_positions=None,
            home_stage='ROUGH_HOME', interruptible=False):
        kwargs = {}
        if startup:
            kwargs['startup'] = True
        if interruptible:
            kwargs['goal_handle'] = self.goal_handle
        if target_positions is not None:
            kwargs['target_positions'] = target_positions
        if str(home_stage) != 'ROUGH_HOME':
            kwargs['home_stage'] = home_stage
        return self.node.prove_return_home_for_shutdown(
            context.session, **kwargs)

    def return_home_diagnostic(self, _context):
        return str(getattr(self.node, 'last_return_home_diagnostic', ''))

    def clear_plan_cache(self, _context):
        return self.node.clear_plan_cache()

    def prepare_acquisition(self, context):
        return self.node.prepare_acquisition(
            context.session, context.target)

    def wait_for_plan(self, context, kind, request_id, timeout):
        return self.node.wait_for_plan(
            self.goal_handle, context.session, kind, request_id, timeout)

    def approve_plan(self, context, plan):
        return self.node.approve_plan(
            self.goal_handle, context.session, plan)

    def wait_for_execution(
            self, context, successes, timeout, failures):
        return self.node.wait_for_execution(
            self.goal_handle, context.session, successes, timeout, failures)

    def start_and_wait_workflow(self, context):
        return self.node.start_and_wait_workflow(
            self.goal_handle, context.session)

    def readiness_rejection(self, _context, mode):
        return self.node.readiness_rejection(mode)

    def capture_count(self, _context):
        return int(self.node.latest_capture.get('captured_frame_count', 0))

    def current_feature_coverage(self, _context):
        return self.node.current_scan_feature_coverage()

    def request_multiview_plan(self, context):
        return self.node.request_multiview_plan(
            self.goal_handle, context.session)

    def wait_for_view_generation(self, context, accepted_views, timeout):
        return self.node.wait_for_view_generation(
            self.goal_handle, context.session, accepted_views, timeout)

    def remaining_time(self, context):
        return context.session.remaining()

    def wait_for_scan_history(self, context, timeout):
        self.node.wait_for(
            self.goal_handle, context.session,
            lambda: int((self.node.latest_scan_history or {}).get(
                'accepted_views', 0)) >= context.session.accepted_captures,
            timeout,
            'scan history did not catch up with the final accepted capture')

    def wait_for_all_motors_disabled(self, _context, timeout):
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            telemetry_store = getattr(self.node, 'telemetry_store', None)
            if telemetry_store is None:
                status = self.node.latest_arm_status
                age = time.monotonic() - self.node.latest_arm_status_at
            else:
                snapshot = telemetry_store.snapshot()
                observation = snapshot.arm.status
                status = None if observation is None else observation.value
                age = (
                    math.inf if observation is None else
                    observation.age_at(snapshot.captured_at))
            if (
                    status is not None
                    and age <= 0.5
                    and bool(getattr(status, 'motor_feedback_valid', False))
                    and not any(motor_driver_states(status))):
                return True
            time.sleep(0.02)
        return False

    def stop_processes(self, _context):
        return self.node.processes.stop_all()

    def stop_processing_processes(self, _context):
        try:
            report = self.node.processes.shutdown(NON_COMMAND_PROCESS_GROUPS)
        except Exception as error:
            self.node.get_logger().error(
                'non-command process cleanup raised %s: %s'
                % (type(error).__name__, error))
            return False
        return report.complete

    def prove_shutdown_hold(self, context):
        return self.node.prove_current_hold_for_shutdown(context.session)


def mission_engine_for(owner, operations):
    """Inject typed production config while retaining old pure test seams."""
    configuration = getattr(owner, 'configuration', None)
    if configuration is None:
        if not hasattr(owner, 'param_bool'):
            return MissionEngine(
                operations,
                motion_config=MissionMotionConfig(
                    enable_real_arm_motion=False,
                    motion_speed_profile_qualified=False,
                    free_motion_speed_percent=30.0,
                    contact_speed_percent=10.0,
                    home_pose_path='',
                    require_staged_home_profile=True,
                ),
                capture_config=MissionCaptureConfig(
                    required_captures=8, maximum_captures=24),
                workflow_config=MissionWorkflowConfig(),
            )
        return MissionEngine(operations)
    return MissionEngine(
        operations,
        motion_config=configuration.motion,
        capture_config=configuration.capture,
        workflow_config=configuration.workflow,
    )


def workflow_config_for(owner):
    """Return frozen production workflow config or characterization defaults."""
    configuration = getattr(owner, 'configuration', None)
    if configuration is None:
        return MissionWorkflowConfig()
    return configuration.workflow


class TargetScanMissionNode(Node):
    def __init__(self):
        super().__init__('target_scan_mission')
        self.configuration = load_mission_configuration(self)
        self.callback_group = ReentrantCallbackGroup()
        self._lock = threading.RLock()
        self.telemetry_store = TelemetryStore(clock=time.monotonic)
        self._spool_seen = set()
        self._pending_missions = {}
        self._action_reservations = {}
        self._prevalidated_goals = {}
        self._cancellation_tokens = {}
        self._active_cancellation_token = None
        self._active_engine_session = None
        self._queue_sequence = 0
        self._dispatch_task_id = ''
        self._process_shutdown_requested = False
        self._process_shutdown_quiescent_since = 0.0
        self.registry = MissionRegistry()
        self.spool = MissionSpool(
            self.configuration.process.mission_spool_root)
        self.load_durable_results()
        self.processes = ProcessSupervisor(
            self.configuration.process.process_log_root)
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
        self.latest_view_generation = None
        self.latest_view_generation_at = 0.0
        self.view_generation_barrier_at = 0.0
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
            String, '/piper/tesseract_view_generation',
            self.view_generation_cb, latched)
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
        self.execute_home_stage_client = self.create_client(
            ExecuteHomeStage,
            '/scan_viewpoint_executor/execute_home_stage',
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
            with self._lock:
                tokens = tuple(self._cancellation_tokens.values())
            for token in tokens:
                token.cancel('coordinator shutdown requested')
            self.get_logger().warn(
                'coordinator shutdown requested; cancelling any active '
                'mission through home, hold, disable, and child cleanup')

    def fresh_all_motors_disabled(self, maximum_age_sec=0.5):
        """Return exact fresh feedback proof that every arm motor is off."""
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            status = self.latest_arm_status
            age = time.monotonic() - self.latest_arm_status_at
        else:
            snapshot = telemetry_store.snapshot()
            observation = snapshot.arm.status
            status = None if observation is None else observation.value
            age = (
                math.inf if observation is None else
                observation.age_at(snapshot.captured_at))
        return bool(
            status is not None
            and age <= float(maximum_age_sec)
            and bool(getattr(status, 'motor_feedback_valid', False))
            and not any(motor_driver_states(status)))

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
        if self.processes.has_live_processes():
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
        value = configured_value(self, name)
        return value.lower() in ('1', 'true', 'yes', 'on') \
            if isinstance(value, str) else bool(value)

    def readiness_cb(self, msg):
        received_at = time.monotonic()
        with self._lock:
            self.latest_readiness, self.latest_readiness_at = msg, received_at
            self.telemetry_store.update_readiness(
                msg, received_at=received_at,
                frame_id=str(getattr(msg.header, 'frame_id', '')))

    def plan_cb(self, msg):
        received_at = time.monotonic()
        with self._lock:
            self.latest_plan, self.latest_plan_at = msg, received_at
            self.telemetry_store.update_plan(
                msg, received_at=received_at,
                frame_id=str(getattr(msg.header, 'frame_id', '')))
            if (
                    str(msg.plan_kind) == 'MULTIVIEW_SCAN'
                    and bool(msg.valid)):
                self.latest_scan_target_center = {
                    'x': float(msg.target_center.x),
                    'y': float(msg.target_center.y),
                    'z': float(msg.target_center.z),
                }

    def execution_cb(self, msg):
        received_at = time.monotonic()
        with self._lock:
            self.latest_execution, self.latest_execution_at = msg, received_at
            self.telemetry_store.update_execution(
                msg, received_at=received_at,
                frame_id=str(getattr(msg.header, 'frame_id', '')))

    def capture_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return
        if isinstance(payload, dict):
            received_at = time.monotonic()
            with self._lock:
                self.latest_capture, self.latest_capture_at = payload, received_at
                self.telemetry_store.update_capture(
                    payload, received_at=received_at)

    def scan_history_cb(self, msg):
        try:
            payload = validate_history_payload(
                json.loads(msg.data),
                int(configured_value(self, 'maximum_captures')))
        except (json.JSONDecodeError, TypeError, ValueError):
            return
        received_at = time.monotonic()
        with self._lock:
            self.latest_scan_history = payload
            self.latest_scan_history_at = received_at
            self.telemetry_store.update_scan_history(
                payload, received_at=received_at)

    def view_generation_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(payload, dict):
            return
        received_at = time.monotonic()
        with self._lock:
            self.latest_view_generation = payload
            self.latest_view_generation_at = received_at

    def wait_for_view_generation(
            self, goal_handle, session, accepted_views, timeout):
        """Wait until the bridge has cached this scan's exact generation."""
        last_reason = 'view generation receipt has not arrived'
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            self.guard(goal_handle, session)
            with self._lock:
                receipt = self.latest_view_generation
                received_at = self.latest_view_generation_at
                barrier = self.view_generation_barrier_at
                history = self.latest_scan_history
            if receipt is None or received_at <= barrier:
                last_reason = (
                    'waiting for a bridge receipt newer than the plan barrier')
            else:
                last_reason = generation_matches_expected(
                    receipt, history, accepted_views)
                if not last_reason:
                    return
            time.sleep(0.05)
        raise MissionFailure(
            'view planner generation did not become ready: %s' % last_reason,
            failure_code='NO_REACHABLE_PLAN', retryable=True)

    def current_scan_feature_coverage(self):
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            history = self.latest_scan_history or {}
            capture = self.latest_capture
        else:
            snapshot = telemetry_store.snapshot()
            history = (
                snapshot.mission.scan_history.value
                if snapshot.mission.scan_history is not None else {})
            capture = (
                snapshot.mission.capture.value
                if snapshot.mission.capture is not None else {})
        persisted = persisted_achieved_history(
            capture.get('scan_dir', ''))
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
            capture.get('scan_dir', ''),
            minimum_views=int(configured_value(self, 'required_captures')))
        coverage = achieved_feature_coverage(
            achieved_entries,
            target_center,
            minimum_views=int(configured_value(self, 'required_captures')),
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
            self.telemetry_store.update_joints(
                msg, received_at=received_at, source_stamp_ns=stamp_ns,
                frame_id=str(getattr(msg.header, 'frame_id', '')))
            self.joint_feedback_rejection = ''

    def arm_status_cb(self, msg):
        received_at = time.monotonic()
        with self._lock:
            self.latest_arm_status = msg
            self.latest_arm_status_at = received_at
            # PiperStatusMsg has no std_msgs/Header.  Keep its existing
            # receipt-time freshness semantics and record an empty frame.
            header = getattr(msg, 'header', None)
            self.telemetry_store.update_arm_status(
                msg, received_at=received_at,
                frame_id=str(getattr(header, 'frame_id', '')))

    def camera_health_cb(self, msg):
        received_at = time.monotonic()
        with self._lock:
            self.latest_camera_health = msg
            self.latest_camera_health_at = received_at
            self.telemetry_store.update_camera(
                msg, received_at=received_at,
                frame_id=str(getattr(msg.header, 'frame_id', '')))

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
                configured_value(self, 'max_pending_missions'))
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
            self._cancellation_tokens[task_id] = CancellationToken()
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

    def cancel_cb(self, goal_handle):
        task_id = str(goal_handle.request.task_id)
        with self._lock:
            token = self._cancellation_tokens.get(task_id)
        if token is not None:
            token.cancel('tracked robot cancelled the task')
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
            cancellation_tokens = getattr(
                self, '_cancellation_tokens', None)
            if cancellation_tokens is None:
                cancellation_tokens = {}
                self._cancellation_tokens = cancellation_tokens
            cancellation = cancellation_tokens.get(task_id)
            if cancellation is None:
                cancellation = CancellationToken()
                cancellation_tokens[task_id] = cancellation
        if bool(goal_handle.is_cancel_requested):
            cancellation.cancel('tracked robot cancelled the task')
        if normalized is None:
            try:
                normalized = validate_goal_payload(
                    self.goal_payload(goal_handle.request))
            except (TypeError, ValueError) as exc:
                goal_handle.abort()
                self.finish_queue_dispatch(task_id)
                return self.action_result('FAILED', str(exc), False)
        if cancellation.cancelled or self._process_shutdown_requested:
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
        context = MissionContext(session=session, cancellation=cancellation)
        operations = _MissionNodeOperations(
            self, goal_handle, cancellation,
            rough_target=goal_handle.request.rough_target)
        self._active_cancellation_token = cancellation
        self._active_engine_session = session
        try:
            engine_result = mission_engine_for(
                self, operations).execute(context)
        finally:
            self._active_cancellation_token = None
            self._active_engine_session = None
        failure = engine_result.failure
        capture = dict(self.latest_capture)
        discarded_dataset_path = ''
        dataset_discarded = False
        dataset_discard_reason = ''
        candidate_dataset_path = str(capture.get('scan_dir', '')).strip()
        process_health = self.processes.health()
        capture_writer_stopped = not bool(
            process_health.get('scan_stack', {}).get('running', False))
        try:
            project_root_value = configured_value(self, 'project_root')
        except (AttributeError, KeyError):
            project_root_value = None
        dataset_root = (
            Path(str(project_root_value)) / 'datasets' / 'active_scan'
            if project_root_value else None)
        if failure is not None and not candidate_dataset_path \
                and capture_writer_stopped and dataset_root is not None:
            candidate_dataset_path, dataset_discard_reason = (
                find_failed_mission_dataset(
                    dataset_root, session.task_id, session.mission_sha256))
        if failure is not None and candidate_dataset_path:
            try:
                reported_captures = int(
                    capture.get('captured_frame_count', 0) or 0)
            except (TypeError, ValueError):
                reported_captures = -1
            if not capture_writer_stopped:
                dataset_discard_reason = (
                    'refused zero-capture cleanup while scan writer is active')
            elif (dataset_root is not None
                  and session.accepted_captures == 0
                  and reported_captures == 0):
                discarded_dataset_path = candidate_dataset_path
                dataset_discarded, dataset_discard_reason = (
                    discard_failed_zero_capture_dataset(
                        discarded_dataset_path, dataset_root,
                        session.task_id, session.mission_sha256))
                if dataset_discarded:
                    capture['scan_dir'] = ''
                    capture['manifest_sha256'] = ''
                    self.get_logger().warn(
                        dataset_discard_reason + ': ' + discarded_dataset_path)
                else:
                    self.get_logger().error(
                        'refused failed-scan dataset cleanup for %s: %s'
                        % (discarded_dataset_path, dataset_discard_reason))
        outcome = engine_result.outcome
        if engine_result.succeeded:
            goal_handle.succeed()
        else:
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
                'rough_target_base_link_xyz': (
                    [float(value) for value in context.target]
                    if context.target is not None else []),
                'processes': self.processes.health(),
                'home_positions_rad': list(session.home_positions_rad),
                'pre_home_positions_rad': list(
                    session.pre_home_positions_rad),
                'pre_home_completed': bool(session.pre_home_completed),
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
                'zero_capture_dataset_discarded': bool(dataset_discarded),
                'discarded_dataset_path': discarded_dataset_path,
                'dataset_discard_reason': dataset_discard_reason,
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
            cancellation_tokens = getattr(
                self, '_cancellation_tokens', None)
            if cancellation_tokens is not None:
                cancellation_tokens.pop(str(task_id), None)

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
        """Delegate the compatibility entry point to the mission engine."""
        cancellation = getattr(self, '_active_cancellation_token', None)
        if cancellation is None:
            cancellation = CancellationToken()
        context = MissionContext(
            session=session, cancellation=cancellation, target=target)
        operations = _MissionNodeOperations(
            self, goal_handle, cancellation)
        return mission_engine_for(self, operations).run_pipeline(context)

    def start_processes(self, goal_handle, session):
        if not self.param_bool('manage_processes'):
            return
        root = Path(str(configured_value(self, 'project_root'))).resolve()
        environment = ProcessSupervisor.build_environment({
            'PIPER_ARM_ROOT': str(root),
            'PIPER_AUTO_ENABLE': 'false',
            'PIPER_ENABLE_REAL_VIEWPOINT_MOTION': (
                '1' if self.param_bool('enable_real_arm_motion') else '0'),
            'PIPER_VIEWPOINT_MISSION_POLICY': '1',
            'PIPER_VIEWPOINT_CLOSED_LOOP_ONE_VIEW': '1',
            'PIPER_VIEWPOINT_SPEED_PERCENT': str(
                configured_value(self, 'free_motion_speed_percent')),
            'PIPER_VIEWPOINT_MAX_VIEWS': str(
                configured_value(self, 'maximum_captures')),
            'PIPER_VIEWPOINT_MIN_VIEWS': str(
                configured_value(self, 'required_captures')),
            'PIPER_RETURN_HOME_POSITIONS_RAD': json.dumps(
                list(session.home_positions_rad), separators=(',', ':')),
            'PIPER_PRE_HOME_POSITIONS_RAD': json.dumps(
                list(session.pre_home_positions_rad), separators=(',', ':')),
            'PIPER_MISSION_TASK_ID': session.task_id,
            'PIPER_MISSION_SHA256': session.mission_sha256,
            'PIPER_TARGET_LABEL': session.goal['target_label'],
            'PIPER_TARGET_PROFILE': session.goal['target_profile'],
            'PIPER_TARGET_PROMPT': session.goal['target_prompt'],
        })
        calibration_path, calibration_sha256 = calibration_identity_for_mission(
            root, environment)
        environment['PIPER_HAND_EYE_CALIBRATION'] = str(calibration_path)
        environment['PIPER_CALIBRATION_SHA256'] = calibration_sha256
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
            telemetry_store = getattr(self, 'telemetry_store', None)
            if telemetry_store is not None:
                telemetry_store.clear_arm_feedback()
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
            telemetry_store = getattr(self, 'telemetry_store', None)
            if telemetry_store is not None:
                telemetry_store.clear_camera()
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
                telemetry_store = getattr(self, 'telemetry_store', None)
                if telemetry_store is None:
                    fresh = (
                        self.latest_joints is not None
                        and now - self.latest_joints_at <= 0.25)
                else:
                    observation = telemetry_store.snapshot().arm.joints
                    fresh = (
                        observation is not None
                        and not observation.is_stale_at(now, 0.25))
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
            telemetry_store = getattr(self, 'telemetry_store', None)
            if telemetry_store is None:
                health = self.latest_camera_health
                health_at = self.latest_camera_health_at
            else:
                observation = telemetry_store.snapshot().perception.camera
                health = None if observation is None else observation.value
                health_at = (
                    0.0 if observation is None else observation.received_at)
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
        configured = str(configured_value(self, 'home_pose_path')).strip()
        if not configured:
            configured = str(
                Path(str(configured_value(self, 'project_root'))).resolve()
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
            if not bool(payload.get('pre_home_configured', False)):
                raise MissionFailure(
                    'configured home pose has no terminal pre-home waypoint; '
                    'record pre-home before enabling autonomous motion',
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

    def authorize_mission(
            self, session, revoke=False, terminal_home=False):
        request = AuthorizeMission.Request()
        request.task_id = session.task_id
        request.mission_sha256 = session.mission_sha256
        if terminal_home and not revoke:
            # A scan deadline ends scan authority, but it must not prevent the
            # already-established terminal home sequence.  Give each direct
            # home stage one fresh, bounded execution window using the
            # existing queue/result limits.  No scan phase is resumed here.
            workflow = workflow_config_for(self)
            expires = time.time() + (
                workflow.plan_request_queue_timeout_sec
                + workflow.plan_result_timeout_sec + 10.0)
        else:
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
            self.prepare_client, request,
            workflow_config_for(self).acquisition_service_timeout_sec,
            'rough acquisition service')
        if not result.accepted or str(result.session_id) != request.session_id:
            raise MissionFailure(str(result.message))
        return request.session_id

    def request_multiview_plan(self, goal_handle, session):
        queue_deadline = (
            time.monotonic()
            + workflow_config_for(self).plan_request_queue_timeout_sec)
        visual_deadline = None
        runtime_refresh_reported = False
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
            if (
                    result.request_id
                    and as_failure(result.message).has(
                        FailureTag.REQUEST_ALREADY_PENDING)):
                time.sleep(0.25)
                continue
            if visual_reacquisition_plan_request_rejection(result.message):
                now = time.monotonic()
                if visual_deadline is None:
                    # No request entered the worker spool and no motion can be
                    # authorized.  Let the existing SAM2/heavy recovery restore
                    # one measured lock, then retry a completely fresh snapshot.
                    visual_deadline = (
                        now + workflow_config_for(
                            self).scan_visual_reacquisition_timeout_sec)
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
            if runtime_freshness_plan_request_rejection(result.message):
                # The bridge has not queued a worker request, so no trajectory
                # can exist or move. Reuse the bounded queue window while the
                # authoritative freshness gate awaits its next valid sample.
                if not runtime_refresh_reported:
                    runtime_refresh_reported = True
                    self.startup_progress(
                        goal_handle, session,
                        'runtime telemetry dipped before scan planning; '
                        'holding without motion for one fresh snapshot')
                self.guard(goal_handle, session)
                time.sleep(0.1)
                continue
            raise MissionFailure(str(result.message))
        if visual_deadline is not None:
            raise MissionFailure(
                'measured target lock did not recover before command-free '
                'scan planning', failure_code='TARGET_NOT_FOUND', retryable=True)
        raise MissionFailure('timed out waiting to queue one fresh multiview plan')

    def wait_for_plan(self, goal_handle, session, kind, request_id, timeout):
        generation_started = time.monotonic()

        def matching():
            telemetry_store = getattr(self, 'telemetry_store', None)
            if telemetry_store is None:
                plan = self.latest_plan
                received_at = self.latest_plan_at
            else:
                observation = telemetry_store.snapshot().mission.plan
                plan = None if observation is None else observation.value
                received_at = (
                    0.0 if observation is None else observation.received_at)
            if plan is None or received_at < generation_started:
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
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            plan = self.latest_plan
        else:
            observation = telemetry_store.snapshot().mission.plan
            plan = None if observation is None else observation.value
        if kind == 'MULTIVIEW_SCAN':
            if int(plan.planned_viewpoints) != 1:
                raise MissionFailure(
                    'closed-loop multiview plan contains %d views; expected 1'
                    % int(plan.planned_viewpoints))
        return plan

    def approve_plan(self, goal_handle, session, plan):
        normal_deadline = (
            time.monotonic()
            + workflow_config_for(self).plan_approval_transient_timeout_sec)
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
                    now + workflow_config_for(
                        self).scan_visual_reacquisition_timeout_sec)
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
        if (
                not result.success
                and not as_failure(result.message).has(
                    FailureTag.WORKFLOW_ALREADY_ACTIVE)):
            raise MissionFailure(str(result.message))
        deadline = (
            time.monotonic()
            + workflow_config_for(self).workflow_assessment_timeout_sec)
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
            % workflow_config_for(self).workflow_assessment_timeout_sec)

    def wait_for_execution(self, goal_handle, session, successes, timeout, failures):
        started = time.monotonic()

        def completed():
            telemetry_store = getattr(self, 'telemetry_store', None)
            if telemetry_store is None:
                status = self.latest_execution
                received_at = self.latest_execution_at
            else:
                observation = telemetry_store.snapshot().mission.execution
                status = None if observation is None else observation.value
                received_at = (
                    0.0 if observation is None else observation.received_at)
            if status is None or received_at < started:
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
                raise MissionFailure(
                    '%s: %s' % (state, status.reason),
                    failure_code=(
                        'MISSION_FAILED' if state == 'ABORTED'
                        else 'NO_REACHABLE_PLAN'),
                )
            return state in successes

        self.wait_for(
            goal_handle, session, completed, timeout,
            'execution did not reach %s before timeout' % '/'.join(successes))
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            return self.latest_execution
        observation = telemetry_store.snapshot().mission.execution
        return None if observation is None else observation.value

    def prove_current_hold(self, goal_handle, session):
        self.require_fresh_joint_feedback()
        # The enable service can return between CAN feedback samples.  Bind
        # the hold to the first sample received after enable completes so a
        # pre-enable sample cannot make a correctly held arm look displaced.
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            previous_sample_at = self.latest_joints_at
        else:
            observation = telemetry_store.snapshot().arm.joints
            previous_sample_at = (
                0.0 if observation is None else observation.received_at)
        fresh_deadline = time.monotonic() + 1.0
        while time.monotonic() < fresh_deadline:
            if telemetry_store is None:
                sample_at = self.latest_joints_at
            else:
                observation = telemetry_store.snapshot().arm.joints
                sample_at = (
                    0.0 if observation is None else observation.received_at)
            if sample_at > previous_sample_at:
                break
            self.guard(goal_handle, session)
            time.sleep(0.01)
        self.require_fresh_joint_feedback()
        if telemetry_store is None:
            joints = self.latest_joints
        else:
            observation = telemetry_store.snapshot().arm.joints
            joints = None if observation is None else observation.value
        # While STARTUP_WRIST is armed, the driver deliberately holds the
        # exact measured extended J6 coordinate instead of clipping it to
        # [-pi, +pi].  The settle reference must describe that same command.
        initial = startup_measured_hold_reference(joints.position[:6])
        result = self.call_service(
            self.hold_client, Trigger.Request(), 8.0,
            'executor current-position hold service')
        if (
                not result.success
                or not as_failure(result.message).has(
                    FailureTag.HOLD_REQUESTED)):
            return False
        # Enabling the controller already holds its commanded joint target.
        # Keep one explicit current-position command so a stale controller
        # target cannot survive across missions, but do not reject ordinary
        # encoder/mechanical settling by trying to prove a 0.005-rad window.
        self.guard(goal_handle, session)
        self.require_fresh_joint_feedback()
        session.current_hold_proved = True
        self.last_hold_diagnostic = (
            'current-position hold acknowledged at %s; position-window proof '
            'is intentionally not a mission gate'
            % [round(float(value), 6) for value in initial])
        return True

    def safe_shutdown(self, session, normal_completion, failure=None):
        """Compatibility entry point for the engine-owned terminal sequence."""
        cancellation = getattr(self, '_active_cancellation_token', None)
        if cancellation is None:
            cancellation = CancellationToken()
        context = MissionContext(session=session, cancellation=cancellation)
        operations = _MissionNodeOperations(
            self, None, cancellation)
        return mission_engine_for(self, operations).shutdown(
            context, normal_completion=normal_completion, failure=failure)

    def at_configured_home(
            self, session, tolerance_rad=0.20, target_positions=None,
            home_stage=None):
        self.require_fresh_joint_feedback()
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            joints = self.latest_joints
        else:
            observation = telemetry_store.snapshot().arm.joints
            joints = None if observation is None else observation.value
        current = np.asarray(joints.position[:6], dtype=float)
        target = np.asarray(
            session.home_positions_rad
            if target_positions is None else target_positions,
            dtype=float)
        if str(home_stage or '').strip().upper() == 'STARTUP_WRIST':
            return bool(
                current.shape == (6,)
                and target.shape == (6,)
                and np.all(np.isfinite(current))
                and np.all(np.isfinite(target))
                and abs(float(current[5]) - float(target[5]))
                <= float(tolerance_rad))
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
        normalized_home_stage = str(home_stage).strip().upper()

        def record_stage_proof():
            # A PRE_HOME proof must not also suppress the following distinct
            # ROUGH_HOME transaction.  Keep each configured stage's proof in
            # its own MissionSession field.
            if normalized_home_stage == 'PRE_HOME':
                session.pre_home_completed = True
            elif normalized_home_stage == 'STORAGE_WRIST':
                session.storage_wrist_proved = True
            else:
                session.return_home_proved = True

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
            names = sorted(set(failed).intersection(('driver', 'scan_stack')))
            return names

        try:
            if self.at_configured_home(
                    session, target_positions=target_positions,
                    home_stage=home_stage):
                record_stage_proof()
                self.last_return_home_diagnostic = (
                    'fresh feedback is already within %s tolerance'
                    % str(home_stage).lower())
                return True
        except MissionFailure as exc:
            return fail_home(exc)
        if startup:
            try:
                self.require_fresh_joint_feedback()
            except MissionFailure as exc:
                return fail_home(exc)
        else:
            result = self.call_service(
                self.cancel_client, Trigger.Request(), 8.0,
                'executor stop-and-hold-before-home service')
            if not result.success:
                return fail_home(
                    result.message or 'executor could not hold before home')
            if not as_failure(result.message).has(
                    FailureTag.HOLD_ACKNOWLEDGED):
                return fail_home(
                    'executor response did not prove a hold before home: '
                    + str(result.message))
            try:
                self.require_fresh_joint_feedback()
            except MissionFailure as exc:
                return fail_home(exc)

        # Terminal recovery is intentionally non-interrupting.  The original
        # scan authorization may have expired (that is what led us here), so
        # renew it after the terminal path has either stopped/held scan motion
        # or refreshed the static-startup joint state.  Startup motion keeps
        # its original interruptible mission authorization.
        if goal_handle is None:
            try:
                self.authorize_mission(session, terminal_home=True)
            except MissionFailure as exc:
                return fail_home(
                    'terminal home authorization failed: %s' % exc)

        request = ExecuteHomeStage.Request()
        request.task_id = session.task_id
        request.mission_sha256 = session.mission_sha256
        request.home_stage = str(home_stage).upper()
        request.joint_goal_positions_rad = [
            float(value) for value in target_positions]
        execution_started = time.monotonic()
        started = self.call_service(
            self.execute_home_stage_client, request,
            workflow_config_for(self).plan_request_queue_timeout_sec,
            'direct configured-home executor service')
        if not started.accepted or not started.execution_id:
            return fail_home(
                'direct home request was rejected: ' + str(started.message))
        execution_id = str(started.execution_id)
        deadline = (
            time.monotonic() + workflow_config_for(
                self).plan_result_timeout_sec)
        while time.monotonic() < deadline:
            guard_active_mission()
            telemetry_store = getattr(self, 'telemetry_store', None)
            if telemetry_store is None:
                status = self.latest_execution
                status_at = self.latest_execution_at
            else:
                observation = telemetry_store.snapshot().mission.execution
                status = None if observation is None else observation.value
                status_at = (
                    0.0 if observation is None else observation.received_at)
            if (
                    status is not None
                    and status_at >= execution_started
                    and str(status.plan_id) == execution_id):
                state = str(status.state)
                reason = str(status.reason)
                if (
                        state == 'ABORTED'
                        and as_failure(reason).has(
                            FailureTag.TERMINAL_HOME_REACHED)):
                    if self.at_configured_home(
                            session, target_positions=target_positions,
                            home_stage=home_stage):
                        record_stage_proof()
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
                    'critical process exited during direct home motion: '
                    + ', '.join(failed))
            time.sleep(0.05)
        return fail_home(
            'timed out waiting for direct configured-home execution proof')

    def prove_current_hold_for_shutdown(self, session):
        try:
            self.guard_motor_control(session)
            self.require_fresh_joint_feedback()
            telemetry_store = getattr(self, 'telemetry_store', None)
            if telemetry_store is None:
                joints = self.latest_joints
            else:
                observation = telemetry_store.snapshot().arm.joints
                joints = None if observation is None else observation.value
            initial = energized_hold_target(joints.position[:6])
            result = self.call_service(
                self.hold_client, Trigger.Request(), 8.0,
                'executor shutdown hold service')
            if (
                    not result.success
                    or not as_failure(result.message).has(
                        FailureTag.HOLD_REQUESTED)):
                return False
            self.guard_motor_control(session)
            self.require_fresh_joint_feedback()
            session.current_hold_proved = True
            self.last_hold_diagnostic = (
                'shutdown hold acknowledged at %s; position-window proof is '
                'intentionally not a disable gate'
                % [round(float(value), 6) for value in initial])
            return True
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
            telemetry_store = getattr(self, 'telemetry_store', None)
            if telemetry_store is None:
                arm_status = self.latest_arm_status
                arm_status_age = time.monotonic() - self.latest_arm_status_at
            else:
                snapshot = telemetry_store.snapshot()
                observation = snapshot.arm.status
                arm_status = (
                    None if observation is None else observation.value)
                arm_status_age = (
                    math.inf if observation is None else
                    observation.age_at(snapshot.captured_at))
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
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            capture = self.latest_capture
        else:
            observation = telemetry_store.snapshot().mission.capture
            capture = {} if observation is None else observation.value
        session.accepted_captures = int(
            capture.get('captured_frame_count', session.accepted_captures))
        cancellation = getattr(self, '_active_cancellation_token', None)
        if self._process_shutdown_requested:
            raise MissionFailure(
                'coordinator shutdown requested', outcome='CANCELLED',
                failure_code='CANCELLED', retryable=True)
        if cancellation is not None and cancellation.cancelled:
            raise MissionFailure(
                cancellation.reason or 'tracked robot cancelled the task',
                outcome='CANCELLED', failure_code='CANCELLED', retryable=True)
        # Compatibility only for Phase 0/1 harnesses and direct helper calls.
        # Production execution always binds the application cancellation token.
        if (
                cancellation is None
                and goal_handle is not None
                and goal_handle.is_cancel_requested):
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
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            readiness = self.latest_readiness
            age = time.monotonic() - self.latest_readiness_at
        else:
            snapshot = telemetry_store.snapshot()
            observation = snapshot.mission.readiness
            readiness = None if observation is None else observation.value
            age = (
                math.inf if observation is None else
                observation.age_at(snapshot.captured_at))
        if readiness is None or age > 1.0:
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
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            joints = self.latest_joints
            age = time.monotonic() - self.latest_joints_at
        else:
            snapshot = telemetry_store.snapshot()
            observation = snapshot.arm.joints
            joints = None if observation is None else observation.value
            age = (
                math.inf if observation is None else
                observation.age_at(snapshot.captured_at))
        if joints is None or age > 1.0:
            raise MissionFailure('joint feedback is missing or stale')
        values = np.asarray(joints.position[:6], dtype=float)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise MissionFailure('joint feedback is not six finite positions')

    def clear_plan_cache(self):
        with self._lock:
            self.latest_plan = None
            self.latest_plan_at = 0.0
            self.view_generation_barrier_at = time.monotonic()
            telemetry_store = getattr(self, 'telemetry_store', None)
            if telemetry_store is not None:
                telemetry_store.clear_plan()

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
            self.latest_view_generation = None
            self.latest_view_generation_at = 0.0
            self.view_generation_barrier_at = 0.0
            self.latest_scan_target_center = None
            self.last_scan_feature_coverage = {}
            telemetry_store = getattr(self, 'telemetry_store', None)
            if telemetry_store is not None:
                telemetry_store.clear_mission_runtime()

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
                configured_value(self, 'required_captures')),
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
        telemetry_store = getattr(self, 'telemetry_store', None)
        if telemetry_store is None:
            capture = self.latest_capture
        else:
            observation = telemetry_store.snapshot().mission.capture
            capture = {} if observation is None else observation.value
        feedback.accepted_captures = int(
            capture.get('captured_frame_count', 0))
        feedback.required_captures = int(
            configured_value(self, 'required_captures'))
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
                configured_value(self, 'required_captures')),
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
                configured_value(self, 'required_captures'))
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
                    float(configured_value(
                        self, 'mission_queue_coalesce_sec')))
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
                    configured_value(self, 'max_pending_missions'))
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
