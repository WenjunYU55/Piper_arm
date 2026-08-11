#!/usr/bin/env python3

import json
import math
import os
import queue
import signal
import subprocess
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import List
import uuid

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from piper_gui_automation import (
    ACQUISITION_PLAN_TIMEOUT_SEC,
    ACQUISITION_SERVICE_TIMEOUT_SEC,
    AcquisitionPhase,
    AutomationSession,
    command_publisher_identity_pending,
    command_publisher_ownership_rejection,
    MULTIVIEW_PLAN_TIMEOUT_SEC,
    MULTIVIEW_SCAN,
    PLAN_REQUEST_QUEUE_TIMEOUT_SEC,
    plan_matches_request,
    plan_rejection,
    readiness_rejection,
    retryable_multiview_terminal,
    ROUGH_ACQUISITION,
    STEP4_BUSY_PHASES,
    STEP45_AUTO_RECOVERY_LOCK_TIMEOUT_SEC,
    STEP45_AUTO_RECOVERY_MAX_ATTEMPTS,
    STEP45_AUTO_RECOVERY_RETRY_SEC,
    Step4Phase,
    step45_auto_recovery_blocker,
    step4_workflow_action,
    tracking_health_rejection,
    tracking_lock_rejection,
    validate_automation_speed,
    validate_rough_coordinates,
    WORKFLOW_ASSESSMENT_TIMEOUT_SEC,
)
from piper_mobile_manipulation.msg import (
    MeshJobStatus,
    ScanExecutionPlan,
    ScanExecutionStatus,
    TesseractReadiness,
    TrackingHealth,
)
from piper_mobile_manipulation.action import RunTargetScan
from piper_mobile_manipulation.home_pose import (
    load_home_pose,
    save_home_pose,
    validate_home_profile_limits,
)
from piper_mobile_manipulation.srv import (
    ApproveScanExecution,
    PrepareAcquisition,
    ReportTrackedRobotHomed,
    RequestTesseractPlan,
)
from piper_mobile_manipulation.scan_motion import (
    energized_hold_target,
    PiperScanKinematics,
    URDF_JOINT_LIMITS,
    interpolate_joint_path,
    validate_joint_path,
)
from piper_msgs.msg import PiperStatusMsg
from piper_msgs.srv import Enable


DEFAULT_JOINTS = [
    ("joint1", -2.8, 2.8, "rad"),
    ("joint2", -2.1, 2.1, "rad"),
    ("joint3", -2.8, 2.8, "rad"),
    ("joint4", -2.8, 2.8, "rad"),
    ("joint5", -2.1, 2.1, "rad"),
    ("joint6", -math.pi, math.pi, "rad"),
    ("gripper", 0.0, 0.08, "m"),
]

BOUNDS_PATH = os.path.join(os.path.dirname(__file__), "piper_joint_bounds.json")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
HOME_POSE_PATH = os.path.join(PROJECT_ROOT, "piper_home_pose.json")
AUTOMATION_CRITICAL_NODES = {
    "/tesseract_plan_bridge",
    "/viewpoint_reachability_filter",
    "/supervised_cube_workflow",
    "/scan_viewpoint_executor",
    "/scan_viewpoint_planner",
    "/scan_target_acquisition",
    "/scan_capture",
}
DISABLED_HOME_DROOP_TOLERANCE_RAD = 0.05


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def load_joint_limits():
    joints = list(DEFAULT_JOINTS)
    if not os.path.exists(BOUNDS_PATH):
        return joints, "default limits"

    try:
        with open(BOUNDS_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return joints, f"bounds file ignored: {exc}"

    saved = data.get("joints", {})
    merged = []
    for name, low, high, unit in joints:
        record = saved.get(name)
        if record is None or record.get("valid", True) is False:
            merged.append((name, low, high, unit))
            continue

        measured_low = float(record.get("min", low))
        measured_high = float(record.get("max", high))
        if measured_low == measured_high:
            merged.append((name, low, high, unit))
            continue

        merged.append((name, min(measured_low, measured_high), max(measured_low, measured_high), unit))

    return merged, f"loaded {BOUNDS_PATH}"


class PiperGuiRos(Node):
    def __init__(self, events: "queue.Queue[tuple]") -> None:
        super().__init__("piper_native_gui")
        self.events = events
        self.latest_feedback = None
        self.latest_feedback_monotonic = None
        self.latest_status = None
        self.latest_tesseract_readiness = None
        self.latest_tesseract_readiness_monotonic = None
        self.latest_tracking_health = None
        self.latest_tracking_health_monotonic = None
        self.manual_commands_enabled = True
        self.command_lock = threading.Lock()
        self.client_lock = threading.Lock()
        self.retired_acquisition_clients = []
        self.service_callback_group = ReentrantCallbackGroup()

        self.joint_pub = None
        self.enable_manual_command_publisher()
        self.feedback_sub = self.create_subscription(
            JointState, "/joint_states_single", self.feedback_callback, 10
        )
        self.status_sub = self.create_subscription(
            PiperStatusMsg, "/arm_status", self.status_callback, 10
        )
        self.preview_set_pub = self.create_publisher(
            JointState, "/piper_gui/preview_set", 10
        )
        self.preview_sub = self.create_subscription(
            JointState,
            "/piper_gui/preview_joint_states",
            self.preview_callback,
            10,
        )
        self.enable_client = self.create_client(
            Enable, "/enable_srv",
            callback_group=self.service_callback_group)
        # Foxy/rclpy requires subscription handles to remain strongly owned for
        # the node lifetime.  Keep the automation readers explicitly so graph
        # churn and shutdown cannot collect an entity still referenced by the
        # executor (previously observed as callback-group weakref assertions).
        self.automation_subscriptions = [
            self.create_subscription(
                ScanExecutionPlan,
                "/piper/scan_execution_plan",
                lambda msg: self.events.put(("scan_plan", msg)),
                10,
            ),
            self.create_subscription(
                ScanExecutionStatus,
                "/piper/scan_execution_status",
                lambda msg: self.events.put(("scan_status", msg)),
                10,
            ),
            self.create_subscription(
                String,
                "/piper/heavy_refresh_status",
                lambda msg: self.events.put(("heavy_status", msg.data)),
                10,
            ),
            self.create_subscription(
                String,
                "/piper/supervised_workflow_status",
                lambda msg: self.events.put(("workflow_status", msg.data)),
                10,
            ),
            self.create_subscription(
                String,
                "/piper/scan_capture_status",
                lambda msg: self.events.put(("capture_status", msg.data)),
                10,
            ),
            self.create_subscription(
                TrackingHealth,
                "/piper/tracking_health",
                self.tracking_health_callback,
                10,
            ),
            self.create_subscription(
                TesseractReadiness,
                "/piper/tesseract_readiness",
                self.tesseract_readiness_callback,
                10,
            ),
            self.create_subscription(
                MeshJobStatus,
                "/piper/mesh_job_status",
                lambda msg: self.events.put(("mesh_job_status", msg)),
                10,
            ),
        ]
        self.acquisition_prepare_client = self.create_client(
            PrepareAcquisition, "/scan_target_acquisition/prepare",
            callback_group=self.service_callback_group)
        self.multiview_plan_client = self.create_client(
            RequestTesseractPlan, "/tesseract_plan_bridge/request_plan",
            callback_group=self.service_callback_group)
        self.workflow_start_client = self.create_client(
            Trigger, "/supervised_cube_workflow/start",
            callback_group=self.service_callback_group)
        self.workflow_diagnostic_client = self.create_client(
            Trigger, "/supervised_cube_workflow/diagnostic_state",
            callback_group=self.service_callback_group)
        self.scan_approve_client = self.create_client(
            ApproveScanExecution, "/scan_viewpoint_executor/approve",
            callback_group=self.service_callback_group)
        self.scan_cancel_client = self.create_client(
            Trigger, "/scan_viewpoint_executor/cancel",
            callback_group=self.service_callback_group)
        self.mission_action_client = ActionClient(
            self, RunTargetScan, '/piper/run_target_scan',
            callback_group=self.service_callback_group)
        self.mission_goal_handle = None
        self.mission_task_id = ''
        self.report_base_home_client = self.create_client(
            ReportTrackedRobotHomed,
            '/piper/report_tracked_robot_homed',
            callback_group=self.service_callback_group)

    def feedback_callback(self, msg: JointState) -> None:
        self.latest_feedback = msg
        self.latest_feedback_monotonic = time.monotonic()
        self.events.put(("feedback", msg))

    def status_callback(self, msg: PiperStatusMsg) -> None:
        self.latest_status = msg
        self.events.put(("status", msg))

    def preview_callback(self, msg: JointState) -> None:
        self.events.put(("preview", msg))

    def tesseract_readiness_callback(self, msg: TesseractReadiness) -> None:
        self.latest_tesseract_readiness = msg
        self.latest_tesseract_readiness_monotonic = time.monotonic()
        self.events.put(("tesseract_readiness", msg))

    def tracking_health_callback(self, msg: TrackingHealth) -> None:
        self.latest_tracking_health = msg
        self.latest_tracking_health_monotonic = time.monotonic()
        self.events.put(("tracking_health", msg))

    def clear_generation_cache(self) -> None:
        self.latest_tesseract_readiness = None
        self.latest_tesseract_readiness_monotonic = None

    def publish_preview_target(self, positions: List[float]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "piper_native_gui_preview_set"
        msg.name = ["joint%d" % index for index in range(1, len(positions) + 1)]
        msg.position = list(positions)
        self.preview_set_pub.publish(msg)

    def publish_joint_target(self, positions: List[float], speed: float, effort: float) -> None:
        with self.command_lock:
            publisher = self.joint_pub if self.manual_commands_enabled else None
        if publisher is None:
            self.events.put((
                "command_blocked",
                "Manual joint commands are disabled while automation owns motion.",
            ))
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "piper_native_gui"
        msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        msg.position = positions
        msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, speed]
        msg.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, effort]
        publisher.publish(msg)
        self.events.put(("command", (positions, speed, effort)))

    def enable_manual_command_publisher(self) -> None:
        with self.command_lock:
            if self.joint_pub is None:
                self.joint_pub = self.create_publisher(
                    JointState, "/joint_ctrl_single", 10)
            self.manual_commands_enabled = True

    def disable_manual_command_publisher(self) -> None:
        with self.command_lock:
            self.manual_commands_enabled = False
            if self.joint_pub is not None:
                self.destroy_publisher(self.joint_pub)
                self.joint_pub = None

    def command_publisher_names(self, resolution_timeout_sec=0.0):
        deadline = time.monotonic() + max(0.0, float(resolution_timeout_sec))
        while True:
            try:
                endpoints = self.get_publishers_info_by_topic("/joint_ctrl_single")
            except Exception:
                endpoints = []
            names = []
            for endpoint in endpoints:
                namespace = str(
                    getattr(endpoint, "node_namespace", "")).rstrip("/")
                name = str(getattr(endpoint, "node_name", ""))
                names.append((namespace + "/" + name).replace("//", "/"))
            unresolved = not names or command_publisher_identity_pending(names)
            if not unresolved or time.monotonic() >= deadline:
                return names
            time.sleep(0.05)

    def automation_nodes_present(self):
        try:
            current = {
                (str(namespace).rstrip("/") + "/" + str(name)).replace("//", "/")
                for name, namespace in self.get_node_names_and_namespaces()
            }
        except Exception:
            return []
        return sorted(AUTOMATION_CRITICAL_NODES.intersection(current))

    def automation_services_ready(self):
        services = {
            "/scan_target_acquisition/prepare":
                self.acquisition_prepare_client,
            "/tesseract_plan_bridge/request_plan":
                self.multiview_plan_client,
            "/supervised_cube_workflow/start":
                self.workflow_start_client,
            "/supervised_cube_workflow/diagnostic_state":
                self.workflow_diagnostic_client,
            "/scan_viewpoint_executor/approve":
                self.scan_approve_client,
            "/scan_viewpoint_executor/cancel":
                self.scan_cancel_client,
        }
        return [
            name for name, client in services.items()
            if not client.service_is_ready()
        ]

    def tesseract_readiness_rejection(
            self, planning_mode, maximum_age_sec=1.0):
        return readiness_rejection(
            self.latest_tesseract_readiness,
            self.latest_tesseract_readiness_monotonic,
            time.monotonic(),
            planning_mode,
            maximum_age_sec=maximum_age_sec,
        )

    def publish_rough_target_and_start(
            self, coordinates, session_id, stack_generation,
            attempt_generation) -> None:
        thread = threading.Thread(
            target=self._publish_rough_target_and_start,
            args=(
                tuple(coordinates),
                str(session_id),
                int(stack_generation),
                int(attempt_generation),
            ),
            daemon=True,
        )
        thread.start()

    def _fresh_acquisition_prepare_client(self):
        with self.client_lock:
            previous = self.acquisition_prepare_client
            client = self.create_client(
                PrepareAcquisition, "/scan_target_acquisition/prepare",
                callback_group=self.service_callback_group)
            self.acquisition_prepare_client = client
            if previous is not None:
                # Do not destroy an endpoint whose timed-out request may still
                # receive a late response in Foxy. Its attempt token makes the
                # response harmless, and the node owns it until shutdown.
                self.retired_acquisition_clients.append(previous)
            return client

    def _publish_rough_target_and_start(
            self, coordinates, session_id, stack_generation,
            attempt_generation) -> None:
        request = PrepareAcquisition.Request()
        request.session_id = session_id
        request.rough_target = PointStamped()
        request.rough_target.header.stamp = self.get_clock().now().to_msg()
        request.rough_target.header.frame_id = "base_link"
        request.rough_target.point.x, request.rough_target.point.y, \
            request.rough_target.point.z = [
                float(value) for value in coordinates]
        for attempt in range(2):
            client = self._fresh_acquisition_prepare_client()
            if not client.wait_for_service(timeout_sec=5.0):
                self.events.put((
                    "acquisition_start",
                    (
                        stack_generation,
                        attempt_generation,
                        "unavailable",
                        "/scan_target_acquisition/prepare is unavailable; "
                        "the managed acquisition node may have exited",
                    ),
                ))
                return
            future = client.call_async(request)
            deadline = (
                time.monotonic() + ACQUISITION_SERVICE_TIMEOUT_SEC)
            while (
                    rclpy.ok()
                    and not future.done()
                    and time.monotonic() < deadline):
                time.sleep(0.05)
            if future.done():
                try:
                    result = future.result()
                    if str(result.session_id) != session_id:
                        self.events.put((
                            "acquisition_start",
                            (
                                stack_generation,
                                attempt_generation,
                                "rejected",
                                "acquisition service returned a different session ID",
                            ),
                        ))
                    else:
                        self.events.put((
                            "acquisition_start",
                            (
                                stack_generation,
                                attempt_generation,
                                "accepted" if result.accepted else "rejected",
                                str(result.message),
                            ),
                        ))
                except Exception as exc:
                    self.events.put((
                        "acquisition_start",
                        (
                            stack_generation,
                            attempt_generation,
                            "error",
                            "acquisition service failed: %s" % exc,
                        ),
                    ))
                return
            try:
                future.cancel()
            except Exception:
                pass
            if not client.service_is_ready():
                self.events.put((
                    "acquisition_start",
                    (
                        stack_generation,
                        attempt_generation,
                        "unavailable",
                        "acquisition service disappeared while the request was "
                        "in flight; the managed node exited",
                    ),
                ))
                return
            if attempt == 0:
                continue
        self.events.put((
            "acquisition_start",
            (
                stack_generation,
                attempt_generation,
                "timeout",
                "acquisition request timed out twice; the exact same session "
                "and coordinates are preserved for a retry",
            ),
        ))

    def approve_scan_plan(self, plan_id, trajectory_sha256, event_name) -> None:
        thread = threading.Thread(
            target=self._approve_scan_plan,
            args=(str(plan_id), str(trajectory_sha256), str(event_name)),
            daemon=True,
        )
        thread.start()

    def _approve_scan_plan(self, plan_id, trajectory_sha256, event_name) -> None:
        if not self.scan_approve_client.wait_for_service(timeout_sec=5.0):
            self.events.put((event_name, (False, "scan approval service unavailable")))
            return
        request = ApproveScanExecution.Request()
        request.plan_id = plan_id
        request.trajectory_sha256 = trajectory_sha256
        request.confirmation = "EXECUTE APPROVED SCAN"
        future = self.scan_approve_client.call_async(request)
        self._wait_for_future(
            future, 10.0, event_name,
            lambda result: (bool(result.accepted), str(result.message)),
        )

    def request_multiview_plan(
            self, stack_generation, attempt_generation) -> None:
        thread = threading.Thread(
            target=self._request_multiview_plan,
            args=(int(stack_generation), int(attempt_generation)),
            daemon=True,
        )
        thread.start()

    def start_workflow_for_current_lock(
            self, stack_generation, attempt_generation) -> None:
        thread = threading.Thread(
            target=self._start_workflow_for_current_lock,
            args=(int(stack_generation), int(attempt_generation)),
            daemon=True,
        )
        thread.start()

    def request_workflow_diagnostic(
            self, stack_generation, attempt_generation) -> None:
        thread = threading.Thread(
            target=self._request_workflow_diagnostic,
            args=(int(stack_generation), int(attempt_generation)),
            daemon=True,
        )
        thread.start()

    def _start_workflow_for_current_lock(
            self, stack_generation, attempt_generation) -> None:
        if not self.workflow_start_client.wait_for_service(timeout_sec=5.0):
            self.events.put((
                "workflow_start",
                (
                    stack_generation,
                    attempt_generation,
                    False,
                    "supervised workflow start service unavailable",
                ),
            ))
            return
        future = self.workflow_start_client.call_async(Trigger.Request())
        deadline = time.monotonic() + 8.0
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done():
            self.events.put((
                "workflow_start",
                (
                    stack_generation,
                    attempt_generation,
                    False,
                    "supervised workflow start service call timed out",
                ),
            ))
            return
        try:
            result = future.result()
            self.events.put((
                "workflow_start",
                (
                    stack_generation,
                    attempt_generation,
                    bool(result.success),
                    str(result.message),
                ),
            ))
        except Exception as exc:
            self.events.put((
                "workflow_start",
                (
                    stack_generation,
                    attempt_generation,
                    False,
                    "supervised workflow start service failed: %s" % exc,
                ),
            ))

    def _request_workflow_diagnostic(
            self, stack_generation, attempt_generation) -> None:
        if not self.workflow_diagnostic_client.wait_for_service(timeout_sec=5.0):
            self.events.put((
                "workflow_diagnostic",
                (
                    stack_generation,
                    attempt_generation,
                    False,
                    {},
                    "supervised workflow diagnostic service unavailable",
                ),
            ))
            return
        future = self.workflow_diagnostic_client.call_async(Trigger.Request())
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done():
            self.events.put((
                "workflow_diagnostic",
                (
                    stack_generation,
                    attempt_generation,
                    False,
                    {},
                    "supervised workflow diagnostic service call timed out",
                ),
            ))
            return
        try:
            result = future.result()
            payload = json.loads(result.message) if result.success else {}
            if result.success and not isinstance(payload, dict):
                raise ValueError(
                    "workflow diagnostic response is not a JSON object")
            self.events.put((
                "workflow_diagnostic",
                (
                    stack_generation,
                    attempt_generation,
                    bool(result.success),
                    payload,
                    str(result.message),
                ),
            ))
        except Exception as exc:
            self.events.put((
                "workflow_diagnostic",
                (
                    stack_generation,
                    attempt_generation,
                    False,
                    {},
                    "supervised workflow diagnostic failed: %s" % exc,
                ),
            ))

    def _request_multiview_plan(
            self, stack_generation, attempt_generation) -> None:
        if not self.multiview_plan_client.wait_for_service(timeout_sec=5.0):
            self.events.put((
                "multiview_plan_request",
                (
                    stack_generation,
                    attempt_generation,
                    False,
                    "",
                    "Tesseract multiview planning service unavailable",
                ),
            ))
            return
        deadline = time.monotonic() + PLAN_REQUEST_QUEUE_TIMEOUT_SEC
        while rclpy.ok() and time.monotonic() < deadline:
            request = RequestTesseractPlan.Request()
            # Never create a duplicate request merely to bypass one already in
            # flight. Wait for it to finish, then take a fresh normal snapshot.
            request.force_refresh = False
            future = self.multiview_plan_client.call_async(request)
            while (
                    rclpy.ok()
                    and not future.done()
                    and time.monotonic() < deadline):
                time.sleep(0.05)
            if not future.done():
                break
            try:
                result = future.result()
            except Exception as exc:
                self.events.put((
                    "multiview_plan_request",
                    (
                        stack_generation,
                        attempt_generation,
                        False,
                        "",
                        str(exc),
                    ),
                ))
                return
            if result.accepted:
                self.events.put((
                    "multiview_plan_request",
                    (
                        stack_generation,
                        attempt_generation,
                        True,
                        str(result.request_id),
                        str(result.message),
                    ),
                ))
                return
            if result.request_id and 'already pending' in str(result.message):
                time.sleep(0.50)
                continue
            self.events.put((
                "multiview_plan_request",
                (
                    stack_generation,
                    attempt_generation,
                    False,
                    str(result.request_id),
                    str(result.message),
                ),
            ))
            return
        self.events.put((
            "multiview_plan_request",
            (
                stack_generation,
                attempt_generation,
                False,
                "",
                "timed out waiting to queue a fresh multiview plan",
            ),
        ))

    def cancel_scan(self) -> None:
        thread = threading.Thread(target=self._cancel_scan, daemon=True)
        thread.start()

    def _cancel_scan(self) -> None:
        if not self.scan_cancel_client.wait_for_service(timeout_sec=3.0):
            self.events.put(("scan_cancel", (False, "scan cancel service unavailable")))
            return
        future = self.scan_cancel_client.call_async(Trigger.Request())
        self._wait_for_future(
            future, 8.0, "scan_cancel",
            lambda result: (bool(result.success), str(result.message)),
        )

    def _wait_for_future(self, future, timeout_sec, event_name, formatter):
        deadline = time.monotonic() + float(timeout_sec)
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done():
            self.events.put((event_name, (False, "service call timed out")))
            return
        try:
            result = future.result()
            self.events.put((event_name, formatter(result)))
        except Exception as exc:
            self.events.put((event_name, (False, str(exc))))

    def submit_simulated_mission(self, coordinates, label='green cube') -> None:
        thread = threading.Thread(
            target=self._submit_simulated_mission,
            args=(tuple(coordinates), str(label)), daemon=True)
        thread.start()

    def _submit_simulated_mission(self, coordinates, label):
        task_id = 'gui-sim-' + uuid.uuid4().hex
        self.mission_task_id = task_id
        if not self.mission_action_client.wait_for_server(timeout_sec=5.0):
            if self.mission_task_id == task_id:
                self.events.put((
                    'mission_state',
                    ('IDLE', 'autonomous mission action server is unavailable; '
                     'start run_target_scan_mission.sh first')))
            return
        goal = RunTargetScan.Goal()
        goal.task_id = task_id
        goal.task_type = 'SCAN_3D'
        goal.target_label = label.strip() or 'green cube'
        goal.target_profile = ''
        goal.target_confidence = 1.0
        goal.deadline_sec = 1200.0
        goal.rough_target = PoseWithCovarianceStamped()
        goal.rough_target.header.stamp = self.get_clock().now().to_msg()
        goal.rough_target.header.frame_id = 'base_link'
        goal.rough_target.pose.pose.position.x = float(coordinates[0])
        goal.rough_target.pose.pose.position.y = float(coordinates[1])
        goal.rough_target.pose.pose.position.z = float(coordinates[2])
        goal.rough_target.pose.pose.orientation.w = 1.0
        covariance = [0.0] * 36
        covariance[0] = covariance[7] = covariance[14] = 0.01
        goal.rough_target.pose.covariance = covariance
        future = self.mission_action_client.send_goal_async(
            goal, feedback_callback=lambda message, bound=task_id:
            self._mission_feedback(bound, message.feedback))
        future.add_done_callback(
            lambda completed, bound=task_id:
            self._mission_goal_response(completed, bound))

    def _mission_feedback(self, task_id, feedback):
        if self.mission_task_id == task_id:
            self.events.put(('mission_feedback', feedback))

    def _mission_goal_response(self, future, task_id):
        if self.mission_task_id != task_id:
            return
        try:
            handle = future.result()
        except Exception as exc:
            self.events.put((
                'mission_state',
                ('IDLE', 'mission goal failed: %s' % exc)))
            return
        if handle is None or not handle.accepted:
            self.events.put((
                'mission_state', ('IDLE', 'mission goal was rejected')))
            return
        self.mission_goal_handle = handle
        self.events.put((
            'mission_state',
            ('ACTIVE', 'mission accepted; automatic startup is beginning')))
        handle.get_result_async().add_done_callback(
            lambda completed, bound=task_id:
            self._mission_result(completed, bound))

    def _mission_result(self, future, task_id):
        if self.mission_task_id != task_id:
            return
        try:
            wrapped = future.result()
            result = wrapped.result
            outcomes = {
                RunTargetScan.Goal.OUTCOME_SUCCEEDED: 'SUCCEEDED',
                RunTargetScan.Goal.OUTCOME_FAILED: 'FAILED',
                RunTargetScan.Goal.OUTCOME_CANCELLED: 'CANCELLED',
                RunTargetScan.Goal.OUTCOME_BUSY: 'BUSY',
                RunTargetScan.Goal.OUTCOME_UNSUPPORTED_TARGET_PROFILE:
                    'UNSUPPORTED_TARGET_PROFILE',
                RunTargetScan.Goal.OUTCOME_NEEDS_OPERATOR: 'NEEDS_OPERATOR',
                RunTargetScan.Goal.OUTCOME_REPOSITION_REQUIRED:
                    'REPOSITION_REQUIRED',
            }
            outcome = outcomes.get(result.outcome, 'UNKNOWN')
            message = ('%s: %s; code=%s; retryable=%s; safe shutdown=%s; '
                       'captures=%d; dataset=%s') % (
                outcome, result.reason,
                result.failure_code or 'none',
                'yes' if result.retryable else 'no',
                'proved' if result.safe_shutdown else 'not proved',
                result.capture_count,
                result.dataset_path or 'unavailable')
            result_payload = {
                'task_id': task_id,
                'outcome': outcome,
                'safe_shutdown': bool(result.safe_shutdown),
                'dataset_path': str(result.dataset_path),
                'manifest_sha256': str(result.manifest_sha256),
                'mesh_job_id': str(result.mesh_job_id),
            }
        except Exception as exc:
            message = 'mission result failed: %s' % exc
            result_payload = None
        if self.mission_task_id != task_id:
            return
        self.mission_goal_handle = None
        self.mission_task_id = ''
        self.events.put(('mission_state', ('IDLE', message)))
        if result_payload is not None:
            self.events.put(('mission_result', result_payload))

    def report_tracked_robot_homed(self, result_payload):
        threading.Thread(
            target=self._report_tracked_robot_homed,
            args=(dict(result_payload),), daemon=True).start()

    def _report_tracked_robot_homed(self, payload):
        if not self.report_base_home_client.wait_for_service(timeout_sec=5.0):
            self.events.put((
                'mesh_job_request',
                (False, 'tracked-robot home service is unavailable')))
            return
        request = ReportTrackedRobotHomed.Request()
        request.task_id = str(payload.get('task_id', ''))
        request.mesh_job_id = str(payload.get('mesh_job_id', ''))
        request.manifest_sha256 = str(payload.get('manifest_sha256', ''))
        request.homed_at = self.get_clock().now().to_msg()
        future = self.report_base_home_client.call_async(request)
        self._wait_for_future(
            future, 10.0, 'mesh_job_request',
            lambda result: (
                bool(result.accepted),
                '%s: %s' % (result.state, result.message)))

    def destroy_node(self):
        client = getattr(self, 'mission_action_client', None)
        if client is not None:
            try:
                client.destroy()
            except Exception:
                pass
            self.mission_action_client = None
        return super().destroy_node()

    def cancel_simulated_mission(self):
        handle = self.mission_goal_handle
        if handle is None:
            self.events.put((
                'mission_state', ('IDLE', 'no GUI mission is active')))
            return
        handle.cancel_goal_async()
        self.events.put((
            'mission_state',
            ('CANCELLING', 'mission cancellation requested; holding current '
             'position, returning to configured home, disabling, and stopping')))

    def call_enable_async(self, enabled: bool) -> None:
        thread = threading.Thread(target=self._call_enable, args=(enabled,), daemon=True)
        thread.start()

    def _call_enable(self, enabled: bool) -> None:
        if not self.enable_client.wait_for_service(timeout_sec=1.0):
            self.events.put((
                "enable_service",
                (enabled, False, "/enable_srv unavailable"),
            ))
            return

        req = Enable.Request()
        req.enable_request = enabled
        future = self.enable_client.call_async(req)
        deadline = time.time() + 20.0

        while rclpy.ok() and not future.done() and time.time() < deadline:
            time.sleep(0.05)

        if not future.done():
            self.events.put((
                "enable_service",
                (
                    enabled,
                    False,
                    f"{'enable' if enabled else 'disable'} timeout",
                ),
            ))
            return

        try:
            result = future.result()
            success = bool(result.enable_response)
            message = (
                f"{'enable' if enabled else 'disable'} -> "
                f"{result.enable_response}"
            )
        except Exception as exc:
            success = False
            message = (
                f"{'enable' if enabled else 'disable'} service failed: {exc}"
            )
        self.events.put((
            "enable_service",
            (enabled, success, message),
        ))


class PiperGuiApp:
    def __init__(self, root: tk.Tk, ros_node: PiperGuiRos, events: "queue.Queue[tuple]") -> None:
        self.root = root
        self.ros_node = ros_node
        self.events = events
        self.vars: List[tk.DoubleVar] = []
        self.feedback_positions = None
        self.joints, self.bounds_message = load_joint_limits()

        self.root.title("PiPER Control")
        self.root.geometry("1320x780")
        self.root.minsize(1120, 680)

        self.speed_var = tk.DoubleVar(value=30.0)
        self.effort_var = tk.DoubleVar(value=1.0)
        self.send_live_var = tk.BooleanVar(value=False)
        self.last_live_publish = 0.0
        self.kinematics = PiperScanKinematics(np.eye(4))
        gui_limits = np.asarray([[joint[1], joint[2]] for joint in self.joints[:6]], dtype=float)
        self.preview_limits = np.column_stack([
            np.maximum(gui_limits[:, 0], URDF_JOINT_LIMITS[:, 0]),
            np.minimum(gui_limits[:, 1], URDF_JOINT_LIMITS[:, 1]),
        ])
        self.preview_speed_var = tk.DoubleVar(value=5.0)
        self.preview_status_var = tk.StringVar(
            value="Open the 3D editor, then load the live arm into the preview."
        )
        self.preview_angle_var = tk.StringVar(value="No 3D preview joint state")
        self.preview_positions = None
        self.preview_process = None
        self.manual_motion_widgets = []
        self.automation_processes = {}
        self.automation_stopping = False
        self.automation_starting = False
        self.automation_failure_handled = False
        self.automation_stack_started_at = 0.0
        self.automation_generation = 0
        self.automation_start_mode = "acquisition"
        self.rough_coordinate_vars = [
            tk.StringVar(value=""),
            tk.StringVar(value=""),
            tk.StringVar(value=""),
        ]
        self.automation_speed_var = tk.DoubleVar(value=5.0)
        self.automation_speed_percent = 5.0
        self.automation_status_var = tk.StringVar(
            value="Driver and camera/perception must be started separately.")
        self.automation_plan_var = tk.StringVar(value="No acquisition plan")
        self.grounding_status_var = tk.StringVar(value="GroundingDINO idle")
        self.automation_tracking_var = tk.StringVar(value="Tracking unavailable")
        self.automation_workflow_var = tk.StringVar(value="Workflow unavailable")
        self.automation_capture_var = tk.StringVar(value="RGB-D capture unavailable")
        self.mission_label_var = tk.StringVar(value="green cube")
        self.mission_status_var = tk.StringVar(value="Autonomous mission idle")
        self.mesh_status_var = tk.StringVar(
            value="Mesh reconstruction is waiting for a completed scan")
        self.last_successful_mission = None
        self.home_status_var = tk.StringVar(
            value="Home: validated compact default")
        self.mission_in_progress = False
        self.load_selected_home()
        self.latest_scan_plan = None
        self.latest_scan_status = None
        self.latest_workflow_status = {}
        self.latest_tracking_health = None
        self.latest_tracking_health_received_at = None
        self.automation_session = AutomationSession()
        self.pending_rough_coordinates = None
        self.pending_acquisition_session_id = ""
        self.pending_acquisition_stack_generation = -1
        self.acquisition_attempt_generation = 0
        self.acquisition_phase = AcquisitionPhase.IDLE
        self.acquisition_plan_deadline = 0.0
        self.pending_early_acquisition_plan = None
        self.awaiting_acquisition_plan = False
        self.step4_attempt_generation = 0
        self.step4_stack_generation = -1
        self.step4_phase = Step4Phase.IDLE
        self.step4_workflow_deadline = 0.0
        self.step4_plan_deadline = 0.0
        self.step4_workflow_started = False
        self.step4_diagnostic_inflight = False
        self.step4_next_diagnostic_at = 0.0
        self.step4_plan_request_sent = False
        self.awaiting_scan_plan = False
        self.expected_scan_request_id = ""
        self.preparing_scan_from_current_lock = False
        self.prepare_scan_after_stack_start = False
        self.acquisition_approval_inflight = False
        self.scan_approval_inflight = False
        # Step 2 is valid only after this GUI instance receives a successful
        # enable acknowledgement.  A pose sampled while disabled is not the
        # same loaded start pose the controller will execute from.
        self.arm_enable_confirmed = False
        self.safe_disable_in_progress = False
        self.safe_disable_waiting_for_hold = False
        self.safe_disable_service_inflight = False
        self.safe_disable_target = None
        self.safe_disable_previous_feedback = None
        self.safe_disable_settled_since = None
        self.safe_disable_deadline = 0.0
        self.cancel_home_shutdown_pending = False
        self.cancel_home_shutdown_report_pending = False
        self.cancel_home_retry_count = 0
        self.pending_scan_cancel_status = ""
        self.step45_auto_recovery_attempts = 0
        self.step45_auto_recovery_pending = False
        self.step45_auto_recovery_deadline = 0.0
        self.step45_auto_recovery_reason = ""
        self.step45_auto_recovery_after_cancel = ""
        self.step45_auto_recovery_token = 0

        self.status_text = tk.StringVar(
            value=f"ROS domain {os.environ.get('ROS_DOMAIN_ID', 'default')} | {self.bounds_message}"
        )
        self.feedback_text = tk.StringVar(value="No feedback")
        self.command_text = tk.StringVar(value="No command sent")
        self.service_text = tk.StringVar(value="No service call")

        self._build()
        self.root.after(100, self.drain_events)
        self.root.after(250, self.poll_automation)

    def _build(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=(14, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        title = ttk.Label(header, text="PiPER Control", font=("TkDefaultFont", 16, "bold"))
        title.grid(row=0, column=0, sticky="w")
        ttk.Button(header, text="Enable", command=lambda: self.ros_node.call_enable_async(True)).grid(row=0, column=1, padx=4)
        self.disable_button = ttk.Button(
            header, text="Disable", command=self.request_safe_disable)
        self.disable_button.grid(row=0, column=2, padx=4)

        body = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        body.grid(row=1, column=0, sticky="nsew")

        notebook = ttk.Notebook(body)
        body.add(notebook, weight=4)

        automatic = ttk.Frame(notebook, padding=14)
        notebook.add(automatic, text="Automatic Scan")
        self._build_automatic_scan(automatic)

        manual = ttk.Frame(notebook, padding=14)
        notebook.add(manual, text="Manual")
        self._build_manual(manual)

        graphical = ttk.Frame(notebook, padding=14)
        notebook.add(graphical, text="Graphical")
        self._build_graphical(graphical)

        acquisition = ttk.Frame(notebook, padding=14)
        notebook.add(acquisition, text="Acquire & Scan")
        self._build_acquisition(acquisition)

        side = ttk.Frame(body, padding=14)
        body.add(side, weight=1)
        self._build_status(side)

    def _build_automatic_scan(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        ttk.Label(
            parent,
            text="Complete automatic target scan",
            font=("TkDefaultFont", 16, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            parent,
            text=(
                "This tab mirrors the final tracked-robot workflow. Enter the "
                "rough target coordinate and label, then press one button. The mission "
                "owns driver/camera/perception startup, arm enable, rough search, "
                "target lock, adaptive 8-to-24-view Tesseract planning, synchronized capture, "
                "return home, current-position hold, disable, and shutdown."
            ),
            wraplength=820,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(8, 16))

        target = ttk.LabelFrame(
            parent, text="Rough target in base_link", padding=12)
        target.grid(row=2, column=0, sticky="ew")
        for column, (axis, variable) in enumerate(
                zip(("X (m)", "Y (m)", "Z (m)"), self.rough_coordinate_vars)):
            ttk.Label(target, text=axis).grid(
                row=0, column=column * 2,
                padx=(0 if column == 0 else 18, 5))
            ttk.Entry(target, textvariable=variable, width=14).grid(
                row=0, column=column * 2 + 1)
        ttk.Label(target, text="Target label").grid(
            row=1, column=0, padx=(0, 5), sticky="w", pady=(10, 0))
        ttk.Entry(
            target, textvariable=self.mission_label_var, width=28).grid(
                row=1, column=1, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Label(
            target,
            text=("Exact 'green cube' uses the calibrated profile; all other "
                  "labels use the open-vocabulary profile (minimum confidence 60%)."),
            foreground="#52606d",
            wraplength=480,
            justify="left",
        ).grid(row=1, column=3, columnspan=3, sticky="w", pady=(10, 0))
        ttk.Button(
            target,
            text="Record Rough / Ready Home",
            command=self.use_current_feedback_as_home,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Button(
            target,
            text="Record Current J6 as Storage",
            command=self.use_current_feedback_as_storage,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(
            target,
            textvariable=self.home_status_var,
            foreground="#52606d",
            wraplength=570,
            justify="left",
        ).grid(row=2, column=2, rowspan=2, columnspan=4, sticky="w", padx=(10, 0),
               pady=(10, 0))

        controls = ttk.Frame(parent)
        controls.grid(row=3, column=0, sticky="w", pady=(18, 10))
        self.mission_start_button = ttk.Button(
            controls,
            text="Start Complete Automated Scan",
            command=self.start_automated_scan,
        )
        self.mission_start_button.grid(row=0, column=0, padx=(0, 8))
        self.mission_cancel_button = ttk.Button(
            controls,
            text="Cancel and Home",
            command=self.ros_node.cancel_simulated_mission,
            state="disabled",
        )
        self.mission_cancel_button.grid(row=0, column=1)
        self.report_base_home_button = ttk.Button(
            controls,
            text="Tracked Robot Homed / Build Mesh",
            command=self.report_tracked_robot_homed,
            state="disabled",
        )
        self.report_base_home_button.grid(row=0, column=2, padx=(8, 0))

        status = ttk.LabelFrame(parent, text="Mission status", padding=12)
        status.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        status.columnconfigure(0, weight=1)
        ttk.Label(
            status,
            textvariable=self.mission_status_var,
            wraplength=790,
            justify="left",
        ).grid(row=0, column=0, sticky="ew")
        ttk.Label(
            status,
            textvariable=self.mesh_status_var,
            foreground="#52606d",
            wraplength=790,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(6, 0))

        ttk.Label(
            parent,
            text=(
                "Obstacle handling: perception and benefit assessment are active, "
                "but physical obstacle removal is NOT enabled yet. A hand is always "
                "a terminal blocker. A beneficial leaf/branch occlusion currently "
                "returns NEEDS_OPERATOR until the gripper, attached-object and "
                "contact-collision model are physically qualified."
            ),
            foreground="#8a3b12",
            wraplength=820,
            justify="left",
        ).grid(row=5, column=0, sticky="ew", pady=(18, 0))

        ttk.Label(
            parent,
            text=(
                "The command-free mission listener must already be running via "
                "run_target_scan_mission.sh, as it will be in the final system. "
                "Real motion and the 30%/10% speed profile remain controlled by "
                "that listener's deployment gates."
            ),
            foreground="#52606d",
            wraplength=820,
            justify="left",
        ).grid(row=6, column=0, sticky="ew", pady=(12, 0))

    def _build_manual(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        joints_frame = ttk.Frame(parent)
        joints_frame.grid(row=0, column=0, sticky="nsew")
        joints_frame.columnconfigure(1, weight=1)

        for index, (name, low, high, unit) in enumerate(self.joints):
            var = tk.DoubleVar(value=0.0)
            self.vars.append(var)

            ttk.Label(joints_frame, text=name, width=10).grid(row=index, column=0, sticky="w", pady=6)
            scale = ttk.Scale(
                joints_frame,
                from_=low,
                to=high,
                variable=var,
                command=lambda _value, i=index: self.on_joint_change(i),
            )
            scale.grid(row=index, column=1, sticky="ew", padx=8, pady=6)
            spin = ttk.Spinbox(
                joints_frame,
                from_=low,
                to=high,
                increment=0.001 if index == 6 else 0.01,
                textvariable=var,
                width=10,
                command=lambda i=index: self.on_joint_change(i),
            )
            spin.grid(row=index, column=2, sticky="e", pady=6)
            spin.bind("<Return>", lambda _event, i=index: self.on_joint_change(i))
            spin.bind("<FocusOut>", lambda _event, i=index: self.on_joint_change(i))
            ttk.Label(joints_frame, text=unit, width=4).grid(row=index, column=3, sticky="w", padx=(6, 0))

        settings = ttk.Frame(parent)
        settings.grid(row=1, column=0, sticky="ew", pady=(18, 8))
        ttk.Label(settings, text="Speed").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(settings, from_=0, to=100, increment=1, textvariable=self.speed_var, width=8).grid(row=0, column=1, padx=(6, 18))
        ttk.Label(settings, text="Grip effort").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(settings, from_=0.5, to=3.0, increment=0.1, textvariable=self.effort_var, width=8).grid(row=0, column=3, padx=(6, 18))
        live_send = ttk.Checkbutton(
            settings, text="Live send", variable=self.send_live_var)
        live_send.grid(row=0, column=4, sticky="w")
        self.manual_motion_widgets.append(live_send)

        actions = ttk.Frame(parent)
        actions.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        send_button = ttk.Button(
            actions, text="Send Joint Target", command=self.send_target)
        send_button.grid(row=0, column=0, padx=(0, 8))
        self.manual_motion_widgets.append(send_button)
        ttk.Button(actions, text="Use Feedback", command=self.use_feedback).grid(row=0, column=1, padx=8)
        ttk.Button(actions, text="Zero Target", command=self.zero_target).grid(row=0, column=2, padx=8)

    def _build_graphical(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        ttk.Label(
            parent,
            text="Draggable 3D PiPER digital twin",
            font=("TkDefaultFont", 14, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        explanation = ttk.Frame(parent, padding=14, relief="solid")
        explanation.grid(row=1, column=0, sticky="nsew")
        explanation.columnconfigure(0, weight=1)
        ttk.Label(
            explanation,
            text=(
                "The 3D editor opens the real STL robot model in RViz. Orange rotation rings are "
                "attached to joint1 through joint6. Drag a ring to rotate that preview joint. "
                "Dragging changes only /piper_gui/preview_joint_states; it cannot move the arm."
            ),
            wraplength=700,
            justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 14))

        ttk.Button(
            explanation,
            text="1. Open 3D Joint Editor",
            command=self.open_3d_joint_editor,
        ).grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=4)
        ttk.Button(
            explanation,
            text="2. Load Live Arm into 3D Preview",
            command=self.load_live_joint_preview,
        ).grid(row=1, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(
            explanation,
            text="Reset Preview to Live",
            command=self.load_live_joint_preview,
        ).grid(row=1, column=2, sticky="ew", padx=(6, 0), pady=4)

        ttk.Label(
            explanation,
            text="Preview joint angles (rad)",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=2, column=0, sticky="w", pady=(18, 3))
        ttk.Label(
            explanation,
            textvariable=self.preview_angle_var,
            wraplength=760,
            justify="left",
        ).grid(row=3, column=0, columnspan=3, sticky="ew")

        move = ttk.Frame(explanation)
        move.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(22, 0))
        ttk.Label(move, text="Mirror speed").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(
            move,
            from_=1.0,
            to=10.0,
            increment=1.0,
            textvariable=self.preview_speed_var,
            width=8,
        ).grid(row=0, column=1, padx=(6, 4))
        ttk.Label(move, text="% (maximum 10%)").grid(row=0, column=2, sticky="w")
        mirror_button = ttk.Button(
            move,
            text="3. Confirm: Mirror 3D Preview on Real Arm",
            command=self.send_joint_preview,
        )
        mirror_button.grid(row=0, column=3, padx=(20, 0))
        self.manual_motion_widgets.append(mirror_button)

        ttk.Label(
            explanation,
            textvariable=self.preview_status_var,
            wraplength=760,
            justify="left",
        ).grid(row=5, column=0, columnspan=3, sticky="ew", pady=(18, 0))
        ttk.Label(
            explanation,
            text=(
                "The 3D model is a separate preview TF tree with preview_ frame names. "
                "Live-arm RViz remains feedback-only and cannot command the arm."
            ),
            foreground="#52606d",
            justify="left",
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(16, 0))

    def _build_acquisition(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        ttk.Label(
            parent,
            text="Rough-coordinate target acquisition and 13-view scan",
            font=("TkDefaultFont", 14, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            parent,
            text=(
                "Enter an approximate cube position in base_link metres. The GUI manages only "
                "the Tesseract/scan stack; start the driver and camera/perception separately. "
                "Automation disables manual GUI motion and never enables the motors."
            ),
            wraplength=820,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(6, 14))

        coordinates = ttk.LabelFrame(parent, text="Rough target in base_link", padding=10)
        coordinates.grid(row=2, column=0, sticky="ew")
        for column, (axis, variable) in enumerate(
                zip(("X (m)", "Y (m)", "Z (m)"), self.rough_coordinate_vars)):
            ttk.Label(coordinates, text=axis).grid(
                row=0, column=column * 2, padx=(0 if column == 0 else 16, 4))
            ttk.Entry(coordinates, textvariable=variable, width=12).grid(
                row=0, column=column * 2 + 1)
        ttk.Label(coordinates, text="Speed (%)").grid(
            row=0, column=6, padx=(16, 4))
        self.automation_speed_spinbox = ttk.Spinbox(
            coordinates,
            from_=1.0,
            to=100.0,
            increment=1.0,
            textvariable=self.automation_speed_var,
            width=8,
        )
        self.automation_speed_spinbox.grid(row=0, column=7)
        ttk.Label(
            coordinates, text="SDK range 1-100%",
            foreground="#52606d",
        ).grid(row=0, column=8, padx=(6, 0))

        actions = ttk.Frame(parent)
        actions.grid(row=3, column=0, sticky="ew", pady=(14, 10))
        self.start_automation_button = ttk.Button(
            actions,
            text="1. Start Tesseract Scan Stack",
            command=self.start_automation_stack,
        )
        self.start_automation_button.grid(row=0, column=0, padx=(0, 6))
        self.prepare_acquisition_button = ttk.Button(
            actions,
            text="2. Prepare Acquisition Plan",
            command=self.prepare_acquisition,
            state="disabled",
        )
        self.prepare_acquisition_button.grid(row=0, column=1, padx=6)
        self.confirm_acquisition_button = ttk.Button(
            actions,
            text="3. Confirm Acquisition & Search",
            command=self.confirm_acquisition,
            state="disabled",
        )
        self.confirm_acquisition_button.grid(row=0, column=2, padx=6)
        self.prepare_scan_button = ttk.Button(
            actions,
            text="4. Prepare Scan from Current Lock",
            command=self.prepare_scan_from_current_lock,
            state="disabled",
        )
        self.prepare_scan_button.grid(row=1, column=0, columnspan=2, padx=(0, 6), pady=(8, 0))
        self.confirm_scan_button = ttk.Button(
            actions,
            text="5. Confirm 13-View Scan",
            command=self.confirm_five_view_scan,
            state="disabled",
        )
        self.confirm_scan_button.grid(row=1, column=2, padx=6, pady=(8, 0))
        ttk.Button(
            actions, text="Cancel and Home", command=self.cancel_automation
        ).grid(row=0, column=3, padx=6)
        ttk.Button(
            actions, text="Stop Scan Stack", command=self.stop_automation_stack
        ).grid(row=1, column=3, padx=6, pady=(8, 0))

        status = ttk.LabelFrame(parent, text="Automation status", padding=10)
        status.grid(row=4, column=0, sticky="nsew", pady=(8, 0))
        status.columnconfigure(1, weight=1)
        for row, (name, variable) in enumerate((
                ("Session", self.automation_status_var),
                ("Plan", self.automation_plan_var),
                ("GroundingDINO", self.grounding_status_var),
                ("Tracking", self.automation_tracking_var),
                ("Workflow", self.automation_workflow_var),
                ("RGB-D capture", self.automation_capture_var),
        )):
            ttk.Label(status, text=name, font=("TkDefaultFont", 10, "bold")).grid(
                row=row, column=0, sticky="nw", padx=(0, 10), pady=4)
            ttk.Label(
                status, textvariable=variable, wraplength=720, justify="left"
            ).grid(row=row, column=1, sticky="ew", pady=4)

        ttk.Label(
            parent,
            text=(
                "Rough acquisition has its own exact plan/hash confirmation. A fresh measured "
                "lock may also be explicitly adopted after a terminal acquisition; this starts "
                "a new workflow/scan phase and does not reuse the acquisition approval. Prepare "
                "and separately confirm one fresh collision-qualified 13-view plan. If the "
                "managed scan stack is stopped, Step 4 starts it before workflow assessment."
            ),
            foreground="#8a3b12",
            wraplength=820,
            justify="left",
        ).grid(row=5, column=0, sticky="ew", pady=(12, 0))

    def start_automated_scan(self):
        if self.mission_in_progress:
            self.mission_status_var.set(
                'an automatic mission is already starting or active')
            return
        try:
            coordinates = validate_rough_coordinates(
                [variable.get() for variable in self.rough_coordinate_vars])
        except ValueError as exc:
            self.mission_status_var.set(str(exc))
            return
        # The automatic mission owns its scan stack and executor.  Invalidate
        # every delayed Step-2/4/5 callback before submitting it so a retry
        # timer or cached terminal message from the manual tab cannot cancel
        # a newly acquired target or mutate the mission state machine.
        self._advance_automation_generation()
        self.automation_session.finish('automatic mission started')
        self.automation_session = AutomationSession()
        self.cancel_home_shutdown_pending = False
        self.cancel_home_shutdown_report_pending = False
        self.cancel_home_retry_count = 0
        self.mission_in_progress = True
        self.last_successful_mission = None
        self.report_base_home_button.configure(state="disabled")
        self.ros_node.disable_manual_command_publisher()
        self.set_manual_motion_enabled(False)
        self.mission_start_button.configure(state="disabled")
        self.mission_cancel_button.configure(state="disabled")
        self.mission_status_var.set(
            'submitting complete task through /piper/run_target_scan')
        self.ros_node.submit_simulated_mission(
            coordinates, self.mission_label_var.get())

    def report_tracked_robot_homed(self):
        if not isinstance(self.last_successful_mission, dict):
            self.mesh_status_var.set(
                'No safely completed acquisition is waiting for reconstruction')
            return
        self.report_base_home_button.configure(state='disabled')
        self.mesh_status_var.set(
            'Reporting tracked-robot home and queueing reconstruction')
        self.ros_node.report_tracked_robot_homed(
            self.last_successful_mission)

    def load_selected_home(self):
        try:
            payload = load_home_pose(HOME_POSE_PATH)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.home_status_var.set('Home file invalid: %s' % exc)
            os.environ.pop('PIPER_RETURN_HOME_POSITIONS_RAD', None)
            return
        if payload is None:
            os.environ.pop('PIPER_RETURN_HOME_POSITIONS_RAD', None)
            return
        try:
            validate_home_profile_limits(payload, URDF_JOINT_LIMITS)
        except (TypeError, ValueError) as exc:
            self.home_status_var.set('Home file invalid: %s' % exc)
            os.environ.pop('PIPER_RETURN_HOME_POSITIONS_RAD', None)
            return
        positions = payload['positions_rad']
        os.environ['PIPER_RETURN_HOME_POSITIONS_RAD'] = json.dumps(positions)
        self.home_status_var.set(
            'Rough home J1-J6: '
            + ', '.join('%.6f' % value for value in positions)
            + '; ready J6 %.6f; storage J6 %.6f; staged=%s'
            % (
                float(payload['mission_ready_joint6_rad']),
                float(payload['storage_joint6_rad']),
                ('yes; startup increasing to zero'
                 if payload.get('staged_home_configured') else 'no')))

    def use_current_feedback_as_home(self):
        if self.mission_in_progress:
            self.home_status_var.set(
                'Home cannot change while an automatic mission is active')
            return
        feedback_age = (
            math.inf if self.ros_node.latest_feedback_monotonic is None
            else time.monotonic() - self.ros_node.latest_feedback_monotonic)
        if (
                self.feedback_positions is None
                or len(self.feedback_positions) < 6
                or feedback_age > 1.0):
            self.home_status_var.set(
                'Fresh six-joint feedback is required to set home')
            return
        observed_positions = np.asarray(
            self.feedback_positions[:6], dtype=float)
        if not np.all(np.isfinite(observed_positions)):
            self.home_status_var.set('Current feedback is not finite')
            return
        observed_limits = URDF_JOINT_LIMITS.copy()
        # Only the two gravity-loaded axes may cross their powered zero while
        # disabled. Keep this allowance small and validate every other joint
        # against the unchanged planning limits.
        observed_limits[1, 0] = min(
            observed_limits[1, 0],
            -DISABLED_HOME_DROOP_TOLERANCE_RAD,
        )
        observed_limits[2, 1] = max(
            observed_limits[2, 1],
            DISABLED_HOME_DROOP_TOLERANCE_RAD,
        )
        if np.any(observed_positions < observed_limits[:, 0]) or np.any(
                observed_positions > observed_limits[:, 1]):
            self.home_status_var.set(
                'Current feedback is outside the planning limits and bounded '
                'disabled J2/J3 droop allowance')
            return
        # A disabled PiPER can droop slightly below powered J2 zero and above
        # powered J3 zero. Persist the nearest controller-representable pose,
        # otherwise an automatic return would request angles the SDK clamps
        # away and the final home proof could never succeed.
        positions = energized_hold_target(observed_positions)
        if np.any(positions < URDF_JOINT_LIMITS[:, 0]) or np.any(
                positions > URDF_JOINT_LIMITS[:, 1]):
            self.home_status_var.set(
                'Powered home target is outside the planning joint limits')
            return
        try:
            existing = load_home_pose(HOME_POSE_PATH)
            if existing is not None:
                validate_home_profile_limits(existing, URDF_JOINT_LIMITS)
            existing_storage = (
                float(existing['storage_joint6_rad'])
                if existing is not None
                and existing.get('staged_home_configured') else None)
            save_home_pose(
                HOME_POSE_PATH,
                positions.tolist(),
                observed_positions=observed_positions.tolist(),
                mission_ready_joint6_rad=float(positions[5]),
                storage_joint6_rad=existing_storage,
                staged_home_configured=existing_storage is not None,
            )
        except (OSError, TypeError, ValueError) as exc:
            self.home_status_var.set('Could not save home: %s' % exc)
            return
        self.load_selected_home()
        self.home_status_var.set(
            self.home_status_var.get()
            + '; saved from current feedback '
            + ', '.join('%.3f' % value for value in observed_positions)
            + ' (disabled J2/J3 droop normalized for powered return)')

    def use_current_feedback_as_storage(self):
        if self.mission_in_progress:
            self.home_status_var.set(
                'Storage J6 cannot change while an automatic mission is active')
            return
        feedback_age = (
            math.inf if self.ros_node.latest_feedback_monotonic is None
            else time.monotonic() - self.ros_node.latest_feedback_monotonic)
        if (
                self.feedback_positions is None
                or len(self.feedback_positions) < 6
                or feedback_age > 1.0):
            self.home_status_var.set(
                'Fresh six-joint feedback is required to record storage J6')
            return
        storage_j6 = float(self.feedback_positions[5])
        if not math.isfinite(storage_j6):
            self.home_status_var.set('Current J6 feedback is not finite')
            return
        if (
                storage_j6 < float(URDF_JOINT_LIMITS[5, 0])
                or storage_j6 > float(URDF_JOINT_LIMITS[5, 1])):
            self.home_status_var.set(
                'Current storage J6 is outside planning limits')
            return
        try:
            profile = load_home_pose(HOME_POSE_PATH)
            if profile is None:
                raise ValueError(
                    'record the rough / mission-ready home first')
            save_home_pose(
                HOME_POSE_PATH,
                profile['positions_rad'],
                observed_positions=profile.get(
                    'observed_disabled_positions_rad'),
                mission_ready_joint6_rad=profile[
                    'mission_ready_joint6_rad'],
                storage_joint6_rad=storage_j6,
                staged_home_configured=True,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.home_status_var.set(
                'Could not save storage J6: %s' % exc)
            return
        self.load_selected_home()

    def _build_status(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        for row, (label, var) in enumerate(
            [
                ("ROS", self.status_text),
                ("Feedback", self.feedback_text),
                ("Last command", self.command_text),
                ("Service", self.service_text),
            ]
        ):
            ttk.Label(parent, text=label, font=("TkDefaultFont", 10, "bold")).grid(row=row * 2, column=0, sticky="w", pady=(0 if row == 0 else 14, 2))
            ttk.Label(parent, textvariable=var, wraplength=250, justify="left").grid(row=row * 2 + 1, column=0, sticky="ew")

    def set_manual_motion_enabled(self, enabled: bool) -> None:
        self.send_live_var.set(False)
        for widget in self.manual_motion_widgets:
            widget.configure(state="normal" if enabled else "disabled")

    def automation_stack_running(self) -> bool:
        stack = self.automation_processes.get("scan_stack")
        worker = self.automation_processes.get("tesseract_worker")
        return (
            stack is not None and stack.poll() is None
            and worker is not None and worker.poll() is None
        )

    def _tracking_health_rejection(self) -> str:
        return tracking_health_rejection(
            self.latest_tracking_health,
            received_at=self.latest_tracking_health_received_at,
            now=time.monotonic(),
        )

    def _tracking_lock_rejection(
            self, workflow, require_scan_ready=True) -> str:
        return tracking_lock_rejection(
            self.latest_tracking_health,
            workflow,
            require_scan_ready=require_scan_ready,
            received_at=self.latest_tracking_health_received_at,
            now=time.monotonic(),
        )

    def _advance_automation_generation(self) -> int:
        self.automation_generation += 1
        self.latest_workflow_status = {}
        self.automation_workflow_var.set("Workflow unavailable")
        self.ros_node.clear_generation_cache()
        self.pending_rough_coordinates = None
        self.pending_acquisition_session_id = ""
        self.pending_acquisition_stack_generation = -1
        self.acquisition_phase = AcquisitionPhase.IDLE
        self.acquisition_plan_deadline = 0.0
        self.pending_early_acquisition_plan = None
        self.awaiting_acquisition_plan = False
        self.step4_stack_generation = -1
        self.step4_phase = Step4Phase.IDLE
        self.step4_workflow_deadline = 0.0
        self.step4_plan_deadline = 0.0
        self.step4_workflow_started = False
        self.step4_diagnostic_inflight = False
        self.step4_next_diagnostic_at = 0.0
        self.step4_plan_request_sent = False
        self.awaiting_scan_plan = False
        self.expected_scan_request_id = ""
        self.preparing_scan_from_current_lock = False
        self.prepare_scan_after_stack_start = False
        self.latest_scan_plan = None
        self.latest_scan_status = None
        self.pending_scan_cancel_status = ""
        self.step45_auto_recovery_attempts = 0
        self.step45_auto_recovery_pending = False
        self.step45_auto_recovery_deadline = 0.0
        self.step45_auto_recovery_reason = ""
        self.step45_auto_recovery_after_cancel = ""
        self.step45_auto_recovery_token += 1
        self.automation_plan_var.set("No current plan")
        self.confirm_acquisition_button.configure(state="disabled")
        self.confirm_scan_button.configure(state="disabled")
        return self.automation_generation

    def _acquisition_fail(self, message, preserve_request=False) -> None:
        self.acquisition_phase = AcquisitionPhase.FAILED
        self.acquisition_plan_deadline = 0.0
        self.awaiting_acquisition_plan = False
        self.latest_scan_plan = None
        self.pending_early_acquisition_plan = None
        self.confirm_acquisition_button.configure(state="disabled")
        if not preserve_request:
            self.pending_rough_coordinates = None
            self.pending_acquisition_session_id = ""
            self.pending_acquisition_stack_generation = -1
        self.automation_status_var.set(str(message))

    def _step4_attempt_is_current(
            self, stack_generation, attempt_generation) -> bool:
        return (
            int(stack_generation) == self.automation_generation
            and int(stack_generation) == self.step4_stack_generation
            and int(attempt_generation) == self.step4_attempt_generation
        )

    def _step4_fail(self, message, allow_auto_recovery=True) -> None:
        self.step4_phase = Step4Phase.FAILED
        self.step4_workflow_deadline = 0.0
        self.step4_plan_deadline = 0.0
        self.step4_diagnostic_inflight = False
        self.step4_plan_request_sent = False
        self.awaiting_scan_plan = False
        self.expected_scan_request_id = ""
        self.latest_scan_plan = None
        self.automation_plan_var.set(
            "No active 13-view plan: " + str(message))
        self.confirm_scan_button.configure(state="disabled")
        self.automation_status_var.set("Step 4 failed: " + str(message))
        if allow_auto_recovery:
            self._schedule_step45_auto_recovery(str(message))

    def _schedule_step45_auto_recovery(self, message) -> None:
        if self.step45_auto_recovery_pending:
            return
        blocker = step45_auto_recovery_blocker(message)
        if blocker:
            self.automation_status_var.set(
                "Step 4/5 stopped for operator action: %s" % message)
            return
        if self.step45_auto_recovery_attempts >= (
                STEP45_AUTO_RECOVERY_MAX_ATTEMPTS):
            self.automation_status_var.set(
                "Step 4/5 automatic recovery exhausted after %d attempts: %s; "
                "correct the blocker, then click Step 4."
                % (STEP45_AUTO_RECOVERY_MAX_ATTEMPTS, message))
            return
        if self.automation_stopping:
            return
        self.step45_auto_recovery_attempts += 1
        self.step45_auto_recovery_pending = True
        self.step45_auto_recovery_deadline = (
            time.monotonic() + STEP45_AUTO_RECOVERY_LOCK_TIMEOUT_SEC)
        self.step45_auto_recovery_reason = str(message)
        self.step45_auto_recovery_token += 1
        token = self.step45_auto_recovery_token
        self.automation_status_var.set(
            "Step 4/5 automatic recovery %d/%d: waiting for a fresh measured "
            "lock before preparing a new plan. A new Step 5 approval is still "
            "required."
            % (
                self.step45_auto_recovery_attempts,
                STEP45_AUTO_RECOVERY_MAX_ATTEMPTS,
            ))
        self.root.after(
            int(STEP45_AUTO_RECOVERY_RETRY_SEC * 1000),
            lambda: self._run_step45_auto_recovery(token),
        )

    def _run_step45_auto_recovery(self, token) -> None:
        if (
                int(token) != self.step45_auto_recovery_token
                or not self.step45_auto_recovery_pending):
            return
        if self.automation_stopping or self.scan_approval_inflight:
            self.step45_auto_recovery_pending = False
            return
        if not self.automation_stack_running():
            self.step45_auto_recovery_pending = False
            self.automation_status_var.set(
                "Step 4/5 automatic recovery paused because the managed scan "
                "stack is not running; click Step 4 to restart it.")
            return
        workflow_state = str(
            self.latest_workflow_status.get("state", "")).upper()
        if workflow_state == "WAIT_CAPTURE":
            if time.monotonic() >= self.step45_auto_recovery_deadline:
                self.step45_auto_recovery_pending = False
                self.automation_status_var.set(
                    "Step 4/5 automatic recovery paused after %.0f seconds: "
                    "the previous view capture did not finish cleanup; Step 4 "
                    "is retryable after the workflow leaves WAIT_CAPTURE."
                    % STEP45_AUTO_RECOVERY_LOCK_TIMEOUT_SEC)
                self.update_automation_buttons()
                return
            self.automation_status_var.set(
                "Step 4/5 automatic recovery %d/%d is waiting for the "
                "previous view capture to finish cleanup. No motion approval "
                "will be reused."
                % (
                    self.step45_auto_recovery_attempts,
                    STEP45_AUTO_RECOVERY_MAX_ATTEMPTS,
                ))
            self.root.after(
                int(STEP45_AUTO_RECOVERY_RETRY_SEC * 1000),
                lambda: self._run_step45_auto_recovery(token),
            )
            return
        health_rejection = self._tracking_health_rejection()
        if health_rejection:
            if time.monotonic() >= self.step45_auto_recovery_deadline:
                self.step45_auto_recovery_pending = False
                self.automation_status_var.set(
                    "Step 4/5 automatic recovery paused after %.0f seconds: %s; "
                    "Step 4 is retryable when measured tracking is fresh."
                    % (
                        STEP45_AUTO_RECOVERY_LOCK_TIMEOUT_SEC,
                        health_rejection,
                    ))
                self.update_automation_buttons()
                return
            self.automation_status_var.set(
                "Step 4/5 automatic recovery %d/%d is waiting: %s. No motion "
                "approval will be reused."
                % (
                    self.step45_auto_recovery_attempts,
                    STEP45_AUTO_RECOVERY_MAX_ATTEMPTS,
                    health_rejection,
                ))
            self.root.after(
                int(STEP45_AUTO_RECOVERY_RETRY_SEC * 1000),
                lambda: self._run_step45_auto_recovery(token),
            )
            return
        self.step45_auto_recovery_pending = False
        self.step45_auto_recovery_deadline = 0.0
        if (
                self.automation_session.scan_plan_id
                and not self.automation_session.scan_approval_used):
            try:
                self.automation_session.discard_scan_plan()
            except ValueError:
                pass
        self.automation_status_var.set(
            "Measured lock recovered; automatically rerunning Step 4. "
            "The resulting exact plan will require a new Step 5 confirmation.")
        self.prepare_scan_from_current_lock(automatic=True)

    def _step4_request_diagnostic(self) -> None:
        if self.step4_phase not in (
                Step4Phase.CHECKING_WORKFLOW,
                Step4Phase.WAITING_SCAN_READY):
            return
        if self.step4_diagnostic_inflight:
            return
        self.step4_diagnostic_inflight = True
        self.ros_node.request_workflow_diagnostic(
            self.step4_stack_generation,
            self.step4_attempt_generation,
        )

    def _step4_begin_workflow_check(self) -> None:
        if not self.automation_stack_running():
            self._step4_fail("managed scan stack is not running")
            return
        if self.step4_stack_generation != self.automation_generation:
            self._step4_fail("managed stack generation changed")
            return
        self.step4_phase = Step4Phase.CHECKING_WORKFLOW
        self.step4_workflow_deadline = (
            time.monotonic() + WORKFLOW_ASSESSMENT_TIMEOUT_SEC)
        self.step4_workflow_started = False
        self.step4_next_diagnostic_at = 0.0
        self.automation_status_var.set(
            "Step 4 is checking the authoritative workflow diagnostic.")
        self._step4_request_diagnostic()

    def _step4_handle_workflow(self, workflow) -> None:
        self.latest_workflow_status = dict(workflow)
        self.automation_workflow_var.set(
            "%s: %s" % (
                workflow.get("state", "unknown"),
                workflow.get("reason", ""),
            ))
        action, message = step4_workflow_action(
            workflow, workflow_started=self.step4_workflow_started)
        if action == "fail":
            self._step4_fail(message)
            return
        if action == "start":
            self.step4_workflow_started = True
            self.step4_phase = Step4Phase.WAITING_SCAN_READY
            self.step4_next_diagnostic_at = time.monotonic() + 0.50
            self.automation_status_var.set(
                "Adopting the measured lock and starting workflow assessment.")
            self.ros_node.start_workflow_for_current_lock(
                self.step4_stack_generation,
                self.step4_attempt_generation,
            )
            return
        if action == "wait":
            self.step4_phase = Step4Phase.WAITING_SCAN_READY
            self.step4_next_diagnostic_at = time.monotonic() + 0.25
            self.automation_status_var.set(
                "Waiting for a fresh authoritative SCAN_READY diagnostic.")
            return
        self._step4_request_plan(workflow)

    def _step4_request_plan(self, workflow) -> None:
        lock_rejection = self._tracking_lock_rejection(
            workflow, require_scan_ready=True)
        if lock_rejection:
            self._step4_fail(lock_rejection)
            return
        readiness_rejection = self.ros_node.tesseract_readiness_rejection(
            "multiview")
        if readiness_rejection:
            self._step4_fail(readiness_rejection)
            return
        publishers = self.ros_node.command_publisher_names(
            resolution_timeout_sec=2.0)
        ownership_rejection = command_publisher_ownership_rejection(
            publishers,
            owned_stack_running=self.automation_stack_running(),
            executor_node_present=(
                "/scan_viewpoint_executor"
                in self.ros_node.automation_nodes_present()),
        )
        if ownership_rejection:
            self._step4_fail(ownership_rejection)
            return
        if self.step4_plan_request_sent:
            return
        accepted_views = int(workflow.get("accepted_views", 0) or 0)
        maximum_views = int(workflow.get("max_views", 13) or 13)
        remaining_views = maximum_views - accepted_views
        if remaining_views < 1:
            self._step4_fail(
                "workflow reports no remaining viewpoints to plan",
                allow_auto_recovery=False,
            )
            return
        try:
            self.automation_session.adopt_current_lock()
            self.automation_session.scan_expected_views = remaining_views
        except ValueError as exc:
            self._step4_fail(str(exc))
            return
        self.step4_plan_request_sent = True
        self.step4_phase = Step4Phase.REQUESTING_PLAN
        self.latest_scan_plan = None
        self.awaiting_scan_plan = True
        self.expected_scan_request_id = ""
        self.confirm_scan_button.configure(state="disabled")
        self.automation_plan_var.set(
            "Waiting for a fresh collision-qualified %d-view remainder plan."
            % remaining_views)
        self.automation_status_var.set(
            "Requesting exactly one correlated plan from the measured lock.")
        self.ros_node.request_multiview_plan(
            self.step4_stack_generation,
            self.step4_attempt_generation,
        )

    def _poll_attempt_deadlines(self) -> None:
        now = time.monotonic()
        if (
                self.acquisition_phase == AcquisitionPhase.WAITING_PLAN
                and self.acquisition_plan_deadline
                and now >= self.acquisition_plan_deadline):
            self._acquisition_fail(
                "Acquisition planning timed out after %.0f seconds; retry "
                "Step 2 with a fresh session."
                % ACQUISITION_PLAN_TIMEOUT_SEC,
                preserve_request=False,
            )
        if self.step4_phase in (
                Step4Phase.CHECKING_WORKFLOW,
                Step4Phase.WAITING_SCAN_READY):
            if (
                    self.step4_workflow_deadline
                    and now >= self.step4_workflow_deadline):
                state = str(
                    self.latest_workflow_status.get("state", "missing"))
                reason = str(
                    self.latest_workflow_status.get("reason", ""))
                self._step4_fail(
                    "workflow assessment timed out after %.0f seconds while "
                    "waiting for SCAN_READY (state=%s%s)"
                    % (
                        WORKFLOW_ASSESSMENT_TIMEOUT_SEC,
                        state,
                        ", reason=" + reason if reason else "",
                    ))
            elif (
                    self.step4_phase == Step4Phase.WAITING_SCAN_READY
                    and not self.step4_diagnostic_inflight
                    and now >= self.step4_next_diagnostic_at):
                self._step4_request_diagnostic()
        if (
                self.step4_phase == Step4Phase.WAITING_PLAN
                and self.step4_plan_deadline
                and now >= self.step4_plan_deadline):
            request_id = self.expected_scan_request_id or "unknown"
            self._step4_fail(
                "13-view plan did not arrive within %.0f seconds for "
                "request %s (bridge limit is 180 seconds)"
                % (MULTIVIEW_PLAN_TIMEOUT_SEC, request_id))

    def start_automation_stack(self, readiness_mode="acquisition") -> bool:
        if self.automation_starting:
            self.automation_status_var.set(
                "Managed Tesseract scan stack startup is already in progress.")
            return False
        if self.automation_stack_running():
            self.automation_status_var.set("Managed Tesseract scan stack is already running.")
            return True
        existing_nodes = self.ros_node.automation_nodes_present()
        if existing_nodes:
            self.automation_status_var.set(
                "Refusing to start over an external scan stack: " + ", ".join(existing_nodes))
            return False
        try:
            self.automation_speed_percent = validate_automation_speed(
                self.automation_speed_var.get())
        except ValueError as exc:
            self.automation_status_var.set(str(exc))
            return False
        self.automation_speed_spinbox.configure(state="disabled")
        self.start_automation_button.configure(state="disabled")
        self.prepare_acquisition_button.configure(state="disabled")
        self.automation_starting = True
        generation = self._advance_automation_generation()
        self.automation_start_mode = str(readiness_mode)
        self.automation_failure_handled = False
        self.ros_node.disable_manual_command_publisher()
        self.set_manual_motion_enabled(False)
        self.automation_status_var.set(
            "Manual command publisher removed; checking command ownership and starting stack.")
        thread = threading.Thread(
            target=self._start_automation_processes,
            args=(generation, self.automation_start_mode),
            daemon=True,
        )
        thread.start()
        return True

    def _start_automation_processes(
            self, generation, readiness_mode) -> None:
        deadline = time.monotonic() + 3.0
        publishers = self.ros_node.command_publisher_names()
        while publishers and time.monotonic() < deadline:
            time.sleep(0.1)
            publishers = self.ros_node.command_publisher_names()
        if publishers:
            self.events.put((
                "automation_start",
                (
                    generation,
                    False,
                    "existing joint command publishers: " + ", ".join(publishers),
                ),
            ))
            return
        environment = os.environ.copy()
        environment.update({
            "PIPER_ENABLE_REAL_VIEWPOINT_MOTION": "1",
            "PIPER_VIEWPOINT_SPEED_PERCENT": (
                "%.3f" % self.automation_speed_percent),
            "PIPER_VIEWPOINT_MAX_VIEWS": "13",
            "PIPER_VIEWPOINT_MIN_VIEWS": "13",
            "PIPER_VIEWPOINT_AUTO_CAPTURE": "1",
            "PIPER_ARM_ROOT": PROJECT_ROOT,
        })
        try:
            worker = subprocess.Popen(
                [os.path.join(
                    PROJECT_ROOT, "motion_planning/tesseract/run_worker.sh")],
                cwd=PROJECT_ROOT,
                env=environment,
                start_new_session=True,
            )
            self.automation_processes["tesseract_worker"] = worker
            time.sleep(1.0)
            if worker.poll() is not None:
                raise RuntimeError(
                    "Tesseract worker exited with code %s; another worker may own its lock"
                    % worker.returncode)
            stack = subprocess.Popen(
                [os.path.join(
                    PROJECT_ROOT,
                    "L515_camera/run_supervised_viewpoint_execution.sh")],
                cwd=PROJECT_ROOT,
                env=environment,
                start_new_session=True,
            )
            self.automation_processes["scan_stack"] = stack
            self.automation_stack_started_at = time.monotonic()
            ready_deadline = time.monotonic() + 25.0
            last_blockers = None
            while time.monotonic() < ready_deadline:
                if worker.poll() is not None:
                    raise RuntimeError(
                        "Tesseract worker exited during startup with code %s"
                        % worker.returncode)
                if stack.poll() is not None:
                    raise RuntimeError(
                        "supervised scan stack exited during startup with code %s"
                        % stack.returncode)
                blockers = []
                missing_services = self.ros_node.automation_services_ready()
                if missing_services:
                    blockers.append(
                        "waiting for services: " + ", ".join(missing_services))
                readiness_rejection = (
                    self.ros_node.tesseract_readiness_rejection(
                        readiness_mode))
                if readiness_rejection:
                    blockers.append(readiness_rejection)
                nodes = self.ros_node.automation_nodes_present()
                missing_nodes = sorted(
                    AUTOMATION_CRITICAL_NODES.difference(nodes))
                if missing_nodes:
                    blockers.append(
                        "waiting for nodes: " + ", ".join(missing_nodes))
                ownership_rejection = command_publisher_ownership_rejection(
                    self.ros_node.command_publisher_names(),
                    owned_stack_running=True,
                    executor_node_present=(
                        "/scan_viewpoint_executor" in nodes),
                )
                if ownership_rejection:
                    blockers.append(ownership_rejection)
                if not blockers:
                    break
                current = tuple(blockers)
                if current != last_blockers:
                    self.events.put((
                        "automation_start_progress",
                        (
                            generation,
                            "Starting scan stack: " + "; ".join(blockers),
                        ),
                    ))
                    last_blockers = current
                time.sleep(0.10)
            else:
                raise RuntimeError(
                    "scan stack readiness timed out: "
                    + "; ".join(last_blockers or ("unknown readiness failure",)))
            self.events.put((
                "automation_start",
                (
                    generation,
                    True,
                    "managed Tesseract worker and %.1f%% scan stack started"
                    % self.automation_speed_percent,
                ),
            ))
        except (OSError, RuntimeError) as exc:
            self._terminate_automation_processes()
            self.events.put((
                "automation_start", (generation, False, str(exc))))

    def prepare_acquisition(self) -> None:
        try:
            coordinates = validate_rough_coordinates(
                [variable.get() for variable in self.rough_coordinate_vars])
        except ValueError as exc:
            self.automation_status_var.set(str(exc))
            return
        if not self.arm_enable_confirmed:
            self.automation_status_var.set(
                "Enable the arm through this GUI before Step 2 so Tesseract "
                "plans from the measured loaded pose.")
            self.update_automation_buttons()
            return
        if not self.automation_stack_running():
            self.automation_status_var.set("Start the managed Tesseract scan stack first.")
            return
        readiness_rejection = self.ros_node.tesseract_readiness_rejection(
            "acquisition")
        if readiness_rejection:
            self.automation_status_var.set(
                "Cannot prepare acquisition: " + readiness_rejection)
            self.update_automation_buttons()
            return
        publishers = self.ros_node.command_publisher_names(
            resolution_timeout_sec=2.0)
        executor_node_present = (
            "/scan_viewpoint_executor"
            in self.ros_node.automation_nodes_present())
        ownership_rejection = command_publisher_ownership_rejection(
            publishers,
            owned_stack_running=self.automation_stack_running(),
            executor_node_present=executor_node_present,
        )
        if ownership_rejection:
            self.automation_status_var.set(ownership_rejection)
            return
        retrying = (
            self.acquisition_phase == AcquisitionPhase.FAILED
            and bool(self.pending_acquisition_session_id)
            and self.pending_acquisition_stack_generation
            == self.automation_generation
        )
        if retrying:
            if tuple(coordinates) != tuple(self.pending_rough_coordinates):
                self.automation_status_var.set(
                    "Step 2 retry is bound to the original coordinates %s; "
                    "use Cancel and Home before starting a changed target."
                    % (self.pending_rough_coordinates,))
                return
        else:
            self.pending_rough_coordinates = coordinates
            self.pending_acquisition_session_id = (
                "acq-" + uuid.uuid4().hex)
            self.pending_acquisition_stack_generation = (
                self.automation_generation)
            self.automation_session = AutomationSession()
        self.acquisition_attempt_generation += 1
        self.acquisition_phase = AcquisitionPhase.REQUESTING_PREPARE
        self.acquisition_plan_deadline = 0.0
        self.pending_early_acquisition_plan = None
        self.latest_scan_plan = None
        self.awaiting_acquisition_plan = True
        self.awaiting_scan_plan = False
        self.expected_scan_request_id = ""
        self.preparing_scan_from_current_lock = False
        self.prepare_scan_after_stack_start = False
        self.confirm_acquisition_button.configure(state="disabled")
        self.prepare_scan_button.configure(state="disabled")
        self.confirm_scan_button.configure(state="disabled")
        self.automation_plan_var.set("Waiting for a collision-qualified acquisition plan.")
        self.automation_status_var.set(
            (
                "Retrying the exact atomic rough-target session through fresh "
                "service endpoints."
                if retrying else
                "Submitting one atomic rough-target session through a fresh "
                "service endpoint."
            ))
        self.ros_node.publish_rough_target_and_start(
            self.pending_rough_coordinates,
            self.pending_acquisition_session_id,
            self.automation_generation,
            self.acquisition_attempt_generation,
        )

    def confirm_acquisition(self) -> None:
        try:
            self.automation_session.prepare_acquisition(
                self.pending_rough_coordinates,
                self.latest_scan_plan,
                self.pending_acquisition_session_id,
            )
        except ValueError as exc:
            self.automation_status_var.set("Cannot confirm acquisition: %s" % exc)
            return
        plan = self.latest_scan_plan
        coordinates = self.automation_session.rough_coordinates
        confirmed = messagebox.askyesno(
            "Confirm rough-coordinate acquisition",
            (
                "This authorizes only the displayed collision-qualified "
                "rough-acquisition plan.\n\n"
                "Rough target: X %.3f, Y %.3f, Z %.3f m in base_link\n"
                "Acquisition poses: %d\nPlan: %s\nHash: %s…\n"
                "Configured speed: %.1f%% (SDK range 1-100%%)\n\n"
                "The later 13-view scan requires a fresh plan and a separate "
                "confirmation after measured lock.\n\n"
                "The GUI will not enable the motors. Keep the workspace clear, "
                "camera cable free, and emergency stop ready."
            ) % (
                coordinates[0], coordinates[1], coordinates[2],
                int(plan.planned_viewpoints), plan.plan_id,
                plan.trajectory_sha256[:16],
                self.automation_speed_percent,
            ),
            parent=self.root,
        )
        if not confirmed:
            self.automation_session.finish("operator declined acquisition confirmation")
            self.automation_status_var.set(
                "Acquisition confirmation cancelled; no motion approval sent.")
            return
        self.automation_session.confirm_acquisition()
        self.acquisition_approval_inflight = True
        self.confirm_acquisition_button.configure(state="disabled")
        self.automation_status_var.set(
            "Submitting the exact acquisition plan and execution hash.")
        self.ros_node.approve_scan_plan(
            plan.plan_id, plan.trajectory_sha256, "acquisition_approval")

    def prepare_scan_from_current_lock(self, automatic=False) -> None:
        if not automatic:
            self.step45_auto_recovery_pending = False
            self.step45_auto_recovery_deadline = 0.0
            self.step45_auto_recovery_reason = ""
            self.step45_auto_recovery_attempts = 0
            self.step45_auto_recovery_token += 1
        health_rejection = self._tracking_health_rejection()
        if health_rejection:
            self.automation_status_var.set(
                "Cannot prepare 13-view scan: " + health_rejection)
            self.update_automation_buttons()
            return
        if self.automation_session.scan_approval_used:
            self.automation_status_var.set(
                "Cannot prepare 13-view scan: the one scan approval was already used.")
            return
        if self.automation_session.scan_plan_id:
            self.automation_status_var.set(
                "Cannot prepare 13-view scan: a scan plan is already prepared.")
            return
        self.step4_attempt_generation += 1
        attempt_generation = self.step4_attempt_generation
        self.step4_workflow_started = False
        self.step4_diagnostic_inflight = False
        self.step4_plan_request_sent = False
        self.step4_plan_deadline = 0.0
        self.expected_scan_request_id = ""
        self.latest_scan_plan = None
        if not self.automation_stack_running():
            self.automation_status_var.set(
                "Fresh measured tracking is available; starting the managed "
                "Tesseract scan stack before workflow assessment.")
            if not self.start_automation_stack(readiness_mode="multiview"):
                self._step4_fail("managed scan stack could not be started")
                self.update_automation_buttons()
                return
            self.step4_attempt_generation = attempt_generation
            self.step4_stack_generation = self.automation_generation
            self.step4_phase = Step4Phase.STARTING_STACK
            self.update_automation_buttons()
            return
        self.step4_stack_generation = self.automation_generation
        self.step4_phase = Step4Phase.CHECKING_WORKFLOW
        self.prepare_scan_button.configure(state="disabled")
        self._step4_begin_workflow_check()

    def _request_scan_after_workflow_ready(self) -> None:
        # Kept as a narrow compatibility entrypoint for older UI callbacks.
        if self.step4_phase not in (
                Step4Phase.CHECKING_WORKFLOW,
                Step4Phase.WAITING_SCAN_READY):
            return
        self._step4_request_diagnostic()

    def confirm_five_view_scan(self) -> None:
        plan = self.latest_scan_plan
        if plan is None:
            self.automation_status_var.set(
                "Cannot confirm 13-view scan: no prepared scan plan.")
            return
        confirmed = messagebox.askyesno(
            "Confirm 13-view scan",
            (
                "This authorizes only the displayed collision-qualified 13-view "
                "scan plan.\n\n"
                "Target: X %.3f, Y %.3f, Z %.3f m in base_link\n"
                "Views: %d\nPlan: %s\nExecution hash: %s…\n"
                "Configured speed: %.1f%% (SDK range 1-100%%)\n\n"
                "After all captures, the approved plan returns to its scan-start "
                "pose. After a non-safety abort at a reached viewpoint, the arm "
                "may retrace only already executed approved targets; any stale "
                "telemetry, obstacle, collision, hardware, or motion fault holds "
                "the current pose instead.\n\n"
                "The GUI will not enable the motors. Keep the workspace clear, "
                "camera cable free, and emergency stop ready."
            ) % (
                float(plan.target_center.x),
                float(plan.target_center.y),
                float(plan.target_center.z),
                int(plan.planned_viewpoints),
                plan.plan_id,
                plan.trajectory_sha256[:16],
                self.automation_speed_percent,
            ),
            parent=self.root,
        )
        if not confirmed:
            self.automation_status_var.set(
                "13-view confirmation cancelled; no scan approval sent.")
            return
        current_plan = self.latest_scan_plan
        if (
                self.step4_phase != Step4Phase.PLAN_READY
                or current_plan is None
                or str(current_plan.plan_id) != str(plan.plan_id)
                or str(current_plan.trajectory_sha256)
                != str(plan.trajectory_sha256)):
            if self.step45_auto_recovery_pending:
                detail = (
                    "automatic recovery is waiting for a fresh measured lock "
                    "and will prepare a replacement")
            else:
                detail = "click Step 4 to prepare a replacement"
            self.automation_status_var.set(
                "Cannot confirm 13-view scan: the exact displayed proposal "
                "was invalidated while the confirmation dialog was open; %s. "
                "No approval was sent." % detail)
            return
        try:
            self.automation_session.confirm_scan(plan)
        except ValueError as exc:
            self.automation_status_var.set(
                "Cannot confirm 13-view scan: %s" % exc)
            return
        self.scan_approval_inflight = True
        self.confirm_scan_button.configure(state="disabled")
        self.automation_status_var.set(
            "Submitting the exact 13-view execution plan and hash.")
        self.ros_node.approve_scan_plan(
            plan.plan_id, plan.trajectory_sha256, "scan_approval")

    def cancel_automation(self) -> None:
        self.automation_session.finish("operator cancelled")
        self.pending_rough_coordinates = None
        self.pending_acquisition_session_id = ""
        self.pending_acquisition_stack_generation = -1
        self.acquisition_phase = AcquisitionPhase.IDLE
        self.acquisition_plan_deadline = 0.0
        self.awaiting_acquisition_plan = False
        self.step4_phase = Step4Phase.FAILED
        self.step4_workflow_deadline = 0.0
        self.step4_plan_deadline = 0.0
        self.step4_diagnostic_inflight = False
        self.step4_plan_request_sent = False
        self.awaiting_scan_plan = False
        self.expected_scan_request_id = ""
        self.preparing_scan_from_current_lock = False
        self.prepare_scan_after_stack_start = False
        self.acquisition_approval_inflight = False
        self.scan_approval_inflight = False
        self.step45_auto_recovery_pending = False
        self.step45_auto_recovery_after_cancel = ""
        self.step45_auto_recovery_token += 1
        self.confirm_acquisition_button.configure(state="disabled")
        self.prepare_scan_button.configure(state="disabled")
        self.confirm_scan_button.configure(state="disabled")
        self.cancel_home_shutdown_pending = True
        self.cancel_home_shutdown_report_pending = False
        self.cancel_home_retry_count = 0
        self.automation_status_var.set(
            "Cancellation requested; holding current position, returning "
            "along the approved path to configured home, then disabling and "
            "stopping the scan stack.")
        self.ros_node.cancel_scan()

    def _retry_cancel_home_after_hold(self) -> None:
        if not self.cancel_home_shutdown_pending:
            return
        self.automation_status_var.set(
            "Current-position hold settled; retrying the bounded approved-path "
            "return to configured home (%d/3)." % self.cancel_home_retry_count)
        self.ros_node.cancel_scan()

    def stop_automation_stack(self) -> None:
        if self.automation_stopping:
            return
        self.automation_stopping = True
        self._advance_automation_generation()
        self.automation_session.finish("managed stack stopped")
        self.confirm_acquisition_button.configure(state="disabled")
        self.prepare_scan_button.configure(state="disabled")
        self.confirm_scan_button.configure(state="disabled")
        self.automation_status_var.set(
            "Cancelling motion before stopping GUI-owned scan processes.")
        self.ros_node.cancel_scan()
        self.root.after(1500, self._finish_stop_automation_stack)

    def _finish_stop_automation_stack(self) -> None:
        self._terminate_automation_processes()
        self.automation_stopping = False
        self.automation_starting = False
        self.automation_failure_handled = False
        self.start_automation_button.configure(state="normal")
        self.prepare_acquisition_button.configure(state="disabled")
        publishers = self.ros_node.command_publisher_names()
        if publishers:
            if self.cancel_home_shutdown_report_pending:
                self.cancel_home_shutdown_pending = False
                self.cancel_home_shutdown_report_pending = False
            self.automation_status_var.set(
                "Managed stack stopped, but command publishers remain; manual controls stay locked: "
                + ", ".join(publishers))
            return
        self.ros_node.enable_manual_command_publisher()
        self.set_manual_motion_enabled(True)
        self.automation_speed_spinbox.configure(state="normal")
        if self.cancel_home_shutdown_report_pending:
            self.cancel_home_shutdown_pending = False
            self.cancel_home_shutdown_report_pending = False
            self.automation_status_var.set(
                "Task failed: cancelled; arm returned to configured home, "
                "disabled, and scan stack stopped. Please retry.")
        else:
            self.automation_status_var.set(
                "Managed scan stack stopped; manual GUI command ownership restored.")

    def _terminate_automation_processes(self) -> None:
        for name in ("scan_stack", "tesseract_worker"):
            process = self.automation_processes.get(name)
            if process is None or process.poll() is not None:
                continue
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
                process.wait(timeout=5.0)
            except OSError:
                pass
            except subprocess.TimeoutExpired:
                if process.poll() is not None:
                    continue
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait(timeout=3.0)
                except OSError:
                    pass
                except subprocess.TimeoutExpired:
                    if process.poll() is None:
                        try:
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                            process.wait(timeout=2.0)
                        except (OSError, subprocess.TimeoutExpired):
                            pass
        self.automation_processes.clear()

    def handle_scan_plan(self, plan) -> None:
        if self.mission_in_progress:
            # The command-free mission listener performs its own full request
            # correlation.  This callback belongs only to the manual
            # Acquire & Scan tab and must never clean up a mission proposal.
            return
        if (
                plan.plan_kind == ROUGH_ACQUISITION
                and self.acquisition_phase
                in (AcquisitionPhase.REQUESTING_PREPARE,
                    AcquisitionPhase.WAITING_PLAN)
                and str(getattr(plan, "source_request_id", ""))
                != self.pending_acquisition_session_id):
            self.automation_status_var.set(
                "Ignoring an acquisition plan from a different rough-target "
                "session.")
            return
        if (
                plan.plan_kind == ROUGH_ACQUISITION
                and self.acquisition_phase != AcquisitionPhase.WAITING_PLAN):
            if (
                    self.acquisition_phase
                    == AcquisitionPhase.REQUESTING_PREPARE
                    and str(getattr(plan, "source_request_id", ""))
                    == self.pending_acquisition_session_id):
                self.pending_early_acquisition_plan = plan
            return
        if (
                plan.plan_kind == MULTIVIEW_SCAN
                and self.step4_phase not in (
                    Step4Phase.REQUESTING_PLAN,
                    Step4Phase.WAITING_PLAN)):
            return
        self.latest_scan_plan = plan
        summary = (
            "%s | valid=%s | views=%d | target=(%.3f, %.3f, %.3f) | "
            "plan=%s | hash=%s… | %s"
            % (
                plan.plan_kind, bool(plan.valid), int(plan.planned_viewpoints),
                float(plan.target_center.x), float(plan.target_center.y),
                float(plan.target_center.z),
                plan.plan_id or "none", plan.trajectory_sha256[:12],
                plan.reason,
            ))
        self.automation_plan_var.set(summary)
        if plan.plan_kind == ROUGH_ACQUISITION:
            if self.acquisition_phase != AcquisitionPhase.WAITING_PLAN:
                if (
                        self.automation_session.acquisition_confirmed
                        and not self.automation_session.ended
                        and plan.plan_id
                        != self.automation_session.acquisition_plan_id):
                    self.automation_session.finish(
                        "acquisition plan changed during the active session")
                    self.automation_status_var.set(
                        "Acquisition replan ended the session; hold requested.")
                    self.ros_node.cancel_scan()
                return
            rejection = plan_rejection(
                plan,
                ROUGH_ACQUISITION,
                expected_source_request_id=(
                    self.pending_acquisition_session_id),
            )
            if rejection:
                self._acquisition_fail(
                    "Acquisition plan rejected by GUI: " + rejection,
                    preserve_request=False,
                )
                return
            if self.pending_rough_coordinates is None:
                self.automation_status_var.set(
                    "Ignoring acquisition plan because no GUI hint is pending.")
                return
            try:
                coordinates = validate_rough_coordinates(
                    self.pending_rough_coordinates)
                planned = (
                    float(plan.target_center.x),
                    float(plan.target_center.y),
                    float(plan.target_center.z),
                )
                if np.linalg.norm(
                        np.asarray(planned) - np.asarray(coordinates)) > 0.001:
                    raise ValueError(
                        "plan target does not match the current rough coordinates")
            except (AttributeError, TypeError, ValueError) as exc:
                self._acquisition_fail(
                    "Acquisition plan rejected by GUI: %s" % exc,
                    preserve_request=False,
                )
                return
            self.awaiting_acquisition_plan = False
            self.acquisition_phase = AcquisitionPhase.PLAN_READY
            self.acquisition_plan_deadline = 0.0
            self.confirm_acquisition_button.configure(state="normal")
            self.automation_status_var.set(
                "Acquisition plan ready. The arm must already be enabled so "
                "this plan starts from the measured loaded pose. Inspect it, "
                "then confirm this exact acquisition plan.")
            return
        if (
                plan.plan_kind != MULTIVIEW_SCAN
                or self.step4_phase == Step4Phase.REQUESTING_PLAN):
            return
        if not self.expected_scan_request_id:
            self.automation_status_var.set(
                "13-view result arrived; waiting for exact request correlation.")
            return
        if not plan_matches_request(plan, self.expected_scan_request_id):
            self.automation_status_var.set(
                "Ignoring a 13-view plan from a different request.")
            return
        if not plan.valid:
            self._step4_fail(
                "13-view planning failed: " + str(plan.reason))
            return
        rejection = self.automation_session.scan_plan_rejection(plan)
        if rejection:
            self._step4_fail(
                "13-view plan rejected by GUI: " + rejection)
            return
        try:
            self.automation_session.prepare_scan(plan)
        except ValueError as exc:
            self._step4_fail(
                "13-view plan rejected by GUI: %s" % exc)
            return
        self.awaiting_scan_plan = False
        self.step4_phase = Step4Phase.PLAN_READY
        self.step4_plan_deadline = 0.0
        self.confirm_scan_button.configure(state="normal")
        if self.step45_auto_recovery_attempts:
            self.automation_status_var.set(
                "Automatic recovery prepared a fresh 13-view plan. Inspect it, "
                "then provide a new Step 5 confirmation for this exact hash.")
        else:
            self.automation_status_var.set(
                "Fresh 13-view plan ready. Inspect it, then confirm this exact hash.")

    def update_automation_buttons(self) -> None:
        acquisition_ready = (
            self.arm_enable_confirmed
            and
            self.automation_stack_running()
            and not self.automation_starting
            and not self.automation_stopping
            and not self.ros_node.automation_services_ready()
            and not self.ros_node.tesseract_readiness_rejection(
                "acquisition")
            and self.acquisition_phase in (
                AcquisitionPhase.IDLE, AcquisitionPhase.FAILED)
            and not self.acquisition_approval_inflight
        )
        self.prepare_acquisition_button.configure(
            state="normal" if acquisition_ready else "disabled")
        lock_rejection = self._tracking_health_rejection()
        lock_ready = lock_rejection == ''
        can_prepare_scan = (
            lock_ready
            and self.step4_phase not in STEP4_BUSY_PHASES
            and self.step4_phase != Step4Phase.PLAN_READY
            and not self.scan_approval_inflight
            and not self.automation_session.scan_approval_used
            and not self.automation_session.scan_plan_id
        )
        self.prepare_scan_button.configure(
            state="normal" if can_prepare_scan else "disabled")
        if (
                lock_ready
                and self.step4_phase == Step4Phase.IDLE
                and self.acquisition_phase in (
                    AcquisitionPhase.IDLE, AcquisitionPhase.FAILED)
                and not self.automation_session.scan_plan_id
                and not self.scan_approval_inflight):
            self.automation_status_var.set(
                "Step 4 ready: adopt this measured lock and prepare "
                "the fresh 13-view plan.")

    def poll_automation(self) -> None:
        if self.automation_processes and not self.automation_stopping:
            stopped = [
                "%s exited %s" % (name, process.returncode)
                for name, process in self.automation_processes.items()
                if process.poll() is not None
            ]
            if stopped and not self.automation_starting:
                step4_was_active = self.step4_phase in STEP4_BUSY_PHASES
                acquisition_was_active = self.acquisition_phase in (
                    AcquisitionPhase.REQUESTING_PREPARE,
                    AcquisitionPhase.WAITING_PLAN,
                )
                self._advance_automation_generation()
                self.automation_session.finish("; ".join(stopped))
                if step4_was_active:
                    self._step4_fail(
                        "managed automation process stopped: "
                        + "; ".join(stopped))
                if acquisition_was_active:
                    self._acquisition_fail(
                        "Managed automation process stopped during Step 2: "
                        + "; ".join(stopped))
                self.prepare_acquisition_button.configure(state="disabled")
                self.confirm_acquisition_button.configure(state="disabled")
                self.prepare_scan_button.configure(state="disabled")
                self.confirm_scan_button.configure(state="disabled")
                self.automation_status_var.set(
                    "Managed automation process stopped: " + "; ".join(stopped))
                if not self.automation_failure_handled:
                    self.automation_failure_handled = True
                    thread = threading.Thread(
                        target=self._clean_up_failed_automation_stack,
                        args=(
                            "; ".join(stopped),
                            self.automation_generation,
                        ),
                        daemon=True,
                    )
                    thread.start()
        self._poll_attempt_deadlines()
        self.update_automation_buttons()
        self.root.after(250, self.poll_automation)

    def _clean_up_failed_automation_stack(
            self, reason, generation) -> None:
        self._terminate_automation_processes()
        self.events.put((
            "automation_failed", (int(generation), str(reason))))

    def current_positions(self) -> List[float]:
        positions = []
        for var, (_, low, high, _) in zip(self.vars, self.joints):
            positions.append(clamp(float(var.get()), low, high))
        return positions

    def on_joint_change(self, _index: int) -> None:
        if not self.send_live_var.get():
            return
        now = time.time()
        if now - self.last_live_publish < 0.12:
            return
        self.last_live_publish = now
        self.send_target()

    def send_target(self) -> None:
        speed = clamp(float(self.speed_var.get()), 0.0, 100.0)
        effort = clamp(float(self.effort_var.get()), 0.5, 3.0)
        self.ros_node.publish_joint_target(self.current_positions(), speed, effort)

    def request_safe_disable(self) -> None:
        """Hold the live feedback pose and prove it settled before disabling."""
        if self.safe_disable_in_progress:
            self.service_text.set(
                "Safe disable is already waiting for the current-pose hold.")
            return
        joints, reason = self.fresh_feedback()
        if reason:
            self.service_text.set("Disable blocked: " + reason)
            return

        self.safe_disable_in_progress = True
        self.safe_disable_waiting_for_hold = False
        self.safe_disable_service_inflight = False
        self.safe_disable_target = energized_hold_target(joints)
        self.safe_disable_previous_feedback = None
        self.safe_disable_settled_since = None
        self.safe_disable_deadline = time.monotonic() + 8.0
        self.disable_button.configure(state="disabled")

        # A disable request is an operator stop, never a trigger for Step-4/5
        # automatic recovery.
        self.step45_auto_recovery_pending = False
        self.step45_auto_recovery_after_cancel = ""
        self.step45_auto_recovery_token += 1

        if self.automation_stack_running():
            # The scan executor is the sole command publisher while the
            # managed stack runs. Its existing cancel service must publish the
            # current feedback as an SDK MoveJ hold even when no motion is
            # active; the cancel response is the acknowledgement for that
            # command.
            self.safe_disable_waiting_for_hold = True
            self.service_text.set(
                "Safe disable: requesting an exact current-feedback hold "
                "from the scan executor.")
            self.ros_node.cancel_scan()
            return

        if not self.ros_node.manual_commands_enabled:
            self._safe_disable_fail(
                "GUI command ownership is unavailable; stop the scan stack "
                "and retry Disable.")
            return

        positions = list(self.safe_disable_target)
        gripper = (
            float(self.feedback_positions[6])
            if self.feedback_positions is not None
            and len(self.feedback_positions) >= 7
            else float(self.vars[6].get())
        )
        positions.append(gripper)
        speed = clamp(float(self.speed_var.get()), 1.0, 5.0)
        effort = clamp(float(self.effort_var.get()), 0.5, 3.0)
        self.ros_node.publish_joint_target(positions, speed, effort)
        self.command_text.set(
            "Safe-disable current feedback target: "
            + ", ".join("%.3f" % value for value in positions)
            + "\nwaiting for settled feedback before Disable")
        self._begin_safe_disable_settle()

    def _begin_safe_disable_settle(self) -> None:
        joints, reason = self.fresh_feedback()
        if reason:
            self._safe_disable_fail(reason)
            return
        # Keep the pose sampled before the hold request as the proof target.
        # Replacing it with feedback received after the request could validate
        # an uncommanded coasting pose. The executor independently samples and
        # publishes its own current feedback in the same bounded transaction;
        # any material disagreement therefore fails the target-error gate and
        # leaves the motors enabled.
        if self.safe_disable_target is None:
            self.safe_disable_target = np.asarray(joints, dtype=float)
        self.safe_disable_waiting_for_hold = False
        self.safe_disable_previous_feedback = None
        self.safe_disable_settled_since = None
        self.service_text.set(
            "Safe disable: current-feedback target sent; verifying the arm "
            "is settled before disabling.")
        self.root.after(100, self._poll_safe_disable_settle)

    def _poll_safe_disable_settle(self) -> None:
        if (
                not self.safe_disable_in_progress
                or self.safe_disable_waiting_for_hold
                or self.safe_disable_service_inflight):
            return
        now = time.monotonic()
        joints, reason = self.fresh_feedback()
        if reason:
            if now < self.safe_disable_deadline:
                self.root.after(100, self._poll_safe_disable_settle)
                return
            self._safe_disable_fail(reason)
            return

        current = np.asarray(joints, dtype=float)
        target_error = float(np.max(np.abs(
            current - self.safe_disable_target)))
        motion_delta = (
            float("inf")
            if self.safe_disable_previous_feedback is None
            else float(np.max(np.abs(
                current - self.safe_disable_previous_feedback)))
        )
        self.safe_disable_previous_feedback = current
        settled = target_error <= 0.025 and motion_delta <= 0.005
        if settled:
            if self.safe_disable_settled_since is None:
                self.safe_disable_settled_since = now
            elif now - self.safe_disable_settled_since >= 1.0:
                self.safe_disable_service_inflight = True
                self.service_text.set(
                    "Safe disable: exact current pose is settled; requesting "
                    "motor disable.")
                self.ros_node.call_enable_async(False)
                return
        else:
            self.safe_disable_settled_since = None

        if now >= self.safe_disable_deadline:
            self._safe_disable_fail(
                "current-feedback hold did not settle within 8 seconds "
                "(target error %.3f rad, latest motion %.3f rad); motors "
                "remain enabled" % (target_error, motion_delta))
            return
        self.root.after(100, self._poll_safe_disable_settle)

    def _safe_disable_fail(self, reason) -> None:
        self.safe_disable_in_progress = False
        self.safe_disable_waiting_for_hold = False
        self.safe_disable_service_inflight = False
        self.safe_disable_target = None
        self.safe_disable_previous_feedback = None
        self.safe_disable_settled_since = None
        self.safe_disable_deadline = 0.0
        self.disable_button.configure(state="normal")
        self.service_text.set(
            "Safe disable blocked: %s. Motors were not disabled." % reason)

    def use_feedback(self) -> None:
        if not self.feedback_positions or len(self.feedback_positions) < 7:
            self.feedback_text.set("No feedback to load")
            return
        for index, value in enumerate(self.feedback_positions[:7]):
            self.vars[index].set(float(value))

    def zero_target(self) -> None:
        for var in self.vars:
            var.set(0.0)

    def fresh_feedback(self):
        if not self.feedback_positions or len(self.feedback_positions) < 6:
            return None, "No six-joint feedback is available."
        feedback_time = self.ros_node.latest_feedback_monotonic
        if feedback_time is None or time.monotonic() - feedback_time > 1.0:
            return None, "Joint feedback is stale; check the driver."
        return np.asarray(self.feedback_positions[:6], dtype=float), ""

    def open_3d_joint_editor(self) -> None:
        if self.preview_process is not None and self.preview_process.poll() is None:
            self.preview_status_var.set("The 3D joint editor is already running.")
            return
        try:
            self.preview_process = subprocess.Popen([
                "ros2",
                "launch",
                "piper_description",
                "joint_preview.launch.py",
            ])
        except OSError as exc:
            self.preview_status_var.set(f"Could not start the 3D editor: {exc}")
            return
        self.preview_status_var.set(
            "Starting the 3D STL editor. Drag the orange joint rings; this is preview-only."
        )

    def close_3d_joint_editor(self) -> None:
        if self.preview_process is None or self.preview_process.poll() is not None:
            return
        self.preview_process.send_signal(signal.SIGINT)
        try:
            self.preview_process.wait(timeout=4.0)
        except subprocess.TimeoutExpired:
            self.preview_process.terminate()

    def load_live_joint_preview(self) -> None:
        joints, reason = self.fresh_feedback()
        if joints is None:
            self.preview_status_var.set(reason)
            return
        positions = list(self.feedback_positions[:8])
        self.ros_node.publish_preview_target(positions)
        self.preview_status_var.set(
            "Live joint feedback copied into the 3D preview; no real command was sent."
        )

    def send_joint_preview(self) -> None:
        joints, reason = self.fresh_feedback()
        if joints is None:
            self.preview_status_var.set(reason)
            return
        if self.preview_positions is None or len(self.preview_positions) < 6:
            self.preview_status_var.set("No 3D preview is available; open and load the editor first.")
            return
        target = np.asarray(self.preview_positions[:6], dtype=float)
        path = interpolate_joint_path(joints, target, 0.025)
        path_reasons = validate_joint_path(
            self.kinematics,
            path,
            self.preview_limits,
            obstacle_boxes=(),
            joint_margin_rad=0.0,
            self_clearance_m=0.060,
        )
        if path_reasons:
            self.preview_status_var.set(
                "3D preview path rejected by the conservative robot/floor model: "
                + path_reasons[0]
            )
            return
        status = self.ros_node.latest_status
        if status is None or int(status.err_code) != 0:
            self.preview_status_var.set("Arm status is missing or reports an error; motion refused.")
            return
        speed = clamp(float(self.preview_speed_var.get()), 1.0, 10.0)
        gripper = self.feedback_positions[6] if len(self.feedback_positions) >= 7 else self.vars[6].get()
        positions = list(target) + [float(gripper)]
        joint_text = ", ".join(f"{value:.3f}" for value in target)
        confirmed = messagebox.askyesno(
            "Confirm real 3D-preview motion",
            "This will make the enabled real arm mirror the 3D preview.\n\n"
            f"Speed: {speed:.0f}%\nJoints: {joint_text}\n\n"
            "Keep the workspace clear and emergency stop ready. Continue?",
            parent=self.root,
        )
        if not confirmed:
            self.preview_status_var.set("3D preview move cancelled; no command sent.")
            return
        self.ros_node.publish_joint_target(positions, speed, clamp(float(self.effort_var.get()), 0.5, 3.0))
        self.preview_status_var.set(
            f"3D preview target sent at {speed:.0f}%. Monitor the live arm and feedback."
        )

    def drain_events(self) -> None:
        try:
            while True:
                name, payload = self.events.get_nowait()
                if name == "feedback":
                    self.feedback_positions = list(payload.position)
                    shown = ", ".join(f"{value:.3f}" for value in self.feedback_positions[:7])
                    self.feedback_text.set(shown)
                elif name == "preview":
                    self.preview_positions = list(payload.position)
                    shown = ", ".join(
                        f"J{index + 1} {value:.3f}"
                        for index, value in enumerate(self.preview_positions[:6])
                    )
                    self.preview_angle_var.set(shown)
                elif name == "status":
                    self.status_text.set(
                        f"domain {os.environ.get('ROS_DOMAIN_ID', 'default')} | "
                        f"mode {payload.ctrl_mode} | arm {payload.arm_status} | err {payload.err_code} | "
                        f"{self.bounds_message}"
                    )
                elif name == "command":
                    positions, speed, effort = payload
                    shown = ", ".join(f"{value:.3f}" for value in positions)
                    self.command_text.set(f"{shown}\nspeed {speed:.0f}, effort {effort:.1f}")
                elif name == "service":
                    self.service_text.set(str(payload))
                elif name == "enable_service":
                    enabled, success, message = payload
                    self.service_text.set(str(message))
                    if not enabled and self.safe_disable_in_progress:
                        self.safe_disable_in_progress = False
                        self.safe_disable_waiting_for_hold = False
                        self.safe_disable_service_inflight = False
                        self.safe_disable_target = None
                        self.safe_disable_previous_feedback = None
                        self.safe_disable_settled_since = None
                        self.safe_disable_deadline = 0.0
                        self.disable_button.configure(state="normal")
                        self.service_text.set(
                            (
                                str(message)
                                + "; exact current-feedback hold was settled "
                                "before disable"
                            )
                            if success else
                            (
                                str(message)
                                + "; disable failed after the settled hold; "
                                "motors may remain enabled"
                            )
                        )
                        if success and self.cancel_home_shutdown_pending:
                            self.cancel_home_shutdown_report_pending = True
                            self.automation_status_var.set(
                                "Configured home and final hold proved; arm "
                                "disabled. Stopping the managed scan stack.")
                            self.stop_automation_stack()
                    # Any failed acknowledgement leaves the hardware state
                    # unconfirmed.  Planning must fail closed until the
                    # operator retries the GUI Enable action successfully.
                    self.arm_enable_confirmed = bool(enabled and success)
                    if (
                            success
                            and self.acquisition_phase
                            in (
                                AcquisitionPhase.REQUESTING_PREPARE,
                                AcquisitionPhase.WAITING_PLAN,
                                AcquisitionPhase.PLAN_READY,
                            )):
                        self.automation_session.finish(
                            "arm enable state changed after Step 2 started")
                        self.automation_session = AutomationSession()
                        self._acquisition_fail(
                            "Arm enable state changed; retry Step 2 for a "
                            "fresh plan from the measured loaded pose.",
                            preserve_request=False,
                        )
                        self.pending_scan_cancel_status = (
                            "Arm enable state changed after Step 2; fresh "
                            "planning is required.")
                        self.ros_node.cancel_scan()
                    self.update_automation_buttons()
                elif name == "command_blocked":
                    self.command_text.set(str(payload))
                elif name == "mission_feedback":
                    self.mission_status_var.set(
                        '%s: %s (%d accepted; model seed floor %d)' % (
                            payload.phase, payload.reason,
                            payload.accepted_captures,
                            payload.required_captures))
                elif name == "mission_state":
                    state, message = payload
                    self.mission_status_var.set(str(message))
                    self.mission_in_progress = state != 'IDLE'
                    if state == 'IDLE':
                        publishers = self.ros_node.command_publisher_names()
                        if not publishers:
                            self.ros_node.enable_manual_command_publisher()
                            self.set_manual_motion_enabled(True)
                        else:
                            self.mission_status_var.set(
                                str(message)
                                + '; manual controls remain locked while '
                                'command publishers exist: '
                                + ', '.join(publishers))
                    self.mission_start_button.configure(
                        state=(
                            "disabled" if self.mission_in_progress
                            else "normal"))
                    self.mission_cancel_button.configure(
                        state=(
                            "normal" if state == 'ACTIVE'
                            else "disabled"))
                elif name == "mission_result":
                    self.last_successful_mission = (
                        dict(payload)
                        if payload.get('outcome') == 'SUCCEEDED'
                        and payload.get('safe_shutdown') is True
                        and payload.get('mesh_job_id')
                        else None)
                    if self.last_successful_mission is not None:
                        self.mesh_status_var.set(
                            'Capture complete and arm shut down; report when '
                            'the tracked robot is home to build mesh %s'
                            % payload['mesh_job_id'])
                        self.report_base_home_button.configure(state='normal')
                elif name == "mesh_job_request":
                    success, message = payload
                    self.mesh_status_var.set(str(message))
                    if not success and self.last_successful_mission is not None:
                        self.report_base_home_button.configure(state='normal')
                elif name == "mesh_job_status":
                    current_job = (
                        self.last_successful_mission.get('mesh_job_id', '')
                        if isinstance(self.last_successful_mission, dict)
                        else '')
                    if current_job and str(payload.mesh_job_id) != current_job:
                        continue
                    self.mesh_status_var.set(
                        '%s: %s%s' % (
                            payload.state, payload.reason,
                            ('; mesh=' + payload.mesh_path)
                            if payload.mesh_path else ''))
                    if payload.state == 'FAILED' and self.last_successful_mission:
                        self.report_base_home_button.configure(state='disabled')
                elif name == "automation_start":
                    generation, success, message = payload
                    if int(generation) != self.automation_generation:
                        continue
                    self.automation_starting = False
                    self.start_automation_button.configure(
                        state="disabled" if success else "normal")
                    self.automation_status_var.set(message)
                    if not success:
                        self.automation_speed_spinbox.configure(state="normal")
                        if (
                                self.step4_phase == Step4Phase.STARTING_STACK
                                and self.step4_stack_generation == generation):
                            self._step4_fail(message)
                        publishers = self.ros_node.command_publisher_names()
                        if not publishers:
                            self.ros_node.enable_manual_command_publisher()
                            self.set_manual_motion_enabled(True)
                        else:
                            self.automation_status_var.set(
                                message + "; manual controls remain locked while "
                                "another command publisher exists")
                    elif (
                            self.step4_phase == Step4Phase.STARTING_STACK
                            and self.step4_stack_generation == generation):
                        self._step4_begin_workflow_check()
                    self.update_automation_buttons()
                elif name == "automation_start_progress":
                    generation, message = payload
                    if int(generation) == self.automation_generation:
                        self.automation_status_var.set(str(message))
                elif name == "automation_failed":
                    generation, failure_message = payload
                    if int(generation) != self.automation_generation:
                        continue
                    self.automation_starting = False
                    self.automation_speed_spinbox.configure(state="normal")
                    self.start_automation_button.configure(state="normal")
                    if self.step4_phase in STEP4_BUSY_PHASES:
                        self._step4_fail(
                            "managed scan stack failed: %s" % failure_message)
                    publishers = self.ros_node.command_publisher_names()
                    if not publishers:
                        self.ros_node.enable_manual_command_publisher()
                        self.set_manual_motion_enabled(True)
                        self.automation_status_var.set(
                            "Managed scan stack failed and was stopped: %s; "
                            "manual GUI command ownership restored"
                            % failure_message)
                    else:
                        self.automation_status_var.set(
                            "Managed scan stack failed: %s; manual controls "
                            "remain locked while command publishers exist: %s"
                            % (failure_message, ", ".join(publishers)))
                    self.update_automation_buttons()
                elif name == "acquisition_start":
                    (
                        stack_generation,
                        attempt_generation,
                        outcome,
                        message,
                    ) = payload
                    if (
                            int(stack_generation) != self.automation_generation
                            or int(attempt_generation)
                            != self.acquisition_attempt_generation
                            or self.acquisition_phase
                            != AcquisitionPhase.REQUESTING_PREPARE):
                        continue
                    if outcome == "accepted":
                        self.acquisition_phase = AcquisitionPhase.WAITING_PLAN
                        self.acquisition_plan_deadline = (
                            time.monotonic()
                            + ACQUISITION_PLAN_TIMEOUT_SEC)
                        self.automation_status_var.set(
                            str(message)
                            + "; waiting for the correlated acquisition plan")
                        if self.pending_early_acquisition_plan is not None:
                            early_plan = self.pending_early_acquisition_plan
                            self.pending_early_acquisition_plan = None
                            self.handle_scan_plan(early_plan)
                    else:
                        self._acquisition_fail(
                            message, preserve_request=(outcome == "timeout"))
                elif name == "multiview_plan_request":
                    (
                        stack_generation,
                        attempt_generation,
                        success,
                        request_id,
                        message,
                    ) = payload
                    if not self._step4_attempt_is_current(
                            stack_generation, attempt_generation):
                        continue
                    if self.step4_phase != Step4Phase.REQUESTING_PLAN:
                        continue
                    if success:
                        self.expected_scan_request_id = str(request_id)
                        self.step4_phase = Step4Phase.WAITING_PLAN
                        self.step4_plan_deadline = (
                            time.monotonic() + MULTIVIEW_PLAN_TIMEOUT_SEC)
                        self.automation_status_var.set(
                            str(message)
                            + "; waiting for the correlated 13-view result")
                        if (
                                self.latest_scan_plan is not None
                                and self.latest_scan_plan.plan_kind == MULTIVIEW_SCAN):
                            self.handle_scan_plan(self.latest_scan_plan)
                    else:
                        self._step4_fail(message)
                elif name == "workflow_start":
                    (
                        stack_generation,
                        attempt_generation,
                        success,
                        message,
                    ) = payload
                    if not self._step4_attempt_is_current(
                            stack_generation, attempt_generation):
                        continue
                    if self.step4_phase != Step4Phase.WAITING_SCAN_READY:
                        continue
                    if success:
                        self.automation_status_var.set(
                            message + "; waiting for a fresh SCAN_READY diagnostic")
                        self.step4_next_diagnostic_at = time.monotonic()
                    elif "already active" in str(message).lower():
                        self.automation_status_var.set(
                            message + "; rechecking authoritative workflow state")
                        self.step4_next_diagnostic_at = time.monotonic()
                    else:
                        self._step4_fail(message)
                elif name == "workflow_diagnostic":
                    (
                        stack_generation,
                        attempt_generation,
                        success,
                        workflow,
                        message,
                    ) = payload
                    if not self._step4_attempt_is_current(
                            stack_generation, attempt_generation):
                        continue
                    self.step4_diagnostic_inflight = False
                    if self.step4_phase not in (
                            Step4Phase.CHECKING_WORKFLOW,
                            Step4Phase.WAITING_SCAN_READY):
                        continue
                    if not success:
                        self._step4_fail(message)
                    else:
                        self._step4_handle_workflow(workflow)
                elif name == "scan_plan":
                    self.handle_scan_plan(payload)
                elif name == "scan_status":
                    self.latest_scan_status = payload
                    if self.mission_in_progress:
                        # Automatic Scan consumes action feedback/result from
                        # the mission node.  Ignore manual-tab retry, cancel
                        # and session-memory transitions while it is active.
                        continue
                    self.automation_status_var.set(
                        "%s %s: %s" % (
                            payload.execution_mode, payload.state, payload.reason))
                    if (
                            self.cancel_home_shutdown_pending
                            and payload.state == "ABORTED"):
                        cancellation_reason = str(payload.reason)
                        lowered_reason = cancellation_reason.lower()
                        if "configured home reached" in lowered_reason:
                            self.automation_status_var.set(
                                "Configured home reached; proving the final "
                                "hold before motor disable.")
                            self.request_safe_disable()
                        elif (
                                self.cancel_home_retry_count < 3
                                and (
                                    "current joint hold" in lowered_reason
                                    or "fresh return-home safety gate failed"
                                    in lowered_reason)):
                            self.cancel_home_retry_count += 1
                            self.automation_status_var.set(
                                "Cancellation stopped at a current-position "
                                "hold; waiting for it to settle before the "
                                "approved-path home retry.")
                            self.root.after(
                                1500, self._retry_cancel_home_after_hold)
                        else:
                            self.cancel_home_shutdown_pending = False
                            self.automation_status_var.set(
                                "Task failed: cancellation held the arm, but "
                                "automatic home/disable was blocked: %s. The "
                                "arm remains enabled; operator attention is "
                                "required." % cancellation_reason)
                        self.update_automation_buttons()
                        continue
                    if retryable_multiview_terminal(
                            payload.execution_mode,
                            payload.state,
                            self.automation_session.scan_approval_used):
                        self.automation_session.finish(payload.reason)
                        # The failed exact approval remains consumed. A fresh
                        # measured lock may begin a new Step-4 session, but it
                        # must receive a new plan and separate confirmation.
                        self.automation_session = AutomationSession()
                        self._step4_fail(
                            "13-view execution %s: %s; retry Step 4 with "
                            "a fresh plan and approval."
                            % (payload.state.lower(), payload.reason))
                    elif (
                            payload.state == "INVALID"
                            and payload.execution_mode == ROUGH_ACQUISITION
                            and self.acquisition_phase in (
                                AcquisitionPhase.REQUESTING_PREPARE,
                                AcquisitionPhase.WAITING_PLAN,
                                AcquisitionPhase.PLAN_READY)):
                        self._acquisition_fail(
                            "Acquisition proposal invalidated: %s; retry Step 2."
                            % payload.reason,
                            preserve_request=False,
                        )
                    elif (
                            payload.state == "INVALID"
                            and payload.execution_mode == MULTIVIEW_SCAN
                            and (
                                self.step4_phase in STEP4_BUSY_PHASES
                                or self.step4_phase == Step4Phase.PLAN_READY)):
                        try:
                            self.automation_session.discard_scan_plan()
                        except ValueError:
                            pass
                        self._step4_fail(
                            "13-view proposal invalidated: %s; retry Step 4."
                            % payload.reason)
                    elif payload.state in (
                            "ABORTED", "INVALID", "ACQUISITION_FAILED"):
                        self.automation_session.finish(payload.reason)
                        self.confirm_acquisition_button.configure(state="disabled")
                        self.prepare_scan_button.configure(state="disabled")
                        self.confirm_scan_button.configure(state="disabled")
                    elif payload.state == "COMPLETE":
                        self.step45_auto_recovery_attempts = 0
                        self.step45_auto_recovery_pending = False
                        self.step45_auto_recovery_token += 1
                        self.automation_session.finish("13-view scan complete")
                        self.automation_status_var.set(
                            "13-view scan and synchronized capture completed: "
                            + str(payload.reason))
                    elif (
                            payload.state == "ACQUIRED"
                            and payload.execution_mode == ROUGH_ACQUISITION):
                        self.automation_session.mark_target_acquired()
                        self.automation_status_var.set(
                            "Measured cube lock acquired. Waiting for SCAN_READY, "
                            "then prepare the fresh 13-view plan.")
                    self.update_automation_buttons()
                elif name == "heavy_status":
                    try:
                        value = json.loads(payload)
                    except (TypeError, json.JSONDecodeError):
                        value = {}
                    request_id = str(value.get("request_id", ""))
                    if (
                            self.latest_scan_status is not None
                            and (
                                not request_id
                                or request_id.startswith(
                                    self.latest_scan_status.plan_id + "-acquire-")
                            )):
                        self.grounding_status_var.set(
                            "%s | request=%s | job=%s" % (
                                value.get("state", "unknown"),
                                request_id or "none",
                                value.get("job_id", "none"),
                            ))
                elif name == "workflow_status":
                    try:
                        value = json.loads(payload)
                    except (TypeError, json.JSONDecodeError):
                        value = {}
                    if self.step4_phase not in (
                            Step4Phase.CHECKING_WORKFLOW,
                            Step4Phase.WAITING_SCAN_READY):
                        self.latest_workflow_status = value
                    self.automation_workflow_var.set(
                        "%s: %s" % (
                            value.get("state", "unknown"),
                            value.get("reason", ""),
                        ))
                    self.update_automation_buttons()
                elif name == "capture_status":
                    try:
                        value = json.loads(payload)
                    except (TypeError, json.JSONDecodeError):
                        value = {}
                    self.automation_capture_var.set(
                        "%s | frames=%s | %s" % (
                            value.get("state", "unknown"),
                            value.get("frames_captured", 0),
                            value.get("scan_dir", "dataset unavailable"),
                        ))
                elif name == "tracking_health":
                    self.latest_tracking_health = payload
                    self.latest_tracking_health_received_at = (
                        self.ros_node.latest_tracking_health_monotonic
                        or time.monotonic())
                    self.automation_tracking_var.set(
                        "%s | settled=%s | prediction=%s | age=%.3fs | %s"
                        % (
                            payload.lifecycle_state,
                            bool(payload.camera_settled),
                            bool(payload.prediction_only),
                            float(payload.measurement_age_sec),
                            payload.reason,
                        ))
                    self.update_automation_buttons()
                elif name == "tesseract_readiness":
                    self.update_automation_buttons()
                elif name == "acquisition_approval":
                    self.acquisition_approval_inflight = False
                    success, message = payload
                    self.automation_status_var.set(message)
                    if success:
                        self.automation_session.mark_acquisition_approved()
                        self.automation_status_var.set(
                            message + "; acquisition search is now running")
                    else:
                        # This exact approval cannot be reused.  Return Step 2
                        # to a genuinely fresh, retryable session instead of
                        # leaving the GUI stranded in PLAN_READY after the
                        # executor clears its proposal.
                        self.automation_session.finish(message)
                        self.automation_session = AutomationSession()
                        self._acquisition_fail(
                            "Acquisition approval rejected: %s; enable the "
                            "arm first, then retry Step 2 for a fresh plan."
                            % message,
                            preserve_request=False,
                        )
                        self.pending_scan_cancel_status = (
                            "Acquisition approval rejected: %s; enable the "
                            "arm first, then retry Step 2 for a fresh plan."
                            % message)
                        self.ros_node.cancel_scan()
                elif name == "scan_approval":
                    self.scan_approval_inflight = False
                    success, message = payload
                    self.automation_status_var.set(message)
                    if not success:
                        self.automation_session.finish(message)
                        # The exact approval is consumed, but a fresh measured
                        # lock may start a new Step-4 session with a new plan
                        # and a new exact approval. Never reuse the rejected
                        # plan/session or leave Step 4 permanently disabled.
                        self.automation_session = AutomationSession()
                        self._step4_fail(
                            "13-view approval rejected: " + message,
                            allow_auto_recovery=False,
                        )
                        self.pending_scan_cancel_status = (
                            "Step 4 failed: 13-view approval rejected: "
                            + message)
                        self.step45_auto_recovery_after_cancel = (
                            "13-view approval rejected: " + message)
                        self.ros_node.cancel_scan()
                    else:
                        self.step45_auto_recovery_pending = False
                        self.step45_auto_recovery_token += 1
                        self.automation_status_var.set(
                            message + "; 13-view RGB-D scan is running")
                elif name == "scan_cancel":
                    success, message = payload
                    if (
                            self.safe_disable_in_progress
                            and self.safe_disable_waiting_for_hold):
                        if (
                                success
                                and "current joint hold requested"
                                in str(message)):
                            self.command_text.set(
                                "Safe-disable current feedback hold requested "
                                "by the scan executor.")
                            self._begin_safe_disable_settle()
                        else:
                            self._safe_disable_fail(message)
                        continue
                    recovery_reason = self.step45_auto_recovery_after_cancel
                    self.step45_auto_recovery_after_cancel = ""
                    if self.pending_scan_cancel_status:
                        preserved = self.pending_scan_cancel_status
                        self.pending_scan_cancel_status = ""
                        self.automation_status_var.set(
                            preserved + "; executor proposal cleared for retry"
                            if success else
                            preserved + "; proposal cleanup failed: " + message)
                    else:
                        self.automation_status_var.set(message)
                    if success and recovery_reason:
                        self._schedule_step45_auto_recovery(recovery_reason)
        except queue.Empty:
            pass
        self.root.after(100, self.drain_events)

    def shutdown(self) -> None:
        if self.automation_processes:
            self.ros_node.cancel_scan()
            time.sleep(1.5)
            self._terminate_automation_processes()
        self.close_3d_joint_editor()


def main() -> None:
    events: "queue.Queue[tuple]" = queue.Queue()
    rclpy.init()
    ros_node = PiperGuiRos(events)
    # Keep high-rate feedback/status subscriptions from starving service
    # responses. Step 2 previously saw the prepare service in the graph but
    # timed out twice before its response could be dispatched.
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(ros_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    root = tk.Tk()
    app = PiperGuiApp(root, ros_node, events)
    try:
        root.mainloop()
    finally:
        app.shutdown()
        executor.shutdown()
        ros_node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
