#!/usr/bin/env python3
"""ROS adapter for the native PiPER GUI."""

import queue
import threading
import time
from typing import List

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from piper_gui.ros_client import MissionActionClient
from piper_mobile_manipulation.action import RunTargetScan
from piper_mobile_manipulation.msg import (
    MeshJobStatus,
    TesseractReadiness,
    TrackingHealth,
)
from piper_mobile_manipulation.srv import ReportTrackedRobotHomed
from piper_msgs.msg import PiperStatusMsg
from piper_msgs.srv import Enable


class PiperGuiRos(Node):
    def __init__(self, events: "queue.Queue[tuple]") -> None:
        super().__init__("piper_native_gui")
        self.events = events
        self.latest_feedback = None
        self.latest_feedback_monotonic = None
        self.latest_status = None
        self.manual_commands_enabled = False
        self.command_lock = threading.Lock()
        self.service_callback_group = ReentrantCallbackGroup()

        self.joint_pub = None
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
        self.mesh_status_sub = self.create_subscription(
            MeshJobStatus,
            "/piper/mesh_job_status",
            lambda msg: self.events.put(("mesh_job_status", msg)),
            10,
        )
        self.diagnostic_subscriptions = [
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
                lambda msg: self.events.put(("tracking_health", msg)),
                10,
            ),
            self.create_subscription(
                TesseractReadiness,
                "/piper/tesseract_readiness",
                lambda msg: self.events.put(("tesseract_readiness", msg)),
                10,
            ),
        ]
        self.mission_action_client = ActionClient(
            self, RunTargetScan, '/piper/run_target_scan',
            callback_group=self.service_callback_group)
        self.mission_client = MissionActionClient(
            action_client=self.mission_action_client,
            goal_builder=self._build_mission_goal,
            outcome_names={
                RunTargetScan.Goal.OUTCOME_SUCCEEDED: 'SUCCEEDED',
                RunTargetScan.Goal.OUTCOME_FAILED: 'FAILED',
                RunTargetScan.Goal.OUTCOME_CANCELLED: 'CANCELLED',
                RunTargetScan.Goal.OUTCOME_BUSY: 'BUSY',
                RunTargetScan.Goal.OUTCOME_UNSUPPORTED_TARGET_PROFILE:
                    'UNSUPPORTED_TARGET_PROFILE',
                RunTargetScan.Goal.OUTCOME_NEEDS_OPERATOR: 'NEEDS_OPERATOR',
                RunTargetScan.Goal.OUTCOME_REPOSITION_REQUIRED:
                    'REPOSITION_REQUIRED',
            },
            event_sink=lambda event: self.events.put(("mission_client", event)),
        )
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
                "Commissioning joint commands are disabled while the "
                "production mission owns motion.",
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
            unresolved = any('UNKNOWN' in name.upper() for name in names)
            if names and not unresolved or time.monotonic() >= deadline:
                return names
            time.sleep(0.05)

    def _build_mission_goal(self, task_id, request):
        goal = RunTargetScan.Goal()
        goal.task_id = str(task_id)
        goal.task_type = 'SCAN_3D'
        goal.target_label = request.target_label
        goal.target_profile = ''
        goal.target_confidence = 1.0
        goal.deadline_sec = 1200.0
        goal.rough_target = PoseWithCovarianceStamped()
        goal.rough_target.header.stamp = self.get_clock().now().to_msg()
        goal.rough_target.header.frame_id = 'base_link'
        goal.rough_target.pose.pose.position.x = request.coordinates[0]
        goal.rough_target.pose.pose.position.y = request.coordinates[1]
        goal.rough_target.pose.pose.position.z = request.coordinates[2]
        goal.rough_target.pose.pose.orientation.w = 1.0
        covariance = [0.0] * 36
        covariance[0] = covariance[7] = covariance[14] = 0.01
        goal.rough_target.pose.covariance = covariance
        return goal

    def submit_mission(self, request):
        return self.mission_client.submit(request)

    def cancel_mission(self):
        return self.mission_client.cancel()

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

    def destroy_node(self):
        client = getattr(self, 'mission_action_client', None)
        if client is not None:
            try:
                client.destroy()
            except Exception:
                pass
            self.mission_action_client = None
        return super().destroy_node()
