#!/usr/bin/env python3

import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import List

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from piper_gui.app import run_gui
from piper_gui.camera_profile import (
    CAMERA_PROFILE_FPS,
    CameraProfile,
    DEFAULT_CAMERA_PROFILE,
    default_camera_profile_path,
    read_camera_profile,
    write_camera_profile,
)
from piper_gui.ros_client import MissionActionClient, MissionClientEvent
from piper_gui.ray_reports import (
    list_ray_reports,
    ray_report_display_name,
    ray_report_selection,
    RayReviewProcess,
    replay_scan_dataset,
)
from piper_gui.scan_policy import (
    FULL_SPHERE_REGION,
    POLICY_LABELS,
    RAY_NBV_POLICY,
    RAY_REGION_LABELS,
    SELECTABLE_POLICIES,
    SELECTABLE_RAY_REGIONS,
    default_scan_policy_path,
    read_scan_settings,
    write_scan_settings,
)
from piper_gui.view_model import MissionViewModel
from reconstruction.gui_support import (
    existing_reconstruction_outputs,
    list_scan_datasets,
    load_quality_report,
    quality_summary,
    reconstruction_command,
    start_reconstruction_process,
    start_viewer_process,
    viewer_command,
)
from piper_mobile_manipulation.msg import (
    MeshJobStatus,
    TesseractReadiness,
    TrackingHealth,
)
from piper_mobile_manipulation.action import RunTargetScan
from piper_mobile_manipulation.collision_environment import (
    FLOOR_PROFILE_LABELS,
    SELECTABLE_FLOOR_PROFILES,
    default_collision_environment_path,
    read_collision_environment,
    write_collision_environment,
)
from piper_mobile_manipulation.home_pose import (
    load_home_pose,
    save_home_pose,
    validate_home_profile_limits,
)
from piper_mobile_manipulation.srv import ReportTrackedRobotHomed
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

        merged.append(
            (
                name,
                min(measured_low, measured_high),
                max(measured_low, measured_high),
                unit,
            )
        )

    return merged, f"loaded {BOUNDS_PATH}"


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


GUI_MAX_WIDTH = 1920
GUI_MAX_HEIGHT = 1080
GUI_MIN_WIDTH = 1120
GUI_MIN_HEIGHT = 680
GUI_HORIZONTAL_SCREEN_MARGIN = 48
GUI_VERTICAL_SCREEN_MARGIN = 96


def fitted_gui_geometry(screen_width: int, screen_height: int):
    """Return a centred GUI rectangle that fits inside the current screen."""
    width = max(1, int(screen_width))
    height = max(1, int(screen_height))
    usable_width = max(1, width - GUI_HORIZONTAL_SCREEN_MARGIN)
    usable_height = max(1, height - GUI_VERTICAL_SCREEN_MARGIN)
    window_width = min(GUI_MAX_WIDTH, usable_width)
    window_height = min(GUI_MAX_HEIGHT, usable_height)
    return (
        window_width,
        window_height,
        max(0, (width - window_width) // 2),
        max(0, (height - window_height) // 2),
    )


def primary_monitor_geometry(xrandr_output: str):
    """Parse the XRandR primary monitor as ``(x, y, width, height)``."""
    fallback = None
    for line in str(xrandr_output).splitlines():
        fields = line.split()
        if len(fields) < 3 or not fields[0].rstrip(':').isdigit():
            continue
        geometry = fields[2]
        try:
            size, x_text, y_text = geometry.rsplit('+', 2)
            width_text, height_text = size.split('x', 1)
            width = int(width_text.split('/', 1)[0])
            height = int(height_text.split('/', 1)[0])
            value = (int(x_text), int(y_text), width, height)
        except (TypeError, ValueError):
            continue
        fallback = fallback or value
        if '*' in fields[1]:
            return value
    return fallback


def detected_monitor_geometry(root: tk.Tk):
    """Return the primary monitor, with a portable Tk screen fallback."""
    try:
        result = subprocess.run(
            ['xrandr', '--listactivemonitors'],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.0,
        )
        monitor = primary_monitor_geometry(result.stdout)
        if result.returncode == 0 and monitor is not None:
            return monitor
    except (OSError, subprocess.SubprocessError):
        pass
    return (0, 0, root.winfo_screenwidth(), root.winfo_screenheight())


def fit_gui_to_screen(root: tk.Tk) -> None:
    """Size the initial window to the display without preventing resizing."""
    root.update_idletasks()
    monitor_x, monitor_y, screen_width, screen_height = (
        detected_monitor_geometry(root))
    width, height, x_position, y_position = fitted_gui_geometry(
        screen_width, screen_height)
    # On smaller displays, lowering the minimum to the fitted size prevents
    # the window manager from forcing controls beyond a screen edge.
    root.minsize(min(GUI_MIN_WIDTH, width), min(GUI_MIN_HEIGHT, height))
    root.geometry(
        "%dx%d+%d+%d" % (
            width, height,
            monitor_x + x_position,
            monitor_y + y_position))


class PiperGuiApp:
    def __init__(self, root: tk.Tk, ros_node: PiperGuiRos, events: "queue.Queue[tuple]") -> None:
        self.root = root
        self.ros_node = ros_node
        self.events = events
        self.vars: List[tk.DoubleVar] = []
        self.feedback_positions = None
        self.joints, self.bounds_message = load_joint_limits()

        self.root.title("PiPER Control")

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
        self.rough_coordinate_vars = [
            tk.StringVar(value=""),
            tk.StringVar(value=""),
            tk.StringVar(value=""),
        ]
        self.mission_label_var = tk.StringVar(value="green cube")
        self.scan_policy_path = default_scan_policy_path(PROJECT_ROOT)
        try:
            saved_settings = read_scan_settings(self.scan_policy_path)
            saved_policy = saved_settings.policy
            policy_status = (
                "Saved for next scan stack: %s; %s; %d rays"
                % (
                    POLICY_LABELS[saved_policy],
                    RAY_REGION_LABELS[saved_settings.ray_region],
                    saved_settings.ray_count))
        except (OSError, ValueError) as exc:
            saved_policy = RAY_NBV_POLICY
            saved_ray_region = FULL_SPHERE_REGION
            saved_ray_count = 175
            policy_status = (
                "Planner policy configuration unavailable: %s" % exc)
        else:
            saved_ray_region = saved_settings.ray_region
            saved_ray_count = saved_settings.ray_count
        self.scan_policy_var = tk.StringVar(value=POLICY_LABELS[saved_policy])
        self.scan_ray_region_var = tk.StringVar(
            value=RAY_REGION_LABELS[saved_ray_region])
        self.scan_ray_count_var = tk.StringVar(
            value=str(saved_ray_count))
        self.scan_policy_status_var = tk.StringVar(value=policy_status)
        self.collision_environment_path = default_collision_environment_path(
            PROJECT_ROOT)
        try:
            saved_environment = read_collision_environment(
                self.collision_environment_path)
            floor_status = (
                'Saved for next mission: %s; combined PiPER/L515/Bunker '
                'geometry is always active.'
                % FLOOR_PROFILE_LABELS[saved_environment.floor_profile])
            saved_floor_profile = saved_environment.floor_profile
        except (OSError, ValueError) as exc:
            saved_floor_profile = SELECTABLE_FLOOR_PROFILES[0]
            floor_status = (
                'Floor configuration unavailable: %s' % exc)
        self.floor_profile_var = tk.StringVar(
            value=FLOOR_PROFILE_LABELS[saved_floor_profile])
        self.floor_profile_status_var = tk.StringVar(value=floor_status)
        self.camera_profile_path = default_camera_profile_path(PROJECT_ROOT)
        try:
            saved_camera_profile = read_camera_profile(
                self.camera_profile_path)
            camera_profile_status = (
                "Saved for next camera startup: %dx%d@%d FPS"
                % (
                    saved_camera_profile.width,
                    saved_camera_profile.height,
                    saved_camera_profile.fps))
        except (OSError, ValueError) as exc:
            saved_camera_profile = CameraProfile(*DEFAULT_CAMERA_PROFILE)
            camera_profile_status = (
                "Camera profile configuration unavailable: %s" % exc)
        self.camera_resolution_var = tk.StringVar(
            value="%dx%d" % (
                saved_camera_profile.width, saved_camera_profile.height))
        self.camera_fps_var = tk.StringVar(
            value=str(saved_camera_profile.fps))
        self.camera_profile_status_var = tk.StringVar(
            value=camera_profile_status)
        self.mission_status_var = tk.StringVar(value="Autonomous mission idle")
        self.mesh_status_var = tk.StringVar(
            value="Mesh reconstruction is waiting for a completed scan")
        self.perception_status_var = tk.StringVar(
            value="GroundingDINO / heavy refresh idle")
        self.tracking_status_var = tk.StringVar(value="Tracking unavailable")
        self.workflow_status_var = tk.StringVar(value="Workflow unavailable")
        self.capture_status_var = tk.StringVar(value="RGB-D capture unavailable")
        self.planning_status_var = tk.StringVar(value="Tesseract readiness unavailable")
        self.last_successful_mission = None
        self.reconstruction_dataset_var = tk.StringVar(value='')
        self.reconstruction_mode_var = tk.StringVar(value='auto')
        self.reconstruction_mask_source_var = tk.StringVar(
            value='captured')
        self.reconstruction_dimension_vars = [
            tk.StringVar(value='35') for _axis in range(3)]
        self.reconstruction_voxel_mm_var = tk.DoubleVar(value=3.0)
        self.reconstruction_voxel_status_var = tk.StringVar(
            value='3.0 mm voxel — baseline mesh detail')
        self.reconstruction_show_input_var = tk.BooleanVar(value=False)
        self.reconstruction_status_var = tk.StringVar(
            value='Select a completed scan dataset')
        self.reconstruction_summary_var = tk.StringVar(
            value='Validation requires visual review; expected cube is 35 mm.')
        self.reconstruction_process = None
        self.reconstruction_report_path = None
        self.reconstruction_output_path = None
        self.ray_report_var = tk.StringVar(value='')
        self.ray_report_paths = ()
        self.ray_replay_dataset_var = tk.StringVar(value='')
        self.ray_report_status_var = tk.StringVar(
            value='No mission ray report selected')
        self.ray_replay_in_progress = False
        self.ray_review_process = RayReviewProcess(PROJECT_ROOT)
        self.home_status_var = tk.StringVar(
            value="Home: validated compact default")
        self.mission_view_model = MissionViewModel()
        self.load_selected_home()
        self.safe_disable_in_progress = False

        ros_domain = os.environ.get("ROS_DOMAIN_ID", "default")
        self.status_text = tk.StringVar(
            value=f"ROS domain {ros_domain} | {self.bounds_message}"
        )
        self.feedback_text = tk.StringVar(value="No feedback")
        self.command_text = tk.StringVar(value="No command sent")
        self.service_text = tk.StringVar(value="No service call")

        self._build()
        fit_gui_to_screen(self.root)
        self.enable_button.configure(state="disabled")
        self.set_manual_motion_enabled(False)
        # Commissioning motion starts locked. Resolve the ROS graph after the
        # UI and executor are running, then create this GUI's command publisher
        # only when no production command owner exists.
        self.root.after(250, self._restore_manual_controls_if_unowned)
        self.root.after(100, self.drain_events)

    def _build(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=(14, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        title = ttk.Label(header, text="PiPER Control", font=("TkDefaultFont", 16, "bold"))
        title.grid(row=0, column=0, sticky="w")
        self.enable_button = ttk.Button(
            header,
            text="Commissioning Enable",
            command=lambda: self.ros_node.call_enable_async(True),
        )
        self.enable_button.grid(row=0, column=1, padx=4)
        self.disable_button = ttk.Button(
            header,
            text="Commissioning Disable (No Home)",
            command=self.request_safe_disable,
        )
        self.disable_button.grid(row=0, column=2, padx=4)

        body = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        body.grid(row=1, column=0, sticky="nsew")

        notebook = ttk.Notebook(body)
        body.add(notebook, weight=4)

        automatic = ttk.Frame(notebook, padding=14)
        notebook.add(automatic, text="Automatic Scan")
        self._build_automatic_scan(automatic)

        manual = ttk.Frame(notebook, padding=14)
        notebook.add(manual, text="Commissioning: Manual")
        self._build_manual(manual)

        graphical = ttk.Frame(notebook, padding=14)
        notebook.add(graphical, text="Commissioning: 3D Preview")
        self._build_graphical(graphical)

        diagnostics = ttk.Frame(notebook, padding=14)
        notebook.add(diagnostics, text="Diagnostics")
        self._build_diagnostics(diagnostics)

        reconstruction = ttk.Frame(notebook, padding=14)
        notebook.add(reconstruction, text='Reconstruction Validation')
        self._build_reconstruction(reconstruction)

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
        ttk.Label(target, text="Configured staged home").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Label(
            target,
            textvariable=self.home_status_var,
            foreground="#52606d",
            wraplength=570,
            justify="left",
        ).grid(row=2, column=2, columnspan=4, sticky="w", padx=(10, 0),
               pady=(10, 0))

        policy = ttk.LabelFrame(
            parent, text="Viewpoint policy for next mission", padding=12)
        policy.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        self.scan_policy_combo = ttk.Combobox(
            policy,
            textvariable=self.scan_policy_var,
            values=tuple(POLICY_LABELS[item] for item in SELECTABLE_POLICIES),
            state="readonly",
            width=34,
        )
        self.scan_policy_combo.grid(row=0, column=0, sticky="w")
        self.scan_ray_region_combo = ttk.Combobox(
            policy,
            textvariable=self.scan_ray_region_var,
            values=tuple(
                RAY_REGION_LABELS[item] for item in SELECTABLE_RAY_REGIONS),
            state="readonly",
            width=28,
        )
        self.scan_ray_region_combo.grid(
            row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(policy, text="Rays").grid(
            row=0, column=2, sticky="e", padx=(10, 4))
        self.scan_ray_count_spinbox = ttk.Spinbox(
            policy,
            from_=1,
            to=1000,
            textvariable=self.scan_ray_count_var,
            width=6,
        )
        self.scan_ray_count_spinbox.grid(row=0, column=3, sticky="w")
        self.scan_policy_apply_button = ttk.Button(
            policy,
            text="Apply for Next Mission",
            command=self.apply_scan_policy,
        )
        self.scan_policy_apply_button.grid(row=0, column=4, padx=(8, 0))
        ttk.Label(
            policy,
            textvariable=self.scan_policy_status_var,
            foreground="#52606d",
            wraplength=760,
            justify="left",
        ).grid(row=1, column=0, columnspan=5, sticky="w", pady=(8, 0))
        ttk.Label(
            policy,
            text=(
                "These change only the saved planner settings. The mission "
                "server starts a fresh scan stack for each mission; it does "
                "not change a stack that is already running. Ray settings "
                "are retained when another policy is selected."
            ),
            foreground="#52606d",
            wraplength=760,
            justify="left",
        ).grid(row=2, column=0, columnspan=5, sticky="w", pady=(4, 0))

        environment = ttk.LabelFrame(
            parent, text='Collision environment for next mission', padding=12)
        environment.grid(row=4, column=0, sticky='ew', pady=(12, 0))
        self.floor_profile_combo = ttk.Combobox(
            environment,
            textvariable=self.floor_profile_var,
            values=tuple(
                FLOOR_PROFILE_LABELS[item]
                for item in SELECTABLE_FLOOR_PROFILES),
            state='readonly',
            width=38,
        )
        self.floor_profile_combo.grid(row=0, column=0, sticky='w')
        self.floor_profile_apply_button = ttk.Button(
            environment,
            text='Apply for Next Mission',
            command=self.apply_floor_profile,
        )
        self.floor_profile_apply_button.grid(row=0, column=1, padx=(8, 0))
        ttk.Label(
            environment,
            textvariable=self.floor_profile_status_var,
            foreground='#52606d',
            wraplength=760,
            justify='left',
        ).grid(row=1, column=0, columnspan=2, sticky='w', pady=(8, 0))
        ttk.Label(
            environment,
            text=(
                'The Bunker chassis and sensor station remain visible and '
                'collision-active in both modes. Only the support-floor '
                'height changes, and it is frozen when the next mission starts.'),
            foreground='#52606d',
            wraplength=760,
            justify='left',
        ).grid(row=2, column=0, columnspan=2, sticky='w', pady=(4, 0))

        camera = ttk.LabelFrame(
            parent, text="L515 RGB profile for next mission", padding=12)
        camera.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        self.camera_resolution_combo = ttk.Combobox(
            camera,
            textvariable=self.camera_resolution_var,
            values=tuple(
                "%dx%d" % resolution
                for resolution in CAMERA_PROFILE_FPS),
            state="readonly",
            width=14,
        )
        self.camera_resolution_combo.grid(row=0, column=0, sticky="w")
        self.camera_resolution_combo.bind(
            "<<ComboboxSelected>>", self._camera_resolution_changed)
        self.camera_fps_combo = ttk.Combobox(
            camera,
            textvariable=self.camera_fps_var,
            state="readonly",
            width=8,
        )
        self.camera_fps_combo.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.camera_profile_apply_button = ttk.Button(
            camera,
            text="Apply for Next Mission",
            command=self.apply_camera_profile,
        )
        self.camera_profile_apply_button.grid(row=0, column=2, padx=(8, 0))
        ttk.Label(
            camera,
            textvariable=self.camera_profile_status_var,
            foreground="#52606d",
            wraplength=760,
            justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(
            camera,
            text=(
                "RGB resolution/FPS is loaded when the next mission starts "
                "the L515. Depth remains at 640x480@30. This never changes "
                "a camera process that is already running."),
            foreground="#52606d",
            wraplength=760,
            justify="left",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self._camera_resolution_changed()

        controls = ttk.Frame(parent)
        controls.grid(row=6, column=0, sticky="w", pady=(18, 10))
        self.mission_start_button = ttk.Button(
            controls,
            text="Start Complete Automated Scan",
            command=self.start_automated_scan,
        )
        self.mission_start_button.grid(row=0, column=0, padx=(0, 8))
        self.mission_cancel_button = ttk.Button(
            controls,
            text="Cancel and Home",
            command=self.cancel_automated_scan,
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
        status.grid(row=7, column=0, sticky="ew", pady=(6, 0))
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
        ).grid(row=8, column=0, sticky="ew", pady=(18, 0))

        ttk.Label(
            parent,
            text=(
                "The command-free mission listener must already be running via "
                "run_target_scan_mission.sh, as it will be in the final system. "
                "The GUI submits and cancels the public action only; mission "
                "phases, safety, retries, motion, and child-process lifecycle "
                "remain owned by that production listener."
            ),
            foreground="#52606d",
            wraplength=820,
            justify="left",
        ).grid(row=9, column=0, sticky="ew", pady=(12, 0))

    def _camera_resolution_changed(self, _event=None):
        try:
            width_text, height_text = self.camera_resolution_var.get().split(
                "x", 1)
            resolution = (int(width_text), int(height_text))
            frame_rates = CAMERA_PROFILE_FPS[resolution]
        except (KeyError, TypeError, ValueError):
            resolution = DEFAULT_CAMERA_PROFILE[:2]
            frame_rates = CAMERA_PROFILE_FPS[resolution]
            self.camera_resolution_var.set("%dx%d" % resolution)
        choices = tuple(str(value) for value in frame_rates)
        self.camera_fps_combo.configure(values=choices)
        if self.camera_fps_var.get() not in choices:
            preferred = str(DEFAULT_CAMERA_PROFILE[2])
            self.camera_fps_var.set(
                preferred if preferred in choices else choices[-1])

    def apply_camera_profile(self):
        if not self.mission_view_model.state.can_start:
            self.camera_profile_status_var.set(
                "Camera profile was not changed: wait for the active mission "
                "to finish.")
            return
        try:
            width_text, height_text = self.camera_resolution_var.get().split(
                "x", 1)
            profile = write_camera_profile(
                self.camera_profile_path,
                width_text,
                height_text,
                self.camera_fps_var.get())
        except (OSError, ValueError) as exc:
            self.camera_profile_status_var.set(
                "Camera profile was not changed: %s" % exc)
            return
        self.camera_profile_status_var.set(
            "Saved for next camera startup: %dx%d@%d FPS"
            % (profile.width, profile.height, profile.fps))

    def _build_reconstruction(self, parent: ttk.Frame) -> None:
        """Build command-free, offline TSDF/GICP validation controls."""
        parent.columnconfigure(0, weight=1)
        ttk.Label(
            parent, text='Offline reconstruction validation',
            font=('TkDefaultFont', 16, 'bold')).grid(
                row=0, column=0, sticky='w')
        ttk.Label(
            parent,
            text=(
                'This tab reads completed datasets only. It never starts ROS '
                'or commands the arm. Robot-pose TSDF is the baseline; bounded '
                'GICP is an optional local registration correction.'),
            wraplength=820, justify='left').grid(
                row=1, column=0, sticky='ew', pady=(8, 16))
        inputs = ttk.LabelFrame(parent, text='Dataset and expected size', padding=12)
        inputs.grid(row=2, column=0, sticky='ew')
        ttk.Label(inputs, text='Dataset').grid(row=0, column=0, sticky='w')
        self.reconstruction_dataset_combo = ttk.Combobox(
            inputs, textvariable=self.reconstruction_dataset_var,
            state='readonly', width=36)
        self.reconstruction_dataset_combo.grid(
            row=0, column=1, columnspan=3, sticky='ew', padx=(8, 8))
        self.reconstruction_dataset_combo.bind(
            '<<ComboboxSelected>>', self.load_existing_reconstruction_outputs)
        ttk.Button(
            inputs, text='Refresh',
            command=self.refresh_reconstruction_datasets).grid(
                row=0, column=4)
        for index, (axis, variable) in enumerate(zip(
                ('X mm', 'Y mm', 'Z mm'), self.reconstruction_dimension_vars)):
            ttk.Label(inputs, text=axis).grid(
                row=1, column=index * 2, sticky='w', pady=(10, 0))
            ttk.Entry(inputs, textvariable=variable, width=9).grid(
                row=1, column=index * 2 + 1, sticky='w', padx=(5, 14),
                pady=(10, 0))
        ttk.Label(inputs, text='Registration').grid(
            row=2, column=0, sticky='w', pady=(10, 0))
        ttk.Combobox(
            inputs, textvariable=self.reconstruction_mode_var,
            values=(
                'auto', 'robot_pose', 'scene_pose_graph',
                'bounded_gicp', 'multiway_gicp'),
            state='readonly', width=23).grid(
                row=2, column=1, sticky='w', padx=(8, 0), pady=(10, 0))
        ttk.Label(inputs, text='Target mask').grid(
            row=3, column=0, sticky='w', pady=(10, 0))
        ttk.Combobox(
            inputs, textvariable=self.reconstruction_mask_source_var,
            values=('offline_resegment', 'captured'),
            state='readonly', width=23).grid(
                row=3, column=1, sticky='w', padx=(8, 0), pady=(10, 0))
        ttk.Label(
            inputs,
            text=(
                'offline_resegment runs fresh GroundingDINO/SAM2 and can only '
                'narrow the immutable confidence-qualified target support.'),
            foreground='#52606d', wraplength=540, justify='left').grid(
                row=3, column=2, columnspan=3, sticky='w', padx=(8, 0),
                pady=(10, 0))
        ttk.Label(inputs, text='Mesh detail').grid(
            row=4, column=0, sticky='w', pady=(12, 0))
        ttk.Label(inputs, text='Coarse').grid(
            row=4, column=1, sticky='w', pady=(12, 0))
        tk.Scale(
            inputs, from_=3.0, to=0.5, resolution=0.1,
            orient='horizontal', showvalue=False, length=330,
            variable=self.reconstruction_voxel_mm_var,
            command=self.update_reconstruction_voxel_status).grid(
                row=4, column=2, columnspan=2, sticky='ew',
                padx=(8, 8), pady=(12, 0))
        ttk.Label(inputs, text='Fine').grid(
            row=4, column=4, sticky='e', pady=(12, 0))
        ttk.Label(
            inputs, textvariable=self.reconstruction_voxel_status_var,
            foreground='#52606d').grid(
            row=5, column=1, columnspan=4, sticky='w', pady=(3, 0))
        controls = ttk.Frame(parent)
        controls.grid(row=3, column=0, sticky='w', pady=(14, 8))
        self.reconstruction_build_button = ttk.Button(
            controls, text='Build Raw + Cleaned',
            command=self.start_reconstruction)
        self.reconstruction_build_button.grid(row=0, column=0)
        self.reconstruction_view_button = ttk.Button(
            controls, text='Open Cleaned',
            command=self.open_reconstruction_viewer, state='disabled')
        self.reconstruction_view_button.grid(row=0, column=1, padx=(8, 0))
        self.reconstruction_raw_view_button = ttk.Button(
            controls, text='Open Raw',
            command=lambda: self.open_reconstruction_viewer('raw'),
            state='disabled')
        self.reconstruction_raw_view_button.grid(
            row=0, column=2, padx=(8, 0))
        self.reconstruction_measured_view_button = ttk.Button(
            controls, text='Open Measured Points',
            command=lambda: self.open_reconstruction_viewer('measured'),
            state='disabled')
        self.reconstruction_measured_view_button.grid(
            row=0, column=3, padx=(8, 0))
        ttk.Checkbutton(
            controls, text='Overlay measured points',
            variable=self.reconstruction_show_input_var).grid(
                row=0, column=4, padx=(12, 0))
        result = ttk.LabelFrame(parent, text='Quality report', padding=12)
        result.grid(row=4, column=0, sticky='ew')
        ttk.Label(
            result, textvariable=self.reconstruction_status_var,
            wraplength=800, justify='left').grid(row=0, column=0, sticky='ew')
        ttk.Label(
            result, textvariable=self.reconstruction_summary_var,
            foreground='#52606d', wraplength=800, justify='left').grid(
                row=1, column=0, sticky='ew', pady=(8, 0))
        self.refresh_reconstruction_datasets()

    def update_reconstruction_voxel_status(self, _value=None) -> None:
        """Explain mesh density without claiming additional measured detail."""
        voxel_mm = round(float(self.reconstruction_voxel_mm_var.get()), 1)
        relative = (3.0 / voxel_mm) ** 2
        self.reconstruction_voxel_status_var.set(
            '%.1f mm voxel — approximately %.1fx baseline surface density; '
            'does not repair missing or misregistered geometry'
            % (voxel_mm, relative))

    def refresh_reconstruction_datasets(self) -> None:
        names = tuple(path.name for path in list_scan_datasets(PROJECT_ROOT))
        self.reconstruction_dataset_combo.configure(values=names)
        if self.reconstruction_dataset_var.get() not in names:
            self.reconstruction_dataset_var.set(names[0] if names else '')
        if not names:
            self.reconstruction_status_var.set(
                'No completed datasets found under datasets/active_scan')
            return
        self.load_existing_reconstruction_outputs()

    def load_existing_reconstruction_outputs(
            self, _event=None, update_status=True) -> bool:
        """Expose saved diagnostic meshes independently of rebuild success."""
        self.reconstruction_report_path = None
        self.reconstruction_output_path = None
        self.reconstruction_view_button.configure(state='disabled')
        self.reconstruction_raw_view_button.configure(state='disabled')
        self.reconstruction_measured_view_button.configure(state='disabled')
        try:
            saved = existing_reconstruction_outputs(
                PROJECT_ROOT, self.reconstruction_dataset_var.get())
        except (OSError, TypeError, ValueError) as exc:
            if update_status:
                self.reconstruction_status_var.set(str(exc))
            return False
        if saved is None:
            if update_status:
                self.reconstruction_status_var.set(
                    'No existing reconstruction for the selected dataset')
            return False
        self.reconstruction_report_path = saved['report_path']
        self.reconstruction_output_path = saved['output_path']
        self.reconstruction_view_button.configure(state='normal')
        if saved['raw_output_path'] is not None:
            self.reconstruction_raw_view_button.configure(state='normal')
        if saved['measured_cloud_path'] is not None:
            self.reconstruction_measured_view_button.configure(state='normal')
        self.reconstruction_summary_var.set(
            quality_summary(saved['report']))
        if update_status:
            self.reconstruction_status_var.set(
                'Existing reconstruction loaded for inspection')
        return True

    def start_reconstruction(self) -> None:
        if (self.reconstruction_process is not None
                and self.reconstruction_process.poll() is None):
            self.reconstruction_status_var.set('Reconstruction is already running')
            return
        try:
            dimensions = tuple(
                float(variable.get())
                for variable in self.reconstruction_dimension_vars)
            command, output = reconstruction_command(
                PROJECT_ROOT, self.reconstruction_dataset_var.get(),
                dimensions, self.reconstruction_mode_var.get(),
                round(float(self.reconstruction_voxel_mm_var.get()), 1),
                self.reconstruction_mask_source_var.get())
        except (OSError, TypeError, ValueError) as exc:
            self.reconstruction_status_var.set(str(exc))
            return
        self.reconstruction_build_button.configure(state='disabled')
        self.reconstruction_view_button.configure(state='disabled')
        self.reconstruction_raw_view_button.configure(state='disabled')
        self.reconstruction_measured_view_button.configure(state='disabled')
        self.reconstruction_status_var.set(
            'Reconstruction is running offline at %.1f mm mesh voxels using %s'
            % (
                round(float(self.reconstruction_voxel_mm_var.get()), 1),
                self.reconstruction_mask_source_var.get()))
        self.reconstruction_output_path = output

        def worker():
            try:
                process = start_reconstruction_process(command)
                self.reconstruction_process = process
                output_text, _unused = process.communicate()
                report_path = output.with_suffix(output.suffix + '.quality.json')
                if process.returncode != 0:
                    raise RuntimeError(
                        (output_text or 'reconstruction failed').strip()[-2000:])
                report = load_quality_report(report_path)
                self.events.put(('reconstruction_complete', {
                    'success': True,
                    'status': 'Reconstruction completed',
                    'summary': quality_summary(report),
                    'report_path': report_path,
                    'output_path': output,
                    'raw_output_path': report.get('raw_mesh_path', ''),
                    'measured_cloud_path': report.get(
                        'measured_cloud_path', ''),
                }))
            except (OSError, RuntimeError, ValueError) as exc:
                self.events.put(('reconstruction_complete', {
                    'success': False,
                    'status': 'Reconstruction failed: %s' % exc,
                    'summary': 'No mesh was accepted.',
                }))

        threading.Thread(target=worker, daemon=True).start()

    def open_reconstruction_viewer(self, mesh_variant='cleaned') -> None:
        try:
            command = viewer_command(
                PROJECT_ROOT, self.reconstruction_report_path,
                show_input=bool(self.reconstruction_show_input_var.get()),
                mesh_variant=mesh_variant)
            start_viewer_process(command)
        except (OSError, TypeError, ValueError) as exc:
            self.reconstruction_status_var.set(str(exc))

    def _build_manual(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        ttk.Label(
            parent,
            text=(
                "Commissioning controls — these directly command an enabled arm "
                "and are not part of the autonomous mission workflow. Disable "
                "does not home the arm. If J6 is on the unfinished negative "
                "startup branch, first send J6 toward ready zero; that command "
                "moves only J6 in its configured positive direction."
            ),
            foreground="#8a3b12",
            wraplength=820,
            justify="left",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 12))

        joints_frame = ttk.Frame(parent)
        joints_frame.grid(row=1, column=0, sticky="nsew")
        joints_frame.columnconfigure(1, weight=1)

        for index, (name, low, high, unit) in enumerate(self.joints):
            var = tk.DoubleVar(value=0.0)
            self.vars.append(var)

            ttk.Label(joints_frame, text=name, width=10).grid(
                row=index, column=0, sticky="w", pady=6
            )
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
            ttk.Label(joints_frame, text=unit, width=4).grid(
                row=index, column=3, sticky="w", padx=(6, 0)
            )

        settings = ttk.Frame(parent)
        settings.grid(row=2, column=0, sticky="ew", pady=(18, 8))
        ttk.Label(settings, text="Speed").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(
            settings,
            from_=0,
            to=100,
            increment=1,
            textvariable=self.speed_var,
            width=8,
        ).grid(row=0, column=1, padx=(6, 18))
        ttk.Label(settings, text="Grip effort").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(
            settings,
            from_=0.5,
            to=3.0,
            increment=0.1,
            textvariable=self.effort_var,
            width=8,
        ).grid(row=0, column=3, padx=(6, 18))
        live_send = ttk.Checkbutton(
            settings, text="Live send", variable=self.send_live_var)
        live_send.grid(row=0, column=4, sticky="w")
        self.manual_motion_widgets.append(live_send)

        actions = ttk.Frame(parent)
        actions.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        send_button = ttk.Button(
            actions, text="Send Joint Target", command=self.send_target)
        send_button.grid(row=0, column=0, padx=(0, 8))
        self.manual_motion_widgets.append(send_button)
        ttk.Button(
            actions, text="Use Feedback", command=self.use_feedback
        ).grid(row=0, column=1, padx=8)
        ttk.Button(
            actions, text="Zero Target", command=self.zero_target
        ).grid(row=0, column=2, padx=8)

        home = ttk.LabelFrame(parent, text="Commissioning: staged home profile", padding=10)
        home.grid(row=4, column=0, sticky="ew", pady=(18, 0))
        ttk.Button(
            home,
            text="Record Rough / Ready Home",
            command=self.use_current_feedback_as_home,
        ).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(
            home,
            text="Record Current J6 as Storage",
            command=self.use_current_feedback_as_storage,
        ).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(
            home,
            text="Record Pre-Home (Shutdown Only)",
            command=self.use_current_feedback_as_pre_home,
        ).grid(row=0, column=2)
        ttk.Label(
            home,
            textvariable=self.home_status_var,
            foreground="#52606d",
            wraplength=760,
            justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

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

    def _build_diagnostics(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        ttk.Label(
            parent,
            text="Read-only mission and perception diagnostics",
            font=("TkDefaultFont", 14, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Label(
            parent,
            text=(
                "These values mirror production ROS topics for commissioning. "
                "They do not authorize motion, advance the mission, retry, or replan."
            ),
            foreground="#52606d",
            wraplength=820,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(0, 14))
        values = (
            ("Perception", self.perception_status_var),
            ("Tracking", self.tracking_status_var),
            ("Workflow", self.workflow_status_var),
            ("Capture", self.capture_status_var),
            ("Planning readiness", self.planning_status_var),
        )
        for row, (label, variable) in enumerate(values, start=2):
            frame = ttk.LabelFrame(parent, text=label, padding=10)
            frame.grid(row=row, column=0, sticky="ew", pady=4)
            ttk.Label(
                frame, textvariable=variable, wraplength=800,
                justify="left").grid(row=0, column=0, sticky="ew")

        ray_frame = ttk.LabelFrame(
            parent, text='Mission ray and robot replay', padding=10)
        ray_frame.grid(row=7, column=0, sticky='ew', pady=(14, 4))
        ray_frame.columnconfigure(1, weight=1)
        ttk.Label(ray_frame, text='Report').grid(
            row=0, column=0, sticky='w')
        self.ray_report_combo = ttk.Combobox(
            ray_frame, textvariable=self.ray_report_var,
            state='readonly', width=42)
        self.ray_report_combo.grid(
            row=0, column=1, sticky='ew', padx=(8, 8))
        ttk.Button(
            ray_frame, text='Refresh', command=self.refresh_ray_reports).grid(
                row=0, column=2)
        ttk.Button(
            ray_frame, text='Open Ray Review',
            command=self.open_selected_ray_report).grid(
                row=1, column=1, sticky='w', pady=(8, 0))
        ttk.Label(ray_frame, text='Historical dataset').grid(
            row=2, column=0, sticky='w', pady=(12, 0))
        self.ray_replay_dataset_combo = ttk.Combobox(
            ray_frame, textvariable=self.ray_replay_dataset_var,
            state='readonly', width=42)
        self.ray_replay_dataset_combo.grid(
            row=2, column=1, sticky='ew', padx=(8, 8), pady=(12, 0))
        self.ray_replay_button = ttk.Button(
            ray_frame, text='Replay Recorded Data',
            command=self.replay_selected_ray_dataset)
        self.ray_replay_button.grid(row=2, column=2, pady=(12, 0))
        ttk.Label(
            ray_frame, textvariable=self.ray_report_status_var,
            foreground='#52606d', wraplength=790, justify='left').grid(
                row=3, column=0, columnspan=3, sticky='ew', pady=(10, 0))
        self.refresh_ray_reports()

    def refresh_ray_reports(self, preferred_report='') -> None:
        reports = list_ray_reports(PROJECT_ROOT)
        previous_index = self.ray_report_combo.current()
        previous_report = (
            self.ray_report_paths[previous_index].parent.name
            if 0 <= previous_index < len(self.ray_report_paths) else '')
        self.ray_report_paths = tuple(reports)
        report_names = tuple(
            ray_report_display_name(path) for path in self.ray_report_paths)
        self.ray_report_combo.configure(values=report_names)
        selected_report = preferred_report or previous_report
        selected_index = next((
            index for index, path in enumerate(self.ray_report_paths)
            if path.parent.name == selected_report),
            0 if report_names else -1)
        if selected_index >= 0:
            self.ray_report_combo.current(selected_index)
        else:
            self.ray_report_var.set('')
        datasets = tuple(
            path.name for path in list_scan_datasets(PROJECT_ROOT))
        self.ray_replay_dataset_combo.configure(values=datasets)
        if self.ray_replay_dataset_var.get() not in datasets:
            self.ray_replay_dataset_var.set(datasets[0] if datasets else '')
        if not report_names:
            self.ray_report_status_var.set(
                'No ray reports yet. Run a mission or replay recorded data.')

    def open_selected_ray_report(self) -> None:
        try:
            selected_index = self.ray_report_combo.current()
            self.ray_review_process.open(
                ray_report_selection(
                    PROJECT_ROOT, self.ray_report_paths, selected_index))
        except (OSError, ValueError) as exc:
            self.ray_report_status_var.set(str(exc))
            return
        self.ray_report_status_var.set(
            'Ray Review opened %s' % self.ray_report_var.get())

    def replay_selected_ray_dataset(self) -> None:
        if self.ray_replay_in_progress:
            self.ray_report_status_var.set(
                'A command-free historical replay is already running')
            return
        selection = self.ray_replay_dataset_var.get()
        self.ray_replay_in_progress = True
        self.ray_replay_button.configure(state='disabled')
        self.ray_report_status_var.set(
            'Replaying recorded metadata only; ROS and arm commands are not used')

        def worker():
            try:
                result = replay_scan_dataset(PROJECT_ROOT, selection)
                self.events.put(('ray_replay_complete', result))
            except (OSError, TypeError, ValueError) as exc:
                self.events.put(('ray_replay_complete', {
                    'error': str(exc),
                }))

        threading.Thread(target=worker, daemon=True).start()

    def start_automated_scan(self):
        try:
            request = self.mission_view_model.begin_submission(
                [variable.get() for variable in self.rough_coordinate_vars],
                self.mission_label_var.get(),
            )
        except (RuntimeError, ValueError) as exc:
            self.mission_status_var.set(str(exc))
            return
        self.last_successful_mission = None
        self.report_base_home_button.configure(state="disabled")
        self.ros_node.disable_manual_command_publisher()
        self.set_manual_motion_enabled(False)
        self._render_mission_state()
        if not self.ros_node.submit_mission(request):
            self.mission_view_model.submission_failed(
                'the GUI ROS client already owns a mission request')
            self._restore_manual_controls_if_unowned()
            self._render_mission_state()

    def apply_scan_policy(self):
        if not self.mission_view_model.state.can_start:
            self.scan_policy_status_var.set(
                "Policy was not changed: wait for the active mission to finish.")
            return
        selected_label = self.scan_policy_var.get()
        label_to_policy = {
            label: policy for policy, label in POLICY_LABELS.items()}
        selected_policy = label_to_policy.get(selected_label)
        if selected_policy is None:
            self.scan_policy_status_var.set(
                "Policy was not changed: invalid selection.")
            return
        label_to_region = {
            label: region for region, label in RAY_REGION_LABELS.items()}
        selected_region = label_to_region.get(self.scan_ray_region_var.get())
        if selected_region is None:
            self.scan_policy_status_var.set(
                "Policy was not changed: invalid ray region.")
            return
        try:
            ray_count = int(self.scan_ray_count_var.get())
            write_scan_settings(
                self.scan_policy_path,
                selected_policy,
                selected_region,
                ray_count,
            )
        except (OSError, ValueError) as exc:
            self.scan_policy_status_var.set(
                "Policy was not changed: %s" % exc)
            return
        self.scan_policy_status_var.set(
            "Saved for next scan stack: %s; %s; %d rays"
            % (selected_label, self.scan_ray_region_var.get(), ray_count))

    def apply_floor_profile(self):
        if not self.mission_view_model.state.can_start:
            self.floor_profile_status_var.set(
                'Floor was not changed: wait for the active mission to finish.')
            return
        label_to_profile = {
            label: profile for profile, label in FLOOR_PROFILE_LABELS.items()}
        selected_profile = label_to_profile.get(self.floor_profile_var.get())
        if selected_profile is None:
            self.floor_profile_status_var.set(
                'Floor was not changed: invalid selection.')
            return
        try:
            environment = write_collision_environment(
                self.collision_environment_path, selected_profile)
        except (OSError, ValueError) as exc:
            self.floor_profile_status_var.set(
                'Floor was not changed: %s' % exc)
            return
        self.floor_profile_status_var.set(
            'Saved for next mission: %s; combined PiPER/L515/Bunker '
            'geometry remains active.'
            % FLOOR_PROFILE_LABELS[environment.floor_profile])

    def cancel_automated_scan(self):
        if self.ros_node.cancel_mission():
            self.mission_view_model.cancellation_requested()
            self._render_mission_state()

    def _render_mission_state(self):
        state = self.mission_view_model.state
        self.mission_status_var.set(state.status)
        self.mission_start_button.configure(
            state="normal" if state.can_start else "disabled")
        self.mission_cancel_button.configure(
            state="normal" if state.can_cancel else "disabled")
        self.disable_button.configure(
            state="disabled" if not state.can_start else "normal")
        self.enable_button.configure(
            state="disabled" if not state.can_start else "normal")
        self.scan_policy_combo.configure(
            state="readonly" if state.can_start else "disabled")
        self.scan_ray_region_combo.configure(
            state="readonly" if state.can_start else "disabled")
        self.scan_ray_count_spinbox.configure(
            state="normal" if state.can_start else "disabled")
        self.scan_policy_apply_button.configure(
            state="normal" if state.can_start else "disabled")
        self.floor_profile_combo.configure(
            state='readonly' if state.can_start else 'disabled')
        self.floor_profile_apply_button.configure(
            state='normal' if state.can_start else 'disabled')
        self.camera_resolution_combo.configure(
            state="readonly" if state.can_start else "disabled")
        self.camera_fps_combo.configure(
            state="readonly" if state.can_start else "disabled")
        self.camera_profile_apply_button.configure(
            state="normal" if state.can_start else "disabled")

    def _restore_manual_controls_if_unowned(self):
        publishers = self.ros_node.command_publisher_names(
            resolution_timeout_sec=0.5)
        if not publishers:
            self.ros_node.enable_manual_command_publisher()
            self.set_manual_motion_enabled(True)
            self.enable_button.configure(state="normal")
            self.disable_button.configure(state="normal")
            return
        self.ros_node.disable_manual_command_publisher()
        self.set_manual_motion_enabled(False)
        self.enable_button.configure(state="disabled")
        # An explicit disable cannot create robot motion. Keep it available so
        # an operator can fail-safe a stranded external controller, while the
        # local active-mission path above still routes through Cancel/Home.
        self.disable_button.configure(
            state="normal" if self.mission_view_model.state.can_start
            else "disabled")
        self.mission_status_var.set(
            self.mission_view_model.state.status
            + '; commissioning motion/enable remain locked while command '
            'publishers exist; explicit no-home Disable remains available: '
            + ', '.join(publishers))

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
            + '; pre-home J1-J6: '
            + ', '.join(
                '%.6f' % value for value in
                payload.get('pre_home_positions_rad', ()))
            + '; ready J6 %.6f; storage J6 %.6f; staged=%s'
            % (
                float(payload['mission_ready_joint6_rad']),
                float(payload['storage_joint6_rad']),
                ('yes; startup increasing to zero'
                 if payload.get('staged_home_configured') else 'no')))

    def use_current_feedback_as_home(self):
        if not self.mission_view_model.state.can_start:
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
            existing_pre_home = (
                list(existing['pre_home_positions_rad'])
                if existing is not None
                and existing.get('pre_home_configured') else None)
            save_home_pose(
                HOME_POSE_PATH,
                positions.tolist(),
                observed_positions=observed_positions.tolist(),
                mission_ready_joint6_rad=float(positions[5]),
                storage_joint6_rad=existing_storage,
                staged_home_configured=existing_storage is not None,
                pre_home_positions_rad=existing_pre_home,
                pre_home_configured=existing_pre_home is not None,
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
        if not self.mission_view_model.state.can_start:
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
                pre_home_positions_rad=profile.get(
                    'pre_home_positions_rad'),
                pre_home_configured=bool(profile.get(
                    'pre_home_configured', False)),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.home_status_var.set(
                'Could not save storage J6: %s' % exc)
            return
        self.load_selected_home()

    def use_current_feedback_as_pre_home(self):
        if not self.mission_view_model.state.can_start:
            self.home_status_var.set(
                'Pre-home cannot change while an automatic mission is active')
            return
        feedback_age = (
            math.inf if self.ros_node.latest_feedback_monotonic is None
            else time.monotonic() - self.ros_node.latest_feedback_monotonic)
        if (
                self.feedback_positions is None
                or len(self.feedback_positions) < 6
                or feedback_age > 1.0):
            self.home_status_var.set(
                'Fresh six-joint feedback is required to record pre-home')
            return
        try:
            pre_home = energized_hold_target(self.feedback_positions[:6])
        except (TypeError, ValueError):
            self.home_status_var.set(
                'Current pre-home feedback is not six finite positions')
            return
        if np.any(pre_home < URDF_JOINT_LIMITS[:, 0]) or np.any(
                pre_home > URDF_JOINT_LIMITS[:, 1]):
            self.home_status_var.set(
                'Current pre-home is outside planning joint limits')
            return
        try:
            profile = load_home_pose(HOME_POSE_PATH)
            if profile is None:
                raise ValueError(
                    'record the rough / mission-ready home first')
            validate_home_profile_limits(profile, URDF_JOINT_LIMITS)
            save_home_pose(
                HOME_POSE_PATH,
                profile['positions_rad'],
                observed_positions=profile.get(
                    'observed_disabled_positions_rad'),
                mission_ready_joint6_rad=profile[
                    'mission_ready_joint6_rad'],
                storage_joint6_rad=profile['storage_joint6_rad'],
                staged_home_configured=bool(profile.get(
                    'staged_home_configured', False)),
                pre_home_positions_rad=pre_home.tolist(),
                pre_home_configured=True,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.home_status_var.set(
                'Could not save pre-home: %s' % exc)
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
            ttk.Label(
                parent, text=label, font=("TkDefaultFont", 10, "bold")
            ).grid(
                row=row * 2,
                column=0,
                sticky="w",
                pady=(0 if row == 0 else 14, 2),
            )
            ttk.Label(
                parent, textvariable=var, wraplength=250, justify="left"
            ).grid(row=row * 2 + 1, column=0, sticky="ew")

    def set_manual_motion_enabled(self, enabled: bool) -> None:
        self.send_live_var.set(False)
        for widget in self.manual_motion_widgets:
            widget.configure(state="normal" if enabled else "disabled")

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
        """Request explicit commissioning disable without motion or homing."""
        if not self.mission_view_model.state.can_start:
            self.service_text.set(
                "Disable is owned by the active production mission; use "
                "Cancel and Home and wait for its terminal result.")
            return
        if self.safe_disable_in_progress:
            self.service_text.set(
                "Commissioning disable is already in progress.")
            return

        self.safe_disable_in_progress = True
        self.disable_button.configure(state="disabled")
        self.service_text.set(
            "Commissioning disable requested directly. No hold target or home "
            "motion is being commanded; support the arm if it can fall.")
        self.ros_node.call_enable_async(False)

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
            self.preview_status_var.set(
                "No 3D preview is available; open and load the editor first."
            )
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
            self.preview_status_var.set(
                "Arm status is missing or reports an error; motion refused."
            )
            return
        speed = clamp(float(self.preview_speed_var.get()), 1.0, 10.0)
        gripper = (
            self.feedback_positions[6]
            if len(self.feedback_positions) >= 7
            else self.vars[6].get()
        )
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
        effort = clamp(float(self.effort_var.get()), 0.5, 3.0)
        self.ros_node.publish_joint_target(positions, speed, effort)
        self.preview_status_var.set(
            f"3D preview target sent at {speed:.0f}%. Monitor the live arm and feedback."
        )

    def drain_events(self) -> None:
        try:
            while True:
                name, payload = self.events.get_nowait()
                if name == "feedback":
                    self.feedback_positions = list(payload.position)
                    self.feedback_text.set(", ".join(
                        "%.3f" % value for value in self.feedback_positions[:7]))
                elif name == "preview":
                    self.preview_positions = list(payload.position)
                    self.preview_angle_var.set(", ".join(
                        "J%d %.3f" % (index + 1, value)
                        for index, value in enumerate(
                            self.preview_positions[:6])))
                elif name == "status":
                    self.status_text.set(
                        "domain %s | mode %s | arm %s | err %s | %s" % (
                            os.environ.get('ROS_DOMAIN_ID', 'default'),
                            payload.ctrl_mode,
                            payload.arm_status,
                            payload.err_code,
                            self.bounds_message,
                        ))
                elif name == "command":
                    positions, speed, effort = payload
                    self.command_text.set(
                        "%s\nspeed %.0f, effort %.1f" % (
                            ", ".join("%.3f" % value for value in positions),
                            speed,
                            effort,
                        ))
                elif name == "service":
                    self.service_text.set(str(payload))
                elif name == "enable_service":
                    enabled, success, message = payload
                    self.service_text.set(str(message))
                    if not enabled and self.safe_disable_in_progress:
                        self.safe_disable_in_progress = False
                        self.disable_button.configure(state="normal")
                        self.service_text.set(
                            str(message)
                            + ("; all-six feedback-proved commissioning "
                               "disable completed without homing" if success
                               else "; disable failed; motors may remain "
                               "enabled"))
                elif name == "command_blocked":
                    self.command_text.set(str(payload))
                elif name == "mission_client":
                    self._handle_mission_client_event(payload)
                elif name == "heavy_status":
                    self.perception_status_var.set(
                        self._format_json_status(payload, "heavy refresh"))
                elif name == "workflow_status":
                    self.workflow_status_var.set(
                        self._format_json_status(payload, "workflow"))
                elif name == "capture_status":
                    self.capture_status_var.set(
                        self._format_json_status(payload, "capture"))
                elif name == "tracking_health":
                    self.tracking_status_var.set(
                        "%s | settled=%s | prediction=%s | age=%.3fs | %s" % (
                            payload.lifecycle_state,
                            bool(payload.camera_settled),
                            bool(payload.prediction_only),
                            float(payload.measurement_age_sec),
                            payload.reason,
                        ))
                elif name == "tesseract_readiness":
                    blockers = list(payload.multiview_blockers)
                    self.planning_status_var.set(
                        "worker=%s | acquisition=%s | multiview=%s%s" % (
                            bool(payload.worker_ready),
                            bool(payload.acquisition_ready),
                            bool(payload.multiview_ready),
                            (" | " + "; ".join(blockers)) if blockers else "",
                        ))
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
                            payload.state,
                            payload.reason,
                            ('; mesh=' + payload.mesh_path)
                            if payload.mesh_path else '',
                        ))
                    if payload.state == 'FAILED' and self.last_successful_mission:
                        self.report_base_home_button.configure(state='disabled')
                elif name == 'reconstruction_complete':
                    self.reconstruction_process = None
                    self.reconstruction_build_button.configure(state='normal')
                    self.reconstruction_status_var.set(str(payload['status']))
                    self.reconstruction_summary_var.set(str(payload['summary']))
                    if bool(payload.get('success')):
                        self.reconstruction_report_path = payload['report_path']
                        self.reconstruction_output_path = Path(
                            payload['output_path'])
                        self.reconstruction_view_button.configure(state='normal')
                        raw_mesh_path = str(payload.get(
                            'raw_output_path', ''))
                        if raw_mesh_path and Path(raw_mesh_path).is_file():
                            self.reconstruction_raw_view_button.configure(
                                state='normal')
                        measured_path = str(payload.get(
                            'measured_cloud_path', ''))
                        if measured_path and Path(measured_path).is_file():
                            self.reconstruction_measured_view_button.configure(
                                state='normal')
                    elif self.load_existing_reconstruction_outputs(
                            update_status=False):
                        self.reconstruction_status_var.set(
                            '%s; previous saved outputs remain available'
                            % payload['status'])
                elif name == 'ray_replay_complete':
                    self.ray_replay_in_progress = False
                    self.ray_replay_button.configure(state='normal')
                    if payload.get('error'):
                        self.ray_report_status_var.set(
                            'Historical replay failed: %s' % payload['error'])
                    else:
                        report_name = Path(payload['html_path']).parent.name
                        self.refresh_ray_reports(
                            preferred_report=report_name)
                        self.ray_report_status_var.set(
                            'Replayed %d accepted captures from %s; %d skipped'
                            % (
                                int(payload['replayed_capture_count']),
                                payload['dataset'],
                                int(payload['skipped_capture_count']),
                            ))
        except queue.Empty:
            pass
        self.root.after(100, self.drain_events)

    @staticmethod
    def _format_json_status(payload, fallback):
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            return "%s: malformed status" % fallback
        state = str(value.get('state', 'unknown'))
        reason = str(value.get('reason', value.get('message', '')))
        return "%s%s" % (state, (": " + reason) if reason else "")

    def _handle_mission_client_event(self, event: MissionClientEvent):
        if event.kind == "accepted":
            self.mission_view_model.goal_accepted()
        elif event.kind == "feedback":
            self.mission_view_model.apply_feedback(event.payload)
        elif event.kind == "cancel_requested":
            self.mission_view_model.cancellation_requested()
        elif event.kind == "cancel_unavailable":
            self.mission_status_var.set(str(event.payload))
            return
        elif event.kind == "submission_failed":
            self.mission_view_model.submission_failed(event.payload)
            self._render_mission_state()
            self._restore_manual_controls_if_unowned()
            return
        elif event.kind == "result":
            result = event.payload
            self.mission_view_model.apply_result(result)
            self.last_successful_mission = result.reconstruction_payload
            if self.last_successful_mission is not None:
                self.mesh_status_var.set(
                    'Capture complete and arm shut down; report when the '
                    'tracked robot is home to build mesh %s'
                    % result.mesh_job_id)
                self.report_base_home_button.configure(state='normal')
            self._render_mission_state()
            self.refresh_ray_reports(preferred_report=result.task_id)
            self._restore_manual_controls_if_unowned()
            return
        self._render_mission_state()

    def shutdown(self) -> None:
        if not self.mission_view_model.state.can_start:
            self.ros_node.cancel_mission()
        self.close_3d_joint_editor()
        self.ray_review_process.shutdown()
        if (self.reconstruction_process is not None
                and self.reconstruction_process.poll() is None):
            self.reconstruction_process.terminate()


def main() -> None:
    run_gui(
        ros_runtime=rclpy,
        node_factory=PiperGuiRos,
        app_factory=PiperGuiApp,
        root_factory=tk.Tk,
        executor_factory=lambda: MultiThreadedExecutor(num_threads=2),
    )


if __name__ == "__main__":
    main()
