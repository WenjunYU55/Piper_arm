#!/usr/bin/env python3

import json
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from piper_msgs.msg import PiperStatusMsg
from piper_msgs.srv import Enable


DEFAULT_JOINTS = [
    ("joint1", -2.8, 2.8, "rad"),
    ("joint2", -2.1, 2.1, "rad"),
    ("joint3", -2.8, 2.8, "rad"),
    ("joint4", -2.8, 2.8, "rad"),
    ("joint5", -2.1, 2.1, "rad"),
    ("joint6", -2.8, 2.8, "rad"),
    ("gripper", 0.0, 0.08, "m"),
]

BOUNDS_PATH = os.path.join(os.path.dirname(__file__), "piper_joint_bounds.json")


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
        if record is None:
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
        self.latest_status = None

        self.joint_pub = self.create_publisher(JointState, "/joint_ctrl_single", 10)
        self.feedback_sub = self.create_subscription(
            JointState, "/joint_states_single", self.feedback_callback, 10
        )
        self.status_sub = self.create_subscription(
            PiperStatusMsg, "/arm_status", self.status_callback, 10
        )
        self.enable_client = self.create_client(Enable, "/enable_srv")

    def feedback_callback(self, msg: JointState) -> None:
        self.latest_feedback = msg
        self.events.put(("feedback", msg))

    def status_callback(self, msg: PiperStatusMsg) -> None:
        self.latest_status = msg
        self.events.put(("status", msg))

    def publish_joint_target(self, positions: List[float], speed: float, effort: float) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "piper_native_gui"
        msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        msg.position = positions
        msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, speed]
        msg.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, effort]
        self.joint_pub.publish(msg)
        self.events.put(("command", (positions, speed, effort)))

    def call_enable_async(self, enabled: bool) -> None:
        thread = threading.Thread(target=self._call_enable, args=(enabled,), daemon=True)
        thread.start()

    def _call_enable(self, enabled: bool) -> None:
        if not self.enable_client.wait_for_service(timeout_sec=1.0):
            self.events.put(("service", f"/enable_srv unavailable"))
            return

        req = Enable.Request()
        req.enable_request = enabled
        future = self.enable_client.call_async(req)
        deadline = time.time() + 20.0

        while rclpy.ok() and not future.done() and time.time() < deadline:
            time.sleep(0.05)

        if not future.done():
            self.events.put(("service", f"{'enable' if enabled else 'disable'} timeout"))
            return

        result = future.result()
        self.events.put(("service", f"{'enable' if enabled else 'disable'} -> {result.enable_response}"))


class PiperGuiApp:
    def __init__(self, root: tk.Tk, ros_node: PiperGuiRos, events: "queue.Queue[tuple]") -> None:
        self.root = root
        self.ros_node = ros_node
        self.events = events
        self.vars: List[tk.DoubleVar] = []
        self.feedback_positions = None
        self.joints, self.bounds_message = load_joint_limits()

        self.root.title("PiPER Control")
        self.root.geometry("980x650")
        self.root.minsize(860, 560)

        self.speed_var = tk.DoubleVar(value=30.0)
        self.effort_var = tk.DoubleVar(value=1.0)
        self.send_live_var = tk.BooleanVar(value=False)
        self.last_live_publish = 0.0

        self.status_text = tk.StringVar(
            value=f"ROS domain {os.environ.get('ROS_DOMAIN_ID', 'default')} | {self.bounds_message}"
        )
        self.feedback_text = tk.StringVar(value="No feedback")
        self.command_text = tk.StringVar(value="No command sent")
        self.service_text = tk.StringVar(value="No service call")

        self._build()
        self.root.after(100, self.drain_events)

    def _build(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=(14, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        title = ttk.Label(header, text="PiPER Control", font=("TkDefaultFont", 16, "bold"))
        title.grid(row=0, column=0, sticky="w")
        ttk.Button(header, text="Enable", command=lambda: self.ros_node.call_enable_async(True)).grid(row=0, column=1, padx=4)
        ttk.Button(header, text="Disable", command=lambda: self.ros_node.call_enable_async(False)).grid(row=0, column=2, padx=4)

        body = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        body.grid(row=1, column=0, sticky="nsew")

        notebook = ttk.Notebook(body)
        body.add(notebook, weight=4)

        manual = ttk.Frame(notebook, padding=14)
        notebook.add(manual, text="Manual")
        self._build_manual(manual)

        graphical = ttk.Frame(notebook, padding=14)
        notebook.add(graphical, text="Graphical")
        self._build_graphical(graphical)

        side = ttk.Frame(body, padding=14)
        body.add(side, weight=1)
        self._build_status(side)

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
        ttk.Checkbutton(settings, text="Live send", variable=self.send_live_var).grid(row=0, column=4, sticky="w")

        actions = ttk.Frame(parent)
        actions.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(actions, text="Send Joint Target", command=self.send_target).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(actions, text="Use Feedback", command=self.use_feedback).grid(row=0, column=1, padx=8)
        ttk.Button(actions, text="Zero Target", command=self.zero_target).grid(row=0, column=2, padx=8)

    def _build_graphical(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        canvas = tk.Canvas(parent, background="#f8faf9", highlightthickness=1, highlightbackground="#b9c2c7")
        canvas.grid(row=0, column=0, sticky="nsew")
        canvas.create_text(
            320,
            220,
            text="Graphical arm dragging will be added after model dimensions/calibration are available.",
            width=460,
            fill="#3b464d",
            font=("TkDefaultFont", 13, "bold"),
        )

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

    def use_feedback(self) -> None:
        if not self.feedback_positions or len(self.feedback_positions) < 7:
            self.feedback_text.set("No feedback to load")
            return
        for index, value in enumerate(self.feedback_positions[:7]):
            self.vars[index].set(float(value))

    def zero_target(self) -> None:
        for var in self.vars:
            var.set(0.0)

    def drain_events(self) -> None:
        try:
            while True:
                name, payload = self.events.get_nowait()
                if name == "feedback":
                    self.feedback_positions = list(payload.position)
                    shown = ", ".join(f"{value:.3f}" for value in self.feedback_positions[:7])
                    self.feedback_text.set(shown)
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
        except queue.Empty:
            pass
        self.root.after(100, self.drain_events)


def main() -> None:
    events: "queue.Queue[tuple]" = queue.Queue()
    rclpy.init()
    ros_node = PiperGuiRos(events)

    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    root = tk.Tk()
    PiperGuiApp(root, ros_node, events)
    try:
        root.mainloop()
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
