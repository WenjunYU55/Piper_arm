#!/usr/bin/env python3

import argparse
import json
import os
import threading
import time
from typing import Dict, List, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


DEFAULT_OUTPUT = "piper_joint_bounds.json"

# Default order requested by the operator. The index is the position index
# from /joint_states_single.
DEFAULT_STEPS: List[Tuple[str, str, int, str]] = [
    ("base rotate", "joint1", 0, "rad"),
    ("base bend", "joint2", 1, "rad"),
    ("elbow", "joint3", 2, "rad"),
    ("rotating wrist", "joint4", 3, "rad"),
    ("wrist", "joint5", 4, "rad"),
    ("rotating_gripper", "joint6", 5, "rad"),
    ("gripper", "joint7", 6, "m"),
]

ALL_GUI_STEPS: List[Tuple[str, str, int, str]] = [
    ("base rotate", "joint1", 0, "rad"),
    ("base bend", "joint2", 1, "rad"),
    ("elbow", "joint3", 2, "rad"),
    ("rotating wrist", "joint4", 3, "rad"),
    ("wrist", "joint5", 4, "rad"),
    ("rotating gripper", "joint6", 5, "rad"),
    ("gripper", "joint7", 6, "m"),
]


class JointFeedbackRecorder(Node):
    def __init__(self) -> None:
        super().__init__("piper_bounds_calibrator")
        self.latest_msg = None
        self.lock = threading.Lock()
        self.sub = self.create_subscription(
            JointState, "/joint_states_single", self.feedback_callback, 10
        )

    def feedback_callback(self, msg: JointState) -> None:
        with self.lock:
            self.latest_msg = msg

    def latest_positions(self) -> List[float]:
        with self.lock:
            if self.latest_msg is None:
                return []
            return list(self.latest_msg.position)


def wait_for_feedback(node: JointFeedbackRecorder, timeout: float) -> List[float]:
    deadline = time.time() + timeout
    while rclpy.ok() and time.time() < deadline:
        positions = node.latest_positions()
        if positions:
            return positions
        time.sleep(0.05)
    raise TimeoutError("No /joint_states_single feedback received. Start PiPER first.")


def load_existing(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_bounds(path: str, bounds: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(bounds, handle, indent=2, sort_keys=True)
        handle.write("\n")


def record_point(
    node: JointFeedbackRecorder,
    label: str,
    joint_name: str,
    index: int,
    bound_name: str,
) -> float:
    print(f"Move the {label} to its {bound_name.upper()} limit.")
    input("Press Enter after the robot is holding that position: ")
    positions = node.latest_positions()
    if index >= len(positions):
        raise IndexError(
            f"Feedback has {len(positions)} positions; cannot read index {index} for {joint_name}."
        )
    value = float(positions[index])
    print(f"  recorded {joint_name} {bound_name} sample = {value:.6f}")
    return value


def build_bounds(
    node: JointFeedbackRecorder,
    steps: List[Tuple[str, str, int, str]],
    existing: Dict,
) -> Dict:
    bounds = dict(existing)
    bounds.setdefault("version", 1)
    bounds["source_topic"] = "/joint_states_single"
    bounds["recorded_at_unix"] = time.time()
    bounds.setdefault("joints", {})

    for label, joint_name, index, unit in steps:
        print("")
        print(f"=== {label} -> {joint_name} ===")
        min_sample = record_point(node, label, joint_name, index, "min")
        max_sample = record_point(node, label, joint_name, index, "max")
        low = min(min_sample, max_sample)
        high = max(min_sample, max_sample)
        bounds["joints"][joint_name] = {
            "label": label,
            "index": index,
            "unit": unit,
            "min": low,
            "max": high,
            "samples": {
                "requested_min": min_sample,
                "requested_max": max_sample,
            },
        }
        print(f"  bounds for {joint_name}: min={low:.6f}, max={high:.6f}")

    return bounds


def main() -> int:
    parser = argparse.ArgumentParser(description="Record PiPER joint bounds from live feedback.")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), DEFAULT_OUTPUT),
        help="Path to write bounds JSON.",
    )
    parser.add_argument(
        "--all-seven",
        action="store_true",
        help="Record all seven GUI controls. This is already the default and is kept for compatibility.",
    )
    args = parser.parse_args()

    rclpy.init()
    node = JointFeedbackRecorder()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        print("Waiting for /joint_states_single feedback...")
        positions = wait_for_feedback(node, timeout=10.0)
        print(f"Feedback received with {len(positions)} positions.")
        print("This tool does not move the arm. You move each joint manually.")
        print("Each joint will be recorded twice: first MIN, then MAX.")

        existing = load_existing(args.output)
        steps = ALL_GUI_STEPS if args.all_seven else DEFAULT_STEPS
        bounds = build_bounds(node, steps, existing)
        save_bounds(args.output, bounds)
        print("")
        print(f"Saved bounds to {args.output}")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    raise SystemExit(main())
