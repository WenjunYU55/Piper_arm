#!/usr/bin/env python3
"""Publish the calibrated eye-in-hand camera pose from PiPER joint feedback."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener
import yaml

from solve_hand_eye import PiperModifiedDhFk


def load_accepted_calibration(path):
    with path.open() as stream:
        data = yaml.safe_load(stream)
    if data.get("status") != "accepted":
        raise RuntimeError("refusing to publish calibration with status %r" % data.get("status"))
    entry = data["camera_to_link6"]
    if "matrix" in entry:
        value = np.asarray(entry["matrix"], dtype=float)
    else:
        value = np.eye(4)
        value[:3, :3] = Rotation.from_quat(entry["quaternion_xyzw"]).as_matrix()
        value[:3, 3] = entry["translation_m"]
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise RuntimeError("camera_to_link6 must be a finite 4x4 transform")
    return value


def message_transform(message):
    translation = message.transform.translation
    quaternion = message.transform.rotation
    value = np.eye(4)
    value[:3, :3] = Rotation.from_quat(
        [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    ).as_matrix()
    value[:3, 3] = [translation.x, translation.y, translation.z]
    return value


def inverse(value):
    rotation = value[:3, :3].T
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = -rotation @ value[:3, 3]
    return result


def wait_for_static_camera_transform(camera_frame, calibration_frame, timeout_sec=10.0):
    """Read the fixed RealSense frame relationship before starting joint I/O."""
    node = Node("hand_eye_static_tf_loader")
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)
    deadline = time.monotonic() + timeout_sec
    last_error = None
    result = None
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        try:
            result = message_transform(
                tf_buffer.lookup_transform(
                    camera_frame, calibration_frame, rclpy.time.Time()
                )
            )
            break
        except TransformException as error:
            last_error = error
    # TransformListener owns its subscriptions and must be released before the
    # temporary node is destroyed.
    del tf_listener
    node.destroy_node()
    if result is None:
        raise RuntimeError(
            "camera static TF %s -> %s unavailable after %.1fs: %s"
            % (camera_frame, calibration_frame, timeout_sec, last_error)
        )
    return result


class HandEyeTfPublisher(Node):
    def __init__(
        self,
        calibration,
        joint_topic,
        base_frame,
        camera_frame,
        calibration_frame,
        camera_from_calibration_frame,
    ):
        super().__init__("hand_eye_tf_publisher")
        self.link6_from_camera = load_accepted_calibration(calibration)
        self.fk = PiperModifiedDhFk()
        self.base_frame = base_frame
        self.camera_frame = camera_frame
        self.calibration_frame = calibration_frame
        self.joint_callback_seen = False
        self.last_joint_at = None
        self.last_joint_subscription_at = time.monotonic()
        self.camera_from_calibration_frame = camera_from_calibration_frame
        self.broadcaster = TransformBroadcaster(self)
        # PiPER publishes reliable feedback at roughly 200 Hz while this
        # callback performs FK and a TF lookup. Match that reliability contract
        # but retain only the newest pose so feedback cannot queue behind arm
        # motion.
        self.joint_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.joint_topic = joint_topic
        self.subscription = None
        self.recreate_joint_subscription(initial=True)
        self.create_timer(0.5, self.check_joint_subscription)
        self.get_logger().info(
            "Publishing %s -> %s from %s using %s calibrated in %s"
            % (base_frame, camera_frame, joint_topic, calibration, calibration_frame)
        )

    def recreate_joint_subscription(self, initial=False):
        if self.subscription is not None:
            self.destroy_subscription(self.subscription)
        self.subscription = self.create_subscription(
            JointState, self.joint_topic, self.callback, self.joint_qos)
        self.last_joint_subscription_at = time.monotonic()
        if not initial:
            self.get_logger().warning(
                "Recreated stale PiPER joint-feedback subscription; calibrated "
                "camera TF remains unavailable until a fresh sample arrives"
            )

    def check_joint_subscription(self):
        now = time.monotonic()
        joint_stale = self.last_joint_at is None or now - self.last_joint_at > 1.0
        if joint_stale and now - self.last_joint_subscription_at >= 2.0:
            self.recreate_joint_subscription()

    def callback(self, message):
        self.last_joint_at = time.monotonic()
        if not self.joint_callback_seen:
            self.joint_callback_seen = True
            self.get_logger().info("Received first PiPER joint feedback sample")
        positions_by_name = dict(zip(message.name, message.position))
        names = ["joint%d" % index for index in range(1, 7)]
        if all(name in positions_by_name for name in names):
            positions = [positions_by_name[name] for name in names]
        elif len(message.position) >= 6:
            positions = list(message.position[:6])
        else:
            self.get_logger().warning("joint state contains fewer than six arm joints")
            return
        base_from_calibration_frame = self.fk.calculate(positions) @ self.link6_from_camera
        base_from_camera = (
            base_from_calibration_frame
            @ inverse(self.camera_from_calibration_frame)
        )
        quaternion = Rotation.from_matrix(base_from_camera[:3, :3]).as_quat()
        outgoing = TransformStamped()
        outgoing.header.stamp = message.header.stamp
        if outgoing.header.stamp.sec == 0 and outgoing.header.stamp.nanosec == 0:
            outgoing.header.stamp = self.get_clock().now().to_msg()
        outgoing.header.frame_id = self.base_frame
        outgoing.child_frame_id = self.camera_frame
        outgoing.transform.translation.x = float(base_from_camera[0, 3])
        outgoing.transform.translation.y = float(base_from_camera[1, 3])
        outgoing.transform.translation.z = float(base_from_camera[2, 3])
        outgoing.transform.rotation.x = float(quaternion[0])
        outgoing.transform.rotation.y = float(quaternion[1])
        outgoing.transform.rotation.z = float(quaternion[2])
        outgoing.transform.rotation.w = float(quaternion[3])
        self.broadcaster.sendTransform(outgoing)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--joint-topic", default="/joint_states_single")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default="camera_link")
    parser.add_argument("--calibration-frame", default="camera_color_optical_frame")
    args = parser.parse_args()
    node = None
    executor = None
    try:
        rclpy.init()
        camera_from_calibration_frame = wait_for_static_camera_transform(
            args.camera_frame, args.calibration_frame)
        node = HandEyeTfPublisher(
            args.calibration,
            args.joint_topic,
            args.base_frame,
            args.camera_frame,
            args.calibration_frame,
            camera_from_calibration_frame,
        )
        # The static RealSense frame relationship was frozen before this node
        # started, leaving only joint feedback and TF publication active.
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
        return 0
    except KeyboardInterrupt:
        return 130
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print("hand-eye TF publisher error: %s" % error, file=sys.stderr)
        return 1
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
