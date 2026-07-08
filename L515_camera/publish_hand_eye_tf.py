#!/usr/bin/env python3
"""Publish the calibrated eye-in-hand camera pose from PiPER joint feedback."""

import argparse
import sys
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
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


class HandEyeTfPublisher(Node):
    def __init__(self, calibration, joint_topic, base_frame, camera_frame, calibration_frame):
        super().__init__("hand_eye_tf_publisher")
        self.link6_from_camera = load_accepted_calibration(calibration)
        self.fk = PiperModifiedDhFk()
        self.base_frame = base_frame
        self.camera_frame = camera_frame
        self.calibration_frame = calibration_frame
        self.wait_warning_emitted = False
        self.broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.subscription = self.create_subscription(JointState, joint_topic, self.callback, 10)
        self.get_logger().info(
            "Publishing %s -> %s from %s using %s calibrated in %s"
            % (base_frame, camera_frame, joint_topic, calibration, calibration_frame)
        )

    def callback(self, message):
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
        if self.camera_frame == self.calibration_frame:
            camera_from_calibration_frame = np.eye(4)
        else:
            try:
                camera_from_calibration_frame = message_transform(
                    self.tf_buffer.lookup_transform(
                        self.camera_frame, self.calibration_frame, rclpy.time.Time()
                    )
                )
            except TransformException as error:
                if not self.wait_warning_emitted:
                    self.get_logger().warning("waiting for camera static TF: %s" % error)
                    self.wait_warning_emitted = True
                return
        self.wait_warning_emitted = False
        base_from_camera = base_from_calibration_frame @ inverse(camera_from_calibration_frame)
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
    try:
        rclpy.init()
        node = HandEyeTfPublisher(
            args.calibration,
            args.joint_topic,
            args.base_frame,
            args.camera_frame,
            args.calibration_frame,
        )
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
        return 0
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print("hand-eye TF publisher error: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
