#!/usr/bin/env python3
"""Capture one validated Boston Dynamics ChArUco hand-eye sample."""

import argparse
from collections import deque
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState

from validate_fixed_board import joint_positions_are_stable


DEFAULT_SQUARE_LENGTH_M = 0.017
DEFAULT_MARKER_LENGTH_M = 0.012


def stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def stamp_dict(stamp):
    return {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)}


def nearest_stamped_joint(samples, image_stamp_seconds, maximum_delta_seconds=0.10):
    """Return the joint message nearest an image stamp, or fail closed."""
    if not np.isfinite(image_stamp_seconds) or image_stamp_seconds <= 0.0:
        return None, None
    candidates = [
        (abs(source_stamp - image_stamp_seconds), message)
        for source_stamp, _received_at, message in samples
        if np.isfinite(source_stamp) and source_stamp > 0.0
    ]
    if not candidates:
        return None, None
    delta, message = min(candidates, key=lambda item: item[0])
    if delta > maximum_delta_seconds:
        return None, float(delta)
    return message, float(delta)


class HandEyeCapture(Node):
    def __init__(self, output_root, timeout, squares_x, squares_y, square_length_m,
                 marker_length_m, dictionary_name, stationary_duration_sec,
                 stationary_position_tolerance_rad):
        super().__init__("hand_eye_sample_capture")
        self.output_root = output_root
        self.timeout = timeout
        self.started = time.monotonic()
        self.bridge = CvBridge()
        self.image = None
        self.info = None
        self.joints = None
        self.end_pose = None
        self.image_received = None
        self.info_received = None
        self.joints_received = None
        self.end_pose_received = None
        self.saved = None
        self.error = None
        self.last_rejection = None
        self.last_checked_image_stamp = None
        self.joint_history = deque()
        self.stamped_joint_history = deque()
        self.stationary_duration_sec = stationary_duration_sec
        self.stationary_position_tolerance_rad = stationary_position_tolerance_rad
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length_m = square_length_m
        self.marker_length_m = marker_length_m
        self.dictionary_name = dictionary_name
        dictionary_id = getattr(cv2.aruco, dictionary_name)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.board = cv2.aruco.CharucoBoard_create(
            squares_x, squares_y, square_length_m, marker_length_m, self.dictionary
        )
        self.expected_marker_ids = sorted(self.board.ids.flatten().astype(int).tolist())
        self.expected_corner_count = (squares_x - 1) * (squares_y - 1)
        self.create_subscription(Image, "/camera/color/image_raw", self.image_cb, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/camera/color/camera_info", self.info_cb, qos_profile_sensor_data)
        self.create_subscription(JointState, "/joint_states_single", self.joints_cb, 10)
        self.create_subscription(Pose, "/end_pose", self.end_pose_cb, 10)

    def image_cb(self, msg):
        self.image = msg
        self.image_received = time.monotonic()

    def info_cb(self, msg):
        self.info = msg
        self.info_received = time.monotonic()

    def joints_cb(self, msg):
        self.joints = msg
        now = time.monotonic()
        self.joints_received = now
        positions = np.asarray(msg.position[:6], dtype=float)
        if positions.size == 6 and np.all(np.isfinite(positions)):
            self.joint_history.append((now, positions.copy()))
            source_stamp = stamp_seconds(msg.header.stamp)
            self.stamped_joint_history.append((source_stamp, now, msg))
            cutoff = now - max(2.0 * self.stationary_duration_sec, 1.0)
            while self.joint_history and self.joint_history[0][0] < cutoff:
                self.joint_history.popleft()
            while (self.stamped_joint_history
                   and self.stamped_joint_history[0][1] < cutoff):
                self.stamped_joint_history.popleft()

    def end_pose_cb(self, msg):
        self.end_pose = msg
        self.end_pose_received = time.monotonic()

    def tick(self):
        if self.saved or self.error:
            return
        if time.monotonic() - self.started > self.timeout:
            if self.last_rejection:
                self.error = "timed out without a valid sample; last rejection: " + self.last_rejection
            else:
                self.error = "timed out waiting for synchronized image, camera_info, and joint state"
            return
        if self.image is None or self.info is None or self.joints is None or self.end_pose is None:
            return
        receipt_times = [self.image_received, self.info_received, self.joints_received, self.end_pose_received]
        if max(receipt_times) - min(receipt_times) > 0.10:
            return
        image_stamp = (self.image.header.stamp.sec, self.image.header.stamp.nanosec)
        if image_stamp == self.last_checked_image_stamp:
            return
        self.last_checked_image_stamp = image_stamp
        try:
            self.saved = self.validate_and_save(max(receipt_times) - min(receipt_times))
        except RuntimeError as exc:
            self.last_rejection = str(exc)

    def validate_and_save(self, receipt_span):
        rgb = self.bridge.imgmsg_to_cv2(self.image, desired_encoding="bgr8")
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)
        marker_count = 0 if marker_ids is None else len(marker_ids)
        detected_marker_ids = [] if marker_ids is None else sorted(marker_ids.flatten().astype(int).tolist())
        if detected_marker_ids != self.expected_marker_ids:
            raise RuntimeError(
                "target rejected: expected marker IDs %s, detected %s"
                % (self.expected_marker_ids, detected_marker_ids)
            )
        camera_matrix = np.asarray(self.info.k, dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(self.info.d, dtype=np.float64)
        count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            self.board,
            cameraMatrix=camera_matrix,
            distCoeffs=distortion,
        )
        if count != self.expected_corner_count:
            raise RuntimeError(
                "target rejected: expected %d ChArUco corners, detected %d"
                % (self.expected_corner_count, count)
            )
        if not joint_positions_are_stable(
                self.joint_history,
                time.monotonic(),
                self.stationary_duration_sec,
                self.stationary_position_tolerance_rad):
            raise RuntimeError(
                "target rejected: joint positions did not remain within %.6f rad for %.2f seconds"
                % (self.stationary_position_tolerance_rad,
                   self.stationary_duration_sec)
            )
        image_stamp_seconds = stamp_seconds(self.image.header.stamp)
        joint_message, joint_image_delta = nearest_stamped_joint(
            self.stamped_joint_history, image_stamp_seconds)
        if joint_message is None:
            raise RuntimeError(
                "target rejected: no joint state correlated within 0.100 seconds "
                "of the image stamp%s"
                % ("" if joint_image_delta is None
                   else " (nearest %.6f seconds)" % joint_image_delta)
            )

        folder = self.output_root / ("capture_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        folder.mkdir(parents=True, exist_ok=False)
        annotated = rgb.copy()
        cv2.aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)
        cv2.aruco.drawDetectedCornersCharuco(annotated, charuco_corners, charuco_ids)
        if not cv2.imwrite(str(folder / "rgb.png"), rgb):
            raise RuntimeError("failed to write rgb.png")
        if not cv2.imwrite(str(folder / "charuco_detection.png"), annotated):
            raise RuntimeError("failed to write charuco_detection.png")

        data = {
            "target": {
                "type": "charuco",
                "source": "Boston Dynamics Spot calibration panel",
                "squares_x": self.squares_x,
                "squares_y": self.squares_y,
                "square_length_m": self.square_length_m,
                "marker_length_m": self.marker_length_m,
                "dictionary": self.dictionary_name,
                "marker_ids": marker_ids.flatten().astype(int).tolist(),
                "charuco_ids": charuco_ids.flatten().astype(int).tolist(),
                "charuco_corners_px": charuco_corners.reshape(-1, 2).astype(float).tolist(),
            },
            "synchronization": {
                "image_stamp": stamp_dict(self.image.header.stamp),
                "joint_stamp": stamp_dict(joint_message.header.stamp),
                "image_to_joint_header_delta_seconds": joint_image_delta,
                "header_clocks_are_not_compared": False,
                "receipt_span_seconds": float(receipt_span),
            },
            "camera": {
                "frame_id": self.image.header.frame_id,
                "width": int(self.info.width),
                "height": int(self.info.height),
                "distortion_model": self.info.distortion_model,
                "d": list(map(float, self.info.d)),
                "k": list(map(float, self.info.k)),
                "r": list(map(float, self.info.r)),
                "p": list(map(float, self.info.p)),
            },
            "joints": {
                "name": list(joint_message.name),
                "position_rad": list(map(float, joint_message.position)),
                "velocity": list(map(float, joint_message.velocity)),
                "effort": list(map(float, joint_message.effort)),
            },
            "controller_end_pose": {
                "position_m": [
                    float(self.end_pose.position.x),
                    float(self.end_pose.position.y),
                    float(self.end_pose.position.z),
                ],
                "orientation_xyzw": [
                    float(self.end_pose.orientation.x),
                    float(self.end_pose.orientation.y),
                    float(self.end_pose.orientation.z),
                    float(self.end_pose.orientation.w),
                ],
            },
        }
        with (folder / "sample.yaml").open("w") as stream:
            yaml.safe_dump(data, stream, sort_keys=False)
        return folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--squares-x", type=int, default=5)
    parser.add_argument("--squares-y", type=int, default=5)
    parser.add_argument("--square-length-m", type=float, default=DEFAULT_SQUARE_LENGTH_M)
    parser.add_argument("--marker-length-m", type=float, default=DEFAULT_MARKER_LENGTH_M)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--stationary-duration-sec", type=float, default=0.75)
    parser.add_argument("--stationary-position-tolerance-rad", type=float, default=0.001)
    args = parser.parse_args()
    if args.squares_x < 2 or args.squares_y < 2:
        parser.error("board must contain at least 2x2 squares")
    if args.marker_length_m <= 0 or args.square_length_m <= 0:
        parser.error("square and marker lengths must be positive")
    if args.marker_length_m >= args.square_length_m:
        parser.error("marker length must be smaller than square length")
    if args.stationary_duration_sec <= 0.0:
        parser.error("stationary duration must be positive")
    if args.stationary_position_tolerance_rad <= 0.0:
        parser.error("stationary position tolerance must be positive")
    if not hasattr(cv2.aruco, args.dictionary):
        parser.error("unknown ArUco dictionary: %s" % args.dictionary)
    rclpy.init()
    node = HandEyeCapture(
        args.output_root,
        args.timeout,
        args.squares_x,
        args.squares_y,
        args.square_length_m,
        args.marker_length_m,
        args.dictionary,
        args.stationary_duration_sec,
        args.stationary_position_tolerance_rad,
    )
    while rclpy.ok() and node.saved is None and node.error is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        # Check readiness explicitly. A timer can be starved by the combined
        # high-rate image, camera-info, joint-state, and end-pose callbacks.
        node.tick()
    if node.saved:
        print(node.saved)
        code = 0
    else:
        print(node.error, file=sys.stderr)
        code = 1
    node.destroy_node()
    rclpy.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
