#!/usr/bin/env python3
"""Measure fixed ChArUco board repeatability in base_link without commanding motion."""

import argparse
from collections import deque
import math
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, TransformException, TransformListener
import yaml


DEFAULT_SQUARE_LENGTH_M = 0.017
DEFAULT_MARKER_LENGTH_M = 0.012


def transform(rotation=None, translation=None):
    value = np.eye(4)
    if rotation is not None:
        value[:3, :3] = rotation
    if translation is not None:
        value[:3, 3] = translation
    return value


def transform_message(message):
    t = message.transform.translation
    q = message.transform.rotation
    return transform(
        Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(),
        [t.x, t.y, t.z],
    )


def mean_transform(values):
    translations = np.asarray([value[:3, 3] for value in values])
    rotations = Rotation.from_matrix(np.asarray([value[:3, :3] for value in values]))
    return transform(rotations.mean().as_matrix(), np.median(translations, axis=0))


def rotation_error_deg(first, second):
    delta = first[:3, :3].T @ second[:3, :3]
    return math.degrees(Rotation.from_matrix(delta).magnitude())


def joint_positions_are_stable(
        history, now, duration_sec, position_tolerance_rad):
    """Require a complete recent window whose per-joint span is bounded."""
    if not history or duration_sec <= 0.0 or position_tolerance_rad <= 0.0:
        return False
    cutoff = now - duration_sec
    window = [positions for stamp, positions in history if stamp >= cutoff]
    if len(window) < 2 or history[0][0] > cutoff:
        return False
    positions = np.asarray(window, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 6 or not np.all(np.isfinite(positions)):
        return False
    return bool(np.max(np.ptp(positions, axis=0)) <= position_tolerance_rad)


class FixedBoardValidator(Node):
    def __init__(self, args):
        super().__init__("fixed_board_validator")
        self.args = args
        self.bridge = CvBridge()
        self.camera_info = None
        self.joints = None
        self.joints_received = 0.0
        self.joint_history = deque()
        self.latest = None
        self.latest_stamp = None
        self.latest_rejection_reason = "waiting for camera and joint feedback"
        self.lock = threading.Lock()
        dictionary_id = getattr(cv2.aruco, args.dictionary)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.board = cv2.aruco.CharucoBoard_create(
            args.squares_x,
            args.squares_y,
            args.square_length_m,
            args.marker_length_m,
            self.dictionary,
        )
        self.expected_markers = sorted(self.board.ids.flatten().astype(int).tolist())
        self.expected_corners = (args.squares_x - 1) * (args.squares_y - 1)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(CameraInfo, args.info_topic, self.info_cb, qos_profile_sensor_data)
        self.create_subscription(JointState, args.joint_topic, self.joints_cb, 10)
        self.create_subscription(Image, args.image_topic, self.image_cb, qos_profile_sensor_data)

    def info_cb(self, message):
        self.camera_info = message

    def joints_cb(self, message):
        self.joints = message
        now = time.monotonic()
        self.joints_received = now
        positions = np.asarray(message.position[:6], dtype=float)
        if positions.size == 6 and np.all(np.isfinite(positions)):
            self.joint_history.append((now, positions.copy()))
            cutoff = now - max(2.0 * self.args.stationary_duration_sec, 1.0)
            while self.joint_history and self.joint_history[0][0] < cutoff:
                self.joint_history.popleft()

    def arm_is_still(self):
        now = time.monotonic()
        if self.joints is None or now - self.joints_received > 0.5:
            return False
        # PiPER's instantaneous velocity field spikes at rest and can also
        # briefly read low while a disabled joint is sagging.  Actual position
        # span over a continuous window is the validation authority.
        return joint_positions_are_stable(
            self.joint_history,
            now,
            self.args.stationary_duration_sec,
            self.args.stationary_position_tolerance_rad,
        )

    def reject_frame(self, reason):
        with self.lock:
            self.latest_rejection_reason = reason

    def image_cb(self, message):
        if self.camera_info is None:
            self.reject_frame("waiting for camera calibration")
            return
        if not self.arm_is_still():
            self.reject_frame(
                "joint positions have not remained within %.6f rad for %.2f seconds"
                % (self.args.stationary_position_tolerance_rad,
                   self.args.stationary_duration_sec)
            )
            return
        stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
        if stamp == self.latest_stamp:
            return
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)
            ids = [] if marker_ids is None else sorted(marker_ids.flatten().astype(int).tolist())
            if ids != self.expected_markers:
                self.reject_frame(
                    "full board not visible: detected %d/%d markers"
                    % (len(ids), len(self.expected_markers))
                )
                return
            camera = self.camera_info
            camera_matrix = np.asarray(camera.k, dtype=np.float64).reshape(3, 3)
            distortion = np.asarray(camera.d, dtype=np.float64)
            count, corners, corner_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids,
                gray,
                self.board,
                cameraMatrix=camera_matrix,
                distCoeffs=distortion,
            )
            if count != self.expected_corners:
                self.reject_frame(
                    "full board not visible: detected %d/%d ChArUco corners"
                    % (count, self.expected_corners)
                )
                return
            object_points = np.asarray(self.board.chessboardCorners, dtype=np.float64)[
                corner_ids.flatten().astype(int)
            ]
            image_points = corners.reshape(-1, 2).astype(np.float64)
            ok, rvec, tvec = cv2.solvePnP(
                object_points, image_points, camera_matrix, distortion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                self.reject_frame("board pose solve failed")
                return
            rotation, _ = cv2.Rodrigues(rvec)
            base_from_camera = transform_message(
                self.tf_buffer.lookup_transform(
                    self.args.base_frame, self.args.camera_frame,
                    Time.from_msg(message.header.stamp)
                )
            )
            base_from_board = base_from_camera @ transform(rotation, tvec.reshape(3))
            projected, _ = cv2.projectPoints(
                object_points, rvec, tvec, camera_matrix, distortion
            )
            error = projected.reshape(-1, 2) - image_points
            rms = float(np.sqrt(np.mean(np.sum(error * error, axis=1))))
            with self.lock:
                self.latest = (stamp, base_from_board, rms)
                self.latest_stamp = stamp
                self.latest_rejection_reason = ""
        except (TransformException, ValueError, cv2.error) as error:
            self.reject_frame("camera TF/board pose unavailable: %s" % error)
            return

    def collect(self):
        deadline = time.monotonic() + self.args.timeout
        frames = []
        seen = set()
        with self.lock:
            self.latest = None
        while len(frames) < self.args.frames_per_pose and time.monotonic() < deadline:
            with self.lock:
                latest = self.latest
            if latest is not None and latest[0] not in seen:
                seen.add(latest[0])
                frames.append(latest)
            time.sleep(0.02)
        if len(frames) < self.args.frames_per_pose:
            with self.lock:
                reason = self.latest_rejection_reason
            raise RuntimeError(
                "only %d/%d valid stationary frames; %s"
                % (len(frames), self.args.frames_per_pose,
                   reason or "check board visibility and TF")
            )
        pose_mean = mean_transform([item[1] for item in frames])
        translation_errors = np.asarray([
            np.linalg.norm(item[1][:3, 3] - pose_mean[:3, 3]) * 1000.0
            for item in frames
        ])
        rotation_errors = np.asarray([
            rotation_error_deg(pose_mean, item[1]) for item in frames
        ])
        reprojection = np.asarray([item[2] for item in frames])
        noise = {
            "frame_count": len(frames),
            "translation_rms_mm": float(np.sqrt(np.mean(translation_errors ** 2))),
            "translation_max_mm": float(np.max(translation_errors)),
            "rotation_rms_deg": float(np.sqrt(np.mean(rotation_errors ** 2))),
            "rotation_max_deg": float(np.max(rotation_errors)),
            "reprojection_mean_px": float(np.mean(reprojection)),
            "reprojection_max_px": float(np.max(reprojection)),
        }
        return pose_mean, float(np.mean(reprojection)), noise


def result_dict(poses, reprojection, translation_limit_mm, rotation_limit_deg,
                validation_metadata, within_pose_noise=None):
    reference = mean_transform(poses)
    measurements = []
    for index, (pose, reprojection_rms) in enumerate(zip(poses, reprojection), 1):
        translation_error = float(np.linalg.norm(pose[:3, 3] - reference[:3, 3]) * 1000.0)
        rotation_error = float(rotation_error_deg(reference, pose))
        measurement = {
            "index": index,
            "position_m": pose[:3, 3].astype(float).tolist(),
            "quaternion_xyzw": Rotation.from_matrix(pose[:3, :3]).as_quat().astype(float).tolist(),
            "translation_error_mm": translation_error,
            "rotation_error_deg": rotation_error,
            "reprojection_rms_px": reprojection_rms,
            "passed": translation_error <= translation_limit_mm and rotation_error <= rotation_limit_deg,
        }
        if within_pose_noise is not None:
            measurement["within_pose_noise"] = within_pose_noise[index - 1]
        measurements.append(measurement)
    return {
        "status": "passed" if measurements and all(item["passed"] for item in measurements) else "failed",
        "validation": validation_metadata,
        "limits": {"translation_mm": translation_limit_mm, "rotation_deg": rotation_limit_deg},
        "reference_position_m": reference[:3, 3].astype(float).tolist(),
        "measurements": measurements,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--info-topic", default="/camera/color/camera_info")
    parser.add_argument("--joint-topic", default="/joint_states_single")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default="camera_color_optical_frame")
    parser.add_argument("--squares-x", type=int, default=5)
    parser.add_argument("--squares-y", type=int, default=5)
    parser.add_argument("--square-length-m", type=float, default=DEFAULT_SQUARE_LENGTH_M)
    parser.add_argument("--marker-length-m", type=float, default=DEFAULT_MARKER_LENGTH_M)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--frames-per-pose", type=int, default=10)
    parser.add_argument("--minimum-poses", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-joint-velocity", type=float, default=0.03,
                        help="deprecated compatibility option; position stability is authoritative")
    parser.add_argument("--stationary-duration-sec", type=float, default=0.75)
    parser.add_argument("--stationary-position-tolerance-rad", type=float, default=0.001)
    parser.add_argument("--translation-limit-mm", type=float, default=15.0)
    parser.add_argument("--rotation-limit-deg", type=float, default=1.5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--stationary-noise-only",
        action="store_true",
        help="collect one multi-frame stationary pose without requiring arm movement",
    )
    args = parser.parse_args()
    if not hasattr(cv2.aruco, args.dictionary):
        parser.error("unknown ArUco dictionary: %s" % args.dictionary)
    if args.frames_per_pose < 1:
        parser.error("--frames-per-pose must be positive")
    if args.minimum_poses < 3:
        parser.error("--minimum-poses must be at least three")
    if args.stationary_duration_sec <= 0.0:
        parser.error("--stationary-duration-sec must be positive")
    if args.stationary_position_tolerance_rad <= 0.0:
        parser.error("--stationary-position-tolerance-rad must be positive")

    rclpy.init()
    node = FixedBoardValidator(args)
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    poses, reprojection, within_pose_noise = [], [], []
    print("Keep the board fixed. Stop the arm, then press Enter to measure each pose.")
    if args.stationary_noise_only:
        print("One stationary multi-frame diagnostic will be collected.")
    else:
        print("Enter q when at least %d different arm poses have been measured." % args.minimum_poses)
    try:
        while True:
            if args.stationary_noise_only and poses:
                break
            command = input("measure [Enter], finish [q]: ").strip().lower()
            if command == "q":
                break
            try:
                pose, rms, noise = node.collect()
                poses.append(pose)
                reprojection.append(rms)
                within_pose_noise.append(noise)
                print(
                    "pose %d: base position [%.4f, %.4f, %.4f] m, "
                    "reprojection %.3f px, within-pose %.3f mm / %.3f deg RMS"
                    % (len(poses), pose[0, 3], pose[1, 3], pose[2, 3], rms,
                       noise["translation_rms_mm"], noise["rotation_rms_deg"])
                )
            except RuntimeError as error:
                print("measurement rejected: %s" % error, file=sys.stderr)
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        thread.join(timeout=2.0)

    required_poses = 1 if args.stationary_noise_only else args.minimum_poses
    if len(poses) < required_poses:
        print(
            "validation requires at least %d measured arm poses" % required_poses,
            file=sys.stderr,
        )
        return 1
    result = result_dict(
        poses,
        reprojection,
        args.translation_limit_mm,
        args.rotation_limit_deg,
        {
            "board": {
                "squares_x": args.squares_x,
                "squares_y": args.squares_y,
                "square_length_m": args.square_length_m,
                "marker_length_m": args.marker_length_m,
                "dictionary": args.dictionary,
            },
            "frames_per_pose": args.frames_per_pose,
            "minimum_poses": args.minimum_poses,
            "stationary_duration_sec": args.stationary_duration_sec,
            "stationary_position_tolerance_rad": args.stationary_position_tolerance_rad,
            "base_frame": args.base_frame,
            "camera_frame": args.camera_frame,
            "stationary_noise_only": args.stationary_noise_only,
        },
        within_pose_noise,
    )
    for item in result["measurements"]:
        print(
            "pose %d: drift %.2f mm, %.2f deg, %s"
            % (item["index"], item["translation_error_mm"], item["rotation_error_deg"],
               "PASS" if item["passed"] else "FAIL")
        )
    print("result: %s" % result["status"].upper())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as stream:
            yaml.safe_dump(result, stream, sort_keys=False)
        print("wrote %s" % args.output)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
