#!/usr/bin/env python3
"""Save one RGB-D perception snapshot for offline analysis."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from piper_mobile_manipulation.msg import Target3D


DEFAULT_OUTPUT_ROOT = Path("/home/prl/Piper_arm/L515_camera/captures")


def stamp_to_dict(stamp: Any) -> Dict[str, int]:
    return {
        "sec": int(stamp.sec),
        "nanosec": int(stamp.nanosec),
    }


def header_to_dict(msg: Any) -> Dict[str, Any]:
    return {
        "stamp": stamp_to_dict(msg.header.stamp),
        "frame_id": str(msg.header.frame_id),
    }


class SnapshotCaptureNode(Node):
    def __init__(self, output_root: Path, timeout_sec: float, continuous: bool, interval_sec: float):
        super().__init__("l515_snapshot_capture")
        self.output_root = output_root
        self.timeout_sec = max(0.1, float(timeout_sec))
        self.continuous = bool(continuous)
        self.interval_sec = max(0.1, float(interval_sec))
        self.bridge = CvBridge()

        self.latest_rgb: Optional[Image] = None
        self.latest_depth: Optional[Image] = None
        self.latest_camera_info: Optional[CameraInfo] = None
        self.latest_mask: Optional[Image] = None
        self.latest_target: Optional[Target3D] = None
        self.latest_scan_quality: Optional[Dict[str, Any]] = None
        self.latest_occlusion_status: Optional[Dict[str, Any]] = None

        self.started_at = time.monotonic()
        self.last_capture = 0.0
        self.saved_folders: List[Path] = []
        self.done = False

        self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.rgb_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            "/camera/aligned_depth_to_color/image_raw",
            self.depth_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            "/camera/color/camera_info",
            self.camera_info_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            "/piper/detection_mask",
            self.mask_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(Target3D, "/piper/target_3d", self.target_cb, 10)
        self.create_subscription(String, "/piper/scan_quality", self.scan_quality_cb, 10)
        self.create_subscription(String, "/piper/occlusion_status", self.occlusion_status_cb, 10)

        self.timer = self.create_timer(0.1, self.timer_cb)
        self.get_logger().info("Waiting for RGB, depth, camera_info, detection_mask, and target_3d.")
        self.get_logger().warn("Snapshot capture is read-only and never publishes /piper/servo_cmd.")

    def rgb_cb(self, msg: Image) -> None:
        self.latest_rgb = msg

    def depth_cb(self, msg: Image) -> None:
        self.latest_depth = msg

    def camera_info_cb(self, msg: CameraInfo) -> None:
        self.latest_camera_info = msg

    def mask_cb(self, msg: Image) -> None:
        self.latest_mask = msg

    def target_cb(self, msg: Target3D) -> None:
        self.latest_target = msg

    def scan_quality_cb(self, msg: String) -> None:
        self.latest_scan_quality = self.parse_string_payload(msg)

    def occlusion_status_cb(self, msg: String) -> None:
        self.latest_occlusion_status = self.parse_string_payload(msg)

    def timer_cb(self) -> None:
        now = time.monotonic()
        ready, missing = self.required_messages_ready()
        if not ready:
            if now - self.started_at > self.timeout_sec:
                self.get_logger().error("Timed out waiting for required topics: %s" % ", ".join(missing))
                self.done = True
            return

        if self.continuous and now - self.last_capture < self.interval_sec:
            return

        try:
            capture_dir = self.save_snapshot()
        except Exception as exc:
            self.get_logger().error("Snapshot save failed: %s" % exc)
            self.done = True
            return

        self.saved_folders.append(capture_dir)
        self.last_capture = now
        print(str(capture_dir), flush=True)
        self.get_logger().info("Saved snapshot to %s" % capture_dir)

        if not self.continuous:
            self.done = True

    def required_messages_ready(self) -> Tuple[bool, List[str]]:
        required = {
            "rgb": self.latest_rgb,
            "depth": self.latest_depth,
            "camera_info": self.latest_camera_info,
            "detection_mask": self.latest_mask,
            "target_3d": self.latest_target,
        }
        missing = [name for name, value in required.items() if value is None]
        return len(missing) == 0, missing

    def save_snapshot(self) -> Path:
        capture_dir = self.next_capture_dir()
        capture_dir.mkdir(parents=True, exist_ok=False)

        rgb = self.bridge.imgmsg_to_cv2(self.latest_rgb, desired_encoding="bgr8")
        depth = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding="passthrough")
        mask = self.bridge.imgmsg_to_cv2(self.latest_mask, desired_encoding="mono8")

        cv2.imwrite(str(capture_dir / "rgb.png"), rgb)
        np.save(str(capture_dir / "depth.npy"), np.asarray(depth))
        cv2.imwrite(str(capture_dir / "detection_mask.png"), mask)

        self.write_yaml(capture_dir / "camera_info.yaml", self.camera_info_to_dict(self.latest_camera_info))
        self.write_yaml(capture_dir / "target_3d.yaml", self.target_3d_to_dict(self.latest_target))

        if self.latest_scan_quality is not None:
            self.write_yaml(capture_dir / "scan_quality.yaml", self.latest_scan_quality)
        if self.latest_occlusion_status is not None:
            self.write_yaml(capture_dir / "occlusion_status.yaml", self.latest_occlusion_status)

        metadata = self.metadata(capture_dir)
        self.write_yaml(capture_dir / "metadata.yaml", metadata)
        return capture_dir

    def next_capture_dir(self) -> Path:
        self.output_root.mkdir(parents=True, exist_ok=True)
        base_name = "capture_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        capture_dir = self.output_root / base_name
        suffix = 1
        while capture_dir.exists():
            capture_dir = self.output_root / ("%s_%02d" % (base_name, suffix))
            suffix += 1
        return capture_dir

    def metadata(self, capture_dir: Path) -> Dict[str, Any]:
        return {
            "capture_type": "one_shot_rgbd_snapshot",
            "capture_folder": str(capture_dir),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "topics": {
                "rgb": "/camera/color/image_raw",
                "depth": "/camera/aligned_depth_to_color/image_raw",
                "camera_info": "/camera/color/camera_info",
                "detection_mask": "/piper/detection_mask",
                "target_3d": "/piper/target_3d",
                "scan_quality": "/piper/scan_quality",
                "occlusion_status": "/piper/occlusion_status",
            },
            "files": {
                "rgb": "rgb.png",
                "depth": "depth.npy",
                "detection_mask": "detection_mask.png",
                "camera_info": "camera_info.yaml",
                "target_3d": "target_3d.yaml",
                "scan_quality": "scan_quality.yaml" if self.latest_scan_quality is not None else "",
                "occlusion_status": "occlusion_status.yaml"
                if self.latest_occlusion_status is not None
                else "",
            },
            "availability": {
                "rgb": True,
                "depth": True,
                "camera_info": True,
                "detection_mask": True,
                "target_3d": True,
                "scan_quality": self.latest_scan_quality is not None,
                "occlusion_status": self.latest_occlusion_status is not None,
            },
            "dry_run": True,
            "real_arm_motion": False,
            "servo_cmd_published": False,
            "continuous": self.continuous,
        }

    @staticmethod
    def parse_string_payload(msg: String) -> Dict[str, Any]:
        try:
            payload = json.loads(msg.data)
            if isinstance(payload, dict):
                return payload
            return {"data": payload}
        except Exception:
            return {"raw_data": msg.data}

    @staticmethod
    def camera_info_to_dict(msg: CameraInfo) -> Dict[str, Any]:
        return {
            "header": header_to_dict(msg),
            "height": int(msg.height),
            "width": int(msg.width),
            "distortion_model": str(msg.distortion_model),
            "d": [float(value) for value in msg.d],
            "k": [float(value) for value in msg.k],
            "r": [float(value) for value in msg.r],
            "p": [float(value) for value in msg.p],
            "binning_x": int(msg.binning_x),
            "binning_y": int(msg.binning_y),
            "roi": {
                "x_offset": int(msg.roi.x_offset),
                "y_offset": int(msg.roi.y_offset),
                "height": int(msg.roi.height),
                "width": int(msg.roi.width),
                "do_rectify": bool(msg.roi.do_rectify),
            },
        }

    @staticmethod
    def target_3d_to_dict(msg: Target3D) -> Dict[str, Any]:
        return {
            "header": header_to_dict(msg),
            "point": {
                "x": float(msg.point.x),
                "y": float(msg.point.y),
                "z": float(msg.point.z),
            },
            "depth": float(msg.depth),
            "valid_depth_ratio": float(msg.valid_depth_ratio),
            "depth_stddev": float(msg.depth_stddev),
            "roi_width": float(msg.roi_width),
            "roi_height": float(msg.roi_height),
            "source_u": float(msg.source_u),
            "source_v": float(msg.source_v),
            "detection_width": float(msg.detection_width),
            "detection_height": float(msg.detection_height),
            "depth_source": str(msg.depth_source),
            "measurement_confidence": float(msg.measurement_confidence),
            "valid": bool(msg.valid),
        }

    @staticmethod
    def write_yaml(path: Path, data: Dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one saved RGB-D snapshot from active ROS topics.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where capture_YYYYMMDD_HHMMSS folders are created.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=30.0,
        help="Maximum time to wait for required messages.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Keep saving snapshots instead of exiting after the first one.",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=2.0,
        help="Delay between captures when --continuous is used.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = SnapshotCaptureNode(
        output_root=args.output_root.expanduser().resolve(),
        timeout_sec=args.timeout_sec,
        continuous=args.continuous,
        interval_sec=args.interval_sec,
    )
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return 0 if node.saved_folders else 1


if __name__ == "__main__":
    raise SystemExit(main())
