#!/usr/bin/env python3
"""Replay deterministic public ROS interfaces through the actual recorder, without arm commands."""

from dataset import convert, validate, atomic_json
from piper_mobile_manipulation.reconstruction_recorder_node import (
    ReconstructionRecorderNode,
)
from piper_mobile_manipulation.msg import ReconstructionKeyframe, ScanExecutionStatus
import argparse
import json
import math
from pathlib import Path
import sys
import threading
import time
import uuid

from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "piper_ros_foxy/src/piper_mobile_manipulation"))


def until(predicate, message, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(message)


class Replay:
    """Publish real ROS images, history, TF and completion, retaining all recorder code."""

    def __init__(self, output, angle):
        self.output = Path(output)
        self.recorder = ReconstructionRecorderNode(
            parameter_overrides=[
                Parameter("output_root", value=str(self.output)),
                Parameter("target_visibility_angle_tolerance_deg", value=float(angle)),
                Parameter("task_id", value="replay-task"),
            ]
        )
        self.publisher = Node("reconstruction_replay_publisher")
        self.tf = TransformBroadcaster(self.publisher)
        self.pubs = {
            name: self.publisher.create_publisher(Image, topic, 10)
            for name, topic in (
                ("rgb", "/camera/color/image_raw"),
                ("depth", "/camera/aligned_depth_to_color/image_raw"),
                ("mask", "/piper/sam2_target_mask"),
            )
        }
        self.info_pub = self.publisher.create_publisher(
            CameraInfo, "/camera/color/camera_info", 10
        )
        self.pin_pub = self.publisher.create_publisher(
            ReconstructionKeyframe, "/piper/reconstruction_keyframe", qos_profile_sensor_data
        )
        self.history_pub = self.publisher.create_publisher(
            String,
            "/piper/scan_session_history",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.workflow_pub = self.publisher.create_publisher(
            String, "/piper/supervised_workflow_status", 10
        )
        self.motion_pub = self.publisher.create_publisher(
            ScanExecutionStatus, "/piper/scan_execution_status", 10
        )
        self.bridge = CvBridge()
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.recorder)
        self.executor.add_node(self.publisher)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()
        self.tick = self.publisher.get_clock().now().nanoseconds
        self.entries = []
        self.session_id = "replay-" + uuid.uuid4().hex
        until(
            lambda: self.pin_pub.get_subscription_count() > 0
            and self.pubs["rgb"].get_subscription_count() > 0,
            "ROS recorder subscriptions did not join",
        )
        time.sleep(0.3)
        assert self.publisher.count_publishers("/joint_ctrl_single") == 0

    def image(
        self, angle=0.0, depth_encoding="16UC1", mask_kind="valid", transform=True
    ):
        self.tick += 200_000_000
        checker = (np.indices((120, 160)).sum(axis=0) % 2 * 255).astype(np.uint8)
        rgb = np.repeat(checker[:, :, None], 3, axis=2)
        depth = (
            np.full((120, 160), 500, np.uint16)
            if depth_encoding == "16UC1"
            else np.full((120, 160), 0.5, np.float32)
        )
        mask = np.zeros((120, 160), np.uint8)
        mask[45:75, 65:95] = 255
        if mask_kind == "empty":
            mask[:] = 0
        if mask_kind == "clipped":
            mask[60, 0] = 255
        messages = [
            self.bridge.cv2_to_imgmsg(rgb, "rgb8"),
            self.bridge.cv2_to_imgmsg(depth, depth_encoding),
            self.bridge.cv2_to_imgmsg(mask, "mono8"),
        ]
        info = CameraInfo(width=160, height=120)
        info.k = [200.0, 0.0, 80.0, 0.0, 200.0, 60.0, 0.0, 0.0, 1.0]
        for msg in messages + [info]:
            msg.header.frame_id = "camera_color_optical_frame"
            msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(
                self.tick, 1_000_000_000
            )
        if transform:
            tf = TransformStamped()
            tf.header.stamp = messages[0].header.stamp
            tf.header.frame_id = "base_link"
            tf.child_frame_id = "camera_color_optical_frame"
            tf.transform.rotation.y = math.sin(math.radians(angle) / 2)
            tf.transform.rotation.w = math.cos(math.radians(angle) / 2)
            self.tf.sendTransform(tf)
            time.sleep(0.05)
        return messages[0], messages[1], info, messages[2]

    def stationary(self, raw):
        rgb, depth, info, mask = raw
        stamp = rgb.header.stamp.sec * 1_000_000_000 + rgb.header.stamp.nanosec
        message = ReconstructionKeyframe(
            task_id="replay-task", rgb=rgb, depth=depth, camera_info=info, mask=mask
        )
        self.pin_pub.publish(message)
        self.entries.append(
            dict(accepted_view=len(self.entries) + 1, reconstruction_stamp_ns=stamp)
        )
        history = dict(
            session_id=self.session_id,
            accepted_views=len(self.entries),
            entries=self.entries,
            coverage_target_center=dict(x=0.0, y=0.0, z=0.5),
        )
        self.history_pub.publish(String(data=json.dumps(history)))
        until(
            lambda: len(self.recorder.processed_views) == len(self.entries),
            "stationary keyframe not correlated",
        )

    def motion(self, moving=True):
        self.motion_pub.publish(
            ScanExecutionStatus(
                execution_mode="MULTIVIEW_SCAN",
                state="MOVING" if moving else "VIEW_COMPLETE",
            )
        )
        until(
            lambda: self.recorder.motion_state
            == ("MOVING" if moving else "VIEW_COMPLETE"),
            "motion event not received",
        )

    def moving(self, raw, expected=True, send_mask=True):
        before = self.recorder.session.accepted
        rgb, depth, info, mask = raw
        self.info_pub.publish(info)
        self.pubs["rgb"].publish(rgb)
        self.pubs["depth"].publish(depth)
        if send_mask:
            self.pubs["mask"].publish(mask)
        if expected:
            until(
                lambda: self.recorder.session.accepted > before,
                "moving frame not recorded",
            )
        else:
            time.sleep(0.15)
            assert self.recorder.session.accepted == before

    def finish(self, state="COMPLETE"):
        self.workflow_pub.publish(
            String(
                data=json.dumps(
                    dict(
                        state=state,
                        session_id=self.session_id,
                        accepted_views=len(self.entries),
                    )
                )
            )
        )
        until(
            lambda: self.recorder.session is None
            and bool(self.recorder.completed_sessions),
            "session did not finalize",
        )
        return self.recorder.completed_sessions[-1]

    def close(self):
        self.executor.shutdown()
        self.spin_thread.join(2)
        self.recorder.destroy_node()
        self.publisher.destroy_node()


def run(output, distorted=False):
    output.mkdir(parents=True, exist_ok=False)
    evidence = []
    rclpy.init()
    try:
        for angle in (20.0, 10.0, 5.0):
            replay = Replay(output / ("angle_%02d" % angle), angle)
            if distorted:
                original_image = replay.image

                def calibrated_image(*args, **kwargs):
                    raw = original_image(*args, **kwargs)
                    raw[2].distortion_model = 'plumb_bob'
                    raw[2].d = [0.169, -0.514, -0.0021, 0.0013, 0.475]
                    return raw

                replay.image = calibrated_image
            try:
                replay.stationary(replay.image())
                replay.motion()
                for i in range(4):
                    replay.moving(
                        replay.image(
                            angle=angle, depth_encoding="32FC1" if i % 2 else "16UC1"
                        )
                    )
                replay.moving(replay.image(angle=angle + 0.001), expected=False)
                replay.moving(replay.image(mask_kind="clipped"), expected=False)
                replay.moving(replay.image(mask_kind="empty"), expected=False)
                if angle == 20.0:
                    # Actual synchronizer delivers a pair inside its diagnostic
                    # 100 ms window; admission must require exact frameset stamps.
                    delayed = replay.image()
                    delayed_stamp = replay.tick - 11_000_000
                    (
                        delayed[1].header.stamp.sec,
                        delayed[1].header.stamp.nanosec,
                    ) = divmod(delayed_stamp, 1_000_000_000)
                    replay.moving(delayed, expected=False)
                    replay.moving(replay.image(transform=False), expected=False)
                    lost = replay.image()
                    replay.moving(lost, expected=False, send_mask=False)
                    stale_stamp = replay.tick - 5_000_000
                    lost[3].header.stamp.sec, lost[3].header.stamp.nanosec = divmod(
                        stale_stamp, 1_000_000_000
                    )
                    replay.pubs["mask"].publish(lost[3])
                    time.sleep(1.1)
                    assert (
                        replay.recorder.session.rejected["rgb_depth_unsynchronised"]
                        >= 1
                    )
                    assert replay.recorder.session.rejected["missing_tf"] >= 1
                    assert replay.recorder.session.rejected["stale_mask"] >= 1
                replay.motion(False)
                replay.stationary(replay.image())
                completed = replay.finish()
                assert completed.manifest()["accepted"] == 6
                assert completed.manifest()["mission_capture_count"] == 2
                assert completed.manifest()["complete"]
                report, _ = validate(completed.final_path)
                convert(completed.final_path, output / ("converted_%02d" % angle))
                evidence.append(completed.manifest())
                # A second mission uses the same node, with reset timestamps and identity.
                replay.entries = []
                replay.session_id = "replay-" + uuid.uuid4().hex
                replay.stationary(replay.image())
                second = replay.finish("ABORTED")
                assert not second.manifest()["complete"]
                assert second.manifest()["accepted"] == 1
                evidence.append(second.manifest())
            finally:
                replay.close()
        atomic_json(
            output / "replay_report.json",
            dict(evidence_class="ROS_SYNTHETIC_REPLAY_NOT_PHYSICAL", sessions=evidence),
        )
        print(output / "replay_report.json")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distorted", action="store_true")
    arguments = parser.parse_args()
    run(arguments.output, arguments.distorted)
