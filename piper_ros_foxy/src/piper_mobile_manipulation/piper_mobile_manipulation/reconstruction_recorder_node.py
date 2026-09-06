#!/usr/bin/env python3
"""Run a passive, bounded RGB-D observer with no mission or robot command interfaces."""

from collections import Counter, OrderedDict, deque
from dataclasses import fields
import json
import queue
import signal
import threading
import time

from cv_bridge import CvBridge
import message_filters
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from piper_mobile_manipulation.msg import ReconstructionKeyframe, ScanExecutionStatus
from piper_mobile_manipulation.reconstruction_recording.core import (
    Frame,
    RecorderConfig,
    RecordingSession,
)
from piper_mobile_manipulation.visibility import frozen_visibility_parameters


def stamp_ns(message):
    """Retain exact ROS integer nanoseconds throughout pairing and persistence."""
    return message.header.stamp.sec * 1000000000 + message.header.stamp.nanosec


class ReconstructionRecorderNode(Node):
    """Process images and disk work outside ROS subscription callbacks."""

    def __init__(self, **kwargs):
        """Create isolated subscriptions and bounded recorder workers."""
        super().__init__("reconstruction_recorder", **kwargs)
        defaults = RecorderConfig()
        values = {}
        for field in fields(defaults):
            self.declare_parameter(field.name, getattr(defaults, field.name))
            values[field.name] = self.get_parameter(field.name).value
        self.config = RecorderConfig(**values)
        options = {
            "output_root": "/home/prl/Piper_arm/datasets/reconstruction_sessions",
            "base_frame": "base_link",
            "camera_optical_frame": "camera_color_optical_frame",
            "task_id": "",
            "color_image_topic": "/camera/color/image_raw",
            "depth_image_topic": "/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/color/camera_info",
            "mask_topic": "/piper/sam2_target_mask",
            "input_queue_size": 8,
        }
        for key, value in options.items():
            self.declare_parameter(key, value)
        self.options = {key: self.get_parameter(key).value for key in options}
        if not 1 <= self.options["input_queue_size"] <= 64:
            raise ValueError("input_queue_size must be within 1..64")
        self.add_on_set_parameters_callback(self.reject_live_parameters)
        self.bridge = CvBridge()
        self.tf_buffer = Buffer(cache_time=Duration(seconds=120.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.images = queue.Queue(maxsize=self.options["input_queue_size"])
        self.events = queue.Queue(maxsize=64)
        self.stop_requested = threading.Event()
        self.rejections = Counter()
        self.session = None
        self.completed_sessions = deque(maxlen=2)
        self.history = None
        self.history_identity = None
        self.processed_views = set()
        self.pins = OrderedDict()
        self.pairs = OrderedDict()
        self.masks = OrderedDict()
        self.motion_state = ""
        self.terminal = None
        self.terminal_at = None
        self.last_status = None
        self.latest_info = None
        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.rgb_sub = message_filters.Subscriber(
            self, Image, self.options["color_image_topic"], qos_profile=sensor_qos
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, self.options["depth_image_topic"], qos_profile=sensor_qos
        )
        self.create_subscription(
            CameraInfo, self.options["camera_info_topic"], self.info_cb, sensor_qos
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 10, 0.1
        )
        self.sync.registerCallback(self.rgbd_cb)
        self.create_subscription(
            Image, self.options["mask_topic"], self.mask_cb, sensor_qos
        )
        self.create_subscription(
            ReconstructionKeyframe,
            "/piper/reconstruction_keyframe",
            self.keyframe_cb,
            sensor_qos,
        )
        history_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            String, "/piper/scan_session_history", self.history_cb, history_qos
        )
        self.create_subscription(
            String, "/piper/supervised_workflow_status", self.workflow_cb, 10
        )
        self.create_subscription(
            ScanExecutionStatus, "/piper/scan_execution_status", self.execution_cb, 10
        )
        self.status_pub = self.create_publisher(
            String, "/piper/reconstruction_recording_status", 10
        )
        self.create_timer(2.0, self.publish_status)
        self.worker = threading.Thread(
            target=self.process_loop, name="rgbd-admission", daemon=True
        )
        self.worker.start()

    def reject_live_parameters(self, parameters):
        """Freeze the complete recorder policy for deterministic session provenance."""
        from rcl_interfaces.msg import SetParametersResult

        visibility = frozen_visibility_parameters(parameters)
        if not visibility.successful:
            return visibility
        return SetParametersResult(
            successful=False,
            reason="Restart recorder between missions to change its frozen configuration",
        )

    def enqueue(self, destination, item, reason):
        """Drop newest work instead of blocking a ROS callback."""
        try:
            destination.put_nowait(item)
            return True
        except queue.Full:
            self.rejections[reason] += 1
            return False

    def info_cb(self, info):
        """Queue calibration independently of RGB/depth delivery order."""
        self.enqueue(self.images, ("info", info), "input_queue_full")

    def rgbd_cb(self, rgb, depth):
        """Queue references only; no decoding, TF wait, encoding or filesystem I/O."""
        self.enqueue(
            self.images,
            (
                "pair",
                (rgb, depth, self.latest_info, time.monotonic(), self.last_status),
            ),
            "input_queue_full",
        )

    def mask_cb(self, mask):
        """Preserve original mask stamp; never synthesize an ROI."""
        self.enqueue(
            self.images, ("mask", (mask, time.monotonic())), "input_queue_full"
        )

    def keyframe_cb(self, message):
        """Pin the exact selected raw tuple while semantic acceptance is pending."""
        if message.task_id == self.options["task_id"]:
            self.enqueue(self.events, ("keyframe", message), "keyframe_queue_full")
        else:
            self.rejections["wrong_task"] += 1

    def history_cb(self, message):
        """Observe only session/count changes; this topic remains executor-owned."""
        try:
            history = json.loads(message.data)
            identity = (history["session_id"], history["accepted_views"])
            if identity != self.history_identity:
                if self.enqueue(self.events, ("history", history), "event_queue_full"):
                    self.history_identity = identity
        except (ValueError, KeyError, TypeError):
            self.rejections["invalid_history"] += 1

    def workflow_cb(self, message):
        """Workflow COMPLETE follows the last accepted capture; abort stays incomplete."""
        try:
            payload = json.loads(message.data)
            if payload.get("state") in ("COMPLETE", "ABORTED"):
                self.enqueue(self.events, ("terminal", payload), "event_queue_full")
        except (ValueError, TypeError):
            self.rejections["invalid_workflow_event"] += 1

    def execution_cb(self, message):
        """Read motion status without publishing any execution input."""
        state = (message.execution_mode, message.state)
        if state != self.last_status:
            self.enqueue(self.events, ("motion", state), "event_queue_full")
            self.last_status = state

    def make_frame(self, rgb, depth, info, mask, received, source, view):
        """Require calibration and TF at the original image stamp, without latest fallback."""
        expected = self.options["camera_optical_frame"]
        if info is None:
            raise ValueError("invalid_intrinsics")
        if any(m.header.frame_id != expected for m in (rgb, depth, info, mask)):
            raise ValueError("camera_frame_mismatch")
        if (
            info.width != rgb.width
            or info.height != rgb.height
            or any(abs(float(x)) > 1e-10 for x in info.d)
        ):
            raise ValueError("invalid_intrinsics")
        # Exact image-time TF lookup, zero wait on this background worker.
        try:
            tf = self.tf_buffer.lookup_transform(
                self.options["base_frame"],
                expected,
                Time.from_msg(rgb.header.stamp),
                timeout=Duration(seconds=0.0),
            )
        except Exception as error:
            raise ValueError("missing_tf") from error
        if stamp_ns(tf) != stamp_ns(rgb):
            raise ValueError("stale_tf")
        q = tf.transform.rotation
        quaternion = np.asarray([q.x, q.y, q.z, q.w])
        if (
            not np.isfinite(quaternion).all()
            or abs(np.linalg.norm(quaternion) - 1.0) > 1e-5
        ):
            raise ValueError("invalid_tf")
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        t = tf.transform.translation
        matrix[:3, 3] = [t.x, t.y, t.z]
        return Frame(
            self.bridge.imgmsg_to_cv2(rgb, "rgb8"),
            self.bridge.imgmsg_to_cv2(depth, "passthrough"),
            self.bridge.imgmsg_to_cv2(mask, "mono8"),
            np.asarray(info.k).reshape(3, 3),
            matrix,
            stamp_ns(rgb),
            stamp_ns(depth),
            stamp_ns(mask),
            (max(0.0, time.monotonic() - received) if source == "moving" else 0.0),
            depth.encoding,
            self.options["base_frame"],
            expected,
            self.session.target_id,
            stamp_ns(tf),
            source=source,
            accepted_view=view,
        )

    def admit(self, raw, source, view=0):
        """Decode and qualify evidence on the recorder worker only."""
        try:
            frame = self.make_frame(*raw, source, view)
            return self.session.submit(frame)
        except ValueError as error:
            return self.session.reject(str(error))
        except Exception:
            return self.session.reject("image_decode_error")

    def reconcile(self):
        """Handle pin/history delivery in either order and never promote pending captures."""
        if self.history is None:
            return
        count = int(self.history["accepted_views"])
        bound_angle = self.history.get("target_visibility_angle_tolerance_deg")
        if (
            bound_angle is not None
            and float(bound_angle) != self.config.target_visibility_angle_tolerance_deg
        ):
            self.rejections["visibility_configuration_mismatch"] += 1
            return
        if count > 0 and self.session is None:
            self.rejections.clear()
            target = self.history.get("coverage_target_center")
            if not isinstance(target, dict):
                self.rejections["missing_frozen_target"] += 1
                return
            self.session = RecordingSession(
                self.options["output_root"],
                self.history["session_id"],
                [target[k] for k in ("x", "y", "z")],
                self.config,
            )
        if self.session is None:
            return
        self.session.mission_capture_count = count
        for entry in self.history.get("entries", []):
            view = int(entry["accepted_view"])
            stamp = int(entry.get("reconstruction_stamp_ns", 0))
            if view in self.processed_views:
                continue
            if stamp in self.pins:
                message, received = self.pins.pop(stamp)
                # The raw tuple was fresh when selected during capture; semantic
                # acceptance may take 75 seconds. Preserve its original freshness.
                raw = (
                    message.rgb,
                    message.depth,
                    message.camera_info,
                    message.mask,
                    received,
                )
                self.admit(
                    raw, "first_capture" if view == 1 else "stationary_capture", view
                )
                self.processed_views.add(view)
        if self.terminal and self.terminal.get("session_id") == self.session.session_id:
            expected = int(self.terminal.get("accepted_views", count))
            if (
                len(self.processed_views) >= expected
                or time.monotonic() - self.terminal_at > 3.0
            ):
                if len(self.processed_views) < expected:
                    self.session.reject("missing_stationary_keyframe")
                self.finish(
                    "complete" if self.terminal["state"] == "COMPLETE" else "failed"
                )

    def finish(self, state):
        """Finalize a session and release all target and sensor caches."""
        if self.session:
            for reason, value in self.rejections.items():
                self.session.rejected[reason] += value
            self.session.close(state)
            self.completed_sessions.append(self.session)
        self.session = None
        self.history = None
        self.terminal = None
        self.processed_views.clear()
        self.pins.clear()
        self.pairs.clear()
        self.masks.clear()
        self.rejections.clear()

    def process_event(self, kind, data):
        """Serialize sensor, acceptance and completion events."""
        if kind == "history":
            if self.history and data["session_id"] != self.history["session_id"]:
                self.finish("incomplete")
            self.history = data
        elif kind == "keyframe":
            self.pins[stamp_ns(data.rgb)] = (data, time.monotonic())
            while len(self.pins) > 4:
                self.pins.popitem(last=False)
                self.rejections["keyframe_evicted"] += 1
        elif kind == "terminal":
            if self.terminal is None or data.get("session_id") != self.terminal.get(
                "session_id"
            ):
                self.terminal_at = time.monotonic()
            self.terminal = data
        elif kind == "motion":
            mode, self.motion_state = data
            if mode not in ("MULTIVIEW_SCAN", "TIMED_STREAM", "TESSERACT_STREAM"):
                self.motion_state = ""
        elif kind == "info":
            self.latest_info = data
        elif kind == "pair":
            self.pairs[stamp_ns(data[0])] = data
        elif kind == "mask":
            self.masks[stamp_ns(data[0])] = data
        for cache in (self.pairs, self.masks):
            while len(cache) > 32:
                cache.popitem(last=False)
                self.rejections["unmatched_sensor_frame"] += 1
        self.reconcile()
        if (
            self.session
            and self.session.accepted
            and self.motion_state == "MOVING"
            and not self.terminal
        ):
            for stamp in sorted(set(self.pairs) & set(self.masks)):
                rgb, depth, info, paired_at, observed_state = self.pairs[stamp]
                info = info if info is not None else self.latest_info
                if info is None and time.monotonic() - paired_at < 0.1:
                    continue
                self.pairs.pop(stamp)
                mask, received = self.masks.pop(stamp)
                if observed_state == ("MULTIVIEW_SCAN", "MOVING"):
                    self.admit((rgb, depth, info, mask, received), "moving")

    def expire_pairs(self):
        """Count masks missing beyond the bounded synchronization cache age."""
        for stamp, raw in list(self.pairs.items()):
            if time.monotonic() - raw[3] > self.config.max_mask_age_sec:
                self.pairs.pop(stamp, None)
                reason = "stale_mask" if self.masks else "missing_mask"
                if self.session:
                    self.session.reject(reason)
                else:
                    self.rejections[reason] += 1

    def process_loop(self):
        """Drain control and image queues without blocking ROS callbacks."""
        try:
            while not self.stop_requested.is_set():
                try:
                    kind, data = self.events.get_nowait()
                except queue.Empty:
                    try:
                        kind, data = self.images.get(timeout=0.02)
                    except queue.Empty:
                        self.process_event("poll", None)
                        self.expire_pairs()
                        continue
                self.process_event(kind, data)
                self.expire_pairs()
        except Exception as error:
            self.get_logger().error("Reconstruction recorder failed: %s" % error)
            self.rejections["processing_error"] += 1
        finally:
            self.finish("incomplete")

    def publish_status(self):
        """Publish throttled aggregate diagnostics without image payloads."""
        session = self.session or (
            self.completed_sessions[-1] if self.completed_sessions else None
        )
        payload = dict(
            state="waiting_for_first_accepted_capture",
            rejected_by_reason=dict(self.rejections),
        )
        if session:
            payload = session.manifest()
            payload.pop("frames", None)
            payload["path"] = str(session.final_path or session.path)
        self.status_pub.publish(String(data=json.dumps(payload, allow_nan=False)))

    def destroy_node(self):
        """Flush incomplete work on SIGINT, node destruction and exceptions."""
        self.stop_requested.set()
        self.worker.join(12.0)
        return super().destroy_node()


def main(args=None):
    """Run the passive recorder until orderly shutdown."""
    rclpy.init(args=args)
    node = None

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    # Background shells may inherit SIGINT=SIG_IGN. Install explicit handlers
    # so bag replay and supervisor shutdown still flush the native journal.
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        node = ReconstructionRecorderNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
