#!/usr/bin/env python3
"""Fail-closed RealSense timestamp watchdog with stationary-only recovery requests."""

import os
from pathlib import Path
import time

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, JointState

from piper_mobile_manipulation.camera_timestamp_health import TimestampHealthMonitor
from piper_mobile_manipulation.msg import CameraTimestampHealth


class CameraTimestampWatchdogNode(Node):
    def __init__(self):
        super().__init__('camera_timestamp_watchdog')
        defaults = {
            'image_topic': '/camera/color/camera_info',
            'joint_states_topic': '/joint_states_single',
            'health_topic': '/piper/camera_timestamp_health',
            'recovery_request_path': '/tmp/piper_vision_recovery/request.yaml',
            'enable_recovery_request': True,
            'max_timestamp_offset_sec': 0.5,
            'backward_tolerance_sec': 0.001,
            'healthy_frames_required': 15,
            'frame_timeout_sec': 2.0,
            'startup_grace_sec': 3.0,
            'joint_state_timeout_sec': 1.0,
            'joint_subscription_retry_sec': 10.0,
            'stationary_velocity_rad_s': 0.03,
            'stationary_position_tolerance_rad': 0.001,
            'stationary_duration_sec': 0.75,
            'unhealthy_frames_before_recovery': 5,
            'publish_period_sec': 0.2,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.monitor = TimestampHealthMonitor(
            self.get_parameter('max_timestamp_offset_sec').value,
            self.get_parameter('backward_tolerance_sec').value,
            self.get_parameter('healthy_frames_required').value,
        )
        self.started_at = time.monotonic()
        self.last_image_at = None
        self.last_joint_at = None
        self.stationary_anchor_positions = None
        self.stationary_since = None
        self.unhealthy_frames = 0
        self.recovery_latched = False
        self.arm_stationary = False
        self.stationarity_detail = 'joint feedback has not arrived'
        self.result = self.monitor.no_frames('waiting for the first camera frame')

        self.health_pub = self.create_publisher(
            CameraTimestampHealth, self.get_parameter('health_topic').value, 10)
        # CameraInfo is emitted with each color frame and carries its measurement
        # timestamp without forcing this safety node to deserialize a 640x480
        # RGB payload. The lightweight timestamp, joint, and timer callbacks use
        # Foxy's standard single-thread executor so subscription recreation can
        # never race an in-flight joint callback.
        self.image_sub = self.create_subscription(
            CameraInfo, self.get_parameter('image_topic').value,
            self.image_cb, qos_profile_sensor_data)
        # PiPER offers reliable joint feedback. Request reliability but retain
        # only the newest sample so the 200 Hz stream cannot build a motion-
        # authority backlog. In this Foxy/Fast DDS runtime a best-effort reader
        # was discovered but stopped delivering callbacks after a restart,
        # leaving stationary-only recovery permanently blocked.
        self.joint_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        # Retain both subscriptions for the node lifetime.  rclpy does not
        # guarantee that an unreferenced Python Subscription wrapper remains
        # alive; the joint endpoint was observed disappearing from the live
        # graph after startup, permanently blocking stationary recovery.
        self.joint_sub = None
        self.last_joint_subscription_at = -float('inf')
        self.recreate_joint_subscription(time.monotonic(), initial=True)
        self.create_timer(
            float(self.get_parameter('publish_period_sec').value), self.tick)
        self.get_logger().warn(
            'Camera timestamp watchdog is fail-closed; recovery requests require '
            'a stationary arm.')

    def ros_now_sec(self):
        return float(self.get_clock().now().nanoseconds) * 1.0e-9

    @staticmethod
    def stamp_sec(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9

    def image_cb(self, message):
        self.last_image_at = time.monotonic()
        self.result = self.monitor.evaluate(
            self.stamp_sec(message.header.stamp), self.ros_now_sec())
        if self.result.healthy:
            self.unhealthy_frames = 0
            self.recovery_latched = False
        else:
            self.unhealthy_frames += 1

    def joint_cb(self, message):
        now = time.monotonic()
        self.last_joint_at = now
        positions = list(message.position[:6])
        have_positions = len(positions) >= 6 and all(
            abs(float(value)) != float('inf') and float(value) == float(value)
            for value in positions)
        if have_positions:
            positions = [float(value) for value in positions]
            tolerance = float(
                self.get_parameter('stationary_position_tolerance_rad').value)
            known_stationary = (
                self.stationary_anchor_positions is None
                or max(abs(value - anchor) for value, anchor in zip(
                    positions, self.stationary_anchor_positions)) <= tolerance
            )
            if not known_stationary or self.stationary_anchor_positions is None:
                self.stationary_anchor_positions = positions
        else:
            # Retain a fail-closed fallback for nonstandard JointState publishers
            # which omit positions. PiPER position stability is preferred because
            # its instantaneous velocity telemetry is noisy at physical rest.
            velocities = list(message.velocity[:6])
            known_stationary = len(velocities) >= 6 and all(
                abs(float(value)) <= float(
                    self.get_parameter('stationary_velocity_rad_s').value)
                for value in velocities)
        if known_stationary:
            if self.stationary_since is None:
                self.stationary_since = now
        else:
            self.stationary_since = now if have_positions else None
        self.update_arm_stationary(now)

    def update_arm_stationary(self, now):
        joint_fresh = self.last_joint_at is not None and (
            now - self.last_joint_at <= float(
                self.get_parameter('joint_state_timeout_sec').value))
        settled = self.stationary_since is not None and (
            now - self.stationary_since >= float(
                self.get_parameter('stationary_duration_sec').value))
        self.arm_stationary = bool(joint_fresh and settled)
        if not joint_fresh:
            self.stationarity_detail = 'joint feedback is missing or stale'
        elif self.arm_stationary:
            self.stationarity_detail = 'joint positions are stable'
        else:
            elapsed = 0.0 if self.stationary_since is None \
                else max(0.0, now - self.stationary_since)
            self.stationarity_detail = 'joint positions settling for %.2fs' % elapsed

    def tick(self):
        now = time.monotonic()
        joint_stale = self.last_joint_at is None or now - self.last_joint_at > float(
            self.get_parameter('joint_state_timeout_sec').value)
        retry_sec = max(0.5, float(
            self.get_parameter('joint_subscription_retry_sec').value))
        if joint_stale and now - self.last_joint_subscription_at >= retry_sec:
            self.recreate_joint_subscription(now)
        self.update_arm_stationary(now)
        if self.last_image_at is None or now - self.last_image_at > float(
                self.get_parameter('frame_timeout_sec').value):
            self.result = self.monitor.no_frames()
            self.unhealthy_frames = max(
                self.unhealthy_frames,
                int(self.get_parameter('unhealthy_frames_before_recovery').value))

        can_request = (
            not self.result.healthy
            and self.arm_stationary
            and not self.recovery_latched
            and self.param_bool('enable_recovery_request')
            and now - self.started_at >= float(
                self.get_parameter('startup_grace_sec').value)
            and self.unhealthy_frames >= int(
                self.get_parameter('unhealthy_frames_before_recovery').value)
        )
        if can_request:
            self.write_recovery_request()
            self.recovery_latched = True
        self.publish_health()

    def recreate_joint_subscription(self, now, initial=False):
        """Restore the read-only joint endpoint without weakening safety gates."""
        if self.joint_sub is not None:
            self.destroy_subscription(self.joint_sub)
        self.joint_sub = self.create_subscription(
            JointState,
            self.get_parameter('joint_states_topic').value,
            self.joint_cb,
            self.joint_qos,
        )
        self.last_joint_subscription_at = now
        if not initial:
            self.get_logger().warn(
                'Recreated stale joint-feedback subscription; recovery remains '
                'blocked until fresh stable positions arrive.')

    def write_recovery_request(self):
        requested = Path(str(self.get_parameter('recovery_request_path').value))
        allowed_root = Path('/tmp/piper_vision_recovery')
        try:
            requested.resolve().relative_to(allowed_root.resolve())
        except ValueError:
            self.get_logger().error(
                'Refusing recovery request outside %s: %s' % (allowed_root, requested))
            return
        requested.parent.mkdir(parents=True, exist_ok=True)
        temporary = requested.with_name(requested.name + '.tmp.%d' % os.getpid())
        payload = {
            'state': self.result.state,
            'reason': self.result.reason,
            'offset_sec': (
                float(self.result.offset_sec)
                if abs(float(self.result.offset_sec)) != float('inf') else None),
            'arm_stationary': True,
            'requested_at_unix_sec': time.time(),
            'dry_run': True,
            'real_arm_motion': False,
        }
        temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
        os.replace(str(temporary), str(requested))
        self.get_logger().error(
            'VISION_RECOVERY_REQUESTED: %s' % self.result.reason)

    def publish_health(self):
        out = CameraTimestampHealth()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'camera_color_optical_frame'
        out.state = self.result.state
        out.healthy = bool(self.result.healthy)
        offset = float(self.result.offset_sec)
        out.offset_sec = offset if abs(offset) != float('inf') else 1.0e9
        out.consecutive_healthy_frames = int(self.result.consecutive_healthy_frames)
        out.image_stamp_monotonic = bool(self.result.monotonic)
        out.arm_stationary = bool(self.arm_stationary)
        out.recovery_requested = bool(self.recovery_latched)
        waiting = not self.result.healthy and not self.arm_stationary
        out.reason = (
            self.result.reason + '; automatic recovery waiting: ' +
            self.stationarity_detail
            if waiting else self.result.reason)
        out.dry_run = True
        out.real_arm_motion = False
        self.health_pub.publish(out)

    def param_bool(self, name):
        value = self.get_parameter(name).value
        return str(value).lower() in ('1', 'true', 'yes', 'on') \
            if isinstance(value, str) else bool(value)


def main(args=None):
    rclpy.init(args=args)
    node = CameraTimestampWatchdogNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
