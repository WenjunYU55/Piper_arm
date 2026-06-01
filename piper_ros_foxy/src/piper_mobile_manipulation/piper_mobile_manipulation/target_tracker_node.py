#!/usr/bin/env python3
import math

import rclpy
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener

from piper_mobile_manipulation.msg import Target3D, TrackedTarget
from piper_mobile_manipulation.utils.kalman_filter import ConstantVelocityKalmanFilter


class TargetTrackerNode(Node):
    def __init__(self):
        super().__init__('target_tracker_node')
        self.declare_parameter('target_topic', '/piper/target_3d')
        self.declare_parameter('tracked_topic', '/piper/tracked_target')
        self.declare_parameter('prediction_horizon_s', 0.3)
        self.declare_parameter('max_missed_frames', 10)
        self.declare_parameter('min_track_frames', 5)
        self.declare_parameter('stable_speed_threshold_mps', 0.03)
        self.declare_parameter('stable_time_s', 0.4)
        self.declare_parameter('process_noise', 0.05)
        self.declare_parameter('measurement_noise', 0.02)
        self.declare_parameter('use_tf_transform', True)
        self.declare_parameter('piper_base_frame', 'piper_base_link')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('transform_timeout_s', 0.2)
        self.declare_parameter('min_measurement_confidence', 0.05)
        self.declare_parameter('confidence_noise_scale', 4.0)

        self.prediction_horizon = float(self.get_parameter('prediction_horizon_s').value)
        self.max_missed = int(self.get_parameter('max_missed_frames').value)
        self.min_track_frames = int(self.get_parameter('min_track_frames').value)
        self.stable_speed_threshold = float(self.get_parameter('stable_speed_threshold_mps').value)
        self.stable_time_s = float(self.get_parameter('stable_time_s').value)
        self.use_tf_transform = bool(self.get_parameter('use_tf_transform').value)
        self.output_frame = self.get_parameter('piper_base_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.transform_timeout_s = float(self.get_parameter('transform_timeout_s').value)
        self.min_measurement_confidence = float(self.get_parameter('min_measurement_confidence').value)
        self.confidence_noise_scale = float(self.get_parameter('confidence_noise_scale').value)
        self.base_measurement_noise = float(self.get_parameter('measurement_noise').value)
        self.filter = ConstantVelocityKalmanFilter(
            self.get_parameter('process_noise').value,
            self.base_measurement_noise,
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.last_time = None
        self.track_frames = 0
        self.missed_frames = 0
        self.stable_since = None

        self.pub = self.create_publisher(
            TrackedTarget, self.get_parameter('tracked_topic').value, 10
        )
        self.sub = self.create_subscription(
            Target3D, self.get_parameter('target_topic').value, self.target_cb, 10
        )
        self.get_logger().info(
            'Target tracker ready; output_frame=%s tf=%s'
            % (self.output_frame, self.use_tf_transform)
        )

    def target_cb(self, msg):
        now = self.get_clock().now()
        out = TrackedTarget()
        out.header = msg.header

        if not msg.valid:
            self.publish_invalid(out)
            return

        measurement_confidence = float(msg.measurement_confidence)
        if measurement_confidence < self.min_measurement_confidence:
            self.get_logger().warn(
                'Target3D rejected low confidence %.2f < %.2f'
                % (measurement_confidence, self.min_measurement_confidence)
            )
            self.publish_invalid(out)
            return

        measurement = self.measurement_in_output_frame(msg)
        if measurement is None:
            self.publish_invalid(out)
            return

        if self.last_time is None:
            dt = 0.033
        else:
            dt = max((now - self.last_time).nanoseconds * 1e-9, 1e-3)
        self.last_time = now
        self.filter.measurement_noise = self.scaled_measurement_noise(measurement_confidence)
        state = self.filter.step(measurement, dt)
        self.track_frames += 1
        self.missed_frames = 0

        x, y, z, vx, vy, vz = state
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        stable_now = speed <= self.stable_speed_threshold
        if stable_now:
            if self.stable_since is None:
                self.stable_since = now
            stable_duration = (now - self.stable_since).nanoseconds * 1e-9
        else:
            self.stable_since = None
            stable_duration = 0.0

        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.output_frame if self.use_tf_transform else msg.header.frame_id
        out.position.x = float(x)
        out.position.y = float(y)
        out.position.z = float(z)
        out.velocity.x = float(vx)
        out.velocity.y = float(vy)
        out.velocity.z = float(vz)
        out.predicted_position.x = float(x + vx * self.prediction_horizon)
        out.predicted_position.y = float(y + vy * self.prediction_horizon)
        out.predicted_position.z = float(z + vz * self.prediction_horizon)
        out.speed = float(speed)
        track_confidence = float(min(1.0, self.track_frames / float(max(self.min_track_frames, 1))))
        out.confidence = float(track_confidence * measurement_confidence)
        out.stable = (
            self.track_frames >= self.min_track_frames
            and stable_now
            and stable_duration >= self.stable_time_s
        )
        out.valid = self.track_frames >= self.min_track_frames
        self.pub.publish(out)
        self.get_logger().info(
            'TrackedTarget frame=%s valid=%s stable=%s pos=(%.3f, %.3f, %.3f) speed=%.3f conf=%.2f'
            % (out.header.frame_id, out.valid, out.stable, x, y, z, speed, out.confidence)
        )

    def publish_invalid(self, out):
        self.missed_frames += 1
        if self.missed_frames > self.max_missed:
            self.filter.reset()
            self.track_frames = 0
            self.stable_since = None
            self.last_time = None
        out.valid = False
        out.confidence = 0.0
        if self.use_tf_transform:
            out.header.frame_id = self.output_frame
        self.pub.publish(out)

    def measurement_in_output_frame(self, msg):
        if not self.use_tf_transform:
            return [msg.point.x, msg.point.y, msg.point.z]

        point = PointStamped()
        point.header = msg.header
        if not point.header.frame_id:
            point.header.frame_id = self.camera_frame
        point.point = msg.point
        try:
            transformed = self.tf_buffer.transform(
                point,
                self.output_frame,
                timeout=Duration(seconds=self.transform_timeout_s),
            )
        except TransformException as exc:
            self.get_logger().warn(
                'TF failed %s -> %s: %s'
                % (point.header.frame_id, self.output_frame, str(exc))
            )
            return None
        return [
            transformed.point.x,
            transformed.point.y,
            transformed.point.z,
        ]

    def scaled_measurement_noise(self, confidence):
        confidence = max(float(confidence), self.min_measurement_confidence)
        confidence = min(confidence, 1.0)
        return self.base_measurement_noise * (1.0 + (1.0 - confidence) * self.confidence_noise_scale)


def main(args=None):
    rclpy.init(args=args)
    node = TargetTrackerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
