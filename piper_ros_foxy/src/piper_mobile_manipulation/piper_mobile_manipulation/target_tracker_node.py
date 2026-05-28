#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

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

        self.prediction_horizon = float(self.get_parameter('prediction_horizon_s').value)
        self.max_missed = int(self.get_parameter('max_missed_frames').value)
        self.min_track_frames = int(self.get_parameter('min_track_frames').value)
        self.stable_speed_threshold = float(self.get_parameter('stable_speed_threshold_mps').value)
        self.stable_time_s = float(self.get_parameter('stable_time_s').value)
        self.filter = ConstantVelocityKalmanFilter(
            self.get_parameter('process_noise').value,
            self.get_parameter('measurement_noise').value,
        )
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
        self.get_logger().info('Target tracker ready; arm should use filtered target only')

    def target_cb(self, msg):
        now = self.get_clock().now()
        out = TrackedTarget()
        out.header = msg.header

        if not msg.valid:
            self.missed_frames += 1
            if self.missed_frames > self.max_missed:
                self.filter.reset()
                self.track_frames = 0
                self.stable_since = None
            out.valid = False
            out.confidence = 0.0
            self.pub.publish(out)
            return

        if self.last_time is None:
            dt = 0.033
        else:
            dt = max((now - self.last_time).nanoseconds * 1e-9, 1e-3)
        self.last_time = now
        state = self.filter.step([msg.point.x, msg.point.y, msg.point.z], dt)
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
        out.confidence = float(min(1.0, self.track_frames / float(max(self.min_track_frames, 1))))
        out.stable = (
            self.track_frames >= self.min_track_frames
            and stable_now
            and stable_duration >= self.stable_time_s
        )
        out.valid = self.track_frames >= self.min_track_frames
        self.pub.publish(out)
        self.get_logger().info(
            'TrackedTarget valid=%s stable=%s pos=(%.3f, %.3f, %.3f) speed=%.3f'
            % (out.valid, out.stable, x, y, z, speed)
        )


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
