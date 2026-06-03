#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node

from piper_mobile_manipulation.msg import ServoCommand, TargetError


class FakeVisualServoNode(Node):
    def __init__(self):
        super().__init__('fake_visual_servo_node')
        self.declare_parameter('target_error_topic', '/piper/target_error')
        self.declare_parameter('servo_cmd_topic', '/piper/servo_cmd')
        self.declare_parameter('gain_xy', 0.8)
        self.declare_parameter('gain_z', 0.6)
        self.declare_parameter('max_lateral_speed_mps', 0.08)
        self.declare_parameter('max_vertical_speed_mps', 0.08)
        self.declare_parameter('max_forward_speed_mps', 0.06)
        self.declare_parameter('command_deadband_m', 0.0)

        self.pub = self.create_publisher(
            ServoCommand,
            self.get_parameter('servo_cmd_topic').value,
            10,
        )
        self.sub = self.create_subscription(
            TargetError,
            self.get_parameter('target_error_topic').value,
            self.error_cb,
            10,
        )
        self.get_logger().info(
            'Fake visual servo publishing %s from %s. No robot motion is commanded.'
            % (
                self.get_parameter('servo_cmd_topic').value,
                self.get_parameter('target_error_topic').value,
            )
        )

    def error_cb(self, msg):
        gain_xy = float(self.get_parameter('gain_xy').value)
        gain_z = float(self.get_parameter('gain_z').value)
        max_lateral = abs(float(self.get_parameter('max_lateral_speed_mps').value))
        max_vertical = abs(float(self.get_parameter('max_vertical_speed_mps').value))
        max_forward = abs(float(self.get_parameter('max_forward_speed_mps').value))
        deadband = max(0.0, float(self.get_parameter('command_deadband_m').value))

        out = ServoCommand()
        out.header = msg.header
        out.gain_xy = gain_xy
        out.gain_z = gain_z
        out.max_speed = max(max_lateral, max_vertical, max_forward)

        if not msg.valid:
            out.command = 'stop_no_target'
            out.aligned = False
            out.valid = False
            self.pub.publish(out)
            return

        out.aligned = bool(msg.centered and msg.at_distance)
        out.valid = True

        if out.aligned:
            out.command = 'stop_aligned'
            self.pub.publish(out)
            return

        ex = self.apply_deadband(float(msg.error.x), msg.position_tolerance, deadband)
        ey = self.apply_deadband(float(msg.error.y), msg.position_tolerance, deadband)
        ez = self.apply_deadband(float(msg.error.z), msg.distance_tolerance, deadband)

        out.linear.x = float(np.clip(gain_z * ez, -max_forward, max_forward))
        out.linear.y = float(np.clip(gain_xy * ex, -max_lateral, max_lateral))
        out.linear.z = float(np.clip(gain_xy * ey, -max_vertical, max_vertical))
        out.command = self.describe_command(out.linear.x, out.linear.y, out.linear.z)
        self.pub.publish(out)

    @staticmethod
    def apply_deadband(error, tolerance, extra_deadband):
        threshold = max(float(tolerance), float(extra_deadband))
        if abs(error) <= threshold:
            return 0.0
        return error

    @staticmethod
    def describe_command(forward, lateral, vertical):
        parts = []
        if lateral > 0.0:
            parts.append('right')
        elif lateral < 0.0:
            parts.append('left')

        if vertical > 0.0:
            parts.append('down')
        elif vertical < 0.0:
            parts.append('up')

        if forward > 0.0:
            parts.append('forward')
        elif forward < 0.0:
            parts.append('back')

        if not parts:
            return 'stop_deadband'
        return '_'.join(parts)


def main(args=None):
    rclpy.init(args=args)
    node = FakeVisualServoNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
