#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from piper_mobile_manipulation.msg import TargetError, TrackedTarget


class TargetErrorNode(Node):
    def __init__(self):
        super().__init__('target_error_node')
        self.declare_parameter('tracked_topic', '/piper/tracked_target')
        self.declare_parameter('error_topic', '/piper/target_error')
        self.declare_parameter('desired_distance_m', 0.55)
        self.declare_parameter('position_tolerance_m', 0.03)
        self.declare_parameter('distance_tolerance_m', 0.04)

        self.pub = self.create_publisher(
            TargetError, self.get_parameter('error_topic').value, 10
        )
        self.sub = self.create_subscription(
            TrackedTarget,
            self.get_parameter('tracked_topic').value,
            self.tracked_cb,
            10,
        )
        self.get_logger().info(
            'Target error publishing %s from %s'
            % (
                self.get_parameter('error_topic').value,
                self.get_parameter('tracked_topic').value,
            )
        )

    def tracked_cb(self, msg):
        desired_distance = float(self.get_parameter('desired_distance_m').value)
        position_tolerance = float(self.get_parameter('position_tolerance_m').value)
        distance_tolerance = float(self.get_parameter('distance_tolerance_m').value)

        out = TargetError()
        out.header = msg.header
        out.desired_distance = desired_distance
        out.position_tolerance = position_tolerance
        out.distance_tolerance = distance_tolerance

        if not msg.valid:
            out.valid = False
            self.pub.publish(out)
            return

        out.error.x = msg.position.x
        out.error.y = msg.position.y
        out.error.z = msg.position.z - desired_distance
        out.distance_error = out.error.z
        out.centered = (
            abs(out.error.x) <= position_tolerance
            and abs(out.error.y) <= position_tolerance
        )
        out.at_distance = abs(out.distance_error) <= distance_tolerance
        out.valid = True
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TargetErrorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
