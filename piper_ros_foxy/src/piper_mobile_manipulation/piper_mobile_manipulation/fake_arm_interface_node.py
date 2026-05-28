#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from piper_mobile_manipulation.msg import ManipulationCommand


class FakeArmInterfaceNode(Node):
    def __init__(self):
        super().__init__('fake_arm_interface_node')
        self.declare_parameter('command_topic', '/piper/manipulation_command')
        self.sub = self.create_subscription(
            ManipulationCommand,
            self.get_parameter('command_topic').value,
            self.command_cb,
            10,
        )
        self.get_logger().info(
            'FAKE arm interface active. It only prints commands and never moves the PiPER arm.'
        )

    def command_cb(self, msg):
        p = msg.target_pose.pose.position
        self.get_logger().info(
            '[FAKE ARM] command=%s execute=%s frame=%s target=(%.3f, %.3f, %.3f) speed=%.3f dist=%.3f'
            % (
                msg.command_type,
                msg.execute,
                msg.target_pose.header.frame_id,
                p.x,
                p.y,
                p.z,
                msg.speed_limit,
                msg.distance_limit,
            )
        )
        self.get_logger().info(
            '[FAKE ARM] Safety note: no real PiPER topic is used. TODO add real PiPER publisher only after command topic and message type are confirmed.'
        )


def main(args=None):
    rclpy.init(args=args)
    node = FakeArmInterfaceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
