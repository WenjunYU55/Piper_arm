#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import os


class ResetArm(Node):
    def __init__(self):
        super().__init__("reset_arm")
        self.current_msg = None
        self.command_msg = None
        self.publish_count = 0
        self.max_publish_count = 120
        self.target_position = [
            -1.5502482800000001,
            -0.040347972,
            0.034103020000000005,
            0.018979072000000003,
            0.320917268,
            1.07777754,
            0.01981,
        ]

        self.sub = self.create_subscription(
            JointState,
            "/joint_states_single",
            self.feedback_callback,
            10,
        )

        self.pub = self.create_publisher(
            JointState,
            os.environ.get("PIPER_JOINT_CTRL_TOPIC", "/joint_ctrl_single"),
            10,
        )

        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info("Waiting for Piper feedback before reset...")

    def feedback_callback(self, msg):
        self.current_msg = msg

    def timer_callback(self):
        if self.current_msg is None:
            return

        if self.command_msg is None:
            cmd = JointState()
            cmd.header.frame_id = "piper_single"
            cmd.name = [
                "joint1", "joint2", "joint3",
                "joint4", "joint5", "joint6", "joint7",
            ]
            cmd.position = self.target_position
            cmd.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
            cmd.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
            self.command_msg = cmd

            current = list(self.current_msg.position[:7])
            self.get_logger().info(f"Current joints: {current}")
            self.get_logger().info(f"Reset target: {self.target_position}")

        if self.publish_count < self.max_publish_count:
            self.command_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(self.command_msg)
            self.publish_count += 1

            if self.publish_count % 20 == 0:
                self.get_logger().info(f"Published reset command {self.publish_count}/120")
        else:
            final_position = list(self.current_msg.position[:7])
            self.get_logger().info(f"Reset command finished. Feedback: {final_position}")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ResetArm()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
