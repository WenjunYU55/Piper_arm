#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import os


class TinyJointTest(Node):
    def __init__(self):
        super().__init__("tiny_joint_test")
        self.current_msg = None
        self.command_msg = None
        self.start_position = None
        self.target_position = [0.2, 0.2, -0.2, 0.3, -0.2, 0.5, 0.01]
        self.publish_count = 0
        self.max_publish_count = 100

        self.sub = self.create_subscription(
            JointState,
            "/joint_states_single",
            self.feedback_callback,
            10
        )

        self.pub = self.create_publisher(
            JointState,
            os.environ.get("PIPER_JOINT_CTRL_TOPIC", "/joint_ctrl_single"),
            10
        )

        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info("Waiting for feedback...")

    def feedback_callback(self, msg):
        self.current_msg = msg

    def timer_callback(self):
        if self.current_msg is None:
            return

        if self.command_msg is None:
            current = list(self.current_msg.position)
            self.start_position = current[:7]

            cmd = JointState()
            cmd.header.frame_id = "piper_single"

            # Command format is 7 values: joint1 to joint6 + gripper.
            cmd.name = [
                "joint1", "joint2", "joint3",
                "joint4", "joint5", "joint6", "joint7"
            ]

            # This is the exact joint target from the Piper ROS foxy README.
            cmd.position = self.target_position

            cmd.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
            cmd.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]

            self.command_msg = cmd

            self.get_logger().info(f"Commanding README target: {cmd.position}")

        if self.publish_count < self.max_publish_count:
            self.command_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(self.command_msg)
            self.publish_count += 1

            if self.publish_count % 10 == 0:
                self.get_logger().info(f"Published {self.publish_count}/100")
        else:
            final_position = list(self.current_msg.position[:7])
            self.get_logger().info(
                f"Finished. feedback start={self.start_position}, "
                f"final={final_position}, target={self.target_position}"
            )
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = TinyJointTest()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
