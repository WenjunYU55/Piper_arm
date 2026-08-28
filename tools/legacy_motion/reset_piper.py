#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import os


class ResetPiper(Node):
    def __init__(self):
        super().__init__("reset_piper")
        self.current_msg = None
        self.publish_count = 0
        self.max_publish_count = 120

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
        self.get_logger().info("Waiting for Piper feedback before reset command...")

    def feedback_callback(self, msg):
        self.current_msg = msg

    def timer_callback(self):
        if self.current_msg is None:
            return

        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "piper_single"
        cmd.name = [
            "joint1", "joint2", "joint3",
            "joint4", "joint5", "joint6", "joint7",
        ]

        # Neutral joint target plus open gripper. Joint values are radians/meters.
        cmd.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        cmd.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
        cmd.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]

        if self.publish_count == 0:
            self.get_logger().info(f"Commanding reset target: {cmd.position}")

        if self.publish_count < self.max_publish_count:
            self.pub.publish(cmd)
            self.publish_count += 1
            if self.publish_count % 20 == 0:
                self.get_logger().info(f"Published reset command {self.publish_count}/{self.max_publish_count}")
        else:
            final_position = list(self.current_msg.position[:7])
            self.get_logger().info(f"Finished reset command. Feedback now: {final_position}")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ResetPiper()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
