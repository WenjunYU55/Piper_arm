#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from piper_mobile_manipulation.msg import HandoffTarget


class TargetHandoffNode(Node):
    def __init__(self):
        super().__init__('target_handoff_node')
        self.declare_parameter('input_topic', '/base/target_pose')
        self.declare_parameter('output_topic', '/piper/handoff_target')
        self.declare_parameter('target_type', 'unknown')
        self.declare_parameter('default_confidence', 0.8)
        self.declare_parameter('min_confidence', 0.1)

        self.target_type = self.get_parameter('target_type').value
        self.default_confidence = float(self.get_parameter('default_confidence').value)
        self.min_confidence = float(self.get_parameter('min_confidence').value)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.pub = self.create_publisher(HandoffTarget, output_topic, 10)
        self.sub = self.create_subscription(PoseStamped, input_topic, self.pose_cb, 10)
        self.get_logger().info('Waiting for base target poses on %s' % input_topic)

    def pose_cb(self, pose):
        msg = HandoffTarget()
        msg.header = pose.header
        msg.pose = pose
        msg.target_type = self.target_type
        msg.confidence = self.default_confidence

        if not pose.header.frame_id:
            msg.valid = False
            msg.reason = 'missing frame_id'
        elif self.default_confidence < self.min_confidence:
            msg.valid = False
            msg.reason = 'confidence below threshold'
        else:
            msg.valid = True
            msg.reason = 'base handoff accepted'

        self.pub.publish(msg)
        self.get_logger().info(
            'Handoff target valid=%s frame=%s pos=(%.3f, %.3f, %.3f) reason=%s'
            % (
                msg.valid,
                pose.header.frame_id,
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
                msg.reason,
            )
        )


def main(args=None):
    rclpy.init(args=args)
    node = TargetHandoffNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
