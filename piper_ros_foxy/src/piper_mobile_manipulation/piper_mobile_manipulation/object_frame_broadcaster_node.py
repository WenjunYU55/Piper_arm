#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from piper_mobile_manipulation.msg import TrackedTarget


class ObjectFrameBroadcasterNode(Node):
    def __init__(self):
        super().__init__('object_frame_broadcaster_node')
        self.declare_parameter('tracked_topic', '/piper/tracked_target')
        self.declare_parameter('object_frame', 'tracked_object_frame')
        self.declare_parameter('min_confidence', 0.05)
        self.declare_parameter('publish_predicted_frame', True)
        self.declare_parameter('predicted_object_frame', 'predicted_object_frame')
        self.declare_parameter('republish_hz', 10.0)

        self.broadcaster = TransformBroadcaster(self)
        self.latest_transforms = []
        self.sub = self.create_subscription(
            TrackedTarget,
            self.get_parameter('tracked_topic').value,
            self.tracked_cb,
            10,
        )
        republish_hz = max(float(self.get_parameter('republish_hz').value), 0.1)
        self.timer = self.create_timer(1.0 / republish_hz, self.timer_cb)
        self.get_logger().info(
            'Object TF broadcaster ready: %s -> tracked object frames'
            % self.get_parameter('tracked_topic').value
        )

    def tracked_cb(self, msg):
        if not msg.valid:
            return
        if float(msg.confidence) < float(self.get_parameter('min_confidence').value):
            return
        if not msg.header.frame_id:
            return
        if not self.finite_point(msg.position):
            return

        transforms = [
            self.make_transform(
                msg.header,
                self.get_parameter('object_frame').value,
                msg.position.x,
                msg.position.y,
                msg.position.z,
            )
        ]
        if self.get_parameter('publish_predicted_frame').value and self.finite_point(msg.predicted_position):
            transforms.append(
                self.make_transform(
                    msg.header,
                    self.get_parameter('predicted_object_frame').value,
                    msg.predicted_position.x,
                    msg.predicted_position.y,
                    msg.predicted_position.z,
                )
            )
        self.latest_transforms = transforms
        self.publish_latest()

    def timer_cb(self):
        self.publish_latest()

    def publish_latest(self):
        if not self.latest_transforms:
            return
        now = self.get_clock().now().to_msg()
        outgoing = []
        for transform in self.latest_transforms:
            copy = TransformStamped()
            copy.header = transform.header
            copy.header.stamp = now
            copy.child_frame_id = transform.child_frame_id
            copy.transform = transform.transform
            outgoing.append(copy)
        self.broadcaster.sendTransform(outgoing)

    def make_transform(self, header, child_frame_id, x, y, z):
        transform = TransformStamped()
        transform.header = header
        transform.child_frame_id = child_frame_id
        transform.transform.translation.x = float(x)
        transform.transform.translation.y = float(y)
        transform.transform.translation.z = float(z)
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0
        return transform

    @staticmethod
    def finite_point(point):
        return (
            math.isfinite(float(point.x))
            and math.isfinite(float(point.y))
            and math.isfinite(float(point.z))
        )


def main(args=None):
    rclpy.init(args=args)
    node = ObjectFrameBroadcasterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
