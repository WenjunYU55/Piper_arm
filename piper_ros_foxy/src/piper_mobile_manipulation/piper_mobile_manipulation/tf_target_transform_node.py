#!/usr/bin/env python3
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener

from piper_mobile_manipulation.msg import HandoffTarget


class TfTargetTransformNode(Node):
    def __init__(self):
        super().__init__('tf_target_transform_node')
        self.declare_parameter('input_topic', '/piper/handoff_target')
        self.declare_parameter('piper_base_frame', 'piper_base_link')
        self.declare_parameter('camera_frame', 'l515_color_optical_frame')
        self.declare_parameter('piper_output_topic', '/piper/target_piper_base')
        self.declare_parameter('camera_output_topic', '/piper/target_camera')
        self.declare_parameter('transform_timeout_s', 0.2)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timeout_s = float(self.get_parameter('transform_timeout_s').value)
        self.piper_base_frame = self.get_parameter('piper_base_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value

        self.piper_pub = self.create_publisher(
            HandoffTarget, self.get_parameter('piper_output_topic').value, 10
        )
        self.camera_pub = self.create_publisher(
            HandoffTarget, self.get_parameter('camera_output_topic').value, 10
        )
        self.sub = self.create_subscription(
            HandoffTarget, self.get_parameter('input_topic').value, self.target_cb, 10
        )
        self.get_logger().info(
            'Transforming targets into %s and %s'
            % (self.piper_base_frame, self.camera_frame)
        )

    def target_cb(self, msg):
        self.publish_transformed(msg, self.piper_base_frame, self.piper_pub)
        self.publish_transformed(msg, self.camera_frame, self.camera_pub)

    def publish_transformed(self, msg, target_frame, publisher):
        out = HandoffTarget()
        out.header = msg.header
        out.pose = msg.pose
        out.target_type = msg.target_type
        out.confidence = msg.confidence
        if not msg.valid:
            out.valid = False
            out.reason = 'input handoff invalid: %s' % msg.reason
            publisher.publish(out)
            return

        try:
            transformed_pose = self.tf_buffer.transform(
                msg.pose, target_frame, timeout=Duration(seconds=self.timeout_s)
            )
            out.header = transformed_pose.header
            out.pose = transformed_pose
            out.valid = True
            out.reason = 'transformed from %s to %s' % (msg.pose.header.frame_id, target_frame)
            self.get_logger().info(
                'TF ok %s -> %s pos=(%.3f, %.3f, %.3f)'
                % (
                    msg.pose.header.frame_id,
                    target_frame,
                    transformed_pose.pose.position.x,
                    transformed_pose.pose.position.y,
                    transformed_pose.pose.position.z,
                )
            )
        except TransformException as exc:
            out.valid = False
            out.reason = 'TF failed %s -> %s: %s' % (
                msg.pose.header.frame_id,
                target_frame,
                str(exc),
            )
            self.get_logger().warn(out.reason)
        publisher.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TfTargetTransformNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
