#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from piper_mobile_manipulation.msg import Target3D, TrackedTarget


class ManipulationTargetNode(Node):
    def __init__(self):
        super().__init__('manipulation_target_node')
        self.declare_parameter('tracked_topic', '/piper/tracked_target')
        self.declare_parameter('target_status_topic', '/piper/target_status')
        self.declare_parameter('manipulation_target_topic', '/piper/manipulation_target')
        self.declare_parameter('approach_offset_m', 0.10)
        self.declare_parameter('target_type', 'plant')
        self.declare_parameter('manipulation_mode', 'inspect')
        self.declare_parameter('stop_on_low_confidence', True)
        self.declare_parameter('stop_on_target_lost', True)

        self.target_status = 'SEARCHING'
        self.pub = self.create_publisher(Target3D, self.get_parameter('manipulation_target_topic').value, 10)
        self.tracked_sub = self.create_subscription(
            TrackedTarget,
            self.get_parameter('tracked_topic').value,
            self.tracked_cb,
            10,
        )
        self.status_sub = self.create_subscription(
            String,
            self.get_parameter('target_status_topic').value,
            self.status_cb,
            10,
        )
        self.get_logger().info('Manipulation target node ready; real arm motion is not commanded here')

    def status_cb(self, msg):
        self.target_status = msg.data

    def tracked_cb(self, msg):
        if not msg.valid:
            return
        if self.param_bool('stop_on_low_confidence') and self.target_status == 'LOW_CONFIDENCE':
            return
        if self.param_bool('stop_on_target_lost') and self.target_status == 'LOST':
            return

        out = Target3D()
        out.header = msg.header
        out.point = msg.position
        out.point.x += float(self.get_parameter('approach_offset_m').value)
        out.depth = out.point.z
        out.measurement_confidence = msg.confidence
        out.valid = True
        self.pub.publish(out)

    def param_bool(self, name):
        value = self.get_parameter(name).value
        if isinstance(value, str):
            return value.lower() in ('1', 'true', 'yes', 'on')
        return bool(value)


def main(args=None):
    rclpy.init(args=args)
    node = ManipulationTargetNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
