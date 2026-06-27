#!/usr/bin/env python3
"""Convert the GPU SAM2 target mask into the Detection2D geometry contract."""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from piper_mobile_manipulation.msg import Detection2D


class MaskToDetectionNode(Node):
    def __init__(self):
        super().__init__('mask_to_detection_node')
        self.declare_parameter('mask_topic', '/piper/sam2_target_mask')
        self.declare_parameter('detection_topic', '/piper/sam2_detection_2d')
        self.declare_parameter('min_area_px', 100)
        self.bridge = CvBridge()
        self.pub = self.create_publisher(
            Detection2D, self.get_parameter('detection_topic').value, 10
        )
        self.create_subscription(
            Image, self.get_parameter('mask_topic').value, self.mask_cb, qos_profile_sensor_data
        )

    def mask_cb(self, msg):
        out = Detection2D()
        out.header = msg.header
        try:
            mask = np.asarray(self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')) > 0
        except Exception as exc:
            self.get_logger().warn('SAM2 mask conversion failed: %s' % exc)
            self.pub.publish(out)
            return
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if count <= 1:
            self.pub.publish(out)
            return
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < int(self.get_parameter('min_area_px').value):
            self.pub.publish(out)
            return
        out.u = float(centroids[component, 0])
        out.v = float(centroids[component, 1])
        out.width = float(stats[component, cv2.CC_STAT_WIDTH])
        out.height = float(stats[component, cv2.CC_STAT_HEIGHT])
        out.confidence = float(min(1.0, area / max(1.0, out.width * out.height)))
        out.valid = True
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = MaskToDetectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
