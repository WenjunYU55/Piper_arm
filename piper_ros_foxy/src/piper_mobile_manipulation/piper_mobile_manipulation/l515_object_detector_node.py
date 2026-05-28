#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from piper_mobile_manipulation.msg import Detection2D


class L515ObjectDetectorNode(Node):
    def __init__(self):
        super().__init__('l515_object_detector_node')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('detection_topic', '/piper/detection_2d')
        self.declare_parameter('debug_image_topic', '/piper/detection_debug_image')
        self.declare_parameter('hsv_lower', [30, 50, 50])
        self.declare_parameter('hsv_upper', [90, 255, 255])
        self.declare_parameter('min_contour_area', 200.0)
        self.declare_parameter('max_contour_area', 100000.0)

        self.bridge = CvBridge()
        self.hsv_lower = np.array(self.get_parameter('hsv_lower').value, dtype=np.uint8)
        self.hsv_upper = np.array(self.get_parameter('hsv_upper').value, dtype=np.uint8)
        self.min_area = float(self.get_parameter('min_contour_area').value)
        self.max_area = float(self.get_parameter('max_contour_area').value)

        self.pub = self.create_publisher(
            Detection2D, self.get_parameter('detection_topic').value, 10
        )
        self.debug_pub = self.create_publisher(
            Image, self.get_parameter('debug_image_topic').value, qos_profile_sensor_data
        )
        self.sub = self.create_subscription(
            Image, self.get_parameter('image_topic').value, self.image_cb, qos_profile_sensor_data
        )
        self.get_logger().info('HSV detector listening on %s' % self.get_parameter('image_topic').value)

    def image_cb(self, image_msg):
        out = Detection2D()
        out.header = image_msg.header
        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as exc:
            out.valid = False
            self.pub.publish(out)
            self.get_logger().warn('cv_bridge failed: %s' % exc)
            return

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area <= area <= self.max_area and area > best_area:
                best = contour
                best_area = area

        if best is None:
            out.valid = False
            out.confidence = 0.0
            debug = image.copy()
        else:
            x, y, w, h = cv2.boundingRect(best)
            out.u = float(x + w / 2.0)
            out.v = float(y + h / 2.0)
            out.width = float(w)
            out.height = float(h)
            out.confidence = float(min(1.0, best_area / self.max_area))
            out.valid = True
            debug = image.copy()
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(debug, (int(out.u), int(out.v)), 4, (0, 0, 255), -1)

        self.pub.publish(out)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))
        if out.valid:
            self.get_logger().info(
                'Detection2D u=%.1f v=%.1f size=(%.1f, %.1f) conf=%.2f'
                % (out.u, out.v, out.width, out.height, out.confidence)
            )


def main(args=None):
    rclpy.init(args=args)
    node = L515ObjectDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
