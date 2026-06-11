#!/usr/bin/env python3
import os
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from piper_mobile_manipulation.msg import Detection2D


class YoloSegDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_seg_detector_node')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('detection_topic', '/piper/detection_2d')
        self.declare_parameter('mask_topic', '/piper/detection_mask')
        self.declare_parameter('debug_image_topic', '/piper/detection_debug_image')
        self.declare_parameter('model_path', '')
        self.declare_parameter('config_path', '')
        self.declare_parameter('detector_rate_hz', 10.0)
        self.declare_parameter('confidence_threshold', 0.35)
        self.declare_parameter('publish_mask', True)
        self.declare_parameter('publish_debug_image', True)

        self.bridge = CvBridge()
        self.last_run_time = 0.0
        self.warned_model = False
        self.pub = self.create_publisher(Detection2D, self.get_parameter('detection_topic').value, 10)
        self.mask_pub = self.create_publisher(Image, self.get_parameter('mask_topic').value, qos_profile_sensor_data)
        self.debug_pub = self.create_publisher(Image, self.get_parameter('debug_image_topic').value, qos_profile_sensor_data)
        self.sub = self.create_subscription(Image, self.get_parameter('image_topic').value, self.image_cb, qos_profile_sensor_data)
        self.get_logger().info('YOLO segmentation detector placeholder ready')

    def image_cb(self, image_msg):
        rate_hz = max(float(self.get_parameter('detector_rate_hz').value), 0.1)
        now = time.monotonic()
        if now - self.last_run_time < 1.0 / rate_hz:
            return
        self.last_run_time = now

        out = Detection2D()
        out.header = image_msg.header
        out.valid = False
        out.confidence = 0.0
        self.pub.publish(out)

        model_path = str(self.get_parameter('model_path').value)
        if not model_path or not os.path.exists(os.path.expanduser(model_path)):
            if not self.warned_model:
                self.get_logger().error(
                    'YOLO-seg model_path is missing or does not exist; publishing invalid detections without crashing.'
                )
                self.warned_model = True
            self.publish_debug(image_msg, 'YOLO-seg model missing')
            return

        self.get_logger().warn('YOLO-seg backend is not implemented yet for model_path=%s' % model_path)
        self.publish_debug(image_msg, 'YOLO-seg backend placeholder')

    def publish_debug(self, image_msg, text):
        if not bool(self.get_parameter('publish_debug_image').value):
            return
        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            cv2.putText(image, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2, cv2.LINE_AA)
            debug_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            debug_msg.header = image_msg.header
            self.debug_pub.publish(debug_msg)
        except Exception as exc:
            self.get_logger().warn('debug image publish failed: %s' % exc)


def main(args=None):
    rclpy.init(args=args)
    node = YoloSegDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
