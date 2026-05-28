#!/usr/bin/env python3
import argparse
import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class ImageViewer(Node):
    def __init__(self, topic):
        super().__init__('l515_opencv_viewer')
        self.topic = topic
        self.bridge = CvBridge()
        self.window = 'L515: %s' % topic
        self.sub = self.create_subscription(Image, topic, self.image_cb, qos_profile_sensor_data)
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        self.get_logger().info('Viewing %s. Press q in the image window to quit.' % topic)

    def image_cb(self, msg):
        try:
            frame = self.to_display_image(msg)
        except Exception as exc:
            self.get_logger().warn('Could not display %s image: %s' % (msg.encoding, exc))
            return

        cv2.imshow(self.window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            rclpy.shutdown()

    def to_display_image(self, msg):
        if msg.encoding in ('16UC1', 'mono16'):
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth = np.asarray(depth, dtype=np.float32)
            depth_m = depth * 0.001
            depth_m[~np.isfinite(depth_m)] = 0.0
            scaled = np.clip(depth_m / 2.0, 0.0, 1.0)
            gray = (scaled * 255.0).astype(np.uint8)
            return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

        if msg.encoding in ('32FC1',):
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth = np.asarray(depth, dtype=np.float32)
            depth[~np.isfinite(depth)] = 0.0
            scaled = np.clip(depth / 2.0, 0.0, 1.0)
            gray = (scaled * 255.0).astype(np.uint8)
            return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

        if msg.encoding in ('mono8', '8UC1'):
            gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')


def main():
    parser = argparse.ArgumentParser(description='Display a ROS 2 Image topic with OpenCV.')
    parser.add_argument('topic', nargs='?', default='/camera/color/image_raw')
    args = parser.parse_args()

    rclpy.init()
    node = ImageViewer(args.topic)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
