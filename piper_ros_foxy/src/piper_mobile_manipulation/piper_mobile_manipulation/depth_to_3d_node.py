#!/usr/bin/env python3
import math

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from piper_mobile_manipulation.msg import Detection2D, Target3D


class DepthTo3DNode(Node):
    def __init__(self):
        super().__init__('depth_to_3d_node')
        self.declare_parameter('detection_topic', '/piper/detection_2d')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('target_topic', '/piper/target_3d')
        self.declare_parameter('depth_min_m', 0.25)
        self.declare_parameter('depth_max_m', 1.0)
        self.declare_parameter('use_median_depth', True)
        self.declare_parameter('min_valid_depth_ratio', 0.4)
        self.declare_parameter('crop_half_size_px', 8)
        self.declare_parameter('max_detection_age_sec', 0.5)

        self.bridge = CvBridge()
        self.latest_detection = None
        self.camera_info = None
        self.depth_min = float(self.get_parameter('depth_min_m').value)
        self.depth_max = float(self.get_parameter('depth_max_m').value)
        self.use_median_depth = bool(self.get_parameter('use_median_depth').value)
        self.min_ratio = float(self.get_parameter('min_valid_depth_ratio').value)
        self.crop_half = int(self.get_parameter('crop_half_size_px').value)
        self.max_detection_age_sec = float(self.get_parameter('max_detection_age_sec').value)

        self.pub = self.create_publisher(Target3D, self.get_parameter('target_topic').value, 10)
        self.det_sub = self.create_subscription(
            Detection2D, self.get_parameter('detection_topic').value, self.detection_cb, 10
        )
        self.info_sub = self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value, self.info_cb, qos_profile_sensor_data
        )
        self.depth_sub = self.create_subscription(
            Image, self.get_parameter('depth_topic').value, self.depth_cb, qos_profile_sensor_data
        )
        self.get_logger().info('Depth projection waiting for Detection2D, depth image, and CameraInfo')

    def detection_cb(self, msg):
        self.latest_detection = msg

    def info_cb(self, msg):
        self.camera_info = msg

    def depth_cb(self, depth_msg):
        out = Target3D()
        out.header = depth_msg.header
        if self.latest_detection is None or self.camera_info is None:
            out.valid = False
            self.pub.publish(out)
            return
        if not self.latest_detection.valid:
            out.valid = False
            self.pub.publish(out)
            return
        detection_age = self.stamp_age_sec(depth_msg.header.stamp, self.latest_detection.header.stamp)
        if detection_age is None or abs(detection_age) > self.max_detection_age_sec:
            out.valid = False
            self.pub.publish(out)
            self.get_logger().warn('Detection rejected age_sec=%s' % self.format_age(detection_age))
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as exc:
            out.valid = False
            self.pub.publish(out)
            self.get_logger().warn('depth cv_bridge failed: %s' % exc)
            return

        depth_m = self.depth_image_to_meters(depth, depth_msg.encoding)
        h, w = depth_m.shape[:2]
        u = int(round(self.latest_detection.u))
        v = int(round(self.latest_detection.v))
        if u < 0 or u >= w or v < 0 or v >= h:
            out.valid = False
            self.pub.publish(out)
            self.get_logger().warn('Detection rejected outside depth image u=%d v=%d size=%dx%d' % (u, v, w, h))
            return

        x0 = max(0, u - self.crop_half)
        x1 = min(w, u + self.crop_half + 1)
        y0 = max(0, v - self.crop_half)
        y1 = min(h, v + self.crop_half + 1)
        crop = depth_m[y0:y1, x0:x1]
        valid = np.isfinite(crop) & (crop > self.depth_min) & (crop < self.depth_max)
        valid_count = int(np.count_nonzero(valid))
        total_count = int(crop.size)
        ratio = float(valid_count) / float(max(total_count, 1))
        out.valid_depth_ratio = ratio

        if valid_count == 0 or ratio < self.min_ratio:
            out.valid = False
            self.pub.publish(out)
            self.get_logger().warn('Depth rejected valid_ratio=%.2f' % ratio)
            return

        if self.use_median_depth:
            z = float(np.median(crop[valid]))
        else:
            z = float(crop[valid][crop[valid].size // 2])
        fx = float(self.camera_info.k[0])
        fy = float(self.camera_info.k[4])
        cx = float(self.camera_info.k[2])
        cy = float(self.camera_info.k[5])
        if fx == 0.0 or fy == 0.0 or math.isnan(z):
            out.valid = False
            self.pub.publish(out)
            return

        out.point.x = (self.latest_detection.u - cx) * z / fx
        out.point.y = (self.latest_detection.v - cy) * z / fy
        out.point.z = z
        out.depth = z
        out.valid = True
        self.pub.publish(out)
        self.get_logger().info(
            'Target3D camera_frame=(%.3f, %.3f, %.3f) ratio=%.2f'
            % (out.point.x, out.point.y, out.point.z, ratio)
        )

    @staticmethod
    def depth_image_to_meters(depth, encoding):
        arr = np.asarray(depth)
        if encoding in ('16UC1', 'mono16'):
            return arr.astype(np.float32) * 0.001
        return arr.astype(np.float32)

    @staticmethod
    def stamp_age_sec(newer, older):
        if newer.sec == 0 and newer.nanosec == 0:
            return None
        if older.sec == 0 and older.nanosec == 0:
            return None
        return float(newer.sec - older.sec) + float(newer.nanosec - older.nanosec) * 1e-9

    @staticmethod
    def format_age(age):
        if age is None:
            return 'unknown'
        return '%.3f' % age


def main(args=None):
    rclpy.init(args=args)
    node = DepthTo3DNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
