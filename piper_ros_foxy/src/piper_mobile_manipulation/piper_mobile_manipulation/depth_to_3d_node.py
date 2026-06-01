#!/usr/bin/env python3
import math

import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
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
        self.declare_parameter('use_detection_bbox', True)
        self.declare_parameter('bbox_scale', 0.8)
        self.declare_parameter('depth_percentile', 50.0)
        self.declare_parameter('max_depth_stddev_m', 0.08)
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop_sec', 0.08)

        self.bridge = CvBridge()
        self.refresh_runtime_params()
        self.sync_queue_size = int(self.get_parameter('sync_queue_size').value)
        self.sync_slop_sec = float(self.get_parameter('sync_slop_sec').value)

        self.pub = self.create_publisher(Target3D, self.get_parameter('target_topic').value, 10)
        self.det_sub = Subscriber(
            self, Detection2D, self.get_parameter('detection_topic').value, qos_profile=10
        )
        self.depth_sub = Subscriber(
            self, Image, self.get_parameter('depth_topic').value, qos_profile=qos_profile_sensor_data
        )
        self.info_sub = Subscriber(
            self, CameraInfo, self.get_parameter('camera_info_topic').value, qos_profile=qos_profile_sensor_data
        )
        self.sync = ApproximateTimeSynchronizer(
            [self.det_sub, self.depth_sub, self.info_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop_sec,
        )
        self.sync.registerCallback(self.synced_cb)
        self.get_logger().info(
            'Depth projection synchronizing Detection2D, depth image, and CameraInfo slop=%.3fs'
            % self.sync_slop_sec
        )

    def synced_cb(self, detection_msg, depth_msg, camera_info):
        self.refresh_runtime_params()
        out = Target3D()
        out.header = depth_msg.header
        if not detection_msg.valid:
            out.valid = False
            self.pub.publish(out)
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
        u = int(round(detection_msg.u))
        v = int(round(detection_msg.v))
        if u < 0 or u >= w or v < 0 or v >= h:
            out.valid = False
            self.pub.publish(out)
            self.get_logger().warn('Detection rejected outside depth image u=%d v=%d size=%dx%d' % (u, v, w, h))
            return

        x0, x1, y0, y1 = self.depth_roi(detection_msg, u, v, w, h)
        out.roi_width = float(x1 - x0)
        out.roi_height = float(y1 - y0)
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

        valid_depths = crop[valid]
        depth_stddev = float(np.std(valid_depths))
        out.depth_stddev = depth_stddev
        if self.max_depth_stddev > 0.0 and depth_stddev > self.max_depth_stddev:
            out.valid = False
            self.pub.publish(out)
            self.get_logger().warn(
                'Depth rejected stddev=%.3f roi=%dx%d ratio=%.2f'
                % (depth_stddev, x1 - x0, y1 - y0, ratio)
            )
            return

        if self.use_median_depth:
            z = float(np.median(valid_depths))
        else:
            percentile = float(np.clip(self.depth_percentile, 0.0, 100.0))
            z = float(np.percentile(valid_depths, percentile))
        fx = float(camera_info.k[0])
        fy = float(camera_info.k[4])
        cx = float(camera_info.k[2])
        cy = float(camera_info.k[5])
        if fx == 0.0 or fy == 0.0 or math.isnan(z):
            out.valid = False
            self.pub.publish(out)
            return

        out.point.x = (detection_msg.u - cx) * z / fx
        out.point.y = (detection_msg.v - cy) * z / fy
        out.point.z = z
        out.depth = z
        out.measurement_confidence = self.measurement_confidence(
            detection_msg.confidence, ratio, depth_stddev
        )
        out.valid = True
        self.pub.publish(out)
        self.get_logger().info(
            'Target3D camera_frame=(%.3f, %.3f, %.3f) ratio=%.2f roi=%dx%d std=%.3f conf=%.2f'
            % (
                out.point.x,
                out.point.y,
                out.point.z,
                ratio,
                x1 - x0,
                y1 - y0,
                depth_stddev,
                out.measurement_confidence,
            )
        )

    def depth_roi(self, detection_msg, u, v, image_width, image_height):
        if not self.use_detection_bbox:
            return (
                max(0, u - self.crop_half),
                min(image_width, u + self.crop_half + 1),
                max(0, v - self.crop_half),
                min(image_height, v + self.crop_half + 1),
            )

        det_width = max(float(detection_msg.width), 1.0)
        det_height = max(float(detection_msg.height), 1.0)
        scale = float(np.clip(self.bbox_scale, 0.05, 1.0))
        half_w = max(1, int(round(det_width * scale * 0.5)))
        half_h = max(1, int(round(det_height * scale * 0.5)))
        return (
            max(0, u - half_w),
            min(image_width, u + half_w + 1),
            max(0, v - half_h),
            min(image_height, v + half_h + 1),
        )

    @staticmethod
    def depth_image_to_meters(depth, encoding):
        arr = np.asarray(depth)
        if encoding in ('16UC1', 'mono16'):
            return arr.astype(np.float32) * 0.001
        return arr.astype(np.float32)

    def measurement_confidence(self, detection_confidence, valid_ratio, depth_stddev):
        std_quality = 1.0
        if self.max_depth_stddev > 0.0:
            std_quality = 1.0 - float(np.clip(depth_stddev / self.max_depth_stddev, 0.0, 1.0))
        return float(np.clip(detection_confidence, 0.0, 1.0) * np.clip(valid_ratio, 0.0, 1.0) * std_quality)

    def refresh_runtime_params(self):
        self.depth_min = float(self.get_parameter('depth_min_m').value)
        self.depth_max = float(self.get_parameter('depth_max_m').value)
        self.use_median_depth = bool(self.get_parameter('use_median_depth').value)
        self.min_ratio = float(self.get_parameter('min_valid_depth_ratio').value)
        self.crop_half = int(self.get_parameter('crop_half_size_px').value)
        self.use_detection_bbox = bool(self.get_parameter('use_detection_bbox').value)
        self.bbox_scale = float(self.get_parameter('bbox_scale').value)
        self.depth_percentile = float(self.get_parameter('depth_percentile').value)
        self.max_depth_stddev = float(self.get_parameter('max_depth_stddev_m').value)


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
