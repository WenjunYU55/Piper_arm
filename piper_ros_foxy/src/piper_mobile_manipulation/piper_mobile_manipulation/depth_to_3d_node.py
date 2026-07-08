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
        self.declare_parameter('detection_topic', '/piper/sam2_detection_2d')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('mask_topic', '/piper/sam2_target_mask')
        self.declare_parameter('target_topic', '/piper/target_3d')
        self.declare_parameter('depth_min_m', 0.25)
        self.declare_parameter('depth_max_m', 1.0)
        self.declare_parameter('min_depth_m', 0.25)
        self.declare_parameter('max_depth_m', 1.20)
        self.declare_parameter('use_median_depth', True)
        self.declare_parameter('min_valid_depth_ratio', 0.4)
        self.declare_parameter('min_valid_depth_pixels', 20)
        self.declare_parameter('crop_half_size_px', 8)
        self.declare_parameter('roi_half_size_px', 10)
        self.declare_parameter('use_detection_bbox', True)
        self.declare_parameter('bbox_scale', 0.8)
        self.declare_parameter('depth_percentile', 50.0)
        self.declare_parameter('max_depth_stddev_m', 0.08)
        self.declare_parameter('confidence_depth_stddev_m', 0.15)
        self.declare_parameter('max_depth_jump_m', 0.20)
        self.declare_parameter('smoothing_alpha', 0.2)
        self.declare_parameter('use_mask_depth', True)
        self.declare_parameter('mask_max_age_s', 0.20)
        self.declare_parameter('mask_erode_px', 2)
        self.declare_parameter('debug', True)
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop_sec', 0.08)

        self.bridge = CvBridge()
        self.latest_mask_msg = None
        self.previous_depth = None
        self.previous_point = None
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
        self.mask_sub = self.create_subscription(
            Image,
            self.get_parameter('mask_topic').value,
            self.mask_cb,
            qos_profile_sensor_data,
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

    def mask_cb(self, mask_msg):
        self.latest_mask_msg = mask_msg

    def synced_cb(self, detection_msg, depth_msg, camera_info):
        self.refresh_runtime_params()
        out = Target3D()
        out.header = depth_msg.header
        out.source_u = float(detection_msg.u)
        out.source_v = float(detection_msg.v)
        out.detection_width = float(detection_msg.width)
        out.detection_height = float(detection_msg.height)
        if not detection_msg.valid:
            self.previous_depth = None
            self.previous_point = None
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

        mask = self.mask_for_depth(depth_msg, w, h)
        if mask is not None:
            x0, x1, y0, y1 = self.bbox_bounds(detection_msg, u, v, w, h)
            roi_mask = mask[y0:y1, x0:x1] > 0
            crop = depth_m[y0:y1, x0:x1]
            valid = roi_mask & np.isfinite(crop) & (crop > self.depth_min) & (crop < self.depth_max)
            out.depth_source = 'mask'
        else:
            x0, x1, y0, y1 = self.depth_roi(detection_msg, u, v, w, h)
            crop = depth_m[y0:y1, x0:x1]
            valid = np.isfinite(crop) & (crop > self.depth_min) & (crop < self.depth_max)
            out.depth_source = 'roi'

        out.roi_width = float(x1 - x0)
        out.roi_height = float(y1 - y0)
        valid_count = int(np.count_nonzero(valid))
        total_count = int(crop.size)
        ratio = float(valid_count) / float(max(total_count, 1))
        out.valid_depth_ratio = ratio

        if valid_count < self.min_valid_depth_pixels or ratio < self.min_ratio:
            out.valid = False
            self.pub.publish(out)
            self.log_debug(
                'Depth rejected source=%s valid_pixels=%d ratio=%.2f roi=%dx%d'
                % (out.depth_source, valid_count, ratio, x1 - x0, y1 - y0),
                warn=True,
            )
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
        if self.previous_depth is not None and abs(z - self.previous_depth) > self.max_depth_jump:
            out.valid = False
            self.pub.publish(out)
            self.log_debug(
                'Depth rejected jump %.3fm -> %.3fm max=%.3fm'
                % (self.previous_depth, z, self.max_depth_jump),
                warn=True,
            )
            return
        fx = float(camera_info.k[0])
        fy = float(camera_info.k[4])
        cx = float(camera_info.k[2])
        cy = float(camera_info.k[5])
        if fx == 0.0 or fy == 0.0 or math.isnan(z):
            out.valid = False
            self.pub.publish(out)
            return

        # Estimate the visible-surface centroid from every valid masked depth
        # sample instead of projecting only the 2D bounding-box centre.
        valid_v, valid_u = np.nonzero(valid)
        full_u = valid_u.astype(np.float64) + float(x0)
        full_v = valid_v.astype(np.float64) + float(y0)
        sample_z = crop[valid].astype(np.float64)
        new_point = np.array([
            float(np.median((full_u - cx) * sample_z / fx)),
            float(np.median((full_v - cy) * sample_z / fy)),
            z,
        ], dtype=np.float64)
        if self.previous_point is not None:
            alpha = float(np.clip(self.smoothing_alpha, 0.0, 1.0))
            filtered_point = alpha * new_point + (1.0 - alpha) * self.previous_point
        else:
            filtered_point = new_point
        self.previous_point = filtered_point
        self.previous_depth = z

        out.point.x = float(filtered_point[0])
        out.point.y = float(filtered_point[1])
        out.point.z = float(filtered_point[2])
        out.depth = float(filtered_point[2])
        out.measurement_confidence = self.measurement_confidence(detection_msg.confidence, valid_count, depth_stddev)
        out.valid = True
        self.pub.publish(out)
        self.log_debug(
            'Target3D source=%s camera_frame=(%.3f, %.3f, %.3f) raw_z=%.3f ratio=%.2f valid=%d roi=%dx%d std=%.3f det_conf=%.2f depth_conf=%.2f conf=%.2f'
            % (
                out.depth_source,
                out.point.x,
                out.point.y,
                out.point.z,
                z,
                ratio,
                valid_count,
                x1 - x0,
                y1 - y0,
                depth_stddev,
                detection_msg.confidence,
                self.depth_confidence(valid_count, depth_stddev),
                out.measurement_confidence,
            )
        )

    def bbox_bounds(self, detection_msg, u, v, image_width, image_height):
        det_width = max(float(detection_msg.width), 1.0)
        det_height = max(float(detection_msg.height), 1.0)
        half_w = max(1, int(round(det_width * 0.5)))
        half_h = max(1, int(round(det_height * 0.5)))
        return (
            max(0, u - half_w),
            min(image_width, u + half_w + 1),
            max(0, v - half_h),
            min(image_height, v + half_h + 1),
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

    def mask_for_depth(self, depth_msg, image_width, image_height):
        if not self.use_mask_depth or self.latest_mask_msg is None:
            return None
        age = abs(
            (
                self.stamp_to_seconds(depth_msg.header.stamp)
                - self.stamp_to_seconds(self.latest_mask_msg.header.stamp)
            )
        )
        if age > self.mask_max_age_s:
            return None
        try:
            mask = self.bridge.imgmsg_to_cv2(self.latest_mask_msg, desired_encoding='mono8')
        except Exception as exc:
            self.log_debug('mask cv_bridge failed: %s' % exc, warn=True)
            return None
        if mask.shape[0] != image_height or mask.shape[1] != image_width:
            return None
        if self.mask_erode_px > 0:
            import cv2

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * self.mask_erode_px + 1, 2 * self.mask_erode_px + 1)
            )
            mask = cv2.erode(mask, kernel)
        return mask

    @staticmethod
    def stamp_to_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def depth_image_to_meters(depth, encoding):
        arr = np.asarray(depth)
        if encoding in ('16UC1', 'mono16'):
            return arr.astype(np.float32) * 0.001
        return arr.astype(np.float32)

    def measurement_confidence(self, detection_confidence, valid_count, depth_stddev):
        return float(np.clip(detection_confidence, 0.0, 1.0) * self.depth_confidence(valid_count, depth_stddev))

    def depth_confidence(self, valid_count, depth_stddev):
        pixel_quality = float(np.clip(valid_count / float(max(self.min_valid_depth_pixels * 2, 1)), 0.0, 1.0))
        std_quality = 1.0
        if self.confidence_depth_stddev > 0.0:
            std_ratio = max(float(depth_stddev), 0.0) / self.confidence_depth_stddev
            std_quality = 1.0 / (1.0 + std_ratio * std_ratio)
        return float(np.clip(0.35 + 0.65 * pixel_quality * std_quality, 0.0, 1.0))

    def refresh_runtime_params(self):
        self.depth_min = float(self.get_parameter('min_depth_m').value)
        self.depth_max = float(self.get_parameter('max_depth_m').value)
        self.use_median_depth = bool(self.get_parameter('use_median_depth').value)
        self.min_ratio = float(self.get_parameter('min_valid_depth_ratio').value)
        self.min_valid_depth_pixels = int(self.get_parameter('min_valid_depth_pixels').value)
        self.crop_half = int(self.get_parameter('roi_half_size_px').value)
        self.use_detection_bbox = bool(self.get_parameter('use_detection_bbox').value)
        self.bbox_scale = float(self.get_parameter('bbox_scale').value)
        self.depth_percentile = float(self.get_parameter('depth_percentile').value)
        self.max_depth_stddev = float(self.get_parameter('max_depth_stddev_m').value)
        self.confidence_depth_stddev = float(self.get_parameter('confidence_depth_stddev_m').value)
        self.max_depth_jump = float(self.get_parameter('max_depth_jump_m').value)
        self.smoothing_alpha = float(self.get_parameter('smoothing_alpha').value)
        self.use_mask_depth = bool(self.get_parameter('use_mask_depth').value)
        self.mask_max_age_s = float(self.get_parameter('mask_max_age_s').value)
        self.mask_erode_px = max(0, int(self.get_parameter('mask_erode_px').value))
        self.debug = bool(self.get_parameter('debug').value)

    def log_debug(self, message, warn=False):
        if warn:
            self.get_logger().warn(message)
        elif self.debug:
            self.get_logger().info(message)


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
