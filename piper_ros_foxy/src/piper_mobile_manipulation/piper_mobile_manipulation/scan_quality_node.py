#!/usr/bin/env python3
import json
import math
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from piper_mobile_manipulation.msg import Detection2D, ScanViewpointArray, Target3D


class ScanQualityNode(Node):
    def __init__(self):
        super().__init__('scan_quality_node')
        self.declare_parameter('color_image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_image_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('mask_topic', '/piper/sam2_target_mask')
        self.declare_parameter('detection_topic', '/piper/sam2_detection_2d')
        self.declare_parameter('target_3d_topic', '/piper/target_3d')
        self.declare_parameter('reachable_scan_viewpoints_topic', '/piper/reachable_scan_viewpoints')
        self.declare_parameter('scan_capture_status_topic', '/piper/scan_capture_status')
        self.declare_parameter('scan_quality_topic', '/piper/scan_quality')
        self.declare_parameter('scan_quality_debug_topic', '/piper/scan_quality_debug')
        self.declare_parameter('useful_scan_coverage_topic', '/piper/useful_scan_coverage')

        self.declare_parameter('evaluation_interval_sec', 1.0)
        self.declare_parameter('stale_timeout_sec', 1.0)
        self.declare_parameter('min_good_scan_quality', 0.65)
        self.declare_parameter('min_acceptable_scan_quality', 0.40)
        self.declare_parameter('max_depth_stddev_good_m', 0.03)
        self.declare_parameter('min_valid_depth_ratio', 0.40)
        self.declare_parameter('min_mask_area_px', 100)
        self.declare_parameter('edge_margin_px', 40)
        self.declare_parameter('min_valid_depth_m', 0.15)
        self.declare_parameter('max_valid_depth_m', 1.20)
        self.declare_parameter('dry_run', True)
        self.declare_parameter('enable_real_arm_motion', False)
        self.declare_parameter('debug', True)

        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None
        self.latest_mask = None
        self.latest_detection = None
        self.latest_target = None
        self.latest_reachable_scan_viewpoints = None
        self.latest_scan_capture_status = None
        self.latest_color_stamp = 0.0
        self.latest_depth_stamp = 0.0
        self.latest_mask_stamp = 0.0
        self.latest_detection_stamp = 0.0
        self.latest_target_stamp = 0.0
        self.latest_reachable_stamp = 0.0
        self.latest_capture_status_stamp = 0.0
        self.last_eval_stamp = 0.0
        self.frame_index = 0
        self.counts = {'GOOD': 0, 'ACCEPTABLE': 0, 'POOR': 0, 'INVALID': 0}
        self.useful_frame_count = 0

        self.quality_pub = self.create_publisher(
            String, self.get_parameter('scan_quality_topic').value, 10
        )
        self.debug_pub = self.create_publisher(
            String, self.get_parameter('scan_quality_debug_topic').value, 10
        )
        self.coverage_pub = self.create_publisher(
            String, self.get_parameter('useful_scan_coverage_topic').value, 10
        )

        self.create_subscription(
            Image,
            self.get_parameter('color_image_topic').value,
            self.color_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            self.get_parameter('depth_image_topic').value,
            self.depth_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            self.get_parameter('mask_topic').value,
            self.mask_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Detection2D,
            self.get_parameter('detection_topic').value,
            self.detection_cb,
            10,
        )
        self.create_subscription(
            Target3D,
            self.get_parameter('target_3d_topic').value,
            self.target_cb,
            10,
        )
        self.create_subscription(
            ScanViewpointArray,
            self.get_parameter('reachable_scan_viewpoints_topic').value,
            self.reachable_scan_viewpoints_cb,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter('scan_capture_status_topic').value,
            self.scan_capture_status_cb,
            10,
        )

        self.timer = self.create_timer(0.20, self.timer_cb)
        self.get_logger().warn(
            'Scan quality is dry-run only; it scores RGB-D views and never publishes /piper/servo_cmd.'
        )

    def color_cb(self, msg):
        self.latest_color = msg
        self.latest_color_stamp = time.monotonic()

    def depth_cb(self, msg):
        self.latest_depth = msg
        self.latest_depth_stamp = time.monotonic()

    def mask_cb(self, msg):
        self.latest_mask = msg
        self.latest_mask_stamp = time.monotonic()

    def detection_cb(self, msg):
        self.latest_detection = msg
        self.latest_detection_stamp = time.monotonic()

    def target_cb(self, msg):
        self.latest_target = msg
        self.latest_target_stamp = time.monotonic()

    def reachable_scan_viewpoints_cb(self, msg):
        self.latest_reachable_scan_viewpoints = msg
        self.latest_reachable_stamp = time.monotonic()

    def scan_capture_status_cb(self, msg):
        self.latest_scan_capture_status = self.parse_json_msg(msg)
        self.latest_capture_status_stamp = time.monotonic()

    def timer_cb(self):
        now = time.monotonic()
        interval = max(0.1, float(self.get_parameter('evaluation_interval_sec').value))
        if now - self.last_eval_stamp < interval:
            return
        self.last_eval_stamp = now

        payload = self.evaluate_current_view(now)
        self.publish_quality(payload)
        self.publish_debug(payload)
        self.publish_useful_coverage()

    def evaluate_current_view(self, now):
        ready, reason = self.ready(now)
        if not ready:
            return self.invalid_payload(reason)

        try:
            depth = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding='passthrough')
            mask = self.bridge.imgmsg_to_cv2(self.latest_mask, desired_encoding='mono8')
        except Exception as exc:
            return self.invalid_payload('image conversion failed: %s' % exc)

        return self.evaluate_arrays(np.asarray(depth), np.asarray(mask))

    def ready(self, now):
        if not self.param_bool('dry_run'):
            return False, 'dry_run is false'
        if self.param_bool('enable_real_arm_motion'):
            return False, 'enable_real_arm_motion is true'
        if self.latest_color is None:
            return False, 'missing color image'
        if self.latest_depth is None:
            return False, 'missing depth image'
        if self.latest_mask is None:
            return False, 'missing detection mask'
        if self.latest_detection is None:
            return False, 'missing detection_2d'
        if self.latest_target is None:
            return False, 'missing target_3d'

        timeout = max(0.1, float(self.get_parameter('stale_timeout_sec').value))
        stale_checks = [
            ('color image', self.latest_color_stamp),
            ('depth image', self.latest_depth_stamp),
            ('detection mask', self.latest_mask_stamp),
            ('detection_2d', self.latest_detection_stamp),
            ('target_3d', self.latest_target_stamp),
        ]
        for name, stamp in stale_checks:
            if now - stamp > timeout:
                return False, 'stale %s' % name
        return True, ''

    def evaluate_arrays(self, depth, mask):
        payload = self.base_payload('INVALID')
        if depth.shape[:2] != mask.shape[:2]:
            return self.invalid_payload('depth and mask dimensions differ')

        mask_bool = mask > 0
        mask_area_px = int(np.count_nonzero(mask_bool))
        bbox = self.mask_bbox(mask_bool)
        depth_m = self.depth_to_meters(depth)
        valid_depth = self.valid_depth_values(depth_m, mask_bool)

        target_valid = self.target_valid()
        valid_depth_ratio = float(valid_depth.size / max(1, mask_area_px))
        depth_mean_m = float(np.mean(valid_depth)) if valid_depth.size > 0 else 0.0
        depth_stddev_m = float(np.std(valid_depth)) if valid_depth.size > 0 else 0.0
        centredness_score = self.centredness_score(mask_bool.shape, bbox)
        edge_margin_score = self.edge_margin_score(mask_bool.shape, bbox)
        quality_score = self.quality_score(
            mask_area_px,
            valid_depth_ratio,
            depth_stddev_m,
            centredness_score,
            edge_margin_score,
            target_valid,
        )
        quality_label = self.quality_label(
            quality_score,
            mask_area_px,
            valid_depth_ratio,
            target_valid,
        )

        self.counts[quality_label] += 1
        if quality_label in ('GOOD', 'ACCEPTABLE'):
            self.useful_frame_count += 1

        payload.update(
            {
                'quality_score': quality_score,
                'quality_label': quality_label,
                'status': quality_label,
                'score': quality_score,
                'mask_area_px': mask_area_px,
                'valid_depth_ratio': valid_depth_ratio,
                'depth_mean_m': depth_mean_m,
                'depth_stddev_m': depth_stddev_m,
                'centredness_score': centredness_score,
                'edge_margin_score': edge_margin_score,
                'target_valid': bool(target_valid),
                'bbox': self.bbox_payload(bbox),
                'detection_valid': bool(self.latest_detection.valid),
                'detection_confidence': float(self.latest_detection.confidence),
                'frame_index': int(self.frame_index),
                'reason': self.reason_for_label(quality_label, mask_area_px, valid_depth_ratio, target_valid),
            }
        )
        self.frame_index += 1
        return payload

    def invalid_payload(self, reason):
        payload = self.base_payload('INVALID')
        payload['reason'] = reason
        self.counts['INVALID'] += 1
        self.frame_index += 1
        return payload

    def base_payload(self, label):
        return {
            'stamp': self.message_stamp(),
            'quality_score': 0.0,
            'quality_label': label,
            'status': label,
            'score': 0.0,
            'mask_area_px': 0,
            'valid_depth_ratio': 0.0,
            'depth_mean_m': 0.0,
            'depth_stddev_m': 0.0,
            'centredness_score': 0.0,
            'edge_margin_score': 0.0,
            'target_valid': False,
            'frame_index': int(self.frame_index),
            'dry_run': True,
            'real_arm_motion': False,
        }

    def quality_score(
        self,
        mask_area_px,
        valid_depth_ratio,
        depth_stddev_m,
        centredness_score,
        edge_margin_score,
        target_valid,
    ):
        if not target_valid:
            return 0.0

        min_mask = max(1.0, float(self.get_parameter('min_mask_area_px').value))
        min_depth_ratio = max(1e-6, float(self.get_parameter('min_valid_depth_ratio').value))
        max_stddev_good = max(1e-6, float(self.get_parameter('max_depth_stddev_good_m').value))
        mask_score = self.clamp(float(mask_area_px) / min_mask)
        depth_ratio_score = self.clamp(float(valid_depth_ratio) / min_depth_ratio)
        depth_noise_score = self.clamp(1.0 - (float(depth_stddev_m) / (max_stddev_good * 2.0)))
        score = (
            0.25 * mask_score
            + 0.30 * depth_ratio_score
            + 0.20 * depth_noise_score
            + 0.15 * edge_margin_score
            + 0.10 * centredness_score
        )
        return self.clamp(score)

    def quality_label(self, quality_score, mask_area_px, valid_depth_ratio, target_valid):
        if not target_valid:
            return 'INVALID'
        if mask_area_px < int(self.get_parameter('min_mask_area_px').value):
            return 'INVALID'
        if valid_depth_ratio < float(self.get_parameter('min_valid_depth_ratio').value):
            return 'POOR'
        if quality_score >= float(self.get_parameter('min_good_scan_quality').value):
            return 'GOOD'
        if quality_score >= float(self.get_parameter('min_acceptable_scan_quality').value):
            return 'ACCEPTABLE'
        return 'POOR'

    def target_valid(self):
        if self.latest_target is None:
            return False
        if hasattr(self.latest_target, 'valid'):
            return bool(self.latest_target.valid)
        point = self.latest_target.point
        values = [float(point.x), float(point.y), float(point.z), float(self.latest_target.depth)]
        return all(math.isfinite(value) for value in values) and any(abs(value) > 1e-6 for value in values)

    def valid_depth_values(self, depth_m, mask_bool):
        masked = depth_m[mask_bool]
        return masked[
            np.isfinite(masked)
            & (masked >= float(self.get_parameter('min_valid_depth_m').value))
            & (masked <= float(self.get_parameter('max_valid_depth_m').value))
        ]

    def centredness_score(self, shape, bbox):
        if bbox is None:
            return 0.0
        height, width = shape
        x0, y0, x1, y1 = bbox
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        dx = abs(cx - (width - 1) * 0.5) / max(1.0, width * 0.5)
        dy = abs(cy - (height - 1) * 0.5) / max(1.0, height * 0.5)
        return self.clamp(1.0 - math.sqrt(dx * dx + dy * dy))

    def edge_margin_score(self, shape, bbox):
        if bbox is None:
            return 0.0
        height, width = shape
        x0, y0, x1, y1 = bbox
        margin = min(x0, y0, width - 1 - x1, height - 1 - y1)
        required = max(1.0, float(self.get_parameter('edge_margin_px').value))
        return self.clamp(float(margin) / required)

    def publish_quality(self, payload):
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self.quality_pub.publish(msg)
        if self.param_bool('debug'):
            self.get_logger().info(
                'scan quality label=%s score=%.2f mask=%d depth=%.2f std=%.3f'
                % (
                    payload['quality_label'],
                    payload['quality_score'],
                    payload['mask_area_px'],
                    payload['valid_depth_ratio'],
                    payload['depth_stddev_m'],
                )
            )

    def publish_debug(self, payload):
        msg = String()
        msg.data = (
            'quality=%s score=%.2f mask_area_px=%d valid_depth_ratio=%.2f '
            'depth_mean_m=%.3f depth_stddev_m=%.3f centredness=%.2f edge_margin=%.2f '
            'target_valid=%s reason=%s'
            % (
                payload['quality_label'],
                payload['quality_score'],
                payload['mask_area_px'],
                payload['valid_depth_ratio'],
                payload['depth_mean_m'],
                payload['depth_stddev_m'],
                payload['centredness_score'],
                payload['edge_margin_score'],
                payload['target_valid'],
                payload.get('reason', ''),
            )
        )
        self.debug_pub.publish(msg)

    def publish_useful_coverage(self):
        reachable_angles = self.reachable_angles()
        reachable_coverage = self.coverage_from_angles(reachable_angles)
        useful_coverage = None
        useful_coverage_note = 'unavailable: no exact viewpoint-to-frame mapping in dry-run capture'
        if reachable_coverage is not None and len(reachable_angles) > 0:
            useful_ratio = min(1.0, float(self.useful_frame_count) / max(1.0, float(len(reachable_angles))))
            useful_coverage = float(reachable_coverage * useful_ratio)
            useful_coverage_note = 'approximate: scaled reachable coverage by useful frame count'

        payload = {
            'good_frame_count': int(self.counts['GOOD']),
            'acceptable_frame_count': int(self.counts['ACCEPTABLE']),
            'poor_frame_count': int(self.counts['POOR']),
            'invalid_frame_count': int(self.counts['INVALID']),
            'useful_frame_count': int(self.useful_frame_count),
            'reachable_viewpoint_count': int(len(reachable_angles)),
            'reachable_coverage_deg': reachable_coverage,
            'useful_coverage_deg': useful_coverage,
            'useful_coverage_note': useful_coverage_note,
            'dry_run': True,
            'real_arm_motion': False,
        }
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self.coverage_pub.publish(msg)

    def reachable_angles(self):
        payload = self.latest_reachable_scan_viewpoints
        if not isinstance(payload, ScanViewpointArray):
            return []
        return [float(viewpoint.view_angle_deg) for viewpoint in payload.viewpoints
                if viewpoint.reachable]

    def message_stamp(self):
        msg = self.latest_depth or self.latest_color or self.latest_mask
        if msg is None:
            return {'available': False}
        return {
            'available': True,
            'sec': int(msg.header.stamp.sec),
            'nanosec': int(msg.header.stamp.nanosec),
            'frame_id': str(msg.header.frame_id),
        }

    @staticmethod
    def reason_for_label(label, mask_area_px, valid_depth_ratio, target_valid):
        if not target_valid:
            return 'target invalid'
        if label == 'INVALID':
            return 'mask too small: %d px' % mask_area_px
        if label == 'POOR':
            return 'low score or valid depth ratio %.2f' % valid_depth_ratio
        return ''

    @staticmethod
    def parse_json_msg(msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return {'parse_error': True, 'raw': msg.data}
        return payload if isinstance(payload, dict) else {'payload': payload}

    @staticmethod
    def mask_bbox(mask_bool):
        ys, xs = np.nonzero(mask_bool)
        if xs.size == 0 or ys.size == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @staticmethod
    def bbox_payload(bbox):
        if bbox is None:
            return {'available': False}
        x0, y0, x1, y1 = bbox
        return {'available': True, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}

    @staticmethod
    def depth_to_meters(depth):
        if np.issubdtype(depth.dtype, np.integer):
            return depth.astype(np.float32, copy=False) * 0.001
        depth = depth.astype(np.float32, copy=False)
        finite = depth[np.isfinite(depth) & (depth > 0.0)]
        if finite.size > 0 and float(np.nanmedian(finite)) > 20.0:
            return depth * 0.001
        return depth

    @staticmethod
    def coverage_from_angles(angles):
        if len(angles) < 2:
            return None
        return float(max(angles) - min(angles))

    @staticmethod
    def clamp(value):
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def is_finite_number(value):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def param_bool(self, name):
        value = self.get_parameter(name).value
        if isinstance(value, str):
            return value.lower() in ('1', 'true', 'yes', 'on')
        return bool(value)


def main(args=None):
    rclpy.init(args=args)
    node = ScanQualityNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
