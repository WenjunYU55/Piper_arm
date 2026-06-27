#!/usr/bin/env python3
import json
import math
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from piper_mobile_manipulation.msg import Detection2D, Target3D


class OcclusionCheckerNode(Node):
    def __init__(self):
        super().__init__('occlusion_checker_node')
        self.declare_parameter('color_image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_image_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('mask_topic', '/piper/sam2_target_mask')
        self.declare_parameter('detection_topic', '/piper/sam2_detection_2d')
        self.declare_parameter('target_3d_topic', '/piper/target_3d')
        self.declare_parameter('scan_quality_topic', '/piper/scan_quality')
        self.declare_parameter('occlusion_status_topic', '/piper/occlusion_status')
        self.declare_parameter('occlusion_debug_topic', '/piper/occlusion_debug')

        self.declare_parameter('evaluation_interval_sec', 1.0)
        self.declare_parameter('stale_timeout_sec', 1.0)
        self.declare_parameter('occlusion_depth_margin_m', 0.03)
        self.declare_parameter('occlusion_persistence_frames', 3)
        self.declare_parameter('min_occluder_area_px', 80)
        self.declare_parameter('near_mask_dilation_px', 20)
        self.declare_parameter('min_valid_depth_ratio', 0.40)
        self.declare_parameter('min_mask_area_px', 100)
        self.declare_parameter('edge_margin_px', 40)
        self.declare_parameter('min_valid_depth_m', 0.15)
        self.declare_parameter('max_valid_depth_m', 1.20)
        self.declare_parameter('partial_occlusion_ratio', 0.05)
        self.declare_parameter('heavy_occlusion_ratio', 0.20)
        self.declare_parameter('state_transition_confirmations', 2)
        self.declare_parameter('lost_transition_confirmations', 3)
        self.declare_parameter('use_reference_mask_area', True)
        self.declare_parameter('reference_update_alpha', 0.05)
        self.declare_parameter('min_reference_mask_area_px', 300)
        self.declare_parameter('partial_visible_ratio', 0.75)
        self.declare_parameter('heavy_visible_ratio', 0.35)
        self.declare_parameter('dry_run', True)
        self.declare_parameter('enable_real_arm_motion', False)
        self.declare_parameter('debug', True)

        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None
        self.latest_mask = None
        self.latest_detection = None
        self.latest_target = None
        self.latest_scan_quality = None
        self.latest_color_stamp = 0.0
        self.latest_depth_stamp = 0.0
        self.latest_mask_stamp = 0.0
        self.latest_detection_stamp = 0.0
        self.latest_target_stamp = 0.0
        self.latest_quality_stamp = 0.0
        self.last_eval_stamp = 0.0
        self.occlusion_history = []
        self.filtered_occlusion_state = None
        self.pending_occlusion_state = None
        self.pending_occlusion_count = 0
        self.reference_mask_area_px = 0.0

        self.status_pub = self.create_publisher(
            String, self.get_parameter('occlusion_status_topic').value, 10
        )
        self.debug_pub = self.create_publisher(
            String, self.get_parameter('occlusion_debug_topic').value, 10
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
            String,
            self.get_parameter('scan_quality_topic').value,
            self.scan_quality_cb,
            10,
        )

        self.timer = self.create_timer(0.20, self.timer_cb)
        self.get_logger().warn(
            'Occlusion checker is depth-only and dry-run; it never publishes /piper/servo_cmd.'
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

    def scan_quality_cb(self, msg):
        self.latest_scan_quality = self.parse_json_msg(msg)
        self.latest_quality_stamp = time.monotonic()

    def timer_cb(self):
        now = time.monotonic()
        interval = max(0.1, float(self.get_parameter('evaluation_interval_sec').value))
        if now - self.last_eval_stamp < interval:
            return
        self.last_eval_stamp = now

        payload = self.stabilize_payload(self.evaluate_current_view(now))
        self.publish_status(payload)
        self.publish_debug(payload)

    def evaluate_current_view(self, now):
        ready, state, reason = self.ready(now)
        if not ready:
            return self.base_payload(state, reason)

        try:
            depth = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding='passthrough')
            mask = self.bridge.imgmsg_to_cv2(self.latest_mask, desired_encoding='mono8')
        except Exception as exc:
            return self.base_payload('UNKNOWN', 'image conversion failed: %s' % exc)

        return self.evaluate_arrays(np.asarray(depth), np.asarray(mask))

    def ready(self, now):
        if not self.param_bool('dry_run'):
            return False, 'UNKNOWN', 'dry_run is false'
        if self.param_bool('enable_real_arm_motion'):
            return False, 'UNKNOWN', 'enable_real_arm_motion is true'

        missing = []
        if self.latest_color is None:
            missing.append('color image')
        if self.latest_depth is None:
            missing.append('depth image')
        if self.latest_mask is None:
            missing.append('detection mask')
        if self.latest_detection is None:
            missing.append('detection_2d')
        if self.latest_target is None:
            missing.append('target_3d')
        if self.latest_scan_quality is None:
            missing.append('scan quality')
        if missing:
            return False, 'UNKNOWN', 'waiting for %s' % ', '.join(missing)

        timeout = max(0.1, float(self.get_parameter('stale_timeout_sec').value))
        stale_checks = [
            ('color image', self.latest_color_stamp),
            ('depth image', self.latest_depth_stamp),
            ('detection mask', self.latest_mask_stamp),
            ('detection_2d', self.latest_detection_stamp),
            ('target_3d', self.latest_target_stamp),
            ('scan quality', self.latest_quality_stamp),
        ]
        for name, stamp in stale_checks:
            if now - stamp > timeout:
                return False, 'UNKNOWN', 'waiting for recent %s' % name
        return True, '', ''

    def evaluate_arrays(self, depth, mask):
        quality = self.scan_quality()
        target_valid = self.target_valid()
        quality_label = quality.get('quality_label', 'UNKNOWN')
        quality_valid_depth_ratio = float(quality.get('valid_depth_ratio', 0.0))
        mask_bool = mask > 0
        mask_area_px = int(np.count_nonzero(mask_bool))

        if not target_valid:
            state, reason = self.classify_by_visible_ratio(mask_area_px)
            payload = self.base_payload(
                state or 'LOST',
                reason or 'target invalid',
                quality=quality,
                mask_area_px=mask_area_px,
            )
            payload.update(self.reference_payload(mask_area_px))
            return payload
        if self.latest_detection is not None and not self.latest_detection.valid:
            state, reason = self.classify_by_visible_ratio(mask_area_px)
            payload = self.base_payload(
                state or 'LOST',
                reason or 'detection_2d invalid',
                quality=quality,
                mask_area_px=mask_area_px,
            )
            payload.update(self.reference_payload(mask_area_px))
            return payload
        if mask_area_px < int(self.get_parameter('min_mask_area_px').value):
            state, reason = self.classify_by_visible_ratio(mask_area_px)
            if state is not None:
                payload = self.base_payload(state, reason, quality=quality, mask_area_px=mask_area_px)
                payload.update(self.reference_payload(mask_area_px))
                return payload
            return self.base_payload('LOST', 'object mask missing or too small', quality=quality, mask_area_px=mask_area_px)
        if depth.shape[:2] != mask_bool.shape[:2]:
            return self.base_payload('UNKNOWN', 'depth and mask dimensions differ', quality=quality, mask_area_px=mask_area_px)

        depth_m = self.depth_to_meters(depth)
        target_depth = self.target_depth_m(depth_m, mask_bool)
        if target_depth is None:
            return self.base_payload('UNKNOWN', 'target depth unavailable', quality=quality, mask_area_px=mask_area_px)

        bbox = self.mask_bbox(mask_bool)
        roi_mask = self.roi_mask(mask_bool, bbox)
        valid_roi_depth = self.valid_depth_mask(depth_m) & roi_mask
        closer = valid_roi_depth & (depth_m < target_depth - float(self.get_parameter('occlusion_depth_margin_m').value))
        closer = self.remove_small_regions(closer, int(self.get_parameter('min_occluder_area_px').value))
        closer_area = int(np.count_nonzero(closer))
        roi_area = int(np.count_nonzero(roi_mask))
        closer_ratio = float(closer_area / max(1, roi_area))
        occlusion_score = self.occlusion_score(closer_ratio, closer_area, quality_label, quality_valid_depth_ratio)
        has_significant_closer_region = self.has_significant_closer_region(closer_area, closer_ratio)
        self.note_occlusion(has_significant_closer_region)
        persisted = self.occlusion_persisted()

        state, reason = self.classify(
            quality_label,
            quality_valid_depth_ratio,
            closer_area,
            closer_ratio,
            persisted,
            mask_area_px,
        )
        if state == 'CLEAR':
            self.update_reference_mask_area(mask_area_px)

        payload = self.base_payload(state, reason, quality=quality, mask_area_px=mask_area_px)
        payload.update(
            {
                'occlusion_score': occlusion_score,
                'closer_region_area_px': closer_area,
                'closer_region_ratio': closer_ratio,
                'target_depth_m': float(target_depth),
                'valid_depth_ratio': quality_valid_depth_ratio,
                'quality_label': quality_label,
                'target_valid': True,
                'closer_region_persisted': bool(persisted),
                'roi_area_px': roi_area,
                'dry_run': True,
                'real_arm_motion': False,
            }
        )
        payload.update(self.reference_payload(mask_area_px))
        return payload

    def stabilize_payload(self, payload):
        raw_state = str(payload.get('occlusion_state', 'UNKNOWN'))
        payload['raw_occlusion_state'] = raw_state

        if self.filtered_occlusion_state is None:
            self.filtered_occlusion_state = raw_state
            payload['state_filter_count'] = 1
            return payload

        if raw_state == self.filtered_occlusion_state:
            self.pending_occlusion_state = None
            self.pending_occlusion_count = 0
            payload['state_filter_count'] = 0
            return payload

        if raw_state != self.pending_occlusion_state:
            self.pending_occlusion_state = raw_state
            self.pending_occlusion_count = 1
        else:
            self.pending_occlusion_count += 1

        required = self.required_state_confirmations(raw_state)
        payload['state_filter_count'] = int(self.pending_occlusion_count)
        if self.pending_occlusion_count >= required:
            self.filtered_occlusion_state = raw_state
            self.pending_occlusion_state = None
            self.pending_occlusion_count = 0
            return payload

        payload['occlusion_state'] = self.filtered_occlusion_state
        payload['state_filter_pending'] = raw_state
        payload['state_filter_required'] = int(required)
        payload['reason'] = '%s; holding %s until %s repeats %d/%d' % (
            payload.get('reason', ''),
            self.filtered_occlusion_state,
            raw_state,
            payload['state_filter_count'],
            required,
        )
        return payload

    def required_state_confirmations(self, state):
        if state == 'LOST':
            return max(1, int(self.get_parameter('lost_transition_confirmations').value))
        return max(1, int(self.get_parameter('state_transition_confirmations').value))

    def has_significant_closer_region(self, closer_area, closer_ratio):
        min_area = int(self.get_parameter('min_occluder_area_px').value)
        partial_ratio = float(self.get_parameter('partial_occlusion_ratio').value)
        return closer_area >= min_area and closer_ratio >= partial_ratio

    def classify(self, quality_label, valid_depth_ratio, closer_area, closer_ratio, persisted, mask_area_px):
        partial_ratio = float(self.get_parameter('partial_occlusion_ratio').value)
        heavy_ratio = float(self.get_parameter('heavy_occlusion_ratio').value)
        min_depth_ratio = float(self.get_parameter('min_valid_depth_ratio').value)
        min_area = int(self.get_parameter('min_occluder_area_px').value)

        has_closer = closer_area >= min_area and closer_ratio >= partial_ratio
        heavy_closer = closer_area >= min_area and closer_ratio >= heavy_ratio
        visible_state, visible_reason = self.classify_by_visible_ratio(mask_area_px)

        if heavy_closer:
            return 'HEAVILY_OCCLUDED', 'large closer depth region'
        if visible_state == 'HEAVILY_OCCLUDED':
            return visible_state, visible_reason
        if visible_state == 'PARTIALLY_OCCLUDED':
            return visible_state, visible_reason
        if has_closer or persisted:
            if quality_label in ('ACCEPTABLE', 'POOR'):
                return 'PARTIALLY_OCCLUDED', 'closer depth region near object and reduced scan quality'
            return 'PARTIALLY_OCCLUDED', 'closer depth region near object mask'
        if quality_label == 'INVALID' or valid_depth_ratio < min_depth_ratio * 0.5:
            return 'LOST', 'target quality invalid without occlusion evidence'
        if quality_label in ('GOOD', 'ACCEPTABLE'):
            return 'CLEAR', 'target visible and no significant closer depth region'
        if quality_label == 'POOR':
            return 'LOST', 'poor scan quality without reliable occlusion evidence'
        return 'UNKNOWN', 'scan quality unavailable'

    def classify_by_visible_ratio(self, mask_area_px):
        if not self.param_bool('use_reference_mask_area') or self.reference_mask_area_px <= 0.0:
            return None, ''
        visible_ratio = float(mask_area_px) / max(1.0, self.reference_mask_area_px)
        heavy_visible_ratio = float(self.get_parameter('heavy_visible_ratio').value)
        partial_visible_ratio = float(self.get_parameter('partial_visible_ratio').value)
        if visible_ratio <= heavy_visible_ratio:
            return 'HEAVILY_OCCLUDED', 'visible mask area dropped to %.2f of clear reference' % visible_ratio
        if visible_ratio <= partial_visible_ratio:
            return 'PARTIALLY_OCCLUDED', 'visible mask area dropped to %.2f of clear reference' % visible_ratio
        return None, ''

    def update_reference_mask_area(self, mask_area_px):
        if not self.param_bool('use_reference_mask_area'):
            return
        min_reference = float(self.get_parameter('min_reference_mask_area_px').value)
        if mask_area_px < min_reference:
            return
        alpha = float(self.get_parameter('reference_update_alpha').value)
        alpha = max(0.0, min(1.0, alpha))
        if self.reference_mask_area_px <= 0.0:
            self.reference_mask_area_px = float(mask_area_px)
            return
        self.reference_mask_area_px = (1.0 - alpha) * self.reference_mask_area_px + alpha * float(mask_area_px)

    def reference_payload(self, mask_area_px):
        visible_ratio = 0.0
        if self.reference_mask_area_px > 0.0:
            visible_ratio = float(mask_area_px) / max(1.0, self.reference_mask_area_px)
        return {
            'reference_mask_area_px': float(self.reference_mask_area_px),
            'visible_mask_ratio': float(visible_ratio),
        }

    def base_payload(self, state, reason, quality=None, mask_area_px=0):
        quality = quality or self.scan_quality()
        return {
            'stamp': self.message_stamp(),
            'occlusion_state': state,
            'occlusion_score': 0.0,
            'closer_region_area_px': 0,
            'closer_region_ratio': 0.0,
            'target_depth_m': self.latest_target_depth_field(),
            'valid_depth_ratio': float(quality.get('valid_depth_ratio', 0.0)),
            'quality_label': str(quality.get('quality_label', 'UNKNOWN')),
            'target_valid': bool(self.target_valid()),
            'mask_area_px': int(mask_area_px),
            'reason': reason,
            'dry_run': True,
            'real_arm_motion': False,
        }

    def target_depth_m(self, depth_m, mask_bool):
        if self.latest_target is not None:
            for value in (self.latest_target.depth, self.latest_target.point.z):
                value = float(value)
                if math.isfinite(value) and value > 0.0:
                    return value

        masked = depth_m[mask_bool & self.valid_depth_mask(depth_m)]
        if masked.size == 0:
            return None
        return float(np.median(masked))

    def latest_target_depth_field(self):
        if self.latest_target is None:
            return 0.0
        for value in (self.latest_target.depth, self.latest_target.point.z):
            value = float(value)
            if math.isfinite(value) and value > 0.0:
                return value
        return 0.0

    def target_valid(self):
        if self.latest_target is None:
            return False
        if hasattr(self.latest_target, 'valid'):
            return bool(self.latest_target.valid)
        values = [
            float(self.latest_target.point.x),
            float(self.latest_target.point.y),
            float(self.latest_target.point.z),
            float(self.latest_target.depth),
        ]
        return all(math.isfinite(value) for value in values) and any(abs(value) > 1e-6 for value in values)

    def scan_quality(self):
        if isinstance(self.latest_scan_quality, dict):
            return self.latest_scan_quality
        return {'quality_label': 'UNKNOWN', 'valid_depth_ratio': 0.0}

    def roi_mask(self, mask_bool, bbox):
        if bbox is None:
            return np.zeros(mask_bool.shape, dtype=bool)
        dilation = max(0, int(self.get_parameter('near_mask_dilation_px').value))
        kernel_size = 2 * dilation + 1
        mask_u8 = mask_bool.astype(np.uint8) * 255
        if kernel_size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
        return mask_u8 > 0

    def valid_depth_mask(self, depth_m):
        return (
            np.isfinite(depth_m)
            & (depth_m >= float(self.get_parameter('min_valid_depth_m').value))
            & (depth_m <= float(self.get_parameter('max_valid_depth_m').value))
        )

    @staticmethod
    def remove_small_regions(mask_bool, min_area):
        mask_u8 = mask_bool.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
        cleaned = np.zeros(mask_bool.shape, dtype=bool)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area:
                cleaned[labels == label] = True
        return cleaned

    def occlusion_score(self, closer_ratio, closer_area, quality_label, valid_depth_ratio):
        area_score = min(1.0, float(closer_area) / max(1.0, float(self.get_parameter('min_occluder_area_px').value) * 4.0))
        ratio_score = min(1.0, float(closer_ratio) / max(1e-6, float(self.get_parameter('heavy_occlusion_ratio').value)))
        quality_penalty = {'GOOD': 0.0, 'ACCEPTABLE': 0.15, 'POOR': 0.35, 'INVALID': 0.60}.get(quality_label, 0.20)
        depth_penalty = max(0.0, float(self.get_parameter('min_valid_depth_ratio').value) - valid_depth_ratio)
        return max(0.0, min(1.0, 0.45 * ratio_score + 0.30 * area_score + quality_penalty + depth_penalty))

    def note_occlusion(self, has_closer_region):
        self.occlusion_history.append(bool(has_closer_region))
        max_len = max(1, int(self.get_parameter('occlusion_persistence_frames').value))
        self.occlusion_history = self.occlusion_history[-max_len:]

    def occlusion_persisted(self):
        required = max(1, int(self.get_parameter('occlusion_persistence_frames').value))
        if len(self.occlusion_history) < required:
            return False
        return all(self.occlusion_history[-required:])

    def publish_status(self, payload):
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self.status_pub.publish(msg)
        if self.param_bool('debug'):
            self.get_logger().info(
                'occlusion state=%s score=%.2f closer=%d ratio=%.2f reason=%s'
                % (
                    payload['occlusion_state'],
                    payload['occlusion_score'],
                    payload['closer_region_area_px'],
                    payload['closer_region_ratio'],
                    payload['reason'],
                )
            )

    def publish_debug(self, payload):
        msg = String()
        msg.data = (
            'state=%s score=%.2f reason=%s target_depth_m=%.3f '
            'closer_region_area_px=%d closer_region_ratio=%.3f '
            'quality_label=%s valid_depth_ratio=%.2f'
            % (
                payload['occlusion_state'],
                payload['occlusion_score'],
                payload['reason'],
                payload['target_depth_m'],
                payload['closer_region_area_px'],
                payload['closer_region_ratio'],
                payload['quality_label'],
                payload['valid_depth_ratio'],
            )
        )
        self.debug_pub.publish(msg)

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
    def depth_to_meters(depth):
        if np.issubdtype(depth.dtype, np.integer):
            return depth.astype(np.float32, copy=False) * 0.001
        depth = depth.astype(np.float32, copy=False)
        finite = depth[np.isfinite(depth) & (depth > 0.0)]
        if finite.size > 0 and float(np.nanmedian(finite)) > 20.0:
            return depth * 0.001
        return depth

    @staticmethod
    def mask_bbox(mask_bool):
        ys, xs = np.nonzero(mask_bool)
        if xs.size == 0 or ys.size == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @staticmethod
    def parse_json_msg(msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return {'parse_error': True, 'raw': msg.data}
        return payload if isinstance(payload, dict) else {'payload': payload}

    def param_bool(self, name):
        value = self.get_parameter(name).value
        if isinstance(value, str):
            return value.lower() in ('1', 'true', 'yes', 'on')
        return bool(value)


def main(args=None):
    rclpy.init(args=args)
    node = OcclusionCheckerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
