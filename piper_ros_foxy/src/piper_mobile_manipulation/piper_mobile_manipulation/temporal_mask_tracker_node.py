#!/usr/bin/env python3
"""Read-only ROS wrapper for lightweight temporal target-mask tracking."""

import json
import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from piper_mobile_manipulation.utils.temporal_tracking import (
    TemporalMaskTracker,
    TemporalTrackerConfig,
)


class TemporalMaskTrackerNode(Node):
    def __init__(self):
        super().__init__('temporal_mask_tracker_node')
        self.declare_parameter('color_image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_image_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('seed_mask_topic', '/piper/heavy_target_mask')
        self.declare_parameter('tracked_mask_topic', '/piper/temporal_target_mask')
        self.declare_parameter('debug_image_topic', '/piper/temporal_tracking_debug_image')
        self.declare_parameter('status_topic', '/piper/temporal_tracking_status')
        self.declare_parameter('heavy_refresh_request_topic', '/piper/heavy_refresh_request')
        self.declare_parameter('min_tracking_confidence', 0.50)
        self.declare_parameter('low_tracking_confidence', 0.25)
        self.declare_parameter('max_missed_frames', 5)
        self.declare_parameter('refresh_interval_frames', 1800)
        self.declare_parameter('scene_change_threshold', 45.0)
        self.declare_parameter('min_mask_area_px', 100)
        self.declare_parameter('min_depth_valid_ratio', 0.40)
        self.declare_parameter('depth_margin_m', 0.03)
        self.declare_parameter('obstacle_persistence_frames', 3)
        self.declare_parameter('enable_color_correction', True)
        self.declare_parameter('hsv_lower', [35, 80, 60])
        self.declare_parameter('hsv_upper', [88, 255, 255])
        self.declare_parameter('color_search_margin_px', 80)
        self.declare_parameter('color_min_area_px', 100)
        self.declare_parameter('color_max_centroid_shift_px', 100.0)
        self.declare_parameter('color_depth_tolerance_m', 0.15)
        self.declare_parameter('require_color_correction', True)
        self.declare_parameter('enable_adaptive_appearance', True)
        self.declare_parameter('appearance_distance_threshold', 3.5)
        self.declare_parameter('appearance_min_chroma_sigma', 8.0)
        self.declare_parameter('appearance_max_chroma_sigma', 35.0)
        self.declare_parameter('appearance_update_rate', 0.05)
        self.declare_parameter('use_hsv_fallback', True)
        self.declare_parameter('dry_run', True)
        self.declare_parameter('enable_real_arm_motion', False)
        self.declare_parameter('debug', True)

        config = TemporalTrackerConfig(
            min_tracking_confidence=float(self.get_parameter('min_tracking_confidence').value),
            low_tracking_confidence=float(self.get_parameter('low_tracking_confidence').value),
            max_missed_frames=int(self.get_parameter('max_missed_frames').value),
            refresh_interval_frames=int(self.get_parameter('refresh_interval_frames').value),
            scene_change_threshold=float(self.get_parameter('scene_change_threshold').value),
            min_mask_area_px=int(self.get_parameter('min_mask_area_px').value),
            min_depth_valid_ratio=float(self.get_parameter('min_depth_valid_ratio').value),
            depth_margin_m=float(self.get_parameter('depth_margin_m').value),
            obstacle_persistence_frames=int(self.get_parameter('obstacle_persistence_frames').value),
            enable_color_correction=bool(self.get_parameter('enable_color_correction').value),
            hsv_lower=tuple(int(value) for value in self.get_parameter('hsv_lower').value),
            hsv_upper=tuple(int(value) for value in self.get_parameter('hsv_upper').value),
            color_search_margin_px=int(self.get_parameter('color_search_margin_px').value),
            color_min_area_px=int(self.get_parameter('color_min_area_px').value),
            color_max_centroid_shift_px=float(self.get_parameter('color_max_centroid_shift_px').value),
            color_depth_tolerance_m=float(self.get_parameter('color_depth_tolerance_m').value),
            require_color_correction=bool(self.get_parameter('require_color_correction').value),
            enable_adaptive_appearance=bool(self.get_parameter('enable_adaptive_appearance').value),
            appearance_distance_threshold=float(self.get_parameter('appearance_distance_threshold').value),
            appearance_min_chroma_sigma=float(self.get_parameter('appearance_min_chroma_sigma').value),
            appearance_max_chroma_sigma=float(self.get_parameter('appearance_max_chroma_sigma').value),
            appearance_update_rate=float(self.get_parameter('appearance_update_rate').value),
            use_hsv_fallback=bool(self.get_parameter('use_hsv_fallback').value),
        )
        self.bridge = CvBridge()
        self.tracker = TemporalMaskTracker(config)
        self.latest_depth = None
        self.pending_seed_mask = None
        self.pending_seed_source = ''
        self.request_sequence = 0
        self.search_request_latched = False
        self.awaiting_heavy_seed = True

        self.mask_pub = self.create_publisher(
            Image, self.get_parameter('tracked_mask_topic').value, qos_profile_sensor_data
        )
        self.debug_pub = self.create_publisher(
            Image, self.get_parameter('debug_image_topic').value, qos_profile_sensor_data
        )
        self.status_pub = self.create_publisher(
            String, self.get_parameter('status_topic').value, 10
        )
        self.refresh_pub = self.create_publisher(
            String, self.get_parameter('heavy_refresh_request_topic').value, 10
        )
        self.create_subscription(
            Image,
            self.get_parameter('depth_image_topic').value,
            self.depth_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            self.get_parameter('seed_mask_topic').value,
            self.seed_mask_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            self.get_parameter('color_image_topic').value,
            self.color_cb,
            qos_profile_sensor_data,
        )
        self.get_logger().warn(
            'Temporal mask tracker is read-only; dry_run=%s real arm motion is always disabled.'
            % bool(self.get_parameter('dry_run').value)
        )

    def depth_cb(self, msg):
        try:
            self.latest_depth = np.asarray(
                self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            )
        except Exception as exc:
            self.get_logger().warn('Depth conversion failed: %s' % exc)

    def seed_mask_cb(self, msg):
        if not self.awaiting_heavy_seed:
            return
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as exc:
            self.get_logger().warn('Seed mask conversion failed: %s' % exc)
            return
        self.pending_seed_mask = np.asarray(mask) > 0
        self.pending_seed_source = msg.header.frame_id or 'heavy_target_mask'
        self.search_request_latched = False

    def color_cb(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn('Color conversion failed: %s' % exc)
            return
        image = np.asarray(image)

        if self.pending_seed_mask is not None:
            seed = self.pending_seed_mask
            self.pending_seed_mask = None
            try:
                if self.tracker.initialized:
                    result = self.tracker.apply_heavy_refresh(image, seed, self.latest_depth)
                else:
                    result = self.tracker.initialize(image, seed, self.latest_depth)
                    result.mode = 'HEAVY_INITIALIZED'
            except Exception as exc:
                self.publish_searching(msg, image, 'seed_rejected: %s' % exc)
                return
            tracked_mask = self.tracker.mask.copy()
            self.search_request_latched = False
            self.awaiting_heavy_seed = False
        elif not self.tracker.initialized:
            self.publish_searching(msg, image, 'heavy_seed_required')
            return
        else:
            result, tracked_mask = self.tracker.step(image, self.latest_depth)

        self.publish_mask(msg, tracked_mask)
        self.publish_debug(msg, image, tracked_mask, result)
        self.publish_status(result.to_dict(), seed_source=self.pending_seed_source)
        if result.heavy_refresh_requested and not self.awaiting_heavy_seed:
            self.publish_refresh_request(result.heavy_refresh_reason, msg, result.to_dict())

    def publish_searching(self, image_msg, image, reason):
        payload = {
            'mode': 'SEARCHING',
            'target_valid': False,
            'tracking_confidence': 0.0,
            'heavy_refresh_requested': True,
            'heavy_refresh_reason': reason,
        }
        self.publish_status(payload)
        if not self.search_request_latched:
            self.publish_refresh_request(reason, image_msg, payload)
            self.search_request_latched = True
        if bool(self.get_parameter('debug').value):
            debug = image.copy()
            self.draw_text(debug, 'SEARCHING: heavy seed required', (0, 0, 255))
            out = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
            out.header = image_msg.header
            self.debug_pub.publish(out)

    def publish_mask(self, image_msg, mask):
        out = self.bridge.cv2_to_imgmsg(mask.astype(np.uint8) * 255, encoding='mono8')
        out.header = image_msg.header
        self.mask_pub.publish(out)

    def publish_debug(self, image_msg, image, mask, result):
        if not bool(self.get_parameter('debug').value):
            return
        debug = image.copy()
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug, contours, -1, (0, 255, 0), 2)
        obstacle_mask = self.tracker.depth_obstacle_mask(mask, self.latest_depth)
        if np.any(obstacle_mask):
            red = np.zeros_like(debug)
            red[:, :, 2] = obstacle_mask.astype(np.uint8) * 255
            debug = cv2.addWeighted(debug, 1.0, red, 0.45, 0.0)
            obstacle_contours, _ = cv2.findContours(
                obstacle_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(debug, obstacle_contours, -1, (0, 0, 255), 2)
        text = '%s conf=%.2f appearance=%s refresh=%s obstacle=%s' % (
            result.mode,
            result.tracking_confidence,
            result.appearance_correction_used,
            result.heavy_refresh_requested,
            result.obstacle_persistent,
        )
        self.draw_text(debug, text, (0, 255, 0))
        out = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        out.header = image_msg.header
        self.debug_pub.publish(out)

    def publish_status(self, payload, seed_source=''):
        message = dict(payload)
        message.update(
            {
                'seed_source': seed_source,
                'dry_run': True,
                'real_arm_motion': False,
            }
        )
        out = String()
        out.data = json.dumps(self.json_safe(message), sort_keys=True)
        self.status_pub.publish(out)

    def publish_refresh_request(self, reason, image_msg, tracking_payload):
        self.request_sequence += 1
        self.awaiting_heavy_seed = True
        out = String()
        out.data = json.dumps(
            self.json_safe(
                {
                    'request_id': self.request_sequence,
                    'reason': reason,
                    'image_stamp': {
                        'sec': int(image_msg.header.stamp.sec),
                        'nanosec': int(image_msg.header.stamp.nanosec),
                    },
                    'color_image_topic': self.get_parameter('color_image_topic').value,
                    'depth_image_topic': self.get_parameter('depth_image_topic').value,
                    'tracked_mask_topic': self.get_parameter('tracked_mask_topic').value,
                    'tracking': tracking_payload,
                    'dry_run': True,
                    'real_arm_motion': False,
                }
            ),
            sort_keys=True,
        )
        self.refresh_pub.publish(out)
        self.get_logger().info('Heavy refresh requested: %s' % reason)

    @staticmethod
    def draw_text(image, text, color):
        cv2.putText(image, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    @classmethod
    def json_safe(cls, value):
        if isinstance(value, dict):
            return {str(key): cls.json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls.json_safe(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, np.generic):
            return cls.json_safe(value.item())
        return value


def main(args=None):
    rclpy.init(args=args)
    node = TemporalMaskTrackerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
