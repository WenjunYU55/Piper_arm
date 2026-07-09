#!/usr/bin/env python3
import json
import math
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from piper_mobile_manipulation.msg import Detection2D, Target3D


class ActiveScanDebugOverlayNode(Node):
    def __init__(self):
        super().__init__('active_scan_debug_overlay_node')
        self.declare_parameter('color_image_topic', '/camera/color/image_raw')
        self.declare_parameter('detection_topic', '/piper/detection_2d')
        self.declare_parameter('detection_debug_image_topic', '/piper/detection_debug_image')
        self.declare_parameter('target_3d_topic', '/piper/target_3d')
        self.declare_parameter('scan_viewpoints_topic', '/piper/scan_viewpoints')
        self.declare_parameter('scan_coverage_topic', '/piper/scan_coverage')
        self.declare_parameter('reachable_scan_viewpoints_topic', '/piper/reachable_scan_viewpoints')
        self.declare_parameter('scan_quality_topic', '/piper/scan_quality')
        self.declare_parameter('useful_scan_coverage_topic', '/piper/useful_scan_coverage')
        self.declare_parameter('occlusion_status_topic', '/piper/occlusion_status')
        self.declare_parameter('debug_image_topic', '/piper/active_scan_debug_image')
        self.declare_parameter('prefer_detection_debug_image', True)
        self.declare_parameter('stale_timeout_s', 1.0)
        self.declare_parameter('scan_stale_timeout_s', 30.0)
        self.declare_parameter('dry_run', True)
        self.declare_parameter('enable_real_arm_motion', False)

        self.bridge = CvBridge()
        self.latest_detection = None
        self.latest_target = None
        self.latest_scan_viewpoints = None
        self.latest_scan_coverage = None
        self.latest_reachable_scan_viewpoints = None
        self.latest_scan_quality = None
        self.latest_useful_scan_coverage = None
        self.latest_occlusion_status = None
        self.latest_color_stamp = 0.0
        self.latest_debug_stamp = 0.0

        self.pub = self.create_publisher(
            Image,
            self.get_parameter('debug_image_topic').value,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            self.get_parameter('color_image_topic').value,
            self.color_image_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            self.get_parameter('detection_debug_image_topic').value,
            self.detection_debug_image_cb,
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
            self.get_parameter('scan_viewpoints_topic').value,
            self.scan_viewpoints_cb,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter('scan_coverage_topic').value,
            self.scan_coverage_cb,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter('reachable_scan_viewpoints_topic').value,
            self.reachable_scan_viewpoints_cb,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter('scan_quality_topic').value,
            self.scan_quality_cb,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter('useful_scan_coverage_topic').value,
            self.useful_scan_coverage_cb,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter('occlusion_status_topic').value,
            self.occlusion_status_cb,
            10,
        )
        self.get_logger().warn(
            'Active scan debug overlay is visual-only; it does not publish /piper/servo_cmd or move the arm.'
        )

    def detection_cb(self, msg):
        self.latest_detection = (msg, time.monotonic())

    def target_cb(self, msg):
        self.latest_target = (msg, time.monotonic())

    def scan_viewpoints_cb(self, msg):
        self.latest_scan_viewpoints = (self.parse_json_msg(msg), time.monotonic())

    def scan_coverage_cb(self, msg):
        self.latest_scan_coverage = (self.parse_json_msg(msg), time.monotonic())

    def reachable_scan_viewpoints_cb(self, msg):
        self.latest_reachable_scan_viewpoints = (self.parse_json_msg(msg), time.monotonic())

    def scan_quality_cb(self, msg):
        self.latest_scan_quality = (self.parse_json_msg(msg), time.monotonic())

    def useful_scan_coverage_cb(self, msg):
        self.latest_useful_scan_coverage = (self.parse_json_msg(msg), time.monotonic())

    def occlusion_status_cb(self, msg):
        self.latest_occlusion_status = (self.parse_json_msg(msg), time.monotonic())

    def color_image_cb(self, msg):
        self.latest_color_stamp = time.monotonic()
        if self.param_bool('prefer_detection_debug_image'):
            debug_age = self.latest_color_stamp - self.latest_debug_stamp
            if self.latest_debug_stamp > 0.0 and debug_age < 0.5:
                return
        self.publish_overlay(msg, base_source='color')

    def detection_debug_image_cb(self, msg):
        self.latest_debug_stamp = time.monotonic()
        self.publish_overlay(msg, base_source='detection_debug')

    def publish_overlay(self, image_msg, base_source):
        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn('Could not convert base image %s: %s' % (image_msg.encoding, exc))
            return

        overlay_lines = self.overlay_lines(base_source)
        self.draw_text_panel(image, overlay_lines)
        out = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        out.header = image_msg.header
        self.pub.publish(out)

    def overlay_lines(self, base_source):
        detection_status = 'detector: unknown'
        if self.latest_detection is not None:
            det, age_stamp = self.latest_detection
            age = time.monotonic() - age_stamp
            if age <= self.stale_timeout():
                detection_status = 'detector: valid=%s conf=%.2f' % (det.valid, det.confidence)
            else:
                detection_status = 'detector: stale %.1fs' % age

        target_status = 'target: unknown'
        target_depth = 'distance: unknown'
        if self.latest_target is not None:
            target, age_stamp = self.latest_target
            age = time.monotonic() - age_stamp
            if age <= self.stale_timeout():
                target_status = 'target valid: %s' % target.valid
                if math.isfinite(float(target.depth)) and target.depth > 0.0:
                    target_depth = 'distance: %.3f m' % target.depth
                elif math.isfinite(float(target.point.z)) and target.point.z > 0.0:
                    target_depth = 'distance: %.3f m' % target.point.z
                else:
                    target_depth = 'distance: unavailable'
            else:
                target_status = 'target: stale %.1fs' % age

        planned = self.planned_viewpoint_count()
        reachable = self.reachable_viewpoint_count()
        coverage = self.scan_coverage_target()
        quality = self.scan_quality_status()
        useful_coverage = self.useful_scan_coverage()
        occlusion = self.occlusion_status()
        real_motion = 'enabled' if self.param_bool('enable_real_arm_motion') else 'disabled'

        return [
            'active scan debug (%s)' % base_source,
            detection_status,
            target_depth,
            target_status,
            'planned viewpoints: %s' % planned,
            'reachable viewpoints: %s' % reachable,
            'scan coverage target: %s' % coverage,
            'view quality: %s' % quality,
            'useful coverage: %s' % useful_coverage,
            'occlusion: %s' % occlusion,
            'dry-run mode: %s' % self.param_bool('dry_run'),
            'real arm motion: %s' % real_motion,
        ]

    def planned_viewpoint_count(self):
        payload = self.latest_scan_payload(self.latest_scan_viewpoints)
        if payload is None:
            return 'unknown'
        if isinstance(payload.get('viewpoints'), list):
            return str(len(payload['viewpoints']))
        value = payload.get('candidate_viewpoints')
        return str(value) if value is not None else 'unknown'

    def reachable_viewpoint_count(self):
        payload = self.latest_scan_payload(self.latest_reachable_scan_viewpoints)
        if payload is None:
            return 'unknown'
        filter_info = payload.get('filter')
        if isinstance(filter_info, dict):
            value = filter_info.get('reachable_viewpoints')
            if value is not None:
                return str(value)
        if isinstance(payload.get('viewpoints'), list):
            return str(sum(1 for viewpoint in payload['viewpoints'] if viewpoint.get('reachable')))
        return 'unknown'

    def scan_coverage_target(self):
        payload = self.latest_scan_payload(self.latest_scan_coverage)
        if payload is None:
            payload = self.latest_scan_payload(self.latest_scan_viewpoints)
        if payload is None:
            return 'unknown'
        for key in ('requested_scan_angle_deg', 'planned_scan_angle_deg'):
            value = payload.get(key)
            if value is not None:
                return '%.1f deg' % float(value)
        viewpoints = payload.get('viewpoints')
        if isinstance(viewpoints, list) and len(viewpoints) >= 2:
            angles = [
                float(v.get('viewpoint_angle_deg'))
                for v in viewpoints
                if isinstance(v, dict) and v.get('viewpoint_angle_deg') is not None
            ]
            if len(angles) >= 2:
                return '%.1f deg' % (max(angles) - min(angles))
        return 'unknown'

    def scan_quality_status(self):
        payload = self.latest_payload(self.latest_scan_quality)
        if payload is None:
            return 'unknown'
        status = payload.get('quality_label', payload.get('status', 'unknown'))
        score = payload.get('quality_score', payload.get('score'))
        if score is None:
            return str(status)
        return '%s %.2f' % (status, float(score))

    def useful_scan_coverage(self):
        payload = self.latest_scan_payload(self.latest_useful_scan_coverage)
        if payload is None:
            return 'unknown'
        coverage = payload.get('useful_coverage_deg', payload.get('useful_scan_coverage_deg'))
        useful_frames = payload.get('useful_frame_count')
        if coverage is None:
            if useful_frames is None:
                return 'unknown'
            return '%s useful frames' % useful_frames
        if useful_frames is None:
            return '%.1f deg' % float(coverage)
        return '%.1f deg (%s useful frames)' % (float(coverage), useful_frames)

    def occlusion_status(self):
        payload = self.latest_payload(self.latest_occlusion_status)
        if payload is None:
            return 'unknown'
        state = payload.get('occlusion_state', 'unknown')
        score = float(payload.get('occlusion_score', 0.0))
        area = int(payload.get('closer_region_area_px', 0))
        return '%s %.2f area=%d' % (state, score, area)

    def latest_payload(self, stored):
        if stored is None:
            return None
        payload, stamp = stored
        if time.monotonic() - stamp > self.stale_timeout():
            return None
        return payload if isinstance(payload, dict) else None

    def latest_scan_payload(self, stored):
        if stored is None:
            return None
        payload, stamp = stored
        if time.monotonic() - stamp > self.scan_stale_timeout():
            return None
        return payload if isinstance(payload, dict) else None

    def stale_timeout(self):
        return max(0.1, float(self.get_parameter('stale_timeout_s').value))

    def scan_stale_timeout(self):
        return max(self.stale_timeout(), float(self.get_parameter('scan_stale_timeout_s').value))

    @staticmethod
    def parse_json_msg(msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return {'parse_error': True}
        return payload if isinstance(payload, dict) else {'payload': payload}

    @staticmethod
    def draw_text_panel(image, lines):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.52
        thickness = 1
        line_height = 22
        pad = 10
        width = 0
        for line in lines:
            size, _ = cv2.getTextSize(line, font, scale, thickness)
            width = max(width, size[0])
        panel_w = min(image.shape[1], width + 2 * pad)
        panel_h = min(image.shape[0], line_height * len(lines) + 2 * pad)
        panel = image[0:panel_h, 0:panel_w].copy()
        dark = panel.copy()
        dark[:, :] = (0, 0, 0)
        blended = cv2.addWeighted(panel, 0.25, dark, 0.75, 0.0)
        image[0:panel_h, 0:panel_w] = blended

        y = pad + 15
        for index, line in enumerate(lines):
            color = (0, 255, 255) if index == 0 else (255, 255, 255)
            cv2.putText(image, line, (pad, y), font, scale, color, thickness, cv2.LINE_AA)
            y += line_height

    def param_bool(self, name):
        value = self.get_parameter(name).value
        if isinstance(value, str):
            return value.lower() in ('1', 'true', 'yes', 'on')
        return bool(value)


def main(args=None):
    rclpy.init(args=args)
    node = ActiveScanDebugOverlayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
