#!/usr/bin/env python3
"""Show the live RGB, SAM target, aligned depth, and L515 confidence streams."""

import os
import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from piper_mobile_manipulation.msg import Target3D


WINDOW = 'PiPER Range Test - RGB | SAM | Depth | Confidence'


class RangeDebugDashboard(Node):
    """Read-only four-panel camera/perception diagnostic viewer."""

    def __init__(self):
        super().__init__('piper_range_debug_dashboard')
        self.bridge = CvBridge()
        self.rgb = None
        self.depth = None
        self.confidence = None
        self.mask = None
        self.target = None
        self.maximum_display_depth_m = max(
            3.00,
            float(os.environ.get(
                'PIPER_RANGE_DASHBOARD_MAX_DEPTH_M', '4.0')),
        )

        topics = (
            ('/camera/color/image_raw', 'rgb'),
            ('/camera/aligned_depth_to_color/image_raw', 'depth'),
            ('/camera/confidence/image_rect_raw', 'confidence'),
            ('/piper/sam2_target_mask', 'mask'),
        )
        for topic, attribute in topics:
            self.create_subscription(
                Image,
                topic,
                lambda message, name=attribute: self.image_callback(
                    message, name),
                qos_profile_sensor_data,
            )
        self.create_subscription(
            Target3D, '/piper/target_3d', self.target_callback, 10)
        self.create_timer(1.0 / 20.0, self.draw)

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1280, 760)
        cv2.moveWindow(WINDOW, 80, 80)
        self.get_logger().info(
            'Range dashboard is read-only. Press q or Escape to quit.')

    def image_callback(self, message, attribute):
        """Cache the newest frame without modifying its ROS-owned storage."""
        try:
            encoding = 'bgr8' if attribute == 'rgb' else 'passthrough'
            image = self.bridge.imgmsg_to_cv2(
                message, desired_encoding=encoding)
            setattr(self, attribute, np.asarray(image).copy())
        except Exception as exc:
            self.get_logger().warning(
                'Could not decode %s: %s' % (attribute, exc))

    def target_callback(self, message):
        """Cache the latest production target-depth assessment."""
        self.target = message

    @staticmethod
    def label(image, title, lines=()):
        """Add a high-contrast title and compact diagnostic lines."""
        output = image.copy()
        height = 34 + 25 * len(lines)
        cv2.rectangle(output, (0, 0), (output.shape[1], height), (0, 0, 0), -1)
        cv2.putText(
            output, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2, cv2.LINE_AA)
        for index, line in enumerate(lines):
            cv2.putText(
                output, line, (12, 55 + 25 * index),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                (255, 255, 255), 1, cv2.LINE_AA)
        return output

    def rgb_with_mask(self):
        """Render the latest SAM target mask on the RGB frame."""
        frame = self.rgb.copy()
        height, width = frame.shape[:2]
        mask_pixels = 0
        if self.mask is not None:
            mask = self.mask[:, :, 0] if self.mask.ndim == 3 else self.mask
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(
                    mask, (width, height), interpolation=cv2.INTER_NEAREST)
            active = mask > 0
            mask_pixels = int(np.count_nonzero(active))
            tint = frame.copy()
            tint[active] = (0, 255, 0)
            frame = cv2.addWeighted(frame, 0.72, tint, 0.28, 0.0)
            contours, _ = cv2.findContours(
                active.astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 255, 255), 2)

        lines = ['SAM mask: %d pixels' % mask_pixels]
        if self.depth is not None and self.mask is not None:
            depth = self.depth.astype(np.float32)
            if self.depth.dtype == np.uint16:
                depth *= 0.001
            raw_support = active & np.isfinite(depth) \
                & (depth >= 0.15) & (depth <= 9.0)
            raw_values = depth[raw_support]
            raw_ratio = float(raw_values.size) / float(max(mask_pixels, 1))
            raw_depth = (
                float(np.median(raw_values)) if raw_values.size else 0.0)
            lines.append(
                'Diagnostic raw mask depth: %.3f m | ratio: %.2f' % (
                    raw_depth, raw_ratio))
        if self.target is not None:
            lines.extend((
                'Target depth: %.3f m | valid: %s' % (
                    self.target.depth, self.target.valid),
                'Depth ratio: %.2f | std: %.4f m | confidence: %.2f' % (
                    self.target.valid_depth_ratio,
                    self.target.depth_stddev,
                    self.target.measurement_confidence),
            ))
        return self.label(frame, 'RGB + SAM target', lines)

    def depth_panel(self):
        """Render aligned depth with a fixed range rather than auto-scaling."""
        if self.depth is None:
            return self.label(np.zeros_like(self.rgb), 'Waiting for aligned depth')
        depth = self.depth.astype(np.float32)
        if self.depth.dtype == np.uint16:
            depth *= 0.001
        valid = np.isfinite(depth) & (depth > 0.0)
        scaled = np.clip(
            (depth - 0.15) / (self.maximum_display_depth_m - 0.15),
            0.0,
            1.0,
        )
        panel = cv2.applyColorMap(
            (255.0 * (1.0 - scaled)).astype(np.uint8), cv2.COLORMAP_TURBO)
        panel[~valid] = 0
        height, width = depth.shape[:2]
        centre = depth[height // 2, width // 2]
        return self.label(
            panel,
            'Aligned depth (raw frame, fixed 0.15-%.2f m)' % (
                self.maximum_display_depth_m),
            ('Centre depth: %.3f m' % centre,),
        )

    def confidence_panel(self):
        """Scale the native four-bit confidence values across 0-255."""
        if self.confidence is None:
            return self.label(np.zeros_like(self.rgb), 'Waiting for confidence')
        confidence = self.confidence
        if confidence.ndim == 3:
            confidence = confidence[:, :, 0]
        scaled = np.clip(
            confidence.astype(np.float32) * (255.0 / 15.0), 0.0, 255.0)
        panel = cv2.applyColorMap(
            scaled.astype(np.uint8), cv2.COLORMAP_TURBO)
        return self.label(
            panel,
            'L515 confidence (native 0-15)',
            ('Median: %.1f | high pixels (>=12): %.1f%%' % (
                float(np.median(confidence)),
                100.0 * float(np.mean(confidence >= 12))),),
        )

    def draw(self):
        """Draw one coherent dashboard from the newest available samples."""
        if self.rgb is None:
            canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(
                canvas, 'Waiting for RGB...', (420, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        else:
            raw = self.label(
                self.rgb,
                'Raw RGB',
                ('Move target nearer/farther; watch all four panels',),
            )
            panels = (
                raw,
                self.rgb_with_mask(),
                self.depth_panel(),
                self.confidence_panel(),
            )
            resized = tuple(
                cv2.resize(panel, (640, 360), interpolation=cv2.INTER_AREA)
                for panel in panels)
            canvas = np.vstack((
                np.hstack(resized[:2]),
                np.hstack(resized[2:]),
            ))
        cv2.imshow(WINDOW, canvas)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            rclpy.shutdown()


def main():
    """Run the read-only ROS viewer."""
    rclpy.init()
    node = RangeDebugDashboard()
    try:
        rclpy.spin(node)
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
