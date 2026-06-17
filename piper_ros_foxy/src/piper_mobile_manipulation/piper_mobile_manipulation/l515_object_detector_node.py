#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from piper_mobile_manipulation.msg import Detection2D


class L515ObjectDetectorNode(Node):
    COLOR_PRESETS = {
        'green': [([35, 45, 45], [90, 255, 255])],
        'red': [([0, 70, 50], [10, 255, 255]), ([170, 70, 50], [180, 255, 255])],
        'blue': [([90, 60, 45], [130, 255, 255])],
        'yellow': [([18, 70, 70], [38, 255, 255])],
        'orange': [([5, 80, 70], [25, 255, 255])],
        'purple': [([125, 45, 45], [165, 255, 255])],
        'custom': None,
    }

    def __init__(self):
        super().__init__('l515_object_detector_node')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('detection_topic', '/piper/detection_2d')
        self.declare_parameter('debug_image_topic', '/piper/detection_debug_image')
        self.declare_parameter('mask_topic', '/piper/detection_mask')
        self.declare_parameter('target_color', 'green')
        self.declare_parameter('use_color_preset', True)
        self.declare_parameter('hsv_lower', [30, 50, 50])
        self.declare_parameter('hsv_upper', [90, 255, 255])
        self.declare_parameter('min_contour_area', 200.0)
        self.declare_parameter('max_contour_area', 100000.0)
        self.declare_parameter('morph_kernel_size', 5)
        self.declare_parameter('morph_open_kernel_size', 0)
        self.declare_parameter('morph_close_kernel_size', 0)
        self.declare_parameter('min_extent', 0.15)
        self.declare_parameter('min_circularity', 0.0)
        self.declare_parameter('min_detection_confidence', 0.0)
        self.declare_parameter('prefer_centered', True)
        self.declare_parameter('area_confidence_full_scale', 5.0)
        self.declare_parameter('log_valid_every_n', 30)

        self.bridge = CvBridge()
        self.target_color = ''
        self.use_color_preset = None
        self.hsv_lower_param = None
        self.hsv_upper_param = None
        self.hsv_ranges = []
        self.refresh_runtime_params()
        self.frame_count = 0

        self.pub = self.create_publisher(
            Detection2D, self.get_parameter('detection_topic').value, 10
        )
        self.debug_pub = self.create_publisher(
            Image, self.get_parameter('debug_image_topic').value, qos_profile_sensor_data
        )
        self.mask_pub = self.create_publisher(
            Image, self.get_parameter('mask_topic').value, qos_profile_sensor_data
        )
        self.sub = self.create_subscription(
            Image, self.get_parameter('image_topic').value, self.image_cb, qos_profile_sensor_data
        )
        self.get_logger().info(
            'Object detector listening on %s target_color=%s'
            % (self.get_parameter('image_topic').value, self.target_color)
        )

    def load_hsv_ranges(self):
        use_color_preset = bool(self.get_parameter('use_color_preset').value)
        preset = self.COLOR_PRESETS.get(self.target_color) if use_color_preset else None
        if preset is None:
            if self.target_color != 'custom':
                if use_color_preset:
                    self.get_logger().warn(
                        'Unknown target_color=%s. Falling back to hsv_lower/hsv_upper.'
                        % self.target_color
                    )
                else:
                    self.get_logger().info(
                        'Using hsv_lower/hsv_upper for target_color=%s.' % self.target_color
                    )
            lower = np.array(self.get_parameter('hsv_lower').value, dtype=np.uint8)
            upper = np.array(self.get_parameter('hsv_upper').value, dtype=np.uint8)
            return [(lower, upper)]
        return [
            (np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
            for lower, upper in preset
        ]

    def image_cb(self, image_msg):
        self.frame_count += 1
        self.refresh_runtime_params()
        out = Detection2D()
        out.header = image_msg.header
        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as exc:
            out.valid = False
            self.pub.publish(out)
            self.get_logger().warn('cv_bridge failed: %s' % exc)
            return

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.hsv_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        mask = self.clean_mask(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = self.choose_best_contour(contours, image.shape[1], image.shape[0])

        debug = image.copy()
        debug_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.addWeighted(debug, 0.82, debug_mask, 0.18, 0.0, debug)

        if best is None:
            out.valid = False
            out.confidence = 0.0
            self.draw_status(debug, 'no %s object' % self.target_color, (0, 0, 255))
        else:
            contour, area, score, extent, circularity = best
            x, y, w, h = cv2.boundingRect(contour)
            out.u = float(x + w / 2.0)
            out.v = float(y + h / 2.0)
            out.width = float(w)
            out.height = float(h)
            out.confidence = float(np.clip(score, 0.0, 1.0))
            out.valid = True
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(debug, (int(out.u), int(out.v)), 4, (0, 0, 255), -1)
            self.draw_status(
                debug,
                '%s conf=%.2f area=%.0f extent=%.2f circ=%.2f'
                % (self.target_color, out.confidence, area, extent, circularity),
                (0, 255, 0),
            )

        self.pub.publish(out)
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_msg.header = image_msg.header
        self.mask_pub.publish(mask_msg)
        debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        debug_msg.header = image_msg.header
        self.debug_pub.publish(debug_msg)
        if out.valid and self.should_log_detection():
            self.get_logger().info(
                'Detection2D color=%s u=%.1f v=%.1f size=(%.1f, %.1f) conf=%.2f'
                % (self.target_color, out.u, out.v, out.width, out.height, out.confidence)
            )

    def clean_mask(self, mask):
        if self.open_kernel_size > 1:
            open_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.open_kernel_size, self.open_kernel_size)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        if self.close_kernel_size > 1:
            close_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.close_kernel_size, self.close_kernel_size)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        return mask

    def choose_best_contour(self, contours, image_width, image_height):
        best = None
        image_center = np.array([image_width / 2.0, image_height / 2.0], dtype=np.float32)
        max_center_dist = float(np.linalg.norm(image_center))
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            bbox_area = float(max(w * h, 1))
            extent = float(area) / bbox_area
            if extent < self.min_extent:
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = 0.0
            if perimeter > 0.0:
                circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
            if circularity < self.min_circularity:
                continue
            area_full_scale = max(self.min_area * self.area_confidence_full_scale, self.min_area + 1.0)
            area_score = float(np.clip((area - self.min_area) / (area_full_scale - self.min_area), 0.0, 1.0))
            center_score = 1.0
            if self.prefer_centered:
                center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32)
                center_dist = float(np.linalg.norm(center - image_center))
                center_score = 1.0 - float(np.clip(center_dist / max_center_dist, 0.0, 1.0))
            extent_score = float(np.clip((extent - self.min_extent) / max(1.0 - self.min_extent, 1e-3), 0.0, 1.0))
            score = 0.45 * area_score + 0.35 * extent_score + 0.20 * center_score
            if score < self.min_detection_confidence:
                continue
            if best is None or score > best[2]:
                best = (contour, area, score, extent, circularity)
        return best

    def should_log_detection(self):
        return self.log_valid_every_n > 0 and self.frame_count % self.log_valid_every_n == 0

    def refresh_runtime_params(self):
        target_color = str(self.get_parameter('target_color').value).lower()
        hsv_lower_param = list(self.get_parameter('hsv_lower').value)
        hsv_upper_param = list(self.get_parameter('hsv_upper').value)
        use_color_preset = bool(self.get_parameter('use_color_preset').value)
        if (
            target_color != self.target_color
            or use_color_preset != self.use_color_preset
            or hsv_lower_param != self.hsv_lower_param
            or hsv_upper_param != self.hsv_upper_param
        ):
            self.target_color = target_color
            self.use_color_preset = use_color_preset
            self.hsv_lower_param = hsv_lower_param
            self.hsv_upper_param = hsv_upper_param
            self.hsv_ranges = self.load_hsv_ranges()

        self.min_area = float(self.get_parameter('min_contour_area').value)
        self.max_area = float(self.get_parameter('max_contour_area').value)
        self.kernel_size = max(1, int(self.get_parameter('morph_kernel_size').value))
        open_size = int(self.get_parameter('morph_open_kernel_size').value)
        close_size = int(self.get_parameter('morph_close_kernel_size').value)
        self.open_kernel_size = self.normalize_kernel_size(open_size or self.kernel_size)
        self.close_kernel_size = self.normalize_kernel_size(close_size or self.kernel_size)
        self.min_extent = float(self.get_parameter('min_extent').value)
        self.min_circularity = float(self.get_parameter('min_circularity').value)
        self.min_detection_confidence = float(self.get_parameter('min_detection_confidence').value)
        self.prefer_centered = bool(self.get_parameter('prefer_centered').value)
        self.area_confidence_full_scale = max(1.1, float(self.get_parameter('area_confidence_full_scale').value))
        self.log_valid_every_n = max(0, int(self.get_parameter('log_valid_every_n').value))

    @staticmethod
    def normalize_kernel_size(size):
        size = max(1, int(size))
        if size > 1 and size % 2 == 0:
            size += 1
        return size

    @staticmethod
    def draw_status(image, text, color):
        cv2.putText(
            image,
            text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


def main(args=None):
    rclpy.init(args=args)
    node = L515ObjectDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
