#!/usr/bin/env python3
"""Display L515 RGB with live SAM2 target and multi-object obstacle overlays."""

import json
import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


class Sam2OverlayViewer(Node):
    PALETTE = (
        (255, 140, 0),    # blue
        (180, 0, 255),    # magenta
        (0, 200, 255),    # yellow
        (255, 80, 180),
        (160, 255, 0),
        (255, 200, 80),
    )

    def __init__(self):
        super().__init__('sam2_overlay_viewer')
        self.bridge = CvBridge()
        self.target = None
        self.object_ids = None
        self.unsafe = None
        self.movable = None
        self.labels = {}
        self.window = 'PiPER SAM2: target and obstacles'

        self.create_subscription(
            Image, '/camera/color/image_raw', self.color_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, '/piper/sam2_target_mask', self.target_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, '/piper/sam2_object_ids', self.ids_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, '/piper/sam2_unsafe_obstacle_mask', self.unsafe_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            Image,
            '/piper/sam2_candidate_movable_obstacle_mask',
            self.movable_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String, '/piper/sam2_tracking_status', self.status_cb, 10
        )
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        self.get_logger().info('Press q or Escape in the image window to quit.')

    def image(self, msg, encoding):
        return np.asarray(self.bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)).copy()

    def target_cb(self, msg):
        self.target = self.image(msg, 'mono8') > 0

    def ids_cb(self, msg):
        self.object_ids = self.image(msg, 'passthrough').astype(np.uint16, copy=False)

    def unsafe_cb(self, msg):
        self.unsafe = self.image(msg, 'mono8') > 0

    def movable_cb(self, msg):
        self.movable = self.image(msg, 'mono8') > 0

    def status_cb(self, msg):
        try:
            payload = json.loads(msg.data)
            self.labels = {
                int(item.get('object_id', 0)): str(item.get('label', 'object'))
                for item in payload.get('objects', [])
                if isinstance(item, dict)
            }
        except (TypeError, ValueError):
            pass

    @staticmethod
    def compatible(mask, frame):
        return mask is not None and mask.shape[:2] == frame.shape[:2]

    @staticmethod
    def blend(frame, mask, color, alpha=0.45):
        if not np.any(mask):
            return
        frame[mask] = (
            frame[mask].astype(np.float32) * (1.0 - alpha)
            + np.asarray(color, dtype=np.float32) * alpha
        ).astype(np.uint8)

    @staticmethod
    def outline(frame, mask, color, thickness=2):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(frame, contours, -1, color, thickness)

    def color_cb(self, msg):
        try:
            frame = self.image(msg, 'bgr8')
        except Exception as exc:
            self.get_logger().warn('RGB conversion failed: %s' % exc)
            return

        legend = []
        if self.compatible(self.object_ids, frame):
            for object_id in np.unique(self.object_ids):
                object_id = int(object_id)
                if object_id <= 0:
                    continue
                mask = self.object_ids == object_id
                color = (0, 220, 0) if object_id == 1 else self.PALETTE[(object_id - 2) % len(self.PALETTE)]
                self.blend(frame, mask, color)
                self.outline(frame, mask, color)
                label = 'target' if object_id == 1 else self.labels.get(object_id, 'obstacle')
                legend.append((color, 'ID %d: %s' % (object_id, label)))
        elif self.compatible(self.target, frame):
            self.blend(frame, self.target, (0, 220, 0))
            self.outline(frame, self.target, (0, 220, 0))
            legend.append(((0, 220, 0), 'ID 1: target'))

        if self.compatible(self.unsafe, frame):
            self.outline(frame, self.unsafe, (0, 0, 255), 3)
        if self.compatible(self.movable, frame):
            self.outline(frame, self.movable, (255, 255, 0), 3)

        legend.extend([
            ((0, 0, 255), 'red outline: unsafe'),
            ((255, 255, 0), 'cyan outline: movable candidate'),
        ])
        for row, (color, text) in enumerate(legend):
            y = 24 + row * 22
            cv2.rectangle(frame, (8, y - 13), (22, y + 1), color, -1)
            cv2.putText(frame, text, (29, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
            cv2.putText(frame, text, (29, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1)

        cv2.imshow(self.window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            rclpy.shutdown()


def main():
    rclpy.init()
    node = Sam2OverlayViewer()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
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
