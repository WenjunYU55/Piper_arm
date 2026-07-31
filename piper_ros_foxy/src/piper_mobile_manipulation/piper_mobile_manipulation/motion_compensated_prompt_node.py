#!/usr/bin/env python3
"""Reproject the last depth-supported target mask through eye-in-hand motion."""

import json
import time
from collections import OrderedDict

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from piper_mobile_manipulation.msg import TrackedTarget
from piper_mobile_manipulation.utils.mask_reprojection import (
    backproject_mask,
    project_points,
    rasterize_prompt,
    transform_matrix,
    transform_points,
)


class MotionCompensatedPromptNode(Node):
    def __init__(self):
        super().__init__('motion_compensated_prompt_node')
        defaults = {
            'color_topic': '/camera/color/image_raw',
            'depth_topic': '/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/color/camera_info',
            'mask_topic': '/piper/sam2_target_mask',
            'tracked_target_topic': '/piper/tracked_target',
            'output_topic': '/piper/motion_compensated_target_prompt',
            'status_topic': '/piper/motion_compensated_prompt_status',
            'base_frame': 'base_link',
            'depth_scale': 0.001,
            'depth_sync_tolerance_sec': 0.08,
            'max_point_age_sec': 1.0,
            'max_prediction_horizon_sec': 0.5,
            'min_support_points': 50,
            'point_stride': 2,
            'prompt_dilation_px': 5,
            'transform_timeout_sec': 0.10,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.camera_matrix = None
        self.depth_cache = OrderedDict()
        self.points_base = None
        self.points_stamp = None
        self.points_updated_at = 0.0
        self.target_velocity = np.zeros(3, dtype=float)
        self.pub = self.create_publisher(
            Image, self.get_parameter('output_topic').value, qos_profile_sensor_data
        )
        self.status_pub = self.create_publisher(
            String, self.get_parameter('status_topic').value, 10
        )
        self.create_subscription(
            Image, self.get_parameter('color_topic').value,
            self.color_cb, qos_profile_sensor_data,
        )
        self.create_subscription(
            Image, self.get_parameter('depth_topic').value,
            self.depth_cb, qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value,
            self.camera_info_cb, qos_profile_sensor_data,
        )
        self.create_subscription(
            Image, self.get_parameter('mask_topic').value,
            self.mask_cb, qos_profile_sensor_data,
        )
        self.create_subscription(
            TrackedTarget, self.get_parameter('tracked_target_topic').value,
            self.tracked_target_cb, 10,
        )
        self.get_logger().warn(
            'Motion-compensated prompts are read-only and never command arm motion.'
        )

    @staticmethod
    def stamp_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def camera_info_cb(self, msg):
        matrix = np.asarray(msg.k, dtype=float).reshape(3, 3)
        if matrix[0, 0] > 0.0 and matrix[1, 1] > 0.0:
            self.camera_matrix = matrix

    def depth_cb(self, msg):
        try:
            depth = np.asarray(
                self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            ).copy()
            if np.issubdtype(depth.dtype, np.integer):
                depth = depth.astype(np.float32) * float(
                    self.get_parameter('depth_scale').value
                )
            else:
                depth = depth.astype(np.float32)
            stamp = self.stamp_seconds(msg.header.stamp)
            self.depth_cache[stamp] = depth
            while len(self.depth_cache) > 60:
                self.depth_cache.popitem(last=False)
        except Exception as exc:
            self.publish_status('depth_rejected', error=str(exc))

    def nearest_depth(self, stamp):
        if not self.depth_cache:
            return None
        key = min(self.depth_cache, key=lambda candidate: abs(candidate - stamp))
        if abs(key - stamp) > float(
            self.get_parameter('depth_sync_tolerance_sec').value
        ):
            return None
        return self.depth_cache[key]

    def mask_cb(self, msg):
        if self.camera_matrix is None:
            return
        try:
            mask = np.asarray(
                self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            )
            stamp = self.stamp_seconds(msg.header.stamp)
            depth = self.nearest_depth(stamp)
            if depth is None:
                self.publish_status('mask_waiting_for_depth')
                return
            points_camera = backproject_mask(
                mask, depth, self.camera_matrix,
                stride=int(self.get_parameter('point_stride').value),
            )
            minimum = int(self.get_parameter('min_support_points').value)
            if points_camera.shape[0] < minimum:
                self.publish_status(
                    'mask_support_rejected', support_points=points_camera.shape[0]
                )
                return
            base_from_camera = self.lookup_matrix(
                str(self.get_parameter('base_frame').value),
                msg.header.frame_id,
                msg.header.stamp,
            )
            self.points_base = transform_points(points_camera, base_from_camera)
            self.points_stamp = stamp
            self.points_updated_at = time.monotonic()
            self.publish_status(
                'support_updated', support_points=self.points_base.shape[0]
            )
        except (ValueError, TransformException) as exc:
            self.publish_status('mask_reprojection_rejected', error=str(exc))

    def color_cb(self, msg):
        if self.camera_matrix is None or self.points_base is None:
            return
        age = time.monotonic() - self.points_updated_at
        if age > float(self.get_parameter('max_point_age_sec').value):
            return
        try:
            stamp = self.stamp_seconds(msg.header.stamp)
            dt = np.clip(
                stamp - self.points_stamp,
                0.0,
                float(self.get_parameter('max_prediction_horizon_sec').value),
            )
            predicted_points = self.points_base + self.target_velocity * dt
            base_from_camera = self.lookup_matrix(
                str(self.get_parameter('base_frame').value),
                msg.header.frame_id,
                msg.header.stamp,
            )
            points_camera = transform_points(
                predicted_points, np.linalg.inv(base_from_camera)
            )
            uv = project_points(
                points_camera, self.camera_matrix,
                (int(msg.height), int(msg.width)),
            )
            minimum = int(self.get_parameter('min_support_points').value)
            if uv.shape[0] < minimum:
                return
            prompt = rasterize_prompt(
                uv, (int(msg.height), int(msg.width)),
                dilation_px=int(self.get_parameter('prompt_dilation_px').value),
            )
            if not np.count_nonzero(prompt):
                return
            out = self.bridge.cv2_to_imgmsg(prompt, encoding='mono8')
            out.header = msg.header
            self.pub.publish(out)
            self.publish_status(
                'prompt_published', support_points=uv.shape[0],
                mask_area_px=int(np.count_nonzero(prompt)), prediction_dt=float(dt),
            )
        except (ValueError, np.linalg.LinAlgError, TransformException) as exc:
            self.publish_status('prompt_rejected', error=str(exc))

    def tracked_target_cb(self, msg):
        if msg.valid:
            self.target_velocity = np.asarray([
                msg.velocity.x, msg.velocity.y, msg.velocity.z
            ], dtype=float)

    def lookup_matrix(self, target, source, stamp):
        if not source:
            raise ValueError('source frame is empty')
        transform = self.tf_buffer.lookup_transform(
            target, source, Time.from_msg(stamp),
            timeout=Duration(
                seconds=float(self.get_parameter('transform_timeout_sec').value)
            ),
        )
        return transform_matrix(transform)

    def publish_status(self, state, **values):
        payload = {'state': state, 'dry_run': True, 'real_arm_motion': False}
        payload.update(values)
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MotionCompensatedPromptNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
