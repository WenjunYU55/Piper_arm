#!/usr/bin/env python3
"""Maintain and reproject a conservative stationary target landmark."""

import json
import math
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from piper_mobile_manipulation.msg import Detection2D
from piper_mobile_manipulation.obstacle_geometry import transform_points
from piper_mobile_manipulation.target_landmark_geometry import (
    direction_angle_degrees,
    maximum_pairwise_distance,
    project_camera_point,
)


class TargetLandmarkNode(Node):
    def __init__(self):
        super().__init__('target_landmark_node')
        defaults = {
            'mask_topic': '/piper/sam2_target_mask',
            'depth_topic': '/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/color/camera_info',
            'landmark_topic': '/piper/target_landmark',
            'projection_topic': '/piper/target_landmark_projection',
            'status_topic': '/piper/target_landmark_status',
            'heavy_request_topic': '/piper/heavy_refresh_request',
            'base_frame': 'base_link',
            'depth_min_m': 0.20,
            'depth_max_m': 1.20,
            'mask_erode_px': 2,
            'min_valid_depth_pixels': 50,
            'min_valid_depth_ratio': 0.40,
            'initial_sample_count': 5,
            'initial_max_spread_m': 0.020,
            'measurement_gate_m': 0.050,
            'new_view_angle_deg': 12.0,
            'landmark_update_alpha': 0.15,
            'projection_disagreement_px': 60.0,
            'sync_queue_size': 20,
            'sync_slop_sec': 0.10,
            'transform_timeout_sec': 0.20,
            'refresh_cooldown_sec': 10.0,
            'request_refresh_on_new_view': True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.initial_measurements = deque(
            maxlen=int(self.get_parameter('initial_sample_count').value))
        self.landmark = None
        self.last_view_direction = None
        self.update_count = 0
        self.last_refresh_time = -math.inf

        self.landmark_pub = self.create_publisher(
            PointStamped, self.get_parameter('landmark_topic').value, 10)
        self.projection_pub = self.create_publisher(
            Detection2D, self.get_parameter('projection_topic').value, 10)
        self.status_pub = self.create_publisher(
            String, self.get_parameter('status_topic').value, 10)
        self.refresh_pub = self.create_publisher(
            String, self.get_parameter('heavy_request_topic').value, 10)

        mask_sub = Subscriber(
            self, Image, self.get_parameter('mask_topic').value,
            qos_profile=qos_profile_sensor_data)
        depth_sub = Subscriber(
            self, Image, self.get_parameter('depth_topic').value,
            qos_profile=qos_profile_sensor_data)
        info_sub = Subscriber(
            self, CameraInfo, self.get_parameter('camera_info_topic').value,
            qos_profile=qos_profile_sensor_data)
        self.sync = ApproximateTimeSynchronizer(
            [mask_sub, depth_sub, info_sub],
            int(self.get_parameter('sync_queue_size').value),
            float(self.get_parameter('sync_slop_sec').value))
        self.sync.registerCallback(self.synced_cb)
        self.get_logger().warn(
            'Target landmark is read-only and never commands robot motion.')

    def synced_cb(self, mask_msg, depth_msg, info_msg):
        try:
            mask = np.asarray(self.bridge.imgmsg_to_cv2(mask_msg, 'mono8')) > 0
            raw_depth = np.asarray(self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough'))
            depth = raw_depth.astype(np.float64)
            if depth_msg.encoding in ('16UC1', 'mono16') or np.issubdtype(
                    raw_depth.dtype, np.integer):
                depth *= 0.001
            measurement_camera, mask_uv, ratio, count = self.measurement(
                mask, depth, info_msg.k)
            camera_to_base = self.lookup(
                self.get_parameter('base_frame').value,
                depth_msg.header.frame_id, depth_msg.header.stamp)
            measurement_base = self.apply_transform(
                measurement_camera, camera_to_base)
            camera_position = np.array([
                camera_to_base.transform.translation.x,
                camera_to_base.transform.translation.y,
                camera_to_base.transform.translation.z,
            ])
            view_direction = camera_position - measurement_base
            self.update_landmark(measurement_base, view_direction)
            self.publish_outputs(
                depth_msg, info_msg, mask, mask_uv, ratio, count,
                measurement_base, view_direction)
        except (ValueError, TransformException) as exc:
            self.publish_invalid(depth_msg, str(exc))

    def measurement(self, mask, depth, camera_matrix):
        if mask.shape != depth.shape:
            raise ValueError('mask_depth_shape_mismatch')
        original_count = int(np.count_nonzero(mask))
        if not original_count:
            raise ValueError('empty_target_mask')
        erode = int(self.get_parameter('mask_erode_px').value)
        if erode > 0:
            kernel = np.ones((erode * 2 + 1, erode * 2 + 1), np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), kernel) > 0
        valid = mask & np.isfinite(depth)
        valid &= depth >= float(self.get_parameter('depth_min_m').value)
        valid &= depth <= float(self.get_parameter('depth_max_m').value)
        count = int(np.count_nonzero(valid))
        ratio = float(count) / float(original_count)
        if count < int(self.get_parameter('min_valid_depth_pixels').value):
            raise ValueError('insufficient_valid_depth_pixels')
        if ratio < float(self.get_parameter('min_valid_depth_ratio').value):
            raise ValueError('insufficient_valid_depth_ratio')
        v, u = np.nonzero(valid)
        z = depth[valid]
        fx, fy = float(camera_matrix[0]), float(camera_matrix[4])
        cx, cy = float(camera_matrix[2]), float(camera_matrix[5])
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError('invalid_camera_intrinsics')
        points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
        return (np.median(points, axis=0), np.median(
            np.column_stack((u, v)), axis=0), ratio, count)

    def update_landmark(self, measurement, view_direction):
        if self.landmark is None:
            self.initial_measurements.append(measurement)
            required = int(self.get_parameter('initial_sample_count').value)
            if len(self.initial_measurements) < required:
                return
            spread = maximum_pairwise_distance(self.initial_measurements)
            if spread > float(self.get_parameter('initial_max_spread_m').value):
                self.initial_measurements.popleft()
                return
            self.landmark = np.median(
                np.asarray(self.initial_measurements), axis=0)
            self.last_view_direction = view_direction
            self.update_count = 1
            return

        error = float(np.linalg.norm(measurement - self.landmark))
        if error > float(self.get_parameter('measurement_gate_m').value):
            return
        angle = direction_angle_degrees(view_direction, self.last_view_direction)
        if angle < float(self.get_parameter('new_view_angle_deg').value):
            return
        alpha = float(np.clip(
            self.get_parameter('landmark_update_alpha').value, 0.0, 1.0))
        self.landmark = (1.0 - alpha) * self.landmark + alpha * measurement
        self.last_view_direction = view_direction
        self.update_count += 1
        self.request_refresh('new_viewpoint')

    def publish_outputs(self, depth_msg, info_msg, mask, mask_uv, ratio, count,
                        measurement, view_direction):
        if self.landmark is None:
            self.publish_status('INITIALIZING', valid=False, depth_ratio=ratio,
                                depth_pixels=count)
            return
        base_to_camera = self.lookup(
            depth_msg.header.frame_id, self.get_parameter('base_frame').value,
            depth_msg.header.stamp)
        landmark_camera = self.apply_transform(self.landmark, base_to_camera)
        projected = project_camera_point(landmark_camera, info_msg.k)
        height, width = mask.shape
        u, v = float(projected[0]), float(projected[1])
        inside_image = 0 <= u < width and 0 <= v < height
        pixel_u = int(np.clip(round(u), 0, width - 1))
        pixel_v = int(np.clip(round(v), 0, height - 1))
        inside_mask = inside_image and bool(mask[pixel_v, pixel_u])
        disagreement = float(np.linalg.norm(projected - mask_uv))
        rescan_needed = (
            not inside_mask
            or disagreement > float(
                self.get_parameter('projection_disagreement_px').value)
        )
        if rescan_needed:
            self.request_refresh('landmark_mask_disagreement')

        point = PointStamped()
        point.header.stamp = depth_msg.header.stamp
        point.header.frame_id = self.get_parameter('base_frame').value
        point.point.x, point.point.y, point.point.z = [
            float(value) for value in self.landmark]
        self.landmark_pub.publish(point)

        projection = Detection2D()
        projection.header = depth_msg.header
        projection.u = u
        projection.v = v
        projection.confidence = float(max(0.0, 1.0 - disagreement / 100.0))
        projection.valid = bool(inside_image)
        self.projection_pub.publish(projection)

        angle = direction_angle_degrees(view_direction, self.last_view_direction)
        self.publish_status(
            'RESCAN_NEEDED' if rescan_needed else 'LOCKED', valid=True,
            projected_u=u, projected_v=v, projection_inside_mask=inside_mask,
            projection_error_px=disagreement, rescan_needed=rescan_needed,
            view_angle_since_update_deg=angle, update_count=self.update_count,
            measurement_error_m=float(np.linalg.norm(measurement - self.landmark)),
            depth_ratio=ratio, depth_pixels=count)

    def request_refresh(self, reason):
        if not bool(self.get_parameter('request_refresh_on_new_view').value):
            return
        now = time.monotonic()
        cooldown = float(self.get_parameter('refresh_cooldown_sec').value)
        if now - self.last_refresh_time < cooldown:
            return
        msg = String()
        msg.data = json.dumps({
            'request_id': 'landmark_%d' % int(time.time() * 1000),
            'reason': reason,
            'tracking': {'tracking_confidence': 0.0},
            'dry_run': True,
            'real_arm_motion': False,
        })
        self.refresh_pub.publish(msg)
        self.last_refresh_time = now

    def lookup(self, target, source, stamp):
        return self.tf_buffer.lookup_transform(
            target, source, rclpy.time.Time.from_msg(stamp),
            timeout=Duration(seconds=float(
                self.get_parameter('transform_timeout_sec').value)))

    @staticmethod
    def apply_transform(point, transform):
        tf = transform.transform
        return transform_points(
            [point],
            (tf.translation.x, tf.translation.y, tf.translation.z),
            (tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w))[0]

    def publish_invalid(self, source_msg, reason):
        projection = Detection2D()
        projection.header = source_msg.header
        projection.valid = False
        self.projection_pub.publish(projection)
        self.publish_status('INVALID', valid=False, reason=reason)

    def publish_status(self, state, **values):
        payload = {'state': state, 'dry_run': True, 'real_arm_motion': False}
        payload.update(values)
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TargetLandmarkNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
