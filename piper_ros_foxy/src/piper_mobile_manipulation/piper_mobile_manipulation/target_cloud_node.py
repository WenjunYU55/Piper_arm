#!/usr/bin/env python3
"""Build a bounded RGB point cloud from the live SAM2 target mask and L515 depth."""

import json
import math
import os
import struct
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import String


class TargetCloudNode(Node):
    def __init__(self):
        super().__init__('target_cloud_node')
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('mask_topic', '/piper/sam2_target_mask')
        self.declare_parameter('refined_mask_topic', '/piper/heavy_target_mask')
        self.declare_parameter('heavy_request_topic', '/piper/heavy_refresh_request')
        self.declare_parameter('cloud_topic', '/piper/target_cloud')
        self.declare_parameter('status_topic', '/piper/target_cloud_status')
        self.declare_parameter('request_topic', '/piper/target_cloud_request')
        self.declare_parameter('target_frame', 'camera_color_optical_frame')
        self.declare_parameter('require_transform', False)
        self.declare_parameter('depth_min_m', 0.20)
        self.declare_parameter('depth_max_m', 1.20)
        self.declare_parameter('mask_max_age_sec', 0.30)
        self.declare_parameter('refined_match_tolerance_sec', 0.08)
        self.declare_parameter('frame_cache_size', 180)
        self.declare_parameter('mask_erode_px', 1)
        self.declare_parameter('accumulate_live_masks', True)
        self.declare_parameter('pixel_stride', 2)
        self.declare_parameter('voxel_size_m', 0.004)
        self.declare_parameter('max_voxels', 250000)
        self.declare_parameter('publish_period_sec', 0.25)
        self.declare_parameter('output_dir', '/home/prl/Piper_arm/datasets/target_clouds')

        self.bridge = CvBridge()
        self.latest_mask = None
        self.latest_mask_stamp = None
        self.voxels = {}
        self.cloud_frame = ''
        self.last_header = None
        self.frame_cache = deque(maxlen=max(10, int(self.get_parameter('frame_cache_size').value)))
        self.awaiting_refined_capture = False
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cloud_pub = self.create_publisher(
            PointCloud2, self.get_parameter('cloud_topic').value, qos_profile_sensor_data
        )
        self.status_pub = self.create_publisher(String, self.get_parameter('status_topic').value, 10)
        self.heavy_request_pub = self.create_publisher(
            String, self.get_parameter('heavy_request_topic').value, 10
        )
        self.create_subscription(
            Image, self.get_parameter('mask_topic').value, self.mask_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, self.get_parameter('refined_mask_topic').value,
            self.refined_mask_cb, qos_profile_sensor_data
        )
        self.create_subscription(String, self.get_parameter('request_topic').value, self.request_cb, 10)
        color_sub = Subscriber(
            self, Image, self.get_parameter('color_topic').value, qos_profile=qos_profile_sensor_data
        )
        depth_sub = Subscriber(
            self, Image, self.get_parameter('depth_topic').value, qos_profile=qos_profile_sensor_data
        )
        info_sub = Subscriber(
            self, CameraInfo, self.get_parameter('camera_info_topic').value, qos_profile=qos_profile_sensor_data
        )
        self.sync = ApproximateTimeSynchronizer([color_sub, depth_sub, info_sub], 10, 0.08)
        self.sync.registerCallback(self.frame_cb)
        self.create_timer(float(self.get_parameter('publish_period_sec').value), self.publish_cloud)

    def mask_cb(self, msg):
        try:
            self.latest_mask = np.asarray(
                self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            ).copy()
            self.latest_mask_stamp = msg.header.stamp
        except Exception as exc:
            self.publish_status('mask_error', error=str(exc))

    def frame_cb(self, color_msg, depth_msg, camera_info):
        self.frame_cache.append((
            self.stamp_seconds(depth_msg.header.stamp), color_msg, depth_msg, camera_info
        ))
        if not bool(self.get_parameter('accumulate_live_masks').value):
            return
        if self.latest_mask is None or self.latest_mask_stamp is None:
            return
        age = abs(self.stamp_seconds(depth_msg.header.stamp) - self.stamp_seconds(self.latest_mask_stamp))
        if age > float(self.get_parameter('mask_max_age_sec').value):
            return
        self.accumulate_frame(
            color_msg, depth_msg, camera_info, self.latest_mask, source='live_tracking'
        )

    def refined_mask_cb(self, msg):
        if not self.awaiting_refined_capture:
            return
        try:
            mask = np.asarray(
                self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            ).copy()
        except Exception as exc:
            self.publish_status('refined_mask_error', error=str(exc))
            return
        if not self.frame_cache:
            self.publish_status('refined_capture_rejected', error='RGB-D frame cache is empty')
            return
        stamp = self.stamp_seconds(msg.header.stamp)
        match = min(self.frame_cache, key=lambda item: abs(item[0] - stamp))
        delta = abs(match[0] - stamp)
        if delta > float(self.get_parameter('refined_match_tolerance_sec').value):
            self.publish_status('refined_capture_rejected', error='matching RGB-D frame expired', delta_sec=delta)
            return
        self.awaiting_refined_capture = False
        self.accumulate_frame(match[1], match[2], match[3], mask, source='full_resolution_refinement')

    def accumulate_frame(self, color_msg, depth_msg, camera_info, mask, source):
        try:
            color = np.asarray(self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8'))
            depth = np.asarray(self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough'))
        except Exception as exc:
            self.publish_status('conversion_error', error=str(exc))
            return
        if mask.shape != depth.shape[:2] or color.shape[:2] != depth.shape[:2]:
            self.publish_status('shape_mismatch')
            return

        erode_px = max(0, int(self.get_parameter('mask_erode_px').value))
        selected_mask = mask > 0
        if erode_px:
            kernel_size = erode_px * 2 + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            selected_mask = cv2.erode(selected_mask.astype(np.uint8), kernel, iterations=1) > 0

        depth_m = depth.astype(np.float32)
        if depth_msg.encoding in ('16UC1', 'mono16') or np.issubdtype(depth.dtype, np.integer):
            depth_m *= 0.001
        stride = max(1, int(self.get_parameter('pixel_stride').value))
        rows, cols = np.indices(depth.shape[:2])
        selected = (
            selected_mask
            & np.isfinite(depth_m)
            & (depth_m >= float(self.get_parameter('depth_min_m').value))
            & (depth_m <= float(self.get_parameter('depth_max_m').value))
            & ((rows % stride) == 0)
            & ((cols % stride) == 0)
        )
        v, u = np.nonzero(selected)
        if not u.size:
            self.publish_status('no_valid_points')
            return
        z = depth_m[v, u]
        fx, fy = float(camera_info.k[0]), float(camera_info.k[4])
        cx, cy = float(camera_info.k[2]), float(camera_info.k[5])
        points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z)).astype(np.float32)
        colors = color[v, u][:, ::-1].astype(np.uint8)
        points, frame = self.transform_points(points, depth_msg.header.frame_id, depth_msg.header.stamp)
        if points is None:
            return
        if self.cloud_frame and frame != self.cloud_frame:
            self.voxels.clear()
            self.publish_status('frame_changed_cloud_cleared', previous_frame=self.cloud_frame, frame=frame)
        self.add_voxels(points, colors)
        self.cloud_frame = frame
        self.last_header = depth_msg.header
        self.publish_status(
            'accumulating', frame=frame, frame_points=int(u.size),
            voxel_count=len(self.voxels), mask_source=source
        )

    def transform_points(self, points, source_frame, stamp):
        target_frame = str(self.get_parameter('target_frame').value)
        if not target_frame or target_frame == source_frame:
            return points, source_frame
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time.from_msg(stamp))
        except Exception as exc:
            if bool(self.get_parameter('require_transform').value):
                self.publish_status('transform_unavailable', error=str(exc))
                return None, ''
            return points, source_frame
        rotation = transform.transform.rotation
        matrix = self.quaternion_matrix(rotation.x, rotation.y, rotation.z, rotation.w)
        translation = transform.transform.translation
        transformed = points.dot(matrix.T)
        transformed += np.array([translation.x, translation.y, translation.z], dtype=np.float32)
        return transformed, target_frame

    def add_voxels(self, points, colors):
        voxel_size = max(1e-4, float(self.get_parameter('voxel_size_m').value))
        limit = max(1, int(self.get_parameter('max_voxels').value))
        for point, color in zip(points, colors):
            key = tuple(np.floor(point / voxel_size).astype(np.int64))
            self.voxels[key] = (point.copy(), color.copy())
            if len(self.voxels) >= limit:
                break

    def publish_cloud(self):
        if not self.voxels or self.last_header is None:
            return
        entries = list(self.voxels.values())
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.cloud_frame
        msg.height = 1
        msg.width = len(entries)
        msg.is_bigendian = False
        msg.is_dense = True
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        data = bytearray(msg.row_step)
        for index, (point, color) in enumerate(entries):
            rgb = (int(color[0]) << 16) | (int(color[1]) << 8) | int(color[2])
            struct.pack_into('<fffI', data, index * msg.point_step, float(point[0]), float(point[1]), float(point[2]), rgb)
        msg.data = bytes(data)
        self.cloud_pub.publish(msg)

    def request_cb(self, msg):
        command = msg.data.strip().lower()
        if command == 'clear':
            self.voxels.clear()
            self.publish_status('cleared')
        elif command in ('save', 'snapshot'):
            self.save_ply()
        elif command in ('capture', 'refine'):
            self.request_refined_capture()

    def request_refined_capture(self):
        if self.awaiting_refined_capture:
            self.publish_status('refined_capture_already_pending')
            return
        request = String()
        request.data = json.dumps({
            'request_id': 'cloud_capture_%d' % int(time.time() * 1000),
            'reason': 'full_resolution_cloud_capture',
            'tracking': {'tracking_confidence': 0.0},
            'dry_run': True,
            'real_arm_motion': False,
        })
        self.heavy_request_pub.publish(request)
        self.awaiting_refined_capture = True
        self.publish_status('full_resolution_refinement_requested')

    def save_ply(self):
        if not self.voxels:
            self.publish_status('save_rejected', error='cloud is empty')
            return
        output_dir = os.path.expanduser(str(self.get_parameter('output_dir').value))
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'target_%s.ply' % datetime.now().strftime('%Y%m%d_%H%M%S'))
        entries = list(self.voxels.values())
        with open(path, 'w', encoding='ascii') as stream:
            stream.write('ply\nformat ascii 1.0\nelement vertex %d\n' % len(entries))
            stream.write('property float x\nproperty float y\nproperty float z\n')
            stream.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
            for point, color in entries:
                stream.write('%.6f %.6f %.6f %d %d %d\n' % (
                    point[0], point[1], point[2], color[0], color[1], color[2]
                ))
        self.publish_status('saved', path=path, voxel_count=len(entries), frame=self.cloud_frame)

    def publish_status(self, state, **values):
        payload = {'state': state, 'voxel_count': len(self.voxels), 'dry_run': True}
        payload.update(values)
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self.status_pub.publish(msg)

    @staticmethod
    def stamp_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def quaternion_matrix(x, y, z, w):
        norm = x * x + y * y + z * z + w * w
        if norm < 1e-12:
            return np.eye(3, dtype=np.float32)
        scale = 2.0 / norm
        return np.array([
            [1 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)],
            [scale * (x * y + z * w), 1 - scale * (x * x + z * z), scale * (y * z - x * w)],
            [scale * (x * z - y * w), scale * (y * z + x * w), 1 - scale * (x * x + y * y)],
        ], dtype=np.float32)


def main(args=None):
    rclpy.init(args=args)
    node = TargetCloudNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
