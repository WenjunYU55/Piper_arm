#!/usr/bin/env python3
"""Publish timestamp-correct, read-only 3D geometry for each SAM2 obstacle."""

import json
import time
from collections import OrderedDict

import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from piper_mobile_manipulation.msg import ObstacleInstance3D, ObstacleInstance3DArray
from piper_mobile_manipulation.obstacle_geometry import (
    MOVABLE, UNSAFE, aabb_corners, effective_classification,
    normalize_label, obstacle_records, project_instance, transform_points,
)


class ObstacleInstance3DNode(Node):
    def __init__(self):
        super().__init__('obstacle_instance_3d_node')
        defaults = {
            'object_ids_topic': '/piper/sam2_object_ids',
            'metadata_topic': '/piper/sam2_tracking_status',
            'depth_topic': '/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/color/camera_info',
            'output_topic': '/piper/obstacle_instances_3d',
            'base_frame': 'base_link', 'depth_min_m': 0.25, 'depth_max_m': 1.20,
            'min_valid_depth_pixels': 20, 'min_valid_depth_ratio': 0.40,
            'mask_erode_px': 2, 'bounds_low_percentile': 2.0,
            'bounds_high_percentile': 98.0, 'sync_slop_sec': 0.08,
            'sync_queue_size': 20, 'metadata_wait_sec': 0.25,
            'max_source_age_sec': 10.0, 'transform_timeout_sec': 0.20,
            'max_transform_age_sec': 0.20,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.declare_parameter('movable_whitelist', ['pen'])
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.metadata = OrderedDict()
        self.pending = OrderedDict()
        self.publisher = self.create_publisher(
            ObstacleInstance3DArray, self.get_parameter('output_topic').value, 10)
        self.create_subscription(
            String, self.get_parameter('metadata_topic').value, self.metadata_cb, 10)
        ids_sub = Subscriber(self, Image, self.get_parameter('object_ids_topic').value,
                             qos_profile=qos_profile_sensor_data)
        depth_sub = Subscriber(self, Image, self.get_parameter('depth_topic').value,
                               qos_profile=qos_profile_sensor_data)
        info_sub = Subscriber(self, CameraInfo, self.get_parameter('camera_info_topic').value,
                              qos_profile=qos_profile_sensor_data)
        self.sync = ApproximateTimeSynchronizer(
            [ids_sub, depth_sub, info_sub],
            int(self.get_parameter('sync_queue_size').value),
            float(self.get_parameter('sync_slop_sec').value))
        self.sync.registerCallback(self.synced_cb)
        self.create_timer(0.05, self.flush_pending)
        self.get_logger().warn(
            'Obstacle instance geometry is read-only; it cannot command arm motion.')

    @staticmethod
    def stamp_key(stamp):
        return (int(stamp.sec), int(stamp.nanosec))

    def metadata_cb(self, msg):
        try:
            payload = json.loads(msg.data)
            stamp = payload.get('image_stamp', {})
            key = (int(stamp['sec']), int(stamp['nanosec']))
            if 'objects' not in payload:
                return
            self.metadata[key] = payload
            self.metadata.move_to_end(key)
            while len(self.metadata) > 100:
                self.metadata.popitem(last=False)
            pending = self.pending.pop(key, None)
            if pending:
                self.process(*pending[1], payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return

    def synced_cb(self, ids_msg, depth_msg, info_msg):
        key = self.stamp_key(ids_msg.header.stamp)
        metadata = self.metadata.get(key)
        if metadata is not None:
            self.process(ids_msg, depth_msg, info_msg, metadata)
            return
        self.pending[key] = (time.monotonic(), (ids_msg, depth_msg, info_msg))
        while len(self.pending) > 30:
            _, (_, messages) = self.pending.popitem(last=False)
            self.process(*messages, None)

    def flush_pending(self):
        cutoff = time.monotonic() - float(self.get_parameter('metadata_wait_sec').value)
        expired = [key for key, value in self.pending.items() if value[0] < cutoff]
        for key in expired:
            _, messages = self.pending.pop(key)
            self.process(*messages, None)

    def config(self):
        names = ('depth_min_m', 'depth_max_m', 'min_valid_depth_pixels',
                 'min_valid_depth_ratio', 'mask_erode_px', 'bounds_low_percentile',
                 'bounds_high_percentile')
        return {name: self.get_parameter(name).value for name in names}

    def process(self, ids_msg, depth_msg, info_msg, metadata):
        out = ObstacleInstance3DArray()
        out.header = ids_msg.header
        try:
            ids = np.asarray(self.bridge.imgmsg_to_cv2(ids_msg, 'passthrough'))
            raw_depth = np.asarray(self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough'))
            depth = raw_depth.astype(np.float64)
            if '16U' in depth_msg.encoding or depth_msg.encoding in ('mono16', '16UC1'):
                depth *= 0.001
            if ids.shape != depth.shape:
                raise ValueError('id_depth_shape_mismatch')
        except Exception as exc:
            out.scene_blocked = True
            out.blocking_reason = 'input_conversion_failed:%s' % exc
            self.publisher.publish(out)
            return
        records = {}
        suppressed_ids = set()
        if metadata:
            records, suppressed_ids = obstacle_records(metadata.get('objects', []))
        ids_in_image = {
            int(value) for value in np.unique(ids)
            if int(value) > 1 and int(value) not in suppressed_ids
        }
        obstacle_ids = sorted(ids_in_image | set(records))
        source_seconds = ids_msg.header.stamp.sec + ids_msg.header.stamp.nanosec * 1e-9
        now_seconds = self.get_clock().now().nanoseconds * 1e-9
        max_source_age = float(self.get_parameter('max_source_age_sec').value)
        source_stale = now_seconds - source_seconds > max_source_age
        transform = None
        transform_error = None
        try:
            transform = self.tf_buffer.lookup_transform(
                self.get_parameter('base_frame').value, ids_msg.header.frame_id,
                rclpy.time.Time.from_msg(ids_msg.header.stamp),
                timeout=Duration(seconds=float(self.get_parameter('transform_timeout_sec').value)))
        except TransformException as exc:
            transform_error = 'transform_unavailable:%s' % exc
        transform_age = 0.0
        if transform is not None:
            tf_stamp = transform.header.stamp.sec + transform.header.stamp.nanosec * 1e-9
            # Static transforms conventionally carry a zero stamp and do not become stale.
            transform_age = 0.0 if tf_stamp == 0.0 else abs(source_seconds - tf_stamp)
            if transform_age > float(self.get_parameter('max_transform_age_sec').value):
                transform_error = 'stale_transform'
        whitelist = list(self.get_parameter('movable_whitelist').value)
        blockers = []
        for object_id in obstacle_ids:
            record = records.get(object_id)
            instance = ObstacleInstance3D()
            instance.header = ids_msg.header
            instance.object_id = object_id
            instance.camera_frame = ids_msg.header.frame_id
            instance.base_frame = self.get_parameter('base_frame').value
            instance.transform_age_sec = -1.0
            if record is None:
                instance.semantic_label = 'unknown'
                instance.classification = UNSAFE
                self.invalidate(instance, 'missing_object_metadata')
            else:
                instance.semantic_label = normalize_label(record.get('label', 'unknown'))
                instance.confidence = float(record.get('confidence', 0.0))
                instance.classification = effective_classification(
                    instance.semantic_label, bool(record.get('unsafe', True)), whitelist)
                reason = None
                if source_stale:
                    reason = 'stale_source_data'
                elif transform_error:
                    reason = transform_error
                try:
                    if reason:
                        raise ValueError(reason)
                    centroid, lower, upper, ratio, count = project_instance(
                        ids == object_id, depth, info_msg.k, self.config())
                    instance.valid_depth_ratio = ratio
                    instance.valid_depth_pixels = count
                    self.set_point(instance.camera_centroid, centroid)
                    self.set_point(instance.camera_bounds_min, lower)
                    self.set_point(instance.camera_bounds_max, upper)
                    tf = transform.transform
                    translation = (tf.translation.x, tf.translation.y, tf.translation.z)
                    quaternion = (tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w)
                    base_centroid = transform_points([centroid], translation, quaternion)[0]
                    base_corners = transform_points(
                        aabb_corners(lower, upper), translation, quaternion)
                    self.set_point(instance.base_centroid, base_centroid)
                    self.set_point(instance.base_bounds_min, np.min(base_corners, axis=0))
                    self.set_point(instance.base_bounds_max, np.max(base_corners, axis=0))
                    instance.transform_age_sec = float(transform_age)
                    instance.valid = True
                    instance.validity_reason = 'ok'
                except ValueError as exc:
                    self.invalidate(instance, str(exc))
            if not instance.valid or instance.classification != MOVABLE:
                reason = (instance.validity_reason if not instance.valid
                          else instance.semantic_label)
                blockers.append('%d:%s' % (object_id, reason))
            out.instances.append(instance)
        out.scene_blocked = bool(blockers)
        out.blocking_reason = ';'.join(blockers) if blockers else 'clear'
        self.publisher.publish(out)

    @staticmethod
    def invalidate(instance, reason):
        instance.valid = False
        instance.validity_reason = reason

    @staticmethod
    def set_point(message, values):
        message.x, message.y, message.z = [float(value) for value in values]


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleInstance3DNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
