#!/usr/bin/env python3
"""Publish timestamp-correct, read-only 3D geometry for each SAM2 obstacle."""

import json
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from piper_mobile_manipulation.msg import ObstacleInstance3D, ObstacleInstance3DArray
from piper_mobile_manipulation.perception.obstacle_geometry import (
    BLOCKED, MOVABLE, aabb_corners, effective_classification,
    normalize_label, obstacle_records, project_instance,
    target_occlusion_evidence, transform_points,
)


def tf_listener_recovery_due(
        requested, failure_started_at, last_reset_at, now, stall_sec,
        retry_sec):
    """Return true only for a continuous TF outage past both time bounds."""
    if not bool(requested) or failure_started_at is None:
        return False
    current = float(now)
    return (
        current - float(failure_started_at) >= max(0.5, float(stall_sec))
        and current - float(last_reset_at) >= max(0.5, float(retry_sec))
    )


class ObstacleInstance3DNode(Node):
    def __init__(self):
        super().__init__('obstacle_instance_3d_node')
        defaults = {
            'object_ids_topic': '/piper/sam2_object_ids',
            'metadata_topic': '/piper/sam2_tracking_status',
            'heavy_status_topic': '/piper/heavy_refresh_status',
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
            'transform_listener_retry_sec': 2.0,
            'transform_listener_stall_sec': 3.0,
            'heavy_spool_dir': '/tmp/piper_heavy_refresh',
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.declare_parameter('movable_whitelist', ['pen', 'marker', 'stick'])
        self.observation_counts = {}
        self.heavy_tracks = {}
        self.track_generation = uuid.uuid4().hex[:12]
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_reset_requested = False
        self.tf_failure_started_at = None
        self.last_tf_listener_reset_at = time.monotonic()
        self.metadata = OrderedDict()
        self.pending = OrderedDict()
        self.publisher = self.create_publisher(
            ObstacleInstance3DArray, self.get_parameter('output_topic').value, 10)
        self.create_subscription(
            String, self.get_parameter('metadata_topic').value, self.metadata_cb, 10)
        self.create_subscription(
            String, self.get_parameter('heavy_status_topic').value,
            self.heavy_status_cb, 10)
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

    def heavy_status_cb(self, msg):
        """Project one request-correlated heavy result with its archived depth."""
        try:
            payload = json.loads(msg.data)
            state = str(payload.get('state', '')).lower()
            reason = str(payload.get('reason', '')).strip()
            exact_empty_rough_look = (
                state == 'worker_result_rejected'
                and payload.get('worker_status') == 'target_mask_missing'
                and reason == 'rough_acquisition_viewpoint'
                and int(payload.get('obstacle_count', -1)) == 0
                and int(payload.get('unsafe_obstacle_count', -1)) == 0
            )
            if state != 'published' and not exact_empty_rough_look:
                return
            stamp = payload.get('image_stamp', {})
            sec = int(stamp['sec'])
            nanosec = int(stamp['nanosec'])
            if sec < 0 or nanosec < 0 or nanosec >= 1_000_000_000:
                return
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        out = ObstacleInstance3DArray()
        out.header.stamp.sec = sec
        out.header.stamp.nanosec = nanosec
        out.header.frame_id = str(self.get_parameter('base_frame').value)
        tracked = payload.get('tracked_objects', [])
        records = [
            item for item in tracked
            if isinstance(item, dict)
            and str(item.get('role', '')).lower() == 'obstacle'
        ] if isinstance(tracked, list) else []
        labels = payload.get('obstacle_labels', [])
        confidences = payload.get('obstacle_confidences', [])
        if not records:
            records = [
                {
                    'object_id': index + 2,
                    'label': label,
                    'confidence': (
                        confidences[index]
                        if isinstance(confidences, list)
                        and index < len(confidences) else 0.0),
                }
                for index, label in enumerate(
                    labels if isinstance(labels, list) else [])
            ]
        declared_count = max(0, int(payload.get('obstacle_count', 0)))
        while len(records) < declared_count:
            records.append({
                'object_id': len(records) + 2,
                'label': 'unknown semantic obstacle',
                'confidence': 0.0,
            })
        if not records:
            out.scene_blocked = False
            out.blocking_reason = 'clear:completed_semantic_result:' + (
                reason or 'unspecified')
            self.publisher.publish(out)
            return
        try:
            job_id = str(payload.get('job_id', ''))
            request_dir, _ = ObstacleInstance3DNode.heavy_job_paths(
                self,
                job_id)
            with (request_dir / 'request.yaml').open(
                    'r', encoding='utf-8') as stream:
                manifest = yaml.safe_load(stream) or {}
            depth = np.load(
                str(request_dir / 'depth.npy'), allow_pickle=False).astype(
                    np.float64)
            encoding = str(manifest.get('depth_encoding', ''))
            if '16U' in encoding or encoding in ('mono16', '16UC1'):
                depth *= 0.001
            camera_matrix = [float(value) for value in manifest['camera_matrix']]
            if len(camera_matrix) != 9:
                raise ValueError('archived camera matrix is invalid')
            target_mask = ObstacleInstance3DNode.heavy_response_mask(
                self, job_id, 'target_mask.png')
            if target_mask is None or target_mask.shape != depth.shape:
                raise ValueError('archived target mask/depth mismatch')
            transform = self.tf_buffer.lookup_transform(
                out.header.frame_id, str(manifest.get('frame_id', '')),
                rclpy.time.Time.from_msg(out.header.stamp),
                timeout=Duration(seconds=float(
                    self.get_parameter('transform_timeout_sec').value)))
            tf = transform.transform
            translation = (tf.translation.x, tf.translation.y, tf.translation.z)
            quaternion = (tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w)
            whitelist = list(self.get_parameter('movable_whitelist').value)
            blockers = []
            for index, record in enumerate(records):
                instance = ObstacleInstance3D()
                instance.header = out.header
                instance.object_id = int(record.get('object_id', index + 2))
                instance.semantic_label = normalize_label(
                    record.get('label', 'unknown semantic obstacle'))
                instance.confidence = float(record.get('confidence', 0.0))
                instance.classification = effective_classification(
                    instance.semantic_label,
                    bool(record.get('unsafe', False)), whitelist)
                instance.camera_frame = str(manifest.get('frame_id', ''))
                instance.base_frame = out.header.frame_id
                instance.transform_age_sec = 0.0
                obstacle_mask = ObstacleInstance3DNode.heavy_response_mask(
                    self, job_id, str(record.get('mask_file', '')))
                if obstacle_mask is None or obstacle_mask.shape != depth.shape:
                    raise ValueError(
                        'archived obstacle mask/depth mismatch for object %d'
                        % instance.object_id)
                centroid, lower, upper, ratio, count = project_instance(
                    obstacle_mask > 0, depth, camera_matrix, self.config())
                instance.valid_depth_ratio = ratio
                instance.valid_depth_pixels = count
                self.set_point(instance.camera_centroid, centroid)
                self.set_point(instance.camera_bounds_min, lower)
                self.set_point(instance.camera_bounds_max, upper)
                base_centroid = transform_points(
                    [centroid], translation, quaternion)[0]
                base_corners = transform_points(
                    aabb_corners(lower, upper), translation, quaternion)
                base_lower = np.min(base_corners, axis=0)
                base_upper = np.max(base_corners, axis=0)
                self.set_point(instance.base_centroid, base_centroid)
                self.set_point(instance.base_bounds_min, base_lower)
                self.set_point(instance.base_bounds_max, base_upper)
                instance.position_uncertainty_m = float(
                    0.005 + 0.02 * (1.0 - ratio))
                overlap, closer = target_occlusion_evidence(
                    target_mask > 0, obstacle_mask > 0, depth)
                instance.target_overlap_ratio = overlap
                instance.closer_depth_occlusion_ratio = closer
                instance.confirmed_in_probe_view = True
                track_id, observations = self.match_heavy_track(
                    instance.semantic_label, base_centroid)
                instance.track_id = track_id
                instance.observation_count = observations
                self.set_footprint(instance.tabletop_footprint, base_lower, base_upper)
                instance.valid = True
                instance.validity_reason = 'ok'
                if not instance.valid or instance.classification != MOVABLE:
                    blockers.append('%d:%s' % (
                        instance.object_id, instance.semantic_label))
                out.instances.append(instance)
            out.scene_blocked = bool(blockers)
            out.blocking_reason = ';'.join(blockers) if blockers else 'clear'
        except (KeyError, OSError, TransformException, ValueError) as exc:
            out.instances = []
            out.scene_blocked = True
            out.blocking_reason = 'semantic_probe_projection_failed:%s' % exc
        self.publisher.publish(out)

    def heavy_job_paths(self, job_id):
        if not job_id:
            raise ValueError('heavy job identity is missing')
        spool = Path(str(self.get_parameter('heavy_spool_dir').value))
        request_dir = spool / 'archive' / job_id
        response_dir = next((
            path for path in (
                spool / 'consumed' / job_id,
                spool / 'responses' / job_id,
            ) if path.is_dir()), None)
        if not request_dir.is_dir() or response_dir is None:
            raise ValueError('archived heavy RGB-D result is unavailable')
        return request_dir, response_dir

    def heavy_response_mask(self, job_id, relative_path):
        """Read a response mask across the atomic responses-to-consumed move."""
        if not job_id:
            raise ValueError('heavy job identity is missing')
        relative = Path(str(relative_path))
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('heavy response mask path is invalid')
        spool = Path(str(self.get_parameter('heavy_spool_dir').value))
        # heavy_refresh_bridge publishes the correlated status and then moves
        # the complete directory from responses/ to consumed/.  A subscriber
        # can run in that narrow interval, so resolve both immutable roots for
        # every read instead of retaining the pre-rename directory path.
        for _attempt in range(2):
            for parent in ('consumed', 'responses'):
                root = (spool / parent / job_id).resolve()
                candidate = (root / relative).resolve()
                if root != candidate and root not in candidate.parents:
                    raise ValueError(
                        'heavy response mask escapes its job directory')
                mask = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    return mask
        return None

    def match_heavy_track(self, label, center):
        """Associate repeated settled probes without trusting detector IDs."""
        normalized = normalize_label(label)
        center = np.asarray(center, dtype=np.float64)
        candidates = self.heavy_tracks.setdefault(normalized, [])
        if candidates:
            selected = min(
                candidates,
                key=lambda item: float(np.linalg.norm(center - item['center'])))
            if float(np.linalg.norm(center - selected['center'])) <= 0.050:
                selected['center'] = center
                selected['count'] += 1
                return selected['track_id'], int(selected['count'])
        record = {
            'track_id': 'heavy-%s-%s-%d' % (
                self.track_generation,
                normalized.replace(' ', '-') or 'unknown',
                len(candidates) + 1),
            'center': center,
            'count': 1,
        }
        candidates.append(record)
        return record['track_id'], 1

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
        self.maybe_recreate_tf_listener()
        cutoff = time.monotonic() - float(self.get_parameter('metadata_wait_sec').value)
        expired = [key for key, value in self.pending.items() if value[0] < cutoff]
        for key in expired:
            _, messages = self.pending.pop(key)
            self.process(*messages, None)

    def maybe_recreate_tf_listener(self):
        now = time.monotonic()
        if not tf_listener_recovery_due(
                self.tf_reset_requested,
                self.tf_failure_started_at,
                self.last_tf_listener_reset_at,
                now,
                self.get_parameter('transform_listener_stall_sec').value,
                self.get_parameter('transform_listener_retry_sec').value):
            return
        self.tf_listener.unregister()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_reset_requested = False
        self.tf_failure_started_at = None
        self.last_tf_listener_reset_at = now
        self.get_logger().warn(
            'Recreated stalled TF listener; obstacle geometry remains blocked '
            'until a fresh base-to-camera transform arrives.')

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
        # A target-only SAM2 frame is already a complete empty obstacle
        # observation.  Do not make its publication wait on an exact-time TF
        # lookup that is only needed to project obstacle geometry.  During arm
        # motion that lookup may legitimately wait for the matching joint-state
        # transform; serialising every empty frame behind that wait can starve
        # the executor's otherwise valid clear-scene heartbeat.
        if not obstacle_ids:
            out.scene_blocked = False
            out.blocking_reason = 'clear:live_target_only_frame'
            self.publisher.publish(out)
            return
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
            self.tf_reset_requested = True
            if self.tf_failure_started_at is None:
                self.tf_failure_started_at = time.monotonic()
        transform_age = 0.0
        if transform is not None:
            self.tf_reset_requested = False
            self.tf_failure_started_at = None
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
            instance.track_id = '%s-sam2-%d' % (
                self.track_generation, object_id)
            self.observation_counts[instance.track_id] = int(
                self.observation_counts.get(instance.track_id, 0)) + 1
            instance.observation_count = self.observation_counts[instance.track_id]
            instance.camera_frame = ids_msg.header.frame_id
            instance.base_frame = self.get_parameter('base_frame').value
            instance.transform_age_sec = -1.0
            if record is None:
                instance.semantic_label = 'unknown'
                instance.classification = BLOCKED
                self.invalidate(instance, 'missing_object_metadata')
            else:
                instance.semantic_label = normalize_label(record.get('label', 'unknown'))
                instance.confidence = float(record.get('confidence', 0.0))
                instance.classification = effective_classification(
                    instance.semantic_label, bool(record.get('unsafe', False)), whitelist)
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
                    instance.position_uncertainty_m = float(min(
                        0.10,
                        0.005 + 0.05 * transform_age
                        + 0.02 * (1.0 - ratio)))
                    self.set_footprint(
                        instance.tabletop_footprint,
                        np.min(base_corners, axis=0),
                        np.max(base_corners, axis=0))
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

    @staticmethod
    def set_footprint(message, lower, upper):
        from geometry_msgs.msg import Point32
        message.points = []
        for x, y in (
                (lower[0], lower[1]), (upper[0], lower[1]),
                (upper[0], upper[1]), (lower[0], upper[1])):
            point = Point32()
            point.x, point.y, point.z = float(x), float(y), float(lower[2])
            message.points.append(point)


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
