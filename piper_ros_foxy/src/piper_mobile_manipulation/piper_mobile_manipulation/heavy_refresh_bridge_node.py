#!/usr/bin/env python3
"""Foxy-side image spool and heavy-mask publisher; contains no AI dependencies."""

import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header, String

from piper_mobile_manipulation.heavy_refresh_contract import (
    image_satisfies_request,
    request_minimum_stamp_ns,
    ros_stamp_to_dict,
)


class HeavyRefreshBridgeNode(Node):
    def __init__(self):
        super().__init__('heavy_refresh_bridge_node')
        self.declare_parameter('color_image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_image_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('tracked_mask_topic', '/piper/temporal_target_mask')
        self.declare_parameter('request_topic', '/piper/heavy_refresh_request')
        self.declare_parameter('output_mask_topic', '/piper/heavy_target_mask')
        self.declare_parameter('movable_obstacle_mask_topic', '/piper/candidate_movable_obstacle_mask')
        self.declare_parameter('unsafe_obstacle_mask_topic', '/piper/unsafe_obstacle_mask')
        self.declare_parameter('all_obstacle_mask_topic', '/piper/heavy_obstacle_mask')
        self.declare_parameter('status_topic', '/piper/heavy_refresh_status')
        self.declare_parameter('spool_dir', '/tmp/piper_heavy_refresh')
        self.declare_parameter('sam2_live_spool_dir', '/tmp/piper_sam2_live')
        self.declare_parameter('seed_sam2_live', True)
        self.declare_parameter('response_poll_period_sec', 0.20)
        self.declare_parameter('rgbd_sync_slop_sec', 0.08)
        self.declare_parameter('rgbd_sync_queue_size', 20)
        self.declare_parameter('max_image_age_sec', 1.0)
        self.declare_parameter('idle_status_interval_sec', 2.0)
        self.declare_parameter('min_target_depth_valid_ratio', 0.05)
        self.declare_parameter('min_target_depth_valid_px', 25)
        self.declare_parameter('dry_run', True)
        self.declare_parameter('enable_real_arm_motion', False)

        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None
        self.latest_mask = None
        self.latest_color_msg = None
        self.latest_depth_msg = None
        self.latest_camera_info = None
        self.latest_color_time = 0.0
        self.pending_request = None
        self.spool = Path(str(self.get_parameter('spool_dir').value))
        for name in ('requests', 'responses', 'consumed'):
            (self.spool / name).mkdir(parents=True, exist_ok=True)

        self.mask_pub = self.create_publisher(
            Image, self.get_parameter('output_mask_topic').value, qos_profile_sensor_data
        )
        self.movable_obstacle_pub = self.create_publisher(
            Image, self.get_parameter('movable_obstacle_mask_topic').value, qos_profile_sensor_data
        )
        self.unsafe_obstacle_pub = self.create_publisher(
            Image, self.get_parameter('unsafe_obstacle_mask_topic').value, qos_profile_sensor_data
        )
        self.all_obstacle_pub = self.create_publisher(
            Image, self.get_parameter('all_obstacle_mask_topic').value, qos_profile_sensor_data
        )
        self.status_pub = self.create_publisher(String, self.get_parameter('status_topic').value, 10)
        color_sub = Subscriber(
            self, Image, self.get_parameter('color_image_topic').value,
            qos_profile=qos_profile_sensor_data)
        depth_sub = Subscriber(
            self, Image, self.get_parameter('depth_image_topic').value,
            qos_profile=qos_profile_sensor_data)
        info_sub = Subscriber(
            self, CameraInfo, self.get_parameter('camera_info_topic').value,
            qos_profile=qos_profile_sensor_data)
        self.rgbd_sync = ApproximateTimeSynchronizer(
            [color_sub, depth_sub, info_sub],
            int(self.get_parameter('rgbd_sync_queue_size').value),
            float(self.get_parameter('rgbd_sync_slop_sec').value))
        self.rgbd_sync.registerCallback(self.rgbd_cb)
        self.create_subscription(Image, self.get_parameter('tracked_mask_topic').value, self.mask_cb, qos_profile_sensor_data)
        self.create_subscription(String, self.get_parameter('request_topic').value, self.request_cb, 10)
        self.create_timer(float(self.get_parameter('response_poll_period_sec').value), self.poll_responses)
        self.create_timer(
            max(0.5, float(self.get_parameter('idle_status_interval_sec').value)),
            self.publish_idle_status,
        )
        self.get_logger().warn('Heavy refresh bridge is read-only; real arm motion is disabled.')

    def rgbd_cb(self, color_msg, depth_msg, info_msg):
        try:
            color = np.asarray(self.bridge.imgmsg_to_cv2(
                color_msg, desired_encoding='bgr8')).copy()
            depth = np.asarray(self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough')).copy()
            if color.shape[:2] != depth.shape[:2]:
                raise ValueError('aligned RGB-D shapes differ')
            self.latest_color = color
            self.latest_depth = depth
            self.latest_color_msg = color_msg
            self.latest_depth_msg = depth_msg
            self.latest_camera_info = info_msg
            self.latest_color_time = time.monotonic()
            self.try_enqueue_pending_request()
        except Exception as exc:
            self.get_logger().warn('Synchronized RGB-D conversion failed: %s' % exc)

    def mask_cb(self, msg):
        try:
            self.latest_mask = np.asarray(self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')).copy()
        except Exception as exc:
            self.get_logger().warn('Tracked-mask conversion failed: %s' % exc)

    def request_cb(self, msg):
        try:
            request = json.loads(msg.data)
        except (TypeError, ValueError) as exc:
            self.publish_status('request_rejected', error='invalid JSON: %s' % exc)
            return
        request_id = str(request.get('request_id', '')).strip()
        if not request_id:
            self.publish_status(
                'request_rejected', error='request_id is required')
            return
        try:
            request_minimum_stamp_ns(request)
        except ValueError as exc:
            self.publish_status(
                'request_rejected', request_id=request_id, error=str(exc))
            return
        if self.pending_request is not None:
            pending_id = str(self.pending_request.get('request_id', ''))
            if pending_id == request_id:
                return
            self.publish_status(
                'request_ignored_busy', request_id=request_id,
                reason=request.get('reason', ''),
            )
            return
        if self.latest_color is None or self.latest_color_msg is None:
            self.pending_request = request
            self.publish_status(
                'waiting_for_fresh_image', request_id=request_id,
                reason=request.get('reason', ''),
            )
            return
        age = time.monotonic() - self.latest_color_time
        if (
                age > float(self.get_parameter('max_image_age_sec').value)
                or not image_satisfies_request(
                    request, self.latest_color_msg.header.stamp)):
            self.pending_request = request
            self.publish_status(
                'waiting_for_fresh_image', request_id=request_id,
                reason=request.get('reason', ''), image_age_sec=age,
                min_image_stamp=request.get('min_image_stamp'),
            )
            return
        if self.worker_busy():
            self.pending_request = request
            self.publish_status(
                'waiting_for_worker', request_id=request_id,
                reason=request.get('reason', ''),
            )
            return
        self.enqueue_request(request)

    def try_enqueue_pending_request(self):
        """Queue the retained request once its frame and worker are ready."""
        request = self.pending_request
        if request is None or self.worker_busy():
            return False
        if self.latest_color is None or self.latest_color_msg is None:
            return False
        age = time.monotonic() - self.latest_color_time
        if age > float(self.get_parameter('max_image_age_sec').value):
            return False
        if not image_satisfies_request(
                request, self.latest_color_msg.header.stamp):
            return False
        self.pending_request = None
        self.enqueue_request(request)
        return True

    def enqueue_request(self, request):
        request_id = request.get('request_id', 'unknown')
        if self.worker_busy():
            if self.pending_request is None:
                self.pending_request = request
            self.publish_status(
                'waiting_for_worker', request_id=request_id,
                reason=request.get('reason', ''),
            )
            return
        stamp = self.latest_color_msg.header.stamp
        job_id = '%d_%09d_request_%s' % (
            int(stamp.sec), int(stamp.nanosec), self.safe_component(request_id)
        )
        final_dir = self.spool / 'requests' / job_id
        if final_dir.exists() or (self.spool / 'responses' / job_id).exists():
            return
        temporary = self.spool / 'requests' / (job_id + '.tmp')
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        try:
            if not cv2.imwrite(str(temporary / 'rgb.png'), self.latest_color):
                raise IOError('could not write rgb.png')
            if self.latest_depth is not None:
                np.save(str(temporary / 'depth.npy'), self.latest_depth)
            mask = self.latest_mask
            if mask is None or mask.shape[:2] != self.latest_color.shape[:2]:
                mask = np.zeros(self.latest_color.shape[:2], dtype=np.uint8)
            cv2.imwrite(str(temporary / 'tracked_mask.png'), mask)
            manifest = {
                'protocol_version': 1,
                'job_id': job_id,
                'request_id': request_id,
                'reason': request.get('reason', ''),
                'tracking_confidence': request.get('tracking', {}).get('tracking_confidence', 0.0),
                'image_stamp': {'sec': int(stamp.sec), 'nanosec': int(stamp.nanosec)},
                'min_image_stamp': request.get('min_image_stamp'),
                'frame_id': self.latest_color_msg.header.frame_id,
                'depth_encoding': str(self.latest_depth_msg.encoding),
                'camera_matrix': [
                    float(value) for value in self.latest_camera_info.k],
                'rgbd_stamp_delta_sec': abs(
                    (int(self.latest_color_msg.header.stamp.sec)
                     + int(self.latest_color_msg.header.stamp.nanosec) * 1e-9)
                    - (int(self.latest_depth_msg.header.stamp.sec)
                       + int(self.latest_depth_msg.header.stamp.nanosec) * 1e-9)),
                'dry_run': True,
                'real_arm_motion': False,
            }
            with (temporary / 'request.yaml').open('w', encoding='utf-8') as stream:
                yaml.safe_dump(manifest, stream, sort_keys=False)
            (temporary / 'READY').touch()
            os.replace(str(temporary), str(final_dir))
            self.publish_status(
                'queued', job_id=job_id, request_id=request_id,
                reason=manifest['reason'], image_stamp=manifest['image_stamp'],
            )
        except Exception as exc:
            shutil.rmtree(temporary, ignore_errors=True)
            self.publish_status(
                'request_failed',
                job_id=job_id,
                request_id=request_id,
                image_stamp=ros_stamp_to_dict(stamp),
                error='%s: %s' % (type(exc).__name__, exc),
            )

    def worker_busy(self):
        for name in ('requests', 'processing', 'responses'):
            directory = self.spool / name
            if directory.is_dir() and any(directory.iterdir()):
                return True
        return False

    def publish_idle_status(self):
        """Confirm quiescence so consumers can recover a lost terminal event."""
        if self.pending_request is None and not self.worker_busy():
            self.publish_status('idle')

    def poll_responses(self):
        response_root = self.spool / 'responses'
        for response in sorted(response_root.iterdir()):
            if not response.is_dir() or response.name.endswith('.tmp') or not (response / 'READY').is_file():
                continue
            try:
                with (response / 'result.yaml').open('r', encoding='utf-8') as stream:
                    result = yaml.safe_load(stream) or {}
                request_manifest = self.request_manifest(response.name)
                request_id = (
                    result.get('request_id')
                    or request_manifest.get('request_id')
                )
                reason = str(request_manifest.get('reason', ''))
                image_stamp = result.get(
                    'image_stamp', request_manifest.get('image_stamp', {}))
                mask = cv2.imread(str(response / 'target_mask.png'), cv2.IMREAD_GRAYSCALE)
                if result.get('status') != 'ok' or mask is None or not np.count_nonzero(mask):
                    self.publish_status(
                        'worker_result_rejected', job_id=response.name,
                        request_id=request_id, reason=reason,
                        worker_status=result.get('status'),
                        image_stamp=image_stamp,
                        target_confidence=result.get('target_confidence'),
                        obstacle_count=result.get('obstacle_count', 0),
                        obstacle_labels=result.get('obstacle_labels', []),
                        obstacle_confidences=result.get(
                            'obstacle_confidences', []),
                        tracked_objects=result.get('tracked_objects', []),
                        unsafe_obstacle_count=result.get('unsafe_obstacle_count', 0),
                    )
                    header = self.result_header(result)
                    self.publish_response_mask(
                        response / 'candidate_movable_obstacle_mask.png',
                        self.movable_obstacle_pub, header)
                    self.publish_response_mask(
                        response / 'unsafe_obstacle_mask.png',
                        self.unsafe_obstacle_pub, header)
                    self.publish_response_mask(
                        response / 'all_obstacle_mask.png',
                        self.all_obstacle_pub, header)
                    if (
                            bool(self.get_parameter('seed_sam2_live').value)
                            and result.get('tracked_objects')):
                        self.queue_sam2_live_seed(response, result)
                else:
                    self.publish_target_depth_diagnostic(mask, response.name, result)
                    out = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
                    stamp = result.get('image_stamp', {})
                    out.header.stamp.sec = int(stamp.get('sec', 0))
                    out.header.stamp.nanosec = int(stamp.get('nanosec', 0))
                    out.header.frame_id = 'heavy_refresh:%s' % response.name
                    self.mask_pub.publish(out)
                    self.publish_response_mask(
                        response / 'candidate_movable_obstacle_mask.png',
                        self.movable_obstacle_pub,
                        out.header,
                    )
                    self.publish_response_mask(
                        response / 'unsafe_obstacle_mask.png', self.unsafe_obstacle_pub, out.header
                    )
                    self.publish_response_mask(
                        response / 'all_obstacle_mask.png', self.all_obstacle_pub, out.header
                    )
                    self.publish_status(
                        'published',
                        job_id=response.name,
                        request_id=request_id,
                        reason=reason,
                        image_stamp=image_stamp,
                        target_confidence=result.get('target_confidence'),
                        obstacle_count=result.get('obstacle_count', 0),
                        obstacle_labels=result.get('obstacle_labels', []),
                        obstacle_confidences=result.get(
                            'obstacle_confidences', []),
                        tracked_objects=result.get('tracked_objects', []),
                        unsafe_obstacle_count=result.get('unsafe_obstacle_count', 0),
                    )
                    if bool(self.get_parameter('seed_sam2_live').value):
                        self.queue_sam2_live_seed(response, result)
                destination = self.spool / 'consumed' / response.name
                shutil.rmtree(destination, ignore_errors=True)
                os.replace(str(response), str(destination))
            except Exception as exc:
                self.get_logger().error('Failed to consume %s: %s' % (response, exc))
        self.try_enqueue_pending_request()

    def request_manifest(self, job_id):
        for path in (
            self.spool / 'archive' / job_id / 'request.yaml',
            self.spool / 'processing' / job_id / 'request.yaml',
            self.spool / 'requests' / job_id / 'request.yaml',
        ):
            if not path.is_file():
                continue
            try:
                with path.open('r', encoding='utf-8') as stream:
                    value = yaml.safe_load(stream) or {}
                    return value if isinstance(value, dict) else {}
            except (OSError, ValueError, TypeError):
                return {}
        return {}

    def request_reason(self, job_id):
        return str(self.request_manifest(job_id).get('reason', ''))

    def publish_response_mask(self, path, publisher, header):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return
        out = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        out.header = header
        publisher.publish(out)

    @staticmethod
    def result_header(result):
        header = Header()
        stamp = result.get('image_stamp', {})
        header.stamp.sec = int(stamp.get('sec', 0))
        header.stamp.nanosec = int(stamp.get('nanosec', 0))
        header.frame_id = str(result.get('frame_id', ''))
        return header

    def publish_target_depth_diagnostic(self, mask, job_id, result):
        if self.latest_depth is None:
            self.publish_status(
                'target_mask_depth_unchecked',
                job_id=job_id,
                request_id=result.get('request_id'),
                reason='no aligned depth frame received yet',
            )
            return
        if mask.shape[:2] != self.latest_depth.shape[:2]:
            self.publish_status(
                'target_mask_depth_shape_mismatch',
                job_id=job_id,
                request_id=result.get('request_id'),
                mask_shape=list(mask.shape[:2]),
                depth_shape=list(self.latest_depth.shape[:2]),
            )
            return
        mask_bool = mask > 0
        mask_px = int(np.count_nonzero(mask_bool))
        if mask_px <= 0:
            return
        depth = self.latest_depth
        if np.issubdtype(depth.dtype, np.floating):
            valid_depth = np.isfinite(depth) & (depth > 0.0)
        else:
            valid_depth = depth > 0
        valid_px = int(np.count_nonzero(valid_depth & mask_bool))
        ratio = valid_px / float(mask_px)
        min_ratio = float(self.get_parameter('min_target_depth_valid_ratio').value)
        min_px = int(self.get_parameter('min_target_depth_valid_px').value)
        if valid_px < min_px or ratio < min_ratio:
            self.publish_status(
                'target_mask_depth_warning',
                job_id=job_id,
                request_id=result.get('request_id'),
                mask_px=mask_px,
                valid_depth_px=valid_px,
                valid_depth_ratio=ratio,
                min_valid_depth_px=min_px,
                min_valid_depth_ratio=min_ratio,
                reason='heavy target mask has little valid aligned depth; publishing mask anyway',
            )
        else:
            self.publish_status(
                'target_mask_depth_ok',
                job_id=job_id,
                request_id=result.get('request_id'),
                mask_px=mask_px,
                valid_depth_px=valid_px,
                valid_depth_ratio=ratio,
            )

    def queue_sam2_live_seed(self, response, result):
        rgb_path = response / 'rgb.jpg'
        objects = result.get('tracked_objects', [])
        if not rgb_path.is_file() or not isinstance(objects, list) or not objects:
            self.publish_status('sam2_seed_skipped', job_id=response.name, error='missing RGB or objects')
            return
        live_spool = Path(str(self.get_parameter('sam2_live_spool_dir').value))
        seed_root = live_spool / 'seeds'
        seed_root.mkdir(parents=True, exist_ok=True)
        final = seed_root / response.name
        temporary = seed_root / (response.name + '.tmp')
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        try:
            shutil.copy2(str(rgb_path), str(temporary / 'rgb.jpg'))
            copied_objects = []
            for record in objects:
                if not isinstance(record, dict):
                    continue
                source = response / str(record.get('mask_file', ''))
                if not source.is_file():
                    continue
                destination_name = 'object_%03d.png' % int(record.get('object_id', 0))
                shutil.copy2(str(source), str(temporary / destination_name))
                copied = dict(record)
                copied['mask_file'] = destination_name
                copied_objects.append(copied)
            if not copied_objects:
                raise ValueError('no valid object masks')
            manifest = {
                'frame_key': response.name,
                'source': 'groundingdino_sam2',
                'image_stamp': result.get('image_stamp', {}),
                'frame_id': result.get('frame_id', ''),
                'objects': copied_objects,
            }
            with (temporary / 'seed.yaml').open('w', encoding='utf-8') as stream:
                yaml.safe_dump(manifest, stream, sort_keys=False)
            (temporary / 'READY').touch()
            if final.exists():
                shutil.rmtree(final)
            os.replace(str(temporary), str(final))
            self.publish_status('sam2_seed_queued', job_id=response.name, object_count=len(copied_objects))
        except Exception as exc:
            shutil.rmtree(temporary, ignore_errors=True)
            self.publish_status('sam2_seed_failed', job_id=response.name, error='%s: %s' % (type(exc).__name__, exc))

    def publish_status(self, state, **values):
        payload = {'state': state, 'dry_run': True, 'real_arm_motion': False}
        payload.update(values)
        out = String()
        out.data = json.dumps(payload, sort_keys=True)
        self.status_pub.publish(out)

    @staticmethod
    def safe_component(value):
        text = ''.join(character if character.isalnum() or character in '-_' else '_' for character in str(value))
        return text[:64] or 'unknown'


def main(args=None):
    rclpy.init(args=args)
    node = HeavyRefreshBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
