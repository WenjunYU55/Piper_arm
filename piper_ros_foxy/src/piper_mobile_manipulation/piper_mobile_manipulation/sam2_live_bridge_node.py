#!/usr/bin/env python3
"""Foxy-side frame bridge and publisher for isolated live SAM2 tracking."""

import json
import os
import shutil
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


class Sam2LiveBridgeNode(Node):
    def __init__(self):
        super().__init__('sam2_live_bridge_node')
        self.declare_parameter('color_image_topic', '/camera/color/image_raw')
        self.declare_parameter('seed_mask_topic', '/piper/heavy_target_mask')
        self.declare_parameter('output_mask_topic', '/piper/sam2_target_mask')
        self.declare_parameter('obstacle_mask_topic', '/piper/sam2_obstacle_mask')
        self.declare_parameter('unsafe_obstacle_mask_topic', '/piper/sam2_unsafe_obstacle_mask')
        self.declare_parameter('movable_obstacle_mask_topic', '/piper/sam2_candidate_movable_obstacle_mask')
        self.declare_parameter('object_ids_topic', '/piper/sam2_object_ids')
        self.declare_parameter('status_topic', '/piper/sam2_tracking_status')
        self.declare_parameter('heavy_request_topic', '/piper/heavy_refresh_request')
        self.declare_parameter('occlusion_status_topic', '/piper/occlusion_status')
        self.declare_parameter('target_status_topic', '/piper/target_status')
        self.declare_parameter('spool_dir', '/tmp/piper_sam2_live')
        self.declare_parameter('frame_rate_hz', 10.0)
        self.declare_parameter('seed_cache_sec', 60.0)
        self.declare_parameter('auto_initial_mask', False)
        self.declare_parameter('allow_heavy_topic_seed', False)
        self.declare_parameter('semantic_refresh_interval_sec', 60.0)
        self.declare_parameter('refresh_cooldown_sec', 5.0)
        self.declare_parameter('lost_refresh_retry_sec', 10.0)
        self.declare_parameter('no_mask_refresh_timeout_sec', 8.0)
        self.declare_parameter('min_target_area_px', 100)
        self.declare_parameter('max_target_area_ratio_change', 2.5)

        self.bridge = CvBridge()
        self.spool = Path(str(self.get_parameter('spool_dir').value))
        for name in ('frames', 'seeds', 'results', 'consumed'):
            (self.spool / name).mkdir(parents=True, exist_ok=True)
        self.jpeg_cache = OrderedDict()
        self.latest_msg = None
        self.latest_bgr = None
        self.last_frame_write = 0.0
        self.initial_requested = False
        self.seed_queued = False
        self.pending_seeds = {}
        self.previous_target_area = None
        self.last_refresh_request = 0.0
        self.last_semantic_refresh = time.monotonic()
        self.last_mask_publish = 0.0
        self.target_lost = False

        self.mask_pub = self.create_publisher(
            Image, self.get_parameter('output_mask_topic').value, qos_profile_sensor_data
        )
        self.obstacle_pub = self.create_publisher(
            Image, self.get_parameter('obstacle_mask_topic').value, qos_profile_sensor_data
        )
        self.unsafe_obstacle_pub = self.create_publisher(
            Image, self.get_parameter('unsafe_obstacle_mask_topic').value, qos_profile_sensor_data
        )
        self.movable_obstacle_pub = self.create_publisher(
            Image, self.get_parameter('movable_obstacle_mask_topic').value, qos_profile_sensor_data
        )
        self.object_ids_pub = self.create_publisher(
            Image, self.get_parameter('object_ids_topic').value, qos_profile_sensor_data
        )
        self.status_pub = self.create_publisher(String, self.get_parameter('status_topic').value, 10)
        self.request_pub = self.create_publisher(
            String, self.get_parameter('heavy_request_topic').value, 10
        )
        self.create_subscription(
            Image, self.get_parameter('color_image_topic').value, self.color_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            String, self.get_parameter('occlusion_status_topic').value, self.occlusion_cb, 10
        )
        self.create_subscription(
            String, self.get_parameter('target_status_topic').value, self.target_status_cb, 10
        )
        self.create_subscription(
            Image, self.get_parameter('seed_mask_topic').value, self.heavy_seed_cb, qos_profile_sensor_data
        )
        self.create_timer(0.02, self.write_frame)
        self.create_timer(0.05, self.poll_results)
        self.create_timer(1.0, self.retry_lost_refresh)
        self.get_logger().warn('SAM2 live bridge is read-only; real arm motion is disabled.')

    @staticmethod
    def stamp_key(stamp):
        return '%010d_%09d' % (int(stamp.sec), int(stamp.nanosec))

    def color_cb(self, msg):
        try:
            image = np.asarray(self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')).copy()
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise IOError('JPEG encoding failed')
            key = self.stamp_key(msg.header.stamp)
            self.jpeg_cache[key] = (time.monotonic(), encoded.tobytes(), msg.header.frame_id)
            self.latest_msg = msg
            self.latest_bgr = image
            cutoff = time.monotonic() - float(self.get_parameter('seed_cache_sec').value)
            while self.jpeg_cache and next(iter(self.jpeg_cache.values()))[0] < cutoff:
                self.jpeg_cache.popitem(last=False)
            pending = self.pending_seeds.pop(key, None)
            if pending is not None:
                self.queue_seed(key, pending[0], pending[1])
            if bool(self.get_parameter('auto_initial_mask').value) and not self.initial_requested:
                request = String()
                request.data = json.dumps({
                    'request_id': 'sam2_initial_%s' % key,
                    'reason': 'sam2_initial_mask',
                    'tracking': {'tracking_confidence': 0.0},
                })
                self.request_pub.publish(request)
                self.initial_requested = True
                self.publish_status('initial_mask_requested', frame_key=key)
        except Exception as exc:
            self.get_logger().warn('SAM2 frame conversion failed: %s' % exc)

    def write_frame(self):
        rate = max(0.1, float(self.get_parameter('frame_rate_hz').value))
        if self.latest_msg is None or time.monotonic() - self.last_frame_write < 1.0 / rate:
            return
        key = self.stamp_key(self.latest_msg.header.stamp)
        cached = self.jpeg_cache.get(key)
        if cached is None:
            return
        final = self.spool / 'frames' / key
        if final.exists():
            return
        temporary = self.spool / 'frames' / (key + '.tmp')
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        (temporary / 'rgb.jpg').write_bytes(cached[1])
        with (temporary / 'frame.yaml').open('w', encoding='utf-8') as stream:
            yaml.safe_dump({
                'image_stamp': {
                    'sec': int(self.latest_msg.header.stamp.sec),
                    'nanosec': int(self.latest_msg.header.stamp.nanosec),
                },
                'frame_id': self.latest_msg.header.frame_id,
            }, stream, sort_keys=False)
        (temporary / 'READY').touch()
        os.replace(str(temporary), str(final))
        queued = sorted(path for path in (self.spool / 'frames').iterdir() if path.is_dir())
        for stale in queued[:-50]:
            shutil.rmtree(stale, ignore_errors=True)
        self.last_frame_write = time.monotonic()

    def heavy_seed_cb(self, msg):
        if bool(self.get_parameter('allow_heavy_topic_seed').value):
            self.seed_cb(msg, 'groundingdino_sam2')

    def occlusion_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        status = str(payload.get('status', payload.get('occlusion_status', ''))).upper()
        if status in ('PARTIALLY_OCCLUDED', 'HEAVILY_OCCLUDED', 'LOST'):
            self.target_lost = True
            self.request_heavy_refresh('occlusion_%s' % status.lower())

    def target_status_cb(self, msg):
        status = str(msg.data).strip().upper()
        if status in ('LOST', 'SEARCHING', 'LOW_CONFIDENCE'):
            self.target_lost = True
            self.request_heavy_refresh('target_status_%s' % status.lower())

    def seed_cb(self, msg, source):
        try:
            mask = np.asarray(self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')).copy()
            if not np.count_nonzero(mask):
                raise ValueError('empty initial mask')
            key = self.stamp_key(msg.header.stamp)
            cached = self.jpeg_cache.get(key)
            if cached is None:
                self.pending_seeds[key] = (mask, source)
                return
            self.queue_seed(key, mask, source)
        except Exception as exc:
            self.publish_status('seed_rejected', error='%s: %s' % (type(exc).__name__, exc))

    def queue_seed(self, key, mask, source):
        try:
            cached = self.jpeg_cache.get(key)
            if cached is None:
                raise ValueError('matching RGB frame expired from seed cache')
            final = self.spool / 'seeds' / ('%s_%s' % (key, source))
            temporary = self.spool / 'seeds' / (key + '.tmp')
            shutil.rmtree(temporary, ignore_errors=True)
            temporary.mkdir(parents=True)
            (temporary / 'rgb.jpg').write_bytes(cached[1])
            cv2.imwrite(str(temporary / 'mask.png'), mask)
            with (temporary / 'seed.yaml').open('w', encoding='utf-8') as stream:
                yaml.safe_dump({'frame_key': key, 'source': source}, stream, sort_keys=False)
            (temporary / 'READY').touch()
            os.replace(str(temporary), str(final))
            self.seed_queued = True
            self.publish_status(
                'seed_queued', frame_key=key, source=source,
                mask_area_px=int(np.count_nonzero(mask))
            )
        except Exception as exc:
            self.publish_status('seed_rejected', error='%s: %s' % (type(exc).__name__, exc))

    def poll_results(self):
        for result_dir in sorted((self.spool / 'results').iterdir()):
            if (
                not result_dir.is_dir()
                or result_dir.name.endswith('.tmp')
                or not (result_dir / 'READY').is_file()
            ):
                continue
            try:
                with (result_dir / 'result.yaml').open('r', encoding='utf-8') as stream:
                    result = yaml.safe_load(stream) or {}
                mask = cv2.imread(str(result_dir / 'mask.png'), cv2.IMREAD_GRAYSCALE)
                if result.get('status') in ('ok', 'empty_target_mask') and mask is not None:
                    out = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
                    stamp = result.get('image_stamp', {})
                    out.header.stamp.sec = int(stamp.get('sec', 0))
                    out.header.stamp.nanosec = int(stamp.get('nanosec', 0))
                    out.header.frame_id = result.get('frame_id', '')
                    self.mask_pub.publish(out)
                    self.last_mask_publish = time.monotonic()
                    self.publish_result_mask(result_dir / 'all_obstacle_mask.png', self.obstacle_pub, out.header)
                    self.publish_result_mask(
                        result_dir / 'unsafe_obstacle_mask.png', self.unsafe_obstacle_pub, out.header
                    )
                    self.publish_result_mask(
                        result_dir / 'candidate_movable_obstacle_mask.png',
                        self.movable_obstacle_pub,
                        out.header,
                    )
                    self.publish_result_mask(
                        result_dir / 'object_ids.png', self.object_ids_pub, out.header, encoding='mono16'
                    )
                    if result.get('status') == 'ok':
                        self.target_lost = False
                        self.publish_status('tracking', **result)
                    else:
                        self.target_lost = True
                        self.publish_status('waiting_for_seed', **result)
                    self.evaluate_refresh_policy(result)
                else:
                    self.publish_status('worker_result_rejected', **result)
                destination = self.spool / 'consumed' / ('result_' + result_dir.name)
                shutil.rmtree(destination, ignore_errors=True)
                os.replace(str(result_dir), str(destination))
                consumed = sorted(
                    path for path in (self.spool / 'consumed').iterdir() if path.is_dir()
                )
                for stale in consumed[:-200]:
                    shutil.rmtree(stale, ignore_errors=True)
            except Exception as exc:
                self.get_logger().error('Failed to consume SAM2 result: %s' % exc)

    def publish_result_mask(self, path, publisher, header, encoding='mono8'):
        read_mode = cv2.IMREAD_UNCHANGED if encoding == 'mono16' else cv2.IMREAD_GRAYSCALE
        mask = cv2.imread(str(path), read_mode)
        if mask is None:
            return
        out = self.bridge.cv2_to_imgmsg(mask, encoding=encoding)
        out.header = header
        publisher.publish(out)

    def evaluate_refresh_policy(self, result):
        area = int(result.get('mask_area_px', 0))
        if area < int(self.get_parameter('min_target_area_px').value):
            self.request_heavy_refresh('sam2_target_lost')
        elif self.previous_target_area:
            ratio = max(area / float(self.previous_target_area), self.previous_target_area / float(area))
            if ratio > float(self.get_parameter('max_target_area_ratio_change').value):
                self.request_heavy_refresh('sam2_target_area_change')
        self.previous_target_area = area
        if time.monotonic() - self.last_semantic_refresh >= float(
            self.get_parameter('semantic_refresh_interval_sec').value
        ):
            self.request_heavy_refresh('periodic_semantic_refresh')

    def retry_lost_refresh(self):
        now = time.monotonic()
        if self.initial_requested and self.last_mask_publish <= 0.0:
            if now - self.last_refresh_request >= float(
                self.get_parameter('no_mask_refresh_timeout_sec').value
            ):
                self.request_heavy_refresh('sam2_no_mask_after_initial')
            return
        if self.last_mask_publish > 0.0:
            age = now - self.last_mask_publish
            if age >= float(self.get_parameter('no_mask_refresh_timeout_sec').value):
                self.target_lost = True
                self.request_heavy_refresh('sam2_no_recent_mask')
                return
        if not self.target_lost:
            return
        retry = max(
            float(self.get_parameter('refresh_cooldown_sec').value),
            float(self.get_parameter('lost_refresh_retry_sec').value),
        )
        if now - self.last_refresh_request >= retry:
            self.request_heavy_refresh('sam2_target_lost_retry')

    def request_heavy_refresh(self, reason):
        now = time.monotonic()
        if now - self.last_refresh_request < float(self.get_parameter('refresh_cooldown_sec').value):
            return
        msg = String()
        msg.data = json.dumps({
            'request_id': 'sam2_event_%d' % int(time.time() * 1000),
            'reason': reason,
            'tracking': {'tracking_confidence': 0.0},
            'dry_run': True,
            'real_arm_motion': False,
        })
        self.request_pub.publish(msg)
        self.last_refresh_request = now
        self.last_semantic_refresh = now
        self.publish_status('heavy_refresh_requested', reason=reason)

    def publish_status(self, state, **values):
        payload = {'state': state, 'dry_run': True, 'real_arm_motion': False}
        payload.update(values)
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Sam2LiveBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
