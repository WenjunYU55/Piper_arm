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
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


class HeavyRefreshBridgeNode(Node):
    def __init__(self):
        super().__init__('heavy_refresh_bridge_node')
        self.declare_parameter('color_image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_image_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('tracked_mask_topic', '/piper/temporal_target_mask')
        self.declare_parameter('request_topic', '/piper/heavy_refresh_request')
        self.declare_parameter('output_mask_topic', '/piper/heavy_target_mask')
        self.declare_parameter('movable_obstacle_mask_topic', '/piper/candidate_movable_obstacle_mask')
        self.declare_parameter('unsafe_obstacle_mask_topic', '/piper/unsafe_obstacle_mask')
        self.declare_parameter('all_obstacle_mask_topic', '/piper/heavy_obstacle_mask')
        self.declare_parameter('status_topic', '/piper/heavy_refresh_status')
        self.declare_parameter('spool_dir', '/tmp/piper_heavy_refresh')
        self.declare_parameter('response_poll_period_sec', 0.20)
        self.declare_parameter('max_image_age_sec', 1.0)
        self.declare_parameter('dry_run', True)
        self.declare_parameter('enable_real_arm_motion', False)

        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None
        self.latest_mask = None
        self.latest_color_msg = None
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
        self.create_subscription(Image, self.get_parameter('color_image_topic').value, self.color_cb, qos_profile_sensor_data)
        self.create_subscription(Image, self.get_parameter('depth_image_topic').value, self.depth_cb, qos_profile_sensor_data)
        self.create_subscription(Image, self.get_parameter('tracked_mask_topic').value, self.mask_cb, qos_profile_sensor_data)
        self.create_subscription(String, self.get_parameter('request_topic').value, self.request_cb, 10)
        self.create_timer(float(self.get_parameter('response_poll_period_sec').value), self.poll_responses)
        self.get_logger().warn('Heavy refresh bridge is read-only; real arm motion is disabled.')

    def color_cb(self, msg):
        try:
            self.latest_color = np.asarray(self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')).copy()
            self.latest_color_msg = msg
            self.latest_color_time = time.monotonic()
            if self.pending_request is not None:
                request = self.pending_request
                self.pending_request = None
                self.enqueue_request(request)
        except Exception as exc:
            self.get_logger().warn('Color conversion failed: %s' % exc)

    def depth_cb(self, msg):
        try:
            self.latest_depth = np.asarray(self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')).copy()
        except Exception as exc:
            self.get_logger().warn('Depth conversion failed: %s' % exc)

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
        if self.latest_color is None or self.latest_color_msg is None:
            self.pending_request = request
            self.publish_status('waiting_for_image', request_id=request.get('request_id'))
            return
        age = time.monotonic() - self.latest_color_time
        if age > float(self.get_parameter('max_image_age_sec').value):
            self.pending_request = request
            self.publish_status('waiting_for_image', request_id=request.get('request_id'), image_age_sec=age)
            return
        self.enqueue_request(request)

    def enqueue_request(self, request):
        request_id = request.get('request_id', 'unknown')
        if self.worker_busy():
            self.publish_status('request_ignored_busy', request_id=request_id)
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
                'frame_id': self.latest_color_msg.header.frame_id,
                'dry_run': True,
                'real_arm_motion': False,
            }
            with (temporary / 'request.yaml').open('w', encoding='utf-8') as stream:
                yaml.safe_dump(manifest, stream, sort_keys=False)
            (temporary / 'READY').touch()
            os.replace(str(temporary), str(final_dir))
            self.publish_status('queued', job_id=job_id, request_id=request_id)
        except Exception as exc:
            shutil.rmtree(temporary, ignore_errors=True)
            self.publish_status('request_failed', job_id=job_id, error='%s: %s' % (type(exc).__name__, exc))

    def worker_busy(self):
        for name in ('requests', 'processing', 'responses'):
            directory = self.spool / name
            if directory.is_dir() and any(directory.iterdir()):
                return True
        return False

    def poll_responses(self):
        response_root = self.spool / 'responses'
        for response in sorted(response_root.iterdir()):
            if not response.is_dir() or response.name.endswith('.tmp') or not (response / 'READY').is_file():
                continue
            try:
                with (response / 'result.yaml').open('r', encoding='utf-8') as stream:
                    result = yaml.safe_load(stream) or {}
                mask = cv2.imread(str(response / 'target_mask.png'), cv2.IMREAD_GRAYSCALE)
                if result.get('status') != 'ok' or mask is None or not np.count_nonzero(mask):
                    self.publish_status('worker_result_rejected', job_id=response.name, worker_status=result.get('status'))
                else:
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
                        request_id=result.get('request_id'),
                        target_confidence=result.get('target_confidence'),
                        obstacle_count=result.get('obstacle_count', 0),
                        obstacle_labels=result.get('obstacle_labels', []),
                        unsafe_obstacle_count=result.get('unsafe_obstacle_count', 0),
                    )
                destination = self.spool / 'consumed' / response.name
                shutil.rmtree(destination, ignore_errors=True)
                os.replace(str(response), str(destination))
            except Exception as exc:
                self.get_logger().error('Failed to consume %s: %s' % (response, exc))

    def publish_response_mask(self, path, publisher, header):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return
        out = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        out.header = header
        publisher.publish(out)

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
