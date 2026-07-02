#!/usr/bin/env python3
"""Operator-triggered fixed-obstacle repeatability validator."""

import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from std_srvs.srv import Trigger

from piper_mobile_manipulation.msg import ObstacleInstance3D, ObstacleInstance3DArray
from piper_mobile_manipulation.obstacle_geometry import canonical_label


class ObstacleRepeatabilityValidator(Node):
    def __init__(self):
        super().__init__('obstacle_repeatability_validator')
        self.declare_parameter('input_topic', '/piper/obstacle_instances_3d')
        self.declare_parameter('expected_label', 'pen')
        self.declare_parameter('scenario', 'clear_view')
        self.declare_parameter('report_dir', '/tmp/piper_obstacle_validation')
        self.declare_parameter('min_samples', 5)
        self.declare_parameter('max_samples', 8)
        self.declare_parameter('max_drift_m', 0.015)
        self.declare_parameter('max_sample_age_sec', 0.5)
        self.declare_parameter('stability_observations', 3)
        self.declare_parameter('stability_max_drift_m', 0.005)
        self.declare_parameter('expect_scene_blocked', False)
        self.history = deque(maxlen=20)
        self.samples = []
        self.create_subscription(
            ObstacleInstance3DArray, self.get_parameter('input_topic').value, self.array_cb, 10)
        self.create_service(Trigger, '~/capture_sample', self.capture_cb)
        self.create_service(Trigger, '~/finalize', self.finalize_cb)
        self.get_logger().warn(
            'Validator is read-only; reposition the arm only with the approved GUI.')

    def array_cb(self, msg):
        label = canonical_label(self.get_parameter('expected_label').value)
        matches = [
            item for item in msg.instances
            if canonical_label(item.semantic_label) == label
        ]
        self.history.append((time.monotonic(), msg, matches))

    def capture_cb(self, _request, response):
        max_samples = int(self.get_parameter('max_samples').value)
        required = int(self.get_parameter('stability_observations').value)
        if len(self.samples) >= max_samples:
            return self.reply(response, False, 'maximum sample count reached')
        recent = list(self.history)[-required:]
        if len(recent) < required:
            return self.reply(response, False, 'not enough recent observations')
        maximum_age = float(self.get_parameter('max_sample_age_sec').value)
        if time.monotonic() - recent[-1][0] > maximum_age:
            return self.reply(response, False, 'latest obstacle snapshot is stale')
        selected = []
        for _, array, matches in recent:
            if len(matches) != 1:
                return self.reply(response, False, 'expected exactly one matching obstacle')
            item = matches[0]
            if not item.valid:
                return self.reply(response, False, 'invalid obstacle: %s' % item.validity_reason)
            if item.classification != ObstacleInstance3D.CLASSIFICATION_MOVABLE:
                return self.reply(response, False, 'marker is not effectively movable')
            selected.append((array, item))
        ids = {item.object_id for _, item in selected}
        if len(ids) != 1:
            return self.reply(response, False, 'identity changed during stability window')
        points = np.array([[item.base_centroid.x, item.base_centroid.y, item.base_centroid.z]
                           for _, item in selected])
        spread = self.maximum_pairwise_distance(points)
        if spread > float(self.get_parameter('stability_max_drift_m').value):
            return self.reply(
                response, False, 'obstacle is not stable (%.1f mm)' % (spread * 1000.0))
        array, item = selected[-1]
        self.samples.append({
            'index': len(self.samples) + 1,
            'stamp': {'sec': int(item.header.stamp.sec),
                      'nanosec': int(item.header.stamp.nanosec)},
            'object_id': int(item.object_id), 'semantic_label': item.semantic_label,
            'classification': int(item.classification), 'confidence': float(item.confidence),
            'base_centroid_m': [float(item.base_centroid.x), float(item.base_centroid.y),
                                float(item.base_centroid.z)],
            'camera_centroid_m': [float(item.camera_centroid.x), float(item.camera_centroid.y),
                                  float(item.camera_centroid.z)],
            'transform_age_sec': float(item.transform_age_sec),
            'scene_blocked': bool(array.scene_blocked),
            'blocking_reason': array.blocking_reason,
            'stability_drift_m': spread,
        })
        return self.reply(
            response, True, 'captured sample %d/%d' % (len(self.samples), max_samples))

    def finalize_cb(self, _request, response):
        minimum = int(self.get_parameter('min_samples').value)
        points = np.array([sample['base_centroid_m'] for sample in self.samples], dtype=np.float64)
        drift = self.maximum_pairwise_distance(points) if len(points) else math.inf
        ids = {sample['object_id'] for sample in self.samples}
        identity_ok = len(ids) == 1
        classification_ok = all(
            sample['classification'] == ObstacleInstance3D.CLASSIFICATION_MOVABLE
            for sample in self.samples)
        transforms_ok = all(sample['transform_age_sec'] >= 0.0 for sample in self.samples)
        expected_blocked = bool(self.get_parameter('expect_scene_blocked').value)
        scene_blocking_ok = all(
            sample['scene_blocked'] == expected_blocked for sample in self.samples)
        count_ok = minimum <= len(self.samples) <= int(self.get_parameter('max_samples').value)
        drift_ok = drift <= float(self.get_parameter('max_drift_m').value)
        passed = (count_ok and drift_ok and identity_ok and classification_ok and transforms_ok
                  and scene_blocking_ok)
        report = {
            'schema_version': 1,
            'created_unix_sec': time.time(),
            'scenario': str(self.get_parameter('scenario').value),
            'expected_label': str(self.get_parameter('expected_label').value),
            'thresholds': {'minimum_samples': minimum,
                           'maximum_samples': int(self.get_parameter('max_samples').value),
                           'maximum_position_drift_m': float(
                               self.get_parameter('max_drift_m').value)},
            'metrics': {'sample_count': len(self.samples),
                        'maximum_pairwise_position_drift_m': None if math.isinf(drift) else drift,
                        'object_ids': sorted(ids)},
            'checks': {'sample_count': count_ok, 'position_drift': drift_ok,
                       'identity_constant': identity_ok, 'classification_safe': classification_ok,
                       'transforms_valid': transforms_ok,
                       'scene_blocking_matches_expected': scene_blocking_ok},
            'passed': passed, 'samples': self.samples,
        }
        directory = Path(str(self.get_parameter('report_dir').value)).expanduser()
        try:
            directory.mkdir(parents=True, exist_ok=True)
            name = 'obstacle_repeatability_%s_%s.yaml' % (
                str(self.get_parameter('scenario').value).replace(' ', '_'),
                time.strftime('%Y%m%d_%H%M%S'))
            path = directory / name
            with path.open('w', encoding='utf-8') as stream:
                yaml.safe_dump(report, stream, sort_keys=False)
        except OSError as exc:
            return self.reply(response, False, 'failed to write report: %s' % exc)
        verdict = 'PASS' if passed else 'FAIL'
        self.get_logger().info('%s drift=%s samples=%d report=%s' % (
            verdict, 'n/a' if math.isinf(drift) else '%.1fmm' % (drift * 1000.0),
            len(self.samples), path))
        return self.reply(response, passed, '%s: %s' % (verdict, path))

    @staticmethod
    def maximum_pairwise_distance(points):
        if len(points) < 2:
            return 0.0
        differences = points[:, None, :] - points[None, :, :]
        return float(np.max(np.linalg.norm(differences, axis=2)))

    @staticmethod
    def reply(response, success, message):
        response.success = bool(success)
        response.message = str(message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleRepeatabilityValidator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
