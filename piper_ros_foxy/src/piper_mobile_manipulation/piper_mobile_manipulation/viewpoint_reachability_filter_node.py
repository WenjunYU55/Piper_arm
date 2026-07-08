#!/usr/bin/env python3
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from piper_mobile_manipulation.msg import ScanViewpoint, ScanViewpointArray
from piper_mobile_manipulation.utils.piper_kinematics import pose_matrix, solve_camera_pose


class ViewpointReachabilityFilterNode(Node):
    def __init__(self):
        super().__init__('viewpoint_reachability_filter_node')
        self.declare_parameter('scan_viewpoints_topic', '/piper/scan_viewpoints')
        self.declare_parameter(
            'reachable_scan_viewpoints_topic', '/piper/reachable_scan_viewpoints')
        self.declare_parameter('joint_states_topic', '/joint_states_single')
        self.declare_parameter('arm_status_topic', '/arm_status')
        self.declare_parameter('target_status_topic', '/piper/target_status')
        self.declare_parameter('base_frame', 'base_link')

        self.declare_parameter('min_reach_m', 0.20)
        self.declare_parameter('max_reach_m', 0.75)
        self.declare_parameter('min_camera_object_distance_m', 0.25)
        self.declare_parameter('max_camera_object_distance_m', 0.80)
        self.declare_parameter('max_height_change_m', 0.40)
        self.declare_parameter('dry_run', True)
        self.declare_parameter('debug', True)
        self.declare_parameter(
            'joint_bounds_path', '/home/prl/Piper_arm/piper_joint_bounds.json')
        self.declare_parameter(
            'hand_eye_path',
            '/home/prl/Piper_arm/L515_camera/calibration/hand_eye/'
            'session_20260701_local/calibration_result.yaml')
        self.declare_parameter('max_joint_step_rad', 2.5)

        self.arm_status = ''
        self.target_status = 'UNKNOWN'
        self.latest_joint_state = None
        self.lower, self.upper = self.load_joint_bounds()
        self.link6_from_camera = self.load_hand_eye()

        self.pub = self.create_publisher(
            ScanViewpointArray,
            self.get_parameter('reachable_scan_viewpoints_topic').value,
            1,
        )
        self.scan_sub = self.create_subscription(
            ScanViewpointArray,
            self.get_parameter('scan_viewpoints_topic').value,
            self.scan_cb,
            1,
        )
        self.joint_sub = self.create_subscription(
            JointState,
            self.get_parameter('joint_states_topic').value,
            self.joint_cb,
            10,
        )
        self.arm_status_sub = self.create_subscription(
            String,
            self.get_parameter('arm_status_topic').value,
            self.arm_status_cb,
            10,
        )
        self.target_status_sub = self.create_subscription(
            String,
            self.get_parameter('target_status_topic').value,
            self.target_status_cb,
            10,
        )
        self.get_logger().warn(
            'Viewpoint reachability filter is dry-run only; it does not publish '
            '/piper/servo_cmd or move the arm.'
        )

    def joint_cb(self, msg):
        self.latest_joint_state = msg

    def arm_status_cb(self, msg):
        self.arm_status = msg.data

    def target_status_cb(self, msg):
        self.target_status = msg.data

    def scan_cb(self, msg):
        viewpoints = msg.viewpoints
        filtered = []
        reachable_count = 0
        safe_count = 0
        message_reasons = self.message_reject_reasons(msg)
        for viewpoint in viewpoints:
            result = viewpoint
            reasons = list(message_reasons)
            reasons.extend(self.reject_reasons(result))
            accepted = len(reasons) == 0
            if accepted:
                solution, converged, details = self.solve_ik(result)
                if not converged:
                    if details.get('reason') == 'missing joint feedback':
                        reasons.append('IK unavailable: missing joint feedback')
                    else:
                        reasons.append(
                            'IK failed position=%.3fm rotation=%.1fdeg' % (
                                details['position_error_m'],
                                math.degrees(details['rotation_error_rad'])))
                else:
                    result.joint_solution = [float(value) for value in solution]
            accepted = len(reasons) == 0
            result.reachable = bool(accepted)
            # Collision checking is unavailable, therefore safe remains false.
            result.safe = False
            result.status = (
                ScanViewpoint.STATUS_REACHABLE if accepted else ScanViewpoint.STATUS_REJECTED)
            result.rejection_reasons = reasons
            result.safety_score = 0.5 if accepted else 0.0
            filtered.append(result)
            if accepted:
                reachable_count += 1
        out = ScanViewpointArray()
        out.header = msg.header
        out.viewpoints = filtered
        out.requested_coverage_deg = msg.requested_coverage_deg
        out.planned_coverage_deg = msg.planned_coverage_deg
        out.reachable_count = reachable_count
        out.dry_run = True
        self.pub.publish(out)

        if self.param_bool('debug'):
            reason_counts = Counter(
                reason for viewpoint in filtered
                for reason in viewpoint.rejection_reasons)
            summary = '; '.join(
                '%s (%d)' % (reason, count)
                for reason, count in reason_counts.most_common(4))
            self.get_logger().info(
                'filtered scan viewpoints: %d/%d reachable safe=%d reasons=%s'
                % (reachable_count, len(filtered), safe_count, summary or 'none')
            )

    def reject_reasons(self, viewpoint):
        reasons = []
        if not self.param_bool('dry_run'):
            reasons.append('dry_run safety config missing or false')

        expected_frame = str(self.get_parameter('base_frame').value)
        if viewpoint.header.frame_id and viewpoint.header.frame_id != expected_frame:
            reasons.append(
                'viewpoint frame %s != %s' % (viewpoint.header.frame_id, expected_frame))

        if self.status_has_error(self.arm_status):
            reasons.append('arm status reports error')

        if self.target_status in ('LOW_CONFIDENCE', 'LOST'):
            reasons.append('target_status=%s' % self.target_status)

        camera_position = viewpoint.camera_pose.position
        target_center = viewpoint.target_center

        reach = self.vector_norm(camera_position)
        min_reach = float(self.get_parameter('min_reach_m').value)
        max_reach = float(self.get_parameter('max_reach_m').value)
        if reach < min_reach:
            reasons.append('camera target position too close %.3fm < %.3fm' % (reach, min_reach))
        if reach > max_reach:
            reasons.append('camera target position too far %.3fm > %.3fm' % (reach, max_reach))

        camera_object_distance = viewpoint.camera_distance_m
        if not self.is_finite_number(camera_object_distance) or camera_object_distance <= 0.0:
            camera_object_distance = self.distance(camera_position, target_center)
        min_dist = float(self.get_parameter('min_camera_object_distance_m').value)
        max_dist = float(self.get_parameter('max_camera_object_distance_m').value)
        if camera_object_distance < min_dist:
            reasons.append(
                'camera-object distance too close %.3fm < %.3fm'
                % (camera_object_distance, min_dist)
            )
        if camera_object_distance > max_dist:
            reasons.append(
                'camera-object distance too far %.3fm > %.3fm'
                % (camera_object_distance, max_dist)
            )

        height_change = abs(float(camera_position.z) - float(target_center.z))
        max_height_change = float(self.get_parameter('max_height_change_m').value)
        if height_change > max_height_change:
            reasons.append(
                'height change too large %.3fm > %.3fm'
                % (height_change, max_height_change)
            )

        return reasons

    def message_reject_reasons(self, msg):
        reasons = []
        expected_frame = str(self.get_parameter('base_frame').value)
        if msg.header.frame_id and msg.header.frame_id != expected_frame:
            reasons.append('viewpoint array frame %s != %s' % (msg.header.frame_id, expected_frame))
        if not msg.dry_run:
            reasons.append('input scan viewpoint array is not dry_run')
        return reasons

    def solve_ik(self, viewpoint):
        if self.latest_joint_state is None or len(self.latest_joint_state.position) < 6:
            return np.zeros(6), False, {
                'position_error_m': float('inf'), 'rotation_error_rad': float('inf'),
                'reason': 'missing joint feedback'}
        seed = np.asarray(self.latest_joint_state.position[:6], dtype=float)
        solution, converged, details = solve_camera_pose(
            pose_matrix(viewpoint.camera_pose), self.link6_from_camera,
            seed, self.lower, self.upper)
        if converged and np.max(np.abs(solution - seed)) > float(
                self.get_parameter('max_joint_step_rad').value):
            converged = False
            details['position_error_m'] = float('inf')
            details['rotation_error_rad'] = float('inf')
        return solution, converged, details

    def load_joint_bounds(self):
        path = Path(str(self.get_parameter('joint_bounds_path').value)).expanduser()
        with path.open('r', encoding='utf-8') as stream:
            payload = json.load(stream)
        lower, upper = [], []
        for index in range(1, 7):
            record = payload['joints']['joint%d' % index]
            lower.append(float(record.get('command_min', record['min'])))
            upper.append(float(record.get('command_max', record['max'])))
        return np.asarray(lower), np.asarray(upper)

    def load_hand_eye(self):
        path = Path(str(self.get_parameter('hand_eye_path').value)).expanduser()
        with path.open('r', encoding='utf-8') as stream:
            payload = yaml.safe_load(stream) or {}
        if payload.get('status') != 'accepted':
            raise RuntimeError('hand-eye calibration is not accepted')
        return np.asarray(payload['camera_to_link6']['matrix'], dtype=float)

    def publish_rejected_payload(self, reason):
        out = ScanViewpointArray()
        out.dry_run = True
        self.pub.publish(out)
        self.get_logger().warn(reason)

    @staticmethod
    def status_has_error(status):
        lowered = status.lower()
        return 'error' in lowered or 'fault' in lowered

    @staticmethod
    def valid_vector(value):
        if not isinstance(value, dict):
            return False
        return all(
            key in value and ViewpointReachabilityFilterNode.is_finite_number(value[key])
            for key in ('x', 'y', 'z')
        )

    @staticmethod
    def is_finite_number(value):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def vector_norm(value):
        return math.sqrt(
            float(value.x) ** 2
            + float(value.y) ** 2
            + float(value.z) ** 2
        )

    @staticmethod
    def distance(a, b):
        return math.sqrt(
            (float(a.x) - float(b.x)) ** 2
            + (float(a.y) - float(b.y)) ** 2
            + (float(a.z) - float(b.z)) ** 2
        )

    def param_bool(self, name):
        value = self.get_parameter(name).value
        if isinstance(value, str):
            return value.lower() in ('1', 'true', 'yes', 'on')
        return bool(value)


def main(args=None):
    rclpy.init(args=args)
    node = ViewpointReachabilityFilterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
