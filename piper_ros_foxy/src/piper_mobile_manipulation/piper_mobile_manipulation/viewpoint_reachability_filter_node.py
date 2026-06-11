#!/usr/bin/env python3
import json
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String


class ViewpointReachabilityFilterNode(Node):
    def __init__(self):
        super().__init__('viewpoint_reachability_filter_node')
        self.declare_parameter('scan_viewpoints_topic', '/piper/scan_viewpoints')
        self.declare_parameter('reachable_scan_viewpoints_topic', '/piper/reachable_scan_viewpoints')
        self.declare_parameter('joint_states_topic', '/joint_states_single')
        self.declare_parameter('arm_status_topic', '/arm_status')
        self.declare_parameter('target_status_topic', '/piper/target_status')

        self.declare_parameter('min_reach_m', 0.20)
        self.declare_parameter('max_reach_m', 0.75)
        self.declare_parameter('min_camera_object_distance_m', 0.25)
        self.declare_parameter('max_camera_object_distance_m', 0.80)
        self.declare_parameter('max_height_change_m', 0.40)
        self.declare_parameter('dry_run', False)
        self.declare_parameter('debug', True)

        self.arm_status = ''
        self.target_status = 'UNKNOWN'
        self.latest_joint_state = None

        self.pub = self.create_publisher(
            String,
            self.get_parameter('reachable_scan_viewpoints_topic').value,
            10,
        )
        self.scan_sub = self.create_subscription(
            String,
            self.get_parameter('scan_viewpoints_topic').value,
            self.scan_cb,
            10,
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
            'Viewpoint reachability filter is dry-run only; it does not publish /piper/servo_cmd or move the arm.'
        )

    def joint_cb(self, msg):
        self.latest_joint_state = msg

    def arm_status_cb(self, msg):
        self.arm_status = msg.data

    def target_status_cb(self, msg):
        self.target_status = msg.data

    def scan_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.publish_rejected_payload('invalid scan viewpoint JSON: %s' % exc)
            return

        viewpoints = payload.get('viewpoints', [])
        if not isinstance(viewpoints, list):
            self.publish_rejected_payload('scan viewpoint JSON has no viewpoints list')
            return

        filtered = []
        reachable_count = 0
        safe_count = 0
        for viewpoint in viewpoints:
            if not isinstance(viewpoint, dict):
                continue
            result = dict(viewpoint)
            reasons = self.reject_reasons(result)
            accepted = len(reasons) == 0
            result['reachable'] = bool(accepted)
            result['safe'] = bool(accepted)
            result['reject_reasons'] = reasons
            filtered.append(result)
            if accepted:
                reachable_count += 1
                safe_count += 1

        output = dict(payload)
        output['dry_run'] = True
        output['filter'] = {
            'node': 'viewpoint_reachability_filter_node',
            'mode': 'conservative_workspace_check',
            'input_viewpoints': len(viewpoints),
            'output_viewpoints': len(filtered),
            'reachable_viewpoints': reachable_count,
            'safe_viewpoints': safe_count,
            'arm_status': self.arm_status,
            'target_status': self.target_status,
            'dry_run_config_loaded': self.param_bool('dry_run'),
        }
        output['viewpoints'] = filtered

        out = String()
        out.data = json.dumps(output, sort_keys=True)
        self.pub.publish(out)

        if self.param_bool('debug'):
            self.get_logger().info(
                'filtered scan viewpoints: %d/%d reachable safe=%d'
                % (reachable_count, len(filtered), safe_count)
            )

    def reject_reasons(self, viewpoint):
        reasons = []
        if not self.param_bool('dry_run'):
            reasons.append('dry_run safety config missing or false')

        if self.status_has_error(self.arm_status):
            reasons.append('arm status reports error')

        if self.target_status in ('LOW_CONFIDENCE', 'LOST'):
            reasons.append('target_status=%s' % self.target_status)

        camera_position = viewpoint.get('desired_camera_position')
        target_center = viewpoint.get('target_object_center')
        if not self.valid_vector(camera_position):
            reasons.append('missing desired camera position')
            return reasons
        if not self.valid_vector(target_center):
            reasons.append('missing target object center')
            return reasons

        reach = self.vector_norm(camera_position)
        min_reach = float(self.get_parameter('min_reach_m').value)
        max_reach = float(self.get_parameter('max_reach_m').value)
        if reach < min_reach:
            reasons.append('camera target position too close %.3fm < %.3fm' % (reach, min_reach))
        if reach > max_reach:
            reasons.append('camera target position too far %.3fm > %.3fm' % (reach, max_reach))

        camera_object_distance = viewpoint.get('camera_object_distance_m')
        if not self.is_finite_number(camera_object_distance):
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

        height_change = abs(float(camera_position['z']) - float(target_center['z']))
        max_height_change = float(self.get_parameter('max_height_change_m').value)
        if height_change > max_height_change:
            reasons.append(
                'height change too large %.3fm > %.3fm'
                % (height_change, max_height_change)
            )

        return reasons

    def publish_rejected_payload(self, reason):
        out = String()
        out.data = json.dumps(
            {
                'dry_run': True,
                'filter': {
                    'node': 'viewpoint_reachability_filter_node',
                    'reachable_viewpoints': 0,
                    'safe_viewpoints': 0,
                    'reject_reasons': [reason],
                    'dry_run_config_loaded': self.param_bool('dry_run'),
                },
                'viewpoints': [],
            },
            sort_keys=True,
        )
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
            float(value['x']) ** 2
            + float(value['y']) ** 2
            + float(value['z']) ** 2
        )

    @staticmethod
    def distance(a, b):
        return math.sqrt(
            (float(a['x']) - float(b['x'])) ** 2
            + (float(a['y']) - float(b['y'])) ** 2
            + (float(a['z']) - float(b['z'])) ** 2
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
