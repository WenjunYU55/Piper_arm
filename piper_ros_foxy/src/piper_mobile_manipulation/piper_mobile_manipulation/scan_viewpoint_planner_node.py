#!/usr/bin/env python3
import json
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String

from piper_mobile_manipulation.msg import Target3D, TrackedTarget


class ScanViewpointPlannerNode(Node):
    def __init__(self):
        super().__init__('scan_viewpoint_planner_node')
        self.declare_parameter('object_topic', '/piper/object_of_interest_3d')
        self.declare_parameter('tracked_target_topic', '/piper/tracked_target')
        self.declare_parameter('fallback_target_topic', '/piper/target_3d')
        self.declare_parameter('target_status_topic', '/piper/target_status')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('scan_viewpoints_topic', '/piper/scan_viewpoints')
        self.declare_parameter('scan_coverage_topic', '/piper/scan_coverage')

        self.declare_parameter('desired_scan_angle_deg', 250)
        self.declare_parameter('viewpoint_step_deg', 15)
        self.declare_parameter('scan_radius_m', 0.45)
        self.declare_parameter('min_scan_radius_m', 0.30)
        self.declare_parameter('max_scan_radius_m', 0.80)
        self.declare_parameter('camera_pitch_deg', -10)
        self.declare_parameter('keep_object_centered', True)
        self.declare_parameter('max_viewpoints', 20)
        self.declare_parameter('dry_run', True)
        self.declare_parameter('debug', True)
        self.declare_parameter('use_predicted_target_for_scan', True)
        self.declare_parameter('tracked_preference_timeout_s', 1.0)

        self.target_status = 'UNKNOWN'
        self.latest_camera_info = None
        self.last_tracked_time = None
        self.pub_viewpoints = self.create_publisher(
            String, self.get_parameter('scan_viewpoints_topic').value, 10
        )
        self.pub_coverage = self.create_publisher(
            String, self.get_parameter('scan_coverage_topic').value, 10
        )

        self.object_sub = self.create_subscription(
            Target3D,
            self.get_parameter('object_topic').value,
            lambda msg: self.target_cb(msg, 'object_of_interest_3d'),
            10,
        )
        self.tracked_sub = self.create_subscription(
            TrackedTarget,
            self.get_parameter('tracked_target_topic').value,
            self.tracked_target_cb,
            10,
        )
        self.fallback_sub = self.create_subscription(
            Target3D,
            self.get_parameter('fallback_target_topic').value,
            lambda msg: self.target_cb(msg, 'target_3d'),
            10,
        )
        self.status_sub = self.create_subscription(
            String,
            self.get_parameter('target_status_topic').value,
            self.status_cb,
            10,
        )
        self.info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self.camera_info_cb,
            10,
        )
        self.get_logger().warn(
            'Scan viewpoint planner is dry-run only; it does not publish /piper/servo_cmd or move the arm.'
        )

    def status_cb(self, msg):
        self.target_status = msg.data

    def camera_info_cb(self, msg):
        self.latest_camera_info = msg

    def target_cb(self, msg, source):
        if not msg.valid:
            return
        if source in ('target_3d', 'object_of_interest_3d') and self.recent_tracked_target_available():
            return

        dry_run = self.param_bool('dry_run')
        if not dry_run:
            self.get_logger().warn('dry_run parameter was false; forcing planner output to dry-run semantics')

        center = {
            'x': float(msg.point.x),
            'y': float(msg.point.y),
            'z': float(msg.point.z),
        }
        frame_id = msg.header.frame_id
        angles = self.viewpoint_angles()
        radius = self.scan_radius()
        viewpoints = []
        for index, angle_deg in enumerate(angles):
            viewpoint = self.make_viewpoint(index, angle_deg, radius, center, frame_id)
            viewpoints.append(viewpoint)

        requested_coverage = float(self.get_parameter('desired_scan_angle_deg').value)
        achieved_dry_run_coverage = self.coverage_from_angles(angles)
        stamp = {
            'sec': int(msg.header.stamp.sec),
            'nanosec': int(msg.header.stamp.nanosec),
        }
        camera_info = self.camera_info_summary()

        view_msg = String()
        view_msg.data = json.dumps(
            {
                'header': {
                    'stamp': stamp,
                    'frame_id': frame_id,
                },
                'dry_run': True,
                'source_topic': source,
                'target_status': self.target_status,
                'target_object_center': center,
                'camera_info': camera_info,
                'viewpoints': viewpoints,
            },
            sort_keys=True,
        )
        self.pub_viewpoints.publish(view_msg)

        coverage_msg = String()
        coverage_msg.data = json.dumps(
            {
                'header': {
                    'stamp': stamp,
                    'frame_id': frame_id,
                },
                'dry_run': True,
                'requested_scan_angle_deg': requested_coverage,
                'planned_scan_angle_deg': achieved_dry_run_coverage,
                'viewpoint_step_deg': float(self.get_parameter('viewpoint_step_deg').value),
                'candidate_viewpoints': len(viewpoints),
                'reachable_viewpoints': 0,
                'safe_viewpoints': 0,
                'note': 'reachability and safety are intentionally false until a later dry-run evaluator is added',
            },
            sort_keys=True,
        )
        self.pub_coverage.publish(coverage_msg)

        if self.param_bool('debug'):
            self.get_logger().info(
                'planned %d dry-run scan viewpoints around target from %s coverage=%.1fdeg radius=%.2fm'
                % (len(viewpoints), source, achieved_dry_run_coverage, radius)
            )

    def tracked_target_cb(self, msg):
        if not msg.valid:
            return
        self.last_tracked_time = self.get_clock().now()
        use_predicted = self.param_bool('use_predicted_target_for_scan')
        point = msg.predicted_position if use_predicted else msg.position
        target = Target3D()
        target.header = msg.header
        target.point.x = float(point.x)
        target.point.y = float(point.y)
        target.point.z = float(point.z)
        target.measurement_confidence = float(msg.confidence)
        target.valid = True
        self.target_cb(
            target,
            'tracked_target_predicted' if use_predicted else 'tracked_target_filtered',
        )

    def recent_tracked_target_available(self):
        if self.last_tracked_time is None:
            return False
        age = (self.get_clock().now() - self.last_tracked_time).nanoseconds * 1e-9
        return age <= float(self.get_parameter('tracked_preference_timeout_s').value)

    def make_viewpoint(self, index, angle_deg, radius, center, frame_id):
        angle_rad = math.radians(angle_deg)
        camera_position = {
            'x': center['x'] + radius * math.cos(angle_rad),
            'y': center['y'] + radius * math.sin(angle_rad),
            'z': center['z'],
        }
        look = {
            'x': center['x'] - camera_position['x'],
            'y': center['y'] - camera_position['y'],
            'z': center['z'] - camera_position['z'],
        }
        look_norm = math.sqrt(look['x'] ** 2 + look['y'] ** 2 + look['z'] ** 2)
        if look_norm > 1e-9:
            look_direction = {
                'x': look['x'] / look_norm,
                'y': look['y'] / look_norm,
                'z': look['z'] / look_norm,
            }
        else:
            look_direction = {'x': 0.0, 'y': 0.0, 'z': 0.0}

        return {
            'index': int(index),
            'frame_id': frame_id,
            'viewpoint_angle_deg': float(angle_deg),
            'target_object_center': center,
            'desired_camera_position': camera_position,
            'desired_look_at_direction': look_direction,
            'camera_object_distance_m': float(radius),
            'camera_pitch_deg': float(self.get_parameter('camera_pitch_deg').value),
            'keep_object_centered': self.param_bool('keep_object_centered'),
            'reachable': False,
            'safe': False,
        }

    def viewpoint_angles(self):
        desired = abs(float(self.get_parameter('desired_scan_angle_deg').value))
        step = max(abs(float(self.get_parameter('viewpoint_step_deg').value)), 1e-3)
        max_viewpoints = max(int(self.get_parameter('max_viewpoints').value), 1)
        half = desired * 0.5
        angles = []
        current = -half
        while current <= half + 1e-6:
            angles.append(round(current, 6))
            current += step
        if angles[-1] < half:
            angles.append(round(half, 6))
        if len(angles) > max_viewpoints:
            angles = self.evenly_downsample(angles, max_viewpoints)
        return angles

    @staticmethod
    def evenly_downsample(values, max_count):
        if max_count == 1:
            return [values[len(values) // 2]]
        last = len(values) - 1
        indexes = [
            int(round(i * last / float(max_count - 1)))
            for i in range(max_count)
        ]
        return [values[i] for i in indexes]

    @staticmethod
    def coverage_from_angles(angles):
        if len(angles) < 2:
            return 0.0
        return float(max(angles) - min(angles))

    def scan_radius(self):
        radius = float(self.get_parameter('scan_radius_m').value)
        min_radius = float(self.get_parameter('min_scan_radius_m').value)
        max_radius = float(self.get_parameter('max_scan_radius_m').value)
        return min(max(radius, min_radius), max_radius)

    def camera_info_summary(self):
        if self.latest_camera_info is None:
            return {'available': False}
        return {
            'available': True,
            'width': int(self.latest_camera_info.width),
            'height': int(self.latest_camera_info.height),
            'frame_id': self.latest_camera_info.header.frame_id,
        }

    def param_bool(self, name):
        value = self.get_parameter(name).value
        if isinstance(value, str):
            return value.lower() in ('1', 'true', 'yes', 'on')
        return bool(value)


def main(args=None):
    rclpy.init(args=args)
    node = ScanViewpointPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
