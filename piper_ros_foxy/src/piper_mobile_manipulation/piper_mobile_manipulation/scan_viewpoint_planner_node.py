#!/usr/bin/env python3
import json
import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from piper_mobile_manipulation.msg import ScanViewpoint, ScanViewpointArray, TargetEstimate


class ScanViewpointPlannerNode(Node):
    def __init__(self):
        super().__init__('scan_viewpoint_planner_node')
        self.declare_parameter('target_topic', '/piper/target/predicted_base')
        self.declare_parameter('target_status_topic', '/piper/target_status')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('scan_viewpoints_topic', '/piper/scan_viewpoints')
        self.declare_parameter('scan_coverage_topic', '/piper/scan_coverage')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('fallback_center_angle_deg', 180.0)

        self.declare_parameter('desired_scan_angle_deg', 250)
        self.declare_parameter('viewpoint_step_deg', 15)
        self.declare_parameter('scan_radius_m', 0.45)
        self.declare_parameter('min_scan_radius_m', 0.30)
        self.declare_parameter('max_scan_radius_m', 0.80)
        self.declare_parameter('camera_pitch_deg', -10)
        self.declare_parameter('camera_height_offset_m', 0.20)
        self.declare_parameter('replan_interval_sec', 1.0)
        self.declare_parameter('replan_target_motion_m', 0.01)
        self.declare_parameter('keep_object_centered', True)
        self.declare_parameter('max_viewpoints', 20)
        self.declare_parameter('dry_run', True)
        self.declare_parameter('debug', True)

        self.target_status = 'UNKNOWN'
        self.latest_camera_info = None
        self.last_plan_time = 0.0
        self.last_plan_center = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub_viewpoints = self.create_publisher(
            ScanViewpointArray, self.get_parameter('scan_viewpoints_topic').value, 1
        )
        self.pub_coverage = self.create_publisher(
            String, self.get_parameter('scan_coverage_topic').value, 10
        )

        self.target_sub = self.create_subscription(
            TargetEstimate,
            self.get_parameter('target_topic').value,
            self.target_cb,
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
            'Scan viewpoint planner is dry-run only; it does not publish '
            '/piper/servo_cmd or move the arm.'
        )

    def status_cb(self, msg):
        self.target_status = msg.data

    def camera_info_cb(self, msg):
        self.latest_camera_info = msg

    def target_cb(self, msg):
        if not msg.valid:
            return
        base_frame = str(self.get_parameter('base_frame').value)
        if msg.header.frame_id != base_frame:
            self.get_logger().error(
                'refusing target in frame %s; required %s' % (msg.header.frame_id, base_frame))
            return

        dry_run = self.param_bool('dry_run')
        if not dry_run:
            self.get_logger().warn(
                'dry_run parameter was false; forcing planner output to dry-run semantics')

        center = {
            'x': float(msg.pose.pose.position.x),
            'y': float(msg.pose.pose.position.y),
            'z': float(msg.pose.pose.position.z),
        }
        now = time.monotonic()
        if self.last_plan_center is not None:
            displacement = math.sqrt(sum(
                (center[key] - self.last_plan_center[key]) ** 2
                for key in ('x', 'y', 'z')))
            interval = float(self.get_parameter('replan_interval_sec').value)
            threshold = float(self.get_parameter('replan_target_motion_m').value)
            if now - self.last_plan_time < interval and displacement < threshold:
                return
        self.last_plan_time = now
        self.last_plan_center = dict(center)
        frame_id = msg.header.frame_id
        offsets = self.viewpoint_angles()
        center_angle = self.current_view_angle(center, msg.header.stamp)
        angles = [center_angle + offset for offset in offsets]
        radius = self.scan_radius()
        viewpoints = []
        for index, angle_deg in enumerate(angles):
            viewpoint = self.make_viewpoint(index, angle_deg, radius, center, frame_id)
            viewpoints.append(viewpoint)

        requested_coverage = float(self.get_parameter('desired_scan_angle_deg').value)
        achieved_dry_run_coverage = self.coverage_from_angles(offsets)
        stamp = {
            'sec': int(msg.header.stamp.sec),
            'nanosec': int(msg.header.stamp.nanosec),
        }

        view_msg = ScanViewpointArray()
        view_msg.header = msg.header
        view_msg.viewpoints = viewpoints
        view_msg.requested_coverage_deg = requested_coverage
        view_msg.planned_coverage_deg = achieved_dry_run_coverage
        view_msg.reachable_count = 0
        view_msg.dry_run = True
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
                'note': (
                    'reachability and safety are false until the IK evaluator runs'),
            },
            sort_keys=True,
        )
        self.pub_coverage.publish(coverage_msg)

        if self.param_bool('debug'):
            self.get_logger().info(
                'planned %d dry-run scan viewpoints around target from %s '
                'coverage=%.1fdeg radius=%.2fm center=%.1fdeg'
                % (len(viewpoints), 'predicted_base', achieved_dry_run_coverage,
                   radius, center_angle)
            )

    def current_view_angle(self, center, stamp):
        base_frame = str(self.get_parameter('base_frame').value)
        camera_frame = str(self.get_parameter('camera_frame').value)
        try:
            transform = self.tf_buffer.lookup_transform(
                base_frame, camera_frame, rclpy.time.Time.from_msg(stamp))
            translation = transform.transform.translation
            return math.degrees(math.atan2(
                float(translation.y) - center['y'],
                float(translation.x) - center['x']))
        except Exception as exc:
            if self.param_bool('debug'):
                self.get_logger().warn(
                    'camera TF unavailable for arc anchor; using fallback: %s' % exc)
            return float(self.get_parameter('fallback_center_angle_deg').value)

    def make_viewpoint(self, index, angle_deg, radius, center, frame_id):
        angle_rad = math.radians(angle_deg)
        camera_position = {
            'x': center['x'] + radius * math.cos(angle_rad),
            'y': center['y'] + radius * math.sin(angle_rad),
            'z': center['z'] + float(
                self.get_parameter('camera_height_offset_m').value),
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

        viewpoint = ScanViewpoint()
        viewpoint.index = int(index)
        viewpoint.header.frame_id = frame_id
        viewpoint.camera_pose.position.x = camera_position['x']
        viewpoint.camera_pose.position.y = camera_position['y']
        viewpoint.camera_pose.position.z = camera_position['z']
        quaternion = self.look_at_quaternion(look_direction)
        viewpoint.camera_pose.orientation.x = quaternion[0]
        viewpoint.camera_pose.orientation.y = quaternion[1]
        viewpoint.camera_pose.orientation.z = quaternion[2]
        viewpoint.camera_pose.orientation.w = quaternion[3]
        viewpoint.target_center.x = center['x']
        viewpoint.target_center.y = center['y']
        viewpoint.target_center.z = center['z']
        viewpoint.view_angle_deg = float(angle_deg)
        viewpoint.camera_distance_m = float(radius)
        viewpoint.coverage_score = 1.0
        viewpoint.safety_score = 0.0
        viewpoint.status = ScanViewpoint.STATUS_UNCHECKED
        viewpoint.reachable = False
        viewpoint.safe = False
        return viewpoint

    @staticmethod
    def look_at_quaternion(direction):
        """ROS optical frame: +Z looks forward and -Y is camera up."""
        import numpy as np
        z_axis = np.asarray([direction['x'], direction['y'], direction['z']], dtype=float)
        norm = np.linalg.norm(z_axis)
        if norm < 1e-9:
            return (0.0, 0.0, 0.0, 1.0)
        z_axis /= norm
        world_up = np.asarray([0.0, 0.0, 1.0])
        x_axis = np.cross(z_axis, world_up)
        if np.linalg.norm(x_axis) < 1e-6:
            world_up = np.asarray([0.0, 1.0, 0.0])
            x_axis = np.cross(z_axis, world_up)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        matrix = np.column_stack((x_axis, y_axis, z_axis))
        trace = float(np.trace(matrix))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            return ((matrix[2, 1] - matrix[1, 2]) / s,
                    (matrix[0, 2] - matrix[2, 0]) / s,
                    (matrix[1, 0] - matrix[0, 1]) / s, 0.25 * s)
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            return (0.25 * s, (matrix[0, 1] + matrix[1, 0]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                    (matrix[2, 1] - matrix[1, 2]) / s)
        if index == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            return ((matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                    (matrix[0, 2] - matrix[2, 0]) / s)
        s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        return ((matrix[0, 2] + matrix[2, 0]) / s,
                (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s,
                (matrix[1, 0] - matrix[0, 1]) / s)

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
