#!/usr/bin/env python3
import json
import math
import time

import rclpy
from piper_msgs.msg import PiperStatusMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from piper_mobile_manipulation.scan_motion import motor_control_reasons


ROUGH_ACQUISITION = 'ROUGH_ACQUISITION'


def target_status_rejection_reason(target_status, plan_kind):
    """Keep normal tracking gates while allowing LOST acquisition bootstrap."""
    status = str(target_status)
    if status == 'LOW_CONFIDENCE':
        return 'target_status=LOW_CONFIDENCE'
    if status == 'LOST' and plan_kind != ROUGH_ACQUISITION:
        return 'target_status=LOST'
    return ''


class ViewpointReachabilityFilterNode(Node):
    def __init__(self):
        super().__init__('viewpoint_reachability_filter_node')
        self.declare_parameter('scan_viewpoints_topic', '/piper/scan_viewpoints')
        self.declare_parameter(
            'reachable_scan_viewpoints_topic', '/piper/reachable_scan_viewpoints')
        self.declare_parameter(
            'acquisition_viewpoints_topic', '/piper/acquisition_viewpoints')
        self.declare_parameter(
            'reachable_acquisition_viewpoints_topic',
            '/piper/reachable_acquisition_viewpoints')
        self.declare_parameter('joint_states_topic', '/joint_states_single')
        self.declare_parameter('arm_status_topic', '/arm_status')
        self.declare_parameter('target_status_topic', '/piper/target_status')

        self.declare_parameter('min_reach_m', 0.20)
        self.declare_parameter('max_reach_m', 0.75)
        self.declare_parameter('min_camera_object_distance_m', 0.25)
        self.declare_parameter('max_camera_object_distance_m', 0.80)
        self.declare_parameter('max_height_change_m', 0.40)
        # Static radial/height boxes are not an authority for the deployed
        # arm. Tesseract IK, joint limits, exact collision validation and the
        # executor's runtime revalidation own physical reachability. Keep the
        # legacy values only for explicit compatibility/debug opt-in.
        self.declare_parameter('enforce_static_reach_bounds', False)
        self.declare_parameter('arm_status_timeout_sec', 1.0)
        self.declare_parameter('dry_run', False)
        self.declare_parameter('debug', True)

        self.arm_status = None
        self.arm_status_at = None
        self.target_status = 'UNKNOWN'
        self.latest_joint_state = None

        self.pub = self.create_publisher(
            String,
            self.get_parameter('reachable_scan_viewpoints_topic').value,
            10,
        )
        self.acquisition_pub = self.create_publisher(
            String,
            self.get_parameter('reachable_acquisition_viewpoints_topic').value,
            10,
        )
        # PiPER publishes reliable status at high rate. Retain only the newest
        # sample. Missing or stale status rejects every viewpoint without
        # destroying/recreating DDS entities in the running safety stack.
        self.arm_status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.arm_status_sub = self.create_subscription(
            PiperStatusMsg,
            self.get_parameter('arm_status_topic').value,
            self.arm_status_cb,
            self.arm_status_qos,
        )
        self.target_status_sub = self.create_subscription(
            String,
            self.get_parameter('target_status_topic').value,
            self.target_status_cb,
            10,
        )
        self.joint_sub = self.create_subscription(
            JointState,
            self.get_parameter('joint_states_topic').value,
            self.joint_cb,
            10,
        )
        # Create the high-rate scan input last so safety-state subscriptions are
        # established before the first candidate burst arrives.
        self.scan_sub = self.create_subscription(
            String,
            self.get_parameter('scan_viewpoints_topic').value,
            self.scan_cb,
            10,
        )
        self.acquisition_sub = self.create_subscription(
            String,
            self.get_parameter('acquisition_viewpoints_topic').value,
            self.acquisition_cb,
            10,
        )
        self.get_logger().warn(
            'Viewpoint reachability filter is dry-run only; it does not publish '
            '/piper/servo_cmd or move the arm.'
        )

    def joint_cb(self, msg):
        self.latest_joint_state = msg

    def arm_status_cb(self, msg):
        self.arm_status = msg
        self.arm_status_at = time.monotonic()

    def target_status_cb(self, msg):
        self.target_status = msg.data

    def scan_cb(self, msg):
        self.filter_payload(
            msg, self.pub, expected_plan_kind='MULTIVIEW_SCAN')

    def acquisition_cb(self, msg):
        self.filter_payload(
            msg, self.acquisition_pub, expected_plan_kind=ROUGH_ACQUISITION)

    def filter_payload(self, msg, publisher, expected_plan_kind):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.publish_rejected_payload(
                'invalid scan viewpoint JSON: %s' % exc,
                publisher,
                expected_plan_kind,
            )
            return

        plan_kind = payload.get('plan_kind', 'MULTIVIEW_SCAN')
        if expected_plan_kind is not None and plan_kind != expected_plan_kind:
            self.publish_rejected_payload(
                'viewpoint plan_kind on this topic must be %s'
                % expected_plan_kind,
                publisher,
                expected_plan_kind,
            )
            return
        viewpoints = payload.get('viewpoints', [])
        if not isinstance(viewpoints, list):
            self.publish_rejected_payload(
                'scan viewpoint JSON has no viewpoints list',
                publisher,
                plan_kind,
            )
            return

        filtered = []
        reachable_count = 0
        safe_count = 0
        for viewpoint in viewpoints:
            if not isinstance(viewpoint, dict):
                continue
            result = dict(viewpoint)
            reasons = self.reject_reasons(result, plan_kind)
            accepted = len(reasons) == 0
            result['prequalified'] = bool(accepted)
            # Compatibility aliases consumed by the unchanged bridge and
            # capture diagnostics.  This stage has not run IK or Tesseract.
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
            'mode': (
                'legacy_static_reach_check'
                if self.param_bool('enforce_static_reach_bounds')
                else 'dynamic_kinematics_deferred_to_tesseract'),
            'input_viewpoints': len(viewpoints),
            'output_viewpoints': len(filtered),
            'prequalified_viewpoints': reachable_count,
            'reachable_viewpoints': reachable_count,
            'safe_viewpoints': safe_count,
            'reachable_field_semantics': 'prequalified_compatibility_alias',
            'arm_status': self.arm_status_summary(),
            'target_status': self.target_status,
            'dry_run_config_loaded': self.param_bool('dry_run'),
        }
        output['viewpoints'] = filtered

        out = String()
        out.data = json.dumps(output, sort_keys=True)
        publisher.publish(out)

        if self.param_bool('debug'):
            self.get_logger().info(
                'prequalified %s viewpoints: %d/%d (Tesseract feasibility pending)'
                % (plan_kind, reachable_count, len(filtered))
            )

    def reject_reasons(self, viewpoint, plan_kind='MULTIVIEW_SCAN'):
        reasons = []
        if not self.param_bool('dry_run'):
            reasons.append('dry_run safety config missing or false')

        reasons.extend(self.arm_status_reasons())

        target_reason = target_status_rejection_reason(
            self.target_status, plan_kind)
        if target_reason:
            reasons.append(target_reason)

        camera_position = viewpoint.get('desired_camera_position')
        target_center = viewpoint.get('target_object_center')
        if not self.valid_vector(camera_position):
            reasons.append('missing desired camera position')
            return reasons
        if not self.valid_vector(target_center):
            reasons.append('missing target object center')
            return reasons

        if self.param_bool('enforce_static_reach_bounds'):
            reach = self.vector_norm(camera_position)
            min_reach = float(self.get_parameter('min_reach_m').value)
            max_reach = float(self.get_parameter('max_reach_m').value)
            if reach < min_reach:
                reasons.append(
                    'camera target position too close %.3fm < %.3fm'
                    % (reach, min_reach))
            if reach > max_reach:
                reasons.append(
                    'camera target position too far %.3fm > %.3fm'
                    % (reach, max_reach))

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

        if self.param_bool('enforce_static_reach_bounds'):
            height_change = abs(
                float(camera_position['z']) - float(target_center['z']))
            max_height_change = float(
                self.get_parameter('max_height_change_m').value)
            if height_change > max_height_change:
                reasons.append(
                    'height change too large %.3fm > %.3fm'
                    % (height_change, max_height_change))

        return reasons

    def publish_rejected_payload(
            self, reason, publisher=None, plan_kind='MULTIVIEW_SCAN'):
        if publisher is None:
            publisher = self.pub
        out = String()
        out.data = json.dumps(
            {
                'plan_kind': plan_kind or 'MULTIVIEW_SCAN',
                'dry_run': True,
                'filter': {
                    'node': 'viewpoint_reachability_filter_node',
                    'prequalified_viewpoints': 0,
                    'reachable_viewpoints': 0,
                    'safe_viewpoints': 0,
                    'reachable_field_semantics': (
                        'prequalified_compatibility_alias'),
                    'reject_reasons': [reason],
                    'dry_run_config_loaded': self.param_bool('dry_run'),
                },
                'viewpoints': [],
            },
            sort_keys=True,
        )
        publisher.publish(out)
        self.get_logger().warn(reason)

    def arm_status_reasons(self):
        if self.arm_status is None or self.arm_status_at is None:
            return ['arm status is missing']
        age = time.monotonic() - self.arm_status_at
        timeout = float(self.get_parameter('arm_status_timeout_sec').value)
        if age > timeout:
            return ['arm status is stale %.3fs > %.3fs' % (age, timeout)]
        reasons = []
        if int(self.arm_status.err_code) != 0:
            reasons.append('arm err_code=%d' % int(self.arm_status.err_code))
        if any((
                self.arm_status.joint_1_angle_limit,
                self.arm_status.joint_2_angle_limit,
                self.arm_status.joint_3_angle_limit,
                self.arm_status.joint_4_angle_limit,
                self.arm_status.joint_5_angle_limit,
                self.arm_status.joint_6_angle_limit)):
            reasons.append('arm reports a joint angle-limit fault')
        if any((
                self.arm_status.communication_status_joint_1,
                self.arm_status.communication_status_joint_2,
                self.arm_status.communication_status_joint_3,
                self.arm_status.communication_status_joint_4,
                self.arm_status.communication_status_joint_5,
                self.arm_status.communication_status_joint_6)):
            reasons.append('arm reports a joint communication fault')
        # Fully disabled is the required preflight state. Partial enable,
        # low-speed feedback loss, or an active FOC fault is never valid
        # planning authority.
        reasons.extend(motor_control_reasons(
            self.arm_status, require_all_enabled=False))
        return reasons

    def arm_status_summary(self):
        if self.arm_status is None or self.arm_status_at is None:
            return {'available': False}
        age = max(0.0, time.monotonic() - self.arm_status_at)
        return {
            'available': True,
            'age_sec': age,
            'err_code': int(self.arm_status.err_code),
            'angle_limit_fault': any((
                self.arm_status.joint_1_angle_limit,
                self.arm_status.joint_2_angle_limit,
                self.arm_status.joint_3_angle_limit,
                self.arm_status.joint_4_angle_limit,
                self.arm_status.joint_5_angle_limit,
                self.arm_status.joint_6_angle_limit,
            )),
            'communication_fault': any((
                self.arm_status.communication_status_joint_1,
                self.arm_status.communication_status_joint_2,
                self.arm_status.communication_status_joint_3,
                self.arm_status.communication_status_joint_4,
                self.arm_status.communication_status_joint_5,
                self.arm_status.communication_status_joint_6,
            )),
        }

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
