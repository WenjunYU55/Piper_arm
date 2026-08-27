#!/usr/bin/env python3
"""Build command-free camera acquisition viewpoints from a rough target hint."""

import json
import re
import time

import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from piper_mobile_manipulation.srv import (
    PrepareAcquisition,
    RequestMotionPlan,
)
from piper_mobile_manipulation.perception.acquisition import (
    ROUGH_ACQUISITION,
    build_acquisition_viewpoints,
    rough_hint_rejection_reason,
    viewpoint_payload_matches,
)
from piper_mobile_manipulation.utils.mask_reprojection import quaternion_matrix

SESSION_ID_PATTERN = re.compile(r'^[A-Za-z0-9_.:-]{8,128}$')


class ScanTargetAcquisitionNode(Node):
    """Publish acquisition proposals only after an explicit start request."""

    def __init__(self):
        super().__init__('scan_target_acquisition')
        defaults = {
            'acquisition_viewpoints_topic': '/piper/acquisition_viewpoints',
            'reachable_acquisition_viewpoints_topic':
                '/piper/reachable_acquisition_viewpoints',
            'request_acquisition_plan_service':
                '/motion_planner/request_acquisition_plan',
            'base_frame': 'base_link',
            'camera_optical_frame': 'camera_color_optical_frame',
            'hint_max_age_sec': 5.0,
            'future_tolerance_sec': 0.1,
            'transform_timeout_sec': 0.25,
            'standoff_m': 0.45,
            'fallback_standoff_m': 0.28,
            'acquisition_camera_pitch_deg': -10.0,
            'sweep_angle_deg': 45.0,
            'handoff_retry_sec': 0.50,
            # Camera-clock health can briefly flap after every process reports
            # ready. Keep the exact session payload fresh long enough for one
            # bounded recovery instead of turning a transient rejection into
            # a permanently stale acquisition request.
            'handoff_timeout_sec': 30.0,
            'dry_run': True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.active_session_id = ''
        self.active_request_signature = None
        self.pending_acquisition_stamp_ns = None
        self.pending_acquisition_payload_ready = False
        self.acquisition_request_sent = False
        self.acquisition_bridge_request_id = ''
        self.pending_acquisition_message = None
        self.acquisition_handoff_started_at = None
        self.last_acquisition_publish_at = -float('inf')
        self.last_acquisition_request_at = -float('inf')
        self.acquisition_handoff_timeout_reported = False
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.publisher = self.create_publisher(
            String, self.get_parameter('acquisition_viewpoints_topic').value, 10)
        self.reachable_acquisition_subscription = self.create_subscription(
            String,
            self.get_parameter('reachable_acquisition_viewpoints_topic').value,
            self.reachable_acquisition_cb,
            10,
        )
        self.acquisition_plan_client = self.create_client(
            RequestMotionPlan,
            self.get_parameter('request_acquisition_plan_service').value,
        )
        self.prepare_service = self.create_service(
            PrepareAcquisition, '~/prepare', self.prepare_cb)
        self.request_timer = self.create_timer(0.25, self.handoff_tick)
        self.get_logger().warn(
            'Rough-target acquisition is command-free; one typed prepare request '
            'atomically binds the session ID and rough coordinate.')

    def validate_hint(self, msg):
        return rough_hint_rejection_reason(
            msg.header.frame_id,
            (msg.point.x, msg.point.y, msg.point.z),
            self.stamp_ns(msg),
            self.get_clock().now().nanoseconds,
            self.get_parameter('hint_max_age_sec').value,
            self.get_parameter('future_tolerance_sec').value,
        )

    @staticmethod
    def request_signature(session_id, look_index, hint=None):
        if hint is None:
            hint = look_index
            look_index = 0
        return (
            str(session_id),
            int(look_index),
            str(hint.header.frame_id),
            float(hint.point.x),
            float(hint.point.y),
            float(hint.point.z),
        )

    def prepare_cb(self, request, response):
        session_id = str(request.session_id)
        response.session_id = session_id
        if SESSION_ID_PATTERN.fullmatch(session_id) is None:
            response.accepted = False
            response.message = (
                'session_id must contain 8..128 letters, digits, _, ., :, or -')
            return response
        hint = request.rough_target
        look_index = int(getattr(request, 'look_index', 0))
        if look_index < 0 or look_index >= 5:
            response.accepted = False
            response.message = 'look_index must be from 0 through 4'
            return response
        signature = self.request_signature(session_id, look_index, hint)
        if session_id == self.active_session_id:
            if signature != self.active_request_signature:
                response.accepted = False
                response.message = (
                    'session_id is already bound to a different rough target')
                return response
            response.accepted = True
            response.message = (
                'duplicate prepare request accepted idempotently; existing '
                'acquisition remains active')
            return response
        reason = self.validate_hint(hint)
        if reason:
            response.accepted = False
            response.message = reason
            return response
        if not self.param_bool('dry_run'):
            response.accepted = False
            response.message = 'dry_run safety config missing or false'
            return response

        try:
            transform = self.tf_buffer.lookup_transform(
                self.get_parameter('base_frame').value,
                self.get_parameter('camera_optical_frame').value,
                Time(),
                timeout=Duration(
                    seconds=float(
                        self.get_parameter('transform_timeout_sec').value)),
            )
            camera = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ]
            rotation = transform.transform.rotation
            camera_look = quaternion_matrix(
                rotation.x, rotation.y, rotation.z, rotation.w).dot(
                    [0.0, 0.0, 1.0]).tolist()
            target = [
                hint.point.x,
                hint.point.y,
                hint.point.z,
            ]
            viewpoints = build_acquisition_viewpoints(
                target,
                camera,
                self.get_parameter('standoff_m').value,
                self.get_parameter('acquisition_camera_pitch_deg').value,
                self.get_parameter('sweep_angle_deg').value,
                self.get_parameter('fallback_standoff_m').value,
                camera_look,
                look_index,
            )
        except (TransformException, ValueError) as exc:
            response.accepted = False
            response.message = 'cannot build acquisition viewpoints: %s' % exc
            return response

        stamp_ns = self.stamp_ns(hint)
        stamp = {
            'sec': int(hint.header.stamp.sec),
            'nanosec': int(hint.header.stamp.nanosec),
        }
        payload = {
            'header': {
                'stamp': stamp,
                'frame_id': 'base_link',
            },
            'plan_kind': ROUGH_ACQUISITION,
            'source_request_id': session_id,
            'dry_run': True,
            'source_service': '/scan_target_acquisition/prepare',
            'acquisition_transaction_index': look_index,
            'rough_hint_stamp_ns': stamp_ns,
            'target_provenance': {
                'source': 'rough_coordinate',
                'source_request_id': session_id,
                'frame_id': 'base_link',
                'stamp': stamp,
            },
            'target_status': 'UNOBSERVED',
            'target_object_center': {
                'x': float(target[0]),
                'y': float(target[1]),
                'z': float(target[2]),
            },
            'acquisition_pattern': {
                'order': [
                    item['acquisition_look'] for item in viewpoints],
                'sweep_angle_deg': abs(float(
                    self.get_parameter('sweep_angle_deg').value)),
                'maximum_standoff_m': float(
                    self.get_parameter('standoff_m').value),
                'fallback_standoff_m': float(
                    self.get_parameter('fallback_standoff_m').value),
                'effective_standoff_m': float(
                    viewpoints[0]['camera_object_distance_m']),
            },
            'viewpoints': viewpoints,
        }
        output = String()
        output.data = json.dumps(payload, sort_keys=True, allow_nan=False)
        self.active_session_id = session_id
        self.active_request_signature = signature
        self.pending_acquisition_stamp_ns = stamp_ns
        self.pending_acquisition_payload_ready = False
        self.acquisition_request_sent = False
        self.acquisition_bridge_request_id = ''
        self.pending_acquisition_message = output
        self.acquisition_handoff_started_at = time.monotonic()
        self.last_acquisition_publish_at = self.acquisition_handoff_started_at
        self.last_acquisition_request_at = -float('inf')
        self.acquisition_handoff_timeout_reported = False
        self.publisher.publish(output)
        response.accepted = True
        response.message = (
            'queued %d command-free rough-acquisition viewpoints for %s; '
            'motion-planner proposal generation is asynchronous'
            % (len(viewpoints), session_id))
        return response

    def reachable_acquisition_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            return
        if self.pending_acquisition_stamp_ns is None:
            return
        if viewpoint_payload_matches(
                payload, ROUGH_ACQUISITION,
                self.pending_acquisition_stamp_ns) \
                and payload.get('source_request_id') == self.active_session_id:
            self.pending_acquisition_payload_ready = True
            self.submit_ready_requests()

    def submit_ready_requests(self):
        now = time.monotonic()
        retry = float(self.get_parameter('handoff_retry_sec').value)
        if (
                self.pending_acquisition_payload_ready
                and not self.acquisition_request_sent
                and now - self.last_acquisition_request_at >= retry
                and self.acquisition_plan_client.service_is_ready()):
            self.acquisition_request_sent = True
            self.last_acquisition_request_at = now
            request = RequestMotionPlan.Request()
            request.force_refresh = False
            future = self.acquisition_plan_client.call_async(request)
            future.add_done_callback(
                lambda result: self.log_plan_request_result(
                    result, ROUGH_ACQUISITION))

    def handoff_tick(self):
        """Retry the command-free Foxy handoff without duplicating a plan."""
        now = time.monotonic()
        if (
                self.pending_acquisition_message is not None
                and not self.acquisition_request_sent):
            started = self.acquisition_handoff_started_at
            timeout = float(self.get_parameter('handoff_timeout_sec').value)
            retry = float(self.get_parameter('handoff_retry_sec').value)
            if started is not None and now - started <= timeout:
                if now - self.last_acquisition_publish_at >= retry:
                    self.publisher.publish(self.pending_acquisition_message)
                    self.last_acquisition_publish_at = now
            elif not self.acquisition_handoff_timeout_reported:
                self.acquisition_handoff_timeout_reported = True
                self.pending_acquisition_message = None
                self.pending_acquisition_payload_ready = False
                self.acquisition_request_sent = False
                self.get_logger().error(
                    'ROUGH_ACQUISITION handoff timed out before planner '
                    'accepted a command-free request')
        self.submit_ready_requests()

    def log_plan_request_result(self, future, plan_kind):
        try:
            response = future.result()
            message = '%s request: %s' % (plan_kind, response.message)
            # Foxy's Python logger keys call sites by source line and rejects a
            # later severity change at that same line. Keep the INFO and WARN
            # calls on distinct lines because this callback legitimately sees
            # rejected retries followed by one accepted request.
            if response.accepted:
                self.get_logger().info(message)
            else:
                self.get_logger().warn(message)
            if plan_kind == ROUGH_ACQUISITION:
                if response.accepted:
                    self.acquisition_bridge_request_id = str(
                        getattr(response, 'request_id', ''))
                    self.pending_acquisition_message = None
                else:
                    # A rejected call queued no plan. Permit the bounded timer
                    # to republish current inputs and retry with one call in
                    # flight at a time.
                    self.acquisition_request_sent = False
        except Exception as exc:  # rclpy futures surface transport errors here.
            if plan_kind == ROUGH_ACQUISITION:
                self.acquisition_request_sent = False
            self.get_logger().error(
                '%s planning service failed: %s' % (plan_kind, exc))

    @staticmethod
    def stamp_ns(msg):
        return (
            int(msg.header.stamp.sec) * 1000000000
            + int(msg.header.stamp.nanosec)
        )

    def param_bool(self, name):
        value = self.get_parameter(name).value
        if isinstance(value, str):
            return value.lower() in ('1', 'true', 'yes', 'on')
        return bool(value)


def main(args=None):
    rclpy.init(args=args)
    node = ScanTargetAcquisitionNode()
    # The start service performs a bounded lookup of the current camera
    # transform. Give TransformListener another executor thread so /tf and
    # /tf_static can continue filling the buffer while that lookup waits.
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
