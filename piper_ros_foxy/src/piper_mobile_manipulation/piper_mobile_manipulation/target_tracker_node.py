#!/usr/bin/env python3
import math

import rclpy
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped conversions)
from geometry_msgs.msg import PointStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Header, String
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from piper_mobile_manipulation.msg import Target3D, TargetEstimate, TrackedTarget
from piper_mobile_manipulation.utils.kalman_filter import ConstantVelocityKalmanFilter


class TargetTrackerNode(Node):
    def __init__(self):
        super().__init__('target_tracker_node')
        self.declare_parameter('target_topic', '/piper/target_3d')
        self.declare_parameter('tracked_topic', '/piper/tracked_target')
        self.declare_parameter('target_status_topic', '/piper/target_status')
        self.declare_parameter('raw_camera_topic', '/piper/target/raw_camera')
        self.declare_parameter('raw_base_topic', '/piper/target/raw_base')
        self.declare_parameter('filtered_base_topic', '/piper/target/filtered_base')
        self.declare_parameter('predicted_base_topic', '/piper/target/predicted_base')
        self.declare_parameter('prediction_horizon_s', 0.05)
        self.declare_parameter('max_missed_frames', 10)
        self.declare_parameter('min_track_frames', 5)
        self.declare_parameter('stable_speed_threshold_mps', 0.03)
        self.declare_parameter('stable_time_s', 0.4)
        self.declare_parameter('process_noise', 0.05)
        self.declare_parameter('measurement_noise', 0.02)
        self.declare_parameter('use_tf_transform', True)
        self.declare_parameter('piper_base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('object_frame', 'tracked_object_frame')
        self.declare_parameter('publish_object_tf', True)
        self.declare_parameter('allow_latest_tf_fallback', True)
        self.declare_parameter('transform_timeout_s', 0.2)
        self.declare_parameter('min_measurement_confidence', 0.05)
        self.declare_parameter('confidence_noise_scale', 4.0)
        self.declare_parameter('depth_gate_m', 0.15)
        self.declare_parameter('max_pixel_jump', 80)
        self.declare_parameter('max_3d_jump_m', 0.10)
        self.declare_parameter('min_area_ratio', 0.5)
        self.declare_parameter('max_area_ratio', 2.0)
        self.declare_parameter('min_confidence', 0.40)
        self.declare_parameter('low_confidence_timeout_s', 0.5)
        self.declare_parameter('lost_timeout_s', 1.0)
        self.declare_parameter('debug', True)

        self.refresh_runtime_params()
        self.filter = ConstantVelocityKalmanFilter(
            self.get_parameter('process_noise').value,
            self.base_measurement_noise,
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.last_time = None
        self.track_frames = 0
        self.missed_frames = 0
        self.stable_since = None
        self.last_seen_time = None
        self.last_source_u = None
        self.last_source_v = None
        self.last_area = None
        self.last_depth = None
        self.last_measurement = None
        self.last_filtered_state = None
        self.last_predicted_state = None
        self.last_valid_header = None
        self.status = 'SEARCHING'

        self.pub = self.create_publisher(
            TrackedTarget, self.get_parameter('tracked_topic').value, 10
        )
        self.status_pub = self.create_publisher(
            String, self.get_parameter('target_status_topic').value, 10
        )
        self.raw_camera_pub = self.create_publisher(
            TargetEstimate, self.get_parameter('raw_camera_topic').value, 10)
        self.raw_base_pub = self.create_publisher(
            TargetEstimate, self.get_parameter('raw_base_topic').value, 10)
        self.filtered_base_pub = self.create_publisher(
            TargetEstimate, self.get_parameter('filtered_base_topic').value, 10)
        self.predicted_base_pub = self.create_publisher(
            TargetEstimate, self.get_parameter('predicted_base_topic').value, 10)
        self.sub = self.create_subscription(
            Target3D, self.get_parameter('target_topic').value, self.target_cb, 10
        )
        self.status_timer = self.create_timer(0.1, self.status_timer_cb)
        self.get_logger().info(
            'Target tracker ready; output_frame=%s tf=%s'
            % (self.output_frame, self.use_tf_transform)
        )

    def target_cb(self, msg):
        self.refresh_runtime_params()
        now = self.get_clock().now()
        out = TrackedTarget()
        out.header = msg.header
        camera_measurement = [msg.point.x, msg.point.y, msg.point.z]
        self.publish_estimate(
            self.raw_camera_pub, msg.header, camera_measurement, [0.0, 0.0, 0.0],
            float(msg.measurement_confidence), bool(msg.valid), False,
            'raw depth and SAM2 mask' if msg.valid else 'invalid depth measurement')

        if not msg.valid:
            self.publish_invalid(out)
            return

        measurement_confidence = float(msg.measurement_confidence)
        if measurement_confidence < self.min_measurement_confidence:
            self.get_logger().warn(
                'Target3D rejected low confidence %.2f < %.2f'
                % (measurement_confidence, self.min_measurement_confidence)
            )
            self.publish_invalid(out)
            return

        measurement, transform_reason = self.measurement_in_output_frame(msg)
        if measurement is None:
            failed_header = self.output_header(msg.header)
            self.publish_estimate(
                self.raw_base_pub, failed_header, self.last_position_or_zero(),
                [0.0, 0.0, 0.0], 0.0, False, False,
                transform_reason or 'TF transform failed')
            self.publish_invalid(out, transform_reason or 'TF transform failed')
            return
        base_header = self.output_header(msg.header)
        self.publish_estimate(
            self.raw_base_pub, base_header, measurement, [0.0, 0.0, 0.0],
            measurement_confidence, True, False, transform_reason)
        gate_reason = self.gate_measurement(msg, measurement, measurement_confidence)
        if gate_reason:
            self.get_logger().warn('Target3D rejected by tracker gate: %s' % gate_reason)
            self.publish_invalid(out, gate_reason)
            return

        if self.last_time is None:
            dt = 0.033
        else:
            dt = max((now - self.last_time).nanoseconds * 1e-9, 1e-3)
        self.last_time = now
        self.filter.measurement_noise = self.scaled_measurement_noise(measurement_confidence)
        state = self.filter.step(measurement, dt)
        self.track_frames += 1
        self.missed_frames = 0
        self.last_seen_time = now
        self.last_source_u = float(msg.source_u)
        self.last_source_v = float(msg.source_v)
        self.last_area = self.detection_area(msg)
        self.last_depth = float(msg.depth)
        self.last_measurement = measurement

        x, y, z, vx, vy, vz = state
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        stable_now = speed <= self.stable_speed_threshold
        if stable_now:
            if self.stable_since is None:
                self.stable_since = now
            stable_duration = (now - self.stable_since).nanoseconds * 1e-9
        else:
            self.stable_since = None
            stable_duration = 0.0

        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.output_frame if self.use_tf_transform else msg.header.frame_id
        out.position.x = float(x)
        out.position.y = float(y)
        out.position.z = float(z)
        out.velocity.x = float(vx)
        out.velocity.y = float(vy)
        out.velocity.z = float(vz)
        out.predicted_position.x = float(x + vx * self.prediction_horizon)
        out.predicted_position.y = float(y + vy * self.prediction_horizon)
        out.predicted_position.z = float(z + vz * self.prediction_horizon)
        out.speed = float(speed)
        track_confidence = float(min(
            1.0, self.track_frames / float(max(self.min_track_frames, 1))))
        out.confidence = float(track_confidence * measurement_confidence)
        out.stable = (
            self.track_frames >= self.min_track_frames
            and stable_now
            and stable_duration >= self.stable_time_s
        )
        out.valid = self.track_frames >= self.min_track_frames
        self.pub.publish(out)
        filtered_state = [x, y, z, vx, vy, vz]
        predicted_state = [
            out.predicted_position.x, out.predicted_position.y, out.predicted_position.z,
            vx, vy, vz,
        ]
        self.last_filtered_state = [float(value) for value in filtered_state]
        self.last_predicted_state = [float(value) for value in predicted_state]
        self.last_valid_header = out.header
        self.publish_estimate(
            self.filtered_base_pub, out.header, [x, y, z], [vx, vy, vz],
            out.confidence, out.valid, False,
            'constant-velocity Kalman estimate')
        self.publish_estimate(
            self.predicted_base_pub, out.header,
            [out.predicted_position.x, out.predicted_position.y, out.predicted_position.z],
            [vx, vy, vz], out.confidence, out.valid, True,
            'Kalman prediction %.3fs ahead' % self.prediction_horizon)
        if out.valid:
            self.publish_object_tf(out.header, [
                out.predicted_position.x,
                out.predicted_position.y,
                out.predicted_position.z,
            ])
        self.publish_status('LOCKED' if out.stable else 'TRACKING')
        self.get_logger().info(
            'TrackedTarget frame=%s valid=%s stable=%s pos=(%.3f, %.3f, %.3f) speed=%.3f conf=%.2f'
            % (out.header.frame_id, out.valid, out.stable, x, y, z, speed, out.confidence)
        )

    def publish_invalid(self, out, reason=None):
        self.missed_frames += 1
        self.update_status_from_timeout()
        if self.missed_frames > self.max_missed:
            self.filter.reset()
            self.track_frames = 0
            self.stable_since = None
            self.last_time = None
        out.valid = False
        out.confidence = 0.0
        if self.use_tf_transform:
            out.header.frame_id = self.output_frame
        stale_age_s = self.stale_age_seconds()
        invalid_reason = str(reason or self.status)
        if self.last_predicted_state is not None:
            invalid_reason = '%s; stale_age_s=%.3f' % (invalid_reason, stale_age_s)
        if self.last_predicted_state is not None:
            out.header = self.output_header(self.last_valid_header or out.header)
            out.position.x = float(self.last_filtered_state[0])
            out.position.y = float(self.last_filtered_state[1])
            out.position.z = float(self.last_filtered_state[2])
            out.velocity.x = float(self.last_filtered_state[3])
            out.velocity.y = float(self.last_filtered_state[4])
            out.velocity.z = float(self.last_filtered_state[5])
            out.predicted_position.x = float(self.last_predicted_state[0])
            out.predicted_position.y = float(self.last_predicted_state[1])
            out.predicted_position.z = float(self.last_predicted_state[2])
            out.speed = math.sqrt(
                out.velocity.x * out.velocity.x
                + out.velocity.y * out.velocity.y
                + out.velocity.z * out.velocity.z
            )
        self.pub.publish(out)
        if self.last_predicted_state is not None and self.last_valid_header is not None:
            header = self.output_header(self.last_valid_header)
        else:
            header = self.output_header(out.header)
        filtered_position = self.last_position_or_zero(filtered=True)
        filtered_velocity = self.last_velocity_or_zero(filtered=True)
        predicted_position = self.last_position_or_zero(filtered=False)
        predicted_velocity = self.last_velocity_or_zero(filtered=False)
        self.publish_estimate(
            self.filtered_base_pub, header, filtered_position, filtered_velocity,
            0.0, False, False, invalid_reason)
        self.publish_estimate(
            self.predicted_base_pub, header, predicted_position, predicted_velocity,
            0.0, False, True, invalid_reason)
        self.publish_status(self.status)

    def publish_estimate(self, publisher, header, position, velocity, confidence,
                         valid, predicted, reason):
        estimate = TargetEstimate()
        estimate.header = header
        estimate.pose.pose.position.x = float(position[0])
        estimate.pose.pose.position.y = float(position[1])
        estimate.pose.pose.position.z = float(position[2])
        estimate.pose.pose.orientation.w = 1.0
        variance = max(float(self.base_measurement_noise), 1e-6)
        estimate.pose.covariance[0] = variance
        estimate.pose.covariance[7] = variance
        estimate.pose.covariance[14] = variance
        estimate.velocity.x = float(velocity[0])
        estimate.velocity.y = float(velocity[1])
        estimate.velocity.z = float(velocity[2])
        estimate.confidence = float(confidence)
        estimate.prediction_horizon_s = self.prediction_horizon if predicted else 0.0
        estimate.predicted = bool(predicted)
        estimate.valid = bool(valid)
        estimate.reason = str(reason)
        publisher.publish(estimate)

    def gate_measurement(self, msg, measurement, confidence):
        if confidence < self.min_confidence:
            return 'confidence %.2f < %.2f' % (confidence, self.min_confidence)
        if (self.last_depth is not None and
                abs(float(msg.depth) - self.last_depth) > self.depth_gate_m):
            return 'depth %.3f outside gate around %.3f +/- %.3f' % (
                float(msg.depth),
                self.last_depth,
                self.depth_gate_m,
            )
        if self.last_source_u is not None and self.last_source_v is not None:
            du = float(msg.source_u) - self.last_source_u
            dv = float(msg.source_v) - self.last_source_v
            pixel_jump = math.sqrt(du * du + dv * dv)
            if pixel_jump > self.max_pixel_jump:
                return 'pixel jump %.1f > %.1f' % (pixel_jump, self.max_pixel_jump)
        if self.last_measurement is not None:
            jump = math.sqrt(
                (measurement[0] - self.last_measurement[0]) ** 2
                + (measurement[1] - self.last_measurement[1]) ** 2
                + (measurement[2] - self.last_measurement[2]) ** 2
            )
            if jump > self.max_3d_jump:
                return '3d jump %.3f > %.3f' % (jump, self.max_3d_jump)
        area = self.detection_area(msg)
        if self.last_area is not None and self.last_area > 0.0 and area > 0.0:
            ratio = area / self.last_area
            if ratio < self.min_area_ratio or ratio > self.max_area_ratio:
                return 'area ratio %.2f outside %.2f..%.2f' % (
                    ratio,
                    self.min_area_ratio,
                    self.max_area_ratio,
                )
        return None

    @staticmethod
    def detection_area(msg):
        return max(float(msg.detection_width), 0.0) * max(float(msg.detection_height), 0.0)

    def status_timer_cb(self):
        self.update_status_from_timeout()
        self.publish_status(self.status)

    def update_status_from_timeout(self):
        if self.last_seen_time is None:
            self.status = 'SEARCHING'
            return
        age = (self.get_clock().now() - self.last_seen_time).nanoseconds * 1e-9
        if age >= self.lost_timeout_s:
            self.status = 'LOST'
        elif age >= self.low_confidence_timeout_s:
            self.status = 'LOW_CONFIDENCE'
        elif self.track_frames >= self.min_track_frames:
            self.status = 'TRACKING'
        else:
            self.status = 'LOCKED'

    def publish_status(self, status):
        self.status = status
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    def measurement_in_output_frame(self, msg):
        if not self.use_tf_transform:
            return [msg.point.x, msg.point.y, msg.point.z], 'TF disabled; camera-frame measurement'

        point = PointStamped()
        point.header = msg.header
        if not point.header.frame_id:
            point.header.frame_id = self.camera_frame
        point.point = msg.point
        try:
            transformed = self.tf_buffer.transform(
                point,
                self.output_frame,
                timeout=Duration(seconds=self.transform_timeout_s),
            )
            reason = 'timestamped TF transform'
        except TransformException as exc:
            exact_error = str(exc)
            if not self.allow_latest_tf_fallback:
                self.get_logger().warn(
                    'TF failed %s -> %s: %s'
                    % (point.header.frame_id, self.output_frame, exact_error)
                )
                return None, 'TF failed %s -> %s: %s' % (
                    point.header.frame_id, self.output_frame, exact_error)
            latest_point = PointStamped()
            latest_point.header.frame_id = point.header.frame_id
            latest_point.header.stamp = rclpy.time.Time().to_msg()
            latest_point.point = point.point
            try:
                transformed = self.tf_buffer.transform(
                    latest_point,
                    self.output_frame,
                    timeout=Duration(seconds=self.transform_timeout_s),
                )
                reason = 'timestamped TF failed; used latest TF'
                if self.debug:
                    self.get_logger().warn(
                        'TF exact failed %s -> %s, using latest TF: %s'
                        % (point.header.frame_id, self.output_frame, exact_error)
                    )
            except TransformException as latest_exc:
                self.get_logger().warn(
                    'TF failed %s -> %s exact=%s latest=%s'
                    % (point.header.frame_id, self.output_frame,
                       exact_error, str(latest_exc))
                )
                return None, 'TF failed %s -> %s exact=%s latest=%s' % (
                    point.header.frame_id, self.output_frame,
                    exact_error, str(latest_exc))
        return [
            transformed.point.x,
            transformed.point.y,
            transformed.point.z,
        ], reason

    def publish_object_tf(self, header, position):
        if not self.publish_object_tf_enabled:
            return
        transform = TransformStamped()
        transform.header = self.output_header(header)
        transform.child_frame_id = self.object_frame
        transform.transform.translation.x = float(position[0])
        transform.transform.translation.y = float(position[1])
        transform.transform.translation.z = float(position[2])
        transform.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(transform)

    def output_header(self, source_header):
        header = Header()
        header.stamp = source_header.stamp
        header.frame_id = self.output_frame
        return header

    def last_position_or_zero(self, filtered=False):
        state = self.last_filtered_state if filtered else self.last_predicted_state
        if state is None:
            return [0.0, 0.0, 0.0]
        return [float(state[0]), float(state[1]), float(state[2])]

    def last_velocity_or_zero(self, filtered=False):
        state = self.last_filtered_state if filtered else self.last_predicted_state
        if state is None:
            return [0.0, 0.0, 0.0]
        return [float(state[3]), float(state[4]), float(state[5])]

    def stale_age_seconds(self):
        if self.last_seen_time is None:
            return 0.0
        return max((self.get_clock().now() - self.last_seen_time).nanoseconds * 1e-9, 0.0)

    def scaled_measurement_noise(self, confidence):
        confidence = max(float(confidence), self.min_measurement_confidence)
        confidence = min(confidence, 1.0)
        return self.base_measurement_noise * (
            1.0 + (1.0 - confidence) * self.confidence_noise_scale)

    def refresh_runtime_params(self):
        self.prediction_horizon = float(self.get_parameter('prediction_horizon_s').value)
        self.max_missed = int(self.get_parameter('max_missed_frames').value)
        self.min_track_frames = int(self.get_parameter('min_track_frames').value)
        self.stable_speed_threshold = float(self.get_parameter('stable_speed_threshold_mps').value)
        self.stable_time_s = float(self.get_parameter('stable_time_s').value)
        self.use_tf_transform = bool(self.get_parameter('use_tf_transform').value)
        self.output_frame = self.get_parameter('piper_base_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.object_frame = self.get_parameter('object_frame').value
        self.publish_object_tf_enabled = bool(self.get_parameter('publish_object_tf').value)
        self.allow_latest_tf_fallback = bool(
            self.get_parameter('allow_latest_tf_fallback').value)
        self.transform_timeout_s = float(self.get_parameter('transform_timeout_s').value)
        self.min_measurement_confidence = float(
            self.get_parameter('min_measurement_confidence').value)
        self.confidence_noise_scale = float(self.get_parameter('confidence_noise_scale').value)
        self.base_measurement_noise = float(self.get_parameter('measurement_noise').value)
        self.depth_gate_m = float(self.get_parameter('depth_gate_m').value)
        self.max_pixel_jump = float(self.get_parameter('max_pixel_jump').value)
        self.max_3d_jump = float(self.get_parameter('max_3d_jump_m').value)
        self.min_area_ratio = float(self.get_parameter('min_area_ratio').value)
        self.max_area_ratio = float(self.get_parameter('max_area_ratio').value)
        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.low_confidence_timeout_s = float(self.get_parameter('low_confidence_timeout_s').value)
        self.lost_timeout_s = float(self.get_parameter('lost_timeout_s').value)
        self.debug = bool(self.get_parameter('debug').value)


def main(args=None):
    rclpy.init(args=args)
    node = TargetTrackerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
