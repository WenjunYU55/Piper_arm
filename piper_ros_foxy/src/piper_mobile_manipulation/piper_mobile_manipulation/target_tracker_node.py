#!/usr/bin/env python3
import math

import rclpy
# Importing this module registers geometry message conversions with tf2.
import tf2_geometry_msgs  # noqa: F401
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from piper_mobile_manipulation.msg import Target3D, TrackedTarget
from piper_mobile_manipulation.utils.kalman_filter import ConstantVelocityKalmanFilter


def finite_target_measurement(measurement):
    """Return whether one transformed 3-D observation is safe to filter."""
    try:
        return bool(
            len(measurement) == 3
            and all(math.isfinite(float(value)) for value in measurement))
    except (TypeError, ValueError):
        return False


def prediction_only_is_valid(
        initialized, track_frames, minimum_track_frames,
        measurement_age_sec, lost_timeout_sec):
    """Return whether a bounded prediction may remain a usable estimate."""
    return bool(
        initialized
        and int(track_frames) >= int(minimum_track_frames)
        and math.isfinite(float(measurement_age_sec))
        and 0.0 <= float(measurement_age_sec) < float(lost_timeout_sec)
    )


class TargetTrackerNode(Node):
    def __init__(self):
        super().__init__('target_tracker_node')
        self.declare_parameter('target_topic', '/piper/target_3d')
        self.declare_parameter('tracked_topic', '/piper/tracked_target')
        self.declare_parameter('target_status_topic', '/piper/target_status')
        self.declare_parameter('prediction_horizon_s', 0.3)
        self.declare_parameter('max_missed_frames', 10)
        self.declare_parameter('min_track_frames', 5)
        self.declare_parameter('stable_speed_threshold_mps', 0.03)
        self.declare_parameter('stable_time_s', 0.4)
        self.declare_parameter('process_noise', 0.01)
        self.declare_parameter('measurement_noise', 0.02)
        self.declare_parameter('use_tf_transform', True)
        self.declare_parameter('piper_base_frame', 'piper_base_link')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('transform_timeout_s', 0.2)
        self.declare_parameter('min_measurement_confidence', 0.05)
        self.declare_parameter('confidence_noise_scale', 4.0)
        self.declare_parameter('depth_gate_m', 0.15)
        self.declare_parameter('max_pixel_jump', 80)
        self.declare_parameter('max_3d_jump_m', 0.10)
        self.declare_parameter('min_area_ratio', 0.5)
        self.declare_parameter('max_area_ratio', 2.0)
        self.declare_parameter('use_camera_space_gates', False)
        self.declare_parameter('max_target_speed_mps', 1.0)
        self.declare_parameter('innovation_gate_threshold', 5.0)
        self.declare_parameter('reject_out_of_order_measurements', True)
        self.declare_parameter('min_confidence', 0.40)
        self.declare_parameter('low_confidence_timeout_s', 0.5)
        self.declare_parameter('lost_timeout_s', 5.0)
        self.declare_parameter('debug', True)

        self.refresh_runtime_params()
        self.filter = ConstantVelocityKalmanFilter(
            self.get_parameter('process_noise').value,
            self.base_measurement_noise,
            velocity_retention=0.0,
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.last_time = None
        self.filter_time = None
        self.track_frames = 0
        self.missed_frames = 0
        self.stable_since = None
        self.last_seen_time = None
        self.last_source_u = None
        self.last_source_v = None
        self.last_area = None
        self.last_depth = None
        self.last_measurement = None
        self.last_measurement_confidence = 0.0
        self.prediction_only = False
        self.last_stable = False
        self.status = 'SEARCHING'

        self.pub = self.create_publisher(
            TrackedTarget, self.get_parameter('tracked_topic').value, 10
        )
        self.status_pub = self.create_publisher(
            String, self.get_parameter('target_status_topic').value, 10
        )
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
        measurement_time = self.measurement_time(msg, now)
        out = TrackedTarget()
        out.header = msg.header

        if (
            self.reject_out_of_order_measurements
            and self.filter_time is not None
            and measurement_time.nanoseconds <= self.filter_time.nanoseconds
        ):
            self.get_logger().warn(
                'Ignoring out-of-order Target3D sample stamp=%d filter=%d'
                % (measurement_time.nanoseconds, self.filter_time.nanoseconds)
            )
            return

        if not msg.valid:
            self.publish_prediction_only(
                out, measurement_time, now,
                str(msg.depth_source or 'no_valid_target_measurement'))
            return

        measurement_confidence = float(msg.measurement_confidence)
        if measurement_confidence < self.min_measurement_confidence:
            self.publish_prediction_only(
                out, measurement_time, now,
                'measurement confidence %.2f < %.2f' % (
                    measurement_confidence,
                    self.min_measurement_confidence))
            return

        measurement = self.measurement_in_output_frame(msg)
        if measurement is None:
            self.publish_prediction_only(
                out, measurement_time, now,
                'measurement frame transform unavailable')
            return
        if not finite_target_measurement(measurement):
            self.publish_prediction_only(
                out, measurement_time, now,
                'transformed measurement is non-finite')
            return

        if self.status == 'LOST' and self.filter.initialized:
            self.reset_tracking_state()

        gate_reason = self.gate_measurement(
            msg, measurement, measurement_confidence
        )
        if gate_reason:
            self.publish_prediction_only(
                out, measurement_time, now, gate_reason)
            return

        measurement_noise = self.scaled_measurement_noise(
            measurement_confidence)
        innovation = [0.0, 0.0, 0.0]
        innovation_score = 0.0
        if self.filter.initialized:
            self.predict_to(measurement_time)
            innovation, _covariance, innovation_score = \
                self.filter.innovation(
                    # Confidence-scaled noise controls correction strength, but
                    # must not make the semantic innovation gate increasingly
                    # permissive as confidence falls.  Gate against the base
                    # sensor model and use the scaled value only for update.
                    measurement,
                    measurement_noise=self.base_measurement_noise)
            gate_reason = self.gate_measurement(
                msg, measurement, measurement_confidence,
                innovation_score=innovation_score,
                innovation_threshold=self.innovation_gate_threshold)
            if gate_reason:
                self.publish_prediction_only(
                    out, measurement_time, now, gate_reason,
                    already_predicted=True, measurement=measurement,
                    innovation=innovation,
                    innovation_score=innovation_score)
                return

        self.filter.measurement_noise = measurement_noise
        if not self.filter.initialized:
            self.filter.initialize(measurement)
            self.filter_time = measurement_time
        else:
            self.filter.update(
                measurement, measurement_noise=measurement_noise)
        self.last_time = measurement_time
        state = self.filter.state
        self.track_frames += 1
        self.missed_frames = 0
        self.last_seen_time = now
        self.last_source_u = float(msg.source_u)
        self.last_source_v = float(msg.source_v)
        self.last_area = self.detection_area(msg)
        self.last_depth = float(msg.depth)
        self.last_measurement = measurement
        self.last_measurement_confidence = measurement_confidence
        self.prediction_only = False

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
            1.0,
            self.track_frames / float(max(self.min_track_frames, 1))))
        out.confidence = float(track_confidence * measurement_confidence)
        out.stable = (
            self.track_frames >= self.min_track_frames
            and stable_now
            and stable_duration >= self.stable_time_s
        )
        self.last_stable = bool(out.stable)
        out.valid = self.track_frames >= self.min_track_frames
        self.pub.publish(out)
        self.publish_status('LOCKED' if out.stable else 'TRACKING')
        self.get_logger().info(
            'Target measurement accepted frame=%s valid=%s stable=%s '
            'pos=(%.3f, %.3f, %.3f) innovation=%.4fm d2=%.3f '
            'position_stddev_max=%.4fm conf=%.2f'
            % (
                out.header.frame_id, out.valid, out.stable, x, y, z,
                math.sqrt(sum(float(value) ** 2 for value in innovation)),
                innovation_score, self.filter.maximum_position_stddev,
                out.confidence,
            )
        )

    @staticmethod
    def measurement_time(msg, fallback):
        stamp = msg.header.stamp
        if int(stamp.sec) == 0 and int(stamp.nanosec) == 0:
            return fallback
        return Time.from_msg(stamp)

    def publish_invalid(self, out):
        """Compatibility entry point for an unavailable measurement."""
        now = self.get_clock().now()
        self.publish_prediction_only(
            out, self.measurement_time(out, now), now,
            'no_valid_target_measurement')

    def publish_prediction_only(
            self, out, measurement_time, now, reason,
            already_predicted=False, measurement=None, innovation=None,
            innovation_score=None):
        """Publish a bounded prediction without correcting from bad vision."""
        self.missed_frames += 1
        if not already_predicted:
            self.predict_to(measurement_time)
        measurement_age = self.measurement_age(now)
        usable = prediction_only_is_valid(
            self.filter.initialized,
            self.track_frames,
            self.min_track_frames,
            measurement_age,
            self.lost_timeout_s,
        )
        out.header.stamp = Time(
            nanoseconds=measurement_time.nanoseconds).to_msg()
        if self.use_tf_transform:
            out.header.frame_id = self.output_frame
        if usable:
            x, y, z, vx, vy, vz = self.filter.state
            out.position.x = float(x)
            out.position.y = float(y)
            out.position.z = float(z)
            out.velocity.x = float(vx)
            out.velocity.y = float(vy)
            out.velocity.z = float(vz)
            out.predicted_position.x = float(x + vx * self.prediction_horizon)
            out.predicted_position.y = float(y + vy * self.prediction_horizon)
            out.predicted_position.z = float(z + vz * self.prediction_horizon)
            out.speed = float(math.sqrt(vx * vx + vy * vy + vz * vz))
            remaining = max(
                0.0, 1.0 - measurement_age / self.lost_timeout_s)
            track_confidence = min(
                1.0,
                self.track_frames / float(max(self.min_track_frames, 1)))
            out.confidence = float(
                track_confidence
                * self.last_measurement_confidence
                * remaining)
            out.stable = False
            out.valid = True
            self.prediction_only = True
            self.last_stable = False
            self.status = 'LOW_CONFIDENCE'
        else:
            out.valid = False
            out.stable = False
            out.confidence = 0.0
            self.prediction_only = False
            self.status = 'LOST' if self.last_seen_time is not None \
                else 'SEARCHING'
        self.pub.publish(out)
        self.publish_status(self.status)
        innovation_magnitude = math.nan
        if innovation is not None:
            innovation_magnitude = math.sqrt(sum(
                float(value) ** 2 for value in innovation))
        measurement_text = 'none' if measurement is None else str([
            round(float(value), 6) for value in measurement])
        score_text = 'none' if innovation_score is None \
            else '%.3f' % float(innovation_score)
        predicted = None
        if self.filter.initialized:
            predicted = [
                round(float(value), 6)
                for value in self.filter.state[:3]]
        self.get_logger().warn(
            'Target measurement rejected reason=%s prediction_only=%s '
            'predicted=%s measurement=%s innovation_m=%.4f d2=%s '
            'measurement_age=%.3fs position_stddev_max=%.4fm'
            % (
                reason, usable, predicted, measurement_text,
                innovation_magnitude, score_text, measurement_age,
                self.filter.maximum_position_stddev,
            )
        )
        expired = (
            self.filter.initialized
            and math.isfinite(measurement_age)
            and measurement_age >= self.lost_timeout_s
        )
        if expired:
            self.reset_tracking_state()

    def predict_to(self, target_time):
        """Advance the one authoritative filter to a source timestamp once."""
        if not self.filter.initialized:
            return self.filter.state
        if self.filter_time is None:
            self.filter_time = target_time
            return self.filter.state
        dt = (target_time - self.filter_time).nanoseconds * 1e-9
        if dt <= 0.0:
            return self.filter.state
        self.filter_time = target_time
        return self.filter.predict(dt)

    def measurement_age(self, now):
        if self.last_seen_time is None:
            return float('inf')
        return max(
            0.0, (now - self.last_seen_time).nanoseconds * 1e-9)

    def reset_tracking_state(self):
        """Reset both the Kalman state and the gates tied to the old track."""
        self.filter.reset()
        self.track_frames = 0
        self.missed_frames = 0
        self.stable_since = None
        self.last_stable = False
        self.last_time = None
        self.filter_time = None
        self.last_source_u = None
        self.last_source_v = None
        self.last_area = None
        self.last_depth = None
        self.last_measurement = None
        self.last_measurement_confidence = 0.0
        self.prediction_only = False
        self.get_logger().info(
            'Tracker reset after prediction-only timeout; '
            'next valid measurement starts a new track'
        )

    def gate_measurement(
            self, msg, measurement, confidence, innovation_score=None,
            innovation_threshold=None):
        if confidence < self.min_confidence:
            return 'confidence %.2f < %.2f' % (confidence, self.min_confidence)
        if self.use_camera_space_gates:
            if (
                self.last_depth is not None
                and abs(float(msg.depth) - self.last_depth) > self.depth_gate_m
            ):
                return 'depth %.3f outside gate around %.3f +/- %.3f' % (
                    float(msg.depth), self.last_depth, self.depth_gate_m
                )
            if self.last_source_u is not None and self.last_source_v is not None:
                du = float(msg.source_u) - self.last_source_u
                dv = float(msg.source_v) - self.last_source_v
                pixel_jump = math.sqrt(du * du + dv * dv)
                if pixel_jump > self.max_pixel_jump:
                    return 'pixel jump %.1f > %.1f' % (
                        pixel_jump, self.max_pixel_jump
                    )
        if innovation_score is not None:
            threshold = self.innovation_gate_threshold \
                if innovation_threshold is None \
                else float(innovation_threshold)
            if (
                    not math.isfinite(float(innovation_score))
                    or float(innovation_score) > threshold):
                filter_state = getattr(
                    getattr(self, 'filter', None), 'state', None)
                innovation_magnitude = math.nan
                if filter_state is not None:
                    innovation_magnitude = math.sqrt(sum(
                        (float(measurement[index])
                         - float(filter_state[index])) ** 2
                        for index in range(3)))
                return (
                    'innovation gate d2=%.3f > %.3f residual=%.4fm'
                    % (
                        float(innovation_score), threshold,
                        innovation_magnitude))
        area = self.detection_area(msg)
        if (
            self.use_camera_space_gates
            and self.last_area is not None
            and self.last_area > 0.0
            and area > 0.0
        ):
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
        elif self.prediction_only or age >= self.low_confidence_timeout_s:
            self.status = 'LOW_CONFIDENCE'
        elif self.last_stable:
            self.status = 'LOCKED'
        elif self.track_frames >= self.min_track_frames:
            self.status = 'TRACKING'
        else:
            self.status = 'SEARCHING'

    def publish_status(self, status):
        self.status = status
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    def measurement_in_output_frame(self, msg):
        if not self.use_tf_transform:
            return [msg.point.x, msg.point.y, msg.point.z]

        point = PointStamped()
        point.header = msg.header
        if not point.header.frame_id:
            point.header.frame_id = self.camera_frame
        point.point = msg.point
        if point.header.frame_id == self.output_frame:
            return [point.point.x, point.point.y, point.point.z]
        try:
            transformed = self.tf_buffer.transform(
                point,
                self.output_frame,
                timeout=Duration(seconds=self.transform_timeout_s),
            )
        except TransformException as exc:
            self.get_logger().warn(
                'TF failed %s -> %s: %s'
                % (point.header.frame_id, self.output_frame, str(exc))
            )
            return None
        return [
            transformed.point.x,
            transformed.point.y,
            transformed.point.z,
        ]

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
        self.use_camera_space_gates = bool(
            self.get_parameter('use_camera_space_gates').value
        )
        self.max_target_speed = float(
            self.get_parameter('max_target_speed_mps').value
        )
        self.innovation_gate_threshold = max(
            0.0,
            float(self.get_parameter('innovation_gate_threshold').value),
        )
        self.reject_out_of_order_measurements = bool(
            self.get_parameter('reject_out_of_order_measurements').value
        )
        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.low_confidence_timeout_s = float(self.get_parameter('low_confidence_timeout_s').value)
        self.lost_timeout_s = float(self.get_parameter('lost_timeout_s').value)
        self.debug = bool(self.get_parameter('debug').value)


def main(args=None):
    rclpy.init(args=args)
    node = TargetTrackerNode()
    # A Target3D callback may wait briefly for a timestamped transform.  TF
    # subscriptions use a re-entrant callback group, so two executor threads
    # let the buffer receive /tf while that lookup is waiting.  A
    # SingleThreadedExecutor can otherwise starve TF continuously when target
    # measurements arrive at the same rate as the transform timeout.
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
