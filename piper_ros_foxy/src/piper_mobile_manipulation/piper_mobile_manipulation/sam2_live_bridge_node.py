#!/usr/bin/env python3
"""Foxy-side frame bridge and publisher for isolated live SAM2 tracking."""

import json
import os
import shutil
import time
from collections import deque, OrderedDict
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String

from piper_mobile_manipulation.msg import (
    CameraTimestampHealth,
    TrackedTarget,
    TrackingHealth,
)


class Sam2LiveBridgeNode(Node):
    def __init__(self):
        super().__init__('sam2_live_bridge_node')
        self.declare_parameter('color_image_topic', '/camera/color/image_raw')
        self.declare_parameter('seed_mask_topic', '/piper/heavy_target_mask')
        self.declare_parameter('output_mask_topic', '/piper/sam2_target_mask')
        self.declare_parameter('obstacle_mask_topic', '/piper/sam2_obstacle_mask')
        self.declare_parameter('unsafe_obstacle_mask_topic', '/piper/sam2_unsafe_obstacle_mask')
        self.declare_parameter(
            'movable_obstacle_mask_topic',
            '/piper/sam2_candidate_movable_obstacle_mask')
        self.declare_parameter('object_ids_topic', '/piper/sam2_object_ids')
        self.declare_parameter('status_topic', '/piper/sam2_tracking_status')
        self.declare_parameter('heavy_request_topic', '/piper/heavy_refresh_request')
        self.declare_parameter('heavy_status_topic', '/piper/heavy_refresh_status')
        self.declare_parameter('occlusion_status_topic', '/piper/occlusion_status')
        self.declare_parameter('target_status_topic', '/piper/target_status')
        self.declare_parameter('tracked_target_topic', '/piper/tracked_target')
        self.declare_parameter('joint_states_topic', '/joint_states_single')
        self.declare_parameter('tracking_health_topic', '/piper/tracking_health')
        self.declare_parameter(
            'camera_timestamp_health_topic', '/piper/camera_timestamp_health')
        self.declare_parameter('camera_timestamp_health_timeout_sec', 1.0)
        self.declare_parameter(
            'motion_prompt_topic', '/piper/motion_compensated_target_prompt'
        )
        self.declare_parameter('health_frame_id', 'base_link')
        self.declare_parameter('spool_dir', '/tmp/piper_sam2_live')
        self.declare_parameter('frame_rate_hz', 10.0)
        self.declare_parameter('seed_cache_sec', 60.0)
        self.declare_parameter('auto_initial_mask', False)
        self.declare_parameter('allow_heavy_topic_seed', False)
        self.declare_parameter('semantic_refresh_interval_sec', 0.0)
        self.declare_parameter('refresh_cooldown_sec', 5.0)
        self.declare_parameter('heavy_request_ack_timeout_sec', 3.0)
        self.declare_parameter('lost_refresh_retry_sec', 10.0)
        self.declare_parameter('absent_retry_sec', 30.0)
        self.declare_parameter('no_mask_refresh_timeout_sec', 8.0)
        self.declare_parameter('min_target_area_px', 100)
        self.declare_parameter('arm_moving_threshold_rad_s', 0.08)
        self.declare_parameter('arm_settled_threshold_rad_s', 0.03)
        self.declare_parameter('arm_motion_window_sec', 0.75)
        self.declare_parameter('arm_moving_position_delta_rad', 0.012)
        self.declare_parameter('arm_settled_position_delta_rad', 0.009)
        self.declare_parameter('camera_settle_time_sec', 0.5)
        self.declare_parameter('max_reacquisition_attempts', 2)
        self.declare_parameter('recovery_valid_frames', 5)
        self.declare_parameter('low_confidence_refresh_threshold', 0.60)
        self.declare_parameter('low_confidence_refresh_duration_sec', 1.0)
        self.declare_parameter('low_confidence_refresh_hysteresis', 0.10)
        self.declare_parameter('degraded_speed_scale', 0.25)
        self.declare_parameter('tracking_measurement_stale_sec', 0.75)
        self.declare_parameter('motion_prompt_max_age_sec', 0.25)
        self.declare_parameter('motion_prompt_recovery_grace_sec', 0.75)

        self.bridge = CvBridge()
        self.spool = Path(str(self.get_parameter('spool_dir').value))
        for name in ('frames', 'seeds', 'results', 'consumed'):
            (self.spool / name).mkdir(parents=True, exist_ok=True)
        self.jpeg_cache = OrderedDict()
        self.latest_msg = None
        self.latest_bgr = None
        self.last_frame_write = 0.0
        self.initial_requested = False
        self.seed_queued = False
        self.pending_seeds = {}
        self.last_refresh_request = 0.0
        self.last_semantic_refresh = time.monotonic()
        self.last_mask_publish = 0.0
        self.target_lost = False
        self.heavy_refresh_in_flight = False
        self.heavy_refresh_acknowledged = False
        self.heavy_refresh_sent_at = 0.0
        self.deferred_refresh_reason = None
        self.next_refresh_allowed = 0.0
        self.loss_episode_active = False
        self.loss_reason = ''
        self.heavy_attempt_count = 0
        self.recovery_valid_frames = 0
        self.has_ever_tracked = False
        self.last_target_status = 'SEARCHING'
        self.low_confidence_since = None
        self.low_confidence_refresh_latched = False
        self.lifecycle_state = 'WAITING_TO_REACQUIRE'
        self.measurement_quality = 0.0
        self.last_joint_positions = None
        self.last_joint_stamp_sec = None
        self.last_joint_monotonic = None
        self.joint_position_history = deque()
        self.joint_position_span_rad = 0.0
        self.joint_position_window_duration_sec = 0.0
        self.calculated_arm_speed = 0.0
        self.reported_arm_speed = 0.0
        self.arm_motion_known = False
        self.arm_moving = False
        self.arm_below_settle_since = time.monotonic()
        self.latest_motion_prompt = None
        self.latest_motion_prompt_key = None
        self.latest_motion_prompt_at = 0.0
        self.motion_prompt_seeded_episode = False
        self.motion_prompt_seeded_at = 0.0
        self.latest_camera_timestamp_health = None
        self.camera_timestamp_health_at = 0.0

        self.mask_pub = self.create_publisher(
            Image, self.get_parameter('output_mask_topic').value, qos_profile_sensor_data
        )
        self.obstacle_pub = self.create_publisher(
            Image, self.get_parameter('obstacle_mask_topic').value, qos_profile_sensor_data
        )
        self.unsafe_obstacle_pub = self.create_publisher(
            Image, self.get_parameter('unsafe_obstacle_mask_topic').value, qos_profile_sensor_data
        )
        self.movable_obstacle_pub = self.create_publisher(
            Image, self.get_parameter('movable_obstacle_mask_topic').value, qos_profile_sensor_data
        )
        self.object_ids_pub = self.create_publisher(
            Image, self.get_parameter('object_ids_topic').value, qos_profile_sensor_data
        )
        self.status_pub = self.create_publisher(
            String, self.get_parameter('status_topic').value, 10)
        self.request_pub = self.create_publisher(
            String, self.get_parameter('heavy_request_topic').value, 10
        )
        self.health_pub = self.create_publisher(
            TrackingHealth, self.get_parameter('tracking_health_topic').value, 10
        )
        self.create_subscription(
            Image, self.get_parameter('color_image_topic').value,
            self.color_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            String, self.get_parameter('occlusion_status_topic').value, self.occlusion_cb, 10
        )
        self.create_subscription(
            String, self.get_parameter('target_status_topic').value, self.target_status_cb, 10
        )
        self.create_subscription(
            TrackedTarget,
            self.get_parameter('tracked_target_topic').value,
            self.tracked_target_cb,
            10,
        )
        self.create_subscription(
            String, self.get_parameter('heavy_status_topic').value, self.heavy_status_cb, 10
        )
        self.create_subscription(
            Image, self.get_parameter('seed_mask_topic').value,
            self.heavy_seed_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            Image,
            self.get_parameter('motion_prompt_topic').value,
            self.motion_prompt_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            JointState,
            self.get_parameter('joint_states_topic').value,
            self.joint_state_cb,
            10,
        )
        self.create_subscription(
            CameraTimestampHealth,
            self.get_parameter('camera_timestamp_health_topic').value,
            self.camera_timestamp_health_cb,
            10,
        )
        self.create_timer(0.02, self.write_frame)
        self.create_timer(0.05, self.poll_results)
        self.create_timer(1.0, self.retry_lost_refresh)
        self.create_timer(0.2, self.publish_tracking_health)
        self.get_logger().warn('SAM2 live bridge is read-only; real arm motion is disabled.')

    @staticmethod
    def stamp_key(stamp):
        return '%010d_%09d' % (int(stamp.sec), int(stamp.nanosec))

    def color_cb(self, msg):
        try:
            image = np.asarray(self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')).copy()
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise IOError('JPEG encoding failed')
            key = self.stamp_key(msg.header.stamp)
            self.jpeg_cache[key] = (time.monotonic(), encoded.tobytes(), msg.header.frame_id)
            self.latest_msg = msg
            self.latest_bgr = image
            cutoff = time.monotonic() - float(self.get_parameter('seed_cache_sec').value)
            while self.jpeg_cache and next(iter(self.jpeg_cache.values()))[0] < cutoff:
                self.jpeg_cache.popitem(last=False)
            pending = self.pending_seeds.pop(key, None)
            if pending is not None:
                self.queue_seed(key, pending[0], pending[1])
            if bool(self.get_parameter('auto_initial_mask').value) and not self.initial_requested:
                self.initial_requested = True
                self.request_heavy_refresh('sam2_initial_mask')
                self.publish_status('initial_mask_requested', frame_key=key)
        except Exception as exc:
            self.get_logger().warn('SAM2 frame conversion failed: %s' % exc)

    def write_frame(self):
        rate = max(0.1, float(self.get_parameter('frame_rate_hz').value))
        if self.latest_msg is None or time.monotonic() - self.last_frame_write < 1.0 / rate:
            return
        key = self.stamp_key(self.latest_msg.header.stamp)
        cached = self.jpeg_cache.get(key)
        if cached is None:
            return
        final = self.spool / 'frames' / key
        if final.exists():
            return
        temporary = self.spool / 'frames' / (key + '.tmp')
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        (temporary / 'rgb.jpg').write_bytes(cached[1])
        with (temporary / 'frame.yaml').open('w', encoding='utf-8') as stream:
            yaml.safe_dump({
                'image_stamp': {
                    'sec': int(self.latest_msg.header.stamp.sec),
                    'nanosec': int(self.latest_msg.header.stamp.nanosec),
                },
                'frame_id': self.latest_msg.header.frame_id,
            }, stream, sort_keys=False)
        (temporary / 'READY').touch()
        os.replace(str(temporary), str(final))
        queued = sorted(path for path in (self.spool / 'frames').iterdir() if path.is_dir())
        for stale in queued[:-50]:
            shutil.rmtree(stale, ignore_errors=True)
        self.last_frame_write = time.monotonic()

    def heavy_seed_cb(self, msg):
        if bool(self.get_parameter('allow_heavy_topic_seed').value):
            self.seed_cb(msg, 'groundingdino_sam2')

    def motion_prompt_cb(self, msg):
        try:
            mask = np.asarray(
                self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            ).copy()
            if not np.count_nonzero(mask):
                return
            self.latest_motion_prompt = mask
            self.latest_motion_prompt_key = self.stamp_key(msg.header.stamp)
            self.latest_motion_prompt_at = time.monotonic()
            if (
                self.loss_episode_active
                and not self.motion_prompt_seeded_episode
                and not self.heavy_refresh_in_flight
            ):
                self.try_motion_prompt_seed()
        except Exception as exc:
            self.publish_status('motion_prompt_rejected', error=str(exc))

    def try_motion_prompt_seed(self):
        if self.latest_motion_prompt is None or self.latest_motion_prompt_key is None:
            return False
        if time.monotonic() - self.latest_motion_prompt_at > float(
            self.get_parameter('motion_prompt_max_age_sec').value
        ):
            return False
        queued = self.queue_seed(
            self.latest_motion_prompt_key,
            self.latest_motion_prompt,
            'motion_compensated',
        )
        if queued:
            self.motion_prompt_seeded_episode = True
            self.motion_prompt_seeded_at = time.monotonic()
            self.lifecycle_state = 'DEGRADED'
            self.publish_status('motion_prompt_recovery_started')
        return queued

    @staticmethod
    def joint_stamp_seconds(msg):
        stamp = msg.header.stamp
        value = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        return value if value > 0.0 else None

    def joint_state_cb(self, msg):
        positions = np.asarray(msg.position[:6], dtype=float)
        if positions.size != 6 or not np.all(np.isfinite(positions)):
            return
        now = time.monotonic()
        stamp = self.joint_stamp_seconds(msg)
        if self.last_joint_positions is not None:
            if stamp is not None and self.last_joint_stamp_sec is not None:
                dt = stamp - self.last_joint_stamp_sec
            else:
                dt = now - self.last_joint_monotonic
            if 1e-3 <= dt <= 1.0:
                self.calculated_arm_speed = float(
                    np.max(np.abs(positions - self.last_joint_positions)) / dt
                )
        velocities = np.asarray(msg.velocity[:6], dtype=float)
        if velocities.size == 6 and np.all(np.isfinite(velocities)):
            self.reported_arm_speed = float(np.max(np.abs(velocities)))
        self.last_joint_positions = positions
        self.last_joint_stamp_sec = stamp
        self.last_joint_monotonic = now
        self.update_position_motion_window(now, positions)
        self.arm_motion_known = True
        self.update_arm_motion_state(now)

    def update_position_motion_window(self, now, positions):
        """
        Track sustained joint displacement without amplifying encoder quantization.

        PiPER publishes at roughly 200 Hz. Differentiating one quantized encoder step
        over a 5 ms interval can look like more than 1 rad/s while the physical arm is
        stationary. A bounded position window still detects real motion, including
        slow moves, while ignoring those isolated sample-to-sample spikes.
        """
        window = max(0.10, float(
            self.get_parameter('arm_motion_window_sec').value))
        self.joint_position_history.append((float(now), np.asarray(positions).copy()))
        cutoff = float(now) - window
        while len(self.joint_position_history) > 1 \
                and self.joint_position_history[1][0] <= cutoff:
            self.joint_position_history.popleft()
        values = np.stack([item[1] for item in self.joint_position_history])
        self.joint_position_span_rad = float(np.max(np.ptp(values, axis=0)))
        self.joint_position_window_duration_sec = max(
            0.0, float(now) - self.joint_position_history[0][0])

    def update_arm_motion_state(self, now=None):
        now = time.monotonic() if now is None else now
        moving_delta = float(
            self.get_parameter('arm_moving_position_delta_rad').value)
        settled_delta = float(
            self.get_parameter('arm_settled_position_delta_rad').value)
        window = max(0.10, float(
            self.get_parameter('arm_motion_window_sec').value))
        if self.joint_position_span_rad >= moving_delta:
            self.arm_moving = True
            self.arm_below_settle_since = None
        elif (
                self.joint_position_window_duration_sec >= 0.80 * window
                and self.joint_position_span_rad <= settled_delta):
            if self.arm_below_settle_since is None:
                self.arm_below_settle_since = now
            settle_time = float(self.get_parameter('camera_settle_time_sec').value)
            if now - self.arm_below_settle_since >= settle_time:
                self.arm_moving = False

    def camera_settled(self, now=None):
        if not self.arm_motion_known:
            return True
        now = time.monotonic() if now is None else now
        self.update_arm_motion_state(now)
        if self.arm_moving or self.arm_below_settle_since is None:
            return False
        return now - self.arm_below_settle_since >= float(
            self.get_parameter('camera_settle_time_sec').value
        )

    def publish_tracking_health(self):
        now = time.monotonic()
        settled = self.camera_settled(now)
        if self.last_mask_publish > 0.0:
            measurement_age = max(0.0, now - self.last_mask_publish)
        else:
            # Keep the typed float32 status finite for all DDS implementations.
            measurement_age = 1.0e6
        camera_reason = self.camera_timestamp_rejection_reason(now)
        lifecycle_state = self.lifecycle_state if camera_reason is None \
            else 'CAMERA_CLOCK_INVALID'
        measurement_stale = max(
            0.05,
            float(self.get_parameter(
                'tracking_measurement_stale_sec').value),
        )
        prediction_only = (
            self.target_lost
            or measurement_age > measurement_stale
            or camera_reason is not None
        )
        if lifecycle_state == 'TRACKING' and not prediction_only:
            speed_scale = (
                float(self.get_parameter('degraded_speed_scale').value)
                if self.arm_moving else 1.0
            )
        elif lifecycle_state == 'DEGRADED':
            speed_scale = float(self.get_parameter('degraded_speed_scale').value)
        else:
            speed_scale = 0.0
        out = TrackingHealth()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = str(self.get_parameter('health_frame_id').value)
        out.lifecycle_state = lifecycle_state
        out.arm_moving = bool(self.arm_moving)
        out.camera_settled = bool(settled)
        out.prediction_only = bool(prediction_only)
        out.measurement_age_sec = float(measurement_age)
        out.measurement_quality = float(self.measurement_quality)
        out.heavy_attempt_count = int(self.heavy_attempt_count)
        out.recommended_speed_scale = float(np.clip(speed_scale, 0.0, 1.0))
        out.reason = camera_reason or self.loss_reason
        out.dry_run = True
        out.real_arm_motion = False
        self.health_pub.publish(out)

    def camera_timestamp_health_cb(self, message):
        self.latest_camera_timestamp_health = message
        self.camera_timestamp_health_at = time.monotonic()

    def camera_timestamp_rejection_reason(self, now=None):
        now = time.monotonic() if now is None else now
        timeout = float(self.get_parameter('camera_timestamp_health_timeout_sec').value)
        if self.latest_camera_timestamp_health is None:
            return 'camera timestamp health is missing'
        if now - self.camera_timestamp_health_at > timeout:
            return 'camera timestamp health is stale'
        if not self.latest_camera_timestamp_health.healthy:
            return 'camera timestamp %s: %s' % (
                self.latest_camera_timestamp_health.state,
                self.latest_camera_timestamp_health.reason,
            )
        return None

    def occlusion_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        status = str(payload.get('status', payload.get('occlusion_status', ''))).upper()
        if self.has_ever_tracked and status in ('HEAVILY_OCCLUDED', 'LOST'):
            self.begin_loss_episode('occlusion_%s' % status.lower())

    def target_status_cb(self, msg):
        status = str(msg.data).strip().upper()
        self.last_target_status = status
        if (
            self.has_ever_tracked
            and status in ('LOST', 'SEARCHING')
        ):
            self.begin_loss_episode('target_status_%s' % status.lower())
        elif status in ('TRACKING', 'LOCKED'):
            self.maybe_complete_recovery()
        elif status == 'LOW_CONFIDENCE' and not self.loss_episode_active:
            self.lifecycle_state = 'DEGRADED'

    def tracked_target_cb(self, msg):
        """Request one heavy refresh after sustained low metric confidence."""
        if not bool(msg.valid):
            return
        now = time.monotonic()
        confidence = float(msg.confidence)
        threshold = float(
            self.get_parameter('low_confidence_refresh_threshold').value)
        hysteresis = max(0.0, float(
            self.get_parameter('low_confidence_refresh_hysteresis').value))
        if confidence >= threshold:
            self.low_confidence_since = None
            if confidence >= min(1.0, threshold + hysteresis):
                self.low_confidence_refresh_latched = False
            return
        if self.low_confidence_refresh_latched or self.loss_episode_active:
            return
        if self.low_confidence_since is None:
            self.low_confidence_since = now
            return
        duration = max(0.0, float(
            self.get_parameter('low_confidence_refresh_duration_sec').value))
        if now - self.low_confidence_since < duration:
            return
        self.low_confidence_refresh_latched = True
        self.begin_loss_episode(
            'tracked_confidence_%.2f_below_%.2f' % (confidence, threshold),
            allow_motion_prompt=False,
        )

    def begin_loss_episode(self, reason, allow_motion_prompt=True):
        """Latch a loss transition; repeated status publications are not new events."""
        new_episode = not self.loss_episode_active
        if new_episode:
            self.loss_episode_active = True
            self.heavy_attempt_count = 0
            self.recovery_valid_frames = 0
            self.loss_reason = reason
            self.motion_prompt_seeded_episode = False
            self.motion_prompt_seeded_at = 0.0
        self.target_lost = True
        if not new_episode:
            return
        if allow_motion_prompt and self.try_motion_prompt_seed():
            return
        if not self.camera_settled():
            self.lifecycle_state = 'WAITING_TO_REACQUIRE'
            self.publish_status('reacquisition_waiting_for_motion', reason=self.loss_reason)
            return
        self.request_heavy_refresh(self.loss_reason)

    def record_valid_mask(self, result):
        self.has_ever_tracked = True
        area = int(result.get('mask_area_px', 0))
        minimum = max(1, int(self.get_parameter('min_target_area_px').value))
        self.measurement_quality = float(np.clip(area / float(4 * minimum), 0.0, 1.0))
        if self.loss_episode_active:
            self.recovery_valid_frames += 1
            self.lifecycle_state = 'DEGRADED'
            self.maybe_complete_recovery()
        else:
            self.target_lost = False
            self.lifecycle_state = 'TRACKING'

    def maybe_complete_recovery(self):
        required = max(1, int(self.get_parameter('recovery_valid_frames').value))
        if (
            self.loss_episode_active
            and self.recovery_valid_frames >= required
            and self.last_target_status in ('TRACKING', 'LOCKED')
        ):
            self.loss_episode_active = False
            self.target_lost = False
            self.heavy_attempt_count = 0
            self.recovery_valid_frames = 0
            self.loss_reason = ''
            self.motion_prompt_seeded_episode = False
            self.motion_prompt_seeded_at = 0.0
            self.lifecycle_state = 'TRACKING'
            self.publish_status('loss_episode_recovered')

    def heavy_status_cb(self, msg):
        """Keep heavy inference single-flight and coalesce requests while it runs."""
        try:
            payload = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        state = str(payload.get('state', '')).lower()
        if state in ('queued', 'waiting_for_image', 'waiting_for_fresh_image'):
            self.heavy_refresh_in_flight = True
            self.heavy_refresh_acknowledged = True
            return
        if state == 'request_ignored_busy':
            # Another heavy job owns the worker. Wait for its terminal status,
            # then the lost-target timer can decide whether another run is needed.
            self.heavy_refresh_in_flight = True
            self.heavy_refresh_acknowledged = True
            return
        if (
            state == 'idle'
            and self.heavy_refresh_in_flight
            and self.heavy_refresh_acknowledged
        ):
            # The heavy bridge is authoritative about worker/spool state.  This
            # recovers a lost terminal status without allowing overlapping jobs.
            self.finish_heavy_refresh()
            return
        if state in ('published', 'worker_result_rejected', 'request_failed', 'request_rejected'):
            self.finish_heavy_refresh()

    def finish_heavy_refresh(self):
        self.heavy_refresh_in_flight = False
        self.heavy_refresh_acknowledged = False
        self.heavy_refresh_sent_at = 0.0
        self.deferred_refresh_reason = None
        self.next_refresh_allowed = time.monotonic() + float(
            self.get_parameter('lost_refresh_retry_sec').value
        )
        if self.loss_episode_active:
            maximum = max(
                1, int(self.get_parameter('max_reacquisition_attempts').value)
            )
            self.lifecycle_state = (
                'ABSENT'
                if self.heavy_attempt_count >= maximum
                else 'WAITING_TO_REACQUIRE'
            )

    def seed_cb(self, msg, source):
        try:
            mask = np.asarray(self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')).copy()
            if not np.count_nonzero(mask):
                raise ValueError('empty initial mask')
            key = self.stamp_key(msg.header.stamp)
            cached = self.jpeg_cache.get(key)
            if cached is None:
                self.pending_seeds[key] = (mask, source)
                return
            self.queue_seed(key, mask, source)
        except Exception as exc:
            self.publish_status('seed_rejected', error='%s: %s' % (type(exc).__name__, exc))

    def queue_seed(self, key, mask, source):
        try:
            cached = self.jpeg_cache.get(key)
            if cached is None:
                raise ValueError('matching RGB frame expired from seed cache')
            final = self.spool / 'seeds' / ('%s_%s' % (key, source))
            temporary = self.spool / 'seeds' / (key + '.tmp')
            shutil.rmtree(temporary, ignore_errors=True)
            temporary.mkdir(parents=True)
            (temporary / 'rgb.jpg').write_bytes(cached[1])
            mask_file = 'object_001.png'
            cv2.imwrite(str(temporary / mask_file), mask)
            objects = [{
                'object_id': 1,
                'role': 'target',
                'label': 'target',
                'mask_file': mask_file,
            }]
            with (temporary / 'seed.yaml').open('w', encoding='utf-8') as stream:
                yaml.safe_dump({
                    'frame_key': key,
                    'source': source,
                    'objects': objects,
                }, stream, sort_keys=False)
            (temporary / 'READY').touch()
            os.replace(str(temporary), str(final))
            self.seed_queued = True
            self.publish_status(
                'seed_queued', frame_key=key, source=source,
                mask_area_px=int(np.count_nonzero(mask))
            )
            return True
        except Exception as exc:
            self.publish_status('seed_rejected', error='%s: %s' % (type(exc).__name__, exc))
            return False

    def poll_results(self):
        for result_dir in sorted((self.spool / 'results').iterdir()):
            if (
                not result_dir.is_dir()
                or result_dir.name.endswith('.tmp')
                or not (result_dir / 'READY').is_file()
            ):
                continue
            try:
                with (result_dir / 'result.yaml').open('r', encoding='utf-8') as stream:
                    result = yaml.safe_load(stream) or {}
                mask = cv2.imread(str(result_dir / 'mask.png'), cv2.IMREAD_GRAYSCALE)
                if result.get('status') in ('ok', 'empty_target_mask') and mask is not None:
                    out = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
                    stamp = result.get('image_stamp', {})
                    out.header.stamp.sec = int(stamp.get('sec', 0))
                    out.header.stamp.nanosec = int(stamp.get('nanosec', 0))
                    out.header.frame_id = result.get('frame_id', '')
                    self.mask_pub.publish(out)
                    self.last_mask_publish = time.monotonic()
                    self.publish_result_mask(
                        result_dir / 'all_obstacle_mask.png',
                        self.obstacle_pub, out.header)
                    self.publish_result_mask(
                        result_dir / 'unsafe_obstacle_mask.png',
                        self.unsafe_obstacle_pub, out.header
                    )
                    self.publish_result_mask(
                        result_dir / 'candidate_movable_obstacle_mask.png',
                        self.movable_obstacle_pub,
                        out.header,
                    )
                    self.publish_result_mask(
                        result_dir / 'object_ids.png', self.object_ids_pub,
                        out.header, encoding='mono16'
                    )
                    if result.get('status') == 'ok':
                        self.record_valid_mask(result)
                        self.publish_status('tracking', **result)
                    else:
                        self.measurement_quality = 0.0
                        self.begin_loss_episode('sam2_empty_target_mask')
                        self.publish_status('waiting_for_seed', **result)
                    self.evaluate_refresh_policy(result)
                else:
                    self.publish_status('worker_result_rejected', **result)
                destination = self.spool / 'consumed' / ('result_' + result_dir.name)
                shutil.rmtree(destination, ignore_errors=True)
                os.replace(str(result_dir), str(destination))
                consumed = sorted(
                    path for path in (self.spool / 'consumed').iterdir() if path.is_dir()
                )
                for stale in consumed[:-200]:
                    shutil.rmtree(stale, ignore_errors=True)
            except Exception as exc:
                self.get_logger().error('Failed to consume SAM2 result: %s' % exc)

    def publish_result_mask(self, path, publisher, header, encoding='mono8'):
        read_mode = cv2.IMREAD_UNCHANGED if encoding == 'mono16' else cv2.IMREAD_GRAYSCALE
        mask = cv2.imread(str(path), read_mode)
        if mask is None:
            return
        out = self.bridge.cv2_to_imgmsg(mask, encoding=encoding)
        out.header = header
        publisher.publish(out)

    def evaluate_refresh_policy(self, result):
        area = int(result.get('mask_area_px', 0))
        if area < int(self.get_parameter('min_target_area_px').value):
            self.begin_loss_episode('sam2_target_lost')
        interval = float(self.get_parameter('semantic_refresh_interval_sec').value)
        if interval > 0.0 and time.monotonic() - self.last_semantic_refresh >= interval:
            self.request_heavy_refresh('periodic_semantic_refresh')

    def retry_lost_refresh(self):
        now = time.monotonic()
        self.update_arm_motion_state(now)
        if self.heavy_refresh_in_flight and not self.heavy_refresh_acknowledged:
            ack_timeout = max(
                0.1, float(self.get_parameter('heavy_request_ack_timeout_sec').value)
            )
            if now - self.heavy_refresh_sent_at >= ack_timeout:
                # Publishing a ROS message is not an acknowledgement.  A request
                # sent while discovery or the camera is recovering can be lost;
                # release only that unacknowledged reservation so it may retry.
                self.heavy_refresh_in_flight = False
                self.heavy_refresh_sent_at = 0.0
                self.publish_status('heavy_refresh_ack_timeout')
        if self.initial_requested and self.last_mask_publish <= 0.0:
            if now - self.last_refresh_request >= float(
                self.get_parameter('no_mask_refresh_timeout_sec').value
            ):
                if not self.loss_episode_active:
                    self.begin_loss_episode('sam2_no_mask_after_initial')
                else:
                    self.retry_loss_episode('sam2_no_mask_after_initial', now)
            return
        if self.last_mask_publish > 0.0:
            age = now - self.last_mask_publish
            if age >= float(self.get_parameter('no_mask_refresh_timeout_sec').value):
                if not self.loss_episode_active:
                    self.begin_loss_episode('sam2_no_recent_mask')
                else:
                    self.retry_loss_episode('sam2_no_recent_mask', now)
                return
        if not self.loss_episode_active:
            return
        self.retry_loss_episode('sam2_target_lost_retry', now)

    def retry_loss_episode(self, reason, now=None):
        now = time.monotonic() if now is None else now
        if self.motion_prompt_seeded_episode and now - self.motion_prompt_seeded_at < float(
            self.get_parameter('motion_prompt_recovery_grace_sec').value
        ):
            return False
        maximum = max(1, int(self.get_parameter('max_reacquisition_attempts').value))
        attempt_limit_reached = self.heavy_attempt_count >= maximum
        if attempt_limit_reached:
            self.lifecycle_state = 'ABSENT'
            absent_retry = float(self.get_parameter('absent_retry_sec').value)
            if absent_retry <= 0.0:
                return False
        if not self.camera_settled(now):
            if not attempt_limit_reached:
                self.lifecycle_state = 'WAITING_TO_REACQUIRE'
            return False
        retry = max(
            float(self.get_parameter('refresh_cooldown_sec').value),
            float(self.get_parameter('lost_refresh_retry_sec').value),
            absent_retry if attempt_limit_reached else 0.0,
        )
        if (
            not self.heavy_refresh_in_flight
            and now >= self.next_refresh_allowed
            and now - self.last_refresh_request >= retry
        ):
            return self.request_heavy_refresh(
                reason, allow_attempt_limit=attempt_limit_reached
            )
        return False

    def request_heavy_refresh(self, reason, allow_attempt_limit=False):
        now = time.monotonic()
        if not self.camera_settled(now):
            if self.loss_episode_active:
                self.lifecycle_state = 'WAITING_TO_REACQUIRE'
            self.publish_status('heavy_refresh_suppressed_motion', reason=reason)
            return False
        if self.loss_episode_active:
            maximum = max(
                1, int(self.get_parameter('max_reacquisition_attempts').value)
            )
            if self.heavy_attempt_count >= maximum and not allow_attempt_limit:
                self.lifecycle_state = 'ABSENT'
                self.publish_status(
                    'heavy_refresh_attempt_limit', reason=self.loss_reason,
                    attempts=self.heavy_attempt_count,
                )
                return False
        if self.heavy_refresh_in_flight:
            self.deferred_refresh_reason = reason
            self.publish_status('heavy_refresh_deferred', reason=reason)
            return False
        if now < self.next_refresh_allowed:
            return False
        if now - self.last_refresh_request < float(
                self.get_parameter('refresh_cooldown_sec').value):
            return False
        msg = String()
        msg.data = json.dumps({
            'request_id': 'sam2_event_%d' % int(time.time() * 1000),
            'reason': reason,
            'tracking': {'tracking_confidence': 0.0},
            'dry_run': True,
            'real_arm_motion': False,
        })
        self.request_pub.publish(msg)
        # Set this before a bridge status arrives so simultaneous callbacks cannot
        # publish duplicate heavy requests.
        self.heavy_refresh_in_flight = True
        self.heavy_refresh_acknowledged = False
        self.heavy_refresh_sent_at = now
        self.last_refresh_request = now
        self.last_semantic_refresh = now
        if self.loss_episode_active:
            self.heavy_attempt_count += 1
            self.lifecycle_state = 'REACQUIRING'
        self.publish_status(
            'heavy_refresh_requested',
            reason=reason,
            retry_mode='periodic_absent' if allow_attempt_limit else 'burst',
        )
        return True

    def publish_status(self, state, **values):
        payload = {'state': state, 'dry_run': True, 'real_arm_motion': False}
        payload.update(values)
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Sam2LiveBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
