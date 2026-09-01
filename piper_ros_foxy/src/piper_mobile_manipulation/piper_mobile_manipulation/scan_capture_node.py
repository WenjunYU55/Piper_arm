#!/usr/bin/env python3
import json
import math
import os
import time
from collections import deque
from datetime import datetime
from threading import Condition

import cv2
import hashlib
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from piper_mobile_manipulation.msg import ScanExecutionStatus, Target3D
from piper_mobile_manipulation.scan_capture import (
    DepthQualityRejected,
    capture_diagnostic_rejection,
    depth_millimetres,
    exact_stamped_item,
    nearest_stamped_item,
    qualify_target_depth,
    rigid_transform_matrix,
    stamp_key,
    stamp_seconds,
    synchronized_bundle_rejection,
    temporal_confident_depth_median,
)
from piper_mobile_manipulation.perception.target_envelope import (
    build_capture_model_seed,
    trusted_silhouette_measurement,
)


CAPTURE_RESPONSE_MARGIN_SEC = 2.0


def capture_model_seed_from_qualified(
        mask, qualified, color_camera_matrix, mask_header,
        camera_transform):
    """Build the model seed from the exact RGB-D record being persisted."""
    if not isinstance(qualified, dict):
        raise ValueError('qualified capture geometry is missing')
    quality = qualified.get('quality')
    if not isinstance(quality, dict):
        raise ValueError('qualified capture quality is missing')
    shape = trusted_silhouette_measurement(
        mask,
        np.asarray(qualified.get('target_support_mask')) > 0,
        (0, 0),
        np.asarray(qualified.get('target_depth_mm'), dtype=float) / 1000.0,
        np.asarray(color_camera_matrix, dtype=float).reshape(3, 3),
        mask_header,
        float(quality['confident_fraction']),
    )
    return build_capture_model_seed(shape, camera_transform)


def capture_view_selection_provenance(plan_provenance, execution):
    """Resolve one executing plan/view to immutable planner provenance."""
    plan_id = str(execution.get('plan_id', ''))
    if not plan_id:
        return {'available': False, 'reason': 'execution plan ID is missing'}
    if not isinstance(plan_provenance, dict):
        return {'available': False, 'reason': 'plan provenance is unavailable'}
    if str(plan_provenance.get('plan_id', '')) != plan_id:
        return {'available': False, 'reason': 'plan provenance ID mismatch'}
    selected = plan_provenance.get('selected_viewpoints')
    if not isinstance(selected, list) or not selected:
        return {'available': False, 'reason': 'selected viewpoint is missing'}
    try:
        current_view = int(execution.get('current_view', 0))
    except (TypeError, ValueError):
        return {'available': False, 'reason': 'execution view index is invalid'}
    index = max(0, current_view - 1)
    if index >= len(selected) or not isinstance(selected[index], dict):
        return {'available': False, 'reason': 'execution view has no provenance'}
    result = {
        'available': True,
        'plan_id': plan_id,
        'request_id': str(plan_provenance.get('request_id', '')),
        'request_sha256': str(plan_provenance.get('request_sha256', '')),
        **dict(selected[index]),
    }
    candidate_diagnostics = plan_provenance.get('candidate_diagnostics')
    if isinstance(candidate_diagnostics, dict):
        result['candidate_diagnostics'] = dict(candidate_diagnostics)
    return result


try:
    import yaml
except ImportError:
    yaml = None


class ScanCaptureNode(Node):
    def __init__(self):
        super().__init__('scan_capture_node')
        self.declare_parameter('color_image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_image_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter(
            'native_depth_image_topic', '/camera/depth/image_rect_raw')
        self.declare_parameter(
            'native_depth_camera_info_topic', '/camera/depth/camera_info')
        self.declare_parameter(
            'confidence_image_topic', '/camera/confidence/image_rect_raw')
        self.declare_parameter('mask_topic', '/piper/sam2_target_mask')
        self.declare_parameter('target_3d_topic', '/piper/target_3d')
        self.declare_parameter('scan_viewpoints_topic', '/piper/scan_viewpoints')
        self.declare_parameter(
            'reachable_scan_viewpoints_topic',
            '/piper/reachable_scan_viewpoints')
        self.declare_parameter('scan_coverage_topic', '/piper/scan_coverage')
        self.declare_parameter('scan_quality_topic', '/piper/scan_quality')
        self.declare_parameter('occlusion_status_topic', '/piper/occlusion_status')
        self.declare_parameter('scan_capture_status_topic', '/piper/scan_capture_status')
        self.declare_parameter('scan_summary_topic', '/piper/scan_summary')
        self.declare_parameter(
            'scan_execution_status_topic', '/piper/scan_execution_status')
        self.declare_parameter(
            'plan_provenance_topic', '/piper/motion_plan_provenance')
        self.declare_parameter('joint_state_topic', '/joint_states_single')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter(
            'camera_optical_frame', 'camera_color_optical_frame')
        self.declare_parameter('require_camera_transform', True)
        self.declare_parameter('camera_transform_timeout_sec', 0.25)
        self.declare_parameter('task_id', '')
        self.declare_parameter('mission_sha256', '')
        self.declare_parameter('target_label', 'green cube')
        self.declare_parameter('target_profile', 'green_cube')
        self.declare_parameter('target_prompt', 'green cube .')
        self.declare_parameter('calibration_sha256', '')

        self.declare_parameter('capture_mode', 'interval')
        self.declare_parameter('capture_interval_sec', 2.0)
        self.declare_parameter('max_frames_per_scan', 30)
        self.declare_parameter('max_bundle_age_sec', 1.0)
        self.declare_parameter('synchronization_slop_sec', 0.08)
        self.declare_parameter('native_depth_confidence_slop_sec', 0.005)
        self.declare_parameter('capture_cache_size', 240)
        self.declare_parameter('capture_burst_frames', 20)
        # This is the same mission capture deadline used by the executor.  The
        # burst collector keeps a small response margin so its Trigger result
        # reaches the executor before that outer deadline expires.
        self.declare_parameter('capture_timeout_sec', 20.0)
        self.declare_parameter(
            'minimum_burst_support_fraction', 0.50)
        self.declare_parameter('minimum_depth_confidence', 8)
        self.declare_parameter('minimum_confident_target_points', 50)
        self.declare_parameter('minimum_confident_target_fraction', 0.50)
        self.declare_parameter('minimum_primary_component_fraction', 0.15)
        self.declare_parameter('depth_component_ambiguity_margin', 0.08)
        self.declare_parameter('require_valid_target', True)
        self.declare_parameter('require_mask', True)
        self.declare_parameter('require_depth', True)
        self.declare_parameter('require_good_quality_for_service', True)
        self.declare_parameter('require_clear_occlusion_for_service', True)
        self.declare_parameter(
            'allow_classified_occlusion_for_service', False)
        self.declare_parameter(
            'allow_classified_occlusion_for_first_capture', False)
        self.declare_parameter('diagnostic_timeout_sec', 1.0)
        self.declare_parameter('minimum_accepted_quality_score', 0.65)
        self.declare_parameter('dataset_root', '/home/prl/Piper_arm/datasets/active_scan')
        self.declare_parameter('dry_run', True)
        self.declare_parameter('enable_real_arm_motion', False)
        self.declare_parameter('debug', True)

        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None
        self.latest_camera_info = None
        self.latest_bundle_received_at = None
        self.latest_mask = None
        self.latest_target = None
        self.latest_joint_state = None
        self.latest_execution_status = None
        self.plan_provenance_by_id = {}
        self.latest_camera_transform = None
        self.latest_color_from_depth_transform = None
        minimum_cache = max(
            1, int(self.get_parameter('capture_burst_frames').value))
        self.color_bundle_cache = deque(maxlen=max(
            minimum_cache,
            int(self.get_parameter('capture_cache_size').value)))
        self.native_bundle_cache = deque(maxlen=max(
            minimum_cache,
            int(self.get_parameter('capture_cache_size').value)))
        self.native_bundle_condition = Condition()
        self.prepared_capture = None
        self.last_capture_result = None
        self.latest_scan_viewpoints = None
        self.latest_reachable_scan_viewpoints = None
        self.latest_scan_coverage = None
        self.latest_scan_quality = None
        self.latest_scan_quality_at = None
        self.latest_occlusion_status = None
        self.latest_occlusion_status_at = None
        self.last_capture_time = None
        self.frame_index = 0
        self.manifest_sha256 = ''
        self.skip_counts = {}
        self.quality_counts = {'GOOD': 0, 'ACCEPTABLE': 0, 'POOR': 0, 'INVALID': 0}
        self.occlusion_counts = {
            'CLEAR': 0,
            'PARTIALLY_OCCLUDED': 0,
            'HEAVILY_OCCLUDED': 0,
            'LOST': 0,
            'UNKNOWN': 0,
        }

        self.scan_dir = self.create_scan_dir()
        self.frames_dir = os.path.join(self.scan_dir, 'frames')
        os.makedirs(self.frames_dir, exist_ok=True)
        self.write_yaml(
            os.path.join(self.scan_dir, 'metadata.yaml'),
            {
                'scan_started_at': self.wall_time_string(),
                'dataset_root': self.get_parameter('dataset_root').value,
                'scan_dir': self.scan_dir,
                'dry_run': self.param_bool('dry_run'),
                'real_arm_motion': False,
                'capture_node_dry_run': self.param_bool('dry_run'),
                'capture_node_real_arm_motion': False,
                'max_frames_per_scan': int(self.get_parameter('max_frames_per_scan').value),
                'capture_mode': self.capture_mode(),
                'capture_interval_sec': float(self.get_parameter('capture_interval_sec').value),
                'task_id': str(self.get_parameter('task_id').value),
                'mission_sha256': str(
                    self.get_parameter('mission_sha256').value),
                'target_label': str(
                    self.get_parameter('target_label').value),
                'target_profile': str(self.get_parameter('target_profile').value),
                'target_prompt': str(
                    self.get_parameter('target_prompt').value),
                'calibration_sha256': str(
                    self.get_parameter('calibration_sha256').value),
                'capture_schema_version': 2,
                'confidence_policy': self.confidence_policy_metadata(),
                'topics': self.topic_metadata(),
            },
        )

        self.status_pub = self.create_publisher(
            String, self.get_parameter('scan_capture_status_topic').value, 10
        )
        self.summary_pub = self.create_publisher(
            String, self.get_parameter('scan_summary_topic').value, 10
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.color_sub = Subscriber(
            self, Image, self.get_parameter('color_image_topic').value,
            qos_profile=qos_profile_sensor_data)
        self.depth_sub = Subscriber(
            self, Image, self.get_parameter('depth_image_topic').value,
            qos_profile=qos_profile_sensor_data)
        self.camera_info_sub = Subscriber(
            self, CameraInfo, self.get_parameter('camera_info_topic').value,
            qos_profile=qos_profile_sensor_data)
        self.rgbd_sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.camera_info_sub],
            10,
            float(self.get_parameter('synchronization_slop_sec').value),
        )
        self.rgbd_sync.registerCallback(self.rgbd_cb)
        self.native_depth_sub = Subscriber(
            self, Image,
            self.get_parameter('native_depth_image_topic').value,
            qos_profile=qos_profile_sensor_data)
        self.native_depth_info_sub = Subscriber(
            self, CameraInfo,
            self.get_parameter('native_depth_camera_info_topic').value,
            qos_profile=qos_profile_sensor_data)
        self.confidence_sub = Subscriber(
            self, Image,
            self.get_parameter('confidence_image_topic').value,
            qos_profile=qos_profile_sensor_data)
        self.native_sync = ApproximateTimeSynchronizer(
            [self.native_depth_sub, self.confidence_sub,
             self.native_depth_info_sub],
            20,
            float(self.get_parameter(
                'native_depth_confidence_slop_sec').value),
        )
        self.native_sync.registerCallback(self.native_bundle_cb)
        self._retained_subscriptions = []
        self._retained_subscriptions.append(self.create_subscription(
            Image,
            self.get_parameter('mask_topic').value,
            self.mask_cb,
            qos_profile_sensor_data,
        ))
        self._retained_subscriptions.append(self.create_subscription(
            Target3D,
            self.get_parameter('target_3d_topic').value,
            self.target_cb,
            10,
        ))
        self._retained_subscriptions.append(self.create_subscription(
            JointState,
            self.get_parameter('joint_state_topic').value,
            self.joint_state_cb,
            10,
        ))
        self._retained_subscriptions.append(self.create_subscription(
            ScanExecutionStatus,
            self.get_parameter('scan_execution_status_topic').value,
            self.execution_status_cb,
            10,
        ))
        self._retained_subscriptions.append(self.create_subscription(
            String,
            self.get_parameter('plan_provenance_topic').value,
            self.plan_provenance_cb,
            10,
        ))
        self._retained_subscriptions.append(self.create_subscription(
            String,
            self.get_parameter('scan_viewpoints_topic').value,
            self.scan_viewpoints_cb,
            10,
        ))
        self._retained_subscriptions.append(self.create_subscription(
            String,
            self.get_parameter('reachable_scan_viewpoints_topic').value,
            self.reachable_scan_viewpoints_cb,
            10,
        ))
        self._retained_subscriptions.append(self.create_subscription(
            String,
            self.get_parameter('scan_coverage_topic').value,
            self.scan_coverage_cb,
            10,
        ))
        # Keep capture-gating evidence responsive while the default callback
        # group services the high-rate RGB-D subscriptions.
        self.diagnostic_callback_group = MutuallyExclusiveCallbackGroup()
        self._retained_subscriptions.append(self.create_subscription(
            String,
            self.get_parameter('scan_quality_topic').value,
            self.scan_quality_cb,
            10,
            callback_group=self.diagnostic_callback_group,
        ))
        self._retained_subscriptions.append(self.create_subscription(
            String,
            self.get_parameter('occlusion_status_topic').value,
            self.occlusion_status_cb,
            10,
            callback_group=self.diagnostic_callback_group,
        ))

        # A service capture waits for 20 new camera frames.  Its own callback
        # group lets the image callbacks continue filling that burst.
        self.capture_callback_group = MutuallyExclusiveCallbackGroup()
        self.capture_service = self.create_service(
            Trigger, '~/capture_view', self.capture_view_cb,
            callback_group=self.capture_callback_group)
        self.timer = None
        if self.capture_mode() == 'interval':
            self.timer = self.create_timer(0.25, self.timer_cb)
        self.publish_status('ready', 'scan capture initialized')
        self.publish_summary()
        self.get_logger().warn(
            'Scan capture is command-free; it saves RGB-D data and never publishes arm commands.'
        )

    def rgbd_cb(self, color, depth, camera_info):
        self.latest_color = color
        self.latest_depth = depth
        self.latest_camera_info = camera_info
        self.latest_bundle_received_at = time.monotonic()
        self.color_bundle_cache.append((
            color, depth, camera_info, self.latest_bundle_received_at))

    def native_bundle_cb(self, depth, confidence, camera_info):
        bundle = (depth, confidence, camera_info, time.monotonic())
        with self.native_bundle_condition:
            self.native_bundle_cache.append(bundle)
            self.native_bundle_condition.notify_all()

    def mask_cb(self, msg):
        self.latest_mask = msg

    def target_cb(self, msg):
        self.latest_target = msg

    def joint_state_cb(self, msg):
        self.latest_joint_state = msg

    def execution_status_cb(self, msg):
        self.latest_execution_status = msg

    def plan_provenance_cb(self, msg):
        payload = self.parse_json_msg(msg)
        if not isinstance(payload, dict):
            return
        try:
            schema_version = int(payload.get('schema_version', 0))
        except (TypeError, ValueError):
            return
        if schema_version != 1:
            return
        plan_id = str(payload.get('plan_id', '')).strip()
        selected = payload.get('selected_viewpoints')
        if not plan_id or not isinstance(selected, list):
            return
        self.plan_provenance_by_id[plan_id] = payload
        while len(self.plan_provenance_by_id) > 64:
            del self.plan_provenance_by_id[next(iter(
                self.plan_provenance_by_id))]

    def scan_viewpoints_cb(self, msg):
        self.latest_scan_viewpoints = self.parse_json_msg(msg)

    def reachable_scan_viewpoints_cb(self, msg):
        self.latest_reachable_scan_viewpoints = self.parse_json_msg(msg)

    def scan_coverage_cb(self, msg):
        self.latest_scan_coverage = self.parse_json_msg(msg)

    def scan_quality_cb(self, msg):
        self.latest_scan_quality = self.parse_json_msg(msg)
        self.latest_scan_quality_at = time.monotonic()

    def occlusion_status_cb(self, msg):
        self.latest_occlusion_status = self.parse_json_msg(msg)
        self.latest_occlusion_status_at = time.monotonic()

    def timer_cb(self):
        if self.frame_index >= int(self.get_parameter('max_frames_per_scan').value):
            self.publish_summary()
            return

        now = self.get_clock().now()
        interval = max(0.1, float(self.get_parameter('capture_interval_sec').value))
        if self.last_capture_time is not None:
            age = (now - self.last_capture_time).nanoseconds * 1e-9
            if age < interval:
                return

        ok, reason = self.capture_ready()
        if not ok:
            self.note_skip(reason)
            self.publish_status('skipped', reason)
            return

        self.capture_frame(now)

    def capture_view_cb(self, _request, response):
        if self.capture_mode() != 'service':
            response.success = False
            response.message = 'capture_mode is not service'
            return response
        if self.frame_index >= int(self.get_parameter('max_frames_per_scan').value):
            response.success = False
            response.message = 'maximum frame count reached'
            return response
        ok, reason = self.capture_prerequisites_ready()
        if not ok:
            self.note_skip(reason)
            self.publish_status('skipped', reason)
            response.success = False
            response.message = reason
            return response
        burst_started_at = time.monotonic()
        burst, reason = self.collect_prospective_native_depth_burst(
            burst_started_at)
        if reason:
            self.note_skip(reason)
            self.publish_status('skipped', reason)
            response.success = False
            response.message = reason
            return response
        ok, reason = self.capture_ready(depth_burst=burst)
        if not ok:
            self.note_skip(reason)
            self.publish_status('skipped', reason)
            response.success = False
            response.message = reason
            return response
        self.last_capture_result = None
        saved, message = self.capture_frame(self.get_clock().now())
        response.success = bool(saved)
        response.message = (
            json.dumps(self.last_capture_result, sort_keys=True)
            if saved and isinstance(self.last_capture_result, dict)
            else str(message))
        return response

    def capture_prerequisites_ready(self):
        """Check the unchanged non-image capture gates."""
        self.prepared_capture = None
        if not self.param_bool('dry_run'):
            return False, 'dry_run is false'
        if self.param_bool('enable_real_arm_motion'):
            return False, 'enable_real_arm_motion is true'
        if self.param_bool('require_mask') and self.latest_mask is None:
            return False, 'missing detection mask'
        if self.param_bool('require_valid_target'):
            if self.latest_target is None:
                return False, 'missing target_3d'
            if not self.latest_target.valid:
                return False, 'target_3d invalid'
        if self.capture_mode() == 'service':
            status = self.latest_execution_status
            if status is None:
                return False, 'missing scan execution status'
            if str(status.execution_mode) != 'MULTIVIEW_SCAN':
                return False, 'scan execution is not MULTIVIEW_SCAN'
            if str(status.state) not in ('CAPTURING', 'CAPTURING_RGBD'):
                return False, 'executor is not at an accepted settled capture'
            if (
                    self.param_bool('require_good_quality_for_service')
                    or self.param_bool('require_clear_occlusion_for_service')):
                now = time.monotonic()
                quality = (
                    self.latest_scan_quality
                    if self.param_bool('require_good_quality_for_service')
                    else {'quality_label': 'GOOD', 'quality_score': 1.0,
                          'target_valid': True})
                occlusion = (
                    self.latest_occlusion_status
                    if self.param_bool('require_clear_occlusion_for_service')
                    else {'occlusion_state': 'CLEAR'})
                classified_occlusion_barrier = bool(
                    self.param_bool(
                        'allow_classified_occlusion_for_service'))
                first_capture_semantic_barrier = bool(
                    self.param_bool(
                        'allow_classified_occlusion_for_first_capture')
                    and int(self.frame_index) == 0
                    and int(status.current_view) == 1)
                reason = capture_diagnostic_rejection(
                    quality,
                    (0.0 if not self.param_bool(
                        'require_good_quality_for_service') else
                     None if self.latest_scan_quality_at is None else
                     now - self.latest_scan_quality_at),
                    occlusion,
                    (0.0 if not self.param_bool(
                        'require_clear_occlusion_for_service') else
                     None if self.latest_occlusion_status_at is None else
                     now - self.latest_occlusion_status_at),
                    float(self.get_parameter('diagnostic_timeout_sec').value),
                    float(self.get_parameter(
                        'minimum_accepted_quality_score').value),
                    allowed_occlusion_states=(
                        ('CLEAR', 'PARTIALLY_OCCLUDED', 'HEAVILY_OCCLUDED')
                        if (classified_occlusion_barrier
                            or first_capture_semantic_barrier)
                        else ('CLEAR',)),
                )
                if reason:
                    return False, reason
        return True, ''

    def capture_ready(self, depth_burst=None):
        """Prepare one capture after all established gates pass."""
        ok, reason = self.capture_prerequisites_ready()
        if not ok:
            return False, reason
        if depth_burst is None:
            depth_burst, reason = self.latest_native_depth_burst()
            if reason:
                return False, reason
        prepared, reason = self.prepare_confidence_capture(depth_burst)
        if reason:
            return False, reason
        self.prepared_capture = prepared
        return True, ''

    @staticmethod
    def transform_matrix(message):
        transform = message.transform
        return rigid_transform_matrix(
            [transform.translation.x, transform.translation.y,
             transform.translation.z],
            [transform.rotation.x, transform.rotation.y,
             transform.rotation.z, transform.rotation.w])

    def refresh_camera_transforms(self, depth_message):
        stamp = Time.from_msg(depth_message.header.stamp)
        try:
            self.latest_camera_transform = self.tf_buffer.lookup_transform(
                str(self.get_parameter('base_frame').value),
                str(self.get_parameter('camera_optical_frame').value),
                stamp,
                timeout=Duration(seconds=float(self.get_parameter(
                    'camera_transform_timeout_sec').value)),
            )
            self.latest_color_from_depth_transform = (
                self.tf_buffer.lookup_transform(
                    str(self.get_parameter('camera_optical_frame').value),
                    str(depth_message.header.frame_id),
                    stamp,
                    timeout=Duration(seconds=float(self.get_parameter(
                        'camera_transform_timeout_sec').value)),
                ))
        except TransformException as exc:
            self.latest_camera_transform = None
            self.latest_color_from_depth_transform = None
            return 'timestamped camera transform is unavailable: %s' % exc
        return ''

    def valid_native_depth_bundles(self, bundles):
        """Return synchronized, unique native bundles in arrival order."""
        requested = int(self.get_parameter('capture_burst_frames').value)
        if requested < 1:
            return [], 'capture_burst_frames must be positive'
        maximum_age = float(self.get_parameter('max_bundle_age_sec').value)
        native_slop = float(self.get_parameter(
            'native_depth_confidence_slop_sec').value)
        selected = []
        seen = set()
        for bundle in bundles:
            depth, confidence, camera_info, received_at = bundle
            key = stamp_key(depth)
            if key in seen:
                continue
            # Prospective frames are fresh by construction.  Use their actual
            # receipt instant to validate timestamp/camera synchronization;
            # do not compare historical frames with the later service-return
            # time and incorrectly discard the beginning of the burst.
            reason = synchronized_bundle_rejection(
                depth, confidence, camera_info, received_at, received_at,
                maximum_age, native_slop)
            if reason:
                continue
            selected.append(bundle)
            seen.add(key)
        return selected, ''

    def latest_native_depth_burst(self):
        """Return the latest complete burst for interval diagnostics."""
        requested = int(self.get_parameter('capture_burst_frames').value)
        with self.native_bundle_condition:
            cached = list(self.native_bundle_cache)
        selected, reason = self.valid_native_depth_bundles(cached)
        if reason:
            return None, reason
        if len(selected) < requested:
            return None, (
                'confidence-qualified native depth is still catching up with '
                'the %d-frame burst (%d/%d available)'
                % (requested, len(selected), requested))
        return selected[-requested:], ''

    def collect_prospective_native_depth_burst(self, started_at):
        """Wait for exactly the requested number of new camera frames."""
        requested = int(self.get_parameter('capture_burst_frames').value)
        if requested < 1:
            return None, 'capture_burst_frames must be positive'
        capture_timeout = float(
            self.get_parameter('capture_timeout_sec').value)
        collection_timeout = capture_timeout - CAPTURE_RESPONSE_MARGIN_SEC
        if collection_timeout <= 0.0:
            return None, (
                'capture_timeout_sec must exceed the %.1f-second service '
                'response margin' % CAPTURE_RESPONSE_MARGIN_SEC)
        deadline = float(started_at) + collection_timeout
        available = 0
        with self.native_bundle_condition:
            while True:
                candidates = [
                    bundle for bundle in self.native_bundle_cache
                    if float(bundle[3]) > float(started_at)]
                selected, reason = self.valid_native_depth_bundles(candidates)
                if reason:
                    return None, reason
                available = len(selected)
                if available >= requested:
                    return selected[:requested], ''
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self.native_bundle_condition.wait(
                    timeout=min(0.1, remaining))
        elapsed = max(0.0, time.monotonic() - float(started_at))
        rate = float(available) / elapsed if elapsed > 0.0 else 0.0
        return None, (
            'timed out collecting %d new settled native depth frames '
            '(%d/%d received over %.2fs; synchronized rate %.2f Hz)'
            % (requested, available, requested, elapsed, rate))

    def prepare_confidence_capture(self, burst):
        """Prepare one immutable mask/RGB plus 20-frame depth observation."""
        if not isinstance(burst, (list, tuple)):
            return None, (
                'confidence-qualified native depth burst is unavailable')
        requested = int(self.get_parameter('capture_burst_frames').value)
        if len(burst) != requested:
            return None, (
                'confidence-qualified native depth burst requires exactly '
                '%d frames; received %d' % (requested, len(burst)))
        mask_message = self.latest_mask
        if mask_message is None:
            return None, 'missing detection mask'
        color_bundle = exact_stamped_item(
            self.color_bundle_cache, mask_message)
        if color_bundle is None:
            return None, (
                'confidence-qualified RGB-D bundle is still catching up with '
                'the exact detection-mask timestamp')
        color, aligned_depth, color_info, color_received_at = color_bundle
        bundle_reason = synchronized_bundle_rejection(
            color, aligned_depth, color_info, color_received_at,
            time.monotonic(),
            float(self.get_parameter('max_bundle_age_sec').value),
            float(self.get_parameter('synchronization_slop_sec').value),
        )
        if bundle_reason:
            return None, (
                'confidence-qualified RGB-D bundle ' +
                bundle_reason.lower().replace('rgb-d bundle ', ''))
        native_bundle = nearest_stamped_item(
            burst, color,
            float(self.get_parameter('synchronization_slop_sec').value))
        if native_bundle is None:
            return None, (
                'RGB and native depth timestamps are not synchronized')
        native_depth, confidence, native_info, native_received_at = \
            native_bundle
        native_slop = float(self.get_parameter(
            'native_depth_confidence_slop_sec').value)
        native_reason = synchronized_bundle_rejection(
            native_depth, confidence, native_info, native_received_at,
            time.monotonic(),
            float(self.get_parameter('max_bundle_age_sec').value),
            native_slop)
        if native_reason:
            return None, (
                'confidence-qualified native depth bundle ' +
                native_reason.lower().replace('rgb-d bundle ', ''))
        rgb_depth_delta = abs(
            stamp_seconds(color) - stamp_seconds(native_depth))
        if rgb_depth_delta > float(
                self.get_parameter('synchronization_slop_sec').value):
            return None, 'RGB and native depth timestamps are not synchronized'
        transform_reason = self.refresh_camera_transforms(native_depth)
        if transform_reason:
            return None, transform_reason
        try:
            rgb = np.asarray(self.bridge.imgmsg_to_cv2(
                color, desired_encoding='bgr8')).copy()
            aligned = np.asarray(self.bridge.imgmsg_to_cv2(
                aligned_depth, desired_encoding='passthrough')).copy()
            raw_depth_frames = []
            depth_encodings = []
            confidence_frames = []
            for burst_depth, burst_confidence, burst_info, _ in burst:
                if str(burst_depth.header.frame_id) != str(
                        native_depth.header.frame_id):
                    raise ValueError(
                        'native depth frame changed within settled burst')
                if (
                        int(burst_info.width) != int(native_info.width)
                        or int(burst_info.height) != int(native_info.height)
                        or not np.allclose(
                            np.asarray(burst_info.k, dtype=float),
                            np.asarray(native_info.k, dtype=float),
                            rtol=0.0, atol=1e-9)):
                    raise ValueError(
                        'native depth camera profile changed within burst')
                if str(burst_confidence.encoding).upper() not in (
                        'MONO8', '8UC1'):
                    raise ValueError(
                        'L515 confidence image encoding must be mono8')
                raw_depth_frames.append(np.asarray(
                    self.bridge.imgmsg_to_cv2(
                        burst_depth, desired_encoding='passthrough')).copy())
                depth_encodings.append(str(burst_depth.encoding))
                confidence_frames.append(np.asarray(
                    self.bridge.imgmsg_to_cv2(
                        burst_confidence,
                        desired_encoding='mono8')).copy())
            raw_depth, grades, burst_report = \
                temporal_confident_depth_median(
                    raw_depth_frames, depth_encodings, confidence_frames,
                    minimum_confidence=int(self.get_parameter(
                        'minimum_depth_confidence').value),
                    minimum_support_fraction=float(self.get_parameter(
                        'minimum_burst_support_fraction').value),
                )
            mask = np.asarray(self.bridge.imgmsg_to_cv2(
                mask_message, desired_encoding='mono8')).copy()
            if mask.shape != rgb.shape[:2]:
                raise ValueError(
                    'detection mask shape does not match exact RGB frame')
            qualified = qualify_target_depth(
                raw_depth, '16UC1', grades, mask,
                native_info.k, color_info.k, color_info.d,
                color_info.distortion_model,
                self.transform_matrix(
                    self.latest_color_from_depth_transform),
                minimum_confidence=int(self.get_parameter(
                    'minimum_depth_confidence').value),
                minimum_points=int(self.get_parameter(
                    'minimum_confident_target_points').value),
                minimum_confident_fraction=float(self.get_parameter(
                    'minimum_confident_target_fraction').value),
                minimum_component_fraction=float(self.get_parameter(
                    'minimum_primary_component_fraction').value),
                component_ambiguity_margin=float(self.get_parameter(
                    'depth_component_ambiguity_margin').value),
            )
            burst_report.update({
                'requested_frames': int(self.get_parameter(
                    'capture_burst_frames').value),
                'start_stamp': self.header_stamp(burst[0][0]),
                'end_stamp': self.header_stamp(burst[-1][0]),
                'span_sec': float(
                    stamp_seconds(burst[-1][0])
                    - stamp_seconds(burst[0][0])),
                'collection': '20_new_frames_after_capture_request',
                'rgb_policy': 'single exact-mask-correlated reference frame',
            })
            qualified['target']['depth_source'] = (
                'l515_temporal_median_confidence_qualified_native_depth')
            qualified['quality']['temporal_aggregation'] = dict(
                burst_report)
            capture_model_seed = capture_model_seed_from_qualified(
                mask,
                qualified,
                color_info.k,
                mask_message.header,
                self.camera_transform_metadata(
                    self.latest_camera_transform),
            )
        except DepthQualityRejected as exc:
            return None, str(exc)
        except (TypeError, ValueError, cv2.error) as exc:
            return None, 'confidence-qualified capture is invalid: %s' % exc
        return {
            'color_message': color,
            'aligned_depth_message': aligned_depth,
            'color_info': color_info,
            'native_depth_message': native_depth,
            'native_depth_info': native_info,
            'confidence_message': confidence,
            'mask_message': mask_message,
            'rgb': rgb,
            'aligned_depth': aligned,
            'mask': mask,
            'qualified': qualified,
            'depth_burst': burst_report,
            'rgb_native_depth_delta_sec': rgb_depth_delta,
            'native_depth_confidence_delta_sec': abs(
                stamp_seconds(native_depth) - stamp_seconds(confidence)),
            'mask_rgb_delta_sec': abs(
                stamp_seconds(mask_message) - stamp_seconds(color)),
            'camera_transform': self.latest_camera_transform,
            'color_from_depth_transform': (
                self.latest_color_from_depth_transform),
            'qualified_target_model_seed': capture_model_seed,
        }, ''

    def capture_frame(self, now):
        prepared = self.prepared_capture
        if not isinstance(prepared, dict):
            return False, 'confidence-qualified capture was not prepared'
        index = self.frame_index
        prefix = 'view_%03d' % index
        paths = {
            'rgb': os.path.join(self.frames_dir, prefix + '_rgb.png'),
            'depth_npy': os.path.join(self.frames_dir, prefix + '_depth.npy'),
            'depth_png': os.path.join(self.frames_dir, prefix + '_depth.png'),
            'mask': os.path.join(self.frames_dir, prefix + '_mask.png'),
            'native_depth_npy': os.path.join(
                self.frames_dir, prefix + '_native_depth.npy'),
            'native_depth_png': os.path.join(
                self.frames_dir, prefix + '_native_depth.png'),
            'confidence': os.path.join(
                self.frames_dir, prefix + '_confidence.png'),
            'target_depth': os.path.join(
                self.frames_dir, prefix + '_target_depth.png'),
            'target_support_mask': os.path.join(
                self.frames_dir, prefix + '_target_support_mask.png'),
            'metadata': os.path.join(
                self.frames_dir, prefix + '_metadata.yaml'),
        }
        aligned_depth_mm = depth_millimetres(
            prepared['aligned_depth'],
            prepared['aligned_depth_message'].encoding)
        qualified = prepared['qualified']
        images = {
            'rgb': prepared['rgb'],
            'depth_png': aligned_depth_mm,
            'mask': prepared['mask'],
            'native_depth_png': qualified['native_depth_mm'],
            'confidence': qualified['confidence'],
            'target_depth': qualified['target_depth_mm'],
            'target_support_mask': qualified['target_support_mask'],
        }
        arrays = {
            'depth_npy': prepared['aligned_depth'],
            'native_depth_npy': qualified['native_depth_mm'],
        }
        metadata = self.frame_metadata(index, now, paths, prepared)
        temporary = {}
        committed = []
        try:
            for key, image in images.items():
                temporary[key] = self.partial_path(paths[key])
                if not cv2.imwrite(temporary[key], image):
                    raise OSError('%s cv2.imwrite returned false' % key)
            for key, array in arrays.items():
                temporary[key] = self.partial_path(paths[key])
                with open(temporary[key], 'wb') as stream:
                    np.save(stream, np.asarray(array), allow_pickle=False)
            temporary['metadata'] = self.partial_path(paths['metadata'])
            self.write_yaml(temporary['metadata'], metadata)
            for key in list(images) + list(arrays) + ['metadata']:
                os.replace(temporary[key], paths[key])
                committed.append(paths[key])
        except Exception as exc:
            for path in temporary.values():
                if os.path.isfile(path):
                    os.remove(path)
            for path in committed:
                if os.path.isfile(path):
                    os.remove(path)
            reason = 'confidence-qualified frame save failed: %s' % exc
            self.note_skip(reason)
            self.publish_status('skipped', reason)
            return False, reason
        self.record_quality_count(metadata)
        self.record_occlusion_count(metadata)

        self.frame_index += 1
        self.last_capture_time = now
        self.write_dataset_manifest()
        self.publish_status('captured', 'saved frame %03d' % index, frame_index=index)
        self.publish_summary()
        self.last_capture_result = {
            'capture_result_schema_version': 1,
            'message': 'saved viewpoint %03d to %s' % (
                index, self.frames_dir),
            'frame_index': int(index),
            'metadata_file_path': paths['metadata'],
            'occlusion_state': str(metadata['occlusion_state']),
            'occlusion_score': float(metadata['occlusion_score']),
            'qualified_target_model_seed': dict(
                metadata['qualified_target_model_seed']),
        }
        if self.param_bool('debug'):
            self.get_logger().info('saved scan frame %03d to %s' % (index, self.frames_dir))
        self.prepared_capture = None
        return True, 'saved viewpoint %03d to %s' % (index, self.frames_dir)

    @staticmethod
    def partial_path(path):
        root, extension = os.path.splitext(path)
        return root + '.partial' + extension

    def frame_metadata(self, index, now, paths, prepared):
        target = self.target_metadata(self.latest_target)
        qualified = prepared['qualified']
        synchronized_target = dict(qualified['target'])
        synchronized_target['available'] = True
        synchronized_target['header'] = self.header_metadata(
            prepared['native_depth_message'].header)
        planned_count = self.planned_viewpoint_count()
        reachable_count = self.reachable_viewpoint_count()
        coverage_target = self.scan_coverage_target()
        quality = self.scan_quality_metadata()
        occlusion = self.occlusion_metadata()
        execution = self.execution_status_metadata(
            self.latest_execution_status)
        view_selection = capture_view_selection_provenance(
            self.plan_provenance_by_id.get(execution.get('plan_id', '')),
            execution)
        return {
            'capture_schema_version': 2,
            'frame_index': int(index),
            'capture_timestamp': self.ros_time_to_dict(now.to_msg()),
            'capture_wall_time': self.wall_time_string(),
            'rgb_topic_timestamp': self.header_stamp(
                prepared['color_message']),
            'depth_topic_timestamp': self.header_stamp(
                prepared['aligned_depth_message']),
            'native_depth_topic_timestamp': self.header_stamp(
                prepared['native_depth_message']),
            'confidence_topic_timestamp': self.header_stamp(
                prepared['confidence_message']),
            'mask_topic_timestamp': self.header_stamp(
                prepared['mask_message']),
            'camera_info': self.camera_info_metadata(prepared['color_info']),
            'native_depth_camera_info': self.camera_info_metadata(
                prepared['native_depth_info']),
            'camera_transform': self.camera_transform_metadata(
                prepared['camera_transform']),
            'qualified_target_model_seed': dict(
                prepared['qualified_target_model_seed']),
            'color_from_depth_transform': self.camera_transform_metadata(
                prepared['color_from_depth_transform']),
            'synchronization': {
                'mask_rgb_delta_sec': float(
                    prepared['mask_rgb_delta_sec']),
                'rgb_native_depth_delta_sec': float(
                    prepared['rgb_native_depth_delta_sec']),
                'native_depth_confidence_delta_sec': float(
                    prepared['native_depth_confidence_delta_sec']),
                'mask_rgb_exact': bool(
                    prepared['mask_rgb_delta_sec'] == 0.0),
                'maximum_rgb_native_depth_delta_sec': float(
                    self.get_parameter('synchronization_slop_sec').value),
                'maximum_native_depth_confidence_delta_sec': float(
                    self.get_parameter(
                        'native_depth_confidence_slop_sec').value),
            },
            'depth_burst': dict(prepared['depth_burst']),
            'task_id': str(self.get_parameter('task_id').value),
            'mission_sha256': str(
                self.get_parameter('mission_sha256').value),
            'target_label': str(self.get_parameter('target_label').value),
            'target_profile': str(self.get_parameter('target_profile').value),
            'target_prompt': str(self.get_parameter('target_prompt').value),
            'calibration_sha256': str(
                self.get_parameter('calibration_sha256').value),
            'target_3d': target,
            'target_3d_role': 'diagnostic_live_provenance_only',
            'synchronized_target_3d': synchronized_target,
            'target_valid': True,
            'confidence_quality': qualified['quality'],
            'planned_viewpoint_count': planned_count,
            'reachable_viewpoint_count': reachable_count,
            'scan_coverage_target': coverage_target,
            'scan_quality_available': quality['scan_quality_available'],
            'scan_quality_score': quality['scan_quality_score'],
            'scan_quality_label': quality['scan_quality_label'],
            'mask_area_px': quality['mask_area_px'],
            'valid_depth_ratio': quality['valid_depth_ratio'],
            'depth_mean_m': quality['depth_mean_m'],
            'depth_stddev_m': quality['depth_stddev_m'],
            'centredness_score': quality['centredness_score'],
            'edge_margin_score': quality['edge_margin_score'],
            'scan_quality_target_valid': quality['target_valid'],
            'occlusion_available': occlusion['occlusion_available'],
            'occlusion_state': occlusion['occlusion_state'],
            'occlusion_score': occlusion['occlusion_score'],
            'closer_region_area_px': occlusion['closer_region_area_px'],
            'closer_region_ratio': occlusion['closer_region_ratio'],
            'occlusion_target_depth_m': occlusion['target_depth_m'],
            'occlusion_reference_mask_area_px': occlusion[
                'reference_mask_area_px'],
            'occlusion_reference_target_depth_m': occlusion[
                'reference_target_depth_m'],
            'occlusion_reference_normalized_mask_area_m2': occlusion[
                'reference_normalized_mask_area_m2'],
            'occlusion_current_normalized_mask_area_m2': occlusion[
                'current_normalized_mask_area_m2'],
            'occlusion_visible_mask_ratio': occlusion['visible_mask_ratio'],
            'occlusion_reference_session_id': occlusion[
                'reference_session_id'],
            'occlusion_reason': occlusion['occlusion_reason'],
            'current_capture_mode': self.capture_mode(),
            # These top-level fields describe the execution that produced the
            # frame.  The capture node itself remains command-free/dry-run;
            # retain that separate authority explicitly instead of labelling
            # physical records as simulations.
            'dry_run': bool(execution.get(
                'dry_run', self.param_bool('dry_run'))),
            'real_arm_motion': bool(execution.get(
                'real_arm_motion', False)),
            'capture_node_dry_run': self.param_bool('dry_run'),
            'capture_node_real_arm_motion': False,
            'rgb_file_path': paths['rgb'],
            'depth_file_path': paths['depth_npy'],
            'depth_png_file_path': paths['depth_png'],
            'depth_encoding': str(
                prepared['aligned_depth_message'].encoding),
            'depth_npy_dtype': str(prepared['aligned_depth'].dtype),
            'depth_png_units': 'millimetres',
            'mask_file_path': paths['mask'],
            'native_depth_file_path': paths['native_depth_npy'],
            'native_depth_png_file_path': paths['native_depth_png'],
            'native_depth_encoding': str(
                prepared['native_depth_message'].encoding),
            'confidence_file_path': paths['confidence'],
            'confidence_encoding': str(
                prepared['confidence_message'].encoding),
            'confidence_grade_range': [0, 15],
            'target_depth_png_file_path': paths['target_depth'],
            'target_support_mask_file_path': paths['target_support_mask'],
            'metadata_file_path': paths['metadata'],
            'joint_state': self.joint_state_metadata(self.latest_joint_state),
            'scan_execution': execution,
            'view_selection': view_selection,
        }

    def publish_status(self, state, reason, frame_index=None):
        msg = String()
        payload = {
            'state': state,
            'reason': reason,
            'scan_dir': self.scan_dir,
            'frames_captured': int(self.frame_index),
            'captured_frame_count': int(self.frame_index),
            'max_frames_per_scan': int(self.get_parameter('max_frames_per_scan').value),
            'manifest_sha256': self.manifest_sha256,
            'manifest_path': os.path.join(self.scan_dir, 'manifest.json'),
            'dry_run': True,
            'real_arm_motion': False,
        }
        if frame_index is not None:
            payload['frame_index'] = int(frame_index)
        msg.data = json.dumps(payload, sort_keys=True)
        self.status_pub.publish(msg)

    def publish_summary(self):
        msg = String()
        payload = {
            'scan_dir': self.scan_dir,
            'frames_captured': int(self.frame_index),
            'captured_frame_count': int(self.frame_index),
            'max_frames_per_scan': int(self.get_parameter('max_frames_per_scan').value),
            'manifest_sha256': self.manifest_sha256,
            'manifest_path': os.path.join(self.scan_dir, 'manifest.json'),
            'planned_viewpoint_count': self.planned_viewpoint_count(),
            'reachable_viewpoint_count': self.reachable_viewpoint_count(),
            'scan_coverage_target': self.scan_coverage_target(),
            'planned_coverage_deg': self.planned_coverage_deg(),
            'reachable_coverage_deg': self.reachable_coverage_deg(),
            'useful_coverage_deg': self.useful_coverage_deg(),
            'useful_coverage_note': self.useful_coverage_note(),
            'good_frame_count': int(self.quality_counts['GOOD']),
            'acceptable_frame_count': int(self.quality_counts['ACCEPTABLE']),
            'poor_frame_count': int(self.quality_counts['POOR']),
            'invalid_frame_count': int(self.quality_counts['INVALID']),
            'useful_frame_count': int(
                self.quality_counts['GOOD'] + self.quality_counts['ACCEPTABLE']
            ),
            'occlusion_summary_available': self.occlusion_summary_available(),
            'clear_frame_count': int(self.occlusion_counts['CLEAR']),
            'partially_occluded_frame_count': int(self.occlusion_counts['PARTIALLY_OCCLUDED']),
            'heavily_occluded_frame_count': int(self.occlusion_counts['HEAVILY_OCCLUDED']),
            'lost_frame_count': int(self.occlusion_counts['LOST']),
            'unknown_occlusion_frame_count': int(self.occlusion_counts['UNKNOWN']),
            'skip_counts': self.skip_counts,
            'dry_run': True,
            'real_arm_motion': False,
        }
        msg.data = json.dumps(payload, sort_keys=True)
        self.summary_pub.publish(msg)

    def note_skip(self, reason):
        self.skip_counts[reason] = int(self.skip_counts.get(reason, 0)) + 1

    def planned_viewpoint_count(self):
        payload = self.latest_scan_viewpoints
        if isinstance(payload, dict):
            viewpoints = payload.get('viewpoints')
            if isinstance(viewpoints, list):
                return len(viewpoints)
            value = payload.get('candidate_viewpoints')
            if value is not None:
                return int(value)
        return 0

    def reachable_viewpoint_count(self):
        payload = self.latest_reachable_scan_viewpoints
        if isinstance(payload, dict):
            filter_info = payload.get('filter')
            if (
                    isinstance(filter_info, dict)
                    and filter_info.get('reachable_viewpoints') is not None):
                return int(filter_info.get('reachable_viewpoints'))
            viewpoints = payload.get('viewpoints')
            if isinstance(viewpoints, list):
                return int(sum(1 for viewpoint in viewpoints if viewpoint.get('reachable')))
        return 0

    def scan_coverage_target(self):
        payload = (
            self.latest_scan_coverage
            if isinstance(self.latest_scan_coverage, dict) else None)
        if payload is None and isinstance(self.latest_scan_viewpoints, dict):
            payload = self.latest_scan_viewpoints
        if payload is None:
            return 0.0
        for key in ('requested_scan_angle_deg', 'planned_scan_angle_deg'):
            value = payload.get(key)
            if value is not None:
                return float(value)
        viewpoints = payload.get('viewpoints')
        if isinstance(viewpoints, list):
            angles = []
            for viewpoint in viewpoints:
                if (
                        isinstance(viewpoint, dict)
                        and viewpoint.get('viewpoint_angle_deg') is not None):
                    angles.append(float(viewpoint.get('viewpoint_angle_deg')))
            if len(angles) >= 2:
                return float(max(angles) - min(angles))
        return 0.0

    def planned_coverage_deg(self):
        return (
            self.scan_coverage_from_payload(self.latest_scan_coverage)
            or self.scan_coverage_from_payload(self.latest_scan_viewpoints))

    def reachable_coverage_deg(self):
        return self.scan_coverage_from_payload(self.latest_reachable_scan_viewpoints)

    def useful_coverage_deg(self):
        reachable = self.reachable_coverage_deg()
        reachable_count = self.reachable_viewpoint_count()
        useful_count = self.quality_counts['GOOD'] + self.quality_counts['ACCEPTABLE']
        if reachable is None or reachable_count <= 0 or useful_count <= 0:
            return None
        useful_ratio = min(1.0, float(useful_count) / float(reachable_count))
        return float(reachable * useful_ratio)

    def useful_coverage_note(self):
        if self.useful_coverage_deg() is None:
            return 'unavailable: no exact viewpoint-to-frame mapping in dry-run capture'
        return 'approximate: scaled reachable coverage by useful captured frame count'

    def scan_coverage_from_payload(self, payload):
        if not isinstance(payload, dict):
            return None
        for key in (
                'planned_scan_angle_deg', 'requested_scan_angle_deg',
                'reachable_coverage_deg'):
            value = payload.get(key)
            if value is not None:
                return float(value)
        viewpoints = payload.get('viewpoints')
        if isinstance(viewpoints, list):
            angles = []
            for viewpoint in viewpoints:
                if not isinstance(viewpoint, dict):
                    continue
                if viewpoint.get('reachable') is False:
                    continue
                angle = viewpoint.get('viewpoint_angle_deg')
                if self.is_finite_number(angle):
                    angles.append(float(angle))
            if len(angles) >= 2:
                return float(max(angles) - min(angles))
        return None

    def scan_quality_metadata(self):
        payload = self.latest_scan_quality if isinstance(self.latest_scan_quality, dict) else None
        if payload is None:
            return self.empty_scan_quality_metadata()

        return {
            'scan_quality_available': True,
            'scan_quality_score': float(payload.get('quality_score', payload.get('score', 0.0))),
            'scan_quality_label': str(payload.get(
                'quality_label', payload.get('status', 'INVALID'))),
            'mask_area_px': int(payload.get('mask_area_px', 0)),
            'valid_depth_ratio': float(payload.get('valid_depth_ratio', 0.0)),
            'depth_mean_m': float(payload.get('depth_mean_m', 0.0)),
            'depth_stddev_m': float(payload.get('depth_stddev_m', 0.0)),
            'centredness_score': float(payload.get('centredness_score', 0.0)),
            'edge_margin_score': float(payload.get('edge_margin_score', 0.0)),
            'target_valid': bool(payload.get('target_valid', False)),
        }

    @staticmethod
    def empty_scan_quality_metadata():
        return {
            'scan_quality_available': False,
            'scan_quality_score': 0.0,
            'scan_quality_label': 'UNAVAILABLE',
            'mask_area_px': 0,
            'valid_depth_ratio': 0.0,
            'depth_mean_m': 0.0,
            'depth_stddev_m': 0.0,
            'centredness_score': 0.0,
            'edge_margin_score': 0.0,
            'target_valid': False,
        }

    def record_quality_count(self, metadata):
        if not metadata.get('scan_quality_available'):
            return
        label = str(metadata.get('scan_quality_label', '')).upper()
        if label in self.quality_counts:
            self.quality_counts[label] += 1

    def occlusion_metadata(self):
        payload = (
            self.latest_occlusion_status
            if isinstance(self.latest_occlusion_status, dict) else None)
        if payload is None:
            return self.empty_occlusion_metadata()
        return {
            'occlusion_available': True,
            'occlusion_state': str(payload.get('occlusion_state', 'UNKNOWN')),
            'occlusion_score': float(payload.get('occlusion_score', 0.0)),
            'closer_region_area_px': int(payload.get('closer_region_area_px', 0)),
            'closer_region_ratio': float(payload.get('closer_region_ratio', 0.0)),
            'target_depth_m': float(payload.get('target_depth_m', 0.0)),
            'reference_mask_area_px': float(
                payload.get('reference_mask_area_px', 0.0)),
            'reference_target_depth_m': float(
                payload.get('reference_target_depth_m', 0.0)),
            'reference_normalized_mask_area_m2': float(
                payload.get('reference_normalized_mask_area_m2', 0.0)),
            'current_normalized_mask_area_m2': float(
                payload.get('current_normalized_mask_area_m2', 0.0)),
            'visible_mask_ratio': float(payload.get('visible_mask_ratio', 0.0)),
            'reference_session_id': str(
                payload.get('reference_session_id', '')),
            'occlusion_reason': str(payload.get('reason', '')),
        }

    @staticmethod
    def empty_occlusion_metadata():
        return {
            'occlusion_available': False,
            'occlusion_state': 'UNAVAILABLE',
            'occlusion_score': 0.0,
            'closer_region_area_px': 0,
            'closer_region_ratio': 0.0,
            'target_depth_m': 0.0,
            'reference_mask_area_px': 0.0,
            'reference_target_depth_m': 0.0,
            'reference_normalized_mask_area_m2': 0.0,
            'current_normalized_mask_area_m2': 0.0,
            'visible_mask_ratio': 0.0,
            'reference_session_id': '',
            'occlusion_reason': '',
        }

    def record_occlusion_count(self, metadata):
        if not metadata.get('occlusion_available'):
            return
        state = str(metadata.get('occlusion_state', '')).upper()
        if state in self.occlusion_counts:
            self.occlusion_counts[state] += 1

    def occlusion_summary_available(self):
        return any(count > 0 for count in self.occlusion_counts.values())

    @staticmethod
    def parse_json_msg(msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return {'parse_error': True, 'raw': msg.data}
        return payload if isinstance(payload, dict) else {'payload': payload}

    @staticmethod
    def target_metadata(msg):
        if msg is None:
            return {'available': False}
        return {
            'available': True,
            'header': ScanCaptureNode.header_metadata(msg.header),
            'point': {
                'x': float(msg.point.x),
                'y': float(msg.point.y),
                'z': float(msg.point.z),
            },
            'depth': float(msg.depth),
            'valid_depth_ratio': float(msg.valid_depth_ratio),
            'depth_stddev': float(msg.depth_stddev),
            'roi_width': float(msg.roi_width),
            'roi_height': float(msg.roi_height),
            'source_u': float(msg.source_u),
            'source_v': float(msg.source_v),
            'detection_width': float(msg.detection_width),
            'detection_height': float(msg.detection_height),
            'depth_source': str(msg.depth_source),
            'measurement_confidence': float(msg.measurement_confidence),
            'valid': bool(msg.valid),
        }

    @staticmethod
    def camera_info_metadata(msg):
        if msg is None:
            return {'available': False}
        return {
            'available': True,
            'header': ScanCaptureNode.header_metadata(msg.header),
            'height': int(msg.height),
            'width': int(msg.width),
            'distortion_model': str(msg.distortion_model),
            'd': [float(v) for v in msg.d],
            'k': [float(v) for v in msg.k],
            'r': [float(v) for v in msg.r],
            'p': [float(v) for v in msg.p],
            'binning_x': int(msg.binning_x),
            'binning_y': int(msg.binning_y),
        }

    @staticmethod
    def camera_transform_metadata(msg):
        if msg is None:
            return {'available': False}
        translation = msg.transform.translation
        rotation = msg.transform.rotation
        matrix = rigid_transform_matrix(
            [translation.x, translation.y, translation.z],
            [rotation.x, rotation.y, rotation.z, rotation.w],
        )
        return {
            'available': True,
            'header': ScanCaptureNode.header_metadata(msg.header),
            'child_frame_id': str(msg.child_frame_id),
            'translation_m': [
                float(translation.x), float(translation.y), float(translation.z)],
            'quaternion_xyzw': [
                float(rotation.x), float(rotation.y),
                float(rotation.z), float(rotation.w)],
            'matrix_4x4': matrix.tolist(),
        }

    @staticmethod
    def joint_state_metadata(msg):
        if msg is None:
            return {'available': False}
        return {
            'available': True,
            'header': ScanCaptureNode.header_metadata(msg.header),
            'name': [str(value) for value in msg.name],
            'position': [float(value) for value in msg.position],
            'velocity': [float(value) for value in msg.velocity],
        }

    @staticmethod
    def execution_status_metadata(msg):
        if msg is None:
            return {'available': False}
        return {
            'available': True,
            'header': ScanCaptureNode.header_metadata(msg.header),
            'plan_id': str(msg.plan_id),
            'execution_mode': str(msg.execution_mode),
            'state': str(msg.state),
            'dry_run': bool(msg.dry_run),
            'real_arm_motion': bool(msg.real_arm_motion),
            'approval_required': bool(msg.approval_required),
            'current_view': int(msg.current_view),
            'total_views': int(msg.total_views),
            'commanded_speed_percent': float(msg.commanded_speed_percent),
        }

    @staticmethod
    def header_metadata(header):
        return {
            'stamp': ScanCaptureNode.ros_time_to_dict(header.stamp),
            'frame_id': str(header.frame_id),
        }

    @staticmethod
    def header_stamp(msg):
        if msg is None:
            return {'available': False}
        return {
            'available': True,
            'stamp': ScanCaptureNode.ros_time_to_dict(msg.header.stamp),
            'frame_id': str(msg.header.frame_id),
        }

    @staticmethod
    def ros_time_to_dict(stamp):
        return {'sec': int(stamp.sec), 'nanosec': int(stamp.nanosec)}

    def topic_metadata(self):
        return {
            'color_image': self.get_parameter('color_image_topic').value,
            'depth_image': self.get_parameter('depth_image_topic').value,
            'camera_info': self.get_parameter('camera_info_topic').value,
            'native_depth_image': self.get_parameter(
                'native_depth_image_topic').value,
            'native_depth_camera_info': self.get_parameter(
                'native_depth_camera_info_topic').value,
            'confidence_image': self.get_parameter(
                'confidence_image_topic').value,
            'mask': self.get_parameter('mask_topic').value,
            'target_3d': self.get_parameter('target_3d_topic').value,
            'scan_viewpoints': self.get_parameter('scan_viewpoints_topic').value,
            'reachable_scan_viewpoints': self.get_parameter(
                'reachable_scan_viewpoints_topic').value,
            'scan_coverage': self.get_parameter('scan_coverage_topic').value,
            'scan_quality': self.get_parameter('scan_quality_topic').value,
            'occlusion_status': self.get_parameter('occlusion_status_topic').value,
            'scan_capture_status': self.get_parameter('scan_capture_status_topic').value,
            'scan_summary': self.get_parameter('scan_summary_topic').value,
            'scan_execution_status': self.get_parameter(
                'scan_execution_status_topic').value,
            'plan_provenance': self.get_parameter(
                'plan_provenance_topic').value,
            'joint_state': self.get_parameter('joint_state_topic').value,
            'camera_transform': '%s -> %s' % (
                self.get_parameter('base_frame').value,
                self.get_parameter('camera_optical_frame').value),
            'capture_service': '/scan_capture/capture_view',
        }

    def confidence_policy_metadata(self):
        return {
            'minimum_grade': int(self.get_parameter(
                'minimum_depth_confidence').value),
            'grade_range': [0, 15],
            'minimum_target_points': int(self.get_parameter(
                'minimum_confident_target_points').value),
            'minimum_confident_fraction': float(self.get_parameter(
                'minimum_confident_target_fraction').value),
            'minimum_primary_component_fraction': float(self.get_parameter(
                'minimum_primary_component_fraction').value),
            'depth_component_ambiguity_margin': float(self.get_parameter(
                'depth_component_ambiguity_margin').value),
            'temporal_depth_burst': {
                'frames': int(self.get_parameter(
                    'capture_burst_frames').value),
                'estimator': 'per_pixel_median',
                'collection': 'new_frames_after_capture_request',
                'minimum_support_fraction': float(self.get_parameter(
                    'minimum_burst_support_fraction').value),
                'rgb_policy': 'single exact-mask-correlated reference frame',
            },
            'mask_erosion': 'adaptive one-native-depth-pixel equivalent',
            'target_cluster': (
                'scored multi-layer 8-connected component; '
                'ambiguous depth layers fail closed'),
        }

    def create_scan_dir(self):
        root = os.path.expanduser(str(self.get_parameter('dataset_root').value))
        stamp = datetime.now().strftime('scan_%Y%m%d_%H%M%S')
        scan_dir = os.path.join(root, stamp)
        os.makedirs(scan_dir, exist_ok=True)
        return scan_dir

    @staticmethod
    def file_sha256(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()

    def write_dataset_manifest(self):
        files = []
        for name in sorted(os.listdir(self.frames_dir)):
            path = os.path.join(self.frames_dir, name)
            if not os.path.isfile(path):
                continue
            files.append({
                'path': os.path.relpath(path, self.scan_dir),
                'bytes': int(os.path.getsize(path)),
                'sha256': self.file_sha256(path),
            })
        payload = {
            'schema_version': 2,
            'capture_schema_version': 2,
            'task_id': str(self.get_parameter('task_id').value),
            'mission_sha256': str(
                self.get_parameter('mission_sha256').value),
            'target_label': str(self.get_parameter('target_label').value),
            'target_profile': str(self.get_parameter('target_profile').value),
            'target_prompt': str(self.get_parameter('target_prompt').value),
            'calibration_sha256': str(
                self.get_parameter('calibration_sha256').value),
            'confidence_policy': self.confidence_policy_metadata(),
            'capture_count': int(self.frame_index),
            'files': files,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(',', ':'),
            ensure_ascii=True).encode('utf-8')
        self.manifest_sha256 = hashlib.sha256(encoded).hexdigest()
        payload['manifest_sha256'] = self.manifest_sha256
        path = os.path.join(self.scan_dir, 'manifest.json')
        temporary = path + '.tmp'
        with open(temporary, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write('\n')
        os.replace(temporary, path)

    @staticmethod
    def wall_time_string():
        return datetime.now().isoformat(timespec='seconds')

    @staticmethod
    def is_finite_number(value):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def write_yaml(path, data):
        with open(path, 'w') as handle:
            if yaml is not None:
                yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)
            else:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.write('\n')

    def param_bool(self, name):
        value = self.get_parameter(name).value
        if isinstance(value, str):
            return value.lower() in ('1', 'true', 'yes', 'on')
        return bool(value)

    def capture_mode(self):
        value = str(self.get_parameter('capture_mode').value).strip().lower()
        return value if value in ('interval', 'service') else 'invalid'


def main(args=None):
    rclpy.init(args=args)
    node = ScanCaptureNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
