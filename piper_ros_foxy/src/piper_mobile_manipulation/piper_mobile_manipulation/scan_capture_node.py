#!/usr/bin/env python3
import json
import math
import os
import time
from datetime import datetime

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from piper_mobile_manipulation.msg import ScanExecutionStatus, Target3D
from piper_mobile_manipulation.scan_capture import (
    depth_millimetres,
    synchronized_bundle_rejection,
)

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
        self.declare_parameter('mask_topic', '/piper/sam2_target_mask')
        self.declare_parameter('target_3d_topic', '/piper/target_3d')
        self.declare_parameter('scan_viewpoints_topic', '/piper/scan_viewpoints')
        self.declare_parameter('reachable_scan_viewpoints_topic', '/piper/reachable_scan_viewpoints')
        self.declare_parameter('scan_coverage_topic', '/piper/scan_coverage')
        self.declare_parameter('scan_quality_topic', '/piper/scan_quality')
        self.declare_parameter('occlusion_status_topic', '/piper/occlusion_status')
        self.declare_parameter('scan_capture_status_topic', '/piper/scan_capture_status')
        self.declare_parameter('scan_summary_topic', '/piper/scan_summary')
        self.declare_parameter(
            'scan_execution_status_topic', '/piper/scan_execution_status')
        self.declare_parameter('joint_state_topic', '/joint_states_single')

        self.declare_parameter('capture_mode', 'interval')
        self.declare_parameter('capture_interval_sec', 2.0)
        self.declare_parameter('max_frames_per_scan', 30)
        self.declare_parameter('max_bundle_age_sec', 1.0)
        self.declare_parameter('synchronization_slop_sec', 0.08)
        self.declare_parameter('require_valid_target', True)
        self.declare_parameter('require_mask', True)
        self.declare_parameter('require_depth', True)
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
        self.latest_scan_viewpoints = None
        self.latest_reachable_scan_viewpoints = None
        self.latest_scan_coverage = None
        self.latest_scan_quality = None
        self.latest_occlusion_status = None
        self.last_capture_time = None
        self.frame_index = 0
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
                'max_frames_per_scan': int(self.get_parameter('max_frames_per_scan').value),
                'capture_mode': self.capture_mode(),
                'capture_interval_sec': float(self.get_parameter('capture_interval_sec').value),
                'topics': self.topic_metadata(),
            },
        )

        self.status_pub = self.create_publisher(
            String, self.get_parameter('scan_capture_status_topic').value, 10
        )
        self.summary_pub = self.create_publisher(
            String, self.get_parameter('scan_summary_topic').value, 10
        )

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
        self._retained_subscriptions.append(self.create_subscription(
            String,
            self.get_parameter('scan_quality_topic').value,
            self.scan_quality_cb,
            10,
        ))
        self._retained_subscriptions.append(self.create_subscription(
            String,
            self.get_parameter('occlusion_status_topic').value,
            self.occlusion_status_cb,
            10,
        ))

        self.capture_service = self.create_service(
            Trigger, '~/capture_view', self.capture_view_cb)
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

    def mask_cb(self, msg):
        self.latest_mask = msg

    def target_cb(self, msg):
        self.latest_target = msg

    def joint_state_cb(self, msg):
        self.latest_joint_state = msg

    def execution_status_cb(self, msg):
        self.latest_execution_status = msg

    def scan_viewpoints_cb(self, msg):
        self.latest_scan_viewpoints = self.parse_json_msg(msg)

    def reachable_scan_viewpoints_cb(self, msg):
        self.latest_reachable_scan_viewpoints = self.parse_json_msg(msg)

    def scan_coverage_cb(self, msg):
        self.latest_scan_coverage = self.parse_json_msg(msg)

    def scan_quality_cb(self, msg):
        self.latest_scan_quality = self.parse_json_msg(msg)

    def occlusion_status_cb(self, msg):
        self.latest_occlusion_status = self.parse_json_msg(msg)

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
        ok, reason = self.capture_ready()
        if not ok:
            self.note_skip(reason)
            self.publish_status('skipped', reason)
            response.success = False
            response.message = reason
            return response
        saved, message = self.capture_frame(self.get_clock().now())
        response.success = bool(saved)
        response.message = str(message)
        return response

    def capture_ready(self):
        if not self.param_bool('dry_run'):
            return False, 'dry_run is false'
        if self.param_bool('enable_real_arm_motion'):
            return False, 'enable_real_arm_motion is true'
        bundle_reason = synchronized_bundle_rejection(
            self.latest_color,
            self.latest_depth,
            self.latest_camera_info,
            self.latest_bundle_received_at,
            time.monotonic(),
            float(self.get_parameter('max_bundle_age_sec').value),
            float(self.get_parameter('synchronization_slop_sec').value),
        )
        if bundle_reason:
            return False, bundle_reason
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
        return True, ''

    def capture_frame(self, now):
        index = self.frame_index
        prefix = 'view_%03d' % index
        rgb_path = os.path.join(self.frames_dir, prefix + '_rgb.png')
        depth_path = os.path.join(self.frames_dir, prefix + '_depth.npy')
        depth_png_path = os.path.join(self.frames_dir, prefix + '_depth.png')
        mask_path = os.path.join(self.frames_dir, prefix + '_mask.png')
        metadata_path = os.path.join(self.frames_dir, prefix + '_metadata.yaml')

        try:
            rgb = self.bridge.imgmsg_to_cv2(self.latest_color, desired_encoding='bgr8')
            if not cv2.imwrite(rgb_path, rgb):
                raise OSError('cv2.imwrite returned false')
        except Exception as exc:
            self.note_skip('RGB save failed')
            self.publish_status('skipped', 'RGB save failed: %s' % exc)
            return False, 'RGB save failed: %s' % exc

        depth_saved = False
        depth_dtype = ''
        if self.latest_depth is not None:
            try:
                depth = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding='passthrough')
                depth = np.asarray(depth)
                depth_dtype = str(depth.dtype)
                np.save(depth_path, depth)
                if not cv2.imwrite(
                        depth_png_path,
                        depth_millimetres(depth, self.latest_depth.encoding)):
                    raise OSError('depth PNG cv2.imwrite returned false')
                depth_saved = True
            except Exception as exc:
                if self.param_bool('require_depth'):
                    self.note_skip('depth save failed')
                    self.publish_status('skipped', 'depth save failed: %s' % exc)
                    return False, 'depth save failed: %s' % exc
                depth_path = ''
                depth_png_path = ''
                self.get_logger().warn('optional depth save failed: %s' % exc)

        mask_saved = False
        if self.latest_mask is not None:
            try:
                mask = self.bridge.imgmsg_to_cv2(self.latest_mask, desired_encoding='mono8')
                if not cv2.imwrite(mask_path, mask):
                    raise OSError('cv2.imwrite returned false')
                mask_saved = True
            except Exception as exc:
                if self.param_bool('require_mask'):
                    self.note_skip('mask save failed')
                    self.publish_status('skipped', 'mask save failed: %s' % exc)
                    return False, 'mask save failed: %s' % exc
                mask_path = ''
                self.get_logger().warn('optional mask save failed: %s' % exc)

        if not depth_saved:
            depth_path = ''
            depth_png_path = ''
        if not mask_saved:
            mask_path = ''

        metadata = self.frame_metadata(
            index,
            now,
            rgb_path,
            depth_path,
            depth_png_path,
            depth_dtype,
            mask_path,
            metadata_path,
        )
        self.write_yaml(metadata_path, metadata)
        self.record_quality_count(metadata)
        self.record_occlusion_count(metadata)

        self.frame_index += 1
        self.last_capture_time = now
        self.publish_status('captured', 'saved frame %03d' % index, frame_index=index)
        self.publish_summary()
        if self.param_bool('debug'):
            self.get_logger().info('saved scan frame %03d to %s' % (index, self.frames_dir))
        return True, 'saved viewpoint %03d to %s' % (index, self.frames_dir)

    def frame_metadata(
            self, index, now, rgb_path, depth_path, depth_png_path, depth_dtype,
            mask_path, metadata_path):
        target = self.target_metadata(self.latest_target)
        planned_count = self.planned_viewpoint_count()
        reachable_count = self.reachable_viewpoint_count()
        coverage_target = self.scan_coverage_target()
        quality = self.scan_quality_metadata()
        occlusion = self.occlusion_metadata()
        return {
            'frame_index': int(index),
            'capture_timestamp': self.ros_time_to_dict(now.to_msg()),
            'capture_wall_time': self.wall_time_string(),
            'rgb_topic_timestamp': self.header_stamp(self.latest_color),
            'depth_topic_timestamp': self.header_stamp(self.latest_depth),
            'camera_info': self.camera_info_metadata(self.latest_camera_info),
            'target_3d': target,
            'target_valid': bool(self.latest_target.valid) if self.latest_target is not None else False,
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
            'dry_run': True,
            'real_arm_motion': False,
            'rgb_file_path': rgb_path,
            'depth_file_path': depth_path,
            'depth_png_file_path': depth_png_path,
            'depth_encoding': str(self.latest_depth.encoding),
            'depth_npy_dtype': depth_dtype,
            'depth_png_units': 'millimetres',
            'mask_file_path': mask_path,
            'metadata_file_path': metadata_path,
            'joint_state': self.joint_state_metadata(self.latest_joint_state),
            'scan_execution': self.execution_status_metadata(
                self.latest_execution_status),
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
            if isinstance(filter_info, dict) and filter_info.get('reachable_viewpoints') is not None:
                return int(filter_info.get('reachable_viewpoints'))
            viewpoints = payload.get('viewpoints')
            if isinstance(viewpoints, list):
                return int(sum(1 for viewpoint in viewpoints if viewpoint.get('reachable')))
        return 0

    def scan_coverage_target(self):
        payload = self.latest_scan_coverage if isinstance(self.latest_scan_coverage, dict) else None
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
                if isinstance(viewpoint, dict) and viewpoint.get('viewpoint_angle_deg') is not None:
                    angles.append(float(viewpoint.get('viewpoint_angle_deg')))
            if len(angles) >= 2:
                return float(max(angles) - min(angles))
        return 0.0

    def planned_coverage_deg(self):
        return self.scan_coverage_from_payload(self.latest_scan_coverage) or self.scan_coverage_from_payload(
            self.latest_scan_viewpoints
        )

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
        for key in ('planned_scan_angle_deg', 'requested_scan_angle_deg', 'reachable_coverage_deg'):
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
            'scan_quality_label': str(payload.get('quality_label', payload.get('status', 'INVALID'))),
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
        payload = self.latest_occlusion_status if isinstance(self.latest_occlusion_status, dict) else None
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
            'mask': self.get_parameter('mask_topic').value,
            'target_3d': self.get_parameter('target_3d_topic').value,
            'scan_viewpoints': self.get_parameter('scan_viewpoints_topic').value,
            'reachable_scan_viewpoints': self.get_parameter('reachable_scan_viewpoints_topic').value,
            'scan_coverage': self.get_parameter('scan_coverage_topic').value,
            'scan_quality': self.get_parameter('scan_quality_topic').value,
            'occlusion_status': self.get_parameter('occlusion_status_topic').value,
            'scan_capture_status': self.get_parameter('scan_capture_status_topic').value,
            'scan_summary': self.get_parameter('scan_summary_topic').value,
            'scan_execution_status': self.get_parameter(
                'scan_execution_status_topic').value,
            'joint_state': self.get_parameter('joint_state_topic').value,
            'capture_service': '/scan_capture/capture_view',
        }

    def create_scan_dir(self):
        root = os.path.expanduser(str(self.get_parameter('dataset_root').value))
        stamp = datetime.now().strftime('scan_%Y%m%d_%H%M%S')
        scan_dir = os.path.join(root, stamp)
        os.makedirs(scan_dir, exist_ok=True)
        return scan_dir

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
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
