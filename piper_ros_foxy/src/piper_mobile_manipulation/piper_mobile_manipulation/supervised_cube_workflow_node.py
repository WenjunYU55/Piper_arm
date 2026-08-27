#!/usr/bin/env python3
"""Coordinate obstacle removal and adaptive cube scanning without moving the arm."""

import json
import math
import struct
import time
import uuid

import rclpy
from geometry_msgs.msg import PointStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster

from piper_mobile_manipulation.msg import (
    ObstacleInstance3DArray,
    TrackedTarget,
    TrackingHealth,
)
from piper_mobile_manipulation.occlusion_policy import (
    OccluderEvidence, evidence_rejection,
)
from piper_mobile_manipulation.execution.modes import (
    measured_target_lock_rejection,
)
from piper_mobile_manipulation.supervised_workflow import (
    capture_cloud_ready, choose_removal_plan, cloud_model,
    corroborated_target_motion_rejection, distance, point,
    semantic_scene_correlation_rejection, should_cache_capture_cloud,
    tracking_allows_target_motion_check,
)


def target_motion_is_terminal(allow_motion_during_scan, workflow_state):
    """Keep the pre-scan lock strict without aborting an active moving-target scan."""
    return not (
        bool(allow_motion_during_scan)
        and str(workflow_state) in ('SCAN_READY', 'WAIT_CAPTURE')
    )


class SupervisedCubeWorkflowNode(Node):
    def __init__(self):
        super().__init__('supervised_cube_workflow')
        defaults = {
            'obstacle_topic': '/piper/obstacle_instances_3d',
            'landmark_topic': '/piper/target_landmark',
            'landmark_status_topic': '/piper/target_landmark_status',
            'tracked_target_topic': '/piper/tracked_target',
            'tracking_health_topic': '/piper/tracking_health',
            'target_status_topic': '/piper/target_status',
            'scan_quality_topic': '/piper/scan_quality',
            'occlusion_status_topic': '/piper/occlusion_status',
            'cloud_topic': '/piper/target_cloud',
            'scene_cloud_topic': '/piper/scene_cloud',
            'cloud_status_topic': '/piper/target_cloud_status',
            'cloud_request_topic': '/piper/target_cloud_request',
            'status_topic': '/piper/supervised_workflow_status',
            'plan_topic': '/piper/removal_plan',
            'target_model_topic': '/piper/target_model',
            'marker_topic': '/piper/supervised_workflow_markers',
            'heavy_refresh_request_topic': '/piper/heavy_refresh_request',
            'heavy_refresh_status_topic': '/piper/heavy_refresh_status',
            'movable_whitelist': ['pen', 'marker', 'stick'],
            'min_views': 13, 'max_views': 13, 'min_quality_score': 0.40,
            'request_optional_cloud_refinement': False,
            'center_convergence_m': 0.005, 'target_motion_abort_m': 0.020,
            'allow_target_motion_during_scan': True,
            'target_surface_measurement_uncertainty_m': 0.015,
            'obstacle_displacement_m': 0.050, 'data_timeout_sec': 2.0,
            'max_tracking_measurement_age_sec': 0.75,
            'target_clearance_m': 0.040, 'drop_target_clearance_m': 0.120,
            'drop_obstacle_clearance_m': 0.080, 'drop_search_radius_m': 0.180,
            'max_grasp_width_m': 0.070, 'approach_height_m': 0.100,
            'pre_push_offset_m': 0.080, 'push_distance_m': 0.060,
            'workspace_x_min': 0.10, 'workspace_x_max': 0.70,
            'workspace_y_min': -0.40, 'workspace_y_max': 0.40,
            'workspace_z_min': 0.02, 'workspace_z_max': 0.60,
            'enforce_static_workspace': False,
            'require_observed_drop_support': True,
            'drop_support_radius_m': 0.055,
            'drop_support_min_points': 16,
            'drop_support_max_stddev_m': 0.008,
            'occlusion_probe_timeout_sec': 70.0,
            'max_contact_actions': 6,
            'first_push_distance_m': 0.010,
            'later_push_distance_m': 0.030,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.state = 'IDLE'
        self.landmark = None
        self.target_landmark = None
        self.initial_landmark = None
        self.target_landmark_state = 'UNKNOWN'
        self.target_landmark_status = {}
        self.tracked_target = None
        self.tracking_health = None
        self.target_status = 'UNKNOWN'
        self.obstacles = None
        self.obstacles_at_plan = {}
        self.plan = None
        self.quality = None
        self.occlusion = None
        self.cloud_points = []
        self.scene_points = []
        self.cloud_frame = ''
        self.cloud_status = None
        self.pending_cloud_msg = None
        self.accepted_views = 0
        self.session_id = ''
        self.modeled_views = 0
        self.centers = []
        self.target_center = None
        self.updated = {}
        self.last_idle_lock_status = None
        self.last_reason = ''
        self.occlusion_probe_request_id = ''
        self.occlusion_probe_status = None
        self.occlusion_probe_scene = None
        self.occlusion_probe_started = None
        self.occlusion_probe_attempt = 0
        self.occlusion_probe_waiting_retry = False
        self.contact_actions = 0
        self.push_attempts_by_track = {}
        self.post_action_verification = False

        self.status_pub = self.create_publisher(String, defaults['status_topic'], 10)
        self.plan_pub = self.create_publisher(String, defaults['plan_topic'], 10)
        self.model_pub = self.create_publisher(String, defaults['target_model_topic'], 10)
        self.marker_pub = self.create_publisher(MarkerArray, defaults['marker_topic'], 10)
        self.target_tf = TransformBroadcaster(self)
        self.cloud_request_pub = self.create_publisher(String, defaults['cloud_request_topic'], 10)
        self.heavy_refresh_pub = self.create_publisher(
            String, defaults['heavy_refresh_request_topic'], 10)
        # Retain explicit references for Foxy: otherwise endpoints can disappear
        # after discovery when the Python wrapper is garbage-collected.
        self._input_subscriptions = [
            self.create_subscription(
                ObstacleInstance3DArray, defaults['obstacle_topic'], self.obstacle_cb, 10),
            self.create_subscription(
                PointStamped, defaults['landmark_topic'], self.landmark_cb, 10),
            self.create_subscription(
                String, defaults['landmark_status_topic'], self.landmark_status_cb, 10),
            self.create_subscription(
                TrackedTarget, defaults['tracked_target_topic'],
                self.tracked_target_cb, 10),
            self.create_subscription(
                TrackingHealth, defaults['tracking_health_topic'],
                self.tracking_health_cb, 10),
            self.create_subscription(
                String, defaults['target_status_topic'], self.target_status_cb, 10),
            self.create_subscription(
                String, defaults['scan_quality_topic'], self.quality_cb, 10),
            self.create_subscription(
                String, defaults['occlusion_status_topic'],
                self.occlusion_cb, 10),
            self.create_subscription(
                PointCloud2, defaults['cloud_topic'], self.cloud_cb,
                qos_profile_sensor_data),
            self.create_subscription(
                PointCloud2, defaults['scene_cloud_topic'],
                self.scene_cloud_cb, qos_profile_sensor_data),
            self.create_subscription(
                String, defaults['cloud_status_topic'], self.cloud_status_cb, 10),
            self.create_subscription(
                String, defaults['heavy_refresh_status_topic'],
                self.heavy_refresh_status_cb, 10),
        ]
        self.create_service(Trigger, '~/start', self.start_cb)
        self.create_service(Trigger, '~/approve_plan', self.approve_cb)
        self.create_service(Trigger, '~/confirm_action_complete', self.confirm_action_cb)
        self.create_service(Trigger, '~/capture_view', self.capture_view_cb)
        self.create_service(Trigger, '~/finish_scan', self.finish_scan_cb)
        self.create_service(Trigger, '~/abort', self.abort_cb)
        self.create_service(Trigger, '~/diagnostic_state', self.diagnostic_state_cb)
        self.create_timer(0.5, self.tick)
        self.publish_status('dry-run coordinator ready; no arm command publisher exists')

    def now(self):
        return time.monotonic()

    def mark(self, key):
        self.updated[key] = self.now()

    def fresh(self, key):
        timeout = float(self.get_parameter('data_timeout_sec').value)
        return self.now() - self.updated.get(key, -1e9) <= timeout

    def obstacle_cb(self, msg):
        self.obstacles = msg
        self.mark('obstacles')
        self.cache_correlated_probe_scene(msg)

    def cache_correlated_probe_scene(self, scene):
        if scene is None or self.occlusion_probe_status is None:
            return
        if not semantic_scene_correlation_rejection(
                self.occlusion_probe_request_id,
                self.occlusion_probe_status,
                scene):
            self.occlusion_probe_scene = scene
            self.mark('occlusion_probe_scene')

    def heavy_refresh_status_cb(self, msg):
        payload = self.parse(msg)
        state = str(payload.get('state', '')).lower()
        if (
                self.state != 'INITIALIZING'
                or not self.occlusion_probe_request_id):
            return
        if state == 'idle' and self.occlusion_probe_status is None:
            if (
                    self.occlusion_probe_waiting_retry
                    and self.occlusion_probe_attempt < 2):
                self.publish_occlusion_probe()
            return
        if str(payload.get('request_id', '')) != self.occlusion_probe_request_id:
            return
        if state == 'request_ignored_busy':
            self.occlusion_probe_waiting_retry = True
            self.publish_status(
                'semantic worker is busy; waiting for one bounded '
                'occlusion-probe retry')
            return
        if state in ('request_rejected', 'request_failed'):
            self.abort(
                'dedicated occlusion probe failed: '
                + str(payload.get('error', state)))
            return
        if state == 'worker_result_rejected':
            self.abort(
                'dedicated occlusion probe could not re-detect the locked target')
            return
        if state == 'published':
            self.occlusion_probe_status = payload
            self.mark('occlusion_probe')
            self.cache_correlated_probe_scene(self.obstacles)
            self.publish_status(
                'dedicated semantic occlusion result ready; waiting for its '
                'exact-stamp 3D obstacle scene')

    def publish_occlusion_probe(self):
        self.occlusion_probe_attempt += 1
        self.occlusion_probe_waiting_retry = False
        stamp = self.get_clock().now().to_msg()
        self.occlusion_probe_request_id = '%s-occlusion-%d' % (
            self.session_id, self.occlusion_probe_attempt)
        request = String()
        request.data = json.dumps({
            'request_id': self.occlusion_probe_request_id,
            'reason': 'workflow_occlusion_probe',
            'min_image_stamp': {
                'sec': int(stamp.sec),
                'nanosec': int(stamp.nanosec),
            },
        }, sort_keys=True)
        self.heavy_refresh_pub.publish(request)
        self.publish_status(
            'requesting a fresh settled semantic occlusion assessment')

    def landmark_cb(self, msg):
        self.target_landmark = point(msg.point)
        self.mark('target_landmark')

    def tracked_target_cb(self, msg):
        self.tracked_target = msg
        self.mark('tracked_target')
        if not msg.valid:
            return
        self.landmark = point(msg.position)

    def landmark_status_cb(self, msg):
        payload = self.parse(msg)
        self.target_landmark_status = payload
        self.target_landmark_state = str(
            payload.get('state', 'UNKNOWN')).upper()
        self.mark('target_landmark_status')

    def tracking_health_cb(self, msg):
        self.tracking_health = msg
        self.mark('tracking_health')
        if (
                self.initial_landmark
                and self.landmark
                and self.fresh('tracked_target')
                and self.fresh('target_status')
                and self.target_status in ('TRACKING', 'LOCKED')
                and tracking_allows_target_motion_check(
                    msg,
                    float(self.get_parameter(
                        'max_tracking_measurement_age_sec').value))):
            threshold = float(
                self.get_parameter('target_motion_abort_m').value)
            rejection = corroborated_target_motion_rejection(
                distance(self.landmark, self.initial_landmark),
                threshold,
                self.target_landmark_status,
                self.fresh('target_landmark_status'),
                float(self.get_parameter(
                    'target_surface_measurement_uncertainty_m').value),
            )
            if (
                    rejection
                    and target_motion_is_terminal(
                        self.get_parameter(
                            'allow_target_motion_during_scan').value,
                        self.state)):
                self.abort(rejection)

    def target_status_cb(self, msg):
        self.target_status = str(msg.data).upper()
        self.mark('target_status')

    def quality_cb(self, msg):
        self.quality = self.parse(msg)
        self.mark('quality')

    def occlusion_cb(self, msg):
        self.occlusion = self.parse(msg)
        self.mark('occlusion')

    def cloud_status_cb(self, msg):
        self.cloud_status = self.parse(msg)
        self.mark('cloud_status')
        if (
                self.state == 'WAIT_CAPTURE'
                and self.cloud_status.get('state') == 'refined_capture_rejected'):
            self.abort(
                'full-resolution cloud capture failed: %s'
                % self.cloud_status.get('error', 'unknown refinement failure'))
        elif (
                self.state == 'WAIT_CAPTURE'
                and self.cloud_status.get('state') == 'accumulating'
                and self.cloud_status.get('mask_source')
                == 'full_resolution_refinement'):
            self.accepted_views += 1
            self.state = 'SCAN_READY'
            self.process_pending_cloud()
            self.publish_status('full-resolution view accepted')

    def cloud_cb(self, msg):
        # The capture cloud and its acceptance status may arrive in either
        # order. Cache the message while WAIT_CAPTURE, but defer expensive
        # deserialization until the corresponding acceptance is known.
        if not should_cache_capture_cloud(
                self.state, self.accepted_views, self.modeled_views):
            return
        self.pending_cloud_msg = msg
        self.mark('cloud')
        self.process_pending_cloud()

    def scene_cloud_cb(self, msg):
        if msg.header.frame_id != 'base_link':
            return
        self.scene_points = self.read_xyz(msg)
        self.mark('scene_cloud')

    def process_pending_cloud(self):
        if not capture_cloud_ready(
                self.accepted_views,
                self.modeled_views,
                self.pending_cloud_msg is not None):
            return
        msg = self.pending_cloud_msg
        if msg.header.frame_id != 'base_link':
            self.abort('target cloud frame is not base_link')
            return
        self.cloud_points = self.read_xyz(msg)
        self.cloud_frame = msg.header.frame_id
        self.publish_model()
        self.modeled_views = self.accepted_views
        self.pending_cloud_msg = None

    def start_cb(self, _request, response):
        if self.state not in ('IDLE', 'COMPLETE', 'ABORTED'):
            return self.reply(response, False, 'workflow already active')
        self.accepted_views, self.modeled_views, self.centers = 0, 0, []
        self.session_id = uuid.uuid4().hex
        self.target_center = None
        self.plan = None
        self.initial_landmark = None
        self.obstacles = None
        self.updated.pop('obstacles', None)
        self.state = 'INITIALIZING'
        self.occlusion_probe_status = None
        self.occlusion_probe_scene = None
        self.occlusion_probe_started = self.now()
        self.occlusion_probe_attempt = 0
        self.occlusion_probe_waiting_retry = False
        self.contact_actions = 0
        self.push_attempts_by_track = {}
        self.post_action_verification = False
        self.publish_occlusion_probe()
        return self.reply(response, True, 'workflow started')

    def approve_cb(self, _request, response):
        if self.state != 'PLAN_READY' or not self.plan or not self.plan.get('valid'):
            return self.reply(response, False, 'no valid removal plan ready')
        self.state = 'WAIT_OPERATOR_ACTION'
        self.publish_status('plan approved; operator may perform the displayed action')
        return self.reply(response, True, 'dry-run plan approved; no motion was commanded')

    def confirm_action_cb(self, _request, response):
        if self.state != 'WAIT_OPERATOR_ACTION':
            return self.reply(response, False, 'not waiting for operator action')
        self.contact_actions += 1
        if self.plan.get('action') == 'push':
            track_id = str(self.plan.get('track_id', ''))
            self.push_attempts_by_track[track_id] = int(
                self.push_attempts_by_track.get(track_id, 0)) + 1
        self.state = 'INITIALIZING'
        self.post_action_verification = True
        self.occlusion_probe_status = None
        self.occlusion_probe_scene = None
        self.occlusion_probe_started = self.now()
        self.occlusion_probe_attempt = 0
        self.obstacles = None
        self.updated.pop('obstacles', None)
        self.publish_occlusion_probe()
        self.publish_status(
            'action recorded; requesting fresh post-action semantic RGB-D')
        return self.reply(response, True, 'post-action verification started')

    def capture_view_cb(self, _request, response):
        if self.state != 'SCAN_READY':
            return self.reply(response, False, 'scan is not ready for a view')
        if self.accepted_views >= int(self.get_parameter('max_views').value):
            return self.reply(response, False, 'maximum view count reached; finish the scan')
        # The executor calls this only after the synchronized RGB-D/mask/
        # metadata files have been saved. Quality, occlusion and accumulated
        # cloud products remain useful diagnostics, but they do not invalidate
        # a completed viewpoint record or stop the fixed 13-view sweep.
        self.accepted_views += 1
        self.modeled_views = self.accepted_views
        if bool(self.get_parameter(
                'request_optional_cloud_refinement').value):
            msg = String()
            msg.data = 'capture'
            self.cloud_request_pub.publish(msg)
        self.publish_status(
            'synchronized RGB-D viewpoint accepted; diagnostic quality and '
            'occlusion recorded')
        return self.reply(
            response, True,
            'synchronized RGB-D viewpoint accepted')

    def finish_scan_cb(self, _request, response):
        if self.state != 'SCAN_READY':
            return self.reply(response, False, 'scan is not ready to finish')
        if self.modeled_views < self.accepted_views:
            return self.reply(response, False, 'waiting for the latest target model')
        minimum = int(self.get_parameter('min_views').value)
        if self.accepted_views < minimum:
            return self.reply(response, False, 'need at least %d accepted views' % minimum)
        convergence = float(self.get_parameter('center_convergence_m').value)
        converged = (len(self.centers) >= 2 and
                     distance(self.centers[-1], self.centers[-2]) <= convergence)
        if not converged and self.accepted_views < int(self.get_parameter('max_views').value):
            return self.reply(response, False, 'center has not converged; capture another view')
        msg = String()
        msg.data = 'save'
        self.cloud_request_pub.publish(msg)
        self.state = 'COMPLETE'
        self.publish_model()
        self.publish_status('scan complete; cloud save requested')
        return self.reply(response, True, 'scan complete')

    def abort_cb(self, _request, response):
        self.abort('operator requested abort')
        return self.reply(response, True, 'workflow aborted; no arm stop command was required')

    def diagnostic_state_cb(self, _request, response):
        now = self.now()
        timeout = float(self.get_parameter('data_timeout_sec').value)
        lock_rejection = self.measured_lock_rejection()
        payload = {
            'state': self.state,
            'reason': self.last_reason,
            'measured_lock_ready': not bool(lock_rejection),
            'measured_lock_rejection': lock_rejection,
            'lock_source': 'tracked_target',
            'target_landmark_diagnostic_state': self.target_landmark_state,
            'landmark_present': self.landmark is not None,
            'target_landmark_present': self.target_landmark is not None,
            'obstacles_present': self.obstacles is not None,
            'quality_present': self.quality is not None,
            'occlusion_present': self.occlusion is not None,
            'occlusion_state': (
                str(self.occlusion.get('occlusion_state', 'UNKNOWN'))
                if isinstance(self.occlusion, dict) else 'UNKNOWN'),
            'occlusion_probe_request_id': self.occlusion_probe_request_id,
            'occlusion_probe_complete': self.occlusion_probe_status is not None,
            'occlusion_probe_scene_complete': self.occlusion_probe_scene is not None,
            'accepted_views': self.accepted_views,
            'session_id': self.session_id,
            'modeled_views': self.modeled_views,
            'data_timeout_sec': timeout,
            'input_age_sec': {
                key: round(now - stamp, 3)
                for key, stamp in sorted(self.updated.items())
            },
        }
        if self.obstacles is not None:
            payload['obstacle_count'] = len(self.obstacles.instances)
            payload['invalid_or_blocked_obstacles'] = sum(
                1 for item in self.obstacles.instances
                if not item.valid
                or int(item.classification) != item.CLASSIFICATION_MOVABLE
            )
        return self.reply(response, True, json.dumps(payload, sort_keys=True))

    def tick(self):
        if self.target_center is not None:
            self.publish_target_frame(self.target_center)
        lock_rejection = self.measured_lock_rejection()
        if self.state == 'IDLE':
            lock_status = (self.state, lock_rejection)
            if lock_status != self.last_idle_lock_status:
                self.last_idle_lock_status = lock_status
                self.publish_status(
                    'measured target lock is available for explicit scan adoption'
                    if not lock_rejection
                    else 'waiting for a measured target lock')
        else:
            self.last_idle_lock_status = None
        if self.state == 'INITIALIZING':
            if (
                    self.occlusion_probe_started is not None
                    and self.now() - self.occlusion_probe_started > float(
                        self.get_parameter('occlusion_probe_timeout_sec').value)):
                self.abort('dedicated semantic occlusion assessment timed out')
                return
            scene_rejection = semantic_scene_correlation_rejection(
                self.occlusion_probe_request_id,
                self.occlusion_probe_status,
                self.occlusion_probe_scene,
            )
            if (
                    not lock_rejection
                    and self.landmark
                    and self.occlusion_probe_scene is not None
                    and not scene_rejection):
                movable_probe = [
                    item for item in self.occlusion_probe_scene.instances
                    if item.valid
                    and int(item.classification)
                    == item.CLASSIFICATION_MOVABLE]
                if (
                        movable_probe
                        and any(int(item.observation_count) < 2
                                for item in movable_probe)
                        and self.occlusion_probe_attempt < 2):
                    self.occlusion_probe_status = None
                    self.occlusion_probe_scene = None
                    self.obstacles = None
                    self.updated.pop('obstacles', None)
                    self.publish_occlusion_probe()
                    self.publish_status(
                        'candidate rigid occluder seen once; requesting the '
                        'mandatory second settled observation')
                    return
                self.obstacles = self.occlusion_probe_scene
                self.mark('obstacles')
                self.initial_landmark = self.landmark
                if self.post_action_verification:
                    self.post_action_verification = False
                    self.publish_status(
                        'fresh post-action scene received; reassessing '
                        'remaining target occlusion')
                self.assess_scene()
        elif (
                self.state == 'VERIFY_ACTION'
                and self.fresh('obstacles')
                and not lock_rejection):
            if self.verify_action():
                self.state = 'INITIALIZING'
                self.publish_status('obstacle action verified; reassessing scene')

    def measured_lock_rejection(self):
        return measured_target_lock_rejection(
            self.tracked_target,
            self.tracking_health,
            self.target_status,
            self.updated.get('tracked_target', -1e9),
            self.updated.get('tracking_health', -1e9),
            self.updated.get('target_status', -1e9),
            self.now(),
            float(self.get_parameter('data_timeout_sec').value),
            float(self.get_parameter(
                'max_tracking_measurement_age_sec').value),
        )

    def assess_scene(self):
        movable = [item for item in self.obstacles.instances
                   if item.valid and int(item.classification) == item.CLASSIFICATION_MOVABLE]
        unsafe = [item for item in self.obstacles.instances
                  if not item.valid or int(item.classification) != item.CLASSIFICATION_MOVABLE]
        if unsafe:
            self.abort(
                'occlusion scene contains unsafe, blocked, or invalid '
                'obstacle geometry')
            return
        if not movable:
            self.state = 'SCAN_READY'
            self.publish_status('scene is clear; ready for first scan view')
            return
        harmful = []
        unqualified = []
        for item in movable:
            evidence = self.occluder_evidence(item)
            rejection = evidence_rejection(evidence)
            if not rejection:
                harmful.append((item, evidence))
            elif (
                    'below 5 percent' not in rejection
                    and 'benefit is below' not in rejection):
                unqualified.append('%s: %s' % (
                    item.semantic_label, rejection))
        if unqualified:
            self.abort(
                'candidate occluder cannot be safely qualified: '
                + '; '.join(unqualified))
            return
        if not harmful:
            self.state = 'SCAN_READY'
            self.publish_status(
                'rigid objects are visible but do not measurably obstruct '
                'target capture; ready to scan')
            return
        if self.contact_actions >= int(
                self.get_parameter('max_contact_actions').value):
            self.abort('bounded six-action occlusion-removal budget exhausted')
            return
        selected, evidence = max(
            harmful,
            key=lambda pair: (
                float(pair[1].predicted_surface_gain),
                float(pair[1].target_overlap_ratio),
                -distance(point(pair[0].base_centroid), self.landmark)))
        config = {name: self.get_parameter(name).value for name in (
            'movable_whitelist', 'target_clearance_m', 'drop_target_clearance_m',
            'drop_obstacle_clearance_m', 'drop_search_radius_m', 'max_grasp_width_m',
            'approach_height_m', 'pre_push_offset_m', 'push_distance_m',
            'workspace_x_min', 'workspace_x_max', 'workspace_y_min', 'workspace_y_max',
            'workspace_z_min', 'workspace_z_max', 'enforce_static_workspace',
            'require_observed_drop_support', 'drop_support_radius_m',
            'drop_support_min_points', 'drop_support_max_stddev_m')}
        push_attempt = int(self.push_attempts_by_track.get(
            str(evidence.track_id), 0))
        config['push_distance_m'] = float(self.get_parameter(
            'first_push_distance_m' if push_attempt == 0
            else 'later_push_distance_m').value)
        self.plan = choose_removal_plan(
            selected, self.landmark, self.obstacles.instances, config,
            self.scene_points if self.fresh('scene_cloud') else [])
        self.plan.update({
            'track_id': evidence.track_id,
            'observation_count': int(evidence.observation_count),
            'target_overlap_ratio': float(evidence.target_overlap_ratio),
            'closer_depth_ratio': float(evidence.closer_depth_ratio),
            'predicted_surface_gain': float(evidence.predicted_surface_gain),
            'predicted_unlocked_viewpoints': int(
                evidence.predicted_unlocked_viewpoints),
            'contact_action_index': int(self.contact_actions + 1),
            'contact_action_limit': int(
                self.get_parameter('max_contact_actions').value),
        })
        self.obstacles_at_plan = {int(item.object_id): point(item.base_centroid)
                                  for item in self.obstacles.instances if item.valid}
        self.publish_json(self.plan_pub, self.plan)
        self.publish_markers(self.plan)
        if not self.plan.get('valid'):
            self.abort('removal planning failed: %s' % self.plan.get('reason', 'unknown'))
            return
        self.state = 'PLAN_READY'
        self.publish_status('removal plan ready for operator review')

    @staticmethod
    def occluder_evidence(instance):
        size = tuple(
            max(0.0, upper - lower)
            for lower, upper in zip(
                point(instance.base_bounds_min),
                point(instance.base_bounds_max)))
        overlap = float(instance.target_overlap_ratio)
        closer = float(instance.closer_depth_occlusion_ratio)
        gain = min(1.0, max(overlap, closer))
        return OccluderEvidence(
            track_id=str(instance.track_id),
            object_id=int(instance.object_id),
            label=str(instance.semantic_label),
            observation_count=int(instance.observation_count),
            confirmed_in_probe=bool(instance.confirmed_in_probe_view),
            target_overlap_ratio=overlap,
            closer_depth_ratio=closer,
            predicted_surface_gain=gain,
            predicted_unlocked_viewpoints=(2 if gain >= 0.10 else 1),
            confidence=float(instance.confidence),
            valid=bool(instance.valid),
            uncertainty_m=float(instance.position_uncertainty_m),
            size_xyz_m=size,
        )

    def verify_action(self):
        object_id = int(self.plan['object_id'])
        current = next(
            (item for item in self.obstacles.instances
             if int(item.object_id) == object_id),
            None)
        if current and current.valid:
            moved = distance(point(current.base_centroid), self.obstacles_at_plan[object_id])
            if moved < float(self.get_parameter('obstacle_displacement_m').value):
                self.publish_status('planned obstacle has not moved far enough')
                return False
        remaining = [item for item in self.obstacles.instances
                     if not item.valid or int(item.classification) != item.CLASSIFICATION_MOVABLE]
        if remaining:
            self.abort('post-action scene contains unsafe or invalid obstacles')
            return False
        return True

    def publish_model(self):
        model = cloud_model(self.cloud_points, self.cloud_frame, self.accepted_views)
        if model.get('valid'):
            center = tuple(model['center'])
            if not self.centers or distance(center, self.centers[-1]) > 1e-9:
                self.centers.append(center)
            model['center_delta_m'] = (distance(self.centers[-1], self.centers[-2])
                                       if len(self.centers) >= 2 else None)
            model['confidence'] = min(1.0, self.accepted_views / float(
                max(1, int(self.get_parameter('min_views').value))))
            self.target_center = center
            self.publish_target_frame(center)
        self.publish_json(self.model_pub, model)

    def publish_target_frame(self, center):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'base_link'
        transform.child_frame_id = 'target_local'
        transform.transform.translation.x = float(center[0])
        transform.transform.translation.y = float(center[1])
        transform.transform.translation.z = float(center[2])
        transform.transform.rotation.w = 1.0
        self.target_tf.sendTransform(transform)

    def publish_status(self, reason):
        self.last_reason = str(reason)
        lock_rejection = self.measured_lock_rejection()
        self.publish_json(self.status_pub, {
            'state': self.state, 'reason': reason, 'dry_run': True,
            'real_arm_motion': False, 'accepted_views': self.accepted_views,
            'session_id': self.session_id,
            'min_views': int(self.get_parameter('min_views').value),
            'max_views': int(self.get_parameter('max_views').value),
            'measured_lock_ready': not bool(lock_rejection),
            'measured_lock_rejection': lock_rejection,
            'lock_source': 'tracked_target',
            'target_landmark_diagnostic_state': self.target_landmark_state,
        })

    def abort(self, reason):
        if self.state == 'ABORTED':
            return
        self.state = 'ABORTED'
        self.publish_status(reason)
        self.get_logger().error(reason)

    def publish_markers(self, plan):
        if not plan.get('valid'):
            return
        array = MarkerArray()
        points = [('object', plan.get('object_center'), (1.0, 0.2, 0.1)),
                  ('approach', plan.get('approach'), (1.0, 0.8, 0.0)),
                  ('destination', plan.get('drop_center', plan.get('push_end')), (0.1, 1.0, 0.2))]
        for index, (name, xyz, color) in enumerate(points):
            if xyz is None:
                continue
            marker = Marker()
            marker.header.frame_id = 'base_link'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'supervised_workflow'
            marker.id = index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = xyz
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.025
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 0.9
            marker.text = name
            array.markers.append(marker)
        self.marker_pub.publish(array)

    @staticmethod
    def parse(msg):
        try:
            return json.loads(msg.data)
        except (ValueError, TypeError):
            return {}

    @staticmethod
    def publish_json(pub, payload):
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        pub.publish(msg)

    @staticmethod
    def reply(response, success, message):
        response.success = bool(success)
        response.message = str(message)
        return response

    @staticmethod
    def read_xyz(msg):
        fields = {field.name: field for field in msg.fields}
        if not all(name in fields for name in ('x', 'y', 'z')):
            return []
        endian = '>' if msg.is_bigendian else '<'
        points = []
        for row in range(msg.height):
            for col in range(msg.width):
                offset = row * msg.row_step + col * msg.point_step
                xyz = tuple(
                    struct.unpack_from(
                        endian + 'f', msg.data, offset + fields[name].offset)[0]
                    for name in ('x', 'y', 'z'))
                if all(math.isfinite(value) for value in xyz):
                    points.append(xyz)
        return points


def main(args=None):
    rclpy.init(args=args)
    node = SupervisedCubeWorkflowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
