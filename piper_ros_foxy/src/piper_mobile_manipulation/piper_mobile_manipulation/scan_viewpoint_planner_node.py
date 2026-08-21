#!/usr/bin/env python3
import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String

from piper_mobile_manipulation.msg import Target3D, TrackedTarget
from piper_mobile_manipulation.nbv_coverage import (
    ObjectCoverageModel,
    rank_next_best_views,
    VoxelCoverageConfig,
)
from piper_mobile_manipulation.scan_motion import orbit_camera_view
from piper_mobile_manipulation.scan_session_memory import (
    filter_and_order_viewpoints,
    history_coverage_target_center,
    validate_history_payload,
)
from piper_mobile_manipulation.view_generation import make_view_generation
from piper_mobile_manipulation.viewpoint_rays import bounded_ray_interval


def build_viewpoint_angles(center_deg, desired_deg, step_deg, max_viewpoints):
    """Build one ordered orbit sector without duplicating a full-circle endpoint."""
    center = float(center_deg)
    desired = min(abs(float(desired_deg)), 360.0)
    step = max(abs(float(step_deg)), 1e-3)
    max_count = max(int(max_viewpoints), 1)
    half = desired * 0.5
    angles = []
    current = center - half
    terminal = center + half
    full_circle = desired >= 360.0 - 1e-9
    while current < terminal - 1e-6 if full_circle else current <= terminal + 1e-6:
        angles.append(round(current, 6))
        current += step
    if not angles:
        angles = [round(center, 6)]
    elif not full_circle and angles[-1] < terminal - 1e-6:
        angles.append(round(terminal, 6))
    if len(angles) > max_count:
        angles = evenly_downsample(angles, max_count)
    return angles


def evenly_downsample(values, max_count):
    if max_count == 1:
        return [values[len(values) // 2]]
    last = len(values) - 1
    indexes = [
        int(round(i * last / float(max_count - 1)))
        for i in range(max_count)
    ]
    return [values[i] for i in indexes]


def viewpoint_replan_required(
        last_center, center, last_history_signature, history_signature,
        translation_threshold_m, elapsed_sec, minimum_period_sec):
    """Suppress tracker-rate duplicates while preserving meaningful replans."""
    if last_center is None or last_history_signature != history_signature:
        return True
    displacement = math.sqrt(sum(
        (float(center[axis]) - float(last_center[axis])) ** 2
        for axis in ('x', 'y', 'z')))
    if displacement < max(0.0, float(translation_threshold_m)):
        return False
    return float(elapsed_sec) >= max(0.0, float(minimum_period_sec))


def viewpoint_refresh_required(elapsed_sec, refresh_period_sec):
    """Keep an unchanged safe candidate set fresh for downstream gates."""
    elapsed = float(elapsed_sec)
    period = float(refresh_period_sec)
    if not math.isfinite(elapsed) or not math.isfinite(period) or period <= 0.0:
        return False
    return elapsed >= period


def target_frame_rejection_reason(frame_id, planning_frame_id='base_link'):
    """
    Reject target coordinates that are not already in the planning frame.

    ``/piper/target_3d`` is normally a camera-optical-frame measurement while
    ``/piper/tracked_target`` is expressed in ``base_link``. Treating the raw
    fallback numbers as base coordinates can move an otherwise valid NBV dome
    to a fictitious target after a short tracker publication gap.
    """
    supplied = str(frame_id).strip()
    expected = str(planning_frame_id).strip()
    if not expected:
        return 'scan planning frame is not configured'
    if supplied != expected:
        return 'target frame %s is not scan planning frame %s' % (
            supplied or '<empty>', expected)
    return ''


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
        self.declare_parameter(
            'scan_session_history_topic', '/piper/scan_session_history')
        self.declare_parameter(
            'scan_capture_status_topic', '/piper/scan_capture_status')
        self.declare_parameter('planning_frame_id', 'base_link')

        self.declare_parameter('desired_scan_angle_deg', 250)
        self.declare_parameter('viewpoint_center_angle_deg', 0.0)
        self.declare_parameter('viewpoint_step_deg', 7.5)
        self.declare_parameter('scan_radius_m', 0.45)
        self.declare_parameter('scan_radius_offsets_m', [0.0, -0.06, 0.06])
        self.declare_parameter('min_scan_radius_m', 0.30)
        self.declare_parameter('max_scan_radius_m', 0.80)
        self.declare_parameter('ray_min_standoff_m', 0.28)
        self.declare_parameter('preferred_scan_radius_m', 0.50)
        self.declare_parameter('camera_pitch_deg', -10)
        self.declare_parameter('camera_pitch_offsets_deg', [0.0])
        self.declare_parameter('keep_object_centered', True)
        self.declare_parameter('max_viewpoints', 25)
        self.declare_parameter('dry_run', True)
        self.declare_parameter('debug', True)
        self.declare_parameter('use_predicted_target_for_scan', True)
        self.declare_parameter('tracked_preference_timeout_s', 1.0)
        self.declare_parameter('session_max_views', 13)
        self.declare_parameter('duplicate_position_tolerance_m', 0.012)
        self.declare_parameter('duplicate_look_tolerance_deg', 2.0)
        self.declare_parameter(
            'minimum_useful_direction_separation_deg', 6.0)
        self.declare_parameter('target_replan_translation_m', 0.01)
        self.declare_parameter('target_replan_min_period_sec', 0.5)
        self.declare_parameter('target_plan_refresh_period_sec', 0.5)
        self.declare_parameter('view_selection_policy', 'voxel_nbv_shadow')
        self.declare_parameter('nbv_voxel_size_m', 0.005)
        self.declare_parameter('nbv_minimum_radius_m', 0.030)
        self.declare_parameter('nbv_maximum_radius_m', 0.250)
        self.declare_parameter('nbv_radius_scale', 2.0)
        self.declare_parameter('nbv_padding_voxels', 2)
        self.declare_parameter('nbv_surface_tolerance_m', 0.007)
        self.declare_parameter('nbv_render_width', 64)
        self.declare_parameter('nbv_render_height', 48)
        self.declare_parameter('nbv_maximum_scoring_voxels', 20000)

        self.target_status = 'UNKNOWN'
        self.latest_camera_info = None
        self.last_tracked_time = None
        self.latest_history = None
        self.last_planned_center = None
        self.last_planned_history_signature = None
        self.last_plan_monotonic = 0.0
        self.last_frame_warning_monotonic = 0.0
        self.latest_capture_status = None
        self.nbv_scan_dir = ''
        self.nbv_model_error = 'no accepted capture is available'
        self.nbv_ranking_cache_key = None
        self.nbv_ranking_cache = None
        self.ray_pool_session_id = ''
        self.ray_pool_target_center = None
        self.ray_pool_frame_id = ''
        self.ray_pool = None
        self.coverage_model = ObjectCoverageModel(VoxelCoverageConfig(
            voxel_size_m=float(
                self.get_parameter('nbv_voxel_size_m').value),
            minimum_radius_m=float(
                self.get_parameter('nbv_minimum_radius_m').value),
            maximum_radius_m=float(
                self.get_parameter('nbv_maximum_radius_m').value),
            radius_scale=float(
                self.get_parameter('nbv_radius_scale').value),
            padding_voxels=int(
                self.get_parameter('nbv_padding_voxels').value),
            surface_tolerance_m=float(
                self.get_parameter('nbv_surface_tolerance_m').value),
            render_width=int(
                self.get_parameter('nbv_render_width').value),
            render_height=int(
                self.get_parameter('nbv_render_height').value),
            maximum_scoring_voxels=int(
                self.get_parameter('nbv_maximum_scoring_voxels').value),
        ))
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
        history_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.history_sub = self.create_subscription(
            String,
            self.get_parameter('scan_session_history_topic').value,
            self.history_cb,
            history_qos,
        )
        self.capture_status_sub = self.create_subscription(
            String,
            self.get_parameter('scan_capture_status_topic').value,
            self.capture_status_cb,
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

    def history_cb(self, msg):
        try:
            payload = json.loads(msg.data)
            self.latest_history = validate_history_payload(
                payload, self.get_parameter('session_max_views').value)
            self.refresh_coverage_model()
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            self.latest_history = None
            self.get_logger().warn('Ignoring invalid scan session history: %s' % error)

    def capture_status_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        self.latest_capture_status = payload
        self.refresh_coverage_model()

    def selection_policy(self):
        policy = str(self.get_parameter('view_selection_policy').value).strip()
        if policy not in ('legacy', 'voxel_nbv_shadow', 'voxel_nbv'):
            raise ValueError('unsupported view_selection_policy %s' % policy)
        return policy

    def refresh_coverage_model(self):
        try:
            policy = self.selection_policy()
        except ValueError as error:
            self.nbv_model_error = str(error)
            return False
        if policy == 'legacy':
            self.nbv_model_error = 'legacy selection policy does not build NBV'
            return False
        history = self.latest_history or {}
        accepted = int(history.get('accepted_views', 0))
        session_id = str(history.get('session_id', ''))
        if accepted <= 0:
            self.coverage_model.reset(session_id)
            self.nbv_scan_dir = ''
            self.nbv_model_error = 'waiting for the first accepted capture'
            return False
        status = self.latest_capture_status or {}
        captured = int(status.get(
            'captured_frame_count', status.get('frames_captured', 0)))
        scan_dir = str(status.get('scan_dir', ''))
        center = history_coverage_target_center(history, None)
        if not session_id or center is None or not scan_dir:
            self.nbv_model_error = (
                'NBV session, target center, or scan path is missing')
            return False
        if captured < accepted:
            self.nbv_model_error = (
                'NBV capture generation is catching up (%d/%d)'
                % (captured, accepted))
            return False
        if (
                self.coverage_model.session_id == session_id
                and self.coverage_model.generation == accepted
                and self.nbv_scan_dir == scan_dir):
            self.nbv_model_error = ''
            return True
        try:
            self.coverage_model.rebuild_from_scan(
                scan_dir, accepted, center, session_id)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) \
                as error:
            self.nbv_model_error = 'NBV coverage update failed: %s' % error
            return False
        self.nbv_scan_dir = scan_dir
        self.nbv_model_error = ''
        return True

    @staticmethod
    def current_achieved_camera(history):
        latest = history.get('latest_achieved_camera')
        if isinstance(latest, dict):
            value = latest.get('camera_position')
            if isinstance(value, dict):
                try:
                    position = [float(value[axis]) for axis in ('x', 'y', 'z')]
                except (KeyError, TypeError, ValueError):
                    position = []
                if len(position) == 3 and all(
                        math.isfinite(item) for item in position):
                    return position
        entries = history.get('accepted_entries', [])
        if not entries:
            return None
        entry = entries[-1]
        value = entry.get(
            'actual_camera_position', entry.get('desired_camera_position'))
        if not isinstance(value, dict):
            return None
        try:
            position = [float(value[axis]) for axis in ('x', 'y', 'z')]
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(item) for item in position):
            return None
        return position

    def apply_view_selection(self, viewpoints, history):
        policy = self.selection_policy()
        accepted = int(history.get('accepted_views', 0))
        session_id = str(history.get('session_id', ''))
        if policy == 'legacy' or accepted <= 0:
            effective = (
                'voxel_nbv_seed'
                if policy == 'voxel_nbv' and accepted <= 0 else policy)
            return [dict(
                item,
                view_selection_policy=effective,
                view_selection_requested_policy=policy,
                view_selection_generation=accepted,
                view_selection_session_id=session_id,
            ) for item in viewpoints]
        ready = self.refresh_coverage_model()
        if not ready:
            if policy == 'voxel_nbv':
                return None
            return [dict(item, nbv_shadow_status=self.nbv_model_error)
                    for item in viewpoints]
        snapshot = self.coverage_model.snapshot()
        current_camera = self.current_achieved_camera(history)
        candidate_geometry = tuple(
            (
                int(item.get('index', -1)),
                round(float(item['desired_camera_position']['x']), 8),
                round(float(item['desired_camera_position']['y']), 8),
                round(float(item['desired_camera_position']['z']), 8),
            )
            for item in viewpoints)
        cache_key = (
            snapshot.session_id,
            snapshot.generation,
            tuple(current_camera or ()),
            candidate_geometry,
        )
        if cache_key == self.nbv_ranking_cache_key:
            ranked = [dict(item) for item in self.nbv_ranking_cache]
        else:
            ranked = rank_next_best_views(
                snapshot, viewpoints, current_camera)
            self.nbv_ranking_cache_key = cache_key
            self.nbv_ranking_cache = [dict(item) for item in ranked]
        ranked_by_index = {
            int(item.get('index', -1)): item for item in ranked}
        if policy == 'voxel_nbv_shadow':
            result = []
            for item in viewpoints:
                candidate = dict(item)
                scored = ranked_by_index.get(int(item.get('index', -1)), {})
                for key, value in scored.items():
                    if key.startswith('nbv_'):
                        candidate[key] = value
                candidate['nbv_shadow_status'] = 'ready'
                candidate['view_selection_policy'] = policy
                candidate['view_selection_requested_policy'] = policy
                candidate['view_selection_generation'] = accepted
                candidate['view_selection_session_id'] = session_id
                result.append(candidate)
            return result
        positive = [
            dict(item) for item in ranked
            if bool(item.get('nbv_positive_information_gain'))]
        for item in positive:
            item['legacy_expected_new_coverage_score'] = float(
                item.get('expected_new_coverage_score', 0.0))
            item['expected_new_coverage_score'] = float(
                item['nbv_rank_score'])
            item['view_selection_policy'] = policy
            item['view_selection_requested_policy'] = policy
            item['view_selection_generation'] = accepted
            item['view_selection_session_id'] = session_id
            item.pop('coverage_objective', None)
            item.pop('coverage_progress_score', None)
        return positive

    def target_cb(self, msg, source):
        if not msg.valid:
            return
        frame_rejection = target_frame_rejection_reason(
            msg.header.frame_id,
            self.get_parameter('planning_frame_id').value)
        if frame_rejection:
            now = time.monotonic()
            if now - self.last_frame_warning_monotonic >= 5.0:
                self.get_logger().warn(
                    'Ignoring %s for multiview planning: %s; waiting for a '
                    'transformed tracked target' % (source, frame_rejection))
                self.last_frame_warning_monotonic = now
            return
        if source in ('target_3d', 'object_of_interest_3d') and \
                self.recent_tracked_target_available():
            return

        dry_run = self.param_bool('dry_run')
        if not dry_run:
            self.get_logger().warn(
                'dry_run parameter was false; forcing planner output to dry-run semantics')

        center = {
            'x': float(msg.point.x),
            'y': float(msg.point.y),
            'z': float(msg.point.z),
        }
        frame_id = msg.header.frame_id
        history = self.latest_history or {
            'session_id': '',
            'accepted_views': 0,
            'max_views': int(self.get_parameter('session_max_views').value),
            'entries': [],
        }
        angles = self.viewpoint_angles()
        ray_policy = self.selection_policy() == 'voxel_nbv'
        frozen_viewpoints = None
        if ray_policy:
            center, frozen_viewpoints = self.frozen_ray_pool(
                history, center, frame_id, angles)
        history_signature = json.dumps(
            history, sort_keys=True, separators=(',', ':'))
        now = time.monotonic()
        replan_required = viewpoint_replan_required(
                self.last_planned_center,
                center,
                self.last_planned_history_signature,
                history_signature,
                self.get_parameter('target_replan_translation_m').value,
                now - self.last_plan_monotonic,
                self.get_parameter('target_replan_min_period_sec').value)
        refresh_required = viewpoint_refresh_required(
            now - self.last_plan_monotonic,
            self.get_parameter('target_plan_refresh_period_sec').value)
        if not replan_required and not refresh_required:
            return
        if not replan_required and self.last_planned_center is not None:
            # A freshness-only publication must preserve the exact previously
            # planned geometry; sub-threshold tracker noise is not a replan.
            center = dict(self.last_planned_center)
        radius = self.scan_radius()
        if ray_policy:
            viewpoints = [dict(item) for item in frozen_viewpoints]
            if viewpoints:
                radius = float(viewpoints[0]['ray_scoring_standoff_m'])
        else:
            viewpoints = []
            index = 0
            for radius_offset_m in self.get_parameter(
                    'scan_radius_offsets_m').value:
                flexible_radius = min(
                    float(self.get_parameter('max_scan_radius_m').value),
                    max(
                        float(self.get_parameter('min_scan_radius_m').value),
                        radius + float(radius_offset_m)))
                for pitch_offset_deg in self.get_parameter(
                        'camera_pitch_offsets_deg').value:
                    pitch_deg = (
                        float(self.get_parameter('camera_pitch_deg').value)
                        + float(pitch_offset_deg))
                    for angle_deg in angles:
                        viewpoint = self.make_viewpoint(
                            index, angle_deg, flexible_radius, center, frame_id,
                            pitch_deg)
                        viewpoints.append(viewpoint)
                        index += 1
        generated_candidate_count = len(viewpoints)
        viewpoints = filter_and_order_viewpoints(
            viewpoints,
            history['entries'],
            self.get_parameter('duplicate_position_tolerance_m').value,
            self.get_parameter('duplicate_look_tolerance_deg').value,
            accepted_entries=(
                history.get('entries', []) if ray_policy
                else history.get('accepted_entries', [])),
            target_center=history_coverage_target_center(history, center),
            minimum_direction_separation_deg=self.get_parameter(
                'minimum_useful_direction_separation_deg').value,
            direction_target_center=center,
        )
        nonduplicate_candidate_count = len(viewpoints)
        selected_viewpoints = self.apply_view_selection(viewpoints, history)
        selection_ready = selected_viewpoints is not None
        if not selection_ready:
            now = time.monotonic()
            if now - self.last_frame_warning_monotonic >= 5.0:
                self.get_logger().warn(
                    'Withholding authoritative NBV candidates: %s'
                    % self.nbv_model_error)
                self.last_frame_warning_monotonic = now
            viewpoints = []
        else:
            viewpoints = selected_viewpoints
        positive_information_count = int(sum(
            bool(item.get('nbv_positive_information_gain', True))
            for item in viewpoints))
        selection_failure_code = (
            'NO_POSITIVE_INFORMATION_CANDIDATE'
            if (
                self.selection_policy() == 'voxel_nbv'
                and int(history.get('accepted_views', 0)) > 0
                and selection_ready
                and not viewpoints)
            else '')
        remaining = max(
            0, int(history['max_views']) - int(history['accepted_views']))
        # Keep extra collision/workspace-qualified candidates as fallbacks.
        # Tesseract still selects exactly ``remaining`` views.  This permits a
        # two-dimensional azimuth/elevation candidate dome without weakening
        # the exact 13-capture contract.
        if remaining == 0:
            viewpoints = []

        requested_coverage = float(self.get_parameter('desired_scan_angle_deg').value)
        achieved_dry_run_coverage = self.coverage_from_angles(angles)
        stamp = {
            'sec': int(msg.header.stamp.sec),
            'nanosec': int(msg.header.stamp.nanosec),
        }
        camera_info = self.camera_info_summary()
        nbv_model_ready = bool(
            not self.nbv_model_error
            and self.coverage_model.session_id == history['session_id']
            and self.coverage_model.generation
            == int(history['accepted_views'])
            and int(history['accepted_views']) > 0)
        nbv_snapshot = (
            self.coverage_model.snapshot() if nbv_model_ready else None)
        requested_policy = self.selection_policy()
        effective_policy = (
            'voxel_nbv_seed'
            if requested_policy == 'voxel_nbv'
            and int(history['accepted_views']) == 0
            else requested_policy)
        view_generation = None
        if str(history['session_id']).strip():
            generation_reason = ''
            if effective_policy == 'voxel_nbv_seed':
                generation_reason = (
                    'first accepted observation seeds measured voxel coverage')
            elif not selection_ready:
                generation_reason = self.nbv_model_error
            view_generation = make_view_generation(
                history['session_id'],
                int(history['accepted_views']),
                effective_policy,
                int(history['accepted_views']),
                selection_ready,
                len(viewpoints),
                generation_reason,
            ).to_dict()

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
                'scan_session': {
                    'session_id': history['session_id'],
                    'accepted_views': int(history['accepted_views']),
                    'max_views': int(history['max_views']),
                    'coverage_target_center': history_coverage_target_center(
                        history, center),
                },
                'remaining_viewpoints': int(remaining),
                'viewpoints': viewpoints,
                'view_generation': view_generation,
                'selection_failure_code': selection_failure_code,
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
                'scan_session_id': str(history['session_id']),
                'session_accepted_views': int(history['accepted_views']),
                'session_remaining_views': int(remaining),
                'reachable_viewpoints': 0,
                'safe_viewpoints': 0,
                'view_selection_policy': self.selection_policy(),
                'effective_view_selection_policy': effective_policy,
                'view_generation': view_generation,
                'nbv_model_generation': int(self.coverage_model.generation),
                'nbv_model_ready': nbv_model_ready,
                'nbv_model_reason': self.nbv_model_error,
                'nbv_unknown_voxels': (
                    nbv_snapshot.unknown_voxels
                    if nbv_snapshot is not None else 0),
                'nbv_surface_voxels': (
                    nbv_snapshot.surface_voxels
                    if nbv_snapshot is not None else 0),
                'nbv_top_candidate_index': (
                    int(viewpoints[0].get('index', -1)) if viewpoints else -1),
                'nbv_top_predicted_unknown_pixels': (
                    int(viewpoints[0].get(
                        'nbv_predicted_unknown_pixels', 0))
                    if viewpoints else 0),
                'selection_failure_code': selection_failure_code,
                'generated_candidates': int(generated_candidate_count),
                'duplicate_rejected_candidates': int(
                    generated_candidate_count - nonduplicate_candidate_count),
                'information_scored_candidates': int(
                    nonduplicate_candidate_count),
                'positive_information_candidates': int(
                    positive_information_count),
                'note': (
                    'reachability and safety are intentionally false until a later '
                    'dry-run evaluator is added'),
            },
            sort_keys=True,
        )
        self.pub_coverage.publish(coverage_msg)
        self.last_planned_center = dict(center)
        self.last_planned_history_signature = history_signature
        self.last_plan_monotonic = now

        if self.param_bool('debug'):
            self.get_logger().info(
                'planned generation session=%s accepted=%d policy=%s '
                'ready=%s candidates=%d around target from %s '
                'coverage=%.1fdeg radius=%.2fm'
                % (
                    history['session_id'], int(history['accepted_views']),
                    effective_policy, selection_ready, len(viewpoints), source,
                    achieved_dry_run_coverage, radius)
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

    def make_viewpoint(
            self, index, angle_deg, radius, center, frame_id,
            pitch_deg=None):
        if pitch_deg is None:
            pitch_deg = float(self.get_parameter('camera_pitch_deg').value)
        pitch_deg = float(pitch_deg)
        camera_vector, look_vector = orbit_camera_view(
            [center['x'], center['y'], center['z']], angle_deg, radius, pitch_deg)
        camera_position = {
            'x': float(camera_vector[0]),
            'y': float(camera_vector[1]),
            'z': float(camera_vector[2]),
        }
        look_direction = {
            'x': float(look_vector[0]),
            'y': float(look_vector[1]),
            'z': float(look_vector[2]),
        }

        return {
            'index': int(index),
            'frame_id': frame_id,
            'viewpoint_angle_deg': float(angle_deg),
            'target_object_center': center,
            'desired_camera_position': camera_position,
            'desired_look_at_direction': look_direction,
            'camera_object_distance_m': float(radius),
            'camera_pitch_deg': pitch_deg,
            'keep_object_centered': self.param_bool('keep_object_centered'),
            'reachable': False,
            'safe': False,
        }

    def make_ray_viewpoint(
            self, ray_id, angle_deg, center, frame_id, pitch_deg):
        """Build one stable direction with a bounded standoff interval."""
        minimum, maximum = bounded_ray_interval(
            [center['x'], center['y'], center['z']],
            self.get_parameter('ray_min_standoff_m').value,
            self.get_parameter('max_scan_radius_m').value,
        )
        preferred_maximum = min(
            maximum,
            max(minimum, float(
                self.get_parameter('preferred_scan_radius_m').value)),
        )
        scoring_standoff = 0.5 * (minimum + preferred_maximum)
        viewpoint = self.make_viewpoint(
            ray_id, angle_deg, scoring_standoff, center, frame_id,
            pitch_deg)
        camera = viewpoint['desired_camera_position']
        direction = {
            axis: (float(camera[axis]) - float(center[axis])) / scoring_standoff
            for axis in ('x', 'y', 'z')
        }
        viewpoint.update({
            'candidate_geometry': 'target_ray',
            'ray_id': int(ray_id),
            'ray_direction': direction,
            'ray_min_standoff_m': float(minimum),
            'ray_max_standoff_m': float(maximum),
            'ray_preferred_max_standoff_m': float(preferred_maximum),
            'ray_scoring_standoff_m': float(scoring_standoff),
        })
        return viewpoint

    def frozen_ray_pool(self, history, center, frame_id, angles):
        """Create one mission-scoped ray pool and otherwise return a copy."""
        session_id = str(history.get('session_id', ''))
        frozen_center = history_coverage_target_center(history, center)
        if frozen_center is None:
            frozen_center = dict(center)
        reusable = bool(
            session_id
            and session_id == self.ray_pool_session_id
            and self.ray_pool is not None)
        if reusable:
            return (
                dict(self.ray_pool_target_center),
                [dict(item) for item in self.ray_pool],
            )

        viewpoints = []
        ray_id = 0
        for pitch_offset_deg in self.get_parameter(
                'camera_pitch_offsets_deg').value:
            pitch_deg = (
                float(self.get_parameter('camera_pitch_deg').value)
                + float(pitch_offset_deg))
            for angle_deg in angles:
                viewpoints.append(self.make_ray_viewpoint(
                    ray_id, angle_deg, frozen_center, frame_id, pitch_deg))
                ray_id += 1
        if session_id:
            self.ray_pool_session_id = session_id
            self.ray_pool_target_center = dict(frozen_center)
            self.ray_pool_frame_id = str(frame_id)
            self.ray_pool = [dict(item) for item in viewpoints]
        return dict(frozen_center), viewpoints

    def viewpoint_angles(self):
        return build_viewpoint_angles(
            self.get_parameter('viewpoint_center_angle_deg').value,
            self.get_parameter('desired_scan_angle_deg').value,
            self.get_parameter('viewpoint_step_deg').value,
            self.get_parameter('max_viewpoints').value,
        )

    @staticmethod
    def evenly_downsample(values, max_count):
        return evenly_downsample(values, max_count)

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
