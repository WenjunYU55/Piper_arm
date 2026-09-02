#!/usr/bin/env python3
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import rclpy
import tf2_ros
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, QoSProfile, ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
import yaml

from piper_mobile_manipulation.msg import Target3D, TrackedTarget
from piper_mobile_manipulation.planning.coverage import (
    candidate_meets_minimum_information,
    MINIMUM_USEFUL_MARGINAL_INFORMATION_FRACTION,
    ObjectCoverageModel,
    persist_coverage_snapshot,
    rank_next_best_views,
    VoxelCoverageConfig,
)
from piper_mobile_manipulation.execution.motion import orbit_camera_view
from piper_mobile_manipulation.perception.target_envelope import (
    build_revolution_envelope,
    coverage_sphere_from_envelope,
    envelope_constrained_ray_interval,
    stamp_nanoseconds,
    validate_capture_model_seed,
    validate_shape_measurement,
)
from piper_mobile_manipulation.utils.mask_reprojection import transform_matrix
from piper_mobile_manipulation.ray_mission_diagnostics import (
    add_capture_event,
    add_target_update_event,
    capture_event_identity,
    planner_generation_snapshot,
    RayMissionDiagnosticsStore,
)
from piper_mobile_manipulation.planning.ray_culls import (
    HardCullLedger,
    prune_hard_culled_rays,
    ray_population_identity,
)
from piper_mobile_manipulation.scan_session_memory import (
    filter_and_order_viewpoints,
    history_coverage_target_center,
    validate_history_payload,
)
from piper_mobile_manipulation.planning.generation import (
    make_view_generation,
    view_policy_capabilities,
)
from piper_mobile_manipulation.planning.rays import (
    bounded_ray_interval,
    build_ray_samples,
)


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


def is_authoritative_nbv_policy(policy):
    """Return whether measured coverage controls candidate ordering."""
    return view_policy_capabilities(policy).authoritative_nbv


def effective_selection_policy(policy, accepted_views):
    """Name seed generations without claiming measured NBV evidence."""
    selected = str(policy)
    if int(accepted_views) > 0:
        return selected
    if selected == 'voxel_nbv':
        return 'voxel_nbv_seed'
    if selected == 'ray_nbv':
        return 'ray_nbv_seed'
    return selected


def provisional_first_ray_allowed(history):
    """Allow generation zero only until an aimed model seed is qualified."""
    return bool(
        int(history.get('accepted_views', 0)) == 0
        and history.get('qualified_target_shape') is None)


def pending_first_ray_framing_retry(history):
    """Return the latest zero-capture same-ray outward retry request."""
    if int(history.get('accepted_views', 0)) != 0:
        return None
    rejected = history.get('rejected_entries', [])
    if not rejected:
        return None
    latest = rejected[-1]
    try:
        ray_id = int(latest['framing_retry_ray_id'])
        minimum = float(latest['framing_retry_min_standoff_m'])
    except (KeyError, TypeError, ValueError):
        return None
    if ray_id < 0 or not math.isfinite(minimum) or minimum <= 0.0:
        return None
    return ray_id, minimum


def restrict_to_framing_retry(viewpoints, history, target_center):
    """Retry the same ray farther out while preserving target-facing aim."""
    request = pending_first_ray_framing_retry(history)
    if request is None:
        return [dict(item) for item in viewpoints], False
    ray_id, requested_minimum = request
    target = np.asarray([
        target_center[axis] for axis in ('x', 'y', 'z')], dtype=float)
    for item in viewpoints:
        if int(item.get('ray_id', -1)) != ray_id:
            continue
        candidate = dict(item)
        maximum = float(candidate['ray_max_standoff_m'])
        minimum = max(
            float(candidate['ray_min_standoff_m']), requested_minimum)
        if minimum > maximum + 1e-9:
            return [], True
        direction = np.asarray([
            candidate['ray_direction'][axis]
            for axis in ('x', 'y', 'z')], dtype=float)
        direction /= np.linalg.norm(direction)
        preferred = max(
            minimum,
            min(maximum, float(candidate['ray_preferred_max_standoff_m'])))
        scoring = 0.5 * (minimum + preferred)
        camera = target + direction * scoring
        candidate.update({
            'ray_min_standoff_m': float(minimum),
            'ray_preferred_max_standoff_m': float(preferred),
            'ray_scoring_standoff_m': float(scoring),
            'camera_object_distance_m': float(scoring),
            'desired_camera_position': dict(zip(
                ('x', 'y', 'z'), (float(value) for value in camera))),
            'desired_look_at_direction': dict(zip(
                ('x', 'y', 'z'),
                (float(value) for value in -direction))),
            'target_framing_retry': True,
        })
        return [candidate], True
    return [], True


def target_shape_failure_code(error):
    """Return the coordinator code carried by a rejected target shape."""
    code = str(error).partition(':')[0].strip()
    if code in ('TARGET_TOO_LARGE_OR_CLOSE', 'TARGET_SCAN_IMPOSSIBLE'):
        return code
    return 'TARGET_SHAPE_UNAVAILABLE'


def retired_view_history(history, ray_policy):
    """Return poses allowed to retire regenerated candidate directions."""
    if ray_policy:
        return list(history.get('accepted_entries', []))
    return list(history.get('entries', []))


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
        self.declare_parameter(
            'ray_hard_culls_topic', '/piper/ray_hard_culls')
        self.declare_parameter('planning_frame_id', 'base_link')

        self.declare_parameter('desired_scan_angle_deg', 250)
        self.declare_parameter('viewpoint_center_angle_deg', 0.0)
        self.declare_parameter('viewpoint_step_deg', 7.5)
        self.declare_parameter('scan_radius_m', 0.45)
        self.declare_parameter('scan_radius_offsets_m', [0.0, -0.06, 0.06])
        self.declare_parameter('min_scan_radius_m', 0.30)
        self.declare_parameter('max_scan_radius_m', 3.0)
        self.declare_parameter('ray_min_standoff_m', 0.28)
        # Compatibility-only parameter.  The former 0.50 m preference acted
        # as a hidden endpoint ceiling.  The current-pose projection may now
        # compete across the complete perception-bounded/capability-supported
        # interval, while exact feasibility remains planner-owned.
        self.declare_parameter('preferred_scan_radius_m', 0.50)
        self.declare_parameter('ray_sampling_region', 'full_sphere')
        self.declare_parameter('ray_count', 175)
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
            'minimum_useful_direction_separation_deg', 15.0)
        self.declare_parameter('target_replan_translation_m', 0.01)
        self.declare_parameter('target_replan_min_period_sec', 0.5)
        self.declare_parameter('target_plan_refresh_period_sec', 0.5)
        self.declare_parameter('view_selection_policy', 'ray_nbv')
        self.declare_parameter('nbv_voxel_size_m', 0.005)
        self.declare_parameter('nbv_minimum_radius_m', 0.030)
        self.declare_parameter('nbv_maximum_radius_m', 0.250)
        self.declare_parameter('nbv_radius_scale', 2.0)
        self.declare_parameter('nbv_padding_voxels', 2)
        self.declare_parameter('nbv_surface_tolerance_m', 0.007)
        self.declare_parameter('nbv_render_width', 64)
        self.declare_parameter('nbv_render_height', 48)
        self.declare_parameter('nbv_maximum_scoring_voxels', 20000)
        project_root = os.environ.get('PIPER_ARM_ROOT', '/home/prl/Piper_arm')
        self.declare_parameter('ray_diagnostics_enabled', True)
        self.declare_parameter(
            'ray_diagnostics_root',
            os.path.join(project_root, 'datasets', 'ray_diagnostics'))

        self._selection_policy = str(
            self.get_parameter('view_selection_policy').value).strip()
        if self._selection_policy not in (
                'legacy', 'voxel_nbv_shadow', 'voxel_nbv', 'ray_nbv'):
            raise ValueError(
                'unsupported view_selection_policy %s'
                % self._selection_policy)
        self._selection_capabilities = view_policy_capabilities(
            self._selection_policy)

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
        self.ray_pool_phase = ''
        self.ray_pool_target_envelope = None
        self.ray_pool_envelope_rejected_rays = 0
        self.current_ray_population = None
        self.hard_cull_ledger = HardCullLedger()
        self.pending_hard_cull_feedback = []
        self.configured_ray_samples = (
            tuple(self.ray_samples())
            if self._selection_capabilities.frozen_candidates else None)
        self.nbv_positive_information_count = 0
        self.nbv_low_information_rejected_count = 0
        self.ray_diagnostics_store = RayMissionDiagnosticsStore(
            self.get_parameter('ray_diagnostics_root').value)
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
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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
        hard_cull_qos = QoSProfile(
            depth=2,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.hard_cull_sub = self.create_subscription(
            String,
            self.get_parameter('ray_hard_culls_topic').value,
            self.hard_cull_cb,
            hard_cull_qos,
        )
        self.get_logger().warn(
            'Scan viewpoint planner is dry-run only; it does not publish '
            '/piper/servo_cmd or move the arm.'
        )

    def status_cb(self, msg):
        self.target_status = msg.data

    def camera_info_cb(self, msg):
        self.latest_camera_info = msg

    def target_envelope_for(self, center, history):
        """Build only from the exact persisted capture-one model seed."""
        session_id = str(history.get('session_id', ''))
        if (
                session_id and session_id == self.ray_pool_session_id
                and self.ray_pool_target_envelope is not None):
            return dict(self.ray_pool_target_envelope), ''
        try:
            model_seed = history.get('qualified_target_model_seed')
            if model_seed is not None:
                model_seed = validate_capture_model_seed(model_seed)
                shape = model_seed['shape']
                transform = np.asarray(
                    model_seed['base_from_camera']['matrix_4x4'],
                    dtype=float)
                envelope = build_revolution_envelope(
                    shape,
                    transform,
                    [center[axis] for axis in ('x', 'y', 'z')],
                )
                return envelope, ''
            shape = history.get('qualified_target_shape')
            if shape is None:
                raise ValueError(
                    'accepted capture target model seed is not yet qualified')
            # Compatibility only for pre-model-seed command-free replays.
            # New live sessions always carry the capture-bound matrix above.
            shape = validate_shape_measurement(shape)
            camera_info = self.camera_info_summary()
            if not camera_info.get('available'):
                raise ValueError(
                    'camera intrinsics are unavailable for target envelope')
            source_frame = str(shape['header']['frame_id'])
            planning_frame = str(
                self.get_parameter('planning_frame_id').value)
            transform = self.tf_buffer.lookup_transform(
                planning_frame, source_frame,
                rclpy.time.Time(nanoseconds=stamp_nanoseconds(
                    shape['header']['stamp'])))
            envelope = build_revolution_envelope(
                shape,
                transform_matrix(transform),
                [center[axis] for axis in ('x', 'y', 'z')],
            )
        except (
                KeyError, TypeError, ValueError,
                tf2_ros.TransformException) as error:
            return None, str(error)
        return envelope, ''

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

    def hard_cull_cb(self, msg):
        """Accept only feedback bound to the current frozen ray universe."""
        try:
            payload = json.loads(msg.data)
            if self.current_ray_population is None:
                self.pending_hard_cull_feedback.append(payload)
                self.pending_hard_cull_feedback = (
                    self.pending_hard_cull_feedback[-8:])
                return
            if self.hard_cull_ledger.update(payload):
                self.nbv_ranking_cache_key = None
                self.nbv_ranking_cache = None
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().warn(
                'Ignoring invalid hard-ray cull feedback: %s' % error)

    def selection_policy(self):
        """Return the policy frozen when this mission stack started."""
        return self._selection_policy

    def authoritative_nbv_policy(self):
        """Return whether measured coverage controls candidate ordering."""
        return is_authoritative_nbv_policy(self.selection_policy())

    def ray_policy(self):
        """Return whether candidates are mission-frozen bounded rays."""
        return self._selection_capabilities.ray_expansion

    def effective_policy(self, accepted_views):
        """Name the seed generation without claiming measured NBV evidence."""
        return effective_selection_policy(
            self.selection_policy(), accepted_views)

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
        model_center = None
        model_radius = None
        model_source = 'accepted_depth_heuristic'
        if self.ray_policy():
            envelope, envelope_error = self.target_envelope_for(
                center, history)
            if envelope is None:
                self.nbv_model_error = (
                    'NBV target envelope is unavailable: %s'
                    % envelope_error)
                return False
            try:
                coverage_sphere = coverage_sphere_from_envelope(envelope)
                model_center = coverage_sphere['center_m']
                model_radius = coverage_sphere['radius_m']
                model_source = coverage_sphere['source']
                center = dict(zip(
                    ('x', 'y', 'z'),
                    (float(value) for value in model_center)))
            except (KeyError, TypeError, ValueError) as error:
                self.nbv_model_error = (
                    'NBV target size is unavailable: %s' % error)
                return False
        if (
                self.coverage_model.session_id == session_id
                and self.coverage_model.generation == accepted
                and self.nbv_scan_dir == scan_dir):
            self.nbv_model_error = ''
            return True
        try:
            self.coverage_model.rebuild_from_scan(
                scan_dir, accepted, center, session_id,
                model_center=model_center, model_radius_m=model_radius,
                model_source=model_source)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) \
                as error:
            self.nbv_model_error = 'NBV coverage update failed: %s' % error
            return False
        try:
            self.record_coverage_diagnostics(
                scan_dir, accepted, center, session_id, history)
        except Exception as error:  # Diagnostics never gate measured NBV.
            self.get_logger().warn(
                'Could not persist coverage diagnostics: %s' % error)
        self.nbv_scan_dir = scan_dir
        self.nbv_model_error = ''
        return True

    def record_coverage_diagnostics(
            self, scan_dir, accepted, center, session_id, history):
        """Bind the just-accepted capture to an exact coverage snapshot."""
        mission_id = os.environ.get('PIPER_MISSION_TASK_ID', '')
        artifact_id = mission_id or session_id
        directory = self.ray_diagnostics_store.session_dir(artifact_id)
        coverage_path = directory / 'coverage' / (
            'capture_%03d.npz' % int(accepted))
        dataset = Path(scan_dir).resolve()
        metadata_paths = sorted((dataset / 'frames').glob(
            'view_*_metadata.yaml'))
        if len(metadata_paths) < int(accepted):
            raise ValueError('accepted capture metadata is unavailable')
        metadata_path = metadata_paths[int(accepted) - 1]
        with metadata_path.open('r', encoding='utf-8') as stream:
            metadata = yaml.safe_load(stream) or {}
        artifacts = [metadata_path]
        for key in (
                'target_depth_png_file_path',
                'target_support_mask_file_path', 'depth_file_path'):
            if metadata.get(key):
                artifacts.append(metadata[key])
        coverage_available = False
        coverage_error = ''
        try:
            persist_coverage_snapshot(
                coverage_path, self.coverage_model.snapshot(),
                capture_artifacts=artifacts,
                configuration_artifacts=[dataset / 'manifest.json'],
                dataset_root=dataset)
            coverage_available = True
        except Exception as error:
            coverage_error = str(error)
            self.get_logger().warn(
                'Could not persist exact coverage snapshot: %s' % error)
        joint_state = metadata.get('joint_state', {})
        names = list(joint_state.get('name', []))
        positions = list(joint_state.get('position', []))
        capture_id, ray_id = capture_event_identity(
            history, metadata, accepted)
        generation = max(0, int(accepted) - 1)
        base = {
            'schema_version': 2,
            'mission_id': mission_id,
            'session_id': session_id,
            'generation': generation,
            'frame_id': 'base_link',
            'target_center_m': [
                float(center[axis]) for axis in ('x', 'y', 'z')]
            if isinstance(center, dict) else [float(value) for value in center],
            'rays': [], 'requests': [], 'events': [],
        }
        base = add_capture_event(
            base, capture_id, True, ray_id=ray_id,
            achieved_camera_matrix_4x4=metadata.get(
                'camera_transform', {}).get('matrix_4x4'),
            joint_names=names[:6], joint_positions=positions[:6],
            gripper_joint_names=names[6:],
            gripper_joint_positions=positions[6:],
            coverage_snapshot_path=(
                str(coverage_path) if coverage_available else ''),
            artifact_bindings=[str(value) for value in artifacts])
        base = add_target_update_event(
            base, capture_id,
            str(coverage_path) if coverage_available else '',
            reason=coverage_error or 'target-model artifacts unavailable')
        self.ray_diagnostics_store.record(base)

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
        capabilities = view_policy_capabilities(policy)
        accepted = int(history.get('accepted_views', 0))
        session_id = str(history.get('session_id', ''))
        self.nbv_positive_information_count = 0
        self.nbv_low_information_rejected_count = 0
        if policy == 'legacy' or accepted <= 0:
            effective = effective_selection_policy(policy, accepted)
            return [dict(
                item,
                view_selection_policy=effective,
                view_selection_requested_policy=policy,
                view_selection_generation=accepted,
                view_selection_session_id=session_id,
            ) for item in viewpoints]
        ready = self.refresh_coverage_model()
        if not ready:
            if is_authoritative_nbv_policy(policy):
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
            policy,
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
        positive_information = [
            dict(item) for item in ranked
            if bool(item.get('nbv_positive_information_gain'))]
        self.nbv_positive_information_count = len(positive_information)
        positive = [
            item for item in positive_information
            if (
                not capabilities.minimum_gain_required
                or candidate_meets_minimum_information(item))]
        self.nbv_low_information_rejected_count = (
            len(positive_information) - len(positive))
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
        ray_policy = self.ray_policy()
        frozen_viewpoints = None
        target_envelope = None
        target_envelope_error = ''
        if ray_policy:
            envelope_center = history_coverage_target_center(history, center)
            if envelope_center is None:
                envelope_center = center
            target_envelope, target_envelope_error = self.target_envelope_for(
                envelope_center, history)
            if target_envelope is None:
                if provisional_first_ray_allowed(history):
                    center, frozen_viewpoints = self.frozen_ray_pool(
                        history, center, frame_id,
                        self.configured_ray_samples, envelope=None)
                else:
                    frozen_viewpoints = []
            else:
                center, frozen_viewpoints = self.frozen_ray_pool(
                    history, center, frame_id, self.configured_ray_samples,
                    envelope=target_envelope)
                target_envelope = self.ray_pool_target_envelope
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
        generated_viewpoints = [dict(item) for item in viewpoints]
        framing_retry = False
        if ray_policy:
            viewpoints, framing_retry = restrict_to_framing_retry(
                viewpoints, history, center)
        ray_population = None
        persistent_culls = {}
        envelope_culls = {
            int(item['ray_id']): {
                'ray_id': int(item['ray_id']),
                'stage': 'target_envelope',
                'reason_code': 'NO_SAFE_TARGET_STANDOFF',
                'reason': str(item.get(
                    'target_envelope_rejection_reason',
                    'target envelope leaves no safe ray interval')),
                'evidence': {
                    'target_envelope_sha256': str(item.get(
                        'target_envelope_sha256', '')),
                    'requested_minimum_standoff_m': float(item.get(
                        'ray_requested_min_standoff_m',
                        item.get('ray_min_standoff_m', 0.0))),
                    'requested_maximum_standoff_m': float(item.get(
                        'ray_requested_max_standoff_m',
                        item.get('ray_max_standoff_m', 0.0))),
                },
            }
            for item in generated_viewpoints
            if item.get('target_envelope_supported') is False
        }
        if ray_policy and str(history.get('session_id', '')).strip():
            ray_population = ray_population_identity(
                generated_viewpoints,
                os.environ.get('PIPER_MISSION_TASK_ID', ''),
                history['session_id'], center, frame_id)
            if self.current_ray_population != ray_population:
                self.current_ray_population = dict(ray_population)
                self.hard_cull_ledger.reset(ray_population)
                pending = self.pending_hard_cull_feedback
                self.pending_hard_cull_feedback = []
                for payload in pending:
                    try:
                        self.hard_cull_ledger.update(payload)
                    except (KeyError, TypeError, ValueError):
                        pass
            persistent_culls = dict(envelope_culls)
            persistent_culls.update(
                self.hard_cull_ledger.entries(ray_population))
        planner_rejections = {}
        viewpoints = prune_hard_culled_rays(
            viewpoints, persistent_culls, planner_rejections)
        # A ray mission has two monotonic retirement sources only: accepted
        # camera directions (including their angular neighbourhood) and the
        # permanent hard-cull ledger above.  Do not turn a path-dependent or
        # framing rejection into an accidental permanent direction cull.
        retired_history = retired_view_history(history, ray_policy)
        viewpoints = filter_and_order_viewpoints(
            viewpoints,
            retired_history,
            self.get_parameter('duplicate_position_tolerance_m').value,
            self.get_parameter('duplicate_look_tolerance_deg').value,
            accepted_entries=history.get('accepted_entries', []),
            target_center=(
                center if target_envelope is not None
                else history_coverage_target_center(history, center)),
            minimum_direction_separation_deg=self.get_parameter(
                'minimum_useful_direction_separation_deg').value,
            direction_target_center=center,
            rejection_reasons=planner_rejections,
        )
        nonduplicate_candidate_count = len(viewpoints)
        history_remaining_viewpoints = [dict(item) for item in viewpoints]
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
        positive_information_count = (
            int(self.nbv_positive_information_count)
            if self.authoritative_nbv_policy()
            and int(history.get('accepted_views', 0)) > 0
            else int(sum(
                bool(item.get('nbv_positive_information_gain', True))
                for item in viewpoints)))
        selection_failure_code = (
            'TARGET_FRAMING_NO_AIMED_ENDPOINT'
            if ray_policy and framing_retry and not viewpoints else
            target_shape_failure_code(target_envelope_error)
            if ray_policy and target_envelope is None and not viewpoints else
            'NO_SAFE_TARGET_STANDOFF'
            if (
                ray_policy and target_envelope is not None
                and generated_candidate_count > 0
                and len(envelope_culls) == generated_candidate_count) else
            'NO_POSITIVE_INFORMATION_CANDIDATE'
            if (
                self.authoritative_nbv_policy()
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

        ray_diagnostics = None
        if ray_policy and str(history['session_id']).strip():
            try:
                ranked_viewpoints = history_remaining_viewpoints
                if (
                        int(history.get('accepted_views', 0)) > 0
                        and selection_ready
                        and isinstance(self.nbv_ranking_cache, list)):
                    ranked_viewpoints = self.nbv_ranking_cache
                ray_diagnostics = planner_generation_snapshot(
                    session_id=history['session_id'],
                    generation=int(history['accepted_views']),
                    target_center=center,
                    frame_id=frame_id,
                    policy=effective_selection_policy(
                        self.selection_policy(), history['accepted_views']),
                    generated=generated_viewpoints,
                    history_remaining=history_remaining_viewpoints,
                    ranked=ranked_viewpoints,
                    selected=viewpoints,
                    planner_rejections=planner_rejections,
                    selection_ready=selection_ready,
                    selection_reason=self.nbv_model_error,
                    remaining_views=remaining,
                    mission_id=os.environ.get('PIPER_MISSION_TASK_ID', ''),
                    persistent_culls=persistent_culls,
                    target_envelope=target_envelope,
                )
                if self.param_bool('ray_diagnostics_enabled'):
                    _json_path, html_path = self.ray_diagnostics_store.record(
                        ray_diagnostics)
                    ray_diagnostics['artifact_html_path'] = html_path
            except Exception as error:  # Diagnostics must never gate planning.
                ray_diagnostics = None
                self.get_logger().warn(
                    'Could not write ray mission diagnostics: %s' % error)

        requested_coverage = float(
            self.get_parameter('desired_scan_angle_deg').value)
        achieved_dry_run_coverage = self.coverage_from_angles(angles)
        if ray_policy and str(self.get_parameter(
                'ray_sampling_region').value) in (
                    'upper_hemisphere', 'full_sphere'):
            requested_coverage = 360.0
            achieved_dry_run_coverage = 360.0
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
        effective_policy = self.effective_policy(history['accepted_views'])
        view_generation = None
        if str(history['session_id']).strip():
            generation_reason = ''
            if effective_policy in ('voxel_nbv_seed', 'ray_nbv_seed'):
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
        view_payload = {
                'header': {
                    'stamp': stamp,
                    'frame_id': frame_id,
                },
                'dry_run': True,
                'source_topic': source,
                'target_status': self.target_status,
                'target_object_center': center,
                'target_envelope': target_envelope,
                'camera_info': camera_info,
                'scan_session': {
                    'session_id': history['session_id'],
                    'accepted_views': int(history['accepted_views']),
                    'max_views': int(history['max_views']),
                    'coverage_target_center': (
                        center if target_envelope is not None
                        else history_coverage_target_center(history, center)),
                },
                'remaining_viewpoints': int(remaining),
                'viewpoints': viewpoints,
                'view_generation': view_generation,
                'selection_failure_code': selection_failure_code,
                'target_envelope_available': target_envelope is not None,
                'target_envelope_sha256': (
                    str(target_envelope.get('envelope_sha256', ''))
                    if target_envelope is not None else ''),
                'target_envelope_rejected_rays': int(
                    self.ray_pool_envelope_rejected_rays
                    if ray_policy else 0),
                'target_envelope_error': target_envelope_error,
            }
        if ray_population is not None:
            view_payload['ray_population'] = ray_population
        if ray_diagnostics is not None:
            view_payload['ray_diagnostics'] = ray_diagnostics
        view_msg.data = json.dumps(view_payload, sort_keys=True)
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
                'candidate_geometry': (
                    'target_ray' if ray_policy else 'exact_point'),
                'ray_sampling_region': (
                    str(self.get_parameter('ray_sampling_region').value)
                    if ray_policy else ''),
                'configured_ray_count': (
                    int(self.get_parameter('ray_count').value)
                    if ray_policy else 0),
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
                'target_envelope_available': target_envelope is not None,
                'target_envelope_sha256': (
                    str(target_envelope.get('envelope_sha256', ''))
                    if target_envelope is not None else ''),
                'target_envelope_rejected_rays': int(
                    self.ray_pool_envelope_rejected_rays
                    if ray_policy else 0),
                'target_envelope_error': target_envelope_error,
                'generated_candidates': int(generated_candidate_count),
                'duplicate_rejected_candidates': int(
                    generated_candidate_count - nonduplicate_candidate_count),
                'information_scored_candidates': int(
                    nonduplicate_candidate_count),
                'positive_information_candidates': int(
                    positive_information_count),
                'minimum_information_fraction': float(
                    MINIMUM_USEFUL_MARGINAL_INFORMATION_FRACTION),
                'low_information_rejected_candidates': int(
                    self.nbv_low_information_rejected_count),
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
            self, ray_id, angle_deg, center, frame_id, pitch_deg,
            envelope=None):
        """Build one stable direction with a bounded standoff interval."""
        requested_minimum, requested_maximum = bounded_ray_interval(
            [center['x'], center['y'], center['z']],
            self.get_parameter('ray_min_standoff_m').value,
            self.get_parameter('max_scan_radius_m').value,
        )
        minimum, maximum = requested_minimum, requested_maximum
        # Retain the private compatibility field without imposing a preferred
        # distance ceiling.  The nearest supported point over the full active
        # interval becomes the numerical seed downstream.
        preferred_maximum = maximum
        scoring_standoff = 0.5 * (minimum + preferred_maximum)
        viewpoint = self.make_viewpoint(
            ray_id, angle_deg, scoring_standoff, center, frame_id,
            pitch_deg)
        camera = viewpoint['desired_camera_position']
        direction = {
            axis: (float(camera[axis]) - float(center[axis])) / scoring_standoff
            for axis in ('x', 'y', 'z')
        }
        envelope_supported = True
        envelope_rejection_reason = ''
        envelope_sha256 = ''
        if envelope is not None:
            envelope_sha256 = str(envelope.get('envelope_sha256', ''))
            interval = envelope_constrained_ray_interval(
                [center[axis] for axis in ('x', 'y', 'z')],
                [direction[axis] for axis in ('x', 'y', 'z')],
                requested_minimum,
                requested_maximum,
                envelope,
                self.camera_info_summary(),
                envelope_is_validated=True,
            )
            if interval is None:
                # Retain the ray in the immutable diagnostic population.  It
                # will be emitted as a permanent hard cull, allowing Full
                # Review to show the actual elimination instead of inventing
                # or silently omitting a candidate.
                envelope_supported = False
                envelope_rejection_reason = (
                    'target envelope leaves no camera standoff with complete '
                    'FOV and 0.250m surface clearance')
            else:
                minimum, maximum = interval
                preferred_maximum = maximum
                scoring_standoff = 0.5 * (minimum + preferred_maximum)
                viewpoint = self.make_viewpoint(
                    ray_id, angle_deg, scoring_standoff, center, frame_id,
                    pitch_deg)
                # Keep the direction calculated before envelope adjustment.
                # Only the representative endpoint/standoff changed.
        viewpoint.update({
            'candidate_geometry': 'target_ray',
            'ray_id': int(ray_id),
            'ray_direction': direction,
            'ray_min_standoff_m': float(minimum),
            'ray_max_standoff_m': float(maximum),
            'ray_preferred_max_standoff_m': float(preferred_maximum),
            'ray_scoring_standoff_m': float(scoring_standoff),
            'ray_requested_min_standoff_m': float(requested_minimum),
            'ray_requested_max_standoff_m': float(requested_maximum),
            'ray_envelope_min_standoff_m': float(minimum),
            'ray_envelope_max_standoff_m': float(maximum),
            'target_envelope_supported': bool(envelope_supported),
            'target_envelope_sha256': envelope_sha256,
            'target_envelope_rejection_reason': envelope_rejection_reason,
        })
        return viewpoint

    def frozen_ray_pool(
            self, history, center, frame_id, ray_samples, envelope=None):
        """Create one bootstrap pool, then one qualified permanent pool."""
        session_id = str(history.get('session_id', ''))
        phase = 'qualified' if envelope is not None else 'bootstrap'
        if envelope is not None:
            anchor = envelope.get('planning_anchor_m')
            if not isinstance(anchor, (list, tuple)) or len(anchor) != 3:
                raise ValueError(
                    'qualified target envelope planning anchor is missing')
            frozen_center = dict(zip(
                ('x', 'y', 'z'), (float(value) for value in anchor)))
        else:
            frozen_center = history_coverage_target_center(history, center)
        if frozen_center is None:
            frozen_center = dict(center)
        reusable = bool(
            session_id
            and session_id == self.ray_pool_session_id
            and self.ray_pool is not None
            and getattr(self, 'ray_pool_phase', '') == phase)
        if reusable:
            return (
                dict(self.ray_pool_target_center),
                [dict(item) for item in self.ray_pool],
            )

        viewpoints = []
        ray_id = 0
        for angle_deg, pitch_deg in ray_samples:
            if envelope is None:
                viewpoint = self.make_ray_viewpoint(
                    ray_id, angle_deg, frozen_center, frame_id, pitch_deg)
            else:
                viewpoint = self.make_ray_viewpoint(
                    ray_id, angle_deg, frozen_center, frame_id, pitch_deg,
                    envelope=envelope)
            viewpoints.append(viewpoint)
            ray_id += 1
        rejected = sum(
            item.get('target_envelope_supported') is False
            for item in viewpoints)
        if session_id:
            self.ray_pool_session_id = session_id
            self.ray_pool_target_center = dict(frozen_center)
            self.ray_pool_frame_id = str(frame_id)
            self.ray_pool = [dict(item) for item in viewpoints]
            self.ray_pool_phase = phase
            self.ray_pool_target_envelope = (
                dict(envelope) if envelope is not None else None)
            self.ray_pool_envelope_rejected_rays = int(rejected)
        return dict(frozen_center), viewpoints

    def ray_samples(self):
        """Return the configured deterministic ray directions."""
        pitches = [
            float(self.get_parameter('camera_pitch_deg').value) + float(offset)
            for offset in self.get_parameter('camera_pitch_offsets_deg').value
        ]
        return build_ray_samples(
            self.get_parameter('ray_sampling_region').value,
            self.get_parameter('ray_count').value,
            self.get_parameter('viewpoint_center_angle_deg').value,
            self.get_parameter('desired_scan_angle_deg').value,
            pitches,
        )

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
        intrinsic = list(self.latest_camera_info.k)
        return {
            'available': True,
            'width': int(self.latest_camera_info.width),
            'height': int(self.latest_camera_info.height),
            'frame_id': self.latest_camera_info.header.frame_id,
            'fx': float(intrinsic[0]),
            'fy': float(intrinsic[4]),
            'cx': float(intrinsic[2]),
            'cy': float(intrinsic[5]),
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
