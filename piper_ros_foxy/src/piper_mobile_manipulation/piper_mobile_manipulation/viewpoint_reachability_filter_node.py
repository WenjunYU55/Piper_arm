#!/usr/bin/env python3
import json
import math
import os
from pathlib import Path
import time

import rclpy
from piper_msgs.msg import PiperStatusMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from piper_mobile_manipulation.scan_motion import motor_control_reasons
from piper_mobile_manipulation.capability_map import (
    load_capability_map,
    sha256_file,
)
from piper_mobile_manipulation.ray_hard_culls import (
    hard_cull_snapshot,
    population_key,
    stable_revision,
)
from piper_mobile_manipulation.ray_mission_diagnostics import (
    add_prequalification,
    RayMissionDiagnosticsStore,
)


ROUGH_ACQUISITION = 'ROUGH_ACQUISITION'
RAY_WORKSPACE_REJECTION = (
    'bounded ray does not intersect configured camera workspace')
CAPABILITY_MAP_REJECTION = (
    'CAPABILITY_MAP_NO_SUPPORT: bounded ray has no nearby '
    'collision-qualified camera capability cell')
CAPABILITY_MAP_MODES = ('off', 'shadow', 'enforce')


def capability_bound_ray(viewpoint, capability):
    """Narrow one requested ray to its map-supported standoff envelope.

    The individual contiguous runs remain attached as evidence.  The active
    min/max envelope is intentionally only a coarse search bound; Tesseract
    still owns exact IK and collision validation and the map is never
    presented as a joint solution.
    """
    result = dict(viewpoint)
    if not isinstance(capability, dict) or not capability.get('supported'):
        return result
    raw_intervals = capability.get('supported_intervals_m', [])
    try:
        requested_minimum = float(result['ray_min_standoff_m'])
        requested_maximum = float(result['ray_max_standoff_m'])
        configured_minimum = float(result.get(
            'ray_requested_min_standoff_m', requested_minimum))
        configured_maximum = float(result.get(
            'ray_requested_max_standoff_m', requested_maximum))
        intervals = []
        for raw in raw_intervals:
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                raise ValueError('capability interval is malformed')
            lower = max(requested_minimum, float(raw[0]))
            upper = min(requested_maximum, float(raw[1]))
            if not all(math.isfinite(value) for value in (lower, upper)):
                raise ValueError('capability interval is non-finite')
            if upper >= lower:
                intervals.append([lower, upper])
    except (KeyError, TypeError, ValueError):
        return result
    if not intervals:
        return result
    intervals.sort(key=lambda item: (item[0], item[1]))
    merged = []
    for lower, upper in intervals:
        if merged and lower <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], upper)
        else:
            merged.append([lower, upper])
    minimum = merged[0][0]
    maximum = merged[-1][1]
    try:
        original_scoring = float(result.get(
            'ray_scoring_standoff_m', 0.5 * (minimum + maximum)))
        original_preferred = float(result.get(
            'ray_preferred_max_standoff_m', maximum))
        target_record = result['target_object_center']
        target = [float(target_record[key]) for key in ('x', 'y', 'z')]
        direction_record = result['ray_direction']
        direction = [float(direction_record[key]) for key in ('x', 'y', 'z')]
        norm = math.sqrt(sum(value * value for value in direction))
        if norm <= 1e-12:
            return result
        direction = [value / norm for value in direction]
    except (KeyError, TypeError, ValueError):
        return result
    scoring_candidates = [
        min(upper, max(lower, original_scoring))
        for lower, upper in merged]
    scoring = min(
        scoring_candidates,
        key=lambda value: (abs(value - original_scoring), value))
    preferred = min(maximum, max(minimum, original_preferred))
    camera = [
        target[index] + direction[index] * scoring for index in range(3)]
    result.update({
        # Preserve the original configured interval for review while naming
        # the size-envelope interval that actually entered atlas lookup.
        'ray_requested_min_standoff_m': configured_minimum,
        'ray_requested_max_standoff_m': configured_maximum,
        'ray_capability_requested_min_standoff_m': requested_minimum,
        'ray_capability_requested_max_standoff_m': requested_maximum,
        'ray_capability_intervals_m': merged,
        'ray_capability_min_standoff_m': minimum,
        'ray_capability_max_standoff_m': maximum,
        'ray_capability_bounded': True,
        'ray_min_standoff_m': minimum,
        'ray_max_standoff_m': maximum,
        'ray_preferred_max_standoff_m': preferred,
        'ray_scoring_standoff_m': scoring,
        'desired_camera_position': {
            axis: float(camera[index])
            for index, axis in enumerate(('x', 'y', 'z'))},
        'camera_object_distance_m': scoring,
    })
    return result


def target_status_rejection_reason(target_status, plan_kind):
    """Keep normal tracking gates while allowing LOST acquisition bootstrap."""
    status = str(target_status)
    if status == 'LOW_CONFIDENCE':
        return 'target_status=LOW_CONFIDENCE'
    if status == 'LOST' and plan_kind != ROUGH_ACQUISITION:
        return 'target_status=LOST'
    return ''


def bounded_ray_intersects_workspace(
        viewpoint, min_reach_m, max_reach_m,
        min_camera_distance_m, max_camera_distance_m,
        max_height_change_m):
    """Return whether any permitted point on a target ray is in workspace."""
    try:
        target_record = viewpoint['target_object_center']
        target = [float(target_record[key]) for key in ('x', 'y', 'z')]
        direction_record = viewpoint['ray_direction']
        if isinstance(direction_record, dict):
            direction = [
                float(direction_record[key]) for key in ('x', 'y', 'z')]
        else:
            # Retain compatibility with recorded/private payloads produced
            # before ray directions were serialized as named XYZ fields.
            direction = [float(value) for value in direction_record]
        minimum = max(
            float(viewpoint['ray_min_standoff_m']),
            float(min_camera_distance_m),
        )
        maximum = min(
            float(viewpoint['ray_max_standoff_m']),
            float(max_camera_distance_m),
        )
        minimum_reach = float(min_reach_m)
        maximum_reach = float(max_reach_m)
        maximum_height = float(max_height_change_m)
    except (KeyError, TypeError, ValueError):
        return False
    if len(direction) != 3:
        return False
    values = (
        target + direction + [
            minimum, maximum, minimum_reach,
            maximum_reach, maximum_height,
        ])
    if not all(math.isfinite(value) for value in values):
        return False
    direction_norm = math.sqrt(sum(value * value for value in direction))
    if (
            direction_norm <= 1e-9 or minimum <= 0.0
            or maximum < minimum or minimum_reach < 0.0
            or maximum_reach < minimum_reach or maximum_height < 0.0):
        return False
    direction = [value / direction_norm for value in direction]

    vertical = abs(direction[2])
    if vertical > 1e-12:
        maximum = min(maximum, maximum_height / vertical)
    if maximum < minimum:
        return False

    projection = sum(
        target[index] * direction[index] for index in range(3))
    target_norm_sq = sum(value * value for value in target)
    discriminant = (
        projection * projection - target_norm_sq
        + maximum_reach * maximum_reach)
    if discriminant < -1e-12:
        return False
    root = math.sqrt(max(0.0, discriminant))
    minimum = max(minimum, -projection - root)
    maximum = min(maximum, -projection + root)
    if maximum < minimum:
        return False

    def reach_sq(standoff):
        return sum(
            (target[index] + direction[index] * standoff) ** 2
            for index in range(3))

    minimum_reach_sq = minimum_reach * minimum_reach
    return max(reach_sq(minimum), reach_sq(maximum)) >= (
        minimum_reach_sq - 1e-12)


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
        self.declare_parameter(
            'ray_hard_culls_topic', '/piper/ray_hard_culls')

        self.declare_parameter('min_reach_m', 0.20)
        self.declare_parameter('max_reach_m', 0.75)
        self.declare_parameter('min_camera_object_distance_m', 0.25)
        self.declare_parameter('max_camera_object_distance_m', 0.80)
        self.declare_parameter('max_height_change_m', 0.40)
        # Exact-point policies keep the legacy opt-in behavior. Target rays
        # always use these bounds as a cheap interval-intersection cull before
        # Tesseract; this does not claim IK or collision feasibility.
        self.declare_parameter('enforce_static_reach_bounds', False)
        project_root = os.environ.get('PIPER_ARM_ROOT', '/home/prl/Piper_arm')
        self.declare_parameter('capability_map_mode', 'enforce')
        self.declare_parameter('capability_map_project_root', project_root)
        self.declare_parameter(
            'capability_map_path', str(Path(project_root) / (
                'piper_ros_foxy/src/piper_mobile_manipulation/'
                'config/piper_camera_capability_map.npz')))
        self.declare_parameter('floor_z_m', 0.005)
        self.declare_parameter('arm_status_timeout_sec', 1.0)
        self.declare_parameter('dry_run', False)
        self.declare_parameter('debug', True)
        self.declare_parameter('ray_diagnostics_enabled', True)
        self.declare_parameter(
            'ray_diagnostics_root',
            os.path.join(project_root, 'datasets', 'active_scan',
                         'ray_diagnostics'))

        self.arm_status = None
        self.arm_status_at = None
        self.target_status = 'UNKNOWN'
        self.latest_joint_state = None
        self.capability_map = None
        self.capability_map_error = ''
        self.capability_map_requested_mode = str(
            self.get_parameter('capability_map_mode').value).strip().lower()
        self.capability_map_effective_mode = 'off'
        self.capability_map_artifact_sha256 = ''
        self.hard_cull_population_key = None
        self.hard_cull_entries = {}
        self.ray_diagnostics_store = RayMissionDiagnosticsStore(
            self.get_parameter('ray_diagnostics_root').value)
        self.initialize_capability_map()

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
        hard_cull_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.hard_cull_pub = self.create_publisher(
            String,
            self.get_parameter('ray_hard_culls_topic').value,
            hard_cull_qos,
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

    def initialize_capability_map(self):
        """Load one hash-bound atlas without making it a safety authority."""
        requested = self.capability_map_requested_mode
        if requested not in CAPABILITY_MAP_MODES:
            self.capability_map_error = (
                'unsupported capability_map_mode=%s' % requested)
            self.get_logger().error(self.capability_map_error)
            return
        if requested == 'off':
            self.capability_map_effective_mode = 'off'
            return
        try:
            self.capability_map = load_capability_map(
                self.get_parameter('capability_map_path').value,
                self.get_parameter('capability_map_project_root').value,
                verify_sources=True,
            )
            self.capability_map_artifact_sha256 = sha256_file(
                self.get_parameter('capability_map_path').value)
        except (OSError, TypeError, ValueError) as error:
            self.capability_map_error = str(error)
            self.capability_map_effective_mode = 'fallback_coarse'
            self.get_logger().warn(
                'Capability map unavailable; retaining coarse ray cull: %s'
                % error)
            return
        qualified = bool(self.capability_map.metadata.get(
            'qualified_for_enforcement', False))
        if requested == 'enforce' and not qualified:
            self.capability_map_error = (
                'capability map convergence is not qualified for enforcement')
            self.capability_map_effective_mode = 'fallback_coarse'
            self.get_logger().warn(self.capability_map_error)
            return
        self.capability_map_effective_mode = requested
        self.get_logger().info(
            'Capability map loaded in %s mode: '
            '%d occupied 5D bins from %d samples'
            % (
                requested,
                len(self.capability_map.keys),
                int(self.capability_map.metadata.get(
                    'selected_checkpoint_samples',
                    self.capability_map.metadata.get(
                        'checkpoint_samples', 0))),
            ))

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
        ray_candidate_count = 0
        workspace_rejected_rays = 0
        capability_supported_rays = 0
        capability_rejected_rays = 0
        capability_query_ms = 0.0
        for viewpoint in viewpoints:
            if not isinstance(viewpoint, dict):
                continue
            result = dict(viewpoint)
            if result.get('candidate_geometry') == 'target_ray':
                ray_candidate_count += 1
            reasons, capability = self.evaluate_viewpoint(result, plan_kind)
            if capability is not None:
                if (
                        capability.get('supported')
                        and getattr(
                            self, 'capability_map_effective_mode', 'off')
                        == 'enforce'):
                    result = capability_bound_ray(result, capability)
                result['capability_map_prequalification'] = capability
                capability_query_ms += float(capability['elapsed_ms'])
                if capability['supported']:
                    capability_supported_rays += 1
                else:
                    capability_rejected_rays += 1
            accepted = len(reasons) == 0
            result['prequalified'] = bool(accepted)
            # Compatibility aliases consumed by the unchanged bridge and
            # capture diagnostics.  This stage has not run IK or Tesseract.
            result['reachable'] = bool(accepted)
            result['safe'] = bool(accepted)
            result['reject_reasons'] = reasons
            if not accepted:
                result['cull_disposition'] = (
                    'permanent' if any(reason in (
                        RAY_WORKSPACE_REJECTION,
                        CAPABILITY_MAP_REJECTION) for reason in reasons)
                    else 'retry_eligible')
            filtered.append(result)
            if RAY_WORKSPACE_REJECTION in reasons:
                workspace_rejected_rays += 1
            if accepted:
                reachable_count += 1
                safe_count += 1

        output = dict(payload)
        output['dry_run'] = True
        output['filter'] = {
            'node': 'viewpoint_reachability_filter_node',
            'mode': (
                'ray_workspace_and_capability_cull_then_tesseract'
                if (
                    ray_candidate_count
                    and self.capability_map_effective_mode == 'enforce')
                else 'ray_workspace_cull_then_tesseract'
                if ray_candidate_count else (
                    'legacy_static_reach_check'
                    if self.param_bool('enforce_static_reach_bounds')
                    else 'dynamic_kinematics_deferred_to_tesseract')),
            'input_viewpoints': len(viewpoints),
            'output_viewpoints': len(filtered),
            'prequalified_viewpoints': reachable_count,
            'reachable_viewpoints': reachable_count,
            'safe_viewpoints': safe_count,
            'ray_candidates': ray_candidate_count,
            'workspace_rejected_rays': workspace_rejected_rays,
            'capability_supported_rays': capability_supported_rays,
            'capability_rejected_rays': capability_rejected_rays,
            'capability_query_total_ms': capability_query_ms,
            'capability_map': self.capability_map_summary(),
            'reachable_field_semantics': 'prequalified_compatibility_alias',
            'arm_status': self.arm_status_summary(),
            'target_status': self.target_status,
            'dry_run_config_loaded': self.param_bool('dry_run'),
        }
        output['viewpoints'] = filtered
        if plan_kind == 'MULTIVIEW_SCAN':
            try:
                self.publish_hard_culls(payload, filtered)
            except (KeyError, TypeError, ValueError) as error:
                self.get_logger().warn(
                    'Could not publish hard-ray cull feedback: %s' % error)
        diagnostics = payload.get('ray_diagnostics')
        if isinstance(diagnostics, dict):
            try:
                diagnostics = add_prequalification(
                    diagnostics, filtered, output['filter'])
                if self.param_bool('ray_diagnostics_enabled'):
                    _json_path, html_path = self.ray_diagnostics_store.record(
                        diagnostics)
                    diagnostics['artifact_html_path'] = html_path
                output['ray_diagnostics'] = diagnostics
            except Exception as error:  # Diagnostics must never gate the filter.
                self.get_logger().warn(
                    'Could not update ray mission diagnostics: %s' % error)

        out = String()
        out.data = json.dumps(output, sort_keys=True)
        publisher.publish(out)

        if self.param_bool('debug'):
            self.get_logger().info(
                'prequalified %s viewpoints: %d/%d (Tesseract feasibility pending)'
                % (plan_kind, reachable_count, len(filtered))
            )

    def hard_cull_source_revision(self):
        return stable_revision({
            'min_reach_m': self.get_parameter('min_reach_m').value,
            'max_reach_m': self.get_parameter('max_reach_m').value,
            'min_camera_object_distance_m': self.get_parameter(
                'min_camera_object_distance_m').value,
            'max_camera_object_distance_m': self.get_parameter(
                'max_camera_object_distance_m').value,
            'max_height_change_m': self.get_parameter(
                'max_height_change_m').value,
            'floor_z_m': self.get_parameter('floor_z_m').value,
            'capability_map_mode': self.capability_map_effective_mode,
            'capability_map_sha256': self.capability_map_artifact_sha256,
        })

    def publish_hard_culls(self, payload, viewpoints):
        """Publish cumulative static rejections for one frozen population."""
        population = payload.get('ray_population')
        if not isinstance(population, dict):
            return
        try:
            key = population_key(population)
        except ValueError:
            return
        if key != self.hard_cull_population_key:
            self.hard_cull_population_key = key
            self.hard_cull_entries = {}
        for viewpoint in viewpoints:
            reasons = [str(value) for value in viewpoint.get(
                'reject_reasons', [])]
            hard_reason = next((value for value in reasons if value in (
                RAY_WORKSPACE_REJECTION, CAPABILITY_MAP_REJECTION)), None)
            if hard_reason is None:
                continue
            ray_id = int(viewpoint.get(
                'ray_id', viewpoint.get('index', -1)))
            if ray_id < 0:
                continue
            capability = viewpoint.get('capability_map_prequalification', {})
            self.hard_cull_entries[ray_id] = {
                'ray_id': ray_id,
                'stage': 'prequalification',
                'reason_code': (
                    'CAPABILITY_MAP_NO_SUPPORT'
                    if hard_reason == CAPABILITY_MAP_REJECTION
                    else 'RAY_WORKSPACE_NO_INTERSECTION'),
                'reason': hard_reason,
                'evidence': dict(capability) if isinstance(
                    capability, dict) else {},
            }
        generation = payload.get('scan_session', {}).get(
            'accepted_views', 0)
        message = String()
        message.data = json.dumps(hard_cull_snapshot(
            population, 'prequalification',
            self.hard_cull_source_revision(), generation,
            self.hard_cull_entries.values()), sort_keys=True)
        self.hard_cull_pub.publish(message)

    def reject_reasons(self, viewpoint, plan_kind='MULTIVIEW_SCAN'):
        """Compatibility wrapper for existing pure characterization tests."""
        reasons, _capability = \
            ViewpointReachabilityFilterNode.evaluate_viewpoint(
                self, viewpoint, plan_kind)
        return reasons

    def evaluate_viewpoint(self, viewpoint, plan_kind='MULTIVIEW_SCAN'):
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
            return reasons, None
        if not self.valid_vector(target_center):
            reasons.append('missing target object center')
            return reasons, None

        ray_geometry = viewpoint.get('candidate_geometry') == 'target_ray'
        if ray_geometry:
            coarse_supported = bounded_ray_intersects_workspace(
                    viewpoint,
                    self.get_parameter('min_reach_m').value,
                    self.get_parameter('max_reach_m').value,
                    self.get_parameter(
                        'min_camera_object_distance_m').value,
                    self.get_parameter(
                        'max_camera_object_distance_m').value,
                    self.get_parameter('max_height_change_m').value)
            if not coarse_supported:
                reasons.append(RAY_WORKSPACE_REJECTION)
                return reasons, None
            capability = ViewpointReachabilityFilterNode.capability_query(
                self, viewpoint)
            if (
                    capability is not None
                    and not capability['supported']
                    and getattr(
                        self, 'capability_map_effective_mode', 'off')
                    == 'enforce'):
                reasons.append(CAPABILITY_MAP_REJECTION)
            return reasons, capability

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

        return reasons, None

    def capability_query(self, viewpoint):
        """Return additive atlas evidence for one already-coarse-valid ray."""
        capability_map = getattr(self, 'capability_map', None)
        if capability_map is None:
            return None
        try:
            target_record = viewpoint['target_object_center']
            target = [float(target_record[key]) for key in ('x', 'y', 'z')]
            direction_record = viewpoint['ray_direction']
            direction = (
                [float(direction_record[key]) for key in ('x', 'y', 'z')]
                if isinstance(direction_record, dict)
                else [float(value) for value in direction_record])
            minimum = max(
                float(viewpoint['ray_min_standoff_m']),
                float(self.get_parameter(
                    'min_camera_object_distance_m').value),
            )
            maximum = min(
                float(viewpoint['ray_max_standoff_m']),
                float(self.get_parameter(
                    'max_camera_object_distance_m').value),
            )
            clearance = float(capability_map.metadata[
                'tool_floor_clearance_m'])
            floor_z = float(self.get_parameter('floor_z_m').value)
            result = capability_map.intersects_ray(
                target, direction, minimum, maximum, floor_z, clearance)
        except (KeyError, TypeError, ValueError) as error:
            return {
                'available': True,
                'supported': False,
                'checked_keys': 0,
                'matching_keys': 0,
                'elapsed_ms': 0.0,
                'reason': str(error),
                'effective_mode': getattr(
                    self, 'capability_map_effective_mode', 'off'),
            }
        return {
            'available': True,
            'supported': bool(result.supported),
            'checked_keys': int(result.checked_keys),
            'matching_keys': int(result.matching_keys),
            'sampled_standoffs': len(result.sample_support),
            'supported_standoff_samples': int(sum(result.sample_support)),
            'supported_intervals_m': [
                [float(lower), float(upper)]
                for lower, upper in result.supported_intervals_m],
            'position_voxel_m': float(capability_map.position_voxel_m),
            'elapsed_ms': float(result.elapsed_ms),
            'reason': str(result.reason),
            'effective_mode': getattr(
                self, 'capability_map_effective_mode', 'off'),
        }

    def capability_map_summary(self):
        capability_map = getattr(self, 'capability_map', None)
        summary = {
            'requested_mode': getattr(
                self, 'capability_map_requested_mode', 'off'),
            'effective_mode': getattr(
                self, 'capability_map_effective_mode', 'off'),
            'available': capability_map is not None,
            'error': getattr(self, 'capability_map_error', ''),
        }
        if capability_map is not None:
            summary.update({
                'occupied_pose_direction_bins': int(len(capability_map.keys)),
                'checkpoint_samples': int(capability_map.metadata.get(
                    'selected_checkpoint_samples',
                    capability_map.metadata.get('checkpoint_samples', 0))),
                'qualified_for_enforcement': bool(
                    capability_map.metadata.get(
                        'qualified_for_enforcement', False)),
            })
        return summary

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
                    'capability_map': self.capability_map_summary(),
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
