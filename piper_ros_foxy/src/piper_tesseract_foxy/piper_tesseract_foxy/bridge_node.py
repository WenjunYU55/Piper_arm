#!/usr/bin/env python3
"""Command-free Foxy bridge for the isolated Tesseract plan worker."""

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Vector3
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from piper_msgs.msg import PiperMotionLimits

from piper_mobile_manipulation.msg import (
    CameraTimestampHealth,
    ObstacleInstance3DArray,
    TesseractPlan,
    TesseractReadiness,
    TesseractPlanStatus,
    TrackingHealth,
)
from piper_mobile_manipulation.scan_motion import (
    load_accepted_hand_eye,
    load_conservative_joint_limits,
    PiperScanKinematics,
)
from piper_mobile_manipulation.scan_execution_modes import (
    commanded_speed_percent,
)
from piper_mobile_manipulation.motion_limit_stability import MotionLimitStability
from piper_mobile_manipulation.srv import RequestTesseractPlan

from piper_tesseract_foxy.contract import (
    attach_digest,
    ContractError,
    JOINT_NAMES,
    PLAN_KINDS,
    SCHEMA_VERSION,
    sha256_file,
    Spool,
    validate_response,
)


def obstacle_scene_rejection_reason(scene):
    """
    Return a blocker only when collision geometry cannot be trusted.

    A valid object classified as unsafe is exactly the kind of obstacle the
    planning worker must receive and route around.  Invalid/missing geometry
    remains fail-closed.
    """
    if scene is None or not scene.scene_blocked:
        return None
    instances = list(scene.instances)
    invalid = [item for item in instances if not item.valid]
    if not instances or invalid:
        return 'obstacle scene is blocked: %s' % scene.blocking_reason
    return None


def select_diverse_smooth_view_path(
        candidates, selected_count=None, start_camera_position=None):
    """
    Select a camera-space-diverse subset, then order it as a smooth route.

    Maximizing every consecutive baseline made a one-dimensional orbit
    alternate between its two endpoints.  Farthest-point sampling still
    spreads the selected captures across the full candidate dome, while a
    nearest-neighbour traversal avoids that pendulum motion.  Unselected
    candidates remain at the end as deterministic IK fallbacks for the worker.
    """
    remaining = [dict(item) for item in candidates]
    if not remaining:
        return remaining
    count = (
        len(remaining)
        if selected_count is None
        else max(1, min(int(selected_count), len(remaining))))
    start = None
    if start_camera_position is not None:
        candidate = np.asarray(start_camera_position, dtype=float)
        if candidate.shape == (3,) and np.all(np.isfinite(candidate)):
            start = candidate

    selected = []
    if start is None:
        first_index = min(
            range(len(remaining)),
            key=lambda index: int(remaining[index]['id']))
    else:
        first_index = min(
            range(len(remaining)),
            key=lambda index: (
                float(np.linalg.norm(
                    np.asarray(
                        remaining[index]['camera_position_m'], dtype=float)
                    - start)),
                int(remaining[index]['id']),
            ))
    selected.append(remaining.pop(first_index))

    while remaining and len(selected) < count:
        index = max(
            range(len(remaining)),
            key=lambda candidate_index: (
                min(
                    float(np.linalg.norm(
                        np.asarray(
                            remaining[candidate_index]['camera_position_m'],
                            dtype=float)
                        - np.asarray(
                            reference['camera_position_m'], dtype=float)))
                    for reference in selected),
                -int(remaining[candidate_index]['id']),
            ))
        selected.append(remaining.pop(index))

    ordered = []
    previous = start
    while selected:
        if previous is None:
            index = min(
                range(len(selected)),
                key=lambda item_index: int(selected[item_index]['id']))
        else:
            index = min(
                range(len(selected)),
                key=lambda item_index: (
                    float(np.linalg.norm(
                        np.asarray(
                            selected[item_index]['camera_position_m'],
                            dtype=float) - previous)),
                    int(selected[item_index]['id']),
                ))
        chosen = selected.pop(index)
        ordered.append(chosen)
        previous = np.asarray(chosen['camera_position_m'], dtype=float)

    # Preserve all unselected candidates as stable worker fallbacks.
    return ordered + sorted(remaining, key=lambda item: int(item['id']))


def maximize_successive_view_distance(candidates):
    """Compatibility alias for callers outside the bridge."""
    return select_diverse_smooth_view_path(candidates)


class TesseractPlanBridge(Node):
    """Freezes live inputs and publishes validated, motion-free proposals."""

    def __init__(self):
        super().__init__('tesseract_plan_bridge')
        defaults = {
            'reachable_viewpoints_topic': '/piper/reachable_scan_viewpoints',
            'reachable_acquisition_viewpoints_topic':
                '/piper/reachable_acquisition_viewpoints',
            'joint_states_topic': '/joint_states_single',
            'motion_limits_topic': '/piper/motion_limits',
            'tracking_health_topic': '/piper/tracking_health',
            'camera_timestamp_health_topic': '/piper/camera_timestamp_health',
            'obstacle_topic': '/piper/obstacle_instances_3d',
            'plan_topic': '/piper/tesseract_plan',
            'status_topic': '/piper/tesseract_plan_status',
            'readiness_topic': '/piper/tesseract_readiness',
            'spool_root': '/tmp/piper_tesseract_plans',
            'hand_eye_calibration_path': '',
            'joint_bounds_path': '',
            'robot_xacro_path': '',
            'srdf_path': '',
            'collision_manifest_path': '',
            'data_timeout_sec': 1.0,
            'motion_limits_timeout_sec': 3.0,
            'motion_limits_change_confirmation_sec': 7.0,
            'motion_limits_change_minimum_samples': 3,
            'worker_heartbeat_timeout_sec': 1.5,
            'request_ttl_sec': 180.0,
            'response_timeout_sec': 180.0,
            'max_tracking_measurement_age_sec': 0.75,
            'max_execution_viewpoints': 13,
            'joint_limit_margin_rad': 0.03,
            'trajectory_joint_step_rad': 0.025,
            'trajectory_command_rate_hz': 100.0,
            'speed_percent': 5.0,
            'roll_samples_rad': [-2.094395102, -1.047197551, 0.0,
                                 1.047197551, 2.094395102, 3.141592654],
            'deterministic_seed': 42,
            'return_home_positions_rad': [
                0.000366362, 0.0, 0.0, 0.0, 0.43869236, 0.0,
            ],
            'manipulation_model_qualified': False,
            'debug': True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        required = [
            'hand_eye_calibration_path', 'joint_bounds_path', 'robot_xacro_path',
            'srdf_path', 'collision_manifest_path',
        ]
        missing = [name for name in required if not self.parameter_path(name).is_file()]
        if missing:
            raise RuntimeError('missing Tesseract bridge assets: ' + ', '.join(missing))

        self.spool = Spool(self.get_parameter('spool_root').value)
        self.hand_eye = load_accepted_hand_eye(
            str(self.parameter_path('hand_eye_calibration_path')))
        self.kinematics = PiperScanKinematics(self.hand_eye)
        self.joint_limits, self.ignored_bounds = load_conservative_joint_limits(
            str(self.parameter_path('joint_bounds_path')))
        self.boot_id = self.read_boot_id()

        self.latest_scan = None
        self.latest_acquisition_scan = None
        self.latest_joints = None
        self.latest_motion_limits = None
        self.motion_limit_stability = MotionLimitStability(
            self.get_parameter(
                'motion_limits_change_confirmation_sec').value,
            self.get_parameter(
                'motion_limits_change_minimum_samples').value,
        )
        self.latest_tracking = None
        self.latest_camera_health = None
        self.latest_obstacles = None
        self.updated = {}
        self.pending = {}
        self.state = 'IDLE'
        self.reason = 'waiting for an explicit plan request'
        self.worker_generation_id = ''

        self.plan_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.plan_pub = self.create_publisher(
            TesseractPlan, self.get_parameter('plan_topic').value, self.plan_qos)
        self.status_pub = self.create_publisher(
            TesseractPlanStatus, self.get_parameter('status_topic').value, 10)
        self.readiness_pub = self.create_publisher(
            TesseractReadiness,
            self.get_parameter('readiness_topic').value,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
        self.scan_sub = self.create_subscription(
            String, self.get_parameter('reachable_viewpoints_topic').value,
            self.scan_cb, 10)
        self.acquisition_scan_sub = self.create_subscription(
            String,
            self.get_parameter('reachable_acquisition_viewpoints_topic').value,
            self.acquisition_scan_cb,
            10,
        )
        self.joint_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.joint_sub = self.create_subscription(
            JointState,
            self.get_parameter('joint_states_topic').value,
            self.joint_cb,
            self.joint_qos,
        )
        self.motion_limits_sub = self.create_subscription(
            PiperMotionLimits,
            self.get_parameter('motion_limits_topic').value,
            self.motion_limits_cb,
            10,
        )
        self.tracking_sub = self.create_subscription(
            TrackingHealth, self.get_parameter('tracking_health_topic').value,
            self.tracking_cb, 10)
        self.camera_health_sub = self.create_subscription(
            CameraTimestampHealth,
            self.get_parameter('camera_timestamp_health_topic').value,
            self.camera_health_cb, 10)
        self.obstacle_sub = self.create_subscription(
            ObstacleInstance3DArray,
            self.get_parameter('obstacle_topic').value,
            self.obstacle_cb,
            10,
        )
        self.request_service = self.create_service(
            RequestTesseractPlan, '~/request_plan', self.request_plan_cb)
        self.request_acquisition_service = self.create_service(
            RequestTesseractPlan, '~/request_acquisition_plan',
            self.request_acquisition_plan_cb)
        self.poll_timer = self.create_timer(0.20, self.poll)
        self.publish_status()
        self.publish_readiness()
        self.get_logger().warn(
            'Tesseract bridge is command-free: it has no /joint_ctrl_single publisher '
            'and no motor-enable client. Ignored saved bounds: %s'
            % (','.join(self.ignored_bounds) or 'none'))

    def parameter_path(self, name):
        return Path(str(self.get_parameter(name).value)).resolve()

    @staticmethod
    def read_boot_id():
        try:
            return Path('/proc/sys/kernel/random/boot_id').read_text(encoding='utf-8').strip()
        except OSError:
            return 'unavailable'

    def now(self):
        return time.monotonic()

    def mark(self, key):
        self.updated[key] = self.now()

    def fresh(self, key, timeout=None):
        maximum = (
            float(self.get_parameter('data_timeout_sec').value)
            if timeout is None else float(timeout))
        return self.now() - self.updated.get(key, -1e9) <= maximum

    def scan_cb(self, msg):
        self.store_scan(msg, 'scan', acquisition=False)

    def acquisition_scan_cb(self, msg):
        self.store_scan(msg, 'acquisition_scan', acquisition=True)

    def store_scan(self, msg, key, acquisition):
        try:
            value = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            return
        if isinstance(value, dict):
            if acquisition:
                self.latest_acquisition_scan = value
            else:
                self.latest_scan = value
            self.mark(key)

    def joint_cb(self, msg):
        self.latest_joints = msg
        self.mark('joints')

    def tracking_cb(self, msg):
        self.latest_tracking = msg
        self.mark('tracking')

    def camera_health_cb(self, msg):
        self.latest_camera_health = msg
        self.mark('camera_clock')

    def obstacle_cb(self, msg):
        self.latest_obstacles = msg
        self.mark('obstacles')

    def worker_health_reasons(self):
        try:
            health = self.spool.read_health()
        except (ContractError, FileNotFoundError, OSError, TypeError, ValueError):
            self.worker_generation_id = ''
            return ['Tesseract worker heartbeat is missing or invalid']
        generation = health.get('generation_id')
        written_at_ns = health.get('written_at_ns')
        if (
                not isinstance(generation, str)
                or len(generation) != 32
                or any(character not in '0123456789abcdef'
                       for character in generation)):
            self.worker_generation_id = ''
            return ['Tesseract worker generation ID is invalid']
        try:
            age_sec = (time.time_ns() - int(written_at_ns)) / 1e9
        except (TypeError, ValueError):
            self.worker_generation_id = ''
            return ['Tesseract worker heartbeat timestamp is invalid']
        timeout = float(
            self.get_parameter('worker_heartbeat_timeout_sec').value)
        if age_sec < -1.0 or age_sec > timeout:
            self.worker_generation_id = generation
            return ['Tesseract worker heartbeat is stale']
        if (
                health.get('schema_version') != SCHEMA_VERSION
                or health.get('backend') != 'tesseract'
                or health.get('worker_ready') is not True):
            self.worker_generation_id = generation
            detail = str(health.get('backend_error', '')).strip()
            return [
                'Tesseract worker is not ready'
                + (': ' + detail if detail else '')
            ]
        self.worker_generation_id = generation
        return []

    def snapshot_reasons(
            self, plan_kind='MULTIVIEW_SCAN', require_viewpoints=True,
            worker_reasons=None):
        if plan_kind not in PLAN_KINDS:
            return ['unsupported plan kind']
        reasons = (
            self.worker_health_reasons()
            if worker_reasons is None else list(worker_reasons))
        required = ['joints', 'camera_clock']
        if plan_kind == 'MULTIVIEW_SCAN':
            required.extend(['tracking', 'obstacles'])
            if require_viewpoints:
                required.append('scan')
        elif require_viewpoints:
            required.append('acquisition_scan')
        for key in required:
            if not self.fresh(key):
                reasons.append('%s data is missing or stale' % key)
        if not self.fresh(
                'motion_limits',
                float(self.get_parameter('motion_limits_timeout_sec').value)):
            reasons.append('controller motion limits are missing or stale')
        limits = self.latest_motion_limits
        if limits is None or not limits.valid:
            reasons.append(
                'controller motion limits are invalid: %s'
                % (limits.reason if limits is not None else 'no message'))
        elif (
                list(limits.joint_names) != list(JOINT_NAMES)
                or len(limits.max_velocity_rad_s) != 6
                or len(limits.max_acceleration_rad_s2) != 6
                or len(str(limits.limits_sha256)) != 64):
            reasons.append('controller motion-limit payload is malformed')
        if self.latest_joints is None or len(self.latest_joints.position) < 6:
            reasons.append('joint feedback has fewer than six positions')
        else:
            values = np.asarray(self.latest_joints.position[:6], dtype=float)
            if not np.all(np.isfinite(values)):
                reasons.append('joint feedback is non-finite')
        if plan_kind == 'MULTIVIEW_SCAN':
            health = self.latest_tracking
            if health is None or health.lifecycle_state != 'TRACKING' \
                    or not health.camera_settled:
                reasons.append('tracking is not settled TRACKING')
            elif health.prediction_only:
                reasons.append('tracking is prediction-only')
            elif float(health.measurement_age_sec) > float(
                    self.get_parameter('max_tracking_measurement_age_sec').value):
                reasons.append('tracking measurement is stale')
        if self.latest_camera_health is None or not self.latest_camera_health.healthy:
            reasons.append('camera timestamp health is not healthy')
        if plan_kind == 'MULTIVIEW_SCAN':
            obstacle_reason = obstacle_scene_rejection_reason(self.latest_obstacles)
            if obstacle_reason:
                reasons.append(obstacle_reason)
        scan = (
            self.latest_scan if plan_kind == 'MULTIVIEW_SCAN'
            else self.latest_acquisition_scan)
        if require_viewpoints and (
                scan is None or scan.get('dry_run') is not True):
            reasons.append('reachable viewpoint source is not explicit dry-run data')
        if plan_kind == 'MULTIVIEW_SCAN' and require_viewpoints:
            session = scan.get('scan_session', {}) if isinstance(scan, dict) else {}
            try:
                session_id = str(session.get('session_id', ''))
                accepted = int(session.get('accepted_views', -1))
                maximum = int(session.get('max_views', -1))
                remaining = int(
                    scan.get('remaining_viewpoints', -1)
                    if isinstance(scan, dict) else -1)
            except (TypeError, ValueError):
                session_id, accepted, maximum, remaining = '', -1, -1, -1
            configured = int(
                self.get_parameter('max_execution_viewpoints').value)
            if not session_id:
                reasons.append('scan session identity is missing')
            if maximum != configured:
                reasons.append('scan session maximum does not match configuration')
            if accepted < 0 or accepted >= maximum:
                reasons.append('scan session has no valid remaining viewpoints')
            if remaining != maximum - accepted:
                reasons.append('scan session remaining count is inconsistent')
        return list(dict.fromkeys(reasons))

    def request_plan_cb(self, request, response):
        return self.request_kind_cb(
            'MULTIVIEW_SCAN', request, response)

    def request_acquisition_plan_cb(self, request, response):
        return self.request_kind_cb(
            'ROUGH_ACQUISITION', request, response)

    def request_kind_cb(self, plan_kind, request, response):
        if self.pending and not request.force_refresh:
            response.accepted = False
            response.request_id = next(iter(self.pending))
            response.message = 'a Tesseract request is already pending'
            return response
        reasons = self.snapshot_reasons(plan_kind)
        if reasons:
            response.accepted = False
            response.request_id = ''
            response.message = 'planning blocked: ' + '; '.join(reasons)
            self.set_status('SNAPSHOT_BLOCKED', response.message)
            return response
        try:
            payload = self.build_request(plan_kind)
            self.spool.write('requests', payload['request_id'], payload)
        except (ContractError, KeyError, OSError, TypeError, ValueError) as error:
            response.accepted = False
            response.request_id = ''
            response.message = 'request creation failed: %s' % error
            self.set_status('REJECTED', response.message)
            return response
        self.pending[payload['request_id']] = {
            'request': payload,
            'started': self.now(),
        }
        response.accepted = True
        response.request_id = payload['request_id']
        response.message = 'command-free Tesseract planning request queued'
        self.set_status('PLANNING', response.message, payload['request_id'])
        return response

    def build_request(self, plan_kind='MULTIVIEW_SCAN'):
        if plan_kind not in PLAN_KINDS:
            raise ContractError('unsupported plan kind')
        now_ns = time.time_ns()
        ttl_ns = int(float(self.get_parameter('request_ttl_sec').value) * 1e9)
        joints = [float(value) for value in self.latest_joints.position[:6]]
        scan = (
            self.latest_scan if plan_kind == 'MULTIVIEW_SCAN'
            else self.latest_acquisition_scan)
        center = self.vector(scan.get('target_object_center'), 'target center')
        provenance = self.target_provenance(scan, plan_kind)
        candidates = []
        for item in scan.get('viewpoints', []):
            if not isinstance(item, dict) or item.get('reachable') is not True \
                    or item.get('safe') is not True:
                continue
            candidates.append({
                'id': int(item.get('index', len(candidates))),
                'camera_position_m': self.vector(
                    item.get('desired_camera_position'), 'camera position'),
                'look_direction': self.vector(
                    item.get('desired_look_at_direction'), 'look direction'),
                'required_first': bool(
                    plan_kind == 'ROUGH_ACQUISITION'
                    and item.get('acquisition_look')
                    in ('center', 'compact_center')),
            })
        configured_maximum = max(
            1, int(self.get_parameter('max_execution_viewpoints').value))
        session = scan.get('scan_session', {}) if isinstance(scan, dict) else {}
        if plan_kind == 'MULTIVIEW_SCAN':
            session_id = str(session.get('session_id', ''))
            accepted_views = int(session.get('accepted_views', -1))
            session_maximum = int(session.get('max_views', -1))
            if not session_id:
                raise ContractError('scan session identity is missing')
            if session_maximum != configured_maximum:
                raise ContractError(
                    'scan session maximum does not match configuration')
            remaining_views = session_maximum - accepted_views
            if remaining_views < 1:
                raise ContractError('scan session has no remaining viewpoints')
            if int(scan.get('remaining_viewpoints', -1)) != remaining_views:
                raise ContractError('scan session remaining count is inconsistent')
        maximum = (
            remaining_views if plan_kind == 'MULTIVIEW_SCAN'
            else min(5, configured_maximum, len(candidates)))
        minimum = maximum if plan_kind == 'MULTIVIEW_SCAN' else 1
        tracking_scale = (
            float(self.latest_tracking.recommended_speed_scale)
            if self.latest_tracking is not None else 1.0)
        execution_speed = commanded_speed_percent(
            float(self.get_parameter('speed_percent').value),
            plan_kind,
            tracking_scale,
        )
        candidates = candidates[:max(maximum * 4, maximum)]
        if plan_kind == 'MULTIVIEW_SCAN':
            current_camera = self.kinematics.camera_transform(joints)[:3, 3]
            candidates = select_diverse_smooth_view_path(
                candidates, maximum, current_camera)
        required_candidates = maximum if plan_kind == 'MULTIVIEW_SCAN' else minimum
        if len(candidates) < required_candidates:
            raise ContractError('only %d safe candidates; need at least %d' % (
                len(candidates), required_candidates))
        observation_mode = (
            'perception_snapshot'
            if plan_kind == 'MULTIVIEW_SCAN'
            else 'bootstrap_static')
        obstacles = []
        if observation_mode == 'perception_snapshot':
            for item in self.latest_obstacles.instances:
                if not item.valid:
                    raise ContractError('invalid obstacle geometry is present')
                obstacles.append({
                    'id': '%s:%d' % (item.semantic_label, int(item.object_id)),
                    'type': 'box',
                    'minimum_m': [
                        float(item.base_bounds_min.x), float(item.base_bounds_min.y),
                        float(item.base_bounds_min.z),
                    ],
                    'maximum_m': [
                        float(item.base_bounds_max.x), float(item.base_bounds_max.y),
                        float(item.base_bounds_max.z),
                    ],
                })
        identity = {
            'created_at_ns': now_ns,
            'joint_positions': [round(value, 9) for value in joints],
            'scan_stamp': scan.get('header', {}).get('stamp', {}),
            'plan_kind': plan_kind,
            'target_provenance': provenance,
            'boot_id': self.boot_id,
        }
        request_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest()[:32]
        payload = {
            'schema_version': SCHEMA_VERSION,
            'plan_kind': plan_kind,
            'target_provenance': provenance,
            'request_id': request_id,
            'boot_id': self.boot_id,
            'created_at_ns': now_ns,
            'expires_at_ns': now_ns + ttl_ns,
            'frames': {
                'world_frame': 'base_link',
                'camera_optical_frame': 'camera_color_optical_frame',
                'tcp_frame': 'camera_optical_frame',
            },
            'start_state': {
                'joint_names': list(JOINT_NAMES),
                'positions_rad': joints,
                'feedback_stamp': {
                    'sec': int(self.latest_joints.header.stamp.sec),
                    'nanosec': int(self.latest_joints.header.stamp.nanosec),
                },
            },
            'scene': {
                'target_center_m': center,
                'target_provenance': provenance,
                'observation_mode': observation_mode,
                'candidate_views': candidates,
                'obstacles': obstacles,
            },
            'scan_session': (
                {
                    'session_id': session_id,
                    'accepted_views': accepted_views,
                    'max_views': session_maximum,
                    'remaining_views': remaining_views,
                }
                if plan_kind == 'MULTIVIEW_SCAN' else {}
            ),
            'model': {
                'mode': 0,
                'xacro_sha256': sha256_file(self.parameter_path('robot_xacro_path')),
                'srdf_sha256': sha256_file(self.parameter_path('srdf_path')),
                'collision_manifest_sha256': sha256_file(
                    self.parameter_path('collision_manifest_path')),
            },
            'calibration': {
                'hand_eye_sha256': sha256_file(
                    self.parameter_path('hand_eye_calibration_path')),
                'T_link6_camera': self.hand_eye.tolist(),
                'convention': 'T_link6_camera_optical',
            },
            'limits': {
                'position_rad': self.joint_limits.tolist(),
                'bootstrap_start_limit_tolerance_rad': (
                    0.04 if plan_kind == 'ROUGH_ACQUISITION' else 0.0),
                'joint_margin_rad': float(
                    self.get_parameter('joint_limit_margin_rad').value),
                'max_velocity_rad_s': [
                    float(value) for value
                    in self.latest_motion_limits.max_velocity_rad_s],
                'max_acceleration_rad_s2': [
                    float(value) for value
                    in self.latest_motion_limits.max_acceleration_rad_s2],
                'motion_limits_sha256':
                    str(self.latest_motion_limits.limits_sha256),
                'source': str(self.latest_motion_limits.source),
            },
            'planning': {
                'planner': 'RRTConnect',
                'pipeline': 'OMPL_ISP',
                'deterministic_seed': int(
                    self.get_parameter('deterministic_seed').value),
                'roll_samples_rad': [float(value) for value in self.get_parameter(
                    'roll_samples_rad').value],
                'min_viewpoints': minimum,
                'max_viewpoints': maximum,
                'max_execution_joint_step_rad': float(
                    self.get_parameter('trajectory_joint_step_rad').value),
                'effective_speed_percent': execution_speed,
                'command_rate_hz': float(
                    self.get_parameter('trajectory_command_rate_hz').value),
                'timing_policy': 'sdk_movej_targets_v1',
                'joint_specific_costs': {},
                'return_home_positions_rad': (
                    [float(value) for value in self.get_parameter(
                        'return_home_positions_rad').value]
                    if plan_kind == 'MULTIVIEW_SCAN' else []
                ),
            },
        }
        return attach_digest(payload, 'request_sha256')

    @staticmethod
    def target_provenance(scan, plan_kind):
        header = scan.get('header', {}) if isinstance(scan, dict) else {}
        stamp = header.get('stamp', {})
        if plan_kind == 'MULTIVIEW_SCAN':
            return {
                'source': 'tracked_target',
                'frame_id': str(header.get('frame_id', '')),
                'stamp': {
                    'sec': int(stamp.get('sec', -1)),
                    'nanosec': int(stamp.get('nanosec', -1)),
                },
            }
        supplied = scan.get('target_provenance')
        if not isinstance(supplied, dict):
            raise ContractError(
                'acquisition viewpoints require target_provenance')
        provenance = dict(supplied)
        source_request_id = scan.get('source_request_id')
        if (
                not isinstance(source_request_id, str)
                or provenance.get('source_request_id') != source_request_id):
            raise ContractError(
                'acquisition source_request_id is missing or inconsistent')
        supplied_stamp = provenance.get('stamp', stamp)
        provenance['stamp'] = {
            'sec': int(supplied_stamp.get('sec', -1)),
            'nanosec': int(supplied_stamp.get('nanosec', -1)),
        }
        provenance.setdefault('frame_id', header.get('frame_id', ''))
        return provenance

    @staticmethod
    def vector(value, label):
        if not isinstance(value, dict):
            raise ContractError('%s is missing' % label)
        result = [float(value[key]) for key in ('x', 'y', 'z')]
        if not all(math.isfinite(item) for item in result):
            raise ContractError('%s is non-finite' % label)
        return result

    def poll(self):
        timeout = float(self.get_parameter('response_timeout_sec').value)
        for request_id, state in list(self.pending.items()):
            response_path = self.spool.path('responses', request_id)
            if response_path.is_file():
                try:
                    payload = self.spool.read('responses', request_id)
                    validate_response(payload, state['request'])
                    self.publish_plan(payload)
                except (ContractError, KeyError, OSError, TypeError, ValueError) as error:
                    self.publish_rejection(request_id, 'RESPONSE_INVALID', str(error))
                finally:
                    # The validated ROS message is the hand-off artifact. Consume
                    # the spool entry so a long-running bridge stays bounded.
                    try:
                        response_path.unlink()
                    except FileNotFoundError:
                        pass
                self.pending.pop(request_id, None)
            elif self.now() - state['started'] > timeout:
                self.pending.pop(request_id, None)
                self.publish_rejection(
                    request_id, 'PLANNER_TIMEOUT', 'Tesseract response timed out')
        self.publish_status()
        self.publish_readiness()

    def motion_limits_cb(self, msg):
        # Preserve the last fully validated controller-limit set through one
        # transient partial CAN-query result.  Its freshness timestamp is not
        # renewed by invalid samples, so a persistent fault still blocks
        # planning after the normal timeout.
        accepted, refreshed = self.motion_limit_stability.observe(
            msg, self.now())
        if accepted is not None:
            self.latest_motion_limits = accepted
        if refreshed:
            self.mark('motion_limits')

    def publish_plan(self, payload):
        if payload.get('status') != 'success':
            codes = payload.get('rejection_codes') or ['PLANNING_FAILED']
            self.publish_rejection(
                payload.get('request_id', ''), str(codes[0]),
                str(payload.get('diagnostic', 'planning failed')))
            return
        msg = TesseractPlan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.plan_id = str(payload['plan_id'])
        msg.plan_kind = str(payload['plan_kind'])
        msg.source_request_id = str(
            payload.get('target_provenance', {}).get(
                'source_request_id', ''))
        msg.request_sha256 = str(payload['request_sha256'])
        msg.trajectory_sha256 = str(payload['trajectory_sha256'])
        msg.motion_limits_sha256 = str(
            payload['trajectory_binding']['limits']['motion_limits_sha256'])
        execution = payload['trajectory_binding']['execution']
        msg.execution_speed_percent = float(
            execution['effective_speed_percent'])
        msg.command_rate_hz = float(execution['command_rate_hz'])
        msg.timing_policy = str(execution['timing_policy'])
        msg.backend = str(payload['backend'])
        msg.backend_version = str(payload['backend_version'])
        msg.valid = True
        msg.dry_run = True
        msg.real_arm_motion = False
        msg.collision_model_qualified = bool(payload['collision_model_qualified'])
        msg.reason = str(payload.get('diagnostic', 'validated Tesseract proposal'))
        msg.rejection_codes = [str(value) for value in payload.get('rejection_codes', [])]
        center = payload['target_center_m']
        msg.target_center = Point(x=float(center[0]), y=float(center[1]), z=float(center[2]))
        for selected in payload['selected_viewpoints']:
            msg.viewpoint_indices.append(int(selected['id']))
            position = selected['camera_position_m']
            direction = selected['look_direction']
            msg.camera_positions.append(Point(
                x=float(position[0]), y=float(position[1]), z=float(position[2])))
            msg.look_directions.append(Vector3(
                x=float(direction[0]), y=float(direction[1]), z=float(direction[2])))
        for segment in payload['segments']:
            trajectory = JointTrajectory()
            trajectory.joint_names = list(JOINT_NAMES)
            for value in segment['points']:
                point = JointTrajectoryPoint()
                point.positions = [float(item) for item in value['positions_rad']]
                point.velocities = [float(item) for item in value['velocities_rad_s']]
                point.accelerations = [float(item) for item in value['accelerations_rad_s2']]
                total_nanoseconds = int(round(
                    float(value['time_from_start_s']) * 1e9))
                point.time_from_start = Duration(
                    sec=total_nanoseconds // 1_000_000_000,
                    nanosec=total_nanoseconds % 1_000_000_000,
                )
                trajectory.points.append(point)
            msg.trajectories.append(trajectory)
            msg.minimum_clearance_m.append(float(segment['minimum_clearance_m']))
            msg.limiting_link_pairs.append(str(segment['limiting_link_pair']))
            recovery_used = bool(segment.get('bootstrap_recovery_used', False))
            msg.bootstrap_recovery_end_points.append(
                int(segment['bootstrap_recovery_end_point'])
                if recovery_used else -1)
            msg.bootstrap_recovery_joints.append(
                int(segment['bootstrap_recovery_joint'])
                if recovery_used else 0)
            msg.bootstrap_recovery_delta_rad.append(
                float(segment['bootstrap_recovery_delta_rad'])
                if recovery_used else 0.0)
            evidence = {
                'used': recovery_used,
                'minimum_clearance_m': (
                    float(segment['bootstrap_recovery_minimum_clearance_m'])
                    if recovery_used else None),
                'limiting_link_pair': (
                    str(segment['bootstrap_recovery_limiting_link_pair'])
                    if recovery_used else ''),
                'validation_samples': (
                    int(segment['bootstrap_recovery_samples'])
                    if recovery_used else 0),
                'start_contacts': (
                    segment.get('bootstrap_start_contacts', [])
                    if recovery_used else []),
                'joint_numbers': (
                    segment.get('bootstrap_recovery_joints', [])
                    if recovery_used else []),
                'delta_rad': (
                    segment.get('bootstrap_recovery_deltas_rad', [])
                    if recovery_used else []),
            }
            msg.bootstrap_recovery_evidence_json.append(
                json.dumps(evidence, sort_keys=True, separators=(',', ':')))
        self.plan_pub.publish(msg)
        self.set_status('PROPOSAL_READY', msg.reason, payload['request_id'])

    def publish_rejection(self, request_id, code, reason):
        msg = TesseractPlan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        # Rejections are plans too for correlation purposes. Preserve the
        # complete request ID so a caller can fail this attempt immediately
        # without accepting or waiting on another generation's result.
        msg.plan_id = str(request_id)
        pending = self.pending.get(request_id, {})
        request = pending.get('request', {})
        msg.plan_kind = str(request.get('plan_kind', ''))
        msg.source_request_id = str(
            request.get('target_provenance', {}).get(
                'source_request_id', ''))
        msg.valid = False
        msg.dry_run = True
        msg.real_arm_motion = False
        msg.reason = '%s: %s' % (code, reason)
        msg.rejection_codes = [code]
        self.plan_pub.publish(msg)
        self.set_status('REJECTED', msg.reason, request_id)

    def set_status(self, state, reason, request_id=''):
        self.state = state
        self.reason = reason
        self.publish_status(request_id)

    def publish_status(self, request_id=''):
        msg = TesseractPlanStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.request_id = request_id or (next(iter(self.pending)) if self.pending else '')
        msg.state = self.state
        msg.reason = self.reason
        msg.dry_run = True
        msg.real_arm_motion = False
        msg.pending_requests = self.spool.pending('requests') + self.spool.pending('processing')
        msg.pending_responses = self.spool.pending('responses')
        self.status_pub.publish(msg)

    def publish_readiness(self):
        worker_blockers = self.worker_health_reasons()
        acquisition_blockers = self.snapshot_reasons(
            'ROUGH_ACQUISITION',
            require_viewpoints=False,
            worker_reasons=worker_blockers,
        )
        multiview_blockers = self.snapshot_reasons(
            'MULTIVIEW_SCAN',
            require_viewpoints=True,
            worker_reasons=worker_blockers,
        )
        manipulation_blockers = list(multiview_blockers)
        # Contact motion remains fail-closed until the installed planning
        # model explicitly contains a qualified gripper TCP, open/closed
        # geometry, attached-object handling, and allowed-contact policy.
        if not bool(self.get_parameter(
                'manipulation_model_qualified').value):
            manipulation_blockers.append(
                'gripper/contact collision model is not qualified')
        manipulation_blockers = list(dict.fromkeys(manipulation_blockers))
        msg = TesseractReadiness()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.generation_id = self.worker_generation_id
        msg.worker_ready = not worker_blockers
        msg.acquisition_blockers = acquisition_blockers
        msg.multiview_blockers = multiview_blockers
        msg.manipulation_blockers = manipulation_blockers
        msg.acquisition_ready = not acquisition_blockers
        msg.multiview_ready = not multiview_blockers
        msg.manipulation_ready = not manipulation_blockers
        self.readiness_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = TesseractPlanBridge()
    except (ContractError, OSError, RuntimeError, ValueError) as error:
        print('Tesseract bridge startup error: %s' % error)
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
