#!/usr/bin/env python3
"""Command-free Foxy bridge for the isolated Tesseract plan worker."""

import hashlib
import json
import math
import os
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
from piper_mobile_manipulation.execution.motion import (
    load_accepted_hand_eye,
    load_conservative_joint_limits,
    PiperScanKinematics,
)
from piper_mobile_manipulation.execution.modes import (
    commanded_speed_percent,
)
from piper_mobile_manipulation.planning.generation import (
    parse_view_generation,
)
from piper_mobile_manipulation.planning.rays import (
    bind_shortlisted_ray_intervals,
)
from piper_mobile_manipulation.motion_limit_stability import MotionLimitStability
from piper_mobile_manipulation.ray_mission_diagnostics import (
    add_bridge_request,
    add_request_rejection,
    add_tesseract_response,
    RayMissionDiagnosticsStore,
)
from piper_mobile_manipulation.planning.ray_culls import (
    hard_cull_snapshot,
    stable_revision,
)
from piper_mobile_manipulation.srv import RequestTesseractPlan

from piper_tesseract_foxy.protocol.contract import (
    attach_digest,
    ContractError,
    JOINT_NAMES,
    MAX_OBSTACLES,
    MAX_CONFIGURED_HOME_START_LIMIT_TOLERANCE_RAD,
    PLAN_KINDS,
    SCHEMA_VERSION,
    TIMING_POLICY,
    sha256_file,
    validate_response,
)
from piper_tesseract_foxy.protocol.spool import Spool

from piper_tesseract_foxy.candidate_selection import (  # noqa: F401
    FINAL_AIM_EXECUTION_MARGIN_DEG,  # noqa: F401 - compatibility export
    RAY_DIRECTION_ATTEMPT_LIMIT,
    balanced_closed_loop_candidates,
    bounded_candidate_attempt_limit,
    bounded_current_look_direction,  # noqa: F401 - compatibility export
    bounded_nbv_candidates,
    exact_target_aim_candidates,
    information_ranked_ray_candidates,
    local_view_frontier_candidates,  # noqa: F401 - compatibility export
    maximize_successive_view_distance,  # noqa: F401 - compatibility export
    obstacle_scene_rejection_reason,
    permanent_ray_ids_from_response,
    relax_closed_loop_candidate_aims,  # noqa: F401 - compatibility export
    select_diverse_smooth_view_path,
    target_envelope_obstacles,
    uses_authoritative_nbv_order,  # noqa: F401 - compatibility export
    validate_candidate_policy_batch,
)


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
            'view_generation_receipt_topic':
                '/piper/tesseract_view_generation',
            'ray_hard_culls_topic': '/piper/ray_hard_culls',
            'plan_provenance_topic': '/piper/tesseract_plan_provenance',
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
            'trajectory_joint_step_rad': 0.05,
            'trajectory_command_rate_hz': 20.0,
            'speed_percent': 5.0,
            'roll_samples_rad': [-2.094395102, -1.047197551, 0.0,
                                 1.047197551, 2.094395102, 3.141592654],
            'deterministic_seed': 42,
            'return_home_positions_rad': [
                0.000366362, 0.0, 0.0, 0.0, 0.43869236, 0.0,
            ],
            'closed_loop_one_view': False,
            # From an achieved pose between fixed 15-degree grid samples, the
            # next grid neighbor can be only about seven degrees away in 3D
            # target-direction space at normal elevation. Six degrees admits
            # that visually distinct neighbor while the
            # independent pose/look duplicate gate still rejects repeats.
            'closed_loop_min_view_step_deg': 6.0,
            'closed_loop_max_view_step_deg': 30.0,
            'closed_loop_candidate_limit': 12,
            # Preserve a comfortable achieved wrist aim when the target stays
            # within this strict subset of the executor's 20-degree cone.
            'closed_loop_max_aim_offset_deg': 5.0,
            'manipulation_model_qualified': False,
            'ray_diagnostics_enabled': True,
            'ray_diagnostics_root': os.path.join(
                os.environ.get('PIPER_ARM_ROOT', '/home/prl/Piper_arm'),
                'datasets', 'ray_diagnostics'),
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
        self.tesseract_exhausted_ray_generation = None
        self.tesseract_exhausted_ray_ids = set()
        self.remaining_ray_pool_session = None
        self.remaining_ray_ids = set()
        self.retired_ray_ids = set()
        self.ray_diagnostics_store = RayMissionDiagnosticsStore(
            self.get_parameter('ray_diagnostics_root').value)
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
        self.view_generation_pub = self.create_publisher(
            String,
            self.get_parameter('view_generation_receipt_topic').value,
            self.plan_qos)
        self.hard_cull_pub = self.create_publisher(
            String, self.get_parameter('ray_hard_culls_topic').value,
            self.plan_qos)
        self.plan_provenance_pub = self.create_publisher(
            String, self.get_parameter('plan_provenance_topic').value,
            QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ))
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
        self.request_return_home_service = self.create_service(
            RequestTesseractPlan, '~/request_return_home_plan',
            self.request_return_home_plan_cb)
        self.request_startup_home_service = self.create_service(
            RequestTesseractPlan, '~/request_startup_home_plan',
            self.request_startup_home_plan_cb)
        self.poll_timer = self.create_timer(0.20, self.poll)
        self.publish_status()
        self.publish_readiness()
        self.get_logger().warn(
            'Tesseract bridge is command-free: it has no /joint_ctrl_single publisher '
            'and no motor-enable client. Ignored saved bounds: %s'
            % (','.join(self.ignored_bounds) or 'none'))

    def parameter_path(self, name):
        return Path(str(self.get_parameter(name).value)).resolve()

    def param_bool(self, name):
        """Return one declared ROS parameter using the shared bool contract."""
        value = self.get_parameter(name).value
        if isinstance(value, str):
            return value.lower() in ('1', 'true', 'yes', 'on')
        return bool(value)

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
            if not acquisition:
                try:
                    generation = parse_view_generation(value)
                except (TypeError, ValueError):
                    return
                receipt = String()
                receipt.data = json.dumps({
                    'bridge_received_at_ns': time.time_ns(),
                    'view_generation': generation.to_dict(),
                }, sort_keys=True)
                self.view_generation_pub.publish(receipt)
                if self.param_bool('debug'):
                    self.get_logger().info(
                        'cached view generation session=%s accepted=%d '
                        'policy=%s ready=%s candidates=%d'
                        % (
                            generation.session_id,
                            generation.accepted_views,
                            generation.policy,
                            generation.ready,
                            generation.candidate_viewpoints))

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
        expected_hashes = {
            'srdf_sha256': sha256_file(self.parameter_path('srdf_path')),
            'collision_manifest_sha256': sha256_file(
                self.parameter_path('collision_manifest_path')),
        }
        mismatches = [
            name for name, expected in expected_hashes.items()
            if health.get(name) != expected
        ]
        if mismatches:
            self.worker_generation_id = generation
            return [
                'Tesseract worker collision profile does not match bridge: '
                + ', '.join(mismatches)
            ]
        self.worker_generation_id = generation
        return []

    def snapshot_reasons(
            self, plan_kind='MULTIVIEW_SCAN', require_viewpoints=True,
            worker_reasons=None, startup_home=False):
        if plan_kind not in PLAN_KINDS:
            return ['unsupported plan kind']
        if startup_home and plan_kind != 'RETURN_HOME':
            return ['startup home is RETURN_HOME-only']
        reasons = (
            self.worker_health_reasons()
            if worker_reasons is None else list(worker_reasons))
        # Dedicated RETURN_HOME plans are direct configured joint targets and
        # intentionally do not consume perception or collision-scene state.
        required = ['joints']
        if plan_kind != 'RETURN_HOME':
            required.append('camera_clock')
        if plan_kind == 'MULTIVIEW_SCAN':
            required.extend(['tracking', 'obstacles'])
            if require_viewpoints:
                required.append('scan')
        elif plan_kind != 'RETURN_HOME' and require_viewpoints:
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
        if (
                plan_kind != 'RETURN_HOME'
                and (
                    self.latest_camera_health is None
                    or not self.latest_camera_health.healthy)):
            reasons.append('camera timestamp health is not healthy')
        if plan_kind == 'MULTIVIEW_SCAN':
            obstacle_reason = obstacle_scene_rejection_reason(self.latest_obstacles)
            if obstacle_reason:
                reasons.append(obstacle_reason)
        scan = (
            self.latest_scan if plan_kind == 'MULTIVIEW_SCAN'
            else self.latest_acquisition_scan)
        if plan_kind != 'RETURN_HOME' and require_viewpoints and (
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
            selection_failure = str(
                scan.get('selection_failure_code', '')
                if isinstance(scan, dict) else '').strip()
            if selection_failure:
                reasons.append(selection_failure)
            elif not (
                    isinstance(scan, dict)
                    and scan.get('viewpoints', [])):
                reasons.append('NO_PREQUALIFIED_VIEWPOINT_CANDIDATE')
            elif not any(
                    isinstance(item, dict)
                    and item.get(
                        'prequalified', item.get('reachable')) is True
                    and item.get('safe') is True
                    for item in scan.get('viewpoints', [])):
                # Preserve the reason supplied by the command-free
                # prequalification stage.  In particular, a transient target
                # status dip must enter the mission's existing bounded visual
                # reacquisition hold instead of being flattened later into the
                # misleading "only 0 safe candidates" request-build error.
                target_status = str(
                    scan.get('filter', {}).get('target_status', '')
                    if isinstance(scan.get('filter'), dict) else '').strip()
                if target_status in (
                        'LOW_CONFIDENCE', 'LOST', 'SEARCHING'):
                    reasons.append('target_status=' + target_status)
                else:
                    reasons.append('NO_PREQUALIFIED_VIEWPOINT_CANDIDATE')
        return list(dict.fromkeys(reasons))

    def request_plan_cb(self, request, response):
        return self.request_kind_cb(
            'MULTIVIEW_SCAN', request, response)

    def request_acquisition_plan_cb(self, request, response):
        return self.request_kind_cb(
            'ROUGH_ACQUISITION', request, response)

    def request_return_home_plan_cb(self, request, response):
        return self.request_kind_cb('RETURN_HOME', request, response)

    def request_startup_home_plan_cb(self, request, response):
        return self.request_kind_cb(
            'RETURN_HOME', request, response, startup_home=True)

    def request_kind_cb(
            self, plan_kind, request, response, startup_home=False):
        if self.pending and not request.force_refresh:
            response.accepted = False
            response.request_id = next(iter(self.pending))
            response.message = 'a Tesseract request is already pending'
            return response
        reasons = self.snapshot_reasons(
            plan_kind, startup_home=startup_home)
        if reasons:
            response.accepted = False
            response.request_id = ''
            response.message = 'planning blocked: ' + '; '.join(reasons)
            state = (
                'NO_POSITIVE_INFORMATION_CANDIDATE'
                if 'NO_POSITIVE_INFORMATION_CANDIDATE' in reasons
                else 'SNAPSHOT_BLOCKED')
            self.set_status(state, response.message)
            return response
        try:
            home_stage = str(getattr(request, 'home_stage', '')).strip()
            joint_goal = [
                float(value) for value in
                getattr(request, 'joint_goal_positions_rad', [])]
            payload = self.build_request(
                plan_kind, startup_home=startup_home,
                home_stage=home_stage, joint_goal=joint_goal)
            self.spool.write('requests', payload['request_id'], payload)
        except (ContractError, KeyError, OSError, TypeError, ValueError) as error:
            response.accepted = False
            response.request_id = ''
            response.message = 'request creation failed: %s' % error
            self.set_status('REJECTED', response.message)
            return response
        ray_diagnostics = self.record_ray_request(payload)
        self.pending[payload['request_id']] = {
            'request': payload,
            'started': self.now(),
            'ray_diagnostics': ray_diagnostics,
        }
        response.accepted = True
        response.request_id = payload['request_id']
        response.message = 'command-free Tesseract planning request queued'
        self.set_status('PLANNING', response.message, payload['request_id'])
        return response

    def build_request(
            self, plan_kind='MULTIVIEW_SCAN', startup_home=False,
            home_stage='', joint_goal=None):
        if plan_kind not in PLAN_KINDS:
            raise ContractError('unsupported plan kind')
        if startup_home and plan_kind != 'RETURN_HOME':
            raise ContractError('startup home is RETURN_HOME-only')
        joint_goal = list(joint_goal or [])
        if plan_kind != 'RETURN_HOME' and (home_stage or joint_goal):
            raise ContractError('home-stage overrides are RETURN_HOME-only')
        if plan_kind == 'RETURN_HOME':
            home_stage = str(home_stage or 'CONFIGURED_HOME').strip().upper()
            if home_stage not in (
                    'CONFIGURED_HOME', 'STARTUP_WRIST', 'ROUGH_HOME',
                    'STORAGE_WRIST'):
                raise ContractError('unsupported home stage: %s' % home_stage)
            if joint_goal and len(joint_goal) != 6:
                raise ContractError(
                    'staged home joint goal must contain six positions')
            if home_stage != 'CONFIGURED_HOME' and len(joint_goal) != 6:
                raise ContractError(
                    '%s requires an explicit six-joint goal' % home_stage)
            if any(not math.isfinite(value) for value in joint_goal):
                raise ContractError('staged home joint goal is non-finite')
        else:
            home_stage = ''
        now_ns = time.time_ns()
        ttl_ns = int(float(self.get_parameter('request_ttl_sec').value) * 1e9)
        joints = [float(value) for value in self.latest_joints.position[:6]]
        scan = (
            self.latest_scan if plan_kind == 'MULTIVIEW_SCAN'
            else self.latest_acquisition_scan)
        center = (
            [0.0, 0.0, 0.0]
            if plan_kind == 'RETURN_HOME'
            else self.vector(scan.get('target_object_center'), 'target center'))
        provenance = self.target_provenance(scan, plan_kind)
        candidates = []
        for item in ([] if plan_kind == 'RETURN_HOME' else scan.get('viewpoints', [])):
            if not isinstance(item, dict) or item.get(
                    'prequalified', item.get('reachable')) is not True \
                    or item.get('safe') is not True:
                continue
            candidate = {
                'id': int(item.get('index', len(candidates))),
                'camera_position_m': self.vector(
                    item.get('desired_camera_position'), 'camera position'),
                'look_direction': self.vector(
                    item.get('desired_look_at_direction'), 'look direction'),
                'required_first': bool(
                    plan_kind == 'ROUGH_ACQUISITION'
                    and item.get('keep_object_centered') is True),
                'coverage_score': float(
                    item.get('expected_new_coverage_score', 0.0)),
                'coverage_objective': str(
                    item.get('coverage_objective', '')),
                'coverage_progress_score': float(
                    item.get('coverage_progress_score', 0.0)),
                'view_selection_policy': str(
                    item.get('view_selection_policy', 'legacy')),
                'view_selection_requested_policy': str(item.get(
                    'view_selection_requested_policy',
                    item.get('view_selection_policy', 'legacy'))),
                'view_selection_generation': int(
                    item.get('view_selection_generation', 0)),
                'view_selection_session_id': str(
                    item.get('view_selection_session_id', '')),
                'nbv_rank': int(item.get('nbv_rank', 0)),
                'nbv_positive_information_gain': bool(
                    item.get('nbv_positive_information_gain', False)),
                'nbv_predicted_unknown_pixels': int(
                    item.get('nbv_predicted_unknown_pixels', 0)),
                'nbv_novel_surface_pixels': int(
                    item.get('nbv_novel_surface_pixels', 0)),
                'nbv_marginal_information_pixels': int(
                    item.get('nbv_marginal_information_pixels', 0)),
                'nbv_marginal_information_fraction': float(
                    item.get('nbv_marginal_information_fraction', 0.0)),
                'nbv_projected_object_pixels': int(
                    item.get('nbv_projected_object_pixels', 0)),
                'nbv_direction_novelty_deg': float(
                    item.get('nbv_direction_novelty_deg', 0.0)),
                'nbv_camera_travel_m': float(
                    item.get('nbv_camera_travel_m', 0.0)),
            }
            if item.get('candidate_geometry') == 'target_ray':
                candidate.update({
                    'candidate_geometry': 'target_ray',
                    'ray_id': int(item.get('ray_id', candidate['id'])),
                    'ray_direction': self.vector(
                        item.get('ray_direction'), 'ray direction'),
                    'ray_min_standoff_m': float(
                        item.get('ray_min_standoff_m')),
                    'ray_max_standoff_m': float(
                        item.get('ray_max_standoff_m')),
                    'ray_preferred_max_standoff_m': float(
                        item.get('ray_preferred_max_standoff_m')),
                    'ray_scoring_standoff_m': float(
                        item.get('ray_scoring_standoff_m')),
                })
                intervals = item.get('ray_capability_intervals_m')
                if intervals is not None:
                    if (
                            not isinstance(intervals, list)
                            or not intervals
                            or any(
                                not isinstance(interval, list)
                                or len(interval) != 2
                                for interval in intervals)):
                        raise ContractError(
                            'target-ray capability intervals are malformed')
                    candidate['ray_capability_intervals_m'] = [
                        [float(interval[0]), float(interval[1])]
                        for interval in intervals]
                for key in (
                        'ray_requested_min_standoff_m',
                        'ray_requested_max_standoff_m'):
                    if item.get(key) is not None:
                        candidate[key] = float(item[key])
            candidates.append(candidate)
        configured_maximum = max(
            1, int(self.get_parameter('max_execution_viewpoints').value))
        session = scan.get('scan_session', {}) if isinstance(scan, dict) else {}
        shortlisted_ray_count = 0
        expanded_ray_candidate_count = 0
        candidate_capabilities = None
        if plan_kind == 'MULTIVIEW_SCAN':
            candidate_capabilities = validate_candidate_policy_batch(
                candidates)
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
            (1 if bool(self.get_parameter('closed_loop_one_view').value)
             else remaining_views) if plan_kind == 'MULTIVIEW_SCAN'
            else (
                0 if plan_kind == 'RETURN_HOME'
                else min(1, configured_maximum, len(candidates))))
        minimum = maximum if plan_kind in ('MULTIVIEW_SCAN', 'RETURN_HOME') else 1
        closed_loop_one_view = bool(
            plan_kind == 'MULTIVIEW_SCAN'
            and self.get_parameter('closed_loop_one_view').value)
        tracking_scale = (
            float(self.latest_tracking.recommended_speed_scale)
            if self.latest_tracking is not None else 1.0)
        execution_speed = commanded_speed_percent(
            float(self.get_parameter('speed_percent').value),
            plan_kind,
            tracking_scale,
        )
        if plan_kind == 'MULTIVIEW_SCAN':
            current_camera_transform = self.kinematics.camera_transform(joints)
            current_camera = current_camera_transform[:3, 3]
            current_look = current_camera_transform[:3, 2]
            candidate_limit = (
                max(1, int(self.get_parameter(
                    'closed_loop_candidate_limit').value))
                if closed_loop_one_view
                else max(20, maximum * 4, maximum))
            if closed_loop_one_view:
                authoritative_nbv = bool(
                    candidate_capabilities is not None
                    and candidate_capabilities.authoritative_nbv)
                if (
                        candidate_capabilities is not None
                        and candidate_capabilities.ray_expansion):
                    # Retire failed rays from the complete prequalified pool
                    # before choosing the next bounded shortlist.  Applying
                    # this after shortlisting mistakes one exhausted batch for
                    # exhaustion of the full ray frontier.
                    candidates = self.exclude_tesseract_exhausted_rays(
                        candidates, session_id, accepted_views)
                if authoritative_nbv:
                    effective_limit = bounded_candidate_attempt_limit(
                        candidate_capabilities, candidate_limit)
                    if (
                            candidate_capabilities is not None
                            and candidate_capabilities.ray_expansion):
                        candidates = information_ranked_ray_candidates(
                            candidates, effective_limit)
                    else:
                        candidates = bounded_nbv_candidates(
                            candidates, current_camera, center,
                            effective_limit)
                else:
                    effective_limit = bounded_candidate_attempt_limit(
                        candidate_capabilities, candidate_limit)
                    candidates = balanced_closed_loop_candidates(
                        candidates, current_camera, effective_limit,
                        compact_first=(accepted_views == 0))
                if (
                        candidate_capabilities is not None
                        and candidate_capabilities.ray_expansion):
                    shortlisted_ray_count = sum(
                        item.get('candidate_geometry') == 'target_ray'
                        for item in candidates)
                    if shortlisted_ray_count != len(candidates):
                        raise ContractError(
                            'ray policy produced mixed candidate geometry')
                    candidates = bind_shortlisted_ray_intervals(
                        candidates, current_camera, center)
                    expanded_ray_candidate_count = len(candidates)
                candidates = exact_target_aim_candidates(
                    candidates,
                    center,
                    current_look,
                    self.get_parameter(
                        'closed_loop_max_aim_offset_deg').value,
                )
            else:
                candidates = candidates[:candidate_limit]
                candidates = select_diverse_smooth_view_path(
                    candidates, maximum, current_camera)
        else:
            candidates = candidates[:max(20, maximum * 4, maximum)]
        required_candidates = (
            0 if plan_kind == 'RETURN_HOME'
            else (maximum if plan_kind == 'MULTIVIEW_SCAN' else minimum))
        if len(candidates) < required_candidates:
            raise ContractError('only %d safe candidates; need at least %d' % (
                len(candidates), required_candidates))
        observation_mode = (
            'bootstrap_static'
            if plan_kind == 'ROUGH_ACQUISITION'
            else 'perception_snapshot')
        obstacles = []
        target_envelope = None
        if plan_kind == 'MULTIVIEW_SCAN':
            target_envelope, target_boxes = target_envelope_obstacles(
                scan, center)
            obstacles.extend(target_boxes)
        if (
                plan_kind != 'RETURN_HOME'
                and observation_mode == 'perception_snapshot'
                and not startup_home):
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
        if len(obstacles) > MAX_OBSTACLES:
            raise ContractError(
                'target envelope and scene exceed the obstacle contract')
        identity = {
            'created_at_ns': now_ns,
            'joint_positions': [round(value, 9) for value in joints],
            'scan_stamp': (
                scan.get('header', {}).get('stamp', {})
                if isinstance(scan, dict) else {}),
            'plan_kind': plan_kind,
            'startup_home': bool(startup_home),
            'home_stage': home_stage,
            'joint_goal_positions_rad': [
                round(value, 9) for value in joint_goal],
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
            'ray_population': (
                dict(scan.get('ray_population', {}))
                if plan_kind == 'MULTIVIEW_SCAN' else {}),
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
                'target_envelope': target_envelope,
                'observation_mode': observation_mode,
                'startup_home_static': bool(startup_home),
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
                # A disabled arm can relax slightly beyond an inclusive
                # controller-coordinate boundary at the configured storage
                # fold.  Only a dedicated direct RETURN_HOME request may
                # carry that measured start back inside the exact limits.
                'configured_home_start_limit_tolerance_rad': (
                    MAX_CONFIGURED_HOME_START_LIMIT_TOLERANCE_RAD
                    if plan_kind == 'RETURN_HOME' else 0.0),
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
                # Automatic one-view transactions always hold for a fresh
                # measured-coverage decision and later use a separate direct
                # configured-home request. Do not reject an otherwise safe
                # capture because an unused embedded contingency home cannot
                # represent the intentional storage-fold collision bypass.
                'include_return_home': bool(
                    plan_kind == 'RETURN_HOME'
                    or (plan_kind == 'MULTIVIEW_SCAN'
                        and not closed_loop_one_view)),
                'max_execution_joint_step_rad': float(
                    self.get_parameter('trajectory_joint_step_rad').value),
                'effective_speed_percent': execution_speed,
                'command_rate_hz': float(
                    self.get_parameter('trajectory_command_rate_hz').value),
                'timing_policy': TIMING_POLICY,
                'joint_specific_costs': {},
                'return_home_positions_rad': (
                    (
                        [float(value) for value in joint_goal]
                        if joint_goal else
                        [float(value) for value in self.get_parameter(
                            'return_home_positions_rad').value]
                    )
                    if plan_kind in ('MULTIVIEW_SCAN', 'RETURN_HOME') else []
                ),
                'home_stage': home_stage,
                'shortlisted_ray_count': int(shortlisted_ray_count),
                'expanded_ray_candidate_count': int(
                    expanded_ray_candidate_count),
                'ray_direction_attempt_limit': int(
                    RAY_DIRECTION_ATTEMPT_LIMIT
                    if shortlisted_ray_count else 0),
            },
        }
        return attach_digest(payload, 'request_sha256')

    def record_ray_request(self, payload):
        """Persist bridge shortlist membership without influencing planning."""
        try:
            return self._record_ray_request(payload)
        except Exception as error:  # Diagnostics must never reject a request.
            self.warn_ray_diagnostics(
                'Could not update ray mission diagnostics: %s' % error)
            return None

    def _record_ray_request(self, payload):
        if payload.get('plan_kind') != 'MULTIVIEW_SCAN':
            return None
        scan = getattr(self, 'latest_scan', None)
        diagnostics = (
            scan.get('ray_diagnostics') if isinstance(scan, dict) else None)
        if not isinstance(diagnostics, dict):
            return None
        shortlisted = {
            int(item['ray_id'])
            for item in payload.get('scene', {}).get('candidate_views', [])
            if item.get('ray_id') is not None}
        diagnostics = add_bridge_request(
            diagnostics,
            payload.get('request_id', ''),
            shortlisted,
            getattr(self, 'retired_ray_ids', set()),
            getattr(self, 'tesseract_exhausted_ray_ids', set()),
        )
        if bool(self.get_parameter('ray_diagnostics_enabled').value):
            self.ray_diagnostics_store.record(diagnostics)
        return diagnostics

    def record_ray_response(self, payload):
        """Persist worker attempt and feasibility evidence for one request."""
        try:
            self._record_ray_response(payload)
        except Exception as error:  # Diagnostics must never reject a response.
            self.warn_ray_diagnostics(
                'Could not record Tesseract ray diagnostics: %s' % error)

    def _record_ray_response(self, payload):
        request_id = str(payload.get('request_id', ''))
        state = getattr(self, 'pending', {}).get(request_id)
        if not isinstance(state, dict):
            return
        diagnostics = state.get('ray_diagnostics')
        if not isinstance(diagnostics, dict):
            return
        diagnostics = add_tesseract_response(
            diagnostics, payload, state.get('request'))
        state['ray_diagnostics'] = diagnostics
        state['ray_response_recorded'] = True
        if bool(self.get_parameter('ray_diagnostics_enabled').value):
            self.ray_diagnostics_store.record(diagnostics)

    def record_ray_rejection(self, request_id, code, reason):
        """Persist transport/validation failures that have no worker result."""
        try:
            self._record_ray_rejection(request_id, code, reason)
        except Exception as error:  # Diagnostics must never mask rejection.
            self.warn_ray_diagnostics(
                'Could not record ray request rejection: %s' % error)

    def _record_ray_rejection(self, request_id, code, reason):
        state = getattr(self, 'pending', {}).get(str(request_id))
        if not isinstance(state, dict) or state.get('ray_response_recorded'):
            return
        diagnostics = state.get('ray_diagnostics')
        if not isinstance(diagnostics, dict):
            return
        diagnostics = add_request_rejection(
            diagnostics, request_id, code, reason)
        state['ray_diagnostics'] = diagnostics
        if bool(self.get_parameter('ray_diagnostics_enabled').value):
            self.ray_diagnostics_store.record(diagnostics)

    def warn_ray_diagnostics(self, message):
        try:
            self.get_logger().warn(str(message))
        except Exception:
            pass

    @staticmethod
    def target_provenance(scan, plan_kind):
        if plan_kind == 'RETURN_HOME':
            now_ns = time.time_ns()
            return {
                'source': 'configured_home',
                'frame_id': 'base_link',
                'stamp': {
                    'sec': int(now_ns // 1_000_000_000),
                    'nanosec': int(now_ns % 1_000_000_000),
                },
            }
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

    def exclude_tesseract_exhausted_rays(
            self, candidates, session_id, accepted_views):
        """Exclude static mission failures and current-generation failures."""
        session = str(session_id)
        generation = (session, int(accepted_views))
        candidate_ids = {
            int(item.get('ray_id', item.get('id', -1)))
            for item in candidates
        }
        if getattr(self, 'remaining_ray_pool_session', None) != session:
            self.remaining_ray_pool_session = session
            self.remaining_ray_ids = set(candidate_ids)
            self.retired_ray_ids = set()
        else:
            # A temporarily absent planner candidate must not erase a frozen
            # mission ray.  Current accepted coverage still controls the
            # candidates presented below, while proven static failures stay
            # retired for the whole session.
            self.remaining_ray_ids.update(
                candidate_ids.difference(self.retired_ray_ids))
        if getattr(
                self, 'tesseract_exhausted_ray_generation', None
        ) != generation:
            self.tesseract_exhausted_ray_generation = generation
            self.tesseract_exhausted_ray_ids = set()
        transient = set(getattr(
            self, 'tesseract_exhausted_ray_ids', set()))
        available = [
            item for item in candidates
            if (
                int(item.get('ray_id', item.get('id', -1)))
                in self.remaining_ray_ids
                and int(item.get('ray_id', item.get('id', -1)))
                not in transient)
        ]
        if not available:
            raise ContractError(
                'RAY_FRONTIER_EXHAUSTED: all %d prequalified rays were '
                'rejected by Tesseract for accepted-view generation %d'
                % (len(candidates), int(accepted_views)))
        self.get_logger().info(
            'mission ray pool session=%s accepted=%d: pool=%d, '
            'transiently exhausted=%d, available=%d'
            % (
                str(session_id), int(accepted_views),
                len(self.remaining_ray_ids), len(transient),
                len(available)))
        return available

    def remember_permanently_infeasible_rays(self, payload):
        """Retire endpoint failures for the frozen target-ray session."""
        request_id = str(payload.get('request_id', ''))
        pending = getattr(self, 'pending', {}).get(request_id, {})
        request = pending.get('request', {})
        if int(request.get('planning', {}).get(
                'shortlisted_ray_count', 0)) < 1:
            return []
        session = request.get('scan_session', {})
        session_id = str(session.get('session_id', ''))
        if not session_id:
            return []
        if getattr(self, 'remaining_ray_pool_session', None) != session_id:
            return []
        request_ray_ids = {
            int(item['ray_id'])
            for item in request.get('scene', {}).get('candidate_views', [])
            if item.get('ray_id') is not None
        }
        reported = set(permanent_ray_ids_from_response(
            request, payload.get('planning_diagnostics', {})))
        reported.intersection_update(request_ray_ids)
        newly_infeasible = sorted(
            reported.intersection(self.remaining_ray_ids))
        self.remaining_ray_ids.difference_update(newly_infeasible)
        self.retired_ray_ids.update(newly_infeasible)
        if newly_infeasible:
            self.get_logger().info(
                'retired static endpoint-infeasible rays for session=%s: %s'
                % (session_id, newly_infeasible))
        self.publish_permanent_hard_culls(
            request, int(session.get('accepted_views', 0)))
        return newly_infeasible

    def publish_permanent_hard_culls(self, request, generation):
        """Return worker-proven endpoint failures to the upstream planner."""
        try:
            population = request.get('ray_population')
            if not isinstance(population, dict) or not population:
                return
            revision = stable_revision({
                'model': request.get('model', {}),
                'calibration_sha256': request.get(
                    'calibration', {}).get('hand_eye_sha256', ''),
                'position_limits': request.get(
                    'limits', {}).get('position_rad', []),
            })
            culls = [{
                'ray_id': int(ray_id),
                'stage': 'tesseract_endpoint',
                'reason_code': 'PERMANENT_ENDPOINT_INFEASIBLE',
                'reason': (
                    'retired after permanent Tesseract endpoint '
                    'infeasibility'),
                'evidence': {'worker_reported_permanent': True},
            } for ray_id in sorted(self.retired_ray_ids)]
            message = String()
            message.data = json.dumps(hard_cull_snapshot(
                population, 'tesseract_endpoint', revision, generation,
                culls), sort_keys=True)
            self.hard_cull_pub.publish(message)
        except (KeyError, TypeError, ValueError) as error:
            self.warn_ray_diagnostics(
                'Could not publish permanent hard-ray culls: %s' % error)

    def remember_tesseract_exhausted_rays(self, payload):
        """Retire only rays actually attempted by one failed worker request."""
        codes = [str(value) for value in payload.get('rejection_codes', [])]
        if 'TESSERACT_EXHAUSTED' not in codes:
            return []
        request_id = str(payload.get('request_id', ''))
        pending = getattr(self, 'pending', {}).get(request_id, {})
        request = pending.get('request', {})
        if int(request.get('planning', {}).get(
                'shortlisted_ray_count', 0)) < 1:
            return []
        session = request.get('scan_session', {})
        generation = (
            str(session.get('session_id', '')),
            int(session.get('accepted_views', -1)),
        )
        if (
                not generation[0]
                or generation != getattr(
                    self, 'tesseract_exhausted_ray_generation', None)):
            return []
        request_ray_ids = {
            int(item['ray_id'])
            for item in request.get('scene', {}).get('candidate_views', [])
            if item.get('ray_id') is not None
        }
        attempted = {
            int(value) for value in payload.get(
                'planning_diagnostics', {}).get('attempted_ray_ids', [])
        }.intersection(request_ray_ids)
        exhausted = self.tesseract_exhausted_ray_ids
        newly_exhausted = sorted(attempted.difference(exhausted))
        exhausted.update(newly_exhausted)
        if newly_exhausted:
            self.get_logger().info(
                'retired Tesseract-infeasible rays for session=%s '
                'accepted=%d: %s'
                % (generation[0], generation[1], newly_exhausted))
        return newly_exhausted

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
                self.record_ray_rejection(
                    request_id, 'PLANNER_TIMEOUT',
                    'Tesseract response timed out')
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
        self.record_ray_response(payload)
        self.remember_permanently_infeasible_rays(payload)
        if payload.get('status') != 'success':
            codes = payload.get('rejection_codes') or ['PLANNING_FAILED']
            exhausted_rays = self.remember_tesseract_exhausted_rays(payload)
            if exhausted_rays:
                self.publish_rejection(
                    payload.get('request_id', ''),
                    'RAY_SHORTLIST_EXHAUSTED',
                    'TESSERACT_EXHAUSTED: %s; retired attempted ray IDs %s'
                    % (
                        str(payload.get('diagnostic', 'planning failed')),
                        exhausted_rays),
                    additional_codes=codes)
                return
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
                'startup_home_static': bool(
                    segment.get('startup_home_static', False)),
                'configured_home_direct_joint_move': bool(
                    segment.get('configured_home_direct_joint_move', False)),
                'configured_home_goal_positions_rad': segment.get(
                    'configured_home_goal_positions_rad', []),
                'collision_validation_bypassed': bool(
                    segment.get('collision_validation_bypassed', False)),
                'home_stage': str(segment.get('home_stage', '')),
                'validation': str(segment.get('validation', '')),
                'trajectory_blending': str(segment.get(
                    'trajectory_blending', '')),
                'pass_through_blending_applied': bool(segment.get(
                    'pass_through_blending_applied', False)),
                'pass_through_blend_fallback_used': bool(segment.get(
                    'pass_through_blend_fallback_used', False)),
                'pass_through_blended_corners': int(segment.get(
                    'pass_through_blended_corners', 0)),
                'pass_through_source_points': int(segment.get(
                    'pass_through_source_points', 0)),
                'pass_through_geometry_points': int(segment.get(
                    'pass_through_geometry_points', 0)),
                'pass_through_maximum_radius_rad': float(segment.get(
                    'pass_through_maximum_radius_rad', 0.0)),
                'pass_through_blend_reason': str(segment.get(
                    'pass_through_blend_reason', '')),
                'sdk_execution_mode': str(segment.get(
                    'sdk_execution_mode', 'TESSERACT_STREAM')),
                'sdk_command_anchor_count': int(segment.get(
                    'sdk_command_anchor_count', 0)),
                'direct_movej_validation': str(segment.get(
                    'direct_movej_validation', '')),
                'direct_movej_source_points': int(segment.get(
                    'direct_movej_source_points', 0)),
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
                'powered_start': {
                    'used': bool(segment.get(
                        'powered_start_recovery_used', False)),
                    'end_point': int(segment.get(
                        'powered_start_recovery_end_point', -1)),
                    'joint_numbers': segment.get(
                        'powered_start_recovery_joints', []),
                    'delta_rad': segment.get(
                        'powered_start_recovery_deltas_rad', []),
                    'minimum_clearance_m': segment.get(
                        'powered_start_recovery_minimum_clearance_m'),
                    'limiting_link_pair': segment.get(
                        'powered_start_recovery_limiting_link_pair', ''),
                    'validation_samples': int(segment.get(
                        'powered_start_recovery_samples', 0)),
                    'start_contacts': segment.get(
                        'powered_start_contacts', []),
                },
            }
            msg.bootstrap_recovery_evidence_json.append(
                json.dumps(evidence, sort_keys=True, separators=(',', ':')))
        provenance = String()
        provenance.data = json.dumps({
            'schema_version': 1,
            'plan_id': str(payload['plan_id']),
            'request_id': str(payload['request_id']),
            'request_sha256': str(payload['request_sha256']),
            'plan_kind': str(payload['plan_kind']),
            'selected_viewpoints': [
                {
                    key: selected[key]
                    for key in (
                        'id',
                        'camera_position_m',
                        'look_direction',
                        'nominal_look_direction',
                        'aim_fallback_used',
                        'aim_offset_deg',
                        'aim_attempt_diagnostics',
                        'view_selection_policy',
                        'view_selection_requested_policy',
                        'view_selection_generation',
                        'view_selection_session_id',
                        'nbv_rank',
                        'nbv_positive_information_gain',
                        'nbv_predicted_unknown_pixels',
                        'nbv_novel_surface_pixels',
                        'nbv_marginal_information_pixels',
                        'nbv_marginal_information_fraction',
                        'nbv_projected_object_pixels',
                        'nbv_direction_novelty_deg',
                        'nbv_camera_travel_m',
                        'coverage_score',
                        'ray_id',
                        'ray_standoff_m',
                        'ray_probe_index',
                        'ray_probe_phase',
                    )
                    if key in selected
                }
                for selected in payload['selected_viewpoints']
            ],
            'candidate_diagnostics': payload.get(
                'planning_diagnostics', {}),
        }, sort_keys=True)
        self.plan_provenance_pub.publish(provenance)
        self.plan_pub.publish(msg)
        if self.param_bool('debug') and payload['selected_viewpoints']:
            selected = payload['selected_viewpoints'][0]
            self.get_logger().info(
                'published plan=%s selected_view=%s policy=%s generation=%s '
                'nbv_rank=%s marginal_fraction=%.4f predicted_unknown=%s '
                'novel_surface=%s'
                % (
                    payload['plan_id'], selected.get('id', ''),
                    selected.get('view_selection_policy', 'legacy'),
                    selected.get('view_selection_generation', 0),
                    selected.get('nbv_rank', 0),
                    selected.get('nbv_marginal_information_fraction', 0.0),
                    selected.get('nbv_predicted_unknown_pixels', 0),
                    selected.get('nbv_novel_surface_pixels', 0)))
        self.set_status('PROPOSAL_READY', msg.reason, payload['request_id'])

    def publish_rejection(
            self, request_id, code, reason, additional_codes=None):
        self.record_ray_rejection(request_id, code, reason)
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
        msg.rejection_codes = [code] + [
            str(value) for value in (additional_codes or [])
            if str(value) != str(code)
        ]
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
