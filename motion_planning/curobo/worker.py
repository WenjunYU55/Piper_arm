#!/usr/bin/env python3
"""ROS-free cuRobo v0.7.8 worker for generic planner spool requests."""

import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

import numpy as np

from motion_planning.curobo import (
    PINNED_COMMIT, PINNED_VERSION, PINNED_WARP_VERSION)
from motion_planning.curobo.adapter import (
    attach_digest,
    CuroboContractError,
    JOINT_NAMES,
    normalize_trajectory,
    obstacle_cuboids,
    trajectory_segment,
    validate_request,
    worker_rejection_code,
)
from motion_planning.curobo.spool import Spool


class BackendUnavailable(RuntimeError):
    """Report a deterministic startup/runtime dependency failure."""


def nvidia_driver_version():
    """Read the installed driver without importing an additional library."""
    try:
        completed = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=3.0)
        return completed.stdout.splitlines()[0].strip()
    except (IndexError, OSError, subprocess.SubprocessError):
        return 'unavailable'


def sha256_file(path):
    """Hash one required model input."""
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def matrix_quaternion_wxyz(matrix):
    """Convert a proper 3x3 rotation matrix to cuRobo's WXYZ quaternion."""
    value = np.asarray(matrix, dtype=float)
    trace = float(np.trace(value))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            0.25 * scale,
            (value[2, 1] - value[1, 2]) / scale,
            (value[0, 2] - value[2, 0]) / scale,
            (value[1, 0] - value[0, 1]) / scale,
        )
    else:
        axis = int(np.argmax(np.diag(value)))
        if axis == 0:
            scale = math.sqrt(
                1.0 + value[0, 0] - value[1, 1] - value[2, 2]) * 2.0
            quaternion = (
                (value[2, 1] - value[1, 2]) / scale,
                0.25 * scale,
                (value[0, 1] + value[1, 0]) / scale,
                (value[0, 2] + value[2, 0]) / scale,
            )
        elif axis == 1:
            scale = math.sqrt(
                1.0 + value[1, 1] - value[0, 0] - value[2, 2]) * 2.0
            quaternion = (
                (value[0, 2] - value[2, 0]) / scale,
                (value[0, 1] + value[1, 0]) / scale,
                0.25 * scale,
                (value[1, 2] + value[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(
                1.0 + value[2, 2] - value[0, 0] - value[1, 1]) * 2.0
            quaternion = (
                (value[1, 0] - value[0, 1]) / scale,
                (value[0, 2] + value[2, 0]) / scale,
                (value[1, 2] + value[2, 1]) / scale,
                0.25 * scale,
            )
    norm = math.sqrt(sum(item * item for item in quaternion))
    if norm <= 1e-12 or not math.isfinite(norm):
        raise CuroboContractError('camera orientation is degenerate')
    return tuple(item / norm for item in quaternion)


def camera_quaternion(look_direction, roll_rad):
    """Construct an optical-frame pose whose +Z axis aims at the target."""
    optical_z = np.asarray(look_direction, dtype=float)
    optical_z /= np.linalg.norm(optical_z)
    reference = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(optical_z, reference))) > 0.95:
        reference = np.asarray([0.0, 1.0, 0.0])
    optical_x = np.cross(optical_z, reference)
    optical_x /= np.linalg.norm(optical_x)
    optical_y = np.cross(optical_z, optical_x)
    cosine = math.cos(float(roll_rad))
    sine = math.sin(float(roll_rad))
    rolled_x = cosine * optical_x + sine * optical_y
    rolled_y = -sine * optical_x + cosine * optical_y
    return matrix_quaternion_wxyz(np.column_stack(
        (rolled_x, rolled_y, optical_z)))


class CuroboBackend:
    """Thin verified adapter around cuRobo v0.7.8 MotionGen."""

    def __init__(self, robot_config_path, floor_z_m):
        try:
            import torch
            import curobo
            import warp
            import yaml
            from curobo.geom.types import Cuboid, WorldConfig
            from curobo.types.base import TensorDeviceType
            from curobo.types.math import Pose
            from curobo.types.state import JointState
            from curobo.wrap.reacher.motion_gen import (
                MotionGen, MotionGenConfig, MotionGenPlanConfig)
        except Exception as error:
            raise BackendUnavailable(
                'cuRobo v%s imports failed: %s' % (PINNED_VERSION, error))
        version = str(getattr(curobo, '__version__', ''))
        if version and not version.startswith(PINNED_VERSION):
            raise BackendUnavailable(
                'cuRobo version %s does not match pinned %s'
                % (version, PINNED_VERSION))
        warp_version = str(getattr(warp, '__version__', ''))
        if warp_version != PINNED_WARP_VERSION:
            raise BackendUnavailable(
                'Warp version %s does not match pinned %s'
                % (warp_version or 'unknown', PINNED_WARP_VERSION))
        if not torch.cuda.is_available():
            raise BackendUnavailable('PyTorch reports CUDA unavailable')
        try:
            torch.zeros(1, device='cuda')
            torch.cuda.synchronize()
        except Exception as error:
            raise BackendUnavailable('CUDA initialization failed: %s' % error)
        config_path = Path(robot_config_path).resolve()
        if not config_path.is_file():
            raise BackendUnavailable(
                'cuRobo robot configuration is missing: %s' % config_path)
        self.torch = torch
        self.Cuboid = Cuboid
        self.WorldConfig = WorldConfig
        self.Pose = Pose
        self.JointState = JointState
        self.MotionGenPlanConfig = MotionGenPlanConfig
        self.tensor_args = TensorDeviceType()
        try:
            config_document = yaml.safe_load(
                config_path.read_text(encoding='utf-8'))
            provenance = config_document.get(
                'piper_curobo_provenance', {})
            if (
                    provenance.get('curobo_version') != PINNED_VERSION
                    or provenance.get('curobo_commit') != PINNED_COMMIT):
                raise ValueError(
                    'robot model is not bound to the pinned cuRobo release')
            configured_urdf = config_document.get(
                'robot_cfg', {}).get('kinematics', {}).get('urdf_path', '')
            if (
                    len(str(provenance.get('source_urdf_sha256', ''))) != 64
                    or sha256_file(configured_urdf) !=
                    provenance['source_urdf_sha256']):
                raise ValueError(
                    'robot model URDF provenance does not match its URDF')
            self.model_provenance = dict(provenance)
            self.robot_config_sha256 = sha256_file(config_path)
            self.fixed_platform_cuboids = list(provenance.get(
                'fixed_world_cuboids', provenance.get(
                    'fixed_platform_cuboids', [])))
        except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
            raise BackendUnavailable(
                'cuRobo PiPER model provenance is malformed: %s' % error)
        try:
            config = MotionGenConfig.load_from_robot_config(
                str(config_path),
                WorldConfig(cuboid=[]),
                tensor_args=self.tensor_args,
                interpolation_dt=0.05,
                collision_cache={'obb': 320, 'mesh': 64},
                evaluate_interpolated_trajectory=True,
            )
            self.motion_gen = MotionGen(config)
            self.motion_gen.warmup(enable_graph=True)
        except Exception as error:
            raise BackendUnavailable(
                'cuRobo PiPER model initialization failed: %s' % error)
        self.version = version or PINNED_VERSION
        self.floor_z_m = float(floor_z_m)
        self.environment = {
            'python_version': sys.version.split()[0],
            'torch_version': str(torch.__version__),
            'torch_cuda_version': str(torch.version.cuda),
            'cuda_available': bool(torch.cuda.is_available()),
            'gpu_name': str(torch.cuda.get_device_name(0)),
            'gpu_memory_bytes': int(
                torch.cuda.get_device_properties(0).total_memory),
            'nvidia_driver': nvidia_driver_version(),
            'cudnn_version': int(torch.backends.cudnn.version() or 0),
            'curobo_version': self.version,
            'curobo_commit': PINNED_COMMIT,
            'warp_version': warp_version,
        }

    def _joint_state(self, positions):
        tensor = self.tensor_args.to_device([list(positions)])
        return self.JointState.from_position(
            tensor, joint_names=list(JOINT_NAMES))

    def _update_world(self, request):
        world_cuboids = list(self.fixed_platform_cuboids)
        world_cuboids.extend(obstacle_cuboids(request, self.floor_z_m))
        cuboids = [self.Cuboid(**value) for value in world_cuboids]
        self.motion_gen.update_world(self.WorldConfig(cuboid=cuboids))

    def _path(self, result, speed):
        if not bool(result.success.item()):
            raise CuroboContractError(
                'cuRobo planning failed: %s' % result.status)
        path = result.get_interpolated_plan()
        positions = path.position.detach().cpu().numpy()
        positions = np.asarray(positions, dtype=float).squeeze()
        if positions.ndim != 2 or positions.shape[1] != 6:
            raise CuroboContractError(
                'cuRobo returned a malformed trajectory shape')
        native_names = tuple(getattr(path, 'joint_names', ()) or ())
        if set(native_names) != set(JOINT_NAMES) or len(native_names) != 6:
            raise CuroboContractError(
                'cuRobo trajectory joint names do not match PiPER')
        positions = positions[:, [
            native_names.index(name) for name in JOINT_NAMES]]
        return normalize_trajectory(
            positions.tolist(), float(result.interpolation_dt), speed)

    def _plan_pose(self, start, candidate, request):
        speed = float(request['planning']['effective_speed_percent'])
        directions = [candidate['look_direction']]
        directions.extend(candidate.get('fallback_look_directions', []))
        diagnostics = []
        for direction in directions:
            for roll in request['planning'].get('roll_samples_rad', [0.0]):
                quaternion = camera_quaternion(direction, roll)
                goal = self.Pose(
                    position=self.tensor_args.to_device(
                        [candidate['camera_position_m']]),
                    quaternion=self.tensor_args.to_device([quaternion]),
                )
                result = self.motion_gen.plan_single(
                    self._joint_state(start), goal,
                    self.MotionGenPlanConfig(
                        enable_graph=True,
                        max_attempts=20,
                        timeout=30.0,
                        fail_on_invalid_query=True,
                    ))
                diagnostics.append({
                    'roll_rad': float(roll),
                    'success': bool(result.success.item()),
                    'status': str(result.status),
                    'planning_duration_sec': float(result.total_time),
                })
                if bool(result.success.item()):
                    points = self._path(result, speed)
                    selected = dict(candidate)
                    selected['look_direction'] = list(direction)
                    nominal = candidate['look_direction']
                    dot = sum(
                        float(left) * float(right)
                        for left, right in zip(nominal, direction))
                    nominal_norm = math.sqrt(sum(
                        float(value) ** 2 for value in nominal))
                    selected_norm = math.sqrt(sum(
                        float(value) ** 2 for value in direction))
                    separation = math.degrees(math.acos(max(
                        -1.0, min(1.0, dot / (
                            nominal_norm * selected_norm)))))
                    selected['nominal_look_direction'] = list(nominal)
                    selected['aim_fallback_used'] = separation > 1e-6
                    selected['aim_offset_deg'] = separation
                    selected['curobo_roll_rad'] = float(roll)
                    selected['curobo_attempt_diagnostics'] = diagnostics
                    return selected, points, float(result.total_time)
        raise CuroboContractError('cuRobo found no collision-safe candidate pose')

    def _plan_joint_goal(self, start, goal, request):
        result = self.motion_gen.plan_single_js(
            self._joint_state(start), self._joint_state(goal),
            self.MotionGenPlanConfig(
                enable_graph=True, max_attempts=20, timeout=30.0,
                fail_on_invalid_query=True))
        return (
            self._path(
                result, request['planning']['effective_speed_percent']),
            float(result.total_time),
        )

    def plan(self, request):
        """Plan selected camera poses and optional home through one backend."""
        self._update_world(request)
        start = list(request['start_state']['positions_rad'])
        selected = []
        segments = []
        duration = 0.0
        kind = request['plan_kind']
        minimum = int(request['planning'].get('min_viewpoints', 1))
        maximum = int(request['planning'].get('max_viewpoints', minimum))
        if kind != 'RETURN_HOME':
            for candidate in request['scene']['candidate_views']:
                if len(selected) >= maximum:
                    break
                try:
                    view, points, elapsed = self._plan_pose(
                        start, candidate, request)
                except CuroboContractError:
                    continue
                selected.append(view)
                segments.append(trajectory_segment(points))
                start = list(points[-1]['positions_rad'])
                duration += elapsed
            if len(selected) < minimum:
                raise CuroboContractError(
                    'cuRobo found %d feasible viewpoints; need %d'
                    % (len(selected), minimum))
        if bool(request['planning'].get('include_return_home', False)):
            goal = request['planning'].get('return_home_positions_rad', [])
            if len(goal) != 6:
                raise CuroboContractError(
                    'RETURN_HOME joint target is missing')
            points, elapsed = self._plan_joint_goal(start, goal, request)
            home = trajectory_segment(points, is_return_home=True)
            home['startup_home_static'] = bool(
                request['scene'].get('startup_home_static', False))
            segments.append(home)
            duration += elapsed
        return selected, segments, duration


class Worker:
    """Own worker readiness, atomic requests, and bounded shutdown."""

    def __init__(self):
        self.spool = Spool(os.environ.get(
            'PIPER_CUROBO_SPOOL', '/tmp/piper_curobo_plans'))
        self.generation_id = uuid.uuid4().hex
        self.running = True
        self.backend = None
        self.backend_error = ''
        self.heartbeat_stop = threading.Event()
        self.model_paths = {
            'srdf_sha256': os.environ.get('PIPER_CUROBO_SRDF', ''),
            'collision_manifest_sha256': os.environ.get(
                'PIPER_CUROBO_COLLISION_MANIFEST', ''),
        }
        self.model_hashes = {}
        for field, path in self.model_paths.items():
            try:
                self.model_hashes[field] = sha256_file(path)
            except OSError:
                self.model_hashes[field] = ''
        try:
            self.backend = CuroboBackend(
                os.environ.get('PIPER_CUROBO_ROBOT_CONFIG', ''),
                os.environ.get('PIPER_CUROBO_FLOOR_Z_M', '0.005'))
            expected_model_hashes = {
                'srdf_sha256': 'source_srdf_sha256',
                'collision_manifest_sha256':
                    'source_collision_manifest_sha256',
            }
            for health_field, provenance_field in (
                    expected_model_hashes.items()):
                if self.model_hashes.get(health_field) != (
                        self.backend.model_provenance.get(provenance_field)):
                    raise BackendUnavailable(
                        'cuRobo model %s does not match the authoritative asset'
                        % provenance_field)
        except (BackendUnavailable, OSError, RuntimeError, ValueError) as error:
            self.backend_error = str(error)
            self.backend = None

    def collision_model_qualified(self):
        """Require model provenance and an explicit operator opt-in."""
        return bool(
            self.backend is not None
            and self.backend.model_provenance.get('hardware_qualified') is True
            and os.environ.get(
                'PIPER_CUROBO_COLLISION_MODEL_QUALIFIED', '0') == '1')

    def stop(self, *_args):
        self.running = False
        self.heartbeat_stop.set()

    def health(self):
        return {
            'schema_version': 5,
            'generation_id': self.generation_id,
            'written_at_ns': time.time_ns(),
            'worker_ready': self.backend is not None,
            'backend': 'curobo',
            'backend_version': (
                self.backend.version if self.backend is not None else 'unavailable'),
            'backend_error': self.backend_error,
            'collision_model_qualified': self.collision_model_qualified(),
            'robot_config_sha256': (
                self.backend.robot_config_sha256
                if self.backend is not None else ''),
            'model_provenance': (
                dict(self.backend.model_provenance)
                if self.backend is not None else {}),
            'environment': (
                dict(self.backend.environment)
                if self.backend is not None else {}),
            **self.model_hashes,
        }

    def publish_health(self):
        self.spool.write_health(self.health())

    def heartbeat_loop(self):
        while not self.heartbeat_stop.wait(0.5):
            try:
                self.publish_health()
            except (OSError, TypeError, ValueError):
                pass

    @staticmethod
    def binding(request):
        return {
            'request_sha256': request['request_sha256'],
            'plan_kind': request['plan_kind'],
            'target_provenance': request['target_provenance'],
            'model': request['model'],
            'calibration': request['calibration']['hand_eye_sha256'],
            'limits': request['limits'],
            'execution': {
                'effective_speed_percent':
                    request['planning']['effective_speed_percent'],
                'command_rate_hz': request['planning']['command_rate_hz'],
                # Private compatibility protocol. The ROS adapter normalizes
                # this to timed_stream_v1 before common execution.
                'timing_policy': request['planning']['timing_policy'],
            },
        }

    def process_once(self):
        request_id, request = self.spool.claim_next()
        if request_id is None:
            return False
        try:
            validate_request(request)
            if self.backend is None:
                raise BackendUnavailable(
                    self.backend_error or 'cuRobo backend unavailable')
            selected, segments, duration = self.backend.plan(request)
            binding = self.binding(request)
            response = {
                'schema_version': 5,
                'plan_kind': request['plan_kind'],
                'target_provenance': request['target_provenance'],
                'request_id': request_id,
                'request_sha256': request['request_sha256'],
                'plan_id': request_id,
                'status': 'success',
                'backend': 'curobo',
                'backend_version': self.backend.version,
                'deterministic_seed': request['planning']['deterministic_seed'],
                'joint_names': list(JOINT_NAMES),
                'collision_model_qualified':
                    self.collision_model_qualified(),
                'diagnostic': 'GPU cuRobo v%s collision-checked proposal'
                % self.backend.version,
                'rejection_codes': [],
                'target_center_m': request['scene']['target_center_m'],
                'selected_viewpoints': selected,
                'segments': segments,
                'trajectory_binding': binding,
                'trajectory_sha256': hashlib.sha256(json.dumps({
                    'joint_names': list(JOINT_NAMES),
                    'segments': segments,
                    'binding': binding,
                }, sort_keys=True, separators=(',', ':')).encode(
                    'utf-8')).hexdigest(),
                'planning_diagnostics': {
                    'planning_duration_sec': duration,
                    'candidate_viewpoints_considered': len(
                        request['scene'].get('candidate_views', [])),
                    'candidate_viewpoints_rejected': max(
                        0, len(request['scene'].get('candidate_views', []))
                        - len(selected)),
                    'feasible_viewpoints': len(selected),
                    'successful_viewpoints': len(selected),
                },
            }
        except (
                BackendUnavailable, CuroboContractError, KeyError, OSError,
                RuntimeError, TypeError, ValueError) as error:
            response = {
                'schema_version': 5,
                'plan_kind': str(request.get('plan_kind', '')),
                'target_provenance': request.get('target_provenance', {}),
                'request_id': request_id,
                'request_sha256': str(request.get('request_sha256', '')),
                'status': 'failed',
                'backend': 'curobo',
                'backend_version': (
                    self.backend.version
                    if self.backend is not None else 'unavailable'),
                'rejection_codes': [worker_rejection_code(error)],
                'diagnostic': str(error),
                'planning_diagnostics': {},
            }
        self.spool.write_response(
            request_id, attach_digest(response, 'response_sha256'))
        processing = self.spool.path('processing', request_id)
        try:
            processing.unlink()
        except FileNotFoundError:
            pass
        return True

    def run(self, once=False):
        self.publish_health()
        heartbeat = threading.Thread(
            target=self.heartbeat_loop, name='curobo-worker-heartbeat',
            daemon=True)
        heartbeat.start()
        try:
            while self.running:
                processed = self.process_once()
                if once:
                    return 0 if processed else 2
                if not processed:
                    time.sleep(0.05)
        finally:
            self.heartbeat_stop.set()
            heartbeat.join(timeout=2.0)
            self.publish_health()
        return 0


def main():
    worker = Worker()
    signal.signal(signal.SIGINT, worker.stop)
    signal.signal(signal.SIGTERM, worker.stop)
    return worker.run(once='--once' in sys.argv[1:])


if __name__ == '__main__':
    raise SystemExit(main())
