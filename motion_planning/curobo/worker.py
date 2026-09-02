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
    PINNED_COMMIT, PINNED_VERSION, PINNED_WARP_VERSION,
    POSITION_LIMIT_CLIP_RAD)
from motion_planning.curobo.adapter import (
    attach_digest,
    CuroboCandidateExhausted,
    CuroboCollisionRejected,
    CuroboContractError,
    CuroboOutputInvalid,
    CuroboPlanningBudgetExceeded,
    JOINT_NAMES,
    normalize_trajectory,
    obstacle_cuboids,
    prepend_bootstrap_recovery,
    target_ray_standoff_samples,
    trajectory_segment,
    validate_request,
    worker_rejection_code,
)
from motion_planning.curobo.spool import Spool
from piper_tesseract_foxy.protocol.contract import (
    ContractError as PlannerContractError,
    validate_response as validate_planner_response,
)


REQUIRED_FIXED_WORLD_MESHES = {
    'bunker_chassis_collision',
    'bunker_sensor_station_collision',
    'piper_base_collision',
}

SCAN_TARGET_MAX_BORESIGHT_DEG = 20.0
SCAN_TARGET_MIN_DISTANCE_M = 0.22
WORKER_PLANNING_BUDGET_SEC = 150.0
AUTOMATIC_ONE_VIEW_PLANNING_BUDGET_SEC = 90.0
AUTOMATIC_ONE_VIEW_ATTEMPT_BUDGET_SEC = 3.0
WORKER_ATTEMPT_BUDGET_SEC = 5.0
WORKER_RESPONSE_RESERVE_SEC = 5.0
TARGET_RAY_MAX_STANDOFF_SAMPLES = 9
CUROBO_FIXED_GOALSET_SIZE = 54


def planning_budgets_for_request(request):
    """Match the bounded transaction policy used by the Tesseract worker."""
    planning = request.get('planning', {})
    automatic_one_view = bool(
        request.get('plan_kind') == 'MULTIVIEW_SCAN'
        and int(planning.get('min_viewpoints', 0)) == 1
        and int(planning.get('max_viewpoints', 0)) == 1
        and not bool(planning.get('include_return_home', True)))
    if automatic_one_view:
        return (
            AUTOMATIC_ONE_VIEW_PLANNING_BUDGET_SEC,
            AUTOMATIC_ONE_VIEW_ATTEMPT_BUDGET_SEC,
        )
    return WORKER_PLANNING_BUDGET_SEC, WORKER_ATTEMPT_BUDGET_SEC


class BackendUnavailable(RuntimeError):
    """Report a deterministic startup/runtime dependency failure."""

    rejection_code = 'PLANNER_UNAVAILABLE'


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


def _validated_hash_bound_path(record, path_key='path'):
    """Resolve one provenance path and prove that its content is unchanged."""
    path = Path(str(record.get(path_key, ''))).resolve()
    expected_hash = str(record.get('sha256', ''))
    if (
            not path.is_file()
            or len(expected_hash) != 64
            or sha256_file(path) != expected_hash):
        raise ValueError('hash-bound model input is invalid: %s' % path)
    return path


def validate_model_provenance(config_document):
    """Validate generated collision evidence before initializing CUDA.

    Schema 1 remains readable for already generated development models.  A
    schema-2 model fails closed unless its moving-link audit and all canonical
    fixed-world meshes are complete and still match their recorded hashes.
    """
    if not isinstance(config_document, dict):
        raise ValueError('robot model document is not a mapping')
    provenance = config_document.get('piper_curobo_provenance', {})
    if not isinstance(provenance, dict):
        raise ValueError('robot model provenance is not a mapping')
    if (
            provenance.get('curobo_version') != PINNED_VERSION
            or provenance.get('curobo_commit') != PINNED_COMMIT):
        raise ValueError('robot model is not bound to the pinned cuRobo release')
    kinematics = config_document.get('robot_cfg', {}).get('kinematics', {})
    configured_urdf = str(kinematics.get('urdf_path', ''))
    if (
            len(str(provenance.get('source_urdf_sha256', ''))) != 64
            or sha256_file(configured_urdf) !=
            provenance['source_urdf_sha256']):
        raise ValueError('robot model URDF provenance does not match its URDF')

    schema_version = int(provenance.get('schema_version', 1))
    fixed_world_meshes = []
    mesh_names = set()
    for item in provenance.get('fixed_world_meshes', []):
        if not isinstance(item, dict):
            raise ValueError('fixed world mesh entry is not a mapping')
        path = _validated_hash_bound_path(item, 'file_path')
        name = str(item.get('name', path.stem))
        pose = item.get('pose')
        if (
                not name or name in mesh_names
                or not isinstance(pose, list) or len(pose) != 7
                or not all(math.isfinite(float(value)) for value in pose)):
            raise ValueError('fixed world mesh provenance is invalid: %s' % path)
        mesh_names.add(name)
        fixed_world_meshes.append({
            'name': name,
            'file_path': str(path),
            'pose': [float(value) for value in pose],
        })

    if schema_version >= 2:
        if mesh_names != REQUIRED_FIXED_WORLD_MESHES:
            raise ValueError(
                'schema-2 model must contain the canonical PiPER base and '
                'Bunker meshes')
        if provenance.get('conservative_geometry') is not False:
            raise ValueError(
                'schema-2 moving-link sphere approximation must declare '
                'conservative_geometry false')
        if not isinstance(provenance.get('hardware_qualified'), bool):
            raise ValueError('hardware qualification flag is missing')
        for path_key, hash_key in (
                ('source_srdf_path', 'source_srdf_sha256'),
                ('source_collision_manifest_path',
                 'source_collision_manifest_sha256')):
            _validated_hash_bound_path({
                'path': provenance.get(path_key, ''),
                'sha256': provenance.get(hash_key, ''),
            })
        collision_links = set(kinematics.get('collision_link_names') or [])
        sphere_sets = kinematics.get('collision_spheres')
        if not isinstance(sphere_sets, dict) or set(sphere_sets) != collision_links:
            raise ValueError(
                'collision sphere sets do not match collision links')
        sphere_counts = provenance.get('sphere_count_by_link')
        if sphere_counts is not None and (
                not isinstance(sphere_counts, dict)
                or sphere_counts != {
                    name: len(sphere_sets[name]) for name in sorted(sphere_sets)
                }):
            raise ValueError('collision sphere counts do not match the model')
        curated_model = provenance.get('curated_sphere_model')
        if curated_model is not None:
            if not isinstance(curated_model, dict):
                raise ValueError('curated collision-sphere provenance is invalid')
            if sphere_counts is None:
                raise ValueError('curated collision-sphere counts are missing')
            _validated_hash_bound_path(curated_model)
            if len(str(curated_model.get('source_usd_sha256', ''))) != 64:
                raise ValueError('curated collision-sphere USD hash is invalid')
            curated_links = provenance.get('curated_collision_links')
            fallback_links = provenance.get(
                'generated_fallback_collision_links')
            if (
                    not isinstance(curated_links, list)
                    or not isinstance(fallback_links, list)
                    or len(set(curated_links)) != len(curated_links)
                    or len(set(fallback_links)) != len(fallback_links)
                    or set(curated_links) & set(fallback_links)
                    or set(curated_links) | set(fallback_links)
                    != collision_links
                    or sorted(fallback_links) != sorted(curated_model.get(
                        'generated_fallback_links', []))):
                raise ValueError(
                    'curated and fallback collision links are inconsistent')
        coverage = provenance.get('moving_link_surface_coverage')
        if not isinstance(coverage, dict) or set(coverage) != collision_links:
            raise ValueError(
                'moving-link surface audit does not match collision links')
        for name, report in coverage.items():
            if not isinstance(report, dict):
                raise ValueError('surface audit is malformed for %s' % name)
            sample_count = int(report.get('sample_count', -1))
            covered_count = int(report.get('covered_sample_count', -1))
            fraction = float(report.get('covered_fraction', math.nan))
            maximum_gap = float(
                report.get('maximum_uncovered_gap_m', math.nan))
            if (
                    sample_count <= 0 or covered_count < 0
                    or covered_count > sample_count
                    or not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0
                    or not math.isfinite(maximum_gap) or maximum_gap < 0.0):
                raise ValueError('surface audit is invalid for %s' % name)
        mesh_sources = provenance.get('moving_link_mesh_sources')
        if not isinstance(mesh_sources, dict):
            raise ValueError('moving-link mesh sources are missing')
        if not set(mesh_sources).issubset(collision_links):
            raise ValueError('moving-link mesh sources contain an unknown link')
        for name, records in mesh_sources.items():
            if not isinstance(records, list):
                raise ValueError('mesh source list is malformed for %s' % name)
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError('mesh source entry is malformed for %s' % name)
                _validated_hash_bound_path(record)

    cspace = kinematics.get('cspace', {})
    if kinematics.get('link_names') != ['link6']:
        raise ValueError(
            'cuRobo link6 FK output required by direct-home policy is missing')
    try:
        position_limit_clip = tuple(float(value) for value in cspace.get(
            'position_limit_clip'))
        provenance_clip = tuple(float(value) for value in provenance.get(
            'position_limit_clip_rad'))
    except (TypeError, ValueError):
        raise ValueError('cuRobo position-limit clip provenance is missing')
    if (
            len(position_limit_clip) != len(POSITION_LIMIT_CLIP_RAD)
            or len(provenance_clip) != len(POSITION_LIMIT_CLIP_RAD)
            or not all(math.isfinite(value) for value in position_limit_clip)
            or not all(math.isfinite(value) for value in provenance_clip)
            or any(
                abs(actual - expected) > 1e-12
                for actual, expected in zip(
                    position_limit_clip, POSITION_LIMIT_CLIP_RAD))
            or any(
                abs(actual - expected) > 1e-12
                for actual, expected in zip(
                    provenance_clip, POSITION_LIMIT_CLIP_RAD))):
        raise ValueError(
            'cuRobo position-limit clip does not match the qualified policy')

    return {
        'provenance': dict(provenance),
        'collision_manifest_path': str(Path(
            provenance.get('source_collision_manifest_path', '')).resolve()),
        'collision_link_names': list(
            kinematics.get('collision_link_names') or []),
        'fixed_platform_cuboids': list(provenance.get(
            'fixed_world_cuboids', provenance.get(
                'fixed_platform_cuboids', []))),
        'fixed_world_meshes': fixed_world_meshes,
    }


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


def quaternion_optical_z_wxyz(quaternions):
    """Return optical +Z axes for finite normalized WXYZ quaternions."""
    values = np.asarray(quaternions, dtype=float)
    if (
            values.ndim != 2 or values.shape[1] != 4
            or not np.all(np.isfinite(values))):
        raise CuroboContractError(
            'cuRobo camera FK quaternions are malformed')
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 1e-9):
        raise CuroboContractError(
            'cuRobo camera FK contains a degenerate quaternion')
    values = values / norms[:, None]
    w, x, y, z = values.T
    return np.column_stack((
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ))


def quaternion_rotation_matrices_wxyz(quaternions):
    """Return finite normalized WXYZ quaternion rotation matrices."""
    values = np.asarray(quaternions, dtype=float)
    if (
            values.ndim != 2 or values.shape[1] != 4
            or not np.all(np.isfinite(values))):
        raise CuroboContractError('cuRobo link quaternions are malformed')
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 1e-9):
        raise CuroboContractError(
            'cuRobo link FK contains a degenerate quaternion')
    w, x, y, z = (values / norms[:, None]).T
    result = np.empty((len(values), 3, 3), dtype=float)
    result[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    result[:, 0, 1] = 2.0 * (x * y - w * z)
    result[:, 0, 2] = 2.0 * (x * z + w * y)
    result[:, 1, 0] = 2.0 * (x * y + w * z)
    result[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    result[:, 1, 2] = 2.0 * (y * z - w * x)
    result[:, 2, 0] = 2.0 * (x * z - w * y)
    result[:, 2, 1] = 2.0 * (y * z + w * x)
    result[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return result


def camera_path_visibility_rejection(
        camera_positions, camera_forwards, target_center,
        maximum_boresight_deg=SCAN_TARGET_MAX_BORESIGHT_DEG,
        minimum_target_distance_m=SCAN_TARGET_MIN_DISTANCE_M,
        initial_alignment=False, final_aim_deg=5.0):
    """Return why one dense camera path cannot retain the locked target.

    This mirrors the established Tesseract-worker and independent executor
    policy.  Keeping the calculation local to the ROS-free backend lets
    cuRobo discard a bad MotionGen solution and try another candidate before
    publishing it; the executor remains the final authority.
    """
    positions = np.asarray(camera_positions, dtype=float)
    forwards = np.asarray(camera_forwards, dtype=float)
    target = np.asarray(target_center, dtype=float)
    try:
        maximum = float(maximum_boresight_deg)
        minimum = float(minimum_target_distance_m)
        final_aim = float(final_aim_deg)
    except (TypeError, ValueError):
        return 'scan target visibility inputs are not numeric'
    if (
            positions.ndim != 2 or positions.shape[1] != 3
            or forwards.shape != positions.shape or len(positions) == 0
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(forwards))
            or target.shape != (3,) or not np.all(np.isfinite(target))
            or not math.isfinite(maximum) or maximum <= 0.0 or maximum > 90.0
            or not math.isfinite(minimum) or minimum <= 0.0
            or not math.isfinite(final_aim) or final_aim <= 0.0
            or final_aim > 90.0):
        return 'scan target visibility inputs are invalid'
    maximum = max(maximum, final_aim)
    initial_angle = None
    entered_normal_cone = False
    final_angle = None
    for index, (camera, forward) in enumerate(zip(positions, forwards)):
        ray = target - camera
        distance_m = float(np.linalg.norm(ray))
        forward_norm = float(np.linalg.norm(forward))
        if distance_m < minimum:
            return (
                'scan camera approaches target to %.3fm at sample %d; '
                'minimum is %.3fm' % (distance_m, index, minimum))
        if forward_norm <= 1e-9:
            return 'scan camera optical axis is invalid at sample %d' % index
        cosine = float(np.dot(
            forward / forward_norm, ray / distance_m))
        angle_deg = math.degrees(math.acos(float(np.clip(
            cosine, -1.0, 1.0))))
        final_angle = angle_deg
        if initial_angle is None:
            initial_angle = angle_deg
            entered_normal_cone = angle_deg <= maximum + 1e-6
        if not initial_alignment and angle_deg > maximum + 1e-6:
            return (
                'scan target leaves the %.1f-degree camera boresight cone '
                'at sample %d (%.1f degrees)'
                % (maximum, index, angle_deg))
        if initial_alignment:
            if angle_deg > max(maximum, initial_angle) + 1e-6:
                return (
                    'initial target-alignment path worsens beyond its '
                    '%.1f-degree acquired aim at sample %d (%.1f degrees)'
                    % (initial_angle, index, angle_deg))
            if entered_normal_cone and angle_deg > maximum + 1e-6:
                return (
                    'initial target-alignment path leaves the %.1f-degree '
                    'camera boresight cone after entering it at sample %d '
                    '(%.1f degrees)' % (maximum, index, angle_deg))
            if angle_deg <= maximum + 1e-6:
                entered_normal_cone = True
    if initial_alignment and final_angle > final_aim + 1e-6:
        return (
            'initial target-alignment endpoint aim %.1f degrees exceeds the '
            '%.1f-degree settled-capture limit' % (final_angle, final_aim))
    return ''


class CuroboBackend:
    """Thin verified adapter around cuRobo v0.7.8 MotionGen."""

    def __init__(self, robot_config_path, floor_z_m):
        try:
            import torch
            import curobo
            import warp
            import yaml
            from curobo.geom.types import Cuboid, Mesh, WorldConfig
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
        config_path = Path(robot_config_path).resolve()
        if not config_path.is_file():
            raise BackendUnavailable(
                'cuRobo robot configuration is missing: %s' % config_path)
        try:
            config_document = yaml.safe_load(
                config_path.read_text(encoding='utf-8'))
            validated = validate_model_provenance(config_document)
            self.model_provenance = validated['provenance']
            self.collision_link_names = validated['collision_link_names']
            manifest_path = Path(
                validated['collision_manifest_path']).resolve()
            self.collision_manifest = yaml.safe_load(
                manifest_path.read_text(encoding='utf-8'))
            if not isinstance(self.collision_manifest, dict):
                raise ValueError('collision manifest is not a mapping')
            self.robot_config_sha256 = sha256_file(config_path)
            self.fixed_platform_cuboids = validated['fixed_platform_cuboids']
            self.fixed_world_meshes = validated['fixed_world_meshes']
        except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
            raise BackendUnavailable(
                'cuRobo PiPER model provenance is malformed: %s' % error)
        if not torch.cuda.is_available():
            raise BackendUnavailable('PyTorch reports CUDA unavailable')
        try:
            torch.zeros(1, device='cuda')
            torch.cuda.synchronize()
        except Exception as error:
            raise BackendUnavailable('CUDA initialization failed: %s' % error)
        self.torch = torch
        self.Cuboid = Cuboid
        self.Mesh = Mesh
        self.WorldConfig = WorldConfig
        self.Pose = Pose
        self.JointState = JointState
        self.MotionGenPlanConfig = MotionGenPlanConfig
        self.tensor_args = TensorDeviceType()
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

    def _restore_collision_constraints(self):
        """Undo a cuRobo v0.7.8 invalid-start state leak fail-closed.

        ``MotionGen.check_start_state`` disables both constraints while it
        classifies an invalid start.  Its world-collision return path exits
        before re-enabling the self-collision constraint.  This backend is a
        persistent worker, so restore both pinned-v0.7.8 constraints around
        every attempt rather than allowing one rejected request to weaken the
        next request.
        """
        rollout = self.motion_gen.rollout_fn
        rollout.primitive_collision_constraint.enable_cost()
        rollout.robot_self_collision_constraint.enable_cost()

    def _start_state_status(self, positions):
        """Classify one state and contain cuRobo's constraint-state leak."""
        try:
            valid, status = self.motion_gen.check_start_state(
                self._joint_state(positions))
        finally:
            self._restore_collision_constraints()
        return bool(valid), str(status)

    def _bootstrap_recovery(self, start, request):
        """Find the same bounded acquisition-only folded-start escape as Tesseract.

        cuRobo rejects an invalid start before MotionGen runs.  The mission's
        first rough-acquisition request is the one qualified exception: search
        at most 0.15 rad on joint 2 or 3, accept only self-collision status
        before the endpoint, and require the endpoint to satisfy all normal
        cuRobo constraints.  Every later segment starts with normal checks.
        """
        valid, status = self._start_state_status(start)
        if valid:
            return None
        bootstrap_scope = bool(
            request.get('plan_kind') == 'ROUGH_ACQUISITION'
            and request.get('scene', {}).get('observation_mode')
            == 'bootstrap_static')
        if not bootstrap_scope:
            raise CuroboContractError(
                'cuRobo start state is invalid outside bootstrap scope: %s'
                % status)
        if status != 'MotionGenStatus.INVALID_START_STATE_SELF_COLLISION':
            raise CuroboContractError(
                'cuRobo bootstrap start has a non-self-collision failure: %s'
                % status)

        bounds = np.asarray(request['limits']['position_rad'], dtype=float)
        step = 0.01
        maximum_steps = 15
        for step_index in range(1, maximum_steps + 1):
            magnitude = float(step_index) * step
            for joint_index in (1, 2):
                for sign in (-1.0, 1.0):
                    endpoint = list(start)
                    endpoint[joint_index] += sign * magnitude
                    if not (
                            bounds[joint_index, 0] <= endpoint[joint_index]
                            <= bounds[joint_index, 1]):
                        continue
                    endpoint_valid, _endpoint_status = (
                        self._start_state_status(endpoint))
                    if not endpoint_valid:
                        continue
                    recovery_positions = []
                    recovery_valid = True
                    for sample_index in range(step_index + 1):
                        sample = list(start)
                        sample[joint_index] += (
                            sign * step * float(sample_index))
                        sample_valid, sample_status = self._start_state_status(
                            sample)
                        if (
                                not sample_valid
                                and sample_status !=
                                'MotionGenStatus.'
                                'INVALID_START_STATE_SELF_COLLISION'):
                            recovery_valid = False
                            break
                        recovery_positions.append(sample)
                    if recovery_valid:
                        return {
                            'positions': recovery_positions,
                            'joint_numbers': [joint_index + 1],
                            'delta_rad': [sign * magnitude],
                            'start_status': status,
                        }
        raise CuroboContractError(
            'no bounded cuRobo bootstrap recovery reaches a normally valid state')

    def _update_world(self, request):
        world_cuboids = list(self.fixed_platform_cuboids)
        world_cuboids.extend(obstacle_cuboids(request, self.floor_z_m))
        cuboids = [self.Cuboid(**value) for value in world_cuboids]
        meshes = [self.Mesh(**value) for value in self.fixed_world_meshes]
        self.motion_gen.update_world(self.WorldConfig(
            cuboid=cuboids, mesh=meshes))
        self._restore_collision_constraints()

    def _path(self, result, speed, position_limits):
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
            positions.tolist(), float(result.interpolation_dt), speed,
            position_limits=position_limits)

    def _target_visibility_rejection(
            self, points, request, candidate):
        """Densely qualify one cuRobo path before exposing it as feasible."""
        if request.get('plan_kind') != 'MULTIVIEW_SCAN':
            return ''
        positions = np.asarray([
            point['positions_rad'] for point in points
        ], dtype=float)
        if (
                positions.ndim != 2 or positions.shape[1] != 6
                or not np.all(np.isfinite(positions))):
            return 'scan target visibility joint path is invalid'
        try:
            state = self.motion_gen.kinematics.get_state(
                self.tensor_args.to_device(positions.tolist()))
            camera_positions = np.asarray(
                state.ee_position.detach().cpu().numpy(), dtype=float)
            camera_quaternions = np.asarray(
                state.ee_quaternion.detach().cpu().numpy(), dtype=float)
            camera_forwards = quaternion_optical_z_wxyz(
                camera_quaternions)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            return (
                'cuRobo camera FK failed during path qualification: %s'
                % error)
        first_alignment = int(request.get(
            'scan_session', {}).get('accepted_views', -1)) == 0
        final_aim = float(candidate.get(
            'maximum_final_aim_offset_deg', 5.0))
        return camera_path_visibility_rejection(
            camera_positions, camera_forwards,
            request['scene']['target_center_m'],
            initial_alignment=first_alignment,
            final_aim_deg=final_aim)

    def _remaining_planning_time(self, context):
        """Bound each MotionGen call and reserve time to emit a response."""
        deadline = getattr(self, 'planning_deadline_monotonic', None)
        attempt_cap = float(getattr(
            self, 'attempt_planning_budget_sec',
            WORKER_ATTEMPT_BUDGET_SEC))
        if deadline is None:
            return attempt_cap
        remaining = (
            float(deadline) - time.monotonic()
            - WORKER_RESPONSE_RESERVE_SEC)
        if remaining <= 0.0:
            budget = float(getattr(
                self, 'planning_budget_sec', WORKER_PLANNING_BUDGET_SEC))
            raise CuroboPlanningBudgetExceeded(
                'cuRobo planning exceeded the internal %.0f-second budget '
                'before the bridge timeout (%s)' % (budget, context),
                getattr(self, 'last_planning_diagnostics', {}))
        return min(attempt_cap, remaining)

    @staticmethod
    def _selected_view(candidate, direction, roll, camera_position):
        selected = dict(candidate)
        selected['camera_position_m'] = [
            float(value) for value in camera_position]
        selected['look_direction'] = [float(value) for value in direction]
        nominal = candidate['look_direction']
        dot = sum(
            float(left) * float(right)
            for left, right in zip(nominal, direction))
        nominal_norm = math.sqrt(sum(
            float(value) ** 2 for value in nominal))
        selected_norm = math.sqrt(sum(
            float(value) ** 2 for value in direction))
        separation = math.degrees(math.acos(max(
            -1.0, min(1.0, dot / (nominal_norm * selected_norm)))))
        selected['nominal_look_direction'] = list(nominal)
        selected['aim_fallback_used'] = separation > 1e-6
        selected['aim_offset_deg'] = separation
        selected['curobo_roll_rad'] = float(roll)
        if candidate.get('candidate_geometry') == 'target_ray':
            center = candidate['_target_center_m']
            ray = candidate['ray_direction']
            selected['ray_standoff_m'] = float(sum(
                (float(value) - float(origin)) * float(axis)
                for value, origin, axis in zip(
                    camera_position, center, ray)))
        return selected

    @staticmethod
    def _goalset_index(result, goal_count):
        value = getattr(result, 'goalset_index', None)
        if value is None:
            if goal_count == 1:
                return 0
            raise CuroboContractError(
                'cuRobo goal-set result omitted its selected goal index')
        try:
            index = int(value.item())
        except (AttributeError, TypeError, ValueError):
            try:
                index = int(np.asarray(value).reshape(-1)[0])
            except (IndexError, TypeError, ValueError) as error:
                raise CuroboContractError(
                    'cuRobo goal-set index is malformed') from error
        if index < 0 or index >= int(goal_count):
            raise CuroboContractError(
                'cuRobo goal-set index is outside the requested goals')
        return index

    @staticmethod
    def _padded_goalset(goals):
        """Keep cuRobo v0.7.8 on one immutable CUDA goal-set shape.

        v0.7.8 can leave ``Goal.current_state`` unset when a persistent worker
        changes goal-set cardinality.  Duplicating the final valid pose does
        not add a new semantic goal, while a fixed tensor shape avoids that
        upstream solver-buffer defect without touching private cuRobo state.
        """
        values = list(goals)
        if not values or len(values) > CUROBO_FIXED_GOALSET_SIZE:
            raise CuroboContractError(
                'cuRobo semantic goal set must contain 1..%d poses'
                % CUROBO_FIXED_GOALSET_SIZE)
        return values + [values[-1]] * (
            CUROBO_FIXED_GOALSET_SIZE - len(values))

    def _plan_target_ray(self, start, candidate, request):
        """Plan one target ray as bounded MotionGen goal sets.

        Exact target-facing aim is exhausted before the optional fallback aim.
        Within one aim variant, cuRobo chooses among capability-supported
        standoffs and wrist rolls in one GPU call instead of multiplying a
        fixed 30-second timeout across individual poses.
        """
        speed = float(request['planning']['effective_speed_percent'])
        target = np.asarray(request['scene']['target_center_m'], dtype=float)
        ray = np.asarray(candidate['ray_direction'], dtype=float)
        ray_norm = float(np.linalg.norm(ray))
        if target.shape != (3,) or ray.shape != (3,) or ray_norm <= 1e-9:
            raise CuroboContractError('target-ray geometry is invalid')
        ray /= ray_norm
        standoffs = target_ray_standoff_samples(
            candidate, TARGET_RAY_MAX_STANDOFF_SAMPLES)
        rolls = [float(value) for value in request["planning"].get(
            'roll_samples_rad', [0.0])]
        directions = [candidate['look_direction']]
        directions.extend(candidate.get('fallback_look_directions', []))
        diagnostics = []
        total_duration = 0.0
        for direction_index, direction in enumerate(directions):
            goals = []
            for standoff in standoffs:
                camera_position = (target + ray * float(standoff)).tolist()
                for roll in rolls:
                    goals.append({
                        'camera_position_m': camera_position,
                        'look_direction': list(direction),
                        'roll_rad': float(roll),
                        'standoff_m': float(standoff),
                    })
            remaining = list(goals)
            while remaining:
                padded = self._padded_goalset(remaining)
                positions = [
                    item['camera_position_m'] for item in padded]
                quaternions = [camera_quaternion(
                    item['look_direction'], item['roll_rad'])
                    for item in padded]
                goal = self.Pose(
                    position=self.tensor_args.to_device([positions]),
                    quaternion=self.tensor_args.to_device([quaternions]),
                )
                timeout = self._remaining_planning_time(
                    'target ray %s (%s aim, %d goal poses)' % (
                        candidate.get('ray_id', candidate.get('id', -1)),
                        'exact' if direction_index == 0 else 'fallback',
                        len(remaining)))
                try:
                    result = self.motion_gen.plan_goalset(
                        self._joint_state(start), goal,
                        self.MotionGenPlanConfig(
                            enable_graph=True,
                            max_attempts=20,
                            timeout=timeout,
                            fail_on_invalid_query=True,
                        ))
                finally:
                    self._restore_collision_constraints()
                elapsed = float(result.total_time)
                total_duration += elapsed
                attempt = {
                    'aim_variant': (
                        'exact' if direction_index == 0 else 'fallback'),
                    'goalset_size': len(remaining),
                    'native_padded_goalset_size': len(padded),
                    'success': bool(result.success.item()),
                    'status': str(result.status),
                    'planning_duration_sec': elapsed,
                    'timeout_sec': float(timeout),
                }
                diagnostics.append(attempt)
                if not bool(result.success.item()):
                    break
                native_index = self._goalset_index(result, len(padded))
                selected_index = min(native_index, len(remaining) - 1)
                selected_goal = remaining[selected_index]
                attempt.update({
                    'native_selected_goal_index': native_index,
                    'selected_goal_index': selected_index,
                    'selected_standoff_m': selected_goal['standoff_m'],
                    'selected_roll_rad': selected_goal['roll_rad'],
                })
                points = self._path(
                    result, speed, request['limits']['position_rad'])
                qualified_candidate = dict(candidate)
                qualified_candidate['_target_center_m'] = target.tolist()
                qualified_candidate['camera_position_m'] = list(
                    selected_goal['camera_position_m'])
                qualified_candidate['look_direction'] = list(
                    selected_goal['look_direction'])
                visibility_rejection = self._target_visibility_rejection(
                    points, request, qualified_candidate)
                if visibility_rejection:
                    attempt.update({
                        'path_qualification': 'rejected',
                        'path_qualification_reason': visibility_rejection,
                    })
                    remaining.pop(selected_index)
                    continue
                attempt['path_qualification'] = 'accepted'
                response_candidate = dict(candidate)
                response_candidate['_target_center_m'] = target.tolist()
                selected = self._selected_view(
                    response_candidate, selected_goal['look_direction'],
                    selected_goal['roll_rad'],
                    selected_goal['camera_position_m'])
                selected.pop('_target_center_m', None)
                selected['curobo_attempt_diagnostics'] = diagnostics
                selected['curobo_goalset_pose_count'] = len(goals)
                return selected, points, total_duration
        visibility_failures = [
            item['path_qualification_reason'] for item in diagnostics
            if item.get('path_qualification') == 'rejected']
        suffix = (
            ': %s' % visibility_failures[0]
            if visibility_failures else '')
        raise CuroboContractError(
            'cuRobo found no collision-safe target-visible pose along ray %s%s'
            % (candidate.get('ray_id', candidate.get('id', -1)), suffix))

    def _plan_pose(self, start, candidate, request):
        if candidate.get('candidate_geometry') == 'target_ray':
            return self._plan_target_ray(start, candidate, request)
        speed = float(request['planning']['effective_speed_percent'])
        directions = [candidate['look_direction']]
        directions.extend(candidate.get('fallback_look_directions', []))
        diagnostics = []
        total_duration = 0.0
        for direction_index, direction in enumerate(directions):
            remaining = [
                float(value) for value in request['planning'].get(
                    'roll_samples_rad', [0.0])]
            while remaining:
                padded_rolls = self._padded_goalset(remaining)
                positions = [
                    list(candidate['camera_position_m'])
                    for _roll in padded_rolls]
                quaternions = [
                    camera_quaternion(direction, roll)
                    for roll in padded_rolls]
                goal = self.Pose(
                    position=self.tensor_args.to_device(
                        [positions]),
                    quaternion=self.tensor_args.to_device([quaternions]),
                )
                timeout = self._remaining_planning_time(
                    'fixed candidate %s (%s aim, %d roll poses)' % (
                        candidate.get('id', -1),
                        'exact' if direction_index == 0 else 'fallback',
                        len(remaining)))
                try:
                    result = self.motion_gen.plan_goalset(
                        self._joint_state(start), goal,
                        self.MotionGenPlanConfig(
                            enable_graph=True,
                            max_attempts=20,
                            timeout=timeout,
                            fail_on_invalid_query=True,
                        ))
                finally:
                    self._restore_collision_constraints()
                total_duration += float(result.total_time)
                attempt = {
                    'aim_variant': (
                        'exact' if direction_index == 0 else 'fallback'),
                    'goalset_size': len(remaining),
                    'native_padded_goalset_size': len(padded_rolls),
                    'success': bool(result.success.item()),
                    'status': str(result.status),
                    'planning_duration_sec': float(result.total_time),
                    'timeout_sec': float(timeout),
                }
                diagnostics.append(attempt)
                if not bool(result.success.item()):
                    break
                native_index = self._goalset_index(result, len(padded_rolls))
                selected_index = min(native_index, len(remaining) - 1)
                selected_roll = remaining[selected_index]
                attempt.update({
                    'native_selected_goal_index': native_index,
                    'selected_goal_index': selected_index,
                    'selected_roll_rad': selected_roll,
                })
                points = self._path(
                    result, speed, request['limits']['position_rad'])
                visibility_rejection = self._target_visibility_rejection(
                    points, request, candidate)
                if visibility_rejection:
                    attempt.update({
                        'path_qualification': 'rejected',
                        'path_qualification_reason': visibility_rejection,
                    })
                    remaining.pop(selected_index)
                    continue
                attempt['path_qualification'] = 'accepted'
                selected = self._selected_view(
                    candidate, direction, selected_roll,
                    candidate['camera_position_m'])
                selected['curobo_attempt_diagnostics'] = diagnostics
                selected['curobo_goalset_pose_count'] = len(
                    request['planning'].get('roll_samples_rad', [0.0]))
                return selected, points, total_duration
        visibility_failures = [
            item['path_qualification_reason'] for item in diagnostics
            if item.get('path_qualification') == 'rejected']
        suffix = (
            ': %s' % visibility_failures[0]
            if visibility_failures else '')
        raise CuroboContractError(
            'cuRobo found no collision-safe target-visible candidate pose%s'
            % suffix)

    def _configured_home_direct_stage(self, request):
        """Resolve the same request-scoped direct-home exception as Tesseract."""
        policy = self.collision_manifest.get(
            'configured_home_direct_joint_move', {})
        if not bool(policy.get('enabled', False)):
            return None
        if str(policy.get('plan_kind', '')).strip().upper() != 'RETURN_HOME':
            raise CuroboContractError(
                'configured home direct policy has an invalid plan kind')
        if str(request.get('plan_kind', '')).strip().upper() != 'RETURN_HOME':
            return None
        allowed = tuple(
            str(value).strip().upper()
            for value in policy.get('allowed_home_stages', []))
        supported = {
            'CONFIGURED_HOME', 'STARTUP_WRIST', 'ROUGH_HOME', 'STORAGE_WRIST'}
        if (
                not allowed or len(set(allowed)) != len(allowed)
                or any(value not in supported for value in allowed)):
            raise CuroboContractError(
                'configured home direct stages are invalid')
        stage = str(request.get('planning', {}).get(
            'home_stage', '') or 'CONFIGURED_HOME').strip().upper()
        if stage not in allowed:
            raise CuroboContractError(
                'configured home direct joint move is not authorized for %s'
                % stage)
        return stage

    def _external_floor_validation(self, positions):
        """Densely validate the manifest's link6-mounted holder against floor."""
        policy = self.collision_manifest.get('external_floor_clearance', {})
        if not bool(policy.get('enabled', False)):
            raise CuroboContractError(
                'configured home requires external-floor clearance policy')
        floor = float(policy.get('floor_z_m'))
        clearance = float(policy.get('clearance_m'))
        origin = np.asarray(policy.get('origin_link6_m'), dtype=float)
        size = np.asarray(policy.get('size_m'), dtype=float)
        if (
                not math.isfinite(floor) or not math.isfinite(clearance)
                or abs(floor - self.floor_z_m) > 1e-9
                or clearance < 0.0 or origin.shape != (3,)
                or size.shape != (3,) or np.any(size <= 0.0)
                or not np.all(np.isfinite(origin))
                or not np.all(np.isfinite(size))):
            raise CuroboContractError(
                'configured home external-floor policy is invalid')
        threshold = floor + clearance
        samples = np.asarray(positions, dtype=float)
        if (
                samples.ndim != 2 or samples.shape[1] != 6
                or not np.all(np.isfinite(samples))):
            raise CuroboContractError(
                'configured home floor-validation samples are invalid')
        chunk_size = 2048
        minimum_z = math.inf
        for offset in range(0, len(samples), chunk_size):
            chunk = samples[offset:offset + chunk_size]
            pose = self.motion_gen.kinematics.get_link_poses(
                self.tensor_args.to_device(chunk.tolist()), ['link6'])
            link_positions = np.asarray(
                pose.position.detach().cpu().numpy(), dtype=float).reshape(
                    -1, 3)
            link_quaternions = np.asarray(
                pose.quaternion.detach().cpu().numpy(), dtype=float).reshape(
                    -1, 4)
            rotations = quaternion_rotation_matrices_wxyz(link_quaternions)
            centers = link_positions + np.einsum(
                'nij,j->ni', rotations, origin)
            half_extent_z = np.sum(
                np.abs(rotations[:, 2, :]) * (size * 0.5), axis=1)
            lowest = centers[:, 2] - half_extent_z
            local_index = int(np.argmin(lowest))
            local_minimum = float(lowest[local_index])
            minimum_z = min(minimum_z, local_minimum)
            if local_minimum < threshold - 1e-9:
                raise CuroboCollisionRejected(
                    'configured home external-floor clearance failed at '
                    'sample %d: %.6fm is below %.6fm' % (
                        offset + local_index, local_minimum, threshold),
                    getattr(self, 'last_planning_diagnostics', {}))
        return minimum_z

    def _plan_configured_home_direct(self, request, start, goal):
        """Build the common, direct SDK home target with external-floor proof."""
        stage = self._configured_home_direct_stage(request)
        if stage is None:
            raise CuroboContractError(
                'configured home direct joint move is disabled')
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        bounds = np.asarray(request['limits']['position_rad'], dtype=float)
        if (
                start.shape != (6,) or goal.shape != (6,)
                or bounds.shape != (6, 2)
                or not np.all(np.isfinite(start))
                or not np.all(np.isfinite(goal))
                or not np.all(np.isfinite(bounds))):
            raise CuroboContractError(
                'configured home direct joint data are invalid')
        policy = self.collision_manifest[
            'configured_home_direct_joint_move']
        maximum_start_violation = float(
            policy.get('maximum_start_limit_violation_rad', -1.0))
        requested_tolerance = float(request['limits'].get(
            'configured_home_start_limit_tolerance_rad', 0.0))
        allowed_start_joints = [int(value) for value in policy.get(
            'allowed_start_limit_joints', [])]
        if (
                not math.isfinite(maximum_start_violation)
                or maximum_start_violation < 0.0
                or maximum_start_violation > 0.3
                or abs(requested_tolerance - maximum_start_violation) > 1e-12
                or allowed_start_joints != [1, 2, 3, 4, 5, 6]):
            raise CuroboContractError(
                'configured home direct start-limit policy is invalid')
        for index, position in enumerate(start):
            low, high = bounds[index]
            violation = max(low - position, position - high, 0.0)
            if (
                    violation > maximum_start_violation + 1e-12
                    or (violation > 0.0
                        and index + 1 not in allowed_start_joints)):
                raise CuroboContractError(
                    'configured home direct start exceeds a position limit')
        if np.any(goal < bounds[:, 0]) or np.any(goal > bounds[:, 1]):
            raise CuroboContractError(
                'configured home direct goal exceeds a position limit')
        maximum_l1 = float(self.collision_manifest.get(
            'validation_max_joint_l1_step_rad', -1.0))
        if not math.isfinite(maximum_l1) or maximum_l1 <= 0.0:
            raise CuroboContractError(
                'configured home floor-validation step is invalid')
        sample_count = max(
            1, int(math.ceil(float(np.sum(np.abs(goal - start)))
                             / maximum_l1)))
        samples = np.linspace(start, goal, sample_count + 1)
        minimum_floor_z = self._external_floor_validation(samples)
        rate = float(request['planning']['command_rate_hz'])
        if not math.isfinite(rate) or rate <= 0.0:
            raise CuroboContractError(
                'configured home command rate is invalid')
        points = [{
            'time_from_start_s': round(index * (1.0 / rate), 9),
            'positions_rad': position.tolist(),
            'velocities_rad_s': [0.0] * 6,
            'accelerations_rad_s2': [0.0] * 6,
        } for index, position in enumerate((start, goal))]
        segment = trajectory_segment(points, is_return_home=True)
        segment.update({
            'minimum_clearance_m': -1.0,
            'limiting_link_pair': 'not_evaluated/configured_home_direct',
            'validation': 'configured_home_collision_validation_bypassed',
            'collision_validation_bypassed': True,
            'configured_home_direct_joint_move': True,
            'configured_home_goal_positions_rad': goal.tolist(),
            'home_stage': stage,
            'validation_samples': 0,
            'external_floor_validation': 'cad_holder_aabb_dense_discrete',
            'external_floor_validation_samples': len(samples),
            'external_floor_minimum_z_m': float(minimum_floor_z),
            'sdk_execution_mode': 'DIRECT_MOVEJ',
            'sdk_command_anchor_count': 1,
        })
        segment['startup_home_static'] = bool(
            request['scene'].get('startup_home_static', False))
        return segment

    def _plan_joint_goal(self, start, goal, request):
        try:
            result = self.motion_gen.plan_single_js(
                self._joint_state(start), self._joint_state(goal),
                self.MotionGenPlanConfig(
                    enable_graph=True, max_attempts=20,
                    timeout=self._remaining_planning_time('joint goal'),
                    fail_on_invalid_query=True))
        finally:
            self._restore_collision_constraints()
        return (
            self._path(
                result, request['planning']['effective_speed_percent'],
                request['limits']['position_rad']),
            float(result.total_time),
        )

    def plan(self, request):
        """Plan selected camera poses and optional home through one backend."""
        planning_budget, attempt_budget = planning_budgets_for_request(request)
        self.planning_budget_sec = planning_budget
        self.attempt_planning_budget_sec = attempt_budget
        self.planning_deadline_monotonic = None
        self.last_planning_diagnostics = {
            'shortlisted_candidates': len(
                request.get('scene', {}).get('candidate_views', [])),
            'shortlisted_rays': int(request.get(
                'planning', {}).get('shortlisted_ray_count', 0)),
            'candidate_attempts': 0,
            'candidate_failures': [],
            'attempted_ray_ids': [],
            'planning_budget_sec': float(planning_budget),
            'attempt_planning_budget_sec': float(attempt_budget),
        }
        self._update_world(request)
        self.planning_deadline_monotonic = time.monotonic() + planning_budget
        start = list(request['start_state']['positions_rad'])
        selected = []
        segments = []
        duration = 0.0
        kind = request['plan_kind']
        if kind == 'RETURN_HOME':
            goal = request['planning'].get('return_home_positions_rad', [])
            if len(goal) != 6:
                raise CuroboContractError(
                    'RETURN_HOME joint target is missing')
            segment = self._plan_configured_home_direct(
                request, start, goal)
            self.last_planning_diagnostics.update({
                'planning_duration_sec': 0.0,
                'candidate_viewpoints_considered': 0,
                'candidate_viewpoints_rejected': 0,
                'feasible_viewpoints': 0,
                'successful_viewpoints': 0,
                'return_home_policy':
                    'configured_home_direct_common_policy',
            })
            return [], [segment], 0.0
        bootstrap_recovery = self._bootstrap_recovery(start, request)
        if bootstrap_recovery is not None:
            start = list(bootstrap_recovery['positions'][-1])
        minimum = int(request['planning'].get('min_viewpoints', 1))
        maximum = int(request['planning'].get('max_viewpoints', minimum))
        candidate_failures = []
        if kind != 'RETURN_HOME':
            for candidate in request['scene']['candidate_views']:
                if len(selected) >= maximum:
                    break
                self._remaining_planning_time(
                    'candidate %s admission' % candidate.get('id', -1))
                self.last_planning_diagnostics['candidate_attempts'] += 1
                ray_id = candidate.get('ray_id')
                if ray_id is not None:
                    attempted = self.last_planning_diagnostics[
                        'attempted_ray_ids']
                    if int(ray_id) not in attempted:
                        attempted.append(int(ray_id))
                try:
                    view, points, elapsed = self._plan_pose(
                        start, candidate, request)
                except CuroboPlanningBudgetExceeded:
                    raise
                except CuroboContractError as error:
                    failure = {
                        'id': int(candidate.get('id', -1)),
                        'reason': str(error),
                    }
                    if ray_id is not None:
                        failure['ray_id'] = int(ray_id)
                    candidate_failures.append(failure)
                    self.last_planning_diagnostics[
                        'candidate_failures'].append(dict(failure))
                    continue
                segment_recovery = None
                if bootstrap_recovery is not None:
                    points, recovery_end = prepend_bootstrap_recovery(
                        points,
                        bootstrap_recovery['positions'],
                        request['planning']['effective_speed_percent'],
                    )
                    segment_recovery = dict(bootstrap_recovery)
                    segment_recovery['end_point'] = recovery_end
                selected.append(view)
                segments.append(trajectory_segment(
                    points, bootstrap_recovery=segment_recovery))
                bootstrap_recovery = None
                start = list(points[-1]['positions_rad'])
                duration += elapsed
            if len(selected) < minimum:
                detail = (
                    '; first candidate failure: view %d: %s' % (
                        candidate_failures[0]['id'],
                        candidate_failures[0]['reason'])
                    if candidate_failures else '')
                raise CuroboCandidateExhausted(
                    'cuRobo found %d feasible viewpoints; need %d%s'
                    % (len(selected), minimum, detail),
                    self.last_planning_diagnostics)
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
        self.last_planning_diagnostics.update({
            'planning_duration_sec': float(duration),
            'candidate_viewpoints_considered': int(
                self.last_planning_diagnostics['candidate_attempts']),
            'candidate_viewpoints_rejected': len(candidate_failures),
            'feasible_viewpoints': len(selected),
            'successful_viewpoints': len(selected),
        })
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
        """Return the compact, frequently refreshed readiness record.

        Full model provenance is intentionally published separately because
        worker_health.json is a bounded liveness contract, not a diagnostic
        transport.
        """
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
            'environment': (
                dict(self.backend.environment)
                if self.backend is not None else {}),
            **self.model_hashes,
        }

    def diagnostics(self):
        """Return full startup and collision-model provenance for inspection."""
        return {
            'schema_version': 1,
            'generation_id': self.generation_id,
            'written_at_ns': time.time_ns(),
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

    def publish_diagnostics(self):
        self.spool.write_diagnostics(self.diagnostics())

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
            planning_diagnostics = dict(getattr(
                self.backend, 'last_planning_diagnostics', {}))
            planning_diagnostics.update({
                'planning_duration_sec': duration,
                'candidate_viewpoints_considered': int(
                    planning_diagnostics.get(
                        'candidate_attempts', len(request[
                            'scene'].get('candidate_views', [])))),
                'candidate_viewpoints_rejected': int(
                    planning_diagnostics.get(
                        'candidate_viewpoints_rejected', max(
                            0, len(request['scene'].get(
                                'candidate_views', [])) - len(selected)))),
                'feasible_viewpoints': len(selected),
                'successful_viewpoints': len(selected),
            })
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
                'planning_diagnostics': planning_diagnostics,
            }
            # A cuRobo proposal is not successful until it satisfies the same
            # complete generic transport/execution contract used by the ROS
            # bridge.  Convert violations into a structured worker failure so
            # malformed GPU output never crosses the backend boundary.
            try:
                validate_planner_response(
                    attach_digest(response, 'response_sha256'), request)
            except PlannerContractError as error:
                raise CuroboOutputInvalid(
                    'cuRobo normalized output failed the generic contract: %s'
                    % error, planning_diagnostics)
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
                'planning_diagnostics': dict(getattr(
                    error, 'planning_diagnostics', getattr(
                        self.backend, 'last_planning_diagnostics', {}))),
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
        self.publish_diagnostics()
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
