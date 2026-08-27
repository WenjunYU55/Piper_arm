"""Offline collision-qualified camera capability-map generator."""

import argparse
import json
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np
from scipy.stats import qmc
import yaml

from piper_mobile_manipulation.planning.capability import (
    capability_key,
    DEFAULT_DIRECTION_BIN_DEG,
    DEFAULT_DIRECTION_TOLERANCE_DEG,
    DEFAULT_POSITION_VOXEL_M,
    DEFAULT_SPATIAL_DILATION_CELLS,
    sha256_file,
    write_capability_map,
)
from piper_mobile_manipulation.execution.motion import (
    load_conservative_joint_limits,
)
from piper_tesseract_foxy.worker import TesseractBackend


DEFAULT_CHECKPOINTS = (100000, 250000, 500000, 1000000, 2000000)
_PROCESS_STATE = None


def _tool_corners(policy):
    origin = np.asarray(policy['origin_link6_m'], dtype=float)
    size = np.asarray(policy['size_m'], dtype=float)
    return np.asarray([
        [
            origin[0] + x * size[0] * 0.5,
            origin[1] + y * size[1] * 0.5,
            origin[2] + z * size[2] * 0.5,
            1.0,
        ]
        for x in (-1.0, 1.0)
        for y in (-1.0, 1.0)
        for z in (-1.0, 1.0)
    ], dtype=float)


def tool_minimum_z(link6_transform, corners):
    """Return the model's existing attached-tool support-plane evidence."""
    transform = np.asarray(link6_transform, dtype=float)
    points = np.asarray(corners, dtype=float)
    if (
            transform.shape != (4, 4) or points.shape != (8, 4)
            or not np.all(np.isfinite(transform))
            or not np.all(np.isfinite(points))):
        raise ValueError('attached-tool floor geometry is malformed')
    return float(np.min((transform @ points.T).T[:, 2]))


class _CollisionSampler:
    def __init__(self, urdf_path, srdf_path, manifest_path, deterministic_seed,
                 position_voxel_m, direction_bin_deg):
        self.backend = TesseractBackend(
            urdf_path, srdf_path, manifest_path,
            deterministic_seed=deterministic_seed)
        _default, report, _maximum_l1, _overrides = \
            self.backend.collision_policy()
        self.report_distance_m = float(report)
        self.manager = self.backend.configure_contact_manager(
            self.backend.robot.env.getDiscreteContactManager(),
            self.report_distance_m)
        policy = self.backend.external_floor_clearance_policy()
        if policy is None:
            raise ValueError(
                'collision model has no attached-tool '
                'floor-clearance geometry')
        self.tool_corners = _tool_corners(policy)
        self.position_voxel_m = float(position_voxel_m)
        self.direction_bin_deg = float(direction_bin_deg)

    def collision_qualified(self, joints):
        backend = self.backend
        backend.robot.env.setState(
            ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
            np.asarray(joints, dtype=float))
        state = backend.robot.env.getState()
        self.manager.setCollisionObjectsTransform(state.link_transforms)
        contacts = backend.api['ContactResultMap']()
        request = backend.api['ContactRequest'](
            backend.api['ContactTestType_ALL'])
        request.calculate_distance = True
        self.manager.contactTest(contacts, request)
        minimums = {}
        for contact in backend.flattened_contacts(contacts):
            distance = float(contact.distance)
            if not math.isfinite(distance):
                return False
            pair = tuple(sorted(str(value) for value in contact.link_names))
            minimums[pair] = min(distance, minimums.get(pair, math.inf))
        return not bool(backend.clearance_violations(minimums))

    def evaluate(self, joint_batch):
        keys = []
        floor_values = []
        collision_rejected = 0
        malformed = 0
        for joints in np.asarray(joint_batch, dtype=float):
            try:
                if not self.collision_qualified(joints):
                    collision_rejected += 1
                    continue
                camera = np.asarray(self.backend.robot.fk(
                    'manipulator', joints,
                    tip_link='camera_optical_frame').matrix, dtype=float)
                link6 = np.asarray(
                    self.backend.robot.fk(
                        'manipulator', joints, tip_link='link6').matrix,
                    dtype=float)
                keys.append(capability_key(
                    camera[:3, 3], camera[:3, 2],
                    self.position_voxel_m, self.direction_bin_deg))
                floor_values.append(tool_minimum_z(link6, self.tool_corners))
            except (RuntimeError, TypeError, ValueError):
                malformed += 1
        return {
            'keys': np.asarray(keys, dtype=np.uint64),
            'floor_values': np.asarray(floor_values, dtype=np.float32),
            'collision_rejected': int(collision_rejected),
            'malformed': int(malformed),
        }


def _initialize_process(urdf_path, srdf_path, manifest_path,
                        deterministic_seed, position_voxel_m,
                        direction_bin_deg):
    global _PROCESS_STATE
    _PROCESS_STATE = _CollisionSampler(
        urdf_path, srdf_path, manifest_path, deterministic_seed,
        position_voxel_m, direction_bin_deg)


def _evaluate_process_batch(joint_batch):
    return _PROCESS_STATE.evaluate(joint_batch)


def merge_capability_records(occupied, keys, floor_values):
    """Keep the best support-plane clearance for every occupied 5D bin."""
    key_array = np.asarray(keys, dtype=np.uint64)
    floor_array = np.asarray(floor_values, dtype=np.float32)
    if key_array.shape != floor_array.shape:
        raise ValueError('capability sample keys and floor values disagree')
    for key, floor in zip(key_array.tolist(), floor_array.tolist()):
        previous = occupied.get(int(key))
        if previous is None or float(floor) > previous:
            occupied[int(key)] = float(floor)


def sorted_capability_records(occupied):
    keys = np.asarray(sorted(occupied), dtype=np.uint64)
    floors = np.asarray(
        [occupied[int(key)] for key in keys], dtype=np.float32)
    return keys, floors


def source_hashes(project_root, paths):
    """Return repo-relative, deterministic source identities."""
    root = Path(project_root).resolve()
    result = {}
    for path in paths:
        source = Path(path).resolve()
        if root not in source.parents or not source.is_file():
            raise ValueError(
                'capability source is outside the project: %s' % path)
        result[str(source.relative_to(root))] = sha256_file(source)
    return result


def _joint_batches(lower, upper, maximum_samples, batch_size, seed):
    sampler = qmc.Halton(d=6, scramble=True, seed=int(seed))
    emitted = 0
    while emitted < maximum_samples:
        count = min(int(batch_size), int(maximum_samples) - emitted)
        unit = sampler.random(count)
        yield lower.reshape((1, 6)) + unit * (upper - lower).reshape((1, 6))
        emitted += count


def generate(args):
    checkpoints = tuple(sorted(set(int(value) for value in args.checkpoints)))
    if not checkpoints or checkpoints[0] < 1:
        raise ValueError('at least one positive checkpoint is required')
    maximum_samples = checkpoints[-1]
    if args.batch_size < 1 or any(
            value % int(args.batch_size) != 0 for value in checkpoints):
        raise ValueError('every checkpoint must be divisible by batch size')
    limits, ignored = load_conservative_joint_limits(args.joint_bounds)
    lower = limits[:, 0] + float(args.joint_margin_rad)
    upper = limits[:, 1] - float(args.joint_margin_rad)
    if np.any(lower >= upper):
        raise ValueError('joint margin collapses capability-map limits')
    sources = source_hashes(args.project_root, args.source)
    with open(args.manifest, 'r', encoding='utf-8') as stream:
        manifest = yaml.safe_load(stream)
    floor_policy = manifest.get('external_floor_clearance', {})
    tool_floor_clearance_m = float(floor_policy.get('clearance_m', -1.0))
    if (
            not bool(floor_policy.get('enabled', False))
            or not math.isfinite(tool_floor_clearance_m)
            or tool_floor_clearance_m < 0.0):
        raise ValueError(
            'collision manifest floor-clearance policy is invalid')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_base = {
        'schema_version': 1,
        'generator': 'halton_collision_qualified_camera_5d_v1',
        'deterministic_seed': int(args.seed),
        'position_voxel_m': float(args.position_voxel_m),
        'direction_bin_deg': float(args.direction_bin_deg),
        'direction_tolerance_deg': float(args.direction_tolerance_deg),
        'spatial_dilation_cells': int(args.spatial_dilation_cells),
        'joint_margin_rad': float(args.joint_margin_rad),
        'joint_position_limits_rad': limits.tolist(),
        'ignored_saved_joint_bounds': list(ignored),
        'source_sha256': sources,
        'tool_floor_clearance_m': tool_floor_clearance_m,
        'collision_scope': (
            'combined PiPER L515 holder and Bunker static endpoint clearance; '
            'support floor selected at query time'),
        'stores_joint_positions': False,
    }
    occupied = {}
    counters = {
        'sampled_configurations': 0,
        'collision_qualified_configurations': 0,
        'collision_rejected_configurations': 0,
        'malformed_configurations': 0,
    }
    started = time.monotonic()
    checkpoint_set = set(checkpoints)
    process_count = max(1, int(args.workers))
    batches = _joint_batches(
        lower, upper, maximum_samples, args.batch_size, args.seed)
    initializer_args = (
        args.urdf, args.srdf, args.manifest, args.seed,
        args.position_voxel_m, args.direction_bin_deg,
    )
    if process_count == 1:
        _initialize_process(*initializer_args)
        results = map(_evaluate_process_batch, batches)
        pool = None
    else:
        context = multiprocessing.get_context('spawn')
        pool = context.Pool(
            process_count,
            initializer=_initialize_process,
            initargs=initializer_args,
        )
        results = pool.imap(_evaluate_process_batch, batches, chunksize=1)
    summaries = []
    try:
        for result in results:
            sampled = (
                len(result['keys']) + int(result['collision_rejected'])
                + int(result['malformed']))
            counters['sampled_configurations'] += sampled
            counters['collision_qualified_configurations'] += len(
                result['keys'])
            counters['collision_rejected_configurations'] += int(
                result['collision_rejected'])
            counters['malformed_configurations'] += int(result['malformed'])
            merge_capability_records(
                occupied, result['keys'], result['floor_values'])
            if counters['sampled_configurations'] not in checkpoint_set:
                continue
            keys, floors = sorted_capability_records(occupied)
            elapsed = time.monotonic() - started
            metadata = dict(metadata_base)
            metadata.update(counters)
            metadata.update({
                'occupied_pose_direction_bins': int(len(keys)),
                'generation_elapsed_sec': float(elapsed),
                'generation_workers': process_count,
                'checkpoint_samples': int(counters['sampled_configurations']),
            })
            path = output_dir / (
                'capability_map_%07d.npz'
                % counters['sampled_configurations'])
            write_capability_map(path, keys, floors, metadata)
            summary = {
                'artifact': path.name,
                'artifact_bytes': int(path.stat().st_size),
                **{key: int(value) for key, value in counters.items()},
                'occupied_pose_direction_bins': int(len(keys)),
                'generation_elapsed_sec': float(elapsed),
                'samples_per_sec': (
                    counters['sampled_configurations'] / elapsed
                    if elapsed > 0.0 else 0.0),
            }
            summaries.append(summary)
            (output_dir / 'generation_progress.json').write_text(
                json.dumps(summaries, indent=2, sort_keys=True) + '\n',
                encoding='utf-8')
            print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return summaries


def parser():
    value = argparse.ArgumentParser(
        description='Generate the command-free PiPER camera capability map.')
    value.add_argument('--project-root', required=True)
    value.add_argument('--urdf', required=True)
    value.add_argument('--srdf', required=True)
    value.add_argument('--manifest', required=True)
    value.add_argument('--joint-bounds', required=True)
    value.add_argument('--source', action='append', required=True)
    value.add_argument('--output-dir', required=True)
    value.add_argument(
        '--checkpoints', type=int, nargs='+', default=DEFAULT_CHECKPOINTS)
    value.add_argument('--workers', type=int, default=8)
    value.add_argument('--batch-size', type=int, default=1000)
    value.add_argument('--seed', type=int, default=42)
    value.add_argument('--joint-margin-rad', type=float, default=0.03)
    value.add_argument(
        '--position-voxel-m', type=float,
        default=DEFAULT_POSITION_VOXEL_M)
    value.add_argument(
        '--direction-bin-deg', type=float,
        default=DEFAULT_DIRECTION_BIN_DEG)
    value.add_argument(
        '--direction-tolerance-deg', type=float,
        default=DEFAULT_DIRECTION_TOLERANCE_DEG)
    value.add_argument(
        '--spatial-dilation-cells', type=int,
        default=DEFAULT_SPATIAL_DILATION_CELLS)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    generate(args)


if __name__ == '__main__':
    main()
