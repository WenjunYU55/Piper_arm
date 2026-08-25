#!/usr/bin/env python3
"""Select the smallest converged PiPER capability-map checkpoint."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
MOBILE_SOURCE = ROOT / (
    'piper_ros_foxy/src/piper_mobile_manipulation')
if str(MOBILE_SOURCE) not in sys.path:
    sys.path.insert(0, str(MOBILE_SOURCE))

from piper_mobile_manipulation.capability_map import (  # noqa: E402
    load_capability_map,
    write_capability_map,
)
from piper_mobile_manipulation.scan_motion import (  # noqa: E402
    orbit_camera_view,
)
from piper_mobile_manipulation.viewpoint_rays import (  # noqa: E402
    build_ray_samples,
)


TABLETOP_FLOOR_Z_M = 0.005
TOOL_FLOOR_CLEARANCE_M = 0.005
RAY_MINIMUM_STANDOFF_M = 0.28
RAY_MAXIMUM_STANDOFF_M = 0.80
VALIDATION_RAY_COUNT = 360
MAXIMUM_VALIDATION_TARGETS = 12


def _finite_three(value):
    result = np.asarray(value, dtype=float)
    valid = result.shape == (3,) and np.all(np.isfinite(result))
    return result if valid else None


def recorded_evidence(dataset_root):
    """Read achieved poses and frozen targets from immutable captures."""
    targets = {}
    poses = []
    pattern = 'scan_*/frames/view_*_metadata.yaml'
    for path in sorted(Path(dataset_root).glob(pattern)):
        try:
            data = yaml.safe_load(path.read_text(encoding='utf-8'))
        except (OSError, yaml.YAMLError):
            continue
        selection = (
            data.get('view_selection') if isinstance(data, dict) else None)
        if (
                not isinstance(selection, dict)
                or not selection.get('available', False)):
            continue
        camera = _finite_three(selection.get('camera_position_m'))
        nominal_look = _finite_three(selection.get('nominal_look_direction'))
        actual_look = _finite_three(selection.get('look_direction'))
        try:
            standoff = float(selection.get('ray_standoff_m'))
        except (TypeError, ValueError):
            continue
        if (
                camera is None or nominal_look is None or actual_look is None
                or not math.isfinite(standoff) or standoff <= 0.0):
            continue
        nominal_norm = float(np.linalg.norm(nominal_look))
        actual_norm = float(np.linalg.norm(actual_look))
        if min(nominal_norm, actual_norm) <= 1e-9:
            continue
        nominal_look /= nominal_norm
        actual_look /= actual_norm
        target = camera + nominal_look * standoff
        target_key = tuple(np.round(target / 0.05).astype(int).tolist())
        targets.setdefault(target_key, target)
        transform = data.get('camera_transform', {})
        actual_camera = _finite_three(transform.get('translation_m'))
        matrix = np.asarray(transform.get('matrix_4x4'), dtype=float)
        if actual_camera is not None and matrix.shape == (4, 4) and np.all(
                np.isfinite(matrix)):
            actual_look = matrix[:3, 2]
        poses.append({
            'dataset': str(path.parents[1].name),
            'frame': str(path.name),
            'camera_position_m': actual_camera.tolist()
            if actual_camera is not None else camera.tolist(),
            'look_direction': actual_look.tolist(),
        })
    ordered_targets = [targets[key] for key in sorted(targets)]
    if len(ordered_targets) > MAXIMUM_VALIDATION_TARGETS:
        indexes = np.linspace(
            0, len(ordered_targets) - 1,
            MAXIMUM_VALIDATION_TARGETS).round().astype(int)
        ordered_targets = [ordered_targets[index] for index in indexes]
    if not ordered_targets:
        ordered_targets = [np.asarray([0.4, 0.0, 0.12], dtype=float)]
    return ordered_targets, poses


def validation_rays(targets):
    samples = build_ray_samples(
        'full_sphere', VALIDATION_RAY_COUNT, center_angle_deg=180.0)
    rays = []
    for target in targets:
        maximum = max(
            RAY_MINIMUM_STANDOFF_M,
            min(float(np.linalg.norm(target)), RAY_MAXIMUM_STANDOFF_M),
        )
        for azimuth, pitch in samples:
            camera, _look = orbit_camera_view(
                target, azimuth, 1.0, pitch)
            direction = camera - target
            direction /= np.linalg.norm(direction)
            rays.append((target, direction, maximum))
    return rays


def evaluate_map(capability_map, rays, poses):
    started = time.perf_counter()
    ray_support = []
    query_times = []
    for target, direction, maximum in rays:
        result = capability_map.intersects_ray(
            target, direction,
            RAY_MINIMUM_STANDOFF_M, maximum,
            TABLETOP_FLOOR_Z_M, TOOL_FLOOR_CLEARANCE_M)
        ray_support.append(result.supported)
        query_times.append(result.elapsed_ms)
    pose_support = []
    for pose in poses:
        result = capability_map.supports_pose(
            pose['camera_position_m'], pose['look_direction'],
            TABLETOP_FLOOR_Z_M, TOOL_FLOOR_CLEARANCE_M)
        pose_support.append(result.supported)
    return {
        'ray_support': np.asarray(ray_support, dtype=bool),
        'pose_support': np.asarray(pose_support, dtype=bool),
        'median_query_ms': (
            float(np.median(query_times)) if query_times else 0.0),
        'p95_query_ms': float(np.percentile(query_times, 95.0))
        if query_times else 0.0,
        'evaluation_elapsed_sec': time.perf_counter() - started,
    }


def _percent(numerator, denominator):
    if not denominator:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def validate(args):
    checkpoints = sorted(
        Path(args.checkpoint_dir).glob('capability_map_*.npz'))
    if not checkpoints:
        raise ValueError('no capability-map checkpoints were found')
    targets, poses = recorded_evidence(args.dataset_root)
    rays = validation_rays(targets)
    loaded = [
        load_capability_map(path, args.project_root, verify_sources=True)
        for path in checkpoints]
    evaluations = [evaluate_map(item, rays, poses) for item in loaded]
    reference = evaluations[-1]['ray_support']
    reference_supported = int(np.count_nonzero(reference))
    summaries = []
    selected_index = None
    for index, (path, capability_map, evaluation) in enumerate(zip(
            checkpoints, loaded, evaluations)):
        supported = evaluation['ray_support']
        true_positive = int(np.count_nonzero(supported & reference))
        recall = _percent(true_positive, reference_supported)
        agreement = _percent(
            int(np.count_nonzero(supported == reference)), len(reference))
        known_supported = int(np.count_nonzero(evaluation['pose_support']))
        known_recall = _percent(known_supported, len(poses))
        summary = {
            'artifact': path.name,
            'checkpoint_samples': int(capability_map.metadata.get(
                'checkpoint_samples', 0)),
            'occupied_pose_direction_bins': int(len(capability_map.keys)),
            'artifact_bytes': int(path.stat().st_size),
            'validation_rays': int(len(rays)),
            'supported_validation_rays': int(np.count_nonzero(supported)),
            'reference_supported_ray_recall_percent': recall,
            'reference_ray_classification_agreement_percent': agreement,
            'known_achieved_poses': int(len(poses)),
            'known_achieved_poses_supported': known_supported,
            'known_achieved_pose_recall_percent': known_recall,
            'median_query_ms': evaluation['median_query_ms'],
            'p95_query_ms': evaluation['p95_query_ms'],
            'evaluation_elapsed_sec': evaluation['evaluation_elapsed_sec'],
        }
        summaries.append(summary)
        if (
                selected_index is None and reference_supported > 0
                and recall >= float(args.minimum_reference_recall_percent)
                and (not poses or known_supported == len(poses))):
            selected_index = index
    qualified = selected_index is not None
    if selected_index is None:
        selected_index = len(checkpoints) - 1
    selected_map = loaded[selected_index]
    metadata = dict(selected_map.metadata)
    metadata.update({
        'qualified_for_enforcement': bool(qualified),
        'selection_policy': (
            'smallest checkpoint with 100 percent known achieved-pose recall '
            'and at least %.3f percent of final supported validation rays'
            % float(args.minimum_reference_recall_percent)),
        'selected_checkpoint_samples': int(metadata.get(
            'checkpoint_samples', 0)),
        'reference_checkpoint_samples': int(loaded[-1].metadata.get(
            'checkpoint_samples', 0)),
        'validation_target_count': int(len(targets)),
        'validation_ray_count': int(len(rays)),
        'known_achieved_pose_count': int(len(poses)),
        'convergence_summary': summaries,
    })
    write_capability_map(
        args.output,
        selected_map.keys,
        selected_map.maximum_tool_minimum_z_m,
        metadata,
    )
    report = {
        'schema_version': 1,
        'qualified_for_enforcement': bool(qualified),
        'selected_artifact': checkpoints[selected_index].name,
        'selected_checkpoint_samples': int(metadata[
            'selected_checkpoint_samples']),
        'reference_checkpoint_samples': int(metadata[
            'reference_checkpoint_samples']),
        'validation_target_count': len(targets),
        'validation_ray_count': len(rays),
        'known_achieved_pose_count': len(poses),
        'checkpoints': summaries,
    }
    Path(args.report_json).write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    lines = [
        '# Capability-map convergence',
        '',
        'The atlas stores occupied 20 mm camera-position cells plus 10-degree '
        'optical-direction bins. It stores neither raw joint samples nor a '
        'solid outer workspace volume.',
        '',
        '| Samples | Occupied 5D bins | Validation support | '
        'Final-support recall | Known poses | Median query | Artifact |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for item in summaries:
        lines.append(
            '| {checkpoint_samples:,} | {occupied_pose_direction_bins:,} | '
            '{supported_validation_rays:,}/{validation_rays:,} | '
            '{reference_supported_ray_recall_percent:.2f}% | '
            '{known_achieved_poses_supported:,}/{known_achieved_poses:,} | '
            '{median_query_ms:.3f} ms | {artifact_bytes:,} B |'.format(**item))
    lines.extend([
        '',
        '**Selected:** `%s` (%s enforcement qualification).' % (
            checkpoints[selected_index].name,
            'passed' if qualified else 'failed'),
        '',
        'The two-million-sample checkpoint is the comparison reference, not '
        'automatically the runtime artifact. Tesseract remains authoritative '
        'for exact IK, collision, path and visibility validation.',
        '',
    ])
    Path(args.report_markdown).write_text(
        '\n'.join(lines), encoding='utf-8')
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parser():
    value = argparse.ArgumentParser()
    value.add_argument('--project-root', default=str(ROOT))
    value.add_argument('--checkpoint-dir', required=True)
    value.add_argument(
        '--dataset-root', default=str(ROOT / 'datasets/active_scan'))
    value.add_argument('--output', required=True)
    value.add_argument('--report-json', required=True)
    value.add_argument('--report-markdown', required=True)
    value.add_argument(
        '--minimum-reference-recall-percent', type=float, default=99.0)
    return value


def main(argv=None):
    validate(parser().parse_args(argv))


if __name__ == '__main__':
    main()
