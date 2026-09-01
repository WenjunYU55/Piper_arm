#!/usr/bin/env python3
"""Build reproducible Chapter 5 method-comparison evidence.

This tool is deliberately command-free: it reads immutable scan artifacts,
historical documentation, Git metadata, capability-map results, and offline
reconstruction reports.  It never imports ROS, opens CAN, or starts a robot
process.  The output is an analysis JSON file plus a multi-sheet Excel
workbook.  Comparisons are labelled by evidence strength so observational
development runs cannot be mistaken for controlled experiments.
"""

import argparse
import collections
import glob
import hashlib
import json
import math
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPOSITORY
MOBILE_SOURCE = (
    REPOSITORY / 'piper_ros_foxy' / 'src' /
    'piper_mobile_manipulation')
if str(MOBILE_SOURCE) not in sys.path:
    sys.path.insert(0, str(MOBILE_SOURCE))

from piper_mobile_manipulation.planning.measured_surface import (  # noqa: E402
    measured_surface_coverage,
)


STRENGTHS = {
    'CONTROLLED_REPLAY',
    'PAIRED_PHYSICAL',
    'MATCHED_RUNS',
    'HISTORICAL_OBSERVATIONAL',
    'EXPLORATORY',
}


def sha256_file(path):
    """Return the SHA-256 digest of one source artifact."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path):
    """Load one YAML mapping."""
    with Path(path).open('r', encoding='utf-8') as stream:
        return yaml.safe_load(stream) or {}


def local_artifact(scan_dir, recorded_path):
    """Resolve absolute capture provenance after a dataset was relocated."""
    value = Path(str(recorded_path or ''))
    if value.is_file():
        return value
    candidate = Path(scan_dir) / 'frames' / value.name
    return candidate if candidate.is_file() else None


def finite_depth(depth, minimum=0.15, maximum=1.20):
    """Return the project's ordinary finite target-depth range mask."""
    return np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)


def depth_stats(depth, support):
    """Return robust descriptive statistics without claiming accuracy."""
    values = np.asarray(depth)[np.asarray(support, dtype=bool)]
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            'valid_pixels': 0,
            'median_depth_m': None,
            'depth_std_mm': None,
            'depth_mad_mm': None,
        }
    median = float(np.median(values))
    return {
        'valid_pixels': int(len(values)),
        'median_depth_m': median,
        'depth_std_mm': float(np.std(values) * 1000.0),
        'depth_mad_mm': float(
            np.median(np.abs(values - median)) * 1000.0),
    }


def roi_support(metadata, shape):
    """Reconstruct the recorded rectangular target ROI support."""
    target = metadata.get('target_3d') or {}
    cx = float(target.get('source_u', float('nan')))
    cy = float(target.get('source_v', float('nan')))
    width = float(target.get('roi_width', float('nan')))
    height = float(target.get('roi_height', float('nan')))
    if not all(math.isfinite(value) for value in (cx, cy, width, height)):
        return None
    x0 = max(0, int(math.floor(cx - width / 2.0)))
    x1 = min(shape[1], int(math.ceil(cx + width / 2.0)))
    y0 = max(0, int(math.floor(cy - height / 2.0)))
    y1 = min(shape[0], int(math.ceil(cy + height / 2.0)))
    if x0 >= x1 or y0 >= y1:
        return None
    support = np.zeros(shape, dtype=bool)
    support[y0:y1, x0:x1] = True
    return support


def analyse_capture_geometry():
    """Replay ROI, semantic-mask, and qualified target-depth extraction."""
    rows = []
    pattern = EVIDENCE_ROOT / 'datasets' / 'active_scan' / 'scan_*' / \
        'frames' / 'view_*_metadata.yaml'
    for metadata_path in sorted(glob.glob(str(pattern))):
        metadata_path = Path(metadata_path)
        scan_dir = metadata_path.parents[1]
        metadata = load_yaml(metadata_path)
        depth_path = local_artifact(
            scan_dir, metadata.get('depth_file_path'))
        mask_path = local_artifact(
            scan_dir, metadata.get('mask_file_path'))
        if depth_path is None or mask_path is None:
            continue
        depth = np.load(str(depth_path), allow_pickle=False).astype(np.float64)
        if '16U' in str(metadata.get('depth_encoding', '')):
            depth *= 0.001
        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_image is None or mask_image.shape != depth.shape:
            continue
        mask = mask_image > 0
        roi = roi_support(metadata, depth.shape)
        if roi is None:
            rows_y, rows_x = np.nonzero(mask)
            if not len(rows_y):
                continue
            roi = np.zeros(depth.shape, dtype=bool)
            roi[
                rows_y.min():rows_y.max() + 1,
                rows_x.min():rows_x.max() + 1,
            ] = True
        valid = finite_depth(depth)
        roi_stats = depth_stats(depth, roi & valid)
        mask_stats = depth_stats(depth, mask & valid)
        qualified_stats = {
            'valid_pixels': None,
            'median_depth_m': None,
            'depth_std_mm': None,
            'depth_mad_mm': None,
        }
        qualified_depth_path = local_artifact(
            scan_dir, metadata.get('target_depth_png_file_path'))
        qualified_mask_path = local_artifact(
            scan_dir, metadata.get('target_support_mask_file_path'))
        if qualified_depth_path is not None and qualified_mask_path is not None:
            qualified_depth = cv2.imread(
                str(qualified_depth_path), cv2.IMREAD_UNCHANGED)
            qualified_mask = cv2.imread(
                str(qualified_mask_path), cv2.IMREAD_GRAYSCALE)
            if (
                    qualified_depth is not None
                    and qualified_mask is not None
                    and qualified_depth.shape == qualified_mask.shape):
                qualified_depth = qualified_depth.astype(np.float64) * 0.001
                qualified_stats = depth_stats(
                    qualified_depth,
                    (qualified_mask > 0) & finite_depth(qualified_depth))
        roi_valid = int(np.count_nonzero(roi & valid))
        off_mask = int(np.count_nonzero(roi & valid & ~mask))
        confidence = metadata.get('confidence_quality') or {}
        layers = confidence.get('depth_layer_selection') or {}
        selection = metadata.get('view_selection') or {}
        rows.append({
            'scan': scan_dir.name,
            'frame': metadata_path.stem,
            'capture_schema_version': int(
                metadata.get('capture_schema_version', 1)),
            'requested_view_policy': selection.get(
                'view_selection_requested_policy', ''),
            'selected_view_policy': selection.get(
                'view_selection_policy', ''),
            'roi_valid_pixels': roi_stats['valid_pixels'],
            'roi_median_depth_m': roi_stats['median_depth_m'],
            'roi_depth_std_mm': roi_stats['depth_std_mm'],
            'roi_depth_mad_mm': roi_stats['depth_mad_mm'],
            'roi_off_mask_valid_pixels': off_mask,
            'roi_off_mask_valid_fraction': (
                float(off_mask) / float(roi_valid) if roi_valid else None),
            'mask_valid_pixels': mask_stats['valid_pixels'],
            'mask_median_depth_m': mask_stats['median_depth_m'],
            'mask_depth_std_mm': mask_stats['depth_std_mm'],
            'mask_depth_mad_mm': mask_stats['depth_mad_mm'],
            'qualified_valid_pixels': qualified_stats['valid_pixels'],
            'qualified_median_depth_m': qualified_stats['median_depth_m'],
            'qualified_depth_std_mm': qualified_stats['depth_std_mm'],
            'qualified_depth_mad_mm': qualified_stats['depth_mad_mm'],
            'qualified_retained_fraction_of_mask': (
                float(qualified_stats['valid_pixels']) /
                float(mask_stats['valid_pixels'])
                if qualified_stats['valid_pixels'] is not None
                and mask_stats['valid_pixels'] else None),
            'depth_layer_candidate_count': layers.get('candidate_count'),
            'depth_layer_selected_m': layers.get('selected_depth_m'),
            'depth_layer_score_margin': layers.get('score_margin'),
            'source_metadata': str(metadata_path.relative_to(EVIDENCE_ROOT)),
        })
    return rows


def median(rows, key):
    """Median of finite numeric values for one key."""
    values = [
        float(row[key]) for row in rows
        if isinstance(row.get(key), (int, float))
        and math.isfinite(float(row[key]))]
    return float(statistics.median(values)) if values else None


def mean(rows, key):
    """Mean of finite numeric values for one key."""
    values = [
        float(row[key]) for row in rows
        if isinstance(row.get(key), (int, float))
        and math.isfinite(float(row[key]))]
    return float(statistics.mean(values)) if values else None


def vector_angle_deg(first, second):
    """Return the smaller angle between two finite directions."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return None
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def circular_span_deg(values):
    """Return the minimum circular arc containing all azimuths."""
    if len(values) < 2:
        return 0.0
    values = sorted(float(value) % 360.0 for value in values)
    gaps = [
        values[index + 1] - values[index]
        for index in range(len(values) - 1)]
    gaps.append(values[0] + 360.0 - values[-1])
    return float(360.0 - max(gaps))


def analyse_scan_runs():
    """Summarize persisted accepted observations by historical policy."""
    rows = []
    for scan_dir in sorted((
            EVIDENCE_ROOT / 'datasets' / 'active_scan').glob('scan_*')):
        metadata_paths = sorted(
            (scan_dir / 'frames').glob('view_*_metadata.yaml'))
        if not metadata_paths:
            continue
        records = [load_yaml(path) for path in metadata_paths]
        requested = [
            (record.get('view_selection') or {}).get(
                'view_selection_requested_policy')
            for record in records]
        requested = [value for value in requested if value]
        policy = (
            collections.Counter(requested).most_common(1)[0][0]
            if requested else 'legacy_or_unrecorded')
        look_directions = [
            (record.get('view_selection') or {}).get('look_direction')
            for record in records]
        look_directions = [value for value in look_directions if value]
        azimuths = [
            math.degrees(math.atan2(value[1], value[0]))
            for value in look_directions]
        elevations = [
            math.degrees(math.asin(max(-1.0, min(1.0, value[2]))))
            for value in look_directions]
        adjacent = [
            vector_angle_deg(look_directions[index - 1],
                             look_directions[index])
            for index in range(1, len(look_directions))]
        adjacent = [value for value in adjacent if value is not None]
        timestamps = []
        for record in records:
            stamp = record.get('capture_timestamp') or {}
            timestamps.append(
                float(stamp.get('sec', 0)) +
                float(stamp.get('nanosec', 0)) * 1e-9)
        coverage = measured_surface_coverage(
            str(scan_dir), minimum_views=1)
        gains = list(coverage.get('per_view_novel_fraction') or [])
        manifest = scan_dir / 'manifest.json'
        rows.append({
            'scan': scan_dir.name,
            'policy': policy,
            'accepted_captures': len(records),
            'capture_span_sec': (
                max(timestamps) - min(timestamps)
                if len(timestamps) > 1 else 0.0),
            'azimuth_span_deg': circular_span_deg(azimuths),
            'elevation_span_deg': (
                max(elevations) - min(elevations) if elevations else 0.0),
            'mean_adjacent_direction_deg': (
                float(statistics.mean(adjacent)) if adjacent else None),
            'adjacent_views_below_15deg': int(sum(
                value < 15.0 for value in adjacent)),
            'adjacent_comparisons': len(adjacent),
            'adjacent_redundancy_rate': (
                float(sum(value < 15.0 for value in adjacent)) /
                float(len(adjacent)) if adjacent else None),
            'mean_capture_quality': float(statistics.mean(
                float(record.get('scan_quality_score', 0.0))
                for record in records)),
            'measured_covered_voxels_10mm': coverage.get('covered_voxels'),
            'mean_novel_fraction': (
                float(statistics.mean(gains)) if gains else None),
            'mean_post_seed_novel_fraction': (
                float(statistics.mean(gains[1:])) if len(gains) > 1
                else None),
            'last_novel_fraction': gains[-1] if gains else None,
            'measured_converged': bool(coverage.get('converged')),
            'manifest_sha256': sha256_file(manifest) if manifest.is_file() else '',
            'comparison_strength': 'HISTORICAL_OBSERVATIONAL',
        })
    return rows


def analyse_offline_resegmentation():
    """Load the complete seven-frame captured-vs-fresh mask replay."""
    root = (
        EVIDENCE_ROOT / 'datasets' / 'active_scan' /
        'scan_20260821_220003' / 'reconstruction' / 'offline_resegment')
    candidates = []
    for path in root.glob('*/index.json'):
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.get('frame_count') == 7 \
                and all('comparison_path' in row for row in value['frames']):
            candidates.append((int(value['identity'].get(
                'pipeline_version', 0)), path, value))
    if not candidates:
        return [], None
    _, path, value = sorted(candidates)[-1]
    rows = []
    for row in value['frames']:
        rows.append({
            'scan': 'scan_20260821_220003',
            'frame': row['frame'],
            'captured_live_mask_pixels': row['source_live_mask_pixels'],
            'fresh_validated_mask_pixels': row['validated_mask_pixels'],
            'fresh_live_intersection_pixels': row[
                'fresh_live_intersection_pixels'],
            'fresh_to_live_iou': row['fresh_to_live_iou'],
            'groundingdino_confidence': row['groundingdino_confidence'],
            'sam2_score': row['sam2_score'],
            'pixels_removed_from_captured_support': (
                row['source_live_mask_pixels'] -
                row['fresh_live_intersection_pixels']),
            'source_index': str(path.relative_to(EVIDENCE_ROOT)),
            'comparison_strength': 'CONTROLLED_REPLAY',
        })
    return rows, {
        'source_index': str(path.relative_to(EVIDENCE_ROOT)),
        'index_sha256': sha256_file(path),
        'generation_sha256': value['generation_sha256'],
    }


def analyse_capability_convergence():
    """Load fixed-validation-set capability-map convergence evidence."""
    path = (
        EVIDENCE_ROOT / 'piper_ros_foxy' / 'src' /
        'piper_mobile_manipulation' / 'config' /
        'capability_map_convergence.json')
    value = json.loads(path.read_text(encoding='utf-8'))
    rows = []
    for row in value['checkpoints']:
        item = dict(row)
        item.update({
            'comparison_strength': 'CONTROLLED_REPLAY',
            'source': str(path.relative_to(EVIDENCE_ROOT)),
            'source_sha256': sha256_file(path),
            'selected_for_enforcement': (
                int(row['checkpoint_samples']) ==
                int(value['selected_checkpoint_samples'])),
        })
        rows.append(item)
    return rows


def analyse_feasibility_events():
    """Extract the first full-sphere prequalification event per mission."""
    root = EVIDENCE_ROOT / 'datasets' / 'ray_diagnostics'
    rows = []
    tesseract = collections.Counter()
    for path in root.glob('**/ray_mission_diagnostics.json'):
        mission = path.parent.name
        if mission.startswith(('offline-', 'replay_')):
            continue
        value = json.loads(path.read_text(encoding='utf-8'))
        candidates = []
        for event in value.get('events', []):
            metrics = event.get('metrics') or {}
            input_count = metrics.get('input_ray_count')
            if event.get('stage') == 'prequalify' \
                    and isinstance(input_count, (int, float)) \
                    and input_count >= 300:
                candidates.append((int(event.get('timestamp_ns', 0)), metrics))
        if candidates:
            _, metrics = sorted(candidates)[0]
            rows.append({
                'mission': mission,
                'generated_rays': int(metrics['input_ray_count']),
                'workspace_rejected_rays': int(
                    metrics.get('workspace_rejected_rays', 0)),
                'capability_rejected_rays': int(
                    metrics.get('capability_rejected_rays', 0)),
                'prequalified_surviving_rays': int(
                    metrics.get('surviving_ray_count', 0)),
                'survival_fraction': (
                    float(metrics.get('surviving_ray_count', 0)) /
                    float(metrics['input_ray_count'])),
                'comparison_strength': 'HISTORICAL_OBSERVATIONAL',
                'source': str(path.relative_to(EVIDENCE_ROOT)),
            })
        for generation in value.get('generations', []):
            if generation.get('historical_replay'):
                continue
            for ray in generation.get('rays') or []:
                status = ray.get('tesseract_status')
                if status:
                    tesseract[str(status)] += 1
    return rows, dict(tesseract)


def reconstruction_metric(path):
    """Normalize one reconstruction quality report."""
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    metrics = value.get('mesh_metrics') or {}
    dimensions = metrics.get('dimension_check') or {}
    residual = metrics.get('point_to_mesh_residual') or {}
    return {
        'scan': 'scan_20260821_220003',
        'registration_mode': value.get('registration_mode'),
        'overall_quality': value.get('overall_quality'),
        'observed_x_mm': (
            dimensions.get('observed_obb_m', [None] * 3)[0] * 1000.0
            if dimensions.get('observed_obb_m') else None),
        'observed_y_mm': (
            dimensions.get('observed_obb_m', [None] * 3)[1] * 1000.0
            if dimensions.get('observed_obb_m') else None),
        'observed_z_mm': (
            dimensions.get('observed_obb_m', [None] * 3)[2] * 1000.0
            if dimensions.get('observed_obb_m') else None),
        'mean_absolute_dimension_error_mm': (
            float(dimensions.get('mean_absolute_error_m')) * 1000.0
            if dimensions.get('mean_absolute_error_m') is not None else None),
        'maximum_absolute_dimension_error_mm': (
            float(dimensions.get('maximum_absolute_error_m')) * 1000.0
            if dimensions.get('maximum_absolute_error_m') is not None else None),
        'dominant_component_triangle_ratio': metrics.get(
            'dominant_component_triangle_ratio'),
        'median_point_to_mesh_residual_mm': (
            float(residual.get('median_m')) * 1000.0
            if residual.get('median_m') is not None else None),
        'p90_point_to_mesh_residual_mm': (
            float(residual.get('p90_m')) * 1000.0
            if residual.get('p90_m') is not None else None),
        'connected_component_count': metrics.get('connected_component_count'),
        'source_report': str(path),
        'source_sha256': sha256_file(path),
        'comparison_strength': 'CONTROLLED_REPLAY',
    }


def analyse_reconstruction(replay_dir):
    """Load fresh same-dataset mode outputs from the supplied replay dir."""
    replay_dir = Path(replay_dir)
    rows = []
    for mode in (
            'robot_pose', 'bounded_gicp', 'multiway_gicp',
            'constrained_superposition', 'scene_pose_graph'):
        path = replay_dir / (
            'scan_20260821_220003.auto.%s.ply.quality.json' % mode)
        if path.is_file():
            rows.append(reconstruction_metric(path))
    return rows


def documented_evidence():
    """Return numeric evidence preserved only in reviewed project records."""
    return {
        'semantic_prompt_replay': {
            'N': 63,
            'former_mixed_prompt_mean_confidence': 0.8435,
            'former_mixed_prompt_min_confidence': 0.7947,
            'target_only_prompt_accepted': 63,
            'target_only_prompt_mean_confidence': 0.8751,
            'target_only_prompt_min_confidence': 0.8342,
            'source': 'docs/historical/supervised_workflow_handoff_2026_07_30.md',
            'comparison_strength': 'CONTROLLED_REPLAY',
            'limitation': (
                'Confirmed-positive replay; confidence is not accuracy and '
                'does not estimate false-positive rate.'),
        },
        'offline_mask_reconstruction': {
            'scan': 'scan_20260821_220003',
            'captured_obb_mm': [58.79, 52.88, 48.29],
            'captured_median_residual_mm': 0.558,
            'captured_dominant_component_percent': 90.5,
            'offline_resegment_obb_mm': [51.48, 49.00, 45.77],
            'offline_resegment_median_residual_mm': 0.530,
            'offline_resegment_dominant_component_percent': 86.65,
            'source': 'docs/ai/60-debt.yaml:offline_resegmentation_followup_2026_08_22',
            'comparison_strength': 'CONTROLLED_REPLAY',
            'limitation': (
                'One seven-view cube dataset; both reconstructions remained '
                'FAIL against the measured 35 mm cube.'),
        },
        'curobo_collision_replay': {
            'N': 2004,
            'mutually_colliding': 43,
            'mutually_clear': 1943,
            'curobo_conservative_false_positives': 18,
            'curobo_state_level_false_negatives': 0,
            'source': (
                'git curobo-integration:docs/architecture/'
                'motion_planner_backends.md at 36d12f5'),
            'comparison_strength': 'CONTROLLED_REPLAY',
            'limitation': (
                'State-level collision-model comparison only; no continuous '
                'path proof and no physical cuRobo motion qualification.'),
        },
    }


def analyse_planner_benchmark(path):
    """Load one command-free, exact-revalidated planner comparison."""
    value = Path(path)
    if not value.is_file():
        return {}
    with value.open('r', encoding='utf-8') as stream:
        report = json.load(stream)
    if (
            report.get('comparison_strength') != 'CONTROLLED_REPLAY'
            or report.get('real_arm_motion') is not False
            or report.get('physical_result_claimed') is not False):
        raise ValueError('planner benchmark is not command-free replay evidence')
    trials = [
        row for row in report.get('trials', [])
        if not row.get('warmup', False)]
    if not trials:
        raise ValueError('planner benchmark contains no measured trials')
    report['trials'] = trials
    report['source'] = str(value.resolve())
    return report


def difference(first, second):
    """Return absolute and relative differences for numeric results."""
    absolute = abs(float(second) - float(first))
    relative = (
        (float(second) - float(first)) / abs(float(first)) * 100.0
        if abs(float(first)) > 1e-12 else None)
    return absolute, relative


def comparison_row(subsystem, method_a, method_b, unit, count, metric,
                   result_a, result_b, strength, sources, interpretation,
                   limitation, selected):
    """Build one normalized Method_Comparison row."""
    if strength not in STRENGTHS:
        raise ValueError('unknown comparison strength: %s' % strength)
    absolute, relative = difference(result_a, result_b)
    return {
        'subsystem': subsystem,
        'method_a': method_a,
        'method_b': method_b,
        'experimental_unit': unit,
        'N': count,
        'metric': metric,
        'method_a_result': result_a,
        'method_b_result': result_b,
        'absolute_difference': absolute,
        'relative_difference_percent': relative,
        'comparison_strength': strength,
        'source_runs': sources,
        'interpretation': interpretation,
        'limitation': limitation,
        'selected_for_final_system': selected,
    }


def row_by_scan(scan_rows, name):
    """Return one persisted scan summary by directory name."""
    for row in scan_rows:
        if row['scan'] == name:
            return row
    raise ValueError('required comparison scan is unavailable: %s' % name)


def build_comparisons(data):
    """Construct thesis-facing pairwise comparisons and ablations."""
    geometry = data['capture_geometry']
    qualified = [
        row for row in geometry
        if row['qualified_valid_pixels'] is not None]
    documented = data['documented_evidence']
    prompt = documented['semantic_prompt_replay']
    voxel = row_by_scan(data['scan_runs'], 'scan_20260820_230845')
    ray = row_by_scan(data['scan_runs'], 'scan_20260821_220003')
    capability = data['capability_convergence']
    low_map = capability[0]
    selected_map = capability[-1]
    feasibility = data['feasibility_initial_pools']
    generated = sum(row['generated_rays'] for row in feasibility)
    workspace_passed = generated - sum(
        row['workspace_rejected_rays'] for row in feasibility)
    map_passed = sum(
        row['prequalified_surviving_rays'] for row in feasibility)
    reconstruction = {
        row['registration_mode']: row for row in data['reconstruction_replay']}
    rows = []
    rows.append(comparison_row(
        'Semantic acquisition', 'former mixed target/obstacle prompt',
        'target-only GroundingDINO prompt', 'confirmed target images',
        prompt['N'], 'mean detector confidence (not accuracy)',
        prompt['former_mixed_prompt_mean_confidence'],
        prompt['target_only_prompt_mean_confidence'], 'CONTROLLED_REPLAY',
        prompt['source'],
        'The target-only prompt produced a stronger usable target hypothesis '
        'on the same confirmed-positive images.', prompt['limitation'],
        'target-only GroundingDINO prompt'))
    rows.append(comparison_row(
        'Depth localisation', 'rectangular detector ROI depth',
        'SAM target-mask depth', 'accepted RGB-D observations', len(geometry),
        'median within-support depth standard deviation [mm]',
        median(geometry, 'roi_depth_std_mm'),
        median(geometry, 'mask_depth_std_mm'), 'CONTROLLED_REPLAY',
        'datasets/active_scan/scan_*/frames/view_*',
        'Mask support reduced depth-layer mixing relative to the rectangular '
        'ROI on identical saved depth images.',
        'No dense ground-truth masks; this is stability, not accuracy.',
        'segmentation-mask support'))
    rows.append(comparison_row(
        'Depth localisation', 'raw SAM-mask valid depth',
        'confidence/layer-qualified target depth',
        'schema-2 accepted observations', len(qualified),
        'median within-support depth standard deviation [mm]',
        median(qualified, 'mask_depth_std_mm'),
        median(qualified, 'qualified_depth_std_mm'), 'CONTROLLED_REPLAY',
        'datasets/active_scan schema-2 target_depth/support artifacts',
        'Confidence and component selection removed secondary depth layers '
        'before target geometry entered coverage/reconstruction.',
        'Spatial surface spread is not sensor accuracy; rejected observations '
        'were not persisted.', 'confidence-qualified target depth'))
    resegmentation = data['offline_resegmentation']
    if resegmentation:
        rows.append(comparison_row(
            'Target segmentation', 'captured live SAM2 mask',
            'fresh offline GroundingDINO + SAM2 image mask',
            'same accepted RGB frames', len(resegmentation),
            'total semantic mask pixels',
            sum(row['captured_live_mask_pixels'] for row in resegmentation),
            sum(row['fresh_validated_mask_pixels'] for row in resegmentation),
            'CONTROLLED_REPLAY',
            data['offline_resegmentation_provenance']['source_index'],
            'Fresh image-mode resegmentation narrowed the stored mask, '
            'including a documented shadow-contaminated view.',
            'One seven-view dataset; the resulting reconstruction still '
            'failed and component coherence decreased, so fewer pixels are '
            'not automatically better segmentation accuracy.',
            'captured live mask remains the default; offline resegmentation '
            'is an optional diagnostic reconstruction mode'))
    rows.append(comparison_row(
        'View planning', 'voxel_nbv matched development run',
        'ray_nbv matched development run', 'physical scan sessions', 2,
        'measured covered 10 mm voxels',
        voxel['measured_covered_voxels_10mm'],
        ray['measured_covered_voxels_10mm'], 'MATCHED_RUNS',
        '%s; %s' % (voxel['scan'], ray['scan']),
        'The selected adjacent ray run accumulated more measured target '
        'voxels and wider angular span with fewer near-duplicate transitions.',
        'Different dates and evolving integration; not a randomized paired '
        'experiment. Aggregate historical results are also reported.',
        'ray_nbv for the final integrated configuration'))
    rows.append(comparison_row(
        'Viewpoint prequalification', '100,000-sample capability atlas',
        '2,000,000-sample capability atlas', 'fixed validation rays',
        low_map['validation_rays'],
        'final-supported-ray recall [%]',
        low_map['reference_supported_ray_recall_percent'],
        selected_map['reference_supported_ray_recall_percent'],
        'CONTROLLED_REPLAY', low_map['source'],
        'The selected atlas preserved all support found by the convergence '
        'reference with sub-millisecond median lookup.',
        'Recall is relative to the densest sampled atlas, not ground-truth IK.',
        '2,000,000-sample capability atlas'))
    rows.append(comparison_row(
        'Viewpoint prequalification', 'workspace-only rays',
        'workspace + capability-map rays', 'initial full-sphere ray pools',
        len(feasibility), 'candidates passed toward expensive planning',
        workspace_passed, map_passed, 'HISTORICAL_OBSERVATIONAL',
        '; '.join(row['mission'] for row in feasibility),
        'Capability prequalification reduced the initial expensive-planning '
        'candidate load by %.1f%%.' % (
            (workspace_passed - map_passed) /
            float(workspace_passed) * 100.0),
        'A capability rejection is coarse prequalification, not proof that '
        'Tesseract would fail every rejected ray.',
        'workspace + capability prequalification before Tesseract'))
    curobo = documented['curobo_collision_replay']
    rows.append(comparison_row(
        'Motion collision model', 'Tesseract exact configured model',
        'cuRobo articulated sphere approximation',
        'identical sampled joint states', curobo['N'],
        'conservative false-positive states vs exact Tesseract reference',
        0, curobo['curobo_conservative_false_positives'],
        'CONTROLLED_REPLAY', curobo['source'],
        'The cuRobo model produced no state-level false negatives in this '
        'sample but rejected 18 exact-model-clear states.',
        curobo['limitation'],
        'Tesseract retained as the physically validated final backend; no '
        'claim of planner-algorithm superiority'))
    planner = data.get('planner_benchmark') or {}
    planner_summary = planner.get('summary', {})
    if 'tesseract' in planner_summary and 'curobo' in planner_summary:
        tesseract_summary = planner_summary['tesseract']
        curobo_summary = planner_summary['curobo']
        rows.append(comparison_row(
            'Motion planning', 'Tesseract 0.35.0.6', 'cuRobo 0.7.8',
            'identical recorded planning requests',
            int(tesseract_summary['positive_trial_count']),
            'median end-to-end planner request wall time [s]',
            tesseract_summary['request_wall_sec']['median'],
            curobo_summary['request_wall_sec']['median'],
            'CONTROLLED_REPLAY', planner['source'],
            'cuRobo completed the same command-free planning transactions '
            'faster on this workstation; every reported successful path was '
            'then accepted by the exact Tesseract geometry validator.',
            'Offline proposal timing only. The articulated collision models '
            'differ and cuRobo remains hardware_qualified=false; this does '
            'not establish physical execution superiority.',
            'Tesseract remains the physically qualified final backend; '
            'cuRobo remains an offline candidate pending qualification'))
        planner_trials = planner.get('trials', [])
        successful = {
            backend: [
                row for row in planner_trials
                if row.get('backend') == backend
                and row.get('expected_role') == 'recorded_achieved_geometry'
                and row.get('status') == 'success']
            for backend in ('tesseract', 'curobo')
        }
        combined = {
            backend: statistics.median([
                float(row['request_wall_sec'])
                + float(row['trajectory_duration_sec'])
                for row in successful[backend]])
            for backend in successful
        }
        rows.append(comparison_row(
            'Motion planning', 'Tesseract 0.35.0.6', 'cuRobo 0.7.8',
            'identical recorded planning requests',
            len(successful['tesseract']),
            'median planning wall + scheduled trajectory duration [s]',
            combined['tesseract'], combined['curobo'],
            'CONTROLLED_REPLAY', planner['source'],
            'The large cuRobo planning-time advantage did not reduce the '
            'median offline planning-plus-scheduled-motion proxy because '
            'several cuRobo paths were longer.',
            'This is not measured physical mission time; perception, capture, '
            'settling, startup, shutdown and controller following are absent.',
            'No backend selected from this proxy; retain Tesseract until '
            'supervised physical qualification'))
    if 'robot_pose' in reconstruction and 'multiway_gicp' in reconstruction:
        rows.append(comparison_row(
            'Reconstruction registration', 'robot-pose TSDF',
            'bounded multiway GICP', 'same seven-view scan', 1,
            'median point-to-mesh residual [mm]',
            reconstruction['robot_pose'][
                'median_point_to_mesh_residual_mm'],
            reconstruction['multiway_gicp'][
                'median_point_to_mesh_residual_mm'], 'CONTROLLED_REPLAY',
            'scan_20260821_220003 current-code replay',
            'Bounded multiway registration reduced cross-view residual and '
            'was selected by auto for this dataset.',
            'All modes still failed 35 mm dimensional quality; lower residual '
            'alone is not reconstruction accuracy.',
            'auto selection with bounded candidates'))
    data['method_comparison'] = rows
    ablations = [
        {
            'component': 'semantic mask support',
            'full_configuration': 'SAM-mask target depth',
            'ablated_configuration': 'rectangular ROI depth',
            'experimental_unit': 'accepted RGB-D observations',
            'N': len(geometry),
            'metric': 'median depth standard deviation [mm]',
            'full_result': median(geometry, 'mask_depth_std_mm'),
            'ablated_result': median(geometry, 'roi_depth_std_mm'),
            'comparison_strength': 'CONTROLLED_REPLAY',
            'source_runs': 'all persisted scan frames',
            'interpretation': 'Removing mask support admitted mixed depth.',
            'limitation': 'No labelled pixel ground truth.',
        },
        {
            'component': 'confidence/depth-layer qualification',
            'full_configuration': 'qualified target support',
            'ablated_configuration': 'all valid depth inside SAM mask',
            'experimental_unit': 'schema-2 accepted RGB-D observations',
            'N': len(qualified),
            'metric': 'median depth standard deviation [mm]',
            'full_result': median(qualified, 'qualified_depth_std_mm'),
            'ablated_result': median(qualified, 'mask_depth_std_mm'),
            'comparison_strength': 'CONTROLLED_REPLAY',
            'source_runs': 'schema-2 target-depth/support artifacts',
            'interpretation': 'Removing the gate retained secondary layers.',
            'limitation': 'Rejected candidate observations were not saved.',
        },
        {
            'component': 'capability-map prequalification',
            'full_configuration': 'workspace + capability map',
            'ablated_configuration': 'workspace only',
            'experimental_unit': 'initial full-sphere ray pools',
            'N': len(feasibility),
            'metric': 'rays forwarded after cheap prequalification',
            'full_result': map_passed,
            'ablated_result': workspace_passed,
            'comparison_strength': 'HISTORICAL_OBSERVATIONAL',
            'source_runs': '; '.join(row['mission'] for row in feasibility),
            'interpretation': 'The map removed coarse unsupported directions.',
            'limitation': 'Does not measure counterfactual Tesseract outcomes.',
        },
    ]
    if 'robot_pose' in reconstruction and 'multiway_gicp' in reconstruction:
        ablations.append({
            'component': 'bounded multiway registration',
            'full_configuration': 'bounded multiway GICP',
            'ablated_configuration': 'robot-pose-only fusion',
            'experimental_unit': 'same seven-view scan',
            'N': 1,
            'metric': 'median point-to-mesh residual [mm]',
            'full_result': reconstruction['multiway_gicp'][
                'median_point_to_mesh_residual_mm'],
            'ablated_result': reconstruction['robot_pose'][
                'median_point_to_mesh_residual_mm'],
            'comparison_strength': 'CONTROLLED_REPLAY',
            'source_runs': 'scan_20260821_220003 current-code replay',
            'interpretation': (
                'Removing bounded registration increased cross-view residual.'),
            'limitation': (
                'Both reconstructions failed the 35 mm dimensional quality '
                'gate; residual alone is not reconstruction accuracy.'),
        })
    data['ablation_study'] = ablations


def build_final_selection(data):
    """Create the concise thesis-ready final method-selection table."""
    geometry = data['capture_geometry']
    qualified = [
        row for row in geometry
        if row['qualified_valid_pixels'] is not None]
    prompt = data['documented_evidence']['semantic_prompt_replay']
    voxel = row_by_scan(data['scan_runs'], 'scan_20260820_230845')
    ray = row_by_scan(data['scan_runs'], 'scan_20260821_220003')
    planner = data.get('planner_benchmark', {}).get('summary', {})
    planner_evidence = (
        'Tesseract is physically integrated; cuRobo remains '
        'hardware_qualified=false.')
    if 'tesseract' in planner and 'curobo' in planner:
        planner_evidence = (
            'Controlled replay median wall time %.3f s Tesseract vs %.3f s '
            'cuRobo; exact-validation pass rate %.1f%% for both successful '
            'path sets. cuRobo remains hardware_qualified=false.' % (
                planner['tesseract']['request_wall_sec']['median'],
                planner['curobo']['request_wall_sec']['median'],
                100.0 * planner['tesseract'][
                    'exact_validated_success_rate']))
    return [
        {
            'Subsystem': 'Semantic acquisition',
            'Alternatives Evaluated': (
                'mixed target/obstacle prompt; target-only prompt'),
            'Evaluation Metric': 'confirmed-positive availability/confidence',
            'Best Performing Tested Approach': (
                'target-only GroundingDINO prompt'),
            'Evidence': (
                '63/63 accepted; mean confidence %.4f vs %.4f' % (
                    prompt['target_only_prompt_mean_confidence'],
                    prompt['former_mixed_prompt_mean_confidence'])),
            'Reason Retained': (
                'More reliable semantic handoff without mixing obstacle '
                'vocabulary into the target query.'),
        },
        {
            'Subsystem': 'Target depth support',
            'Alternatives Evaluated': (
                'rectangular ROI; SAM mask; confidence/layer-qualified mask'),
            'Evaluation Metric': 'within-support depth spread and contamination',
            'Best Performing Tested Approach': (
                'confidence/layer-qualified target-mask depth'),
            'Evidence': (
                'median depth std %.1f -> %.1f -> %.1f mm; %d/%d schema-2 '
                'frames contained multiple candidate depth layers' % (
                    median(geometry, 'roi_depth_std_mm'),
                    median(geometry, 'mask_depth_std_mm'),
                    median(qualified, 'qualified_depth_std_mm'),
                    sum((row.get('depth_layer_candidate_count') or 0) > 1
                        for row in qualified), len(qualified))),
            'Reason Retained': (
                'Prevents background/secondary layers from directly driving '
                'tracking, coverage, and reconstruction.'),
        },
        {
            'Subsystem': 'Temporal target estimate',
            'Alternatives Evaluated': (
                'raw updates; near-static filtered/gated prediction'),
            'Evaluation Metric': 'software fault replay only',
            'Best Performing Tested Approach': (
                'near-static Kalman estimate with innovation gating'),
            'Evidence': (
                'Characterization tests reject corrupted updates and bound '
                'prediction-only operation; no continuous empirical trace.'),
            'Reason Retained': (
                'Fail-closed continuity contract, but quantitative physical '
                'tracking superiority remains unproven.'),
        },
        {
            'Subsystem': 'View planning',
            'Alternatives Evaluated': (
                'fixed/dome historical routes; voxel_nbv; ray_nbv'),
            'Evaluation Metric': (
                'measured voxels, novel gain, angular diversity, redundancy'),
            'Best Performing Tested Approach': (
                'ray_nbv in the selected matched development comparison'),
            'Evidence': (
                '%d vs %d covered voxels; %.1f vs %.1f deg azimuth span' % (
                    ray['measured_covered_voxels_10mm'],
                    voxel['measured_covered_voxels_10mm'],
                    ray['azimuth_span_deg'], voxel['azimuth_span_deg'])),
            'Reason Retained': (
                'Supports full-sphere directions, bounded standoff, '
                'size-aware envelopes, and capability prequalification.'),
        },
        {
            'Subsystem': 'Viewpoint feasibility',
            'Alternatives Evaluated': (
                'workspace-only; capability atlas; exact Tesseract checks'),
            'Evaluation Metric': 'reference recall and candidate reduction',
            'Best Performing Tested Approach': (
                '2M-sample atlas as coarse gate, Tesseract as exact authority'),
            'Evidence': (
                '100%% final-support recall; 0.280 ms median query; initial '
                'full-sphere survival 263/2520 rays across seven pools.'),
            'Reason Retained': (
                'Reduces expensive attempts without treating coarse support '
                'as collision/path proof.'),
        },
        {
            'Subsystem': 'Motion planning',
            'Alternatives Evaluated': (
                'Tesseract physical integration; cuRobo branch exploration'),
            'Evaluation Metric': 'qualification status and collision replay',
            'Best Performing Tested Approach': (
                'Tesseract for the physically validated final system'),
            'Evidence': planner_evidence,
            'Reason Retained': (
                'Strongest physically validated integration evidence; no '
                'claim of algorithmic superiority.'),
        },
        {
            'Subsystem': 'Reconstruction registration',
            'Alternatives Evaluated': (
                'robot pose; bounded sequential/multiway GICP; constrained '
                'superposition; static-scene pose graph'),
            'Evaluation Metric': 'residual, dimensions, components, quality gate',
            'Best Performing Tested Approach': (
                'bounded auto-selection, not one universally fixed mode'),
            'Evidence': (
                'Current seven-view replay selected multiway GICP, but every '
                'mode remained FAIL against 35 mm dimensions.'),
            'Reason Retained': (
                'Permits bounded residual correction while preserving the '
                'robot-pose baseline and failing closed.'),
        },
    ]


def unsupported_comparisons():
    """List comparisons that current artifacts cannot support scientifically."""
    return [
        {
            'comparison': 'raw target position vs Kalman/gated tracking',
            'reason': (
                'No continuous synchronized raw-measurement and filtered-state '
                'sequence is stored; accepted frame metadata is too sparse.'),
            'required_data': (
                'timestamped raw Target3D, filtered TrackedTarget, validity, '
                'innovation score, camera motion state, and optional ground truth'),
        },
        {
            'comparison': 'segmentation IoU/Dice/precision/recall',
            'reason': 'No manually labelled pixel ground truth exists.',
            'required_data': 'independent frame-level target masks',
        },
        {
            'comparison': 'detector precision/recall/false-positive rate',
            'reason': (
                'The 63-image replay is confirmed-positive; the absent-scene '
                'sample is insufficient for a false-positive estimate.'),
            'required_data': 'labelled positive and negative acquisition set',
        },
        {
            'comparison': '20-frame burst vs single-frame temporal noise',
            'reason': (
                'Only the aggregated target depth is persisted, not all twenty '
                'per-pixel input frames needed for paired temporal error.'),
            'required_data': 'raw synchronized burst frames and static reference',
        },
        {
            'comparison': 'quality gates: rejected vs admitted observations',
            'reason': (
                'Rejected observations are intentionally not committed to the '
                'scan dataset, so false-admission counterfactuals cannot be '
                'reconstructed from accepted frames alone.'),
            'required_data': 'diagnostic-only rejected-frame archive',
        },
        {
            'comparison': 'full simple pipeline vs final integrated pipeline',
            'reason': (
                'Historical fixed-route raw inputs are absent and no matched '
                'current full mission completed the present convergence plus '
                'reconstruction quality criteria.'),
            'required_data': 'paired runs or replayable complete candidate streams',
        },
    ]


def evidence_register(data):
    """Return source-level provenance for the workbook."""
    capability = data['capability_convergence'][0]
    sources = [
        ('E01', 'Prompt replay summary',
         'docs/historical/supervised_workflow_handoff_2026_07_30.md',
         'CONTROLLED_REPLAY'),
        ('E02', 'Immutable accepted RGB-D captures',
         'datasets/active_scan/scan_*/frames', 'CONTROLLED_REPLAY'),
        ('E03', 'Persisted scan policy and achieved-pose histories',
         'datasets/active_scan/scan_*', 'HISTORICAL_OBSERVATIONAL'),
        ('E04', 'Ray generation/prequalification/Tesseract diagnostics',
         'datasets/ray_diagnostics', 'HISTORICAL_OBSERVATIONAL'),
        ('E05', 'Capability atlas convergence benchmark',
         capability['source'], 'CONTROLLED_REPLAY'),
        ('E06', 'Seven-frame offline mask replay',
         data['offline_resegmentation_provenance']['source_index'],
         'CONTROLLED_REPLAY'),
        ('E07', 'Fresh same-dataset reconstruction replay',
         '/tmp/piper_chapter5_reconstruction (metrics copied into analysis)',
         'CONTROLLED_REPLAY'),
        ('E08', 'cuRobo branch collision qualification',
         'git curobo-integration @ cca6844', 'CONTROLLED_REPLAY'),
        ('E09', 'Historical fixed/dome scan implementation',
         'docs/historical/active_scan_notes.md; '
         'docs/historical/session_resume_2026-07-30.md',
         'HISTORICAL_OBSERVATIONAL'),
    ]
    if data.get('planner_benchmark'):
        sources.append((
            'E10', 'Same-request Tesseract/curobo planner replay with exact '
            'path revalidation',
            data['planner_benchmark']['source'], 'CONTROLLED_REPLAY'))
    return [
        {'evidence_id': item[0], 'description': item[1], 'source': item[2],
         'comparison_strength': item[3]} for item in sources]


def workbook_sheets(data):
    """Return ordered workbook sheet names and row mappings."""
    perception = []
    prompt = data['documented_evidence']['semantic_prompt_replay']
    perception.extend([
        {
            'comparison': 'GroundingDINO prompt replay',
            'method': 'former mixed prompt',
            'N': prompt['N'],
            'metric': 'mean confidence (not accuracy)',
            'result': prompt['former_mixed_prompt_mean_confidence'],
            'comparison_strength': prompt['comparison_strength'],
            'source': prompt['source'],
            'limitation': prompt['limitation'],
        },
        {
            'comparison': 'GroundingDINO prompt replay',
            'method': 'target-only prompt',
            'N': prompt['N'],
            'metric': 'mean confidence (not accuracy)',
            'result': prompt['target_only_prompt_mean_confidence'],
            'comparison_strength': prompt['comparison_strength'],
            'source': prompt['source'],
            'limitation': prompt['limitation'],
        },
    ])
    for row in data['offline_resegmentation']:
        item = dict(row)
        item['comparison'] = 'captured live mask vs fresh image-mode mask'
        perception.append(item)
    gate_rows = []
    for row in data['capture_geometry']:
        gate_rows.append({
            'scan': row['scan'],
            'frame': row['frame'],
            'raw_mask_valid_pixels': row['mask_valid_pixels'],
            'qualified_target_valid_pixels': row['qualified_valid_pixels'],
            'retained_fraction': row[
                'qualified_retained_fraction_of_mask'],
            'raw_mask_depth_std_mm': row['mask_depth_std_mm'],
            'qualified_depth_std_mm': row['qualified_depth_std_mm'],
            'depth_layer_candidate_count': row[
                'depth_layer_candidate_count'],
            'source': row['source_metadata'],
        })
    planner = [
        {
            'planner_or_model': 'Tesseract exact configured collision model',
            'evaluation': 'physical integration qualification',
            'N': '',
            'metric': 'qualification status',
            'result': 'physically validated final backend',
            'comparison_strength': 'HISTORICAL_OBSERVATIONAL',
            'limitation': (
                'Physical qualification is not an algorithmic timing '
                'comparison; see the controlled replay rows below.'),
        },
        {
            'planner_or_model': 'cuRobo articulated sphere approximation',
            'evaluation': 'same-state collision classification vs Tesseract',
            'N': 2004,
            'metric': 'state-level false negatives / conservative positives',
            'result': '0 / 18',
            'comparison_strength': 'CONTROLLED_REPLAY',
            'limitation': data['documented_evidence'][
                'curobo_collision_replay']['limitation'],
        },
    ]
    benchmark = data.get('planner_benchmark') or {}
    for row in benchmark.get('trials', []):
        planner.append({
            'planner_or_model': row.get('backend'),
            'evaluation': row.get('fixture'),
            'N': 1,
            'metric': 'end-to-end request wall time [s]',
            'result': row.get('request_wall_sec'),
            'status': row.get('status'),
            'expected_role': row.get('expected_role'),
            'exact_collision_validation': row.get(
                'exact_collision_validation'),
            'trajectory_duration_sec': row.get('trajectory_duration_sec'),
            'joint_space_path_length_rad': row.get(
                'joint_space_path_length_rad'),
            'comparison_strength': 'CONTROLLED_REPLAY',
            'source': benchmark.get('source'),
            'limitation': (
                'Command-free proposal timing; cuRobo remains physically '
                'unqualified and uses articulated collision spheres.'),
        })
    return collections.OrderedDict([
        ('Method_Comparison', data['method_comparison']),
        ('Ablation_Study', data['ablation_study']),
        ('Perception_Comparison', perception),
        ('Geometry_Comparison', data['capture_geometry']),
        ('NBV_Comparison', data['scan_runs']),
        ('Feasibility_Comparison',
         data['capability_convergence'] + data['feasibility_initial_pools']),
        ('Planner_Comparison', planner),
        ('Gate_Comparison', gate_rows),
        ('Reconstruction_Comparison', data['reconstruction_replay']),
        ('Final_Method_Selection', data['final_method_selection']),
        ('Scan_Run_Summary', data['scan_runs']),
        ('Evidence_Register', data['evidence_register']),
        ('Unsupported_Comparisons', data['unsupported_comparisons']),
    ])


def uno_property(name, value):
    """Construct a UNO property without importing generated helper modules."""
    from com.sun.star.beans import PropertyValue
    result = PropertyValue()
    result.Name = name
    result.Value = value
    return result


def available_port():
    """Reserve and release one local TCP port for the isolated office process."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.bind(('127.0.0.1', 0))
        return int(stream.getsockname()[1])


def write_workbook(path, sheets):
    """Write a formatted multi-sheet XLSX through the installed LibreOffice."""
    import uno
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = Path(tempfile.mkdtemp(prefix='piper_chapter5_lo_'))
    port = available_port()
    command = [
        'libreoffice', '--headless', '--nologo', '--nodefault', '--norestore',
        '--nofirststartwizard',
        '-env:UserInstallation=%s' % uno.systemPathToFileUrl(str(profile)),
        '--accept=socket,host=127.0.0.1,port=%d;urp;' % port,
    ]
    office = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    document = None
    try:
        local_context = uno.getComponentContext()
        resolver = local_context.ServiceManager.createInstanceWithContext(
            'com.sun.star.bridge.UnoUrlResolver', local_context)
        context = None
        for _ in range(100):
            if office.poll() is not None:
                raise RuntimeError('LibreOffice exited before workbook creation')
            try:
                context = resolver.resolve(
                    'uno:socket,host=127.0.0.1,port=%d;urp;'
                    'StarOffice.ComponentContext' % port)
                break
            except Exception:
                time.sleep(0.1)
        if context is None:
            raise RuntimeError('unable to connect to isolated LibreOffice')
        desktop = context.ServiceManager.createInstanceWithContext(
            'com.sun.star.frame.Desktop', context)
        document = desktop.loadComponentFromURL(
            'private:factory/scalc', '_blank', 0, ())
        workbook = document.getSheets()
        default_name = workbook.getElementNames()[0]
        first = True
        for sheet_name, rows in sheets.items():
            if len(sheet_name) > 31:
                raise ValueError('Excel sheet name is too long: %s' % sheet_name)
            if first:
                workbook.getByName(default_name).setName(sheet_name)
                sheet = workbook.getByName(sheet_name)
                first = False
            else:
                workbook.insertNewByName(sheet_name, len(workbook.getElementNames()))
                sheet = workbook.getByName(sheet_name)
            rows = list(rows)
            columns = []
            for row in rows:
                for key in row:
                    if key not in columns:
                        columns.append(key)
            if not columns:
                columns = ['note']
                rows = [{'note': 'No genuine comparison data available.'}]
            for column, key in enumerate(columns):
                cell = sheet.getCellByPosition(column, 0)
                cell.setString(str(key))
            header = sheet.getCellRangeByPosition(0, 0, len(columns) - 1, 0)
            header.CharWeight = 150.0
            header.CharColor = 0xFFFFFF
            header.CellBackColor = 0x1F4E78
            header.IsTextWrapped = True
            for row_index, row in enumerate(rows, 1):
                for column, key in enumerate(columns):
                    value = row.get(key)
                    cell = sheet.getCellByPosition(column, row_index)
                    if isinstance(value, bool):
                        cell.setString('TRUE' if value else 'FALSE')
                    elif isinstance(value, (int, float)) and not isinstance(value, bool):
                        if math.isfinite(float(value)):
                            cell.setValue(float(value))
                        else:
                            cell.setString('')
                    elif value is None:
                        cell.setString('')
                    elif isinstance(value, (dict, list, tuple)):
                        cell.setString(json.dumps(value, sort_keys=True))
                    else:
                        cell.setString(str(value))
            used = sheet.getCellRangeByPosition(
                0, 0, len(columns) - 1, len(rows))
            used.IsTextWrapped = True
            for column, key in enumerate(columns):
                width = max(2800, min(14000, 500 + 230 * len(str(key))))
                if key in (
                        'interpretation', 'limitation', 'source_runs', 'source',
                        'Evidence', 'Reason Retained', 'required_data'):
                    width = 12000
                sheet.getColumns().getByIndex(column).Width = width
            document.getCurrentController().setActiveSheet(sheet)
            document.getCurrentController().freezeAtPosition(0, 1)
        document.getCurrentController().setActiveSheet(
            workbook.getByName('Method_Comparison'))
        document.storeAsURL(
            uno.systemPathToFileUrl(str(output)),
            (uno_property('FilterName', 'Calc MS Excel 2007 XML'),
             uno_property('Overwrite', True)))
        document.close(True)
        document = None
    finally:
        if document is not None:
            try:
                document.close(True)
            except Exception:
                pass
        office.terminate()
        try:
            office.wait(timeout=5)
        except subprocess.TimeoutExpired:
            office.kill()
            office.wait(timeout=5)
        shutil.rmtree(str(profile), ignore_errors=True)


def git_state(root=REPOSITORY):
    """Capture the exact code state used for the analysis."""
    head = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=str(root), text=True).strip()
    branch = subprocess.check_output(
        ['git', 'branch', '--show-current'], cwd=str(root),
        text=True).strip()
    status = subprocess.check_output(
        ['git', 'status', '--short'], cwd=str(root), text=True)
    return {
        'root': str(Path(root).resolve()),
        'head': head,
        'branch': branch,
        'worktree_dirty': bool(status.strip()),
    }


def main():
    """Build the analysis JSON and workbook without touching source data."""
    global EVIDENCE_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--evidence-root', default=str(REPOSITORY),
        help='Read-only repository/worktree containing saved datasets')
    parser.add_argument(
        '--reconstruction-replay-dir',
        default='/tmp/piper_chapter5_reconstruction',
        help='Directory containing current same-dataset reconstruction reports')
    parser.add_argument(
        '--json-output',
        default=str(REPOSITORY / 'docs' / 'chapter5_method_comparison.json'))
    parser.add_argument(
        '--workbook-output',
        default=str(REPOSITORY / 'docs' / 'chapter5_method_comparison.xlsx'))
    parser.add_argument(
        '--planner-benchmark', default=str(
            REPOSITORY / 'benchmarks' / 'planner_backends' / 'results' /
            '20260901_tabletop_controlled_replay' /
            'planner_benchmark.json'))
    args = parser.parse_args()
    EVIDENCE_ROOT = Path(args.evidence_root).resolve()
    data = {
        'schema_version': 1,
        'analysis_scope': (
            'Command-free comparison of implemented/tested PiPER acquisition '
            'methods using repository evidence available on 2026-09-01.'),
        'git': git_state(REPOSITORY),
        'evidence_git': git_state(EVIDENCE_ROOT),
        'capture_geometry': analyse_capture_geometry(),
        'scan_runs': analyse_scan_runs(),
        'documented_evidence': documented_evidence(),
        'capability_convergence': analyse_capability_convergence(),
        'reconstruction_replay': analyse_reconstruction(
            args.reconstruction_replay_dir),
        'planner_benchmark': analyse_planner_benchmark(
            args.planner_benchmark),
    }
    resegment, provenance = analyse_offline_resegmentation()
    data['offline_resegmentation'] = resegment
    data['offline_resegmentation_provenance'] = provenance
    feasibility, tesseract = analyse_feasibility_events()
    data['feasibility_initial_pools'] = feasibility
    data['tesseract_ray_outcomes'] = tesseract
    build_comparisons(data)
    data['final_method_selection'] = build_final_selection(data)
    data['unsupported_comparisons'] = unsupported_comparisons()
    data['evidence_register'] = evidence_register(data)
    output = Path(args.json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    write_workbook(args.workbook_output, workbook_sheets(data))
    print('analysis:', output)
    print('workbook:', Path(args.workbook_output))
    print('captures:', len(data['capture_geometry']))
    print('scan runs:', len(data['scan_runs']))
    print('method comparisons:', len(data['method_comparison']))


if __name__ == '__main__':
    main()
