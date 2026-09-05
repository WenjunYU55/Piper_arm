"""Offline metrics derived from immutable PiPER campaign evidence."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree


def finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def stamp_delta_sec(end_ns: Any, start_ns: Any) -> Optional[float]:
    try:
        delta = (int(end_ns) - int(start_ns)) / 1e9
    except (TypeError, ValueError):
        return None
    return delta if delta >= 0.0 and math.isfinite(delta) else None


def circular_span_deg(angles: Sequence[float]) -> float:
    """Return the minimum circular arc containing all finite angles."""
    values = sorted(float(value) % 360.0 for value in angles if finite_float(value) is not None)
    if len(values) < 2:
        return 0.0
    gaps = [values[index + 1] - values[index] for index in range(len(values) - 1)]
    gaps.append(values[0] + 360.0 - values[-1])
    return 360.0 - max(gaps)


def camera_angles(camera_m: Sequence[float], centre_m: Sequence[float]) -> Tuple[float, float]:
    vector = np.asarray(camera_m, dtype=float) - np.asarray(centre_m, dtype=float)
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        return float('nan'), float('nan')
    azimuth = math.degrees(math.atan2(float(vector[1]), float(vector[0]))) % 360.0
    elevation = math.degrees(math.asin(float(np.clip(vector[2] / length, -1.0, 1.0))))
    return azimuth, elevation


def transform_target_centre(frame: Dict[str, Any]) -> Optional[np.ndarray]:
    matrix = frame.get('base_from_camera_4x4')
    point = frame.get('target_point_camera_m')
    try:
        transform = np.asarray(matrix, dtype=float).reshape(4, 4)
        target = np.asarray(point, dtype=float).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(transform).all() or not np.isfinite(target).all():
        return None
    return (transform @ np.r_[target, 1.0])[:3]


def mission_target_centre(frames: Sequence[Dict[str, Any]]) -> Optional[np.ndarray]:
    centres = [centre for centre in (transform_target_centre(frame) for frame in frames) if centre is not None]
    if not centres:
        return None
    return np.median(np.vstack(centres), axis=0)


def _resolve_artifact(dataset: Path, text: str) -> Optional[Path]:
    if not text:
        return None
    path = Path(text)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((dataset / path, dataset / 'frames' / path.name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def frame_cloud_base(frame: Dict[str, Any], dataset: Path, stride: int = 1) -> np.ndarray:
    """Project persisted qualified colour-depth support into ``base_link``."""
    depth_path = _resolve_artifact(dataset, frame.get('target_depth_png_file_path', ''))
    support_path = _resolve_artifact(dataset, frame.get('target_support_mask_file_path', ''))
    if depth_path is None:
        return np.empty((0, 3), dtype=float)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    support = cv2.imread(str(support_path), cv2.IMREAD_GRAYSCALE) if support_path else None
    if depth is None or depth.ndim != 2:
        return np.empty((0, 3), dtype=float)
    valid = np.isfinite(depth) & (depth > 0)
    if support is not None and support.shape == depth.shape:
        valid &= support > 0
    rows, columns = np.nonzero(valid)
    if stride > 1:
        rows, columns = rows[::stride], columns[::stride]
    if not len(rows):
        return np.empty((0, 3), dtype=float)
    camera_info = frame.get('camera_info') or {}
    try:
        intrinsic = np.asarray(camera_info['k'], dtype=float).reshape(3, 3)
        distortion = np.asarray(camera_info.get('d', []), dtype=float)
        transform = np.asarray(frame['base_from_camera_4x4'], dtype=float).reshape(4, 4)
    except (KeyError, TypeError, ValueError):
        return np.empty((0, 3), dtype=float)
    pixels = np.column_stack((columns, rows)).astype(np.float64).reshape(-1, 1, 2)
    normal = cv2.undistortPoints(pixels, intrinsic, distortion).reshape(-1, 2)
    # Persisted target-depth PNG is documented as millimetres.
    z = depth[rows, columns].astype(np.float64) * 0.001
    camera = np.column_stack((normal[:, 0] * z, normal[:, 1] * z, z))
    homogeneous = np.column_stack((camera, np.ones(len(camera))))
    base = (transform @ homogeneous.T).T[:, :3]
    return base[np.isfinite(base).all(axis=1)]


def cube_surface_points(dimensions_mm=(35.0, 35.0, 35.0), spacing_mm=1.0) -> np.ndarray:
    dimensions = np.asarray(dimensions_mm, dtype=float) * 0.001
    axes = [np.linspace(-value / 2.0, value / 2.0, max(2, int(round(value * 1000.0 / spacing_mm)) + 1)) for value in dimensions]
    faces = []
    for fixed_axis in range(3):
        free = [axis for axis in range(3) if axis != fixed_axis]
        grid_a, grid_b = np.meshgrid(axes[free[0]], axes[free[1]], indexing='ij')
        for sign in (-1.0, 1.0):
            face = np.zeros((grid_a.size, 3), dtype=float)
            face[:, fixed_axis] = sign * dimensions[fixed_axis] / 2.0
            face[:, free[0]] = grid_a.ravel()
            face[:, free[1]] = grid_b.ravel()
            faces.append(face)
    # Deduplicate edge/corner samples so percentages have one physical denominator.
    return np.unique(np.round(np.vstack(faces), 9), axis=0)


def cumulative_cube_coverage(
        frame_clouds: Sequence[np.ndarray], centre_m: Sequence[float],
        dimensions_mm=(35.0, 35.0, 35.0), tolerance_mm=2.0) -> List[Dict[str, Any]]:
    reference = cube_surface_points(dimensions_mm) + np.asarray(centre_m, dtype=float)
    observed = np.zeros(len(reference), dtype=bool)
    output = []
    for index, cloud in enumerate(frame_clouds):
        new_count = 0
        if len(cloud):
            distances, _ = cKDTree(cloud).query(reference, k=1)
            now = distances <= float(tolerance_mm) * 0.001
            new_count = int(np.count_nonzero(now & ~observed))
            observed |= now
        output.append({
            'capture_index': index,
            'reference_surface_points': int(len(reference)),
            'new_surface_points': new_count,
            'new_surface_fraction': new_count / len(reference),
            'cumulative_surface_points': int(np.count_nonzero(observed)),
            'cumulative_coverage_fraction': float(np.mean(observed)),
            'coverage_tolerance_mm': float(tolerance_mm),
        })
    return output


def cloud_dimension_metrics(cloud: np.ndarray, known_dimensions_mm=(35.0, 35.0, 35.0)) -> Dict[str, Any]:
    if len(cloud) < 10:
        return {'point_count': int(len(cloud)), 'qualified': False}
    low, high = np.percentile(cloud, [1.0, 99.0], axis=0)
    observed = (high - low) * 1000.0
    known = np.asarray(known_dimensions_mm, dtype=float)
    error = np.abs(observed - known)
    return {
        'point_count': int(len(cloud)),
        'observed_x_mm': float(observed[0]),
        'observed_y_mm': float(observed[1]),
        'observed_z_mm': float(observed[2]),
        'known_x_mm': float(known[0]),
        'known_y_mm': float(known[1]),
        'known_z_mm': float(known[2]),
        'mean_absolute_dimension_error_mm': float(np.mean(error)),
        'maximum_absolute_dimension_error_mm': float(np.max(error)),
        'qualified': True,
        'extent_estimator': '1st-to-99th percentile in base_link',
    }


def summarize_evidence(evidence: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Create normalized workbook rows for one mission evidence record."""
    submission = evidence.get('submission', {})
    expected = submission.get('expected_trial', {})
    task_id = str(evidence.get('task_id', ''))
    backend = str(evidence.get('planner_backend', ''))
    common = {
        'campaign_id': evidence.get('campaign_id'),
        'trial_id': evidence.get('trial_id'),
        'pair_index': expected.get('pair_index'),
        'task_id': task_id,
        'planner_backend': backend,
        'target_x_m': expected.get('x_m'),
        'target_y_m': expected.get('y_m'),
        'target_z_m': expected.get('z_m'),
        'evidence_class': evidence.get('evidence_class'),
        'matches_schedule': evidence.get('matches_schedule'),
        'code_branch': submission.get(
            'configuration_snapshot', {}).get('git', {}).get('branch'),
        'code_commit': submission.get(
            'configuration_snapshot', {}).get('git', {}).get('commit'),
        'code_dirty': submission.get(
            'configuration_snapshot', {}).get('git', {}).get('dirty'),
    }
    mission = evidence.get('mission_result', {})
    terminal = evidence.get('terminal', {})
    action = mission.get('action_summary', {}) if isinstance(mission, dict) else {}
    feature = action.get('scan_feature_coverage', {}) if isinstance(action, dict) else {}
    surface = feature.get('surface_coverage', {}) if isinstance(feature, dict) else {}
    phase = evidence.get('phase_timing', {})
    configuration_files = submission.get(
        'configuration_snapshot', {}).get('files', [])
    planner_model = submission.get(
        'configuration_snapshot', {}).get(
            'planner_models', {}).get(backend, {})
    if backend == 'curobo':
        selected_tokens = (
            'motion_planning/curobo/model/piper_collision_spheres.yaml',
            'piper_description.urdf', 'collision_environment.yaml')
        qualification = (
            'hardware-qualified: %s'
            % planner_model.get(
                'qualification_scope', 'scope not recorded')
            if planner_model.get('hardware_qualified') is True
            else 'hardware qualification not recorded')
    else:
        selected_tokens = (
            'piper_tesseract_foxy/model/collision_model',
            'piper_tesseract_foxy/model/piper',
            'piper_description.urdf', 'collision_environment.yaml')
        qualification = 'current supervised Tesseract reference'
    selected_collision_files = [
        item for item in configuration_files
        if item.get('available') and any(
            token in item.get('path', '') for token in selected_tokens)]
    run = dict(common)
    run.update({
        'outcome': terminal.get('outcome', mission.get('outcome')),
        'reason': terminal.get('reason', mission.get('reason')),
        'failure_code': terminal.get('failure_code', mission.get('failure_code')),
        'safe_shutdown': terminal.get('safe_shutdown', mission.get('safe_shutdown')),
        'accepted_captures': terminal.get('capture_count', mission.get('capture_count', len(evidence.get('frames', [])))),
        'total_mission_sec': phase.get('total_elapsed_sec') if isinstance(phase, dict) else None,
        'dataset_path': evidence.get('dataset_path'),
        'collision_model_configuration': ', '.join(
            item.get('path', '') + ':' + item.get('sha256', '')[:12]
            for item in selected_collision_files),
        'collision_model_qualification': qualification,
        'collision_model_hardware_qualified': planner_model.get(
            'hardware_qualified'),
        'collision_model_qualification_date': planner_model.get(
            'qualification_date'),
        'collision_model_qualification_scope': planner_model.get(
            'qualification_scope'),
        'collision_model_qualification_basis': planner_model.get(
            'qualification_basis'),
        'collision_model_qualified_floor_profile': planner_model.get(
            'qualified_floor_profile'),
        'collision_model_qualified_free_motion_speed_percent': (
            planner_model.get('qualified_free_motion_speed_percent')),
        'collision_model_qualified_contact_speed_percent': (
            planner_model.get('qualified_contact_speed_percent')),
        'collision_model_conservative_geometry': planner_model.get(
            'conservative_geometry'),
        'system_accepted_achieved_views': feature.get('accepted_achieved_views') if isinstance(feature, dict) else None,
        'system_azimuth_span_deg': feature.get('azimuth_span_deg') if isinstance(feature, dict) else None,
        'system_elevation_span_deg': feature.get('elevation_span_deg') if isinstance(feature, dict) else None,
        'system_covered_10mm_voxels': surface.get('covered_voxels') if isinstance(surface, dict) else None,
        'system_surface_converged': surface.get('converged') if isinstance(surface, dict) else None,
        'system_surface_sufficient': surface.get('sufficient') if isinstance(surface, dict) else None,
    })
    rows = {'Runs': [run], 'Acquisition': [], 'Capture_Timeline': [], 'Coverage': [], 'Planning': [], 'PointCloud': [], 'Phase_Timing': []}

    acquisition = dict(common)
    first = evidence.get('first_acquisition', {})
    acquisition.update({
        'request_id': first.get('request_id'),
        'request_reason': first.get('request_reason'),
        'requested_target_label': first.get('requested_target_label'),
        'target_profile': first.get('target_profile'),
        'target_prompt': first.get('target_prompt'),
        'target_source': first.get('target_source'),
        'obstacle_count': first.get('obstacle_count'),
        'grounding_dino_confidence': first.get('grounding_dino_confidence'),
        'sam2_score': first.get('sam2_score'),
        'mask_area_px': first.get('mask_area_px'),
        'valid_depth_ratio': first.get('valid_depth_ratio'),
        'target_depth_m': first.get('median_depth_m'),
        'depth_layer_candidate_count': first.get('depth_layer_candidate_count'),
        'note': 'Confidence values are not detection accuracy.',
    })
    rows['Acquisition'].append(acquisition)

    frames = evidence.get('frames', [])
    centre = mission_target_centre(frames)
    frame_clouds = [frame_cloud_base(frame, Path(evidence.get('dataset_path', '.'))) for frame in frames]
    coverages = cumulative_cube_coverage(frame_clouds, centre) if centre is not None else [{} for _ in frames]
    azimuths, elevations = [], []
    previous_capture_ns = None
    submitted_ns = submission.get('submitted_wall_time_ns')
    for position, (frame, coverage, cloud) in enumerate(zip(frames, coverages, frame_clouds)):
        capture_ns = frame.get('capture_timestamp_ns')
        transaction_ns = frame.get('capture_transaction_start_ns')
        camera = frame.get('camera_position_m')
        azimuth = elevation = None
        if centre is not None and isinstance(camera, (list, tuple)) and len(camera) == 3:
            azimuth, elevation = camera_angles(camera, centre)
            if math.isfinite(azimuth):
                azimuths.append(azimuth)
            if math.isfinite(elevation):
                elevations.append(elevation)
        capture = dict(common)
        capture.update({
            'capture_index': frame.get('frame_index', position),
            'capture_timestamp_ns': capture_ns,
            'submit_to_capture_sec': stamp_delta_sec(capture_ns, submitted_ns),
            'capture_transaction_sec': stamp_delta_sec(capture_ns, transaction_ns),
            'time_since_previous_capture_sec': stamp_delta_sec(capture_ns, previous_capture_ns),
            'azimuth_deg': azimuth,
            'elevation_deg': elevation,
            'mask_area_px': frame.get('mask_area_px'),
            'valid_depth_ratio': frame.get('valid_depth_ratio'),
            'raw_mask_depth_stddev_mm': frame.get('raw_mask_depth_stddev_mm'),
            'qualified_depth_stddev_mm': frame.get('qualified_depth_stddev_mm'),
            'depth_layer_candidate_count': frame.get('depth_layer_candidate_count'),
            'scan_quality_score': frame.get('scan_quality_score'),
            'scan_quality_label': frame.get('scan_quality_label'),
            'occlusion_state': frame.get('occlusion_state'),
        })
        rows['Capture_Timeline'].append(capture)
        coverage_row = dict(common)
        coverage_row.update(coverage)
        coverage_row.update({
            'capture_index': frame.get('frame_index', position),
            'azimuth_deg': azimuth,
            'elevation_deg': elevation,
            'cumulative_azimuth_span_deg': circular_span_deg(azimuths),
            'cumulative_elevation_span_deg': (
                max(elevations) - min(elevations)
                if len(elevations) > 1 else 0.0),
            'cumulative_uncovered_fraction': (
                1.0 - coverage.get('cumulative_coverage_fraction', 0.0)),
            'system_new_surface_fraction': (
                surface.get('per_view_novel_fraction', [])[position]
                if isinstance(surface, dict)
                and position < len(surface.get('per_view_novel_fraction', []))
                else None),
        })
        rows['Coverage'].append(coverage_row)
        planning = dict(common)
        diagnostics = frame.get('candidate_diagnostics', {})
        planning.update({
            'record_type': 'capture_candidate',
            'capture_index': frame.get('frame_index', position),
            **(diagnostics if isinstance(diagnostics, dict) else {}),
        })
        rows['Planning'].append(planning)
        previous_capture_ns = capture_ns

    ray_diagnostics = evidence.get('ray_diagnostics', {})
    for event_index, event in enumerate(
            ray_diagnostics.get('planning_events', [])
            if isinstance(ray_diagnostics, dict) else []):
        if not isinstance(event, dict):
            continue
        metrics = event.get('metrics', {})
        planning = dict(common)
        planning.update({
            'record_type': 'ray_event',
            'request_id': event.get('request_id', ''),
            'request_status': event.get('request_status', ''),
            'plan_kind': event.get('plan_kind', ''),
            'event_index': event_index,
            'event_timestamp_ns': event.get('timestamp_ns'),
            'event_message': event.get('message'),
            'planner_revision': event.get('planner_revision'),
            'accepted_view_cycle': event.get('accepted_view_cycle'),
            **(metrics if isinstance(metrics, dict) else {}),
        })
        rows['Planning'].append(planning)

        # Scalar rows keep the existing Planning CSV/XLSX directly usable.
        # Nested stages are inclusive; never sum them as total request time.
        for timing_key in ('timing', 'worker_timing'):
            timing = metrics.get(timing_key, {}) if isinstance(metrics, dict) else {}
            if not isinstance(timing, dict):
                continue
            stages = timing.get('stages', {})
            for stage, values in stages.items() if isinstance(stages, dict) else []:
                if isinstance(values, dict):
                    rows['Planning'].append({
                        **common, 'record_type': 'planner_stage',
                        'request_id': event.get('request_id', ''),
                        'request_status': event.get('request_status', ''),
                        'plan_kind': event.get('plan_kind', ''),
                        'event_index': event_index, 'timing_scope': timing_key,
                        'stage': stage, 'stage_calls': values.get('calls'),
                        'stage_wall_sec': values.get('wall_sec'),
                    })
            events = timing.get('events', [])
            for attempt_index, attempt in enumerate(events if isinstance(events, list) else []):
                if isinstance(attempt, dict):
                    rows['Planning'].append({
                        **common, 'record_type': 'planner_attempt',
                        'request_id': event.get('request_id', ''),
                        'request_status': event.get('request_status', ''),
                        'plan_kind': event.get('plan_kind', ''),
                        'event_index': event_index, 'timing_scope': timing_key,
                        'attempt_index': attempt_index, **attempt,
                    })

    if rows['Runs']:
        capture_times = [
            row.get('submit_to_capture_sec')
            for row in rows['Capture_Timeline']
            if row.get('submit_to_capture_sec') is not None]
        transaction_times = [
            row.get('capture_transaction_sec')
            for row in rows['Capture_Timeline']
            if row.get('capture_transaction_sec') is not None]
        accepted_count = finite_float(rows['Runs'][0].get(
            'accepted_captures'))
        total_sec = finite_float(rows['Runs'][0].get('total_mission_sec'))
        rows['Runs'][0].update({
            'final_cube_coverage_fraction': coverages[-1].get('cumulative_coverage_fraction') if coverages else None,
            'azimuth_span_deg': circular_span_deg(azimuths),
            'elevation_span_deg': max(elevations) - min(elevations) if len(elevations) > 1 else 0.0,
            'submit_to_first_capture_sec': (
                min(capture_times) if capture_times else None),
            'median_capture_transaction_sec': (
                float(np.median(transaction_times))
                if transaction_times else None),
            'seconds_per_accepted_capture': (
                total_sec / accepted_count
                if total_sec is not None and accepted_count
                and accepted_count > 0.0 else None),
            'accepted_captures_per_minute': (
                accepted_count * 60.0 / total_sec
                if total_sec is not None and total_sec > 0.0
                and accepted_count is not None else None),
        })
    combined = np.vstack([cloud for cloud in frame_clouds if len(cloud)]) if any(len(cloud) for cloud in frame_clouds) else np.empty((0, 3))
    point_row = dict(common)
    point_row.update(cloud_dimension_metrics(combined))
    point_row['method'] = 'robot-pose fused qualified depth points'
    point_row['limitation'] = 'Dimensions are geometric stability evidence, not independent metrology.'
    rows['PointCloud'].append(point_row)
    if isinstance(phase, dict):
        for phase_name, duration in phase.get('phase_totals_sec', {}).items():
            row = dict(common)
            row.update({'phase': phase_name, 'duration_sec': duration})
            rows['Phase_Timing'].append(row)
    return rows
