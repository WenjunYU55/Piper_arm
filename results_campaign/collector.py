"""Read-only collection of evidence emitted by an existing PiPER mission.

The collector never imports ROS and never writes into a mission dataset.  It
copies small, immutable summaries into the campaign attempt directory so that
ephemeral ``/tmp`` diagnostics survive a normal coordinator shutdown.
"""

from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from .campaign import EVIDENCE_SCHEMA_VERSION, atomic_write_json, load_json, sha256_file, utc_now


def _load_yaml(path: Path) -> Any:
    try:
        with path.open('r', encoding='utf-8') as stream:
            return yaml.safe_load(stream)
    except (OSError, yaml.YAMLError):
        return None


def _source(path: Path, project_root: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    try:
        display = str(resolved.relative_to(project_root.resolve()))
    except ValueError:
        display = str(resolved)
    return {
        'path': display,
        'absolute_path': str(resolved),
        'sha256': sha256_file(resolved) if resolved.is_file() else '',
        'size_bytes': int(resolved.stat().st_size) if resolved.is_file() else 0,
    }


def _stamp_ns(stamp: Any) -> Optional[int]:
    if not isinstance(stamp, dict):
        return None
    try:
        return int(stamp['sec']) * 1_000_000_000 + int(stamp.get('nanosec', 0))
    except (KeyError, TypeError, ValueError):
        return None


def _task_matches(record: Any, task_id: str) -> bool:
    if not isinstance(record, dict):
        return False
    context = record.get('mission_context')
    return str(record.get('task_id', '')) == task_id or (
        isinstance(context, dict) and str(context.get('task_id', '')) == task_id)


def _heavy_roots(minimum_mtime_ns: int = 0) -> Iterable[Path]:
    root = Path('/tmp/piper_heavy_refresh')
    if not root.is_dir():
        return ()
    paths = []
    for path in root.glob('**/result.yaml'):
        try:
            if path.stat().st_mtime_ns >= minimum_mtime_ns:
                paths.append(path)
        except FileNotFoundError:
            continue
    return paths


def _target_mask_summary(result: Dict[str, Any], result_path: Path) -> Dict[str, Any]:
    sam_path_text = str(result.get('sam2_masks_yaml', '')).strip()
    sam_path = Path(sam_path_text) if sam_path_text else None
    sam = _load_yaml(sam_path) if sam_path and sam_path.is_file() else None
    target_mask = None
    if isinstance(sam, dict):
        for mask in sam.get('masks', []):
            if isinstance(mask, dict) and (
                    mask.get('mask_role') == 'target' or
                    mask.get('is_target_candidate') is True):
                target_mask = mask
                break
    selection = sam.get('target_depth_selection', {}) if isinstance(sam, dict) else {}
    context = result.get('mission_context', {})
    return {
        'result_path': str(result_path),
        'result_sha256': sha256_file(result_path),
        'request_id': str(result.get('request_id', '')),
        'request_reason': str(context.get('request_reason', '')) if isinstance(context, dict) else '',
        'requested_target_label': str(result.get(
            'requested_target_label', context.get('target_label', '')
            if isinstance(context, dict) else '')),
        'target_profile': str(result.get(
            'target_profile', context.get('target_profile', '')
            if isinstance(context, dict) else '')),
        'target_prompt': str(result.get(
            'target_prompt', context.get('target_prompt', '')
            if isinstance(context, dict) else '')),
        'target_source': str(result.get('target_source', '')),
        'obstacle_count': result.get('obstacle_count'),
        'image_stamp_ns': _stamp_ns(result.get('image_stamp')),
        'status': str(result.get('status', '')),
        'target_valid': not bool(result.get('target_rejection_reason')),
        'target_rejection_reason': str(result.get('target_rejection_reason', '')),
        'grounding_dino_confidence': result.get('target_confidence'),
        'sam2_score': target_mask.get('sam2_score') if isinstance(target_mask, dict) else None,
        'mask_area_px': target_mask.get('mask_area_px') if isinstance(target_mask, dict) else None,
        'valid_depth_ratio': target_mask.get('valid_depth_ratio') if isinstance(target_mask, dict) else None,
        'median_depth_m': target_mask.get('median_depth_m') if isinstance(target_mask, dict) else (
            sam.get('target_depth_m') if isinstance(sam, dict) else None),
        'depth_layer_candidate_count': selection.get('candidate_count') if isinstance(selection, dict) else None,
        'depth_layer_candidate_depths_m': selection.get('candidate_depths_m', []) if isinstance(selection, dict) else [],
        'depth_layer_score_margin': selection.get('score_margin') if isinstance(selection, dict) else None,
        'sam2_masks_path': sam_path_text,
        'sam2_masks_sha256': sha256_file(sam_path) if sam_path and sam_path.is_file() else '',
    }


def _find_dataset(project_root: Path, task_id: str, mission_result: Dict[str, Any]) -> Optional[Path]:
    candidates: List[Path] = []
    result_path = str(mission_result.get('dataset_path', '')).strip()
    if result_path:
        candidates.append(Path(result_path))
    action = mission_result.get('action_summary', {})
    if isinstance(action, dict):
        for key in ('dataset_path', 'discarded_dataset_path'):
            value = str(action.get(key, '')).strip()
            if value:
                candidates.append(Path(value))
    active = project_root / 'datasets' / 'active_scan'
    if active.is_dir():
        candidates.extend(path.parent for path in active.glob('*/manifest.json'))
    for candidate in candidates:
        manifest = load_json(candidate / 'manifest.json', {})
        if isinstance(manifest, dict) and str(manifest.get('task_id', '')) == task_id:
            return candidate.resolve()
    return None


def _resolve_frame_artifact(dataset: Path, text: str) -> Optional[Path]:
    """Resolve only an already-persisted capture artifact for hashing."""
    if not text:
        return None
    path = Path(text)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((dataset / path, dataset / 'frames' / path.name))
    return next((candidate.resolve() for candidate in candidates
                 if candidate.is_file()), None)


def _deduplicate_sources(sources: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique = {}
    for source in sources:
        if isinstance(source, dict) and source.get('absolute_path'):
            unique[str(source['absolute_path'])] = source
    return [unique[key] for key in sorted(unique)]


def _frame_summary(path: Path) -> Optional[Dict[str, Any]]:
    frame = _load_yaml(path)
    if not isinstance(frame, dict):
        return None
    confidence = frame.get('confidence_quality', {})
    layer = confidence.get('depth_layer_selection', {}) if isinstance(confidence, dict) else {}
    temporal = confidence.get('temporal_aggregation', {}) if isinstance(confidence, dict) else {}
    execution = frame.get('scan_execution', {})
    selection = frame.get('view_selection', {})
    transform = frame.get('camera_transform', {})
    # Bind the target point to the same optical frame as camera_transform.
    # Schema-2 also carries a native-depth-frame synchronized target; mixing
    # that point with base_from_color would create a false centre offset.
    transform_child = str(transform.get('child_frame_id', '')) if isinstance(transform, dict) else ''
    target = frame.get('target_3d', {})
    target_frame = str(target.get('header', {}).get('frame_id', '')) if isinstance(target, dict) else ''
    if (not isinstance(target, dict) or not target.get('available')
            or (transform_child and target_frame != transform_child)):
        synchronized = frame.get('synchronized_target_3d', {})
        synchronized_frame = str(
            synchronized.get('header', {}).get('frame_id', '')) \
            if isinstance(synchronized, dict) else ''
        if (isinstance(synchronized, dict) and synchronized.get('available')
                and (not transform_child or synchronized_frame == transform_child)):
            target = synchronized
    return {
        'frame_index': frame.get('frame_index'),
        'metadata_path': str(path.resolve()),
        'metadata_sha256': sha256_file(path),
        'capture_timestamp_ns': _stamp_ns(frame.get('capture_timestamp')),
        'capture_transaction_start_ns': _stamp_ns(
            execution.get('header', {}).get('stamp') if isinstance(execution, dict) else None),
        'capture_wall_time': frame.get('capture_wall_time'),
        'camera_position_m': transform.get('translation_m') if isinstance(transform, dict) else None,
        'base_from_camera_4x4': transform.get('matrix_4x4') if isinstance(transform, dict) else None,
        'target_point_camera_m': [
            target.get('point', {}).get(axis) for axis in ('x', 'y', 'z')
        ] if isinstance(target, dict) and isinstance(target.get('point'), dict) else None,
        'target_depth_m': target.get('depth') if isinstance(target, dict) else None,
        'mask_area_px': frame.get('mask_area_px'),
        'valid_depth_ratio': frame.get('valid_depth_ratio'),
        'raw_mask_depth_stddev_mm': (
            float(frame['depth_stddev_m']) * 1000.0
            if frame.get('depth_stddev_m') is not None else None),
        'qualified_depth_stddev_mm': (
            float(target['depth_stddev']) * 1000.0
            if isinstance(target, dict) and target.get('depth_stddev') is not None else None),
        'qualified_support_points': confidence.get('projected_output_points') if isinstance(confidence, dict) else None,
        'qualified_support_fraction': layer.get('selected_support_fraction') if isinstance(layer, dict) else None,
        'depth_layer_candidate_count': layer.get('candidate_count') if isinstance(layer, dict) else None,
        'depth_layer_candidate_depths_m': layer.get('candidate_depths_m', []) if isinstance(layer, dict) else [],
        'temporal_depth_frames': temporal.get('input_frames') if isinstance(temporal, dict) else None,
        'temporal_depth_span_sec': temporal.get('span_sec') if isinstance(temporal, dict) else None,
        'viewpoint_id': selection.get('id') if isinstance(selection, dict) else None,
        'ray_id': selection.get('ray_id') if isinstance(selection, dict) else None,
        'azimuth_deg': selection.get('azimuth_deg') if isinstance(selection, dict) else None,
        'elevation_deg': selection.get('elevation_deg') if isinstance(selection, dict) else None,
        'camera_position_requested_m': selection.get('camera_position_m') if isinstance(selection, dict) else None,
        'look_direction': selection.get('look_direction') if isinstance(selection, dict) else None,
        'nbv_marginal_information_fraction': selection.get('nbv_marginal_information_fraction') if isinstance(selection, dict) else None,
        'nbv_novel_surface_pixels': selection.get('nbv_novel_surface_pixels') if isinstance(selection, dict) else None,
        'candidate_diagnostics': selection.get('candidate_diagnostics', {}) if isinstance(selection, dict) else {},
        'scan_quality_score': frame.get('scan_quality_score'),
        'scan_quality_label': frame.get('scan_quality_label'),
        'occlusion_state': frame.get('occlusion_state'),
        'occlusion_score': frame.get('occlusion_score'),
        'rgb_file_path': str(frame.get('rgb_file_path', '')),
        'target_depth_png_file_path': str(frame.get('target_depth_png_file_path', '')),
        'target_support_mask_file_path': str(frame.get('target_support_mask_file_path', '')),
        'native_depth_file_path': str(frame.get('native_depth_file_path', '')),
        'camera_info': frame.get('camera_info'),
    }


def _ray_summary(project_root: Path, task_id: str) -> Tuple[Optional[Path], Dict[str, Any]]:
    root = project_root / 'datasets' / 'ray_diagnostics' / task_id
    paths = sorted(root.glob('*.json'), key=lambda item: item.stat().st_mtime_ns) if root.is_dir() else []
    if not paths:
        return None, {}
    path = paths[-1]
    value = load_json(path, {})
    if not isinstance(value, dict):
        return path, {}
    events = value.get('events', [])
    planning = []
    for event in events if isinstance(events, list) else []:
        if not isinstance(event, dict):
            continue
        metrics = event.get('metrics', {})
        if not isinstance(metrics, dict):
            continue
        if any(key in metrics for key in (
                'planning_duration_sec', 'generated_ray_count',
                'surviving_ray_count', 'candidate_viewpoints_considered')):
            planning.append({
                'request_id': event.get('request_id', ''),
                'request_status': event.get('request_status', ''),
                'plan_kind': event.get('plan_kind', 'MULTIVIEW_SCAN'),
                'timestamp_ns': event.get('timestamp_ns', event.get('created_at_ns')),
                'message': event.get('message', ''),
                'planner_revision': event.get('planner_revision'),
                'accepted_view_cycle': event.get('accepted_view_cycle'),
                'metrics': metrics,
            })
    return path, {
        'schema_version': value.get('schema_version'),
        'artifact_id': value.get('artifact_id'),
        'journal_complete': value.get('journal_complete'),
        'generations': value.get('generations', []),
        'planning_events': planning,
    }


def _planner_log_events(path: Path, task_id: str) -> List[Dict[str, Any]]:
    """Recover acquisition/failure timings absent from multiview ray reports."""
    events = []
    try:
        with path.open(encoding='utf-8', errors='replace') as stream:
            for line in stream:
                if not line.startswith('CUROBO_REQUEST_TIMING '):
                    continue
                try:
                    value = json.loads(line.split(' ', 1)[1])
                except (ValueError, TypeError):
                    continue
                if (isinstance(value, dict) and value.get('task_id') == task_id
                        and isinstance(value.get('metrics'), dict)):
                    events.append(value)
    except OSError:
        pass
    return events


def collect_task(project_root: Path, attempt: Path) -> Path:
    """Refresh one campaign evidence summary from existing output files."""
    root = Path(project_root).resolve()
    attempt = Path(attempt).resolve()
    submission = load_json(attempt / 'submission.json')
    if not isinstance(submission, dict):
        raise ValueError('attempt has no valid submission.json')
    task_id = str(submission.get('task_id', '')).strip()
    if not task_id:
        raise ValueError('submission has no task ID')

    sources: List[Dict[str, Any]] = [_source(attempt / 'submission.json', root)]
    mission_path = Path('/tmp/piper_target_scan_missions/results') / (task_id + '.json')
    mission_result = load_json(mission_path, {})
    if not isinstance(mission_result, dict):
        mission_result = {}
    if mission_path.is_file():
        sources.append(_source(mission_path, root))

    heavy = []
    submission_ns = int(submission.get('submitted_wall_time_ns') or 0)
    # Only inspect files plausibly belonging to this mission. The two-minute
    # lead accommodates an already-running perception job at GUI submission.
    minimum_heavy_mtime_ns = max(0, submission_ns - 120_000_000_000)
    for path in _heavy_roots(minimum_heavy_mtime_ns):
        value = _load_yaml(path)
        if _task_matches(value, task_id):
            summary = _target_mask_summary(value, path)
            heavy.append(summary)
            sources.append(_source(path, root))
            sam_path = Path(str(summary.get('sam2_masks_path', '')))
            if sam_path.is_file():
                sources.append(_source(sam_path, root))
    heavy.sort(key=lambda item: item.get('image_stamp_ns') or 0)

    dataset = _find_dataset(root, task_id, mission_result)
    manifest = {}
    frames = []
    if dataset is not None:
        manifest_path = dataset / 'manifest.json'
        manifest = load_json(manifest_path, {})
        if manifest_path.is_file():
            sources.append(_source(manifest_path, root))
        for path in sorted((dataset / 'frames').glob('view_*_metadata.yaml')):
            summary = _frame_summary(path)
            if summary is not None:
                frames.append(summary)
                sources.append(_source(path, root))
                for field in (
                        'rgb_file_path', 'target_depth_png_file_path',
                        'target_support_mask_file_path',
                        'native_depth_file_path'):
                    artifact = _resolve_frame_artifact(
                        dataset, str(summary.get(field, '')))
                    if artifact is not None:
                        sources.append(_source(artifact, root))

    ray_path, ray = _ray_summary(root, task_id)
    if ray_path is not None:
        sources.append(_source(ray_path, root))

    terminal = load_json(attempt / 'terminal.json', {})
    action = mission_result.get('action_summary', {})
    processes = action.get('processes', {}) if isinstance(action, dict) else {}
    worker = processes.get('curobo_worker', {}) if isinstance(processes, dict) else {}
    log_name = worker.get('log') if isinstance(worker, dict) else None
    if log_name:
        log_path = Path(log_name)
        log_events = _planner_log_events(log_path, task_id)
        if log_events:
            sources.append(_source(log_path, root))
            planning_events = ray.setdefault('planning_events', [])
            by_request = {event.get('request_id'): event for event in planning_events
                          if event.get('request_id')}
            for event in log_events:
                previous = by_request.get(event.get('request_id'))
                if previous is not None:
                    # Keep bridge wall time while adding post-publication evidence.
                    previous['metrics'].update(event['metrics'])
                    previous['plan_kind'] = event.get('plan_kind', '')
                else:
                    planning_events.append(event)
                    by_request[event.get('request_id')] = event
    consistency_errors = []
    actual_backend = action.get('planner_backend') if isinstance(action, dict) else None
    if actual_backend and str(actual_backend) != str(submission.get('requested_backend')):
        consistency_errors.append('mission-result planner backend differs from submitted backend')
    actual_coordinates = action.get('rough_target_base_link_xyz') if isinstance(action, dict) else None
    if actual_coordinates is not None:
        try:
            requested_coordinates = [float(value) for value in submission.get('requested_coordinates_m', [])]
            actual_values = [float(value) for value in actual_coordinates]
            if len(actual_values) != 3 or len(requested_coordinates) != 3 or any(abs(a - b) > 1e-9 for a, b in zip(actual_values, requested_coordinates)):
                consistency_errors.append('mission-result target coordinates differ from submitted coordinates')
        except (TypeError, ValueError):
            consistency_errors.append('mission-result target coordinates are malformed')
    campaign_definition = load_json(
        attempt.parents[3] / 'campaign.json', {}) or {}
    evidence = {
        'schema_version': EVIDENCE_SCHEMA_VERSION,
        'collected_utc': utc_now(),
        'collected_wall_time_ns': time.time_ns(),
        'campaign_id': submission.get('campaign_id'),
        'trial_id': submission.get('trial_id'),
        'task_id': task_id,
        'matches_schedule': submission.get('matches_schedule') is True and not consistency_errors,
        'consistency_errors': consistency_errors,
        'evidence_class': campaign_definition.get(
            'evidence_class', 'EXPLORATORY'),
        'submission': submission,
        'terminal': terminal if isinstance(terminal, dict) else {},
        'mission_result': mission_result,
        'planner_backend': (
            action.get('planner_backend') if isinstance(action, dict) else None
        ) or submission.get('requested_backend'),
        'phase_timing': action.get('phase_timing', {}) if isinstance(action, dict) else {},
        'processes': action.get('processes', {}) if isinstance(action, dict) else {},
        'heavy_perception': heavy,
        'first_acquisition': next((
            item for item in heavy
            if any(token in item.get('request_reason', '').lower()
                   for token in ('rough', 'acquisition'))), heavy[0] if heavy else {}),
        'dataset_path': str(dataset) if dataset is not None else '',
        'dataset_manifest': manifest if isinstance(manifest, dict) else {},
        'frames': frames,
        'ray_diagnostics': ray,
        'sources': _deduplicate_sources(sources),
        'collection_limitations': [
            'Observer reads existing files and cannot recover diagnostics that were never persisted.',
            'Grounding DINO confidence and SAM2 score are model confidence values, not detection accuracy.',
            'Per-capture timing uses persisted capture transaction and capture timestamps.',
        ],
    }
    path = attempt / 'evidence.json'
    atomic_write_json(path, evidence)
    from .bookmarks import write_campaign_bookmarks
    write_campaign_bookmarks(root, attempt.parents[3])
    return path


def collect_campaign(project_root: Path, campaign_root: Path) -> List[Path]:
    outputs = []
    for attempt in sorted(Path(campaign_root).glob('trials/*/attempts/*')):
        if (attempt / 'submission.json').is_file():
            outputs.append(collect_task(project_root, attempt))
    return outputs
