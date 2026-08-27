#!/usr/bin/env python3
"""
Create a command-free size-aware Ray Review artifact.

This exercises the production silhouette envelope, 360-ray generator,
committed capability atlas, and NBV scorer without creating a ROS node or
claiming that IK/motion was executed.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOBILE_SOURCE = PROJECT_ROOT / (
    'piper_ros_foxy/src/piper_mobile_manipulation')
if str(MOBILE_SOURCE) not in sys.path:
    sys.path.insert(0, str(MOBILE_SOURCE))

from piper_mobile_manipulation.capability_map import (  # noqa: E402
    load_capability_map,
    sha256_file,
)
from piper_mobile_manipulation.nbv_coverage import (  # noqa: E402
    candidate_meets_minimum_information,
    CoverageSnapshot,
    direction_bin,
    ObjectCoverageModel,
    persist_coverage_snapshot,
    rank_next_best_views,
    SURFACE,
    UNKNOWN,
    VoxelCoverageConfig,
)
from piper_mobile_manipulation.ray_mission_diagnostics import (  # noqa: E402
    add_bridge_request,
    add_capture_event,
    add_prequalification,
    add_request_rejection,
    add_target_update_event,
    add_terminal_event,
    planner_generation_snapshot,
    RayMissionDiagnosticsStore,
)
from piper_mobile_manipulation.target_envelope import (  # noqa: E402
    build_revolution_envelope,
    canonical_sha256,
    coverage_sphere_from_envelope,
    envelope_constrained_ray_interval,
    trusted_silhouette_measurement,
    validate_envelope,
)
from piper_mobile_manipulation.scan_session_memory import (  # noqa: E402
    viewpoint_direction_is_redundant,
)
from piper_mobile_manipulation.viewpoint_rays import (  # noqa: E402
    build_ray_samples,
)
from piper_mobile_manipulation.viewpoint_reachability_filter_node import (  # noqa: E402
    capability_bound_ray,
    CAPABILITY_MAP_REJECTION,
)


def large_target_envelope(target):
    """Return an inflated envelope from a synthetic 10 x 14 cm silhouette."""
    height, width = 480, 640
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[135:345, 245:395] = 255
    support = np.ones((210, 150), dtype=bool)
    depth = np.full((210, 150), 0.40, dtype=float)
    header = SimpleNamespace(
        stamp=SimpleNamespace(sec=int(time.time()), nanosec=0),
        frame_id='camera_color_optical_frame')
    intrinsic = np.asarray([
        [600.0, 0.0, 319.5],
        [0.0, 600.0, 239.5],
        [0.0, 0.0, 1.0],
    ])
    shape = trusted_silhouette_measurement(
        mask, support, (245, 135), depth, intrinsic, header, 0.99)
    # Optical +Z is base +X; place the synthetic camera at the target height.
    base_from_camera = np.asarray([
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, float(target[2])],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return build_revolution_envelope(
        shape, base_from_camera, target)


def generated_rays(target, envelope=None, count=360):
    center_vector = np.asarray(target, dtype=float)
    center = dict(zip(('x', 'y', 'z'), center_vector.tolist()))
    camera_info = {
        'available': True, 'width': 640, 'height': 480,
        'fx': 600.0, 'fy': 600.0, 'cx': 319.5, 'cy': 239.5,
        'frame_id': 'camera_color_optical_frame',
    }
    result = []
    for ray_id, (angle, pitch) in enumerate(
            build_ray_samples('full_sphere', count)):
        azimuth = math.radians(float(angle))
        elevation = math.radians(float(pitch))
        direction_vector = np.asarray([
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            -math.sin(elevation),
        ])
        interval = (
            envelope_constrained_ray_interval(
                center_vector, direction_vector, 0.28, 0.80,
                envelope, camera_info, envelope_is_validated=True)
            if envelope is not None else (0.28, 0.80))
        supported = interval is not None
        minimum, maximum = interval if supported else (0.28, 0.80)
        preferred_maximum = min(maximum, max(minimum, 0.50))
        scoring = 0.5 * (minimum + preferred_maximum)
        camera = center_vector + direction_vector * scoring
        direction = dict(zip(
            ('x', 'y', 'z'), direction_vector.tolist()))
        result.append({
            'index': ray_id,
            'ray_id': ray_id,
            'frame_id': 'base_link',
            'viewpoint_angle_deg': float(angle),
            'camera_pitch_deg': float(pitch),
            'target_object_center': dict(center),
            'desired_camera_position': dict(zip(
                ('x', 'y', 'z'), camera.tolist())),
            'desired_look_at_direction': dict(zip(
                ('x', 'y', 'z'), (-direction_vector).tolist())),
            'camera_object_distance_m': float(scoring),
            'keep_object_centered': True,
            'reachable': False,
            'safe': False,
            'candidate_geometry': 'target_ray',
            'ray_direction': direction,
            'ray_min_standoff_m': float(minimum),
            'ray_max_standoff_m': float(maximum),
            'ray_preferred_max_standoff_m': float(preferred_maximum),
            'ray_scoring_standoff_m': float(scoring),
            'ray_requested_min_standoff_m': 0.28,
            'ray_requested_max_standoff_m': 0.80,
            'ray_envelope_min_standoff_m': float(minimum),
            'ray_envelope_max_standoff_m': float(maximum),
            'target_envelope_supported': bool(supported),
            'target_envelope_sha256': (
                envelope['envelope_sha256'] if envelope is not None else ''),
            'target_envelope_rejection_reason': '' if supported else (
                'target envelope leaves no camera standoff with complete FOV '
                'and 0.250m surface clearance'),
        })
    return result


def capability_prequalification(candidates, capability):
    output = []
    hard_culls = {}
    for candidate in candidates:
        if candidate.get('target_envelope_supported') is False:
            ray_id = int(candidate['ray_id'])
            hard_culls[ray_id] = {
                'ray_id': ray_id,
                'stage': 'target_envelope',
                'reason_code': 'NO_SAFE_TARGET_STANDOFF',
                'reason': candidate['target_envelope_rejection_reason'],
                'evidence': {
                    'target_envelope_sha256': candidate[
                        'target_envelope_sha256'],
                },
            }
            continue
        center = candidate['target_object_center']
        direction = candidate['ray_direction']
        query = capability.intersects_ray(
            [center[key] for key in ('x', 'y', 'z')],
            [direction[key] for key in ('x', 'y', 'z')],
            candidate['ray_min_standoff_m'],
            candidate['ray_max_standoff_m'],
            0.005,
            float(capability.metadata['tool_floor_clearance_m']))
        evidence = {
            'available': True,
            'supported': bool(query.supported),
            'checked_keys': int(query.checked_keys),
            'matching_keys': int(query.matching_keys),
            'sampled_standoffs': len(query.sample_support),
            'supported_standoff_samples': int(sum(query.sample_support)),
            'supported_intervals_m': [
                [float(lower), float(upper)]
                for lower, upper in query.supported_intervals_m],
            'position_voxel_m': float(capability.position_voxel_m),
            'elapsed_ms': float(query.elapsed_ms),
            'reason': str(query.reason),
            'effective_mode': 'enforce',
        }
        value = dict(candidate)
        if query.supported:
            value = capability_bound_ray(value, evidence)
            value.update({
                'prequalified': True,
                'reachable': True,
                'safe': True,
                'reject_reasons': [],
                'capability_map_prequalification': evidence,
            })
        else:
            value.update({
                'prequalified': False,
                'reachable': False,
                'safe': False,
                'reject_reasons': [CAPABILITY_MAP_REJECTION],
                'cull_disposition': 'permanent',
                'capability_map_prequalification': evidence,
            })
            ray_id = int(value['ray_id'])
            hard_culls[ray_id] = {
                'ray_id': ray_id,
                'stage': 'prequalification',
                'reason_code': 'CAPABILITY_MAP_NO_SUPPORT',
                'reason': CAPABILITY_MAP_REJECTION,
                'evidence': evidence,
            }
        output.append(value)
    return output, hard_culls


def seed_coverage(target, radius, seed_candidate, session_id):
    voxel = 0.010
    coordinates = np.arange(-radius, radius + voxel * 0.5, voxel)
    grid = np.stack(np.meshgrid(
        coordinates, coordinates, coordinates, indexing='ij'), axis=-1)
    offsets = grid.reshape((-1, 3))
    lengths = np.linalg.norm(offsets, axis=1)
    inside = lengths <= radius + 1e-9
    offsets = offsets[inside]
    lengths = lengths[inside]
    centers = offsets + np.asarray(target, dtype=float)
    states = np.full(len(centers), UNKNOWN, dtype=np.uint8)
    shell = lengths >= radius - voxel * 1.5
    states[shell] = SURFACE
    camera = np.asarray([
        seed_candidate['desired_camera_position'][key]
        for key in ('x', 'y', 'z')], dtype=float)
    seed_direction = camera - np.asarray(target, dtype=float)
    seed_direction /= np.linalg.norm(seed_direction)
    normals = np.zeros_like(offsets)
    nonzero = lengths > 1e-9
    normals[nonzero] = offsets[nonzero] / lengths[nonzero, None]
    measured = shell & ((normals @ seed_direction) > 0.05)
    bits = np.zeros(len(centers), dtype=np.uint32)
    bits[measured] = np.uint32(1 << direction_bin(seed_direction))
    return CoverageSnapshot(
        session_id=session_id,
        generation=1,
        target_center=tuple(float(value) for value in target),
        radius_m=float(radius),
        voxel_size_m=voxel,
        states=states,
        surface_view_bits=bits,
        voxel_centers=centers,
        view_directions=(tuple(float(value) for value in seed_direction),),
        tan_half_fov_x=320.0 / 600.0,
        tan_half_fov_y=240.0 / 600.0,
        render_width=64,
        render_height=48,
        maximum_scoring_voxels=20000,
    )


def camera_matrix(candidate):
    position = np.asarray([
        candidate['desired_camera_position'][key]
        for key in ('x', 'y', 'z')], dtype=float)
    optical_z = np.asarray([
        candidate['desired_look_at_direction'][key]
        for key in ('x', 'y', 'z')], dtype=float)
    optical_z /= np.linalg.norm(optical_z)
    up = np.asarray([0.0, 0.0, 1.0])
    optical_x = np.cross(optical_z, up)
    if np.linalg.norm(optical_x) < 1e-6:
        optical_x = np.asarray([1.0, 0.0, 0.0])
    optical_x /= np.linalg.norm(optical_x)
    optical_y = np.cross(optical_z, optical_x)
    result = np.eye(4)
    result[:3, :3] = np.column_stack((optical_x, optical_y, optical_z))
    result[:3, 3] = position
    return result.tolist()


def accepted_capture_events(document):
    """Return recorded accepted captures in achieved-view order."""
    captures = []
    for generation in document.get('generations', []):
        for event in generation.get('events', []):
            ray_ids = event.get('captured_ray_ids', [])
            matrix = event.get('achieved_camera_matrix_4x4')
            if event.get('capture_accepted') is True and ray_ids and matrix:
                captures.append({
                    'ray_id': int(ray_ids[0]),
                    'matrix': matrix,
                    'capture_id': str(event.get('capture_id', '')),
                    'artifacts': list(event.get('artifact_bindings', [])),
                })
    return captures


def replay_source(document):
    """Extract one immutable cube replay source from a mission journal."""
    generations = [
        value for value in document.get('generations', [])
        if value.get('generated_ray_population')]
    envelope = next((
        value.get('target_envelope') for value in reversed(generations)
        if isinstance(value.get('target_envelope'), dict)), None)
    captures = accepted_capture_events(document)
    if envelope is None or not captures:
        raise ValueError('source report has no envelope or accepted capture')
    target = tuple(float(value) for value in generations[-1][
        'target_center_m'])
    first_artifact = Path(captures[0]['artifacts'][0]).resolve()
    scan_dir = first_artifact.parent.parent
    if not scan_dir.is_dir():
        raise ValueError('source capture directory is unavailable')
    return target, envelope, captures, scan_dir


def qualified_anchor_envelope(source):
    """Upgrade recorded pre-boundary geometry for current-policy replay."""
    value = json.loads(json.dumps(source))
    old_anchor = list(value['planning_anchor_m'])
    qualified = list(value['axis_origin_m'])
    value.setdefault('bootstrap_anchor_m', old_anchor)
    value['planning_anchor_m'] = qualified
    bounds_min = np.asarray(value['bounds_min_m'], dtype=float)
    bounds_max = np.asarray(value['bounds_max_m'], dtype=float)
    corners = np.asarray([
        [x, y, z]
        for x in (bounds_min[0], bounds_max[0])
        for y in (bounds_min[1], bounds_max[1])
        for z in (bounds_min[2], bounds_max[2])])
    value['bounding_radius_from_anchor_m'] = round(float(np.max(
        np.linalg.norm(corners - np.asarray(qualified), axis=1))), 8)
    value.pop('envelope_sha256', None)
    value['envelope_sha256'] = canonical_sha256(value)
    return validate_envelope(value)


def recorded_exhaustive_ray_ik_failures(document):
    """Return rays with recorded typed exhaustive continuous-IK failure."""
    result = set()
    for generation in document.get('generations', []):
        for request in generation.get('requests', []):
            if (
                    request.get('status') == 'failed'
                    and 'RAY_IK_FAILURE' in str(request.get(
                        'diagnostic', ''))):
                result.update(int(value) for value in request.get(
                    'attempted_ray_ids', []))
    return sorted(result)


def history_entry(capture, target):
    """Adapt one achieved camera matrix to accepted-direction history."""
    matrix = np.asarray(capture['matrix'], dtype=float)
    camera = matrix[:3, 3]
    look = np.asarray(target, dtype=float) - camera
    look /= np.linalg.norm(look)
    return {
        'ray_id': int(capture['ray_id']),
        'desired_camera_position': dict(zip(
            ('x', 'y', 'z'), camera.tolist())),
        'desired_look_at_direction': dict(zip(
            ('x', 'y', 'z'), look.tolist())),
        'target_estimate_used': dict(zip(('x', 'y', 'z'), target)),
    }


def corrected_coverage(
        scan_dir, accepted_views, target, envelope, session_id):
    """Rebuild recorded RGB-D into the corrected object-sized model."""
    sphere = coverage_sphere_from_envelope(envelope)
    model = ObjectCoverageModel(VoxelCoverageConfig())
    snapshot = model.rebuild_from_scan(
        scan_dir, accepted_views, target, session_id,
        model_center=sphere['center_m'],
        model_radius_m=sphere['radius_m'],
        model_source=sphere['source'])
    return snapshot, sphere


def replay_cube_pipeline(args):
    """Replay real cube captures through the current command-free planner."""
    source_path = Path(args.source_report).resolve()
    document = json.loads(source_path.read_text(encoding='utf-8'))
    bootstrap_target, recorded_envelope, captures, scan_dir = replay_source(
        document)
    envelope = qualified_anchor_envelope(recorded_envelope)
    target = tuple(float(value) for value in envelope['planning_anchor_m'])
    recorded_ik_culls = recorded_exhaustive_ray_ik_failures(document)
    if args.capture_count is not None:
        captures = captures[:max(1, int(args.capture_count))]
    session_id = args.name or time.strftime(
        'offline-cube-current-replay-%Y%m%d-%H%M%S')
    report_root = PROJECT_ROOT / 'datasets/active_scan/ray_diagnostics'
    capability_path = MOBILE_SOURCE / (
        'config/piper_camera_capability_map.npz')
    before = (sha256_file(capability_path), capability_path.stat().st_mtime_ns)
    capability = load_capability_map(
        capability_path, PROJECT_ROOT, verify_sources=True)

    provisional = generated_rays(bootstrap_target, None, args.ray_count)
    provisional_results, provisional_culls = capability_prequalification(
        provisional, capability)
    bounded = generated_rays(target, envelope, args.ray_count)
    bounded_results, bounded_culls = capability_prequalification(
        bounded, capability)
    # Bootstrap culls and recorded old-centre endpoint failures belong to a
    # different population identity.  The qualified population is evaluated
    # afresh around the mask-derived object centre.
    persistent_culls = dict(bounded_culls)
    static_survivors = [
        item for item in bounded_results
        if item.get('prequalified') is True
        and int(item['ray_id']) not in persistent_culls]
    if not static_survivors:
        raise RuntimeError('cube replay produced no statically surviving ray')

    bootstrap_common = dict(
        session_id=session_id, target_center=bootstrap_target,
        frame_id='base_link', policy='ray_nbv_seed', mission_id=session_id,
        remaining_views=13)
    common = dict(
        session_id=session_id, target_center=target, frame_id='base_link',
        policy='ray_nbv', mission_id=session_id, remaining_views=13)
    first = captures[0]
    generation0_survivors = [
        item for item in provisional_results
        if item.get('prequalified') is True]
    generation0 = planner_generation_snapshot(
        generation=0, generated=provisional,
        history_remaining=provisional,
        ranked=provisional, selected=provisional,
        persistent_culls={}, target_envelope=None, **bootstrap_common)
    generation0 = add_prequalification(generation0, provisional_results, {
        'candidate_viewpoints': len(provisional),
        'prequalified_viewpoints': len(generation0_survivors),
        'capability_map_mode': 'enforce',
        'capability_map_sha256': before[0],
        'simulation': True,
        'source_report': str(source_path),
    })
    generation0 = add_bridge_request(
        generation0, 'recorded-capture-001', [first['ray_id']])
    generation0 = add_capture_event(
        generation0, 'recorded-capture-001', True,
        ray_id=first['ray_id'],
        achieved_camera_matrix_4x4=first['matrix'],
        artifact_bindings=first['artifacts'])
    coverage, sphere = corrected_coverage(
        scan_dir, 1, target, envelope, session_id)
    coverage_path = report_root / session_id / 'coverage' / 'capture_001.npz'
    persist_coverage_snapshot(
        coverage_path, coverage, capture_artifacts=first['artifacts'],
        configuration_artifacts=[capability_path, MOBILE_SOURCE / (
            'config/scan_planning_params.yaml')], dataset_root=PROJECT_ROOT)
    generation0 = add_target_update_event(
        generation0, 'recorded-capture-001', str(coverage_path))
    for event in generation0['events']:
        event['simulation'] = True

    accepted_history = [history_entry(first, target)]
    target_mapping = dict(zip(('x', 'y', 'z'), target))
    store = RayMissionDiagnosticsStore(report_root)
    json_path, html_path = store.record(generation0)
    for capture_index, capture in enumerate(captures[1:], start=2):
        rejected = {}
        survivors = []
        for candidate in static_survivors:
            if viewpoint_direction_is_redundant(
                    candidate, accepted_history, target_mapping, 15.0):
                rejected[int(candidate['ray_id'])] = [
                    'direction is within the accepted-view redundancy floor']
            else:
                survivors.append(candidate)
        ranked = rank_next_best_views(
            coverage, survivors,
            [accepted_history[-1]['desired_camera_position'][axis]
             for axis in ('x', 'y', 'z')])
        positive = [
            item for item in ranked
            if candidate_meets_minimum_information(item)]
        generation = planner_generation_snapshot(
            generation=capture_index - 1, generated=bounded,
            history_remaining=survivors, ranked=ranked, selected=positive,
            planner_rejections=rejected,
            persistent_culls=persistent_culls,
            target_envelope=envelope, **common)
        generation = add_prequalification(generation, positive, {
            'candidate_viewpoints': len(bounded),
            'prequalified_viewpoints': len(positive),
            'capability_map_mode': 'enforce', 'simulation': True})
        if capture_index == 2 and recorded_ik_culls:
            generation = add_bridge_request(
                generation, 'recorded-exhaustive-ray-ik',
                recorded_ik_culls)
            generation = add_request_rejection(
                generation, 'recorded-exhaustive-ray-ik',
                'TESSERACT_EXHAUSTED',
                'Recorded typed RAY_IK_FAILURE now persists for this session')
        generation = add_bridge_request(
            generation, 'recorded-capture-%03d' % capture_index,
            [capture['ray_id']])
        generation = add_capture_event(
            generation, 'recorded-capture-%03d' % capture_index, True,
            ray_id=capture['ray_id'],
            achieved_camera_matrix_4x4=capture['matrix'],
            artifact_bindings=capture['artifacts'])
        coverage, sphere = corrected_coverage(
            scan_dir, capture_index, target, envelope, session_id)
        coverage_path = report_root / session_id / 'coverage' / (
            'capture_%03d.npz' % capture_index)
        persist_coverage_snapshot(
            coverage_path, coverage,
            capture_artifacts=[
                value for item in captures[:capture_index]
                for value in item['artifacts']],
            configuration_artifacts=[capability_path, MOBILE_SOURCE / (
                'config/scan_planning_params.yaml')],
            dataset_root=PROJECT_ROOT)
        generation = add_target_update_event(
            generation, 'recorded-capture-%03d' % capture_index,
            str(coverage_path))
        for event in generation['events']:
            event['simulation'] = True
        json_path, html_path = store.record(generation)
        accepted_history.append(history_entry(capture, target))

    rejected = {}
    survivors = []
    for candidate in static_survivors:
        if viewpoint_direction_is_redundant(
                candidate, accepted_history, target_mapping, 15.0):
            rejected[int(candidate['ray_id'])] = [
                'direction is within the accepted-view redundancy floor']
        else:
            survivors.append(candidate)
    ranked = rank_next_best_views(
        coverage, survivors,
        [accepted_history[-1]['desired_camera_position'][axis]
         for axis in ('x', 'y', 'z')])
    positive = [
        item for item in ranked if candidate_meets_minimum_information(item)]
    preview = planner_generation_snapshot(
        generation=len(captures), generated=bounded,
        history_remaining=survivors, ranked=ranked, selected=positive,
        planner_rejections=rejected, persistent_culls=persistent_culls,
        target_envelope=envelope, **common)
    shortlist = [int(item['ray_id']) for item in positive[:6]]
    preview = add_prequalification(preview, positive, {
        'candidate_viewpoints': len(bounded),
        'prequalified_viewpoints': len(positive),
        'capability_map_mode': 'enforce', 'simulation': True})
    preview = add_bridge_request(
        preview, 'current-policy-preview', shortlist)
    preview = add_request_rejection(
        preview, 'current-policy-preview', 'OFFLINE_SIMULATION_NO_IK',
        'Recorded RGB-D replay ended before new IK; no motion claim')
    preview = add_terminal_event(
        preview, 'cancelled',
        'Command-free current-policy cube replay completed')
    for event in preview['events']:
        event['simulation'] = True
    json_path, html_path = store.record(preview)
    after = (sha256_file(capability_path), capability_path.stat().st_mtime_ns)
    if before != after:
        raise RuntimeError('capability map changed during simulation')
    print('report=%s' % json_path)
    print('html=%s' % html_path)
    print('source_report=%s' % source_path)
    print('recorded_captures=%s' % [item['ray_id'] for item in captures])
    print('recorded_permanent_ray_ik_culls=%s' % recorded_ik_culls)
    print('recorded_old_population_ik_culls_reused=false')
    print('bootstrap_center_m=%s' % (list(bootstrap_target),))
    print('qualified_ray_center_m=%s' % (list(target),))
    print('grey_diameter_m=%.8f' % (2.0 * sphere['radius_m']))
    print('grey_center_m=%s' % sphere['center_m'])
    print('static_culls=%d accepted_neighbour_culls=%d survivors=%d'
          % (len(persistent_culls), len(rejected), len(survivors)))
    print('nbv_positive=%d shortlisted=%s' % (len(positive), shortlist))
    print('capability_map_sha256=%s unchanged=true' % before[0])
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', nargs=3, type=float,
                        default=(0.4, 0.0, 0.12), metavar=('X', 'Y', 'Z'))
    parser.add_argument('--ray-count', type=int, default=360)
    parser.add_argument('--name', default='')
    parser.add_argument('--source-report', default='')
    parser.add_argument('--capture-count', type=int, default=None)
    args = parser.parse_args()

    if args.source_report:
        return replay_cube_pipeline(args)

    bootstrap_target = tuple(args.target)
    session_id = args.name or time.strftime(
        'offline-size-aware-large-target-%Y%m%d-%H%M%S')
    report_root = PROJECT_ROOT / 'datasets/active_scan/ray_diagnostics'
    capability_path = MOBILE_SOURCE / (
        'config/piper_camera_capability_map.npz')
    before = (sha256_file(capability_path), capability_path.stat().st_mtime_ns)
    capability = load_capability_map(capability_path, verify_sources=False)
    envelope = large_target_envelope(bootstrap_target)
    target = tuple(float(value) for value in envelope['planning_anchor_m'])
    generated = generated_rays(target, envelope, args.ray_count)
    prequalified, hard_culls = capability_prequalification(
        generated, capability)
    envelope_survivors = [
        item for item in generated
        if item.get('target_envelope_supported') is not False]
    accepted = [item for item in prequalified if item['prequalified']]
    if not accepted:
        raise RuntimeError('large-target simulation produced no supported ray')

    common = dict(
        session_id=session_id,
        target_center=target,
        frame_id='base_link',
        policy='ray_nbv',
        generated=generated,
        mission_id=session_id,
        target_envelope=envelope,
        remaining_views=13,
    )
    generation0 = planner_generation_snapshot(
        generation=0,
        history_remaining=envelope_survivors,
        ranked=envelope_survivors,
        selected=envelope_survivors,
        persistent_culls={
            key: value for key, value in hard_culls.items()
            if value['stage'] == 'target_envelope'},
        **common)
    generation0 = add_prequalification(generation0, prequalified, {
        'candidate_viewpoints': len(envelope_survivors),
        'prequalified_viewpoints': len(accepted),
        'capability_supported_rays': len(accepted),
        'capability_rejected_rays': sum(
            not item['prequalified'] for item in prequalified),
        'capability_map_mode': 'enforce',
        'capability_map_sha256': before[0],
        'simulation': True,
    })
    seed = min(accepted, key=lambda item: int(item['ray_id']))
    generation0 = add_bridge_request(
        generation0, 'offline-synthetic-seed', [seed['ray_id']])
    # The report explicitly records this as synthetic policy evidence. No IK,
    # trajectory, controller, or ROS entity exists in this tool.
    generation0 = add_capture_event(
        generation0, 'offline-synthetic-seed', True,
        ray_id=seed['ray_id'],
        achieved_camera_matrix_4x4=camera_matrix(seed))
    coverage = seed_coverage(
        target,
        min(0.24, float(envelope['bounding_radius_from_anchor_m'])),
        seed,
        session_id)
    coverage_path = report_root / session_id / 'coverage' / 'capture_001.npz'
    persist_coverage_snapshot(
        coverage_path, coverage,
        configuration_artifacts=[capability_path, MOBILE_SOURCE / (
            'config/scan_planning_params.yaml')],
        dataset_root=PROJECT_ROOT)
    generation0 = add_target_update_event(
        generation0, 'offline-synthetic-seed', str(coverage_path))
    for event in generation0['events']:
        event['simulation'] = True
        if event.get('capture_id') == 'offline-synthetic-seed':
            event['message'] = (
                'Synthetic accepted seed for command-free NBV scoring; no IK '
                'or motion was run')

    remaining = [
        item for item in accepted if int(item['ray_id']) != int(seed['ray_id'])]
    ranked = rank_next_best_views(
        coverage, remaining,
        [seed['desired_camera_position'][key] for key in ('x', 'y', 'z')])
    positive = [item for item in ranked
                if candidate_meets_minimum_information(item)]
    generation1 = planner_generation_snapshot(
        generation=1,
        history_remaining=remaining,
        ranked=ranked,
        selected=positive,
        planner_rejections={
            int(seed['ray_id']): ['already used by synthetic seed capture']},
        persistent_culls=hard_culls,
        **common)
    generation1 = add_prequalification(
        generation1,
        [dict(item, prequalified=True, reachable=True, safe=True,
              reject_reasons=[]) for item in positive],
        {'candidate_viewpoints': len(positive),
         'prequalified_viewpoints': len(positive),
         'capability_map_mode': 'enforce', 'simulation': True})
    shortlist = [int(item['ray_id']) for item in positive[:6]]
    generation1 = add_bridge_request(
        generation1, 'offline-nbv-policy-preview', shortlist)
    generation1 = add_request_rejection(
        generation1, 'offline-nbv-policy-preview',
        'OFFLINE_SIMULATION_NO_IK',
        'Policy preview ended before IK; no reachability claim or motion')
    generation1 = add_terminal_event(
        generation1, 'cancelled',
        'Command-free large-target policy preview completed')
    for event in generation1['events']:
        event['simulation'] = True

    store = RayMissionDiagnosticsStore(report_root)
    json_path, html_path = store.record(generation0)
    json_path, html_path = store.record(generation1)
    after = (sha256_file(capability_path), capability_path.stat().st_mtime_ns)
    if before != after:
        raise RuntimeError('capability map changed during simulation')
    print('report=%s' % json_path)
    print('html=%s' % html_path)
    print('target_envelope_sha256=%s' % envelope['envelope_sha256'])
    print('envelope_bounding_radius_m=%.4f' % float(
        envelope['bounding_radius_from_anchor_m']))
    print('rays=%d envelope_rejected=%d capability_rejected=%d supported=%d'
          % (len(generated),
             sum(item.get('target_envelope_supported') is False
                 for item in generated),
             sum(not item['prequalified'] for item in prequalified),
             len(accepted)))
    print('nbv_positive=%d shortlisted=%s' % (len(positive), shortlist))
    print('capability_map_sha256=%s mtime_ns=%d unchanged=true' % before)


if __name__ == '__main__':
    main()
