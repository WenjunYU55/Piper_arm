#!/usr/bin/env python3
"""Replay accepted RGB-D scan artifacts through the command-free NBV model."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_PACKAGE = (
    REPOSITORY / 'piper_ros_foxy' / 'src' / 'piper_mobile_manipulation')
if str(SOURCE_PACKAGE) not in sys.path:
    sys.path.insert(0, str(SOURCE_PACKAGE))

from piper_mobile_manipulation.nbv_coverage import (  # noqa: E402
    candidate_information,
    ObjectCoverageModel,
)


FINAL_CAPTURE_AIM_LIMIT_DEG = 5.0


def metadata_records(scan_dir):
    manifest_path = scan_dir / 'manifest.json'
    with manifest_path.open('r', encoding='utf-8') as stream:
        manifest = json.load(stream)
    records = sorted(
        (item for item in manifest.get('files', [])
         if str(item.get('path', '')).endswith('_metadata.yaml')),
        key=lambda item: str(item.get('path', '')))
    count = int(manifest.get('capture_count', len(records)))
    if count < 1 or len(records) < count:
        raise ValueError('scan manifest has no complete accepted generation')
    return records[:count]


def local_metadata_path(scan_dir, record):
    path = Path(str(record['path']))
    if not path.is_absolute():
        path = scan_dir / path
    if not path.is_file():
        path = scan_dir / 'frames' / path.name
    if not path.is_file():
        raise ValueError('capture metadata is unavailable: %s' % path)
    return path


def measured_center(metadata):
    point = metadata.get('target_3d', {}).get('point', {})
    transform = np.asarray(
        metadata['camera_transform']['matrix_4x4'], dtype=np.float64)
    camera = np.asarray([
        float(point['x']), float(point['y']), float(point['z']), 1.0])
    if transform.shape != (4, 4) or not np.all(np.isfinite(camera)):
        raise ValueError('capture target provenance is invalid')
    return transform.dot(camera)[:3]


def synchronized_measured_center(metadata):
    """Return the capture-time qualified target point in ``base_link``.

    Synchronized target depth is expressed in the depth optical frame, while
    the recorded camera transform is for the colour optical frame.  Preserve
    that distinction when replaying the aim that was actually achieved.
    """
    target = metadata.get('synchronized_target_3d', {})
    depth_to_color = metadata.get('color_from_depth_transform', {})
    if not target.get('available') or not depth_to_color.get('available'):
        return None
    point = target.get('point', {})
    camera_point = np.asarray([
        float(point['x']), float(point['y']), float(point['z']), 1.0])
    color_from_depth = np.asarray(
        depth_to_color['matrix_4x4'], dtype=np.float64)
    base_from_color = np.asarray(
        metadata['camera_transform']['matrix_4x4'], dtype=np.float64)
    if (
            camera_point.shape != (4,)
            or color_from_depth.shape != (4, 4)
            or base_from_color.shape != (4, 4)
            or not np.all(np.isfinite(camera_point))
            or not np.all(np.isfinite(color_from_depth))
            or not np.all(np.isfinite(base_from_color))):
        raise ValueError('synchronized target transform provenance is invalid')
    return base_from_color.dot(color_from_depth).dot(camera_point)[:3]


def camera_position(metadata):
    transform = np.asarray(
        metadata['camera_transform']['matrix_4x4'], dtype=np.float64)
    return transform[:3, 3]


def camera_look_direction(metadata):
    transform = np.asarray(
        metadata['camera_transform']['matrix_4x4'], dtype=np.float64)
    direction = transform[:3, :3].dot(
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12 or not np.all(np.isfinite(direction)):
        raise ValueError('capture optical-axis provenance is invalid')
    return direction / norm


def angular_error_deg(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        raise ValueError('aim directions must be non-zero')
    cosine = float(np.clip(
        np.dot(first / first_norm, second / second_norm), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def capture_aim_diagnostic(metadata):
    camera = camera_position(metadata)
    achieved = camera_look_direction(metadata)
    target = synchronized_measured_center(metadata)
    selected = metadata.get('view_selection', {}).get('look_direction')
    result = {
        'synchronized_target_available': target is not None,
        'achieved_camera_position_m': camera.tolist(),
        'achieved_look_direction': achieved.tolist(),
    }
    if selected is not None:
        result['selected_to_achieved_aim_error_deg'] = angular_error_deg(
            selected, achieved)
    if target is None:
        result['final_target_aim_error_deg'] = None
        result['passes_repaired_final_aim_gate'] = None
        return result
    exact = target - camera
    error = angular_error_deg(achieved, exact)
    result.update({
        'synchronized_target_center_m': target.tolist(),
        'exact_capture_time_look_direction': (
            exact / np.linalg.norm(exact)).tolist(),
        'final_target_aim_error_deg': error,
        'passes_repaired_final_aim_gate': bool(
            error <= FINAL_CAPTURE_AIM_LIMIT_DEG + 1e-6),
        'repaired_final_aim_limit_deg': FINAL_CAPTURE_AIM_LIMIT_DEG,
    })
    return result


def parse_center(value):
    if value is None:
        return None
    result = np.asarray([float(item) for item in value.split(',')])
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError('--target-center must be x,y,z in metres')
    return result


def replay(scan_dir, supplied_center=None):
    records = metadata_records(scan_dir)
    metadata = []
    for record in records:
        with local_metadata_path(scan_dir, record).open(
                'r', encoding='utf-8') as stream:
            metadata.append(yaml.safe_load(stream) or {})
    center = (
        supplied_center if supplied_center is not None
        else np.median(np.asarray([
            measured_center(item) for item in metadata]), axis=0))
    previous_unknown = None
    rows = []
    for generation, item in enumerate(metadata, 1):
        model = ObjectCoverageModel()
        snapshot = model.rebuild_from_scan(
            scan_dir, generation, center, 'offline-replay')
        if previous_unknown is not None \
                and snapshot.unknown_voxels > previous_unknown:
            raise ValueError('unknown coverage increased between generations')
        next_information = None
        if generation < len(metadata):
            next_information = candidate_information(
                snapshot, camera_position(metadata[generation]),
                camera_position(item))
        rows.append({
            'generation': generation,
            'unknown_voxels': snapshot.unknown_voxels,
            'surface_voxels': snapshot.surface_voxels,
            'next_actual_unknown_pixels': (
                int(next_information['predicted_unknown_pixels'])
                if next_information is not None else None),
            'next_actual_novel_surface_pixels': (
                int(next_information['novel_surface_pixels'])
                if next_information is not None else None),
            'capture_aim': capture_aim_diagnostic(item),
        })
        previous_unknown = snapshot.unknown_voxels
    return {
        'scan_dir': str(scan_dir),
        'capture_count': len(records),
        'target_center_m': center.tolist(),
        'monotonic_unknown_reduction': True,
        'generations': rows,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Replay a committed scan without ROS, CAN, or motion')
    parser.add_argument('scan_dir', type=Path)
    parser.add_argument(
        '--target-center', help='frozen base-frame x,y,z override in metres')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    result = replay(args.scan_dir.resolve(), parse_center(args.target_center))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print('scan:', result['scan_dir'])
    print('captures:', result['capture_count'])
    print('target center [m]:', ', '.join(
        '%.5f' % value for value in result['target_center_m']))
    print(
        'gen  unknown  surface  next-unknown  next-novel-surface  aim-error')
    for row in result['generations']:
        aim_error = row['capture_aim']['final_target_aim_error_deg']
        print('%3d %8d %8d %13s %18s' % (
            row['generation'], row['unknown_voxels'],
            row['surface_voxels'],
            '-' if row['next_actual_unknown_pixels'] is None
            else row['next_actual_unknown_pixels'],
            '-' if row['next_actual_novel_surface_pixels'] is None
            else row['next_actual_novel_surface_pixels']), end='')
        print(' %9s' % (
            '-' if aim_error is None else '%.3fdeg' % aim_error))


if __name__ == '__main__':
    main()
