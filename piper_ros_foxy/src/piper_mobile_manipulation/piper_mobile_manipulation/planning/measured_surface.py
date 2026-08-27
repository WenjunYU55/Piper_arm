"""Measured object-surface coverage from persisted RGB-D capture records."""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml


def _inside(root, value):
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if root != path and root not in path.parents:
        raise ValueError('capture artifact escapes the dataset root')
    return path


def _frame_voxels(dataset, metadata, voxel_size_m, pixel_stride):
    depth_path = _inside(dataset, metadata['depth_file_path'])
    mask_path = _inside(dataset, metadata['mask_file_path'])
    depth = np.load(str(depth_path), allow_pickle=False).astype(np.float64)
    if '16U' in str(metadata.get('depth_encoding', '')) or str(
            metadata.get('depth_encoding', '')) in ('mono16', '16UC1'):
        depth *= 0.001
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != depth.shape:
        raise ValueError('capture mask/depth shape mismatch')
    k = np.asarray(metadata['camera_info']['k'], dtype=np.float64)
    transform = np.asarray(
        metadata['camera_transform']['matrix_4x4'], dtype=np.float64)
    if k.shape != (9,) or transform.shape != (4, 4):
        raise ValueError('capture calibration metadata is invalid')
    step = max(1, int(pixel_stride))
    rows, cols = np.nonzero(
        (mask[::step, ::step] > 0)
        & np.isfinite(depth[::step, ::step])
        & (depth[::step, ::step] > 0.10)
        & (depth[::step, ::step] < 1.50))
    if not len(rows):
        return set()
    rows = rows * step
    cols = cols * step
    z = depth[rows, cols]
    fx, fy, cx, cy = k[0], k[4], k[2], k[5]
    if min(fx, fy) <= 0.0 or not all(
            math.isfinite(value) for value in (fx, fy, cx, cy)):
        raise ValueError('capture intrinsics are invalid')
    camera = np.column_stack((
        (cols - cx) * z / fx,
        (rows - cy) * z / fy,
        z,
        np.ones_like(z),
    ))
    base = camera.dot(transform.T)[:, :3]
    quantized = np.floor(base / float(voxel_size_m) + 0.5).astype(np.int64)
    return {tuple(int(value) for value in row) for row in quantized}


def persisted_achieved_history(scan_dir):
    """
    Recover achieved camera poses from immutable accepted-frame metadata.

    Transient ROS history normally supplies these values.  A persisted capture
    can arrive just before that callback, however, so durable frame metadata is
    the fallback authority for completion and terminal reporting.
    """
    dataset = Path(str(scan_dir or '')).resolve()
    manifest_path = dataset / 'manifest.json'
    result = {
        'available': False,
        'entries': [],
        'target_center': None,
        'reason': 'scan dataset is unavailable',
    }
    if not manifest_path.is_file():
        return result
    try:
        with manifest_path.open('r', encoding='utf-8') as stream:
            manifest = json.load(stream)
        metadata_files = sorted(
            (item for item in manifest.get('files', [])
             if str(item.get('path', '')).endswith('_metadata.yaml')),
            key=lambda item: str(item.get('path', '')))
        entries = []
        target_center = None
        for record in metadata_files:
            metadata_path = _inside(dataset, record['path'])
            with metadata_path.open('r', encoding='utf-8') as stream:
                metadata = yaml.safe_load(stream) or {}
            transform = np.asarray(
                metadata['camera_transform']['matrix_4x4'],
                dtype=np.float64)
            if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
                raise ValueError('capture camera transform is invalid')
            camera_position = transform[:3, 3]
            optical_z = transform[:3, 2]
            optical_norm = float(np.linalg.norm(optical_z))
            if optical_norm <= 1e-9:
                raise ValueError('capture optical direction is invalid')
            optical_z = optical_z / optical_norm
            entries.append({
                'index': int(metadata.get('frame_index', len(entries))),
                'actual_camera_position': dict(zip(
                    ('x', 'y', 'z'),
                    (float(value) for value in camera_position))),
                'actual_look_at_direction': dict(zip(
                    ('x', 'y', 'z'),
                    (float(value) for value in optical_z))),
                'plan_id': str(
                    (metadata.get('scan_execution') or {}).get(
                        'plan_id', '')),
                'source': 'persisted_capture_metadata',
            })
            if target_center is None:
                target = metadata.get('target_3d') or {}
                point = target.get('point') or {}
                candidate = np.asarray([
                    point.get('x'), point.get('y'), point.get('z')],
                    dtype=np.float64)
                if candidate.shape == (3,) and np.all(np.isfinite(candidate)):
                    frame_id = str(
                        (target.get('header') or {}).get('frame_id', ''))
                    if frame_id and frame_id != str(
                            (metadata.get('camera_transform') or {}).get(
                                'header', {}).get('frame_id', '')):
                        candidate = transform.dot(np.append(candidate, 1.0))[:3]
                    target_center = dict(zip(
                        ('x', 'y', 'z'),
                        (float(value) for value in candidate)))
        declared = int(manifest.get('capture_count', len(entries)))
        if declared != len(entries):
            raise ValueError(
                'manifest capture count does not match frame metadata')
        result.update({
            'available': True,
            'entries': entries,
            'target_center': target_center,
            'reason': 'persisted achieved history recovered',
        })
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        result['reason'] = 'persisted achieved history invalid: %s' % exc
    return result


def measured_surface_coverage(
        scan_dir, minimum_views=8, voxel_size_m=0.010,
        convergence_gain=0.02, convergence_views=3, pixel_stride=3):
    """
    Return measured novel-surface gain for every accepted capture.

    Voxels live in ``base_link`` using each capture-time camera transform, so
    repeated images do not manufacture progress merely by increasing the
    capture count.  Invalid records fail closed in the returned diagnostics.
    """
    dataset = Path(str(scan_dir or '')).resolve()
    manifest_path = dataset / 'manifest.json'
    result = {
        'available': False,
        'sufficient': False,
        'valid_surface_views': 0,
        'covered_voxels': 0,
        'per_view_novel_fraction': [],
        'recent_gain': None,
        'converged': False,
        'reason': 'scan dataset is unavailable',
    }
    if not manifest_path.is_file():
        return result
    try:
        with manifest_path.open('r', encoding='utf-8') as stream:
            manifest = json.load(stream)
        metadata_files = sorted(
            (item for item in manifest.get('files', [])
             if str(item.get('path', '')).endswith('_metadata.yaml')),
            key=lambda item: str(item.get('path', '')))
        occupied = set()
        gains = []
        for record in metadata_files:
            metadata_path = _inside(dataset, record['path'])
            with metadata_path.open('r', encoding='utf-8') as stream:
                metadata = yaml.safe_load(stream) or {}
            frame = _frame_voxels(
                dataset, metadata, voxel_size_m, pixel_stride)
            if not frame:
                continue
            added = len(frame - occupied)
            gains.append(float(added) / float(max(1, len(frame))))
            occupied.update(frame)
        recent = gains[-max(1, int(convergence_views)):]
        converged = bool(
            len(recent) >= max(1, int(convergence_views))
            and all(value <= float(convergence_gain) for value in recent))
        enough_views = len(gains) >= int(minimum_views)
        enough_geometry = len(occupied) >= 100
        result.update({
            'available': True,
            'sufficient': bool(enough_views and enough_geometry and converged),
            'valid_surface_views': len(gains),
            'covered_voxels': len(occupied),
            'per_view_novel_fraction': gains,
            'recent_gain': gains[-1] if gains else None,
            'converged': converged,
            'reason': (
                'measured surface gain converged'
                if enough_views and enough_geometry and converged
                else 'measured surface gain has not converged'),
        })
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        result['reason'] = 'surface coverage invalid: %s' % exc
    return result
