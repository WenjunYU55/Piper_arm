import json

import cv2
import numpy as np
import yaml

from piper_mobile_manipulation.surface_coverage import (
    measured_surface_coverage,
    persisted_achieved_history,
)


def write_frame(root, index, depth, mask, transform=None):
    frames = root / 'frames'
    frames.mkdir(exist_ok=True)
    depth_path = frames / ('frame_%03d_depth.npy' % index)
    mask_path = frames / ('frame_%03d_mask.png' % index)
    metadata_path = frames / ('frame_%03d_metadata.yaml' % index)
    np.save(str(depth_path), depth)
    assert cv2.imwrite(str(mask_path), mask)
    metadata = {
        'depth_file_path': str(depth_path),
        'mask_file_path': str(mask_path),
        'depth_encoding': '32FC1',
        'camera_info': {'k': [
            100.0, 0.0, depth.shape[1] / 2.0,
            0.0, 100.0, depth.shape[0] / 2.0,
            0.0, 0.0, 1.0]},
        'camera_transform': {'matrix_4x4': (
            transform if transform is not None else np.eye(4)).tolist()},
        'frame_index': index,
        'target_3d': {'point': {'x': 0.4, 'y': 0.0, 'z': 0.0}},
        'scan_execution': {'plan_id': 'plan-%d' % index},
    }
    metadata_path.write_text(yaml.safe_dump(metadata), encoding='utf-8')
    return {'path': str(metadata_path.relative_to(root))}


def test_persisted_achieved_history_recovers_every_durable_camera_pose(
        tmp_path):
    depth = np.full((60, 60), 0.4, dtype=np.float32)
    mask = np.full((60, 60), 255, dtype=np.uint8)
    files = []
    for index in range(3):
        transform = np.eye(4)
        transform[1, 3] = 0.05 * index
        files.append(write_frame(tmp_path, index, depth, mask, transform))
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'capture_count': 3,
        'files': files,
    }), encoding='utf-8')

    result = persisted_achieved_history(tmp_path)

    assert result['available']
    assert len(result['entries']) == 3
    assert result['entries'][-1]['actual_camera_position'] == {
        'x': 0.0, 'y': 0.1, 'z': 0.0}
    assert result['entries'][-1]['actual_look_at_direction'] == {
        'x': 0.0, 'y': 0.0, 'z': 1.0}
    assert result['target_center'] == {'x': 0.4, 'y': 0.0, 'z': 0.0}


def test_repeated_measured_surface_converges(tmp_path):
    depth = np.full((60, 60), 0.4, dtype=np.float32)
    mask = np.full((60, 60), 255, dtype=np.uint8)
    files = [write_frame(tmp_path, index, depth, mask) for index in range(3)]
    (tmp_path / 'manifest.json').write_text(
        json.dumps({'files': files}), encoding='utf-8')

    result = measured_surface_coverage(
        tmp_path, minimum_views=3, pixel_stride=1,
        convergence_views=2)

    assert result['available']
    assert result['sufficient']
    assert result['per_view_novel_fraction'][0] == 1.0
    assert result['per_view_novel_fraction'][-2:] == [0.0, 0.0]


def test_schema_two_coverage_uses_qualified_target_depth_not_occluder(tmp_path):
    target_depth = np.full((60, 60), 400, dtype=np.uint16)
    raw_depth = np.full((60, 60), 0.20, dtype=np.float32)
    mask = np.full((60, 60), 255, dtype=np.uint8)
    files = []
    for index in range(2):
        record = write_frame(tmp_path, index, raw_depth, mask)
        metadata_path = tmp_path / record['path']
        metadata = yaml.safe_load(metadata_path.read_text(encoding='utf-8'))
        target_depth_path = (
            tmp_path / 'frames' / ('frame_%03d_target_depth.png' % index))
        target_support_path = (
            tmp_path / 'frames' / ('frame_%03d_target_support.png' % index))
        assert cv2.imwrite(str(target_depth_path), target_depth)
        assert cv2.imwrite(str(target_support_path), mask)
        metadata.update({
            'capture_schema_version': 2,
            'target_depth_png_file_path': str(target_depth_path),
            'target_support_mask_file_path': str(target_support_path),
        })
        metadata_path.write_text(
            yaml.safe_dump(metadata), encoding='utf-8')
        files.append(record)
    (tmp_path / 'manifest.json').write_text(
        json.dumps({'files': files}), encoding='utf-8')

    result = measured_surface_coverage(
        tmp_path, minimum_views=2, pixel_stride=1,
        convergence_views=1)

    assert result['available']
    assert result['per_view_novel_fraction'] == [1.0, 0.0]


def test_new_surface_does_not_claim_convergence(tmp_path):
    depth = np.full((60, 60), 0.4, dtype=np.float32)
    mask = np.full((60, 60), 255, dtype=np.uint8)
    files = []
    for index in range(3):
        transform = np.eye(4)
        transform[0, 3] = index * 0.10
        files.append(write_frame(tmp_path, index, depth, mask, transform))
    (tmp_path / 'manifest.json').write_text(
        json.dumps({'files': files}), encoding='utf-8')

    result = measured_surface_coverage(
        tmp_path, minimum_views=3, pixel_stride=1,
        convergence_views=2)

    assert result['available']
    assert not result['sufficient']
    assert not result['converged']
