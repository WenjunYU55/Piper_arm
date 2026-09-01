import numpy as np
import pytest

from reconstruction.native_depth import (
    GEOMETRY_SOURCES,
    project_native_depth,
    replay_native_target_geometry,
)


def metadata():
    identity = np.eye(4)
    color_from_depth = identity.copy()
    color_from_depth[0, 3] = 0.01
    base_from_color = identity.copy()
    base_from_color[1, 3] = 0.02
    return {
        'native_depth_camera_info': {
            'available': True,
            'k': [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            'd': [0.0] * 5,
        },
        'camera_info': {
            'available': True,
            'k': [2.0, 0.0, 3.0, 0.0, 2.0, 3.0, 0.0, 0.0, 1.0],
            'd': [0.0] * 5,
            'distortion_model': 'plumb_bob',
        },
        'color_from_depth_transform': {
            'available': True,
            'matrix_4x4': color_from_depth.tolist(),
        },
        'camera_transform': {
            'available': True,
            'matrix_4x4': base_from_color.tolist(),
        },
        'confidence_quality': {'confidence_threshold': 8},
    }


def projected_capture_inputs():
    capture = metadata()
    depth = np.full((3, 3), 1000, dtype=np.uint16)
    confidence = np.full((3, 3), 15, dtype=np.uint8)
    color_depth = np.zeros((7, 7), dtype=np.uint16)
    color_support = np.zeros((7, 7), dtype=np.uint8)
    color = np.arange(7 * 7 * 3, dtype=np.uint8).reshape(7, 7, 3)
    projection = project_native_depth(
        depth, capture['native_depth_camera_info']['k'],
        capture['camera_info']['k'], capture['camera_info']['d'],
        capture['camera_info']['distortion_model'],
        capture['color_from_depth_transform']['matrix_4x4'],
        color_depth.shape)
    rows, columns = np.nonzero(projection['valid'])
    u = projection['u'][rows, columns]
    v = projection['v'][rows, columns]
    color_depth[v, u] = np.rint(
        projection['color_depth_m'][rows, columns] * 1000.0).astype(
            np.uint16)
    color_support[v, u] = 255
    return capture, depth, confidence, color_depth, color_support, color


def test_native_replay_recovers_contiguous_grid_without_expanding_support():
    capture, depth, confidence, color_depth, support, color = \
        projected_capture_inputs()
    assert np.count_nonzero(support) == 9
    assert np.count_nonzero(support[1::2, 1::2]) == 9

    replay = replay_native_target_geometry(
        capture, depth, confidence, color_depth, support, color)

    assert GEOMETRY_SOURCES == ('projected_color_depth', 'native_depth')
    assert np.array_equal(replay['depth_mm'], depth)
    assert np.all(replay['mask'] == 255)
    assert replay['report']['recovered_native_target_points'] == 9
    assert replay['report']['target_support_expanded'] is False
    expected = (
        np.asarray(capture['camera_transform']['matrix_4x4'])
        @ np.asarray(
            capture['color_from_depth_transform']['matrix_4x4']))
    assert replay['base_from_camera'] == pytest.approx(expected)


def test_native_replay_fails_if_projected_support_cannot_be_correlated():
    capture, depth, confidence, color_depth, support, color = \
        projected_capture_inputs()
    support[0, 0] = 255
    color_depth[0, 0] = 1000
    with pytest.raises(ValueError, match='does not correlate every accepted'):
        replay_native_target_geometry(
            capture, depth, confidence, color_depth, support, color)


def test_native_replay_fails_closed_on_unqualified_confidence():
    capture, depth, confidence, color_depth, support, color = \
        projected_capture_inputs()
    confidence[:] = 7
    with pytest.raises(ValueError, match='no correlated target samples'):
        replay_native_target_geometry(
            capture, depth, confidence, color_depth, support, color)
