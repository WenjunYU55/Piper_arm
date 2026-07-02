import numpy as np
import pytest

from piper_mobile_manipulation.obstacle_geometry import (
    BLOCKED, MOVABLE, UNSAFE, aabb_corners, canonical_label,
    effective_classification, normalize_label, obstacle_records,
    project_instance, transform_points,
)


CONFIG = {
    'depth_min_m': 0.25, 'depth_max_m': 1.2,
    'min_valid_depth_pixels': 4, 'min_valid_depth_ratio': 0.4,
    'mask_erode_px': 0, 'bounds_low_percentile': 0.0,
    'bounds_high_percentile': 100.0,
}


def test_whitelist_is_effective_not_advisory():
    assert normalize_label('  Felt_Tip   Marker ') == 'felt tip marker'
    assert canonical_label('whiteboard marker') == 'pen'
    assert effective_classification('marker', False, ['pen']) == MOVABLE
    assert effective_classification('whiteboard marker', False, ['pen']) == MOVABLE
    assert effective_classification('paper', False, ['pen']) == BLOCKED
    assert effective_classification('marker', True, ['pen']) == UNSAFE
    assert effective_classification('unknown', False, ['pen']) == UNSAFE


def test_explained_depth_duplicate_is_suppressed():
    records, suppressed = obstacle_records([
        {'object_id': 1, 'role': 'target', 'label': 'green cube'},
        {'object_id': 2, 'role': 'obstacle', 'label': 'whiteboard marker'},
        {'object_id': 3, 'role': 'obstacle',
         'label': 'depth foreground (whiteboard marker)'},
    ])
    assert set(records) == {2}
    assert suppressed == {3}


def test_unexplained_depth_foreground_remains_blocking():
    records, suppressed = obstacle_records([
        {'object_id': 4, 'role': 'obstacle', 'label': 'unknown depth foreground'},
    ])
    assert set(records) == {4}
    assert not suppressed


def test_projection_centroid_and_bounds():
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[1:5, 2:6] = 1
    depth = np.ones(mask.shape, dtype=np.float64) * 0.5
    k = [100.0, 0.0, 4.0, 0.0, 100.0, 3.0, 0.0, 0.0, 1.0]
    centroid, lower, upper, ratio, count = project_instance(mask, depth, k, CONFIG)
    assert count == 16
    assert ratio == 1.0
    np.testing.assert_allclose(centroid, [-0.0025, -0.0025, 0.5])
    np.testing.assert_allclose(lower, [-0.01, -0.01, 0.5])
    np.testing.assert_allclose(upper, [0.005, 0.005, 0.5])


def test_projection_rejects_bad_depth_support():
    mask = np.ones((3, 3), dtype=np.uint8)
    depth = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match='insufficient_valid_depth_pixels'):
        project_instance(mask, depth, [1, 0, 0, 0, 1, 0, 0, 0, 1], CONFIG)


def test_transform_and_aabb_corner_rotation():
    corners = aabb_corners(np.array([0., 0., 0.]), np.array([1., 2., 3.]))
    transformed = transform_points(corners, (1., 2., 3.), (0., 0., 0., 1.))
    np.testing.assert_allclose(np.min(transformed, axis=0), [1., 2., 3.])
    np.testing.assert_allclose(np.max(transformed, axis=0), [2., 4., 6.])
