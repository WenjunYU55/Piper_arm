"""Regression tests for trusted silhouette target envelopes."""

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from piper_mobile_manipulation.target_envelope import (
    build_capture_model_seed,
    build_revolution_envelope,
    canonical_sha256,
    clipped_shape_rejection,
    classify_centered_silhouette,
    coverage_sphere_from_envelope,
    envelope_constrained_ray_interval,
    point_to_envelope_distance,
    TargetSilhouetteClippedError,
    trusted_silhouette_measurement,
    validate_envelope,
    validate_capture_model_seed,
    validate_shape_measurement,
    validate_shape_rejection,
)


CAMERA_MATRIX = np.asarray([
    [400.0, 0.0, 320.0],
    [0.0, 400.0, 240.0],
    [0.0, 0.0, 1.0],
])
CAMERA_INFO = {
    'available': True,
    'width': 640,
    'height': 480,
    'fx': 400.0,
    'fy': 400.0,
}


def header():
    return SimpleNamespace(
        stamp=SimpleNamespace(sec=12, nanosec=345),
        frame_id='camera_color_optical_frame')


def rectangle_shape(width_px=35, height_px=35):
    mask = np.zeros((480, 640), dtype=np.uint8)
    x0, y0 = 320 - width_px // 2, 240 - height_px // 2
    mask[y0:y0 + height_px, x0:x0 + width_px] = 255
    support = np.ones((height_px, width_px), dtype=bool)
    depth = np.full((height_px, width_px), 0.40, dtype=float)
    return trusted_silhouette_measurement(
        mask, support, (x0, y0), depth, CAMERA_MATRIX, header(), 0.9)


def envelope(width_px=35, height_px=35, anchor=(0.0, 0.0, 0.40)):
    return build_revolution_envelope(
        rectangle_shape(width_px, height_px), np.eye(4), anchor)


def capture_transform(stamp=(12, 30_000_345)):
    return {
        'available': True,
        'header': {
            'stamp': {'sec': stamp[0], 'nanosec': stamp[1]},
            'frame_id': 'base_link',
        },
        'child_frame_id': 'camera_color_optical_frame',
        'matrix_4x4': np.eye(4).tolist(),
    }


def test_capture_model_seed_binds_shape_to_synchronized_base_transform():
    shape = rectangle_shape()
    seed = build_capture_model_seed(shape, capture_transform())

    assert validate_capture_model_seed(seed) == seed
    assert seed['shape']['measurement_sha256'] == \
        shape['measurement_sha256']
    assert seed['shape_transform_delta_sec'] == pytest.approx(0.03)

    changed = copy.deepcopy(seed)
    changed['base_from_camera']['matrix_4x4'][0][3] = 0.1
    with pytest.raises(ValueError, match='digest'):
        validate_capture_model_seed(changed)


def test_capture_model_seed_rejects_transform_outside_capture_sync_window():
    with pytest.raises(ValueError, match='not synchronized'):
        build_capture_model_seed(
            rectangle_shape(), capture_transform(stamp=(13, 0)))


def test_only_semantic_component_overlapping_qualified_depth_is_used():
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    mask[50:80, 70:105] = 255
    support = np.ones((20, 20), dtype=bool)
    depth = np.full((20, 20), 0.40, dtype=float)
    camera = np.asarray([
        [100.0, 0.0, 60.0],
        [0.0, 100.0, 50.0],
        [0.0, 0.0, 1.0],
    ])

    shape = trusted_silhouette_measurement(
        mask, support, (10, 10), depth, camera, header(), 0.8)

    assert shape['mask_pixel_count'] == 400
    assert shape['qualified_depth_pixel_count'] == 400
    assert len(shape['silhouette_points_camera_m']) >= 4


@pytest.mark.parametrize('x0, y0, width, height', (
    (0, 20, 20, 20),
    (100, 20, 20, 20),
    (20, 0, 20, 20),
    (20, 80, 20, 20),
))
def test_component_touching_any_image_border_is_rejected_as_clipped(
        x0, y0, width, height):
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[y0:y0 + height, x0:x0 + width] = 255
    support = np.ones((height, width), dtype=bool)
    depth = np.full((height, width), 0.40, dtype=float)
    camera = np.asarray([
        [100.0, 0.0, 60.0],
        [0.0, 100.0, 50.0],
        [0.0, 0.0, 1.0],
    ])

    with pytest.raises(TargetSilhouetteClippedError) as caught:
        trusted_silhouette_measurement(
            mask, support, (x0, y0), depth, camera, header(), 0.8)

    assert caught.value.near_depth_m == pytest.approx(0.40)


def test_one_pixel_image_border_margin_is_a_complete_silhouette():
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[1:99, 1:119] = 255
    support = np.ones((98, 118), dtype=bool)
    depth = np.full((98, 118), 0.40, dtype=float)
    camera = np.asarray([
        [100.0, 0.0, 60.0],
        [0.0, 100.0, 50.0],
        [0.0, 0.0, 1.0],
    ])

    shape = trusted_silhouette_measurement(
        mask, support, (1, 1), depth, camera, header(), 0.8)

    assert shape['valid'] is True


def test_clipped_rejection_is_exact_stamp_depth_and_digest_bound():
    rejection = clipped_shape_rejection(header(), 0.73)

    assert rejection['valid'] is False
    assert rejection['rejection_code'] == 'TARGET_SILHOUETTE_CLIPPED'
    assert rejection['near_depth_m'] == pytest.approx(0.73)
    assert validate_shape_rejection(rejection) == rejection
    changed = copy.deepcopy(rejection)
    changed['near_depth_m'] = 0.80
    with pytest.raises(ValueError, match='digest'):
        validate_shape_rejection(changed)


def test_centered_crop_classification_uses_25_cm_and_3_m_center_distance():
    assert classify_centered_silhouette(
        clipped_shape_rejection(header(), 0.40), 0.25)[0] == 'TOO_CLOSE'
    assert classify_centered_silhouette(
        clipped_shape_rejection(header(), 0.20), 0.40)[0] == 'RETRY_FARTHER'
    assert classify_centered_silhouette(
        clipped_shape_rejection(header(), 0.20), 0.80)[0] == 'RETRY_FARTHER'
    assert classify_centered_silhouette(
        clipped_shape_rejection(header(), 0.20), 3.00)[0] == 'TOO_LARGE'


def test_shape_digest_rejects_mutated_or_nonfinite_geometry():
    shape = rectangle_shape()
    assert validate_shape_measurement(shape)['valid'] is True
    changed = copy.deepcopy(shape)
    changed['near_depth_m'] = 0.7
    with pytest.raises(ValueError, match='digest'):
        validate_shape_measurement(changed)
    changed = copy.deepcopy(shape)
    changed['silhouette_points_camera_m'][0][0] = math.nan
    with pytest.raises(ValueError, match='malformed'):
        validate_shape_measurement(changed)


def test_square_silhouette_uses_stable_vertical_fallback_and_metric_size():
    result = envelope()

    assert result['axis_source'] in (
        'base_vertical_projection', 'camera_image_vertical_fallback')
    assert result['visible_axial_span_m'] == pytest.approx(0.034, abs=0.003)
    assert result['visible_transverse_span_m'] == pytest.approx(
        0.034, abs=0.003)
    assert result['rotation_origin_depth_m'] == pytest.approx(
        0.5 * result['visible_transverse_span_m'], abs=1e-8)
    surface = np.asarray(result['visible_surface_center_m'])
    origin = np.asarray(result['axis_origin_m'])
    normal = np.asarray(result['camera_normal_at_lock'])
    assert float(np.dot(origin - surface, normal)) == pytest.approx(
        result['rotation_origin_depth_m'], abs=1e-8)
    assert result['planning_anchor_m'] == result['axis_origin_m']
    assert result['bootstrap_anchor_m'] == [0.0, 0.0, 0.4]
    assert len(result['profile_sections']) <= 24
    assert len(result['collision_boxes']) == len(result['profile_sections'])
    assert 3 <= len(result['visible_silhouette_points_m']) <= 256
    assert validate_envelope(result)['envelope_sha256'] == (
        result['envelope_sha256'])


@pytest.mark.parametrize('width_px, height_px', ((40, 40), (300, 40)))
def test_coverage_sphere_diameter_matches_largest_uninflated_span(
        width_px, height_px):
    result = envelope(width_px=width_px, height_px=height_px)

    sphere = coverage_sphere_from_envelope(result)

    expected = max(
        result['visible_axial_span_m'],
        result['visible_transverse_span_m'])
    assert sphere['diameter_m'] == pytest.approx(expected)
    assert 2.0 * sphere['radius_m'] == pytest.approx(expected)
    assert sphere['center_m'] == result['axis_origin_m']
    assert sphere['source'] == 'qualified_revolved_target_size'


def test_recorded_visible_outline_is_exact_base_frame_source_geometry():
    shape = rectangle_shape(width_px=60, height_px=25)
    transform = np.eye(4)
    transform[:3, 3] = [0.1, -0.2, 0.3]

    result = build_revolution_envelope(
        shape, transform, [0.1, -0.2, 0.7])
    source = np.asarray(shape['silhouette_points_camera_m'])
    expected = source + transform[:3, 3]

    assert np.allclose(result['visible_silhouette_points_m'], expected)


def test_legacy_envelope_without_visible_outline_remains_valid():
    result = envelope()
    result.pop('visible_silhouette_points_m')
    unsigned = dict(result)
    unsigned.pop('envelope_sha256')
    result['envelope_sha256'] = canonical_sha256(unsigned)

    assert validate_envelope(result) == result


def test_legacy_envelope_without_bootstrap_anchor_remains_valid():
    result = envelope()
    result.pop('bootstrap_anchor_m')
    unsigned = dict(result)
    unsigned.pop('envelope_sha256')
    result['envelope_sha256'] = canonical_sha256(unsigned)

    assert validate_envelope(result) == result


def test_malformed_visible_outline_fails_before_rendering_or_handoff():
    result = envelope()
    result['visible_silhouette_points_m'] = [[math.nan, 0.0, 0.4]] * 3

    with pytest.raises(ValueError, match='outline'):
        validate_envelope(result)


def test_elongated_silhouette_uses_mask_major_axis():
    result = envelope(width_px=100, height_px=20)

    assert result['axis_source'] == 'mask_major_axis'
    assert result['axis_anisotropy_ratio'] > 1.2
    assert result['visible_axial_span_m'] > (
        3.0 * result['visible_transverse_span_m'])


def test_surface_clearance_raises_minimum_standoff_for_large_target():
    result = envelope(width_px=100, height_px=100)
    anchor = result['planning_anchor_m']

    interval = envelope_constrained_ray_interval(
        anchor, [1.0, 0.0, 0.0], 0.28, 0.80, result, CAMERA_INFO)

    assert interval is not None
    assert interval[0] > 0.28
    camera = np.asarray(anchor) + np.asarray([interval[0], 0.0, 0.0])
    assert point_to_envelope_distance(camera, result) >= 0.25 - 1e-6


def test_no_safe_interval_when_clearance_and_fov_exceed_ray_bound():
    result = envelope(width_px=220, height_px=220)

    assert envelope_constrained_ray_interval(
        result['planning_anchor_m'], [1.0, 0.0, 0.0],
        0.28, 0.30, result, CAMERA_INFO) is None


def test_ray_query_does_not_mutate_frozen_envelope_record():
    result = envelope()
    snapshot = copy.deepcopy(result)

    envelope_constrained_ray_interval(
        result['planning_anchor_m'], [0.0, 1.0, 0.0],
        0.28, 0.80, result, CAMERA_INFO)

    assert result == snapshot
