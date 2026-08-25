"""Regression tests for trusted silhouette target envelopes."""

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from piper_mobile_manipulation.target_envelope import (
    build_revolution_envelope,
    envelope_constrained_ray_interval,
    point_to_envelope_distance,
    trusted_silhouette_measurement,
    validate_envelope,
    validate_shape_measurement,
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
    assert len(result['profile_sections']) <= 24
    assert len(result['collision_boxes']) == len(result['profile_sections'])
    assert validate_envelope(result)['envelope_sha256'] == (
        result['envelope_sha256'])


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
