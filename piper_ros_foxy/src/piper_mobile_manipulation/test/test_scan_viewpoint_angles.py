from types import SimpleNamespace

import numpy as np
import pytest

import piper_mobile_manipulation.scan_viewpoint_planner_node as planner_module
from piper_mobile_manipulation.scan_viewpoint_planner_node import (
    build_viewpoint_angles,
    pending_first_ray_framing_retry,
    provisional_first_ray_allowed,
    retired_view_history,
    restrict_to_framing_retry,
    ScanViewpointPlannerNode,
    target_shape_failure_code,
    target_frame_rejection_reason,
    viewpoint_replan_required,
    viewpoint_refresh_required,
)
from piper_mobile_manipulation.viewpoint_rays import build_ray_samples
from piper_mobile_manipulation.target_envelope import (
    build_capture_model_seed,
    canonical_sha256,
)


def shape_header():
    return SimpleNamespace(
        stamp=SimpleNamespace(sec=12, nanosec=345),
        frame_id='camera_color_optical_frame')


def test_reachable_workstation_sector_has_five_ordered_views():
    assert build_viewpoint_angles(165, 60, 15, 20) == [
        135.0, 150.0, 165.0, 180.0, 195.0,
    ]


def test_full_circle_does_not_duplicate_equivalent_endpoint():
    angles = build_viewpoint_angles(0, 360, 90, 20)
    assert angles == [-180.0, -90.0, 0.0, 90.0]


def test_downsampling_retains_sector_endpoints():
    assert build_viewpoint_angles(10, 60, 10, 3) == [-20.0, 10.0, 40.0]


def test_live_thirteen_view_sector_retains_reachable_endpoints():
    angles = build_viewpoint_angles(147.5, 55, 55 / 12, 20)
    assert len(angles) == 13
    assert angles[0] == 120.0
    assert angles[-1] == 175.0


def test_live_ray_region_spans_both_y_sides_and_elevations_once():
    angles = build_viewpoint_angles(180, 180, 7.5, 25)
    pitches = [
        -50.0 + offset
        for offset in (35.0, 25.0, 15.0, 5.0, -5.0, -15.0, -25.0)
    ]
    candidates = [
        (angle, pitch)
        for pitch in pitches for angle in angles]

    assert len(angles) == 25
    assert len(candidates) == 175
    assert angles[0] == 90.0
    assert angles[-1] == 270.0
    # The failed physical pose landed at 195.6 degrees.  The denser region has
    # a meaningful but compact successor instead of jumping directly to 210.
    assert 6.0 <= 202.5 - 195.6 <= 8.0
    assert sorted(set(pitch for _angle, pitch in candidates)) == [
        -75.0, -65.0, -55.0, -45.0, -35.0, -25.0, -15.0]

    samples = build_ray_samples(
        'target_sector', 175, 180.0, 180.0, sorted(set(pitches)))
    assert len(samples) == 175
    assert min(angle for angle, _pitch in samples) == 90.0
    assert max(angle for angle, _pitch in samples) == 270.0


def test_full_sphere_has_exact_requested_count_and_both_vertical_halves():
    samples = build_ray_samples('full_sphere', 175)
    assert len(samples) == 175
    assert len(set(samples)) == 175
    assert min(pitch for _angle, pitch in samples) < -80.0
    assert max(pitch for _angle, pitch in samples) > 80.0
    assert all(-180.0 <= angle < 180.0 for angle, _pitch in samples)


def test_upper_hemisphere_is_360_degree_and_never_below_target():
    samples = build_ray_samples('upper_hemisphere', 120)
    assert len(samples) == 120
    assert all(-90.0 <= pitch < 0.0 for _angle, pitch in samples)
    quadrants = {
        int((angle + 180.0) // 90.0) for angle, _pitch in samples}
    assert quadrants == {0, 1, 2, 3}


def complete_shape():
    shape = {
        'schema_version': 1,
        'valid': True,
        'header': {
            'stamp': {'sec': 12, 'nanosec': 345},
            'frame_id': 'camera_color_optical_frame',
        },
        'source': 'fresh_mask_qualified_depth',
        'silhouette_points_camera_m': [
            [-0.05, -0.10, 0.40],
            [0.05, -0.10, 0.40],
            [0.05, 0.10, 0.40],
            [-0.05, 0.10, 0.40],
        ],
        'near_depth_m': 0.40,
        'mask_pixel_count': 1000,
        'qualified_depth_pixel_count': 900,
        'measurement_confidence': 0.9,
        'camera_info': {
            'width': 640, 'height': 480,
            'fx': 600.0, 'fy': 600.0, 'cx': 320.0, 'cy': 240.0,
        },
    }
    shape['measurement_sha256'] = canonical_sha256(shape)
    return shape


def complete_capture_model_seed():
    return build_capture_model_seed(complete_shape(), {
        'header': {
            'stamp': {'sec': 12, 'nanosec': 30_000_345},
            'frame_id': 'base_link',
        },
        'child_frame_id': 'camera_color_optical_frame',
        'matrix_4x4': np.eye(4).tolist(),
    })


def test_ray_pool_is_generated_once_and_ignores_later_target_shift():
    calls = []
    parameters = {
        'camera_pitch_offsets_deg': [0.0, -10.0],
        'camera_pitch_deg': -20.0,
    }
    harness = SimpleNamespace(
        ray_pool_session_id='',
        ray_pool_target_center=None,
        ray_pool_frame_id='',
        ray_pool=None,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
    )

    def make_ray(ray_id, angle, center, frame_id, pitch):
        calls.append((ray_id, angle, dict(center), frame_id, pitch))
        return {'ray_id': ray_id, 'target_object_center': dict(center)}

    harness.make_ray_viewpoint = make_ray
    history = {'session_id': 'static-ray-session'}
    first_center = {'x': 0.4, 'y': 0.0, 'z': 0.0}
    shifted_center = {'x': 0.46, 'y': 0.0, 'z': 0.0}

    frozen, first = ScanViewpointPlannerNode.frozen_ray_pool(
        harness, history, first_center, 'base_link',
        [(90.0, -20.0), (180.0, -30.0)])
    reused, second = ScanViewpointPlannerNode.frozen_ray_pool(
        harness, history, shifted_center, 'base_link', [(0.0, 40.0)])

    assert frozen == first_center
    assert reused == first_center
    assert first == second
    assert len(first) == 2
    assert len(calls) == 2


def test_target_envelope_culls_are_retained_before_pool_is_frozen():
    calls = []
    harness = SimpleNamespace(
        ray_pool_session_id='',
        ray_pool_target_center=None,
        ray_pool_frame_id='',
        ray_pool=None,
        ray_pool_target_envelope=None,
        ray_pool_envelope_rejected_rays=0,
    )

    def make_ray(ray_id, angle, center, frame_id, pitch, envelope=None):
        calls.append((ray_id, envelope))
        return {
            'ray_id': ray_id,
            'target_object_center': dict(center),
            'target_envelope_supported': ray_id != 0,
        }

    harness.make_ray_viewpoint = make_ray
    history = {'session_id': 'envelope-session'}
    envelope = {
        'envelope_sha256': 'a' * 64,
        'planning_anchor_m': [0.4, 0.0, 0.0],
    }
    center = {'x': 0.4, 'y': 0.0, 'z': 0.0}

    _frozen, first = ScanViewpointPlannerNode.frozen_ray_pool(
        harness, history, center, 'base_link',
        [(0.0, 0.0), (180.0, 0.0)], envelope=envelope)
    _reused, second = ScanViewpointPlannerNode.frozen_ray_pool(
        harness, history, {'x': 0.5, 'y': 0.0, 'z': 0.0}, 'base_link',
        [(90.0, 0.0)], envelope={
            'envelope_sha256': 'b' * 64,
            'planning_anchor_m': [0.5, 0.0, 0.0],
        })

    assert first == second
    assert [item['ray_id'] for item in first] == [0, 1]
    assert first[0]['target_envelope_supported'] is False
    assert len(calls) == 2
    assert harness.ray_pool_envelope_rejected_rays == 1
    assert harness.ray_pool_target_envelope == envelope


def test_qualified_pool_replaces_bootstrap_once_at_model_centre():
    calls = []
    harness = SimpleNamespace(
        ray_pool_session_id='',
        ray_pool_target_center=None,
        ray_pool_frame_id='',
        ray_pool=None,
        ray_pool_phase='',
        ray_pool_target_envelope=None,
        ray_pool_envelope_rejected_rays=0,
    )

    def make_ray(ray_id, angle, center, frame_id, pitch, envelope=None):
        calls.append((ray_id, dict(center), envelope is not None))
        return {
            'ray_id': ray_id,
            'target_object_center': dict(center),
            'target_envelope_supported': True,
        }

    harness.make_ray_viewpoint = make_ray
    history = {
        'session_id': 'two-phase-session',
        'coverage_target_center': {'x': 0.40, 'y': 0.0, 'z': 0.10},
    }
    bootstrap_center, _bootstrap = ScanViewpointPlannerNode.frozen_ray_pool(
        harness, history, history['coverage_target_center'], 'base_link',
        [(0.0, 0.0), (180.0, 0.0)])
    envelope = {
        'envelope_sha256': 'a' * 64,
        'planning_anchor_m': [0.35, -0.01, 0.14],
    }
    final_center, final = ScanViewpointPlannerNode.frozen_ray_pool(
        harness, history, history['coverage_target_center'], 'base_link',
        [(0.0, 0.0), (180.0, 0.0)], envelope=envelope)
    reused_center, reused = ScanViewpointPlannerNode.frozen_ray_pool(
        harness, history, {'x': 0.8, 'y': 0.8, 'z': 0.8}, 'base_link',
        [(90.0, 0.0)], envelope=envelope)

    assert bootstrap_center == history['coverage_target_center']
    assert final_center == {'x': 0.35, 'y': -0.01, 'z': 0.14}
    assert reused_center == final_center
    assert reused == final
    assert harness.ray_pool_phase == 'qualified'
    assert len(calls) == 4


def test_generation_zero_is_provisional_until_executor_qualifies_shape():
    assert provisional_first_ray_allowed({
        'session_id': 'session-a',
        'accepted_views': 0,
        'qualified_target_shape': None,
    })
    assert not provisional_first_ray_allowed({
        'session_id': 'session-a',
        'accepted_views': 0,
        'qualified_target_shape': complete_shape(),
    })
    assert not provisional_first_ray_allowed({
        'session_id': 'session-a',
        'accepted_views': 1,
        'qualified_target_shape': None,
    })


def test_planner_refuses_to_revolve_unqualified_rough_shape():
    harness = SimpleNamespace(
        ray_pool_session_id='',
        ray_pool_target_envelope=None,
        get_parameter=lambda name: SimpleNamespace(
            value={'planning_frame_id': 'base_link'}[name]),
        camera_info_summary=lambda: {'available': True},
    )
    envelope, error = ScanViewpointPlannerNode.target_envelope_for(
        harness, {'x': 0.4, 'y': 0.0, 'z': 0.1}, {
            'session_id': 'session-a',
            'qualified_target_shape': None,
        })

    assert envelope is None
    assert 'accepted capture target model seed' in error
    assert target_shape_failure_code(error) == 'TARGET_SHAPE_UNAVAILABLE'


def test_planner_revolves_capture_seed_without_delayed_live_tf(monkeypatch):
    observed = {}
    harness = SimpleNamespace(
        ray_pool_session_id='',
        ray_pool_target_envelope=None,
        get_parameter=lambda name: SimpleNamespace(
            value={'planning_frame_id': 'base_link'}[name]),
        camera_info_summary=lambda: {'available': True},
        tf_buffer=SimpleNamespace(lookup_transform=lambda *_args: pytest.fail(
            'capture-bound model seeds must not query live TF')),
    )

    def build(shape, matrix, anchor):
        observed.update({'shape': shape, 'matrix': matrix, 'anchor': anchor})
        return {'envelope_sha256': 'e' * 64}

    monkeypatch.setattr(planner_module, 'build_revolution_envelope', build)
    shape = complete_shape()
    seed = complete_capture_model_seed()
    envelope, error = ScanViewpointPlannerNode.target_envelope_for(
        harness, {'x': 0.4, 'y': -0.02, 'z': 0.1}, {
            'session_id': 'session-a',
            'qualified_target_shape': shape,
            'qualified_target_model_seed': seed,
        })

    assert error == ''
    assert envelope == {'envelope_sha256': 'e' * 64}
    assert observed['shape'] == shape
    assert np.allclose(observed['matrix'], np.eye(4))
    assert observed['anchor'] == [0.4, -0.02, 0.1]


def test_first_crop_retry_keeps_same_ray_and_moves_endpoint_farther():
    history = {
        'accepted_views': 0,
        'rejected_entries': [{
            'framing_retry_ray_id': 7,
            'framing_retry_min_standoff_m': 0.50,
        }],
    }
    viewpoints = [{
        'ray_id': 7,
        'ray_direction': {'x': 1.0, 'y': 0.0, 'z': 0.0},
        'ray_min_standoff_m': 0.28,
        'ray_max_standoff_m': 0.80,
        'ray_preferred_max_standoff_m': 0.50,
        'desired_camera_position': {'x': 0.79, 'y': 0.0, 'z': 0.1},
        'desired_look_at_direction': {'x': -1.0, 'y': 0.0, 'z': 0.0},
    }, {
        'ray_id': 8,
        'ray_direction': {'x': 0.0, 'y': 1.0, 'z': 0.0},
        'ray_min_standoff_m': 0.28,
        'ray_max_standoff_m': 0.80,
        'ray_preferred_max_standoff_m': 0.50,
    }]

    assert pending_first_ray_framing_retry(history) == (7, 0.50)
    retried, active = restrict_to_framing_retry(
        viewpoints, history, {'x': 0.4, 'y': 0.0, 'z': 0.1})

    assert active
    assert len(retried) == 1
    assert retried[0]['ray_id'] == 7
    assert retried[0]['ray_min_standoff_m'] == pytest.approx(0.50)
    assert retried[0]['desired_look_at_direction'] == {
        'x': -1.0, 'y': -0.0, 'z': -0.0}


def test_ray_retirement_uses_accepted_views_not_path_rejections():
    accepted = {'ray_id': 3}
    rejected = {'ray_id': 7, 'framing_retry_ray_id': 7}
    history = {
        'entries': [accepted, rejected],
        'accepted_entries': [accepted],
    }

    assert retired_view_history(history, True) == [accepted]
    assert retired_view_history(history, False) == [accepted, rejected]


def test_tracker_rate_duplicates_do_not_regenerate_candidates():
    center = {'x': 0.4, 'y': 0.0, 'z': 0.05}
    assert not viewpoint_replan_required(
        center, {'x': 0.404, 'y': 0.0, 'z': 0.05},
        'history-1', 'history-1', 0.01, 5.0, 0.5)
    assert not viewpoint_replan_required(
        center, {'x': 0.42, 'y': 0.0, 'z': 0.05},
        'history-1', 'history-1', 0.01, 0.1, 0.5)
    assert viewpoint_replan_required(
        center, {'x': 0.42, 'y': 0.0, 'z': 0.05},
        'history-1', 'history-1', 0.01, 0.5, 0.5)
    assert viewpoint_replan_required(
        center, center, 'history-1', 'history-2', 0.01, 0.0, 0.5)


def test_stable_candidates_refresh_before_bridge_freshness_expires():
    assert not viewpoint_refresh_required(0.49, 0.50)
    assert viewpoint_refresh_required(0.50, 0.50)
    assert not viewpoint_refresh_required(10.0, 0.0)


def test_raw_camera_frame_target_cannot_replace_base_link_nbv_center():
    assert target_frame_rejection_reason('base_link') == ''
    assert target_frame_rejection_reason(
        'camera_color_optical_frame') == (
            'target frame camera_color_optical_frame is not scan planning '
            'frame base_link')
    assert target_frame_rejection_reason('') == (
        'target frame <empty> is not scan planning frame base_link')
