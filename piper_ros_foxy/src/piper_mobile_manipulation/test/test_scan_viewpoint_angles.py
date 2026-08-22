from types import SimpleNamespace

from piper_mobile_manipulation.scan_viewpoint_planner_node import (
    build_viewpoint_angles,
    ScanViewpointPlannerNode,
    target_frame_rejection_reason,
    viewpoint_replan_required,
    viewpoint_refresh_required,
)
from piper_mobile_manipulation.viewpoint_rays import build_ray_samples


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
