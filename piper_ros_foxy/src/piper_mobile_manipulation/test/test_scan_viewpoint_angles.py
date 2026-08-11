from piper_mobile_manipulation.scan_viewpoint_planner_node import (
    build_viewpoint_angles,
    target_frame_rejection_reason,
    viewpoint_replan_required,
    viewpoint_refresh_required,
)


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


def test_live_flexible_region_spans_both_y_sides_distances_and_elevations():
    angles = build_viewpoint_angles(180, 180, 7.5, 25)
    pitches = [
        -50.0 + offset
        for offset in (35.0, 25.0, 15.0, 5.0, -5.0, -15.0, -25.0)
    ]
    radii = [0.30, 0.33, 0.36, 0.39, 0.42, 0.45]
    candidates = [
        (angle, pitch, radius)
        for radius in radii for pitch in pitches for angle in angles]

    assert len(angles) == 25
    assert len(candidates) == 1050
    assert angles[0] == 90.0
    assert angles[-1] == 270.0
    # The failed physical pose landed at 195.6 degrees.  The denser region has
    # a meaningful but compact successor instead of jumping directly to 210.
    assert 6.0 <= 202.5 - 195.6 <= 8.0
    assert sorted(set(pitch for _angle, pitch, _radius in candidates)) == [
        -75.0, -65.0, -55.0, -45.0, -35.0, -25.0, -15.0]


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
