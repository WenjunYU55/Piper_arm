from piper_mobile_manipulation.scan_viewpoint_planner_node import (
    build_viewpoint_angles,
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


def test_live_diverse_dome_uses_seven_azimuths_for_three_elevations():
    angles = build_viewpoint_angles(147.5, 55, 55 / 6, 21)
    pitches = [
        -55.0 + offset for offset in (0.0, 10.0, -10.0)
    ]
    candidates = [(angle, pitch) for pitch in pitches for angle in angles]

    assert len(angles) == 7
    assert len(candidates) == 21
    assert angles[0] == 120.0
    assert angles[-1] == 175.0
    assert sorted(set(pitch for _angle, pitch in candidates)) == [
        -65.0, -55.0, -45.0]
