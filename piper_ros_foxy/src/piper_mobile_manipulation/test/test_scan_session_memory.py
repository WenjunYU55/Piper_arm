import math

import pytest

from piper_mobile_manipulation.motion_limit_stability import MotionLimitStability
from piper_mobile_manipulation.scan_session_memory import (
    achieved_feature_coverage,
    feature_coverage_progress,
    angular_separation_deg,
    feature_coverage_priority,
    filter_and_order_viewpoints,
    history_coverage_target_center,
    validate_history_payload,
    viewpoint_is_duplicate,
)


def achieved_entry(x, y, z):
    return {
        'actual_camera_position': {'x': x, 'y': y, 'z': z},
        'desired_camera_position': {'x': x, 'y': y, 'z': z},
        'desired_look_at_direction': {'x': 1.0, 'y': 0.0, 'z': 0.0},
    }


def spherical_entry(azimuth_deg, elevation_deg, radius=0.30):
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    horizontal = radius * math.cos(elevation)
    return achieved_entry(
        horizontal * math.cos(azimuth),
        horizontal * math.sin(azimuth),
        radius * math.sin(elevation),
    )


def test_feature_coverage_requires_both_y_sides_and_elevation_diversity():
    target = {'x': 0.4, 'y': 0.0, 'z': 0.03}
    diverse = [
        achieved_entry(0.40, 0.30, 0.18),
        achieved_entry(0.30, 0.28, 0.32),
        achieved_entry(0.10, 0.18, 0.20),
        achieved_entry(0.10, 0.00, 0.40),
        achieved_entry(0.10, -0.18, 0.20),
        achieved_entry(0.30, -0.28, 0.32),
        achieved_entry(0.40, -0.30, 0.18),
        achieved_entry(0.16, 0.12, 0.16),
        achieved_entry(0.16, -0.12, 0.38),
    ]

    coverage = achieved_feature_coverage(diverse, target)

    assert coverage['sufficient']
    assert coverage['positive_y_side_views'] >= 2
    assert coverage['negative_y_side_views'] >= 2
    assert coverage['azimuth_span_deg'] >= 120.0
    assert coverage['elevation_span_deg'] >= 25.0


def test_front_and_top_cluster_is_not_distinctive_feature_coverage():
    target = {'x': 0.36, 'y': -0.07, 'z': 0.03}
    clustered = [
        achieved_entry(0.11, -0.025, 0.271),
        achieved_entry(0.238, -0.022, 0.281),
        achieved_entry(0.204, 0.066, 0.246),
        achieved_entry(0.181, 0.015, 0.329),
        achieved_entry(0.190, -0.054, 0.275),
        achieved_entry(0.168, 0.018, 0.241),
        achieved_entry(0.236, 0.046, 0.271),
    ]

    coverage = achieved_feature_coverage(clustered, target)

    assert not coverage['sufficient']
    assert coverage['negative_y_side_views'] == 0
    assert any('-Y side' in blocker for blocker in coverage['blockers'])


def view(index, angle_deg, radius=0.30):
    angle = math.radians(angle_deg)
    camera = {
        'x': radius * math.cos(angle),
        'y': radius * math.sin(angle),
        'z': 0.20,
    }
    return {
        'index': index,
        'desired_camera_position': camera,
        'desired_look_at_direction': {
            'x': -math.cos(angle), 'y': -math.sin(angle), 'z': 0.0,
        },
    }


def test_duplicate_requires_close_position_and_direction():
    accepted = [view(0, 0.0)]
    assert viewpoint_is_duplicate(view(1, 1.0), accepted, 0.012, 2.0)
    assert not viewpoint_is_duplicate(view(2, 5.0), accepted, 0.012, 2.0)
    shifted = view(3, 1.0)
    shifted['desired_camera_position']['z'] += 0.02
    assert not viewpoint_is_duplicate(shifted, accepted, 0.012, 2.0)


def test_duplicate_and_diversity_use_achieved_pose_over_stale_proposal():
    candidate = view(1, 1.0)
    history = view(0, 1.0)
    history['actual_camera_position'] = view(2, 30.0)[
        'desired_camera_position']
    history['actual_look_at_direction'] = view(2, 30.0)[
        'desired_look_at_direction']

    assert not viewpoint_is_duplicate(candidate, [history], 0.012, 2.0)


def test_feature_priority_finishes_negative_then_positive_y_faces():
    target = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    negative = view(1, 240.0)
    positive = view(2, 120.0)

    negative_score, objective = feature_coverage_priority(
        negative, [], target)
    positive_score, _ = feature_coverage_priority(positive, [], target)
    assert objective == 'negative_y_face'
    assert negative_score > positive_score

    two_negative = [
        achieved_entry(-0.15, -0.26, 0.10),
        achieved_entry(-0.20, -0.24, 0.12),
    ]
    negative_score, objective = feature_coverage_priority(
        negative, two_negative, target)
    positive_score, _ = feature_coverage_priority(
        positive, two_negative, target)
    assert objective == 'positive_y_face'
    assert positive_score > negative_score


def test_feature_priority_finishes_the_approached_positive_side_first():
    target = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    positive_start = [spherical_entry(165.0, 50.0)]
    positive = view(1, 120.0)
    negative = view(2, 240.0)

    positive_score, objective = feature_coverage_priority(
        positive, positive_start, target)
    negative_score, _ = feature_coverage_priority(
        negative, positive_start, target)

    assert objective == 'positive_y_face'
    assert positive_score > negative_score


def test_side_switch_progress_uses_latest_pose_not_historical_extreme():
    target = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    history = [
        spherical_entry(120.0, 50.0),
        spherical_entry(135.0, 50.0),
        spherical_entry(150.0, 50.0),
    ]
    toward_negative = view(1, 165.0)
    back_toward_positive = view(2, 135.0)

    assert feature_coverage_progress(
        toward_negative, history, target, 'negative_y_face') >= 0.0
    assert feature_coverage_progress(
        back_toward_positive, history, target, 'negative_y_face') < 0.0


def test_feature_priority_does_not_reverse_before_missing_side_is_reached():
    target = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    front_history = [
        achieved_entry(-0.28, 0.05, 0.24),
        achieved_entry(-0.28, -0.01, 0.24),
        achieved_entry(-0.27, -0.08, 0.23),
    ]
    continue_negative = view(1, 225.0)
    reverse_positive = view(2, 150.0)

    continue_score, objective = feature_coverage_priority(
        continue_negative, front_history, target)
    reverse_score, _ = feature_coverage_priority(
        reverse_positive, front_history, target)

    assert objective == 'negative_y_face'
    assert continue_score > reverse_score


def test_feature_progress_allows_elevation_change_but_rejects_y_reversal():
    target = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    achieved = [spherical_entry(199.0, 51.0)]
    lower_same_side = spherical_entry(195.0, 30.0)
    lower_same_side['index'] = 1
    reverse_to_positive_y = spherical_entry(165.0, 50.0)
    reverse_to_positive_y['index'] = 2

    assert feature_coverage_progress(
        lower_same_side, achieved, target, 'negative_y_face') >= 0.0
    assert feature_coverage_progress(
        reverse_to_positive_y, achieved, target, 'negative_y_face') < 0.0


def test_ordered_candidates_publish_objective_progress_margin():
    target = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    achieved = [spherical_entry(199.0, 51.0)]
    candidates = [
        dict(spherical_entry(195.0, 30.0), index=1),
        dict(spherical_entry(165.0, 50.0), index=2),
    ]

    result = filter_and_order_viewpoints(
        candidates, achieved, accepted_entries=achieved,
        target_center=target)

    margins = {
        item['index']: item['coverage_progress_score'] for item in result}
    assert margins[1] >= 0.0
    assert margins[2] < 0.0


def test_feature_priority_extends_azimuth_then_elevation_span():
    target = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    narrow_azimuth = [
        spherical_entry(220.0, 40.0),
        spherical_entry(230.0, 40.0),
        spherical_entry(130.0, 40.0),
        spherical_entry(140.0, 40.0),
    ]
    wider_azimuth = view(1, 100.0)
    taller_only = spherical_entry(180.0, 70.0)
    taller_only['index'] = 2

    wide_score, objective = feature_coverage_priority(
        wider_azimuth, narrow_azimuth, target)
    tall_score, _ = feature_coverage_priority(
        taller_only, narrow_azimuth, target)
    assert objective == 'azimuth_span'
    assert wide_score > tall_score

    broad_but_flat = [
        spherical_entry(240.0, 40.0),
        spherical_entry(250.0, 40.0),
        spherical_entry(110.0, 40.0),
        spherical_entry(120.0, 40.0),
    ]
    high_view = spherical_entry(180.0, 70.0)
    high_view['index'] = 3
    flat_view = spherical_entry(180.0, 45.0)
    flat_view['index'] = 4
    high_score, objective = feature_coverage_priority(
        high_view, broad_but_flat, target)
    flat_score, _ = feature_coverage_priority(
        flat_view, broad_but_flat, target)
    assert objective == 'elevation_span'
    assert high_score > flat_score


def test_retry_filters_accepted_views_and_prioritizes_new_coverage():
    candidates = [view(index, index * 5.0) for index in range(13)]
    accepted = [candidates[0], candidates[4], candidates[8]]
    result = filter_and_order_viewpoints(candidates, accepted, 0.001, 0.1)
    assert len(result) == 10
    assert not {item['index'] for item in result}.intersection({0, 4, 8})
    scores = [item['expected_new_coverage_score'] for item in result]
    assert scores == sorted(scores, reverse=True)
    assert result[0]['index'] == 12


def test_history_payload_is_session_bounded_and_correlated():
    entry = view(0, 0.0)
    payload = validate_history_payload({
        'session_id': 'session-a',
        'accepted_views': 1,
        'max_views': 13,
        'entries': [entry],
        'coverage_target_center': {'x': 0.4, 'y': -0.08, 'z': 0.03},
    }, 13)
    assert payload['accepted_views'] == 1
    assert payload['coverage_target_center'] == {
        'x': 0.4, 'y': -0.08, 'z': 0.03}
    with pytest.raises(ValueError, match='count'):
        validate_history_payload({
            'session_id': 'session-a',
            'accepted_views': 2,
            'max_views': 13,
            'entries': [entry],
        }, 13)


def test_frozen_coverage_center_prevents_tracker_shift_from_reopening_y_face():
    frozen = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    shifted_live = {'x': 0.0, 'y': 0.08, 'z': 0.0}
    history = {
        'coverage_target_center': frozen,
        'accepted_entries': [
            achieved_entry(-0.20, 0.15, 0.10),
            achieved_entry(-0.20, 0.14, 0.10),
        ],
    }

    coverage_center = history_coverage_target_center(history, shifted_live)
    negative_score, objective = feature_coverage_priority(
        view(1, 240.0), history['accepted_entries'], coverage_center)
    positive_score, _ = feature_coverage_priority(
        view(2, 120.0), history['accepted_entries'], coverage_center)

    assert coverage_center == frozen
    assert objective == 'negative_y_face'
    assert negative_score > positive_score


def test_rejected_view_is_filtered_without_incrementing_accepted_count():
    accepted = view(0, 0.0)
    rejected = view(1, 5.0)
    payload = validate_history_payload({
        'session_id': 'session-a',
        'accepted_views': 1,
        'max_views': 13,
        'entries': [accepted],
        'rejected_entries': [rejected],
    }, 13)
    assert payload['accepted_views'] == 1
    assert len(payload['entries']) == 2
    remaining = filter_and_order_viewpoints(
        [accepted, rejected, view(2, 10.0)], payload['entries'],
        0.001, 0.1)
    assert [item['index'] for item in remaining] == [2]


def test_motion_limit_change_requires_persistence_and_cancels_on_recovery():
    class Limits:
        valid = True

        def __init__(self, digest):
            self.limits_sha256 = digest

    stable = Limits('a' * 64)
    transient = Limits('b' * 64)
    tracker = MotionLimitStability(7.0, 3)
    assert tracker.observe(stable, 0.0) == (stable, True)
    assert tracker.observe(transient, 1.0) == (stable, False)
    assert tracker.observe(transient, 2.0) == (stable, False)
    assert tracker.observe(stable, 5.0) == (stable, True)
    assert tracker.observe(transient, 10.0) == (stable, False)
    assert tracker.observe(transient, 14.0) == (stable, False)
    accepted, refreshed = tracker.observe(transient, 17.1)
    assert accepted is transient
    assert refreshed is True


def test_direction_angle_is_scale_independent():
    assert angular_separation_deg([1, 0, 0], [2, 0, 0]) == pytest.approx(0.0)
    assert angular_separation_deg([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
