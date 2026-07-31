import math

import pytest

from piper_mobile_manipulation.motion_limit_stability import MotionLimitStability
from piper_mobile_manipulation.scan_session_memory import (
    angular_separation_deg,
    filter_and_order_viewpoints,
    validate_history_payload,
    viewpoint_is_duplicate,
)


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


def test_retry_filters_accepted_views_and_preserves_candidate_geometry_order():
    candidates = [view(index, index * 5.0) for index in range(13)]
    accepted = [candidates[0], candidates[4], candidates[8]]
    result = filter_and_order_viewpoints(candidates, accepted, 0.001, 0.1)
    assert len(result) == 10
    assert not {item['index'] for item in result}.intersection({0, 4, 8})
    # The bridge owns diverse subset selection and smooth routing because it
    # also knows the current calibrated camera position.
    assert [item['index'] for item in result] == [
        1, 2, 3, 5, 6, 7, 9, 10, 11, 12]


def test_history_payload_is_session_bounded_and_correlated():
    entry = view(0, 0.0)
    payload = validate_history_payload({
        'session_id': 'session-a',
        'accepted_views': 1,
        'max_views': 13,
        'entries': [entry],
    }, 13)
    assert payload['accepted_views'] == 1
    with pytest.raises(ValueError, match='count'):
        validate_history_payload({
            'session_id': 'session-a',
            'accepted_views': 2,
            'max_views': 13,
            'entries': [entry],
        }, 13)


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
