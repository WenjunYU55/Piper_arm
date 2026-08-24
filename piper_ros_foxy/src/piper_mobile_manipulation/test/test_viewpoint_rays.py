"""Pure regression tests for mission-frozen target-centred viewpoint rays."""

import numpy as np
import pytest

from piper_mobile_manipulation.viewpoint_rays import (
    bind_shortlisted_ray_intervals,
    bounded_ray_interval,
    decoded_ray_id,
)


def ray(ray_id, direction, maximum=0.60, preferred=0.50):
    return {
        'id': ray_id,
        'candidate_geometry': 'target_ray',
        'ray_id': ray_id,
        'ray_direction': direction,
        'ray_min_standoff_m': 0.28,
        'ray_max_standoff_m': maximum,
        'ray_preferred_max_standoff_m': preferred,
    }


@pytest.mark.parametrize(
    'target, expected', (
        ([0.20, 0.0, 0.0], (0.28, 0.28)),
        ([0.40, 0.0, 0.0], (0.28, 0.40)),
        ([0.90, 0.0, 0.0], (0.28, 0.80)),
    ),
)
def test_ray_interval_uses_target_range_with_existing_scan_cap(
        target, expected):
    assert bounded_ray_interval(target, 0.28, 0.80) == pytest.approx(
        expected)


def test_only_shortlisted_rays_bind_to_one_complete_interval_each():
    target = np.asarray([0.4, 0.0, 0.1])
    bound = bind_shortlisted_ray_intervals(
        [ray(3, [1.0, 0.0, 0.0]), ray(9, [0.0, 1.0, 0.0])],
        start_camera_position=[0.85, 0.0, 0.1],
        target_center=target,
    )

    assert len(bound) == 2
    assert [item['ray_id'] for item in bound] == [3, 9]
    assert len({item['id'] for item in bound}) == len(bound)
    for item in bound:
        position = np.asarray(item['camera_position_m'])
        direction = np.asarray(item['ray_direction'])
        standoff = float(item['ray_standoff_m'])
        assert position == pytest.approx(target + direction * standoff)
        assert item['look_direction'] == pytest.approx(-direction)
        assert item['ray_probe_phase'] == 'interval_search'
        assert item['ray_min_standoff_m'] <= standoff
        assert standoff <= item['ray_max_standoff_m']
        assert decoded_ray_id(item['id']) == item['ray_id']


def test_close_target_degenerate_ray_binds_one_zero_length_interval():
    bound = bind_shortlisted_ray_intervals(
        [ray(4, [1.0, 0.0, 0.0], maximum=0.28, preferred=0.28)],
        start_camera_position=[0.0, 0.0, 0.0],
        target_center=[0.20, 0.0, 0.0],
    )

    assert len(bound) == 1
    assert bound[0]['ray_standoff_m'] == pytest.approx(0.28)
    assert bound[0]['ray_probe_phase'] == 'interval_search'


def test_ordinary_legacy_viewpoint_id_is_not_a_ray_probe():
    assert decoded_ray_id(42) is None
