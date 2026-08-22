"""Pure regression tests for mission-frozen target-centred viewpoint rays."""

import numpy as np
import pytest

from piper_mobile_manipulation.viewpoint_rays import (
    bounded_ray_interval,
    decoded_ray_id,
    expand_shortlisted_rays,
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


def test_only_shortlisted_rays_expand_to_exact_target_facing_probes():
    target = np.asarray([0.4, 0.0, 0.1])
    expanded = expand_shortlisted_rays(
        [ray(3, [1.0, 0.0, 0.0]), ray(9, [0.0, 1.0, 0.0])],
        start_camera_position=[0.85, 0.0, 0.1],
        target_center=target,
    )

    assert {item['ray_id'] for item in expanded} == {3, 9}
    ray_order = [item['ray_id'] for item in expanded]
    assert [
        ray_id for index, ray_id in enumerate(ray_order)
        if index == 0 or ray_order[index - 1] != ray_id
    ] == [3, 9]
    for ray_id in (3, 9):
        phases = [
            item['ray_probe_phase'] for item in expanded
            if item['ray_id'] == ray_id]
        if 'reserve' in phases:
            first_reserve = phases.index('reserve')
            assert phases[:first_reserve] == ['preferred'] * first_reserve
            assert phases[first_reserve:] == ['reserve'] * (
                len(phases) - first_reserve)
    assert len(expanded) <= 10
    assert len({item['id'] for item in expanded}) == len(expanded)
    for item in expanded:
        position = np.asarray(item['camera_position_m'])
        direction = np.asarray(item['ray_direction'])
        standoff = float(item['ray_standoff_m'])
        assert position == pytest.approx(target + direction * standoff)
        assert item['look_direction'] == pytest.approx(-direction)
        assert decoded_ray_id(item['id']) == item['ray_id']


def test_close_target_degenerate_ray_emits_one_exact_probe():
    expanded = expand_shortlisted_rays(
        [ray(4, [1.0, 0.0, 0.0], maximum=0.28, preferred=0.28)],
        start_camera_position=[0.0, 0.0, 0.0],
        target_center=[0.20, 0.0, 0.0],
    )

    assert len(expanded) == 1
    assert expanded[0]['ray_standoff_m'] == pytest.approx(0.28)
    assert expanded[0]['ray_probe_phase'] == 'preferred'


def test_ordinary_legacy_viewpoint_id_is_not_a_ray_probe():
    assert decoded_ray_id(42) is None
