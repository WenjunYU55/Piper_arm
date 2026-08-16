import numpy as np

from piper_mobile_manipulation.depth_to_3d_node import (
    depth_jump_reacquisition,
    primary_depth_component,
)
from piper_mobile_manipulation.utils.target_depth import (
    select_target_depth_component,
)


def test_depth_jump_accepts_change_and_disables_with_nonpositive_limit():
    assert depth_jump_reacquisition(
        0.40, 0.45, 0.20, None, 0, 3, 0.03
    ) == (True, None, 0, False)
    assert depth_jump_reacquisition(
        0.40, 0.80, 0.0, None, 0, 3, 0.03
    ) == (True, None, 0, False)


def test_depth_jump_requires_consistent_reacquisition_samples():
    accepted, pending, count, resynced = depth_jump_reacquisition(
        0.30, 0.62, 0.20, None, 0, 3, 0.03
    )
    assert (accepted, count, resynced) == (False, 1, False)

    accepted, pending, count, resynced = depth_jump_reacquisition(
        0.30, 0.61, 0.20, pending, count, 3, 0.03
    )
    assert (accepted, count, resynced) == (False, 2, False)

    accepted, pending, count, resynced = depth_jump_reacquisition(
        0.30, 0.615, 0.20, pending, count, 3, 0.03
    )
    assert (accepted, pending, count, resynced) == (True, None, 0, True)


def test_depth_jump_restarts_consistency_count_for_an_unrelated_outlier():
    accepted, pending, count, _ = depth_jump_reacquisition(
        0.30, 0.62, 0.20, None, 0, 3, 0.03
    )
    assert not accepted
    accepted, pending, count, resynced = depth_jump_reacquisition(
        0.30, 0.90, 0.20, pending, count, 3, 0.03
    )
    assert (accepted, pending, count, resynced) == (
        False, 0.90, 1, False)


def test_primary_depth_component_rejects_adjacent_background_mode():
    depth = np.full((7, 9), 0.42, dtype=float)
    depth[1:6, 2:7] = np.asarray([
        [0.330, 0.334, 0.338, 0.342, 0.346],
        [0.332, 0.336, 0.340, 0.344, 0.348],
        [0.334, 0.338, 0.342, 0.346, 0.350],
        [0.336, 0.340, 0.344, 0.348, 0.352],
        [0.338, 0.342, 0.346, 0.350, 0.354],
    ])
    valid = np.ones_like(depth, dtype=bool)

    selected, seed_depth = primary_depth_component(
        valid, depth, 4.0, 3.0, 0.03, 0.015)

    assert seed_depth == 0.342
    assert int(np.count_nonzero(selected)) == 25
    assert np.all(depth[selected] < 0.40)


def test_primary_depth_component_keeps_only_nearest_semantic_surface():
    depth = np.full((5, 9), np.nan, dtype=float)
    depth[1:4, 1:4] = 0.34
    depth[1:4, 6:9] = 0.34
    valid = np.isfinite(depth)

    selected, _ = primary_depth_component(
        valid, depth, 2.0, 2.0, 0.03, 0.015)

    assert int(np.count_nonzero(selected)) == 9
    assert np.all(selected[1:4, 1:4])
    assert not np.any(selected[:, 6:9])


def test_primary_depth_component_can_be_disabled():
    depth = np.asarray([[0.3, 0.7]], dtype=float)
    valid = np.ones_like(depth, dtype=bool)

    selected, seed_depth = primary_depth_component(
        valid, depth, 0.0, 0.0, 0.0, 0.0)

    assert seed_depth == 0.3
    assert np.array_equal(selected, valid)


def test_depth_modes_do_not_leak_across_a_gradual_background_bridge():
    depth = np.full((30, 40), 0.42, dtype=float)
    depth[5:25, 5:24] = 0.34
    for column, value in enumerate(np.linspace(0.35, 0.41, 8), start=23):
        depth[12:18, column] = value
    support = np.ones_like(depth, dtype=bool)

    selected, report = select_target_depth_component(
        support, depth, center_u=15.0, center_v=15.0,
        minimum_points=20, minimum_support_fraction=0.15)

    assert report['selected_depth_m'] < 0.36
    assert np.all(depth[selected] < 0.38)


def test_ambiguous_depth_layers_fail_closed():
    depth = np.full((20, 20), np.nan, dtype=float)
    depth[:, :10] = 0.34
    depth[:, 10:] = 0.42
    support = np.isfinite(depth)

    with np.testing.assert_raises_regex(ValueError, 'ambiguous'):
        select_target_depth_component(
            support, depth, center_u=9.5, center_v=9.5,
            minimum_points=20, ambiguity_margin=0.25)
