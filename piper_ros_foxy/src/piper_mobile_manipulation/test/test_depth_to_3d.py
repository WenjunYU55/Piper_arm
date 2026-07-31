from piper_mobile_manipulation.depth_to_3d_node import (
    depth_jump_reacquisition,
)


def test_depth_jump_accepts_normal_change_and_disables_with_nonpositive_limit():
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
