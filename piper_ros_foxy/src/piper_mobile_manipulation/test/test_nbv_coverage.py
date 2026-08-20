"""Command-free tests for live voxel next-best-view scoring."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from piper_mobile_manipulation import nbv_coverage
from piper_mobile_manipulation.nbv_coverage import (
    candidate_information,
    ObjectCoverageModel,
    rank_next_best_views,
    VoxelCoverageConfig,
)
from piper_mobile_manipulation.scan_viewpoint_planner_node import (
    ScanViewpointPlannerNode,
)


def synthetic_observation(camera_transform=None):
    height = 32
    width = 32
    depth = np.full((height, width), 1.0, dtype=np.float64)
    target = np.zeros((height, width), dtype=np.float64)
    support = np.zeros((height, width), dtype=bool)
    support[10:22, 10:22] = True
    target[support] = 0.4
    depth[support] = 0.4
    intrinsic = np.asarray([
        80.0, 0.0, 15.5,
        0.0, 80.0, 15.5,
        0.0, 0.0, 1.0,
    ])
    return (
        target,
        support,
        depth,
        intrinsic,
        np.eye(4) if camera_transform is None else camera_transform,
    )


def viewpoint(index, position, target=(0.0, 0.0, 0.4)):
    camera = np.asarray(position, dtype=float)
    look = np.asarray(target, dtype=float) - camera
    look /= np.linalg.norm(look)
    return {
        'index': index,
        'desired_camera_position': dict(zip(('x', 'y', 'z'), position)),
        'desired_look_at_direction': dict(zip(
            ('x', 'y', 'z'), (float(value) for value in look))),
    }


def initialized_model():
    model = ObjectCoverageModel(VoxelCoverageConfig(
        voxel_size_m=0.005,
        minimum_radius_m=0.03,
        maximum_radius_m=0.10,
        render_width=32,
        render_height=24,
    ))
    model.session_id = 'test-session'
    model.integrate(
        *synthetic_observation(), target_center=[0.0, 0.0, 0.4])
    return model


def test_repeated_front_view_loses_to_unobserved_back_view():
    model = initialized_model()
    snapshot = model.snapshot()
    front = viewpoint(0, [0.0, 0.0, 0.0])
    back = viewpoint(1, [0.0, 0.0, 0.8])

    ranked = rank_next_best_views(snapshot, [front, back], [0.0, 0.0, 0.0])

    assert ranked[0]['index'] == 1
    assert ranked[0]['nbv_predicted_unknown_pixels'] > \
        ranked[1]['nbv_predicted_unknown_pixels']


def test_information_precedes_motion_cost():
    snapshot = initialized_model().snapshot()
    near_repeat = viewpoint(0, [0.0, 0.0, 0.01])
    farther_unseen = viewpoint(1, [0.0, 0.0, 0.8])

    ranked = rank_next_best_views(
        snapshot, [near_repeat, farther_unseen], [0.0, 0.0, 0.0])

    assert ranked[0]['index'] == 1
    assert ranked[0]['nbv_camera_travel_m'] > \
        ranked[1]['nbv_camera_travel_m']


def test_marginal_fraction_prevents_raw_unknown_pixel_bias(monkeypatch):
    snapshot = initialized_model().snapshot()
    steep = viewpoint(10, [0.3, 0.0, 0.4])
    diverse = viewpoint(20, [0.0, 0.3, 0.4])

    def information(_snapshot, camera_position, _current, _look):
        if float(camera_position[0]) > 0.0:
            unknown, novel, projected, novelty = 900, 0, 1000, 10.0
        else:
            unknown, novel, projected, novelty = 400, 50, 450, 60.0
        marginal = unknown + novel
        return {
            'predicted_unknown_pixels': unknown,
            'novel_surface_pixels': novel,
            'marginal_information_pixels': marginal,
            'marginal_information_fraction': marginal / projected,
            'projected_object_pixels': projected,
            'direction_novelty_deg': novelty,
            'camera_travel_m': 0.0,
            'positive_information_gain': True,
        }

    monkeypatch.setattr(nbv_coverage, 'candidate_information', information)
    ranked = rank_next_best_views(
        snapshot, [steep, diverse], [0.0, 0.0, 0.4])

    assert ranked[0]['index'] == 20
    assert ranked[0]['nbv_predicted_unknown_pixels'] < \
        ranked[1]['nbv_predicted_unknown_pixels']
    assert ranked[0]['nbv_marginal_information_fraction'] > \
        ranked[1]['nbv_marginal_information_fraction']


def test_direction_novelty_breaks_equal_marginal_fraction(monkeypatch):
    snapshot = initialized_model().snapshot()
    adjacent = viewpoint(10, [0.3, 0.0, 0.4])
    diverse = viewpoint(20, [0.0, 0.3, 0.4])

    def information(_snapshot, camera_position, _current, _look):
        adjacent_view = float(camera_position[0]) > 0.0
        unknown = 900 if adjacent_view else 90
        projected = 1000 if adjacent_view else 100
        return {
            'predicted_unknown_pixels': unknown,
            'novel_surface_pixels': 0,
            'marginal_information_pixels': unknown,
            'marginal_information_fraction': 0.9,
            'projected_object_pixels': projected,
            'direction_novelty_deg': 10.0 if adjacent_view else 60.0,
            'camera_travel_m': 0.0,
            'positive_information_gain': True,
        }

    monkeypatch.setattr(nbv_coverage, 'candidate_information', information)
    ranked = rank_next_best_views(
        snapshot, [adjacent, diverse], [0.0, 0.0, 0.4])

    assert ranked[0]['index'] == 20
    assert ranked[0]['nbv_direction_novelty_deg'] == 60.0


def test_candidate_information_reports_normalized_marginal_gain():
    metrics = candidate_information(
        initialized_model().snapshot(),
        [0.0, 0.0, 0.8],
        [0.0, 0.0, 0.0],
    )

    assert metrics['marginal_information_pixels'] == (
        metrics['predicted_unknown_pixels']
        + metrics['novel_surface_pixels'])
    assert metrics['marginal_information_fraction'] == pytest.approx(
        metrics['marginal_information_pixels']
        / metrics['projected_object_pixels'])
    assert 0.0 <= metrics['marginal_information_fraction'] <= 1.0


def test_global_nbv_can_select_seventy_degrees_over_low_gain_twenty_degrees():
    snapshot = initialized_model().snapshot()
    center = np.asarray(snapshot.target_center, dtype=float)
    radius = 0.40

    def rotated_candidate(index, angle_deg):
        angle = np.deg2rad(float(angle_deg))
        position = center + radius * np.asarray([
            np.sin(angle), 0.0, -np.cos(angle)])
        return viewpoint(index, position.tolist(), center)

    low_gain_nearby = rotated_candidate(20, 20.0)
    high_gain_global = rotated_candidate(70, 70.0)

    ranked = rank_next_best_views(
        snapshot, [low_gain_nearby, high_gain_global], [0.0, 0.0, 0.0])

    assert ranked[0]['index'] == 70
    assert ranked[0]['nbv_predicted_unknown_pixels'] > \
        ranked[1]['nbv_predicted_unknown_pixels']
    assert ranked[0]['nbv_camera_travel_m'] > \
        ranked[1]['nbv_camera_travel_m']


def test_rejected_observation_cannot_change_accepted_coverage_generation():
    model = initialized_model()
    before = model.snapshot()

    # A rejected observation is deliberately not passed to integrate().
    after = model.snapshot()

    assert after.generation == before.generation
    assert np.array_equal(after.states, before.states)
    assert after.view_directions == before.view_directions


def test_next_motion_uses_latest_achieved_fk_even_when_capture_was_rejected():
    history = {
        'accepted_entries': [{
            'actual_camera_position': {'x': 0.1, 'y': 0.0, 'z': 0.3},
        }],
        'latest_achieved_camera': {
            'camera_position': {'x': 0.2, 'y': -0.1, 'z': 0.4},
        },
    }

    current = ScanViewpointPlannerNode.current_achieved_camera(history)

    assert current == [0.2, -0.1, 0.4]


def test_radius_fallbacks_do_not_displace_distinct_view_directions():
    snapshot = initialized_model().snapshot()
    viewpoints = [
        viewpoint(0, [0.0, 0.0, 0.8]),
        viewpoint(1, [0.0, 0.0, 0.9]),
        viewpoint(2, [0.1, 0.0, 0.8]),
        viewpoint(3, [0.2, 0.0, 0.8]),
    ]

    ranked = rank_next_best_views(snapshot, viewpoints, [0.0, 0.0, 0.0])

    # The same +Z ray's second radius appears only after each distinct ray has
    # received one opportunity.
    assert ranked.index(next(
        item for item in ranked if item['index'] == 1)) >= 3


def test_opposite_capture_reduces_back_view_information():
    model = initialized_model()
    before = candidate_information(
        model.snapshot(), [0.0, 0.0, 0.8], [0.0, 0.0, 0.0])
    transform = np.eye(4)
    transform[:3, :3] = np.diag([1.0, -1.0, -1.0])
    transform[:3, 3] = [0.0, 0.0, 0.8]
    model.integrate(
        *synthetic_observation(transform), target_center=[0.0, 0.0, 0.4])
    after = candidate_information(
        model.snapshot(), [0.0, 0.0, 0.8], [0.0, 0.0, 0.0])

    assert after['predicted_unknown_pixels'] < \
        before['predicted_unknown_pixels']


def test_snapshot_is_immutable_and_consistent():
    snapshot = initialized_model().snapshot()

    assert snapshot.generation == 1
    assert snapshot.surface_voxels > 0
    assert snapshot.unknown_voxels > 0
    assert not snapshot.states.flags.writeable
    with pytest.raises(ValueError):
        snapshot.states[0] = 9
    with pytest.raises(FrozenInstanceError):
        snapshot.generation = 2


def test_target_center_change_is_rejected():
    model = initialized_model()

    with pytest.raises(ValueError, match='target center changed'):
        model.integrate(
            *synthetic_observation(), target_center=[0.02, 0.0, 0.4])


@pytest.mark.parametrize('value', [0.0, -0.1, float('nan')])
def test_invalid_voxel_size_is_rejected(value):
    with pytest.raises(ValueError, match='finite and positive'):
        ObjectCoverageModel(VoxelCoverageConfig(voxel_size_m=value))


def test_scoring_budget_has_a_safe_lower_bound():
    with pytest.raises(ValueError, match='at least 1000'):
        ObjectCoverageModel(VoxelCoverageConfig(maximum_scoring_voxels=999))


def test_authoritative_voxel_policy_labels_first_view_as_seed():
    planner = type('Planner', (), {
        'selection_policy': lambda _self: 'voxel_nbv',
    })()

    selected = ScanViewpointPlannerNode.apply_view_selection(
        planner,
        [viewpoint(4, [0.3, 0.0, 0.4])],
        {'session_id': 'scan-a', 'accepted_views': 0},
    )

    assert selected[0]['view_selection_policy'] == 'voxel_nbv_seed'
    assert selected[0]['view_selection_requested_policy'] == 'voxel_nbv'
    assert selected[0]['view_selection_generation'] == 0
    assert selected[0]['view_selection_session_id'] == 'scan-a'
