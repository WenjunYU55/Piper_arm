"""No-hardware tests for the closed-loop view-generation handoff."""

from dataclasses import FrozenInstanceError

import pytest

from piper_mobile_manipulation.view_generation import (
    generation_matches_expected,
    make_view_generation,
    parse_view_generation,
    view_policy_capabilities,
)


def _receipt(session='scan-a', accepted=2, policy='voxel_nbv', ready=True):
    return {
        'bridge_received_at_ns': 123,
        'view_generation': make_view_generation(
            session, accepted, policy, accepted, ready, 18,
            '' if ready else 'coverage model is rebuilding').to_dict(),
    }


def test_matching_bridge_generation_is_ready_for_one_planning_request():
    reason = generation_matches_expected(
        _receipt(), {'session_id': 'scan-a', 'accepted_views': 2}, 2)

    assert reason == ''


def test_first_voxel_view_is_an_explicit_seed_generation():
    reason = generation_matches_expected(
        _receipt(accepted=0, policy='voxel_nbv_seed'),
        {'session_id': 'scan-a', 'accepted_views': 0}, 0)

    assert reason == ''


def test_ray_nbv_generation_and_seed_are_supported():
    assert generation_matches_expected(
        _receipt(accepted=0, policy='ray_nbv_seed'),
        {'session_id': 'scan-a', 'accepted_views': 0}, 0) == ''
    assert generation_matches_expected(
        _receipt(accepted=2, policy='ray_nbv'),
        {'session_id': 'scan-a', 'accepted_views': 2}, 2) == ''


def test_view_policy_capabilities_keep_generators_exchangeable():
    legacy = view_policy_capabilities('legacy')
    shadow = view_policy_capabilities('voxel_nbv_shadow')
    voxel = view_policy_capabilities('voxel_nbv')
    ray = view_policy_capabilities('ray_nbv')

    assert legacy.candidate_geometry == 'exact_point'
    assert not legacy.authoritative_nbv
    assert not shadow.minimum_gain_required
    assert voxel.candidate_geometry == 'exact_point'
    assert voxel.authoritative_nbv and voxel.minimum_gain_required
    assert ray.candidate_geometry == 'target_ray'
    assert ray.authoritative_nbv and ray.minimum_gain_required
    assert ray.frozen_candidates and ray.ray_expansion


def test_unknown_view_policy_fails_closed():
    with pytest.raises(ValueError, match='unsupported'):
        view_policy_capabilities('future-unregistered-policy')


def test_previous_session_receipt_cannot_start_a_repeated_mission():
    reason = generation_matches_expected(
        _receipt(session='old-scan'),
        {'session_id': 'new-scan', 'accepted_views': 2}, 2)

    assert 'different scan session' in reason


def test_previous_capture_generation_cannot_plan_the_next_view():
    reason = generation_matches_expected(
        _receipt(accepted=1),
        {'session_id': 'scan-a', 'accepted_views': 2}, 2)

    assert 'generation is 1' in reason


def test_model_rebuild_receipt_remains_not_ready_without_loosening_freshness():
    reason = generation_matches_expected(
        _receipt(ready=False),
        {'session_id': 'scan-a', 'accepted_views': 2}, 2)

    assert reason == 'coverage model is rebuilding'


def test_generation_is_immutable_and_rejects_counter_mismatch():
    generation = parse_view_generation(_receipt())

    with pytest.raises(FrozenInstanceError):
        generation.ready = False
    with pytest.raises(ValueError, match='does not match'):
        make_view_generation('scan-a', 2, 'voxel_nbv', 1, True, 4)


@pytest.mark.parametrize('payload', [
    {},
    {'view_generation': None},
    {'view_generation': {'schema_version': 9}},
])
def test_malformed_receipts_fail_closed(payload):
    assert generation_matches_expected(
        payload, {'session_id': 'scan-a', 'accepted_views': 2}, 2)
