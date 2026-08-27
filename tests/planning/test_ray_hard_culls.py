"""Persistent hard-cull planning regressions."""

from piper_gui.ray_review_model import state_at_event
from piper_mobile_manipulation.ray_hard_culls import (
    HardCullLedger,
    hard_cull_snapshot,
    prune_hard_culled_rays,
    ray_population_identity,
)
from piper_mobile_manipulation.ray_mission_diagnostics import (
    add_bridge_request,
    add_prequalification,
    planner_generation_snapshot,
)


def ray(ray_id, direction):
    return {
        'index': ray_id,
        'ray_id': ray_id,
        'ray_direction': {
            'x': direction[0], 'y': direction[1], 'z': direction[2]},
        'desired_camera_position': {
            'x': direction[0] * 0.4,
            'y': direction[1] * 0.4,
            'z': direction[2] * 0.4,
        },
        'ray_min_standoff_m': 0.28,
        'ray_max_standoff_m': 0.80,
        'ray_scoring_standoff_m': 0.40,
    }


def population():
    rays = [
        ray(0, [1.0, 0.0, 0.0]),
        ray(1, [0.0, 1.0, 0.0]),
        ray(2, [0.0, 0.0, 1.0]),
    ]
    identity = ray_population_identity(
        rays, 'mission-a', 'session-a', [0.0, 0.0, 0.0], 'base_link')
    return rays, identity


def hard(ray_id, reason='CAPABILITY_MAP_NO_SUPPORT'):
    return {
        'ray_id': ray_id,
        'stage': 'prequalification',
        'reason_code': 'CAPABILITY_MAP_NO_SUPPORT',
        'reason': reason,
        'evidence': {'supported': False},
    }


def test_hard_cull_ledger_is_population_bound_and_monotonic_per_revision():
    rays, identity = population()
    ledger = HardCullLedger()
    ledger.reset(identity)
    assert ledger.update(hard_cull_snapshot(
        identity, 'prequalification', 'revision-a', 0, [hard(0)]))
    # A restarted source may see only survivors. The same immutable revision
    # cannot resurrect its previously proven static failure.
    assert ledger.update(hard_cull_snapshot(
        identity, 'prequalification', 'revision-a', 1, []))
    assert list(ledger.entries(identity)) == [0]
    assert [item['ray_id'] for item in prune_hard_culled_rays(
        rays, ledger.entries(identity))] == [1, 2]

    # A changed capability/configuration revision replaces that source.
    assert ledger.update(hard_cull_snapshot(
        identity, 'prequalification', 'revision-b', 1, []))
    assert ledger.entries(identity) == {}

    _other_rays, other = population()
    other['session_id'] = 'different-session'
    assert not ledger.update(hard_cull_snapshot(
        other, 'prequalification', 'revision-b', 1, [hard(1)]))


def test_envelope_bound_changes_do_not_create_a_new_ray_universe():
    rays, before = population()
    bounded = [dict(item) for item in rays]
    for item in bounded:
        item['ray_min_standoff_m'] = 0.41
        item['ray_scoring_standoff_m'] = 0.53
        direction = item['ray_direction']
        item['desired_camera_position'] = {
            axis: float(direction[axis]) * 0.53
            for axis in ('x', 'y', 'z')}

    after = ray_population_identity(
        bounded, 'mission-a', 'session-a', [0.0, 0.0, 0.0], 'base_link')

    assert after == before


def test_bootstrap_culls_do_not_cross_qualified_center_transition():
    rays, identity = population()
    ledger = HardCullLedger()
    ledger.reset(identity)
    assert ledger.update(hard_cull_snapshot(
        identity, 'prequalification', 'revision-a', 0, [hard(0)]))
    bounded = [dict(item, ray_min_standoff_m=0.41) for item in rays]
    after = ray_population_identity(
        bounded, 'mission-a', 'session-a', [0.05, 0.0, 0.0], 'base_link')

    assert [item['ray_id'] for item in prune_hard_culled_rays(
        bounded, ledger.entries(after))] == [0, 1, 2]
    ledger.reset(after)
    assert ledger.entries(after) == {}


def test_later_generation_ranks_only_survivors_of_hard_culls():
    rays, identity = population()
    first = planner_generation_snapshot(
        'session-a', 0, [0.0, 0.0, 0.0], 'base_link', 'ray_nbv',
        rays, rays, rays, rays, mission_id='mission-a')
    first = add_prequalification(first, [
        dict(rays[0], prequalified=False, reachable=False,
             reject_reasons=['CAPABILITY_MAP_NO_SUPPORT'],
             cull_disposition='permanent'),
        dict(rays[1], prequalified=True, reachable=True, reject_reasons=[]),
        dict(rays[2], prequalified=True, reachable=True, reject_reasons=[]),
    ], {'safe_viewpoints': 2})
    persistent = {0: hard(0)}
    second = planner_generation_snapshot(
        'session-a', 1, [0.0, 0.0, 0.0], 'base_link', 'ray_nbv',
        rays, [rays[2]], [rays[2]], [rays[2]],
        planner_rejections={1: ['duplicate of an accepted camera pose']},
        mission_id='mission-a', persistent_culls=persistent)

    cull = second['events'][0]
    rank = second['events'][1]
    assert first['ray_population_sha256'] == identity['sha256']
    assert second['ray_population_sha256'] == identity['sha256']
    assert cull['ray_deltas']['0']['status'] == 'culled'
    assert cull['ray_deltas']['0']['culled'] is True
    assert cull['ray_deltas']['0']['cull_disposition'] == 'permanent'
    assert 0 not in cull['newly_culled_ray_ids']
    assert cull['metrics'] == {
        'input_ray_count': 2,
        'eliminated_ray_count': 1,
        'surviving_ray_count': 1,
        'carried_hard_culled_count': 1,
    }
    assert set(rank['ray_deltas']) == {'2'}


def test_historical_cull_return_is_labelled_as_reevaluated():
    document = {'events': [{
        'event_id': 'generate', 'sequence': 0,
        'ray_deltas': {'7': {'ray_id': 7, 'direction': [1, 0, 0]}},
    }, {
        'event_id': 'culled', 'sequence': 1,
        'newly_culled_ray_ids': [7],
        'ray_deltas': {'7': {
            'ray_id': 7, 'culled': True, 'status': 'culled',
            'cull_stage': 'prequalification', 'reasons': ['unsupported']}},
    }, {
        'event_id': 'returned', 'sequence': 2,
        'ray_deltas': {'7': {
            'ray_id': 7, 'culled': False, 'status': 'surviving',
            'cull_stage': '', 'reasons': []}},
    }]}

    state = state_at_event(document, 2)['rays'][7]

    assert state['reevaluated_at_event'] is True
    assert state['previous_cull_stage'] == 'prequalification'
    assert state['previous_cull_reasons'] == ['unsupported']


def test_unshortlisted_rays_are_deferred_not_falsely_culled():
    rays, _identity = population()
    snapshot = planner_generation_snapshot(
        'session-a', 0, [0.0, 0.0, 0.0], 'base_link', 'ray_nbv',
        rays, rays, rays, rays, mission_id='mission-a')

    result = add_bridge_request(snapshot, 'request-a', [2])
    by_id = {item['ray_id']: item for item in result['rays']}
    event = result['events'][-1]

    assert by_id[0]['bridge_status'] == 'deferred_shortlist'
    assert event['ray_deltas']['0']['status'] == 'deferred'
    assert event['ray_deltas']['0']['culled'] is False
    assert event['newly_culled_ray_ids'] == []
