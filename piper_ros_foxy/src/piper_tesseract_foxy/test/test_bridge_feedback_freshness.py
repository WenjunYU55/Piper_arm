import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

import piper_tesseract_foxy.bridge_node as bridge_module
import piper_tesseract_foxy.candidate_selection as candidate_selection_module
from piper_tesseract_foxy.candidate_selection import (
    balanced_closed_loop_candidates,
    bounded_candidate_attempt_limit,
    bounded_nbv_candidates,
    bounded_current_look_direction,
    exact_target_aim_candidates,
    information_ranked_ray_candidates,
    local_view_frontier_candidates,
    maximize_successive_view_distance,
    obstacle_scene_rejection_reason,
    permanent_ray_ids_from_response,
    relax_closed_loop_candidate_aims,
    select_diverse_smooth_view_path,
    target_envelope_obstacles,
    uses_authoritative_nbv_order,
    validate_candidate_policy_batch,
)
from piper_tesseract_foxy.bridge_node import TesseractPlanBridge
from piper_tesseract_foxy.contract import ContractError
from piper_mobile_manipulation.view_generation import view_policy_capabilities
from piper_mobile_manipulation.motion_limit_stability import MotionLimitStability
from piper_mobile_manipulation.target_envelope import (
    build_revolution_envelope,
    trusted_silhouette_measurement,
)


def test_bridge_preserves_extracted_candidate_selection_exports():
    """Keep historical bridge helper imports identical during migration."""
    names = (
        'FINAL_AIM_EXECUTION_MARGIN_DEG',
        'RAY_DIRECTION_ATTEMPT_LIMIT',
        'balanced_closed_loop_candidates',
        'bounded_candidate_attempt_limit',
        'bounded_current_look_direction',
        'bounded_nbv_candidates',
        'exact_target_aim_candidates',
        'information_ranked_ray_candidates',
        'local_view_frontier_candidates',
        'maximize_successive_view_distance',
        'obstacle_scene_rejection_reason',
        'permanent_ray_ids_from_response',
        'relax_closed_loop_candidate_aims',
        'select_diverse_smooth_view_path',
        'target_envelope_obstacles',
        'uses_authoritative_nbv_order',
        'validate_candidate_policy_batch',
    )
    for name in names:
        assert getattr(bridge_module, name) is getattr(
            candidate_selection_module, name)


class _Recorder:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


def _target_envelope_fixture():
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[20:55, 30:65] = 255
    support = np.ones((35, 35), dtype=bool)
    depth = np.full((35, 35), 0.40, dtype=float)
    header = SimpleNamespace(
        stamp=SimpleNamespace(sec=1, nanosec=2),
        frame_id='camera_color_optical_frame')
    shape = trusted_silhouette_measurement(
        mask, support, (30, 20), depth,
        np.asarray([
            [100.0, 0.0, 50.0],
            [0.0, 100.0, 40.0],
            [0.0, 0.0, 1.0],
        ]),
        header, 0.9)
    return build_revolution_envelope(
        shape, np.eye(4), [0.0, 0.0, 0.40])


def test_target_envelope_reuses_box_contract_and_hash_binding():
    envelope = _target_envelope_fixture()
    payload = {
        'target_envelope': envelope,
        'viewpoints': [{
            'candidate_geometry': 'target_ray',
            'target_envelope_sha256': envelope['envelope_sha256'],
        }],
    }

    validated, boxes = target_envelope_obstacles(
        payload, envelope['planning_anchor_m'])

    assert validated['envelope_sha256'] == envelope['envelope_sha256']
    assert boxes == envelope['collision_boxes']
    assert all(box['type'] == 'box' for box in boxes)


def test_target_envelope_rejects_wrong_anchor_or_unbound_ray():
    envelope = _target_envelope_fixture()
    payload = {
        'target_envelope': envelope,
        'viewpoints': [{
            'candidate_geometry': 'target_ray',
            'target_envelope_sha256': '0' * 64,
        }],
    }
    with pytest.raises(ContractError, match='not bound'):
        target_envelope_obstacles(payload, envelope['planning_anchor_m'])
    payload['viewpoints'][0]['target_envelope_sha256'] = (
        envelope['envelope_sha256'])
    with pytest.raises(ContractError, match='anchor disagrees'):
        target_envelope_obstacles(payload, [0.1, 0.0, 0.40])


def test_point_and_ray_nbv_share_ranking_but_seed_and_legacy_do_not():
    assert uses_authoritative_nbv_order([
        {'view_selection_policy': 'voxel_nbv'}])
    assert uses_authoritative_nbv_order([
        {'view_selection_policy': 'ray_nbv'}])
    assert not uses_authoritative_nbv_order([
        {'view_selection_policy': 'ray_nbv_seed'}])
    assert not uses_authoritative_nbv_order([
        {'view_selection_policy': 'legacy'}])
    assert not uses_authoritative_nbv_order([])


def _policy_candidate(policy, geometry='exact_point', item_id=1):
    candidate = {
        'id': item_id,
        'view_selection_policy': policy,
        'candidate_geometry': geometry,
        'camera_position_m': [0.3, 0.0, 0.2],
        'nbv_rank': 1,
        'nbv_marginal_information_fraction': 0.5,
    }
    if geometry == 'target_ray':
        candidate.update({
            'ray_id': item_id,
            'ray_direction': [1.0, 0.0, 0.0],
            'ray_min_standoff_m': 0.28,
            'ray_max_standoff_m': 0.50,
            'ray_preferred_max_standoff_m': 0.45,
        })
    return candidate


def test_candidate_policy_batch_preserves_point_and_ray_seams():
    voxel = validate_candidate_policy_batch([
        _policy_candidate('voxel_nbv')])
    ray = validate_candidate_policy_batch([
        _policy_candidate('ray_nbv', 'target_ray')])

    assert voxel.candidate_geometry == 'exact_point'
    assert not voxel.ray_expansion
    assert ray.candidate_geometry == 'target_ray'
    assert ray.ray_expansion

    ray_seed = validate_candidate_policy_batch([
        _policy_candidate('ray_nbv_seed', 'target_ray')])
    assert not ray_seed.authoritative_nbv
    assert ray_seed.ray_expansion


def test_candidate_policy_batch_rejects_mixed_policy_or_geometry():
    with pytest.raises(ContractError, match='mixes'):
        validate_candidate_policy_batch([
            _policy_candidate('voxel_nbv', item_id=1),
            _policy_candidate('ray_nbv', 'target_ray', item_id=2),
        ])
    with pytest.raises(ContractError, match='requires target_ray'):
        validate_candidate_policy_batch([
            _policy_candidate('ray_nbv', 'exact_point')])


def test_only_ray_policy_reduces_direction_attempt_bound_to_six():
    assert bounded_candidate_attempt_limit(
        view_policy_capabilities('ray_nbv'), 12) == 6
    assert bounded_candidate_attempt_limit(
        view_policy_capabilities('ray_nbv_seed'), 12) == 6
    assert bounded_candidate_attempt_limit(
        view_policy_capabilities('voxel_nbv'), 12) == 12


def test_only_contextual_failures_reset_after_an_accepted_view():
    bridge = object.__new__(TesseractPlanBridge)
    bridge.tesseract_exhausted_ray_generation = None
    bridge.tesseract_exhausted_ray_ids = set()
    bridge.remaining_ray_pool_session = None
    bridge.remaining_ray_ids = set()
    bridge.retired_ray_ids = set()
    bridge.get_logger = lambda: _Logger()
    candidates = [
        _policy_candidate('ray_nbv', 'target_ray', item_id=item_id)
        for item_id in (10, 11, 12)
    ]

    assert TesseractPlanBridge.exclude_tesseract_exhausted_rays(
        bridge, candidates, 'session-a', 2) == candidates
    bridge.tesseract_exhausted_ray_ids.update((10, 12))
    available = TesseractPlanBridge.exclude_tesseract_exhausted_rays(
        bridge, candidates, 'session-a', 2)

    assert [item['ray_id'] for item in available] == [11]
    bridge.remaining_ray_ids.discard(10)
    bridge.retired_ray_ids.add(10)
    available = TesseractPlanBridge.exclude_tesseract_exhausted_rays(
        bridge, candidates, 'session-a', 3)
    assert [item['ray_id'] for item in available] == [11, 12]
    assert bridge.tesseract_exhausted_ray_ids == set()
    assert bridge.remaining_ray_ids == {11, 12}
    assert bridge.retired_ray_ids == {10}
    assert TesseractPlanBridge.exclude_tesseract_exhausted_rays(
        bridge, candidates, 'session-b', 0) == candidates
    assert bridge.remaining_ray_ids == {10, 11, 12}
    assert bridge.retired_ray_ids == set()


def test_successful_plan_remembers_static_failures_before_selected_ray():
    bridge = object.__new__(TesseractPlanBridge)
    bridge.remaining_ray_pool_session = 'session-a'
    bridge.remaining_ray_ids = {10, 11}
    bridge.retired_ray_ids = set()
    bridge.get_logger = lambda: _Logger()
    bridge.pending = {'request-a': {'request': {
        'planning': {'shortlisted_ray_count': 2},
        'scan_session': {'session_id': 'session-a', 'accepted_views': 2},
        'scene': {'candidate_views': [
            {'id': 100, 'ray_id': 10},
            {'id': 101, 'ray_id': 10},
            {'id': 110, 'ray_id': 11},
        ]},
    }}}
    payload = {
        'request_id': 'request-a',
        'status': 'success',
        'planning_diagnostics': {'permanent_infeasible_ray_ids': [10, 999]},
    }

    remembered = TesseractPlanBridge.remember_permanently_infeasible_rays(
        bridge, payload)

    assert remembered == [10]
    assert bridge.remaining_ray_ids == {11}
    assert bridge.retired_ray_ids == {10}


def test_exhausted_continuous_ray_ik_is_permanent_but_path_failure_is_not():
    request = {'scene': {'candidate_views': [
        {'id': 100, 'ray_id': 10},
        {'id': 110, 'ray_id': 11},
    ]}}
    diagnostics = {'candidate_failures': [
        {'id': 100, 'stage': 'RAY_IK_FAILURE'},
        {'id': 110, 'stage': 'PLANNING_FAILURE'},
    ]}

    assert permanent_ray_ids_from_response(request, diagnostics) == [10]


def test_temporary_candidate_absence_does_not_delete_frozen_ray():
    bridge = object.__new__(TesseractPlanBridge)
    bridge.tesseract_exhausted_ray_generation = None
    bridge.tesseract_exhausted_ray_ids = set()
    bridge.remaining_ray_pool_session = None
    bridge.remaining_ray_ids = set()
    bridge.retired_ray_ids = set()
    bridge.get_logger = lambda: _Logger()
    candidates = [
        _policy_candidate('ray_nbv', 'target_ray', item_id=item_id)
        for item_id in (10, 11, 12)
    ]

    TesseractPlanBridge.exclude_tesseract_exhausted_rays(
        bridge, candidates, 'session-a', 0)
    assert TesseractPlanBridge.exclude_tesseract_exhausted_rays(
        bridge, candidates[:1], 'session-a', 0) == candidates[:1]
    assert TesseractPlanBridge.exclude_tesseract_exhausted_rays(
        bridge, candidates, 'session-a', 0) == candidates


def test_all_tesseract_exhausted_rays_end_the_true_ray_frontier():
    bridge = object.__new__(TesseractPlanBridge)
    bridge.tesseract_exhausted_ray_generation = ('session-a', 2)
    bridge.tesseract_exhausted_ray_ids = {10, 11}
    bridge.get_logger = lambda: _Logger()
    candidates = [
        _policy_candidate('ray_nbv', 'target_ray', item_id=item_id)
        for item_id in (10, 11)
    ]

    with pytest.raises(ContractError, match='RAY_FRONTIER_EXHAUSTED'):
        TesseractPlanBridge.exclude_tesseract_exhausted_rays(
            bridge, candidates, 'session-a', 2)


def test_next_ray_shortlist_is_selected_from_full_untried_frontier():
    bridge = object.__new__(TesseractPlanBridge)
    bridge.tesseract_exhausted_ray_generation = ('session-a', 2)
    bridge.tesseract_exhausted_ray_ids = set(range(1, 7))
    bridge.get_logger = lambda: _Logger()
    candidates = [
        _policy_candidate('ray_nbv', 'target_ray', item_id=item_id)
        for item_id in range(1, 13)
    ]

    available = TesseractPlanBridge.exclude_tesseract_exhausted_rays(
        bridge, candidates, 'session-a', 2)
    shortlisted = bounded_nbv_candidates(
        available, [0.0, 0.0, 0.0], [0.4, 0.0, 0.0], 6)

    assert len(available) == 6
    assert {item['ray_id'] for item in shortlisted} == set(range(7, 13))


def test_information_ray_shortlist_does_not_insert_nearby_fallback():
    candidates = [
        dict(
            _policy_candidate(
                'ray_nbv', 'target_ray', item_id=item_id),
            nbv_rank=rank,
            coverage_score=1.0 / rank,
        )
        for item_id, rank in (
            (1, 99), (2, 1), (3, 2), (4, 3),
            (5, 4), (6, 5), (7, 6))
    ]

    shortlisted = information_ranked_ray_candidates(candidates, 6)

    assert [item['ray_id'] for item in shortlisted] == [2, 3, 4, 5, 6, 7]


def test_worker_exhaustion_retires_only_attempted_request_rays():
    bridge = object.__new__(TesseractPlanBridge)
    bridge.tesseract_exhausted_ray_generation = ('session-a', 2)
    bridge.tesseract_exhausted_ray_ids = set()
    bridge.get_logger = lambda: _Logger()
    bridge.pending = {'request-a': {'request': {
        'planning': {'shortlisted_ray_count': 2},
        'scan_session': {'session_id': 'session-a', 'accepted_views': 2},
        'scene': {'candidate_views': [
            {'id': 100, 'ray_id': 10},
            {'id': 101, 'ray_id': 10},
            {'id': 110, 'ray_id': 11},
        ]},
    }}}
    payload = {
        'request_id': 'request-a',
        'rejection_codes': ['TESSERACT_EXHAUSTED'],
        'planning_diagnostics': {'attempted_ray_ids': [10, 11, 999]},
    }

    retired = TesseractPlanBridge.remember_tesseract_exhausted_rays(
        bridge, payload)

    assert retired == [10, 11]
    assert bridge.tesseract_exhausted_ray_ids == {10, 11}
    assert TesseractPlanBridge.remember_tesseract_exhausted_rays(
        bridge, payload) == []


def test_ray_worker_exhaustion_publishes_retryable_batch_diagnostic():
    bridge = object.__new__(TesseractPlanBridge)
    bridge.tesseract_exhausted_ray_generation = ('session-a', 2)
    bridge.tesseract_exhausted_ray_ids = set()
    bridge.get_logger = lambda: _Logger()
    bridge.pending = {'request-a': {'request': {
        'planning': {'shortlisted_ray_count': 1},
        'scan_session': {'session_id': 'session-a', 'accepted_views': 2},
        'scene': {'candidate_views': [{'id': 100, 'ray_id': 10}]},
    }}}
    published = []
    bridge.publish_rejection = lambda *args, **kwargs: published.append(
        (args, kwargs))

    TesseractPlanBridge.publish_plan(bridge, {
        'request_id': 'request-a',
        'status': 'failed',
        'diagnostic': 'no ray was reachable',
        'rejection_codes': ['TESSERACT_EXHAUSTED'],
        'planning_diagnostics': {'attempted_ray_ids': [10]},
    })

    assert published[0][0][1] == 'RAY_SHORTLIST_EXHAUSTED'
    assert published[0][1]['additional_codes'] == ['TESSERACT_EXHAUSTED']
    assert bridge.tesseract_exhausted_ray_ids == {10}


def test_bridge_caches_and_logs_view_generation_with_debug_enabled():
    """The generation receipt path must not crash after a valid plan update."""
    bridge = object.__new__(TesseractPlanBridge)
    bridge.latest_scan = None
    bridge.latest_acquisition_scan = None
    bridge.updated = {}
    bridge.view_generation_pub = _Recorder()
    logger = _Logger()
    bridge.get_logger = lambda: logger
    bridge.get_parameter = lambda name: SimpleNamespace(
        value=True if name == 'debug' else None)
    bridge.now = lambda: 12.5
    payload = {
        'view_generation': {
            'schema_version': 1,
            'session_id': 'scan-session',
            'accepted_views': 2,
            'policy': 'voxel_nbv',
            'generation': 2,
            'ready': True,
            'candidate_viewpoints': 18,
            'reason': '',
        },
    }

    bridge.store_scan(
        SimpleNamespace(data=json.dumps(payload)), 'scan', acquisition=False)

    assert bridge.latest_scan == payload
    assert bridge.updated == {'scan': 12.5}
    assert len(bridge.view_generation_pub.messages) == 1
    receipt = json.loads(bridge.view_generation_pub.messages[0].data)
    assert receipt['view_generation']['session_id'] == 'scan-session'
    assert receipt['view_generation']['accepted_views'] == 2
    assert logger.messages == [
        'cached view generation session=scan-session accepted=2 '
        'policy=voxel_nbv ready=True candidates=18',
    ]


def test_multiview_candidates_follow_smooth_camera_route():
    candidates = [
        {
            'id': index,
            'camera_position_m': [0.0, 0.1 * index, 0.0],
        }
        for index in range(5)
    ]

    ordered = maximize_successive_view_distance(candidates)

    assert [item['id'] for item in ordered] == [0, 1, 2, 3, 4]


def test_qualified_thirteen_view_orbit_does_not_pendulum_between_endpoints():
    radius = 0.30
    angles = [
        120.0 + index * (55.0 / 12.0)
        for index in range(13)
    ]
    candidates = [
        {
            'id': index,
            'camera_position_m': [
                radius * math.cos(math.radians(angle)),
                radius * math.sin(math.radians(angle)),
                0.0,
            ],
        }
        for index, angle in enumerate(angles)
    ]

    ordered = maximize_successive_view_distance(candidates)
    distances = [
        math.dist(
            first['camera_position_m'], second['camera_position_m'])
        for first, second in zip(ordered[:-1], ordered[1:])
    ]

    assert [item['id'] for item in ordered] == list(range(13))
    assert max(distances) == pytest.approx(0.0240, abs=0.0001)


def test_bridge_selects_spread_subset_then_orders_nearest_neighbour():
    candidates = [
        {
            'id': index,
            'camera_position_m': [float(index), 0.0, 0.0],
        }
        for index in range(7)
    ]

    ordered = select_diverse_smooth_view_path(
        candidates, selected_count=4, start_camera_position=[3.1, 0.0, 0.0])

    # Four preferred candidates cover both ends of the interval, but their
    # executed prefix is a smooth route rather than endpoint alternation.
    assert [item['id'] for item in ordered[:4]] == [3, 1, 0, 6]
    assert sorted(item['id'] for item in ordered) == list(range(7))


def test_closed_loop_frontier_excludes_large_first_view_jump():
    center = [0.4, 0.0, 0.0]
    start = [0.1, 0.0, 0.2]
    candidates = []
    for item_id, angle_deg, score in ((1, 10.0, 0.4), (2, 19.0, 0.8),
                                      (3, 35.0, 1.0)):
        angle = math.radians(angle_deg)
        candidates.append({
            'id': item_id,
            'camera_position_m': [
                center[0] - 0.3 * math.cos(angle),
                center[1] + 0.3 * math.sin(angle),
                center[2] + 0.2,
            ],
            'coverage_score': score,
        })

    # Build angles against the actual current target-to-camera elevation.
    current = [dict(item) for item in candidates]
    reference = [start[0] - center[0], start[1], start[2]]
    reference_norm = math.sqrt(sum(value * value for value in reference))
    for item, angle_deg in zip(current, (10.0, 19.0, 35.0)):
        # Rotate the normalized reference around Z so the requested angular
        # offsets are exact and independent of the test fixture's elevation.
        x, y, z = [value / reference_norm for value in reference]
        angle = math.radians(angle_deg)
        rotated = [x * math.cos(angle) - y * math.sin(angle),
                   x * math.sin(angle) + y * math.cos(angle), z]
        item['camera_position_m'] = [
            center[index] + 0.3 * rotated[index] for index in range(3)]

    frontier = local_view_frontier_candidates(current, start, center, 20.0)

    assert [item['id'] for item in frontier] == [2, 1]
    assert all(item['_frontier_angle_deg'] <= 20.0 for item in frontier)


def test_closed_loop_frontier_excludes_nonmeaningful_capture_step():
    center = [0.0, 0.0, 0.0]
    start = [-0.36, 0.0, 0.0]
    reference = np.asarray(start, dtype=float)
    reference /= np.linalg.norm(reference)
    candidates = []
    for item_id, angle_deg in ((1, 3.0), (2, 12.0), (3, 48.0)):
        angle = math.radians(angle_deg)
        direction = [
            reference[0] * math.cos(angle) - reference[1] * math.sin(angle),
            reference[0] * math.sin(angle) + reference[1] * math.cos(angle),
            reference[2],
        ]
        candidates.append({
            'id': item_id,
            'camera_position_m': [0.36 * value for value in direction],
        })

    frontier = local_view_frontier_candidates(
        candidates, start, center, 45.0, 6.0)

    assert [item['id'] for item in frontier] == [2]


def test_elevated_fifteen_degree_orbit_neighbor_remains_eligible():
    center = np.zeros(3)
    elevation = math.radians(52.0)

    def position(azimuth_deg):
        azimuth = math.radians(azimuth_deg)
        return [
            0.30 * math.cos(elevation) * math.cos(azimuth),
            0.30 * math.cos(elevation) * math.sin(azimuth),
            0.30 * math.sin(elevation),
        ]

    # The arm achieved about 184 degrees from an exact 180-degree request.
    # Its next fixed grid neighbor is 195 degrees, which is only about seven
    # degrees apart in 3D target-ray space at this elevation.
    start = position(184.0)
    candidates = [{
        'id': 1,
        'camera_position_m': position(195.0),
        'coverage_score': 1.0,
    }]

    frontier = local_view_frontier_candidates(
        candidates, start, center, 45.0, 6.0)

    assert [item['id'] for item in frontier] == [1]
    assert 6.0 < frontier[0]['_frontier_angle_deg'] < 8.0


def test_closed_loop_aim_preserves_current_direction_inside_bound():
    nominal = [1.0, 0.0, 0.0]
    angle = math.radians(9.0)
    current = [math.cos(angle), math.sin(angle), 0.0]

    relaxed = bounded_current_look_direction(nominal, current, 12.0)

    assert relaxed == pytest.approx(current)


def test_closed_loop_aim_stops_at_strict_target_offset_bound():
    nominal = [1.0, 0.0, 0.0]
    angle = math.radians(30.0)
    current = [math.cos(angle), math.sin(angle), 0.0]

    relaxed = bounded_current_look_direction(nominal, current, 12.0)

    achieved_offset = math.degrees(math.acos(np.clip(
        np.dot(nominal, relaxed), -1.0, 1.0)))
    remaining_wrist_change = math.degrees(math.acos(np.clip(
        np.dot(current, relaxed), -1.0, 1.0)))
    assert achieved_offset == pytest.approx(12.0)
    assert remaining_wrist_change == pytest.approx(18.0)


def test_closed_loop_aim_relaxation_preserves_candidate_membership():
    candidates = [
        {'id': 1, 'camera_position_m': [0.1, 0.0, 0.2],
         'look_direction': [1.0, 0.0, 0.0]},
        {'id': 2, 'camera_position_m': [0.2, 0.0, 0.2],
         'look_direction': [0.0, 1.0, 0.0]},
    ]

    relaxed = relax_closed_loop_candidate_aims(
        candidates, [math.cos(math.radians(8.0)),
                     math.sin(math.radians(8.0)), 0.0], 12.0)

    assert [item['id'] for item in relaxed] == [1, 2]
    assert [item['camera_position_m'] for item in relaxed] == [
        item['camera_position_m'] for item in candidates]
    assert relaxed[0]['look_direction'] != candidates[0]['look_direction']
    assert candidates[0]['look_direction'] == [1.0, 0.0, 0.0]


def test_exact_target_aim_is_primary_and_fallback_is_bounded_to_five_degrees():
    target = [0.4, 0.0, 0.0]
    candidate = {
        'id': 7,
        'camera_position_m': [0.1, 0.0, 0.2],
        'look_direction': [0.0, 1.0, 0.0],
    }
    exact = np.asarray(target) - np.asarray(candidate['camera_position_m'])
    exact /= np.linalg.norm(exact)
    current = [0.0, 0.0, 1.0]

    bound = exact_target_aim_candidates(
        [candidate], target, current, 5.0)[0]

    assert bound['look_direction'] == pytest.approx(exact)
    assert len(bound['fallback_look_directions']) == 1
    offset = math.degrees(math.acos(float(np.clip(np.dot(
        exact, bound['fallback_look_directions'][0]), -1.0, 1.0))))
    assert offset == pytest.approx(4.0)
    assert bound['maximum_final_aim_offset_deg'] == 5.0


def test_closed_loop_shortlist_interleaves_coverage_and_nearby_fallbacks():
    candidates = [
        {
            'id': item_id,
            'camera_position_m': [distance, 0.0, 0.0],
            'coverage_score': score,
            '_frontier_angle_deg': frontier,
        }
        for item_id, distance, score, frontier in (
            (10, 0.90, 0.90, 45.0),
            (11, 0.10, 0.10, 5.0),
            (12, 0.70, 0.70, 35.0),
            (13, 0.20, 0.20, 10.0),
            (14, 0.50, 0.50, 25.0),
            (15, 0.30, 0.30, 15.0),
        )
    ]

    selected = balanced_closed_loop_candidates(
        candidates, [0.0, 0.0, 0.0], 4)

    assert [item['id'] for item in selected] == [10, 11, 12, 13]
    assert len({item['id'] for item in selected}) == len(selected)


def test_nbv_shortlist_keeps_information_leaders_and_local_ik_fallbacks():
    candidates = [
        {
            'id': item_id,
            'camera_position_m': [distance, 0.0, 0.0],
            'coverage_score': float(20 - rank),
            'nbv_rank': rank,
        }
        for item_id, rank, distance in (
            (10, 1, 0.80), (11, 2, 0.70), (12, 3, 0.60),
            (13, 4, 0.50), (14, 5, 0.40), (15, 6, 0.30),
            (16, 7, 0.20), (17, 8, 0.01),
        )
    ]

    selected = bounded_nbv_candidates(
        candidates, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 4,
        leader_fraction=0.75)

    assert [item['id'] for item in selected] == [10, 11, 12, 17]
    assert [item['nbv_rank'] for item in selected] == [1, 2, 3, 8]


def test_nbv_shortlist_rejects_invalid_policy_fraction():
    with pytest.raises(ValueError, match='leader fraction'):
        bounded_nbv_candidates(
            [{'id': 1, 'camera_position_m': [0.1, 0.0, 0.0]}],
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1,
            leader_fraction=0.0)


def test_nbv_shortlist_reserves_leaders_from_distinct_view_directions():
    candidates = [
        {
            'id': rank,
            'camera_position_m': [0.30 + rank * 0.01, 0.0, 0.0],
            'coverage_score': float(100 - rank),
            'nbv_rank': rank,
        }
        for rank in range(1, 9)
    ]
    candidates.extend([
        {
            'id': 9,
            'camera_position_m': [0.30, 0.18, 0.0],
            'coverage_score': 91.0,
            'nbv_rank': 9,
        },
        {
            'id': 10,
            'camera_position_m': [0.30, 0.0, 0.18],
            'coverage_score': 90.0,
            'nbv_rank': 10,
        },
    ])

    selected = bounded_nbv_candidates(
        candidates, [0.25, 0.0, 0.0], [0.0, 0.0, 0.0], 4,
        leader_fraction=0.75)

    assert 9 in [item['id'] for item in selected]
    assert 10 in [item['id'] for item in selected]
    assert [item['nbv_rank'] for item in selected] == sorted(
        item['nbv_rank'] for item in selected)


def test_nbv_shortlist_pairs_informative_azimuths_with_elevation_ik_escapes():
    def position(azimuth_deg, elevation_deg, radius=0.30):
        azimuth = np.deg2rad(float(azimuth_deg))
        elevation = np.deg2rad(float(elevation_deg))
        return [
            radius * np.cos(elevation) * np.cos(azimuth),
            radius * np.cos(elevation) * np.sin(azimuth),
            radius * np.sin(elevation),
        ]

    candidates = [
        {'id': 1, 'camera_position_m': position(90, 75), 'nbv_rank': 1},
        {'id': 2, 'camera_position_m': position(120, 75), 'nbv_rank': 2},
        # These retain the leaders' lateral sectors but offer a much less
        # wrist-down elevation for Tesseract.
        {'id': 10, 'camera_position_m': position(90, 40), 'nbv_rank': 20},
        {'id': 11, 'camera_position_m': position(120, 40), 'nbv_rank': 21},
        # A generic pose nearest the current camera must not consume both
        # reserved escape slots and erase the informative lateral directions.
        {'id': 99, 'camera_position_m': position(0, 40), 'nbv_rank': 99},
    ]

    selected = bounded_nbv_candidates(
        candidates, position(0, 40), [0.0, 0.0, 0.0], 5,
        leader_fraction=0.5)

    assert [item['id'] for item in selected] == [1, 2, 10, 11, 99]


def test_nbv_shortlist_cannot_spend_every_fallback_on_unreachable_sectors():
    def position(azimuth_deg, elevation_deg, radius=0.30):
        azimuth = np.deg2rad(float(azimuth_deg))
        elevation = np.deg2rad(float(elevation_deg))
        return [
            radius * np.cos(elevation) * np.cos(azimuth),
            radius * np.cos(elevation) * np.sin(azimuth),
            radius * np.sin(elevation),
        ]

    current = position(180, 35, 0.39)
    candidates = []
    for rank, azimuth in enumerate(range(90, 211, 15), 1):
        candidates.append({
            'id': rank,
            'camera_position_m': position(azimuth, 75),
            'nbv_rank': rank,
        })
        candidates.append({
            'id': 100 + rank,
            'camera_position_m': position(azimuth, 45),
            'nbv_rank': 100 + rank,
        })
    candidates.append({
        'id': 999,
        'camera_position_m': position(172.5, 35, 0.39),
        'nbv_rank': 999,
    })

    selected = bounded_nbv_candidates(
        candidates, current, [0.0, 0.0, 0.0], 12)

    assert 999 in [item['id'] for item in selected]
    assert len(selected) == 12
    assert [item['nbv_rank'] for item in selected] == sorted(
        item['nbv_rank'] for item in selected)


def test_nbv_shortlist_escapes_live_steep_band_without_losing_information():
    def position(azimuth_deg, elevation_deg, radius=0.30):
        azimuth = np.deg2rad(float(azimuth_deg))
        elevation = np.deg2rad(float(elevation_deg))
        return [
            radius * np.cos(elevation) * np.cos(azimuth),
            radius * np.cos(elevation) * np.sin(azimuth),
            radius * np.sin(elevation),
        ]

    # Shape this like the physical 2026-08-20 failure: the best information
    # directions are all steep, while useful target-facing IK exists in the
    # middle elevation band at nearby azimuths.
    leaders = [-135.0, -105.0, -142.5, 135.0, -165.0, 172.5]
    candidates = [
        {
            'id': rank,
            'camera_position_m': position(azimuth, 75.0),
            'nbv_rank': rank,
        }
        for rank, azimuth in enumerate(leaders, 1)
    ]
    next_rank = 20
    for elevation in (65.0, 55.0, 45.0, 35.0, 25.0, 15.0):
        for azimuth in (
                -150.0, -142.5, -135.0, -105.0,
                135.0, 142.5, 165.0, 172.5):
            candidates.append({
                'id': next_rank,
                'camera_position_m': position(azimuth, elevation),
                'nbv_rank': next_rank,
            })
            next_rank += 1
    candidates.append({
        'id': 999,
        'camera_position_m': position(172.5, 65.0, 0.39),
        'nbv_rank': 999,
    })

    selected = bounded_nbv_candidates(
        candidates, position(172.5, 65.0, 0.39),
        [0.0, 0.0, 0.0], 12)
    selected_angles = []
    for item in selected:
        camera = np.asarray(item['camera_position_m'])
        radius = float(np.linalg.norm(camera))
        selected_angles.append((
            float(np.rad2deg(np.arctan2(camera[1], camera[0]))),
            float(np.rad2deg(np.arcsin(camera[2] / radius))),
        ))

    assert all(item['id'] in [candidate['id'] for candidate in candidates]
               for item in selected)
    selected_ranks = {item['nbv_rank'] for item in selected}
    # Ranks 1 and 3 occupy the same 30-degree direction sector, so retaining
    # either one is sufficient; the other global sectors must remain.
    assert len(selected_ranks.intersection(range(1, 7))) >= 5
    assert 999 in [item['id'] for item in selected]
    assert any(
        azimuth == pytest.approx(142.5)
        and elevation == pytest.approx(45.0)
        for azimuth, elevation in selected_angles)
    assert any(
        azimuth == pytest.approx(-150.0)
        and elevation == pytest.approx(45.0)
        for azimuth, elevation in selected_angles)


def test_closed_loop_shortlist_keeps_proven_local_fallback_under_diversity_bias():
    candidates = [
        {
            'id': item_id,
            'camera_position_m': [distance, 0.0, 0.0],
            'coverage_score': float(100 - item_id),
            '_frontier_angle_deg': float(40 - item_id),
        }
        for item_id, distance in enumerate(
            [0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.01])
    ]

    selected = balanced_closed_loop_candidates(
        candidates, [0.0, 0.0, 0.0], 4)

    assert selected[0]['id'] == 0
    assert selected[1]['id'] == 7


def test_first_closed_loop_view_orders_seed_only_by_current_camera_travel():
    candidates = [
        {
            'id': item_id,
            'camera_position_m': [distance, 0.0, 0.0],
            'coverage_score': float(100 - item_id),
            '_frontier_angle_deg': float(40 - item_id),
        }
        for item_id, distance in enumerate([0.30, 0.20, 0.10, 0.01])
    ]

    selected = balanced_closed_loop_candidates(
        candidates, [0.0, 0.0, 0.0], 4, compact_first=True)

    assert [item['id'] for item in selected] == [3, 2, 1, 0]


def test_first_ray_seed_uses_distance_to_bounded_interval_not_scoring_point():
    candidates = [
        {
            'id': 1,
            'camera_position_m': [0.06, 0.0, 0.0],
            'coverage_score': 1000.0,
            'candidate_geometry': 'target_ray',
            'ray_direction': [-1.0, 0.0, 0.0],
            'ray_scoring_standoff_m': 0.34,
            'ray_min_standoff_m': 0.10,
            'ray_max_standoff_m': 0.30,
        },
        {
            'id': 2,
            'camera_position_m': [0.19, 0.0, 0.0],
            'coverage_score': -1000.0,
        },
    ]

    selected = balanced_closed_loop_candidates(
        candidates, [0.20, 0.0, 0.0], 2, compact_first=True)

    assert [item['id'] for item in selected] == [1, 2]


def test_closed_loop_fallback_steps_toward_missing_feature_leader():
    candidates = [
        {
            'id': 1,
            'camera_position_m': [0.90, 0.0, 0.0],
            'coverage_score': 10.0,
            'coverage_progress_score': 0.50,
            '_frontier_angle_deg': 45.0,
        },
        {
            'id': 2,
            'camera_position_m': [-0.05, 0.0, 0.0],
            'coverage_score': 0.1,
            'coverage_progress_score': -0.10,
            '_frontier_angle_deg': 5.0,
        },
        {
            'id': 3,
            'camera_position_m': [0.10, 0.0, 0.0],
            'coverage_score': 1.0,
            'coverage_progress_score': 0.01,
            '_frontier_angle_deg': 10.0,
        },
    ]

    selected = balanced_closed_loop_candidates(
        candidates, [0.0, 0.0, 0.0], 3)

    assert [item['id'] for item in selected[:2]] == [1, 3]
    assert 2 not in [item['id'] for item in selected]


def test_closed_loop_tries_material_feature_progress_before_near_duplicates():
    candidates = [
        {
            'id': item_id,
            'camera_position_m': [distance, 0.0, 0.0],
            'coverage_score': score,
            'coverage_progress_score': progress,
            'coverage_objective': 'negative_y_face',
            '_frontier_angle_deg': frontier,
        }
        for item_id, distance, score, progress, frontier in (
            (1, 0.40, 10.0, 0.20, 18.0),
            (2, 0.35, 9.0, 0.15, 17.0),
            (3, 0.30, 8.0, 0.10, 16.0),
            (4, 0.02, 1.0, 0.01, 4.0),
            (5, 0.03, 0.5, 0.00, 5.0),
            (6, 0.04, 0.2, 0.02, 6.0),
        )
    ]

    selected = balanced_closed_loop_candidates(
        candidates, [0.0, 0.0, 0.0], 6)

    assert [item['id'] for item in selected[:3]] == [1, 2, 3]
    assert set(item['id'] for item in selected[3:]) == {4, 5, 6}


def test_progress_reordering_preserves_balanced_ik_fallback_membership():
    candidates = [
        {
            'id': item_id,
            'camera_position_m': [distance, 0.0, 0.0],
            'coverage_score': score,
            'coverage_progress_score': progress,
            'coverage_objective': 'negative_y_face',
            '_frontier_angle_deg': frontier,
        }
        for item_id, distance, score, progress, frontier in (
            (1, 0.40, 10.0, 0.20, 18.0),
            (2, 0.35, 9.0, 0.15, 17.0),
            (3, 0.30, 8.0, 0.10, 16.0),
            (4, 0.02, 1.0, 0.01, 4.0),
            (5, 0.03, 0.5, 0.00, 5.0),
            (6, 0.04, 0.2, 0.02, 6.0),
        )
    ]

    selected = balanced_closed_loop_candidates(
        candidates, [0.0, 0.0, 0.0], 4)

    assert [item['id'] for item in selected] == [1, 2, 4, 5]


def test_live_candidate_dome_selects_all_three_elevations():
    candidates = []
    index = 0
    for pitch_deg in (-55.0, -45.0, -65.0):
        pitch = math.radians(pitch_deg)
        horizontal = 0.30 * math.cos(pitch)
        z = -0.30 * math.sin(pitch)
        for angle_deg in (
                120.0, 129.1666667, 138.3333333, 147.5,
                156.6666667, 165.8333333, 175.0):
            angle = math.radians(angle_deg)
            candidates.append({
                'id': index,
                'camera_position_m': [
                    horizontal * math.cos(angle),
                    horizontal * math.sin(angle),
                    z,
                ],
            })
            index += 1

    ordered = select_diverse_smooth_view_path(
        candidates, selected_count=13,
        start_camera_position=candidates[0]['camera_position_m'])
    selected_ids = [item['id'] for item in ordered[:13]]

    assert {item_id // 7 for item_id in selected_ids} == {0, 1, 2}
    assert len({item_id % 7 for item_id in selected_ids}) >= 6
    assert len(set(selected_ids)) == 13


def test_bridge_preserves_fresh_valid_limits_through_transient_invalid_sample():
    valid = SimpleNamespace(valid=True, limits_sha256='a' * 64)
    invalid = SimpleNamespace(valid=False, limits_sha256='0' * 64)
    marked = []
    bridge = SimpleNamespace(
        latest_motion_limits=valid,
        motion_limit_stability=MotionLimitStability(),
        now=lambda: 1.0,
        mark=lambda key: marked.append(key),
    )
    bridge.motion_limit_stability.observe(valid, 0.0)

    TesseractPlanBridge.motion_limits_cb(bridge, invalid)

    assert bridge.latest_motion_limits is valid
    assert marked == []


def test_initial_multiview_readiness_reports_missing_scan_without_crashing():
    bridge = snapshot_bridge()
    bridge.latest_scan = None
    bridge.fresh = lambda key, _timeout=None: key != 'scan'
    reasons = TesseractPlanBridge.snapshot_reasons(
        bridge, 'MULTIVIEW_SCAN')
    assert 'scan data is missing or stale' in reasons
    assert 'scan session identity is missing' in reasons


def test_valid_unsafe_obstacle_is_forwarded_to_planner():
    scene = SimpleNamespace(
        scene_blocked=True,
        blocking_reason='2:unknown depth foreground',
        instances=[SimpleNamespace(valid=True)],
    )

    assert obstacle_scene_rejection_reason(scene) is None


def test_invalid_obstacle_geometry_remains_fail_closed():
    scene = SimpleNamespace(
        scene_blocked=True,
        blocking_reason='2:transform_unavailable',
        instances=[SimpleNamespace(valid=False)],
    )

    assert obstacle_scene_rejection_reason(scene) == (
        'obstacle scene is blocked: 2:transform_unavailable')


def test_scene_level_failure_without_instances_remains_fail_closed():
    scene = SimpleNamespace(
        scene_blocked=True,
        blocking_reason='input_conversion_failed',
        instances=[],
    )

    assert obstacle_scene_rejection_reason(scene) == (
        'obstacle scene is blocked: input_conversion_failed')


def test_fresh_worker_heartbeat_is_required_for_readiness(monkeypatch):
    now_ns = 20_000_000_000
    monkeypatch.setattr(
        'piper_tesseract_foxy.bridge_node.time.time_ns',
        lambda: now_ns)
    monkeypatch.setattr(
        'piper_tesseract_foxy.bridge_node.sha256_file',
        lambda path: {'srdf': 's' * 64, 'manifest': 'm' * 64}[str(path)])
    bridge = SimpleNamespace(
        spool=SimpleNamespace(read_health=lambda: {
            'schema_version': 5,
            'generation_id': '1' * 32,
            'written_at_ns': now_ns - 100_000_000,
            'worker_ready': True,
            'backend': 'tesseract',
            'backend_version': '0.35.0.6',
            'backend_error': '',
            'srdf_sha256': 's' * 64,
            'collision_manifest_sha256': 'm' * 64,
        }),
        worker_generation_id='',
        get_parameter=lambda name: SimpleNamespace(value={
            'worker_heartbeat_timeout_sec': 1.5,
        }[name]),
        parameter_path=lambda name: {
            'srdf_path': 'srdf',
            'collision_manifest_path': 'manifest',
        }[name],
    )

    assert TesseractPlanBridge.worker_health_reasons(bridge) == []
    assert bridge.worker_generation_id == '1' * 32

    bridge.spool = SimpleNamespace(read_health=lambda: {
        'schema_version': 5,
        'generation_id': '2' * 32,
        'written_at_ns': now_ns - 2_000_000_000,
        'worker_ready': True,
        'backend': 'tesseract',
    })
    assert TesseractPlanBridge.worker_health_reasons(bridge) == [
        'Tesseract worker heartbeat is stale']


def test_worker_collision_profile_hash_mismatch_blocks_readiness(monkeypatch):
    now_ns = 20_000_000_000
    monkeypatch.setattr(
        'piper_tesseract_foxy.bridge_node.time.time_ns', lambda: now_ns)
    monkeypatch.setattr(
        'piper_tesseract_foxy.bridge_node.sha256_file', lambda path: 'a' * 64)
    bridge = SimpleNamespace(
        spool=SimpleNamespace(read_health=lambda: {
            'schema_version': 5,
            'generation_id': '3' * 32,
            'written_at_ns': now_ns,
            'worker_ready': True,
            'backend': 'tesseract',
            'srdf_sha256': 'b' * 64,
            'collision_manifest_sha256': 'a' * 64,
        }),
        worker_generation_id='',
        get_parameter=lambda name: SimpleNamespace(value=1.5),
        parameter_path=lambda name: name,
    )
    assert TesseractPlanBridge.worker_health_reasons(bridge) == [
        'Tesseract worker collision profile does not match bridge: '
        'srdf_sha256']


def test_poll_keeps_stale_subscription_entities_stable():
    published = []
    bridge = SimpleNamespace(
        latest_joints=object(),
        updated={'joints': 1.0},
        pending={},
        now=lambda: 10.0,
        get_parameter=lambda name: SimpleNamespace(value={
            'response_timeout_sec': 180.0,
        }[name]),
        publish_status=lambda: published.append(True),
        publish_readiness=lambda: published.append(True),
    )

    TesseractPlanBridge.poll(bridge)

    assert bridge.latest_joints is not None
    assert bridge.updated == {'joints': 1.0}
    assert published == [True, True]


def test_poll_does_not_require_subscription_recovery_methods():
    bridge = SimpleNamespace(
        pending={},
        now=lambda: 10.0,
        get_parameter=lambda name: SimpleNamespace(value={
            'response_timeout_sec': 180.0,
        }[name]),
        publish_status=lambda: None,
        publish_readiness=lambda: None,
    )

    TesseractPlanBridge.poll(bridge)

    assert not hasattr(bridge, 'recreate_joint_subscription')


def snapshot_bridge():
    return SimpleNamespace(
        latest_joints=SimpleNamespace(position=[0.0] * 6),
        latest_motion_limits=SimpleNamespace(
            valid=True,
            reason='fresh controller limits',
            joint_names=[
                'joint1', 'joint2', 'joint3',
                'joint4', 'joint5', 'joint6'],
            max_velocity_rad_s=[1.0] * 6,
            max_acceleration_rad_s2=[2.0] * 6,
            limits_sha256='a' * 64,
        ),
        latest_tracking=None,
        latest_camera_health=SimpleNamespace(healthy=True),
        latest_obstacles=SimpleNamespace(scene_blocked=False, instances=[]),
        latest_scan=SimpleNamespace(),
        latest_acquisition_scan={'dry_run': True},
        worker_health_reasons=lambda: [],
        fresh=lambda _key, _timeout=None: True,
        get_parameter=lambda name: SimpleNamespace(value={
            'max_tracking_measurement_age_sec': 0.75,
            'motion_limits_timeout_sec': 3.0,
            'max_execution_viewpoints': 13,
        }[name]),
    )


def test_acquisition_snapshot_does_not_require_target_tracking():
    bridge = snapshot_bridge()

    assert TesseractPlanBridge.snapshot_reasons(
        bridge, 'ROUGH_ACQUISITION') == []


def test_acquisition_snapshot_keeps_camera_gate_but_bootstraps_without_scene():
    bridge = snapshot_bridge()
    bridge.latest_camera_health.healthy = False
    bridge.latest_obstacles = SimpleNamespace(
        scene_blocked=True,
        blocking_reason='transform_unavailable',
        instances=[SimpleNamespace(valid=False)],
    )

    reasons = TesseractPlanBridge.snapshot_reasons(
        bridge, 'ROUGH_ACQUISITION')

    assert 'camera timestamp health is not healthy' in reasons
    assert not any('obstacle scene' in reason for reason in reasons)


def test_return_home_snapshot_requires_only_robot_and_worker_state():
    bridge = snapshot_bridge()
    bridge.latest_acquisition_scan = None
    assert TesseractPlanBridge.snapshot_reasons(
        bridge, 'RETURN_HOME') == []

    bridge.latest_obstacles = SimpleNamespace(
        scene_blocked=True,
        blocking_reason='transform_unavailable',
        instances=[SimpleNamespace(valid=False)],
    )
    reasons = TesseractPlanBridge.snapshot_reasons(
        bridge, 'RETURN_HOME')
    assert reasons == []

    bridge.latest_camera_health.healthy = False
    assert TesseractPlanBridge.snapshot_reasons(
        bridge, 'RETURN_HOME') == []


def test_all_dedicated_home_stages_ignore_perception_scene_state():
    bridge = snapshot_bridge()
    bridge.latest_obstacles = SimpleNamespace(
        scene_blocked=True,
        blocking_reason='no target-derived scene exists yet',
        instances=[SimpleNamespace(valid=False)],
    )
    bridge.fresh = lambda key, _timeout=None: key not in (
        'obstacles', 'acquisition_scan')

    assert TesseractPlanBridge.snapshot_reasons(
        bridge, 'RETURN_HOME', startup_home=True) == []
    normal = TesseractPlanBridge.snapshot_reasons(
        bridge, 'RETURN_HOME')
    assert normal == []


def test_normal_snapshot_keeps_scene_gate():
    bridge = snapshot_bridge()
    bridge.latest_scan = {
        'dry_run': True,
        'scan_session': {
            'session_id': 'session-a',
            'accepted_views': 0,
            'max_views': 13,
        },
        'remaining_viewpoints': 13,
    }
    bridge.latest_tracking = SimpleNamespace(
        lifecycle_state='TRACKING',
        camera_settled=True,
        prediction_only=False,
        measurement_age_sec=0.1,
    )
    bridge.latest_obstacles = SimpleNamespace(
        scene_blocked=True,
        blocking_reason='transform_unavailable',
        instances=[SimpleNamespace(valid=False)],
    )

    reasons = TesseractPlanBridge.snapshot_reasons(
        bridge, 'MULTIVIEW_SCAN')

    assert 'obstacle scene is blocked: transform_unavailable' in reasons


def test_normal_snapshot_still_requires_tracking():
    bridge = snapshot_bridge()
    bridge.latest_scan = {
        'dry_run': True,
        'scan_session': {
            'session_id': 'session-a',
            'accepted_views': 0,
            'max_views': 13,
        },
        'remaining_viewpoints': 13,
    }

    reasons = TesseractPlanBridge.snapshot_reasons(
        bridge, 'MULTIVIEW_SCAN')

    assert 'tracking is not settled TRACKING' in reasons


def test_zero_prequalified_views_preserve_transient_target_status_reason():
    bridge = snapshot_bridge()
    bridge.latest_scan = {
        'dry_run': True,
        'scan_session': {
            'session_id': 'session-a',
            'accepted_views': 1,
            'max_views': 13,
        },
        'remaining_viewpoints': 12,
        'filter': {
            'prequalified_viewpoints': 0,
            'target_status': 'LOW_CONFIDENCE',
        },
        'viewpoints': [{
            'prequalified': False,
            'reachable': False,
            'safe': False,
            'reject_reasons': ['target_status=LOW_CONFIDENCE'],
        }],
    }
    bridge.latest_tracking = SimpleNamespace(
        lifecycle_state='TRACKING',
        camera_settled=True,
        prediction_only=False,
        measurement_age_sec=0.1,
    )

    reasons = TesseractPlanBridge.snapshot_reasons(
        bridge, 'MULTIVIEW_SCAN')

    assert 'target_status=LOW_CONFIDENCE' in reasons
    assert 'NO_PREQUALIFIED_VIEWPOINT_CANDIDATE' not in reasons


def test_zero_prequalified_views_without_visual_reason_fail_as_no_candidate():
    bridge = snapshot_bridge()
    bridge.latest_scan = {
        'dry_run': True,
        'scan_session': {
            'session_id': 'session-a',
            'accepted_views': 1,
            'max_views': 13,
        },
        'remaining_viewpoints': 12,
        'filter': {
            'prequalified_viewpoints': 0,
            'target_status': 'LOCKED',
        },
        'viewpoints': [{
            'prequalified': False,
            'reachable': False,
            'safe': False,
            'reject_reasons': ['camera-object distance too close'],
        }],
    }
    bridge.latest_tracking = SimpleNamespace(
        lifecycle_state='TRACKING',
        camera_settled=True,
        prediction_only=False,
        measurement_age_sec=0.1,
    )

    reasons = TesseractPlanBridge.snapshot_reasons(
        bridge, 'MULTIVIEW_SCAN')

    assert 'NO_PREQUALIFIED_VIEWPOINT_CANDIDATE' in reasons


def test_clipped_target_failure_is_preserved_for_the_coordinator():
    bridge = snapshot_bridge()
    bridge.latest_scan = {
        'dry_run': True,
        'scan_session': {
            'session_id': 'session-a',
            'accepted_views': 0,
            'max_views': 13,
        },
        'remaining_viewpoints': 13,
        'selection_failure_code': 'TARGET_TOO_LARGE_OR_CLOSE',
        'viewpoints': [],
    }

    reasons = TesseractPlanBridge.snapshot_reasons(
        bridge, 'MULTIVIEW_SCAN')

    assert 'TARGET_TOO_LARGE_OR_CLOSE' in reasons
    assert 'NO_PREQUALIFIED_VIEWPOINT_CANDIDATE' not in reasons
