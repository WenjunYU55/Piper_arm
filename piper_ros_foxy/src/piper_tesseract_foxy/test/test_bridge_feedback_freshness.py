import math
from types import SimpleNamespace

import numpy as np
import pytest

from piper_tesseract_foxy.bridge_node import (
    balanced_closed_loop_candidates,
    bounded_current_look_direction,
    local_view_frontier_candidates,
    maximize_successive_view_distance,
    obstacle_scene_rejection_reason,
    relax_closed_loop_candidate_aims,
    select_diverse_smooth_view_path,
    TesseractPlanBridge,
)
from piper_mobile_manipulation.motion_limit_stability import MotionLimitStability


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


def test_first_closed_loop_view_tries_compact_progress_before_ambitious_leader():
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

    assert selected[0]['id'] == 3
    assert selected[1]['id'] == 0


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
    bridge = SimpleNamespace(
        spool=SimpleNamespace(read_health=lambda: {
            'schema_version': 5,
            'generation_id': '1' * 32,
            'written_at_ns': now_ns - 100_000_000,
            'worker_ready': True,
            'backend': 'tesseract',
            'backend_version': '0.35.0.6',
            'backend_error': '',
        }),
        worker_generation_id='',
        get_parameter=lambda name: SimpleNamespace(value={
            'worker_heartbeat_timeout_sec': 1.5,
        }[name]),
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
