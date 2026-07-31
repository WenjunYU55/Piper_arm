import math
from types import SimpleNamespace

import pytest

from piper_tesseract_foxy.bridge_node import (
    maximize_successive_view_distance,
    obstacle_scene_rejection_reason,
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
