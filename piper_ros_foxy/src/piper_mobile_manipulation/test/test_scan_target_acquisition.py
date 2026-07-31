import math
from types import SimpleNamespace

import numpy as np
from rclpy.logging import get_logger

from piper_mobile_manipulation.target_acquisition import (
    ROUGH_ACQUISITION,
    build_acquisition_viewpoints,
    rough_hint_rejection_reason,
    viewpoint_payload_matches,
)
from piper_mobile_manipulation.scan_target_acquisition_node import (
    ScanTargetAcquisitionNode,
)
from piper_mobile_manipulation.scan_execution_modes import (
    acquired_target_rejection,
    correlated_obstacle_scene_status,
    heavy_refresh_status_action,
    uses_bootstrap_static_scene,
)
from piper_mobile_manipulation.heavy_refresh_contract import (
    image_satisfies_request,
)
from piper_mobile_manipulation.viewpoint_reachability_filter_node import (
    ViewpointReachabilityFilterNode,
    target_status_rejection_reason,
)


def vector(value):
    return np.asarray([value['x'], value['y'], value['z']], dtype=float)


def angle_degrees(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dot = float(np.clip(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)),
        -1.0,
        1.0,
    ))
    return math.degrees(math.acos(dot))


def circular_difference_degrees(a, b):
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def test_rough_hint_requires_fresh_finite_base_link_point():
    assert rough_hint_rejection_reason(
        'base_link', [0.4, 0.1, 0.2], 9_500_000_000, 10_000_000_000, 5.0) == ''
    assert rough_hint_rejection_reason(
        'camera_link', [0.4, 0.1, 0.2], 9_500_000_000, 10_000_000_000, 5.0
    ) == 'rough target frame must be base_link'
    assert rough_hint_rejection_reason(
        'base_link', [0.4, math.nan, 0.2],
        9_500_000_000, 10_000_000_000, 5.0
    ) == 'rough target point must contain three finite coordinates'
    assert 'stale' in rough_hint_rejection_reason(
        'base_link', [0.4, 0.1, 0.2], 4_000_000_000, 10_000_000_000, 5.0)
    assert rough_hint_rejection_reason(
        'base_link', [0.4, 0.1, 0.2], 0, 10_000_000_000, 5.0
    ) == 'rough target stamp is missing'
    assert rough_hint_rejection_reason(
        'base_link', [0.4, 0.1, 0.2],
        10_200_000_000, 10_000_000_000, 5.0
    ) == 'rough target stamp is in the future'


def test_acquisition_builds_distinct_bounded_orientation_cone():
    target = np.asarray([0.35, 0.05, 0.20])
    current_camera = np.asarray([0.65, 0.05, 0.35])
    viewpoints = build_acquisition_viewpoints(
        target, current_camera,
        standoff_m=0.45, camera_pitch_deg=-10.0, sweep_angle_deg=15.0)

    assert [item['acquisition_look'] for item in viewpoints] == [
        'center', 'left', 'right', 'up', 'down',
        'left_up', 'right_up', 'left_down', 'right_down']
    positions = [vector(item['desired_camera_position']) for item in viewpoints]
    effective_standoff = np.linalg.norm(current_camera - target)
    assert effective_standoff < 0.45
    assert np.allclose(positions[0], current_camera)
    assert all(np.allclose(position, current_camera) for position in positions)
    assert all(np.isclose(
        item['camera_object_distance_m'], effective_standoff)
        for item in viewpoints)
    assert all(np.isclose(item['maximum_standoff_m'], 0.45)
               for item in viewpoints)
    assert len({tuple(np.round(position, 6)) for position in positions}) == 1

    looks = [vector(item['desired_look_at_direction']) for item in viewpoints]
    assert all(np.isclose(np.linalg.norm(look), 1.0) for look in looks)
    expected_center = (target - current_camera) / np.linalg.norm(
        target - current_camera)
    assert np.allclose(looks[0], expected_center)
    center_azimuth = math.degrees(math.atan2(looks[0][1], looks[0][0]))
    left_azimuth = math.degrees(math.atan2(looks[1][1], looks[1][0]))
    right_azimuth = math.degrees(math.atan2(looks[2][1], looks[2][0]))
    assert np.isclose(
        circular_difference_degrees(left_azimuth, center_azimuth), 15.0)
    assert np.isclose(
        circular_difference_degrees(right_azimuth, center_azimuth), 15.0)
    assert np.isclose(angle_degrees(looks[0], looks[3]), 15.0)
    assert np.isclose(angle_degrees(looks[0], looks[4]), 15.0)
    assert looks[3][2] > looks[0][2]
    assert looks[4][2] < looks[0][2]
    assert viewpoints[0]['keep_object_centered']
    assert not any(item['keep_object_centered'] for item in viewpoints[1:])


def test_default_cone_covers_deliberately_wrong_lateral_cube_hint():
    rough = np.asarray([0.25, 0.0, 0.0])
    actual = np.asarray([0.25, -0.25, 0.0])
    current_camera = np.asarray([0.17354935, 0.03613366, 0.28642369])
    viewpoints = build_acquisition_viewpoints(rough, current_camera)
    actual_direction = actual - current_camera
    actual_direction /= np.linalg.norm(actual_direction)

    look_errors = [
        angle_degrees(
            vector(item['desired_look_at_direction']), actual_direction)
        for item in viewpoints
    ]

    assert min(look_errors) < 5.0
    best = viewpoints[int(np.argmin(look_errors))]
    assert best['acquisition_look'] == 'right_up'
    assert np.isclose(best['acquisition_yaw_offset_deg'], -45.0)
    assert np.isclose(best['acquisition_pitch_offset_deg'], 30.0)


def test_near_base_hint_does_not_push_center_view_behind_robot():
    target = np.asarray([0.38, -0.12, 0.0])
    current_camera = np.asarray([0.095, 0.001, 0.232])
    viewpoints = build_acquisition_viewpoints(
        target, current_camera,
        standoff_m=0.45, camera_pitch_deg=-10.0, sweep_angle_deg=15.0)

    positions = [vector(item['desired_camera_position']) for item in viewpoints]
    base_reaches = [np.linalg.norm(position) for position in positions]
    assert np.allclose(positions[0], current_camera)
    assert base_reaches[0] >= 0.20
    assert sum(reach >= 0.20 for reach in base_reaches) >= 4
    assert all(np.allclose(position, current_camera) for position in positions)


def test_acquisition_checks_the_settled_current_camera_before_the_cone():
    target = np.asarray([0.25, 0.0, 0.0])
    current_camera = np.asarray([0.12, -0.04, 0.27])
    current_look = np.asarray([0.0, -0.8, -0.6])
    viewpoints = build_acquisition_viewpoints(
        target, current_camera,
        standoff_m=0.45, camera_pitch_deg=-10.0, sweep_angle_deg=15.0,
        fallback_standoff_m=0.30,
        current_camera_look_direction=current_look)

    assert len(viewpoints) <= 20
    assert viewpoints[0]['acquisition_look'] == 'current_view'
    assert viewpoints[0]['acquisition_search_stage'] == 'current_camera'
    assert np.allclose(
        vector(viewpoints[0]['desired_camera_position']), current_camera)
    assert np.allclose(
        vector(viewpoints[0]['desired_look_at_direction']),
        current_look / np.linalg.norm(current_look))
    assert not viewpoints[0]['keep_object_centered']
    assert 'center' in [
        item['acquisition_look'] for item in viewpoints[1:]]


def test_centerline_hint_adds_distinct_compact_fallback_candidates():
    target = np.asarray([0.25, 0.0, 0.0])
    current_camera = np.asarray([0.074991, -0.002784, 0.280012])
    viewpoints = build_acquisition_viewpoints(
        target, current_camera,
        standoff_m=0.45, camera_pitch_deg=-10.0, sweep_angle_deg=15.0,
        fallback_standoff_m=0.30)

    assert len(viewpoints) == 20
    assert [item['acquisition_look'] for item in viewpoints[:9]] == [
        'center', 'left', 'right', 'up', 'down',
        'left_up', 'right_up', 'left_down', 'right_down']
    assert all(item['acquisition_search_stage'] == 'orientation_cone'
               for item in viewpoints[:9])
    assert all(item['acquisition_search_stage'] == 'compact_fallback'
               for item in viewpoints[9:])
    assert all(np.isclose(item['camera_object_distance_m'], 0.30)
               for item in viewpoints[9:])
    positions = [
        tuple(np.round(vector(item['desired_camera_position']), 9))
        for item in viewpoints]
    assert len(set(positions[:9])) == 1
    assert len(set(positions[9:])) == len(positions[9:])


def test_compact_fallback_deduplicates_primary_poses_at_same_radius():
    target = np.asarray([0.25, 0.0, 0.0])
    current_camera = np.asarray([
        0.10, 0.0, math.sqrt(0.30 ** 2 - 0.15 ** 2)])
    viewpoints = build_acquisition_viewpoints(
        target, current_camera,
        standoff_m=0.45, camera_pitch_deg=-10.0, sweep_angle_deg=15.0,
        fallback_standoff_m=0.30)

    pose_keys = {
        tuple(np.round(np.concatenate((
            vector(item['desired_camera_position']),
            vector(item['desired_look_at_direction']))), 9))
        for item in viewpoints
    }
    assert len(pose_keys) == len(viewpoints)
    assert 5 < len(viewpoints) <= 20


def test_compact_fallback_standoff_is_bounded_by_primary_maximum():
    with np.testing.assert_raises_regex(
            ValueError, 'fallback standoff.*no greater'):
        build_acquisition_viewpoints(
            [0.25, 0.0, 0.0], [0.075, 0.0, 0.28],
            standoff_m=0.45, fallback_standoff_m=0.46)


def test_only_rough_acquisition_relaxes_lost_target_gate():
    assert target_status_rejection_reason('LOST', 'MULTIVIEW_SCAN') == (
        'target_status=LOST')
    assert target_status_rejection_reason('LOST', ROUGH_ACQUISITION) == ''
    assert target_status_rejection_reason(
        'LOW_CONFIDENCE', ROUGH_ACQUISITION) == 'target_status=LOW_CONFIDENCE'


def test_plan_request_gate_matches_kind_and_exact_rough_hint_stamp():
    acquisition = {
        'dry_run': True,
        'plan_kind': ROUGH_ACQUISITION,
        'rough_hint_stamp_ns': 123,
    }
    assert viewpoint_payload_matches(acquisition, ROUGH_ACQUISITION, 123)
    assert not viewpoint_payload_matches(acquisition, ROUGH_ACQUISITION, 124)
    assert not viewpoint_payload_matches(acquisition, 'MULTIVIEW_SCAN')
    assert viewpoint_payload_matches(
        {'dry_run': True, 'plan_kind': 'MULTIVIEW_SCAN'}, 'MULTIVIEW_SCAN')
    assert not viewpoint_payload_matches(
        {'dry_run': False, 'plan_kind': 'MULTIVIEW_SCAN'}, 'MULTIVIEW_SCAN')


def acquisition_hint(x=0.4):
    return SimpleNamespace(
        header=SimpleNamespace(
            frame_id='base_link',
            stamp=SimpleNamespace(sec=10, nanosec=20)),
        point=SimpleNamespace(x=x, y=0.0, z=0.2),
    )


def test_duplicate_atomic_prepare_request_is_idempotent():
    session_id = 'acq-0123456789abcdef'
    hint = acquisition_hint()
    signature = ScanTargetAcquisitionNode.request_signature(session_id, hint)
    node = SimpleNamespace(
        active_session_id=session_id,
        active_request_signature=signature,
        validate_hint=lambda _hint: (_ for _ in ()).throw(
            AssertionError('an accepted duplicate must not age out')),
        request_signature=ScanTargetAcquisitionNode.request_signature,
    )
    request = SimpleNamespace(session_id=session_id, rough_target=hint)
    response = SimpleNamespace(accepted=False, session_id='', message='')

    result = ScanTargetAcquisitionNode.prepare_cb(node, request, response)

    assert result.accepted
    assert result.session_id == session_id
    assert 'idempotently' in result.message


def test_duplicate_atomic_prepare_accepts_a_refreshed_transport_stamp():
    session_id = 'acq-0123456789abcdef'
    original = acquisition_hint()
    signature = ScanTargetAcquisitionNode.request_signature(
        session_id, original)
    refreshed = acquisition_hint()
    refreshed.header.stamp.sec = 11
    refreshed.header.stamp.nanosec = 30
    node = SimpleNamespace(
        active_session_id=session_id,
        active_request_signature=signature,
        validate_hint=lambda _hint: (_ for _ in ()).throw(
            AssertionError('an accepted duplicate must not age out')),
        request_signature=ScanTargetAcquisitionNode.request_signature,
    )
    request = SimpleNamespace(
        session_id=session_id, rough_target=refreshed)
    response = SimpleNamespace(accepted=False, session_id='', message='')

    result = ScanTargetAcquisitionNode.prepare_cb(node, request, response)

    assert result.accepted
    assert result.session_id == session_id
    assert 'idempotently' in result.message


def test_atomic_session_id_cannot_be_rebound_to_another_target():
    session_id = 'acq-0123456789abcdef'
    original = acquisition_hint()
    node = SimpleNamespace(
        active_session_id=session_id,
        active_request_signature=ScanTargetAcquisitionNode.request_signature(
            session_id, original),
        validate_hint=lambda _hint: '',
        request_signature=ScanTargetAcquisitionNode.request_signature,
    )
    request = SimpleNamespace(
        session_id=session_id, rough_target=acquisition_hint(x=0.5))
    response = SimpleNamespace(accepted=False, session_id='', message='')

    result = ScanTargetAcquisitionNode.prepare_cb(node, request, response)

    assert not result.accepted
    assert 'different rough target' in result.message


def test_rejected_nonqueued_acquisition_call_rearms_handoff_retry():
    messages = []
    bridge = SimpleNamespace(
        acquisition_request_sent=True,
        pending_acquisition_message=object(),
        get_logger=lambda: SimpleNamespace(
            info=messages.append,
            warn=messages.append,
            error=messages.append,
        ),
    )
    future = SimpleNamespace(result=lambda: SimpleNamespace(
        accepted=False,
        message='planning blocked: acquisition_scan data is missing or stale',
    ))

    ScanTargetAcquisitionNode.log_plan_request_result(
        bridge, future, ROUGH_ACQUISITION)

    assert not bridge.acquisition_request_sent
    assert bridge.pending_acquisition_message is not None
    assert messages


def test_accepted_acquisition_call_stops_candidate_republish():
    bridge = SimpleNamespace(
        acquisition_request_sent=True,
        pending_acquisition_message=object(),
        get_logger=lambda: SimpleNamespace(
            info=lambda _message: None,
            warn=lambda _message: None,
            error=lambda _message: None,
        ),
    )
    future = SimpleNamespace(result=lambda: SimpleNamespace(
        accepted=True,
        message='command-free Tesseract planning request queued',
    ))

    ScanTargetAcquisitionNode.log_plan_request_result(
        bridge, future, ROUGH_ACQUISITION)

    assert bridge.acquisition_request_sent
    assert bridge.pending_acquisition_message is None


def test_rejected_then_accepted_request_can_change_log_severity_on_foxy():
    bridge = SimpleNamespace(
        acquisition_request_sent=True,
        pending_acquisition_message=object(),
        get_logger=lambda: get_logger('acquisition_handoff_severity_test'),
    )
    rejected = SimpleNamespace(result=lambda: SimpleNamespace(
        accepted=False,
        message='planning inputs are not fresh yet',
    ))
    accepted = SimpleNamespace(result=lambda: SimpleNamespace(
        accepted=True,
        message='command-free Tesseract planning request queued',
    ))

    ScanTargetAcquisitionNode.log_plan_request_result(
        bridge, rejected, ROUGH_ACQUISITION)
    assert not bridge.acquisition_request_sent

    bridge.acquisition_request_sent = True
    ScanTargetAcquisitionNode.log_plan_request_result(
        bridge, accepted, ROUGH_ACQUISITION)

    assert bridge.acquisition_request_sent
    assert bridge.pending_acquisition_message is None


def test_acquisition_still_applies_non_tracking_reachability_checks():
    fixture = SimpleNamespace(
        target_status='LOST',
        param_bool=lambda _name: True,
        arm_status_reasons=lambda: [],
        valid_vector=ViewpointReachabilityFilterNode.valid_vector,
    )
    assert ViewpointReachabilityFilterNode.reject_reasons(
        fixture, {}, ROUGH_ACQUISITION) == ['missing desired camera position']


def test_grounding_status_is_correlated_and_requires_post_settle_frame():
    minimum = 20_000_000_000
    other = {
        'state': 'published',
        'request_id': 'other',
        'image_stamp': {'sec': 21, 'nanosec': 0},
    }
    assert heavy_refresh_status_action(other, 'wanted', minimum)[0] == 'ignore'

    stale = dict(other)
    stale['request_id'] = 'wanted'
    stale['image_stamp'] = {'sec': 19, 'nanosec': 999999999}
    action, reason, _stamp = heavy_refresh_status_action(
        stale, 'wanted', minimum)
    assert action == 'abort'
    assert 'pre-settle' in reason

    queued = dict(stale)
    queued['state'] = 'queued'
    queued['image_stamp'] = {'sec': 20, 'nanosec': 1}
    assert heavy_refresh_status_action(
        queued, 'wanted', minimum) == ('queued', '', 20_000_000_001)

    missing = dict(queued)
    missing.update({
        'state': 'worker_result_rejected',
        'worker_status': 'target_mask_missing',
    })
    assert heavy_refresh_status_action(
        missing, 'wanted', minimum)[0] == 'not_found'

    missing_clear = dict(missing)
    missing_clear['obstacle_count'] = 0
    assert heavy_refresh_status_action(
        missing_clear, 'wanted', minimum) == (
            'not_found_clear',
            'GroundingDINO did not find the target or any obstacles',
            20_000_000_001,
        )

    missing_with_obstacle = dict(missing)
    missing_with_obstacle['obstacle_count'] = 1
    assert heavy_refresh_status_action(
        missing_with_obstacle, 'wanted', minimum)[0] == 'not_found'

    missing_invalid_count = dict(missing)
    missing_invalid_count['obstacle_count'] = 'unknown'
    assert heavy_refresh_status_action(
        missing_invalid_count, 'wanted', minimum)[0] == 'not_found'


def test_only_first_rough_segment_uses_static_bootstrap_scene():
    assert uses_bootstrap_static_scene(ROUGH_ACQUISITION, 0)
    assert not uses_bootstrap_static_scene(ROUGH_ACQUISITION, 1)
    assert not uses_bootstrap_static_scene('MULTIVIEW_SCAN', 0)


def test_subsequent_acquisition_move_requires_correlated_obstacle_scene():
    scene = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=21, nanosec=0)),
        scene_blocked=False,
        blocking_reason='clear',
        instances=[],
    )
    assert correlated_obstacle_scene_status(
        scene, 10.0, 10.2, 1.0, 20_000_000_000) == ('ready', '')

    old = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=19, nanosec=0)),
        scene_blocked=False,
        blocking_reason='clear',
        instances=[],
    )
    assert correlated_obstacle_scene_status(
        old, 10.0, 10.2, 1.0, 20_000_000_000)[0] == 'waiting'

    valid_obstacle = SimpleNamespace(valid=True)
    blocked_but_projected = SimpleNamespace(
        header=scene.header,
        scene_blocked=True,
        blocking_reason='2:hand',
        instances=[valid_obstacle],
    )
    assert correlated_obstacle_scene_status(
        blocked_but_projected, 10.0, 10.2, 1.0,
        20_000_000_000) == ('ready', '')

    invalid = SimpleNamespace(
        header=scene.header,
        scene_blocked=True,
        blocking_reason='2:transform_unavailable',
        instances=[SimpleNamespace(valid=False)],
    )
    assert correlated_obstacle_scene_status(
        invalid, 10.0, 10.2, 1.0, 20_000_000_000)[0] == 'blocked'


def test_heavy_refresh_waits_for_first_frame_after_settle_stamp():
    request = {'min_image_stamp': {'sec': 20, 'nanosec': 100}}
    assert not image_satisfies_request(
        request, SimpleNamespace(sec=20, nanosec=99))
    assert image_satisfies_request(
        request, SimpleNamespace(sec=20, nanosec=100))


def test_acquired_lock_is_new_measured_stable_and_near_rough_hint():
    stamp = SimpleNamespace(sec=21, nanosec=0)
    target = SimpleNamespace(
        header=SimpleNamespace(frame_id='base_link', stamp=stamp),
        position=SimpleNamespace(x=0.55, y=0.0, z=0.20),
        valid=True,
        stable=True,
    )
    health = SimpleNamespace(
        lifecycle_state='TRACKING',
        prediction_only=False,
        camera_settled=True,
        measurement_age_sec=0.1,
    )
    args = (
        target, health, 'LOCKED',
        11.0, 11.0, 11.0, 10.0, 11.1,
        1.0, 0.75, 20_000_000_000,
        np.asarray([0.30, 0.0, 0.20]), 0.30,
    )
    assert acquired_target_rejection(*args) == ''

    too_far = SimpleNamespace(
        header=target.header,
        position=SimpleNamespace(x=0.61, y=0.0, z=0.20),
        valid=True,
        stable=True,
    )
    assert 'rough hint' in acquired_target_rejection(too_far, *args[1:])
    assert 'not LOCKED' in acquired_target_rejection(
        target, health, 'TRACKING', *args[3:])
    old_target = SimpleNamespace(
        header=SimpleNamespace(
            frame_id='base_link',
            stamp=SimpleNamespace(sec=19, nanosec=0),
        ),
        position=target.position,
        valid=True,
        stable=True,
    )
    assert 'predates' in acquired_target_rejection(old_target, *args[1:])
