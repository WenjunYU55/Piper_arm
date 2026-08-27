"""Phase 1 characterization of executor safety and capture decisions."""

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from piper_mobile_manipulation.safety_evaluator import (
    ObstacleAuthority,
    runtime_gate_policy,
    SafetyMode,
)
from piper_mobile_manipulation.scan_viewpoint_executor_node import (
    MAX_RGBD_CAPTURE_READINESS_RETRIES,
    ScanViewpointExecutorNode,
    abort_return_home_blocker,
    retryable_rgbd_capture_rejection,
    runtime_gate_action,
    terminal_home_hold_required,
    visual_capture_rejection,
    waypoint_motion_action,
)
from piper_mobile_manipulation.target_envelope import (
    build_capture_model_seed,
    canonical_sha256,
)


def test_executor_isolates_timer_services_clients_and_telemetry_callbacks():
    constructor = inspect.getsource(ScanViewpointExecutorNode.__init__)
    module = inspect.getmodule(ScanViewpointExecutorNode)
    main_source = inspect.getsource(module.main)

    assert 'control_callback_group' in constructor
    assert 'timer_callback_group' in constructor
    assert 'telemetry_callback_group' in constructor
    assert 'client_callback_group' in constructor
    assert 'serialized_control_callback(self.tick)' in constructor
    assert 'callback_group=self.control_callback_group' in constructor
    assert 'callback_group=self.timer_callback_group' in constructor
    assert 'callback_group=self.telemetry_callback_group' in constructor
    assert 'callback_group=self.client_callback_group' in constructor
    assert 'MultiThreadedExecutor' in main_source
    assert 'except KeyboardInterrupt' in main_source


class CompletedFuture:
    """Already-complete ROS future double."""

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    @staticmethod
    def done():
        return True

    def result(self):
        if self._error is not None:
            raise self._error
        return self._result


def capture_model_seed_response():
    shape = {
        'schema_version': 1,
        'valid': True,
        'header': {
            'stamp': {'sec': 12, 'nanosec': 345},
            'frame_id': 'camera_color_optical_frame',
        },
        'silhouette_points_camera_m': [
            [-0.05, -0.05, 0.40], [0.05, -0.05, 0.40],
            [0.05, 0.05, 0.40], [-0.05, 0.05, 0.40],
        ],
    }
    shape['measurement_sha256'] = canonical_sha256(shape)
    seed = build_capture_model_seed(shape, {
        'header': {
            'stamp': {'sec': 12, 'nanosec': 30_000_345},
            'frame_id': 'base_link',
        },
        'child_frame_id': 'camera_color_optical_frame',
        'matrix_4x4': np.eye(4).tolist(),
    })
    return json.dumps({
        'capture_result_schema_version': 1,
        'occlusion_state': 'CLEAR',
        'occlusion_score': 0.0,
        'qualified_target_model_seed': seed,
    }), seed


def _runtime_reasons(
        harness, mode=SafetyMode.SCAN_APPROVAL,
        require_settled=False,
        obstacle_authority=ObstacleAuthority.LIVE):
    policy = runtime_gate_policy(
        mode, require_settled=require_settled,
        obstacle_authority=obstacle_authority)
    return ScanViewpointExecutorNode.runtime_reasons(harness, policy)


@pytest.mark.parametrize(
    'missing_key,expected', [
        ('camera_clock', 'camera_clock data missing or stale'),
        ('joints', 'joints data missing or stale'),
        ('arm_status', 'arm_status data missing or stale'),
        ('tracking', 'tracking data missing or stale'),
        ('target_status', 'target_status data missing or stale'),
        ('obstacles', 'obstacles data missing or stale'),
        ('motion_limits', 'motion_limits data missing or stale'),
    ],
)
def test_missing_runtime_inputs_hold_for_bounded_refresh(
        executor_runtime_harness, missing_key, expected):
    executor_runtime_harness.freshness[missing_key] = False

    reasons = _runtime_reasons(executor_runtime_harness)

    assert expected in reasons
    assert runtime_gate_action(reasons) == 'hold_for_refresh'


def test_fresh_but_unhealthy_camera_clock_enters_bounded_refresh_hold(
        executor_runtime_harness):
    executor_runtime_harness.latest_camera_timestamp_health = SimpleNamespace(
        healthy=False,
        state='CLOCK_OFFSET',
        reason='camera timestamp is in the future',
    )

    reasons = _runtime_reasons(executor_runtime_harness)

    assert reasons == [
        'camera timestamp CLOCK_OFFSET: camera timestamp is in the future']
    assert runtime_gate_action(reasons) == 'hold_for_refresh'


@pytest.mark.parametrize('target_state', [
    'LOW_CONFIDENCE', 'LOST', 'SEARCHING'])
def test_unlocked_target_blocks_new_motion_but_not_issued_segment(
        executor_runtime_harness, target_state):
    executor_runtime_harness.latest_target_status = target_state

    before_motion = _runtime_reasons(
        executor_runtime_harness, mode=SafetyMode.SCAN_APPROVAL,
    )
    issued_segment = _runtime_reasons(
        executor_runtime_harness,
        mode=SafetyMode.SCAN_MOTION,
        obstacle_authority=ObstacleAuthority.APPROVED_SNAPSHOT,
    )

    assert 'target_status=%s' % target_state in before_motion
    assert 'target_status=%s' % target_state not in issued_segment


def test_prediction_only_target_blocks_settled_view_approval(
        executor_runtime_harness):
    health = executor_runtime_harness.latest_tracking_health
    health.prediction_only = True

    reasons = _runtime_reasons(
        executor_runtime_harness,
        require_settled=True,
    )

    assert 'tracking is prediction-only' in reasons


@pytest.mark.parametrize(
    'reason,expected_blocker', [
        ('joint feedback became invalid', 'joint feedback became invalid'),
        ('arm status is missing or stale', 'arm status'),
        ('command publisher is unavailable', 'command publisher'),
        ('motion limits data missing or stale', 'motion limits'),
        ('obstacle collision is present', ''),
        ('camera timestamp health is stale', ''),
        ('target lock was lost', ''),
    ],
)
def test_direct_home_blocker_classification_is_frozen(
        reason, expected_blocker):
    assert abort_return_home_blocker(reason) == expected_blocker


@pytest.mark.parametrize(
    'message,retry_same_pose,reject_view', [
        ('missing target_3d', True, False),
        ('missing detection mask', True, False),
        ('quality_rejected: scan quality is stale', True, False),
        ('occlusion_rejected: occlusion evidence is missing', True, False),
        ('target_3d invalid', False, True),
        ('quality_rejected: score 0.42 is below GOOD', False, True),
        (
            'occlusion_rejected: settled target view is PARTIAL',
            False,
            True,
        ),
        ('timestamped camera transform is malformed', False, False),
    ],
)
def test_capture_rejection_classifier_preserves_retry_scope(
        message, retry_same_pose, reject_view):
    assert retryable_rgbd_capture_rejection(message) is retry_same_pose
    assert visual_capture_rejection(message) is reject_view


def _capture_harness(message, attempts=1):
    events = []
    result = SimpleNamespace(success=False, message=message)
    harness = SimpleNamespace(
        now=lambda: 1.0,
        state_started=0.0,
        get_parameter=lambda name: SimpleNamespace(value={
            'capture_timeout_sec': 20.0,
            'capture_status_propagation_sec': 0.25,
        }[name]),
        rgbd_capture_future=CompletedFuture(result),
        rgbd_capture_attempts=attempts,
        rgbd_capture_client=SimpleNamespace(
            call_async=lambda _request: events.append('rgbd_request')),
        capture_client=SimpleNamespace(
            call_async=lambda _request: events.append('workflow_capture')),
        record_rejected_view=lambda reason: events.append(
            ('record_rejected', reason)),
        command_target=object(),
        current_path=[object()],
        current_path_velocities=[object()],
        current_path_accelerations=[object()],
        current_path_times=[1.0],
        capture_heavy_refresh_request_id='',
        request_capture_heavy_refresh=lambda reason: events.append(
            ('refresh', reason)),
        reject_achieved_capture_view=lambda reason: events.append(
            ('reject_achieved', reason)),
        publish_hold=lambda: events.append('hold') or True,
        publish_status=lambda: events.append('status'),
        set_state=lambda state, reason: events.append(
            ('state', state, reason)),
        abort_motion=lambda reason: events.append(('abort', reason)),
    )
    return harness, events


def test_missing_capture_evidence_retries_same_pose_without_motion():
    harness, events = _capture_harness('missing target_3d')

    ScanViewpointExecutorNode.capturing_rgbd_tick(harness)

    assert harness.rgbd_capture_future is None
    assert events[0][0:2] == ('state', 'CAPTURING_RGBD')
    assert 'hold' not in events
    assert not any(isinstance(item, tuple) and item[0] == 'abort'
                   for item in events)


def test_retry_budget_exhaustion_aborts_instead_of_moving():
    harness, events = _capture_harness(
        'missing target_3d', MAX_RGBD_CAPTURE_READINESS_RETRIES)

    ScanViewpointExecutorNode.capturing_rgbd_tick(harness)

    assert events == [(
        'abort',
        'RGB-D viewpoint capture was rejected: missing target_3d',
    )]


def test_fresh_visual_capture_rejection_requests_one_heavy_refresh():
    harness, events = _capture_harness('target_3d invalid')

    ScanViewpointExecutorNode.capturing_rgbd_tick(harness)

    assert events == [('refresh', 'target_3d invalid')]


def test_persistent_visual_rejection_replans_without_second_refresh():
    harness, events = _capture_harness('target_3d invalid')
    harness.capture_heavy_refresh_request_id = 'already-used'

    ScanViewpointExecutorNode.capturing_rgbd_tick(harness)

    assert events == [('reject_achieved', 'target_3d invalid')]


def test_capture_transport_failure_aborts_without_recording_acceptance():
    harness, events = _capture_harness(
        'timestamped camera transform is malformed')

    ScanViewpointExecutorNode.capturing_rgbd_tick(harness)

    assert events == [(
        'abort',
        'RGB-D viewpoint capture was rejected: '
        'timestamped camera transform is malformed',
    )]


def test_successful_rgbd_capture_advances_to_workflow_acceptance():
    events = []
    harness, _unused = _capture_harness('')
    response, seed = capture_model_seed_response()
    harness.rgbd_capture_future = CompletedFuture(
        SimpleNamespace(success=True, message=response))
    harness.capture_client = SimpleNamespace(
        call_async=lambda _request: events.append('workflow_capture') or
        CompletedFuture())
    harness.set_state = lambda state, reason: events.append(
        ('state', state, reason))

    ScanViewpointExecutorNode.capturing_rgbd_tick(harness)

    assert events[0] == 'workflow_capture'
    assert events[1][0:2] == ('state', 'CAPTURING')
    assert harness.pending_scan_qualified_target_model_seed == seed
    assert harness.pending_scan_qualified_target_shape == seed['shape']
    assert harness.pending_capture_occlusion_state == 'CLEAR'
    assert harness.capture_semantic_probe_pending


def test_first_capture_without_capture_bound_model_seed_aborts():
    harness, events = _capture_harness('')
    harness.rgbd_capture_future = CompletedFuture(
        SimpleNamespace(success=True, message='saved'))

    ScanViewpointExecutorNode.capturing_rgbd_tick(harness)

    assert events == [(
        'abort',
        'accepted RGB-D capture result is malformed: capture response does '
        'not contain model-seed JSON',
    )]


def test_waiting_capture_propagates_occlusion_cancellation_immediately():
    aborted = []
    harness = SimpleNamespace(
        now=lambda: 1.0,
        state_started=0.0,
        get_parameter=lambda _name: SimpleNamespace(value=20.0),
        telemetry_store=None,
        latest_workflow={
            'state': 'ABORTED',
            'reason': 'target is occluded by pen; mission cancelled',
        },
        workflow_ready=lambda: False,
        abort_motion=lambda reason: aborted.append(reason),
    )

    ScanViewpointExecutorNode.wait_capture_tick(harness)

    assert aborted == [
        'workflow rejected first capture: target is occluded by pen; '
        'mission cancelled',
    ]


def test_semantic_probe_wait_uses_extended_timeout_for_any_capture():
    aborted = []
    parameters = {
        'capture_timeout_sec': 20.0,
        'first_capture_acceptance_timeout_sec': 75.0,
    }
    harness = SimpleNamespace(
        now=lambda: 74.0,
        state_started=0.0,
        scan_history=[],
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        telemetry_store=None,
        latest_workflow={'state': 'INITIALIZING', 'accepted_views': 0},
        abort_motion=aborted.append,
    )

    ScanViewpointExecutorNode.wait_capture_tick(harness)
    assert aborted == []

    harness.now = lambda: 76.0
    ScanViewpointExecutorNode.wait_capture_tick(harness)
    assert aborted == [
        'capture semantic acceptance did not return workflow to '
        'SCAN_READY',
    ]

    aborted.clear()
    harness.scan_history = [{}]
    harness.capture_semantic_probe_pending = False
    harness.now = lambda: 21.0
    ScanViewpointExecutorNode.wait_capture_tick(harness)
    assert aborted == [
        'accepted capture did not return workflow to SCAN_READY',
    ]

    aborted.clear()
    harness.capture_semantic_probe_pending = True
    harness.now = lambda: 74.0
    ScanViewpointExecutorNode.wait_capture_tick(harness)
    assert aborted == []


def _achieved_view_harness(look_angle_deg=0.0):
    angle = np.deg2rad(float(look_angle_deg))
    transform = np.eye(4)
    transform[:3, 2] = [np.sin(angle), 0.0, np.cos(angle)]
    transform[:3, 3] = [0.1, 0.0, 0.2]
    events = []
    return SimpleNamespace(
        plan_kind='MULTIVIEW_SCAN',
        plan_id='plan-a',
        current_view=0,
        plan_viewpoints=[{
            'index': 4,
            'desired_camera_position': {'x': 0.1, 'y': 0.0, 'z': 0.2},
            'desired_look_at_direction': {'x': 0.0, 'y': 0.0, 'z': 1.0},
        }],
        plan_target_center=np.asarray([0.1, 0.0, 0.6]),
        latest_tracked_target=None,
        fresh=lambda _key: False,
        get_parameter=lambda name: SimpleNamespace(value={
            'max_target_drift_before_approval_m': 0.015,
        }[name]),
        kinematics=SimpleNamespace(
            camera_transform=lambda _joints: transform),
        current_joints=lambda: np.asarray([0.1] * 6),
        now=lambda: 12.5,
        latest_achieved_scan_view=None,
        publish_scan_history=lambda: events.append('history'),
        abort_motion=lambda reason: events.append(('abort', reason)),
    ), events


def test_settled_fk_is_recorded_even_before_capture_acceptance():
    harness, events = _achieved_view_harness()

    assert ScanViewpointExecutorNode.record_latest_achieved_scan_view(harness)

    achieved = harness.latest_achieved_scan_view
    assert achieved['viewpoint_index'] == 4
    assert achieved['camera_position'] == pytest.approx(
        {'x': 0.1, 'y': 0.0, 'z': 0.2})
    assert achieved['look_direction'] == pytest.approx(
        {'x': 0.0, 'y': 0.0, 'z': 1.0})
    assert achieved['joint_positions_rad'] == pytest.approx([0.1] * 6)
    assert events == ['history']


def test_final_capture_aim_accepts_exact_and_rejects_more_than_five_degrees():
    exact, _events = _achieved_view_harness(0.0)
    assert ScanViewpointExecutorNode.record_latest_achieved_scan_view(exact)
    assert ScanViewpointExecutorNode.final_capture_aim_rejection(exact) == ''

    off_axis, _events = _achieved_view_harness(6.0)
    assert ScanViewpointExecutorNode.record_latest_achieved_scan_view(off_axis)
    rejection = ScanViewpointExecutorNode.final_capture_aim_rejection(
        off_axis)
    assert rejection.startswith('FINAL_AIM_EXCEEDED:')
    assert off_axis.latest_achieved_scan_view[
        'final_aim_error_deg'] == pytest.approx(6.0)


def test_post_motion_target_drift_rejects_capture_and_requests_closed_loop_replan():
    harness, _events = _achieved_view_harness(0.0)
    harness.latest_tracked_target = SimpleNamespace(
        valid=True,
        header=SimpleNamespace(frame_id='base_link'),
        position=SimpleNamespace(x=0.13, y=0.0, z=0.6),
    )
    harness.fresh = lambda key: key == 'tracked_target'
    assert ScanViewpointExecutorNode.record_latest_achieved_scan_view(harness)

    rejection = ScanViewpointExecutorNode.final_capture_aim_rejection(harness)

    assert rejection.startswith('TARGET_DRIFT_REPLAN:')
    assert harness.latest_achieved_scan_view[
        'target_estimate_drift_m'] == pytest.approx(0.03)


def test_frozen_ray_capture_aim_uses_plan_target_not_tracker_jitter():
    harness, _events = _achieved_view_harness(0.0)
    harness.plan_viewpoints[0]['ray_id'] = 7
    harness.latest_tracked_target = SimpleNamespace(
        valid=True,
        header=SimpleNamespace(frame_id='base_link'),
        position=SimpleNamespace(x=0.16, y=0.0, z=0.6),
    )
    harness.fresh = lambda key: key == 'tracked_target'
    assert ScanViewpointExecutorNode.record_latest_achieved_scan_view(harness)

    assert ScanViewpointExecutorNode.final_capture_aim_rejection(harness) == ''
    assert harness.latest_achieved_scan_view['ray_id'] == 7


@pytest.mark.parametrize(
    'error,elapsed,progress,expected', [
        (0.005, 1.0, 1.0, 'advance'),
        (0.020, 1.0, 1.0, 'wait'),
        (0.020, 91.0, 1.0, 'abort_timeout'),
        (0.020, 1.0, 21.0, 'abort_stalled'),
        (float('nan'), 1.0, 1.0, 'abort_invalid'),
    ],
)
def test_trajectory_feedback_terminal_decisions_are_characterized(
        error, elapsed, progress, expected):
    assert waypoint_motion_action(
        error,
        reached_tolerance_rad=0.012,
        waypoint_elapsed_sec=elapsed,
        waypoint_timeout_sec=90.0,
        progress_elapsed_sec=progress,
        progress_timeout_sec=20.0,
    ) == expected


@pytest.mark.parametrize('state', [
    'WAITING_FOR_RUNTIME_REFRESH',
    'MOVING',
    'SETTLING',
    'CAPTURING',
    'CAPTURING_RGBD',
    'WAIT_CAPTURE',
    'WAITING_FOR_CAPTURE_REFRESH',
])
def test_cancel_during_executor_active_states_holds_then_aborts(state):
    events = []
    harness = SimpleNamespace(
        state=state,
        reason='active',
        publish_hold=lambda: events.append('hold') or True,
        _terminal_abort=lambda reason: events.append(('abort', reason)),
    )
    response = SimpleNamespace(success=False, message='')

    ScanViewpointExecutorNode.cancel_cb(harness, None, response)

    assert events[0] == 'hold'
    assert events[1][0] == 'abort'
    assert 'dedicated current-state return-home replanning' in events[1][1]
    assert response.success is True
    assert 'current joint hold requested' in response.message


def test_cancel_after_configured_home_only_refreshes_hold():
    events = []
    harness = SimpleNamespace(
        state='ABORTED',
        reason='operator cancelled; configured home reached',
        publish_hold=lambda: events.append('hold') or True,
        _terminal_abort=lambda reason: events.append(('abort', reason)),
    )
    response = SimpleNamespace(success=False, message='')

    ScanViewpointExecutorNode.cancel_cb(harness, None, response)

    assert terminal_home_hold_required(harness.state, harness.reason)
    assert events == ['hold']
    assert response.success is True
    assert 'configured home reached' in response.message
