"""Phase 1 characterization of executor safety and capture decisions."""

from types import SimpleNamespace

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


def test_fresh_visual_capture_rejection_records_pose_and_holds():
    harness, events = _capture_harness('target_3d invalid')

    ScanViewpointExecutorNode.capturing_rgbd_tick(harness)

    assert events[0] == ('record_rejected', 'target_3d invalid')
    assert events[1] == 'hold'
    assert events[2][0:2] == ('state', 'VIEW_REJECTED')
    assert harness.command_target is None
    assert harness.current_path == []


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
    harness.rgbd_capture_future = CompletedFuture(
        SimpleNamespace(success=True, message='saved'))
    harness.capture_client = SimpleNamespace(
        call_async=lambda _request: events.append('workflow_capture') or
        CompletedFuture())
    harness.set_state = lambda state, reason: events.append(
        ('state', state, reason))

    ScanViewpointExecutorNode.capturing_rgbd_tick(harness)

    assert events[0] == 'workflow_capture'
    assert events[1][0:2] == ('state', 'CAPTURING')


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
