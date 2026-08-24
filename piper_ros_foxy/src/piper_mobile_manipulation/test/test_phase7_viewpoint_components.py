"""Pure Phase 7 characterization of viewpoint application components."""

from dataclasses import replace

import pytest

from piper_mobile_manipulation.capture_coordinator import (
    CaptureAction,
    CaptureCoordinator,
)
from piper_mobile_manipulation.executor_recovery import (
    RecoveryAction,
    RecoveryContext,
    RecoveryPolicy,
)
from piper_mobile_manipulation.failure_model import (
    as_failure,
    Failure,
    FailureCode,
    FailureTag,
)
from piper_mobile_manipulation.plan_authorizer import (
    PlanAuthorizationRequest,
    PlanAuthorizationStatus,
    PlanAuthorizer,
)
from piper_mobile_manipulation.trajectory_runner import (
    TrajectoryAction,
    TrajectoryRunner,
)


def authorization_request(**changes):
    """Return one valid exact-authorization request with selected changes."""
    request = PlanAuthorizationRequest(
        state='PROPOSAL_READY',
        loaded_plan_id='plan-1',
        requested_plan_id='plan-1',
        confirmation='EXECUTE APPROVED SCAN',
        expected_confirmation='EXECUTE APPROVED SCAN',
        real_motion_enabled=True,
        plan_age_sec=1.0,
        plan_max_age_sec=300.0,
        loaded_trajectory_sha256='a' * 64,
        requested_trajectory_sha256='a' * 64,
    )
    return replace(request, **changes)


def failure(code=FailureCode.MISSION_FAILED, *tags, detail='diagnostic'):
    """Create one typed failure for recovery/capture tests."""
    return Failure(code, detail, frozenset(tags))


def test_valid_plan_is_authorized():
    """Accept a matching, fresh, motion-enabled exact proposal."""
    decision = PlanAuthorizer.evaluate(authorization_request())
    assert decision.permitted
    assert decision.status is PlanAuthorizationStatus.AUTHORIZED
    assert decision.failure is None


def test_stale_plan_is_rejected_without_changing_public_wording():
    """Reject a proposal beyond its established maximum age."""
    decision = PlanAuthorizer.evaluate(authorization_request(
        plan_age_sec=301.0))
    assert not decision.permitted
    assert decision.status is PlanAuthorizationStatus.STALE_PLAN
    assert 'expired' in decision.detail


def test_wrong_mission_authorization_is_rejected_first():
    """Reject mission confirmation without live matching authorization."""
    decision = PlanAuthorizer.evaluate(authorization_request(
        confirmation='MISSION_POLICY:' + 'b' * 64,
        expected_confirmation='MISSION_POLICY:' + 'b' * 64,
        mission_authorization_required=True,
        mission_authorization_granted=False,
    ))
    assert not decision.permitted
    assert decision.status is (
        PlanAuthorizationStatus.WRONG_MISSION_AUTHORIZATION)


def test_target_drift_and_stale_target_are_distinct():
    """Keep unavailable target evidence distinct from excessive drift."""
    stale = PlanAuthorizer.evaluate(authorization_request(
        target_required=True,
        target_available=False,
    ))
    drift = PlanAuthorizer.evaluate(authorization_request(
        target_required=True,
        target_available=True,
        target_drift_m=0.016,
        maximum_target_drift_m=0.015,
    ))
    assert stale.status is PlanAuthorizationStatus.TARGET_STALE
    assert drift.status is PlanAuthorizationStatus.TARGET_DRIFT


def test_planner_failure_is_typed():
    """Represent invalid Tesseract results with a stable status and code."""
    decision = PlanAuthorizer.planner_result(False, 'no IK')
    assert not decision.permitted
    assert decision.status is PlanAuthorizationStatus.PLANNER_FAILURE
    assert decision.failure.code is FailureCode.NO_REACHABLE_PLAN


def test_path_failure_is_rejected_after_exact_identity():
    """Reject a freshly revalidated path without changing its detail."""
    decision = PlanAuthorizer.evaluate(authorization_request(
        path_reasons=('collision path invalid',)))
    assert decision.status is PlanAuthorizationStatus.PATH_INVALID
    assert decision.detail == (
        'fresh trajectory validation failed: collision path invalid')


def test_successful_streaming_trajectory_progression():
    """Wait, publish one due sample, then complete and converge."""
    runner = TrajectoryRunner()
    session = runner.begin(
        'plan-1',
        ((0.0,) * 6, (0.1,) * 6),
        (0.05, 0.10),
        True,
    )
    assert session.streaming
    assert runner.stream_decision(
        0, session.times_sec, 0.01).action is TrajectoryAction.WAIT
    publish = runner.stream_decision(0, session.times_sec, 0.051)
    assert publish.action is TrajectoryAction.PUBLISH
    assert publish.sample_index == 0
    assert runner.stream_decision(
        2, session.times_sec, 0.11).action is TrajectoryAction.COMPLETE
    assert runner.feedback_decision(
        0.01, 0.025, 1.0, 90.0, 1.0, 20.0,
    ).action is TrajectoryAction.ADVANCE


@pytest.mark.parametrize('decision,expected', [
    (TrajectoryRunner.feedback_decision(
        float('nan'), 0.025, 1.0, 90.0, 1.0, 20.0),
     TrajectoryAction.FAILED_INVALID),
    (TrajectoryRunner.feedback_decision(
        0.1, 0.025, 91.0, 90.0, 1.0, 20.0),
     TrajectoryAction.FAILED_TIMEOUT),
    (TrajectoryRunner.feedback_decision(
        0.1, 0.025, 2.0, 90.0, 21.0, 20.0),
     TrajectoryAction.FAILED_STALLED),
    (TrajectoryRunner.following_decision(1.0, 0.31, 1.0, 0.30),
     TrajectoryAction.FAILED_FOLLOWING),
])
def test_failed_trajectory_decisions(decision, expected):
    """Report every monitored trajectory failure as a typed action."""
    assert decision.action is expected


def test_late_stream_tick_preserves_next_unsent_sample_and_reports_delay():
    decision = TrajectoryRunner.stream_decision(
        0, (0.05, 0.10), 0.101)

    assert decision.action is TrajectoryAction.PUBLISH
    assert decision.sample_index == 0
    assert decision.missed_samples == 1
    assert decision.schedule_delay_sec == pytest.approx(0.051)


def test_trajectory_cancellation_is_explicit():
    """Represent cancellation independently from ROS service objects."""
    assert TrajectoryRunner.cancellation_decision(
        True).action is TrajectoryAction.CANCELLED
    assert TrajectoryRunner.cancellation_decision(
        False).action is TrajectoryAction.WAIT


def test_following_corridor_can_hold_before_existing_bounded_failure():
    decision = TrajectoryRunner.following_decision(
        2.0, 0.31, 1.0, 0.30, over_limit_elapsed_sec=0.5)
    assert decision.action is TrajectoryAction.HOLD_FOLLOWING
    expired = TrajectoryRunner.following_decision(
        3.0, 0.31, 1.0, 0.30, over_limit_elapsed_sec=1.01)
    assert expired.action is TrajectoryAction.FAILED_FOLLOWING


def test_capture_success_retry_and_failure():
    """Preserve accepted, bounded retry, replacement, and abort outcomes."""
    coordinator = CaptureCoordinator(maximum_readiness_retries=10)
    retry = failure(
        FailureCode.SENSOR_UNAVAILABLE,
        FailureTag.CAPTURE_RETRY_SAME_VIEW,
    )
    rejected = failure(
        FailureCode.INSUFFICIENT_CAPTURE_QUALITY,
        FailureTag.CAPTURE_REJECT_VIEW,
    )
    refreshable = failure(
        FailureCode.INSUFFICIENT_CAPTURE_QUALITY,
        FailureTag.CAPTURE_REFRESH_SAME_VIEW,
        FailureTag.CAPTURE_REJECT_VIEW,
    )
    fatal = failure(FailureCode.CONTROL_UNTRUSTWORTHY)
    assert coordinator.classify_result(
        True, None, 1).action is CaptureAction.ACCEPT
    assert coordinator.classify_result(
        False, retry, 9).action is CaptureAction.RETRY_SAME_VIEW
    assert coordinator.classify_result(
        False, retry, 10).action is CaptureAction.ABORT
    assert coordinator.classify_result(
        False, refreshable, 1, False).action is (
            CaptureAction.REFRESH_SAME_VIEW)
    assert coordinator.classify_result(
        False, refreshable, 1, True).action is CaptureAction.REPLAN_VIEW
    assert coordinator.classify_result(
        False, rejected, 1).action is CaptureAction.REPLAN_VIEW
    assert coordinator.classify_result(
        False, fatal, 1).action is CaptureAction.ABORT


def test_capture_handoff_preserves_status_propagation_order():
    """Publish settled authority before requesting RGB-D persistence."""
    assert CaptureCoordinator.handoff(
        False, 0.10, 0.25).action is CaptureAction.PUBLISH_AUTHORIZATION
    assert CaptureCoordinator.handoff(
        False, 0.25, 0.25).action is CaptureAction.REQUEST_CAPTURE
    assert CaptureCoordinator.handoff(
        True, 10.0, 0.25).action is CaptureAction.WAIT_RESPONSE


def test_stale_capture_authorization_retries_the_same_settled_view():
    """A delayed status subscription must not abort an otherwise valid view."""
    coordinator = CaptureCoordinator(maximum_readiness_retries=10)
    delayed = as_failure(
        'executor is not at an accepted settled capture')

    assert delayed.has(FailureTag.CAPTURE_RETRY_SAME_VIEW)
    assert coordinator.classify_result(
        False, delayed, 1).action is CaptureAction.RETRY_SAME_VIEW


def test_capture_settle_requires_one_continuous_window():
    """Reset interrupted settling and accept only the complete window."""
    waiting = CaptureCoordinator.settle(
        1.0, False, 0.5, 1.0, 1.5, 15.0)
    start = CaptureCoordinator.settle(
        1.0, True, None, 1.0, 1.5, 15.0)
    ready = CaptureCoordinator.settle(
        2.6, True, 1.0, 2.6, 1.5, 15.0)
    timeout = CaptureCoordinator.settle(
        15.1, True, 1.0, 15.1, 1.5, 15.0)
    assert waiting.reset_settle_window
    assert start.action is CaptureAction.START_SETTLE_WINDOW
    assert ready.action is CaptureAction.READY
    assert timeout.action is CaptureAction.ABORT


@pytest.mark.parametrize('context,typed_failure,expected', [
    (RecoveryContext.RUNTIME,
     failure(FailureCode.SENSOR_UNAVAILABLE,
             FailureTag.RUNTIME_FRESHNESS_GAP),
     RecoveryAction.RETRY),
    (RecoveryContext.PLANNING,
     failure(FailureCode.TARGET_NOT_FOUND,
             FailureTag.PLAN_REQUEST_VISUAL_REACQUISITION),
     RecoveryAction.REACQUIRE),
    (RecoveryContext.PLANNING,
     failure(FailureCode.NO_REACHABLE_PLAN,
             FailureTag.TARGET_DRIFT_REPLAN),
     RecoveryAction.REPLAN),
    (RecoveryContext.CAPTURE,
     failure(FailureCode.SENSOR_UNAVAILABLE,
             FailureTag.CAPTURE_RETRY_SAME_VIEW),
     RecoveryAction.RETRY),
    (RecoveryContext.CAPTURE,
     failure(FailureCode.INSUFFICIENT_CAPTURE_QUALITY,
             FailureTag.CAPTURE_REJECT_VIEW),
     RecoveryAction.REPLAN),
    (RecoveryContext.TRAJECTORY,
     failure(FailureCode.CONTROL_UNTRUSTWORTHY),
     RecoveryAction.ABORT),
    (RecoveryContext.ACQUISITION,
     failure(FailureCode.CANCELLED),
     RecoveryAction.ABORT),
])
def test_recovery_decision_matrix(context, typed_failure, expected):
    """Map typed executor failures to their explicit recovery action."""
    decision = RecoveryPolicy.decide(context, (typed_failure,))
    assert decision.action is expected


def test_recovery_does_not_depend_on_human_wording():
    """Prove recovery is stable when only operator wording changes."""
    original = failure(
        FailureCode.SENSOR_UNAVAILABLE,
        FailureTag.RUNTIME_FRESHNESS_GAP,
        detail='camera data missing or stale',
    )
    rewritten = original.with_detail('wording changed completely')
    assert RecoveryPolicy.decide(
        RecoveryContext.RUNTIME, (original,)).action is RecoveryAction.RETRY
    assert RecoveryPolicy.decide(
        RecoveryContext.RUNTIME, (rewritten,)).action is RecoveryAction.RETRY
