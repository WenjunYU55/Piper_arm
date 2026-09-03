"""Regression tests for typed failures and legacy string compatibility."""

from dataclasses import replace

import pytest

from piper_mobile_manipulation.failure_model import (
    as_failure,
    Failure,
    FailureCode,
    FailureTag,
    legacy_failure_adapter,
)
from piper_mobile_manipulation.scan_viewpoint_executor_node import (
    abort_return_home_blocker,
    retryable_rgbd_capture_rejection,
    runtime_gate_action,
    terminal_home_hold_required,
    visual_capture_rejection,
)
from piper_mobile_manipulation.target_scan_mission_node import (
    failure_code_for_reason,
    MissionFailure,
    planning_rejection_allows_current_state_home,
    retryable_plan_approval_rejection,
    runtime_freshness_plan_request_rejection,
    target_drift_requires_replan,
    visual_reacquisition_plan_approval_rejection,
    visual_reacquisition_plan_request_rejection,
)


@pytest.mark.parametrize('detail, expected', (
    ('tracked robot cancelled the task', FailureCode.CANCELLED),
    ('camera vision startup timed out', FailureCode.SENSOR_UNAVAILABLE),
    ('mission deadline expired', FailureCode.DEADLINE_EXPIRED),
    ('capture quality was insufficient',
     FailureCode.INSUFFICIENT_CAPTURE_QUALITY),
    ('occlusion was not cleared', FailureCode.OCCLUSION_NOT_CLEARED),
    ('target lock was not found', FailureCode.TARGET_NOT_FOUND),
    ('TARGET_TOO_LARGE_OR_CLOSE: item is cropped',
     FailureCode.TARGET_TOO_LARGE_OR_CLOSE),
    ('TARGET_SCAN_IMPOSSIBLE: item remains cropped at maximum distance',
     FailureCode.TARGET_SCAN_IMPOSSIBLE),
    ('no reachable IK plan exists', FailureCode.NO_REACHABLE_PLAN),
    ('fresh trajectory validation failed: camera holder/L515 envelope floor '
     'clearance 0.004064m is below 0.005000m',
     FailureCode.NO_REACHABLE_PLAN),
    ('CAN bus feedback is unavailable',
     FailureCode.CONTROL_UNTRUSTWORTHY),
    ('visual replacement budget exhausted', FailureCode.MISSION_FAILED),
))
def test_legacy_public_failure_code_mapping_is_preserved(detail, expected):
    failure = legacy_failure_adapter(detail)

    assert failure.code is expected
    assert failure_code_for_reason(detail) == expected.value


def test_explicit_failure_code_overrides_legacy_detail_classification():
    failure = legacy_failure_adapter(
        'camera wording that would historically imply a sensor failure',
        code=FailureCode.TARGET_NOT_FOUND,
        retryable=False,
    )

    assert failure.code is FailureCode.TARGET_NOT_FOUND
    assert failure.retryable is False


@pytest.mark.parametrize('detail, tag', (
    ('TARGET_FRAMING_RETRY_FARTHER: cropped',
     FailureTag.TARGET_FRAMING_RETRY_FARTHER),
    ('TARGET_FRAMING_TOO_CLOSE: cropped',
     FailureTag.TARGET_FRAMING_TOO_CLOSE),
    ('TARGET_FRAMING_TOO_LARGE: cropped',
     FailureTag.TARGET_FRAMING_TOO_LARGE),
    ('planning failed: TARGET_FRAMING_NO_AIMED_ENDPOINT',
     FailureTag.TARGET_FRAMING_NO_AIMED_ENDPOINT),
))
def test_target_framing_legacy_boundary_adds_typed_tag(detail, tag):
    assert legacy_failure_adapter(detail).has(tag)


@pytest.mark.parametrize('tag, decision', (
    (
        FailureTag.PLAN_APPROVAL_RETRY,
        retryable_plan_approval_rejection,
    ),
    (
        FailureTag.PLAN_APPROVAL_VISUAL_REACQUISITION,
        visual_reacquisition_plan_approval_rejection,
    ),
    (
        FailureTag.PLAN_REQUEST_VISUAL_REACQUISITION,
        visual_reacquisition_plan_request_rejection,
    ),
    (
        FailureTag.RUNTIME_FRESHNESS_GAP,
        runtime_freshness_plan_request_rejection,
    ),
    (FailureTag.TARGET_DRIFT_REPLAN, target_drift_requires_replan),
    (
        FailureTag.PLAN_REJECTION_HOME_ALLOWED,
        planning_rejection_allows_current_state_home,
    ),
    (
        FailureTag.CAPTURE_RETRY_SAME_VIEW,
        retryable_rgbd_capture_rejection,
    ),
    (FailureTag.CAPTURE_REJECT_VIEW, visual_capture_rejection),
))
def test_machine_decisions_do_not_depend_on_detail_wording(tag, decision):
    failure = Failure(
        code=FailureCode.MISSION_FAILED,
        detail='first operator-facing explanation',
        tags=frozenset((tag,)),
    )

    assert decision(failure)
    assert decision(
        failure.with_detail('completely different human wording'))


def test_terminal_home_decision_is_independent_of_reason_wording():
    failure = Failure(
        code=FailureCode.CONTROL_UNTRUSTWORTHY,
        detail='old home wording',
        tags=frozenset((FailureTag.TERMINAL_HOME_REACHED,)),
    )

    assert terminal_home_hold_required('ABORTED', failure)
    assert terminal_home_hold_required(
        'ABORTED', failure.with_detail('home proof wording revised'))
    assert not terminal_home_hold_required('INVALID', failure)


def test_runtime_freshness_decision_uses_tags_for_typed_failures():
    gap = Failure(
        code=FailureCode.SENSOR_UNAVAILABLE,
        detail='legacy freshness wording',
        tags=frozenset((FailureTag.RUNTIME_FRESHNESS_GAP,)),
    )
    fault = Failure(
        code=FailureCode.CONTROL_UNTRUSTWORTHY,
        detail='legacy safety wording',
    )

    assert runtime_gate_action([
        gap.with_detail('camera producer has not caught up'),
    ]) == 'hold_for_refresh'
    assert runtime_gate_action([gap, fault]) == 'abort'


def test_return_home_blocker_is_machine_data_not_reparsed_detail():
    failure = Failure(
        code=FailureCode.CONTROL_UNTRUSTWORTHY,
        detail='old controller wording',
        tags=frozenset((FailureTag.RETURN_HOME_BLOCKED,)),
        blocker='arm status',
        needs_operator=True,
        retryable=False,
    )

    assert abort_return_home_blocker(failure) == 'arm status'
    assert abort_return_home_blocker(
        failure.with_detail('the explanation has been rewritten')) \
        == 'arm status'


@pytest.mark.parametrize('detail', [
    'trajectory waypoint did not reach target',
    'SDK MoveJ waypoint made no measurable joint progress before timeout',
])
def test_motion_execution_failure_does_not_suppress_fresh_home_attempt(detail):
    failure = legacy_failure_adapter(detail)

    assert not failure.has(FailureTag.RETURN_HOME_BLOCKED)
    assert failure.blocker == ''


def test_gui_recovery_blocker_is_machine_data_not_reparsed_detail():
    failure = Failure(
        code=FailureCode.CONTROL_UNTRUSTWORTHY,
        detail='old workspace wording',
        tags=frozenset((FailureTag.GUI_AUTO_RECOVERY_BLOCKED,)),
        recovery_blocker='clear the workspace',
    )

    assert as_failure(failure).recovery_blocker == 'clear the workspace'
    assert as_failure(
        failure.with_detail('operator guidance was rewritten')
    ).recovery_blocker == 'clear the workspace'


def test_mission_exception_carries_typed_failure_and_legacy_attributes():
    typed = Failure(
        code=FailureCode.TARGET_NOT_FOUND,
        detail='target observation was unavailable',
        tags=frozenset((
            FailureTag.PLAN_REQUEST_VISUAL_REACQUISITION,
        )),
        retryable=True,
    )

    error = MissionFailure(typed)

    assert str(error) == typed.detail
    assert error.failure is typed
    assert error.failure_code == 'TARGET_NOT_FOUND'
    assert error.retryable is True
    assert error.needs_operator is False
    assert as_failure(error) is typed


def test_failure_is_immutable_and_with_detail_preserves_machine_fields():
    original = Failure(
        code=FailureCode.NO_REACHABLE_PLAN,
        detail='old detail',
        tags=frozenset((FailureTag.EMPTY_VIEW_FRONTIER,)),
        retryable=False,
        blocker='command publisher',
    )
    changed = original.with_detail('new detail')

    assert changed == replace(original, detail='new detail')
    assert changed.code is original.code
    assert changed.tags is original.tags
    assert changed.retryable is original.retryable
    assert changed.blocker == original.blocker


@pytest.mark.parametrize('detail, tag', (
    (
        'execution blocked: target_status=LOW_CONFIDENCE',
        FailureTag.PLAN_APPROVAL_VISUAL_REACQUISITION,
    ),
    (
        'planning blocked: tracking measurement is stale',
        FailureTag.PLAN_REQUEST_VISUAL_REACQUISITION,
    ),
    (
        'quality_rejected: scan quality is stale',
        FailureTag.CAPTURE_RETRY_SAME_VIEW,
    ),
    (
        'executor is not at an accepted settled capture',
        FailureTag.CAPTURE_RETRY_SAME_VIEW,
    ),
    (
        'executor is not at an accepted settled capture '
        '(cached mode=MULTIVIEW_SCAN state=SETTLING)',
        FailureTag.CAPTURE_RETRY_SAME_VIEW,
    ),
    (
        'quality_rejected: target is clipped',
        FailureTag.CAPTURE_REJECT_VIEW,
    ),
    (
        'proposal cancelled; configured home reached; hold requested',
        FailureTag.TERMINAL_HOME_REACHED,
    ),
))
def test_legacy_adapter_tags_string_only_ros_boundaries(detail, tag):
    assert legacy_failure_adapter(detail).has(tag)


def test_completed_capture_transaction_timeout_is_not_retried_externally():
    failure = legacy_failure_adapter(
        'CAPTURE_EVIDENCE_TIMEOUT: '
        'QUALITY_REJECTED: scan quality is stale')

    assert not failure.has(FailureTag.CAPTURE_RETRY_SAME_VIEW)


def test_ray_shortlist_and_complete_frontier_have_distinct_decisions():
    shortlist = legacy_failure_adapter(
        'RAY_SHORTLIST_EXHAUSTED: TESSERACT_EXHAUSTED: six rays failed')
    frontier = legacy_failure_adapter(
        'RAY_FRONTIER_EXHAUSTED: every prequalified ray was attempted')

    assert shortlist.has(FailureTag.RAY_SHORTLIST_EXHAUSTED)
    assert shortlist.has(FailureTag.TESSERACT_EXHAUSTED)
    assert frontier.has(FailureTag.EMPTY_VIEW_FRONTIER)
    assert frontier.has(FailureTag.SAFE_VIEW_EXHAUSTED)
    assert not frontier.has(FailureTag.RAY_SHORTLIST_EXHAUSTED)
