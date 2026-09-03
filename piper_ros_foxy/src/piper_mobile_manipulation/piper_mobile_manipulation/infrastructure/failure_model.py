"""Typed failure information with one explicit legacy-text boundary."""

from dataclasses import dataclass, replace
from enum import Enum
from typing import FrozenSet, Optional, Union


class FailureCode(str, Enum):
    """Stable machine-readable failure codes exposed to mission clients."""

    CANCELLED = 'CANCELLED'
    SENSOR_UNAVAILABLE = 'SENSOR_UNAVAILABLE'
    DEADLINE_EXPIRED = 'DEADLINE_EXPIRED'
    INSUFFICIENT_CAPTURE_QUALITY = 'INSUFFICIENT_CAPTURE_QUALITY'
    OCCLUSION_NOT_CLEARED = 'OCCLUSION_NOT_CLEARED'
    TARGET_NOT_FOUND = 'TARGET_NOT_FOUND'
    TARGET_TOO_FAR = 'TARGET_TOO_FAR'
    TARGET_TOO_LARGE_OR_CLOSE = 'TARGET_TOO_LARGE_OR_CLOSE'
    TARGET_SCAN_IMPOSSIBLE = 'TARGET_SCAN_IMPOSSIBLE'
    NO_REACHABLE_PLAN = 'NO_REACHABLE_PLAN'
    CONTROL_UNTRUSTWORTHY = 'CONTROL_UNTRUSTWORTHY'
    MISSION_FAILED = 'MISSION_FAILED'


class FailureTag(str, Enum):
    """Internal decisions that are finer grained than the public code."""

    PLAN_APPROVAL_RETRY = 'PLAN_APPROVAL_RETRY'
    PLAN_APPROVAL_VISUAL_REACQUISITION = (
        'PLAN_APPROVAL_VISUAL_REACQUISITION')
    PLAN_REQUEST_VISUAL_REACQUISITION = (
        'PLAN_REQUEST_VISUAL_REACQUISITION')
    TARGET_DRIFT_REPLAN = 'TARGET_DRIFT_REPLAN'
    SAFE_VIEW_EXHAUSTED = 'SAFE_VIEW_EXHAUSTED'
    EMPTY_VIEW_FRONTIER = 'EMPTY_VIEW_FRONTIER'
    PLAN_REJECTION_HOME_ALLOWED = 'PLAN_REJECTION_HOME_ALLOWED'
    CAPTURE_RETRY_SAME_VIEW = 'CAPTURE_RETRY_SAME_VIEW'
    CAPTURE_REFRESH_SAME_VIEW = 'CAPTURE_REFRESH_SAME_VIEW'
    CAPTURE_REJECT_VIEW = 'CAPTURE_REJECT_VIEW'
    NO_POSITIVE_INFORMATION = 'NO_POSITIVE_INFORMATION'
    TESSERACT_EXHAUSTED = 'TESSERACT_EXHAUSTED'
    RAY_SHORTLIST_EXHAUSTED = 'RAY_SHORTLIST_EXHAUSTED'
    FINAL_AIM_EXCEEDED = 'FINAL_AIM_EXCEEDED'
    TERMINAL_HOME_REACHED = 'TERMINAL_HOME_REACHED'
    HOME_REACHED = 'HOME_REACHED'
    REQUEST_ALREADY_PENDING = 'REQUEST_ALREADY_PENDING'
    WORKFLOW_ALREADY_ACTIVE = 'WORKFLOW_ALREADY_ACTIVE'
    HOLD_ACKNOWLEDGED = 'HOLD_ACKNOWLEDGED'
    HOLD_REQUESTED = 'HOLD_REQUESTED'
    RUNTIME_FRESHNESS_GAP = 'RUNTIME_FRESHNESS_GAP'
    OBSTACLE_TRANSFORM_TRANSIENT = 'OBSTACLE_TRANSFORM_TRANSIENT'
    RETURN_HOME_BLOCKED = 'RETURN_HOME_BLOCKED'
    SELF_COLLISION_CLEARANCE_DUPLICATE = (
        'SELF_COLLISION_CLEARANCE_DUPLICATE')
    GUI_RETURN_HOME_RETRY = 'GUI_RETURN_HOME_RETRY'
    GUI_AUTO_RECOVERY_BLOCKED = 'GUI_AUTO_RECOVERY_BLOCKED'
    TARGET_FRAMING_RETRY_FARTHER = 'TARGET_FRAMING_RETRY_FARTHER'
    TARGET_FRAMING_TOO_CLOSE = 'TARGET_FRAMING_TOO_CLOSE'
    TARGET_FRAMING_TOO_LARGE = 'TARGET_FRAMING_TOO_LARGE'
    TARGET_FRAMING_NO_AIMED_ENDPOINT = (
        'TARGET_FRAMING_NO_AIMED_ENDPOINT')


@dataclass(frozen=True)
class Failure:
    """Carry machine decisions separately from operator-facing detail."""

    code: FailureCode
    detail: str
    tags: FrozenSet[FailureTag] = frozenset()
    retryable: bool = True
    needs_operator: bool = False
    outcome: str = 'FAILED'
    blocker: str = ''
    recovery_blocker: str = ''

    def has(self, tag: FailureTag) -> bool:
        """Return whether this failure carries an internal decision tag."""
        return tag in self.tags

    def with_detail(self, detail: str) -> 'Failure':
        """Change presentation wording without changing machine decisions."""
        return replace(self, detail=str(detail))


FailureLike = Union[Failure, BaseException, str]


def _failure_code_from_legacy_detail(detail: str) -> FailureCode:
    """Preserve the Phase 1 public failure-code mapping exactly."""
    lowered = detail.lower()
    if 'target_scan_impossible' in lowered:
        return FailureCode.TARGET_SCAN_IMPOSSIBLE
    if 'target_too_large_or_close' in lowered:
        return FailureCode.TARGET_TOO_LARGE_OR_CLOSE
    if 'cancel' in lowered:
        return FailureCode.CANCELLED
    if (
            'trajectory validation failed' in lowered
            or 'envelope floor clearance' in lowered
            or 'envelope intersects obstacle' in lowered):
        return FailureCode.NO_REACHABLE_PLAN
    if (
            'camera' in lowered
            or 'vision' in lowered
            or 'sensor' in lowered):
        return FailureCode.SENSOR_UNAVAILABLE
    if 'deadline' in lowered or 'timed out' in lowered:
        return FailureCode.DEADLINE_EXPIRED
    if 'capture' in lowered or 'quality' in lowered:
        return FailureCode.INSUFFICIENT_CAPTURE_QUALITY
    if (
            'occlud' in lowered
            or 'occlusion' in lowered
            or 'manipulation' in lowered):
        return FailureCode.OCCLUSION_NOT_CLEARED
    if 'target' in lowered and (
            'not found' in lowered or 'lock' in lowered):
        return FailureCode.TARGET_NOT_FOUND
    if (
            'plan' in lowered
            or 'reachable' in lowered
            or 'ik' in lowered
            or 'scan candidate' in lowered
            or 'view frontier' in lowered):
        return FailureCode.NO_REACHABLE_PLAN
    if any(term in lowered for term in (
            'joint feedback', 'collision', 'hold', 'disable',
            'control', 'arm status', 'trajectory waypoint',
            'joint progress')) or any(
                token == 'can'
                for token in lowered.replace(':', ' ').split()):
        return FailureCode.CONTROL_UNTRUSTWORTHY
    return FailureCode.MISSION_FAILED


def _normalized_code(value: Optional[Union[FailureCode, str]],
                     detail: str) -> FailureCode:
    if isinstance(value, FailureCode):
        return value
    if value:
        try:
            return FailureCode(str(value))
        except ValueError:
            pass
    return _failure_code_from_legacy_detail(detail)


def legacy_failure_adapter(
        detail: object, *,
        code: Optional[Union[FailureCode, str]] = None,
        retryable: Optional[bool] = None,
        needs_operator: bool = False,
        outcome: str = 'FAILED') -> Failure:
    """
    Translate a string-only legacy boundary into typed failure data.

    This is the only production function allowed to infer behavior from old
    human-readable ROS service/status/exception text. Producers will migrate
    to typed information in later phases without changing the public wire
    messages retained here.
    """
    text = str(detail)
    lowered = text.lower().strip()
    tags = set()

    execution_blocked = lowered.startswith('execution blocked:')
    plan_retry_markers = (
        'tracking is not settled tracking',
        'tracking is prediction-only',
        'tracking speed scale is below the motion threshold',
        'camera timestamp health is stale',
        'joint feedback is not settled',
        'target_status=low_confidence',
        'target_status=lost',
        'target_status=searching',
    )
    visual_status_markers = (
        'target_status=low_confidence',
        'target_status=lost',
        'target_status=searching',
    )
    if execution_blocked and any(
            marker in lowered for marker in plan_retry_markers):
        tags.add(FailureTag.PLAN_APPROVAL_RETRY)
    if execution_blocked and any(
            marker in lowered for marker in visual_status_markers):
        tags.add(FailureTag.PLAN_APPROVAL_VISUAL_REACQUISITION)
    if lowered.startswith('planning blocked:') and any(
            marker in lowered for marker in (
                'tracking is not settled tracking',
                'tracking is prediction-only',
                'tracking measurement is stale',
                'target_status=low_confidence',
                'target_status=lost',
                'target_status=searching')):
        tags.add(FailureTag.PLAN_REQUEST_VISUAL_REACQUISITION)
    if (
            lowered.startswith('target moved ')
            and ' after planning; refresh the plan' in lowered):
        tags.add(FailureTag.TARGET_DRIFT_REPLAN)
    if lowered.startswith('target_drift_replan:'):
        tags.add(FailureTag.TARGET_DRIFT_REPLAN)
        tags.add(FailureTag.CAPTURE_REJECT_VIEW)

    framing_tags = (
        ('target_framing_retry_farther:',
         FailureTag.TARGET_FRAMING_RETRY_FARTHER),
        ('target_framing_too_close:', FailureTag.TARGET_FRAMING_TOO_CLOSE),
        ('target_framing_too_large:', FailureTag.TARGET_FRAMING_TOO_LARGE),
        ('target_framing_no_aimed_endpoint',
         FailureTag.TARGET_FRAMING_NO_AIMED_ENDPOINT),
    )
    for marker, tag in framing_tags:
        if marker in lowered:
            tags.add(tag)

    empty_view_frontier = (
        'only 0 viewpoints planned; require at least 1 of 1' in lowered)
    if empty_view_frontier:
        tags.add(FailureTag.EMPTY_VIEW_FRONTIER)
    if (
            'multiview_scan planning failed:' in lowered
            and empty_view_frontier
            and 'no finite bounded collision-free ik goal for any roll'
            in lowered):
        tags.add(FailureTag.SAFE_VIEW_EXHAUSTED)
    if 'no_positive_information_candidate' in lowered:
        tags.add(FailureTag.NO_POSITIVE_INFORMATION)
        tags.add(FailureTag.EMPTY_VIEW_FRONTIER)
        tags.add(FailureTag.SAFE_VIEW_EXHAUSTED)
    if (
            'tesseract_exhausted' in lowered
            or 'planner_exhausted' in lowered):
        tags.add(FailureTag.TESSERACT_EXHAUSTED)
        tags.add(FailureTag.SAFE_VIEW_EXHAUSTED)
    if 'ray_shortlist_exhausted' in lowered:
        tags.add(FailureTag.RAY_SHORTLIST_EXHAUSTED)
    if 'ray_frontier_exhausted' in lowered:
        tags.add(FailureTag.EMPTY_VIEW_FRONTIER)
        tags.add(FailureTag.SAFE_VIEW_EXHAUSTED)
    if (
            (
                'planning failed: tesseract proposal rejected:' in lowered
                and 'planning_failed:' in lowered)
            or (
                'planning failed: motion-planner proposal rejected:' in lowered
                and 'planning_failed:' in lowered)
            or 'runtime safety gate: invalid obstacle geometry is present'
            in lowered
            or (
                'fresh runtime telemetry did not arrive' in lowered
                and 'obstacles data missing or stale' in lowered)):
        tags.add(FailureTag.PLAN_REJECTION_HOME_ALLOWED)

    # A recorder-owned transaction has already spent its complete bounded
    # wait.  Preserve the underlying reason for diagnostics, but never feed
    # that terminal timeout back into the historical external retry loop.
    capture_transaction_expired = lowered.startswith(
        'capture_evidence_timeout:')
    capture_retry = not capture_transaction_expired and (
        (
            'timestamped camera transform is unavailable' in lowered
            and 'extrapolation into the future' in lowered)
        or lowered == 'missing scan execution status'
        or lowered.startswith('scan execution is not multiview_scan')
        or lowered.startswith(
            'executor is not at an accepted settled capture')
        or 'quality_rejected: scan quality is missing' in lowered
        or 'quality_rejected: scan quality is stale' in lowered
        or 'occlusion_rejected: occlusion evidence is missing' in lowered
        or 'occlusion_rejected: occlusion evidence is stale' in lowered
        or lowered == 'missing target_3d'
        or lowered == 'missing detection mask'
        or 'confidence-qualified rgb-d bundle is still catching up' in lowered
        or 'confidence-qualified native depth is still catching up' in lowered
        or 'confidence-qualified rgb-d bundle is stale' in lowered
        or 'confidence-qualified native depth bundle is stale' in lowered
        or 'confidence-qualified rgb-d bundle timestamps are not synchronized'
        in lowered
        or (
            'confidence-qualified native depth bundle timestamps are not '
            'synchronized' in lowered)
        or 'confidence-qualified rgb-d bundle has an invalid timestamp'
        in lowered
        or 'confidence-qualified native depth bundle has an invalid timestamp'
        in lowered
        or 'confidence-qualified rgb-d bundle receipt time is unavailable'
        in lowered
        or (
            'confidence-qualified native depth bundle receipt time is '
            'unavailable' in lowered)
        or lowered == (
            'native depth and confidence timestamps are not synchronized')
        or lowered == 'rgb and native depth timestamps are not synchronized')
    if capture_retry:
        tags.add(FailureTag.CAPTURE_RETRY_SAME_VIEW)
    refreshable_visual_rejection = (
        lowered.startswith('quality_rejected:')
        or lowered == 'target_3d invalid'
        or lowered.startswith('depth_quality_rejected:')
    )
    if not capture_retry and refreshable_visual_rejection:
        tags.add(FailureTag.CAPTURE_REFRESH_SAME_VIEW)
    if not capture_retry and (
            refreshable_visual_rejection
            or lowered.startswith(
                'occlusion_rejected: settled target view is ')):
        tags.add(FailureTag.CAPTURE_REJECT_VIEW)
    if lowered.startswith('final_aim_exceeded:'):
        tags.add(FailureTag.FINAL_AIM_EXCEEDED)
        tags.add(FailureTag.CAPTURE_REJECT_VIEW)

    if 'home reached' in lowered:
        tags.add(FailureTag.HOME_REACHED)
    if 'configured home reached' in lowered:
        tags.add(FailureTag.TERMINAL_HOME_REACHED)
    if 'already pending' in lowered:
        tags.add(FailureTag.REQUEST_ALREADY_PENDING)
    if 'already active' in lowered:
        tags.add(FailureTag.WORKFLOW_ALREADY_ACTIVE)
    if 'hold' in lowered:
        tags.add(FailureTag.HOLD_ACKNOWLEDGED)
    if 'hold requested' in lowered:
        tags.add(FailureTag.HOLD_REQUESTED)
    if (
            'current joint hold' in lowered
            or 'fresh return-home safety gate failed' in lowered):
        tags.add(FailureTag.GUI_RETURN_HOME_RETRY)
    if (
            lowered.endswith((
                'data missing or stale',
                'data is missing or stale',
            ))
            or lowered.endswith(
                'controller motion limits are missing or stale')
            or lowered.startswith('camera timestamp ')):
        tags.add(FailureTag.RUNTIME_FRESHNESS_GAP)
    if lowered.startswith((
            'transform_unavailable:',
            'stale_transform',
            'stale_source_data')):
        tags.add(FailureTag.OBSTACLE_TRANSFORM_TRANSIENT)
    if 'self-collision clearance between link segments' in lowered:
        tags.add(FailureTag.SELF_COLLISION_CLEARANCE_DUPLICATE)

    blockers = (
        'emergency stop',
        'joint feedback became invalid',
        'outside configured limits',
        'motion limits',
        'arm status',
        'arm is not enabled',
        'err_code',
        'command publisher',
    )
    blocker = next((item for item in blockers if item in lowered), '')
    if blocker:
        tags.add(FailureTag.RETURN_HOME_BLOCKED)

    recovery_blockers = (
        'movable clutter',
        'clear the workspace',
        'obstacle scene is blocked',
        'obstacle geometry is invalid',
        'incompatible active state',
        'command publisher',
        'collision model is not qualified',
        'outside configured limits',
        'arm is not enabled',
        'managed scan stack failed',
        'managed automation process stopped',
        'operator cancelled',
        'emergency stop',
    )
    recovery_blocker = next((
        item for item in recovery_blockers if item in lowered), '')
    if recovery_blocker:
        tags.add(FailureTag.GUI_AUTO_RECOVERY_BLOCKED)

    operator = bool(needs_operator)
    return Failure(
        code=_normalized_code(code, text),
        detail=text,
        tags=frozenset(tags),
        retryable=(not operator if retryable is None else bool(retryable)),
        needs_operator=operator,
        outcome=str(outcome),
        blocker=blocker,
        recovery_blocker=recovery_blocker,
    )


def as_failure(value: FailureLike) -> Failure:
    """Return typed failure data, adapting only untyped legacy values."""
    if isinstance(value, Failure):
        return value
    failure = getattr(value, 'failure', None)
    if isinstance(failure, Failure):
        return failure
    return legacy_failure_adapter(value)
