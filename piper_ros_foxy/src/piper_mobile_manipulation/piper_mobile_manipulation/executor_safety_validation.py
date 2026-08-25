"""Pure safety and path-validation classifications for the executor."""

from piper_mobile_manipulation.failure_model import as_failure, FailureTag
from piper_mobile_manipulation.scan_execution_modes import (
    MULTIVIEW_SCAN,
    ROUGH_ACQUISITION,
    uses_bootstrap_static_scene,
)


def approved_return_home_obstacle_snapshot(
        returning_home, collision_model_qualified, obstacles):
    """Keep the approval-time scene for the exact planned home segment."""
    return bool(
        returning_home
        and collision_model_qualified
        and obstacles is not None
    )


def approved_multiview_motion_obstacle_snapshot(
        plan_kind, state, collision_model_qualified, obstacles):
    """Keep the approval scene only while one exact scan target moves."""
    return bool(
        plan_kind == MULTIVIEW_SCAN
        and state == 'MOVING'
        and collision_model_qualified
        and obstacles is not None
    )


def bootstrap_abort_retrace_uses_static_scene(
        plan_kind, viewpoint_index, collision_model_qualified):
    """Mirror the approval scene for the exact first acquisition retrace."""
    return bool(
        collision_model_qualified
        and uses_bootstrap_static_scene(plan_kind, viewpoint_index)
    )


def obstacle_scene_runtime_reasons(scene):
    """Classify temporary transform gaps separately from unsafe geometry."""
    instances = list(getattr(scene, 'instances', []))
    if bool(getattr(scene, 'scene_blocked', True)) and not instances:
        return ['scene_blocked: %s' % str(
            getattr(scene, 'blocking_reason', 'unknown reason'))]
    invalid = [item for item in instances if not bool(
        getattr(item, 'valid', False))]
    if not invalid:
        return []
    validity_failures = [
        as_failure(getattr(item, 'validity_reason', ''))
        for item in invalid
    ]
    if validity_failures and all(
            failure.has(FailureTag.OBSTACLE_TRANSFORM_TRANSIENT)
            for failure in validity_failures):
        return ['obstacles data missing or stale']
    return ['invalid obstacle geometry is present']


def approved_retrace_validation_reasons(reasons):
    """Suppress only duplicate static self-clearance on approved retraces."""
    return [
        str(reason) for reason in reasons
        if not as_failure(reason).has(
            FailureTag.SELF_COLLISION_CLEARANCE_DUPLICATE)
    ]


def missing_obstacles_can_wait(
        plan_kind, viewpoint_index, state,
        bootstrap_abort_retrace=False):
    """Let stationary phases reach the bounded pre-motion refresh."""
    if bool(bootstrap_abort_retrace):
        return True
    if (
            plan_kind == ROUGH_ACQUISITION
            and (
                uses_bootstrap_static_scene(plan_kind, viewpoint_index)
                or state == 'WAITING_FOR_OBSTACLE_SCENE')):
        return True
    return (
        plan_kind == MULTIVIEW_SCAN
        and state in (
            'SETTLING', 'CAPTURING', 'CAPTURING_RGBD', 'WAIT_CAPTURE',
            'WAITING_FOR_CAPTURE_REFRESH'))
