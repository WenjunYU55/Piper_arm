"""Compatibility facade for execution authorization policy."""

from piper_mobile_manipulation.execution.authorization import (
    PlanAuthorizationDecision,
    PlanAuthorizationRequest,
    PlanAuthorizationStatus,
    PlanAuthorizer,
    configured_home_endpoint_rejection,
    direct_home_stage_rejection,
    direct_home_stage_targets,
    target_drift_before_approval_rejection,
    trajectory_count_rejection,
)

__all__ = [
    'PlanAuthorizationDecision', 'PlanAuthorizationRequest',
    'PlanAuthorizationStatus', 'PlanAuthorizer',
    'configured_home_endpoint_rejection', 'direct_home_stage_rejection',
    'direct_home_stage_targets', 'target_drift_before_approval_rejection',
    'trajectory_count_rejection',
]
