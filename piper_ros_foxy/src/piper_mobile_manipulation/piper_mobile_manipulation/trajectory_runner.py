"""Compatibility facade for authorized trajectory monitoring."""

from piper_mobile_manipulation.execution.trajectory import (
    AuthorizedTrajectory,
    TrajectoryAction,
    TrajectoryDecision,
    TrajectoryRunner,
    joint_progress_error,
    waypoint_motion_action,
)

__all__ = [
    'AuthorizedTrajectory', 'TrajectoryAction', 'TrajectoryDecision',
    'TrajectoryRunner', 'joint_progress_error', 'waypoint_motion_action',
]
