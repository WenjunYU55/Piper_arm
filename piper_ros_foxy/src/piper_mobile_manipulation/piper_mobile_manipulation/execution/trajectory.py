"""Pure monitoring decisions for one already-authorized trajectory."""

import math
from bisect import bisect_right
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class TrajectoryAction(str, Enum):
    """Next action for the ROS command adapter."""

    WAIT = 'wait'
    PUBLISH = 'publish'
    ADVANCE = 'advance'
    COMPLETE = 'complete'
    CANCELLED = 'cancelled'
    FAILED_INVALID = 'abort_invalid'
    FAILED_TIMEOUT = 'abort_timeout'
    FAILED_STALLED = 'abort_stalled'
    HOLD_FOLLOWING = 'hold_following'
    FAILED_FOLLOWING = 'abort_following'
    FAILED_OVERRUN = 'abort_overrun'


@dataclass(frozen=True)
class AuthorizedTrajectory:
    """Immutable identity and schedule for one authorized segment."""

    plan_id: str
    positions: Tuple[Tuple[float, ...], ...]
    times_sec: Tuple[float, ...]
    streaming: bool


@dataclass(frozen=True)
class TrajectoryDecision:
    """Typed trajectory-monitoring result."""

    action: TrajectoryAction
    sample_index: Optional[int] = None
    missed_samples: int = 0
    schedule_delay_sec: float = 0.0


class TrajectoryRunner:
    """Own pure schedule and feedback decisions; publish no commands."""

    # Resume inside the entry corridor, with independent feedback evidence.
    # These only tighten recovery; the configured entry limit is unchanged.
    FOLLOWING_RESUME_FRACTION = 0.9
    FOLLOWING_RESUME_SAMPLES = 3

    @staticmethod
    def following_recovery_decision(
            error_rad, limit_rad, hold_elapsed_sec, timeout_sec,
            consecutive_samples):
        """Keep a latched hold from chattering or restarting its deadline."""
        if (not math.isfinite(float(error_rad))
                or float(hold_elapsed_sec) > float(timeout_sec)):
            return TrajectoryDecision(TrajectoryAction.FAILED_FOLLOWING)
        if (float(error_rad) <= float(limit_rad)
                * TrajectoryRunner.FOLLOWING_RESUME_FRACTION
                and consecutive_samples >= TrajectoryRunner.FOLLOWING_RESUME_SAMPLES):
            return TrajectoryDecision(TrajectoryAction.PUBLISH)
        return TrajectoryDecision(TrajectoryAction.HOLD_FOLLOWING)

    @staticmethod
    def begin(plan_id, positions, times_sec, streaming):
        """Freeze one already-authorized trajectory for execution."""
        frozen_positions = tuple(
            tuple(float(value) for value in point) for point in positions)
        frozen_times = tuple(float(value) for value in times_sec)
        if not str(plan_id) or not frozen_positions:
            raise ValueError('authorized trajectory identity or path is empty')
        if bool(streaming) and len(frozen_times) != len(frozen_positions):
            raise ValueError(
                'authorized trajectory schedule length differs')
        if any(len(point) != 6 for point in frozen_positions):
            raise ValueError(
                'authorized trajectory points must have six joints')
        return AuthorizedTrajectory(
            str(plan_id), frozen_positions, frozen_times, bool(streaming))

    @staticmethod
    def feedback_decision(
            error_rad, reached_tolerance_rad, waypoint_elapsed_sec,
            waypoint_timeout_sec, progress_elapsed_sec,
            progress_timeout_sec):
        """Preserve the existing endpoint/no-progress watchdog ordering."""
        error = float(error_rad)
        if not math.isfinite(error):
            return TrajectoryDecision(TrajectoryAction.FAILED_INVALID)
        if error <= float(reached_tolerance_rad):
            return TrajectoryDecision(TrajectoryAction.ADVANCE)
        if float(waypoint_elapsed_sec) > float(waypoint_timeout_sec):
            return TrajectoryDecision(TrajectoryAction.FAILED_TIMEOUT)
        if float(progress_elapsed_sec) > float(progress_timeout_sec):
            return TrajectoryDecision(TrajectoryAction.FAILED_STALLED)
        return TrajectoryDecision(TrajectoryAction.WAIT)

    @staticmethod
    def stream_decision(path_index, times_sec, elapsed_sec):
        """Select the next unsent sample without bursting or skipping it."""
        index = int(path_index)
        # The admitted schedule is already numeric and ordered. Avoid copying
        # the entire trajectory on every high-rate control tick.
        schedule = times_sec
        elapsed = float(elapsed_sec)
        if index >= len(schedule):
            return TrajectoryDecision(TrajectoryAction.COMPLETE)
        if elapsed + 1e-6 < schedule[index]:
            return TrajectoryDecision(TrajectoryAction.WAIT)
        due_index = max(index, bisect_right(schedule, elapsed + 1e-6) - 1)
        missed = due_index - index
        return TrajectoryDecision(
            TrajectoryAction.PUBLISH,
            sample_index=index,
            missed_samples=missed,
            schedule_delay_sec=(
                max(0.0, elapsed - schedule[index]) if missed else 0.0),
        )

    @staticmethod
    def following_decision(
            elapsed_sec, following_error_rad, grace_sec, limit_rad,
            over_limit_elapsed_sec=None):
        """Bound command lead while retaining the existing hard failure."""
        error = float(following_error_rad)
        if float(elapsed_sec) < float(grace_sec):
            return TrajectoryDecision(TrajectoryAction.WAIT)
        if not math.isfinite(error):
            return TrajectoryDecision(TrajectoryAction.FAILED_FOLLOWING)
        if error > float(limit_rad):
            if (
                    over_limit_elapsed_sec is not None
                    and float(over_limit_elapsed_sec) <= float(grace_sec)):
                return TrajectoryDecision(TrajectoryAction.HOLD_FOLLOWING)
            return TrajectoryDecision(TrajectoryAction.FAILED_FOLLOWING)
        return TrajectoryDecision(TrajectoryAction.WAIT)

    @staticmethod
    def cancellation_decision(cancelled):
        """Represent cancellation without consulting ROS goal objects."""
        return TrajectoryDecision(
            TrajectoryAction.CANCELLED if bool(cancelled)
            else TrajectoryAction.WAIT)


def waypoint_motion_action(
        error_rad, reached_tolerance_rad, waypoint_elapsed_sec,
        waypoint_timeout_sec, progress_elapsed_sec, progress_timeout_sec):
    """Compatibility wrapper returning the historical string action."""
    return TrajectoryRunner.feedback_decision(
        error_rad,
        reached_tolerance_rad,
        waypoint_elapsed_sec,
        waypoint_timeout_sec,
        progress_elapsed_sec,
        progress_timeout_sec,
    ).action.value


def joint_progress_error(current, target):
    """Measure total remaining motion so progress by any joint is visible."""
    current_values = np.asarray(current, dtype=float)
    target_values = np.asarray(target, dtype=float)
    if (
            current_values.shape != (6,)
            or target_values.shape != (6,)
            or not np.all(np.isfinite(current_values))
            or not np.all(np.isfinite(target_values))):
        return math.inf
    return float(np.sum(np.abs(current_values - target_values)))
