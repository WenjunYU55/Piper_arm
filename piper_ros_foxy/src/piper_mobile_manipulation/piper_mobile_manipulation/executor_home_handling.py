"""Pure return-home and endpoint-settling decisions."""

import numpy as np

from piper_mobile_manipulation.failure_model import as_failure, FailureTag


def terminal_home_hold_required(state, reason):
    """Recognize an abort that has completed its bounded home retrace."""
    return bool(
        str(state) == 'ABORTED'
        and as_failure(reason).has(FailureTag.TERMINAL_HOME_REACHED)
    )


def home_position_sample_settled(
        current, target, previous, target_tolerance, motion_tolerance):
    """Prove a home sample from position, independent of noisy SDK speed."""
    current_values = np.asarray(current, dtype=float)
    target_values = np.asarray(target, dtype=float)
    if (
            current_values.shape != (6,)
            or target_values.shape != (6,)
            or not np.all(np.isfinite(current_values))
            or not np.all(np.isfinite(target_values))):
        return False
    if float(np.max(np.abs(current_values - target_values))) > float(
            target_tolerance):
        return False
    if previous is None:
        return False
    previous_values = np.asarray(previous, dtype=float)
    if (
            previous_values.shape != (6,)
            or not np.all(np.isfinite(previous_values))):
        return False
    return float(np.max(np.abs(
        current_values - previous_values))) <= float(motion_tolerance)


def abort_return_home_blocker(reason):
    """Block direct home only when command/feedback authority is untrusted."""
    return as_failure(reason).blocker
