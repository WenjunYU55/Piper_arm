"""Fail-closed validation for scheduled planner-produced PiPER paths."""

import math

import numpy as np


EXECUTION_TIMING_POLICY_VERSION = 'timed_stream_v1'
LEGACY_TESSERACT_TIMING_POLICY_VERSION = 'tesseract_stream_v3'
# Source-compatible legacy name. New production code uses the neutral value.
TIMING_POLICY_VERSION = LEGACY_TESSERACT_TIMING_POLICY_VERSION
MOVEJ_NOMINAL_VELOCITY_RAD_S = np.asarray(
    [5.0, 5.0, 5.0, 5.0, 5.0, 3.0], dtype=float)
JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']


def validate_planner_point(
        positions,
        velocities,
        accelerations,
        time_from_start_s,
        previous_time_s=None,
):
    """Validate and normalize one complete six-joint planner sample."""
    vectors = []
    for label, values in (
            ('positions', positions),
            ('velocities', velocities),
            ('accelerations', accelerations)):
        vector = np.asarray(values, dtype=float)
        if vector.shape != (6,) or not np.all(np.isfinite(vector)):
            raise ValueError('%s must contain six finite values' % label)
        vectors.append(vector)
    when = float(time_from_start_s)
    if not math.isfinite(when) or when < 0.0:
        raise ValueError('timestamp must be finite and nonnegative')
    if previous_time_s is not None and when <= float(previous_time_s):
        raise ValueError('timestamps must be strictly increasing')
    return vectors[0], vectors[1], vectors[2], when


def validate_timed_execution_path(
        positions,
        velocities,
        accelerations,
        times,
        command_rate_hz,
        maximum_step_rad=0.05,
        velocity_limits_rad_s=None,
        acceleration_limits_rad_s2=None,
        speed_percent=100.0,
):
    """
    Validate an exact, time-followed position path without altering it.

    The PiPER boundary still consumes position targets plus one aggregate
    speed percentage, so qdot/qddot remain zero transport placeholders.  In
    this policy the timestamps are an execution schedule and every adjacent
    position is a collision-qualified planner sample. Direct configured-home transactions
    remain a separately scoped two-point exception in the caller.
    """
    q = np.asarray(positions, dtype=float)
    qd = np.asarray(velocities, dtype=float)
    qdd = np.asarray(accelerations, dtype=float)
    t = np.asarray(times, dtype=float)
    if q.ndim != 2 or q.shape[0] < 2 or q.shape[1] != 6:
        raise ValueError('trajectory must contain at least two six-joint points')
    if qd.shape != q.shape or qdd.shape != q.shape:
        raise ValueError(
            'trajectory derivatives must match the position matrix')
    if (
            not np.all(np.isfinite(q))
            or not np.all(np.isfinite(qd))
            or not np.all(np.isfinite(qdd))):
        raise ValueError('trajectory contains non-finite joint values')
    if t.shape != (q.shape[0],) or not np.all(np.isfinite(t)):
        raise ValueError(
            'trajectory times must contain one finite value per position')
    if abs(float(t[0])) > 1e-9 or np.any(np.diff(t) <= 0.0):
        raise ValueError(
            'trajectory times must start at zero and strictly increase')
    rate = float(command_rate_hz)
    maximum_step = float(maximum_step_rad)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError('command_rate_hz must be finite and positive')
    if not math.isfinite(maximum_step) or maximum_step <= 0.0:
        raise ValueError('maximum_step_rad must be finite and positive')
    period = 1.0 / rate
    if np.any(np.diff(t) < period - 1e-6):
        raise ValueError(
            'trajectory asks for commands faster than the declared rate')
    if np.any(np.abs(np.diff(q, axis=0)) > maximum_step + 1e-9):
        raise ValueError(
            'trajectory contains a joint step larger than maximum_step_rad')
    if np.any(np.abs(qd) > 1e-12) or np.any(np.abs(qdd) > 1e-12):
        raise ValueError(
            'SDK MoveJ target derivatives must be zero because the '
            'controller interface accepts positions and aggregate speed only')
    if (
            velocity_limits_rad_s is not None
            or acceleration_limits_rad_s2 is not None):
        velocity_limits = np.asarray(velocity_limits_rad_s, dtype=float)
        acceleration_limits = np.asarray(
            acceleration_limits_rad_s2, dtype=float)
        speed = float(speed_percent)
        if (
                velocity_limits.shape != (6,)
                or acceleration_limits.shape != (6,)
                or not np.all(np.isfinite(velocity_limits))
                or not np.all(np.isfinite(acceleration_limits))
                or np.any(velocity_limits <= 0.0)
                or np.any(acceleration_limits <= 0.0)):
            raise ValueError(
                'controller velocity and acceleration limits must contain '
                'six finite positive values')
        if not math.isfinite(speed) or speed < 1.0 or speed > 100.0:
            raise ValueError('speed_percent must be within 1..100')
        scale = speed / 100.0
        # The aggregate speed field is a percentage of the PiPER MoveJ model;
        # live limit feedback remains plan health evidence and is not another
        # percentage multiplier on this position-target schedule.
        scheduled_velocity_limits = MOVEJ_NOMINAL_VELOCITY_RAD_S * scale
        intervals = np.diff(t)
        interval_velocities = np.diff(q, axis=0) / intervals[:, None]
        if np.any(
                np.abs(interval_velocities)
                > scheduled_velocity_limits[None, :] + 1e-6):
            raise ValueError(
                'trajectory exceeds a speed-scaled MoveJ model velocity limit')
    return q.copy(), qd.copy(), qdd.copy(), t.copy()


# Kept as a source-compatible import for downstream packages while the policy
# string prevents an old endpoint-only proposal from entering this executor.
validate_sdk_movej_waypoint_path = validate_timed_execution_path
validate_tesseract_point = validate_planner_point
validate_timed_tesseract_path = validate_timed_execution_path
