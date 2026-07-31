"""Fail-closed validation for exact PiPER SDK MoveJ target paths."""

import math

import numpy as np


TIMING_POLICY_VERSION = 'sdk_movej_targets_v1'
JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']


def validate_tesseract_point(
        positions,
        velocities,
        accelerations,
        time_from_start_s,
        previous_time_s=None,
):
    """Validate and normalize one complete six-joint Tesseract sample."""
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


def validate_sdk_movej_waypoint_path(
        positions,
        velocities,
        accelerations,
        times,
        command_rate_hz,
        maximum_step_rad=0.025,
):
    """Validate the exact position-only target path without altering it.

    PiPER's SDK MoveJ command accepts a joint-position target and one aggregate
    speed percentage.  It does not accept the per-joint velocity or
    acceleration vectors carried by JointTrajectoryPoint. The path therefore
    contains the bound plan start, one final SDK target, and at most one
    acquisition-only bootstrap target. Derivatives are exactly zero;
    timestamps preserve target order but are not an execution schedule.
    """
    q = np.asarray(positions, dtype=float)
    qd = np.asarray(velocities, dtype=float)
    qdd = np.asarray(accelerations, dtype=float)
    t = np.asarray(times, dtype=float)
    if q.ndim != 2 or q.shape[0] not in (2, 3) or q.shape[1] != 6:
        raise ValueError(
            'trajectory must contain start/final targets and at most one '
            'bootstrap target')
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
    if np.any(np.abs(qd) > 1e-12) or np.any(np.abs(qdd) > 1e-12):
        raise ValueError(
            'SDK MoveJ target derivatives must be zero because the '
            'controller interface accepts positions and aggregate speed only')
    return q.copy(), qd.copy(), qdd.copy(), t.copy()
