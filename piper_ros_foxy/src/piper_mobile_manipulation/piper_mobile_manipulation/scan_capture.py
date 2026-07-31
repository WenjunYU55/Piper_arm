"""Pure validation and encoding helpers for event-triggered RGB-D scan capture."""

import math

import numpy as np


def stamp_seconds(message):
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def synchronized_bundle_rejection(
        color, depth, camera_info, received_at, now_monotonic,
        maximum_age_sec, synchronization_slop_sec):
    if color is None:
        return 'missing RGB image'
    if depth is None:
        return 'missing depth image'
    if camera_info is None:
        return 'missing camera_info'
    values = [stamp_seconds(item) for item in (color, depth, camera_info)]
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        return 'RGB-D bundle has an invalid timestamp'
    if max(values) - min(values) > max(0.0, float(synchronization_slop_sec)):
        return 'RGB-D bundle timestamps are not synchronized'
    if received_at is None:
        return 'RGB-D bundle receipt time is unavailable'
    age = float(now_monotonic) - float(received_at)
    if not math.isfinite(age) or age < 0.0 or age > float(maximum_age_sec):
        return 'RGB-D bundle is stale'
    return ''


def depth_millimetres(depth, encoding):
    """Return a lossless uint16 depth image in millimetres for normal L515 encodings."""
    values = np.asarray(depth)
    if values.ndim != 2:
        raise ValueError('depth image must be two-dimensional')
    name = str(encoding).upper()
    if name in ('16UC1', 'MONO16') or np.issubdtype(values.dtype, np.integer):
        millimetres = values.astype(np.float64)
    else:
        millimetres = values.astype(np.float64) * 1000.0
    millimetres[~np.isfinite(millimetres)] = 0.0
    return np.clip(np.rint(millimetres), 0, 65535).astype(np.uint16)
