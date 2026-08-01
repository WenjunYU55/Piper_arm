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


def rigid_transform_matrix(translation, quaternion_xyzw):
    """Return a finite 4x4 transform matrix for reconstruction metadata."""
    t = np.asarray(translation, dtype=float)
    q = np.asarray(quaternion_xyzw, dtype=float)
    if t.shape != (3,) or q.shape != (4,) or not np.all(np.isfinite(t)) \
            or not np.all(np.isfinite(q)):
        raise ValueError('camera transform must be finite XYZ and XYZW')
    norm = float(np.linalg.norm(q))
    if norm <= 1e-9:
        raise ValueError('camera transform quaternion is zero')
    x, y, z, w = q / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
         2 * (x * z + y * w), t[0]],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
         2 * (y * z - x * w), t[1]],
        [2 * (x * z - y * w), 2 * (y * z + x * w),
         1 - 2 * (x * x + y * y), t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)
