"""Pure helpers for a stationary target landmark."""

import math

import numpy as np


def maximum_pairwise_distance(points):
    values = np.asarray(points, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    differences = values[:, None, :] - values[None, :, :]
    return float(np.max(np.linalg.norm(differences, axis=2)))


def direction_angle_degrees(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return 0.0
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def project_camera_point(point, camera_matrix):
    x, y, z = [float(value) for value in point]
    if not np.isfinite([x, y, z]).all() or z <= 0.0:
        raise ValueError('landmark_behind_camera')
    fx, fy = float(camera_matrix[0]), float(camera_matrix[4])
    cx, cy = float(camera_matrix[2]), float(camera_matrix[5])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError('invalid_camera_intrinsics')
    return np.array([fx * x / z + cx, fy * y / z + cy], dtype=np.float64)
