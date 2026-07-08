"""Small confidence-gated point-to-point ICP implementation."""

import numpy as np
from scipy.spatial import cKDTree


def rigid_transform(source, target):
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T.dot(target - target_center)
    u, _singular, vt = np.linalg.svd(covariance)
    rotation = vt.T.dot(u.T)
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T.dot(u.T)
    translation = target_center - rotation.dot(source_center)
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def transform_points(points, transform):
    return points.dot(transform[:3, :3].T) + transform[:3, 3]


def voxel_downsample(points, voxel_size):
    if not len(points):
        return np.empty((0, 3), dtype=float)
    keys = np.floor(np.asarray(points) / float(voxel_size)).astype(np.int64)
    _, indexes = np.unique(keys, axis=0, return_index=True)
    return np.asarray(points)[np.sort(indexes)]


def register(source, target, initial=None, max_correspondence_m=0.015,
             max_iterations=40, voxel_size_m=0.004):
    source = voxel_downsample(np.asarray(source, dtype=float), voxel_size_m)
    target = voxel_downsample(np.asarray(target, dtype=float), voxel_size_m)
    transform = np.eye(4) if initial is None else np.asarray(initial, dtype=float).copy()
    if len(source) < 20 or len(target) < 20:
        return transform, {'accepted': False, 'fitness': 0.0, 'rmse_m': float('inf'),
                           'reason': 'insufficient points'}
    tree = cKDTree(target)
    previous_rmse = float('inf')
    for _iteration in range(int(max_iterations)):
        moved = transform_points(source, transform)
        distances, indexes = tree.query(moved, k=1)
        keep = distances <= float(max_correspondence_m)
        if np.count_nonzero(keep) < 20:
            break
        delta = rigid_transform(moved[keep], target[indexes[keep]])
        transform = delta.dot(transform)
        rmse = float(np.sqrt(np.mean(np.square(distances[keep]))))
        if abs(previous_rmse - rmse) < 1e-5:
            break
        previous_rmse = rmse
    moved = transform_points(source, transform)
    distances, _indexes = tree.query(moved, k=1)
    keep = distances <= float(max_correspondence_m)
    fitness = float(np.count_nonzero(keep)) / float(max(len(source), 1))
    rmse = float(np.sqrt(np.mean(np.square(distances[keep])))) if np.any(keep) else float('inf')
    return transform, {'accepted': True, 'fitness': fitness, 'rmse_m': rmse, 'reason': 'ok'}
