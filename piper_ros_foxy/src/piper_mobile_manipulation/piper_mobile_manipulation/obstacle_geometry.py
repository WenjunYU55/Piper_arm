"""Pure geometry and classification helpers for obstacle instance projection."""

import math

import cv2
import numpy as np


BLOCKED = 0
MOVABLE = 1
UNSAFE = 2


def normalize_label(label):
    return ' '.join(str(label or '').strip().lower().replace('_', ' ').split())


def canonical_label(label):
    """Map detector label variants onto the deliberately small permission vocabulary."""
    normalized = normalize_label(label)
    words = set(normalized.replace('(', ' ').replace(')', ' ').split())
    if 'pen' in words or 'marker' in words:
        return 'pen'
    return normalized


def obstacle_records(records):
    """Return obstacle metadata keyed by ID and redundant explained depth IDs."""
    obstacles = [item for item in records if item.get('role') == 'obstacle']
    semantic_labels = {
        canonical_label(item.get('label'))
        for item in obstacles
        if not normalize_label(item.get('label')).startswith('depth foreground (')
    }
    suppressed = {
        int(item.get('object_id', 0))
        for item in obstacles
        if normalize_label(item.get('label')).startswith('depth foreground (')
        and canonical_label(item.get('label')) in semantic_labels
    }
    return {
        int(item.get('object_id', 0)): item
        for item in obstacles
        if int(item.get('object_id', 0)) not in suppressed
    }, suppressed


def effective_classification(label, upstream_unsafe, whitelist):
    normalized = canonical_label(label)
    allowed = {canonical_label(item) for item in whitelist}
    if upstream_unsafe or normalized in ('', 'unknown'):
        return UNSAFE
    if normalized in allowed:
        return MOVABLE
    return BLOCKED


def project_instance(mask, depth_m, camera_matrix, config):
    """Return centroid, robust AABB, valid ratio and count, or raise ValueError."""
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != depth_m.shape:
        raise ValueError('mask_depth_shape_mismatch')
    mask_pixels = int(np.count_nonzero(binary))
    if not mask_pixels:
        raise ValueError('empty_instance_mask')
    erode_px = int(config.get('mask_erode_px', 0))
    if erode_px > 0:
        size = erode_px * 2 + 1
        binary = cv2.erode(binary.astype(np.uint8), np.ones((size, size), np.uint8)) > 0
        if not np.any(binary):
            raise ValueError('mask_empty_after_erosion')
    valid = binary & np.isfinite(depth_m)
    valid &= depth_m >= float(config['depth_min_m'])
    valid &= depth_m <= float(config['depth_max_m'])
    count = int(np.count_nonzero(valid))
    ratio = float(count) / float(mask_pixels)
    if count < int(config['min_valid_depth_pixels']):
        raise ValueError('insufficient_valid_depth_pixels')
    if ratio < float(config['min_valid_depth_ratio']):
        raise ValueError('insufficient_valid_depth_ratio')
    v, u = np.nonzero(valid)
    z = depth_m[valid].astype(np.float64)
    fx, fy = float(camera_matrix[0]), float(camera_matrix[4])
    cx, cy = float(camera_matrix[2]), float(camera_matrix[5])
    if not all(math.isfinite(x) for x in (fx, fy, cx, cy)) or fx <= 0.0 or fy <= 0.0:
        raise ValueError('invalid_camera_intrinsics')
    points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    if not np.all(np.isfinite(points)):
        raise ValueError('non_finite_camera_geometry')
    low = float(config['bounds_low_percentile'])
    high = float(config['bounds_high_percentile'])
    centroid = np.median(points, axis=0)
    bounds_min = np.percentile(points, low, axis=0)
    bounds_max = np.percentile(points, high, axis=0)
    return centroid, bounds_min, bounds_max, ratio, count


def transform_points(points, translation, quaternion):
    qx, qy, qz, qw = [float(value) for value in quaternion]
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError('invalid_transform_quaternion')
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    rotation = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ])
    return np.asarray(points, dtype=np.float64).dot(rotation.T) + np.asarray(translation)


def aabb_corners(bounds_min, bounds_max):
    return np.array([
        [x, y, z]
        for x in (bounds_min[0], bounds_max[0])
        for y in (bounds_min[1], bounds_max[1])
        for z in (bounds_min[2], bounds_max[2])
    ], dtype=np.float64)
