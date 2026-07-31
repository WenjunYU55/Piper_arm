"""Depth-supported mask reprojection for a moving eye-in-hand camera."""

import cv2
import numpy as np


def backproject_mask(mask, depth_m, camera_matrix, stride=2):
    mask = np.asarray(mask) > 0
    depth_m = np.asarray(depth_m, dtype=float)
    if mask.shape != depth_m.shape:
        raise ValueError('mask and depth dimensions differ')
    valid = mask & np.isfinite(depth_m) & (depth_m > 0.0)
    rows, cols = np.nonzero(valid)
    stride = max(1, int(stride))
    rows, cols = rows[::stride], cols[::stride]
    if not rows.size:
        return np.empty((0, 3), dtype=float)
    k = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    z = depth_m[rows, cols]
    x = (cols.astype(float) - k[0, 2]) * z / k[0, 0]
    y = (rows.astype(float) - k[1, 2]) * z / k[1, 1]
    return np.column_stack((x, y, z))


def transform_points(points, transform):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    transform = np.asarray(transform, dtype=float).reshape(4, 4)
    if not points.size:
        return points.copy()
    homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    return (homogeneous @ transform.T)[:, :3]


def project_points(points_camera, camera_matrix, image_shape):
    points = np.asarray(points_camera, dtype=float).reshape(-1, 3)
    k = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    height, width = image_shape[:2]
    if not points.size:
        return np.empty((0, 2), dtype=np.int32)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-4)
    points = points[valid]
    if not points.size:
        return np.empty((0, 2), dtype=np.int32)
    u = k[0, 0] * points[:, 0] / points[:, 2] + k[0, 2]
    v = k[1, 1] * points[:, 1] / points[:, 2] + k[1, 2]
    uv = np.rint(np.column_stack((u, v))).astype(np.int32)
    inside = (
        (uv[:, 0] >= 0) & (uv[:, 0] < width)
        & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    return uv[inside]


def rasterize_prompt(uv, image_shape, dilation_px=5):
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    uv = np.asarray(uv, dtype=np.int32).reshape(-1, 2)
    if uv.shape[0] < 3:
        return mask
    hull = cv2.convexHull(uv.reshape(-1, 1, 2))
    cv2.fillConvexPoly(mask, hull, 255)
    dilation_px = max(0, int(dilation_px))
    if dilation_px:
        size = 2 * dilation_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.dilate(mask, kernel)
    return mask


def quaternion_matrix(x, y, z, w):
    quaternion = np.asarray([x, y, z, w], dtype=float)
    norm = float(np.dot(quaternion, quaternion))
    if norm < 1e-12:
        return np.eye(3)
    x, y, z, w = quaternion / np.sqrt(norm)
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def transform_matrix(transform):
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_matrix(
        rotation.x, rotation.y, rotation.z, rotation.w
    )
    matrix[:3, 3] = [translation.x, translation.y, translation.z]
    return matrix
