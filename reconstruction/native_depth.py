"""Replay confidence-qualified target geometry on the native L515 depth grid.

Capture stores both the contiguous native depth image and its sparse,
z-buffered projection into the colour image.  This module correlates those
immutable artifacts in reverse so reconstruction can use the original depth
sampling without expanding the accepted target support.
"""

import numpy as np


GEOMETRY_SOURCES = ('projected_color_depth', 'native_depth')
DEPTH_CORRELATION_TOLERANCE_MM = 1


def _camera_matrix(values, name):
    matrix = np.asarray(values, dtype=float)
    if matrix.size != 9:
        raise ValueError('%s must contain nine values' % name)
    matrix = matrix.reshape(3, 3)
    if not np.all(np.isfinite(matrix)) or matrix[0, 0] <= 0.0 \
            or matrix[1, 1] <= 0.0:
        raise ValueError('%s is not a finite pinhole matrix' % name)
    return matrix


def _homogeneous_matrix(value, name):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)) \
            or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError('%s is not a finite homogeneous transform' % name)
    return matrix


def _distort_normalized(x, y, distortion, model):
    name = str(model or '').lower()
    if name not in ('', 'none', 'plumb_bob'):
        raise ValueError('unsupported colour distortion model: %s' % model)
    coefficients = np.zeros(5, dtype=float)
    supplied = np.asarray(distortion, dtype=float).reshape(-1)
    if supplied.size:
        if supplied.size < 4 or not np.all(np.isfinite(supplied)):
            raise ValueError('colour distortion coefficients are invalid')
        coefficients[:min(5, supplied.size)] = supplied[:5]
    k1, k2, p1, p2, k3 = coefficients
    radius2 = x * x + y * y
    radial = 1.0 + k1 * radius2 + k2 * radius2 ** 2 + k3 * radius2 ** 3
    return (
        x * radial + 2.0 * p1 * x * y
        + p2 * (radius2 + 2.0 * x * x),
        y * radial + p1 * (radius2 + 2.0 * y * y)
        + 2.0 * p2 * x * y,
    )


def project_native_depth(depth_mm, depth_k, color_k, color_distortion,
                         color_distortion_model, color_from_depth,
                         color_shape):
    """Project valid native-depth samples into the raw colour image."""
    depth = np.asarray(depth_mm)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError('native depth must be a two-dimensional uint16 image')
    kd = _camera_matrix(depth_k, 'native depth camera matrix')
    kc = _camera_matrix(color_k, 'colour camera matrix')
    transform = _homogeneous_matrix(
        color_from_depth, 'depth-to-colour transform')
    height, width = [int(value) for value in color_shape]
    if height <= 0 or width <= 0:
        raise ValueError('colour image shape is invalid')
    rows, columns = np.indices(depth.shape)
    z = depth.astype(np.float64) / 1000.0
    valid = z > 0.0
    x = (columns.astype(np.float64) - kd[0, 2]) * z / kd[0, 0]
    y = (rows.astype(np.float64) - kd[1, 2]) * z / kd[1, 1]
    points = np.stack((x, y, z), axis=-1)
    color_points = points @ transform[:3, :3].T + transform[:3, 3]
    color_z = color_points[..., 2]
    valid &= np.isfinite(color_z) & (color_z > 0.0)
    normalized_x = np.zeros_like(color_z)
    normalized_y = np.zeros_like(color_z)
    normalized_x[valid] = color_points[..., 0][valid] / color_z[valid]
    normalized_y[valid] = color_points[..., 1][valid] / color_z[valid]
    distorted_x, distorted_y = _distort_normalized(
        normalized_x, normalized_y, color_distortion,
        color_distortion_model)
    u_float = kc[0, 0] * distorted_x + kc[0, 2]
    v_float = kc[1, 1] * distorted_y + kc[1, 2]
    u = np.rint(u_float).astype(np.int32)
    v = np.rint(v_float).astype(np.int32)
    valid &= np.isfinite(u_float) & np.isfinite(v_float) \
        & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return {
        'valid': valid,
        'u': u,
        'v': v,
        'color_depth_m': color_z,
    }


def replay_native_target_geometry(
        metadata, native_depth_mm, confidence, color_target_depth_mm,
        color_target_support, raw_color_bgr):
    """Recover one accepted native-grid sample for every projected winner."""
    depth = np.asarray(native_depth_mm)
    grades = np.asarray(confidence)
    color_depth = np.asarray(color_target_depth_mm)
    color_support = np.asarray(color_target_support) > 0
    color = np.asarray(raw_color_bgr)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError('native depth artifact is invalid')
    if grades.shape != depth.shape or not np.issubdtype(
            grades.dtype, np.integer):
        raise ValueError('native confidence artifact is invalid')
    if color_depth.ndim != 2 or color_depth.dtype != np.uint16 \
            or color_support.shape != color_depth.shape:
        raise ValueError('projected target depth/support is invalid')
    if color.shape != color_depth.shape + (3,) or color.dtype != np.uint8:
        raise ValueError('raw colour artifact is invalid')

    depth_info = metadata.get('native_depth_camera_info', {})
    color_info = metadata.get('camera_info', {})
    color_from_depth_record = metadata.get('color_from_depth_transform', {})
    base_from_color_record = metadata.get('camera_transform', {})
    if depth_info.get('available') is not True \
            or color_info.get('available') is not True \
            or color_from_depth_record.get('available') is not True \
            or base_from_color_record.get('available') is not True:
        raise ValueError('native-depth replay calibration is unavailable')
    depth_distortion = np.asarray(depth_info.get('d', []), dtype=float)
    if depth_distortion.size and (
            not np.all(np.isfinite(depth_distortion))
            or np.max(np.abs(depth_distortion)) > 1e-9):
        raise ValueError(
            'native depth distortion is nonzero and cannot be integrated '
            'without rectification')
    depth_k = _camera_matrix(
        depth_info.get('k', []), 'native depth camera matrix')
    color_k = _camera_matrix(
        color_info.get('k', []), 'colour camera matrix')
    color_from_depth = _homogeneous_matrix(
        color_from_depth_record.get('matrix_4x4'),
        'depth-to-colour transform')
    base_from_color = _homogeneous_matrix(
        base_from_color_record.get('matrix_4x4'),
        'base-from-colour transform')
    projection = project_native_depth(
        depth, depth_k, color_k, color_info.get('d', []),
        color_info.get('distortion_model', ''), color_from_depth,
        color_depth.shape)

    threshold = int(metadata.get(
        'confidence_quality', {}).get('confidence_threshold', -1))
    if threshold < 8 or threshold > 15:
        raise ValueError('native confidence threshold is not qualified')
    rows, columns = np.nonzero(projection['valid'])
    u = projection['u'][rows, columns]
    v = projection['v'][rows, columns]
    projected_depth = np.rint(
        projection['color_depth_m'][rows, columns] * 1000.0).astype(np.int64)
    recorded_depth = color_depth[v, u].astype(np.int64)
    difference = np.abs(recorded_depth - projected_depth)
    candidate = color_support[v, u] \
        & (recorded_depth > 0) \
        & (difference <= DEPTH_CORRELATION_TOLERANCE_MM) \
        & (grades[rows, columns] >= threshold)
    rows = rows[candidate]
    columns = columns[candidate]
    u = u[candidate]
    v = v[candidate]
    difference = difference[candidate]
    projected_depth = projected_depth[candidate]
    if not rows.size:
        raise ValueError(
            'native-depth replay found no correlated target samples')

    width = color_depth.shape[1]
    linear = v.astype(np.int64) * width + u.astype(np.int64)
    order = np.lexsort((projected_depth, difference, linear))
    _unique, first = np.unique(linear[order], return_index=True)
    winners = order[first]
    rows = rows[winners]
    columns = columns[winners]
    u = u[winners]
    v = v[winners]
    recovered_linear = linear[winners]
    expected_linear = np.flatnonzero(color_support.reshape(-1))
    if not np.array_equal(
            np.sort(recovered_linear), np.sort(expected_linear)):
        raise ValueError(
            'native-depth replay does not correlate every accepted projected '
            'target sample')

    native_mask = np.zeros(depth.shape, dtype=np.uint8)
    native_mask[rows, columns] = 255
    native_target_depth = np.zeros(depth.shape, dtype=np.uint16)
    native_target_depth[rows, columns] = depth[rows, columns]
    native_color = np.zeros(depth.shape + (3,), dtype=np.uint8)
    native_color[rows, columns] = color[v, u]
    base_from_depth = base_from_color @ color_from_depth
    return {
        'depth_mm': native_target_depth,
        'mask': native_mask,
        'color_bgr': native_color,
        'camera_matrix': depth_k,
        'base_from_camera': base_from_depth,
        'registration_base_from_camera': base_from_color,
        'geometry_from_registration_camera': color_from_depth,
        'report': {
            'mode': 'native_l515_depth_grid_reverse_correlated',
            'projected_target_points': int(len(expected_linear)),
            'recovered_native_target_points': int(len(rows)),
            'one_native_sample_per_projected_winner': True,
            'depth_correlation_tolerance_mm':
                DEPTH_CORRELATION_TOLERANCE_MM,
            'confidence_threshold': threshold,
            'target_support_expanded': False,
            'source_artifacts_immutable': True,
            'rgb_sampling':
                'nearest correlated raw-colour pixel for visualization only',
        },
    }
