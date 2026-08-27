"""Pure validation and geometry helpers for event-triggered RGB-D capture."""

from collections import deque
import math

import cv2
import numpy as np

from piper_mobile_manipulation.utils.target_depth import (
    select_target_depth_component,
)


class DepthQualityRejected(ValueError):
    """A complete observation exists, but its target depth is not usable."""


def stamp_seconds(message):
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def stamp_key(message):
    """Return an exact ROS timestamp key without floating-point comparison."""
    stamp = message.header.stamp
    return int(stamp.sec), int(stamp.nanosec)


def exact_stamped_item(items, reference):
    """Select the newest item whose first message has the exact stamp."""
    wanted = stamp_key(reference)
    for item in reversed(list(items)):
        if item and stamp_key(item[0]) == wanted:
            return item
    return None


def nearest_stamped_item(items, reference, maximum_delta_sec):
    """Select a cached item by its first message's nearest valid timestamp."""
    if not items:
        return None
    target = stamp_seconds(reference)
    candidate = min(
        items, key=lambda item: abs(stamp_seconds(item[0]) - target))
    if abs(stamp_seconds(candidate[0]) - target) > float(maximum_delta_sec):
        return None
    return candidate


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


def capture_diagnostic_rejection(
        quality, quality_age_sec, occlusion, occlusion_age_sec,
        maximum_age_sec=1.0, minimum_quality_score=0.65,
        allowed_occlusion_states=('CLEAR',)):
    """Require one fresh GOOD observation and an explicitly allowed scene."""
    if not isinstance(quality, dict):
        return 'QUALITY_REJECTED: scan quality is missing'
    try:
        quality_age = float(quality_age_sec)
    except (TypeError, ValueError):
        quality_age = math.inf
    if (
            not math.isfinite(quality_age) or quality_age < 0.0
            or quality_age > float(maximum_age_sec)):
        return 'QUALITY_REJECTED: scan quality is stale'
    label = str(quality.get(
        'quality_label', quality.get('status', ''))).upper()
    try:
        score = float(quality.get('quality_score', quality.get('score', 0.0)))
    except (TypeError, ValueError):
        score = math.nan
    if quality.get('target_valid') is not True:
        return 'QUALITY_REJECTED: target is not valid in the settled frame'
    if (
            label != 'GOOD' or not math.isfinite(score)
            or score < float(minimum_quality_score)):
        return (
            'QUALITY_REJECTED: quality %s %.3f is below GOOD %.3f'
            % (label or 'MISSING', score, float(minimum_quality_score)))
    if not isinstance(occlusion, dict):
        return 'OCCLUSION_REJECTED: occlusion evidence is missing'
    try:
        occlusion_age = float(occlusion_age_sec)
    except (TypeError, ValueError):
        occlusion_age = math.inf
    if (
            not math.isfinite(occlusion_age) or occlusion_age < 0.0
            or occlusion_age > float(maximum_age_sec)):
        return 'OCCLUSION_REJECTED: occlusion evidence is stale'
    state = str(occlusion.get('occlusion_state', 'UNKNOWN')).upper()
    allowed = {
        str(value).strip().upper()
        for value in allowed_occlusion_states
        if str(value).strip()
    }
    if state not in allowed:
        return 'OCCLUSION_REJECTED: settled target view is %s' % state
    return ''


def depth_millimetres(depth, encoding):
    """Return lossless uint16 millimetres for normal L515 encodings."""
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


def temporal_confident_depth_median(
        depth_frames, depth_encodings, confidence_frames,
        minimum_confidence=8, minimum_support_fraction=0.50):
    """
    Fuse a settled native-depth burst without averaging invalid pixels.

    Only nonzero depth samples with a qualifying L515 confidence grade
    contribute.  A pixel is retained when it is supported by the configured
    fraction of the burst.  Median, rather than arithmetic mean, prevents one
    flying depth sample from pulling the reconstructed surface.
    """
    depths = list(depth_frames)
    encodings = list(depth_encodings)
    confidences = list(confidence_frames)
    if not depths:
        raise ValueError('depth burst must contain at least one frame')
    if len(depths) != len(encodings) or len(depths) != len(confidences):
        raise ValueError(
            'depth, encoding and confidence burst lengths must match')
    fraction = float(minimum_support_fraction)
    if not math.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
        raise ValueError(
            'minimum temporal support fraction must be in (0, 1]')
    threshold = int(minimum_confidence)
    if threshold < 0 or threshold > 15:
        raise ValueError('minimum confidence must be in the range 0..15')

    depth_stack = np.stack([
        depth_millimetres(frame, encoding)
        for frame, encoding in zip(depths, encodings)
    ])
    confidence_stack = np.stack([
        normalize_l515_confidence(frame)[0] for frame in confidences
    ])
    if depth_stack.ndim != 3:
        raise ValueError('every native depth burst frame must be 2D')
    if confidence_stack.shape != depth_stack.shape:
        raise ValueError(
            'confidence burst shapes must match native depth shapes')

    valid = (depth_stack > 0) & (confidence_stack >= threshold)
    support = np.count_nonzero(valid, axis=0)
    required = max(1, int(math.ceil(len(depths) * fraction)))
    retained = support >= required

    def valid_integer_median(stack, sentinel):
        values = np.where(valid, stack, sentinel)
        values.sort(axis=0)
        lower_index = np.maximum(0, (support - 1) // 2)[None, ...]
        upper_index = np.maximum(0, support // 2)[None, ...]
        lower = np.take_along_axis(values, lower_index, axis=0)[0]
        upper = np.take_along_axis(values, upper_index, axis=0)[0]
        return (
            lower.astype(np.uint32) + upper.astype(np.uint32)
        ) / 2.0

    median_depth = valid_integer_median(depth_stack, np.uint16(65535))
    median_confidence = valid_integer_median(
        confidence_stack, np.uint8(255))
    output_depth = np.zeros(depth_stack.shape[1:], dtype=np.uint16)
    output_confidence = np.zeros(depth_stack.shape[1:], dtype=np.uint8)
    output_depth[retained] = np.clip(
        np.rint(median_depth[retained]), 0, 65535).astype(np.uint16)
    output_confidence[retained] = np.clip(
        np.rint(median_confidence[retained]), 0, 15).astype(np.uint8)

    retained_support = support[retained]
    return output_depth, output_confidence, {
        'estimator': 'per_pixel_median',
        'input_frames': int(len(depths)),
        'minimum_confidence_grade': threshold,
        'minimum_support_fraction': fraction,
        'minimum_support_frames': required,
        'retained_pixels': int(np.count_nonzero(retained)),
        'retained_support_min_frames': (
            int(np.min(retained_support)) if retained_support.size else 0),
        'retained_support_median_frames': (
            float(np.median(retained_support))
            if retained_support.size else 0.0),
        'retained_support_max_frames': (
            int(np.max(retained_support)) if retained_support.size else 0),
    }


def camera_matrix(values, name):
    """Validate and return a three-by-three pinhole camera matrix."""
    matrix = np.asarray(values, dtype=float)
    if matrix.size != 9:
        raise ValueError('%s must contain nine values' % name)
    matrix = matrix.reshape(3, 3)
    if not np.all(np.isfinite(matrix)) or matrix[0, 0] <= 0.0 \
            or matrix[1, 1] <= 0.0:
        raise ValueError('%s is not a finite pinhole matrix' % name)
    return matrix


def adaptive_eroded_mask(mask, native_width, minimum_retained_fraction=0.70,
                         minimum_retained_pixels=64):
    """Erode by one native-depth pixel without destroying a small RGB mask."""
    binary = np.asarray(mask) > 0
    if binary.ndim != 2 or not np.any(binary):
        raise DepthQualityRejected(
            'DEPTH_QUALITY_REJECTED: target mask is empty')
    radius = max(1, int(round(float(binary.shape[1]) / float(native_width))))
    kernel_size = radius * 2 + 1
    eroded = cv2.erode(
        binary.astype(np.uint8),
        np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1) > 0
    before = int(np.count_nonzero(binary))
    after = int(np.count_nonzero(eroded))
    retained = float(after) / float(before)
    applied = bool(
        after >= int(minimum_retained_pixels)
        and retained >= float(minimum_retained_fraction))
    return (eroded if applied else binary), {
        'native_pixel_radius_in_rgb_px': radius,
        'mask_pixels_before': before,
        'mask_pixels_after_candidate_erosion': after,
        'retained_fraction': retained,
        'erosion_applied': applied,
    }


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
        x * radial + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x),
        y * radial + p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y,
    )


def project_native_depth(depth_mm, depth_k, color_k, color_distortion,
                         color_distortion_model, color_from_depth,
                         color_shape):
    """Project every valid native depth sample into the raw colour image."""
    depth = np.asarray(depth_mm, dtype=np.uint16)
    if depth.ndim != 2:
        raise ValueError('native depth must be a two-dimensional image')
    kd = camera_matrix(depth_k, 'native depth camera matrix')
    kc = camera_matrix(color_k, 'colour camera matrix')
    transform = np.asarray(color_from_depth, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(
            'depth-to-colour transform must be a finite 4x4 matrix')
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
        normalized_x, normalized_y, color_distortion, color_distortion_model)
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
        'color_points_m': color_points,
        'color_depth_m': color_z,
    }


def depth_connected_component(candidate, depth_mm, seed, threshold_mm):
    """Return the 8-connected, depth-continuous component containing seed."""
    support = np.asarray(candidate, dtype=bool)
    depth = np.asarray(depth_mm, dtype=np.uint16)
    if support.shape != depth.shape:
        raise ValueError('candidate and depth shapes differ')
    seed_row, seed_column = [int(value) for value in seed]
    if not support[seed_row, seed_column]:
        raise ValueError('component seed is outside candidate support')
    selected = np.zeros_like(support)
    selected[seed_row, seed_column] = True
    pending = deque([(seed_row, seed_column)])
    height, width = support.shape
    limit = float(threshold_mm)
    while pending:
        row, column = pending.popleft()
        reference = float(depth[row, column])
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                yy, xx = row + dy, column + dx
                if yy < 0 or xx < 0 or yy >= height or xx >= width \
                        or selected[yy, xx] or not support[yy, xx]:
                    continue
                if abs(float(depth[yy, xx]) - reference) <= limit:
                    selected[yy, xx] = True
                    pending.append((yy, xx))
    return selected


def _zbuffer_selected(projection, selected, confidence, color_shape):
    height, width = [int(value) for value in color_shape]
    rows, columns = np.nonzero(selected)
    if rows.size == 0:
        raise DepthQualityRejected(
            'DEPTH_QUALITY_REJECTED: target component is empty')
    u = projection['u'][rows, columns]
    v = projection['v'][rows, columns]
    z = projection['color_depth_m'][rows, columns]
    linear = v.astype(np.int64) * width + u.astype(np.int64)
    order = np.argsort(z, kind='stable')
    _, first = np.unique(linear[order], return_index=True)
    winners = order[first]
    output_depth = np.zeros(height * width, dtype=np.uint16)
    output_confidence = np.zeros(height * width, dtype=np.uint8)
    winner_linear = linear[winners]
    output_depth[winner_linear] = np.clip(
        np.rint(z[winners] * 1000.0), 0, 65535).astype(np.uint16)
    output_confidence[winner_linear] = np.asarray(
        confidence, dtype=np.uint8)[rows[winners], columns[winners]]
    output_mask = output_depth > 0
    return (
        output_depth.reshape(height, width),
        (output_mask.reshape(height, width).astype(np.uint8) * 255),
        output_confidence.reshape(height, width),
    )


def normalize_l515_confidence(confidence):
    """Return logical 0..15 L515 grades from either RAW8 representation."""
    values = np.asarray(confidence)
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError('confidence image must use an integer mono8 encoding')
    if np.any(values < 0):
        raise ValueError('L515 confidence grades must be nonnegative')
    if np.all(values <= 15):
        return values.astype(np.uint8), 'unpacked_4bit_grade'
    if np.all(values <= 240) and np.all((values % 16) == 0):
        return (values.astype(np.uint8) >> 4), 'left_justified_4bit_raw8'
    raise ValueError(
        'L515 confidence bytes must be 0..15 grades or high-nibble RAW8 '
        'values 0,16,...,240')


def qualify_target_depth(
        depth, depth_encoding, confidence, mask, depth_k, color_k,
        color_distortion, color_distortion_model, color_from_depth,
        minimum_confidence=8, minimum_points=50,
        minimum_confident_fraction=0.50, minimum_component_fraction=0.15,
        component_ambiguity_margin=0.08):
    """Build a confidence-qualified target from one correlated set."""
    depth_mm = depth_millimetres(depth, depth_encoding)
    confidence_input = np.asarray(confidence)
    if confidence_input.ndim != 2 or confidence_input.shape != depth_mm.shape:
        raise ValueError('confidence image shape does not match native depth')
    confidence_values, confidence_representation = \
        normalize_l515_confidence(confidence_input)
    threshold = int(minimum_confidence)
    if threshold < 0 or threshold > 15:
        raise ValueError('minimum confidence must be in the range 0..15')
    binary_mask = np.asarray(mask) > 0
    eroded_mask, erosion = adaptive_eroded_mask(
        binary_mask, depth_mm.shape[1])
    projection = project_native_depth(
        depth_mm, depth_k, color_k, color_distortion,
        color_distortion_model, color_from_depth, binary_mask.shape)
    projected = projection['valid']
    projected_inside = np.zeros_like(projected)
    projected_inside[projected] = eroded_mask[
        projection['v'][projected], projection['u'][projected]]
    depth_supported = projected_inside & (depth_mm > 0)
    confident = depth_supported & (confidence_values >= threshold)
    supported_count = int(np.count_nonzero(depth_supported))
    confident_count = int(np.count_nonzero(confident))
    confident_fraction = (
        float(confident_count) / float(supported_count)
        if supported_count else 0.0)
    if confident_count < int(minimum_points):
        raise DepthQualityRejected(
            'DEPTH_QUALITY_REJECTED: only %d confident target points; need %d'
            % (confident_count, int(minimum_points)))
    if confident_fraction < float(minimum_confident_fraction):
        raise DepthQualityRejected(
            'DEPTH_QUALITY_REJECTED: confident target fraction %.3f is '
            'below %.3f'
            % (confident_fraction, float(minimum_confident_fraction)))
    mask_rows, mask_columns = np.nonzero(eroded_mask)
    try:
        component, component_report = select_target_depth_component(
            confident, depth_mm.astype(np.float64) / 1000.0,
            semantic_u=projection['u'], semantic_v=projection['v'],
            center_u=float(np.median(mask_columns)),
            center_v=float(np.median(mask_rows)),
            minimum_points=minimum_points,
            minimum_support_fraction=minimum_component_fraction,
            ambiguity_margin=component_ambiguity_margin)
    except ValueError as exc:
        raise DepthQualityRejected(
            'DEPTH_QUALITY_REJECTED: %s' % exc) from exc
    component_count = int(np.count_nonzero(component))
    component_fraction = float(component_count) / float(confident_count)
    if component_count < int(minimum_points) \
            or component_fraction < float(minimum_component_fraction):
        raise DepthQualityRejected(
            'DEPTH_QUALITY_REJECTED: primary target component has %d points '
            'and %.3f support; need %d and %.3f'
            % (component_count, component_fraction, int(minimum_points),
               float(minimum_component_fraction)))
    target_depth, target_mask, aligned_confidence = _zbuffer_selected(
        projection, component, confidence_values, binary_mask.shape)
    points = projection['color_points_m'][component]
    centroid = np.median(points, axis=0)
    grades = np.bincount(
        confidence_values[depth_supported].astype(np.int64), minlength=16)
    return {
        'native_depth_mm': depth_mm,
        'confidence': confidence_values,
        'eroded_mask': eroded_mask.astype(np.uint8) * 255,
        'target_depth_mm': target_depth,
        'target_support_mask': target_mask,
        'aligned_confidence': aligned_confidence,
        'target': {
            'point': {
                'x': float(centroid[0]), 'y': float(centroid[1]),
                'z': float(centroid[2]),
            },
            'depth': float(np.median(points[:, 2])),
            'depth_stddev': float(np.std(points[:, 2])),
            'valid': True,
            'depth_source': 'l515_confidence_qualified_native_depth',
        },
        'quality': {
            'confidence_threshold': threshold,
            'confidence_input_representation': confidence_representation,
            'confidence_grade_histogram_0_to_15': grades.astype(int).tolist(),
            'depth_supported_points': supported_count,
            'confident_points': confident_count,
            'confident_fraction': confident_fraction,
            'primary_component_points': component_count,
            'primary_component_fraction': component_fraction,
            'depth_layer_selection': component_report,
            'projected_output_points': int(np.count_nonzero(target_mask)),
            'erosion': erosion,
        },
    }


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
