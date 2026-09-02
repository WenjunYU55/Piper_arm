"""Trusted silhouette geometry for target-aware viewpoint prequalification."""

import hashlib
import json
import math

import cv2
import numpy as np


SHAPE_SCHEMA_VERSION = 1
CAPTURE_MODEL_SEED_SCHEMA_VERSION = 1
ENVELOPE_SCHEMA_VERSION = 1
MAX_SILHOUETTE_POINTS = 256
MAX_PROFILE_SECTIONS = 24
AXIS_ANISOTROPY_RATIO = 1.2
ENVELOPE_INFLATION_M = 0.010
CAMERA_SURFACE_CLEARANCE_M = 0.250
TARGET_SILHOUETTE_CLIPPED = 'TARGET_SILHOUETTE_CLIPPED'
CROPPED_TOO_CLOSE_DISTANCE_M = 0.250
CROPPED_TOO_LARGE_DISTANCE_M = 3.000
CAPTURE_MODEL_SEED_MAX_TRANSFORM_DELTA_SEC = 0.100


class TargetSilhouetteClippedError(ValueError):
    """Report that a measured component cannot prove its complete outline."""

    def __init__(self, near_depth_m):
        self.near_depth_m = float(near_depth_m)
        super().__init__('target silhouette touches the image border')


def canonical_sha256(value):
    """Return a stable digest for one private JSON-compatible record."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _finite_array(value, shape, label):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError('%s is malformed or non-finite' % label)
    return result


def _stamp_record(header):
    stamp = header.stamp
    return {'sec': int(stamp.sec), 'nanosec': int(stamp.nanosec)}


def clipped_shape_rejection(header, near_depth_m):
    """Build one exact-stamp rejection for a border-clipped silhouette."""
    depth = float(near_depth_m)
    if not math.isfinite(depth) or depth <= 0.0:
        raise ValueError('clipped target near depth is invalid')
    rejection = {
        'schema_version': SHAPE_SCHEMA_VERSION,
        'valid': False,
        'header': {
            'stamp': _stamp_record(header),
            'frame_id': str(header.frame_id),
        },
        'source': 'fresh_mask_qualified_depth',
        'rejection_code': TARGET_SILHOUETTE_CLIPPED,
        'rejection_reason': (
            'the item is cropped by the camera frame, so its complete size '
            'cannot be measured'),
        'near_depth_m': round(depth, 8),
    }
    rejection['measurement_sha256'] = canonical_sha256(rejection)
    return rejection


def stamp_nanoseconds(stamp):
    """Return a validated ROS-style timestamp as integer nanoseconds."""
    if not isinstance(stamp, dict):
        raise ValueError('shape timestamp is missing')
    try:
        sec = int(stamp['sec'])
        nanosec = int(stamp['nanosec'])
    except (KeyError, TypeError, ValueError):
        raise ValueError('shape timestamp is malformed')
    if sec < 0 or nanosec < 0 or nanosec >= 1_000_000_000:
        raise ValueError('shape timestamp is outside ROS bounds')
    return sec * 1_000_000_000 + nanosec


def trusted_silhouette_measurement(
        mask, qualified_support, crop_origin, qualified_depth_m,
        camera_matrix, header, measurement_confidence):
    """Build one bounded metric silhouette from already-qualified RGB-D."""
    semantic = np.asarray(mask, dtype=bool)
    support = np.asarray(qualified_support, dtype=bool)
    depth = np.asarray(qualified_depth_m, dtype=float)
    if semantic.ndim != 2 or support.ndim != 2 or support.shape != depth.shape:
        raise ValueError('mask and qualified depth support are inconsistent')
    try:
        x0, y0 = (int(crop_origin[0]), int(crop_origin[1]))
    except (IndexError, TypeError, ValueError):
        raise ValueError('crop origin is invalid')
    y1, x1 = y0 + support.shape[0], x0 + support.shape[1]
    if x0 < 0 or y0 < 0 or x1 > semantic.shape[1] or y1 > semantic.shape[0]:
        raise ValueError('qualified support lies outside the semantic mask')
    qualified = support & np.isfinite(depth) & (depth > 0.0)
    if not np.any(qualified):
        raise ValueError('qualified target support is empty')
    depths = depth[qualified]
    near_depth = float(np.percentile(depths, 10.0))
    if not math.isfinite(near_depth) or near_depth <= 0.0:
        raise ValueError('trusted target near depth is invalid')

    labels_count, labels = cv2.connectedComponents(
        semantic.astype(np.uint8), connectivity=8)
    qualified_full = np.zeros_like(semantic, dtype=bool)
    qualified_full[y0:y1, x0:x1] = qualified
    best_label = 0
    best_overlap = 0
    for label in range(1, labels_count):
        overlap = int(np.count_nonzero((labels == label) & qualified_full))
        if overlap > best_overlap:
            best_label, best_overlap = label, overlap
    if best_label == 0 or best_overlap == 0:
        raise ValueError('no semantic component overlaps qualified depth')
    component = labels == best_label
    if (
            np.any(component[0, :]) or np.any(component[-1, :])
            or np.any(component[:, 0]) or np.any(component[:, -1])):
        raise TargetSilhouetteClippedError(near_depth)
    contours, _hierarchy = cv2.findContours(
        component.astype(np.uint8), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError('trusted target silhouette has no contour')
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if contour.shape[0] < 3:
        raise ValueError('trusted target silhouette contour is degenerate')
    if contour.shape[0] > MAX_SILHOUETTE_POINTS:
        indexes = np.linspace(
            0, contour.shape[0] - 1, MAX_SILHOUETTE_POINTS,
            dtype=np.int64)
        contour = contour[indexes]

    intrinsic = _finite_array(camera_matrix, (3, 3), 'camera matrix')
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError('camera focal lengths must be positive')
    columns = contour[:, 0].astype(float)
    rows = contour[:, 1].astype(float)
    points = np.column_stack((
        (columns - cx) * near_depth / fx,
        (rows - cy) * near_depth / fy,
        np.full(columns.shape, near_depth, dtype=float),
    ))
    geometry = {
        'schema_version': SHAPE_SCHEMA_VERSION,
        'valid': True,
        'header': {
            'stamp': _stamp_record(header),
            'frame_id': str(header.frame_id),
        },
        'source': 'fresh_mask_qualified_depth',
        'silhouette_points_camera_m': np.round(points, 8).tolist(),
        'near_depth_m': round(near_depth, 8),
        'mask_pixel_count': int(np.count_nonzero(component)),
        'qualified_depth_pixel_count': int(np.count_nonzero(qualified)),
        'measurement_confidence': round(float(measurement_confidence), 8),
        'camera_info': {
            'width': int(semantic.shape[1]),
            'height': int(semantic.shape[0]),
            'fx': round(fx, 8),
            'fy': round(fy, 8),
            'cx': round(cx, 8),
            'cy': round(cy, 8),
        },
    }
    geometry['measurement_sha256'] = canonical_sha256(geometry)
    return geometry


def validate_shape_measurement(payload):
    """Validate and normalize one private silhouette measurement."""
    if not isinstance(payload, dict) or payload.get('valid') is not True:
        raise ValueError('target shape measurement is not valid')
    if int(payload.get('schema_version', -1)) != SHAPE_SCHEMA_VERSION:
        raise ValueError('target shape measurement schema is unsupported')
    header = payload.get('header')
    if not isinstance(header, dict) or not str(header.get('frame_id', '')):
        raise ValueError('target shape frame is missing')
    stamp_nanoseconds(header.get('stamp'))
    points = np.asarray(
        payload.get('silhouette_points_camera_m'), dtype=float)
    if (
            points.ndim != 2 or points.shape[1] != 3
            or points.shape[0] < 3
            or points.shape[0] > MAX_SILHOUETTE_POINTS
            or not np.all(np.isfinite(points))):
        raise ValueError('target silhouette points are malformed')
    supplied = str(payload.get('measurement_sha256', ''))
    unsigned = dict(payload)
    unsigned.pop('measurement_sha256', None)
    if supplied != canonical_sha256(unsigned):
        raise ValueError('target shape measurement digest does not match')
    return dict(payload)


def build_capture_model_seed(shape_measurement, camera_transform):
    """Bind capture-one silhouette evidence to its immutable base transform."""
    shape = validate_shape_measurement(shape_measurement)
    if not isinstance(camera_transform, dict):
        raise ValueError('capture camera transform is missing')
    header = camera_transform.get('header')
    if not isinstance(header, dict):
        raise ValueError('capture camera transform header is missing')
    if str(header.get('frame_id', '')) != 'base_link':
        raise ValueError('capture camera transform must start in base_link')
    transform_stamp = header.get('stamp')
    transform_ns = stamp_nanoseconds(transform_stamp)
    shape_ns = stamp_nanoseconds(shape['header']['stamp'])
    delta_sec = abs(transform_ns - shape_ns) * 1e-9
    if delta_sec > CAPTURE_MODEL_SEED_MAX_TRANSFORM_DELTA_SEC:
        raise ValueError(
            'capture silhouette and camera transform are not synchronized')
    child = str(camera_transform.get('child_frame_id', ''))
    if child != str(shape['header']['frame_id']):
        raise ValueError(
            'capture camera transform child does not match silhouette frame')
    matrix = _finite_array(
        camera_transform.get('matrix_4x4'), (4, 4),
        'capture base-from-camera transform')
    if not np.allclose(
            matrix[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-9):
        raise ValueError('capture camera transform is not homogeneous')
    seed = {
        'schema_version': CAPTURE_MODEL_SEED_SCHEMA_VERSION,
        'shape': shape,
        'base_from_camera': {
            'header': {
                'stamp': dict(transform_stamp),
                'frame_id': 'base_link',
            },
            'child_frame_id': child,
            'matrix_4x4': np.round(matrix, 12).tolist(),
        },
        'shape_transform_delta_sec': round(delta_sec, 9),
    }
    seed['model_seed_sha256'] = canonical_sha256(seed)
    return seed


def validate_capture_model_seed(payload):
    """Validate one capture-bound shape/transform transaction."""
    if not isinstance(payload, dict):
        raise ValueError('capture target model seed is missing')
    if int(payload.get('schema_version', -1)) != \
            CAPTURE_MODEL_SEED_SCHEMA_VERSION:
        raise ValueError('capture target model seed schema is unsupported')
    supplied = str(payload.get('model_seed_sha256', ''))
    unsigned = dict(payload)
    unsigned.pop('model_seed_sha256', None)
    if supplied != canonical_sha256(unsigned):
        raise ValueError('capture target model seed digest does not match')
    rebuilt = build_capture_model_seed(
        payload.get('shape'), payload.get('base_from_camera'))
    if rebuilt['model_seed_sha256'] != supplied:
        raise ValueError('capture target model seed normalization changed')
    return dict(payload)


def validate_shape_rejection(payload):
    """Validate one exact-stamp border-clipping rejection record."""
    if not isinstance(payload, dict) or payload.get('valid') is not False:
        raise ValueError('target shape rejection is not valid')
    if int(payload.get('schema_version', -1)) != SHAPE_SCHEMA_VERSION:
        raise ValueError('target shape rejection schema is unsupported')
    header = payload.get('header')
    if not isinstance(header, dict) or not str(header.get('frame_id', '')):
        raise ValueError('target shape rejection frame is missing')
    stamp_nanoseconds(header.get('stamp'))
    if payload.get('rejection_code') != TARGET_SILHOUETTE_CLIPPED:
        raise ValueError('target shape rejection code is unsupported')
    try:
        near_depth = float(payload['near_depth_m'])
    except (KeyError, TypeError, ValueError):
        raise ValueError('target shape rejection depth is malformed')
    if not math.isfinite(near_depth) or near_depth <= 0.0:
        raise ValueError('target shape rejection depth is invalid')
    supplied = str(payload.get('measurement_sha256', ''))
    unsigned = dict(payload)
    unsigned.pop('measurement_sha256', None)
    if supplied != canonical_sha256(unsigned):
        raise ValueError('target shape rejection digest does not match')
    return dict(payload)


def classify_centered_silhouette(
        payload, camera_target_distance_m,
        too_close_distance_m=CROPPED_TOO_CLOSE_DISTANCE_M,
        too_large_distance_m=CROPPED_TOO_LARGE_DISTANCE_M):
    """
    Classify one fresh target-facing silhouette observation.

    A complete silhouette is immediately usable.  Border contact is only
    diagnosed after the caller has separately proved the camera was settled
    and aimed at the measured target.  The middle depth band deliberately
    requests one farther framing view rather than guessing object size.
    """
    if not isinstance(payload, dict):
        raise ValueError('target framing observation is missing')
    if payload.get('valid') is True:
        validate_shape_measurement(payload)
        return 'CLEAR', 'complete silhouette is inside the camera frame'
    validate_shape_rejection(payload)
    distance = float(camera_target_distance_m)
    close = float(too_close_distance_m)
    large = float(too_large_distance_m)
    if (
            not math.isfinite(distance) or distance <= 0.0
            or not math.isfinite(close) or not math.isfinite(large)
            or close <= 0.0 or large <= close):
        raise ValueError('crop classification distance bounds are invalid')
    if distance <= close:
        return (
            'TOO_CLOSE',
            'item remains cropped with the camera %.3fm from the target '
            'centre, within the %.3fm too-close boundary' % (distance, close),
        )
    if distance >= large:
        return (
            'TOO_LARGE',
            'item remains cropped with the camera %.3fm from the target '
            'centre, at the %.3fm maximum framing distance'
            % (distance, large),
        )
    return (
        'RETRY_FARTHER',
        'item remains cropped with the camera %.3fm from the target centre; '
        'move farther outward toward the %.3fm maximum'
        % (distance, large),
    )


def _deterministic_axis_sign(axis):
    result = np.asarray(axis, dtype=float)
    for value in result:
        if abs(float(value)) > 1e-9:
            return result if value > 0.0 else -result
    return result


def build_revolution_envelope(
        shape_measurement, base_from_camera, target_anchor_base,
        inflation_m=ENVELOPE_INFLATION_M,
        maximum_sections=MAX_PROFILE_SECTIONS):
    """Rotate a trusted silhouette into one conservative frozen envelope."""
    shape = validate_shape_measurement(shape_measurement)
    transform = _finite_array(
        base_from_camera, (4, 4), 'base-from-camera transform')
    bootstrap_anchor = _finite_array(
        target_anchor_base, (3,), 'bootstrap target anchor')
    margin = float(inflation_m)
    section_limit = int(maximum_sections)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError('target envelope inflation is invalid')
    if section_limit < 3 or section_limit > MAX_PROFILE_SECTIONS:
        raise ValueError('target envelope section limit is invalid')
    points_camera = np.asarray(
        shape['silhouette_points_camera_m'], dtype=float)
    homogeneous = np.column_stack((
        points_camera, np.ones(points_camera.shape[0], dtype=float)))
    points = (homogeneous @ transform.T)[:, :3]
    rotation = transform[:3, :3]
    camera_normal = rotation @ np.asarray([0.0, 0.0, 1.0])
    camera_normal /= np.linalg.norm(camera_normal)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / float(max(points.shape[0] - 1, 1))
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    major = vectors[:, order[0]]
    major -= camera_normal * float(np.dot(major, camera_normal))
    major_norm = float(np.linalg.norm(major))
    planar_values = sorted(
        [max(float(values[index]), 0.0) for index in order[:2]],
        reverse=True)
    ratio = planar_values[0] / max(planar_values[1], 1e-12)
    if ratio >= AXIS_ANISOTROPY_RATIO and major_norm > 1e-9:
        axis = major / major_norm
        axis_source = 'mask_major_axis'
    else:
        base_vertical = np.asarray([0.0, 0.0, 1.0])
        projected = base_vertical - camera_normal * float(
            np.dot(base_vertical, camera_normal))
        if float(np.linalg.norm(projected)) <= 1e-9:
            projected = rotation @ np.asarray([0.0, 1.0, 0.0])
            projected -= camera_normal * float(
                np.dot(projected, camera_normal))
            axis_source = 'camera_image_vertical_fallback'
        else:
            axis_source = 'base_vertical_projection'
        axis = projected / np.linalg.norm(projected)
    axis = _deterministic_axis_sign(axis)
    transverse = np.cross(camera_normal, axis)
    transverse /= np.linalg.norm(transverse)
    axial = centered @ axis
    lateral = centered @ transverse
    lateral_min = float(np.min(lateral))
    lateral_max = float(np.max(lateral))
    lateral_midpoint = 0.5 * (lateral_min + lateral_max)
    visible_half_width = 0.5 * (lateral_max - lateral_min)
    lateral = lateral - lateral_midpoint
    axial_min = float(np.min(axial)) - margin
    axial_max = float(np.max(axial)) + margin
    axial_span = axial_max - axial_min
    if not math.isfinite(axial_span) or axial_span <= 1e-6:
        raise ValueError('target silhouette axial span is degenerate')
    sections = min(
        section_limit, max(8, int(math.ceil(axial_span / 0.005))))
    edges = np.linspace(axial_min, axial_max, sections + 1)
    raw_radii = []
    for index in range(sections):
        lower, upper = float(edges[index]), float(edges[index + 1])
        selected = (
            (axial >= lower) &
            (axial <= upper if index == sections - 1 else axial < upper))
        raw_radii.append(
            float(np.max(np.abs(lateral[selected])))
            if np.any(selected) else math.nan)
    valid_indexes = [
        index for index, value in enumerate(raw_radii)
        if math.isfinite(value)]
    if not valid_indexes:
        raise ValueError('target silhouette profile is empty')
    for index, value in enumerate(raw_radii):
        if not math.isfinite(value):
            nearest = min(valid_indexes, key=lambda item: abs(item - index))
            raw_radii[index] = raw_radii[nearest]
    maximum_raw_radius = max(raw_radii)
    if maximum_raw_radius <= 1e-6 or visible_half_width <= 1e-6:
        raise ValueError('target silhouette radial span is degenerate')

    # Qualified depth samples lie on the visible surface.  Move the revolution
    # axis to the silhouette's cross-axis midpoint and one exact half-width
    # away from the camera.  The measured layer is therefore the near surface
    # and the rotation origin lies at the centre of the assumed object depth.
    visible_surface_center = centroid + transverse * lateral_midpoint
    axis_origin = (
        visible_surface_center + camera_normal * visible_half_width)
    profile = []
    boxes = []
    for index in range(sections):
        lower, upper = float(edges[index]), float(edges[index + 1])
        radius = max(raw_radii[max(0, index - 1):min(
            sections, index + 2)]) + margin
        center_s = 0.5 * (lower + upper)
        half_length = 0.5 * (upper - lower)
        center = axis_origin + axis * center_s
        radial_extent = radius * np.sqrt(np.maximum(0.0, 1.0 - axis ** 2))
        half_extents = np.abs(axis) * half_length + radial_extent
        minimum = center - half_extents
        maximum = center + half_extents
        profile.append({
            'center_s_m': round(center_s, 8),
            'half_length_m': round(half_length, 8),
            'radius_m': round(radius, 8),
        })
        boxes.append({
            'id': 'target_envelope:%02d' % index,
            'type': 'box',
            'minimum_m': np.round(minimum, 8).tolist(),
            'maximum_m': np.round(maximum, 8).tolist(),
        })
    bounds_min = np.min(np.asarray([
        box['minimum_m'] for box in boxes]), axis=0)
    bounds_max = np.max(np.asarray([
        box['maximum_m'] for box in boxes]), axis=0)
    corners = np.asarray([
        [x, y, z]
        for x in (bounds_min[0], bounds_max[0])
        for y in (bounds_min[1], bounds_max[1])
        for z in (bounds_min[2], bounds_max[2])])
    # Qualified depth samples lie on a visible surface.  Once the settled,
    # aimed silhouette is accepted, the revolution axis origin is the
    # qualified object-volume centre and the anchor of the permanent rays.
    # Retain the earlier surface lock only as provenance.
    planning_anchor = axis_origin
    bounding_radius = float(np.max(np.linalg.norm(
        corners - planning_anchor, axis=1)))
    envelope = {
        'schema_version': ENVELOPE_SCHEMA_VERSION,
        'frame_id': 'base_link',
        'planning_anchor_m': np.round(planning_anchor, 8).tolist(),
        'bootstrap_anchor_m': np.round(bootstrap_anchor, 8).tolist(),
        'axis_origin_m': np.round(axis_origin, 8).tolist(),
        'visible_surface_center_m': np.round(
            visible_surface_center, 8).tolist(),
        'visible_silhouette_points_m': np.round(points, 8).tolist(),
        'axis_direction': np.round(axis, 10).tolist(),
        'axis_source': axis_source,
        'axis_anisotropy_ratio': round(float(ratio), 8),
        'camera_normal_at_lock': np.round(camera_normal, 10).tolist(),
        'inflation_m': round(margin, 8),
        'surface_clearance_m': CAMERA_SURFACE_CLEARANCE_M,
        'maximum_raw_radius_m': round(maximum_raw_radius, 8),
        'rotation_origin_depth_m': round(visible_half_width, 8),
        'visible_axial_span_m': round(
            float(np.max(axial) - np.min(axial)), 8),
        'visible_transverse_span_m': round(
            2.0 * visible_half_width, 8),
        'bounding_radius_from_anchor_m': round(bounding_radius, 8),
        'bounds_min_m': np.round(bounds_min, 8).tolist(),
        'bounds_max_m': np.round(bounds_max, 8).tolist(),
        'profile_sections': profile,
        'collision_boxes': boxes,
        'shape_measurement_sha256': shape['measurement_sha256'],
        'shape_stamp': dict(shape['header']['stamp']),
    }
    envelope['envelope_sha256'] = canonical_sha256(envelope)
    return envelope


def validate_envelope(envelope):
    """Validate the frozen envelope before it crosses subsystem boundaries."""
    if not isinstance(envelope, dict):
        raise ValueError('target envelope is missing')
    if int(envelope.get('schema_version', -1)) != ENVELOPE_SCHEMA_VERSION:
        raise ValueError('target envelope schema is unsupported')
    if envelope.get('frame_id') != 'base_link':
        raise ValueError('target envelope must be in base_link')
    _finite_array(envelope.get('planning_anchor_m'), (3,), 'planning anchor')
    bootstrap_anchor = envelope.get('bootstrap_anchor_m')
    if bootstrap_anchor is not None:
        _finite_array(
            bootstrap_anchor, (3,), 'bootstrap target anchor')
    _finite_array(envelope.get('axis_origin_m'), (3,), 'envelope axis origin')
    _finite_array(
        envelope.get('visible_surface_center_m'), (3,),
        'visible surface center')
    outline = envelope.get('visible_silhouette_points_m')
    if outline is not None:
        outline = np.asarray(outline, dtype=float)
        if (
                outline.ndim != 2 or outline.shape[1] != 3
                or outline.shape[0] < 3
                or outline.shape[0] > MAX_SILHOUETTE_POINTS
                or not np.all(np.isfinite(outline))):
            raise ValueError('visible silhouette outline is malformed')
    axis = _finite_array(
        envelope.get('axis_direction'), (3,), 'envelope axis direction')
    if abs(float(np.linalg.norm(axis)) - 1.0) > 1e-6:
        raise ValueError('target envelope axis is not unit length')
    try:
        origin_depth = float(envelope['rotation_origin_depth_m'])
        visible_span = float(envelope['visible_transverse_span_m'])
    except (KeyError, TypeError, ValueError):
        raise ValueError('target envelope rotation origin is malformed')
    if (
            not math.isfinite(origin_depth) or origin_depth <= 0.0
            or not math.isfinite(visible_span) or visible_span <= 0.0
            or abs(origin_depth - 0.5 * visible_span) > 1e-6):
        raise ValueError(
            'target envelope rotation origin is not half the mask width')
    profile = envelope.get('profile_sections')
    boxes = envelope.get('collision_boxes')
    if (
            not isinstance(profile, list) or not 3 <= len(profile) <=
            MAX_PROFILE_SECTIONS or not isinstance(boxes, list)
            or len(boxes) != len(profile)):
        raise ValueError('target envelope profile is malformed')
    for section, box in zip(profile, boxes):
        values = [
            float(section[key]) for key in (
                'center_s_m', 'half_length_m', 'radius_m')]
        if (
                not all(math.isfinite(value) for value in values)
                or values[1] <= 0.0 or values[2] <= 0.0):
            raise ValueError('target envelope section is invalid')
        minimum = _finite_array(box.get('minimum_m'), (3,), 'box minimum')
        maximum = _finite_array(box.get('maximum_m'), (3,), 'box maximum')
        if np.any(minimum >= maximum) or box.get('type') != 'box':
            raise ValueError('target envelope collision box is invalid')
    supplied = str(envelope.get('envelope_sha256', ''))
    unsigned = dict(envelope)
    unsigned.pop('envelope_sha256', None)
    if supplied != canonical_sha256(unsigned):
        raise ValueError('target envelope digest does not match')
    return dict(envelope)


def _point_to_validated_envelope_distance(point, envelope):
    """Return point distance after the caller has validated the envelope."""
    value = envelope
    origin = np.asarray(value['axis_origin_m'], dtype=float)
    axis = np.asarray(value['axis_direction'], dtype=float)
    minimum_distance = math.inf
    for section in value['profile_sections']:
        center = origin + axis * float(section['center_s_m'])
        relative = point - center
        axial = abs(float(np.dot(relative, axis)))
        radial = float(np.linalg.norm(
            relative - axis * float(np.dot(relative, axis))))
        outside_axial = max(
            0.0, axial - float(section['half_length_m']))
        outside_radial = max(0.0, radial - float(section['radius_m']))
        distance = math.hypot(outside_axial, outside_radial)
        minimum_distance = min(minimum_distance, distance)
    return float(minimum_distance)


def point_to_envelope_distance(point, envelope):
    """Return conservative distance from a point to the revolved solid."""
    value = validate_envelope(envelope)
    position = _finite_array(point, (3,), 'camera point')
    return _point_to_validated_envelope_distance(position, value)


def _minimum_fov_standoff_validated(value, camera_info):
    """Return FOV distance after the caller validates the envelope."""
    if not isinstance(camera_info, dict) or not camera_info.get('available'):
        raise ValueError('camera intrinsics are unavailable')
    try:
        width = float(camera_info['width'])
        height = float(camera_info['height'])
        fx = float(camera_info['fx'])
        fy = float(camera_info['fy'])
    except (KeyError, TypeError, ValueError):
        raise ValueError('camera intrinsics are malformed')
    if not all(math.isfinite(item) and item > 0.0 for item in (
            width, height, fx, fy)):
        raise ValueError('camera intrinsics are invalid')
    half_angle = min(
        math.atan(0.5 * width / fx), math.atan(0.5 * height / fy))
    radius = float(value['bounding_radius_from_anchor_m'])
    return radius / max(math.sin(half_angle), 1e-9)


def minimum_fov_standoff(envelope, camera_info):
    """Return centre distance required to contain the complete envelope."""
    return _minimum_fov_standoff_validated(
        validate_envelope(envelope), camera_info)


def coverage_sphere_from_envelope(envelope):
    """Return an object-sized NBV sphere from the uninflated revolution."""
    value = validate_envelope(envelope)
    diameter = max(
        float(value['visible_axial_span_m']),
        float(value['visible_transverse_span_m']))
    if not math.isfinite(diameter) or diameter <= 0.0:
        raise ValueError('target envelope coverage diameter is invalid')
    return {
        'center_m': [float(item) for item in value['axis_origin_m']],
        'diameter_m': float(diameter),
        'radius_m': 0.5 * float(diameter),
        'source': 'qualified_revolved_target_size',
    }


def envelope_constrained_ray_interval(
        target_anchor, direction, minimum_standoff, maximum_standoff,
        envelope, camera_info,
        clearance_m=CAMERA_SURFACE_CLEARANCE_M, envelope_is_validated=False):
    """Return a safe continuous ray interval or ``None`` when none exists."""
    anchor = _finite_array(target_anchor, (3,), 'ray target anchor')
    ray = _finite_array(direction, (3,), 'ray direction')
    norm = float(np.linalg.norm(ray))
    if norm <= 1e-9:
        raise ValueError('ray direction is zero')
    ray /= norm
    lower, upper = float(minimum_standoff), float(maximum_standoff)
    clearance = float(clearance_m)
    if (
            not all(math.isfinite(value) for value in (
                lower, upper, clearance))
            or lower <= 0.0 or upper < lower or clearance < 0.0):
        raise ValueError('ray interval or clearance is invalid')
    value = (
        envelope if envelope_is_validated else validate_envelope(envelope))
    lower = max(lower, _minimum_fov_standoff_validated(value, camera_info))
    if lower > upper:
        return None

    def safe(standoff):
        point = anchor + ray * float(standoff)
        return _point_to_validated_envelope_distance(
            point, value) >= clearance

    if safe(lower):
        return float(lower), float(upper)
    samples = np.linspace(lower, upper, 65)
    previous = float(samples[0])
    for sample in samples[1:]:
        current = float(sample)
        if safe(current):
            low, high = previous, current
            for _iteration in range(20):
                middle = 0.5 * (low + high)
                if safe(middle):
                    high = middle
                else:
                    low = middle
            return float(high), float(upper)
        previous = current
    return None
