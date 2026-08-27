"""Pure geometry helpers for bounded target-centred viewpoint rays."""

import math

import numpy as np


RAY_PROBE_ID_BASE = 1_000_000
RAY_PROBE_ID_STRIDE = 8
RAY_REGIONS = (
    'target_sector',
    'upper_hemisphere',
    'full_sphere',
)
MIN_RAY_COUNT = 1
MAX_RAY_COUNT = 1000


def _wrapped_angle_deg(value):
    return (float(value) + 180.0) % 360.0 - 180.0


def _evenly_downsample(values, count):
    if len(values) == count:
        return list(values)
    if count == 1:
        return [values[len(values) // 2]]
    last = len(values) - 1
    indexes = [
        int(round(index * last / float(count - 1)))
        for index in range(count)
    ]
    return [values[index] for index in indexes]


def build_ray_samples(
        region, ray_count, center_angle_deg=180.0,
        sector_angle_deg=180.0, sector_pitch_degrees=()):
    """
    Return exactly ``ray_count`` deterministic target-centred directions.

    ``target_sector`` retains the existing azimuth/elevation grid convention.
    The hemisphere and sphere use equal-area Fibonacci samples so their poles
    are not duplicated for every azimuth.
    """
    selected_region = str(region).strip()
    count = int(ray_count)
    center = float(center_angle_deg)
    if selected_region not in RAY_REGIONS:
        raise ValueError('unsupported ray sampling region: %s' % region)
    if count < MIN_RAY_COUNT or count > MAX_RAY_COUNT:
        raise ValueError(
            'ray count must be between %d and %d'
            % (MIN_RAY_COUNT, MAX_RAY_COUNT))
    if not math.isfinite(center):
        raise ValueError('ray center angle must be finite')

    if selected_region == 'target_sector':
        span = min(abs(float(sector_angle_deg)), 360.0)
        pitches = [float(value) for value in sector_pitch_degrees]
        if not math.isfinite(span) or span <= 0.0:
            raise ValueError('target-sector angle must be positive and finite')
        if not pitches or not all(
                math.isfinite(value) and -90.0 <= value <= 90.0
                for value in pitches):
            raise ValueError(
                'target-sector pitches must be finite values within 90 degrees')
        azimuth_count = max(1, int(math.ceil(count / float(len(pitches)))))
        if span >= 360.0 - 1e-9:
            step = 360.0 / float(azimuth_count)
            azimuths = [
                center - 180.0 + index * step
                for index in range(azimuth_count)
            ]
        elif azimuth_count == 1:
            azimuths = [center]
        else:
            start = center - 0.5 * span
            step = span / float(azimuth_count - 1)
            azimuths = [start + index * step for index in range(azimuth_count)]
        grid = [
            (float(azimuth), float(pitch))
            for pitch in pitches for azimuth in azimuths
        ]
        return _evenly_downsample(grid, count)

    golden_angle_deg = 137.50776405003785
    samples = []
    for index in range(count):
        unit = (index + 0.5) / float(count)
        vertical = (
            unit if selected_region == 'upper_hemisphere'
            else 1.0 - 2.0 * unit)
        pitch = -math.degrees(math.asin(vertical))
        azimuth = _wrapped_angle_deg(center + index * golden_angle_deg)
        samples.append((azimuth, pitch))
    return samples


def bounded_ray_interval(
        target_center, minimum_standoff_m, maximum_standoff_m):
    """
    Return the configured target-relative mission ray interval.

    The target position validates the frame geometry, but it must not shorten
    the ray.  Workspace and immutable capability-map evidence own subsequent
    per-direction culling/bounding before continuous IK runs.
    """
    target = np.asarray(target_center, dtype=float)
    minimum = float(minimum_standoff_m)
    maximum = float(maximum_standoff_m)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError('ray target center must contain three finite values')
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise ValueError('ray minimum standoff must be positive and finite')
    if not math.isfinite(maximum) or maximum < minimum:
        raise ValueError('ray maximum standoff is below its minimum')
    return minimum, maximum


def ray_probe_id(ray_id, probe_index):
    """Encode one exact standoff probe without changing public messages."""
    ray = int(ray_id)
    probe = int(probe_index)
    if ray < 0 or probe < 0 or probe >= RAY_PROBE_ID_STRIDE:
        raise ValueError('ray/probe identity is outside its bounded range')
    return RAY_PROBE_ID_BASE + ray * RAY_PROBE_ID_STRIDE + probe


def decoded_ray_id(viewpoint_id):
    """Return the stable ray ID for an expanded probe, otherwise ``None``."""
    value = int(viewpoint_id)
    if value < RAY_PROBE_ID_BASE:
        return None
    return (value - RAY_PROBE_ID_BASE) // RAY_PROBE_ID_STRIDE


def bind_shortlisted_ray_intervals(
        candidates, start_camera_position, target_center):
    """
    Bind each shortlisted ray to one bounded Tesseract search interval.

    The representative point keeps the existing request/provenance contract,
    but it is not the only endpoint the worker may select.  The isolated
    worker solves for a continuous standoff inside this interval while
    preserving the ray direction and target-facing aim.
    """
    start = np.asarray(start_camera_position, dtype=float)
    target = np.asarray(target_center, dtype=float)
    if start.shape != (3,) or not np.all(np.isfinite(start)):
        raise ValueError('ray expansion camera start is invalid')
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError('ray expansion target center is invalid')

    bound = []
    for item in candidates:
        candidate = dict(item)
        if candidate.get('candidate_geometry') != 'target_ray':
            bound.append(candidate)
            continue
        ray_id = int(candidate['ray_id'])
        direction = np.asarray(candidate.get('ray_direction'), dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError('ray direction is invalid')
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            raise ValueError('ray direction is zero')
        direction /= norm
        minimum = float(candidate['ray_min_standoff_m'])
        maximum = float(candidate['ray_max_standoff_m'])
        preferred_maximum = min(
            maximum, float(candidate['ray_preferred_max_standoff_m']))
        if (
                not all(math.isfinite(value) for value in (
                    minimum, maximum, preferred_maximum))
                or minimum <= 0.0 or maximum < minimum
                or preferred_maximum < minimum):
            raise ValueError('ray standoff interval is invalid')
        projected = float(np.dot(start - target, direction))
        raw_intervals = candidate.get('ray_capability_intervals_m')
        intervals = []
        if raw_intervals is not None:
            if not isinstance(raw_intervals, (list, tuple)):
                raise ValueError('ray capability intervals are invalid')
            for raw in raw_intervals:
                if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                    raise ValueError('ray capability interval is invalid')
                lower, upper = float(raw[0]), float(raw[1])
                if (
                        not all(math.isfinite(value) for value in (
                            lower, upper))
                        or lower < minimum - 1e-9
                        or upper > maximum + 1e-9 or upper < lower):
                    raise ValueError('ray capability interval is invalid')
                intervals.append((lower, upper))
        if not intervals:
            intervals = [(minimum, maximum)]
        primary_candidates = [
            min(upper, max(lower, projected))
            for lower, upper in intervals
            if lower <= preferred_maximum + 1e-9]
        if not primary_candidates:
            primary_candidates = [intervals[0][0]]
        primary = min(
            primary_candidates,
            key=lambda value: (abs(value - projected), value))
        primary = min(preferred_maximum, max(minimum, primary))
        # Preserve the current-pose projection as the first numerical seed.
        # The worker owns the rest of the one-dimensional interval search.
        representative = primary
        candidate['id'] = ray_probe_id(ray_id, 0)
        candidate['camera_position_m'] = (
            target + direction * float(representative)).tolist()
        candidate['look_direction'] = (-direction).tolist()
        candidate['ray_id'] = ray_id
        candidate['ray_standoff_m'] = float(representative)
        candidate['ray_probe_index'] = 0
        candidate['ray_probe_phase'] = 'interval_search'
        candidate['ray_direction'] = direction.tolist()
        bound.append(candidate)
    return bound
