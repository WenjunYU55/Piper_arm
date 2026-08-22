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
    """Return the static mission ray interval for one measured target."""
    target = np.asarray(target_center, dtype=float)
    minimum = float(minimum_standoff_m)
    maximum = float(maximum_standoff_m)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError('ray target center must contain three finite values')
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise ValueError('ray minimum standoff must be positive and finite')
    if not math.isfinite(maximum) or maximum < minimum:
        raise ValueError('ray maximum standoff is below its minimum')
    target_radius = float(np.linalg.norm(target))
    return minimum, max(minimum, min(target_radius, maximum))


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


def _distinct_standoffs(values, tolerance_m):
    selected = []
    for value in values:
        candidate = float(value)
        if not any(abs(candidate - previous) <= tolerance_m
                   for previous in selected):
            selected.append(candidate)
    return selected


def expand_shortlisted_rays(
        candidates, start_camera_position, target_center,
        standoff_tolerance_m=0.005):
    """
    Expand only shortlisted rays into bounded exact Tesseract endpoints.

    Keep every probe for one information-ranked ray adjacent.  The worker can
    therefore exhaust that direction before attempting the next ray without
    changing the existing exact-pose Tesseract contract or restoring the
    planner's former full radial lattice.
    """
    start = np.asarray(start_camera_position, dtype=float)
    target = np.asarray(target_center, dtype=float)
    tolerance = float(standoff_tolerance_m)
    if start.shape != (3,) or not np.all(np.isfinite(start)):
        raise ValueError('ray expansion camera start is invalid')
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError('ray expansion target center is invalid')
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError('ray standoff tolerance is invalid')

    expanded = []
    for item in candidates:
        candidate = dict(item)
        if candidate.get('candidate_geometry') != 'target_ray':
            expanded.append(candidate)
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
        primary = min(preferred_maximum, max(minimum, projected))
        preferred_values = _distinct_standoffs(
            (primary, minimum, preferred_maximum), tolerance)
        reserve_values = _distinct_standoffs(
            (
                0.5 * (preferred_maximum + maximum),
                maximum,
            ) if maximum > preferred_maximum + tolerance else (),
            tolerance,
        )

        def exact_probe(standoff, probe_index, phase):
            result = dict(candidate)
            result['id'] = ray_probe_id(ray_id, probe_index)
            result['camera_position_m'] = (
                target + direction * float(standoff)).tolist()
            result['look_direction'] = (-direction).tolist()
            result['ray_id'] = ray_id
            result['ray_standoff_m'] = float(standoff)
            result['ray_probe_index'] = int(probe_index)
            result['ray_probe_phase'] = str(phase)
            result['ray_direction'] = direction.tolist()
            return result

        for probe_index, standoff in enumerate(preferred_values):
            expanded.append(exact_probe(
                standoff, probe_index, 'preferred'))
        reserve_offset = len(preferred_values)
        for offset, standoff in enumerate(reserve_values):
            expanded.append(exact_probe(
                standoff, reserve_offset + offset, 'reserve'))
    return expanded
