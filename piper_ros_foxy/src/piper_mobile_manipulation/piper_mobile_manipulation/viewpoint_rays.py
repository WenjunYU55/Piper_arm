"""Pure geometry helpers for bounded target-centred viewpoint rays."""

import math

import numpy as np


RAY_PROBE_ID_BASE = 1_000_000
RAY_PROBE_ID_STRIDE = 8


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

    Every preferred-band probe is ordered before every reserve-band probe.
    This retains the existing exact-pose Tesseract contract without restoring
    the planner's former full radial lattice.
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

    preferred = []
    reserve = []
    for item in candidates:
        candidate = dict(item)
        if candidate.get('candidate_geometry') != 'target_ray':
            preferred.append(candidate)
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
            preferred.append(exact_probe(standoff, probe_index, 'preferred'))
        reserve_offset = len(preferred_values)
        for offset, standoff in enumerate(reserve_values):
            reserve.append(exact_probe(
                standoff, reserve_offset + offset, 'reserve'))
    return preferred + reserve
