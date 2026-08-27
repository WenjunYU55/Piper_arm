"""Compatibility facade for bounded planning rays."""

from piper_mobile_manipulation.planning.rays import (
    MAX_RAY_COUNT,
    MIN_RAY_COUNT,
    RAY_PROBE_ID_BASE,
    RAY_PROBE_ID_STRIDE,
    RAY_REGIONS,
    bind_shortlisted_ray_intervals,
    bounded_ray_interval,
    build_ray_samples,
    decoded_ray_id,
    ray_probe_id,
)

__all__ = [
    'MAX_RAY_COUNT', 'MIN_RAY_COUNT', 'RAY_PROBE_ID_BASE',
    'RAY_PROBE_ID_STRIDE', 'RAY_REGIONS', 'bind_shortlisted_ray_intervals',
    'bounded_ray_interval', 'build_ray_samples', 'decoded_ray_id',
    'ray_probe_id',
]
