"""Compatibility facade for permanent planning-ray elimination."""

from piper_mobile_manipulation.planning.ray_culls import (
    HARD_CULL_SOURCES,
    PROTOCOL_VERSION,
    HardCullLedger,
    canonical_ray_population,
    canonical_ray_universe,
    hard_cull_snapshot,
    population_key,
    population_sha256,
    prune_hard_culled_rays,
    ray_population_identity,
    ray_universe_sha256,
    stable_revision,
)

__all__ = [
    'HARD_CULL_SOURCES', 'PROTOCOL_VERSION', 'HardCullLedger',
    'canonical_ray_population', 'canonical_ray_universe',
    'hard_cull_snapshot', 'population_key', 'population_sha256',
    'prune_hard_culled_rays', 'ray_population_identity',
    'ray_universe_sha256', 'stable_revision',
]
