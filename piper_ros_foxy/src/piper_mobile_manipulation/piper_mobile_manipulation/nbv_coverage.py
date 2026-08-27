"""Compatibility facade for object-centric planning coverage."""

from piper_mobile_manipulation.planning.coverage import (
    FREE,
    MINIMUM_USEFUL_MARGINAL_INFORMATION_FRACTION,
    SURFACE,
    UNKNOWN,
    CoverageSnapshot,
    ObjectCoverageModel,
    VoxelCoverageConfig,
    candidate_information,
    candidate_meets_minimum_information,
    direction_bin,
    persist_coverage_snapshot,
    rank_next_best_views,
)

__all__ = [
    'FREE', 'MINIMUM_USEFUL_MARGINAL_INFORMATION_FRACTION', 'SURFACE',
    'UNKNOWN', 'CoverageSnapshot', 'ObjectCoverageModel',
    'VoxelCoverageConfig', 'candidate_information',
    'candidate_meets_minimum_information', 'direction_bin',
    'persist_coverage_snapshot', 'rank_next_best_views',
]
