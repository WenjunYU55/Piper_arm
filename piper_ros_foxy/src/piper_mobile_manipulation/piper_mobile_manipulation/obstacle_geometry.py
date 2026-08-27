"""Compatibility facade for obstacle perception geometry."""

from piper_mobile_manipulation.perception.obstacle_geometry import (
    BLOCKED,
    HUMAN_LABELS,
    MOVABLE,
    UNSAFE,
    aabb_corners,
    canonical_label,
    effective_classification,
    normalize_label,
    obstacle_records,
    project_instance,
    target_occlusion_evidence,
    transform_points,
)

__all__ = [
    'BLOCKED', 'HUMAN_LABELS', 'MOVABLE', 'UNSAFE', 'aabb_corners',
    'canonical_label', 'effective_classification', 'normalize_label',
    'obstacle_records', 'project_instance', 'target_occlusion_evidence',
    'transform_points',
]
