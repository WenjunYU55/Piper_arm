"""Compatibility facade for occlusion and bounded-contact policy."""

from piper_mobile_manipulation.perception.occlusion import (
    HAND_LABELS,
    MOVABLE_LABELS,
    OccluderEvidence,
    canonical_label,
    evidence_rejection,
    placement_rejection,
    select_action,
)

__all__ = [
    'HAND_LABELS', 'MOVABLE_LABELS', 'OccluderEvidence', 'canonical_label',
    'evidence_rejection', 'placement_rejection', 'select_action',
]
