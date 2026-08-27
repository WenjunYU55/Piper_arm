"""Compatibility facade for target acquisition policy."""

from piper_mobile_manipulation.perception.acquisition import (
    MULTIVIEW_SCAN,
    ROUGH_ACQUISITION,
    build_acquisition_viewpoints,
    rough_hint_rejection_reason,
    viewpoint_payload_matches,
)

__all__ = [
    'MULTIVIEW_SCAN', 'ROUGH_ACQUISITION', 'build_acquisition_viewpoints',
    'rough_hint_rejection_reason', 'viewpoint_payload_matches',
]
