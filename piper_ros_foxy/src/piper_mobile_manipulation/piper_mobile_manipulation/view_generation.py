"""Compatibility facade for planning-generation identity."""

from piper_mobile_manipulation.planning.generation import (
    VIEW_POLICY_CAPABILITIES,
    VIEW_SELECTION_POLICIES,
    ViewGeneration,
    ViewPolicyCapabilities,
    generation_matches_expected,
    make_view_generation,
    parse_view_generation,
    view_policy_capabilities,
)

__all__ = [
    'VIEW_POLICY_CAPABILITIES', 'VIEW_SELECTION_POLICIES',
    'ViewGeneration', 'ViewPolicyCapabilities', 'generation_matches_expected',
    'make_view_generation', 'parse_view_generation',
    'view_policy_capabilities',
]
