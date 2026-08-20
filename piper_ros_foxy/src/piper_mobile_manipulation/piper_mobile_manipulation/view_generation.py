"""Typed generation identity for closed-loop scan-view proposals."""

from dataclasses import dataclass


VIEW_SELECTION_POLICIES = (
    'legacy',
    'voxel_nbv_shadow',
    'voxel_nbv_seed',
    'voxel_nbv',
)


@dataclass(frozen=True)
class ViewGeneration:
    """Immutable identity and readiness of one planner output generation."""

    session_id: str
    accepted_views: int
    policy: str
    generation: int
    ready: bool
    candidate_viewpoints: int
    reason: str = ''

    def to_dict(self):
        """Return the additive JSON representation used on existing topics."""
        return {
            'schema_version': 1,
            'session_id': self.session_id,
            'accepted_views': self.accepted_views,
            'policy': self.policy,
            'generation': self.generation,
            'ready': self.ready,
            'candidate_viewpoints': self.candidate_viewpoints,
            'reason': self.reason,
        }


def make_view_generation(
        session_id, accepted_views, policy, generation, ready,
        candidate_viewpoints, reason=''):
    """Build and validate one immutable generation description."""
    value = ViewGeneration(
        session_id=str(session_id).strip(),
        accepted_views=int(accepted_views),
        policy=str(policy).strip(),
        generation=int(generation),
        ready=bool(ready),
        candidate_viewpoints=int(candidate_viewpoints),
        reason=str(reason),
    )
    if not value.session_id:
        raise ValueError('view generation session identity is missing')
    if value.accepted_views < 0 or value.generation < 0:
        raise ValueError('view generation counters must be non-negative')
    if value.candidate_viewpoints < 0:
        raise ValueError('view generation candidate count is negative')
    if value.policy not in VIEW_SELECTION_POLICIES:
        raise ValueError('view generation policy is unsupported')
    if value.generation != value.accepted_views:
        raise ValueError('view generation does not match accepted views')
    return value


def parse_view_generation(payload):
    """Parse the nested generation object from a planner/bridge JSON value."""
    if not isinstance(payload, dict):
        raise ValueError('view generation payload is not an object')
    source = payload.get('view_generation', payload)
    if not isinstance(source, dict):
        raise ValueError('view generation is not an object')
    if int(source.get('schema_version', 0)) != 1:
        raise ValueError('view generation schema version is unsupported')
    return make_view_generation(
        source.get('session_id', ''),
        source.get('accepted_views', -1),
        source.get('policy', ''),
        source.get('generation', -1),
        source.get('ready', False),
        source.get('candidate_viewpoints', -1),
        source.get('reason', ''),
    )


def generation_matches_expected(receipt, history, expected_accepted_views):
    """Return a reason until a bridge-cached generation matches the mission."""
    if not isinstance(history, dict):
        return 'scan history is unavailable'
    session_id = str(history.get('session_id', '')).strip()
    if not session_id:
        return 'scan history session identity is missing'
    try:
        accepted = int(history.get('accepted_views', -1))
        expected = int(expected_accepted_views)
    except (TypeError, ValueError):
        return 'scan history accepted-view count is invalid'
    if accepted != expected:
        return 'scan history generation is %d; waiting for %d' % (
            accepted, expected)
    try:
        generation = parse_view_generation(receipt)
    except (TypeError, ValueError) as error:
        return str(error)
    if generation.session_id != session_id:
        return 'view generation belongs to a different scan session'
    if generation.accepted_views != expected:
        return 'view generation is %d; waiting for %d' % (
            generation.accepted_views, expected)
    if generation.generation != expected:
        return 'view model generation is %d; waiting for %d' % (
            generation.generation, expected)
    if not generation.ready:
        return generation.reason or 'view generation is not ready'
    return ''
