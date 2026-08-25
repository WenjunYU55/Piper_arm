"""Candidate-policy selection kept independent of ROS bridge lifecycle."""

import math

import numpy as np

from piper_mobile_manipulation.target_envelope import validate_envelope
from piper_mobile_manipulation.view_generation import view_policy_capabilities
from piper_tesseract_foxy.contract import ContractError


RAY_DIRECTION_ATTEMPT_LIMIT = 6


def target_envelope_obstacles(scan, target_center):
    """Validate one frozen envelope and return its existing box contract."""
    supplied = scan.get('target_envelope') if isinstance(scan, dict) else None
    if supplied is None:
        return None, []
    try:
        envelope = validate_envelope(supplied)
    except ValueError as error:
        raise ContractError('target envelope is invalid: %s' % error)
    anchor = [float(value) for value in envelope['planning_anchor_m']]
    center = [float(value) for value in target_center]
    if any(
            abs(actual - expected) > 1e-6
            for actual, expected in zip(anchor, center)):
        raise ContractError(
            'target envelope planning anchor disagrees with tracked '
            'target center')
    envelope_hash = str(envelope.get('envelope_sha256', ''))
    if any(
            item.get('candidate_geometry') == 'target_ray'
            and str(item.get('target_envelope_sha256', '')) != envelope_hash
            for item in scan.get('viewpoints', [])
            if isinstance(item, dict)):
        raise ContractError('target ray is not bound to the frozen envelope')
    return envelope, [
        dict(item) for item in envelope['collision_boxes']]


def obstacle_scene_rejection_reason(scene):
    """Return a blocker only when collision geometry cannot be trusted."""
    if scene is None or not scene.scene_blocked:
        return None
    instances = list(scene.instances)
    invalid = [item for item in instances if not item.valid]
    if not instances or invalid:
        return 'obstacle scene is blocked: %s' % scene.blocking_reason
    return None


def uses_authoritative_nbv_order(candidates):
    """Return whether every candidate carries an active NBV policy."""
    if not candidates:
        return False
    policies = {
        str(item.get('view_selection_policy', '')).strip()
        for item in candidates}
    if len(policies) != 1:
        return False
    try:
        return view_policy_capabilities(policies.pop()).authoritative_nbv
    except ValueError:
        return False


def validate_candidate_policy_batch(candidates):
    """Fail closed when one multiview batch crosses policy/geometry seams."""
    if not candidates:
        return None
    policies = {
        str(item.get('view_selection_policy', '')).strip()
        for item in candidates}
    if len(policies) != 1:
        raise ContractError('candidate batch mixes view-selection policies')
    policy = policies.pop()
    try:
        capabilities = view_policy_capabilities(policy)
    except ValueError as error:
        raise ContractError(str(error))
    for item in candidates:
        geometry = str(item.get(
            'candidate_geometry', 'exact_point')).strip()
        if geometry != capabilities.candidate_geometry:
            raise ContractError(
                'candidate policy %s requires %s geometry, received %s'
                % (policy, capabilities.candidate_geometry, geometry))
        if capabilities.authoritative_nbv:
            try:
                rank = int(item.get('nbv_rank', 0))
                fraction = float(item.get(
                    'nbv_marginal_information_fraction', float('nan')))
            except (TypeError, ValueError):
                raise ContractError(
                    'authoritative NBV candidate score is malformed')
            if (
                    rank < 1 or not math.isfinite(fraction)
                    or fraction < 0.0 or fraction > 1.0):
                raise ContractError(
                    'authoritative NBV candidate score is invalid')
        if capabilities.ray_expansion:
            try:
                ray_id = int(item.get('ray_id', -1))
                direction = np.asarray(
                    item.get('ray_direction'), dtype=float)
                minimum = float(item.get('ray_min_standoff_m'))
                maximum = float(item.get('ray_max_standoff_m'))
                preferred = float(item.get(
                    'ray_preferred_max_standoff_m'))
            except (TypeError, ValueError):
                raise ContractError('target-ray candidate is malformed')
            if (
                    ray_id < 0 or direction.shape != (3,)
                    or not np.all(np.isfinite(direction))
                    or float(np.linalg.norm(direction)) <= 1e-9
                    or not all(math.isfinite(value) for value in (
                        minimum, maximum, preferred))
                    or minimum <= 0.0 or maximum < minimum
                    or preferred < minimum or preferred > maximum):
                raise ContractError('target-ray candidate is invalid')
    return capabilities


def bounded_candidate_attempt_limit(capabilities, configured_limit):
    """Keep exact-point behavior while bounding ray directions to six."""
    limit = max(1, int(configured_limit))
    if capabilities is not None and capabilities.ray_expansion:
        return min(limit, RAY_DIRECTION_ATTEMPT_LIMIT)
    return limit


def information_ranked_ray_candidates(candidates, candidate_limit):
    """Return the best ray directions without inserting travel fallbacks."""
    limit = max(1, int(candidate_limit))
    return sorted(
        (dict(item) for item in candidates),
        key=lambda item: (
            int(item.get('nbv_rank', 2 ** 31 - 1)),
            -float(item.get('coverage_score', 0.0)),
            int(item['id']),
        ),
    )[:limit]
