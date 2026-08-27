"""ROS-free candidate validation, ranking, and path selection."""

import math

import numpy as np

from piper_mobile_manipulation.perception.target_envelope import (
    validate_envelope,
)
from piper_mobile_manipulation.planning.generation import (
    view_policy_capabilities,
)
from piper_tesseract_foxy.protocol.contract import ContractError


RAY_DIRECTION_ATTEMPT_LIMIT = 6
FINAL_AIM_EXECUTION_MARGIN_DEG = 1.0


def permanent_ray_ids_from_response(request, diagnostics):
    """Classify endpoint-static ray failures without changing the IK worker."""
    candidates = request.get('scene', {}).get('candidate_views', [])
    candidate_ids_by_ray = {}
    for candidate in candidates:
        if candidate.get('ray_id') is None:
            continue
        candidate_ids_by_ray.setdefault(int(candidate['ray_id']), set()).add(
            int(candidate['id']))
    failures = {
        int(item['id']): item
        for item in diagnostics.get('candidate_failures', [])
        if item.get('id') is not None
    }
    permanent = {
        int(value)
        for value in diagnostics.get('permanent_infeasible_ray_ids', [])
    }
    for ray_id, candidate_ids in candidate_ids_by_ray.items():
        if candidate_ids and all(
                candidate_id in failures
                and (
                    bool(failures[candidate_id].get(
                        'permanent_endpoint_failure', False))
                    or str(failures[candidate_id].get('stage', ''))
                    == 'RAY_IK_FAILURE')
                for candidate_id in candidate_ids):
            # RAY_IK_FAILURE is emitted only after the continuous ray solve
            # exhausts its bounded standoff, roll, exact-aim and fallback-aim
            # variants.  It is endpoint-static; path failures remain retryable.
            permanent.add(ray_id)
    return sorted(permanent)


def target_envelope_obstacles(scan, target_center):
    """Validate one frozen envelope and adapt it to the box contract."""
    supplied = scan.get('target_envelope') if isinstance(scan, dict) else None
    if supplied is None:
        return None, []
    try:
        envelope = validate_envelope(supplied)
    except ValueError as error:
        raise ContractError('target envelope is invalid: %s' % error)
    anchor = [float(value) for value in envelope['planning_anchor_m']]
    center = [float(value) for value in target_center]
    if any(abs(actual - expected) > 1e-6
           for actual, expected in zip(anchor, center)):
        raise ContractError(
            'target envelope planning anchor disagrees with tracked target '
            'center')
    envelope_hash = str(envelope.get('envelope_sha256', ''))
    if any(
            item.get('candidate_geometry') == 'target_ray'
            and str(item.get('target_envelope_sha256', '')) != envelope_hash
            for item in scan.get('viewpoints', [])
            if isinstance(item, dict)):
        raise ContractError('target ray is not bound to the frozen envelope')
    return envelope, [dict(item) for item in envelope['collision_boxes']]


def obstacle_scene_rejection_reason(scene):
    """
    Return a blocker only when collision geometry cannot be trusted.

    A valid object classified as unsafe is exactly the kind of obstacle the
    planning worker must receive and route around.  Invalid/missing geometry
    remains fail-closed.
    """
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


def local_view_frontier_candidates(
        candidates, start_camera_position, target_center,
        maximum_angular_step_deg, minimum_angular_step_deg=0.0):
    """Keep only views reached by one compact target-centred direction step."""
    start = np.asarray(start_camera_position, dtype=float)
    center = np.asarray(target_center, dtype=float)
    maximum = float(maximum_angular_step_deg)
    minimum = max(0.0, float(minimum_angular_step_deg))
    if (
            start.shape != (3,) or center.shape != (3,)
            or not np.all(np.isfinite(start))
            or not np.all(np.isfinite(center))
            or not math.isfinite(maximum) or maximum <= 0.0
            or not math.isfinite(minimum) or minimum >= maximum):
        raise ValueError('local view frontier inputs are invalid')
    reference = start - center
    reference_norm = float(np.linalg.norm(reference))
    if reference_norm <= 1e-6:
        raise ValueError('current camera position coincides with target center')
    reference /= reference_norm
    accepted = []
    for item in candidates:
        position = np.asarray(item.get('camera_position_m'), dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            continue
        direction = position - center
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            continue
        direction /= norm
        angle = math.degrees(math.acos(float(np.clip(
            np.dot(reference, direction), -1.0, 1.0))))
        if minimum - 1e-6 <= angle <= maximum + 1e-6:
            candidate = dict(item)
            candidate['_frontier_angle_deg'] = angle
            accepted.append(candidate)
    return sorted(
        accepted,
        key=lambda item: (
            -float(item.get('coverage_score', 0.0)),
            float(item['_frontier_angle_deg']),
            float(np.linalg.norm(
                np.asarray(item['camera_position_m'], dtype=float) - start)),
            int(item['id']),
        ),
    )


def bounded_current_look_direction(
        nominal_look_direction, current_look_direction,
        maximum_offset_deg):
    """
    Keep a new view's aim close to the achieved safe wrist orientation.

    Camera position owns surface coverage. Requiring every nearby position to
    use an exact target-centred orientation can nevertheless force a compact
    arm into wrist/link self-collision. This spherical interpolation preserves
    the measured current optical direction whenever it already lies inside the
    permitted target cone, otherwise it moves only far enough toward the target
    to meet the configured bound.
    """
    nominal = np.asarray(nominal_look_direction, dtype=float)
    current = np.asarray(current_look_direction, dtype=float)
    maximum = float(maximum_offset_deg)
    nominal_norm = float(np.linalg.norm(nominal))
    current_norm = float(np.linalg.norm(current))
    if (
            nominal.shape != (3,) or current.shape != (3,)
            or not np.all(np.isfinite(nominal))
            or not np.all(np.isfinite(current))
            or min(nominal_norm, current_norm) <= 1e-9
            or not math.isfinite(maximum)
            or maximum < 0.0 or maximum >= 90.0):
        raise ValueError('closed-loop aim-relaxation inputs are invalid')
    nominal /= nominal_norm
    current /= current_norm
    angle = math.acos(float(np.clip(np.dot(nominal, current), -1.0, 1.0)))
    maximum_rad = math.radians(maximum)
    if angle <= maximum_rad + 1e-12:
        return current.tolist()
    # An antiparallel current look cannot occur while the target is inside the
    # executor's live visibility cone. Retain exact target aim if malformed
    # upstream state somehow reaches this pure helper.
    sine = math.sin(angle)
    if abs(sine) <= 1e-9 or maximum_rad <= 0.0:
        return nominal.tolist()
    fraction = maximum_rad / angle
    relaxed = (
        math.sin((1.0 - fraction) * angle) / sine * nominal
        + math.sin(fraction * angle) / sine * current)
    relaxed /= np.linalg.norm(relaxed)
    return relaxed.tolist()


def relax_closed_loop_candidate_aims(
        candidates, current_look_direction, maximum_offset_deg):
    """Return exact request candidates with bounded current-look-biased aim."""
    relaxed = []
    for item in candidates:
        candidate = dict(item)
        candidate['look_direction'] = bounded_current_look_direction(
            candidate.get('look_direction'), current_look_direction,
            maximum_offset_deg)
        relaxed.append(candidate)
    return relaxed


def exact_target_aim_candidates(
        candidates, target_center, current_look_direction,
        maximum_fallback_offset_deg):
    """Bind exact target aim and at most one bounded fallback per candidate."""
    center = np.asarray(target_center, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError('target-centred aim requires a finite target center')
    result = []
    for item in candidates:
        candidate = dict(item)
        position = np.asarray(candidate.get('camera_position_m'), dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError('target-centred aim requires a finite camera position')
        nominal = center - position
        norm = float(np.linalg.norm(nominal))
        if norm <= 1e-9:
            raise ValueError('target-centred aim camera coincides with target')
        nominal /= norm
        # Leave physical following/calibration margin inside the public final
        # aim limit. Exact aim remains first and there is still only one
        # current-look-biased fallback.
        fallback_limit = max(
            0.0,
            float(maximum_fallback_offset_deg)
            - FINAL_AIM_EXECUTION_MARGIN_DEG,
        )
        fallback = np.asarray(bounded_current_look_direction(
            nominal, current_look_direction,
            fallback_limit), dtype=float)
        cosine = float(np.clip(np.dot(nominal, fallback), -1.0, 1.0))
        offset = math.degrees(math.acos(cosine))
        candidate['look_direction'] = nominal.tolist()
        candidate['fallback_look_directions'] = (
            [fallback.tolist()] if offset > 1e-6 else [])
        candidate['maximum_final_aim_offset_deg'] = float(
            maximum_fallback_offset_deg)
        result.append(candidate)
    return result


def balanced_closed_loop_candidates(
        candidates, start_camera_position, candidate_limit,
        compact_first=False, meaningful_progress=0.03):
    """
    Interleave ambitious coverage views with nearby IK fallbacks.

    Session history deliberately scores distant directions highest.  Taking a
    prefix of that ordering can remove every pose near the current, already
    proven reachable configuration before the worker gets a chance to solve
    IK.  Before any capture has established measured coverage, order the seed
    shortlist only by motion from the achieved camera pose.  Once history
    identifies a missing feature axis, the worker sees a block of materially
    advancing candidates before compact non-regressing fallbacks.  This
    retains IK escape candidates without letting comfortable radius/elevation
    variants consume the bounded feature-capture budget.
    """
    start = np.asarray(start_camera_position, dtype=float)
    if start.shape != (3,) or not np.all(np.isfinite(start)):
        raise ValueError('closed-loop camera start is invalid')
    limit = max(1, int(candidate_limit))
    if not candidates:
        return []
    coverage_order = sorted(
        (dict(item) for item in candidates),
        key=lambda item: (
            -float(item.get('coverage_score', 0.0)),
            -float(item.get('_frontier_angle_deg', 0.0)),
            float(np.linalg.norm(
                np.asarray(item['camera_position_m'], dtype=float) - start)),
            int(item['id']),
        ),
    )
    leader_position = np.asarray(
        coverage_order[0]['camera_position_m'], dtype=float)
    # Planner history provides an objective-specific margin against achieved
    # coverage.  This admits elevation/radius changes that can escape a hard
    # IK branch while excluding genuine regression toward an already covered
    # face.  Legacy/manual candidates without the margin remain unfiltered.
    progress_candidates = [
        item for item in coverage_order
        if 'coverage_progress_score' not in item
        or float(item['coverage_progress_score']) >= -1e-9
    ]
    coverage_order = progress_candidates

    def camera_travel(item):
        position = np.asarray(item['camera_position_m'], dtype=float)
        if item.get('candidate_geometry') != 'target_ray':
            return float(np.linalg.norm(position - start))
        try:
            direction = np.asarray(item['ray_direction'], dtype=float)
            direction_norm = float(np.linalg.norm(direction))
            scoring_standoff = float(item['ray_scoring_standoff_m'])
            minimum = float(item['ray_min_standoff_m'])
            maximum = float(item['ray_max_standoff_m'])
            if (
                    direction.shape != (3,)
                    or not np.all(np.isfinite(direction))
                    or not math.isfinite(direction_norm)
                    or direction_norm <= 1e-9
                    or not all(math.isfinite(value) for value in (
                        scoring_standoff, minimum, maximum))
                    or minimum > maximum):
                raise ValueError('invalid bounded target ray')
            direction /= direction_norm
            target = position - direction * scoring_standoff
            closest_standoff = float(np.clip(
                np.dot(start - target, direction), minimum, maximum))
            closest = target + direction * closest_standoff
            return float(np.linalg.norm(closest - start))
        except (KeyError, TypeError, ValueError, FloatingPointError):
            return float(np.linalg.norm(position - start))

    fallback_order = sorted(
        (dict(item) for item in progress_candidates),
        key=lambda item: (
            camera_travel(item),
            float(np.linalg.norm(
                np.asarray(item['camera_position_m'], dtype=float)
                - leader_position)),
            float(item.get('_frontier_angle_deg', 0.0)),
            -float(item.get('coverage_score', 0.0)),
            int(item['id']),
        ),
    )
    has_progress_contract = any(
        'coverage_progress_score' in item for item in coverage_order)

    def progress_threshold(item):
        objective = str(item.get('coverage_objective', ''))
        if objective in ('positive_y_face', 'negative_y_face'):
            # scan_session_memory includes 0.02 lateral noise tolerance in the
            # score, so 0.05 proves at least 0.03 normalized side advance.
            return 0.05
        if objective in ('azimuth_span', 'elevation_span'):
            return 2.0
        return float(meaningful_progress)

    materially_advancing = [
        item for item in coverage_order
        if float(item.get('coverage_progress_score', 0.0))
        >= progress_threshold(item)
    ]
    if compact_first:
        return fallback_order[:min(limit, len(fallback_order))]
    selected = []
    selected_ids = set()
    target_count = min(limit, len(progress_candidates))
    for coverage, fallback in zip(coverage_order, fallback_order):
        pair = (coverage, fallback)
        for item in pair:
            item_id = int(item['id'])
            if item_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item_id)
            if len(selected) >= target_count:
                break
        if len(selected) >= target_count:
            break
    if has_progress_contract and materially_advancing and not compact_first:
        # Preserve the proven balanced shortlist membership: changing it can
        # discard the only reachable radius/elevation IK escape.  Reorder that
        # same bounded set so every materially feature-advancing member is
        # attempted before near-zero-progress fallbacks.
        advancing_ids = {int(item['id']) for item in materially_advancing}
        return (
            [item for item in selected if int(item['id']) in advancing_ids]
            + [item for item in selected if int(item['id']) not in advancing_ids]
        )
    return selected


def bounded_nbv_candidates(
        candidates, start_camera_position, target_center, candidate_limit,
        leader_fraction=0.5, direction_bin_deg=30.0):
    """
    Bound an information-first shortlist while retaining direction spread.

    The voxel planner has already compared every candidate using measured
    coverage. Tesseract still needs a bounded number of IK attempts. Selecting
    a plain rank prefix can fill that budget with different radii from one
    direction and hide the best candidate from every other visible face.
    Choose the best-ranked member of distinct azimuth/elevation bins first.
    A bin spans several samples from the 7.5-degree candidate grid so its
    mid-elevation escape can shift slightly in azimuth when the exact
    high-gain ray has no IK branch.
    Use the reserved fallback slots for mid-elevation poses in those same
    informative azimuth sectors before retaining a generic nearby fallback.
    This keeps a steep high-gain direction from collapsing back to the current
    azimuth merely because its first elevation has no feasible IK solution.
    The final attempt order remains NBV rank, so Tesseract still executes the
    highest-information feasible member.
    """
    start = np.asarray(start_camera_position, dtype=float)
    center = np.asarray(target_center, dtype=float)
    if start.shape != (3,) or not np.all(np.isfinite(start)):
        raise ValueError('NBV camera start is invalid')
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError('NBV target center is invalid')
    limit = max(1, int(candidate_limit))
    fraction = float(leader_fraction)
    bin_width = float(direction_bin_deg)
    if not math.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
        raise ValueError('NBV leader fraction is invalid')
    if not math.isfinite(bin_width) or bin_width <= 0.0 or bin_width > 90.0:
        raise ValueError('NBV direction bin size is invalid')
    ordered = sorted(
        (dict(item) for item in candidates),
        key=lambda item: (
            int(item.get('nbv_rank', 2 ** 31 - 1)),
            -float(item.get('coverage_score', 0.0)),
            int(item['id']),
        ),
    )
    target_count = min(limit, len(ordered))
    leader_count = min(
        target_count, max(1, int(math.ceil(target_count * fraction))))

    def direction_angles(item):
        position = np.asarray(item.get('camera_position_m'), dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            return None
        direction = position - center
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            return None
        direction /= norm
        return (
            math.degrees(math.atan2(direction[1], direction[0])),
            math.degrees(math.asin(float(np.clip(
                direction[2], -1.0, 1.0)))),
        )

    start_direction = start - center
    start_direction_norm = float(np.linalg.norm(start_direction))
    start_elevation = 0.0
    if start_direction_norm > 1e-9:
        start_direction /= start_direction_norm
        start_elevation = math.degrees(math.asin(float(np.clip(
            start_direction[2], -1.0, 1.0))))
    candidate_elevations = [
        angles[1] for angles in (direction_angles(item) for item in ordered)
        if angles is not None
    ]
    middle_elevation = (
        float(np.median(candidate_elevations))
        if candidate_elevations else start_elevation)

    selected = []
    direction_bins = set()
    for item in ordered:
        angles = direction_angles(item)
        if angles is None:
            continue
        azimuth, elevation = angles
        direction_bin = (
            int(math.floor((azimuth + 180.0) / bin_width)),
            int(math.floor((elevation + 90.0) / bin_width)),
        )
        if direction_bin in direction_bins:
            continue
        selected.append(item)
        direction_bins.add(direction_bin)
        if len(selected) >= leader_count:
            break
    if len(selected) < leader_count:
        selected_ids = {int(item['id']) for item in selected}
        selected.extend(
            item for item in ordered
            if int(item['id']) not in selected_ids
        )
        selected = selected[:leader_count]
    selected_ids = {int(item['id']) for item in selected}
    fallback_slots = target_count - len(selected)
    # Keep one informative continuity candidate close to the achieved physical
    # camera pose.  Previously every fallback slot could be consumed by
    # ambitious sector/elevation alternatives; the live 2026-08-20 run then
    # exhausted twelve IK-infeasible views even though nearby positive-gain
    # candidates had valid IK branches.  Information leaders still run first
    # because the returned shortlist remains sorted by authoritative NBV rank.
    continuity_slots = 1 if fallback_slots > 0 else 0
    sector_slots = max(0, fallback_slots - continuity_slots)
    sector_fallbacks = []
    sector_fallback_ids = set()
    for leader in selected:
        if len(sector_fallbacks) >= sector_slots:
            break
        leader_angles = direction_angles(leader)
        if leader_angles is None:
            continue
        leader_azimuth, leader_elevation = leader_angles
        leader_azimuth_bin = int(math.floor(
            (leader_azimuth + 180.0) / bin_width))
        alternatives = []
        for item in ordered:
            item_id = int(item['id'])
            if item_id in selected_ids or item_id in sector_fallback_ids:
                continue
            angles = direction_angles(item)
            if angles is None:
                continue
            azimuth, elevation = angles
            if int(math.floor(
                    (azimuth + 180.0) / bin_width)) != leader_azimuth_bin:
                continue
            elevation_change = abs(elevation - leader_elevation)
            alternatives.append((
                # Prefer a genuinely different camera elevation over only a
                # radial duplicate of the failed pose.  Then prefer the middle
                # of the configured elevation region, not the achieved
                # elevation: a live run at 65 degrees otherwise retained only
                # the same unreachable 65/75-degree IK band.  Camera travel is
                # a tie-break inside the leader's information-bearing azimuth
                # sector; it cannot displace a global information leader.
                0 if elevation_change >= 5.0 else 1,
                abs(elevation - middle_elevation),
                float(np.linalg.norm(
                    np.asarray(item['camera_position_m'], dtype=float)
                    - start)),
                abs(azimuth - leader_azimuth),
                int(item.get('nbv_rank', 2 ** 31 - 1)),
                item_id,
                item,
            ))
        if not alternatives:
            continue
        fallback_item = min(alternatives)[-1]
        sector_fallbacks.append(fallback_item)
        sector_fallback_ids.add(int(fallback_item['id']))

    selected.extend(sector_fallbacks)
    selected_ids = {int(item['id']) for item in selected}
    fallbacks = sorted(
        (item for item in ordered if int(item['id']) not in selected_ids),
        key=lambda item: (
            float(np.linalg.norm(
                np.asarray(item['camera_position_m'], dtype=float) - start)),
            int(item.get('nbv_rank', 2 ** 31 - 1)),
            int(item['id']),
        ),
    )
    selected.extend(fallbacks[:target_count - len(selected)])
    return sorted(
        selected,
        key=lambda item: (
            int(item.get('nbv_rank', 2 ** 31 - 1)),
            -float(item.get('coverage_score', 0.0)),
            int(item['id']),
        ),
    )


def select_diverse_smooth_view_path(
        candidates, selected_count=None, start_camera_position=None):
    """
    Select a camera-space-diverse subset, then order it as a smooth route.

    Maximizing every consecutive baseline made a one-dimensional orbit
    alternate between its two endpoints.  Farthest-point sampling still
    spreads the selected captures across the full candidate dome, while a
    nearest-neighbour traversal avoids that pendulum motion.  Unselected
    candidates remain at the end as deterministic IK fallbacks for the worker.
    """
    remaining = [dict(item) for item in candidates]
    if not remaining:
        return remaining
    count = (
        len(remaining)
        if selected_count is None
        else max(1, min(int(selected_count), len(remaining))))
    start = None
    if start_camera_position is not None:
        candidate = np.asarray(start_camera_position, dtype=float)
        if candidate.shape == (3,) and np.all(np.isfinite(candidate)):
            start = candidate

    selected = []
    if count == 1 and any(
            'coverage_score' in item for item in remaining):
        first_index = min(
            range(len(remaining)),
            key=lambda index: (
                -float(remaining[index].get('coverage_score', 0.0)),
                float(np.linalg.norm(
                    np.asarray(
                        remaining[index]['camera_position_m'], dtype=float)
                    - start)) if start is not None else 0.0,
                int(remaining[index]['id']),
            ))
    elif start is None:
        first_index = min(
            range(len(remaining)),
            key=lambda index: int(remaining[index]['id']))
    else:
        first_index = min(
            range(len(remaining)),
            key=lambda index: (
                float(np.linalg.norm(
                    np.asarray(
                        remaining[index]['camera_position_m'], dtype=float)
                    - start)),
                int(remaining[index]['id']),
            ))
    selected.append(remaining.pop(first_index))

    while remaining and len(selected) < count:
        index = max(
            range(len(remaining)),
            key=lambda candidate_index: (
                min(
                    float(np.linalg.norm(
                        np.asarray(
                            remaining[candidate_index]['camera_position_m'],
                            dtype=float)
                        - np.asarray(
                            reference['camera_position_m'], dtype=float)))
                    for reference in selected),
                -int(remaining[candidate_index]['id']),
            ))
        selected.append(remaining.pop(index))

    ordered = []
    previous = start
    while selected:
        if previous is None:
            index = min(
                range(len(selected)),
                key=lambda item_index: int(selected[item_index]['id']))
        else:
            index = min(
                range(len(selected)),
                key=lambda item_index: (
                    float(np.linalg.norm(
                        np.asarray(
                            selected[item_index]['camera_position_m'],
                            dtype=float) - previous)),
                    int(selected[item_index]['id']),
                ))
        chosen = selected.pop(index)
        ordered.append(chosen)
        previous = np.asarray(chosen['camera_position_m'], dtype=float)

    # Preserve unselected candidates as worker fallbacks. For a one-view
    # closed-loop request keep coverage priority first and use camera travel
    # only as the tie-break; later observations will score the next request.
    if count == 1 and any('coverage_score' in item for item in remaining):
        remaining = sorted(
            remaining,
            key=lambda item: (
                -float(item.get('coverage_score', 0.0)),
                float(np.linalg.norm(
                    np.asarray(item['camera_position_m'], dtype=float) - start))
                if start is not None else 0.0,
                int(item['id']),
            ))
    else:
        remaining = sorted(remaining, key=lambda item: int(item['id']))
    return ordered + remaining


def maximize_successive_view_distance(candidates):
    """Compatibility alias for callers outside the bridge."""
    return select_diverse_smooth_view_path(candidates)
