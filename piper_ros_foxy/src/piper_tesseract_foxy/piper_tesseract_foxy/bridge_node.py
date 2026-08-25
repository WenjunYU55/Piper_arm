#!/usr/bin/env python3
"""Command-free Foxy bridge for the isolated Tesseract plan worker."""

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Vector3
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from piper_msgs.msg import PiperMotionLimits

from piper_mobile_manipulation.msg import (
    CameraTimestampHealth,
    ObstacleInstance3DArray,
    TesseractPlan,
    TesseractReadiness,
    TesseractPlanStatus,
    TrackingHealth,
)
from piper_mobile_manipulation.scan_motion import (
    load_accepted_hand_eye,
    load_conservative_joint_limits,
    PiperScanKinematics,
)
from piper_mobile_manipulation.scan_execution_modes import (
    commanded_speed_percent,
)
from piper_mobile_manipulation.view_generation import (
    parse_view_generation,
    view_policy_capabilities,
)
from piper_mobile_manipulation.viewpoint_rays import (
    bind_shortlisted_ray_intervals,
)
from piper_mobile_manipulation.motion_limit_stability import MotionLimitStability
from piper_mobile_manipulation.srv import RequestTesseractPlan

from piper_tesseract_foxy.contract import (
    attach_digest,
    ContractError,
    JOINT_NAMES,
    MAX_CONFIGURED_HOME_START_LIMIT_TOLERANCE_RAD,
    PLAN_KINDS,
    SCHEMA_VERSION,
    TIMING_POLICY,
    sha256_file,
    Spool,
    validate_response,
)


RAY_DIRECTION_ATTEMPT_LIMIT = 6
FINAL_AIM_EXECUTION_MARGIN_DEG = 1.0


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


class TesseractPlanBridge(Node):
    """Freezes live inputs and publishes validated, motion-free proposals."""

    def __init__(self):
        super().__init__('tesseract_plan_bridge')
        defaults = {
            'reachable_viewpoints_topic': '/piper/reachable_scan_viewpoints',
            'reachable_acquisition_viewpoints_topic':
                '/piper/reachable_acquisition_viewpoints',
            'joint_states_topic': '/joint_states_single',
            'motion_limits_topic': '/piper/motion_limits',
            'tracking_health_topic': '/piper/tracking_health',
            'camera_timestamp_health_topic': '/piper/camera_timestamp_health',
            'obstacle_topic': '/piper/obstacle_instances_3d',
            'plan_topic': '/piper/tesseract_plan',
            'view_generation_receipt_topic':
                '/piper/tesseract_view_generation',
            'plan_provenance_topic': '/piper/tesseract_plan_provenance',
            'status_topic': '/piper/tesseract_plan_status',
            'readiness_topic': '/piper/tesseract_readiness',
            'spool_root': '/tmp/piper_tesseract_plans',
            'hand_eye_calibration_path': '',
            'joint_bounds_path': '',
            'robot_xacro_path': '',
            'srdf_path': '',
            'collision_manifest_path': '',
            'data_timeout_sec': 1.0,
            'motion_limits_timeout_sec': 3.0,
            'motion_limits_change_confirmation_sec': 7.0,
            'motion_limits_change_minimum_samples': 3,
            'worker_heartbeat_timeout_sec': 1.5,
            'request_ttl_sec': 180.0,
            'response_timeout_sec': 180.0,
            'max_tracking_measurement_age_sec': 0.75,
            'max_execution_viewpoints': 13,
            'joint_limit_margin_rad': 0.03,
            'trajectory_joint_step_rad': 0.05,
            'trajectory_command_rate_hz': 20.0,
            'speed_percent': 5.0,
            'roll_samples_rad': [-2.094395102, -1.047197551, 0.0,
                                 1.047197551, 2.094395102, 3.141592654],
            'deterministic_seed': 42,
            'return_home_positions_rad': [
                0.000366362, 0.0, 0.0, 0.0, 0.43869236, 0.0,
            ],
            'closed_loop_one_view': False,
            # From an achieved pose between fixed 15-degree grid samples, the
            # next grid neighbor can be only about seven degrees away in 3D
            # target-direction space at normal elevation. Six degrees admits
            # that visually distinct neighbor while the
            # independent pose/look duplicate gate still rejects repeats.
            'closed_loop_min_view_step_deg': 6.0,
            'closed_loop_max_view_step_deg': 30.0,
            'closed_loop_candidate_limit': 12,
            # Preserve a comfortable achieved wrist aim when the target stays
            # within this strict subset of the executor's 20-degree cone.
            'closed_loop_max_aim_offset_deg': 5.0,
            'manipulation_model_qualified': False,
            'debug': True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        required = [
            'hand_eye_calibration_path', 'joint_bounds_path', 'robot_xacro_path',
            'srdf_path', 'collision_manifest_path',
        ]
        missing = [name for name in required if not self.parameter_path(name).is_file()]
        if missing:
            raise RuntimeError('missing Tesseract bridge assets: ' + ', '.join(missing))

        self.spool = Spool(self.get_parameter('spool_root').value)
        self.hand_eye = load_accepted_hand_eye(
            str(self.parameter_path('hand_eye_calibration_path')))
        self.kinematics = PiperScanKinematics(self.hand_eye)
        self.joint_limits, self.ignored_bounds = load_conservative_joint_limits(
            str(self.parameter_path('joint_bounds_path')))
        self.boot_id = self.read_boot_id()

        self.latest_scan = None
        self.latest_acquisition_scan = None
        self.latest_joints = None
        self.latest_motion_limits = None
        self.motion_limit_stability = MotionLimitStability(
            self.get_parameter(
                'motion_limits_change_confirmation_sec').value,
            self.get_parameter(
                'motion_limits_change_minimum_samples').value,
        )
        self.latest_tracking = None
        self.latest_camera_health = None
        self.latest_obstacles = None
        self.updated = {}
        self.pending = {}
        self.tesseract_exhausted_ray_generation = None
        self.tesseract_exhausted_ray_ids = set()
        self.remaining_ray_pool_session = None
        self.remaining_ray_ids = set()
        self.retired_ray_ids = set()
        self.state = 'IDLE'
        self.reason = 'waiting for an explicit plan request'
        self.worker_generation_id = ''

        self.plan_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.plan_pub = self.create_publisher(
            TesseractPlan, self.get_parameter('plan_topic').value, self.plan_qos)
        self.view_generation_pub = self.create_publisher(
            String,
            self.get_parameter('view_generation_receipt_topic').value,
            self.plan_qos)
        self.plan_provenance_pub = self.create_publisher(
            String, self.get_parameter('plan_provenance_topic').value,
            QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ))
        self.status_pub = self.create_publisher(
            TesseractPlanStatus, self.get_parameter('status_topic').value, 10)
        self.readiness_pub = self.create_publisher(
            TesseractReadiness,
            self.get_parameter('readiness_topic').value,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
        self.scan_sub = self.create_subscription(
            String, self.get_parameter('reachable_viewpoints_topic').value,
            self.scan_cb, 10)
        self.acquisition_scan_sub = self.create_subscription(
            String,
            self.get_parameter('reachable_acquisition_viewpoints_topic').value,
            self.acquisition_scan_cb,
            10,
        )
        self.joint_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.joint_sub = self.create_subscription(
            JointState,
            self.get_parameter('joint_states_topic').value,
            self.joint_cb,
            self.joint_qos,
        )
        self.motion_limits_sub = self.create_subscription(
            PiperMotionLimits,
            self.get_parameter('motion_limits_topic').value,
            self.motion_limits_cb,
            10,
        )
        self.tracking_sub = self.create_subscription(
            TrackingHealth, self.get_parameter('tracking_health_topic').value,
            self.tracking_cb, 10)
        self.camera_health_sub = self.create_subscription(
            CameraTimestampHealth,
            self.get_parameter('camera_timestamp_health_topic').value,
            self.camera_health_cb, 10)
        self.obstacle_sub = self.create_subscription(
            ObstacleInstance3DArray,
            self.get_parameter('obstacle_topic').value,
            self.obstacle_cb,
            10,
        )
        self.request_service = self.create_service(
            RequestTesseractPlan, '~/request_plan', self.request_plan_cb)
        self.request_acquisition_service = self.create_service(
            RequestTesseractPlan, '~/request_acquisition_plan',
            self.request_acquisition_plan_cb)
        self.request_return_home_service = self.create_service(
            RequestTesseractPlan, '~/request_return_home_plan',
            self.request_return_home_plan_cb)
        self.request_startup_home_service = self.create_service(
            RequestTesseractPlan, '~/request_startup_home_plan',
            self.request_startup_home_plan_cb)
        self.poll_timer = self.create_timer(0.20, self.poll)
        self.publish_status()
        self.publish_readiness()
        self.get_logger().warn(
            'Tesseract bridge is command-free: it has no /joint_ctrl_single publisher '
            'and no motor-enable client. Ignored saved bounds: %s'
            % (','.join(self.ignored_bounds) or 'none'))

    def parameter_path(self, name):
        return Path(str(self.get_parameter(name).value)).resolve()

    def param_bool(self, name):
        """Return one declared ROS parameter using the shared bool contract."""
        value = self.get_parameter(name).value
        if isinstance(value, str):
            return value.lower() in ('1', 'true', 'yes', 'on')
        return bool(value)

    @staticmethod
    def read_boot_id():
        try:
            return Path('/proc/sys/kernel/random/boot_id').read_text(encoding='utf-8').strip()
        except OSError:
            return 'unavailable'

    def now(self):
        return time.monotonic()

    def mark(self, key):
        self.updated[key] = self.now()

    def fresh(self, key, timeout=None):
        maximum = (
            float(self.get_parameter('data_timeout_sec').value)
            if timeout is None else float(timeout))
        return self.now() - self.updated.get(key, -1e9) <= maximum

    def scan_cb(self, msg):
        self.store_scan(msg, 'scan', acquisition=False)

    def acquisition_scan_cb(self, msg):
        self.store_scan(msg, 'acquisition_scan', acquisition=True)

    def store_scan(self, msg, key, acquisition):
        try:
            value = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            return
        if isinstance(value, dict):
            if acquisition:
                self.latest_acquisition_scan = value
            else:
                self.latest_scan = value
            self.mark(key)
            if not acquisition:
                try:
                    generation = parse_view_generation(value)
                except (TypeError, ValueError):
                    return
                receipt = String()
                receipt.data = json.dumps({
                    'bridge_received_at_ns': time.time_ns(),
                    'view_generation': generation.to_dict(),
                }, sort_keys=True)
                self.view_generation_pub.publish(receipt)
                if self.param_bool('debug'):
                    self.get_logger().info(
                        'cached view generation session=%s accepted=%d '
                        'policy=%s ready=%s candidates=%d'
                        % (
                            generation.session_id,
                            generation.accepted_views,
                            generation.policy,
                            generation.ready,
                            generation.candidate_viewpoints))

    def joint_cb(self, msg):
        self.latest_joints = msg
        self.mark('joints')

    def tracking_cb(self, msg):
        self.latest_tracking = msg
        self.mark('tracking')

    def camera_health_cb(self, msg):
        self.latest_camera_health = msg
        self.mark('camera_clock')

    def obstacle_cb(self, msg):
        self.latest_obstacles = msg
        self.mark('obstacles')

    def worker_health_reasons(self):
        try:
            health = self.spool.read_health()
        except (ContractError, FileNotFoundError, OSError, TypeError, ValueError):
            self.worker_generation_id = ''
            return ['Tesseract worker heartbeat is missing or invalid']
        generation = health.get('generation_id')
        written_at_ns = health.get('written_at_ns')
        if (
                not isinstance(generation, str)
                or len(generation) != 32
                or any(character not in '0123456789abcdef'
                       for character in generation)):
            self.worker_generation_id = ''
            return ['Tesseract worker generation ID is invalid']
        try:
            age_sec = (time.time_ns() - int(written_at_ns)) / 1e9
        except (TypeError, ValueError):
            self.worker_generation_id = ''
            return ['Tesseract worker heartbeat timestamp is invalid']
        timeout = float(
            self.get_parameter('worker_heartbeat_timeout_sec').value)
        if age_sec < -1.0 or age_sec > timeout:
            self.worker_generation_id = generation
            return ['Tesseract worker heartbeat is stale']
        if (
                health.get('schema_version') != SCHEMA_VERSION
                or health.get('backend') != 'tesseract'
                or health.get('worker_ready') is not True):
            self.worker_generation_id = generation
            detail = str(health.get('backend_error', '')).strip()
            return [
                'Tesseract worker is not ready'
                + (': ' + detail if detail else '')
            ]
        expected_hashes = {
            'srdf_sha256': sha256_file(self.parameter_path('srdf_path')),
            'collision_manifest_sha256': sha256_file(
                self.parameter_path('collision_manifest_path')),
        }
        mismatches = [
            name for name, expected in expected_hashes.items()
            if health.get(name) != expected
        ]
        if mismatches:
            self.worker_generation_id = generation
            return [
                'Tesseract worker collision profile does not match bridge: '
                + ', '.join(mismatches)
            ]
        self.worker_generation_id = generation
        return []

    def snapshot_reasons(
            self, plan_kind='MULTIVIEW_SCAN', require_viewpoints=True,
            worker_reasons=None, startup_home=False):
        if plan_kind not in PLAN_KINDS:
            return ['unsupported plan kind']
        if startup_home and plan_kind != 'RETURN_HOME':
            return ['startup home is RETURN_HOME-only']
        reasons = (
            self.worker_health_reasons()
            if worker_reasons is None else list(worker_reasons))
        # Dedicated RETURN_HOME plans are direct configured joint targets and
        # intentionally do not consume perception or collision-scene state.
        required = ['joints']
        if plan_kind != 'RETURN_HOME':
            required.append('camera_clock')
        if plan_kind == 'MULTIVIEW_SCAN':
            required.extend(['tracking', 'obstacles'])
            if require_viewpoints:
                required.append('scan')
        elif plan_kind != 'RETURN_HOME' and require_viewpoints:
            required.append('acquisition_scan')
        for key in required:
            if not self.fresh(key):
                reasons.append('%s data is missing or stale' % key)
        if not self.fresh(
                'motion_limits',
                float(self.get_parameter('motion_limits_timeout_sec').value)):
            reasons.append('controller motion limits are missing or stale')
        limits = self.latest_motion_limits
        if limits is None or not limits.valid:
            reasons.append(
                'controller motion limits are invalid: %s'
                % (limits.reason if limits is not None else 'no message'))
        elif (
                list(limits.joint_names) != list(JOINT_NAMES)
                or len(limits.max_velocity_rad_s) != 6
                or len(limits.max_acceleration_rad_s2) != 6
                or len(str(limits.limits_sha256)) != 64):
            reasons.append('controller motion-limit payload is malformed')
        if self.latest_joints is None or len(self.latest_joints.position) < 6:
            reasons.append('joint feedback has fewer than six positions')
        else:
            values = np.asarray(self.latest_joints.position[:6], dtype=float)
            if not np.all(np.isfinite(values)):
                reasons.append('joint feedback is non-finite')
        if plan_kind == 'MULTIVIEW_SCAN':
            health = self.latest_tracking
            if health is None or health.lifecycle_state != 'TRACKING' \
                    or not health.camera_settled:
                reasons.append('tracking is not settled TRACKING')
            elif health.prediction_only:
                reasons.append('tracking is prediction-only')
            elif float(health.measurement_age_sec) > float(
                    self.get_parameter('max_tracking_measurement_age_sec').value):
                reasons.append('tracking measurement is stale')
        if (
                plan_kind != 'RETURN_HOME'
                and (
                    self.latest_camera_health is None
                    or not self.latest_camera_health.healthy)):
            reasons.append('camera timestamp health is not healthy')
        if plan_kind == 'MULTIVIEW_SCAN':
            obstacle_reason = obstacle_scene_rejection_reason(self.latest_obstacles)
            if obstacle_reason:
                reasons.append(obstacle_reason)
        scan = (
            self.latest_scan if plan_kind == 'MULTIVIEW_SCAN'
            else self.latest_acquisition_scan)
        if plan_kind != 'RETURN_HOME' and require_viewpoints and (
                scan is None or scan.get('dry_run') is not True):
            reasons.append('reachable viewpoint source is not explicit dry-run data')
        if plan_kind == 'MULTIVIEW_SCAN' and require_viewpoints:
            session = scan.get('scan_session', {}) if isinstance(scan, dict) else {}
            try:
                session_id = str(session.get('session_id', ''))
                accepted = int(session.get('accepted_views', -1))
                maximum = int(session.get('max_views', -1))
                remaining = int(
                    scan.get('remaining_viewpoints', -1)
                    if isinstance(scan, dict) else -1)
            except (TypeError, ValueError):
                session_id, accepted, maximum, remaining = '', -1, -1, -1
            configured = int(
                self.get_parameter('max_execution_viewpoints').value)
            if not session_id:
                reasons.append('scan session identity is missing')
            if maximum != configured:
                reasons.append('scan session maximum does not match configuration')
            if accepted < 0 or accepted >= maximum:
                reasons.append('scan session has no valid remaining viewpoints')
            if remaining != maximum - accepted:
                reasons.append('scan session remaining count is inconsistent')
            selection_failure = str(
                scan.get('selection_failure_code', '')
                if isinstance(scan, dict) else '').strip()
            if selection_failure:
                reasons.append(selection_failure)
            elif not (
                    isinstance(scan, dict)
                    and scan.get('viewpoints', [])):
                reasons.append('NO_PREQUALIFIED_VIEWPOINT_CANDIDATE')
            elif not any(
                    isinstance(item, dict)
                    and item.get(
                        'prequalified', item.get('reachable')) is True
                    and item.get('safe') is True
                    for item in scan.get('viewpoints', [])):
                # Preserve the reason supplied by the command-free
                # prequalification stage.  In particular, a transient target
                # status dip must enter the mission's existing bounded visual
                # reacquisition hold instead of being flattened later into the
                # misleading "only 0 safe candidates" request-build error.
                target_status = str(
                    scan.get('filter', {}).get('target_status', '')
                    if isinstance(scan.get('filter'), dict) else '').strip()
                if target_status in (
                        'LOW_CONFIDENCE', 'LOST', 'SEARCHING'):
                    reasons.append('target_status=' + target_status)
                else:
                    reasons.append('NO_PREQUALIFIED_VIEWPOINT_CANDIDATE')
        return list(dict.fromkeys(reasons))

    def request_plan_cb(self, request, response):
        return self.request_kind_cb(
            'MULTIVIEW_SCAN', request, response)

    def request_acquisition_plan_cb(self, request, response):
        return self.request_kind_cb(
            'ROUGH_ACQUISITION', request, response)

    def request_return_home_plan_cb(self, request, response):
        return self.request_kind_cb('RETURN_HOME', request, response)

    def request_startup_home_plan_cb(self, request, response):
        return self.request_kind_cb(
            'RETURN_HOME', request, response, startup_home=True)

    def request_kind_cb(
            self, plan_kind, request, response, startup_home=False):
        if self.pending and not request.force_refresh:
            response.accepted = False
            response.request_id = next(iter(self.pending))
            response.message = 'a Tesseract request is already pending'
            return response
        reasons = self.snapshot_reasons(
            plan_kind, startup_home=startup_home)
        if reasons:
            response.accepted = False
            response.request_id = ''
            response.message = 'planning blocked: ' + '; '.join(reasons)
            state = (
                'NO_POSITIVE_INFORMATION_CANDIDATE'
                if 'NO_POSITIVE_INFORMATION_CANDIDATE' in reasons
                else 'SNAPSHOT_BLOCKED')
            self.set_status(state, response.message)
            return response
        try:
            home_stage = str(getattr(request, 'home_stage', '')).strip()
            joint_goal = [
                float(value) for value in
                getattr(request, 'joint_goal_positions_rad', [])]
            payload = self.build_request(
                plan_kind, startup_home=startup_home,
                home_stage=home_stage, joint_goal=joint_goal)
            self.spool.write('requests', payload['request_id'], payload)
        except (ContractError, KeyError, OSError, TypeError, ValueError) as error:
            response.accepted = False
            response.request_id = ''
            response.message = 'request creation failed: %s' % error
            self.set_status('REJECTED', response.message)
            return response
        self.pending[payload['request_id']] = {
            'request': payload,
            'started': self.now(),
        }
        response.accepted = True
        response.request_id = payload['request_id']
        response.message = 'command-free Tesseract planning request queued'
        self.set_status('PLANNING', response.message, payload['request_id'])
        return response

    def build_request(
            self, plan_kind='MULTIVIEW_SCAN', startup_home=False,
            home_stage='', joint_goal=None):
        if plan_kind not in PLAN_KINDS:
            raise ContractError('unsupported plan kind')
        if startup_home and plan_kind != 'RETURN_HOME':
            raise ContractError('startup home is RETURN_HOME-only')
        joint_goal = list(joint_goal or [])
        if plan_kind != 'RETURN_HOME' and (home_stage or joint_goal):
            raise ContractError('home-stage overrides are RETURN_HOME-only')
        if plan_kind == 'RETURN_HOME':
            home_stage = str(home_stage or 'CONFIGURED_HOME').strip().upper()
            if home_stage not in (
                    'CONFIGURED_HOME', 'STARTUP_WRIST', 'ROUGH_HOME',
                    'STORAGE_WRIST'):
                raise ContractError('unsupported home stage: %s' % home_stage)
            if joint_goal and len(joint_goal) != 6:
                raise ContractError(
                    'staged home joint goal must contain six positions')
            if home_stage != 'CONFIGURED_HOME' and len(joint_goal) != 6:
                raise ContractError(
                    '%s requires an explicit six-joint goal' % home_stage)
            if any(not math.isfinite(value) for value in joint_goal):
                raise ContractError('staged home joint goal is non-finite')
        else:
            home_stage = ''
        now_ns = time.time_ns()
        ttl_ns = int(float(self.get_parameter('request_ttl_sec').value) * 1e9)
        joints = [float(value) for value in self.latest_joints.position[:6]]
        scan = (
            self.latest_scan if plan_kind == 'MULTIVIEW_SCAN'
            else self.latest_acquisition_scan)
        center = (
            [0.0, 0.0, 0.0]
            if plan_kind == 'RETURN_HOME'
            else self.vector(scan.get('target_object_center'), 'target center'))
        provenance = self.target_provenance(scan, plan_kind)
        candidates = []
        for item in ([] if plan_kind == 'RETURN_HOME' else scan.get('viewpoints', [])):
            if not isinstance(item, dict) or item.get(
                    'prequalified', item.get('reachable')) is not True \
                    or item.get('safe') is not True:
                continue
            candidate = {
                'id': int(item.get('index', len(candidates))),
                'camera_position_m': self.vector(
                    item.get('desired_camera_position'), 'camera position'),
                'look_direction': self.vector(
                    item.get('desired_look_at_direction'), 'look direction'),
                'required_first': bool(
                    plan_kind == 'ROUGH_ACQUISITION'
                    and item.get('keep_object_centered') is True),
                'coverage_score': float(
                    item.get('expected_new_coverage_score', 0.0)),
                'coverage_objective': str(
                    item.get('coverage_objective', '')),
                'coverage_progress_score': float(
                    item.get('coverage_progress_score', 0.0)),
                'view_selection_policy': str(
                    item.get('view_selection_policy', 'legacy')),
                'view_selection_requested_policy': str(item.get(
                    'view_selection_requested_policy',
                    item.get('view_selection_policy', 'legacy'))),
                'view_selection_generation': int(
                    item.get('view_selection_generation', 0)),
                'view_selection_session_id': str(
                    item.get('view_selection_session_id', '')),
                'nbv_rank': int(item.get('nbv_rank', 0)),
                'nbv_positive_information_gain': bool(
                    item.get('nbv_positive_information_gain', False)),
                'nbv_predicted_unknown_pixels': int(
                    item.get('nbv_predicted_unknown_pixels', 0)),
                'nbv_novel_surface_pixels': int(
                    item.get('nbv_novel_surface_pixels', 0)),
                'nbv_marginal_information_pixels': int(
                    item.get('nbv_marginal_information_pixels', 0)),
                'nbv_marginal_information_fraction': float(
                    item.get('nbv_marginal_information_fraction', 0.0)),
                'nbv_projected_object_pixels': int(
                    item.get('nbv_projected_object_pixels', 0)),
                'nbv_direction_novelty_deg': float(
                    item.get('nbv_direction_novelty_deg', 0.0)),
                'nbv_camera_travel_m': float(
                    item.get('nbv_camera_travel_m', 0.0)),
            }
            if item.get('candidate_geometry') == 'target_ray':
                candidate.update({
                    'candidate_geometry': 'target_ray',
                    'ray_id': int(item.get('ray_id', candidate['id'])),
                    'ray_direction': self.vector(
                        item.get('ray_direction'), 'ray direction'),
                    'ray_min_standoff_m': float(
                        item.get('ray_min_standoff_m')),
                    'ray_max_standoff_m': float(
                        item.get('ray_max_standoff_m')),
                    'ray_preferred_max_standoff_m': float(
                        item.get('ray_preferred_max_standoff_m')),
                    'ray_scoring_standoff_m': float(
                        item.get('ray_scoring_standoff_m')),
                })
            candidates.append(candidate)
        configured_maximum = max(
            1, int(self.get_parameter('max_execution_viewpoints').value))
        session = scan.get('scan_session', {}) if isinstance(scan, dict) else {}
        shortlisted_ray_count = 0
        expanded_ray_candidate_count = 0
        candidate_capabilities = None
        if plan_kind == 'MULTIVIEW_SCAN':
            candidate_capabilities = validate_candidate_policy_batch(
                candidates)
            session_id = str(session.get('session_id', ''))
            accepted_views = int(session.get('accepted_views', -1))
            session_maximum = int(session.get('max_views', -1))
            if not session_id:
                raise ContractError('scan session identity is missing')
            if session_maximum != configured_maximum:
                raise ContractError(
                    'scan session maximum does not match configuration')
            remaining_views = session_maximum - accepted_views
            if remaining_views < 1:
                raise ContractError('scan session has no remaining viewpoints')
            if int(scan.get('remaining_viewpoints', -1)) != remaining_views:
                raise ContractError('scan session remaining count is inconsistent')
        maximum = (
            (1 if bool(self.get_parameter('closed_loop_one_view').value)
             else remaining_views) if plan_kind == 'MULTIVIEW_SCAN'
            else (
                0 if plan_kind == 'RETURN_HOME'
                else min(1, configured_maximum, len(candidates))))
        minimum = maximum if plan_kind in ('MULTIVIEW_SCAN', 'RETURN_HOME') else 1
        closed_loop_one_view = bool(
            plan_kind == 'MULTIVIEW_SCAN'
            and self.get_parameter('closed_loop_one_view').value)
        tracking_scale = (
            float(self.latest_tracking.recommended_speed_scale)
            if self.latest_tracking is not None else 1.0)
        execution_speed = commanded_speed_percent(
            float(self.get_parameter('speed_percent').value),
            plan_kind,
            tracking_scale,
        )
        if plan_kind == 'MULTIVIEW_SCAN':
            current_camera_transform = self.kinematics.camera_transform(joints)
            current_camera = current_camera_transform[:3, 3]
            current_look = current_camera_transform[:3, 2]
            candidate_limit = (
                max(1, int(self.get_parameter(
                    'closed_loop_candidate_limit').value))
                if closed_loop_one_view
                else max(20, maximum * 4, maximum))
            if closed_loop_one_view:
                authoritative_nbv = bool(
                    candidate_capabilities is not None
                    and candidate_capabilities.authoritative_nbv)
                if (
                        candidate_capabilities is not None
                        and candidate_capabilities.ray_expansion):
                    # Retire failed rays from the complete prequalified pool
                    # before choosing the next bounded shortlist.  Applying
                    # this after shortlisting mistakes one exhausted batch for
                    # exhaustion of the full ray frontier.
                    candidates = self.exclude_tesseract_exhausted_rays(
                        candidates, session_id, accepted_views)
                if authoritative_nbv:
                    effective_limit = bounded_candidate_attempt_limit(
                        candidate_capabilities, candidate_limit)
                    if (
                            candidate_capabilities is not None
                            and candidate_capabilities.ray_expansion):
                        candidates = information_ranked_ray_candidates(
                            candidates, effective_limit)
                    else:
                        candidates = bounded_nbv_candidates(
                            candidates, current_camera, center,
                            effective_limit)
                else:
                    effective_limit = bounded_candidate_attempt_limit(
                        candidate_capabilities, candidate_limit)
                    candidates = balanced_closed_loop_candidates(
                        candidates, current_camera, effective_limit,
                        compact_first=(accepted_views == 0))
                if (
                        candidate_capabilities is not None
                        and candidate_capabilities.ray_expansion):
                    shortlisted_ray_count = sum(
                        item.get('candidate_geometry') == 'target_ray'
                        for item in candidates)
                    if shortlisted_ray_count != len(candidates):
                        raise ContractError(
                            'ray policy produced mixed candidate geometry')
                    candidates = bind_shortlisted_ray_intervals(
                        candidates, current_camera, center)
                    expanded_ray_candidate_count = len(candidates)
                candidates = exact_target_aim_candidates(
                    candidates,
                    center,
                    current_look,
                    self.get_parameter(
                        'closed_loop_max_aim_offset_deg').value,
                )
            else:
                candidates = candidates[:candidate_limit]
                candidates = select_diverse_smooth_view_path(
                    candidates, maximum, current_camera)
        else:
            candidates = candidates[:max(20, maximum * 4, maximum)]
        required_candidates = (
            0 if plan_kind == 'RETURN_HOME'
            else (maximum if plan_kind == 'MULTIVIEW_SCAN' else minimum))
        if len(candidates) < required_candidates:
            raise ContractError('only %d safe candidates; need at least %d' % (
                len(candidates), required_candidates))
        observation_mode = (
            'bootstrap_static'
            if plan_kind == 'ROUGH_ACQUISITION'
            else 'perception_snapshot')
        obstacles = []
        if (
                plan_kind != 'RETURN_HOME'
                and observation_mode == 'perception_snapshot'
                and not startup_home):
            for item in self.latest_obstacles.instances:
                if not item.valid:
                    raise ContractError('invalid obstacle geometry is present')
                obstacles.append({
                    'id': '%s:%d' % (item.semantic_label, int(item.object_id)),
                    'type': 'box',
                    'minimum_m': [
                        float(item.base_bounds_min.x), float(item.base_bounds_min.y),
                        float(item.base_bounds_min.z),
                    ],
                    'maximum_m': [
                        float(item.base_bounds_max.x), float(item.base_bounds_max.y),
                        float(item.base_bounds_max.z),
                    ],
                })
        identity = {
            'created_at_ns': now_ns,
            'joint_positions': [round(value, 9) for value in joints],
            'scan_stamp': (
                scan.get('header', {}).get('stamp', {})
                if isinstance(scan, dict) else {}),
            'plan_kind': plan_kind,
            'startup_home': bool(startup_home),
            'home_stage': home_stage,
            'joint_goal_positions_rad': [
                round(value, 9) for value in joint_goal],
            'target_provenance': provenance,
            'boot_id': self.boot_id,
        }
        request_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest()[:32]
        payload = {
            'schema_version': SCHEMA_VERSION,
            'plan_kind': plan_kind,
            'target_provenance': provenance,
            'request_id': request_id,
            'boot_id': self.boot_id,
            'created_at_ns': now_ns,
            'expires_at_ns': now_ns + ttl_ns,
            'frames': {
                'world_frame': 'base_link',
                'camera_optical_frame': 'camera_color_optical_frame',
                'tcp_frame': 'camera_optical_frame',
            },
            'start_state': {
                'joint_names': list(JOINT_NAMES),
                'positions_rad': joints,
                'feedback_stamp': {
                    'sec': int(self.latest_joints.header.stamp.sec),
                    'nanosec': int(self.latest_joints.header.stamp.nanosec),
                },
            },
            'scene': {
                'target_center_m': center,
                'target_provenance': provenance,
                'observation_mode': observation_mode,
                'startup_home_static': bool(startup_home),
                'candidate_views': candidates,
                'obstacles': obstacles,
            },
            'scan_session': (
                {
                    'session_id': session_id,
                    'accepted_views': accepted_views,
                    'max_views': session_maximum,
                    'remaining_views': remaining_views,
                }
                if plan_kind == 'MULTIVIEW_SCAN' else {}
            ),
            'model': {
                'mode': 0,
                'xacro_sha256': sha256_file(self.parameter_path('robot_xacro_path')),
                'srdf_sha256': sha256_file(self.parameter_path('srdf_path')),
                'collision_manifest_sha256': sha256_file(
                    self.parameter_path('collision_manifest_path')),
            },
            'calibration': {
                'hand_eye_sha256': sha256_file(
                    self.parameter_path('hand_eye_calibration_path')),
                'T_link6_camera': self.hand_eye.tolist(),
                'convention': 'T_link6_camera_optical',
            },
            'limits': {
                'position_rad': self.joint_limits.tolist(),
                'bootstrap_start_limit_tolerance_rad': (
                    0.04 if plan_kind == 'ROUGH_ACQUISITION' else 0.0),
                # A disabled arm can relax slightly beyond an inclusive
                # controller-coordinate boundary at the configured storage
                # fold.  Only a dedicated direct RETURN_HOME request may
                # carry that measured start back inside the exact limits.
                'configured_home_start_limit_tolerance_rad': (
                    MAX_CONFIGURED_HOME_START_LIMIT_TOLERANCE_RAD
                    if plan_kind == 'RETURN_HOME' else 0.0),
                'joint_margin_rad': float(
                    self.get_parameter('joint_limit_margin_rad').value),
                'max_velocity_rad_s': [
                    float(value) for value
                    in self.latest_motion_limits.max_velocity_rad_s],
                'max_acceleration_rad_s2': [
                    float(value) for value
                    in self.latest_motion_limits.max_acceleration_rad_s2],
                'motion_limits_sha256':
                    str(self.latest_motion_limits.limits_sha256),
                'source': str(self.latest_motion_limits.source),
            },
            'planning': {
                'planner': 'RRTConnect',
                'pipeline': 'OMPL_ISP',
                'deterministic_seed': int(
                    self.get_parameter('deterministic_seed').value),
                'roll_samples_rad': [float(value) for value in self.get_parameter(
                    'roll_samples_rad').value],
                'min_viewpoints': minimum,
                'max_viewpoints': maximum,
                # Automatic one-view transactions always hold for a fresh
                # measured-coverage decision and later use a separate direct
                # configured-home request. Do not reject an otherwise safe
                # capture because an unused embedded contingency home cannot
                # represent the intentional storage-fold collision bypass.
                'include_return_home': bool(
                    plan_kind == 'RETURN_HOME'
                    or (plan_kind == 'MULTIVIEW_SCAN'
                        and not closed_loop_one_view)),
                'max_execution_joint_step_rad': float(
                    self.get_parameter('trajectory_joint_step_rad').value),
                'effective_speed_percent': execution_speed,
                'command_rate_hz': float(
                    self.get_parameter('trajectory_command_rate_hz').value),
                'timing_policy': TIMING_POLICY,
                'joint_specific_costs': {},
                'return_home_positions_rad': (
                    (
                        [float(value) for value in joint_goal]
                        if joint_goal else
                        [float(value) for value in self.get_parameter(
                            'return_home_positions_rad').value]
                    )
                    if plan_kind in ('MULTIVIEW_SCAN', 'RETURN_HOME') else []
                ),
                'home_stage': home_stage,
                'shortlisted_ray_count': int(shortlisted_ray_count),
                'expanded_ray_candidate_count': int(
                    expanded_ray_candidate_count),
                'ray_direction_attempt_limit': int(
                    RAY_DIRECTION_ATTEMPT_LIMIT
                    if shortlisted_ray_count else 0),
            },
        }
        return attach_digest(payload, 'request_sha256')

    @staticmethod
    def target_provenance(scan, plan_kind):
        if plan_kind == 'RETURN_HOME':
            now_ns = time.time_ns()
            return {
                'source': 'configured_home',
                'frame_id': 'base_link',
                'stamp': {
                    'sec': int(now_ns // 1_000_000_000),
                    'nanosec': int(now_ns % 1_000_000_000),
                },
            }
        header = scan.get('header', {}) if isinstance(scan, dict) else {}
        stamp = header.get('stamp', {})
        if plan_kind == 'MULTIVIEW_SCAN':
            return {
                'source': 'tracked_target',
                'frame_id': str(header.get('frame_id', '')),
                'stamp': {
                    'sec': int(stamp.get('sec', -1)),
                    'nanosec': int(stamp.get('nanosec', -1)),
                },
            }
        supplied = scan.get('target_provenance')
        if not isinstance(supplied, dict):
            raise ContractError(
                'acquisition viewpoints require target_provenance')
        provenance = dict(supplied)
        source_request_id = scan.get('source_request_id')
        if (
                not isinstance(source_request_id, str)
                or provenance.get('source_request_id') != source_request_id):
            raise ContractError(
                'acquisition source_request_id is missing or inconsistent')
        supplied_stamp = provenance.get('stamp', stamp)
        provenance['stamp'] = {
            'sec': int(supplied_stamp.get('sec', -1)),
            'nanosec': int(supplied_stamp.get('nanosec', -1)),
        }
        provenance.setdefault('frame_id', header.get('frame_id', ''))
        return provenance

    @staticmethod
    def vector(value, label):
        if not isinstance(value, dict):
            raise ContractError('%s is missing' % label)
        result = [float(value[key]) for key in ('x', 'y', 'z')]
        if not all(math.isfinite(item) for item in result):
            raise ContractError('%s is non-finite' % label)
        return result

    def exclude_tesseract_exhausted_rays(
            self, candidates, session_id, accepted_views):
        """Exclude static mission failures and current-generation failures."""
        session = str(session_id)
        generation = (session, int(accepted_views))
        candidate_ids = {
            int(item.get('ray_id', item.get('id', -1)))
            for item in candidates
        }
        if getattr(self, 'remaining_ray_pool_session', None) != session:
            self.remaining_ray_pool_session = session
            self.remaining_ray_ids = set(candidate_ids)
            self.retired_ray_ids = set()
        else:
            # A temporarily absent planner candidate must not erase a frozen
            # mission ray.  Current accepted coverage still controls the
            # candidates presented below, while proven static failures stay
            # retired for the whole session.
            self.remaining_ray_ids.update(
                candidate_ids.difference(self.retired_ray_ids))
        if getattr(
                self, 'tesseract_exhausted_ray_generation', None
        ) != generation:
            self.tesseract_exhausted_ray_generation = generation
            self.tesseract_exhausted_ray_ids = set()
        transient = set(getattr(
            self, 'tesseract_exhausted_ray_ids', set()))
        available = [
            item for item in candidates
            if (
                int(item.get('ray_id', item.get('id', -1)))
                in self.remaining_ray_ids
                and int(item.get('ray_id', item.get('id', -1)))
                not in transient)
        ]
        if not available:
            raise ContractError(
                'RAY_FRONTIER_EXHAUSTED: all %d prequalified rays were '
                'rejected by Tesseract for accepted-view generation %d'
                % (len(candidates), int(accepted_views)))
        self.get_logger().info(
            'mission ray pool session=%s accepted=%d: pool=%d, '
            'transiently exhausted=%d, available=%d'
            % (
                str(session_id), int(accepted_views),
                len(self.remaining_ray_ids), len(transient),
                len(available)))
        return available

    def remember_permanently_infeasible_rays(self, payload):
        """Retire endpoint failures for the frozen target-ray session."""
        request_id = str(payload.get('request_id', ''))
        pending = getattr(self, 'pending', {}).get(request_id, {})
        request = pending.get('request', {})
        if int(request.get('planning', {}).get(
                'shortlisted_ray_count', 0)) < 1:
            return []
        session = request.get('scan_session', {})
        session_id = str(session.get('session_id', ''))
        if not session_id:
            return []
        if getattr(self, 'remaining_ray_pool_session', None) != session_id:
            return []
        request_ray_ids = {
            int(item['ray_id'])
            for item in request.get('scene', {}).get('candidate_views', [])
            if item.get('ray_id') is not None
        }
        reported = {
            int(value) for value in payload.get(
                'planning_diagnostics', {}).get(
                    'permanent_infeasible_ray_ids', [])
        }.intersection(request_ray_ids)
        newly_infeasible = sorted(
            reported.intersection(self.remaining_ray_ids))
        self.remaining_ray_ids.difference_update(newly_infeasible)
        self.retired_ray_ids.update(newly_infeasible)
        if newly_infeasible:
            self.get_logger().info(
                'retired static endpoint-infeasible rays for session=%s: %s'
                % (session_id, newly_infeasible))
        return newly_infeasible

    def remember_tesseract_exhausted_rays(self, payload):
        """Retire only rays actually attempted by one failed worker request."""
        codes = [str(value) for value in payload.get('rejection_codes', [])]
        if 'TESSERACT_EXHAUSTED' not in codes:
            return []
        request_id = str(payload.get('request_id', ''))
        pending = getattr(self, 'pending', {}).get(request_id, {})
        request = pending.get('request', {})
        if int(request.get('planning', {}).get(
                'shortlisted_ray_count', 0)) < 1:
            return []
        session = request.get('scan_session', {})
        generation = (
            str(session.get('session_id', '')),
            int(session.get('accepted_views', -1)),
        )
        if (
                not generation[0]
                or generation != getattr(
                    self, 'tesseract_exhausted_ray_generation', None)):
            return []
        request_ray_ids = {
            int(item['ray_id'])
            for item in request.get('scene', {}).get('candidate_views', [])
            if item.get('ray_id') is not None
        }
        attempted = {
            int(value) for value in payload.get(
                'planning_diagnostics', {}).get('attempted_ray_ids', [])
        }.intersection(request_ray_ids)
        exhausted = self.tesseract_exhausted_ray_ids
        newly_exhausted = sorted(attempted.difference(exhausted))
        exhausted.update(newly_exhausted)
        if newly_exhausted:
            self.get_logger().info(
                'retired Tesseract-infeasible rays for session=%s '
                'accepted=%d: %s'
                % (generation[0], generation[1], newly_exhausted))
        return newly_exhausted

    def poll(self):
        timeout = float(self.get_parameter('response_timeout_sec').value)
        for request_id, state in list(self.pending.items()):
            response_path = self.spool.path('responses', request_id)
            if response_path.is_file():
                try:
                    payload = self.spool.read('responses', request_id)
                    validate_response(payload, state['request'])
                    self.publish_plan(payload)
                except (ContractError, KeyError, OSError, TypeError, ValueError) as error:
                    self.publish_rejection(request_id, 'RESPONSE_INVALID', str(error))
                finally:
                    # The validated ROS message is the hand-off artifact. Consume
                    # the spool entry so a long-running bridge stays bounded.
                    try:
                        response_path.unlink()
                    except FileNotFoundError:
                        pass
                self.pending.pop(request_id, None)
            elif self.now() - state['started'] > timeout:
                self.pending.pop(request_id, None)
                self.publish_rejection(
                    request_id, 'PLANNER_TIMEOUT', 'Tesseract response timed out')
        self.publish_status()
        self.publish_readiness()

    def motion_limits_cb(self, msg):
        # Preserve the last fully validated controller-limit set through one
        # transient partial CAN-query result.  Its freshness timestamp is not
        # renewed by invalid samples, so a persistent fault still blocks
        # planning after the normal timeout.
        accepted, refreshed = self.motion_limit_stability.observe(
            msg, self.now())
        if accepted is not None:
            self.latest_motion_limits = accepted
        if refreshed:
            self.mark('motion_limits')

    def publish_plan(self, payload):
        self.remember_permanently_infeasible_rays(payload)
        if payload.get('status') != 'success':
            codes = payload.get('rejection_codes') or ['PLANNING_FAILED']
            exhausted_rays = self.remember_tesseract_exhausted_rays(payload)
            if exhausted_rays:
                self.publish_rejection(
                    payload.get('request_id', ''),
                    'RAY_SHORTLIST_EXHAUSTED',
                    'TESSERACT_EXHAUSTED: %s; retired attempted ray IDs %s'
                    % (
                        str(payload.get('diagnostic', 'planning failed')),
                        exhausted_rays),
                    additional_codes=codes)
                return
            self.publish_rejection(
                payload.get('request_id', ''), str(codes[0]),
                str(payload.get('diagnostic', 'planning failed')))
            return
        msg = TesseractPlan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.plan_id = str(payload['plan_id'])
        msg.plan_kind = str(payload['plan_kind'])
        msg.source_request_id = str(
            payload.get('target_provenance', {}).get(
                'source_request_id', ''))
        msg.request_sha256 = str(payload['request_sha256'])
        msg.trajectory_sha256 = str(payload['trajectory_sha256'])
        msg.motion_limits_sha256 = str(
            payload['trajectory_binding']['limits']['motion_limits_sha256'])
        execution = payload['trajectory_binding']['execution']
        msg.execution_speed_percent = float(
            execution['effective_speed_percent'])
        msg.command_rate_hz = float(execution['command_rate_hz'])
        msg.timing_policy = str(execution['timing_policy'])
        msg.backend = str(payload['backend'])
        msg.backend_version = str(payload['backend_version'])
        msg.valid = True
        msg.dry_run = True
        msg.real_arm_motion = False
        msg.collision_model_qualified = bool(payload['collision_model_qualified'])
        msg.reason = str(payload.get('diagnostic', 'validated Tesseract proposal'))
        msg.rejection_codes = [str(value) for value in payload.get('rejection_codes', [])]
        center = payload['target_center_m']
        msg.target_center = Point(x=float(center[0]), y=float(center[1]), z=float(center[2]))
        for selected in payload['selected_viewpoints']:
            msg.viewpoint_indices.append(int(selected['id']))
            position = selected['camera_position_m']
            direction = selected['look_direction']
            msg.camera_positions.append(Point(
                x=float(position[0]), y=float(position[1]), z=float(position[2])))
            msg.look_directions.append(Vector3(
                x=float(direction[0]), y=float(direction[1]), z=float(direction[2])))
        for segment in payload['segments']:
            trajectory = JointTrajectory()
            trajectory.joint_names = list(JOINT_NAMES)
            for value in segment['points']:
                point = JointTrajectoryPoint()
                point.positions = [float(item) for item in value['positions_rad']]
                point.velocities = [float(item) for item in value['velocities_rad_s']]
                point.accelerations = [float(item) for item in value['accelerations_rad_s2']]
                total_nanoseconds = int(round(
                    float(value['time_from_start_s']) * 1e9))
                point.time_from_start = Duration(
                    sec=total_nanoseconds // 1_000_000_000,
                    nanosec=total_nanoseconds % 1_000_000_000,
                )
                trajectory.points.append(point)
            msg.trajectories.append(trajectory)
            msg.minimum_clearance_m.append(float(segment['minimum_clearance_m']))
            msg.limiting_link_pairs.append(str(segment['limiting_link_pair']))
            recovery_used = bool(segment.get('bootstrap_recovery_used', False))
            msg.bootstrap_recovery_end_points.append(
                int(segment['bootstrap_recovery_end_point'])
                if recovery_used else -1)
            msg.bootstrap_recovery_joints.append(
                int(segment['bootstrap_recovery_joint'])
                if recovery_used else 0)
            msg.bootstrap_recovery_delta_rad.append(
                float(segment['bootstrap_recovery_delta_rad'])
                if recovery_used else 0.0)
            evidence = {
                'startup_home_static': bool(
                    segment.get('startup_home_static', False)),
                'configured_home_direct_joint_move': bool(
                    segment.get('configured_home_direct_joint_move', False)),
                'configured_home_goal_positions_rad': segment.get(
                    'configured_home_goal_positions_rad', []),
                'collision_validation_bypassed': bool(
                    segment.get('collision_validation_bypassed', False)),
                'home_stage': str(segment.get('home_stage', '')),
                'validation': str(segment.get('validation', '')),
                'trajectory_blending': str(segment.get(
                    'trajectory_blending', '')),
                'pass_through_blending_applied': bool(segment.get(
                    'pass_through_blending_applied', False)),
                'pass_through_blend_fallback_used': bool(segment.get(
                    'pass_through_blend_fallback_used', False)),
                'pass_through_blended_corners': int(segment.get(
                    'pass_through_blended_corners', 0)),
                'pass_through_source_points': int(segment.get(
                    'pass_through_source_points', 0)),
                'pass_through_geometry_points': int(segment.get(
                    'pass_through_geometry_points', 0)),
                'pass_through_maximum_radius_rad': float(segment.get(
                    'pass_through_maximum_radius_rad', 0.0)),
                'pass_through_blend_reason': str(segment.get(
                    'pass_through_blend_reason', '')),
                'sdk_execution_mode': str(segment.get(
                    'sdk_execution_mode', 'TESSERACT_STREAM')),
                'sdk_command_anchor_count': int(segment.get(
                    'sdk_command_anchor_count', 0)),
                'direct_movej_validation': str(segment.get(
                    'direct_movej_validation', '')),
                'direct_movej_source_points': int(segment.get(
                    'direct_movej_source_points', 0)),
                'used': recovery_used,
                'minimum_clearance_m': (
                    float(segment['bootstrap_recovery_minimum_clearance_m'])
                    if recovery_used else None),
                'limiting_link_pair': (
                    str(segment['bootstrap_recovery_limiting_link_pair'])
                    if recovery_used else ''),
                'validation_samples': (
                    int(segment['bootstrap_recovery_samples'])
                    if recovery_used else 0),
                'start_contacts': (
                    segment.get('bootstrap_start_contacts', [])
                    if recovery_used else []),
                'joint_numbers': (
                    segment.get('bootstrap_recovery_joints', [])
                    if recovery_used else []),
                'delta_rad': (
                    segment.get('bootstrap_recovery_deltas_rad', [])
                    if recovery_used else []),
                'powered_start': {
                    'used': bool(segment.get(
                        'powered_start_recovery_used', False)),
                    'end_point': int(segment.get(
                        'powered_start_recovery_end_point', -1)),
                    'joint_numbers': segment.get(
                        'powered_start_recovery_joints', []),
                    'delta_rad': segment.get(
                        'powered_start_recovery_deltas_rad', []),
                    'minimum_clearance_m': segment.get(
                        'powered_start_recovery_minimum_clearance_m'),
                    'limiting_link_pair': segment.get(
                        'powered_start_recovery_limiting_link_pair', ''),
                    'validation_samples': int(segment.get(
                        'powered_start_recovery_samples', 0)),
                    'start_contacts': segment.get(
                        'powered_start_contacts', []),
                },
            }
            msg.bootstrap_recovery_evidence_json.append(
                json.dumps(evidence, sort_keys=True, separators=(',', ':')))
        provenance = String()
        provenance.data = json.dumps({
            'schema_version': 1,
            'plan_id': str(payload['plan_id']),
            'request_id': str(payload['request_id']),
            'request_sha256': str(payload['request_sha256']),
            'plan_kind': str(payload['plan_kind']),
            'selected_viewpoints': [
                {
                    key: selected[key]
                    for key in (
                        'id',
                        'camera_position_m',
                        'look_direction',
                        'nominal_look_direction',
                        'aim_fallback_used',
                        'aim_offset_deg',
                        'aim_attempt_diagnostics',
                        'view_selection_policy',
                        'view_selection_requested_policy',
                        'view_selection_generation',
                        'view_selection_session_id',
                        'nbv_rank',
                        'nbv_positive_information_gain',
                        'nbv_predicted_unknown_pixels',
                        'nbv_novel_surface_pixels',
                        'nbv_marginal_information_pixels',
                        'nbv_marginal_information_fraction',
                        'nbv_projected_object_pixels',
                        'nbv_direction_novelty_deg',
                        'nbv_camera_travel_m',
                        'coverage_score',
                        'ray_id',
                        'ray_standoff_m',
                        'ray_probe_index',
                        'ray_probe_phase',
                    )
                    if key in selected
                }
                for selected in payload['selected_viewpoints']
            ],
            'candidate_diagnostics': payload.get(
                'planning_diagnostics', {}),
        }, sort_keys=True)
        self.plan_provenance_pub.publish(provenance)
        self.plan_pub.publish(msg)
        if self.param_bool('debug') and payload['selected_viewpoints']:
            selected = payload['selected_viewpoints'][0]
            self.get_logger().info(
                'published plan=%s selected_view=%s policy=%s generation=%s '
                'nbv_rank=%s marginal_fraction=%.4f predicted_unknown=%s '
                'novel_surface=%s'
                % (
                    payload['plan_id'], selected.get('id', ''),
                    selected.get('view_selection_policy', 'legacy'),
                    selected.get('view_selection_generation', 0),
                    selected.get('nbv_rank', 0),
                    selected.get('nbv_marginal_information_fraction', 0.0),
                    selected.get('nbv_predicted_unknown_pixels', 0),
                    selected.get('nbv_novel_surface_pixels', 0)))
        self.set_status('PROPOSAL_READY', msg.reason, payload['request_id'])

    def publish_rejection(
            self, request_id, code, reason, additional_codes=None):
        msg = TesseractPlan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        # Rejections are plans too for correlation purposes. Preserve the
        # complete request ID so a caller can fail this attempt immediately
        # without accepting or waiting on another generation's result.
        msg.plan_id = str(request_id)
        pending = self.pending.get(request_id, {})
        request = pending.get('request', {})
        msg.plan_kind = str(request.get('plan_kind', ''))
        msg.source_request_id = str(
            request.get('target_provenance', {}).get(
                'source_request_id', ''))
        msg.valid = False
        msg.dry_run = True
        msg.real_arm_motion = False
        msg.reason = '%s: %s' % (code, reason)
        msg.rejection_codes = [code] + [
            str(value) for value in (additional_codes or [])
            if str(value) != str(code)
        ]
        self.plan_pub.publish(msg)
        self.set_status('REJECTED', msg.reason, request_id)

    def set_status(self, state, reason, request_id=''):
        self.state = state
        self.reason = reason
        self.publish_status(request_id)

    def publish_status(self, request_id=''):
        msg = TesseractPlanStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.request_id = request_id or (next(iter(self.pending)) if self.pending else '')
        msg.state = self.state
        msg.reason = self.reason
        msg.dry_run = True
        msg.real_arm_motion = False
        msg.pending_requests = self.spool.pending('requests') + self.spool.pending('processing')
        msg.pending_responses = self.spool.pending('responses')
        self.status_pub.publish(msg)

    def publish_readiness(self):
        worker_blockers = self.worker_health_reasons()
        acquisition_blockers = self.snapshot_reasons(
            'ROUGH_ACQUISITION',
            require_viewpoints=False,
            worker_reasons=worker_blockers,
        )
        multiview_blockers = self.snapshot_reasons(
            'MULTIVIEW_SCAN',
            require_viewpoints=True,
            worker_reasons=worker_blockers,
        )
        manipulation_blockers = list(multiview_blockers)
        # Contact motion remains fail-closed until the installed planning
        # model explicitly contains a qualified gripper TCP, open/closed
        # geometry, attached-object handling, and allowed-contact policy.
        if not bool(self.get_parameter(
                'manipulation_model_qualified').value):
            manipulation_blockers.append(
                'gripper/contact collision model is not qualified')
        manipulation_blockers = list(dict.fromkeys(manipulation_blockers))
        msg = TesseractReadiness()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.generation_id = self.worker_generation_id
        msg.worker_ready = not worker_blockers
        msg.acquisition_blockers = acquisition_blockers
        msg.multiview_blockers = multiview_blockers
        msg.manipulation_blockers = manipulation_blockers
        msg.acquisition_ready = not acquisition_blockers
        msg.multiview_ready = not multiview_blockers
        msg.manipulation_ready = not manipulation_blockers
        self.readiness_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = TesseractPlanBridge()
    except (ContractError, OSError, RuntimeError, ValueError) as error:
        print('Tesseract bridge startup error: %s' % error)
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
