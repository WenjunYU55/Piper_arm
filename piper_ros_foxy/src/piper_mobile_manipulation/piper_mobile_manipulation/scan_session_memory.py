"""Pure helpers for session-scoped scan-view diversity and resumption."""

import math

import numpy as np

from piper_mobile_manipulation.perception.target_envelope import (
    validate_capture_model_seed,
    validate_shape_measurement,
)


def vector3(mapping, field):
    if not isinstance(mapping, dict):
        raise ValueError('%s is missing' % field)
    values = np.asarray([
        mapping.get('x'), mapping.get('y'), mapping.get('z'),
    ], dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError('%s must contain three finite values' % field)
    return values


def angular_separation_deg(first, second):
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left.shape != (3,) or right.shape != (3,) or min(left_norm, right_norm) <= 0:
        return math.inf
    cosine = float(np.dot(left, right) / (left_norm * right_norm))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def history_vector3(entry, actual_field, desired_field, label):
    """Prefer achieved geometry, with legacy desired geometry as fallback."""
    if not isinstance(entry, dict):
        raise ValueError('%s is missing' % label)
    for field in (actual_field, desired_field):
        try:
            return vector3(entry.get(field), label)
        except (TypeError, ValueError):
            continue
    raise ValueError('%s is missing' % label)


def viewpoint_is_duplicate(
        viewpoint, entries, position_tolerance_m=0.012,
        look_tolerance_deg=2.0):
    camera = vector3(
        viewpoint.get('desired_camera_position'), 'camera position')
    look = vector3(
        viewpoint.get('desired_look_at_direction'), 'look direction')
    for entry in entries:
        try:
            previous_camera = history_vector3(
                entry, 'actual_camera_position', 'desired_camera_position',
                'history camera position')
            previous_look = history_vector3(
                entry, 'actual_look_at_direction',
                'desired_look_at_direction', 'history look direction')
        except (TypeError, ValueError):
            continue
        if (
                float(np.linalg.norm(camera - previous_camera))
                <= float(position_tolerance_m)
                and angular_separation_deg(look, previous_look)
                <= float(look_tolerance_deg)):
            return True
    return False


def viewpoint_direction_is_redundant(
        viewpoint, accepted_entries, target_center,
        minimum_separation_deg=6.0):
    """Reject an already-observed target-to-camera direction, not travel."""
    threshold = max(0.0, float(minimum_separation_deg))
    if threshold <= 0.0 or not accepted_entries:
        return False
    current_center = vector3(target_center, 'current target center')
    camera = vector3(
        viewpoint.get('desired_camera_position'), 'camera position')
    candidate_direction = camera - current_center
    if float(np.linalg.norm(candidate_direction)) <= 1e-9:
        return True
    for entry in accepted_entries:
        try:
            previous_camera = history_vector3(
                entry, 'actual_camera_position', 'desired_camera_position',
                'accepted camera position')
            previous_center = vector3(
                entry.get('target_estimate_used', target_center),
                'accepted target estimate')
        except (TypeError, ValueError):
            continue
        previous_direction = previous_camera - previous_center
        if angular_separation_deg(
                candidate_direction, previous_direction) <= threshold + 1e-9:
            return True
    return False


def diversity_distance(viewpoint, references):
    """Score a view by its closest combined position/direction separation."""
    if not references:
        return math.inf
    camera = vector3(
        viewpoint.get('desired_camera_position'), 'camera position')
    look = vector3(
        viewpoint.get('desired_look_at_direction'), 'look direction')
    scores = []
    for reference in references:
        try:
            previous_camera = history_vector3(
                reference, 'actual_camera_position',
                'desired_camera_position', 'reference camera')
            previous_look = history_vector3(
                reference, 'actual_look_at_direction',
                'desired_look_at_direction', 'reference look')
        except (TypeError, ValueError):
            continue
        # One degree contributes one centimetre to the diversity score.
        scores.append(
            float(np.linalg.norm(camera - previous_camera))
            + angular_separation_deg(look, previous_look) * 0.01)
    return min(scores) if scores else math.inf


def camera_offset_geometry(entries, target_center):
    """Return achieved target-to-camera offsets and spherical coordinates."""
    center = vector3(target_center, 'coverage target center')
    geometry = []
    for entry in entries if isinstance(entries, list) else []:
        try:
            camera = history_vector3(
                entry, 'actual_camera_position', 'desired_camera_position',
                'achieved camera position')
        except (TypeError, ValueError):
            continue
        offset = camera - center
        distance = float(np.linalg.norm(offset))
        if distance <= 1e-6 or not np.all(np.isfinite(offset)):
            continue
        horizontal = float(np.linalg.norm(offset[:2]))
        azimuth = math.degrees(math.atan2(offset[1], offset[0]))
        if azimuth < 0.0:
            azimuth += 360.0
        geometry.append({
            'offset': offset,
            'distance': distance,
            'lateral_fraction': float(offset[1] / distance),
            'azimuth_deg': azimuth,
            'elevation_deg': math.degrees(
                math.atan2(offset[2], horizontal)),
        })
    return geometry


def feature_coverage_priority(
        viewpoint, accepted_entries, target_center,
        minimum_views_per_y_side=2, minimum_elevation_span_deg=25.0):
    """
    Score the next view against the most important unmet feature axis.

    A generic farthest-point score can reverse an orbit before it reaches a
    missing cube face.  This staged score keeps walking toward one missing
    lateral face until it has two achieved observations, then crosses to the
    other face, extends elevation, and only then maximizes ordinary pose
    diversity.  The stages are recomputed after every accepted achieved pose,
    so no desired pose is treated as accomplished.
    """
    center = vector3(target_center, 'coverage target center')
    camera = vector3(
        viewpoint.get('desired_camera_position'), 'camera position')
    offset = camera - center
    distance = float(np.linalg.norm(offset))
    if distance <= 1e-6 or not np.all(np.isfinite(offset)):
        raise ValueError('candidate camera position coincides with target center')
    horizontal = float(np.linalg.norm(offset[:2]))
    lateral_fraction = float(offset[1] / distance)
    elevation = math.degrees(math.atan2(offset[2], horizontal))

    achieved = camera_offset_geometry(accepted_entries, target_center)
    positive_y = sum(
        item['lateral_fraction'] >= 0.35 for item in achieved)
    negative_y = sum(
        item['lateral_fraction'] <= -0.35 for item in achieved)
    elevations = [item['elevation_deg'] for item in achieved]
    elevation_span = (
        max(elevations) - min(elevations)
        if len(elevations) >= 2 else 0.0)

    # Finish the side the achieved camera is already approaching, then cross
    # the front hemisphere once to the opposite side. Hard-coding -Y first
    # made a physical +Y-side acquisition cross the workspace twice.
    both_sides_missing = (
        negative_y < int(minimum_views_per_y_side)
        and positive_y < int(minimum_views_per_y_side))
    approaching_positive = bool(
        achieved and achieved[-1]['lateral_fraction'] >= 0.0)
    if (
            positive_y < int(minimum_views_per_y_side)
            and both_sides_missing and approaching_positive):
        objective = 'positive_y_face'
        primary = lateral_fraction
    elif negative_y < int(minimum_views_per_y_side):
        objective = 'negative_y_face'
        primary = -lateral_fraction
    elif positive_y < int(minimum_views_per_y_side):
        objective = 'positive_y_face'
        primary = lateral_fraction
    elif elevation_span < float(minimum_elevation_span_deg):
        objective = 'elevation_span'
        projected = elevations + [elevation]
        primary = (max(projected) - min(projected)) / 90.0
    else:
        objective = 'residual_pose_diversity'
        value = diversity_distance(viewpoint, accepted_entries)
        primary = 1.0 if math.isinf(value) else float(value)

    # Marginal elevation and ordinary baseline separate otherwise equivalent
    # face directions without overpowering the active hard-coverage goal.
    projected_elevations = elevations + [elevation]
    elevation_gain = (
        (max(projected_elevations) - min(projected_elevations)
         - elevation_span) if elevations else 0.0)
    diversity = diversity_distance(viewpoint, accepted_entries)
    if math.isinf(diversity) or objective != 'residual_pose_diversity':
        diversity = 0.0
    score = (
        1000.0 * float(primary)
        + 2.0 * max(0.0, float(elevation_gain))
        + min(10.0, max(0.0, float(diversity))))
    return score, objective


def feature_coverage_progress(
        viewpoint, accepted_entries, target_center, objective,
        lateral_tolerance=0.02):
    """
    Measure non-regression against achieved history for one objective.

    Positive values advance the active feature floor.  Zero permits a useful
    radius/elevation configuration change without surrendering already
    achieved coverage.  A small lateral tolerance absorbs measured target/FK
    noise, while a genuine orbit reversal remains negative.
    """
    center = vector3(target_center, 'coverage target center')
    camera = vector3(
        viewpoint.get('desired_camera_position'), 'camera position')
    offset = camera - center
    distance = float(np.linalg.norm(offset))
    if distance <= 1e-6 or not np.all(np.isfinite(offset)):
        raise ValueError('candidate camera position coincides with target center')
    horizontal = float(np.linalg.norm(offset[:2]))
    candidate_lateral = float(offset[1] / distance)
    candidate_elevation = math.degrees(
        math.atan2(offset[2], horizontal))
    achieved = camera_offset_geometry(accepted_entries, target_center)
    if not achieved:
        return 0.0
    if objective == 'negative_y_face':
        current = achieved[-1]['lateral_fraction']
        return current - candidate_lateral + float(lateral_tolerance)
    if objective == 'positive_y_face':
        current = achieved[-1]['lateral_fraction']
        return candidate_lateral - current + float(lateral_tolerance)
    if objective == 'elevation_span':
        values = [item['elevation_deg'] for item in achieved]
        previous = max(values) - min(values) if len(values) >= 2 else 0.0
        projected = values + [candidate_elevation]
        return max(projected) - min(projected) - previous
    return max(0.0, float(diversity_distance(
        viewpoint, accepted_entries)))


def achieved_feature_coverage(
        entries, target_center, minimum_views=9,
        minimum_views_per_y_side=2, minimum_elevation_span_deg=25.0,
        surface_coverage=None):
    """
    Measure achieved face/axis diversity from persisted camera poses.

    The base-link X direction defines the front hemisphere and Y separates
    the two lateral cube faces.  This is an execution-result gate, not a
    reachability claim: only accepted achieved FK positions contribute.
    """
    try:
        geometry = camera_offset_geometry(entries, target_center)
    except (TypeError, ValueError):
        geometry = []
    azimuths = []
    elevations = []
    positive_y = 0
    negative_y = 0
    for item in geometry:
        lateral_fraction = item['lateral_fraction']
        if lateral_fraction >= 0.35:
            positive_y += 1
        if lateral_fraction <= -0.35:
            negative_y += 1
        azimuths.append(item['azimuth_deg'])
        elevations.append(item['elevation_deg'])
    azimuth_span = (
        max(azimuths) - min(azimuths) if len(azimuths) >= 2 else 0.0)
    elevation_span = (
        max(elevations) - min(elevations) if len(elevations) >= 2 else 0.0)
    geometric_sufficient = bool(
        len(geometry) >= int(minimum_views)
        and positive_y >= int(minimum_views_per_y_side)
        and negative_y >= int(minimum_views_per_y_side)
        and elevation_span >= float(minimum_elevation_span_deg))
    surface_required = isinstance(surface_coverage, dict)
    surface_sufficient = bool(
        surface_coverage.get('sufficient')) if surface_required else True
    sufficient = bool(geometric_sufficient and surface_sufficient)
    blockers = []
    if len(geometry) < int(minimum_views):
        blockers.append(
            'only %d/%d accepted achieved views' % (
                len(geometry), int(minimum_views)))
    if positive_y < int(minimum_views_per_y_side):
        blockers.append(
            '+Y side has %d/%d views' % (
                positive_y, int(minimum_views_per_y_side)))
    if negative_y < int(minimum_views_per_y_side):
        blockers.append(
            '-Y side has %d/%d views' % (
                negative_y, int(minimum_views_per_y_side)))
    if elevation_span < float(minimum_elevation_span_deg):
        blockers.append(
            'elevation span %.1f/%.1f deg' % (
                elevation_span, float(minimum_elevation_span_deg)))
    if surface_required and not surface_sufficient:
        blockers.append(str(surface_coverage.get(
            'reason', 'measured surface coverage is insufficient')))
    return {
        'sufficient': sufficient,
        'geometric_sufficient': geometric_sufficient,
        'accepted_achieved_views': len(geometry),
        'positive_y_side_views': positive_y,
        'negative_y_side_views': negative_y,
        'azimuth_span_deg': float(azimuth_span),
        'elevation_span_deg': float(elevation_span),
        'surface_coverage': dict(surface_coverage or {}),
        'blockers': blockers,
    }


def filter_and_order_viewpoints(
        viewpoints, entries, position_tolerance_m=0.012,
        look_tolerance_deg=2.0, accepted_entries=None, target_center=None,
        minimum_direction_separation_deg=0.0,
        direction_target_center=None, rejection_reasons=None):
    """
    Remove viewpoints already captured in this session.

    Preserve the planner's deterministic geometric order.  Diversity
    selection and smooth route ordering happen once in the Tesseract bridge,
    where the requested remaining-view count and the current calibrated
    camera position are both available.  Greedily choosing the farthest pose
    here and again in the bridge made the live arm alternate between the two
    ends of one orbit sector.
    """
    accepted_entries = entries if accepted_entries is None else accepted_entries
    remaining = []
    for item in viewpoints:
        if viewpoint_is_duplicate(
                item, entries, position_tolerance_m, look_tolerance_deg):
            if rejection_reasons is not None:
                rejection_reasons[int(item.get('ray_id', item.get(
                    'index', -1)))] = ['duplicate of an accepted camera pose']
            continue
        if (
                (direction_target_center is not None or target_center is not None)
                and viewpoint_direction_is_redundant(
                    item, accepted_entries,
                    direction_target_center
                    if direction_target_center is not None else target_center,
                    minimum_direction_separation_deg)):
            if rejection_reasons is not None:
                rejection_reasons[int(item.get('ray_id', item.get(
                    'index', -1)))] = [
                        'direction is within the accepted-view redundancy floor']
            continue
        remaining.append(item)
    if target_center is None:
        accepted_entries = entries
    if not entries and target_center is None:
        return remaining
    scored = []
    for item in remaining:
        candidate = dict(item)
        if target_center is not None:
            value, objective = feature_coverage_priority(
                item, accepted_entries, target_center)
            candidate['coverage_objective'] = objective
            candidate['coverage_progress_score'] = feature_coverage_progress(
                item, accepted_entries, target_center, objective)
        else:
            value = float(diversity_distance(item, entries))
        candidate['expected_new_coverage_score'] = value
        scored.append(candidate)
    return sorted(
        scored,
        key=lambda item: (
            -float(item['expected_new_coverage_score']),
            int(item.get('index', 0))),
    )


def validate_history_payload(payload, maximum_views):
    """Return a normalized session payload or raise a precise ValueError."""
    if not isinstance(payload, dict):
        raise ValueError('scan history is not an object')
    session_id = str(payload.get('session_id', ''))
    if not session_id:
        raise ValueError('scan history session_id is missing')
    entries = payload.get('entries', [])
    if not isinstance(entries, list):
        raise ValueError('scan history entries are not a list')
    accepted = int(payload.get('accepted_views', len(entries)))
    maximum = int(payload.get('max_views', maximum_views))
    if maximum != int(maximum_views):
        raise ValueError('scan history maximum does not match configuration')
    if accepted != len(entries):
        raise ValueError('scan history count does not match its entries')
    rejected_entries = payload.get('rejected_entries', [])
    if not isinstance(rejected_entries, list):
        raise ValueError('rejected scan history entries are not a list')
    if accepted < 0 or accepted > maximum:
        raise ValueError('scan history count is outside the session bounds')
    coverage_target_center = payload.get('coverage_target_center')
    if coverage_target_center is not None:
        normalized_center = vector3(
            coverage_target_center, 'coverage target center')
        coverage_target_center = dict(zip(
            ('x', 'y', 'z'), (float(value) for value in normalized_center)))
    qualified_target_shape = payload.get('qualified_target_shape')
    if qualified_target_shape is not None:
        qualified_target_shape = validate_shape_measurement(
            qualified_target_shape)
    qualified_target_model_seed = payload.get('qualified_target_model_seed')
    if qualified_target_model_seed is not None:
        qualified_target_model_seed = validate_capture_model_seed(
            qualified_target_model_seed)
        if accepted < 1:
            raise ValueError(
                'capture target model seed requires an accepted view')
        seed_shape = qualified_target_model_seed['shape']
        if (
                qualified_target_shape is not None
                and seed_shape['measurement_sha256'] !=
                qualified_target_shape['measurement_sha256']):
            raise ValueError(
                'qualified target shape does not match capture model seed')
    for entry in entries:
        vector3(entry.get('desired_camera_position'), 'history camera position')
        vector3(entry.get('desired_look_at_direction'), 'history look direction')
    for entry in rejected_entries:
        vector3(entry.get('desired_camera_position'), 'rejected camera position')
        vector3(entry.get('desired_look_at_direction'), 'rejected look direction')
        if 'ray_population_phase' in entry:
            population_phase = entry['ray_population_phase']
            if population_phase not in ('bootstrap', 'qualified'):
                raise ValueError('rejected ray population phase is invalid')
        has_retry_ray = 'framing_retry_ray_id' in entry
        has_retry_minimum = 'framing_retry_min_standoff_m' in entry
        if has_retry_ray != has_retry_minimum:
            raise ValueError('rejected framing retry metadata is incomplete')
        if has_retry_ray:
            try:
                retry_ray = int(entry['framing_retry_ray_id'])
                retry_minimum = float(
                    entry['framing_retry_min_standoff_m'])
            except (TypeError, ValueError):
                raise ValueError('rejected framing retry metadata is invalid')
            if (
                    retry_ray < 0 or not math.isfinite(retry_minimum)
                    or retry_minimum <= 0.0):
                raise ValueError('rejected framing retry metadata is invalid')
    latest_achieved = payload.get('latest_achieved_camera')
    if latest_achieved is not None:
        if not isinstance(latest_achieved, dict):
            raise ValueError('latest achieved camera is not an object')
        camera = vector3(
            latest_achieved.get('camera_position'),
            'latest achieved camera position')
        look = vector3(
            latest_achieved.get('look_direction'),
            'latest achieved camera look direction')
        latest_achieved = dict(latest_achieved)
        latest_achieved['camera_position'] = dict(zip(
            ('x', 'y', 'z'), (float(value) for value in camera)))
        latest_achieved['look_direction'] = dict(zip(
            ('x', 'y', 'z'), (float(value) for value in look)))
    return {
        'session_id': session_id,
        'accepted_views': accepted,
        'max_views': maximum,
        # Non-ray policies exclude accepted and rejected poses positionally.
        # Ray policies use accepted entries for permanent angular retirement
        # and population-tagged rejections for temporary exact-ray
        # quarantine.
        'entries': list(entries) + list(rejected_entries),
        'accepted_entries': list(entries),
        'rejected_entries': list(rejected_entries),
        'latest_achieved_camera': latest_achieved,
        # This measured center is frozen once per scan session.  Live target
        # measurements still place each new candidate, but achieved coverage
        # must be compared in one immutable frame so tracker noise cannot make
        # an already-covered cube face appear missing again.
        'coverage_target_center': coverage_target_center,
        # Only the executor may populate this after it proves that the first
        # scan endpoint is settled, aimed, and observed by a post-settle frame.
        # The planner uses this exact immutable measurement as its model seed.
        'qualified_target_shape': qualified_target_shape,
        # Capture one binds its exact persisted mask/depth silhouette to the
        # synchronized base-from-camera transform.  The planner must consume
        # this immutable record instead of querying live TF later.
        'qualified_target_model_seed': qualified_target_model_seed,
    }


def history_coverage_target_center(history, live_target_center):
    """Prefer the session's frozen measured coverage frame over live tracking."""
    candidate = (
        history.get('coverage_target_center')
        if isinstance(history, dict) else None)
    if candidate is None:
        candidate = live_target_center
    try:
        center = vector3(candidate, 'coverage target center')
    except (TypeError, ValueError):
        return None
    return dict(zip(('x', 'y', 'z'), (float(value) for value in center)))
