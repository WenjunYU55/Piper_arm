"""Pure helpers for session-scoped scan-view diversity and resumption."""

import math

import numpy as np


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


def viewpoint_is_duplicate(
        viewpoint, entries, position_tolerance_m=0.012,
        look_tolerance_deg=2.0):
    camera = vector3(
        viewpoint.get('desired_camera_position'), 'camera position')
    look = vector3(
        viewpoint.get('desired_look_at_direction'), 'look direction')
    for entry in entries:
        try:
            previous_camera = vector3(
                entry.get('desired_camera_position'), 'history camera position')
            previous_look = vector3(
                entry.get('desired_look_at_direction'), 'history look direction')
        except (TypeError, ValueError):
            continue
        if (
                float(np.linalg.norm(camera - previous_camera))
                <= float(position_tolerance_m)
                and angular_separation_deg(look, previous_look)
                <= float(look_tolerance_deg)):
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
            previous_camera = vector3(
                reference.get('desired_camera_position'), 'reference camera')
            previous_look = vector3(
                reference.get('desired_look_at_direction'), 'reference look')
        except (TypeError, ValueError):
            continue
        # One degree contributes one centimetre to the diversity score.
        scores.append(
            float(np.linalg.norm(camera - previous_camera))
            + angular_separation_deg(look, previous_look) * 0.01)
    return min(scores) if scores else math.inf


def filter_and_order_viewpoints(
        viewpoints, entries, position_tolerance_m=0.012,
        look_tolerance_deg=2.0):
    """
    Remove viewpoints already captured in this session.

    Preserve the planner's deterministic geometric order.  Diversity
    selection and smooth route ordering happen once in the Tesseract bridge,
    where the requested remaining-view count and the current calibrated
    camera position are both available.  Greedily choosing the farthest pose
    here and again in the bridge made the live arm alternate between the two
    ends of one orbit sector.
    """
    return [
        item for item in viewpoints
        if not viewpoint_is_duplicate(
            item, entries, position_tolerance_m, look_tolerance_deg)
    ]


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
    if accepted < 0 or accepted > maximum:
        raise ValueError('scan history count is outside the session bounds')
    for entry in entries:
        vector3(entry.get('desired_camera_position'), 'history camera position')
        vector3(entry.get('desired_look_at_direction'), 'history look direction')
    return {
        'session_id': session_id,
        'accepted_views': accepted,
        'max_views': maximum,
        'entries': list(entries),
    }
