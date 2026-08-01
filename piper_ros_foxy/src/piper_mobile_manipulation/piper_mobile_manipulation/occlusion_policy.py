"""Target-centred occluder qualification and bounded contact policy."""

from dataclasses import dataclass
import math


HAND_LABELS = frozenset(('hand', 'finger', 'person', 'human'))
MOVABLE_LABELS = frozenset(('leaf', 'branch'))


@dataclass(frozen=True)
class OccluderEvidence:
    track_id: str
    object_id: int
    label: str
    observation_count: int
    confirmed_in_probe: bool
    target_overlap_ratio: float
    closer_depth_ratio: float
    predicted_surface_gain: float
    predicted_unlocked_viewpoints: int
    confidence: float
    valid: bool
    uncertainty_m: float
    size_xyz_m: tuple


def canonical_label(label):
    return ' '.join(str(label or '').strip().lower().replace('_', ' ').split())


def evidence_rejection(evidence):
    """Require semantic and geometric corroboration before contact planning."""
    label = canonical_label(evidence.label)
    if label in HAND_LABELS:
        return 'hand or person is a terminal workspace blocker'
    if label not in MOVABLE_LABELS:
        return 'object is not in the qualified movable leaf/branch profile'
    if not evidence.valid:
        return 'occluder geometry is invalid'
    if not evidence.track_id:
        return 'occluder track identity is missing'
    if evidence.observation_count < 2 or not evidence.confirmed_in_probe:
        return 'occluder was not confirmed in both initial and probe views'
    values = (
        evidence.target_overlap_ratio,
        evidence.closer_depth_ratio,
        evidence.predicted_surface_gain,
        evidence.confidence,
        evidence.uncertainty_m,
    )
    if not all(math.isfinite(float(value)) for value in values):
        return 'occluder evidence contains a non-finite value'
    if evidence.closer_depth_ratio < 0.05:
        return 'closer-depth target occlusion is below 5 percent'
    if evidence.target_overlap_ratio < 0.05:
        return 'target overlap is below 5 percent'
    if (
            evidence.predicted_unlocked_viewpoints < 2
            and evidence.predicted_surface_gain < 0.10):
        return 'predicted removal benefit is below the scan threshold'
    if evidence.uncertainty_m > 0.025:
        return 'occluder position uncertainty exceeds 25mm'
    return ''


def select_action(evidence, pick_path_valid, push_path_valid,
                  destination_valid, push_increment_index=0):
    """Prefer a qualified pick, otherwise one bounded outward push increment."""
    rejection = evidence_rejection(evidence)
    if rejection:
        return {'action': 'none', 'valid': False, 'reason': rejection}
    size = tuple(float(value) for value in evidence.size_xyz_m)
    if len(size) != 3 or not all(math.isfinite(value) and value >= 0 for value in size):
        return {'action': 'none', 'valid': False, 'reason': 'object size is invalid'}
    if max(size[0], size[1]) <= 0.070 and pick_path_valid and destination_valid:
        return {
            'action': 'pick_and_place',
            'valid': True,
            'reason': 'qualified grasp section and destination are collision-valid',
            'contact_speed_percent': 10.0,
        }
    increment = int(push_increment_index)
    if increment < 0 or increment >= 3:
        return {'action': 'none', 'valid': False, 'reason': 'push increment limit reached'}
    if push_path_valid:
        return {
            'action': 'push',
            'valid': True,
            'reason': 'pick unavailable; bounded outward push is collision-valid',
            'push_distance_m': 0.010 if increment == 0 else 0.030,
            'contact_speed_percent': 10.0,
        }
    return {
        'action': 'route_around',
        'valid': False,
        'reason': 'no safe contact action; retain object in the planning scene',
    }


def placement_rejection(candidate, target, obstacles, table_bounds,
                        forbidden_footprints=()):
    """Validate a target-centred tabletop placement footprint."""
    point = tuple(float(value) for value in candidate)
    if len(point) != 3 or not all(math.isfinite(value) for value in point):
        return 'placement point is invalid'
    if math.dist(point, tuple(target)) < 0.120:
        return 'placement is within 120mm of the target'
    for obstacle in obstacles:
        if math.dist(point, tuple(obstacle)) < 0.080:
            return 'placement is within 80mm of another object'
    x_min, x_max, y_min, y_max = [float(value) for value in table_bounds]
    if not (
            x_min + 0.050 <= point[0] <= x_max - 0.050
            and y_min + 0.050 <= point[1] <= y_max - 0.050):
        return 'placement is within 50mm of a table edge'
    for footprint in forbidden_footprints:
        fx_min, fx_max, fy_min, fy_max = [float(value) for value in footprint]
        if fx_min <= point[0] <= fx_max and fy_min <= point[1] <= fy_max:
            return 'placement intersects a robot, camera, home, or scan footprint'
    return ''
