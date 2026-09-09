"""Conservative clearance estimates for tightening a streamed command lead.

This monitor supplements the approved planner model; it cannot qualify physical
braking distance, unobserved obstacles or gaps in the compact collision proxy.
"""
import math

import numpy as np

from .motion import attached_box_world_corners, segment_distance


def segment_box_distance(start, end, minimum, maximum):
    """Compute segment-to-AABB distance over piecewise quadratic intervals."""
    start, end, minimum, maximum = [np.asarray(v, dtype=float)
                                   for v in (start, end, minimum, maximum)]
    delta = end - start
    breaks = [0., 1.]
    for i in range(3):
        if abs(delta[i]) > 1e-12:
            breaks.extend(float(t) for t in ((minimum[i] - start[i]) / delta[i],
                                             (maximum[i] - start[i]) / delta[i])
                          if 0. < t < 1.)
    breaks = sorted(set(breaks))
    best = math.inf
    for left, right in zip(breaks[:-1], breaks[1:]):
        point = start + .5 * (left + right) * delta
        active = (point < minimum) | (point > maximum)
        bound = np.where(point < minimum, minimum, maximum)
        a = delta[active]
        b = (start - bound)[active]
        denominator = float(np.dot(a, a))
        t = float(np.clip(-np.dot(a, b) / denominator, left, right)) if denominator else left
        point = start + t * delta
        residual = np.maximum(np.maximum(minimum - point, point - maximum), 0.)
        best = min(best, float(np.linalg.norm(residual)))
    return best


def clearance_margin(kinematics, joints, boxes, floor, radius, self_clearance,
                     holder_center, holder_size, holder_clearance, include_self=True):
    """Return remaining margin beyond the existing proxy clearance rules."""
    joints = np.asarray(joints, dtype=float)
    if joints.shape != (6,) or not np.all(np.isfinite(joints)):
        raise ValueError('measured collision-monitor joints are invalid')
    transforms = kinematics.chain_transforms(joints)
    points = [np.zeros(3)] + [t[:3, 3] for t in transforms]
    points.append((transforms[-1] @ kinematics.link6_from_camera)[:3, 3])
    segments = [(i, points[i], points[i + 1]) for i in range(len(points) - 1)
                if np.linalg.norm(points[i + 1] - points[i]) > 1e-7]
    # The first two joint origins are fixed at base height; their constant
    # distance cannot be consumed by tracking error. Admission still checks them.
    margins = [(min(float(p[2]) for p in points[3:]) - floor - radius, 'arm/table')]
    for i, a, b in segments:
        for j, c, d in segments:
            if include_self and j > i + 2:
                margins.append((segment_distance(a, b, c, d) - self_clearance, 'arm/self'))
    corners = attached_box_world_corners(kinematics, joints, holder_center, holder_size)
    margins.append((float(corners[:, 2].min()) - floor - holder_clearance, 'holder/table'))
    holder_min, holder_max = corners.min(axis=0), corners.max(axis=0)
    for box in boxes:
        for _, a, b in segments:
            margins.append((segment_box_distance(a, b, box.minimum, box.maximum) - radius,
                            'arm/obstacle'))
        residual = np.maximum(np.maximum(box.minimum - holder_max, holder_min - box.maximum), 0.)
        margins.append((float(np.linalg.norm(residual)) - holder_clearance, 'holder/obstacle'))
    if any(not math.isfinite(value) for value, _ in margins):
        raise ValueError('collision-monitor clearance is nonfinite')
    return min(margins)


def tracking_limit(clearance_m, closing_speed_m_s, reaction_sec,
                   near_limit_rad, open_limit_rad, transition_m):
    """Tighten early for closing motion; never loosen the existing open limit."""
    values = [clearance_m, closing_speed_m_s, reaction_sec,
              near_limit_rad, open_limit_rad, transition_m]
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError('collision-monitor inputs are nonfinite')
    if min(near_limit_rad, open_limit_rad, transition_m) <= 0 or reaction_sec < 0:
        raise ValueError('collision-monitor configuration is invalid')
    near = min(near_limit_rad, open_limit_rad)
    usable = clearance_m - max(0., closing_speed_m_s) * reaction_sec
    fraction = min(1., max(0., usable / transition_m))
    return min(open_limit_rad, near + (open_limit_rad - near) * fraction)
