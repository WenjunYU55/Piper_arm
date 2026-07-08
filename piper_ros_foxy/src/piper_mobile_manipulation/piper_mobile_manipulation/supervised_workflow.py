"""Pure planning and validation helpers for the supervised dry-run workflow."""

import math


def point(value):
    return (float(value.x), float(value.y), float(value.z))


def distance(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def canonical_label(label):
    words = set(str(label or '').lower().replace('_', ' ').split())
    if words.intersection({'pen', 'marker'}):
        return 'pen'
    if 'ground' in words:
        return 'ground'
    return ' '.join(sorted(words))


def choose_removal_plan(instance, target, obstacles, config):
    """Return a conservative dry-run pick/push plan or a rejection."""
    result = {
        'valid': False, 'dry_run': True, 'execute': False,
        'object_id': int(instance.object_id),
        'label': canonical_label(instance.semantic_label),
    }
    if not instance.valid:
        result['reason'] = 'invalid obstacle geometry: %s' % instance.validity_reason
        return result
    if result['label'] not in set(config['movable_whitelist']):
        result['reason'] = 'label is not whitelisted'
        return result
    center = point(instance.base_centroid)
    lower = point(instance.base_bounds_min)
    upper = point(instance.base_bounds_max)
    if not in_workspace(center, config):
        result['reason'] = 'obstacle center is outside configured workspace'
        return result
    size = tuple(max(0.0, hi - lo) for lo, hi in zip(lower, upper))
    target_clearance = distance(center, target)
    if target_clearance < config['target_clearance_m']:
        result['reason'] = 'obstacle is inside target clearance'
        return result

    graspable = max(size[0], size[1]) <= config['max_grasp_width_m']
    drop = find_drop_zone(target, center, obstacles, config)
    if graspable and drop is not None:
        result.update({
            'valid': True, 'action': 'pick_and_place', 'reason': 'graspable with clear drop zone',
            'object_center': list(center), 'object_size': list(size), 'drop_center': list(drop),
            'approach': [center[0], center[1], upper[2] + config['approach_height_m']],
            'retreat': [drop[0], drop[1], drop[2] + config['approach_height_m']],
            'risk_score': min(1.0, config['target_clearance_m'] / max(target_clearance, 1e-6)),
            'drop_support_verified': False,
            'drop_zone_note': 'dry-run candidate; dense support-surface verification unavailable',
        })
        return result

    dx, dy = center[0] - target[0], center[1] - target[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        result['reason'] = 'no push direction away from target'
        return result
    direction = (dx / norm, dy / norm, 0.0)
    end = (center[0] + direction[0] * config['push_distance_m'],
           center[1] + direction[1] * config['push_distance_m'], center[2])
    if (not in_workspace(end, config) or
            not clearance_ok(end, target, obstacles, instance.object_id, config)):
        result['reason'] = 'no safe pick or outward push path'
        return result
    result.update({
        'valid': True, 'action': 'push', 'reason': 'pick unavailable; outward push is clear',
        'object_center': list(center), 'object_size': list(size),
        'push_direction': list(direction), 'push_end': list(end),
        'approach': [center[0] - direction[0] * config['pre_push_offset_m'],
                     center[1] - direction[1] * config['pre_push_offset_m'], center[2]],
        'retreat': [end[0], end[1], end[2] + config['approach_height_m']],
        'risk_score': min(1.0, config['target_clearance_m'] / max(target_clearance, 1e-6) + 0.2),
    })
    return result


def find_drop_zone(target, source, obstacles, config):
    """Search observed tabletop-height candidates outside target/obstacle clearances."""
    radius = config['drop_search_radius_m']
    for ring in (1.0, 1.35):
        for degrees in range(0, 360, 30):
            angle = math.radians(degrees)
            candidate = (target[0] + ring * radius * math.cos(angle),
                         target[1] + ring * radius * math.sin(angle), source[2])
            if in_workspace(candidate, config) and clearance_ok(
                    candidate, target, obstacles, None, config):
                return candidate
    return None


def in_workspace(p, config):
    return (config['workspace_x_min'] <= p[0] <= config['workspace_x_max'] and
            config['workspace_y_min'] <= p[1] <= config['workspace_y_max'] and
            config['workspace_z_min'] <= p[2] <= config['workspace_z_max'])


def clearance_ok(candidate, target, obstacles, ignored_id, config):
    if distance(candidate, target) < config['drop_target_clearance_m']:
        return False
    for item in obstacles:
        if int(item.object_id) == ignored_id or not item.valid:
            continue
        # Ground is semantically safe for placement, but trajectory generation
        # still treats its fitted plane as a no-penetration boundary.
        if canonical_label(item.semantic_label) == 'ground':
            continue
        if distance(candidate, point(item.base_centroid)) < config['drop_obstacle_clearance_m']:
            return False
    return True


def cloud_model(points, frame_id, accepted_views):
    """Compute robust point-cloud center and percentile bounds."""
    import numpy as np
    values = np.asarray(points, dtype=np.float64)
    values = values[np.all(np.isfinite(values), axis=1)] if values.size else values.reshape((0, 3))
    if not len(values):
        return {'valid': False, 'reason': 'empty cloud', 'frame_id': frame_id,
                'accepted_views': accepted_views}
    lower = np.percentile(values, 2.0, axis=0)
    upper = np.percentile(values, 98.0, axis=0)
    clipped = values[np.all((values >= lower) & (values <= upper), axis=1)]
    center = np.median(clipped if len(clipped) else values, axis=0)
    return {
        'valid': True, 'frame_id': frame_id, 'accepted_views': int(accepted_views),
        'point_count': int(len(values)), 'center': center.tolist(),
        'bounds_min': lower.tolist(), 'bounds_max': upper.tolist(),
    }
