"""Pure planning and validation helpers for the supervised dry-run workflow."""

import math

import numpy as np


def point(value):
    return (float(value.x), float(value.y), float(value.z))


def distance(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def tracking_allows_target_motion_check(health, maximum_measurement_age_sec):
    """Use only a settled measured observation to declare target motion.

    A valid tracked position can be prediction-influenced while the
    eye-in-hand camera is moving. That transient is useful for monitoring but
    must not be compared with the pre-motion landmark as if it were a settled
    physical displacement.
    """
    return (
        health is not None
        and str(health.lifecycle_state) == 'TRACKING'
        and not bool(health.arm_moving)
        and bool(health.camera_settled)
        and not bool(health.prediction_only)
        and float(health.measurement_age_sec)
        <= float(maximum_measurement_age_sec)
    )


def heavy_refinement_status_action(payload, request_id):
    """Classify a heavy-worker status for one correlated cloud capture."""
    if not request_id or not isinstance(payload, dict):
        return 'ignore'
    payload_request_id = str(payload.get('request_id', ''))
    if payload_request_id != str(request_id):
        return 'ignore'
    state = str(payload.get('state', ''))
    if state == 'request_ignored_busy':
        return 'retry'
    if state in (
            'request_rejected', 'request_failed', 'worker_result_rejected'):
        return 'fail'
    return 'wait'


def semantic_scene_correlation_rejection(
        request_id, heavy_status, obstacle_scene):
    """Require one completed semantic result and exact-stamp 3D scene."""
    if not request_id:
        return 'occlusion probe request identity is missing'
    if not isinstance(heavy_status, dict):
        return 'dedicated occlusion semantic result is missing'
    if str(heavy_status.get('request_id', '')) != str(request_id):
        return 'dedicated occlusion semantic result is not request-correlated'
    state = str(heavy_status.get('state', '')).lower()
    if state != 'published':
        return 'dedicated occlusion semantic result is not a published target result'
    try:
        stamp = heavy_status['image_stamp']
        result_stamp = (
            int(stamp['sec']) * 1_000_000_000 + int(stamp['nanosec']))
        scene_stamp = (
            int(obstacle_scene.header.stamp.sec) * 1_000_000_000
            + int(obstacle_scene.header.stamp.nanosec))
    except (AttributeError, KeyError, TypeError, ValueError):
        return 'dedicated occlusion scene timestamp is invalid'
    if result_stamp <= 0 or scene_stamp != result_stamp:
        return 'dedicated occlusion 3D scene does not match the semantic image stamp'
    return ''


def corroborated_target_motion_rejection(
        tracked_displacement_m, threshold_m, landmark_status,
        landmark_status_fresh, surface_measurement_uncertainty_m=0.0):
    """Require agreeing independent geometry before declaring target motion.

    ``measurement_error_m`` is a residual against the stationary landmark.
    A gross mask/depth/TF outlier can therefore be large without representing
    the same displacement reported by the tracker. A bounded uncertainty also
    accounts for the visible surface centroid changing as a stationary cube is
    viewed from a new angle. Treat motion as corroborated only when both
    magnitudes exceed that uncertainty plus the physical-motion threshold and
    agree within the physical threshold.
    """
    displacement = float(tracked_displacement_m)
    threshold = float(threshold_m)
    uncertainty = max(0.0, float(surface_measurement_uncertainty_m))
    physical_displacement = max(0.0, displacement - uncertainty)
    if physical_displacement <= threshold:
        return ''
    if not landmark_status_fresh or not isinstance(landmark_status, dict):
        return (
            'tracked target shifted %.3fm but stationary-landmark '
            'corroboration is missing or stale' % displacement)
    state = str(landmark_status.get('state', '')).upper()
    if state not in ('LOCKED', 'RESCAN_NEEDED'):
        return (
            'tracked target shifted %.3fm and stationary-landmark state is %s'
            % (displacement, state or 'MISSING'))
    try:
        landmark_error = float(landmark_status['measurement_error_m'])
    except (KeyError, TypeError, ValueError):
        return (
            'tracked target shifted %.3fm but stationary-landmark '
            'measurement error is unavailable' % displacement)
    if not math.isfinite(landmark_error):
        return (
            'tracked target shifted %.3fm but stationary-landmark '
            'measurement error is invalid' % displacement)
    if (
            max(0.0, landmark_error - uncertainty) > threshold
            and abs(landmark_error - displacement) <= threshold):
        return (
            'cube landmark moved beyond tolerance '
            '(tracked %.3fm, independent landmark %.3fm)'
            % (displacement, landmark_error))
    return ''


def canonical_label(label):
    words = set(str(label or '').lower().replace('_', ' ').split())
    if words.intersection({'pen', 'marker'}):
        return 'pen'
    if 'stick' in words:
        return 'stick'
    return ' '.join(sorted(words))


def choose_removal_plan(
        instance, target, obstacles, config, support_points=None):
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
    drop = find_drop_zone(
        target, center, obstacles, config, support_points, size)
    if graspable and drop is not None:
        result.update({
            'valid': True, 'action': 'pick_and_place', 'reason': 'graspable with clear drop zone',
            'object_center': list(center), 'object_size': list(size), 'drop_center': list(drop),
            'approach': [center[0], center[1], upper[2] + config['approach_height_m']],
            'retreat': [drop[0], drop[1], drop[2] + config['approach_height_m']],
            'risk_score': min(1.0, config['target_clearance_m'] / max(target_clearance, 1e-6)),
            'drop_support_verified': bool(
                config.get('require_observed_drop_support', False)),
            'drop_zone_note': (
                'dense observed support patch verified'
                if config.get('require_observed_drop_support', False)
                else 'legacy geometric drop candidate'),
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


def find_drop_zone(
        target, source, obstacles, config, support_points=None,
        object_size=(0.0, 0.0, 0.0)):
    """Search observed tabletop-height candidates outside target/obstacle clearances."""
    if config.get('require_observed_drop_support', False):
        return observed_support_drop_zone(
            target, source, obstacles, config,
            support_points or [], object_size)
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


def observed_support_drop_zone(
        target, source, obstacles, config, support_points, object_size):
    """Choose a flat, locally supported and visibly clear placement patch."""
    points = np.asarray(support_points, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,):
        return None
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < int(config.get('drop_support_min_points', 16)):
        return None
    radius = max(
        float(config.get('drop_support_radius_m', 0.055)),
        0.5 * max(float(object_size[0]), float(object_size[1])) + 0.020)
    flatness = float(config.get('drop_support_max_stddev_m', 0.008))
    resolution = max(0.015, radius * 0.5)
    cells = {}
    for point_value in points:
        key = (
            int(math.floor(point_value[0] / resolution)),
            int(math.floor(point_value[1] / resolution)),
        )
        cells.setdefault(key, []).append(point_value)
    candidates = []
    for values in cells.values():
        patch_center = np.median(np.asarray(values), axis=0)
        lateral = np.linalg.norm(points[:, :2] - patch_center[:2], axis=1)
        local = points[lateral <= radius]
        if len(local) < int(config.get('drop_support_min_points', 16)):
            continue
        surface_z = float(np.median(local[:, 2]))
        support = local[np.abs(local[:, 2] - surface_z) <= flatness * 2.0]
        if (
                len(support) < int(config.get('drop_support_min_points', 16))
                or float(np.std(support[:, 2])) > flatness):
            continue
        relative = support[:, :2] - patch_center[:2]
        quadrants = {
            (int(delta[0] >= 0.0), int(delta[1] >= 0.0))
            for delta in relative
            if float(np.linalg.norm(delta)) >= radius * 0.45
        }
        if len(quadrants) < 4:
            continue
        footprint = lateral <= radius * 0.75
        overhead = points[
            footprint
            & (points[:, 2] > surface_z + 0.020)]
        if len(overhead):
            continue
        candidate = (
            float(patch_center[0]), float(patch_center[1]),
            surface_z + 0.5 * float(object_size[2]))
        if not clearance_ok(candidate, target, obstacles, None, config):
            continue
        candidates.append(candidate)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: (
            distance(candidate, target), -distance(candidate, source)))


def in_workspace(p, config):
    if not bool(config.get('enforce_static_workspace', False)):
        return True
    return (config['workspace_x_min'] <= p[0] <= config['workspace_x_max'] and
            config['workspace_y_min'] <= p[1] <= config['workspace_y_max'] and
            config['workspace_z_min'] <= p[2] <= config['workspace_z_max'])


def clearance_ok(candidate, target, obstacles, ignored_id, config):
    if distance(candidate, target) < config['drop_target_clearance_m']:
        return False
    for item in obstacles:
        if int(item.object_id) == ignored_id or not item.valid:
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


def should_cache_capture_cloud(state, accepted_views, modeled_views):
    """Keep the capture cloud across either valid message ordering."""
    return (
        str(state) == 'WAIT_CAPTURE'
        or int(accepted_views) > int(modeled_views)
    )


def capture_cloud_ready(accepted_views, modeled_views, cloud_available):
    """A cloud can be modeled only after its acceptance event is known."""
    return (
        bool(cloud_available)
        and int(accepted_views) > int(modeled_views)
    )


def occlusion_capture_rejection(payload, fresh):
    """Reject capture unless the depth checker freshly confirms a clear view."""
    if not fresh or not isinstance(payload, dict):
        return 'occlusion status is missing or stale'
    state = str(payload.get('occlusion_state', 'UNKNOWN')).upper()
    if state != 'CLEAR':
        reason = str(payload.get('reason', '')).strip()
        suffix = ': %s' % reason if reason else ''
        return 'view occlusion is %s%s' % (state, suffix)
    return ''
