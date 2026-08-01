"""Pure helpers for distinguishing scan and rough-target acquisition execution."""

import math

import numpy as np

from piper_mobile_manipulation.heavy_refresh_contract import stamp_dict_to_ns


MULTIVIEW_SCAN = 'MULTIVIEW_SCAN'
ROUGH_ACQUISITION = 'ROUGH_ACQUISITION'
VALID_PLAN_KINDS = (MULTIVIEW_SCAN, ROUGH_ACQUISITION)


def uses_bootstrap_static_scene(plan_kind, viewpoint_index):
    """Only the first rough-acquisition segment omits perception obstacles."""
    return plan_kind == ROUGH_ACQUISITION and int(viewpoint_index) == 0


def plan_count_rejection(
        plan_kind, count, minimum_scan_views, maximum_acquisition_views,
        session_accepted_views=0, session_maximum_views=None):
    """Return a fail-closed reason for an invalid Tesseract plan shape."""
    if plan_kind not in VALID_PLAN_KINDS:
        return 'unsupported plan_kind=%s' % plan_kind
    if count < 1:
        return 'Tesseract proposal has no viewpoints'
    if plan_kind == MULTIVIEW_SCAN:
        maximum = (
            int(minimum_scan_views)
            if session_maximum_views is None else int(session_maximum_views))
        accepted = int(session_accepted_views)
        if accepted < 0 or accepted >= maximum:
            return 'scan session accepted-view count is invalid'
        if count != maximum - accepted:
            return (
                'Tesseract proposal does not contain every remaining '
                'session viewpoint')
    if (
            plan_kind == ROUGH_ACQUISITION
            and count > max(1, int(maximum_acquisition_views))):
        return 'rough acquisition proposal exceeds the bounded viewpoint count'
    return ''


def commanded_speed_percent(configured_speed, plan_kind, tracking_scale):
    """Return the operator-selected SDK MoveJ speed within its 1-100% range.

    Tracking health is a readiness/diagnostic input, not a second motion
    controller. Binding a plan to a live confidence-derived multiplier made an
    otherwise identical 5% plan reject itself while Tesseract was computing.
    """
    del plan_kind, tracking_scale
    configured = max(1.0, min(100.0, float(configured_speed)))
    return configured


def planned_speed_rejection(
        configured_speed, plan_kind, tracking_scale, planned_speed):
    """Validate a plan's bound speed without requiring a volatile exact scale."""
    configured = max(1.0, min(100.0, float(configured_speed)))
    try:
        planned = float(planned_speed)
    except (TypeError, ValueError):
        return 'Tesseract execution speed is invalid'
    if not math.isfinite(planned) or planned < 1.0 or planned > configured + 1e-6:
        return 'Tesseract execution speed is outside the configured limit'
    del tracking_scale
    if plan_kind == ROUGH_ACQUISITION:
        if abs(planned - configured) > 1e-4:
            return 'Tesseract acquisition speed does not match the selected speed'
    elif plan_kind == MULTIVIEW_SCAN:
        # The configured SDK percentage is the complete motion-speed contract.
        # Tracking cannot raise this bound and must not invalidate it.
        pass
    else:
        return 'unsupported plan_kind=%s' % plan_kind
    return ''


def acquisition_tracking_locked(
        health,
        target_status,
        tracking_updated_at,
        target_updated_at,
        refresh_started_at,
        now,
        data_timeout_sec,
        max_measurement_age_sec):
    """Require fresh, measured tracking produced after the acquisition refresh."""
    if health is None or refresh_started_at is None:
        return False
    timestamps = (
        tracking_updated_at, target_updated_at, refresh_started_at, now,
        data_timeout_sec, max_measurement_age_sec,
    )
    if not all(math.isfinite(float(value)) for value in timestamps):
        return False
    if tracking_updated_at < refresh_started_at or target_updated_at < refresh_started_at:
        return False
    if now - tracking_updated_at > data_timeout_sec:
        return False
    if now - target_updated_at > data_timeout_sec:
        return False
    if str(getattr(health, 'lifecycle_state', '')) != 'TRACKING':
        return False
    if bool(getattr(health, 'prediction_only', True)):
        return False
    if not bool(getattr(health, 'camera_settled', False)):
        return False
    try:
        measurement_age = float(health.measurement_age_sec)
    except (AttributeError, TypeError, ValueError):
        return False
    if not math.isfinite(measurement_age) or measurement_age > max_measurement_age_sec:
        return False
    return str(target_status).upper() in ('TRACKING', 'LOCKED')


def measured_target_lock_rejection(
        target,
        health,
        target_status,
        target_updated_at,
        tracking_updated_at,
        status_updated_at,
        now,
        data_timeout_sec,
        max_measurement_age_sec):
    """Return why a stationary measured target is not ready for scan planning."""
    if target is None or health is None:
        return 'measured target or tracking health is missing'
    times = (
        target_updated_at, tracking_updated_at, status_updated_at, now,
        data_timeout_sec, max_measurement_age_sec,
    )
    if not all(math.isfinite(float(value)) for value in times):
        return 'measured-lock timing is invalid'
    if now - target_updated_at > data_timeout_sec:
        return 'tracked target is stale'
    if now - tracking_updated_at > data_timeout_sec:
        return 'tracking health is stale'
    if now - status_updated_at > data_timeout_sec:
        return 'target status is stale'
    if str(target_status).upper() != 'LOCKED':
        return 'target status is not LOCKED'
    if str(getattr(health, 'lifecycle_state', '')) != 'TRACKING':
        return 'tracking lifecycle is not TRACKING'
    if bool(getattr(health, 'prediction_only', True)):
        return 'tracking is prediction-only'
    if not bool(getattr(health, 'camera_settled', False)):
        return 'camera is not settled'
    try:
        measurement_age = float(health.measurement_age_sec)
    except (AttributeError, TypeError, ValueError):
        return 'tracking measurement age is invalid'
    if (
            not math.isfinite(measurement_age)
            or measurement_age > max_measurement_age_sec):
        return 'tracking measurement is stale'
    if str(getattr(target.header, 'frame_id', '')) != 'base_link':
        return 'tracked target frame is not base_link'
    if not bool(getattr(target, 'valid', False)):
        return 'tracked target is invalid'
    if not bool(getattr(target, 'stable', False)):
        return 'tracked target is not stable'
    position = np.asarray([
        target.position.x, target.position.y, target.position.z,
    ], dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        return 'tracked target position is invalid'
    return ''


def heavy_refresh_status_action(payload, request_id, minimum_image_stamp_ns):
    """Classify only the GroundingDINO status belonging to this exact look."""
    if not isinstance(payload, dict):
        return 'ignore', 'invalid status payload', None
    state = str(payload.get('state', '')).lower()
    if state == 'idle':
        return 'idle', '', None
    if str(payload.get('request_id', '')) != str(request_id):
        return 'ignore', 'status belongs to another request', None
    if state == 'request_ignored_busy':
        return 'busy', 'heavy worker is busy', None
    if state in ('waiting_for_fresh_image', 'waiting_for_image'):
        return 'waiting_for_frame', '', None
    if state == 'queued':
        try:
            image_stamp_ns = stamp_dict_to_ns(payload.get('image_stamp'))
        except ValueError as exc:
            return 'abort', str(exc), None
        if image_stamp_ns < int(minimum_image_stamp_ns):
            return 'abort', 'GroundingDINO job used a pre-settle image', image_stamp_ns
        return 'queued', '', image_stamp_ns
    if state in ('published', 'worker_result_rejected'):
        try:
            image_stamp_ns = stamp_dict_to_ns(payload.get('image_stamp'))
        except ValueError as exc:
            return 'abort', str(exc), None
        if image_stamp_ns < int(minimum_image_stamp_ns):
            return 'abort', 'GroundingDINO result used a pre-settle image', image_stamp_ns
        if state == 'published':
            return 'detected', '', image_stamp_ns
        worker_status = str(payload.get('worker_status', ''))
        if worker_status == 'target_mask_missing':
            try:
                obstacle_count = int(payload.get('obstacle_count', -1))
            except (TypeError, ValueError):
                obstacle_count = -1
            if obstacle_count == 0:
                return (
                    'not_found_clear',
                    'GroundingDINO did not find the target or any obstacles',
                    image_stamp_ns,
                )
            return 'not_found', 'GroundingDINO did not find the target', image_stamp_ns
        return 'abort', 'GroundingDINO result rejected: %s' % (
            worker_status or 'unknown worker status'), image_stamp_ns
    if state in ('request_failed', 'request_rejected'):
        return 'abort', str(payload.get('error', state)), None
    return 'ignore', 'non-terminal status', None


def correlated_obstacle_scene_status(
        scene, updated_at, now, timeout_sec, minimum_image_stamp_ns):
    """Classify the scene needed before a subsequent acquisition move."""
    values = (updated_at, now, timeout_sec)
    if scene is None or not all(math.isfinite(float(value)) for value in values):
        return 'waiting', 'post-settle obstacle scene is missing'
    if now - updated_at > timeout_sec:
        return 'waiting', 'post-settle obstacle scene is stale'
    try:
        stamp = scene.header.stamp
        scene_stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError):
        return 'blocked', 'post-settle obstacle scene has an invalid timestamp'
    if scene_stamp_ns < int(minimum_image_stamp_ns):
        return 'waiting', 'obstacle scene predates the matching semantic result'
    instances = list(getattr(scene, 'instances', []))
    if any(not bool(getattr(item, 'valid', False)) for item in instances):
        return 'blocked', 'invalid obstacle geometry is present'
    if bool(getattr(scene, 'scene_blocked', True)) and not instances:
        return 'blocked', 'scene_blocked: %s' % str(
            getattr(scene, 'blocking_reason', 'unknown reason'))
    return 'ready', ''


def acquired_target_rejection(
        target, health, target_status, target_updated_at, tracking_updated_at,
        status_updated_at, detection_completed_at, now, data_timeout_sec,
        max_measurement_age_sec, minimum_image_stamp_ns, rough_target,
        max_target_offset_m):
    """Require a new measured stable base-frame lock for the processed image."""
    if target is None or health is None or detection_completed_at is None:
        return 'measured target or tracking health is missing'
    times = (
        target_updated_at, tracking_updated_at, status_updated_at,
        detection_completed_at, now,
    )
    if not all(math.isfinite(float(value)) for value in times):
        return 'acquisition timing is invalid'
    if min(target_updated_at, tracking_updated_at, status_updated_at) < (
            detection_completed_at):
        return 'tracking lock predates the matching GroundingDINO result'
    if now - target_updated_at > data_timeout_sec:
        return 'tracked target is stale'
    if now - tracking_updated_at > data_timeout_sec:
        return 'tracking health is stale'
    if now - status_updated_at > data_timeout_sec:
        return 'target status is stale'
    if str(target_status).upper() != 'LOCKED':
        return 'target status is not LOCKED'
    if str(getattr(health, 'lifecycle_state', '')) != 'TRACKING':
        return 'tracking lifecycle is not TRACKING'
    if bool(getattr(health, 'prediction_only', True)):
        return 'tracking is prediction-only'
    if not bool(getattr(health, 'camera_settled', False)):
        return 'camera is not settled'
    try:
        measurement_age = float(health.measurement_age_sec)
    except (AttributeError, TypeError, ValueError):
        return 'tracking measurement age is invalid'
    if not math.isfinite(measurement_age) or measurement_age > max_measurement_age_sec:
        return 'tracking measurement is stale'
    if str(getattr(target.header, 'frame_id', '')) != 'base_link':
        return 'tracked target frame is not base_link'
    stamp = target.header.stamp
    target_stamp_ns = int(stamp.sec) * 1000000000 + int(stamp.nanosec)
    if target_stamp_ns < int(minimum_image_stamp_ns):
        return 'tracked target measurement predates the GroundingDINO frame'
    if not bool(getattr(target, 'valid', False)) or not bool(
            getattr(target, 'stable', False)):
        return 'tracked target is not valid and stable'
    position = np.asarray([
        target.position.x, target.position.y, target.position.z,
    ], dtype=float)
    rough = np.asarray(rough_target, dtype=float)
    if position.shape != (3,) or rough.shape != (3,) or not np.all(
            np.isfinite(position)) or not np.all(np.isfinite(rough)):
        return 'target association coordinates are invalid'
    offset = float(np.linalg.norm(position - rough))
    if offset > float(max_target_offset_m):
        return 'measured target is %.3fm from the rough hint (maximum %.3fm)' % (
            offset, float(max_target_offset_m))
    return ''
