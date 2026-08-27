"""Pure validation helpers for sequential automatic-mission startup gates."""

import math


MAX_JOINT_SOURCE_AGE_NS = 500_000_000


def post_enable_sample_rejection(
        joints, joint_received_at, status, status_received_at,
        enable_completed_at, now, joint_max_age_sec=0.25,
        status_max_age_sec=0.5):
    """Reject feedback that cannot prove a healthy powered startup sample."""
    try:
        values = [float(value) for value in joints]
        joint_at = float(joint_received_at)
        status_at = float(status_received_at)
        enabled_at = float(enable_completed_at)
        current = float(now)
    except (TypeError, ValueError):
        return 'post-enable feedback fields are invalid'
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        return 'post-enable joint feedback is not six finite positions'
    if joint_at <= enabled_at or status_at <= enabled_at:
        return 'waiting for joint and arm feedback received after enable'
    if (
            joint_at > current or status_at > current
            or current - joint_at > float(joint_max_age_sec)
            or current - status_at > float(status_max_age_sec)):
        return 'post-enable joint or arm feedback is stale or in the future'
    if status is None:
        return 'post-enable arm status is missing'
    if int(getattr(status, 'err_code', 0)) != 0:
        return 'post-enable arm err_code=%d' % int(status.err_code)
    if any(bool(getattr(
            status, 'joint_%d_angle_limit' % index, False))
            for index in range(1, 7)):
        return 'post-enable arm reports a joint angle-limit fault'
    if any(bool(getattr(
            status, 'communication_status_joint_%d' % index, False))
            for index in range(1, 7)):
        return 'post-enable arm reports a joint communication fault'
    if not bool(getattr(status, 'motor_feedback_valid', False)):
        return 'post-enable low-speed motor feedback is invalid'
    states = tuple(bool(getattr(
        status, 'motor_%d_driver_enabled' % index, False))
        for index in range(1, 7))
    if not all(states):
        return 'post-enable all-six motor authority is unproved: %s' % (
            states,)
    faults = tuple(str(value) for value in getattr(
        status, 'motor_faults', ()) if str(value))
    if faults:
        return 'post-enable motor faults=%s' % ','.join(faults)
    watchdog = str(getattr(status, 'motor_watchdog_reason', '')).strip()
    if watchdog:
        return 'post-enable motor watchdog=%s' % watchdog
    return ''


def joint_sample_rejection(
        previous_positions, previous_stamp_ns, positions, stamp_ns,
        receive_ns, generation_started_ns):
    """Reject cached, out-of-order, or malformed joint feedback.

    Joint-pair coherence is enforced by the driver's independent raw-CAN
    assembler before this gate. A stable zero from fresh advancing pair frames
    is therefore measured feedback, not a missing position.
    """
    try:
        values = [float(value) for value in positions]
        source_ns = int(stamp_ns)
        arrival_ns = int(receive_ns)
        generation_ns = int(generation_started_ns)
    except (TypeError, ValueError):
        return 'joint feedback fields are invalid'
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        return 'joint feedback is not six finite positions'
    if source_ns <= 0:
        return 'joint feedback source timestamp is invalid'
    if source_ns < generation_ns:
        return 'joint feedback predates the current driver generation'
    source_age_ns = arrival_ns - source_ns
    if abs(source_age_ns) > MAX_JOINT_SOURCE_AGE_NS:
        return 'joint feedback source timestamp is stale or in the future'
    if previous_stamp_ns and source_ns <= int(previous_stamp_ns):
        return 'joint feedback source timestamp is not increasing'
    if previous_positions is not None and len(previous_positions) != 6:
        return 'previous joint feedback is invalid'
    return ''


def joint_stability_update(reference, stable_since, positions, received_at,
                           max_window_drift_rad=0.005):
    """Track a bounded-position window instead of only adjacent deltas."""
    try:
        values = [float(value) for value in positions]
        now = float(received_at)
        baseline = (
            None if reference is None
            else [float(value) for value in reference])
    except (TypeError, ValueError):
        return None, 0.0
    if (
            len(values) != 6
            or not all(math.isfinite(value) for value in values)
            or not math.isfinite(now)):
        return None, 0.0
    if baseline is None or len(baseline) != 6:
        return values, now
    drift = max(abs(value - start) for value, start in zip(values, baseline))
    if drift > float(max_window_drift_rad):
        return values, now
    return (
        baseline,
        float(stable_since) if float(stable_since) > 0.0 else now,
    )


def worker_health_rejection(
        health, now_ns, previous_generation='', maximum_age_sec=1.5,
        expected_backend='tesseract'):
    """Require one fresh, ready heartbeat from a newly started worker."""
    backend = str(expected_backend).strip().lower()
    label = 'Tesseract' if backend == 'tesseract' else 'cuRobo'
    if not isinstance(health, dict):
        return '%s worker heartbeat is missing or invalid' % label
    generation = health.get('generation_id')
    if not isinstance(generation, str) or len(generation) != 32:
        return '%s worker generation ID is invalid' % label
    if any(character not in '0123456789abcdef'
           for character in generation):
        return '%s worker generation ID is invalid' % label
    if previous_generation and generation == previous_generation:
        return '%s worker has not started a new generation' % label
    try:
        age_sec = (int(now_ns) - int(health.get('written_at_ns'))) / 1e9
    except (TypeError, ValueError):
        return '%s worker heartbeat timestamp is invalid' % label
    if age_sec < -1.0 or age_sec > float(maximum_age_sec):
        return '%s worker heartbeat is stale' % label
    if health.get('worker_ready') is not True:
        detail = str(health.get('backend_error', '')).strip()
        return '%s worker is not ready' % label + (
            ': ' + detail if detail else '')
    if health.get('backend') != backend:
        return '%s worker backend is invalid' % label
    return ''


def readiness_stability_update(stable_since, rejection, observed_at):
    """Require a continuous ready window instead of accepting one good tick."""
    try:
        started = float(stable_since)
        now = float(observed_at)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(now):
        return 0.0
    if str(rejection):
        return 0.0
    return started if started > 0.0 else now
