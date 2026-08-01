"""Pure validation helpers for sequential automatic-mission startup gates."""

import math


MAX_JOINT_SOURCE_AGE_NS = 500_000_000


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
        health, now_ns, previous_generation='', maximum_age_sec=1.5):
    """Require one fresh, ready heartbeat from a newly started worker."""
    if not isinstance(health, dict):
        return 'Tesseract worker heartbeat is missing or invalid'
    generation = health.get('generation_id')
    if not isinstance(generation, str) or len(generation) != 32:
        return 'Tesseract worker generation ID is invalid'
    if any(character not in '0123456789abcdef'
           for character in generation):
        return 'Tesseract worker generation ID is invalid'
    if previous_generation and generation == previous_generation:
        return 'Tesseract worker has not started a new generation'
    try:
        age_sec = (int(now_ns) - int(health.get('written_at_ns'))) / 1e9
    except (TypeError, ValueError):
        return 'Tesseract worker heartbeat timestamp is invalid'
    if age_sec < -1.0 or age_sec > float(maximum_age_sec):
        return 'Tesseract worker heartbeat is stale'
    if health.get('worker_ready') is not True:
        detail = str(health.get('backend_error', '')).strip()
        return 'Tesseract worker is not ready' + (
            ': ' + detail if detail else '')
    if health.get('backend') != 'tesseract':
        return 'Tesseract worker backend is invalid'
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
