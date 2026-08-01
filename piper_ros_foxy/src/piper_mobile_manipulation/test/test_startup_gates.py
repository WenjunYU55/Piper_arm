from piper_mobile_manipulation.startup_gates import (
    joint_sample_rejection,
    joint_stability_update,
    readiness_stability_update,
    worker_health_rejection,
)


def valid_joint_sample(previous=None, previous_stamp=0):
    return joint_sample_rejection(
        previous, previous_stamp,
        [-0.02, -0.03, 0.03, 0.01, 0.28, 0.02],
        10_100_000_000, 10_110_000_000, 10_000_000_000)


def test_current_generation_ordered_joint_sample_is_accepted():
    assert valid_joint_sample() == ''


def test_previous_generation_and_delayed_joint_samples_are_rejected():
    assert 'predates' in joint_sample_rejection(
        None, 0, [0.0] * 6, 9_000_000_000, 10_000_000_000,
        10_000_000_000)
    assert 'stale' in joint_sample_rejection(
        None, 0, [0.0] * 6, 10_000_000_000, 11_000_000_000,
        9_000_000_000)


def test_out_of_order_is_rejected_but_coherent_controller_zero_is_accepted():
    previous = [-0.02, -0.03, 0.03, 0.01, 0.28, 0.02]
    assert 'not increasing' in joint_sample_rejection(
        previous, 10_100_000_000, previous, 10_100_000_000,
        10_110_000_000, 10_000_000_000)
    assert joint_sample_rejection(
        previous, 10_100_000_000,
        [-0.02, 0.0, 0.0, 0.01, 0.28, 0.02],
        10_105_000_000, 10_110_000_000, 10_000_000_000) == ''


def test_stability_window_accumulates_small_motion_instead_of_adjacent_deltas():
    reference, started = joint_stability_update(
        None, 0.0, [0.0] * 6, 1.0)
    reference, started = joint_stability_update(
        reference, started, [0.003] * 6, 1.1)
    assert reference == [0.0] * 6 and started == 1.0
    reference, started = joint_stability_update(
        reference, started, [0.006] * 6, 1.2)
    assert reference == [0.006] * 6 and started == 1.2


def test_stability_window_preserves_start_for_bounded_stationary_feedback():
    reference, started = joint_stability_update(
        [0.1] * 6, 2.0, [0.102] * 6, 3.0)
    assert reference == [0.1] * 6 and started == 2.0


def test_worker_requires_new_fresh_ready_generation():
    health = {
        'generation_id': 'a' * 32,
        'written_at_ns': 10_000_000_000,
        'worker_ready': True,
        'backend': 'tesseract',
        'backend_error': '',
    }
    assert worker_health_rejection(health, 10_100_000_000) == ''
    assert 'new generation' in worker_health_rejection(
        health, 10_100_000_000, previous_generation='a' * 32)
    assert 'stale' in worker_health_rejection(health, 12_000_000_000)


def test_worker_backend_failure_is_reported_verbatim():
    health = {
        'generation_id': 'b' * 32,
        'written_at_ns': 10_000_000_000,
        'worker_ready': False,
        'backend': 'tesseract',
        'backend_error': 'OMPL plugin unavailable',
    }
    assert worker_health_rejection(health, 10_100_000_000).endswith(
        'OMPL plugin unavailable')


def test_readiness_requires_one_continuous_good_window():
    started = readiness_stability_update(0.0, '', 10.0)
    assert started == 10.0
    assert readiness_stability_update(started, '', 11.5) == 10.0
    assert readiness_stability_update(started, 'camera unhealthy', 11.6) == 0.0
    assert readiness_stability_update(0.0, '', 12.0) == 12.0
