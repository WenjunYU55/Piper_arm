import numpy as np
import pytest
from types import SimpleNamespace
from rclpy.time import Time
from sensor_msgs.msg import Image

from piper_mobile_manipulation.depth_to_3d_node import (
    DepthTo3DNode,
    depth_jump_reacquisition,
    median_component_camera_point,
    primary_depth_component,
)
from piper_mobile_manipulation.target_tracker_node import (
    finite_target_measurement,
    motion_measurement_rejection,
    prediction_only_is_valid,
    TargetTrackerNode,
)
from piper_mobile_manipulation.utils.kalman_filter import (
    ConstantVelocityKalmanFilter,
)
from piper_mobile_manipulation.utils.target_depth import (
    select_target_depth_component,
)
from piper_mobile_manipulation.msg import TrackedTarget


def test_depth_jump_accepts_change_and_disables_with_nonpositive_limit():
    assert depth_jump_reacquisition(
        0.40, 0.45, 0.20, None, 0, 3, 0.03
    ) == (True, None, 0, False)
    assert depth_jump_reacquisition(
        0.40, 0.80, 0.0, None, 0, 3, 0.03
    ) == (True, None, 0, False)


def test_depth_jump_requires_consistent_reacquisition_samples():
    accepted, pending, count, resynced = depth_jump_reacquisition(
        0.30, 0.62, 0.20, None, 0, 3, 0.03
    )
    assert (accepted, count, resynced) == (False, 1, False)

    accepted, pending, count, resynced = depth_jump_reacquisition(
        0.30, 0.61, 0.20, pending, count, 3, 0.03
    )
    assert (accepted, count, resynced) == (False, 2, False)

    accepted, pending, count, resynced = depth_jump_reacquisition(
        0.30, 0.615, 0.20, pending, count, 3, 0.03
    )
    assert (accepted, pending, count, resynced) == (True, None, 0, True)


def test_depth_jump_restarts_consistency_count_for_an_unrelated_outlier():
    accepted, pending, count, _ = depth_jump_reacquisition(
        0.30, 0.62, 0.20, None, 0, 3, 0.03
    )
    assert not accepted
    accepted, pending, count, resynced = depth_jump_reacquisition(
        0.30, 0.90, 0.20, pending, count, 3, 0.03
    )
    assert (accepted, pending, count, resynced) == (
        False, 0.90, 1, False)


def test_primary_depth_component_rejects_adjacent_background_mode():
    depth = np.full((7, 9), 0.42, dtype=float)
    depth[1:6, 2:7] = np.asarray([
        [0.330, 0.334, 0.338, 0.342, 0.346],
        [0.332, 0.336, 0.340, 0.344, 0.348],
        [0.334, 0.338, 0.342, 0.346, 0.350],
        [0.336, 0.340, 0.344, 0.348, 0.352],
        [0.338, 0.342, 0.346, 0.350, 0.354],
    ])
    valid = np.ones_like(depth, dtype=bool)

    selected, seed_depth = primary_depth_component(
        valid, depth, 4.0, 3.0, 0.03, 0.015)

    assert seed_depth == 0.342
    assert int(np.count_nonzero(selected)) == 25
    assert np.all(depth[selected] < 0.40)


def test_primary_depth_component_keeps_only_nearest_semantic_surface():
    depth = np.full((5, 9), np.nan, dtype=float)
    depth[1:4, 1:4] = 0.34
    depth[1:4, 6:9] = 0.34
    valid = np.isfinite(depth)

    selected, _ = primary_depth_component(
        valid, depth, 2.0, 2.0, 0.03, 0.015)

    assert int(np.count_nonzero(selected)) == 9
    assert np.all(selected[1:4, 1:4])
    assert not np.any(selected[:, 6:9])


def test_primary_depth_component_can_be_disabled():
    depth = np.asarray([[0.3, 0.7]], dtype=float)
    valid = np.ones_like(depth, dtype=bool)

    selected, seed_depth = primary_depth_component(
        valid, depth, 0.0, 0.0, 0.0, 0.0)

    assert seed_depth == 0.3
    assert np.array_equal(selected, valid)


def test_depth_modes_do_not_leak_across_a_gradual_background_bridge():
    depth = np.full((30, 40), 0.42, dtype=float)
    depth[5:25, 5:24] = 0.34
    for column, value in enumerate(np.linspace(0.35, 0.41, 8), start=23):
        depth[12:18, column] = value
    support = np.ones_like(depth, dtype=bool)

    selected, report = select_target_depth_component(
        support, depth, center_u=15.0, center_v=15.0,
        minimum_points=20, minimum_support_fraction=0.15)

    assert report['selected_depth_m'] < 0.36
    assert np.all(depth[selected] < 0.38)


def test_ambiguous_depth_layers_fail_closed():
    depth = np.full((20, 20), np.nan, dtype=float)
    depth[:, :10] = 0.34
    depth[:, 10:] = 0.42
    support = np.isfinite(depth)

    with np.testing.assert_raises_regex(ValueError, 'ambiguous'):
        select_target_depth_component(
            support, depth, center_u=9.5, center_v=9.5,
            minimum_points=20, ambiguity_margin=0.25)


def test_component_point_uses_only_the_depth_qualified_support():
    depth = np.full((5, 7), 0.80, dtype=float)
    depth[1:4, 1:4] = 0.40
    support = np.zeros_like(depth, dtype=bool)
    support[1:4, 1:4] = True
    intrinsic = [100.0, 0.0, 3.0, 0.0, 100.0, 2.0, 0.0, 0.0, 1.0]

    point = median_component_camera_point(depth, support, intrinsic)

    assert point == pytest.approx([-0.004, 0.0, 0.40])


class FakeMaskBridge:
    def __init__(self, mask):
        self.mask = mask

    def imgmsg_to_cv2(self, _message, desired_encoding=None):
        assert desired_encoding in ('mono8', 'passthrough')
        return self.mask


class RecordingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class SilentLogger:
    def info(self, _message):
        pass

    def warn(self, _message):
        pass


def image_stamp(seconds, frame='camera_color_optical_frame'):
    whole = int(seconds)
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(
                sec=whole,
                nanosec=int(round((seconds - whole) * 1.0e9))),
            frame_id=frame,
        )
    )


def mask_qualification_node(mask_message, mask_array=None):
    return SimpleNamespace(
        use_mask_depth=True,
        latest_mask_msg=mask_message,
        mask_max_age_s=0.20,
        mask_erode_px=0,
        bridge=FakeMaskBridge(
            np.ones((8, 10), dtype=np.uint8)
            if mask_array is None else mask_array),
        stamp_to_seconds=DepthTo3DNode.stamp_to_seconds,
        log_debug=lambda *_args, **_kwargs: None,
    )


def test_fresh_valid_mask_is_the_only_depth_measurement_support():
    node = mask_qualification_node(image_stamp(10.0))

    mask, rejection = DepthTo3DNode.mask_for_depth(
        node, image_stamp(10.1), 10, 8)

    assert rejection == ''
    assert mask.shape == (8, 10)
    assert np.count_nonzero(mask) == 80


def test_missing_mask_produces_no_target_measurement_support():
    node = mask_qualification_node(None)

    mask, rejection = DepthTo3DNode.mask_for_depth(
        node, image_stamp(10.0), 10, 8)

    assert mask is None
    assert rejection == 'mask_missing'


def test_stale_mask_produces_no_target_measurement_support():
    node = mask_qualification_node(image_stamp(9.0))

    mask, rejection = DepthTo3DNode.mask_for_depth(
        node, image_stamp(10.0), 10, 8)

    assert mask is None
    assert rejection == 'mask_stale'


def test_empty_mask_cannot_turn_clean_background_into_target_measurement():
    node = mask_qualification_node(
        image_stamp(10.0), np.zeros((8, 10), dtype=np.uint8))
    clean_background_depth = np.full((8, 10), 0.42, dtype=np.float32)
    assert float(np.std(clean_background_depth)) < 1.0e-6

    mask, rejection = DepthTo3DNode.mask_for_depth(
        node, image_stamp(10.0), 10, 8)

    assert mask is None
    assert rejection == 'mask_empty'


def test_wrong_frame_mask_produces_no_target_measurement_support():
    node = mask_qualification_node(image_stamp(10.0, frame='other_camera'))

    mask, rejection = DepthTo3DNode.mask_for_depth(
        node, image_stamp(10.0), 10, 8)

    assert mask is None
    assert rejection == 'mask_frame_mismatch'


def test_missing_mask_callback_never_uses_clean_background_roi():
    background = np.full((8, 10), 420, dtype=np.uint16)
    bridge = FakeMaskBridge(background)
    publisher = RecordingPublisher()
    node = SimpleNamespace(
        refresh_runtime_params=lambda: None,
        bridge=bridge,
        pub=publisher,
        latest_mask_msg=None,
        use_mask_depth=True,
        mask_max_age_s=0.20,
        mask_erode_px=0,
        depth_min=0.25,
        depth_max=1.20,
        previous_depth=None,
        pending_jump_depth=None,
        pending_jump_count=0,
        depth_image_to_meters=DepthTo3DNode.depth_image_to_meters,
        stamp_to_seconds=DepthTo3DNode.stamp_to_seconds,
        mask_for_depth=lambda depth, width, height:
            DepthTo3DNode.mask_for_depth(node, depth, width, height),
        depth_roi=lambda *_args: pytest.fail(
            'unmasked ROI fallback was called'),
        log_debug=lambda *_args, **_kwargs: None,
        get_logger=lambda: SilentLogger(),
    )
    detection = SimpleNamespace(
        valid=True, u=5.0, v=4.0, width=6.0, height=6.0,
        confidence=1.0)
    depth = Image()
    depth.header.stamp.sec = 10
    depth.header.frame_id = 'camera_color_optical_frame'
    depth.encoding = '16UC1'
    camera_info = SimpleNamespace(k=[100.0, 0.0, 5.0,
                                     0.0, 100.0, 4.0,
                                     0.0, 0.0, 1.0])

    DepthTo3DNode.synced_cb(node, detection, depth, camera_info)

    assert len(publisher.messages) == 1
    assert not publisher.messages[0].valid
    assert publisher.messages[0].depth_source == 'mask_missing'


@pytest.mark.parametrize('measurement', [
    [float('nan'), 0.0, 0.4],
    [0.0, float('inf'), 0.4],
    [0.0, 0.4],
    None,
])
def test_nonfinite_or_malformed_target_measurements_fail_closed(measurement):
    assert not finite_target_measurement(measurement)


def test_finite_target_measurement_is_accepted_for_filtering():
    assert finite_target_measurement([0.4, 0.0, 0.1])


def tracker_gate(last_measurement=None):
    return type('TrackerGate', (), {
        'min_confidence': 0.4,
        'use_camera_space_gates': False,
        'last_depth': None,
        'depth_gate_m': 0.15,
        'last_source_u': None,
        'last_source_v': None,
        'max_pixel_jump': 80.0,
        'last_measurement': last_measurement,
        'max_3d_jump': 0.10,
        'max_target_speed': 1.0,
        'last_area': None,
        'min_area_ratio': 0.5,
        'max_area_ratio': 2.0,
        'detection_area': TargetTrackerNode.detection_area,
    })()


def target_message():
    return type('TargetMessage', (), {
        'depth': 0.4,
        'source_u': 320.0,
        'source_v': 240.0,
        'detection_width': 60.0,
        'detection_height': 60.0,
    })()


def test_base_frame_target_gate_rejects_an_implausible_single_frame_jump():
    tracker = tracker_gate(last_measurement=[0.40, 0.0, 0.05])

    reason = TargetTrackerNode.gate_measurement(
        tracker, target_message(), [0.65, 0.0, 0.05],
        confidence=0.9, innovation_score=12.0,
        innovation_threshold=5.0)

    assert reason.startswith('innovation gate')


def test_base_frame_filter_smooths_qualified_measurement_jitter_and_resets():
    estimator = ConstantVelocityKalmanFilter(
        process_noise=0.01, measurement_noise=0.04,
        velocity_retention=0.0)
    first = estimator.step([0.40, 0.0, 0.05], 0.033)
    second = estimator.step([0.42, 0.0, 0.05], 0.033)

    assert first[0] == pytest.approx(0.40)
    assert 0.40 < second[0] < 0.42
    estimator.reset()
    assert not estimator.initialized
    restarted = estimator.step([0.50, 0.0, 0.05], 0.033)
    assert restarted[0] == pytest.approx(0.50)


def test_high_confidence_corrupted_measurement_fails_innovation_gate():
    estimator = ConstantVelocityKalmanFilter(
        process_noise=0.05, measurement_noise=0.03,
        velocity_retention=0.0)
    for _index in range(12):
        estimator.step([0.40, 0.0, 0.05], 0.033)
    estimator.predict(0.033)
    before = estimator.state.copy()

    _residual, _covariance, score = estimator.innovation(
        [0.49, 0.0, 0.05], measurement_noise=0.03)

    assert score > 5.0
    assert estimator.state == pytest.approx(before)


def test_short_prediction_only_outage_retains_state_and_grows_uncertainty():
    estimator = ConstantVelocityKalmanFilter(
        process_noise=0.05, measurement_noise=0.03,
        velocity_retention=0.0)
    estimator.step([0.40, 0.0, 0.05], 0.033)
    initial_uncertainty = estimator.maximum_position_stddev

    for _index in range(4):
        state = estimator.predict(0.10)

    assert state[:3] == pytest.approx([0.40, 0.0, 0.05])
    assert state[3:] == pytest.approx([0.0, 0.0, 0.0])
    assert estimator.maximum_position_stddev > initial_uncertainty
    assert prediction_only_is_valid(True, 5, 5, 0.40, 1.0)


def test_long_prediction_only_outage_expires_track():
    assert prediction_only_is_valid(True, 5, 5, 3.89, 5.0)
    assert not prediction_only_is_valid(True, 5, 5, 5.01, 5.0)


def test_fresh_explicit_camera_motion_forces_prediction_only_correction_gate():
    moving = SimpleNamespace(arm_moving=True, camera_settled=False)
    settling = SimpleNamespace(arm_moving=False, camera_settled=False)
    settled = SimpleNamespace(arm_moving=False, camera_settled=True)

    assert 'moving' in motion_measurement_rejection(moving, 0.1, 1.0)
    assert 'not settled' in motion_measurement_rejection(settling, 0.1, 1.0)
    assert motion_measurement_rejection(settled, 0.1, 1.0) is None


def test_missing_or_stale_motion_health_does_not_create_an_impossible_gate():
    stale_moving = SimpleNamespace(arm_moving=True, camera_settled=False)

    assert motion_measurement_rejection(None, float('inf'), 1.0) is None
    assert motion_measurement_rejection(stale_moving, 1.01, 1.0) is None


def test_stationary_gate_rejects_bad_measurement_after_bursty_gpu_gap():
    estimator = ConstantVelocityKalmanFilter(
        process_noise=0.01, measurement_noise=0.02,
        velocity_retention=0.0)
    for _index in range(12):
        estimator.step([0.44, -0.045, 0.015], 0.033)
    estimator.predict(3.89)
    before = estimator.state.copy()

    _residual, _covariance, score = estimator.innovation(
        [0.52, -0.045, 0.015], measurement_noise=0.02)

    assert score > 5.0
    assert estimator.state == pytest.approx(before)


def test_prediction_only_publication_retains_bounded_filtered_position():
    estimator = ConstantVelocityKalmanFilter(
        process_noise=0.05, measurement_noise=0.03,
        velocity_retention=0.0)
    estimator.step([0.40, 0.0, 0.05], 0.033)
    publisher = RecordingPublisher()
    status_publisher = RecordingPublisher()
    accepted_at = Time(nanoseconds=10_000_000_000)
    prediction_at = Time(nanoseconds=10_200_000_000)
    tracker = SimpleNamespace(
        filter=estimator,
        filter_time=accepted_at,
        last_seen_time=accepted_at,
        missed_frames=0,
        track_frames=5,
        min_track_frames=5,
        lost_timeout_s=1.0,
        prediction_horizon=0.15,
        last_measurement_confidence=0.9,
        prediction_only=False,
        last_stable=True,
        use_tf_transform=True,
        output_frame='base_link',
        status='LOCKED',
        pub=publisher,
        status_pub=status_publisher,
        predict_to=lambda stamp: TargetTrackerNode.predict_to(tracker, stamp),
        measurement_age=lambda now: TargetTrackerNode.measurement_age(
            tracker, now),
        publish_status=lambda status: TargetTrackerNode.publish_status(
            tracker, status),
        reset_tracking_state=lambda: pytest.fail(
            'short prediction-only period reset the track'),
        get_logger=lambda: SilentLogger(),
    )
    output = TrackedTarget()

    TargetTrackerNode.publish_prediction_only(
        tracker, output, prediction_at, prediction_at, 'mask_stale')

    assert len(publisher.messages) == 1
    assert publisher.messages[0].valid
    assert not publisher.messages[0].stable
    assert publisher.messages[0].position.x == pytest.approx(0.40)
    assert tracker.prediction_only
    assert tracker.status == 'LOW_CONFIDENCE'


def test_stationary_filter_reduces_noise_without_learning_false_velocity():
    estimator = ConstantVelocityKalmanFilter(
        process_noise=0.01, measurement_noise=0.015,
        velocity_retention=0.0)
    raw = []
    filtered = []
    for index in range(80):
        measurement = 0.40 + 0.012 * np.sin(index * 1.7)
        raw.append(measurement)
        filtered.append(estimator.step(
            [measurement, 0.0, 0.05], 0.033)[0])

    assert np.std(filtered[10:]) < np.std(raw[10:]) * 0.75
    assert estimator.state[3:] == pytest.approx([0.0, 0.0, 0.0])
    assert np.mean(filtered[-20:]) == pytest.approx(0.40, abs=0.004)
