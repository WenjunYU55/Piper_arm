from types import SimpleNamespace
import time
from threading import Condition, Thread

import numpy as np

from piper_mobile_manipulation.scan_capture import (
    DepthQualityRejected,
    adaptive_eroded_mask,
    capture_diagnostic_rejection,
    depth_connected_component,
    depth_millimetres,
    exact_stamped_item,
    nearest_stamped_item,
    normalize_l515_confidence,
    qualify_target_depth,
    rigid_transform_matrix,
    synchronized_bundle_rejection,
    temporal_confident_depth_median,
)
from piper_mobile_manipulation.scan_capture_node import (
    capture_view_selection_provenance,
    ScanCaptureNode,
)


def message(seconds):
    sec = int(seconds)
    nanosec = int(round((float(seconds) - sec) * 1e9))
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=sec, nanosec=nanosec)))


def test_synchronized_rgbd_bundle_requires_fresh_matching_stamps():
    assert synchronized_bundle_rejection(
        message(10.00), message(10.03), message(10.01),
        received_at=100.0, now_monotonic=100.2,
        maximum_age_sec=1.0, synchronization_slop_sec=0.08,
    ) == ''
    assert 'not synchronized' in synchronized_bundle_rejection(
        message(10.00), message(10.20), message(10.01),
        received_at=100.0, now_monotonic=100.2,
        maximum_age_sec=1.0, synchronization_slop_sec=0.08,
    )
    assert 'stale' in synchronized_bundle_rejection(
        message(10.00), message(10.03), message(10.01),
        received_at=100.0, now_monotonic=101.1,
        maximum_age_sec=1.0, synchronization_slop_sec=0.08,
    )


def test_depth_png_is_uint16_millimetres_for_l515_encodings():
    raw = np.asarray([[0, 301, 65535]], dtype=np.uint16)
    assert np.array_equal(depth_millimetres(raw, '16UC1'), raw)
    metres = np.asarray([[0.0, 0.301, np.nan]], dtype=np.float32)
    converted = depth_millimetres(metres, '32FC1')
    assert converted.dtype == np.uint16
    assert converted.tolist() == [[0, 301, 0]]


def test_twenty_frame_confident_depth_burst_rejects_flying_pixels():
    frames = []
    confidence = []
    for index in range(20):
        frame = np.asarray([
            [399 + (index % 3), 400],
            [400, 400],
        ], dtype=np.uint16)
        if index == 19:
            frame[0, 0] = 900
        if index >= 9:
            frame[1, 1] = 0
        frames.append(frame)
        confidence.append(np.full((2, 2), 10, dtype=np.uint8))

    depth, grades, report = temporal_confident_depth_median(
        frames, ['16UC1'] * 20, confidence,
        minimum_confidence=8, minimum_support_fraction=0.50)

    assert int(depth[0, 0]) == 400
    assert int(grades[0, 0]) == 10
    assert int(depth[1, 1]) == 0
    assert int(grades[1, 1]) == 0
    assert report['input_frames'] == 20
    assert report['minimum_support_frames'] == 10
    assert report['estimator'] == 'per_pixel_median'


def test_depth_burst_never_uses_low_confidence_samples_as_support():
    frames = [np.full((2, 2), 400, dtype=np.uint16) for _ in range(20)]
    grades = [np.full((2, 2), 10, dtype=np.uint8) for _ in range(20)]
    for index in range(11):
        grades[index][0, 0] = 2

    depth, confidence, report = temporal_confident_depth_median(
        frames, ['16UC1'] * 20, grades,
        minimum_confidence=8, minimum_support_fraction=0.50)

    assert int(depth[0, 0]) == 0
    assert int(confidence[0, 0]) == 0
    assert report['retained_pixels'] == 3


def _native_bundle(seconds, received_at):
    depth = message(seconds)
    confidence = message(seconds)
    camera_info = message(seconds)
    return depth, confidence, camera_info, received_at


class BurstHarness:
    valid_native_depth_bundles = ScanCaptureNode.valid_native_depth_bundles
    collect_prospective_native_depth_burst = (
        ScanCaptureNode.collect_prospective_native_depth_burst)

    def __init__(self, bundles):
        self.native_bundle_cache = list(bundles)
        self.native_bundle_condition = Condition()

    @staticmethod
    def get_parameter(name):
        return SimpleNamespace(value={
            'capture_burst_frames': 20,
            'max_bundle_age_sec': 1.0,
            'native_depth_confidence_slop_sec': 0.005,
        }[name])


def test_capture_burst_collects_twenty_new_frames_after_request():
    now = time.monotonic()
    old = [_native_bundle(9.0, now - 0.1)]
    fresh = [
        _native_bundle(10.0 + index / 10.0, now + 0.01 + index / 10.0)
        for index in range(20)]
    harness = BurstHarness(old + fresh)

    burst, reason = harness.collect_prospective_native_depth_burst(now)

    assert reason == ''
    assert len(burst) == 20
    assert burst[0] is fresh[0]
    assert burst[-1] is fresh[-1]


def test_capture_burst_timeout_reports_actual_new_frame_count():
    now = time.monotonic()
    fresh = [
        _native_bundle(10.0 + index / 10.0, now - 5.0 + index / 10.0)
        for index in range(19)]
    harness = BurstHarness(fresh)

    burst, reason = harness.collect_prospective_native_depth_burst(now - 6.0)

    assert burst is None
    assert '19/20 received' in reason


def test_capture_burst_waits_while_camera_callbacks_fill_it():
    started_at = time.monotonic()
    harness = BurstHarness([])
    result = {}

    thread = Thread(target=lambda: result.update(zip(
        ('burst', 'reason'),
        harness.collect_prospective_native_depth_burst(started_at))))
    thread.start()
    frames = [
        _native_bundle(
            20.0 + index / 10.0,
            started_at + 0.01 + index / 10.0)
        for index in range(20)]
    with harness.native_bundle_condition:
        harness.native_bundle_cache.extend(frames)
        harness.native_bundle_condition.notify_all()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert result['reason'] == ''
    assert result['burst'] == frames


def test_service_capture_collects_burst_before_preparing_and_saving():
    events = []
    frames = [_native_bundle(30.0 + index / 10.0, 1.0 + index)
              for index in range(20)]
    harness = SimpleNamespace(
        frame_index=0,
        capture_mode=lambda: 'service',
        get_parameter=lambda name: SimpleNamespace(value={
            'max_frames_per_scan': 30,
        }[name]),
        capture_prerequisites_ready=lambda: events.append(
            'prerequisites') or (True, ''),
        collect_prospective_native_depth_burst=lambda _started_at: (
            events.append('collect') or (frames, '')),
        capture_ready=lambda depth_burst: events.append(
            ('prepare', depth_burst)) or (True, ''),
        capture_frame=lambda _now: events.append('save') or (True, 'saved'),
        get_clock=lambda: SimpleNamespace(now=lambda: object()),
        note_skip=lambda reason: events.append(('skip', reason)),
        publish_status=lambda *args: events.append(('status', args)),
    )
    response = SimpleNamespace(success=False, message='')

    result = ScanCaptureNode.capture_view_cb(harness, object(), response)

    assert result.success
    assert result.message == 'saved'
    assert events == [
        'prerequisites', 'collect', ('prepare', frames), 'save']


def test_rigid_camera_transform_metadata_matrix_is_exact_and_finite():
    matrix = rigid_transform_matrix([1, 2, 3], [0, 0, 0, 1])
    assert matrix.tolist() == [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_exact_and_nearest_timestamp_cache_selection_are_distinct():
    items = [(message(10.00), 'first'), (message(10.03), 'second')]
    assert exact_stamped_item(items, message(10.03))[1] == 'second'
    assert exact_stamped_item(items, message(10.02)) is None
    assert nearest_stamped_item(items, message(10.02), 0.02)[1] == 'second'
    assert nearest_stamped_item(items, message(10.20), 0.02) is None


def test_mask_erosion_falls_back_before_destroying_a_small_target():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[4:8, 4:8] = 255
    result, report = adaptive_eroded_mask(mask, native_width=6)
    assert report['erosion_applied'] is False
    assert np.array_equal(result, mask > 0)


def test_depth_connected_component_does_not_cross_a_depth_step():
    candidate = np.ones((3, 4), dtype=bool)
    depth = np.asarray([
        [400, 401, 440, 441],
        [400, 402, 439, 441],
        [401, 402, 440, 442],
    ], dtype=np.uint16)
    selected = depth_connected_component(candidate, depth, (1, 0), 10.0)
    assert int(np.count_nonzero(selected)) == 6
    assert np.all(selected[:, :2])
    assert not np.any(selected[:, 2:])


def _qualified_fixture(confidence_grade=10):
    depth = np.full((6, 6), 400, dtype=np.uint16)
    confidence = np.full((6, 6), confidence_grade, dtype=np.uint8)
    mask = np.full((12, 12), 255, dtype=np.uint8)
    depth_k = [100.0, 0.0, 2.5, 0.0, 100.0, 2.5, 0.0, 0.0, 1.0]
    color_k = [200.0, 0.0, 5.5, 0.0, 200.0, 5.5, 0.0, 0.0, 1.0]
    return qualify_target_depth(
        depth, '16UC1', confidence, mask, depth_k, color_k,
        [0.0] * 5, 'plumb_bob', np.eye(4), minimum_confidence=8,
        minimum_points=10, minimum_confident_fraction=0.5,
        minimum_component_fraction=0.6)


def test_confidence_target_uses_native_geometry_and_hardware_grades():
    result = _qualified_fixture()
    assert result['target']['valid'] is True
    assert abs(result['target']['depth'] - 0.4) < 1e-9
    assert result['quality']['confidence_threshold'] == 8
    assert result['quality']['confident_fraction'] == 1.0
    assert result['quality']['primary_component_fraction'] == 1.0
    assert np.count_nonzero(result['target_support_mask']) >= 10


def test_l515_left_justified_raw8_confidence_is_normalized_to_grades():
    raw = np.asarray([[0, 16, 128, 240]], dtype=np.uint8)
    grades, representation = normalize_l515_confidence(raw)
    assert grades.tolist() == [[0, 1, 8, 15]]
    assert representation == 'left_justified_4bit_raw8'

    result = _qualified_fixture(confidence_grade=160)
    assert result['quality']['confidence_input_representation'] == \
        'left_justified_4bit_raw8'
    assert result['quality']['confidence_grade_histogram_0_to_15'][10] == 36
    assert int(result['confidence'].max()) == 10


def test_l515_confidence_rejects_arbitrary_or_mixed_eight_bit_values():
    with np.testing.assert_raises(ValueError):
        normalize_l515_confidence(
            np.asarray([[0, 15, 16, 17]], dtype=np.uint8))


def test_complete_low_confidence_observation_is_a_visual_depth_rejection():
    with np.testing.assert_raises(DepthQualityRejected):
        _qualified_fixture(confidence_grade=4)


def test_capture_requires_fresh_good_and_clear_visual_evidence():
    quality = {
        'quality_label': 'GOOD', 'quality_score': 0.80,
        'target_valid': True,
    }
    clear = {'occlusion_state': 'CLEAR'}
    assert capture_diagnostic_rejection(
        quality, 0.1, clear, 0.1, 1.0, 0.65) == ''
    assert capture_diagnostic_rejection(
        dict(quality, quality_label='POOR'), 0.1,
        clear, 0.1, 1.0, 0.65).startswith('QUALITY_REJECTED')
    assert capture_diagnostic_rejection(
        quality, 0.1, {'occlusion_state': 'PARTIALLY_OCCLUDED'},
        0.1, 1.0, 0.65).startswith('OCCLUSION_REJECTED')
    assert 'stale' in capture_diagnostic_rejection(
        quality, 1.1, clear, 0.1, 1.0, 0.65)


def test_capture_diagnostic_fails_closed_on_malformed_or_nonfinite_values():
    clear = {'occlusion_state': 'CLEAR'}
    assert capture_diagnostic_rejection({
        'quality_label': 'GOOD', 'quality_score': 'not-a-number',
        'target_valid': True,
    }, 0.1, clear, 0.1).startswith('QUALITY_REJECTED:')
    assert capture_diagnostic_rejection({
        'quality_label': 'GOOD', 'quality_score': float('nan'),
        'target_valid': True,
    }, 0.1, clear, 0.1).startswith('QUALITY_REJECTED:')
    assert capture_diagnostic_rejection({
        'quality_label': 'GOOD', 'quality_score': 0.9,
        'target_valid': 'true',
    }, 0.1, clear, 0.1).startswith('QUALITY_REJECTED:')
    assert capture_diagnostic_rejection({
        'quality_label': 'GOOD', 'quality_score': 0.9,
        'target_valid': True,
    }, -0.1, clear, 0.1).startswith('QUALITY_REJECTED:')


def test_execution_metadata_preserves_physical_capture_provenance():
    status = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=12, nanosec=34),
            frame_id='base_link'),
        plan_id='physical-plan',
        execution_mode='MULTIVIEW_SCAN',
        state='CAPTURING_RGBD',
        dry_run=False,
        real_arm_motion=True,
        approval_required=False,
        current_view=1,
        total_views=1,
        commanded_speed_percent=5.0,
    )

    metadata = ScanCaptureNode.execution_status_metadata(status)

    assert metadata['dry_run'] is False
    assert metadata['real_arm_motion'] is True
    assert metadata['approval_required'] is False


def test_capture_resolves_exact_nbv_policy_rank_and_gain_from_plan_id():
    provenance = {
        'schema_version': 1,
        'plan_id': 'physical-plan',
        'request_id': 'request-2',
        'request_sha256': 'a' * 64,
        'candidate_diagnostics': {
            'candidate_attempts': 3,
            'candidate_failures': [
                {'id': 2, 'stage': 'AIM_VARIANTS_EXHAUSTED'}],
        },
        'selected_viewpoints': [{
            'id': 41,
            'view_selection_policy': 'voxel_nbv',
            'view_selection_generation': 2,
            'view_selection_session_id': 'scan-a',
            'nbv_rank': 7,
            'nbv_predicted_unknown_pixels': 318,
            'nbv_novel_surface_pixels': 22,
            'nbv_marginal_information_pixels': 340,
            'nbv_marginal_information_fraction': 0.85,
        }],
    }

    result = capture_view_selection_provenance(
        provenance, {'plan_id': 'physical-plan', 'current_view': 1})

    assert result == {
        'available': True,
        'plan_id': 'physical-plan',
        'request_id': 'request-2',
        'request_sha256': 'a' * 64,
        **provenance['selected_viewpoints'][0],
        'candidate_diagnostics': provenance['candidate_diagnostics'],
    }


def test_capture_never_attaches_provenance_from_a_different_plan():
    result = capture_view_selection_provenance(
        {'plan_id': 'old-plan', 'selected_viewpoints': [{'id': 1}]},
        {'plan_id': 'new-plan', 'current_view': 1})

    assert result['available'] is False
    assert result['reason'] == 'plan provenance ID mismatch'
