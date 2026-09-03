from types import SimpleNamespace
import time
from threading import Condition, Thread
from pathlib import Path

import numpy as np
import pytest

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
    capture_model_seed_from_qualified,
    capture_view_selection_provenance,
    ScanCaptureNode,
)


def message(seconds):
    sec = int(seconds)
    nanosec = int(round((float(seconds) - sec) * 1e9))
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=sec, nanosec=nanosec),
            frame_id='camera_color_optical_frame'))


def test_capture_model_seed_uses_exact_persisted_mask_depth_and_transform():
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[30:70, 40:80] = 255
    support = np.zeros_like(mask)
    support[30:70, 40:80] = 255
    depth = np.zeros_like(mask, dtype=np.uint16)
    depth[30:70, 40:80] = 400
    transform = {
        'header': {
            'stamp': {'sec': 12, 'nanosec': 30_000_000},
            'frame_id': 'base_link',
        },
        'child_frame_id': 'camera_color_optical_frame',
        'matrix_4x4': np.eye(4).tolist(),
    }
    seed = capture_model_seed_from_qualified(
        mask,
        {
            'target_support_mask': support,
            'target_depth_mm': depth,
            'quality': {'confident_fraction': 0.95},
        },
        [100.0, 0.0, 60.0, 0.0, 100.0, 50.0, 0.0, 0.0, 1.0],
        message(12.0).header,
        transform,
    )

    assert seed['shape']['near_depth_m'] == 0.4
    assert seed['shape']['mask_pixel_count'] == 1600
    assert seed['base_from_camera']['matrix_4x4'] == np.eye(4).tolist()


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
            'capture_timeout_sec': 20.0,
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
        _native_bundle(10.0 + index / 10.0, now - 17.0 + index / 10.0)
        for index in range(19)]
    harness = BurstHarness(fresh)

    burst, reason = harness.collect_prospective_native_depth_burst(now - 19.0)

    assert burst is None
    assert '19/20 received' in reason
    assert 'synchronized rate' in reason


def test_capture_burst_can_complete_after_former_five_second_limit():
    now = time.monotonic()
    started_at = now - 6.0
    fresh = [
        _native_bundle(
            15.0 + index / 10.0,
            started_at + 0.1 + index * 0.25)
        for index in range(20)]
    harness = BurstHarness(fresh)

    burst, reason = harness.collect_prospective_native_depth_burst(started_at)

    assert reason == ''
    assert burst == fresh


def test_capture_recorder_reuses_outer_capture_deadline_configuration():
    package_root = Path(__file__).resolve().parents[1]
    launch_source = (
        package_root / 'launch' / 'supervised_viewpoint_execution.launch.py'
    ).read_text()
    capture_block = launch_source.split('capture = Node(', 1)[1].split(
        'return LaunchDescription', 1)[0]

    assert "self.declare_parameter('capture_timeout_sec', 20.0)" in (
        package_root / 'piper_mobile_manipulation' /
        'scan_capture_node.py').read_text()
    assert capture_block.index('execution_params,') < capture_block.index(
        'scan_params,') < capture_block.index('capture_params,')


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
    admission = {'scan_quality_raw': {'quality_label': 'GOOD'}}
    admissions = iter((admission, admission))
    mask = object()
    color_bundle = object()
    frames = [_native_bundle(30.0 + index / 10.0, 1.0 + index)
              for index in range(20)]
    harness = SimpleNamespace(
        frame_index=0,
        prepared_capture=None,
        capture_mode=lambda: 'service',
        get_parameter=lambda name: SimpleNamespace(value={
            'max_frames_per_scan': 30,
            'capture_timeout_sec': 20.0,
        }[name]),
        capture_transaction_deadline=lambda started_at: started_at + 18.0,
        wait_for_capture_prerequisite_evidence=lambda _deadline: (
            events.append('admit') or (True, '', next(admissions))),
        collect_prospective_native_depth_burst=lambda _started_at, deadline: (
            events.append('collect') or (frames, '')),
        wait_for_correlated_capture_pair=lambda depth_burst, deadline: (
            events.append(('pair', depth_burst))
            or (mask, color_bundle, '')),
        prepare_capture_until_ready=lambda depth_burst, selected_mask,
        selected_color, deadline: (
            events.append(
                ('prepare', depth_burst, selected_mask, selected_color))
            or ({'prepared': True}, '')),
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
        'admit', 'collect', ('pair', frames),
        ('prepare', frames, mask, color_bundle), 'admit', 'save']
    assert harness.prepared_capture['capture_admission_diagnostics'] is (
        admission)


def test_capture_transaction_waits_for_delayed_diagnostic_without_restarting():
    condition = Condition()
    calls = []
    responses = [
        (False, 'QUALITY_REJECTED: scan quality is stale', None),
        (True, '', {'scan_quality_age_sec': 0.01}),
    ]
    harness = SimpleNamespace(
        capture_evidence_condition=condition,
        capture_prerequisite_evidence=lambda: (
            calls.append('check') or responses.pop(0)),
    )

    def wake():
        time.sleep(0.02)
        with condition:
            condition.notify_all()

    thread = Thread(target=wake)
    thread.start()
    ready, reason, diagnostics = (
        ScanCaptureNode.wait_for_capture_prerequisite_evidence(
            harness, time.monotonic() + 0.5))
    thread.join(timeout=1.0)

    assert ready
    assert reason == ''
    assert diagnostics['scan_quality_age_sec'] == 0.01
    assert calls == ['check', 'check']


def test_capture_transaction_timeout_preserves_last_evidence_reason():
    harness = SimpleNamespace(
        capture_evidence_condition=None,
        capture_prerequisite_evidence=lambda: (
            False, 'QUALITY_REJECTED: scan quality is stale', None),
    )

    ready, reason, diagnostics = (
        ScanCaptureNode.wait_for_capture_prerequisite_evidence(
            harness, time.monotonic() - 0.01))

    assert not ready
    assert diagnostics is None
    assert reason == (
        'CAPTURE_EVIDENCE_TIMEOUT: '
        'QUALITY_REJECTED: scan quality is stale')


def test_capture_pair_skips_latest_unmatched_mask_without_weakening_identity():
    now = time.monotonic()
    matched_mask = message(10.02)
    unmatched_latest_mask = message(10.03)
    matched_color_bundle = (
        message(10.02), message(10.02), message(10.02), now)
    burst = [_native_bundle(10.00 + index * 0.005, now)
             for index in range(20)]
    parameters = {
        'synchronization_slop_sec': 0.08,
        'max_bundle_age_sec': 1.0,
    }
    harness = SimpleNamespace(
        mask_cache=[matched_mask, unmatched_latest_mask],
        color_bundle_cache=[matched_color_bundle],
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
    )

    selected_mask, selected_color, reason = (
        ScanCaptureNode.correlated_capture_pair(
            harness, burst, now_monotonic=now + 0.1))

    assert reason == ''
    assert selected_mask is matched_mask
    assert selected_color is matched_color_bundle


def test_capture_pair_requires_overlap_with_the_settled_depth_burst():
    now = time.monotonic()
    old_mask = message(9.0)
    old_color_bundle = (
        message(9.0), message(9.0), message(9.0), now)
    burst = [_native_bundle(10.00 + index * 0.005, now)
             for index in range(20)]
    parameters = {
        'synchronization_slop_sec': 0.08,
        'max_bundle_age_sec': 1.0,
    }
    harness = SimpleNamespace(
        mask_cache=[old_mask],
        color_bundle_cache=[old_color_bundle],
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
    )

    selected_mask, selected_color, reason = (
        ScanCaptureNode.correlated_capture_pair(
            harness, burst, now_monotonic=now + 0.1))

    assert selected_mask is None
    assert selected_color is None
    assert 'overlaps the settled native-depth burst' in reason


def test_capture_pair_waits_for_a_later_provable_mask_rgb_match():
    now = time.monotonic()
    condition = Condition()
    burst = [_native_bundle(10.00 + index * 0.005, now)
             for index in range(20)]
    matching_mask = message(10.04)
    matching_color = (
        message(10.04), message(10.04), message(10.04), now)
    parameters = {
        'synchronization_slop_sec': 0.08,
        'max_bundle_age_sec': 1.0,
    }
    harness = SimpleNamespace(
        capture_evidence_condition=condition,
        mask_cache=[message(10.03)],
        color_bundle_cache=[],
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        correlated_capture_pair=lambda received_burst: (
            ScanCaptureNode.correlated_capture_pair(
                harness, received_burst)),
    )

    def publish_pair():
        time.sleep(0.02)
        with condition:
            harness.mask_cache.append(matching_mask)
            harness.color_bundle_cache.append(matching_color)
            condition.notify_all()

    thread = Thread(target=publish_pair)
    thread.start()
    selected_mask, selected_color, reason = (
        ScanCaptureNode.wait_for_correlated_capture_pair(
            harness, burst, time.monotonic() + 0.5))
    thread.join(timeout=1.0)

    assert reason == ''
    assert selected_mask is matching_mask
    assert selected_color is matching_color


def test_capture_preparation_keeps_one_burst_mask_and_rgbd_pair():
    condition = Condition()
    burst = [object()]
    mask = object()
    color_bundle = object()
    calls = []
    responses = [
        (None, 'timestamped camera transform is unavailable: '
         'extrapolation into the future'),
        ({'prepared': True}, ''),
    ]
    harness = SimpleNamespace(
        capture_evidence_condition=condition,
        prepare_confidence_capture=lambda received_burst, mask_message,
        color_bundle: (
            calls.append((received_burst, mask_message, color_bundle))
            or responses.pop(0)),
    )

    def wake():
        time.sleep(0.02)
        with condition:
            condition.notify_all()

    thread = Thread(target=wake)
    thread.start()
    prepared, reason = ScanCaptureNode.prepare_capture_until_ready(
        harness, burst, mask, color_bundle, time.monotonic() + 0.5)
    thread.join(timeout=1.0)

    assert reason == ''
    assert prepared == {'prepared': True}
    assert calls == [
        (burst, mask, color_bundle), (burst, mask, color_bundle)]


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


def test_pending_first_capture_allows_only_fresh_classified_occlusion():
    quality = {
        'quality_label': 'GOOD', 'quality_score': 0.80,
        'target_valid': True,
    }
    allowed = ('CLEAR', 'PARTIALLY_OCCLUDED', 'HEAVILY_OCCLUDED')

    for state in allowed:
        assert capture_diagnostic_rejection(
            quality, 0.1, {'occlusion_state': state}, 0.1,
            1.0, 0.65, allowed) == ''
    assert 'UNKNOWN' in capture_diagnostic_rejection(
        quality, 0.1, {'occlusion_state': 'UNKNOWN'}, 0.1,
        1.0, 0.65, allowed)
    assert 'stale' in capture_diagnostic_rejection(
        quality, 0.1, {'occlusion_state': 'PARTIALLY_OCCLUDED'}, 1.1,
        1.0, 0.65, allowed)


def test_capture_node_allows_classified_occlusion_for_every_service_view():
    now = time.monotonic()
    parameters = {
        'dry_run': True,
        'enable_real_arm_motion': False,
        'require_mask': True,
        'require_valid_target': True,
        'require_good_quality_for_service': True,
        'require_clear_occlusion_for_service': True,
        'allow_classified_occlusion_for_service': True,
        'allow_classified_occlusion_for_first_capture': False,
        'diagnostic_timeout_sec': 1.0,
        'minimum_accepted_quality_score': 0.65,
    }
    node = SimpleNamespace(
        prepared_capture=None,
        param_bool=lambda name: bool(parameters[name]),
        capture_mode=lambda: 'service',
        latest_mask=np.ones((2, 2), dtype=np.uint8),
        latest_target=SimpleNamespace(valid=True),
        latest_execution_status=SimpleNamespace(
            execution_mode='MULTIVIEW_SCAN',
            state='CAPTURING_RGBD', current_view=1),
        frame_index=0,
        latest_scan_quality={
            'quality_label': 'GOOD', 'quality_score': 0.80,
            'target_valid': True,
        },
        latest_scan_quality_at=now,
        latest_occlusion_status={
            'occlusion_state': 'PARTIALLY_OCCLUDED'},
        latest_occlusion_status_at=now,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
    )

    ready, reason = ScanCaptureNode.capture_prerequisites_ready(node)
    assert ready, reason

    node.frame_index = 1
    node.latest_execution_status.current_view = 2
    ready, reason = ScanCaptureNode.capture_prerequisites_ready(node)
    assert ready, reason

    node.latest_occlusion_status = {'occlusion_state': 'UNKNOWN'}
    ready, reason = ScanCaptureNode.capture_prerequisites_ready(node)
    assert not ready
    assert 'UNKNOWN' in reason


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


def test_capture_metadata_keeps_commit_admitted_diagnostics_after_live_flip():
    admitted_quality = {
        'scan_quality_available': True,
        'scan_quality_score': 0.96,
        'scan_quality_label': 'GOOD',
        'mask_area_px': 7446,
        'valid_depth_ratio': 0.998,
        'depth_mean_m': 0.269,
        'depth_stddev_m': 0.008,
        'centredness_score': 0.93,
        'edge_margin_score': 1.0,
        'target_valid': True,
    }
    admitted_occlusion = dict(
        ScanCaptureNode.empty_occlusion_metadata(),
        occlusion_available=True,
        occlusion_state='CLEAR',
        occlusion_reason='target visible',
    )
    node = SimpleNamespace(
        scan_quality_metadata=lambda: pytest.fail(
            'live quality must not replace admitted evidence'),
        occlusion_metadata=lambda: pytest.fail(
            'live occlusion must not replace admitted evidence'),
    )
    prepared = {
        'capture_admission_diagnostics': {
            'scan_quality_metadata': admitted_quality,
            'occlusion_metadata': admitted_occlusion,
        },
    }

    quality, occlusion = ScanCaptureNode.capture_diagnostics_for_metadata(
        node, prepared)

    assert quality == admitted_quality
    assert occlusion['occlusion_state'] == 'CLEAR'
    # The persisted copies cannot mutate the frozen admission record.
    quality['scan_quality_label'] = 'INVALID'
    occlusion['occlusion_state'] = 'LOST'
    assert prepared['capture_admission_diagnostics'][
        'scan_quality_metadata']['scan_quality_label'] == 'GOOD'
    assert prepared['capture_admission_diagnostics'][
        'occlusion_metadata']['occlusion_state'] == 'CLEAR'


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
