from types import SimpleNamespace

import numpy as np

from piper_mobile_manipulation.scan_capture import (
    capture_diagnostic_rejection,
    depth_millimetres,
    rigid_transform_matrix,
    synchronized_bundle_rejection,
)
from piper_mobile_manipulation.scan_capture_node import ScanCaptureNode


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


def test_rigid_camera_transform_metadata_matrix_is_exact_and_finite():
    matrix = rigid_transform_matrix([1, 2, 3], [0, 0, 0, 1])
    assert matrix.tolist() == [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


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
