from collections import deque
import inspect

import numpy as np

from capture_hand_eye_sample import (
    DEFAULT_MARKER_LENGTH_M as CAPTURE_MARKER_LENGTH_M,
    DEFAULT_SQUARE_LENGTH_M as CAPTURE_SQUARE_LENGTH_M,
    nearest_stamped_joint,
)
from validate_fixed_board import (
    DEFAULT_MARKER_LENGTH_M as VALIDATION_MARKER_LENGTH_M,
    DEFAULT_SQUARE_LENGTH_M as VALIDATION_SQUARE_LENGTH_M,
    joint_positions_are_stable,
    result_dict,
    FixedBoardValidator,
)


def history(samples):
    return deque((stamp, np.asarray(positions, dtype=float))
                 for stamp, positions in samples)


def test_measured_board_dimensions_are_shared_by_capture_and_validation():
    assert CAPTURE_SQUARE_LENGTH_M == VALIDATION_SQUARE_LENGTH_M == 0.017
    assert CAPTURE_MARKER_LENGTH_M == VALIDATION_MARKER_LENGTH_M == 0.012


def test_validation_report_persists_exact_board_provenance():
    pose = np.eye(4)
    metadata = {
        "board": {
            "squares_x": 5,
            "squares_y": 5,
            "square_length_m": VALIDATION_SQUARE_LENGTH_M,
            "marker_length_m": VALIDATION_MARKER_LENGTH_M,
            "dictionary": "DICT_4X4_50",
        }
    }
    report = result_dict([pose], [0.1], 15.0, 1.5, metadata)
    assert report["validation"]["board"] == metadata["board"]


def test_stable_positions_pass_complete_window():
    samples = history([
        (9.20, [0.0] * 6),
        (9.60, [0.0004] * 6),
        (10.00, [0.0008] * 6),
    ])
    assert joint_positions_are_stable(samples, 10.0, 0.75, 0.001)


def test_incomplete_window_fails_closed():
    samples = history([
        (9.60, [0.0] * 6),
        (10.00, [0.0] * 6),
    ])
    assert not joint_positions_are_stable(samples, 10.0, 0.75, 0.001)


def test_disabled_joint_sag_is_rejected_even_if_endpoints_are_finite():
    samples = history([
        (9.20, [0.0] * 6),
        (9.60, [0.0, 0.0, 0.0, 0.0, 0.012, 0.0]),
        (10.00, [0.0, 0.0, 0.0, 0.0, 0.026, 0.0]),
    ])
    assert not joint_positions_are_stable(samples, 10.0, 0.75, 0.001)


def test_nonfinite_feedback_fails_closed():
    samples = history([
        (9.20, [0.0] * 6),
        (10.00, [0.0, 0.0, 0.0, 0.0, float("nan"), 0.0]),
    ])
    assert not joint_positions_are_stable(samples, 10.0, 0.75, 0.001)


def test_board_pose_uses_the_exact_image_timestamp_for_tf():
    source = inspect.getsource(FixedBoardValidator.image_cb)
    assert 'Time.from_msg(message.header.stamp)' in source
    assert 'rclpy.time.Time()' not in source


def test_capture_selects_joint_feedback_nearest_the_image_timestamp():
    first, second = object(), object()
    samples = deque([
        (10.000, 20.0, first),
        (10.035, 20.1, second),
    ])
    selected, delta = nearest_stamped_joint(samples, 10.031)
    assert selected is second
    assert abs(delta - 0.004) < 1e-12


def test_capture_rejects_uncorrelated_or_invalid_joint_timestamps():
    samples = deque([(10.0, 20.0, object())])
    selected, delta = nearest_stamped_joint(samples, 10.2)
    assert selected is None
    assert abs(delta - 0.2) < 1e-12
    assert nearest_stamped_joint(samples, 0.0) == (None, None)


def test_capture_and_validation_use_intrinsics_for_charuco_interpolation():
    capture_source = inspect.getsource(
        __import__('capture_hand_eye_sample').HandEyeCapture.validate_and_save)
    validation_source = inspect.getsource(FixedBoardValidator.image_cb)
    for source in (capture_source, validation_source):
        assert 'cameraMatrix=' in source
        assert 'distCoeffs=' in source


def test_validation_report_can_persist_within_pose_noise():
    pose = np.eye(4)
    noise = {
        'frame_count': 30,
        'translation_rms_mm': 0.4,
        'translation_max_mm': 0.8,
        'rotation_rms_deg': 0.1,
        'rotation_max_deg': 0.2,
        'reprojection_mean_px': 0.15,
        'reprojection_max_px': 0.2,
    }
    report = result_dict(
        [pose], [0.15], 15.0, 1.5, {'board': {}}, [noise])
    assert report['measurements'][0]['within_pose_noise'] == noise
