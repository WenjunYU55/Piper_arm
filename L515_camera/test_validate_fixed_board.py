from collections import deque

import numpy as np

from capture_hand_eye_sample import (
    DEFAULT_MARKER_LENGTH_M as CAPTURE_MARKER_LENGTH_M,
    DEFAULT_SQUARE_LENGTH_M as CAPTURE_SQUARE_LENGTH_M,
)
from validate_fixed_board import (
    DEFAULT_MARKER_LENGTH_M as VALIDATION_MARKER_LENGTH_M,
    DEFAULT_SQUARE_LENGTH_M as VALIDATION_SQUARE_LENGTH_M,
    joint_positions_are_stable,
    result_dict,
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
