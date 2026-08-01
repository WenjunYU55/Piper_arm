from types import SimpleNamespace

import numpy as np

from piper_mobile_manipulation.scan_capture import (
    depth_millimetres,
    rigid_transform_matrix,
    synchronized_bundle_rejection,
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


def test_rigid_camera_transform_metadata_matrix_is_exact_and_finite():
    matrix = rigid_transform_matrix([1, 2, 3], [0, 0, 0, 1])
    assert matrix.tolist() == [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
