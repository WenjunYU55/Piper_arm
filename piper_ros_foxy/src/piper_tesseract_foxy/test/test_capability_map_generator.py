import numpy as np
import pytest

from piper_tesseract_foxy.capability_map_generator import (
    merge_capability_records,
    sorted_capability_records,
    tool_minimum_z,
)


def test_duplicate_capability_bin_keeps_best_floor_clearance():
    occupied = {}
    merge_capability_records(
        occupied,
        np.asarray([9, 3, 9], dtype=np.uint64),
        np.asarray([0.10, -0.20, 0.14], dtype=np.float32),
    )

    keys, floors = sorted_capability_records(occupied)

    assert keys.tolist() == [3, 9]
    assert floors.tolist() == pytest.approx([-0.20, 0.14])


def test_tool_minimum_z_uses_all_eight_transformed_corners():
    corners = np.asarray([
        [x, y, z, 1.0]
        for x in (-0.1, 0.1)
        for y in (-0.2, 0.2)
        for z in (-0.3, 0.3)
    ])
    transform = np.eye(4)
    transform[2, 3] = 0.5

    assert tool_minimum_z(transform, corners) == pytest.approx(0.2)


def test_generator_rejects_malformed_tool_geometry():
    with pytest.raises(ValueError, match='floor geometry'):
        tool_minimum_z(np.eye(3), np.ones((8, 4)))
