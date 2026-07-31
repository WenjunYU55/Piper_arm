import numpy as np

from piper_mobile_manipulation.utils.mask_reprojection import (
    backproject_mask,
    project_points,
    rasterize_prompt,
    transform_points,
)


def test_depth_mask_round_trip_projection():
    k = np.asarray([[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]])
    mask = np.zeros((7, 9), dtype=np.uint8)
    mask[2:5, 3:6] = 255
    depth = np.ones(mask.shape, dtype=np.float32)

    points = backproject_mask(mask, depth, k, stride=1)
    uv = project_points(points, k, mask.shape)
    prompt = rasterize_prompt(uv, mask.shape, dilation_px=0)

    assert points.shape == (9, 3)
    assert np.count_nonzero(prompt) == 9


def test_camera_translation_changes_reprojected_pixels():
    k = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    points_base = np.asarray([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]])
    camera_from_base = np.eye(4)
    camera_from_base[0, 3] = -0.1

    moved = transform_points(points_base, camera_from_base)
    uv = project_points(moved, k, (80, 100))

    assert set(map(tuple, uv)) == {(40, 40), (50, 40), (40, 50)}
