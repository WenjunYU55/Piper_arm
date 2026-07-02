import numpy as np
import pytest

from piper_mobile_manipulation.target_landmark_geometry import (
    direction_angle_degrees,
    maximum_pairwise_distance,
    project_camera_point,
)


def test_stationary_measurement_spread():
    points = [[0.5, 0.0, 0.1], [0.503, 0.004, 0.1], [0.5, 0.0, 0.1]]
    assert maximum_pairwise_distance(points) == pytest.approx(0.005)


def test_viewpoint_angle():
    assert direction_angle_degrees([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)


def test_project_landmark():
    uv = project_camera_point([0.1, -0.05, 0.5], [500, 0, 320, 0, 500, 240, 0, 0, 1])
    np.testing.assert_allclose(uv, [420, 190])


def test_reject_landmark_behind_camera():
    with pytest.raises(ValueError, match='landmark_behind_camera'):
        project_camera_point([0, 0, -1], [500, 0, 320, 0, 500, 240, 0, 0, 1])
