"""Geometry-based checks of offline rectification and planar depth diagnostics."""

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/bundlesdf'))
from distortion import plane_residuals, rectification_maps  # noqa: E402

K = np.array([[598.87, 0., 330.25], [0., 599.38, 239.76], [0., 0., 1.]])
D = np.array([.16904977, -.51425225, -.00214692, .00131909, .47541139])


def test_rectification_map_matches_independent_brown_conrady_projection():
    # Independently project rectified pixel rays with the documented polynomial.
    mx, my = rectification_maps(K, D, 640, 480)
    v, u = np.mgrid[10:470:31, 10:630:31]
    x, y = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
    r2 = x*x + y*y
    radial = 1 + D[0]*r2 + D[1]*r2**2 + D[4]*r2**3
    xd = x*radial + 2*D[2]*x*y + D[3]*(r2 + 2*x*x)
    yd = y*radial + D[2]*(r2 + 2*y*y) + 2*D[3]*x*y
    assert np.max(np.abs(mx[v, u] - (K[0, 0]*xd + K[0, 2]))) < 0.00004
    assert np.max(np.abs(my[v, u] - (K[1, 1]*yd + K[1, 2]))) < 0.00004


def test_zero_distortion_is_identity():
    mx, my = rectification_maps(K, np.zeros(5), 640, 480)
    y, x = np.indices(mx.shape)
    assert np.allclose(mx, x, atol=1e-5)
    assert np.allclose(my, y, atol=1e-5)


def test_tilted_plane_uses_axial_depth_and_camera_distortion():
    y, x = np.indices((480, 640))
    pixels = np.stack((x, y), axis=-1).astype(float).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(pixels, K, D).reshape(480, 640, 2)
    depth = .6 / (1 + .4*rays[:, :, 0] + .2*rays[:, :, 1])
    support = np.zeros(depth.shape, bool)
    support[20:460, 20:620] = True
    correct = plane_residuals(depth, K, support, D)
    wrong = plane_residuals(depth, K, support)
    assert correct['rms_mm'] < 1e-8
    assert wrong['rms_mm'] > .1
    # A flat frontoparallel plane has constant Z even away from the optical axis.
    flat = plane_residuals(np.full(depth.shape, .6), K, support, D)
    assert flat['rms_mm'] < 1e-8


def test_bowed_depth_is_reported_not_flattened():
    y, x = np.indices((480, 640))
    depth = .6 + .03*((x-320)/320.)**2
    original = depth.copy()
    result = plane_residuals(depth, K, np.ones(depth.shape, bool))
    assert result['rms_mm'] > 5
    assert np.array_equal(depth, original)


def test_invalid_or_tiny_patch_rejected():
    with pytest.raises(ValueError, match='100 valid'):
        plane_residuals(np.zeros((20, 20)), K, np.ones((20, 20), bool))
    with pytest.raises(ValueError, match='matching'):
        plane_residuals(np.ones((20, 20)), K, np.ones((10, 10), bool))
