import math

import numpy as np

from piper_mobile_manipulation.utils.icp import register, transform_points


def test_icp_recovers_rigid_transform():
    rng = np.random.RandomState(7)
    target = rng.uniform(-0.04, 0.04, size=(600, 3))
    angle = math.radians(5.0)
    rotation = np.asarray([
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    source = target.dot(rotation.T) + np.asarray([0.006, -0.004, 0.003])
    transform, metrics = register(
        source, target, max_correspondence_m=0.02, voxel_size_m=0.002)
    aligned = transform_points(source, transform)
    assert metrics['fitness'] > 0.9
    assert metrics['rmse_m'] < 0.004
    assert np.median(np.linalg.norm(aligned - target, axis=1)) < 0.004


def test_icp_rejects_too_few_points():
    _transform, metrics = register(np.zeros((5, 3)), np.zeros((5, 3)))
    assert not metrics['accepted']
