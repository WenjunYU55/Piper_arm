import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from solve_hand_eye import (
    HAND_EYE_METHODS,
    compare_solvers,
    inverse,
    pose_geometry,
    solve,
    transform,
)


def synthetic_samples():
    link6_from_camera = transform(
        Rotation.from_euler('xyz', [0.12, -0.08, 0.2]).as_matrix(),
        [0.08, -0.01, 0.04],
    )
    base_from_target = transform(
        Rotation.from_euler('xyz', [0.02, 0.05, -0.1]).as_matrix(),
        [0.42, 0.03, 0.12],
    )
    poses = [
        ([0.0, 0.0, 0.0], [0.20, 0.00, 0.30]),
        ([0.4, 0.1, 0.0], [0.22, 0.02, 0.29]),
        ([-0.3, 0.35, 0.1], [0.18, -0.03, 0.31]),
        ([0.2, -0.25, 0.45], [0.21, 0.04, 0.28]),
        ([-0.35, -0.15, -0.3], [0.19, -0.02, 0.32]),
        ([0.1, 0.5, -0.2], [0.23, 0.01, 0.30]),
    ]
    samples = []
    for index, (rpy, xyz) in enumerate(poses):
        base_from_link6 = transform(Rotation.from_euler('xyz', rpy).as_matrix(), xyz)
        camera_from_target = (
            inverse(link6_from_camera)
            @ inverse(base_from_link6)
            @ base_from_target
        )
        samples.append({
            'name': 'sample_%02d' % index,
            'base_from_link6': base_from_link6,
            'camera_from_target': camera_from_target,
            'reprojection_rms_px': 0.1,
        })
    return samples, link6_from_camera


def test_park_solver_recovers_exact_synthetic_transform():
    samples, expected = synthetic_samples()
    actual = solve(samples, cv2.CALIB_HAND_EYE_PARK)
    assert np.allclose(actual, expected, atol=1e-8)


def test_pose_geometry_reports_multi_axis_rotation_diversity():
    samples, _expected = synthetic_samples()
    report = pose_geometry(samples)
    assert report['pair_count'] == 15
    assert report['relative_rotation_deg']['pairs_at_least_15_deg'] > 5
    assert len(report['rotation_axis_scatter_eigenvalues']) == 3
    assert report['rotation_axis_scatter_eigenvalues'][0] > 0.01


def test_solver_comparison_is_diagnostic_and_reports_each_method():
    samples, _expected = synthetic_samples()
    report = compare_solvers(samples, samples[-2:])
    assert set(report) == set(HAND_EYE_METHODS)
    assert report['PARK']['status'] == 'completed'
    assert report['PARK']['held_out_validation']['translation_max_mm'] < 1e-5
