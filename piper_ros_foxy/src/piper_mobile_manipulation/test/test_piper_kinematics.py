import numpy as np

from piper_mobile_manipulation.utils.piper_kinematics import forward_matrix, solve_link6_pose


def test_numerical_ik_round_trip():
    desired_joints = np.asarray([0.2, 0.5, -1.0, 0.25, 0.4, -0.3])
    desired_pose = forward_matrix(desired_joints)
    lower = np.asarray([-2.7, -0.05, -3.03, -1.8, -1.3, -4.23])
    upper = np.asarray([2.7, 3.38, 0.04, 1.8, 1.34, 4.23])
    solution, converged, details = solve_link6_pose(
        desired_pose, desired_joints + 0.02, lower, upper)
    assert converged, details
    assert np.linalg.norm(forward_matrix(solution)[:3, 3] - desired_pose[:3, 3]) < 0.005


def test_ik_respects_bounds_for_unreachable_pose():
    desired = np.eye(4)
    desired[:3, 3] = [2.0, 2.0, 2.0]
    lower = np.full(6, -0.1)
    upper = np.full(6, 0.1)
    solution, converged, _details = solve_link6_pose(
        desired, np.zeros(6), lower, upper, max_iterations=5)
    assert not converged
    assert np.all(solution >= lower) and np.all(solution <= upper)
