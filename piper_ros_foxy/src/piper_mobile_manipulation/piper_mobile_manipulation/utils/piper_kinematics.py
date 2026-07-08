"""PiPER mode-0 kinematics and bounded numerical IK for dry-run validation."""

import math

import numpy as np
from scipy.optimize import least_squares


_A = [0.0, 0.0, 0.28503, -0.02198, 0.0, 0.0]
_ALPHA = [0.0, -math.pi / 2.0, 0.0, math.pi / 2.0, -math.pi / 2.0, math.pi / 2.0]
_THETA = [0.0, -math.radians(174.22), -math.radians(100.78), 0.0, 0.0, 0.0]
_D = [0.123, 0.0, 0.0, 0.25075, 0.0, 0.091]


def _link(alpha, a, theta, d):
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.asarray([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -sa * d],
        [st * sa, ct * sa, ca, ca * d],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)


def forward_matrix(joints):
    transform = np.eye(4, dtype=float)
    for index in range(6):
        transform = transform.dot(_link(
            _ALPHA[index], _A[index], float(joints[index]) + _THETA[index], _D[index]))
    return transform


def quaternion_matrix(quaternion):
    x, y, z, w = (float(value) for value in quaternion)
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-12:
        return np.eye(3)
    scale = 2.0 / norm
    return np.asarray([
        [1 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)],
        [scale * (x * y + z * w), 1 - scale * (x * x + z * z), scale * (y * z - x * w)],
        [scale * (x * z - y * w), scale * (y * z + x * w), 1 - scale * (x * x + y * y)],
    ])


def pose_matrix(pose):
    result = np.eye(4, dtype=float)
    result[:3, :3] = quaternion_matrix((
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))
    result[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return result


def rotation_vector(matrix):
    cosine = np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0)
    angle = math.acos(float(cosine))
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.asarray([
        matrix[2, 1] - matrix[1, 2],
        matrix[0, 2] - matrix[2, 0],
        matrix[1, 0] - matrix[0, 1],
    ]) / (2.0 * math.sin(angle))
    return axis * angle


def pose_error(current, desired):
    translation = desired[:3, 3] - current[:3, 3]
    rotation = rotation_vector(desired[:3, :3].dot(current[:3, :3].T))
    return np.concatenate((translation, rotation))


def solve_camera_pose(desired_base_camera, link6_from_camera, seed, lower, upper,
                      max_iterations=100, damping=0.04, position_tolerance=0.005,
                      rotation_tolerance=math.radians(3.0)):
    desired_base_link6 = desired_base_camera.dot(np.linalg.inv(link6_from_camera))
    return solve_link6_pose(
        desired_base_link6, seed, lower, upper, max_iterations, damping,
        position_tolerance, rotation_tolerance)


def solve_link6_pose(desired_base_link6, seed, lower, upper,
                     max_iterations=100, damping=0.04, position_tolerance=0.005,
                     rotation_tolerance=math.radians(3.0)):
    joints = np.clip(np.asarray(seed, dtype=float), lower, upper)
    epsilon = 1e-4
    for iteration in range(int(max_iterations)):
        current = forward_matrix(joints)
        error = pose_error(current, desired_base_link6)
        if (np.linalg.norm(error[:3]) <= position_tolerance and
                np.linalg.norm(error[3:]) <= rotation_tolerance):
            return joints, True, {
                'iterations': iteration,
                'position_error_m': float(np.linalg.norm(error[:3])),
                'rotation_error_rad': float(np.linalg.norm(error[3:])),
            }
        jacobian = np.zeros((6, 6), dtype=float)
        for column in range(6):
            perturbed = joints.copy()
            perturbed[column] += epsilon
            jacobian[:, column] = (
                pose_error(current, forward_matrix(perturbed)) / epsilon)
        system = jacobian.dot(jacobian.T) + float(damping) ** 2 * np.eye(6)
        step = jacobian.T.dot(np.linalg.solve(system, error))
        joints = np.clip(joints + np.clip(step, -0.15, 0.15), lower, upper)
    # Refine with a bounded trust-region solver. This is slower than the DLS
    # loop but substantially more robust for large wrist-orientation changes.
    refined = least_squares(
        lambda values: pose_error(forward_matrix(values), desired_base_link6),
        joints, bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
        max_nfev=300, xtol=1e-8, ftol=1e-8, gtol=1e-8,
    )
    joints = np.asarray(refined.x, dtype=float)
    final_error = pose_error(forward_matrix(joints), desired_base_link6)
    converged = (
        np.linalg.norm(final_error[:3]) <= position_tolerance
        and np.linalg.norm(final_error[3:]) <= rotation_tolerance)
    return joints, bool(converged), {
        'iterations': int(max_iterations),
        'least_squares_evaluations': int(refined.nfev),
        'position_error_m': float(np.linalg.norm(final_error[:3])),
        'rotation_error_rad': float(np.linalg.norm(final_error[3:])),
    }
