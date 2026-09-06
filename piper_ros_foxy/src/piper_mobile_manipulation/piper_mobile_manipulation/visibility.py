"""ROS-free camera optical-axis geometry shared by all planner consumers."""

import math
import numpy as np

DEFAULT_VISIBILITY_ANGLE_DEG = 20.0
ANGLE_EPSILON_DEG = 1e-6
VISIBILITY_PARAMETER = "target_visibility_angle_tolerance_deg"


def visibility_tolerance(value):
    """Validate a strict forward-hemisphere cone, in degrees."""
    angle = float(value)
    if not math.isfinite(angle) or not 0.0 < angle < 90.0:
        raise ValueError(VISIBILITY_PARAMETER + " must be finite and within (0, 90)")
    return angle


def angle_degrees(forward, target_vector):
    """Measure angle from optical +Z; reject invalid or behind-camera rays."""
    a, b = np.asarray(forward, dtype=float), np.asarray(target_vector, dtype=float)
    if a.shape != (3,) or b.shape != (3,) or not np.isfinite([a, b]).all():
        raise ValueError("invalid target vector or optical axis")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not np.isfinite([na, nb]).all() or min(na, nb) <= 1e-9:
        raise ValueError("zero-length target vector or optical axis")
    cosine = float(np.dot(a / na, b / nb))
    if cosine <= 0.0:
        raise ValueError("target behind camera")
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def inside_cone(angle, tolerance):
    """Use inclusive boundaries with one microdegree floating-point allowance."""
    return (
        math.isfinite(float(angle))
        and float(angle) <= visibility_tolerance(tolerance) + ANGLE_EPSILON_DEG
    )


def validate_transform(value):
    """Validate T_base_camera: p_base = R_base_camera p_camera + t_base_camera."""
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
        or not math.isclose(np.linalg.det(matrix[:3, :3]), 1.0, abs_tol=1e-6)
    ):
        raise ValueError("invalid T_base_camera rigid transform")
    return matrix


def frozen_visibility_parameters(parameters):
    """Reject partial live updates; relaunch all consumers between missions."""
    from rcl_interfaces.msg import SetParametersResult

    for parameter in parameters:
        if parameter.name == VISIBILITY_PARAMETER:
            return SetParametersResult(
                successful=False,
                reason=(
                    "Visibility is frozen with planner/executor/recorder; restart the "
                    "scan stack between missions with a shared launch override."
                ),
            )
    return SetParametersResult(successful=True)
