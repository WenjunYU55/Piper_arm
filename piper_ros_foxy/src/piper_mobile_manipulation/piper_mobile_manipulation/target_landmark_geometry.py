"""Compatibility facade for target landmark geometry."""

from piper_mobile_manipulation.perception.landmark_geometry import (
    direction_angle_degrees,
    maximum_pairwise_distance,
    project_camera_point,
)

__all__ = [
    'direction_angle_degrees', 'maximum_pairwise_distance',
    'project_camera_point',
]
