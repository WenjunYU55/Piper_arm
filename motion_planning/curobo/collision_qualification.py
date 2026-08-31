"""ROS- and CUDA-free qualification helpers for cuRobo collision spheres.

The helpers intentionally mirror only URDF forward kinematics and sphere-pair
distance.  They make reviewed safe/colliding poses auditable in ordinary unit
tests without importing cuRobo or changing the runtime collision policy.
"""

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class SphereOverlap:
    """One non-ignored sphere overlap in a robot configuration."""

    first_link: str
    first_sphere: int
    second_link: str
    second_sphere: int
    penetration_m: float


def _vector(text, default):
    if text is None:
        return np.asarray(default, dtype=float)
    result = np.asarray([float(value) for value in text.split()], dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError('malformed URDF three-vector')
    return result


def _rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _transform(translation=None, rotation=None):
    result = np.eye(4, dtype=float)
    if translation is not None:
        result[:3, 3] = translation
    if rotation is not None:
        result[:3, :3] = rotation
    return result


def _axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError('URDF joint axis must be non-zero')
    x, y, z = axis / norm
    cosine, sine = math.cos(angle), math.sin(angle)
    one_minus = 1.0 - cosine
    return np.asarray([
        [cosine + x * x * one_minus,
         x * y * one_minus - z * sine,
         x * z * one_minus + y * sine],
        [y * x * one_minus + z * sine,
         cosine + y * y * one_minus,
         y * z * one_minus - x * sine],
        [z * x * one_minus - y * sine,
         z * y * one_minus + x * sine,
         cosine + z * z * one_minus],
    ])


def urdf_link_transforms(urdf_path, joint_positions):
    """Return world transforms for every reachable URDF link."""
    root = ET.parse(Path(urdf_path)).getroot()
    joints_by_parent = {}
    child_links = set()
    for joint in root.findall('joint'):
        parent = joint.find('parent')
        child = joint.find('child')
        if parent is None or child is None:
            raise ValueError('URDF joint is missing parent or child')
        parent_name = str(parent.get('link', ''))
        child_name = str(child.get('link', ''))
        if not parent_name or not child_name:
            raise ValueError('URDF joint has an empty parent or child')
        joints_by_parent.setdefault(parent_name, []).append(joint)
        child_links.add(child_name)
    link_names = {
        str(link.get('name', '')) for link in root.findall('link')
        if link.get('name')}
    roots = sorted(link_names - child_links)
    if len(roots) != 1:
        raise ValueError('URDF must have exactly one root link')

    transforms = {roots[0]: np.eye(4, dtype=float)}
    pending = [roots[0]]
    while pending:
        parent_name = pending.pop()
        for joint in joints_by_parent.get(parent_name, []):
            child_name = str(joint.find('child').get('link'))
            origin = joint.find('origin')
            xyz = _vector(
                None if origin is None else origin.get('xyz'), [0.0] * 3)
            rpy = _vector(
                None if origin is None else origin.get('rpy'), [0.0] * 3)
            relative = _transform(xyz, _rpy_matrix(rpy))
            joint_type = str(joint.get('type', 'fixed'))
            position = float(joint_positions.get(str(joint.get('name')), 0.0))
            if not math.isfinite(position):
                raise ValueError('joint position is not finite')
            if joint_type in ('revolute', 'continuous'):
                axis = _vector(
                    None if joint.find('axis') is None
                    else joint.find('axis').get('xyz'), [1.0, 0.0, 0.0])
                relative = relative @ _transform(
                    rotation=_axis_angle(axis, position))
            elif joint_type == 'prismatic':
                axis = _vector(
                    None if joint.find('axis') is None
                    else joint.find('axis').get('xyz'), [1.0, 0.0, 0.0])
                relative = relative @ _transform(translation=axis * position)
            elif joint_type != 'fixed':
                raise ValueError('unsupported URDF joint type: %s' % joint_type)
            transforms[child_name] = transforms[parent_name] @ relative
            pending.append(child_name)
    if set(transforms) != link_names:
        raise ValueError('URDF contains unreachable links')
    return transforms


def sphere_overlaps(urdf_path, kinematics, joint_positions):
    """Report all non-ignored self overlaps for one joint configuration."""
    transforms = urdf_link_transforms(urdf_path, joint_positions)
    sphere_sets = kinematics.get('collision_spheres')
    if not isinstance(sphere_sets, dict) or not sphere_sets:
        raise ValueError('collision sphere model is missing')
    ignored = {
        str(link): {str(partner) for partner in partners}
        for link, partners in (
            kinematics.get('self_collision_ignore') or {}).items()
    }
    buffer_m = float(kinematics.get('collision_sphere_buffer', 0.0))
    if not math.isfinite(buffer_m):
        raise ValueError('collision sphere buffer is not finite')
    world_spheres = {}
    for link, spheres in sphere_sets.items():
        if link not in transforms:
            raise ValueError('collision-sphere link is absent from URDF: %s' % link)
        transform = transforms[link]
        converted = []
        for index, sphere in enumerate(spheres):
            center = np.asarray(sphere.get('center'), dtype=float)
            radius = float(sphere.get('radius', math.nan)) + buffer_m
            if (
                    center.shape != (3,) or not np.all(np.isfinite(center))
                    or not math.isfinite(radius) or radius <= 0.0):
                raise ValueError('invalid collision sphere for %s' % link)
            world_center = transform[:3, :3] @ center + transform[:3, 3]
            converted.append((index, world_center, radius))
        world_spheres[str(link)] = converted

    overlaps = []
    links = sorted(world_spheres)
    for first_index, first_link in enumerate(links):
        for second_link in links[first_index + 1:]:
            if (
                    second_link in ignored.get(first_link, set())
                    or first_link in ignored.get(second_link, set())):
                continue
            for first_sphere, first_center, first_radius in world_spheres[first_link]:
                for second_sphere, second_center, second_radius in world_spheres[
                        second_link]:
                    penetration = (
                        first_radius + second_radius
                        - float(np.linalg.norm(first_center - second_center)))
                    if penetration > 0.0:
                        overlaps.append(SphereOverlap(
                            first_link=first_link,
                            first_sphere=first_sphere,
                            second_link=second_link,
                            second_sphere=second_sphere,
                            penetration_m=penetration,
                        ))
    return sorted(overlaps, key=lambda item: (
        -item.penetration_m,
        item.first_link,
        item.first_sphere,
        item.second_link,
        item.second_sphere,
    ))
