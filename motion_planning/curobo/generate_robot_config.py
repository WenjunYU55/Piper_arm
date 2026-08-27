#!/usr/bin/env python3
"""Derive a conservative cuRobo sphere model from the canonical planning URDF."""

import argparse
import hashlib
import math
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from motion_planning.curobo import PINNED_COMMIT, PINNED_VERSION


JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
# The canonical PiPER URDF includes the two independently commanded gripper
# fingers.  Arm motion plans intentionally remain six-DOF, so cuRobo must fold
# both fingers into the fixed kinematic model at the neutral position rather
# than trying to reconcile them with the arm-only cspace below.
LOCKED_JOINTS = {'joint7': 0.0, 'joint8': 0.0}
FIXED_WORLD_LINKS = {
    'bunker_chassis_collision', 'bunker_sensor_station_collision'}
RIGID_MOUNT_SEAM_Z_M = -0.005


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def values(text, length, default):
    if not text:
        return np.asarray(default, dtype=float)
    result = np.asarray([float(item) for item in text.split()], dtype=float)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError('malformed URDF numeric vector')
    return result


def rotation_rpy(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def mesh_path(uri, description_root):
    prefix = 'package://piper_description/'
    if not str(uri).startswith(prefix):
        raise ValueError('unsupported collision mesh URI: %s' % uri)
    root = Path(description_root).resolve()
    path = (root / str(uri)[len(prefix):]).resolve()
    if path != root and root not in path.parents:
        raise ValueError('collision mesh escapes piper_description')
    if not path.is_file():
        raise ValueError('collision mesh is missing: %s' % path)
    return path


def stl_vertices(path):
    """Read binary or ASCII STL vertices without a geometry dependency."""
    payload = Path(path).read_bytes()
    if len(payload) >= 84:
        triangles = struct.unpack_from('<I', payload, 80)[0]
        if len(payload) == 84 + triangles * 50:
            vertices = []
            for triangle in range(triangles):
                offset = 84 + triangle * 50 + 12
                for vertex in range(3):
                    vertices.append(struct.unpack_from(
                        '<fff', payload, offset + vertex * 12))
            result = np.asarray(vertices, dtype=float)
            if result.size:
                return result
    try:
        text = payload.decode('ascii')
    except UnicodeDecodeError as error:
        raise ValueError('malformed STL mesh: %s' % path) from error
    matches = re.findall(
        r'(?im)^\s*vertex\s+('
        r'[-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$',
        text)
    result = np.asarray(matches, dtype=float)
    if result.ndim != 2 or result.shape[1:] != (3,) or not result.size:
        raise ValueError('STL mesh contains no vertices: %s' % path)
    return result


def collision_bounds(collision, description_root):
    origin = collision.find('origin')
    translation = values(
        None if origin is None else origin.get('xyz'), 3, [0.0, 0.0, 0.0])
    rotation = rotation_rpy(values(
        None if origin is None else origin.get('rpy'), 3, [0.0, 0.0, 0.0]))
    geometry = collision.find('geometry')
    if geometry is None:
        raise ValueError('collision geometry is missing')
    mesh = geometry.find('mesh')
    box = geometry.find('box')
    cylinder = geometry.find('cylinder')
    sphere = geometry.find('sphere')
    if mesh is not None:
        path = mesh_path(mesh.get('filename'), description_root)
        if path.suffix.lower() != '.stl':
            raise ValueError('only STL collision meshes are supported: %s' % path)
        vertices = stl_vertices(path)
        scale = values(mesh.get('scale'), 3, [1.0, 1.0, 1.0])
        vertices = (vertices * scale[None, :]) @ rotation.T + translation
        return vertices.min(axis=0), vertices.max(axis=0)
    if box is not None:
        half = values(box.get('size'), 3, [0.0, 0.0, 0.0]) * 0.5
    elif cylinder is not None:
        radius = float(cylinder.get('radius'))
        half = np.asarray([radius, radius, float(cylinder.get('length')) * 0.5])
    elif sphere is not None:
        radius = float(sphere.get('radius'))
        half = np.asarray([radius, radius, radius])
    else:
        raise ValueError('unsupported collision geometry')
    corners = np.asarray([
        [x, y, z]
        for x in (-half[0], half[0])
        for y in (-half[1], half[1])
        for z in (-half[2], half[2])])
    corners = corners @ rotation.T + translation
    return corners.min(axis=0), corners.max(axis=0)


def covering_spheres(low, high, cell_size):
    """Approximate an AABB with a deterministic regular sphere grid.

    Using the cell diagonal as the radius covers every AABB corner but
    substantially inflates PiPER's narrow links and creates false self
    collisions.  cuRobo robot models are sphere approximations, so use an
    explicitly recorded 4 mm inset from half the largest cell dimension.
    """
    extent = np.asarray(high) - np.asarray(low)
    counts = np.maximum(1, np.ceil(extent / float(cell_size)).astype(int))
    steps = extent / counts
    radius = float(max(0.002, np.max(steps) * 0.5 - 0.004))
    result = []
    for ix in range(int(counts[0])):
        for iy in range(int(counts[1])):
            for iz in range(int(counts[2])):
                center = np.asarray(low) + steps * np.asarray([
                    ix + 0.5, iy + 0.5, iz + 0.5])
                result.append({
                    'center': [round(float(value), 9) for value in center],
                    'radius': round(radius, 9),
                })
    return result


def deduplicate_spheres(spheres, cell_size):
    """Keep one deterministic representative per overlapping grid cell."""
    selected = {}
    size = float(cell_size)
    for sphere in spheres:
        center = np.asarray(sphere['center'], dtype=float)
        key = tuple(int(round(value / size)) for value in center)
        grid_center = np.asarray(key, dtype=float) * size
        rank = (
            float(np.linalg.norm(center - grid_center)),
            float(sphere['radius']),
            tuple(float(value) for value in center),
        )
        if key not in selected or rank < selected[key][0]:
            selected[key] = (rank, sphere)
    return [selected[key][1] for key in sorted(selected)]


def fixed_world_bounds(name, low, high):
    """Apply the SRDF rigid-mount contact seam to fixed world geometry."""
    low = np.asarray(low, dtype=float).copy()
    high = np.asarray(high, dtype=float).copy()
    if name == 'bunker_chassis_collision':
        high[2] = min(high[2], RIGID_MOUNT_SEAM_Z_M)
    if np.any(low >= high):
        raise ValueError('fixed world collision bounds are empty')
    return low, high


def disabled_collision_pairs(srdf_path):
    ignored = {}
    root = ET.parse(srdf_path).getroot()
    for item in root.findall('disable_collisions'):
        left, right = item.get('link1'), item.get('link2')
        if left and right:
            ignored.setdefault(left, []).append(right)
            ignored.setdefault(right, []).append(left)
    return {name: sorted(set(partners)) for name, partners in ignored.items()}


def build(
        urdf_path, srdf_path, collision_manifest_path,
        description_root, cell_size_m):
    root = ET.parse(urdf_path).getroot()
    spheres = {}
    collision_links = []
    platform_cuboids = []
    for link in root.findall('link'):
        name = str(link.get('name', ''))
        collisions = link.findall('collision')
        if not name or not collisions:
            continue
        link_spheres = []
        for collision_index, collision in enumerate(collisions):
            low, high = collision_bounds(collision, description_root)
            if name in FIXED_WORLD_LINKS:
                low, high = fixed_world_bounds(name, low, high)
                platform_cuboids.append({
                    'name': '%s_%03d' % (name, collision_index),
                    'pose': [
                        round(float((low[axis] + high[axis]) * 0.5), 9)
                        for axis in range(3)
                    ] + [1.0, 0.0, 0.0, 0.0],
                    'dims': [
                        round(float(high[axis] - low[axis]), 9)
                        for axis in range(3)],
                })
                continue
            link_spheres.extend(covering_spheres(low, high, cell_size_m))
        if link_spheres:
            collision_links.append(name)
            spheres[name] = deduplicate_spheres(link_spheres, cell_size_m)
    if not spheres:
        raise ValueError('planning URDF contains no collision geometry')
    collision_link_set = set(collision_links)
    self_collision_ignore = {
        name: [partner for partner in partners
               if partner in collision_link_set]
        for name, partners in disabled_collision_pairs(srdf_path).items()
        if name in collision_link_set
    }
    config = {
        'robot_cfg': {
            'kinematics': {
                'urdf_path': str(Path(urdf_path).resolve()),
                'asset_root_path': str(Path(description_root).resolve()),
                'base_link': 'base_link',
                # model_builder.py creates this calibrated eye-in-hand frame.
                'ee_link': 'camera_optical_frame',
                'link_names': None,
                'lock_joints': LOCKED_JOINTS,
                'extra_links': None,
                'collision_link_names': collision_links,
                'collision_spheres': spheres,
                'collision_sphere_buffer': 0.0,
                'extra_collision_spheres': {},
                'self_collision_ignore': self_collision_ignore,
                'self_collision_buffer': {},
                'use_global_cumul': True,
                'mesh_link_names': None,
                'external_asset_path': str(Path(description_root).resolve()),
                'cspace': {
                    'joint_names': JOINT_NAMES,
                    'retract_config': [0.0, 0.0, 0.0, 0.0, 0.43869236, 0.0],
                    'null_space_weight': [1.0] * 6,
                    'cspace_distance_weight': [1.0] * 6,
                    'max_jerk': 50.0,
                    'max_acceleration': 5.0,
                },
            },
        },
        'piper_curobo_provenance': {
            'schema_version': 1,
            'source_urdf_sha256': sha256_file(urdf_path),
            'source_srdf_sha256': sha256_file(srdf_path),
            'source_collision_manifest_sha256': sha256_file(
                collision_manifest_path),
            'curobo_version': PINNED_VERSION,
            'curobo_commit': PINNED_COMMIT,
            'covering_shape': 'per_collision_aabb_regular_sphere_grid',
            'cell_size_m': float(cell_size_m),
            'sphere_surface_inset_m': 0.004,
            'deduplication_grid_m': float(cell_size_m),
            'rigid_mount_seam_z_m': RIGID_MOUNT_SEAM_Z_M,
            'conservative_geometry': False,
            'hardware_qualified': False,
            # The Bunker is fixed in base_link during arm planning. Per-piece
            # AABBs remain world geometry so they do not consume thousands of
            # robot spheres while still colliding with every moving arm link.
            'fixed_world_cuboids': platform_cuboids,
        },
    }
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', required=True)
    parser.add_argument('--srdf', required=True)
    parser.add_argument('--collision-manifest', required=True)
    parser.add_argument('--description-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cell-size-m', type=float, default=0.04)
    args = parser.parse_args()
    if not 0.01 <= args.cell_size_m <= 0.10:
        raise SystemExit('cell-size-m must be within 0.01..0.10')
    payload = build(
        args.urdf, args.srdf, args.collision_manifest,
        args.description_root, args.cell_size_m)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
