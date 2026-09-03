#!/usr/bin/env python3
"""Derive and audit cuRobo collision geometry from the canonical planning model.

cuRobo 0.7.8 represents articulated robot collision geometry with spheres, but
supports triangle meshes for fixed world obstacles.  The generated document
therefore accepts a reviewed, hash-bound sphere model for moving links, records
its measured error against every source collision surface, and uses the exact
hash-bound Bunker meshes for the fixed world.  Links not present in the reviewed
model retain the deterministic bounded generator as an explicit fallback.
"""

import argparse
import hashlib
import math
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from motion_planning.curobo import (
    PINNED_COMMIT,
    PINNED_VERSION,
    POSITION_LIMIT_CLIP_RAD,
)


JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
# The canonical PiPER URDF includes the two independently commanded gripper
# fingers.  Arm motion plans intentionally remain six-DOF, so cuRobo must fold
# both fingers into the fixed kinematic model at the neutral position rather
# than trying to reconcile them with the arm-only cspace below.
LOCKED_JOINTS = {'joint7': 0.0, 'joint8': 0.0}
FIXED_WORLD_LINKS = {
    'bunker_chassis_collision', 'bunker_sensor_station_collision'}
FIXED_WORLD_MESH_FILES = {
    # PiPER's base is rigidly bolted to the Bunker.  Keeping it as a robot
    # sphere makes cuRobo test it against the overlapping platform mesh, while
    # Tesseract correctly disables that rigid-mount pair.  Model the exact base
    # mesh as fixed world geometry instead: moving links still collide with it,
    # but fixed world objects are not incorrectly tested against each other.
    'piper_base_collision': 'base_link.STL',
    'bunker_chassis_collision': 'bunker_chassis_collision.STL',
    'bunker_sensor_station_collision':
        'bunker_sensor_station_collision.STL',
}
FIXED_ROBOT_COLLISION_LINKS = {'base_link'}
CURATED_SPHERE_SCHEMA_VERSION = 1


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
    vertices, _sources = collision_vertices(collision, description_root)
    return vertices.min(axis=0), vertices.max(axis=0)


def collision_vertices(collision, description_root):
    """Return collision-surface samples and their canonical source records."""
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
        return vertices, [{
            'path': str(path),
            'sha256': sha256_file(path),
            'scale': [float(item) for item in scale],
        }]
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
    return corners, []


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


def deduplicate_spheres(spheres, cell_size, separation_ratio=0.75):
    """Prune redundant overlapping centres without merging grid neighbours.

    The previous cell-size rounding merged legitimate adjacent grid centres
    (Python's ties-to-even rounding made ``-0.5`` and ``+0.5`` especially
    destructive) and then retained the smaller radius.  Process largest-first
    and use an explicit Euclidean separation smaller than one grid step.  This
    removes duplicate/near-duplicate decomposition cells, retains 40 mm grid
    neighbours, and keeps cuRobo below its slow large-self-collision kernel.
    """
    separation = float(cell_size) * float(separation_ratio)
    if not math.isfinite(separation) or separation <= 0.0:
        raise ValueError('sphere centre separation must be positive')
    selected = []
    ordered = sorted(spheres, key=lambda item: (
        -float(item['radius']),
        tuple(float(value) for value in item['center']),
    ))
    for sphere in ordered:
        center = np.asarray(sphere['center'], dtype=float)
        if all(
                float(np.linalg.norm(
                    center - np.asarray(item['center'], dtype=float)))
                >= separation
                for item in selected):
            selected.append(sphere)
    return sorted(selected, key=lambda item: tuple(item['center']))


def load_curated_spheres(path, urdf_path):
    """Load a portable Isaac-authored sphere model and bind it to the URDF.

    The USD itself is intentionally not a runtime dependency.  Its reviewed
    sphere centres are stored in a small YAML source asset whose own hash is
    carried into generated-model provenance.  The source URDF hash prevents a
    tuned model from being silently reused after kinematic geometry changes.
    """
    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise ValueError('curated collision-sphere model is missing: %s' % path)
    try:
        document = yaml.safe_load(source_path.read_text(encoding='utf-8'))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError('curated collision-sphere model is malformed') from error
    if not isinstance(document, dict):
        raise ValueError('curated collision-sphere model is not a mapping')
    if int(document.get('schema_version', 0)) != CURATED_SPHERE_SCHEMA_VERSION:
        raise ValueError('unsupported curated collision-sphere schema')
    source = document.get('source')
    if not isinstance(source, dict):
        raise ValueError('curated collision-sphere source is missing')
    urdf_hash = str(source.get('planning_urdf_sha256', ''))
    if len(urdf_hash) != 64 or urdf_hash != sha256_file(urdf_path):
        raise ValueError('curated collision spheres do not match the planning URDF')
    usd_hash = str(source.get('usd_sha256', ''))
    if len(usd_hash) != 64:
        raise ValueError('curated collision-sphere USD provenance is missing')
    configured = document.get('collision_spheres')
    if not isinstance(configured, dict) or not configured:
        raise ValueError('curated collision-sphere model contains no links')
    cleaned = {}
    source_links = {}
    for link_name, entries in configured.items():
        name = str(link_name)
        if not name or not isinstance(entries, list) or not entries:
            raise ValueError('curated sphere list is malformed for %s' % name)
        link_spheres = []
        provenance_links = set()
        centers = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError('curated sphere entry is malformed for %s' % name)
            center = np.asarray(entry.get('center'), dtype=float)
            radius = float(entry.get('radius', math.nan))
            source_link = str(entry.get('source_link', name))
            if (
                    center.shape != (3,) or not np.all(np.isfinite(center))
                    or not math.isfinite(radius) or not 0.002 <= radius <= 0.15
                    or not source_link):
                raise ValueError('curated sphere geometry is invalid for %s' % name)
            key = tuple(round(float(value), 9) for value in center)
            if key in centers:
                raise ValueError('curated sphere centres are duplicated for %s' % name)
            centers.add(key)
            provenance_links.add(source_link)
            link_spheres.append({
                'center': list(key),
                'radius': round(radius, 9),
            })
        cleaned[name] = link_spheres
        source_links[name] = sorted(provenance_links)
    fallbacks = document.get('generated_fallback_links', [])
    if not isinstance(fallbacks, list) or not all(
            isinstance(item, str) and item for item in fallbacks):
        raise ValueError('curated generated-fallback link list is malformed')
    qualification_notes = document.get('qualification_notes', [])
    if not isinstance(qualification_notes, list) or not all(
            isinstance(note, str) and note for note in qualification_notes):
        raise ValueError('curated qualification notes are malformed')
    qualification = document.get('qualification')
    if not isinstance(qualification, dict):
        raise ValueError('curated hardware qualification is missing')
    if qualification.get('hardware_qualified') is not True:
        raise ValueError('curated collision model is not hardware-qualified')
    if (
            str(qualification.get('scope', '')) !=
            'supervised_5_percent_target_scan'
            or str(qualification.get('basis', '')) !=
            'operator_reported_physical_e2e'
            or str(qualification.get('floor_profile', '')) != 'tabletop'
            or float(qualification.get(
                'free_motion_speed_percent', float('nan'))) != 5.0
            or float(qualification.get(
                'contact_speed_percent', float('nan'))) != 5.0
            or qualification.get('real_motion_requires_explicit_opt_in') is not True):
        raise ValueError('curated hardware qualification scope is invalid')
    qualification_date = str(qualification.get('qualification_date', ''))
    if not re.fullmatch(r'20[0-9]{2}-[0-9]{2}-[0-9]{2}', qualification_date):
        raise ValueError('curated hardware qualification date is invalid')
    return cleaned, {
        'path': str(source_path),
        'sha256': sha256_file(source_path),
        'model_name': str(document.get('model_name', '')),
        'source_usd_sha256': usd_hash,
        'qualification_notes': list(qualification_notes),
        'hardware_qualification': dict(qualification),
        'source_links_by_collision_owner': source_links,
        'generated_fallback_links': sorted(set(fallbacks)),
    }


def surface_coverage(vertices, spheres, chunk_size=20000):
    """Measure signed surface gap to a sphere union without large allocations.

    A positive gap is uncovered source surface.  A non-positive gap lies in at
    least one sphere.  This audit is deliberately independent from cuRobo so it
    also runs in ordinary CPU-only tests.
    """
    points = np.asarray(vertices, dtype=float).reshape((-1, 3))
    centers = np.asarray([item['center'] for item in spheres], dtype=float)
    radii = np.asarray([item['radius'] for item in spheres], dtype=float)
    if not len(points) or not len(centers):
        raise ValueError('surface coverage requires vertices and spheres')
    covered = 0
    maximum_gap = -math.inf
    for start in range(0, len(points), int(chunk_size)):
        block = points[start:start + int(chunk_size)]
        gaps = np.min(
            np.linalg.norm(
                block[:, None, :] - centers[None, :, :], axis=2)
            - radii[None, :],
            axis=1,
        )
        covered += int(np.count_nonzero(gaps <= 1e-9))
        maximum_gap = max(maximum_gap, float(np.max(gaps)))
    return {
        'sample_count': int(len(points)),
        'covered_sample_count': covered,
        'covered_fraction': round(float(covered) / float(len(points)), 9),
        'maximum_uncovered_gap_m': round(max(0.0, maximum_gap), 9),
    }


def fixed_world_meshes(description_root):
    """Describe the exact base-frame Bunker meshes with immutable hashes."""
    mesh_root = Path(description_root).resolve() / 'meshes'
    result = []
    for name in sorted(FIXED_WORLD_MESH_FILES):
        path = (mesh_root / FIXED_WORLD_MESH_FILES[name]).resolve()
        if not path.is_file():
            raise ValueError('fixed world collision mesh is missing: %s' % path)
        result.append({
            'name': name,
            'file_path': str(path),
            'pose': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            'sha256': sha256_file(path),
        })
    return result


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
        description_root, cell_size_m, sphere_model_path=None):
    root = ET.parse(urdf_path).getroot()
    curated_spheres = {}
    curated_provenance = None
    if sphere_model_path is not None:
        curated_spheres, curated_provenance = load_curated_spheres(
            sphere_model_path, urdf_path)
    spheres = {}
    collision_links = []
    coverage = {}
    moving_mesh_sources = {}
    for link in root.findall('link'):
        name = str(link.get('name', ''))
        collisions = link.findall('collision')
        if not name or not collisions:
            continue
        link_spheres = []
        link_vertices = []
        link_sources = []
        for collision in collisions:
            if name in FIXED_WORLD_LINKS or name in FIXED_ROBOT_COLLISION_LINKS:
                continue
            vertices, sources = collision_vertices(
                collision, description_root)
            low, high = vertices.min(axis=0), vertices.max(axis=0)
            link_spheres.extend(covering_spheres(low, high, cell_size_m))
            link_vertices.append(vertices)
            link_sources.extend(sources)
        if link_spheres:
            collision_links.append(name)
            if name in curated_spheres:
                spheres[name] = curated_spheres[name]
            else:
                spheres[name] = deduplicate_spheres(link_spheres, cell_size_m)
            coverage[name] = surface_coverage(
                np.concatenate(link_vertices), spheres[name])
            unique_sources = {
                (item['path'], item['sha256'], tuple(item['scale'])):
                    item
                for item in link_sources
            }
            moving_mesh_sources[name] = [
                unique_sources[key] for key in sorted(unique_sources)]
    if not spheres:
        raise ValueError('planning URDF contains no collision geometry')
    unknown_curated_links = set(curated_spheres) - set(collision_links)
    if unknown_curated_links:
        raise ValueError(
            'curated spheres contain unknown planning links: %s'
            % ', '.join(sorted(unknown_curated_links)))
    curated_links = sorted(set(curated_spheres))
    generated_links = sorted(set(collision_links) - set(curated_spheres))
    if curated_provenance is not None:
        expected_fallbacks = curated_provenance['generated_fallback_links']
        if expected_fallbacks != generated_links:
            raise ValueError(
                'generated fallback links do not match the curated model: %s'
                % ', '.join(generated_links))
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
                # Retain link6 FK for the common direct-home CAD-holder
                # support-plane check as well as the camera end-effector FK.
                'link_names': ['link6'],
                'lock_joints': LOCKED_JOINTS,
                'extra_links': None,
                'collision_link_names': collision_links,
                'collision_spheres': spheres,
                'collision_sphere_buffer': 0.0,
                'extra_collision_spheres': {},
                'self_collision_ignore': self_collision_ignore,
                # Link 1's four operator-fitted spheres leave a measured gap in
                # exact-mesh collision samples.  A 7 mm buffer closes every
                # observed false-negative gap while preserving the exact-clear
                # zero, neutral, and qualified-scan reference poses;
                # adjacent pairs remain governed by the unchanged SRDF ignores.
                'self_collision_buffer': {'link1': 0.007},
                'use_global_cumul': True,
                'mesh_link_names': None,
                'external_asset_path': str(Path(description_root).resolve()),
                'cspace': {
                    'joint_names': JOINT_NAMES,
                    'retract_config': [0.0, 0.0, 0.0, 0.0, 0.43869236, 0.0],
                    'null_space_weight': [1.0] * 6,
                    'cspace_distance_weight': [1.0] * 6,
                    # cuRobo uses float32 internally.  Keep generated paths a
                    # meaningful distance inside the raw PiPER limits rather
                    # than allowing a nominal boundary value to round outside.
                    'position_limit_clip': list(POSITION_LIMIT_CLIP_RAD),
                    'max_jerk': 50.0,
                    'max_acceleration': 5.0,
                },
            },
        },
        'piper_curobo_provenance': {
            'schema_version': 3,
            'source_srdf_path': str(Path(srdf_path).resolve()),
            'source_urdf_sha256': sha256_file(urdf_path),
            'source_srdf_sha256': sha256_file(srdf_path),
            'source_collision_manifest_path': str(
                Path(collision_manifest_path).resolve()),
            'source_collision_manifest_sha256': sha256_file(
                collision_manifest_path),
            'curobo_version': PINNED_VERSION,
            'curobo_commit': PINNED_COMMIT,
            'position_limit_clip_rad': list(POSITION_LIMIT_CLIP_RAD),
            'covering_shape': (
                'isaac_lula_hand_tuned_with_generated_fallback'
                if curated_provenance is not None
                else 'per_collision_aabb_regular_sphere_grid'),
            'cell_size_m': float(cell_size_m),
            'sphere_surface_inset_m': 0.004,
            'minimum_sphere_center_separation_m': round(
                float(cell_size_m) * 0.75, 9),
            'moving_link_surface_coverage': coverage,
            'moving_link_mesh_sources': moving_mesh_sources,
            'curated_sphere_model': curated_provenance,
            'curated_collision_links': curated_links,
            'generated_fallback_collision_links': generated_links,
            'sphere_count_by_link': {
                name: len(spheres[name]) for name in sorted(spheres)},
            'conservative_geometry': False,
            # Qualification is owned by the reviewed, hash-bound sphere source
            # rather than being granted by the generator itself.
            'hardware_qualified': bool(
                curated_provenance is not None
                and curated_provenance['hardware_qualification'][
                    'hardware_qualified'] is True),
            'hardware_qualification': (
                dict(curated_provenance['hardware_qualification'])
                if curated_provenance is not None else {
                    'hardware_qualified': False,
                    'scope': 'generated_model_unqualified',
                    'floor_profile': '',
                    'free_motion_speed_percent': None,
                    'contact_speed_percent': None,
                    'real_motion_requires_explicit_opt_in': True,
                }),
            # The rigid PiPER base and both Bunker meshes are already expressed
            # in base_link by the canonical model and hash-locked manifest.
            'fixed_world_meshes': fixed_world_meshes(description_root),
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
    parser.add_argument('--sphere-model')
    args = parser.parse_args()
    if not 0.01 <= args.cell_size_m <= 0.10:
        raise SystemExit('cell-size-m must be within 0.01..0.10')
    payload = build(
        args.urdf, args.srdf, args.collision_manifest,
        args.description_root, args.cell_size_m, args.sphere_model)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
