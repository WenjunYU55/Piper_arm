"""Pure canonical-model conversion tests; CUDA is deliberately absent."""

from pathlib import Path
import struct

import numpy as np
import pytest

from motion_planning.curobo import PINNED_WARP_VERSION
from motion_planning.curobo.generate_robot_config import (
    FIXED_WORLD_LINKS,
    JOINT_NAMES,
    LOCKED_JOINTS,
    covering_spheres,
    deduplicate_spheres,
    fixed_world_bounds,
    stl_vertices,
)


ROOT = Path(__file__).resolve().parents[2]


def test_runtime_constraints_pin_warp_api_used_by_curobo_v078():
    constraints = (
        ROOT / 'motion_planning/curobo/constraints.txt').read_text(
            encoding='utf-8')
    assert PINNED_WARP_VERSION == '1.11.1'
    assert 'warp-lang==%s' % PINNED_WARP_VERSION in constraints


def test_ascii_stl_vertices_are_read_without_geometry_dependencies(tmp_path):
    path = tmp_path / 'triangle.stl'
    path.write_text(
        'solid test\n'
        ' facet normal 0 0 1\n'
        '  outer loop\n'
        '   vertex 0 0 0\n'
        '   vertex 1 0 0\n'
        '   vertex 0 1 0\n'
        '  endloop\n'
        ' endfacet\n'
        'endsolid test\n',
        encoding='ascii',
    )
    assert stl_vertices(path) == pytest.approx(np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]))


def test_binary_stl_vertices_are_read_deterministically(tmp_path):
    path = tmp_path / 'triangle.stl'
    payload = bytearray(80)
    payload.extend(struct.pack('<I', 1))
    payload.extend(struct.pack('<fff', 0.0, 0.0, 1.0))
    for vertex in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
        payload.extend(struct.pack('<fff', *vertex))
    payload.extend(struct.pack('<H', 0))
    path.write_bytes(payload)
    assert stl_vertices(path) == pytest.approx(np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]))


def test_sphere_grid_uses_bounded_regular_approximation():
    low = np.asarray([-0.04, -0.02, -0.01])
    high = np.asarray([0.04, 0.02, 0.01])
    spheres = covering_spheres(low, high, 0.04)
    assert len(spheres) == 2
    assert {sphere['radius'] for sphere in spheres} == {0.016}
    assert [sphere['center'] for sphere in spheres] == [
        [-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]]


def test_overlapping_decomposed_mesh_spheres_are_deduplicated():
    spheres = [
        {'center': [0.039, 0.0, 0.0], 'radius': 0.016},
        {'center': [0.041, 0.0, 0.0], 'radius': 0.015},
        {'center': [0.079, 0.0, 0.0], 'radius': 0.016},
    ]
    assert deduplicate_spheres(spheres, 0.04) == [
        spheres[1], spheres[2]]


def test_arm_planning_contract_locks_canonical_gripper_joints():
    assert JOINT_NAMES == [
        'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    assert LOCKED_JOINTS == {'joint7': 0.0, 'joint8': 0.0}

    urdf = ROOT / 'piper_ros_foxy/src/piper_description/urdf/piper_description.urdf'
    text = urdf.read_text(encoding='utf-8')
    for joint_name in LOCKED_JOINTS:
        assert 'name="%s"' % joint_name in text


def test_robot_base_is_not_misclassified_as_world_geometry():
    assert 'base_link' not in FIXED_WORLD_LINKS
    assert FIXED_WORLD_LINKS == {
        'bunker_chassis_collision', 'bunker_sensor_station_collision'}


def test_fixed_chassis_stops_below_the_srdf_rigid_mount_seam():
    low, high = fixed_world_bounds(
        'bunker_chassis_collision', [-1.0, -0.3, -0.4], [0.1, 0.3, 0.0])
    assert low == pytest.approx([-1.0, -0.3, -0.4])
    assert high == pytest.approx([0.1, 0.3, -0.005])

    _, sensor_high = fixed_world_bounds(
        'bunker_sensor_station_collision',
        [-1.0, -0.3, 0.0], [-0.2, 0.3, 0.4])
    assert sensor_high == pytest.approx([-0.2, 0.3, 0.4])


def test_curobo_scripts_discard_inherited_pythonpath():
    worker = (ROOT / 'motion_planning/curobo/run_worker.sh').read_text(
        encoding='utf-8')
    model = (ROOT / 'motion_planning/curobo/prepare_model.sh').read_text(
        encoding='utf-8')
    assert 'export PYTHONPATH="$ROOT:$ROOT/piper_ros_foxy/src/piper_tesseract_foxy"' in worker
    assert 'PYTHONPATH="$ROOT" \\\n' in model
    assert '${PYTHONPATH:+' not in worker
    assert '${PYTHONPATH:+' not in model
    assert 'PIPER_CUROBO_CUDA_HOME' in worker
    assert 'export CUDA_HOME="$CUROBO_CUDA_HOME"' in worker
