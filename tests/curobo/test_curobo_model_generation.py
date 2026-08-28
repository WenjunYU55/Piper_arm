"""Pure canonical-model conversion tests; CUDA is deliberately absent."""

from pathlib import Path
import struct

import numpy as np
import pytest

from motion_planning.curobo import PINNED_WARP_VERSION
from motion_planning.curobo.generate_robot_config import (
    FIXED_WORLD_MESH_FILES,
    FIXED_WORLD_LINKS,
    JOINT_NAMES,
    LOCKED_JOINTS,
    covering_spheres,
    deduplicate_spheres,
    fixed_world_meshes,
    stl_vertices,
    surface_coverage,
)
from motion_planning.curobo.worker import validate_model_provenance


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


def test_sphere_pruning_keeps_grid_neighbours_and_largest_overlap():
    spheres = [
        {'center': [0.039, 0.0, 0.0], 'radius': 0.016},
        {'center': [0.041, 0.0, 0.0], 'radius': 0.015},
        {'center': [0.039, 0.0, 0.0], 'radius': 0.018},
        {'center': [0.079, 0.0, 0.0], 'radius': 0.016},
    ]
    assert deduplicate_spheres(spheres, 0.04) == [
        spheres[2], spheres[3]]


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


def test_fixed_bunker_uses_full_hash_bound_base_frame_meshes():
    description = ROOT / 'piper_ros_foxy/src/piper_description'
    records = fixed_world_meshes(description)
    assert [item['name'] for item in records] == sorted(FIXED_WORLD_LINKS)
    assert FIXED_WORLD_MESH_FILES == {
        'bunker_chassis_collision': 'bunker_chassis_collision.STL',
        'bunker_sensor_station_collision':
            'bunker_sensor_station_collision.STL',
    }
    for item in records:
        path = Path(item['file_path'])
        assert path.is_file()
        assert path.parent == description.resolve() / 'meshes'
        assert len(item['sha256']) == 64
        assert item['pose'] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def test_surface_coverage_reports_uncovered_mesh_samples():
    vertices = np.asarray([
        [0.0, 0.0, 0.0],
        [0.01, 0.0, 0.0],
        [0.10, 0.0, 0.0],
    ])
    report = surface_coverage(vertices, [{
        'center': [0.0, 0.0, 0.0],
        'radius': 0.02,
    }], chunk_size=2)
    assert report == {
        'sample_count': 3,
        'covered_sample_count': 2,
        'covered_fraction': pytest.approx(2.0 / 3.0),
        'maximum_uncovered_gap_m': 0.08,
    }


def test_schema_two_collision_provenance_fails_closed_when_incomplete(
        tmp_path):
    urdf = tmp_path / 'robot.urdf'
    srdf = tmp_path / 'robot.srdf'
    manifest = tmp_path / 'collision.yaml'
    for path in (urdf, srdf, manifest):
        path.write_text(path.name, encoding='utf-8')
    from motion_planning.curobo import PINNED_COMMIT, PINNED_VERSION
    from motion_planning.curobo.worker import sha256_file
    document = {
        'robot_cfg': {'kinematics': {
            'urdf_path': str(urdf),
            'collision_link_names': ['link1'],
        }},
        'piper_curobo_provenance': {
            'schema_version': 2,
            'source_urdf_sha256': sha256_file(urdf),
            'source_srdf_path': str(srdf),
            'source_srdf_sha256': sha256_file(srdf),
            'source_collision_manifest_path': str(manifest),
            'source_collision_manifest_sha256': sha256_file(manifest),
            'curobo_version': PINNED_VERSION,
            'curobo_commit': PINNED_COMMIT,
            'conservative_geometry': False,
            'hardware_qualified': False,
            'moving_link_surface_coverage': {
                'link1': {
                    'sample_count': 1,
                    'covered_sample_count': 1,
                    'covered_fraction': 1.0,
                    'maximum_uncovered_gap_m': 0.0,
                },
            },
            'moving_link_mesh_sources': {},
            'fixed_world_meshes': [],
        },
    }
    with pytest.raises(ValueError, match='both canonical Bunker meshes'):
        validate_model_provenance(document)


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
