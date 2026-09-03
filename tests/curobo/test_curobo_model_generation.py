"""Pure canonical-model conversion tests; CUDA is deliberately absent."""

import copy
from pathlib import Path
import os
import struct
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from motion_planning.curobo import (
    PINNED_WARP_VERSION,
    POSITION_LIMIT_CLIP_RAD,
)
from motion_planning.curobo.generate_robot_config import (
    build,
    FIXED_WORLD_MESH_FILES,
    FIXED_WORLD_LINKS,
    FIXED_ROBOT_COLLISION_LINKS,
    JOINT_NAMES,
    LOCKED_JOINTS,
    covering_spheres,
    deduplicate_spheres,
    fixed_world_meshes,
    stl_vertices,
    surface_coverage,
)
from motion_planning.curobo.collision_qualification import sphere_overlaps
from motion_planning.curobo.worker import CuroboBackend, validate_model_provenance


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def curated_model(tmp_path_factory):
    output_root = tmp_path_factory.mktemp('curated_curobo_model')
    urdf = output_root / 'piper_planning.urdf'
    description = ROOT / 'piper_ros_foxy/src/piper_description'
    tesseract_package = ROOT / 'piper_ros_foxy/src/piper_tesseract_foxy'
    model = tesseract_package / 'model'
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(tesseract_package)
    subprocess.run([
        '/usr/bin/python3', '-m', 'piper_tesseract_foxy.model_builder',
        '--xacro', str(description / 'urdf/piper_description.xacro'),
        '--calibration', str(
            ROOT / 'L515_camera/calibration/hand_eye/'
            'session_20260808_straight_mount/calibration_result.yaml'),
        '--manifest', str(model / 'collision_model.yaml'),
        '--output', str(urdf),
    ], check=True, env=environment)
    return build(
        urdf,
        model / 'piper_bunker.srdf',
        model / 'collision_model.yaml',
        description,
        0.04,
        ROOT / 'motion_planning/curobo/model/piper_collision_spheres.yaml',
    )


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


def test_generated_model_keeps_motiongen_inside_raw_joint_limits(
        curated_model):
    kinematics = curated_model['robot_cfg']['kinematics']
    provenance = curated_model['piper_curobo_provenance']
    assert POSITION_LIMIT_CLIP_RAD == pytest.approx(
        (0.005, 0.0, 0.0, 0.005, 0.005, 0.005))
    assert kinematics['cspace']['position_limit_clip'] == pytest.approx(
        POSITION_LIMIT_CLIP_RAD)
    assert kinematics['link_names'] == ['link6']
    assert provenance['schema_version'] == 3
    assert provenance['position_limit_clip_rad'] == pytest.approx(
        POSITION_LIMIT_CLIP_RAD)
    validate_model_provenance(curated_model)

    tampered = copy.deepcopy(curated_model)
    tampered['robot_cfg']['kinematics']['cspace'][
        'position_limit_clip'] = [0.0] * 6
    with pytest.raises(ValueError, match='position-limit clip'):
        validate_model_provenance(tampered)


def test_rigid_robot_base_is_exact_fixed_world_geometry():
    assert 'base_link' not in FIXED_WORLD_LINKS
    assert FIXED_ROBOT_COLLISION_LINKS == {'base_link'}
    assert FIXED_WORLD_LINKS == {
        'bunker_chassis_collision', 'bunker_sensor_station_collision'}


def test_fixed_bunker_uses_full_hash_bound_base_frame_meshes():
    description = ROOT / 'piper_ros_foxy/src/piper_description'
    records = fixed_world_meshes(description)
    assert [item['name'] for item in records] == sorted(FIXED_WORLD_MESH_FILES)
    assert FIXED_WORLD_MESH_FILES == {
        'piper_base_collision': 'base_link.STL',
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


def test_curated_spheres_are_bounded_and_preserve_collision_evidence(
        curated_model):
    kinematics = curated_model['robot_cfg']['kinematics']
    provenance = curated_model['piper_curobo_provenance']
    joint_names = [
        'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    clear_poses = (
        [0.0] * 6,
        [0.0, 0.8, -0.7, 0.0, 0.7, 0.0],
        [
            0.3189509166, 0.7800870124, -1.6258884709,
            -0.6660237320, -0.2154052887, 0.0403545644,
        ],
    )
    for positions in clear_poses:
        assert sphere_overlaps(
            kinematics['urdf_path'],
            kinematics,
            dict(zip(joint_names, positions)),
        ) == []

    colliding = sphere_overlaps(
        kinematics['urdf_path'],
        kinematics,
        dict(zip(joint_names, [0.0, 0.0, 0.0, 0.0, 0.43869236, 0.0])),
    )
    # Tesseract reports both link1/link5 and link2/link5 at this folded state.
    # A sphere approximation need not reproduce every mesh contact pair, but it
    # must reject the same state through at least one real contact pair.
    assert any(
        {item.first_link, item.second_link} == {'link1', 'link5'}
        for item in colliding)
    assert sum(provenance['sphere_count_by_link'].values()) == 69
    assert 'base_link' not in provenance['sphere_count_by_link']
    assert provenance['hardware_qualified'] is True
    assert provenance['hardware_qualification'] == {
        'hardware_qualified': True,
        'qualification_date': '2026-09-02',
        'scope': 'supervised_5_percent_target_scan',
        'basis': 'operator_reported_physical_e2e',
        'floor_profile': 'tabletop',
        'free_motion_speed_percent': 5.0,
        'contact_speed_percent': 5.0,
        'real_motion_requires_explicit_opt_in': True,
    }
    assert provenance['conservative_geometry'] is False


def test_curated_sphere_provenance_is_hash_bound(curated_model):
    validate_model_provenance(curated_model)
    tampered = copy.deepcopy(curated_model)
    tampered['piper_curobo_provenance'][
        'curated_sphere_model']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='hash-bound model input'):
        validate_model_provenance(tampered)


def test_curated_sphere_counts_fail_closed(curated_model):
    tampered = copy.deepcopy(curated_model)
    tampered['piper_curobo_provenance']['sphere_count_by_link']['link5'] -= 1
    with pytest.raises(ValueError, match='sphere counts'):
        validate_model_provenance(tampered)


def test_backend_restores_both_persistent_collision_constraints():
    class Constraint:
        def __init__(self):
            self.enabled = False

        def enable_cost(self):
            self.enabled = True

    primitive = Constraint()
    self_collision = Constraint()
    backend = object.__new__(CuroboBackend)
    backend.motion_gen = SimpleNamespace(rollout_fn=SimpleNamespace(
        primitive_collision_constraint=primitive,
        robot_self_collision_constraint=self_collision,
    ))
    backend._restore_collision_constraints()
    assert primitive.enabled is True
    assert self_collision.enabled is True


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
    with pytest.raises(ValueError, match='canonical PiPER base and Bunker meshes'):
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
