import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import xml.etree.ElementTree as ET

import pytest
import yaml

from piper_tesseract_foxy.collision_meshes import read_binary_stl
TRIANGLE_STRUCT = struct.Struct('<12fH')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mesh_bounds(path):
    points = []
    for record in read_binary_stl(path):
        values = TRIANGLE_STRUCT.unpack(record)
        points.extend((values[3:6], values[6:9], values[9:12]))
    return (
        tuple(min(point[axis] for point in points) for axis in range(3)),
        tuple(max(point[axis] for point in points) for axis in range(3)),
    )


def repository_root():
    return Path(__file__).resolve().parents[4]


def test_imported_platform_sources_and_transforms_are_hash_locked():
    meshes = repository_root() / 'piper_ros_foxy/src/piper_description/meshes'
    assert sha256(meshes / 'platform_sources/bunker_pro2_base_link.STL') == \
        'aa9f81c3d75c28e6720fda247bc01d052d17d9343b1e050d13ea2f7a5e60ce9d'
    assert sha256(meshes / 'platform_sources/bunker_pro2_FullCase.STL') == \
        '62148eb78b3e3294210e01cab448c4b0f099aacecfbe38d4409f8c2bf8bdd8d5'
    chassis_bounds = mesh_bounds(meshes / 'bunker_chassis_collision.STL')
    assert chassis_bounds[0] == pytest.approx(
        (-1.00921173, -0.394, -0.45525956), abs=1e-6)
    assert chassis_bounds[1] == pytest.approx(
        (0.05812238, 0.3945, 0.0), abs=1e-6)
    station_bounds = mesh_bounds(
        meshes / 'bunker_sensor_station_collision.STL')
    assert station_bounds[0] == pytest.approx(
        (-0.87125, -0.2335, 0.0), abs=1e-6)
    assert station_bounds[1] == pytest.approx(
        (-0.33375, 0.2335, 0.44755), abs=1e-6)


def test_bunker_decomposition_retains_every_source_triangle_and_is_unqualified():
    root = repository_root()
    description = root / 'piper_ros_foxy/src/piper_description/meshes'
    manifest_path = (
        description / 'platform_planning_150mm/collision_mesh_manifest.json')
    assert sha256(manifest_path) == \
        '58d79534efba2ae68ea1e409a0f3c07321561f5622ceb9177f1b2c3c7d89fcdf'
    document = json.loads(manifest_path.read_text(encoding='utf-8'))
    entries = {entry['link']: entry for entry in document['links']}
    assert len(entries['bunker_chassis_collision']['pieces']) == 46
    assert len(entries['bunker_sensor_station_collision']['pieces']) == 16
    for entry in entries.values():
        assert entry['triangle_assignments'] >= entry['source_triangle_count']
        for piece in entry['pieces']:
            path = description / 'platform_planning_150mm' / piece['filename']
            assert sha256(path) == piece['sha256']

    model = yaml.safe_load((
        root / 'piper_ros_foxy/src/piper_tesseract_foxy/model/'
        'collision_model_ground.yaml').read_text(encoding='utf-8'))
    assert model['collision_profile'] == 'combined_platform_ground_floor'
    assert model['qualified_for_hardware'] is False
    assert model['external_floor_clearance']['floor_z_m'] == -0.466


def test_tabletop_floor_profile_uses_qualified_platform_geometry():
    root = repository_root()
    model = yaml.safe_load((
        root / 'piper_ros_foxy/src/piper_tesseract_foxy/model/'
        'collision_model.yaml').read_text(encoding='utf-8'))
    assert model['collision_profile'] == 'combined_platform_tabletop_floor'
    assert model['qualified_for_hardware'] is True
    assert model['external_floor_clearance']['floor_z_m'] == 0.005
    policies = model['collision_mesh_decompositions']
    assert len(policies) == 2
    assert policies[1]['manifest_uri'].endswith(
        'platform_planning_150mm/collision_mesh_manifest.json')


def test_bunker_srdf_disables_only_required_fixed_platform_pairs():
    srdf = ET.parse(str(
        repository_root()
        / 'piper_ros_foxy/src/piper_tesseract_foxy/model/piper_bunker.srdf'))
    pairs = {
        frozenset((item.get('link1'), item.get('link2')))
        for item in srdf.getroot().findall('disable_collisions')
    }
    assert frozenset(('base_link', 'bunker_chassis_collision')) in pairs
    assert frozenset((
        'bunker_chassis_collision',
        'bunker_sensor_station_collision')) in pairs
    moving_links = {
        'link1', 'link2', 'link3', 'link4', 'link5', 'link6',
        'gripper_base', 'link7', 'link8', 'l515_attached_assembly',
    }
    assert not any(
        platform in pair and any(link in pair for link in moving_links)
        for pair in pairs
        for platform in (
            'bunker_chassis_collision',
            'bunker_sensor_station_collision')
    )


def test_floor_profiles_share_platform_geometry_and_srdf():
    root = repository_root()
    models = root / 'piper_ros_foxy/src/piper_tesseract_foxy/model'
    tabletop = yaml.safe_load((models / 'collision_model.yaml').read_text(
        encoding='utf-8'))
    ground = yaml.safe_load((models / 'collision_model_ground.yaml').read_text(
        encoding='utf-8'))

    tabletop_policies = tabletop['collision_mesh_decompositions']
    ground_policies = ground['collision_mesh_decompositions']
    assert len(tabletop_policies) == len(ground_policies) == 2
    for tabletop_policy, ground_policy in zip(
            tabletop_policies, ground_policies):
        assert tabletop_policy['manifest_uri'] == ground_policy['manifest_uri']
        assert tabletop_policy['manifest_sha256'] == (
            ground_policy['manifest_sha256'])
        assert tabletop_policy['mesh_uri_prefix'] == (
            ground_policy['mesh_uri_prefix'])
    assert tabletop['external_floor_clearance']['floor_z_m'] == 0.005
    assert ground['external_floor_clearance']['floor_z_m'] == -0.466

    helper = root / 'motion_planning/tesseract/floor_profile.sh'
    for profile, manifest in (
            ('tabletop', 'collision_model.yaml'),
            ('ground', 'collision_model_ground.yaml')):
        environment = dict(os.environ, PIPER_FLOOR_PROFILE=profile)
        result = subprocess.run(
            ['bash', '-c', (
                'source "$1"\n'
                'printf "%s|%s" "$COLLISION_MANIFEST_NAME" '
                '"$COLLISION_SRDF_NAME"'), 'profile-test', str(helper)],
            check=True, capture_output=True, text=True, env=environment)
        assert result.stdout == '%s|piper_bunker.srdf' % manifest
