from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from piper_mobile_manipulation.capability_map import (
    CapabilityMap,
    capability_key,
    load_capability_map,
    pack_capability_key,
    position_indices,
    sha256_file,
    unpack_capability_key,
    write_capability_map,
)
from piper_mobile_manipulation.viewpoint_reachability_filter_node import (
    CAPABILITY_MAP_REJECTION,
    ViewpointReachabilityFilterNode,
)


def synthetic_map(keys, floor_values=None, qualified=True, sources=None):
    ordered = np.asarray(sorted(int(value) for value in keys), dtype=np.uint64)
    floors = np.asarray(
        floor_values if floor_values is not None else [0.10] * len(ordered),
        dtype=np.float32,
    )
    return CapabilityMap(ordered, floors, {
        'schema_version': 1,
        'position_voxel_m': 0.020,
        'direction_bin_deg': 10.0,
        'direction_tolerance_deg': 15.0,
        'spatial_dilation_cells': 1,
        'tool_floor_clearance_m': 0.005,
        'qualified_for_enforcement': bool(qualified),
        'checkpoint_samples': 100000,
        'source_sha256': dict(sources or {'source.txt': 'unused'}),
    })


def test_packed_key_round_trip_preserves_signed_position_and_direction():
    key = pack_capability_key([-31, 22, -7], [35, 17])
    position, direction = unpack_capability_key(key)
    assert position.tolist() == [-31, 22, -7]
    assert direction.tolist() == [35, 17]


def test_pose_and_ray_lookup_use_sparse_occupied_cells_not_solid_volume():
    target = np.asarray([0.40, 0.0, 0.12])
    direction = np.asarray([0.0, 1.0, 0.0])
    camera = target + direction * 0.34
    key = capability_key(camera, -direction, 0.020, 10.0)
    capability = synthetic_map([key])

    pose = capability.supports_pose(
        camera, -direction, floor_z_m=0.005, clearance_m=0.005)
    ray = capability.intersects_ray(
        target, direction, 0.28, 0.50,
        floor_z_m=0.005, clearance_m=0.005)
    unrelated = capability.intersects_ray(
        target, [-1.0, 0.0, 0.0], 0.28, 0.50,
        floor_z_m=0.005, clearance_m=0.005)

    assert pose.supported
    assert ray.supported
    assert not unrelated.supported
    assert unrelated.reason


def test_query_applies_selected_support_floor_without_claiming_exact_ik():
    target = np.asarray([0.40, 0.0, 0.12])
    direction = np.asarray([0.0, 1.0, 0.0])
    camera = target + direction * 0.34
    key = capability_key(camera, -direction, 0.020, 10.0)
    capability = synthetic_map([key], floor_values=[0.015])

    ground = capability.supports_pose(
        camera, -direction, floor_z_m=-0.466, clearance_m=0.005)
    tabletop = capability.supports_pose(
        camera, -direction, floor_z_m=0.020, clearance_m=0.005)

    assert ground.supported
    assert not tabletop.supported


def test_artifact_is_pickle_free_and_rejects_changed_source_hash(tmp_path):
    source = tmp_path / 'source.txt'
    source.write_text('first', encoding='utf-8')
    target = np.asarray([0.40, 0.0, 0.12])
    direction = np.asarray([0.0, 1.0, 0.0])
    key = capability_key(target + direction * 0.34, -direction, 0.020, 10.0)
    artifact = tmp_path / 'map.npz'
    capability = synthetic_map(
        [key], sources={'source.txt': sha256_file(source)})
    write_capability_map(
        artifact, capability.keys, capability.maximum_tool_minimum_z_m,
        capability.metadata)

    loaded = load_capability_map(artifact, tmp_path, verify_sources=True)
    assert loaded.keys.tolist() == capability.keys.tolist()
    source.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='source hash mismatch'):
        load_capability_map(artifact, tmp_path, verify_sources=True)


def target_ray():
    return {
        'candidate_geometry': 'target_ray',
        'desired_camera_position': {'x': 0.40, 'y': 0.34, 'z': 0.12},
        'target_object_center': {'x': 0.40, 'y': 0.0, 'z': 0.12},
        'ray_direction': {'x': 0.0, 'y': 1.0, 'z': 0.0},
        'ray_min_standoff_m': 0.28,
        'ray_max_standoff_m': 0.50,
    }


def filter_fixture(capability, mode):
    parameters = {
        'dry_run': True,
        'enforce_static_reach_bounds': False,
        'min_reach_m': 0.20,
        'max_reach_m': 0.75,
        'min_camera_object_distance_m': 0.25,
        'max_camera_object_distance_m': 0.80,
        'max_height_change_m': 0.40,
        'floor_z_m': 0.005,
    }
    node = SimpleNamespace(
        target_status='LOCKED',
        capability_map=capability,
        capability_map_effective_mode=mode,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        param_bool=lambda name: bool(parameters[name]),
        arm_status_reasons=lambda: [],
        valid_vector=ViewpointReachabilityFilterNode.valid_vector,
        vector_norm=ViewpointReachabilityFilterNode.vector_norm,
        is_finite_number=ViewpointReachabilityFilterNode.is_finite_number,
        distance=ViewpointReachabilityFilterNode.distance,
    )
    node.capability_query = lambda viewpoint: (
        ViewpointReachabilityFilterNode.capability_query(node, viewpoint))
    return node


def test_shadow_mode_reports_missing_capability_without_rejecting_ray():
    unrelated = capability_key(
        [0.0, -0.4, 0.5], [1.0, 0.0, 0.0], 0.020, 10.0)
    filter_node = filter_fixture(synthetic_map([unrelated]), 'shadow')

    reasons, evidence = ViewpointReachabilityFilterNode.evaluate_viewpoint(
        filter_node, target_ray())

    assert reasons == []
    assert evidence['supported'] is False


def test_enforce_mode_culls_only_map_unsupported_target_ray():
    unrelated = capability_key(
        [0.0, -0.4, 0.5], [1.0, 0.0, 0.0], 0.020, 10.0)
    filter_node = filter_fixture(synthetic_map([unrelated]), 'enforce')

    reasons, evidence = ViewpointReachabilityFilterNode.evaluate_viewpoint(
        filter_node, target_ray())

    assert evidence['supported'] is False
    assert CAPABILITY_MAP_REJECTION in reasons


def test_fallback_without_map_preserves_existing_coarse_ray_decision():
    filter_node = filter_fixture(None, 'fallback_coarse')

    reasons, evidence = ViewpointReachabilityFilterNode.evaluate_viewpoint(
        filter_node, target_ray())

    assert reasons == []
    assert evidence is None


def test_position_quantization_is_stable_at_negative_voxel_boundary():
    assert position_indices(
        [-0.020, -0.0001, 0.0199], 0.020).tolist() == [-1, -1, 0]


def test_committed_map_is_hash_validated_and_enforcement_qualified():
    root = Path(__file__).resolve().parents[4]
    artifact = root / (
        'piper_ros_foxy/src/piper_mobile_manipulation/config/'
        'piper_camera_capability_map.npz')

    capability = load_capability_map(artifact, root, verify_sources=True)

    assert capability.metadata['qualified_for_enforcement'] is True
    assert capability.metadata['selected_checkpoint_samples'] == 2000000
    assert len(capability.keys) == 1479561
    assert capability.metadata['stores_joint_positions'] is False
