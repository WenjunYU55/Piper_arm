import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from reconstruction import input_provenance as INPUTS

cv2 = pytest.importorskip('cv2')


PATH = Path(__file__).with_name('tsdf_reconstruct.py')
SPEC = importlib.util.spec_from_file_location('tsdf_reconstruct', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_tsdf_module_preserves_input_admission_exports():
    names = (
        'MINIMUM_CAPTURE_VIEWS', 'MAXIMUM_CAPTURE_VIEWS', 'MASK_SOURCES',
        'canonical_sha256', 'camera_extrinsic_from_metadata', 'sha256_file',
        'load_metadata', 'capture_set_provenance', 'validate_capture_set',
        'validate_manifest_integrity', 'manifest_artifact_index',
        'prepare_offline_mask_context', 'load_offline_target_mask',
        'metadata_paths_from_manifest', 'resolve_frame_artifacts',
        'confidence_capture_provenance', 'calibration_provenance',
        'capture_schema_provenance',
    )
    for name in names:
        assert getattr(MODULE, name) is getattr(INPUTS, name)


def test_default_reconstruction_preserves_one_millimetre_detail():
    assert MODULE.DEFAULT_VOXEL_LENGTH_M == pytest.approx(0.003)
    assert MODULE.DEFAULT_SDF_TRUNC_M == pytest.approx(0.015)
    assert 'robot_pose' in MODULE.REGISTRATION_MODES
    assert 'scene_pose_graph' in MODULE.REGISTRATION_MODES
    assert 'constrained_superposition' in MODULE.REGISTRATION_MODES
    assert MODULE.SUPERPOSITION_MAX_ROTATION_DEG == pytest.approx(3.0)
    assert MODULE.MASK_SOURCES == ('captured', 'offline_resegment')
    assert MODULE.GEOMETRY_SOURCES == (
        'projected_color_depth', 'native_depth')
    assert MODULE.HOLE_REPAIR_MODES == ('none', 'measured_wall')


def test_offline_mask_loader_is_hash_and_rgb_bound(tmp_path):
    root = tmp_path / 'derived'
    masks = root / 'masks'
    masks.mkdir(parents=True)
    mask_path = masks / 'view_000_mask.png'
    assert cv2.imwrite(
        str(mask_path), np.pad(
            np.full((2, 2), 255, dtype=np.uint8), 1))
    rgb_path = tmp_path / 'view_000_rgb.png'
    assert cv2.imwrite(
        str(rgb_path), np.zeros((4, 4, 3), dtype=np.uint8))
    metadata_path = tmp_path / 'view_000_metadata.yaml'
    record = {
        'frame': metadata_path.name,
        'mask_path': 'masks/view_000_mask.png',
        'mask_sha256': MODULE.sha256_file(mask_path),
        'source_rgb_sha256': MODULE.sha256_file(rgb_path),
    }
    context = {'root': root, 'by_frame': {metadata_path.name: record}}
    loaded, returned = MODULE.load_offline_target_mask(
        context, metadata_path, rgb_path, (4, 4), cv2)
    assert np.array_equal(loaded, cv2.imread(str(mask_path), 0))
    assert returned is record
    record['source_rgb_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='different RGB'):
        MODULE.load_offline_target_mask(
            context, metadata_path, rgb_path, (4, 4), cv2)


def test_scene_registration_excludes_target_and_invalid_depth():
    depth = np.full((7, 7), 400, dtype=np.uint16)
    depth[0, 0] = 0
    depth[0, 1] = 1600
    target = np.zeros((7, 7), dtype=np.uint8)
    target[3, 3] = 255
    support = MODULE.scene_registration_support_mask(
        cv2, depth, target, depth_trunc=1.5, exclusion_radius_px=1)
    assert not support[0, 0]
    assert not support[0, 1]
    assert not np.any(support[2:5, 2:5])
    assert support[6, 6]


def test_stored_base_camera_pose_is_inverted_for_open3d():
    base_camera = np.eye(4)
    base_camera[0, 3] = 0.25
    extrinsic = MODULE.camera_extrinsic_from_metadata({
        'camera_transform': {
            'available': True,
            'matrix_4x4': base_camera.tolist(),
        },
    })
    assert extrinsic[0, 3] == pytest.approx(-0.25)


def test_missing_timestamped_pose_fails_closed():
    with pytest.raises(ValueError, match='timestamped camera transform'):
        MODULE.camera_extrinsic_from_metadata({})


@pytest.mark.parametrize('count', [1, 3, 7, 8, 12, 19, 24])
def test_feature_driven_capture_count_is_accepted(count):
    paths = [Path('view_%03d_metadata.yaml' % index) for index in range(count)]
    assert MODULE.validate_capture_set({'capture_count': count}, paths) == paths


@pytest.mark.parametrize('count', [0, 25])
def test_capture_count_outside_bounded_contract_is_rejected(count):
    with pytest.raises(ValueError, match='1-24'):
        MODULE.validate_capture_set(
            {'capture_count': count}, [Path(str(index)) for index in range(count)])


@pytest.mark.parametrize('count', [1, 3, 7])
def test_legacy_partial_admission_flag_is_a_compatible_no_op(count):
    paths = [Path(str(index)) for index in range(count)]
    assert MODULE.validate_capture_set(
        {'capture_count': count}, paths,
        allow_partial_view_set=True) == paths
    report = MODULE.capture_set_provenance(
        {'capture_count': count}, allow_partial_view_set=True)
    assert report['classification'] == 'VIEW_COUNT_ELIGIBLE'
    assert report['ordinary_feature_minimum'] == 1


def test_zero_capture_set_remains_rejected_with_legacy_flag():
    with pytest.raises(ValueError, match='1-24'):
        MODULE.validate_capture_set(
            {'capture_count': 0}, [], allow_partial_view_set=True)


def test_manifest_and_frame_metadata_count_must_match():
    with pytest.raises(ValueError, match='does not match'):
        MODULE.validate_capture_set(
            {'capture_count': 14}, [Path(str(index)) for index in range(13)])


def test_registration_correction_is_bounded():
    correction = np.eye(4)
    correction[0, 3] = 0.01
    assert MODULE.correction_rejection(
        correction, fitness=0.5, inlier_rmse=0.004) == ''
    correction[0, 3] = 0.03
    assert 'translation' in MODULE.correction_rejection(
        correction, fitness=0.5, inlier_rmse=0.004)
    assert 'overlap' in MODULE.correction_rejection(
        np.eye(4), fitness=0.01, inlier_rmse=0.004)


def test_multiway_pair_selection_is_bounded_and_keeps_consecutive_views():
    poses = []
    for index in range(8):
        pose = np.eye(4)
        angle = np.radians(index * 10.0)
        pose[:3, 2] = [np.sin(angle), 0.0, np.cos(angle)]
        poses.append(pose)
    pairs = MODULE.multiway_registration_pairs(poses)
    assert all((index - 1, index) in pairs for index in range(1, 8))
    assert len(pairs) <= 7 + 8 * MODULE.MULTIWAY_MAX_PRIOR_NEIGHBORS


def test_multiway_spanning_tree_requires_connected_accepted_edges():
    edges = [
        {'source': 0, 'target': 1, 'inlier_rmse_m': 0.001},
        {'source': 1, 'target': 2, 'inlier_rmse_m': 0.002},
        {'source': 0, 'target': 2, 'inlier_rmse_m': 0.003},
    ]
    selected = MODULE.registration_spanning_tree(3, edges)
    assert len(selected) == 2
    with pytest.raises(ValueError, match='disconnected'):
        MODULE.registration_spanning_tree(4, edges)


def test_constrained_superposition_solves_global_translation_without_rotation():
    constraints = []
    expected = np.asarray([
        [0.0, 0.0, 0.0],
        [0.004, -0.002, 0.001],
        [-0.003, 0.003, -0.002],
    ])
    normals = np.eye(3)
    for source, target in ((0, 1), (1, 2), (0, 2)):
        for normal in normals:
            constraints.append({
                'source': source,
                'target': target,
                'normal_base': normal,
                'offset_m': float(np.dot(
                    normal, expected[source] - expected[target])),
                'edge_constraint_count': 3,
            })
    solved = MODULE.solve_constrained_translations(
        3, constraints, prior_sigma_m=1.0, data_sigma_m=0.0001,
        huber_delta_m=0.1)
    assert solved[0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert solved[1] == pytest.approx(expected[1], abs=1e-5)
    assert solved[2] == pytest.approx(expected[2], abs=1e-5)


def test_constrained_superposition_fails_closed_outside_pose_bound():
    constraints = [{
        'source': 0,
        'target': 1,
        'normal_base': axis,
        'offset_m': -0.03 if index == 0 else 0.0,
        'edge_constraint_count': 3,
    } for index, axis in enumerate(np.eye(3))]
    with pytest.raises(ValueError, match='exceeds 20mm'):
        MODULE.solve_constrained_translations(
            2, constraints, prior_sigma_m=1.0, data_sigma_m=0.0001,
            huber_delta_m=0.1)


def test_constrained_rigid_superposition_recovers_small_camera_tilt():
    expected_translation = np.asarray([0.0020, -0.0010, 0.0005])
    expected_rotation = np.radians(np.asarray([1.0, -0.5, 0.75]))
    camera_origins = np.asarray([
        [0.0, 0.0, -0.30],
        [0.0, 0.0, -0.30],
    ])
    constraints = []
    points = (
        np.asarray([x, y, z], dtype=float)
        for x in (-0.02, 0.02)
        for y in (-0.02, 0.02)
        for z in (-0.02, 0.02)
    )
    for point in points:
        for normal in np.eye(3):
            target_point = (
                point - expected_translation
                - np.cross(expected_rotation, point - camera_origins[1]))
            constraints.append({
                'source': 0,
                'target': 1,
                'source_point_base': point,
                'target_point_base': target_point,
                'normal_base': normal,
                'edge_constraint_count': 24,
            })

    translations, rotations = MODULE.solve_constrained_rigid_corrections(
        2, constraints, camera_origins,
        prior_sigma_m=100.0, rotation_prior_sigma_rad=100.0,
        data_sigma_m=0.0001, huber_delta_m=0.1)

    # Equal robot-pose priors select the minimum-total-correction solution,
    # distributing the relative correction instead of pinning capture zero.
    assert translations[0] == pytest.approx(
        -0.5 * expected_translation, abs=1.5e-4)
    assert translations[1] == pytest.approx(
        0.5 * expected_translation, abs=1.5e-4)
    assert rotations[0] == pytest.approx(
        -0.5 * expected_rotation, abs=2e-5)
    assert rotations[1] == pytest.approx(
        0.5 * expected_rotation, abs=2e-5)
    assert translations[1] - translations[0] == pytest.approx(
        expected_translation, abs=1.5e-4)
    assert rotations[1] - rotations[0] == pytest.approx(
        expected_rotation, abs=2e-5)


def test_constrained_rigid_superposition_rejects_excessive_tilt():
    camera_origins = np.zeros((2, 3), dtype=float)
    rotation = np.radians(np.asarray([8.0, 0.0, 0.0]))
    constraints = []
    for point in (
            np.asarray([x, y, z], dtype=float)
            for x in (-0.02, 0.02)
            for y in (-0.02, 0.02)
            for z in (-0.02, 0.02)):
        for normal in np.eye(3):
            target_point = point - np.cross(rotation, point)
            constraints.append({
                'source': 0,
                'target': 1,
                'source_point_base': point,
                'target_point_base': target_point,
                'normal_base': normal,
                'edge_constraint_count': 24,
            })

    with pytest.raises(ValueError, match='rotation exceeds 5.0deg'):
        MODULE.solve_constrained_rigid_corrections(
            2, constraints, camera_origins,
            prior_sigma_m=100.0, rotation_prior_sigma_rad=100.0,
            data_sigma_m=0.0001, huber_delta_m=0.1)


def test_rigid_camera_correction_rotates_about_camera_origin():
    origin = np.asarray([0.10, -0.20, 0.30])
    translation = np.asarray([0.002, -0.001, 0.0005])
    rotation = np.radians(np.asarray([0.5, -0.25, 0.75]))

    correction = MODULE.rigid_camera_correction_matrix(
        origin, translation, rotation)
    transformed_origin = correction @ np.append(origin, 1.0)

    assert transformed_origin[:3] == pytest.approx(origin + translation)
    assert np.degrees(MODULE.rotation_angle_rad(correction)) == pytest.approx(
        np.degrees(np.linalg.norm(rotation)), abs=1e-9)


def test_capture_zero_translation_anchor_has_no_displacement_limit():
    expected = np.asarray([0.25, -0.12, 0.08])
    constraints = [{
        'source': 0,
        'target': 1,
        'normal_base': normal,
        'offset_m': float(-np.dot(normal, expected)),
        'edge_constraint_count': 3,
    } for normal in np.eye(3)]

    translations = MODULE.solve_constrained_translations(
        2, constraints, prior_sigma_m=1.0e6,
        data_sigma_m=0.0001, huber_delta_m=1.0,
        maximum_translation_m=None)

    assert translations[0] == pytest.approx(np.zeros(3), abs=1e-8)
    assert translations[1] == pytest.approx(expected, abs=1e-6)


def test_capture_zero_rigid_anchor_is_fixed_and_rotation_is_bounded():
    expected_translation = np.asarray([0.002, -0.001, 0.0005])
    expected_rotation = np.radians(np.asarray([1.0, -0.5, 0.75]))
    origins = np.asarray([[0.0, 0.0, -0.3], [0.0, 0.0, -0.3]])
    constraints = []
    for point in (
            np.asarray([x, y, z], dtype=float)
            for x in (-0.02, 0.02)
            for y in (-0.02, 0.02)
            for z in (-0.02, 0.02)):
        for normal in np.eye(3):
            target = (
                point - expected_translation
                - np.cross(expected_rotation, point - origins[1]))
            constraints.append({
                'source': 0, 'target': 1,
                'source_point_base': point,
                'target_point_base': target,
                'normal_base': normal,
                'edge_constraint_count': 24,
            })

    translations, rotations = MODULE.solve_constrained_rigid_corrections(
        2, constraints, origins, prior_sigma_m=1.0e6,
        rotation_prior_sigma_rad=100.0, data_sigma_m=0.0001,
        huber_delta_m=0.1, maximum_translation_m=None,
        maximum_rotation_deg=MODULE.SUPERPOSITION_MAX_ROTATION_DEG,
        fixed_reference_index=0)

    assert translations[0] == pytest.approx(np.zeros(3), abs=1e-12)
    assert rotations[0] == pytest.approx(np.zeros(3), abs=1e-12)
    assert translations[1] == pytest.approx(expected_translation, abs=1.5e-4)
    assert rotations[1] == pytest.approx(expected_rotation, abs=2e-5)


def test_capture_zero_rigid_anchor_rejects_rotation_above_3_degrees():
    expected_rotation = np.radians(np.asarray([60.0, 0.0, 0.0]))
    origins = np.zeros((2, 3), dtype=float)
    constraints = []
    for point in (
            np.asarray([x, y, z], dtype=float)
            for x in (-0.02, 0.02)
            for y in (-0.02, 0.02)
            for z in (-0.02, 0.02)):
        for normal in np.eye(3):
            constraints.append({
                'source': 0, 'target': 1,
                'source_point_base': point,
                'target_point_base': point - np.cross(
                    expected_rotation, point),
                'normal_base': normal,
                'edge_constraint_count': 24,
            })

    with pytest.raises(ValueError, match='rotation exceeds 3.0deg'):
        MODULE.solve_constrained_rigid_corrections(
            2, constraints, origins, prior_sigma_m=1.0e6,
            rotation_prior_sigma_rad=100.0, data_sigma_m=0.0001,
            huber_delta_m=0.1, maximum_translation_m=None,
            maximum_rotation_deg=MODULE.SUPERPOSITION_MAX_ROTATION_DEG,
            fixed_reference_index=0)


def test_constrained_superposition_single_capture_is_fixed_noop():
    pose = np.eye(4)
    pose[:3, 3] = [0.4, -0.1, 0.2]

    refined, diagnostics = MODULE.constrained_superposition_camera_poses(
        [{'nominal_base_camera': pose}], o3d=None)

    assert len(refined) == 1
    assert refined[0] == pytest.approx(pose)
    assert diagnostics['single_capture_noop'] is True
    assert diagnostics['candidate_pair_count'] == 0
    assert diagnostics['accepted_pair_count'] == 0
    assert diagnostics['spanning_tree_pair_count'] == 0
    assert diagnostics['pose_corrections'] == [{
        'frame_index': 0,
        'translation_correction_m': 0.0,
        'translation_vector_m': [0.0, 0.0, 0.0],
        'rotation_correction_deg': 0.0,
        'rotation_vector_rad': [0.0, 0.0, 0.0],
        'fixed_reference': True,
    }]


def test_constrained_superposition_rejects_zero_captures():
    with pytest.raises(ValueError, match='at least one view'):
        MODULE.constrained_superposition_camera_poses([], o3d=None)


def test_cross_capture_consensus_allows_only_one_vote_per_view():
    surfaces = [
        {
            'points': np.asarray([
                [0.0010, 0.0010, 0.0010],
                [0.0029, 0.0029, 0.0029],
            ]),
            'normals': np.asarray([[0.0, 0.0, 1.0]] * 2),
        },
        {
            'points': np.asarray([[0.0031, 0.0029, 0.0030]]),
            'normals': np.asarray([[0.0, 0.0, 1.0]]),
        },
        {
            'points': np.asarray([[0.0027, 0.0030, 0.0028]]),
            'normals': np.asarray([[0.0, 0.0, 1.0]]),
        },
    ]

    points, normals, support, diagnostics = (
        MODULE.robust_cross_capture_consensus(
            surfaces, voxel_length_m=0.010))

    assert points.shape == (1, 3)
    assert points[0] == pytest.approx(
        np.mean([
            [0.0029, 0.0029, 0.0029],
            [0.0031, 0.0029, 0.0030],
            [0.0027, 0.0030, 0.0028],
        ], axis=0))
    assert normals[0] == pytest.approx([0.0, 0.0, 1.0])
    assert support.tolist() == [3]
    assert diagnostics['same_capture_points_averaged_together'] is False


def test_cross_capture_consensus_rejects_positional_outlier_before_mean():
    surfaces = [
        {
            'points': np.asarray([[x, 0.0010, 0.0010]]),
            'normals': np.asarray([[0.0, 0.0, 1.0]]),
        }
        for x in (0.0010, 0.0011, 0.0009, 0.0090)
    ]

    points, _normals, support, diagnostics = (
        MODULE.robust_cross_capture_consensus(
            surfaces, voxel_length_m=0.010))

    assert points[0, 0] == pytest.approx(0.0010, abs=1e-9)
    assert support.tolist() == [3]
    assert diagnostics['maximum_capture_support'] == 3


def test_cross_capture_consensus_excludes_single_view_geometry():
    surfaces = [
        {
            'points': np.asarray([[0.001, 0.001, 0.001]]),
            'normals': np.asarray([[0.0, 0.0, 1.0]]),
        },
        {
            'points': np.asarray([[0.021, 0.001, 0.001]]),
            'normals': np.asarray([[0.0, 0.0, 1.0]]),
        },
    ]

    with pytest.raises(ValueError, match='no multi-view correspondences'):
        MODULE.robust_cross_capture_consensus(
            surfaces, voxel_length_m=0.010)


def test_cross_capture_consensus_rejects_malformed_surface():
    with pytest.raises(ValueError, match='malformed'):
        MODULE.robust_cross_capture_consensus([{
            'points': np.asarray([[0.0, 0.0]]),
            'normals': np.asarray([[0.0, 0.0, 1.0]]),
        }], voxel_length_m=0.003)


def test_manifest_integrity_rejects_changed_artifact(tmp_path):
    artifact = tmp_path / 'frame.bin'
    artifact.write_bytes(b'original')
    unsigned = {
        'capture_count': 13,
        'files': [{
            'path': 'frame.bin',
            'bytes': len(b'original'),
            'sha256': hashlib.sha256(b'original').hexdigest(),
        }],
    }
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode('utf-8')
    manifest = dict(unsigned)
    manifest['manifest_sha256'] = hashlib.sha256(encoded).hexdigest()
    assert MODULE.validate_manifest_integrity(tmp_path, manifest)
    artifact.write_bytes(b'changed')
    with pytest.raises(ValueError, match='size changed'):
        MODULE.validate_manifest_integrity(tmp_path, manifest)


def test_manifest_rejects_escaping_and_duplicate_artifacts(tmp_path):
    outside = tmp_path.parent / 'outside.bin'
    outside.write_bytes(b'outside')
    unsigned = {
        'capture_count': 8,
        'files': [{
            'path': '../outside.bin',
            'bytes': outside.stat().st_size,
            'sha256': MODULE.sha256_file(outside),
        }],
    }
    manifest = dict(unsigned, manifest_sha256=MODULE.canonical_sha256(unsigned))
    with pytest.raises(ValueError, match='escapes'):
        MODULE.validate_manifest_integrity(tmp_path, manifest)


def test_frame_artifacts_are_resolved_from_manifest_not_absolute_metadata(tmp_path):
    frames = tmp_path / 'frames'
    frames.mkdir()
    metadata_path = frames / 'view_000_metadata.yaml'
    paths = {}
    records = []
    for suffix in ('rgb.png', 'depth.png', 'mask.png'):
        path = frames / ('view_000_' + suffix)
        path.write_bytes(suffix.encode('ascii'))
        relative = path.relative_to(tmp_path).as_posix()
        paths[suffix] = path
        records.append({
            'path': relative, 'bytes': path.stat().st_size,
            'sha256': MODULE.sha256_file(path),
        })
    manifest = {'files': records}
    metadata = {
        'rgb_file_path': '/old/machine/view_000_rgb.png',
        'depth_png_file_path': '/old/machine/view_000_depth.png',
        'mask_file_path': '/old/machine/view_000_mask.png',
    }
    resolved = MODULE.resolve_frame_artifacts(
        tmp_path, metadata_path, metadata, manifest)
    assert resolved == {
        'rgb': paths['rgb.png'], 'depth': paths['depth.png'],
        'mask': paths['mask.png']}
    metadata['rgb_file_path'] = '/tmp/unrelated.png'
    with pytest.raises(ValueError, match='manifest-listed rgb'):
        MODULE.resolve_frame_artifacts(
            tmp_path, metadata_path, metadata, manifest)


def _schema_v2_artifacts(tmp_path):
    frames = tmp_path / 'frames'
    frames.mkdir(exist_ok=True)
    metadata_path = frames / 'view_000_metadata.yaml'
    suffixes = {
        'rgb_file_path': 'rgb.png',
        'depth_png_file_path': 'depth.png',
        'mask_file_path': 'mask.png',
        'native_depth_file_path': 'native_depth.npy',
        'native_depth_png_file_path': 'native_depth.png',
        'confidence_file_path': 'confidence.png',
        'target_depth_png_file_path': 'target_depth.png',
        'target_support_mask_file_path': 'target_support_mask.png',
    }
    metadata = {'capture_schema_version': 2}
    records = []
    for key, suffix in suffixes.items():
        path = frames / ('view_000_' + suffix)
        if suffix == 'native_depth.npy':
            with path.open('wb') as stream:
                np.save(stream, np.full((4, 5), 400, dtype=np.uint16))
        elif suffix == 'native_depth.png':
            assert cv2.imwrite(
                str(path), np.full((4, 5), 400, dtype=np.uint16))
        elif suffix == 'confidence.png':
            assert cv2.imwrite(str(path), np.full((4, 5), 8, dtype=np.uint8))
        elif suffix == 'target_depth.png':
            assert cv2.imwrite(
                str(path), np.full((5, 6), 400, dtype=np.uint16))
        elif suffix == 'target_support_mask.png':
            assert cv2.imwrite(
                str(path), np.full((5, 6), 255, dtype=np.uint8))
        else:
            path.write_bytes(suffix.encode('ascii'))
        metadata[key] = '/old/machine/' + path.name
        records.append({
            'path': path.relative_to(tmp_path).as_posix(),
            'bytes': path.stat().st_size,
            'sha256': MODULE.sha256_file(path),
        })
    manifest = {
        'capture_schema_version': 2,
        'confidence_policy': {'minimum_grade': 8},
        'files': records,
    }
    return metadata_path, metadata, manifest


def test_schema_v2_artifacts_and_confidence_provenance_are_required(tmp_path):
    metadata_path, metadata, manifest = _schema_v2_artifacts(tmp_path)
    artifacts = MODULE.resolve_frame_artifacts(
        tmp_path, metadata_path, metadata, manifest)
    metadata.update({
        'synchronization': {
            'mask_rgb_exact': True,
            'rgb_native_depth_delta_sec': 0.012,
            'native_depth_confidence_delta_sec': 0.001,
            'maximum_rgb_native_depth_delta_sec': 0.04,
            'maximum_native_depth_confidence_delta_sec': 0.005,
        },
        'confidence_quality': {
            'confidence_threshold': 8,
            'confident_fraction': 0.92,
            'projected_output_points': 30,
        },
        'target_valid': True,
        'synchronized_target_3d': {'valid': True},
    })
    provenance = MODULE.confidence_capture_provenance(
        metadata, manifest, artifacts, cv2)
    assert provenance['confidence_qualified'] is True
    assert provenance['minimum_confidence_grade'] == 8
    metadata['synchronization']['mask_rgb_exact'] = False
    with pytest.raises(ValueError, match='exactly RGB-correlated'):
        MODULE.confidence_capture_provenance(
            metadata, manifest, artifacts, cv2)


def test_capture_schema_controls_certification():
    certified = {
        'classification': 'CERTIFIED',
        'calibration_sha256': 'a' * 64,
        'reason': 'calibration bound',
    }
    manifest = {
        'capture_schema_version': 2,
        'confidence_policy': {'minimum_grade': 8},
    }
    result = MODULE.capture_schema_provenance(
        manifest, [{'capture_schema_version': 2}] * 8, certified)
    assert result['classification'] == 'CERTIFIED'
    assert result['confidence_qualified'] is True
    with pytest.raises(ValueError, match='predates confidence-qualified'):
        MODULE.capture_schema_provenance(
            {}, [{'capture_schema_version': 1}] * 8, certified)
    diagnostic = MODULE.capture_schema_provenance(
        {}, [{'capture_schema_version': 1}] * 8, certified,
        allow_historical=True)
    assert diagnostic['classification'] == 'DIAGNOSTIC_ONLY'
    assert diagnostic['confidence_qualified'] is False


def test_missing_calibration_requires_explicit_diagnostic_mode():
    frames = [{'calibration_sha256': ''} for _ in range(8)]
    with pytest.raises(ValueError, match='no calibration identity'):
        MODULE.calibration_provenance(
            {'calibration_sha256': ''}, frames, allow_missing=False)
    assert MODULE.calibration_provenance(
        {'calibration_sha256': ''}, frames,
        allow_missing=True)['classification'] == 'DIAGNOSTIC_ONLY'


def test_calibration_identity_must_match_every_frame():
    identity = 'a' * 64
    frames = [{'calibration_sha256': identity} for _ in range(8)]
    assert MODULE.calibration_provenance(
        {'calibration_sha256': identity}, frames)['classification'] == 'CERTIFIED'
    frames[-1]['calibration_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='one calibration identity'):
        MODULE.calibration_provenance(
            {'calibration_sha256': identity}, frames)


def test_quality_thresholds_are_explicit():
    assert MODULE.quality_classification(0.004, 0.96) == 'PASS'
    assert MODULE.quality_classification(0.007, 0.96) == 'WARN'
    assert MODULE.quality_classification(0.004, 0.90) == 'WARN'
    assert MODULE.quality_classification(0.011, 1.0) == 'FAIL'
    assert MODULE.quality_classification(0.004, 0.79) == 'FAIL'
    assert MODULE.dimension_classification(0.004) == 'GOOD'
    assert MODULE.dimension_classification(0.007) == 'WARN'
    assert MODULE.dimension_classification(0.011) == 'POOR'


def test_connected_target_policy_removes_only_tiny_fragments():
    report = MODULE.target_component_policy(
        [900, 30, 20], [0.009, 0.0003, 0.0001])
    assert report['retained_component_indices'] == [0]
    assert report['removed_fragment_component_count'] == 2
    assert report['retained_triangle_count'] == 900
    assert report['connectivity_valid'] is True
    assert report['decision'] == 'SINGLE_CONNECTED_TARGET'


def test_connected_target_policy_retains_and_rejects_substantial_split():
    report = MODULE.target_component_policy(
        [800, 160, 10], [0.008, 0.0016, 0.0001])
    assert report['retained_component_indices'] == [0, 1]
    assert report['removed_fragment_component_count'] == 1
    assert report['connectivity_valid'] is False
    assert report['decision'] == 'MULTIPLE_SUBSTANTIAL_COMPONENTS'


def test_connected_target_policy_keeps_component_at_five_percent_boundary():
    report = MODULE.target_component_policy(
        [1000, 50], [0.010, 0.0005])
    assert report['retained_component_indices'] == [0, 1]
    assert report['connectivity_valid'] is False


def test_connected_target_policy_retains_small_measured_component():
    report = MODULE.target_component_policy(
        [1000, 2, 1], [0.010, 0.00001, 0.000001],
        measured_supported_indices=[1])
    assert report['retained_component_indices'] == [0, 1]
    assert report['retained_only_by_measured_support_indices'] == [1]
    assert report['removed_fragment_component_count'] == 1
    assert report['connectivity_valid'] is False
    assert report['decision'] == 'MULTIPLE_MEASURED_TARGET_COMPONENTS'


def test_connected_target_policy_rejects_invalid_measured_component_index():
    with pytest.raises(ValueError, match='indices'):
        MODULE.target_component_policy(
            [1000, 2], [0.010, 0.00001],
            measured_supported_indices=[2])


def test_component_cleanup_keeps_two_view_supported_surface_and_removes_noise():
    o3d = pytest.importorskip('open3d')
    vertices = np.asarray([
        [0.000, 0.000, 0.000], [0.020, 0.000, 0.000],
        [0.020, 0.020, 0.000], [0.000, 0.020, 0.000],
        [0.030, 0.000, 0.000], [0.031, 0.000, 0.000],
        [0.030, 0.001, 0.000],
        [0.060, 0.000, 0.000], [0.061, 0.000, 0.000],
        [0.060, 0.001, 0.000],
    ], dtype=float)
    triangles = np.asarray([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [7, 8, 9]],
        dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    views = [
        np.asarray([[0.0302, 0.0002, 0.0],
                    [0.0304, 0.0003, 0.0]], dtype=float),
        np.asarray([[0.0303, 0.0002, 0.0],
                    [0.0302, 0.0004, 0.0]], dtype=float),
    ]

    filtered, report = MODULE.filter_target_mesh_components(
        mesh, o3d=o3d, measured_support_views=views)

    assert len(filtered.triangles) == 3
    assert len(report['retained_only_by_measured_support_indices']) == 1
    assert report['removed_fragment_component_count'] == 1
    support = report['measured_component_support']
    assert support['available']
    assert len(support['qualified_component_indices']) == 1
    retained = support['components'][
        support['qualified_component_indices'][0]]
    assert retained['total_support_points'] == 4
    assert retained['supporting_view_count'] == 2


def test_registration_selection_uses_pre_cleanup_component_coherence():
    robot = _selection_report(0.006, 0.004, component=1.0)
    robot['raw_mesh_metrics'] = {
        **robot['mesh_metrics'],
        'dominant_component_triangle_ratio': 0.70,
    }
    gicp = _selection_report(0.004, 0.004, component=1.0)
    gicp['raw_mesh_metrics'] = {
        **gicp['mesh_metrics'],
        'dominant_component_triangle_ratio': 0.90,
    }
    selected, comparison = MODULE.select_registration_report(robot, gicp)
    assert selected == 'robot_pose'
    assert comparison['component_coherence_acceptable'] is False


def test_raw_mesh_path_is_adjacent_and_does_not_replace_cleaned_output():
    output = Path('/tmp/scan/target_mesh.ply')
    assert MODULE.raw_mesh_output_path(output) == \
        Path('/tmp/scan/target_mesh.raw.ply')


def test_known_target_dimensions_reject_aligned_depth_background():
    depth = np.asarray([
        [0, 400, 405],
        [395, 440, 500],
    ], dtype=np.uint16)
    mask = np.asarray([
        [0, 255, 255],
        [255, 255, 255],
    ], dtype=np.uint8)
    supported, report = MODULE.target_depth_support_mask(
        depth, mask, 0.400, np.asarray([0.04, 0.04, 0.04]))
    assert supported.tolist() == [[False, True, True], [True, False, False]]
    assert report['masked_points_before'] == 5
    assert report['masked_points_after'] == 3


def test_unknown_target_size_keeps_binary_mask_without_dimension_claim():
    mask = np.asarray([[0, 255]], dtype=np.uint8)
    supported, report = MODULE.target_depth_support_mask(
        np.asarray([[400, 900]], dtype=np.uint16), mask, 0.4, None)
    assert supported.tolist() == [[False, True]]
    assert report is None


def _selection_report(residual, dimension_error, component=0.98,
                      quality='PASS'):
    return {
        'structural_quality': quality,
        'mesh_metrics': {
            'dominant_component_triangle_ratio': component,
            'point_to_mesh_residual': {'median_m': residual},
            'dimension_check': {'mean_absolute_error_m': dimension_error},
        },
    }


def test_auto_selection_requires_material_safe_gicp_improvement():
    robot = _selection_report(0.006, 0.004)
    gicp = _selection_report(0.005, 0.005)
    assert MODULE.select_registration_report(robot, gicp)[0] == 'bounded_gicp'
    weak = _selection_report(0.0057, 0.003)
    assert MODULE.select_registration_report(robot, weak)[0] == 'robot_pose'
    fragmented = _selection_report(0.004, 0.003, component=0.80)
    assert MODULE.select_registration_report(robot, fragmented)[0] == 'robot_pose'


def test_auto_selection_can_choose_coherent_bounded_multiway_refinement():
    robot = _selection_report(0.0040, 0.020, component=0.40, quality='FAIL')
    sequential = _selection_report(
        0.0033, 0.018, component=0.62, quality='FAIL')
    multiway = _selection_report(
        0.0025, 0.016, component=0.90, quality='FAIL')
    selected, comparison = MODULE.select_registration_reports({
        'robot_pose': robot,
        'bounded_gicp': sequential,
        'multiway_gicp': multiway,
    })
    assert selected == 'multiway_gicp'
    assert comparison['candidate_assessments'][
        'multiway_gicp']['eligible'] is True
    assert comparison['candidate_assessments'][
        'bounded_gicp']['eligible'] is False


def test_auto_selection_can_choose_scene_pose_graph_refinement():
    robot = _selection_report(0.0040, 0.010, component=0.70, quality='WARN')
    target_only = _selection_report(
        0.0038, 0.009, component=0.78, quality='WARN')
    scene = _selection_report(
        0.0020, 0.008, component=0.96, quality='WARN')
    selected, comparison = MODULE.select_registration_reports({
        'robot_pose': robot,
        'multiway_gicp': target_only,
        'scene_pose_graph': scene,
    })
    assert selected == 'scene_pose_graph'
    assert comparison['candidate_assessments'][
        'scene_pose_graph']['eligible'] is True


def test_auto_selection_retains_robot_pose_when_refinement_is_incoherent():
    robot = _selection_report(0.0040, 0.005, component=0.85, quality='WARN')
    refinement = _selection_report(
        0.0020, 0.004, component=0.79, quality='WARN')
    selected, comparison = MODULE.select_registration_reports({
        'robot_pose': robot,
        'multiway_gicp': refinement,
    })
    assert selected == 'robot_pose'
    assert comparison['candidate_assessments'][
        'multiway_gicp']['eligible'] is False
