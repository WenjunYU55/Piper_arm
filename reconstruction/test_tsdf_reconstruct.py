import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')


PATH = Path(__file__).with_name('tsdf_reconstruct.py')
SPEC = importlib.util.spec_from_file_location('tsdf_reconstruct', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_default_reconstruction_preserves_one_millimetre_detail():
    assert MODULE.DEFAULT_VOXEL_LENGTH_M == pytest.approx(0.001)
    assert MODULE.DEFAULT_SDF_TRUNC_M == pytest.approx(0.005)
    assert 'robot_pose' in MODULE.REGISTRATION_MODES
    assert 'scene_pose_graph' in MODULE.REGISTRATION_MODES
    assert MODULE.MASK_SOURCES == ('captured', 'offline_resegment')


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


@pytest.mark.parametrize('count', [8, 12, 19, 24])
def test_feature_driven_capture_count_is_accepted(count):
    paths = [Path('view_%03d_metadata.yaml' % index) for index in range(count)]
    assert MODULE.validate_capture_set({'capture_count': count}, paths) == paths


@pytest.mark.parametrize('count', [0, 7, 25])
def test_capture_count_outside_bounded_contract_is_rejected(count):
    with pytest.raises(ValueError, match='8-24'):
        MODULE.validate_capture_set(
            {'capture_count': count}, [Path(str(index)) for index in range(count)])


@pytest.mark.parametrize('count', [1, 3, 7])
def test_partial_capture_count_requires_explicit_admission(count):
    paths = [Path(str(index)) for index in range(count)]
    assert MODULE.validate_capture_set(
        {'capture_count': count}, paths,
        allow_partial_view_set=True) == paths
    report = MODULE.capture_set_provenance(
        {'capture_count': count}, allow_partial_view_set=True)
    assert report['classification'] == 'PARTIAL_VIEW_SET'
    assert report['ordinary_feature_minimum'] == 8


def test_zero_capture_set_remains_rejected_when_partial_is_allowed():
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
