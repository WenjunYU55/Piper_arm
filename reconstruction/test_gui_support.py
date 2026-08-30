import json

import pytest

from reconstruction.gui_support import (
    constrained_output_paths, existing_reconstruction_outputs,
    list_scan_datasets, quality_summary, reconstruction_command,
    validated_dataset_path, viewer_command)


def project(tmp_path):
    root = tmp_path / 'project'
    scan = root / 'datasets' / 'active_scan' / 'scan_001'
    scan.mkdir(parents=True)
    (scan / 'manifest.json').write_text('{}', encoding='utf-8')
    python = root / 'reconstruction' / '.venv' / 'bin' / 'python'
    python.parent.mkdir(parents=True)
    python.write_text('', encoding='utf-8')
    return root, scan


def test_dataset_selection_cannot_escape_scan_root(tmp_path):
    root, scan = project(tmp_path)
    assert list_scan_datasets(root) == [scan]
    assert validated_dataset_path(root, 'scan_001') == scan
    with pytest.raises(ValueError, match='escapes'):
        validated_dataset_path(root, '../../outside')


def test_gui_command_is_argument_only_and_diagnostic(tmp_path):
    root, scan = project(tmp_path)
    command, output = reconstruction_command(
        root, scan.name, (40.0, 40.0, 40.0))
    assert '--allow-missing-calibration-id' in command
    assert '--allow-partial-view-set' in command
    assert command[command.index('--registration-mode') + 1] == 'auto'
    assert command[command.index('--voxel-length') + 1] == '0.003'
    assert output == scan / 'reconstruction' / 'validation' / 'target_mesh.ply'
    assert all('\n' not in argument for argument in command)


def test_gui_command_accepts_bounded_mesh_detail(tmp_path):
    root, scan = project(tmp_path)
    command, _output = reconstruction_command(
        root, scan.name, (35.0, 35.0, 35.0), voxel_length_mm=0.5)
    assert command[command.index('--voxel-length') + 1] == '0.0005'

    for invalid in (0.49, 3.01, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='voxel size'):
            reconstruction_command(
                root, scan.name, (35.0, 35.0, 35.0),
                voxel_length_mm=invalid)


def test_gui_command_accepts_scene_pose_graph_mode(tmp_path):
    root, scan = project(tmp_path)
    command, _output = reconstruction_command(
        root, scan.name, (35.0, 35.0, 35.0),
        registration_mode='scene_pose_graph', voxel_length_mm=0.8)
    assert command[command.index('--registration-mode') + 1] == \
        'scene_pose_graph'
    assert command[command.index('--voxel-length') + 1] == '0.0008'


def test_gui_command_selects_offline_resegmentation_without_changing_registration(tmp_path):
    root, scan = project(tmp_path)
    command, _output = reconstruction_command(
        root, scan.name, (35.0, 35.0, 35.0),
        registration_mode='robot_pose', mask_source='offline_resegment')
    assert command[command.index('--mask-source') + 1] == \
        'offline_resegment'
    assert command[command.index('--registration-mode') + 1] == 'robot_pose'
    with pytest.raises(ValueError, match='mask source'):
        reconstruction_command(
            root, scan.name, (35.0, 35.0, 35.0),
            mask_source='untrusted')


def test_viewer_report_cannot_escape_dataset_root(tmp_path):
    root, scan = project(tmp_path)
    report = scan / 'reconstruction' / 'validation' / 'target_mesh.ply.quality.json'
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({}), encoding='utf-8')
    assert '--report' in viewer_command(root, report)
    assert '--show-input' in viewer_command(root, report, show_input=True)
    raw_command = viewer_command(root, report, mesh_variant='raw')
    assert raw_command[raw_command.index('--mesh') + 1] == 'raw'
    measured_command = viewer_command(root, report, mesh_variant='measured')
    assert measured_command[measured_command.index('--mesh') + 1] == 'measured'
    consensus_command = viewer_command(root, report, mesh_variant='consensus')
    assert consensus_command[consensus_command.index('--mesh') + 1] == 'consensus'
    superposition_command = viewer_command(
        root, report, mesh_variant='superposition')
    assert superposition_command[
        superposition_command.index('--mesh') + 1] == 'superposition'
    textured_command = viewer_command(root, report, mesh_variant='textured')
    assert textured_command[textured_command.index('--mesh') + 1] == \
        'textured'
    with pytest.raises(ValueError, match='mesh variant'):
        viewer_command(root, report, mesh_variant='unknown')
    outside = root / 'report.json'
    outside.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='escapes'):
        viewer_command(root, outside)


def test_existing_failed_outputs_remain_inspectable(tmp_path):
    root, scan = project(tmp_path)
    output = scan / 'reconstruction' / 'validation' / 'target_mesh.ply'
    raw = output.with_name('target_mesh.raw.ply')
    measured = output.with_name('target_mesh.measured_points.ply')
    consensus = output.with_name('target_mesh.consensus_points.ply')
    textured = output.with_name('target_mesh.textured.obj')
    report = output.with_suffix(output.suffix + '.quality.json')
    report.parent.mkdir(parents=True)
    output.write_text('cleaned', encoding='utf-8')
    raw.write_text('raw', encoding='utf-8')
    measured.write_text('measured', encoding='utf-8')
    consensus.write_text('consensus', encoding='utf-8')
    textured.write_text('textured', encoding='utf-8')
    report.write_text(json.dumps({
        'overall_quality': 'FAIL',
        'registration_mode': 'constrained_superposition',
        'mesh_path': str(output),
        'raw_mesh_path': str(raw),
        'measured_cloud_path': str(measured),
        'consensus_cloud_path': str(consensus),
        'textured_mesh_path': str(textured),
    }), encoding='utf-8')

    saved = existing_reconstruction_outputs(root, scan.name)

    assert saved['report']['overall_quality'] == 'FAIL'
    assert saved['report_path'] == report
    assert saved['output_path'] == output
    assert saved['raw_output_path'] == raw
    assert saved['measured_cloud_path'] == measured
    assert saved['consensus_cloud_path'] == consensus
    assert saved['superposition_cloud_path'] == measured
    assert saved['textured_mesh_path'] == textured


def test_auto_report_exposes_constrained_candidate_outputs():
    report = {
        'registration_mode': 'robot_pose',
        'measured_cloud_path': '/tmp/robot.ply',
        'candidate_reports': {
            'constrained_superposition': {
                'measured_cloud_path': '/tmp/superposition.ply',
                'consensus_cloud_path': '/tmp/consensus.ply',
                'textured_mesh_path': '/tmp/textured.obj',
            },
        },
    }

    assert constrained_output_paths(report) == {
        'superposition_cloud_path': '/tmp/superposition.ply',
        'consensus_cloud_path': '/tmp/consensus.ply',
        'textured_mesh_path': '/tmp/textured.obj',
    }


def test_auto_summary_reports_constrained_textured_candidate():
    report = {
        'structural_quality': 'FAIL',
        'overall_quality': 'FAIL',
        'provenance': {'classification': 'QUALIFIED'},
        'registration_mode': 'robot_pose',
        'integrated_views': 3,
        'vertex_count': 20,
        'triangle_count': 30,
        'configuration': {'voxel_length_m': 0.003},
        'registration_summary': {'median_rmse_m': 0.001},
        'mesh_metrics': {
            'connected_component_count': 1,
            'dominant_component_triangle_ratio': 1.0,
            'dimension_check': None,
        },
        'component_filter': {
            'decision': 'SINGLE_CONNECTED_TARGET',
            'removed_fragment_component_count': 0,
        },
        'candidate_reports': {
            'constrained_superposition': {
                'textured_mesh_path': '/tmp/textured.obj',
                'texture_baking': {
                    'textured_triangle_count': 27,
                    'triangle_count': 30,
                    'atlas_width_px': 96,
                    'atlas_height_px': 80,
                },
            },
        },
    }

    summary = quality_summary(report)

    assert 'Textured mesh (unknown): 27/30 triangles' in summary


def test_existing_output_paths_cannot_escape_selected_dataset(tmp_path):
    root, scan = project(tmp_path)
    output = scan / 'reconstruction' / 'validation' / 'target_mesh.ply'
    report = output.with_suffix(output.suffix + '.quality.json')
    outside = root / 'outside.ply'
    report.parent.mkdir(parents=True)
    outside.write_text('outside', encoding='utf-8')
    report.write_text(json.dumps({
        'mesh_path': str(outside),
    }), encoding='utf-8')

    with pytest.raises(ValueError, match='escapes'):
        existing_reconstruction_outputs(root, scan.name)


def test_quality_summary_calls_out_provisional_dimension_and_visual_review():
    summary = quality_summary({
        'structural_quality': 'WARN',
        'overall_quality': 'FAIL',
        'dimension_quality': 'POOR',
        'provenance': {'classification': 'DIAGNOSTIC_ONLY'},
        'registration_mode': 'robot_pose', 'integrated_views': 15,
        'vertex_count': 100, 'triangle_count': 200,
        'raw_mesh_path': '/tmp/target_mesh.raw.ply',
        'measured_cloud_path': '/tmp/target_mesh.measured_points.ply',
        'measured_point_count': 11947,
        'consensus_cloud_path': '/tmp/target_mesh.consensus_points.ply',
        'consensus_point_count': 3210,
        'cross_capture_consensus': {
            'median_capture_support': 3.0,
            'median_maximum_cross_capture_spread_m': 0.0008,
        },
        'configuration': {'voxel_length_m': 0.0005},
        'registration_summary': {'median_rmse_m': 0.006},
        'mesh_metrics': {
            'connected_component_count': 1,
            'dominant_component_triangle_ratio': 0.9,
            'dimension_check': {
                'observed_obb_m': [0.041, 0.04, 0.039],
                'maximum_absolute_error_m': 0.001,
            },
        },
        'raw_mesh_metrics': {
            'connected_component_count': 12,
            'dominant_component_triangle_ratio': 0.8,
            'dimension_check': {
                'observed_obb_m': [0.05, 0.048, 0.041],
                'maximum_absolute_error_m': 0.015,
            },
        },
        'component_filter': {
            'decision': 'SINGLE_CONNECTED_TARGET',
            'removed_fragment_component_count': 11,
        },
    })
    assert 'FAIL / DIAGNOSTIC_ONLY' in summary
    assert 'Provisional cube OBB' in summary
    assert 'TSDF mesh voxel: 0.50 mm' in summary
    assert 'Target mask: captured' in summary
    assert '12 raw components -> 1 cleaned components' in summary
    assert '11,947 accepted depth points' not in summary
    assert '11947 accepted depth points' in summary
    assert 'removed 11 tiny fragments' in summary
    assert 'Cleaned mesh OBB' in summary
    assert 'Visual review is required' in summary


def test_quality_summary_explains_single_capture_consensus_unavailable():
    summary = quality_summary({
        'structural_quality': 'FAIL',
        'overall_quality': 'FAIL',
        'provenance': {'classification': 'QUALIFIED'},
        'registration_mode': 'constrained_superposition',
        'integrated_views': 1,
        'vertex_count': 8,
        'triangle_count': 4,
        'measured_cloud_path': '/tmp/measured.ply',
        'measured_point_count': 3415,
        'consensus_cloud_path': '',
        'cross_capture_consensus': {
            'available': False,
            'reason': (
                'cross-capture consensus requires at least two '
                'distinct captures'),
        },
        'configuration': {'voxel_length_m': 0.003},
        'registration_summary': {'median_rmse_m': float('inf')},
        'mesh_metrics': {
            'connected_component_count': 1,
            'dominant_component_triangle_ratio': 1.0,
            'dimension_check': None,
        },
        'raw_mesh_metrics': {
            'connected_component_count': 1,
            'dominant_component_triangle_ratio': 1.0,
            'dimension_check': None,
        },
        'component_filter': {
            'decision': 'SINGLE_CONNECTED_TARGET',
            'removed_fragment_component_count': 0,
        },
    })

    assert (
        'Cross-view consensus unavailable: cross-capture consensus '
        'requires at least two distinct captures' in summary)
