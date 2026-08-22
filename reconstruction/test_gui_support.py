import json

import pytest

from reconstruction.gui_support import (
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
    assert command[command.index('--voxel-length') + 1] == '0.001'
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


def test_viewer_report_cannot_escape_dataset_root(tmp_path):
    root, scan = project(tmp_path)
    report = scan / 'reconstruction' / 'validation' / 'target_mesh.ply.quality.json'
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({}), encoding='utf-8')
    assert '--report' in viewer_command(root, report)
    assert '--show-input' in viewer_command(root, report, show_input=True)
    outside = root / 'report.json'
    outside.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='escapes'):
        viewer_command(root, outside)


def test_quality_summary_calls_out_provisional_dimension_and_visual_review():
    summary = quality_summary({
        'structural_quality': 'WARN',
        'overall_quality': 'FAIL',
        'dimension_quality': 'POOR',
        'provenance': {'classification': 'DIAGNOSTIC_ONLY'},
        'registration_mode': 'robot_pose', 'integrated_views': 15,
        'vertex_count': 100, 'triangle_count': 200,
        'configuration': {'voxel_length_m': 0.0005},
        'registration_summary': {'median_rmse_m': 0.006},
        'mesh_metrics': {
            'dominant_component_triangle_ratio': 0.9,
            'dimension_check': {
                'observed_obb_m': [0.041, 0.04, 0.039],
                'maximum_absolute_error_m': 0.001,
            },
        },
    })
    assert 'FAIL / DIAGNOSTIC_ONLY' in summary
    assert 'Provisional cube OBB' in summary
    assert 'TSDF mesh voxel: 0.50 mm' in summary
    assert 'Visual review is required' in summary
