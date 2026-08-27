"""Pure command construction and reporting helpers for the Tk GUI."""

import json
import math
from pathlib import Path
import subprocess


MINIMUM_GUI_VOXEL_LENGTH_MM = 0.5
MAXIMUM_GUI_VOXEL_LENGTH_MM = 3.0
RECONSTRUCTION_MASK_SOURCES = ('captured', 'offline_resegment')


def dataset_root(project_root):
    return Path(project_root).resolve() / 'datasets' / 'active_scan'


def list_scan_datasets(project_root):
    root = dataset_root(project_root)
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.glob('scan_*')
         if path.is_dir() and (path / 'manifest.json').is_file()),
        key=lambda path: path.name, reverse=True)


def validated_dataset_path(project_root, selection):
    root = dataset_root(project_root)
    selected = Path(selection)
    if not selected.is_absolute():
        selected = root / selected
    selected = selected.resolve()
    try:
        selected.relative_to(root)
    except ValueError as exc:
        raise ValueError('dataset selection escapes datasets/active_scan') from exc
    if not selected.is_dir() or not (selected / 'manifest.json').is_file():
        raise ValueError('selected scan dataset is missing its manifest')
    return selected


def reconstruction_python(project_root):
    return Path(project_root).resolve() / 'reconstruction' / '.venv' / 'bin' / 'python'


def existing_reconstruction_outputs(project_root, selection):
    """Return inspectable saved outputs for one selected dataset, if present."""
    dataset = validated_dataset_path(project_root, selection)
    expected_root = dataset.resolve()
    report_path = (
        dataset / 'reconstruction' / 'validation' /
        'target_mesh.ply.quality.json')
    if not report_path.is_file():
        return None
    report = load_quality_report(report_path)

    def saved_mesh(key):
        value = str(report.get(key, '')).strip()
        if not value:
            return None
        path = Path(value).resolve()
        try:
            path.relative_to(expected_root)
        except ValueError as exc:
            raise ValueError('reported mesh escapes selected dataset') from exc
        return path if path.is_file() else None

    cleaned = saved_mesh('mesh_path')
    if cleaned is None:
        return None
    return {
        'report': report,
        'report_path': report_path.resolve(),
        'output_path': cleaned,
        'raw_output_path': saved_mesh('raw_mesh_path'),
        'measured_cloud_path': saved_mesh('measured_cloud_path'),
    }


def reconstruction_command(project_root, selection, dimensions_mm,
                           registration_mode='auto', voxel_length_mm=3.0,
                           mask_source='captured'):
    root = Path(project_root).resolve()
    dataset = validated_dataset_path(root, selection)
    python = reconstruction_python(root)
    if not python.is_file():
        raise ValueError(
            'reconstruction environment is missing; run '
            'reconstruction/setup_environment.sh')
    dimensions = tuple(float(value) for value in dimensions_mm)
    if len(dimensions) != 3 or any(
            value <= 0.0 or value > 2000.0 for value in dimensions):
        raise ValueError('expected X/Y/Z must be between 0 and 2000 mm')
    voxel_mm = float(voxel_length_mm)
    if not math.isfinite(voxel_mm) or not (
            MINIMUM_GUI_VOXEL_LENGTH_MM
            <= voxel_mm <= MAXIMUM_GUI_VOXEL_LENGTH_MM):
        raise ValueError('mesh voxel size must be between 0.5 and 3.0 mm')
    source = str(mask_source)
    if source not in RECONSTRUCTION_MASK_SOURCES:
        raise ValueError(
            'mask source must be captured or offline_resegment')
    output = dataset / 'reconstruction' / 'validation' / 'target_mesh.ply'
    command = [
        str(python), str(root / 'reconstruction' / 'tsdf_reconstruct.py'),
        str(dataset), '--output', str(output), '--registration-mode',
        str(registration_mode), '--expected-dimensions-mm',
        *(format(value, '.12g') for value in dimensions),
        '--voxel-length', format(voxel_mm / 1000.0, '.12g'),
        '--mask-source', source,
        '--allow-missing-calibration-id',
        '--allow-partial-view-set',
    ]
    return command, output


def viewer_command(project_root, report_path, show_input=False,
                   mesh_variant='cleaned'):
    root = Path(project_root).resolve()
    python = reconstruction_python(root)
    report = Path(report_path).resolve()
    expected_root = dataset_root(root)
    try:
        report.relative_to(expected_root)
    except ValueError as exc:
        raise ValueError('quality report escapes datasets/active_scan') from exc
    if not python.is_file() or not report.is_file():
        raise ValueError('viewer environment or quality report is missing')
    variant = str(mesh_variant)
    if variant not in ('cleaned', 'raw', 'measured'):
        raise ValueError('mesh variant must be cleaned, raw or measured')
    command = [
        str(python), str(root / 'reconstruction' / 'view_reconstruction.py'),
        '--report', str(report), '--mesh', variant,
    ]
    if show_input:
        command.append('--show-input')
    return command


def start_reconstruction_process(command):
    """Start one offline reconstruction worker with captured diagnostics."""
    return subprocess.Popen(
        list(command), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)


def start_viewer_process(command):
    """Start one detached offline mesh viewer."""
    return subprocess.Popen(list(command), start_new_session=True)


def load_quality_report(path):
    with open(path, 'r', encoding='utf-8') as stream:
        report = json.load(stream)
    if not isinstance(report, dict):
        raise ValueError('quality report is not an object')
    return report


def quality_summary(report):
    provenance = report.get('provenance', {})
    registration = report.get('registration_summary', {})
    mesh = report.get('mesh_metrics', {})
    raw_mesh = report.get('raw_mesh_metrics', mesh)
    component_filter = report.get('component_filter', {})
    dimensions = raw_mesh.get('dimension_check') or {}
    cleaned_dimensions = mesh.get('dimension_check') or {}
    configuration = report.get('configuration', {})
    lines = [
        '%s / %s' % (
            report.get('overall_quality',
                       report.get('structural_quality', 'UNKNOWN')),
            provenance.get('classification', 'UNKNOWN')),
        'Selected mode: %s; views: %s; mesh: %s vertices / %s triangles' % (
            report.get('registration_mode', 'unknown'),
            report.get('integrated_views', '?'), report.get('vertex_count', '?'),
            report.get('triangle_count', '?')),
    ]
    if report.get('raw_mesh_path'):
        lines.append(
            'Raw + cleaned outputs: %s raw components -> %s cleaned components'
            % (
                raw_mesh.get('connected_component_count', '?'),
                mesh.get('connected_component_count', '?')))
    if report.get('measured_cloud_path'):
        lines.append(
            'Full-resolution measured cloud: %s accepted depth points'
            % report.get('measured_point_count', '?'))
    lines.extend([
        'TSDF mesh voxel: %.2f mm (smaller permits more polygons)' % (
            1000.0 * float(configuration.get(
                'voxel_length_m', float('nan')))),
        'Target mask: %s' % configuration.get(
            'semantic_mask_source', 'captured'),
        'Registration median RMSE: %.2f mm; dominant component: %.1f%%' % (
            1000.0 * float(registration.get('median_rmse_m', float('nan'))),
            100.0 * float(raw_mesh.get(
                'dominant_component_triangle_ratio', float('nan')))),
    ])
    if component_filter.get('decision'):
        lines.append(
            'Connected-target cleanup: %s; removed %s tiny fragments' % (
                component_filter['decision'],
                component_filter.get(
                    'removed_fragment_component_count', '?')))
    if dimensions:
        lines.append(
            'Provisional cube OBB: %s mm; maximum size error: %.2f mm (%s)' % (
                ' x '.join('%.2f' % (1000.0 * float(value))
                           for value in dimensions.get('observed_obb_m', [])),
                1000.0 * float(dimensions.get(
                    'maximum_absolute_error_m', float('nan'))),
                report.get('dimension_quality', 'UNKNOWN')))
    if cleaned_dimensions and cleaned_dimensions != dimensions:
        lines.append(
            'Cleaned mesh OBB: %s mm; maximum size error: %.2f mm' % (
                ' x '.join('%.2f' % (1000.0 * float(value))
                           for value in cleaned_dimensions.get(
                               'observed_obb_m', [])),
                1000.0 * float(cleaned_dimensions.get(
                    'maximum_absolute_error_m', float('nan')))))
    lines.append('Visual review is required before accepting this mesh.')
    return '\n'.join(lines)
