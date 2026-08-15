"""Pure command construction and reporting helpers for the Tk GUI."""

import json
from pathlib import Path
import subprocess


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


def reconstruction_command(project_root, selection, dimensions_mm,
                           registration_mode='auto'):
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
    output = dataset / 'reconstruction' / 'validation' / 'target_mesh.ply'
    command = [
        str(python), str(root / 'reconstruction' / 'tsdf_reconstruct.py'),
        str(dataset), '--output', str(output), '--registration-mode',
        str(registration_mode), '--expected-dimensions-mm',
        *(format(value, '.12g') for value in dimensions),
        '--allow-missing-calibration-id',
    ]
    return command, output


def viewer_command(project_root, report_path, show_input=False):
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
    command = [
        str(python), str(root / 'reconstruction' / 'view_reconstruction.py'),
        '--report', str(report),
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
    dimensions = mesh.get('dimension_check') or {}
    lines = [
        '%s / %s' % (
            report.get('overall_quality',
                       report.get('structural_quality', 'UNKNOWN')),
            provenance.get('classification', 'UNKNOWN')),
        'Selected mode: %s; views: %s; mesh: %s vertices / %s triangles' % (
            report.get('registration_mode', 'unknown'),
            report.get('integrated_views', '?'), report.get('vertex_count', '?'),
            report.get('triangle_count', '?')),
        'Registration median RMSE: %.2f mm; dominant component: %.1f%%' % (
            1000.0 * float(registration.get('median_rmse_m', float('nan'))),
            100.0 * float(mesh.get(
                'dominant_component_triangle_ratio', float('nan')))),
    ]
    if dimensions:
        lines.append(
            'Provisional cube OBB: %s mm; maximum size error: %.2f mm (%s)' % (
                ' x '.join('%.2f' % (1000.0 * float(value))
                           for value in dimensions.get('observed_obb_m', [])),
                1000.0 * float(dimensions.get(
                    'maximum_absolute_error_m', float('nan'))),
                report.get('dimension_quality', 'UNKNOWN')))
    lines.append('Visual review is required before accepting this mesh.')
    return '\n'.join(lines)
