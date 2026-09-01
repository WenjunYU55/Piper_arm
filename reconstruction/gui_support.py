"""Pure command construction and reporting helpers for the Tk GUI."""

import json
import math
from pathlib import Path
import subprocess


MINIMUM_GUI_VOXEL_LENGTH_MM = 0.5
MAXIMUM_GUI_VOXEL_LENGTH_MM = 3.0
MINIMUM_GUI_SDF_TRUNC_MM = 2.0
MAXIMUM_GUI_SDF_TRUNC_MM = 30.0
RECONSTRUCTION_MASK_SOURCES = ('captured', 'offline_resegment')
RECONSTRUCTION_GEOMETRY_SOURCES = (
    'projected_color_depth', 'native_depth')
RECONSTRUCTION_GEOMETRY_SOURCE_LABELS = {
    'Projected colour depth (legacy)': 'projected_color_depth',
    'Native L515 depth (dense)': 'native_depth',
}
RECONSTRUCTION_HOLE_REPAIR_MODES = ('none', 'measured_wall')
RECONSTRUCTION_HOLE_REPAIR_LABELS = {
    'None (measured TSDF only)': 'none',
    'Conservative measured-wall repair (6 mm)': 'measured_wall',
}


def geometry_source_from_label(value):
    """Resolve one GUI label to the reconstruction CLI contract."""
    try:
        return RECONSTRUCTION_GEOMETRY_SOURCE_LABELS[str(value)]
    except KeyError as exc:
        raise ValueError('unknown reconstruction geometry source') from exc


def hole_repair_from_label(value):
    """Resolve one GUI label to the reconstruction CLI contract."""
    try:
        return RECONSTRUCTION_HOLE_REPAIR_LABELS[str(value)]
    except KeyError as exc:
        raise ValueError('unknown reconstruction hole repair mode') from exc


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


def _output_path(dataset, geometry_source, hole_repair='none'):
    source = str(geometry_source)
    if source not in RECONSTRUCTION_GEOMETRY_SOURCES:
        raise ValueError('unknown reconstruction geometry source')
    repair = str(hole_repair)
    if repair not in RECONSTRUCTION_HOLE_REPAIR_MODES:
        raise ValueError('unknown reconstruction hole repair mode')
    filename = (
        'target_mesh.ply'
        if source == 'projected_color_depth'
        else 'target_mesh.native_depth.ply')
    if repair == 'measured_wall':
        filename = filename[:-4] + '.wall_repaired.ply'
    return dataset / 'reconstruction' / 'validation' / filename


def existing_reconstruction_outputs(
        project_root, selection, geometry_source='projected_color_depth',
        hole_repair='none'):
    """Return inspectable saved outputs for one selected dataset, if present."""
    dataset = validated_dataset_path(project_root, selection)
    expected_root = dataset.resolve()
    output = _output_path(dataset, geometry_source, hole_repair)
    report_path = output.with_suffix(output.suffix + '.quality.json')
    if not report_path.is_file():
        return None
    report = load_quality_report(report_path)

    def saved_path(value):
        value = str(value or '').strip()
        if not value:
            return None
        path = Path(value).resolve()
        try:
            path.relative_to(expected_root)
        except ValueError as exc:
            raise ValueError('reported mesh escapes selected dataset') from exc
        return path if path.is_file() else None

    constrained = constrained_output_paths(report)
    cleaned = saved_path(report.get('mesh_path'))
    if cleaned is None:
        return None
    return {
        'report': report,
        'report_path': report_path.resolve(),
        'output_path': cleaned,
        'raw_output_path': saved_path(report.get('raw_mesh_path')),
        'measured_cloud_path': saved_path(report.get('measured_cloud_path')),
        'superposition_cloud_path': saved_path(
            constrained['superposition_cloud_path']),
        'consensus_cloud_path': saved_path(
            constrained['consensus_cloud_path']),
        'textured_mesh_path': saved_path(
            constrained['textured_mesh_path']),
    }


def constrained_output_paths(report):
    """Return constrained-result paths from explicit or auto reconstruction."""
    source = report
    if str(report.get('registration_mode', '')) != 'constrained_superposition':
        candidates = report.get('candidate_reports') or {}
        candidate = candidates.get('constrained_superposition')
        source = candidate if isinstance(candidate, dict) else {}
    return {
        'superposition_cloud_path': source.get('measured_cloud_path', ''),
        'consensus_cloud_path': source.get('consensus_cloud_path', ''),
        'textured_mesh_path': source.get('textured_mesh_path', ''),
    }


def reconstruction_command(project_root, selection, dimensions_mm,
                           registration_mode='auto', voxel_length_mm=3.0,
                           mask_source='captured',
                           geometry_source='projected_color_depth',
                           sdf_trunc_mm=15.0, hole_repair='none'):
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
    trunc_mm = float(sdf_trunc_mm)
    if not math.isfinite(trunc_mm) or not (
            MINIMUM_GUI_SDF_TRUNC_MM <= trunc_mm
            <= MAXIMUM_GUI_SDF_TRUNC_MM):
        raise ValueError('TSDF truncation must be between 2 and 30 mm')
    if trunc_mm < 2.0 * voxel_mm:
        raise ValueError(
            'TSDF truncation must be at least twice the mesh voxel size')
    source = str(mask_source)
    if source not in RECONSTRUCTION_MASK_SOURCES:
        raise ValueError(
            'mask source must be captured or offline_resegment')
    geometry = str(geometry_source)
    if geometry not in RECONSTRUCTION_GEOMETRY_SOURCES:
        raise ValueError('unknown reconstruction geometry source')
    repair = str(hole_repair)
    if repair not in RECONSTRUCTION_HOLE_REPAIR_MODES:
        raise ValueError('unknown reconstruction hole repair mode')
    output = _output_path(dataset, geometry, repair)
    command = [
        str(python), str(root / 'reconstruction' / 'tsdf_reconstruct.py'),
        str(dataset), '--output', str(output), '--registration-mode',
        str(registration_mode), '--expected-dimensions-mm',
        *(format(value, '.12g') for value in dimensions),
        '--voxel-length', format(voxel_mm / 1000.0, '.12g'),
        '--sdf-trunc', format(trunc_mm / 1000.0, '.12g'),
        '--mask-source', source,
        '--geometry-source', geometry,
        '--hole-repair', repair,
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
    if variant not in (
            'cleaned', 'raw', 'measured', 'superposition', 'consensus',
            'textured'):
        raise ValueError(
            'mesh variant must be cleaned, raw, measured, superposition or '
            'consensus, or textured')
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
        'Depth geometry: %s' % report.get(
            'geometry_source', configuration.get(
                'geometry_source', 'projected_color_depth')),
    ]
    repair = report.get('hole_repair') or {}
    if repair.get('applied'):
        lines.append(
            'Wall-hole repair: %s interpolated triangles; boundaries %s -> '
            '%s (6 mm maximum radius)' % (
                repair.get('interpolated_triangle_count', '?'),
                (repair.get('before') or {}).get(
                    'boundary_component_count', '?'),
                (repair.get('after') or {}).get(
                    'boundary_component_count', '?')))
    else:
        lines.append('Wall-hole repair: disabled (measured TSDF only)')
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
    consensus = report.get('cross_capture_consensus') or {}
    if report.get('consensus_cloud_path'):
        lines.append(
            'Cross-view consensus: %s points; median support %.1f captures; '
            'median spread %.2f mm' % (
                report.get('consensus_point_count', '?'),
                float(consensus.get(
                    'median_capture_support', float('nan'))),
                1000.0 * float(consensus.get(
                    'median_maximum_cross_capture_spread_m', float('nan')))))
    elif consensus.get('available') is False and consensus.get('reason'):
        lines.append(
            'Cross-view consensus unavailable: %s'
            % consensus['reason'])
    texture_source = report
    if not report.get('textured_mesh_path'):
        candidate = (report.get('candidate_reports') or {}).get(
            'constrained_superposition')
        if isinstance(candidate, dict):
            texture_source = candidate
    texture = texture_source.get('texture_baking') or {}
    if texture_source.get('textured_mesh_path'):
        lines.append(
            'Textured mesh (%s): %s/%s triangles textured from source RGB; '
            'atlas %sx%s px' % (
                texture.get('surface_method', 'unknown'),
                texture.get('textured_triangle_count', '?'),
                texture.get('triangle_count', '?'),
                texture.get('atlas_width_px', '?'),
                texture.get('atlas_height_px', '?')))
    lines.extend([
        'TSDF mesh voxel: %.2f mm (smaller permits more polygons)' % (
            1000.0 * float(configuration.get(
                'voxel_length_m', float('nan')))),
        'TSDF truncation band: %.2f mm' % (
            1000.0 * float(configuration.get(
                'sdf_trunc_m', float('nan')))),
        'Target mask: %s' % configuration.get(
            'semantic_mask_source', 'captured'),
        'Registration median RMSE: %.2f mm; dominant component: %.1f%%' % (
            1000.0 * float(registration.get('median_rmse_m', float('nan'))),
            100.0 * float(raw_mesh.get(
                'dominant_component_triangle_ratio', float('nan')))),
    ])
    if component_filter.get('decision'):
        corroborated_count = len(component_filter.get(
            'retained_only_by_measured_support_indices', []))
        lines.append(
            'Connected-target cleanup: %s; removed %s unsupported '
            'fragments; retained %s corroborated small surfaces' % (
                component_filter['decision'],
                component_filter.get(
                    'removed_fragment_component_count', '?'),
                corroborated_count))
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
