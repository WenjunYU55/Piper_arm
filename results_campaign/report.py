"""Generate campaign CSV, Excel, figures, and reconstruction comparisons."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Mapping, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .campaign import (  # noqa: E402
    CampaignStore, atomic_write_json, load_json, sha256_file, utc_now)
from .collector import collect_campaign  # noqa: E402
from .metrics import summarize_evidence  # noqa: E402
from .workbook import write_xlsx  # noqa: E402


BLUE = '#003E74'
LIGHT_BLUE = '#00A6D6'
GREEN = '#008A68'
ORANGE = '#F28E2B'
GREY = '#5B6770'
MODES = ('robot_pose', 'bounded_gicp', 'multiway_gicp', 'constrained_superposition', 'scene_pose_graph')


def _new_output_directory(project_root: Path, campaign_id: str, requested: Optional[Path] = None) -> Path:
    base = Path(requested) if requested else project_root / 'datasets' / 'experiment_results' / campaign_id
    if not base.exists():
        base.mkdir(parents=True)
        return base.resolve()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = base.parent / (base.name + '_' + timestamp)
    counter = 2
    while output.exists():
        output = base.parent / (base.name + '_' + timestamp + '_%d' % counter)
        counter += 1
    output.mkdir(parents=True)
    return output.resolve()


def _write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    if not columns:
        columns = ['note']
        rows = [{'note': 'No data recorded yet.'}]
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def _nested(data: Mapping[str, Any], *keys, default=None):
    value = data
    for key in keys:
        if not isinstance(value, Mapping):
            return default
        value = value.get(key)
    return default if value is None else value


def _evidence_manifest_rows(
        evidence: Mapping[str, Any], attempt: Path,
        included: bool, exclusion_reason: str) -> List[Dict[str, Any]]:
    """Bind each reported mission to exact source and configuration hashes."""
    submission = evidence.get('submission', {})
    configuration = submission.get('configuration_snapshot', {})
    expected = submission.get('expected_trial', {})
    common = {
        'campaign_id': evidence.get('campaign_id'),
        'trial_id': evidence.get('trial_id'),
        'pair_index': expected.get('pair_index'),
        'task_id': evidence.get('task_id'),
        'planner_backend': evidence.get('planner_backend'),
        'evidence_class': evidence.get('evidence_class'),
        'included_in_comparison': included,
        'exclusion_reason': exclusion_reason,
        'code_branch': configuration.get('git', {}).get('branch'),
        'code_commit': configuration.get('git', {}).get('commit'),
        'code_dirty': configuration.get('git', {}).get('dirty'),
        'configuration_captured_utc': configuration.get('captured_utc'),
    }
    rows = []
    seen = set()

    def add(role, path, digest, size):
        key = (str(path), str(digest))
        if key in seen:
            return
        seen.add(key)
        row = dict(common)
        row.update({
            'source_role': role,
            'source_path': str(path),
            'source_sha256': str(digest),
            'source_size_bytes': size,
            'evidence_snapshot_path': str((attempt / 'evidence.json').resolve()),
        })
        rows.append(row)

    evidence_path = attempt / 'evidence.json'
    if evidence_path.is_file():
        add('evidence_snapshot', evidence_path.resolve(),
            sha256_file(evidence_path), evidence_path.stat().st_size)
    for source in evidence.get('sources', []):
        if isinstance(source, Mapping):
            add('mission_evidence', source.get('path', ''),
                source.get('sha256', ''), source.get('size_bytes'))
    for source in configuration.get('files', []):
        if isinstance(source, Mapping) and source.get('available'):
            add('configuration', source.get('path', ''),
                source.get('sha256', ''), source.get('size_bytes'))
    return rows


def _quality_row(quality_path: Path, common: Mapping[str, Any]) -> Dict[str, Any]:
    quality = load_json(quality_path, {})
    mesh_metrics = quality.get('mesh_metrics', {})
    dimensions = mesh_metrics.get(
        'oriented_bounding_box_extents_m',
        mesh_metrics.get('dimensions_m', mesh_metrics.get('dimensions')))
    if not dimensions:
        raw_metrics = quality.get('raw_mesh_metrics', {})
        dimensions = raw_metrics.get(
            'oriented_bounding_box_extents_m',
            raw_metrics.get('dimensions_m', raw_metrics.get('dimensions')))
    dimensions = list(dimensions or [None, None, None])
    expected = _nested(quality, 'configuration', 'expected_dimensions_m', default=[0.035] * 3)
    observed_mm = [float(value) * 1000.0 if value is not None else None for value in dimensions[:3]]
    expected_mm = [float(value) * 1000.0 for value in expected[:3]] if expected else [35.0] * 3
    errors = [abs(a - b) for a, b in zip(observed_mm, expected_mm) if a is not None]
    registration = quality.get('registration_summary', {})
    component = quality.get('component_filter', {})
    residual = mesh_metrics.get('point_to_mesh_residual', {})
    row = dict(common)
    row.update({
        'registration_mode': quality.get('registration_mode'),
        'geometry_source': quality.get('geometry_source'),
        'integrated_views': quality.get('integrated_views'),
        'observed_x_mm': observed_mm[0], 'observed_y_mm': observed_mm[1], 'observed_z_mm': observed_mm[2],
        'known_x_mm': expected_mm[0], 'known_y_mm': expected_mm[1], 'known_z_mm': expected_mm[2],
        'mean_absolute_dimension_error_mm': float(np.mean(errors)) if errors else None,
        'maximum_absolute_dimension_error_mm': float(np.max(errors)) if errors else None,
        'median_point_to_mesh_residual_mm': residual.get(
            'median_m', _nested(
                registration, 'point_to_mesh_residual_median_m')),
        'p90_point_to_mesh_residual_mm': residual.get(
            'p90_m', _nested(
                registration, 'point_to_mesh_residual_p90_m')),
        'connected_components': mesh_metrics.get(
            'connected_component_count',
            component.get('original_component_count')),
        'dominant_component_ratio': mesh_metrics.get(
            'dominant_component_triangle_ratio',
            component.get('original_dominant_component_surface_area_ratio')),
        'vertex_count': quality.get('vertex_count'),
        'triangle_count': quality.get('triangle_count'),
        'overall_quality': quality.get('overall_quality'),
        'dimension_quality': quality.get('dimension_quality'),
        'quality_report': str(quality_path.resolve()),
        'evidence_class': 'CONTROLLED_REPLAY',
        'limitation': 'All modes use identical accepted captures; lower registration residual alone is not reconstruction accuracy.',
    })
    for key in ('median_point_to_mesh_residual_mm', 'p90_point_to_mesh_residual_mm'):
        if row[key] is not None:
            row[key] = float(row[key]) * 1000.0
    return row


def _run_reconstructions(project_root: Path, output: Path, run_rows: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    interpreter = project_root / 'reconstruction' / '.venv' / 'bin' / 'python'
    if not interpreter.is_file():
        interpreter = Path('/usr/bin/python3')
    for run in run_rows:
        dataset_text = str(run.get('dataset_path', ''))
        dataset = Path(dataset_text) if dataset_text else None
        if dataset is None or not dataset.is_dir():
            continue
        common = {key: run.get(key) for key in ('campaign_id', 'trial_id', 'pair_index', 'task_id', 'planner_backend', 'target_x_m', 'target_y_m', 'target_z_m')}
        for mode in MODES:
            directory = output / 'reconstruction' / str(run.get('trial_id')) / mode
            directory.mkdir(parents=True, exist_ok=True)
            mesh = directory / 'target_mesh.ply'
            command = [str(interpreter), str(project_root / 'reconstruction' / 'tsdf_reconstruct.py'), str(dataset), '--output', str(mesh), '--registration-mode', mode, '--expected-dimensions-mm', '35', '35', '35', '--mask-source', 'captured', '--geometry-source', 'native_depth', '--hole-repair', 'none', '--voxel-length', '0.003', '--sdf-trunc', '0.015']
            completed = subprocess.run(command, cwd=str(project_root), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log = directory / 'reconstruction.log'
            log.write_text(completed.stdout, encoding='utf-8')
            quality_candidates = list(directory.glob('*.quality.json'))
            if quality_candidates:
                row = _quality_row(quality_candidates[0], common)
                row['command'] = command
                row['returncode'] = completed.returncode
            else:
                row = dict(common)
                row.update({'registration_mode': mode, 'overall_quality': 'FAIL', 'returncode': completed.returncode, 'quality_report': '', 'evidence_class': 'CONTROLLED_REPLAY', 'limitation': 'Reconstruction produced no quality report; inspect reconstruction.log.'})
            rows.append(row)
    return rows


def _existing_reconstructions(run_rows: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for run in run_rows:
        dataset_text = str(run.get('dataset_path', ''))
        dataset = Path(dataset_text) if dataset_text else None
        if dataset is None or not dataset.is_dir():
            continue
        common = {key: run.get(key) for key in ('campaign_id', 'trial_id', 'pair_index', 'task_id', 'planner_backend', 'target_x_m', 'target_y_m', 'target_z_m')}
        for path in sorted((dataset / 'reconstruction').glob('**/*.quality.json')):
            rows.append(_quality_row(path, common))
    return rows


def _save_figure(fig, output: Path, stem: str) -> None:
    fig.savefig(str(output / (stem + '.png')), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(str(output / (stem + '.pdf')), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _no_data(ax, title):
    ax.set_title(title)
    ax.text(0.5, 0.5, 'No completed campaign data yet', ha='center', va='center', transform=ax.transAxes, color=GREY, fontsize=13)
    ax.set_axis_off()


def _figures(output: Path, sheets: Mapping[str, List[Mapping[str, Any]]],
             evidence_class: str) -> None:
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    acquisition = sheets.get('Acquisition', [])
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    x_values, z_values = (0.30, 0.50, 0.70, 0.90, 1.10), (0.0, 0.12, 0.30)
    for row_index, (metric, metric_title) in enumerate((('grounding_dino_confidence', 'Grounding DINO'), ('sam2_score', 'SAM2'))):
        for column_index, backend in enumerate(('tesseract', 'curobo')):
            ax = axes[row_index, column_index]
            matrix = np.full((len(z_values), len(x_values)), np.nan)
            for record in acquisition:
                if (record.get('planner_backend') != backend
                        or record.get(metric) is None):
                    continue
                try:
                    xi = min(range(len(x_values)), key=lambda index: abs(x_values[index] - float(record['target_x_m'])))
                    zi = min(range(len(z_values)), key=lambda index: abs(z_values[index] - float(record['target_z_m'])))
                    matrix[zi, xi] = float(record[metric])
                except (TypeError, ValueError):
                    continue
            if np.isfinite(matrix).any():
                ax.imshow(
                    matrix, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
                for zi in range(len(z_values)):
                    for xi in range(len(x_values)):
                        if math.isfinite(matrix[zi, xi]):
                            ax.text(xi, zi, '%.1f%%' % (100 * matrix[zi, xi]), ha='center', va='center', fontsize=9, color='white' if matrix[zi, xi] > .6 else 'black')
                ax.set_xticks(range(len(x_values)))
                ax.set_xticklabels(['%.2f' % value for value in x_values])
                ax.set_yticks(range(len(z_values)))
                ax.set_yticklabels(['%.2f' % value for value in z_values])
                ax.set_title('%s — %s (N=%d)' % (backend.title(), metric_title, int(np.isfinite(matrix).sum())))
                ax.set_xlabel('Target X from base_link (m)')
                ax.set_ylabel('Target Z from base_link (m)')
            else:
                _no_data(ax, '%s — %s' % (backend.title(), metric_title))
    fig.suptitle('First acquisition confidence — %s (model confidence, not accuracy)' % evidence_class, fontweight='bold')
    _save_figure(fig, output, '01_acquisition_confidence')

    coverage = sheets.get('Coverage', [])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharey=True)
    for ax, z in zip(axes, (0.0, 0.12, 0.30)):
        plotted = False
        rows_z = [row for row in coverage if row.get('target_z_m') is not None and abs(float(row['target_z_m']) - z) < 1e-6]
        for row_group_key in sorted(set((row.get('task_id'), row.get('planner_backend'), row.get('target_x_m')) for row in rows_z)):
            task, backend, x = row_group_key
            values = sorted([row for row in rows_z if row.get('task_id') == task], key=lambda row: row.get('capture_index', 0))
            if not values:
                continue
            ax.plot([int(row['capture_index']) + 1 for row in values], [100.0 * float(row.get('cumulative_coverage_fraction', 0)) for row in values], color=BLUE if backend == 'tesseract' else ORANGE, linestyle='-' if backend == 'tesseract' else '--', alpha=.75, label='%s, x=%.1f m' % (backend.title(), x))
            plotted = True
        if plotted:
            ax.set(
                title='Target Z = %.2f m' % z,
                xlabel='Accepted capture number')
            ax.grid(alpha=.22)
        else:
            _no_data(ax, 'Target Z = %.2f m' % z)
    axes[0].set_ylabel('Known six-face cube surface covered (%)')
    if coverage and axes[-1].get_legend_handles_labels()[0]:
        axes[-1].legend(fontsize=7, loc='best')
    fig.suptitle('Coverage growth by accepted capture — %s; 35 mm cube, 2 mm association' % evidence_class, fontweight='bold')
    _save_figure(fig, output, '02_coverage_by_capture')

    runs = sheets.get('Runs', [])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    valid = [row for row in runs if row.get('final_cube_coverage_fraction') is not None]
    if valid:
        labels = ['%s\nX %.1f Z %.2f' % (row.get('planner_backend', '').title(), row.get('target_x_m'), row.get('target_z_m')) for row in valid]
        xloc = np.arange(len(valid))
        axes[0].bar(
            xloc,
            [100 * row['final_cube_coverage_fraction'] for row in valid],
            color=[BLUE if row.get('planner_backend') == 'tesseract'
                   else ORANGE for row in valid])
        axes[0].set_xticks(xloc)
        axes[0].set_xticklabels(
            labels, rotation=55, ha='right', fontsize=7)
        axes[0].set_ylabel('Final six-face coverage (%)')
        axes[0].grid(axis='y', alpha=.22)
        axes[1].scatter(
            [row.get('azimuth_span_deg', 0) for row in valid],
            [row.get('accepted_captures', 0) for row in valid],
            c=[BLUE if row.get('planner_backend') == 'tesseract'
               else ORANGE for row in valid], s=65)
        axes[1].set(
            xlabel='Azimuth span (deg)', ylabel='Accepted captures',
            title='View diversity and captures')
        axes[1].grid(alpha=.22)
    else:
        _no_data(axes[0], 'Final cube coverage')
        _no_data(axes[1], 'View diversity and captures')
    fig.suptitle('Mission-level NBV results — %s' % evidence_class, fontweight='bold')
    _save_figure(fig, output, '03_backend_results_grid')

    timeline = sheets.get('Capture_Timeline', [])
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, 15))
    plotted = False
    for pair_index in range(1, 16):
        rows_pair = [row for row in timeline if int(row.get('pair_index') or 0) == pair_index]
        for backend in ('tesseract', 'curobo'):
            values = sorted([row for row in rows_pair if row.get('planner_backend') == backend and row.get('submit_to_capture_sec') is not None], key=lambda row: row.get('capture_index', 0))
            if not values:
                continue
            times = [0.0] + [float(row['submit_to_capture_sec']) for row in values]
            counts = [0] + list(range(1, len(values) + 1))
            first = values[0]
            label = 'P%02d X=%.2f Z=%.2f %s' % (pair_index, first['target_x_m'], first['target_z_m'], backend.title())
            ax.step(times, counts, where='post', color=colors[pair_index - 1], linestyle='-' if backend == 'tesseract' else '--', linewidth=1.8, alpha=.9, label=label)
            plotted = True
    if plotted:
        ax.axvline(120.0, color='black', linestyle=':', linewidth=2.2, label='120 s reference')
        ax.set(xlabel='Elapsed from GUI mission submission (s)', ylabel='Accepted captures (N)')
        ax.grid(alpha=.22)
        ax.legend(fontsize=6.5, ncol=3, loc='upper left')
    else:
        _no_data(ax, 'Accepted captures over mission time')
    fig.suptitle('Tesseract (solid) vs cuRobo (dashed) operating runtime — %s' % evidence_class, fontweight='bold')
    _save_figure(fig, output, '04_runtime_by_capture')

    point = sheets.get('PointCloud', [])
    recon = sheets.get('Reconstruction', [])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    point_valid = [row for row in point if row.get('mean_absolute_dimension_error_mm') is not None]
    if point_valid:
        axes[0].bar(
            np.arange(len(point_valid)),
            [row['mean_absolute_dimension_error_mm'] for row in point_valid],
            color=[BLUE if row.get('planner_backend') == 'tesseract'
                   else ORANGE for row in point_valid])
        axes[0].set(
            xlabel='Mission',
            ylabel='Mean absolute dimension error (mm)',
            title='Fused qualified point cloud')
        axes[0].grid(axis='y', alpha=.22)
    else:
        _no_data(axes[0], 'Fused qualified point cloud')
    recon_valid = [row for row in recon if row.get('mean_absolute_dimension_error_mm') is not None]
    if recon_valid:
        modes = list(dict.fromkeys(
            row.get('registration_mode') for row in recon_valid))
        values = [
            [row['mean_absolute_dimension_error_mm'] for row in recon_valid
             if row.get('registration_mode') == mode]
            for mode in modes]
        axes[1].boxplot(
            values, labels=[str(mode).replace('_', '\n') for mode in modes],
            showfliers=True)
        axes[1].set(
            ylabel='Mean absolute dimension error (mm)',
            title='Reconstruction on identical captures')
        axes[1].tick_params(axis='x', labelsize=8)
        axes[1].grid(axis='y', alpha=.22)
    else:
        _no_data(axes[1], 'Reconstruction on identical captures')
    fig.suptitle('Geometry evidence — point-cloud %s; reconstruction CONTROLLED_REPLAY' % evidence_class, fontweight='bold')
    _save_figure(fig, output, '05_pointcloud_reconstruction_accuracy')


def build_report(project_root: Path, campaign_id: str, output: Optional[Path] = None, run_reconstruction: bool = False) -> Path:
    root = Path(project_root).resolve()
    store = CampaignStore(root, campaign_id)
    definition = store.create_or_load()
    evidence_class = str(definition.get('evidence_class', 'EXPLORATORY'))
    backend_order = str(definition.get('design', {}).get(
        'backend_order', 'unknown'))
    if backend_order == 'curobo_block_then_tesseract_block':
        evidence_definition = (
            '%s: one Tesseract and one cuRobo mission at the same declared '
            'target position, collected in backend blocks. N=1 per cell; '
            'elapsed-time and environmental drift are confounded with '
            'backend.' % evidence_class)
        design_description = (
            'backend-blocked matched physical missions, all cuRobo first '
            'and then all Tesseract, with one run per backend at each of 15 '
            'declared target positions')
        order_limitation = (
            'backend-block collection confounds planner with elapsed time '
            'and environmental drift; ')
    else:
        evidence_definition = (
            '%s: one Tesseract and one cuRobo mission at the same declared '
            'target position; order alternates. N=1 per cell.' %
            evidence_class)
        design_description = (
            'alternating physical missions, Tesseract versus cuRobo, with '
            'one run per backend at each of 15 declared target positions')
        order_limitation = ''
    collect_campaign(root, store.root)
    destination = _new_output_directory(root, campaign_id, output)
    sheets: Dict[str, List[Dict[str, Any]]] = {
        'Campaign': [definition], 'Runs': [], 'Acquisition': [], 'Capture_Timeline': [], 'Coverage': [], 'Planning': [], 'PointCloud': [], 'Phase_Timing': [], 'Reconstruction': [], 'Evidence_Manifest': [], 'Exclusions': [], 'Definitions': [],
    }
    for attempt in store.task_attempts():
        evidence = load_json(attempt / 'evidence.json', {})
        if not isinstance(evidence, dict):
            continue
        if evidence.get('matches_schedule') is not True:
            reason = 'Submitted coordinates or planner backend did not match the loaded campaign trial.'
            sheets['Exclusions'].append({'task_id': evidence.get('task_id'), 'trial_id': evidence.get('trial_id'), 'reason': reason, 'evidence_path': str(attempt / 'evidence.json')})
            sheets['Evidence_Manifest'].extend(
                _evidence_manifest_rows(evidence, attempt, False, reason))
            continue
        sheets['Evidence_Manifest'].extend(
            _evidence_manifest_rows(evidence, attempt, True, ''))
        normalized = summarize_evidence(evidence)
        for name, rows in normalized.items():
            sheets[name].extend(rows)
    sheets['Reconstruction'] = _run_reconstructions(root, destination, sheets['Runs']) if run_reconstruction else _existing_reconstructions(sheets['Runs'])
    sheets['Definitions'] = [
        {'metric': 'Evidence class', 'definition': evidence_definition},
        {'metric': 'Coverage', 'definition': 'Fraction of a 1 mm sampled 35 mm six-face cube surface within 2 mm of any persisted qualified depth point. Axis-aligned cube assumption.'},
        {'metric': 'Acquisition confidence', 'definition': 'Grounding DINO target confidence and SAM2 mask score from first rough/acquisition heavy inference. These are not detection accuracy.'},
        {'metric': 'Capture duration', 'definition': 'Persisted capture timestamp minus CAPTURING_RGBD transaction timestamp.'},
        {'metric': 'Dimension error', 'definition': 'Absolute difference between 1st-to-99th percentile fused cloud extents and the 35 mm reference cube; not independent metrology.'},
        {'metric': '120 s line', 'definition': 'Reference only; 30 cm is the only confirmed target distance and 50 cm remains unconfirmed.'},
    ]
    for name, rows in sheets.items():
        _write_csv(destination / (name.lower() + '.csv'), rows)
    write_xlsx(destination / 'PiPER_results_campaign.xlsx', sheets)
    _figures(destination, sheets, evidence_class)
    bookmark_document = store.refresh_bookmarks()
    for name in ('campaign_bookmarks.json', 'campaign_bookmarks.csv'):
        shutil.copy2(store.root / name, destination / name)
    progress = store.progress()
    readme = """# PiPER results campaign output

Generated: {generated}\

Campaign: `{campaign}`

Evidence design: {design_description}. Evidence class: `{evidence_class}`.

Progress: {completed}/{total} scheduled trials have a matching terminal record.

The recorder is passive and file-only. It reads existing mission results, schema-2 capture metadata, heavy-perception outputs, ray diagnostics, and reconstruction quality reports. It never publishes ROS data or commands hardware.

`campaign_bookmarks.json` and `campaign_bookmarks.csv` are the durable one-row-per-attempt ledger. `evidence_manifest.csv` maps included and excluded attempts to every source/configuration hash used by this report. `PiPER_results_campaign.xlsx` contains all normalized result tables, including the evidence manifest. Matching CSV files are provided for reproducibility. PNG/PDF figures are generated from those tables. Reconstruction rows are CONTROLLED_REPLAY only when the same stored capture set is processed by each mode.

Important limitations: {order_limitation}one run per backend/location does not support confidence intervals; acquisition confidence is not accuracy; depth spread is stability/contamination evidence rather than metric sensor accuracy; cube coverage assumes a 35 mm axis-aligned six-face reference; 30 cm is confirmed but 50 cm is not.
""".format(generated=utc_now(), campaign=campaign_id, evidence_class=evidence_class, design_description=design_description, order_limitation=order_limitation, completed=progress['completed'], total=progress['total'])
    (destination / 'README.md').write_text(readme, encoding='utf-8')
    artifacts = []
    for path in sorted(destination.rglob('*')):
        if path.is_file() and path.name != 'report_manifest.json':
            artifacts.append({
                'path': str(path.relative_to(destination)),
                'sha256': sha256_file(path),
                'size_bytes': path.stat().st_size,
            })
    atomic_write_json(destination / 'report_manifest.json', {
        'schema_version': 1,
        'generated_utc': utc_now(),
        'campaign_id': campaign_id,
        'campaign_definition': str(store.definition_path),
        'campaign_definition_sha256': sha256_file(store.definition_path),
        'campaign_bookmark_entry_count': bookmark_document['entry_count'],
        'completed_trials': progress['completed'],
        'total_trials': progress['total'],
        'run_reconstruction': run_reconstruction,
        'output_directory': str(destination),
        'artifacts': artifacts,
    })
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run-reconstruction', action='store_true')
    args = parser.parse_args(argv)
    print(build_report(args.project_root, args.campaign, args.output, args.run_reconstruction))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
