"""Pure artifact analysis and reporting for the camera range experiment."""

import csv
from datetime import datetime, timezone
import html
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import cv2
import numpy as np
import yaml

from piper_mobile_manipulation.utils.target_depth import (
    select_target_depth_component,
)


FIELDS = (
    'recorded_utc',
    'reference_surface_distance_m',
    'request_id',
    'job_id',
    'worker_status',
    'production_depth_status',
    'groundingdino_detected',
    'groundingdino_confidence',
    'sam2_score',
    'mask_area_px',
    'mask_touches_border',
    'raw_depth_valid_ratio_0p15_to_9m',
    'current_0p24_to_3p00m_valid_ratio',
    'selected_depth_m',
    'absolute_depth_error_m',
    'selected_depth_stddev_m',
    'selected_depth_mad_m',
    'selected_depth_points',
    'selected_support_fraction',
    'usable',
    'failure_reason',
)


def finite_float(value, default=math.nan):
    """Return a finite float or the supplied default."""
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if math.isfinite(converted) else default


def depth_metres(depth):
    """Convert a stored L515 depth image into metres."""
    values = np.asarray(depth, dtype=np.float32)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size and float(np.median(positive)) > 20.0:
        values *= 0.001
    return values


def find_job_file(spool, job_id, filename):
    """Find an exact job artifact without accepting another request."""
    for root in ('consumed', 'responses', 'archive', 'processing', 'requests'):
        candidate = spool / root / job_id / filename
        if candidate.is_file():
            return candidate
    return None


def load_sam2_score(result):
    """Read the target SAM2 score from the worker's immutable result."""
    path = Path(str(result.get('sam2_masks_yaml', '')))
    if not path.is_file():
        return math.nan
    try:
        payload = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except (OSError, yaml.YAMLError):
        return math.nan
    for item in payload.get('masks', []):
        if not isinstance(item, dict):
            continue
        if item.get('mask_role') == 'target' or item.get('is_target_candidate'):
            return finite_float(item.get('sam2_score'))
    return math.nan


def empty_row(request_id, reference_distance_m, terminal_status):
    """Build a typed failure-first result row."""
    row = {field: '' for field in FIELDS}
    row.update({
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'reference_surface_distance_m': float(reference_distance_m),
        'request_id': request_id,
        'job_id': str(terminal_status.get('job_id', '')),
        'worker_status': str(terminal_status.get('state', 'unknown')),
        'production_depth_status': str(
            terminal_status.get('target_depth_status', 'UNKNOWN')),
        'groundingdino_detected': False,
        'groundingdino_confidence': 0.0,
        'sam2_score': 0.0,
        'mask_area_px': 0,
        'mask_touches_border': False,
        'raw_depth_valid_ratio_0p15_to_9m': 0.0,
        'current_0p24_to_3p00m_valid_ratio': 0.0,
        'selected_depth_m': 0.0,
        'absolute_depth_error_m': 0.0,
        'selected_depth_stddev_m': 0.0,
        'selected_depth_mad_m': 0.0,
        'selected_depth_points': 0,
        'selected_support_fraction': 0.0,
        'usable': False,
        'failure_reason': '',
    })
    return row


def analyse_job(spool, request_id, reference_distance_m, terminal_status):
    """Build one evidence row from an exact heavy-refresh transaction."""
    row = empty_row(request_id, reference_distance_m, terminal_status)
    job_id = row['job_id']
    if not job_id:
        row['failure_reason'] = str(
            terminal_status.get('error', 'heavy refresh returned no job ID'))
        return row

    result_path = mask_path = depth_path = None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        result_path = find_job_file(spool, job_id, 'result.yaml')
        mask_path = find_job_file(spool, job_id, 'target_mask.png')
        depth_path = find_job_file(spool, job_id, 'depth.npy')
        if result_path and mask_path and depth_path:
            break
        time.sleep(0.1)
    if result_path is None:
        row['failure_reason'] = 'exact worker result artifact is unavailable'
        return row
    try:
        result = yaml.safe_load(result_path.read_text(encoding='utf-8')) or {}
    except (OSError, yaml.YAMLError) as exc:
        row['failure_reason'] = 'worker result unreadable: %s' % exc
        return row

    row['worker_status'] = str(result.get('status', row['worker_status']))
    row['groundingdino_confidence'] = finite_float(
        result.get('target_confidence'), 0.0)
    row['sam2_score'] = finite_float(load_sam2_score(result), 0.0)
    detected = result.get('status') == 'ok' and mask_path is not None
    row['groundingdino_detected'] = bool(detected)
    if not detected:
        row['failure_reason'] = str(
            result.get('target_rejection_reason')
            or result.get('status') or 'target not detected')
        return row
    if depth_path is None:
        row['failure_reason'] = 'exact request-correlated depth is unavailable'
        return row

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    try:
        depth = depth_metres(np.load(str(depth_path), allow_pickle=False))
    except (OSError, ValueError) as exc:
        row['failure_reason'] = 'depth artifact unreadable: %s' % exc
        return row
    if mask is None or mask.shape != depth.shape:
        row['failure_reason'] = 'mask/depth shape mismatch'
        return row

    support = mask > 0
    mask_area = int(np.count_nonzero(support))
    row['mask_area_px'] = mask_area
    row['mask_touches_border'] = bool(
        np.any(support[0, :]) or np.any(support[-1, :])
        or np.any(support[:, 0]) or np.any(support[:, -1]))
    if mask_area == 0:
        row['failure_reason'] = 'SAM2 target mask is empty'
        return row
    eroded = cv2.erode(
        support.astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
        iterations=2) > 0
    if not np.any(eroded):
        eroded = support

    broad = eroded & np.isfinite(depth) & (depth >= 0.15) & (depth <= 9.0)
    current = eroded & np.isfinite(depth) & (depth >= 0.24) & (depth <= 3.00)
    denominator = float(max(int(np.count_nonzero(eroded)), 1))
    row['raw_depth_valid_ratio_0p15_to_9m'] = float(
        np.count_nonzero(broad)) / denominator
    row['current_0p24_to_3p00m_valid_ratio'] = float(
        np.count_nonzero(current)) / denominator
    try:
        selected_mask, report = select_target_depth_component(
            broad,
            depth,
            minimum_points=20,
            minimum_support_fraction=0.15,
            ambiguity_margin=0.08,
        )
    except ValueError as exc:
        row['failure_reason'] = str(exc)
        return row

    selected = depth[selected_mask]
    measured = float(report['selected_depth_m'])
    error = abs(measured - float(reference_distance_m))
    row.update({
        'selected_depth_m': measured,
        'absolute_depth_error_m': error,
        'selected_depth_stddev_m': float(np.std(selected)),
        'selected_depth_mad_m': float(report['selected_depth_mad_m']),
        'selected_depth_points': int(report['selected_points']),
        'selected_support_fraction': float(report['selected_support_fraction']),
    })

    reasons = []
    if row['mask_touches_border']:
        reasons.append('mask touches frame border')
    if row['raw_depth_valid_ratio_0p15_to_9m'] < 0.50:
        reasons.append('raw target depth ratio below 0.50')
    if row['selected_depth_points'] < 50:
        reasons.append('fewer than 50 coherent target points')
    if row['selected_support_fraction'] < 0.50:
        reasons.append('selected target layer support below 0.50')
    if row['selected_depth_stddev_m'] > 0.03:
        reasons.append('selected target depth stddev above 0.03 m')
    allowed_error = max(0.03, 0.05 * float(reference_distance_m))
    if error > allowed_error:
        reasons.append('depth error exceeds max(0.03 m, 5%)')
    row['usable'] = not reasons
    row['failure_reason'] = '; '.join(reasons)
    return row


def csv_value(value):
    """Use readable values while retaining numeric spreadsheet cells."""
    if isinstance(value, float) and not math.isfinite(value):
        return ''
    return value


def write_csv(path, rows):
    """Atomically write the range evidence table."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: csv_value(row.get(key, '')) for key in FIELDS})
    os.replace(str(temporary), str(path))


def write_html(path, rows):
    """Write a dependency-free measured-versus-reference plot and table."""
    width, height, margin = 900, 430, 62
    valid = [row for row in rows if finite_float(
        row.get('reference_surface_distance_m'), 0.0) > 0.0]
    maximum = max(
        [3.00]
        + [finite_float(row.get('reference_surface_distance_m'), 0.0)
           for row in valid]
        + [finite_float(row.get('selected_depth_m'), 0.0) for row in valid]
    ) * 1.10

    def point(x_value, y_value):
        x = margin + x_value / maximum * (width - 2 * margin)
        y = height - margin - y_value / maximum * (height - 2 * margin)
        return x, y

    x0, y0 = point(0.0, 0.0)
    x1, y1 = point(maximum, maximum)
    chart = [
        '<svg viewBox="0 0 %d %d">' % (width, height),
        '<rect width="100%%" height="100%%" fill="#101820"/>',
        '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" '
        'stroke="white"/>' % (x0, y0, x1, y0),
        '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" '
        'stroke="white"/>' % (x0, y0, x0, y1),
        '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" '
        'stroke="#64b5f6" stroke-width="2" stroke-dasharray="8 6"/>'
        % (x0, y0, x1, y1),
    ]
    for row in valid:
        reference = finite_float(row.get('reference_surface_distance_m'), 0.0)
        measured = finite_float(row.get('selected_depth_m'), 0.0)
        if measured <= 0.0:
            continue
        x, y = point(reference, measured)
        passed = row.get('usable') is True \
            or str(row.get('usable')).lower() == 'true'
        colour = '#35d07f' if passed else '#ff5c5c'
        chart.append(
            '<circle cx="%.1f" cy="%.1f" r="6" fill="%s">'
            '<title>reference %.3f m, measured %.3f m</title></circle>'
            % (x, y, colour, reference, measured))
    chart.extend((
        '<text x="450" y="422" fill="white" text-anchor="middle">'
        'Tape-measured camera-to-visible-surface distance (m)</text>',
        '<text x="18" y="215" fill="white" text-anchor="middle" '
        'transform="rotate(-90 18 215)">Selected L515 depth (m)</text>',
        '</svg>',
    ))
    header = ''.join('<th>%s</th>' % html.escape(field) for field in FIELDS)
    body = ''.join('<tr>%s</tr>' % ''.join(
        '<td>%s</td>' % html.escape(str(csv_value(row.get(field, ''))))
        for field in FIELDS) for row in rows)
    document = '''<!doctype html><html><head><meta charset="utf-8">
<title>PiPER perception range test</title><style>
body{font-family:sans-serif;background:#18242d;color:#eef;margin:24px}
table{border-collapse:collapse;font-size:12px;background:#fff;color:#111}
th,td{border:1px solid #888;padding:5px}th{background:#ddd}
.table{overflow:auto;max-height:520px}.note{max-width:1000px}
</style></head><body><h1>PiPER perception range test</h1>
<p class="note">Blue is ideal depth. Green points pass the diagnostic criteria;
red points do not. This is camera-to-visible-surface range, not base_link range
or physical-motion qualification.</p>%s<div class="table"><table><thead><tr>%s
</tr></thead><tbody>%s</tbody></table></div></body></html>''' % (
        ''.join(chart), header, body)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(document, encoding='utf-8')
    os.replace(str(temporary), str(path))


def write_xlsx(csv_path, xlsx_path):
    """Convert the canonical CSV to XLSX using installed LibreOffice."""
    executable = shutil.which('libreoffice')
    if executable is None:
        return False, 'libreoffice unavailable; CSV and HTML were written'
    with tempfile.TemporaryDirectory(prefix='piper_range_xlsx_') as directory:
        root = Path(directory)
        source = root / csv_path.name
        shutil.copy2(str(csv_path), str(source))
        profile = root / 'libreoffice_profile'
        command = [
            executable, '--headless',
            '-env:UserInstallation=file://%s' % profile,
            '--convert-to', 'xlsx', '--outdir', str(root), str(source),
        ]
        try:
            result = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, str(exc)
        generated = source.with_suffix('.xlsx')
        if result.returncode != 0 or not generated.is_file():
            return False, result.stdout.strip() or 'LibreOffice conversion failed'
        os.replace(str(generated), str(xlsx_path))
    return True, ''


def refresh_reports(directory, rows):
    """Refresh CSV, XLSX, and plotted HTML after each measurement."""
    csv_path = directory / 'perception_range_results.csv'
    xlsx_path = directory / 'perception_range_results.xlsx'
    html_path = directory / 'perception_range_plot.html'
    write_csv(csv_path, rows)
    write_html(html_path, rows)
    converted, reason = write_xlsx(csv_path, xlsx_path)
    return csv_path, xlsx_path if converted else None, html_path, reason
