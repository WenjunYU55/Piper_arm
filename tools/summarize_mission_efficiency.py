#!/usr/bin/env python3
"""Summarize phase timings from saved target-scan action results."""

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motion_planning.mission_efficiency import (  # noqa: E402
    mission_efficiency_row, summarize_mission_rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'results', nargs='+', type=Path,
        help='Saved JSON result payloads or directories containing JSON')
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def input_paths(values):
    for value in values:
        if value.is_dir():
            yield from sorted(value.rglob('*.json'))
        else:
            yield value


def read_result(path):
    with path.open('r', encoding='utf-8') as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or 'outcome' not in value:
        raise ValueError('not a target-scan result payload')
    return value


def write_csv(path, rows):
    fields = [
        'source', 'task_id', 'planner_backend', 'outcome', 'safe_shutdown',
        'capture_count', 'timing_available', 'total_elapsed_sec',
        'seconds_per_capture', 'captures_per_minute', 'phase_totals_json',
    ]
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            value = dict(row)
            value['phase_totals_json'] = json.dumps(
                value.pop('phase_totals_sec'), sort_keys=True)
            writer.writerow(value)


def main():
    args = parse_args()
    rows = []
    skipped = []
    for path in input_paths(args.results):
        try:
            rows.append(mission_efficiency_row(
                read_result(path), source=str(path.resolve())))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            skipped.append({'source': str(path), 'reason': str(error)})
    report = {
        'schema_version': 1,
        'measurement_scope': 'saved result diagnostics only',
        'physical_motion_started': False,
        'rows': rows,
        'summary_by_planner_backend': summarize_mission_rows(rows),
        'skipped_inputs': skipped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    write_csv(args.output.with_suffix('.csv'), rows)
    print('wrote %d mission rows to %s' % (len(rows), args.output))
    if skipped:
        print('skipped %d non-result JSON files' % len(skipped))


if __name__ == '__main__':
    main()
