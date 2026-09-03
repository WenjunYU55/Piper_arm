"""Durable campaign-wide indexes for passive physical result evidence."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List

from .campaign import atomic_write_json, load_json, sha256_file, utc_now


BOOKMARK_SCHEMA_VERSION = 1
BOOKMARK_COLUMNS = (
    'bookmark_id', 'campaign_id', 'trial_id', 'pair_index', 'task_id',
    'state', 'planner_backend', 'target_x_m', 'target_y_m', 'target_z_m',
    'submitted_utc', 'terminal_recorded_utc', 'evidence_collected_utc',
    'outcome', 'failure_code', 'safe_shutdown', 'accepted_captures',
    'matches_schedule', 'included_in_comparison', 'exclusion_reason',
    'dataset_path', 'dataset_manifest_sha256', 'ray_diagnostics_path',
    'ray_diagnostics_sha256', 'first_acquisition_path',
    'first_acquisition_sha256', 'evidence_path', 'evidence_sha256',
    'source_file_count', 'source_manifest_sha256', 'code_branch',
    'code_commit', 'code_dirty', 'configuration_captured_utc',
    'collision_model_hardware_qualified',
    'collision_model_qualification_scope', 'bookmark_sha256',
)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _source_for_suffix(
        sources: Iterable[Dict[str, Any]], suffix: str) -> Dict[str, Any]:
    return next((
        source for source in sources
        if str(source.get('path', '')).endswith(suffix)), {})


def _bookmark_entry(attempt: Path) -> Dict[str, Any]:
    submission = load_json(attempt / 'submission.json', {}) or {}
    terminal = load_json(attempt / 'terminal.json', {}) or {}
    failure = load_json(attempt / 'submission_failure.json', {}) or {}
    evidence_path = attempt / 'evidence.json'
    evidence = load_json(evidence_path, {}) or {}
    expected = submission.get('expected_trial', {})
    configuration = submission.get('configuration_snapshot', {})
    backend = str(
        evidence.get('planner_backend')
        or submission.get('requested_backend', ''))
    planner_model = configuration.get(
        'planner_models', {}).get(backend, {})
    sources = evidence.get('sources', [])
    if not isinstance(sources, list):
        sources = []
    source_manifest = sorted((
        {
            'path': str(source.get('path', '')),
            'sha256': str(source.get('sha256', '')),
            'size_bytes': source.get('size_bytes'),
        }
        for source in sources if isinstance(source, dict)),
        key=lambda source: source['path'])
    dataset_source = _source_for_suffix(sources, '/manifest.json')
    ray_source = next((
        source for source in sources
        if '/ray_diagnostics/' in str(source.get('path', ''))
        and str(source.get('path', '')).endswith('.json')), {})
    acquisition = evidence.get('first_acquisition', {})
    matches = (
        evidence.get('matches_schedule')
        if evidence else submission.get('matches_schedule'))
    if failure:
        state = 'SUBMISSION_FAILED'
    elif evidence and terminal:
        state = 'EVIDENCE_CAPTURED'
    elif terminal:
        state = 'TERMINAL'
    else:
        state = 'SUBMITTED'
    exclusion = ''
    if failure:
        exclusion = str(failure.get(
            'exclusion_reason', 'mission action submission failed'))
    elif matches is False:
        errors = evidence.get('consistency_errors', [])
        exclusion = '; '.join(str(value) for value in errors) or (
            'submitted coordinates or planner backend did not match schedule')
    entry = {
        'bookmark_id': '%s/%s/%s' % (
            submission.get('campaign_id', ''),
            submission.get('trial_id', ''), submission.get('task_id', '')),
        'campaign_id': submission.get('campaign_id'),
        'trial_id': submission.get('trial_id'),
        'pair_index': expected.get('pair_index'),
        'task_id': submission.get('task_id'),
        'state': state,
        'planner_backend': backend,
        'target_x_m': expected.get('x_m'),
        'target_y_m': expected.get('y_m'),
        'target_z_m': expected.get('z_m'),
        'submitted_utc': submission.get('submitted_utc'),
        'terminal_recorded_utc': terminal.get('recorded_utc'),
        'evidence_collected_utc': evidence.get('collected_utc'),
        'outcome': terminal.get('outcome'),
        'failure_code': terminal.get('failure_code'),
        'safe_shutdown': terminal.get('safe_shutdown'),
        'accepted_captures': terminal.get('capture_count'),
        'matches_schedule': matches,
        'included_in_comparison': bool(
            evidence and terminal and matches is True),
        'exclusion_reason': exclusion,
        'dataset_path': evidence.get('dataset_path', ''),
        'dataset_manifest_sha256': dataset_source.get('sha256', ''),
        'ray_diagnostics_path': ray_source.get('path', ''),
        'ray_diagnostics_sha256': ray_source.get('sha256', ''),
        'first_acquisition_path': acquisition.get('result_path', ''),
        'first_acquisition_sha256': acquisition.get('result_sha256', ''),
        'evidence_path': str(evidence_path.resolve()) if evidence else '',
        'evidence_sha256': sha256_file(evidence_path)
        if evidence_path.is_file() else '',
        'source_file_count': len(source_manifest),
        'source_manifest_sha256': _canonical_hash(source_manifest),
        'code_branch': configuration.get('git', {}).get('branch'),
        'code_commit': configuration.get('git', {}).get('commit'),
        'code_dirty': configuration.get('git', {}).get('dirty'),
        'configuration_captured_utc': configuration.get('captured_utc'),
        'collision_model_hardware_qualified': planner_model.get(
            'hardware_qualified'),
        'collision_model_qualification_scope': planner_model.get(
            'qualification_scope'),
    }
    entry['bookmark_sha256'] = _canonical_hash(entry)
    return entry


def _atomic_write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + '.', suffix='.partial', dir=str(path.parent))
    try:
        with os.fdopen(descriptor, 'w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=BOOKMARK_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in BOOKMARK_COLUMNS})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(path))
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def write_campaign_bookmarks(
        project_root: Path, campaign_root: Path) -> Dict[str, Any]:
    """Atomically rebuild the human-readable campaign bookmark ledger."""
    root = Path(project_root).resolve()
    campaign = Path(campaign_root).resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    lock_path = campaign / '.bookmarks.lock'
    with lock_path.open('a+', encoding='utf-8') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        attempts = sorted(campaign.glob('trials/*/attempts/*'))
        entries = [
            _bookmark_entry(attempt) for attempt in attempts
            if (attempt / 'submission.json').is_file()]
        definition = campaign / 'campaign.json'
        document = {
            'schema_version': BOOKMARK_SCHEMA_VERSION,
            'generated_utc': utc_now(),
            'campaign_id': load_json(
                definition, {}).get('campaign_id', campaign.name),
            'project_root': str(root),
            'campaign_definition_path': str(definition),
            'campaign_definition_sha256': (
                sha256_file(definition) if definition.is_file() else ''),
            'entry_count': len(entries),
            'entries': entries,
        }
        atomic_write_json(campaign / 'campaign_bookmarks.json', document)
        _atomic_write_csv(campaign / 'campaign_bookmarks.csv', entries)
        return document
