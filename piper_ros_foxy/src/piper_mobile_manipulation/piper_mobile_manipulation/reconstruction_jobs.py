"""Pure durable contracts for delayed target-mesh reconstruction jobs."""

from pathlib import Path
import math
import time

from piper_mobile_manipulation.mission_core import sha256_value


MESH_STATES = frozenset((
    'WAITING_FOR_BASE_HOME', 'QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED'))


def reconstruction_terminal_decision(report):
    """Convert the worker's measured mesh quality into a durable job result."""
    if not isinstance(report, dict):
        raise ValueError('reconstruction quality report is missing')
    quality = str(report.get('overall_quality', '')).strip().upper()
    if quality == 'FAIL':
        return (
            'FAILED',
            'target mesh was generated but failed reconstruction quality '
            'validation; inspect the diagnostic mesh and quality report')
    if quality in ('PASS', 'WARN'):
        return (
            'SUCCEEDED',
            'target mesh reconstruction completed with %s quality; visual '
            'review remains required' % quality)
    raise ValueError('reconstruction worker returned an invalid quality result')


def mesh_job_id(task_id, manifest_sha256):
    task = str(task_id).strip()
    manifest = str(manifest_sha256).strip().lower()
    if not task or len(manifest) != 64:
        raise ValueError('mesh job requires task ID and manifest SHA-256')
    return 'mesh-' + sha256_value({
        'task_id': task, 'manifest_sha256': manifest})[:24]


def waiting_job(result):
    if not isinstance(result, dict):
        raise ValueError('mission result is missing')
    if result.get('outcome') != 'SUCCEEDED' or result.get('safe_shutdown') is not True:
        raise ValueError('mesh job requires a safely completed acquisition')
    task_id = str(result.get('task_id', '')).strip()
    dataset = Path(str(result.get('dataset_path', ''))).expanduser().resolve()
    manifest = str(result.get('manifest_sha256', '')).strip().lower()
    job_id = str(result.get('mesh_job_id', '')).strip()
    expected = mesh_job_id(task_id, manifest)
    if job_id != expected:
        raise ValueError('mission mesh job identity is inconsistent')
    if not dataset.is_dir():
        raise ValueError('mesh job dataset directory is missing')
    return {
        'task_id': task_id,
        'mesh_job_id': job_id,
        'state': 'WAITING_FOR_BASE_HOME',
        'reason': 'capture complete; waiting for matching tracked-robot home report',
        'dataset_path': str(dataset),
        'manifest_sha256': manifest,
        'mesh_path': '',
        'mesh_sha256': '',
        'quality_report': {},
        'updated_at_sec': time.time(),
    }


def validate_home_report(
        task_id, job_id, manifest_sha256, homed_at_sec, result,
        now_sec=None):
    if not isinstance(result, dict):
        raise ValueError('completed mission result is unavailable')
    if str(task_id) != str(result.get('task_id', '')):
        raise ValueError('home report task ID does not match mission result')
    expected_job = mesh_job_id(
        result.get('task_id', ''), result.get('manifest_sha256', ''))
    if str(job_id) != expected_job:
        raise ValueError('home report mesh job ID does not match mission result')
    if str(manifest_sha256).lower() != str(
            result.get('manifest_sha256', '')).lower():
        raise ValueError('home report manifest hash does not match mission result')
    if result.get('outcome') != 'SUCCEEDED' or result.get('safe_shutdown') is not True:
        raise ValueError('home report cannot start an incomplete acquisition')
    stamp = float(homed_at_sec)
    current = time.time() if now_sec is None else float(now_sec)
    if not math.isfinite(stamp) or stamp <= 0.0:
        raise ValueError('home report timestamp is invalid')
    if stamp > current + 1.0 or current - stamp > 600.0:
        raise ValueError('home report timestamp is stale or in the future')
    return expected_job


def transition_job(job, state, reason, **fields):
    if not isinstance(job, dict):
        raise ValueError('mesh job is missing')
    next_state = str(state).upper()
    if next_state not in MESH_STATES:
        raise ValueError('mesh job state is unsupported')
    current = str(job.get('state', '')).upper()
    allowed = {
        'WAITING_FOR_BASE_HOME': {'QUEUED'},
        'QUEUED': {'RUNNING', 'FAILED'},
        'RUNNING': {'SUCCEEDED', 'FAILED'},
        'SUCCEEDED': set(),
        'FAILED': set(),
    }
    if next_state != current and next_state not in allowed.get(current, set()):
        raise ValueError(
            'mesh job transition %s -> %s is invalid' % (current, next_state))
    updated = dict(job)
    updated.update(fields)
    updated['state'] = next_state
    updated['reason'] = str(reason)
    updated['updated_at_sec'] = time.time()
    return updated
