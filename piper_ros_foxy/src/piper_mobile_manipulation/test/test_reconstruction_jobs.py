import time

import pytest

from piper_mobile_manipulation.reconstruction_jobs import (
    mesh_job_id, reconstruction_terminal_decision, transition_job,
    validate_home_report, waiting_job)


def result(tmp_path):
    manifest = 'a' * 64
    return {
        'task_id': 'task-12345678',
        'outcome': 'SUCCEEDED',
        'safe_shutdown': True,
        'dataset_path': str(tmp_path),
        'manifest_sha256': manifest,
        'mesh_job_id': mesh_job_id('task-12345678', manifest),
    }


def test_waiting_job_and_home_report_are_correlated(tmp_path):
    completed = result(tmp_path)
    job = waiting_job(completed)
    assert job['state'] == 'WAITING_FOR_BASE_HOME'
    assert validate_home_report(
        completed['task_id'], completed['mesh_job_id'],
        completed['manifest_sha256'], time.time(), completed)
    queued = transition_job(job, 'QUEUED', 'base is home')
    running = transition_job(queued, 'RUNNING', 'worker started')
    assert transition_job(running, 'SUCCEEDED', 'done')['state'] == 'SUCCEEDED'


def test_home_report_rejects_cross_task_and_stale_events(tmp_path):
    completed = result(tmp_path)
    with pytest.raises(ValueError, match='task ID'):
        validate_home_report(
            'other-task', completed['mesh_job_id'],
            completed['manifest_sha256'], time.time(), completed)
    with pytest.raises(ValueError, match='stale'):
        validate_home_report(
            completed['task_id'], completed['mesh_job_id'],
            completed['manifest_sha256'], time.time() - 601.0, completed)


def test_terminal_mesh_job_cannot_restart(tmp_path):
    job = waiting_job(result(tmp_path))
    job = transition_job(job, 'QUEUED', 'queued')
    job = transition_job(job, 'FAILED', 'failed')
    with pytest.raises(ValueError, match='invalid'):
        transition_job(job, 'RUNNING', 'retry without a new job')


@pytest.mark.parametrize('quality', ['PASS', 'WARN'])
def test_usable_reconstruction_quality_completes_job(quality):
    state, reason = reconstruction_terminal_decision({
        'overall_quality': quality})
    assert state == 'SUCCEEDED'
    assert quality in reason


def test_failed_reconstruction_quality_fails_job_but_preserves_diagnostics():
    state, reason = reconstruction_terminal_decision({
        'overall_quality': 'FAIL'})
    assert state == 'FAILED'
    assert 'diagnostic mesh' in reason


def test_missing_reconstruction_quality_fails_closed():
    with pytest.raises(ValueError, match='quality'):
        reconstruction_terminal_decision({})
