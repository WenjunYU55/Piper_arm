"""Offline orchestration tests with explicitly mocked child commands."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/bundlesdf'))
from reconstruct_job import run_job  # noqa: E402
import reconstruct_job  # noqa: E402


def test_archive_uses_dense_extraction_and_sam2_before_validation(tmp_path, monkeypatch):
    archive = tmp_path / 'archive with spaces'
    (archive / 'bag').mkdir(parents=True)
    (archive / 'recording.json').write_text('{}')
    (archive / 'bag/metadata.yaml').write_text('{}')
    job, output = tmp_path / 'job', tmp_path / 'result'
    calls = []

    def child(command, **kwargs):
        calls.append(command)
        if 'extract_video.py' in ' '.join(command):
            root = Path(command[-1])
            root.mkdir()
            (root / 'extraction.json').write_text(json.dumps(dict(complete=False, frames=[{}]*100)))
        elif 'segment_video.py' in ' '.join(command):
            root = Path(command[-1])
            root.mkdir()
            (root / 'manifest.json').write_text(json.dumps(dict(frames=[{}]*100)))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(reconstruct_job.subprocess, 'run', child)
    run_job(archive, output, job, allow_incomplete=True, validate_only=True)
    assert len(calls) == 3
    assert calls[0][-2] == str(archive / 'bag')
    assert Path(calls[0][-3]).name == 'extract_video.py'
    assert Path(calls[1][0]).name == 'python_sam2.sh'
    assert '--validate-only' in calls[2]
    assert 'import_bag.py' not in str(calls)
    assert json.loads((job / 'job.json').read_text())['stage'] == 'validated'


def test_stride_failure_and_child_failure_are_durable(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'manifest.json').write_text(json.dumps(dict(frames=[{}]*6)))
    with pytest.raises(ValueError, match='fewer than five'):
        run_job(source, tmp_path / 'out', tmp_path / 'bad-stride', stride=2)
    assert json.loads((tmp_path / 'bad-stride/job.json').read_text())['state'] == 'failed'
    monkeypatch.setattr(reconstruct_job.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=2))
    with pytest.raises(RuntimeError, match='validate_input failed'):
        run_job(source, tmp_path / 'out', tmp_path / 'bad-child')
    assert json.loads((tmp_path / 'bad-child/job.json').read_text())['state'] == 'failed'


def test_cancelled_child_retains_cancelled_status(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'manifest.json').write_text(json.dumps(dict(frames=[{}]*6)))

    def stop(*args, **kwargs):
        raise KeyboardInterrupt('test cancellation')

    monkeypatch.setattr(reconstruct_job.subprocess, 'run', stop)
    with pytest.raises(KeyboardInterrupt):
        run_job(source, tmp_path / 'out', tmp_path / 'job')
    assert json.loads((tmp_path / 'job/job.json').read_text())['state'] == 'cancelled'
