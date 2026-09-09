"""Offline BundleSDF GUI selection, provenance and process-boundary tests."""

import json
import sys
import tkinter as tk
from types import SimpleNamespace

import pytest

from piper_gui import bundlesdf_results as module


def make_result(root, synthetic=False):
    (root / 'result').mkdir(parents=True)
    (root / 'run.json').write_text('{}')
    (root / 'result/textured_mesh.obj').write_text('v 0 0 0\n')
    if synthetic:
        (root / 'mesh-inspection.json').write_text(json.dumps({
            'evidence_class': 'SYNTHETIC_GPU_RECONSTRUCTION_NOT_PHYSICAL'}))
    return root


def test_synthetic_is_explicit_and_unknown_is_not_called_live(tmp_path):
    synthetic = make_result(tmp_path / 'synthetic', True)
    assert module.result_details(synthetic)[2] == 'Synthetic — not a live recording'
    assert module.result_details(synthetic / 'result/textured_mesh.obj')[0] == synthetic
    unknown = make_result(tmp_path / 'unknown')
    assert 'origin not verified' in module.result_details(unknown)[2]
    with pytest.raises(ValueError, match='output folder'):
        module.result_details(tmp_path)


def test_viewer_command_uses_isolated_python_and_literal_paths(tmp_path, monkeypatch):
    result = make_result(tmp_path / 'space ; dollar $ folder')
    python = tmp_path / 'python'
    python.touch()
    monkeypatch.setenv('PIPER_BUNDLESDF_PYTHON', str(python))
    command = module.viewer_command(tmp_path, result)
    assert command[:2] == [str(tmp_path / 'tools/bundlesdf/python_native.sh'),
                           str(tmp_path / 'reconstruction/view_bundlesdf.py')]
    assert command[2] == str(result / 'result/textured_mesh.obj')
    python.unlink()
    with pytest.raises(ValueError, match='setup_native.sh'):
        module.viewer_command(tmp_path, result)


def test_recorded_recovery_has_distinct_origin_label(tmp_path):
    result = make_result(tmp_path / 'recorded')
    (result / 'recovery.json').write_text(json.dumps(dict(
        evidence_class='RECORDED_ROS_BAG_CALLBACK_RECOVERY_NOT_DDS_OR_NEW_PHYSICAL_TEST',
        source_bag='/retained/recording/bag')))
    assert module.result_details(result)[2] == (
        'Recorded ROS RGB-D session — offline reconstruction')
    (result / 'mesh-inspection.json').write_text(json.dumps(dict(
        evidence_class='SYNTHETIC_GPU_RECONSTRUCTION_NOT_PHYSICAL')))
    assert module.result_details(result)[2] == 'Synthetic — not a live recording'


def test_dense_archive_command_preserves_literal_paths_and_partial_selection(tmp_path, monkeypatch):
    archive = tmp_path / 'recording ; with spaces.closed'
    archive.mkdir()
    (archive / 'recording.json').write_text(json.dumps(dict(
        state='closed', topic_counts={'/camera/color/image_raw': 1000})))
    monkeypatch.setenv('PIPER_BUNDLESDF_RUNTIME', str(tmp_path / 'runtime'))
    assert 'target tracking runs offline' in module.input_details(archive)[1]
    command, output, job = module.reconstruction_command(tmp_path, archive, True, 1)
    assert command[2] == str(archive)
    assert '--allow-incomplete' in command
    assert output != job
    assert module.reconstruction_command(tmp_path, archive, True, 1)[1] != output
    with pytest.raises(ValueError, match='positive'):
        module.reconstruction_command(tmp_path, archive, True, 0)


def test_active_mission_blocks_offline_gpu_job(tmp_path):
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('Tk display required')
    try:
        panel = module.BundleSDFResults(root, tmp_path, can_reconstruct=lambda: False)
        panel.reconstruct()
        assert panel.job_process is None
        assert 'Finish the active mission' in panel.status.get()
    finally:
        root.destroy()


def test_cancel_signals_only_owned_real_child_group(tmp_path, monkeypatch):
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('Tk display required')
    monkeypatch.setattr(module, 'reconstruction_command', lambda *a: (
        [sys.executable, '-c', 'import time; time.sleep(30)'], tmp_path / 'out', tmp_path / 'job'))
    panel = module.BundleSDFResults(root, tmp_path)
    process = None
    try:
        panel.reconstruct()
        process = panel.job_process
        assert process is not None
        panel.cancel_button.invoke()
        assert process.wait(timeout=5) != 0
        panel.after_cancel(panel.job_poll_id)
        panel.poll_job()
        assert panel.job_process is None
        assert 'cancelled' in panel.status.get()
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        root.destroy()


def test_progress_reports_mask_counts_without_promoting_pose_files(tmp_path):
    job = tmp_path / 'job'
    job.mkdir()
    (job / 'track_target_masks.log').write_text('SAM2 50/100 frames, 48 accepted, 2 rejected\n')
    assert '50/100 frames (48 usable, 2 rejected)' in module.job_progress(job, tmp_path, 'track_target_masks')
    poses = tmp_path / 'result/ob_in_cam'
    poses.mkdir(parents=True)
    (poses / '00000000.txt').touch()
    message = module.job_progress(job, tmp_path, 'reconstruct')
    assert '1 pose files written' in message
    assert 'tracking quality checked at completion' in message


def test_real_tk_button_launch_and_failure_reporting(tmp_path, monkeypatch):
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('Tk display required')
    runtime = tmp_path / 'runtime'
    result = make_result(runtime / 'evidence/demo', True)
    monkeypatch.setenv('PIPER_BUNDLESDF_RUNTIME', str(runtime))
    python = tmp_path / 'python'
    python.touch()
    monkeypatch.setenv('PIPER_BUNDLESDF_PYTHON', str(python))
    calls = []

    def start(command, **kwargs):
        calls.append((command, kwargs))
        kwargs['stdout'].write('display unavailable')
        return SimpleNamespace(poll=lambda: 2)

    monkeypatch.setattr(module.subprocess, 'Popen', start)
    try:
        panel = module.BundleSDFResults(root, tmp_path)
        panel.pack()
        root.update()
        assert panel.selection.get() == str(result)
        assert 'Synthetic' in panel.status.get()
        panel.open_button.invoke()
        assert len(calls) == 1
        assert 'shell' not in calls[0][1]
        assert calls[0][1]['start_new_session']
        panel.after_cancel(panel.poll_id)
        panel.poll_viewer()
        assert 'display unavailable' in panel.status.get()
        assert panel.process is None
        assert str(panel.open_button['state']) == 'normal'
    finally:
        root.destroy()


def test_selector_explains_eligibility_and_keeps_path_identity(tmp_path, monkeypatch):
    monkeypatch.setenv('PIPER_BUNDLESDF_RUNTIME', str(tmp_path / 'runtime'))
    archive = tmp_path / 'datasets/mission_recordings/opaque.closed'
    archive.mkdir(parents=True)
    metadata = dict(state='closed', task_id='task', started_unix_ns=1788787580021363932,
                    topic_counts={'/camera/color/image_raw': 4358, '/piper/reconstruction_keyframe': 3})
    (archive / 'recording.json').write_text(json.dumps(metadata))
    scan = tmp_path / 'datasets/active_scan/scan'
    scan.mkdir(parents=True)
    (scan / 'metadata.yaml').write_text('task_id: task\ntarget_label: green cube\n')
    mesh = make_result(tmp_path / 'runtime/evidence/comparison', True)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('Tk display required')
    try:
        panel = module.BundleSDFResults(root, tmp_path)
        label = next(label for label, path in panel.choices.items() if path == str(archive))
        assert 'green cube' in label and '4,358 RGB frames' in label
        assert 'Reconstruct input' in label
        panel.display_selection.set(label)
        panel.choose_selection()
        assert panel.selection.get() == str(archive)
        assert str(panel.build_button['state']) == 'normal'
        assert str(panel.open_button['state']) == 'disabled'
        panel.refresh()
        assert panel.selection.get() == str(archive)
        panel.selection.set(str(mesh))
        panel.show_selection()
        assert 'View only' in panel.display_selection.get()
        assert str(panel.build_button['state']) == 'disabled'
        assert str(panel.open_button['state']) == 'normal'
        metadata['topic_counts']['/piper/reconstruction_keyframe'] = 0
        (archive / 'recording.json').write_text(json.dumps(metadata))
        panel.selection.set(str(archive))
        panel.show_selection()
        assert 'No accepted reconstruction keyframe' in panel.status.get()
        assert str(panel.build_button['state']) == 'disabled'
        panel.selection.set(str(tmp_path / 'missing'))
        panel.show_selection()
        assert str(panel.open_button['state']) == 'disabled'
    finally:
        root.destroy()


def test_incomplete_output_and_bad_metadata_are_unavailable(tmp_path):
    (tmp_path / 'run.json').write_text('{}')
    title, detail, build, view = module.selection_details(str(tmp_path))
    assert 'Unavailable' in title and 'No exported mesh' in detail
    assert not build and not view
    (tmp_path / 'recording.json').write_text('[]')
    assert module.selection_details(str(tmp_path))[2:] == (False, False)
