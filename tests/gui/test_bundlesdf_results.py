"""Offline BundleSDF GUI selection, provenance and process-boundary tests."""

import json
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
