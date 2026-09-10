"""Next-mission occlusion controls preserve both modes and unrelated settings."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from piper_gui.occlusion_policy import (
    OCCLUSION_POLICY_LABELS, read_occlusion_policy, write_occlusion_policy,
)


def config(tmp_path):
    path = tmp_path / 'workflow.yaml'
    path.write_text('/**:\n  ros__parameters:\n'
                    '    occlusion_probe_every_viewpoint: false # choice\n'
                    '    max_views: 13\n')
    return path


def test_roundtrip_changes_only_probe_choice(tmp_path):
    path = config(tmp_path)
    original = path.read_text()
    assert read_occlusion_policy(path) is False
    write_occlusion_policy(path, True)
    assert read_occlusion_policy(path) is True
    assert path.read_text() == original.replace(': false', ': true')
    write_occlusion_policy(path, False)
    assert path.read_text() == original


@pytest.mark.parametrize('value', [None, 'false', 1, 0])
def test_invalid_selection_leaves_config_intact(tmp_path, value):
    path = config(tmp_path)
    original = path.read_bytes()
    with pytest.raises(ValueError):
        write_occlusion_policy(path, value)
    assert path.read_bytes() == original


@pytest.mark.parametrize('active', [True, False])
@pytest.mark.parametrize('choice', [True, False])
def test_gui_apply_and_active_mission_guard(tmp_path, active, choice):
    path = config(tmp_path)
    write_occlusion_policy(path, not choice)
    source = Path('piper_gui/native_app.py').read_text()
    method = next(node for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.FunctionDef) and node.name == 'apply_occlusion_policy')
    namespace = dict(OCCLUSION_POLICY_LABELS=OCCLUSION_POLICY_LABELS,
                     write_occlusion_policy=write_occlusion_policy)
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<GUI handler>', 'exec'), namespace)
    messages = []
    fake = SimpleNamespace(
        occlusion_policy_path=path,
        mission_view_model=SimpleNamespace(state=SimpleNamespace(can_start=not active)),
        occlusion_policy_var=SimpleNamespace(get=lambda: OCCLUSION_POLICY_LABELS[choice]),
        occlusion_policy_status_var=SimpleNamespace(set=messages.append))
    namespace['apply_occlusion_policy'](fake)
    assert read_occlusion_policy(path) is (not choice if active else choice)
    assert ('wait for the active mission' if active else 'Saved for next mission') in messages[-1]
