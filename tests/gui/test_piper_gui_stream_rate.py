"""Next-mission rate selection without starting a GUI, ROS or hardware."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from piper_gui.stream_rate import (
    STREAM_RATE_LABELS, read_stream_rate, write_stream_rate,
)

ROOT = Path(__file__).resolve().parents[2]


def config(tmp_path):
    path = tmp_path / 'execution.yaml'
    path.write_text('/**:\n  ros__parameters:\n'
                    '    trajectory_command_rate_hz: 20.0 # commands\n'
                    '    executor_tick_rate_hz: 200.0 # timer\n'
                    '    speed_percent: 5.0\n'
                    '    enable_real_arm_motion: false\n')
    return path


@pytest.mark.parametrize('rate,tick', [(20, 200), (100, 200), (200, 400)])
def test_selection_updates_both_authoritative_values_only(tmp_path, rate, tick):
    from piper_mobile_manipulation.configuration import load_stream_rates
    path = config(tmp_path)
    original = path.read_text()
    frozen = load_stream_rates(path)
    saved = write_stream_rate(path, rate)
    assert read_stream_rate(path) == saved
    assert load_stream_rates(path) == dict(
        trajectory_command_rate_hz=rate, executor_tick_rate_hz=tick)
    assert frozen == dict(trajectory_command_rate_hz=20, executor_tick_rate_hz=200)
    assert path.read_text() == original.replace(
        'rate_hz: 20.0 # commands', 'rate_hz: %.1f # commands' % rate).replace(
        'rate_hz: 200.0 # timer', 'rate_hz: %.1f # timer' % tick)
    write_stream_rate(path, 20)
    assert path.read_text() == original


@pytest.mark.parametrize('value', [0, 50, 201, float('nan'), float('inf'), 'bad'])
def test_invalid_selection_cannot_modify_file(tmp_path, value):
    path = config(tmp_path)
    original = path.read_bytes()
    with pytest.raises(ValueError):
        write_stream_rate(path, value)
    assert path.read_bytes() == original


def test_duplicate_missing_and_invalid_saved_rates_fail_closed(tmp_path):
    path = config(tmp_path)
    original = path.read_text()
    for text in (original + 'trajectory_command_rate_hz: 100\n',
                 original.replace('executor_tick_rate_hz', 'wrong_key'),
                 original.replace('20.0 # commands', '50.0 # commands'),
                 original.replace('200.0 # timer', '10.0 # timer')):
        path.write_text(text)
        with pytest.raises(ValueError):
            read_stream_rate(path)


@pytest.mark.parametrize('active', [True, False])
def test_actual_gui_handler_blocks_active_mission(tmp_path, active):
    path = config(tmp_path)
    source = (ROOT / 'piper_gui/native_app.py').read_text()
    method = next(node for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.FunctionDef) and node.name == 'apply_stream_rate')
    namespace = dict(STREAM_RATE_LABELS=STREAM_RATE_LABELS, write_stream_rate=write_stream_rate)
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<GUI handler>', 'exec'), namespace)
    messages = []
    fake = SimpleNamespace(
        scan_aim_path=path,
        mission_view_model=SimpleNamespace(state=SimpleNamespace(can_start=not active)),
        stream_rate_var=SimpleNamespace(get=lambda: STREAM_RATE_LABELS[100]),
        stream_rate_status_var=SimpleNamespace(set=messages.append))
    namespace['apply_stream_rate'](fake)
    assert read_stream_rate(path).trajectory_command_rate_hz == (20 if active else 100)
    assert ('wait for the active mission' if active else '100 Hz') in messages[-1]
    assert "values=tuple(STREAM_RATE_LABELS.values()), state='readonly'" in source
