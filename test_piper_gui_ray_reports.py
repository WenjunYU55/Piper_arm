from pathlib import Path

import pytest

from piper_gui.ray_reports import (
    list_ray_reports,
    open_ray_report,
    validated_ray_report,
)


def _report(project, name):
    path = (
        project / 'datasets' / 'active_scan' / 'ray_diagnostics' / name
        / 'ray_mission_diagnostics.html'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('<html></html>', encoding='utf-8')
    path.with_suffix('.json').write_text(
        '{"schema_version":2,"events":[],"generations":[]}\n',
        encoding='utf-8')
    return path


def test_gui_discovers_and_opens_a_validated_local_report(tmp_path):
    older = _report(tmp_path, 'mission-old')
    newer = _report(tmp_path, 'mission-new')
    older.touch()
    newer.touch()
    opened = []

    reports = list_ray_reports(tmp_path)
    report = open_ray_report(
        tmp_path, 'mission-new', opener=lambda url: opened.append(url) or True)

    assert set(reports) == {older.with_suffix('.json'), newer.with_suffix('.json')}
    assert report == newer
    assert opened == [newer.as_uri()]


def test_live_missions_are_listed_before_newer_historical_replays(tmp_path):
    mission = _report(tmp_path, 'mission-full').with_suffix('.json')
    replay = _report(tmp_path, 'replay_scan_newer').with_suffix('.json')
    mission.touch()
    replay.touch()

    reports = list_ray_reports(tmp_path)

    assert reports == [mission, replay]


def test_gui_report_selection_cannot_escape_diagnostics_root(tmp_path):
    with pytest.raises(ValueError, match='escapes'):
        validated_ray_report(tmp_path, '../../outside')


def test_native_gui_exposes_command_free_report_controls():
    source = Path('piper_gui_native.py').read_text(encoding='utf-8')

    assert 'Open Ray Review' in source
    assert 'Open 3D Ray Report' not in source
    assert 'Replay Recorded Data' in source
    assert 'replay_scan_dataset' in source
    assert 'self.refresh_ray_reports(preferred_report=result.task_id)' in source
