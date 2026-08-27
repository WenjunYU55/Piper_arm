from datetime import timezone
from pathlib import Path

import pytest

from piper_gui.ray_reports import (
    list_ray_reports,
    open_ray_report,
    ray_report_display_name,
    ray_report_root,
    ray_report_selection,
    validated_ray_report,
)


def test_future_reports_use_top_level_datasets_root(tmp_path):
    assert ray_report_root(tmp_path) == (
        tmp_path / 'datasets' / 'ray_diagnostics')


def _report(
        project, name, basename='RayProcesses - 14:35 - 27-08-2026',
        legacy=False):
    relative_root = (
        Path('datasets/active_scan/ray_diagnostics')
        if legacy else Path('datasets/ray_diagnostics'))
    path = (
        project / relative_root / name
        / (basename + '.html')
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

    assert set(reports) == {
        older.with_suffix('.json'), newer.with_suffix('.json')}
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


def test_gui_still_opens_a_legacy_internal_filename(tmp_path):
    legacy = _report(
        tmp_path, 'legacy-mission', basename='ray_mission_diagnostics')

    assert validated_ray_report(
        tmp_path, 'legacy-mission', suffix='.html') == legacy


def test_gui_discovers_and_opens_reports_from_legacy_root(tmp_path):
    legacy = _report(tmp_path, 'old-location', legacy=True)

    reports = list_ray_reports(tmp_path)
    selection = ray_report_selection(tmp_path, reports, 0)

    assert reports == [legacy.with_suffix('.json')]
    assert selection == 'legacy:old-location'
    assert validated_ray_report(
        tmp_path, selection, suffix='.html') == legacy


def test_root_qualified_selection_disambiguates_duplicate_ids(tmp_path):
    current = _report(tmp_path, 'same-id')
    legacy = _report(tmp_path, 'same-id', legacy=True)
    reports = list_ray_reports(tmp_path)
    selections = {
        ray_report_selection(tmp_path, reports, index): report
        for index, report in enumerate(reports)
    }

    assert validated_ray_report(
        tmp_path, 'current:same-id', suffix='.html') == current
    assert validated_ray_report(
        tmp_path, 'legacy:same-id', suffix='.html') == legacy
    assert set(selections) == {'current:same-id', 'legacy:same-id'}


def test_ray_report_operator_name_uses_first_event_to_minute(tmp_path):
    path = tmp_path / 'ray_mission_diagnostics.json'
    path.write_text(
        '{"events":['
        '{"timestamp_ns":1787841359000000000},'
        '{"timestamp_ns":1787841419000000000}]}'
        '\n', encoding='utf-8')

    assert ray_report_display_name(path, timezone.utc) == (
        'RayProcesses - 14:35 - 27-08-2026')


def test_duplicate_minute_labels_keep_unique_internal_selections(tmp_path):
    first = _report(tmp_path, 'mission-first').with_suffix('.json')
    second = _report(tmp_path, 'mission-second').with_suffix('.json')
    evidence = '{"events":[{"timestamp_ns":1787841359000000000}]}\n'
    first.write_text(evidence, encoding='utf-8')
    second.write_text(evidence, encoding='utf-8')
    reports = list_ray_reports(tmp_path)

    assert len({ray_report_display_name(
        path, timezone.utc) for path in reports}) == 1
    assert {ray_report_selection(tmp_path, reports, 0),
            ray_report_selection(tmp_path, reports, 1)} == {
                'current:mission-first', 'current:mission-second'}
    with pytest.raises(ValueError, match='no ray report'):
        ray_report_selection(tmp_path, reports, -1)


def test_native_gui_exposes_command_free_report_controls():
    source = Path('piper_gui_native.py').read_text(encoding='utf-8')

    assert 'Open Ray Review' in source
    assert 'Open 3D Ray Report' not in source
    assert 'Replay Recorded Data' in source
    assert 'replay_scan_dataset' in source
    assert (
        'self.refresh_ray_reports(preferred_report=result.task_id)' in source)
    assert 'ray_report_display_name(path)' in source
    selection_source = (
        'ray_report_selection(\n'
        '                    PROJECT_ROOT, self.ray_report_paths, '
        'selected_index)')
    assert selection_source in source
    viewer = Path('piper_gui/ray_review_viewer.py').read_text(encoding='utf-8')
    assert 'ray_report_display_name(source)' in viewer
