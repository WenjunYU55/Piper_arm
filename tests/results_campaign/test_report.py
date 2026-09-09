import json
import csv
from pathlib import Path
import zipfile

from results_campaign.campaign import CampaignStore
from results_campaign.report import build_report
from piper_gui.view_model import MissionRequest


def test_empty_campaign_report_has_parseable_tables_workbook_and_figures(tmp_path):
    project = tmp_path / 'project'
    project.mkdir()
    CampaignStore(project, 'empty-campaign').create_or_load()
    output = build_report(project, 'empty-campaign', tmp_path / 'report')
    with (output / 'definitions.csv').open(
            newline='', encoding='utf-8') as stream:
        definitions = list(csv.DictReader(stream))
    assert definitions[0]['metric'] == 'Evidence class'
    assert 'MATCHED_RUNS' in definitions[0]['definition']
    assert 'order alternates' in definitions[0]['definition']
    readme = (output / 'README.md').read_text(encoding='utf-8')
    assert 'alternating physical missions' in readme
    assert 'all cuRobo first and then all Tesseract' not in readme
    with (output / 'runs.csv').open(newline='', encoding='utf-8') as stream:
        assert list(csv.DictReader(stream)) == [{'note': 'No data recorded yet.'}]
    with (output / 'evidence_manifest.csv').open(
            newline='', encoding='utf-8') as stream:
        assert list(csv.DictReader(stream)) == [
            {'note': 'No data recorded yet.'}]
    assert (output / 'campaign_bookmarks.csv').is_file()
    assert (output / 'campaign_bookmarks.json').is_file()
    with zipfile.ZipFile(output / 'PiPER_results_campaign.xlsx') as archive:
        assert 'xl/workbook.xml' in archive.namelist()
        assert archive.testzip() is None
    for stem in (
            '01_acquisition_confidence', '02_coverage_by_capture',
            '03_backend_results_grid', '04_runtime_by_capture',
            '05_pointcloud_reconstruction_accuracy'):
        assert (output / (stem + '.png')).stat().st_size > 1000
        assert (output / (stem + '.pdf')).stat().st_size > 1000
    manifest = json.loads(
        (output / 'report_manifest.json').read_text(encoding='utf-8'))
    assert manifest['schema_version'] == 1
    assert manifest['campaign_bookmark_entry_count'] == 0
    assert any(
        item['path'] == 'PiPER_results_campaign.xlsx'
        and len(item['sha256']) == 64
        for item in manifest['artifacts'])


def test_recorder_and_analysis_modules_have_no_ros_or_command_topics():
    root = Path(__file__).resolve().parents[2]
    for relative in (
            'results_campaign/campaign.py', 'results_campaign/collector.py',
            'results_campaign/bookmarks.py',
            'results_campaign/recorder.py', 'results_campaign/metrics.py',
            'results_campaign/report.py'):
        text = (root / relative).read_text(encoding='utf-8')
        assert 'import rclpy' not in text
        assert '/joint_ctrl_single' not in text
        assert 'create_publisher' not in text
        assert 'create_client' not in text


def test_report_manifest_bookmarks_included_attempt_sources(tmp_path):
    project = tmp_path / 'project'
    project.mkdir()
    store = CampaignStore(project, 'bookmarked-campaign')
    store.create_or_load()
    trial = store.load_next_trial()
    store.record_submission(
        'bookmark-source-task', MissionRequest(
            (trial['x_m'], trial['y_m'], trial['z_m']),
            'green cube', trial['backend']))
    store.record_terminal('bookmark-source-task', {
        'outcome': 'FAILED', 'safe_shutdown': True, 'capture_count': 0,
    })
    output = build_report(
        project, 'bookmarked-campaign', tmp_path / 'bookmarked-report')
    with (output / 'evidence_manifest.csv').open(
            newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert {row['task_id'] for row in rows} == {'bookmark-source-task'}
    assert {row['included_in_comparison'] for row in rows} == {'True'}
    assert all(len(row['source_sha256']) == 64 for row in rows)
    bookmarks = json.loads(
        (output / 'campaign_bookmarks.json').read_text(encoding='utf-8'))
    entry = bookmarks['entries'][0]
    assert entry['state'] == 'EVIDENCE_CAPTURED'
    assert entry['included_in_comparison'] is True
    assert len(entry['evidence_sha256']) == 64


def test_skipped_trials_export_separately_without_measured_runs(tmp_path):
    import hashlib
    store = CampaignStore(tmp_path, 'skipped-report')
    record = store.skip_next_trial()
    trial_id = record['expected_trial']['trial_id']
    source = store.root / 'trials' / trial_id / 'skip.json'
    before = source.read_bytes()
    output = build_report(tmp_path, 'skipped-report', tmp_path / 'report')
    with (output / 'skipped_trials.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]['trial_id'] == trial_id
    assert rows[0]['evidence_class'] == 'OPERATOR_SKIP_NOT_A_MISSION'
    with (output / 'runs.csv').open(newline='') as stream:
        assert list(csv.DictReader(stream)) == [{'note': 'No data recorded yet.'}]
    manifest = json.loads((output / 'report_manifest.json').read_text())
    assert manifest['completed_trials'] == 0
    assert manifest['skipped_trials'] == 1
    assert manifest['remaining_trials'] == 29
    assert (output / 'skipped_trials' / (trial_id + '.json')).read_bytes() == before
    assert source.read_bytes() == before
    with zipfile.ZipFile(output / 'PiPER_results_campaign.xlsx') as archive:
        assert archive.testzip() is None
        assert b'Skipped_Trials' in archive.read('xl/workbook.xml')
    for artifact in manifest['artifacts']:
        path = output / artifact['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact['sha256']


def test_completed_retry_replaces_summary_but_preserves_excluded_evidence(tmp_path):
    store = CampaignStore(tmp_path, 'retry-report')
    store.create_or_load()
    trial = store.load_next_trial()
    request = MissionRequest((trial['x_m'], trial['y_m'], trial['z_m']),
                             'green cube', trial['backend'])
    store.record_submission('old-attempt', request)
    store.record_terminal('old-attempt', {'outcome': 'SUCCEEDED', 'capture_count': 1})
    store.retry_last_attempt()
    store.record_submission('new-attempt', request)
    store.record_terminal('new-attempt', {'outcome': 'FAILED', 'capture_count': 0})
    output = build_report(tmp_path, 'retry-report', tmp_path / 'retry-report-output')
    with (output / 'runs.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert {row['task_id'] for row in rows} == {'new-attempt'}
    with (output / 'exclusions.csv').open(newline='') as stream:
        excluded = list(csv.DictReader(stream))
    assert any(row['task_id'] == 'old-attempt' and 'Superseded' in row['reason']
               for row in excluded)
    assert len(store.task_attempts()) == 2
