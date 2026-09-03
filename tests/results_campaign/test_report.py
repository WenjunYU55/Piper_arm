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
    assert 'backend blocks' in definitions[0]['definition']
    readme = (output / 'README.md').read_text(encoding='utf-8')
    assert 'all cuRobo first and then all Tesseract' in readme
    assert 'confounds planner with elapsed time' in readme
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
