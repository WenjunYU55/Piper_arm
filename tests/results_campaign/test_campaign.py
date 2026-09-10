import csv
import json

import yaml

from results_campaign.campaign import (
    CampaignStore,
    alternating_trial_schedule,
    blocked_trial_schedule,
    configuration_snapshot,
    default_trial_schedule,
)
from piper_gui.view_model import MissionRequest


def test_fixed_schedule_pairs_planners_and_alternates_first_at_each_position():
    trials = default_trial_schedule()
    assert len(trials) == 30
    pairs = {}
    for trial in trials:
        pairs.setdefault(trial.pair_index, []).append(trial)
    assert len(pairs) == 15
    assert all({item.backend for item in pair} == {'tesseract', 'curobo'} for pair in pairs.values())
    assert [item.sequence for item in trials] == list(range(1, 31))
    for index, pair in enumerate(pairs.values()):
        assert pair[0].coordinates == pair[1].coordinates
        expected = ['curobo', 'tesseract'] if index % 2 == 0 else ['tesseract', 'curobo']
        assert [item.backend for item in pair] == expected
        assert trials[index * 2:index * 2 + 2] == pair
    assert {item.x_m for item in trials} == {0.30, 0.50, 0.70, 0.90, 1.10}
    assert {item.z_m for item in trials} == {0.0, 0.12, 0.30}
    assert {item.y_m for item in trials} == {0.0}


def test_original_alternating_schedule_remains_load_compatible():
    trials = alternating_trial_schedule()
    assert [item.backend for item in trials[:4]] == [
        'tesseract', 'curobo', 'curobo', 'tesseract']


def test_campaign_records_only_matching_terminal_as_completed(tmp_path):
    store = CampaignStore(tmp_path, 'test-campaign')
    store.create_or_load()
    first = store.load_next_trial()
    request = MissionRequest(
        (first['x_m'], first['y_m'], first['z_m']), 'green cube', first['backend'])
    submission = store.record_submission('task-one', request, 100, 50)
    assert submission['matches_schedule'] is True
    assert store.progress()['completed'] == 0
    bookmarks = json.loads(
        (store.root / 'campaign_bookmarks.json').read_text(encoding='utf-8'))
    assert bookmarks['entries'][0]['state'] == 'SUBMITTED'
    assert len(bookmarks['entries'][0]['bookmark_sha256']) == 64
    store.record_terminal('task-one', {'outcome': 'FAILED', 'safe_shutdown': True})
    assert store.progress()['completed'] == 1
    with (store.root / 'campaign_bookmarks.csv').open(
            newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['state'] == 'TERMINAL'
    assert rows[0]['outcome'] == 'FAILED'
    assert rows[0]['safe_shutdown'] == 'True'


def test_mismatched_request_is_preserved_but_excluded(tmp_path):
    store = CampaignStore(tmp_path, 'test-campaign')
    store.create_or_load()
    store.load_next_trial()
    submission = store.record_submission(
        'task-wrong', MissionRequest((9.0, 0.0, 0.0), 'green cube', 'tesseract'))
    assert submission['matches_schedule'] is False
    store.record_terminal('task-wrong', {'outcome': 'FAILED'})
    assert store.progress()['completed'] == 0


def test_configuration_snapshot_reads_hash_bound_planner_qualification(tmp_path):
    curobo = tmp_path / 'motion_planning/curobo/model'
    curobo.mkdir(parents=True)
    (curobo / 'piper_collision_spheres.yaml').write_text(
        yaml.safe_dump({
            'qualification': {
                'hardware_qualified': True,
                'qualification_date': '2026-09-02',
                'scope': 'supervised_5_percent_target_scan',
                'basis': 'operator_reported_physical_e2e',
                'floor_profile': 'tabletop',
                'free_motion_speed_percent': 5.0,
                'contact_speed_percent': 5.0,
            },
        }),
        encoding='utf-8')
    snapshot = configuration_snapshot(tmp_path)
    model = snapshot['planner_models']['curobo']
    assert model['hardware_qualified'] is True
    assert model['qualification_scope'] == (
        'supervised_5_percent_target_scan')
    assert model['qualification_basis'] == 'operator_reported_physical_e2e'
    assert model['qualified_floor_profile'] == 'tabletop'
    assert model['qualified_free_motion_speed_percent'] == 5.0
    assert model['qualified_contact_speed_percent'] == 5.0


def test_existing_schedules_resume_without_rewriting_definitions(tmp_path):
    from dataclasses import asdict
    for schedule in (blocked_trial_schedule, alternating_trial_schedule, default_trial_schedule):
        store = CampaignStore(tmp_path, schedule.__name__)
        definition = store.create_or_load()
        definition['trials'] = [asdict(item) for item in schedule()]
        store.definition_path.write_text(json.dumps(definition))
        before = store.definition_path.read_bytes()
        first = store.load_next_trial()
        store.record_submission('first', MissionRequest(
            (first['x_m'], first['y_m'], first['z_m']), 'green cube', first['backend']))
        store.record_terminal('first', {'outcome': 'FAILED', 'safe_shutdown': True})
        loaded = CampaignStore(tmp_path, schedule.__name__)
        loaded.create_or_load()
        assert loaded.load_next_trial() == definition['trials'][1]
        assert store.definition_path.read_bytes() == before


def test_skips_persist_without_becoming_mission_results(tmp_path):
    store = CampaignStore(tmp_path, 'skip-campaign')
    definition = store.create_or_load()
    original = store.definition_path.read_bytes()
    first = store.load_next_trial()
    record = store.skip_next_trial()
    assert record['expected_trial'] == first
    resumed = CampaignStore(tmp_path, 'skip-campaign')
    assert resumed.load_next_trial() == definition['trials'][1]
    assert resumed.load_next_trial() == definition['trials'][1]
    assert resumed.progress()['completed'] == 0
    assert resumed.progress()['skipped'] == 1
    bookmark = resumed.refresh_bookmarks()['entries'][0]
    assert bookmark['state'] == 'SKIPPED'
    assert bookmark['included_in_comparison'] is False
    assert 'task_id' not in bookmark
    assert len(bookmark['skip_sha256']) == 64
    for _ in range(29):
        resumed.skip_next_trial()
    assert resumed.next_trial() is None
    assert resumed.skip_next_trial() is None
    assert resumed.progress()['remaining'] == 0
    assert resumed.progress()['completed'] == 0
    assert resumed.progress()['skipped'] == 30
    assert store.definition_path.read_bytes() == original
    assert list(store.task_attempts()) == []


def test_skip_blocks_pending_submission_even_after_reprepare(tmp_path):
    import pytest
    store = CampaignStore(tmp_path, 'skip-pending')
    trial = store.load_next_trial()
    store.record_submission('pending', MissionRequest(
        (trial['x_m'], trial['y_m'], trial['z_m']), 'cube', trial['backend']))
    store.load_next_trial()
    with pytest.raises(ValueError, match='submitted mission result'):
        store.skip_next_trial()
    assert store.skip_records() == []
    store.record_terminal('pending', {'outcome': 'FAILED', 'safe_shutdown': True})
    record = store.skip_next_trial()
    assert record['expected_trial']['sequence'] == 2
    assert store.progress()['completed'] == 1


def test_skip_rejects_record_for_different_trial(tmp_path):
    import pytest
    store = CampaignStore(tmp_path, 'invalid-skip')
    record = store.skip_next_trial()
    path = store.root / 'trials' / record['expected_trial']['trial_id'] / 'skip.json'
    record['expected_trial']['backend'] = 'invalid'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='invalid campaign skip'):
        store.next_trial()


def test_retry_keeps_failed_trial_and_prior_files_then_resumes_schedule(tmp_path):
    store = CampaignStore(tmp_path, 'retry')
    store.create_or_load()
    trial = store.load_next_trial()
    request = MissionRequest((trial['x_m'], trial['y_m'], trial['z_m']),
                             'green cube', trial['backend'])
    store.record_submission('original', request)
    old = store.record_terminal('original', {'outcome': 'FAILED'})
    original_bytes = old.read_bytes()
    following = store.next_trial()
    assert following != trial
    assert store.retry_last_attempt() == trial
    reopened = CampaignStore(tmp_path, 'retry')
    assert reopened.load_next_trial() == trial
    submission = reopened.record_submission('retry', request)
    assert submission['trial_id'] == trial['trial_id']
    assert submission['retry_of_task_id'] == 'original'
    assert not reopened.superseded_tasks()
    reopened.record_terminal('retry', {'outcome': 'SUCCEEDED'})
    assert reopened.superseded_tasks() == {'original'}
    assert reopened.next_trial() == following
    assert old.read_bytes() == original_bytes
    assert len(reopened.task_attempts()) == 2


def test_retry_ignores_mismatched_attempt_and_keeps_retry_on_mismatch(tmp_path):
    store = CampaignStore(tmp_path, 'retry-mismatch')
    store.create_or_load()
    trial = store.load_next_trial()
    request = MissionRequest((trial['x_m'], trial['y_m'], trial['z_m']),
                             'green cube', trial['backend'])
    store.record_submission('failed', request)
    store.record_terminal('failed', {'outcome': 'FAILED'})
    store.record_submission('wrong-slot', request)
    store.record_terminal('wrong-slot', {'outcome': 'FAILED'})
    assert store.retry_last_attempt() == trial
    wrong = MissionRequest((9, 0, 0), 'green cube', trial['backend'])
    store.record_submission('wrong-retry', wrong)
    store.record_terminal('wrong-retry', {'outcome': 'FAILED'})
    assert store.next_trial() == trial
    assert store.superseded_tasks() == set()


def test_retry_refuses_unresolved_submission_and_can_skip_pending_retry(tmp_path):
    import pytest
    store = CampaignStore(tmp_path, 'retry-active')
    store.create_or_load()
    trial = store.load_next_trial()
    request = MissionRequest((trial['x_m'], trial['y_m'], trial['z_m']),
                             'green cube', trial['backend'])
    store.record_submission('failed', request)
    with pytest.raises(ValueError, match='unresolved'):
        store.retry_last_attempt()
    store.record_terminal('failed', {'outcome': 'FAILED'})
    following = store.next_trial()
    store.retry_last_attempt()
    store.skip_next_trial()
    assert store.next_trial() == following
    assert store.progress()['skipped'] == 0
    assert store.progress()['completed'] == 1


def test_retry_last_attempt_accepts_success_and_no_history_is_clear(tmp_path):
    import pytest
    store = CampaignStore(tmp_path, 'retry-success')
    store.create_or_load()
    with pytest.raises(ValueError, match='no completed'):
        store.retry_last_attempt()
    trial = store.load_next_trial()
    request = MissionRequest((trial['x_m'], trial['y_m'], trial['z_m']),
                             'green cube', trial['backend'])
    store.record_submission('success', request)
    store.record_terminal('success', {'outcome': 'SUCCEEDED'})
    assert store.retry_last_attempt() == trial
    assert json.loads(store.active_path.read_text())['retry_of_task_id'] == 'success'


def test_selected_retry_older_trial_and_table_preserve_schedule(tmp_path):
    import pytest
    store = CampaignStore(tmp_path, 'selected-retry')
    store.create_or_load()
    first = store.load_next_trial()
    for task, outcome in [('older', 'SUCCEEDED'), ('newer', 'FAILED')]:
        trial = store.load_next_trial()
        request = MissionRequest((trial['x_m'], trial['y_m'], trial['z_m']),
                                 'green cube', trial['backend'])
        store.record_submission(task, request)
        store.record_terminal(task, {'outcome': outcome})
    rows = store.trial_rows()
    assert [r['status'] for r in rows[:3]] == ['Done', 'Done', 'To do']
    assert [r['outcome'] for r in rows[:2]] == ['SUCCEEDED', 'FAILED']
    assert store.retry_last_attempt(first['trial_id']) == first
    assert CampaignStore(tmp_path, 'selected-retry').load_next_trial() == first
    request = MissionRequest((first['x_m'], first['y_m'], first['z_m']),
                             'green cube', first['backend'])
    submitted = store.record_submission('selected-redo', request)
    assert submitted['retry_of_task_id'] == 'older'
    with pytest.raises(ValueError, match='unresolved'):
        store.retry_last_attempt(rows[1]['trial_id'])
    store.record_terminal('selected-redo', {'outcome': 'FAILED'})
    assert store.superseded_tasks() == {'older'}
    assert store.next_trial()['trial_id'] == rows[2]['trial_id']
    assert store.trial_rows()[0]['attempts'] == 2
    assert store.trial_rows()[0]['outcome'] == 'FAILED'
    with pytest.raises(ValueError, match='unknown'):
        store.retry_last_attempt('not-a-trial')
    with pytest.raises(ValueError, match='no completed'):
        store.retry_last_attempt(rows[2]['trial_id'])


def test_imported_unknown_submission_time_displays_done_and_allows_retry(tmp_path):
    store = CampaignStore(tmp_path, 'imported-time')
    store.create_or_load()
    trial = store.load_next_trial()
    request = MissionRequest((trial['x_m'], trial['y_m'], trial['z_m']),
                             'green cube', trial['backend'])
    store.record_submission('imported', request)
    store.record_terminal('imported', {'outcome': 'FAILED', 'capture_count': 8})
    path = store.attempt_for_task('imported') / 'submission.json'
    submission = json.loads(path.read_text())
    submission['submitted_wall_time_ns'] = None
    path.write_text(json.dumps(submission))
    row = store.trial_rows()[0]
    assert row['status'] == 'Done'
    assert row['outcome'] == 'FAILED'
    assert row['attempts'] == 1
    assert store.retry_last_attempt(trial['trial_id']) == trial
    store.record_submission('retry', request)
    store.record_terminal('retry', {'outcome': 'SUCCEEDED'})
    assert store.superseded_tasks() == {'imported'}
