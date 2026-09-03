import csv
import json

import yaml

from results_campaign.campaign import (
    CampaignStore,
    alternating_trial_schedule,
    configuration_snapshot,
    default_trial_schedule,
)
from piper_gui.view_model import MissionRequest


def test_fixed_schedule_has_15_matched_positions_in_backend_blocks():
    trials = default_trial_schedule()
    assert len(trials) == 30
    pairs = {}
    for trial in trials:
        pairs.setdefault(trial.pair_index, []).append(trial)
    assert len(pairs) == 15
    assert all({item.backend for item in pair} == {'tesseract', 'curobo'} for pair in pairs.values())
    assert [item.backend for item in trials[:15]] == ['curobo'] * 15
    assert [item.backend for item in trials[15:]] == ['tesseract'] * 15
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
