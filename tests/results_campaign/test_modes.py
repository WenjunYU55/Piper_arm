import csv
import json

import numpy as np
import pytest

from piper_gui.camera_profile import CAMERA_PROFILE_FPS
from piper_gui.view_model import MissionRequest
from results_campaign.campaign import CampaignStore
from results_campaign.metrics import cloud_dimension_metrics, summarize_evidence
from results_campaign.modes import mode_trials
from results_campaign.report import build_report


def settings():
    return {
        'backend': 'curobo', 'camera_profile': [640, 480, 30],
        'camera_profiles': [[w, h, fps] for (w, h), rates in CAMERA_PROFILE_FPS.items() for fps in rates],
        'aim_tolerance_deg': 5.0, 'height_m': 0.30, 'cube_label': 'green cube',
    }


@pytest.mark.parametrize('mode,count', [('camera_profiles', 8), ('aim_tolerance', 30), ('objects', 4)])
def test_sweep_schedule_and_resume_preserve_frozen_conditions(tmp_path, mode, count):
    store = CampaignStore(tmp_path, mode)
    definition = store.create_or_load(mode, settings())
    before = store.definition_path.read_bytes()
    trials = definition['trials']
    assert len(trials) == count
    assert {trial['backend'] for trial in trials} == {'curobo'}
    if mode == 'aim_tolerance':
        assert [t['aim_tolerance_deg'] for t in trials[:6]] == [1, 5, 10, 15, 20, 25]
        assert [trials[i]['x_m'] for i in range(0, 30, 6)] == [0.3, 0.5, 0.7, 0.9, 1.1]
        assert {t['z_m'] for t in trials} == {0.3}
    elif mode == 'camera_profiles':
        assert [t['camera_profile'] for t in trials] == settings()['camera_profiles']
        assert {(t['x_m'], t['y_m'], t['z_m']) for t in trials} == {(0.3, 0.0, 0.3)}
        assert {t['aim_tolerance_deg'] for t in trials} == {5}
    else:
        assert [t['object_key'] for t in trials] == ['cube', 'cup', 'stick', 'flower']
        assert {t['z_m'] for t in trials} == {0}
        assert trials[1]['reference_dimensions_mm'] is None
        assert {t['aim_tolerance_deg'] for t in trials} == {5}
    store.skip_next_trial()
    resumed = CampaignStore(tmp_path, mode)
    resumed.create_or_load('objects', dict(settings(), camera_profile=[1920, 1080, 15]))
    assert resumed.next_trial() == trials[1]
    assert resumed.definition_path.read_bytes() == before
    assert resumed.progress()['skipped'] == 1


def write_applied(root, profile, aim):
    path = root / 'L515_camera/rgb_profile.conf'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(','.join(str(x) for x in profile))
    path = root / 'piper_ros_foxy/src/piper_mobile_manipulation/config/scan_execution_params.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('executor:\n  ros__parameters:\n    final_capture_aim_tolerance_deg: %s\n' % aim)


def test_mismatched_applied_camera_aim_and_object_are_not_completed(tmp_path):
    store = CampaignStore(tmp_path, 'camera')
    trial = store.create_or_load('camera_profiles', settings())['trials'][0]
    request = MissionRequest((.3, 0, .3), 'green cube', 'curobo')
    write_applied(tmp_path, [640, 480, 30], 10)
    submission = store.record_submission('wrong', request)
    assert not submission['matches_schedule']
    assert len(submission['condition_mismatches']) == 2
    store.record_terminal('wrong', {'outcome': 'SUCCESS'})
    assert store.progress()['completed'] == 0
    write_applied(tmp_path, trial['camera_profile'], trial['aim_tolerance_deg'])
    submission = store.record_submission('right', request)
    assert submission['matches_schedule']
    store.record_terminal('right', {'outcome': 'SUCCESS'})
    assert store.progress()['completed'] == 1
    object_trial = mode_trials('objects', settings())[1]
    from results_campaign.modes import condition_mismatches
    assert condition_mismatches(object_trial, 'green cube', {
        'camera_profile': [640, 480, 30], 'aim_tolerance_deg': 5})


def test_non_cube_metrics_have_no_invented_reference(monkeypatch):
    cloud = np.arange(60).reshape(20, 3) / 1000.0
    metrics = cloud_dimension_metrics(cloud, None)
    assert metrics['observed_x_mm'] > 0
    assert 'known_x_mm' not in metrics
    assert 'mean_absolute_dimension_error_mm' not in metrics
    trial = mode_trials('objects', settings())[1]
    monkeypatch.setattr('results_campaign.metrics.frame_cloud_base', lambda *args: cloud)
    monkeypatch.setattr('results_campaign.metrics.mission_target_centre', lambda *args: np.zeros(3))
    rows = summarize_evidence({
        'submission': {'expected_trial': trial}, 'frames': [{}], 'dataset_path': '.',
        'planner_backend': 'curobo', 'task_id': 'cup', 'terminal': {},
    })
    assert rows['Runs'][0]['object_key'] == 'cup'
    assert rows['Coverage'][0]['cumulative_uncovered_fraction'] is None
    assert 'cumulative_coverage_fraction' not in rows['Coverage'][0]
    assert 'mean_absolute_dimension_error_mm' not in rows['PointCloud'][0]


def test_sweep_report_keeps_conditions_and_avoids_backend_grid(tmp_path):
    store = CampaignStore(tmp_path, 'objects')
    store.create_or_load('objects', settings())
    store.skip_next_trial()
    output = build_report(tmp_path, 'objects', tmp_path / 'report')
    with (output / 'scheduled_trials.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row['object_key'] for row in rows] == ['cube', 'cup', 'stick', 'flower']
    readme = (output / 'README.md').read_text()
    assert 'PHYSICAL_PARAMETER_SWEEP' in readme
    assert 'alternating physical missions' not in readme
    assert (output / '03_mission_duration.png').is_file()
    assert not (output / '03_backend_results_grid.png').exists()
    assert json.loads((output / 'report_manifest.json').read_text())['skipped_trials'] == 1
