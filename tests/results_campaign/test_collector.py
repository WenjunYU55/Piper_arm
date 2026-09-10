import yaml

from results_campaign.collector import (
    _deduplicate_sources,
    _frame_summary,
    _resolve_frame_artifact,
    _target_mask_summary,
)


def test_frame_summary_never_mixes_native_target_with_colour_transform(tmp_path):
    path = tmp_path / 'view_000_metadata.yaml'
    path.write_text(yaml.safe_dump({
        'frame_index': 0,
        'camera_transform': {
            'available': True,
            'child_frame_id': 'camera_color_optical_frame',
            'translation_m': [0.0, 0.0, 0.0],
            'matrix_4x4': [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        'target_3d': {
            'available': True,
            'header': {'frame_id': 'camera_color_optical_frame'},
            'point': {'x': 1.0, 'y': 2.0, 'z': 3.0},
        },
        'synchronized_target_3d': {
            'available': True,
            'header': {'frame_id': 'camera_depth_optical_frame'},
            'point': {'x': 9.0, 'y': 9.0, 'z': 9.0},
        },
    }), encoding='utf-8')
    summary = _frame_summary(path)
    assert summary['target_point_camera_m'] == [1.0, 2.0, 3.0]


def test_target_summary_preserves_acquisition_prompt_context(tmp_path):
    path = tmp_path / 'result.yaml'
    path.write_text('status: ok\n', encoding='utf-8')
    summary = _target_mask_summary({
        'request_id': 'acquire-0',
        'requested_target_label': 'green cube',
        'target_profile': 'green_cube',
        'target_prompt': 'green cube .',
        'target_source': 'grounding_dino',
        'obstacle_count': 2,
        'mission_context': {'request_reason': 'rough_acquisition'},
    }, path)
    assert summary['requested_target_label'] == 'green cube'
    assert summary['target_prompt'] == 'green cube .'
    assert summary['target_source'] == 'grounding_dino'
    assert summary['obstacle_count'] == 2


def test_capture_artifacts_are_resolved_and_bookmarked_without_mutation(
        tmp_path):
    dataset = tmp_path / 'scan'
    frames = dataset / 'frames'
    frames.mkdir(parents=True)
    depth = frames / 'target_depth.png'
    depth.write_bytes(b'depth-evidence')
    before = depth.read_bytes()
    assert _resolve_frame_artifact(dataset, 'target_depth.png') == depth
    sources = _deduplicate_sources([
        {'absolute_path': str(depth), 'sha256': 'a'},
        {'absolute_path': str(depth), 'sha256': 'a'},
    ])
    assert len(sources) == 1
    assert depth.read_bytes() == before
def test_planner_logs_filter_task_and_ignore_partial_json(tmp_path):
    import json
    from results_campaign.collector import _planner_log_events
    path = tmp_path / 'worker.log'
    rows = [
        {'task_id': 'other', 'metrics': {}},
        {'task_id': 'wanted', 'request_id': 'r', 'metrics': {'worker_request_wall_sec': 1.2}},
    ]
    path.write_text('noise\n' + ''.join(
        'CUROBO_REQUEST_TIMING ' + json.dumps(row) + '\n' for row in rows
    ) + 'CUROBO_REQUEST_TIMING {partial\n')
    events = _planner_log_events(path, 'wanted')
    assert len(events) == 1
    assert events[0]['request_id'] == 'r'


def test_collector_recovers_missed_terminal_callback_and_retry(tmp_path, monkeypatch):
    import json
    from results_campaign.campaign import CampaignStore
    from results_campaign import collector
    from piper_gui.view_model import MissionRequest
    store = CampaignStore(tmp_path, 'recovery')
    store.create_or_load()
    trial = store.load_next_trial()
    request = MissionRequest((trial['x_m'], trial['y_m'], trial['z_m']), 'cube', trial['backend'])
    store.record_submission('earlier', request)
    store.record_terminal('earlier', {'outcome': 'FAILED', 'safe_shutdown': True})
    store.retry_last_attempt()
    store.record_submission('latest', request)
    attempt = store.attempt_for_task('latest')
    spool = tmp_path / 'results'
    spool.mkdir()
    result = dict(task_id='latest', outcome='FAILED', failure_code='DEADLINE_EXPIRED',
                  safe_shutdown=True, capture_count=7, reason='mission deadline expired')
    source = spool / 'latest.json'
    source.write_text(json.dumps(result))
    original = source.read_bytes()
    monkeypatch.setattr(collector, 'MISSION_RESULTS_ROOT', spool)
    monkeypatch.setattr(collector, '_heavy_roots', lambda *args: [])
    assert store.trial_rows()[0]['status'] == 'In progress'
    collector.collect_task(tmp_path, attempt)
    terminal_path = attempt / 'terminal.json'
    terminal = json.loads(terminal_path.read_text())
    assert terminal['capture_count'] == 7
    assert terminal['failure_code'] == 'DEADLINE_EXPIRED'
    assert terminal['recovered_from_mission_result']['sha256']
    assert store.trial_rows()[0]['status'] == 'Done'
    assert store.trial_rows()[0]['outcome'] == 'FAILED'
    assert 'earlier' in store.superseded_tasks()
    assert not store.active_path.exists()
    assert source.read_bytes() == original
    before = terminal_path.read_bytes()
    collector.collect_task(tmp_path, attempt)
    assert terminal_path.read_bytes() == before
    assert json.loads((attempt / 'evidence.json').read_text())['terminal']['capture_count'] == 7


def test_terminal_recovery_rejects_foreign_and_unfinished_results(tmp_path):
    import json
    from results_campaign.campaign import CampaignStore
    from results_campaign.collector import _recover_terminal
    from piper_gui.view_model import MissionRequest
    store = CampaignStore(tmp_path, 'guards')
    store.create_or_load()
    trial = store.load_next_trial()
    submission = store.record_submission('wanted', MissionRequest(
        (trial['x_m'], trial['y_m'], trial['z_m']), 'cube', trial['backend']))
    attempt = store.attempt_for_task('wanted')
    source = tmp_path / 'result.json'
    for result in [dict(task_id='foreign', outcome='FAILED', safe_shutdown=True, capture_count=7),
                   dict(task_id='wanted', outcome='RUNNING', safe_shutdown=True, capture_count=7),
                   dict(task_id='wanted', outcome='FAILED'),
                   dict(task_id='wanted', outcome='FAILED', safe_shutdown=True, capture_count=-1)]:
        source.write_text(json.dumps(result))
        _recover_terminal(tmp_path, attempt, submission, result, source)
        assert not (attempt / 'terminal.json').exists()
        assert store.active_path.exists()
