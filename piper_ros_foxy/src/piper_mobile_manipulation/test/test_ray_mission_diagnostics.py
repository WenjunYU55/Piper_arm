import json

import numpy as np
import yaml

from piper_mobile_manipulation.ray_mission_diagnostics import (
    add_capture_event,
    add_bridge_request,
    add_prequalification,
    add_tesseract_response,
    add_target_update_event,
    add_terminal_event,
    capture_event_identity,
    historical_replay_snapshot,
    planner_generation_snapshot,
    RayMissionDiagnosticsStore,
    replay_historical_dataset,
)


def ray(ray_id, direction):
    return {
        'index': ray_id,
        'ray_id': ray_id,
        'ray_direction': {
            'x': direction[0], 'y': direction[1], 'z': direction[2]},
        'desired_camera_position': {
            'x': direction[0] * 0.4,
            'y': direction[1] * 0.4,
            'z': direction[2] * 0.4,
        },
        'ray_min_standoff_m': 0.28,
        'ray_max_standoff_m': 0.80,
        'ray_scoring_standoff_m': 0.40,
    }


def snapshot():
    generated = [
        ray(0, [1.0, 0.0, 0.0]),
        ray(1, [0.0, 1.0, 0.0]),
        ray(2, [0.0, 0.0, 1.0]),
    ]
    ranked = [
        dict(generated[2], nbv_rank=1, nbv_positive_information_gain=True,
             nbv_marginal_information_fraction=0.25),
        dict(generated[1], nbv_rank=2, nbv_positive_information_gain=False),
    ]
    return planner_generation_snapshot(
        'mission-a', 2, {'x': 0.2, 'y': -0.1, 'z': 0.3}, 'base_link',
        'ray_nbv', generated, generated[1:], ranked, [ranked[0]],
        planner_rejections={0: ['duplicate of an accepted camera pose']},
        selection_ready=True, remaining_views=11)


def test_planner_snapshot_retains_every_generated_ray_and_exact_rank():
    value = snapshot()
    rays = {item['ray_id']: item for item in value['rays']}

    assert value['generated_ray_count'] == 3
    assert value['ray_population_complete'] is True
    assert len(value['generated_ray_population']) == 3
    assert len(value['ray_population_sha256']) == 64
    assert rays[0]['planner_status'] == 'culled_history'
    assert rays[0]['planner_reasons'] == [
        'duplicate of an accepted camera pose']
    assert rays[1]['planner_status'] == 'culled_information'
    assert rays[1]['rank'] == 2
    assert rays[2]['planner_status'] == 'remaining'
    assert rays[2]['rank'] == 1
    assert rays[2]['elevation_deg'] == 90.0


def test_prequalification_bridge_and_worker_evidence_are_correlated_by_ray():
    value = snapshot()
    prequalified = [
        dict(ray(1, [0.0, 1.0, 0.0]), prequalified=False, reachable=False,
             safe=False, reject_reasons=['CAPABILITY_MAP_NO_SUPPORT']),
        dict(
            ray(2, [0.0, 0.0, 1.0]),
            prequalified=True, reachable=True, safe=True, reject_reasons=[],
            ray_capability_bounded=True,
            ray_requested_min_standoff_m=0.28,
            ray_requested_max_standoff_m=0.80,
            ray_min_standoff_m=0.30,
            ray_max_standoff_m=0.54,
            ray_scoring_standoff_m=0.34,
            ray_capability_intervals_m=[[0.30, 0.34], [0.50, 0.54]],
            desired_camera_position={'x': 0.0, 'y': 0.0, 'z': 0.34}),
    ]
    value = add_prequalification(value, prequalified, {'safe_viewpoints': 1})
    value = add_bridge_request(value, 'request-1', [2])
    response = {
        'request_id': 'request-1',
        'status': 'failure',
        'selected_viewpoints': [],
        'rejection_codes': ['TESSERACT_EXHAUSTED'],
        'diagnostic': 'no endpoint',
        'planning_diagnostics': {
            'attempted_ray_ids': [2],
            'candidate_failures': [{
                'id': 200,
                'stage': 'IK_FAILED',
                'detail': 'no exact solution',
                'permanent_endpoint_failure': True,
            }],
        },
    }
    request = {'scene': {'candidate_views': [{'id': 200, 'ray_id': 2}]}}
    value = add_tesseract_response(value, response, request)
    rays = {item['ray_id']: item for item in value['rays']}

    assert rays[1]['prequalification_status'] == 'culled'
    assert rays[2]['bridge_status'] == 'shortlisted'
    assert rays[2]['tesseract_status'] == 'culled'
    assert rays[2]['tesseract_reasons'][0]['stage'] == 'IK_FAILED'
    assert rays[2]['requested_maximum_standoff_m'] == 0.80
    assert rays[2]['maximum_standoff_m'] == 0.54
    assert rays[2]['capability_intervals_m'] == [
        [0.30, 0.34], [0.50, 0.54]]
    prequalify = next(
        event for event in value['events'] if event['stage'] == 'prequalify')
    assert prequalify['ray_deltas']['2']['maximum_standoff_m'] == 0.54
    assert value['requests'][0]['status'] == 'failure'


def test_schema2_journal_has_golden_later_cycle_and_terminal_order():
    value = snapshot()
    prequalified = [
        dict(ray(2, [0.0, 0.0, 1.0]), prequalified=True, reachable=True,
             safe=True, reject_reasons=[]),
    ]
    value = add_prequalification(value, prequalified, {'safe_viewpoints': 1})
    value = add_bridge_request(value, 'request-1', [2])
    value = add_capture_event(
        value, 'capture-1', True, ray_id=2,
        achieved_camera_matrix_4x4=np.eye(4).tolist(),
        joint_names=['joint%d' % index for index in range(1, 7)],
        joint_positions=[0.0] * 6, coverage_snapshot_path='coverage.npz')
    value = add_target_update_event(
        value, 'capture-1', 'coverage.npz')
    value = add_terminal_event(value, 'completed', 'safe frontier complete')

    assert value['schema_version'] == 2
    assert [event['stage'] for event in value['events']] == [
        'cull_used_redundant', 'nbv_rank', 'information_cull',
        'prequalify', 'plan', 'capture', 'update_target', 'completed']
    assert len({event['event_id'] for event in value['events']}) == len(
        value['events'])
    cull = value['events'][0]
    assert len(cull['ray_deltas']) == value['generated_ray_count']
    assert cull['ray_deltas']['0']['culled'] is True
    assert cull['ray_deltas']['2']['culled'] is False


def test_schema2_journal_has_golden_seed_cycle_prefix():
    generated = [ray(0, [1.0, 0.0, 0.0]), ray(1, [0.0, 1.0, 0.0])]
    value = planner_generation_snapshot(
        'mission-seed', 0, [0.0, 0.0, 0.2], 'base_link', 'ray_nbv',
        generated, generated, generated, generated, selection_ready=True)
    value = add_prequalification(value, [
        dict(item, prequalified=True, reachable=True, reject_reasons=[])
        for item in generated], {'safe_viewpoints': 2})

    assert [event['stage'] for event in value['events']] == [
        'generate', 'cull', 'prequalify', 'seed_rank']
    assert value['events'][0]['metrics']['surviving_ray_count'] == 2
    assert value['events'][1]['metrics']['eliminated_ray_count'] == 0


def test_seed_cycle_records_size_envelope_and_eliminates_without_omission():
    generated = [ray(0, [1.0, 0.0, 0.0]), ray(1, [0.0, 1.0, 0.0])]
    generated[0].update({
        'target_envelope_supported': False,
        'target_envelope_sha256': 'e' * 64,
        'target_envelope_rejection_reason': 'no safe target standoff',
        'ray_requested_min_standoff_m': 0.28,
        'ray_requested_max_standoff_m': 0.80,
    })
    envelope = {
        'envelope_sha256': 'e' * 64,
        'collision_boxes': [],
    }
    hard_culls = {0: {
        'ray_id': 0,
        'stage': 'target_envelope',
        'reason_code': 'NO_SAFE_TARGET_STANDOFF',
        'reason': 'no safe target standoff',
    }}

    value = planner_generation_snapshot(
        'mission-size-aware', 0, [0.4, 0.0, 0.12], 'base_link',
        'ray_nbv', generated, generated[1:], generated[1:], generated[1:],
        selection_ready=True, persistent_culls=hard_culls,
        target_envelope=envelope)

    assert value['generated_ray_count'] == 2
    assert value['target_envelope'] == envelope
    assert value['events'][0]['target_envelope'] == envelope
    assert value['events'][0]['ray_deltas']['0'][
        'target_envelope_supported'] is False
    cull = value['events'][1]
    assert cull['newly_culled_ray_ids'] == [0]
    assert cull['ray_deltas']['0']['cull_stage'] == 'target_envelope'
    assert cull['metrics']['input_ray_count'] == 2
    assert cull['metrics']['surviving_ray_count'] == 1
    value = add_prequalification(value, [
        dict(generated[1], prequalified=True, reachable=True,
             reject_reasons=[]),
    ], {'safe_viewpoints': 1})
    assert set(value['events'][-1]['ray_deltas']) == {'1'}


def test_tesseract_only_marks_eliminations_from_the_current_response():
    value = snapshot()
    first = {
        'request_id': 'request-1', 'status': 'failure',
        'selected_viewpoints': [], 'rejection_codes': [],
        'planning_diagnostics': {
            'attempted_ray_ids': [2],
            'candidate_failures': [{
                'ray_id': 2, 'stage': 'IK_FAILED', 'detail': 'first',
                'permanent_endpoint_failure': True,
            }],
        },
    }
    value = add_tesseract_response(value, first)
    second = {
        'request_id': 'request-2', 'status': 'failure',
        'selected_viewpoints': [], 'rejection_codes': [],
        'planning_diagnostics': {
            'attempted_ray_ids': [1],
            'candidate_failures': [{
                'ray_id': 1, 'stage': 'IK_FAILED', 'detail': 'second',
                'permanent_endpoint_failure': True,
            }],
        },
    }
    value = add_tesseract_response(value, second)

    assert value['events'][-1]['newly_culled_ray_ids'] == [1]
    assert value['events'][-1]['metrics']['eliminated_ray_count'] == 1


def test_capture_identity_falls_back_to_immutable_frame_metadata():
    capture_id, ray_id = capture_event_identity(
        {'accepted_entries': [{}]},
        {'view_selection': {
            'plan_id': 'plan-from-frame', 'ray_id': 144}}, 1)

    assert capture_id == 'plan-from-frame'
    assert ray_id == 144


def test_selected_worker_endpoint_supplies_the_urdf_replay_pose():
    value = snapshot()
    response = {
        'request_id': 'request-1',
        'status': 'success',
        'selected_viewpoints': [{
            'id': 202,
            'ray_id': 2,
            'camera_position_m': [0.2, 0.0, 0.5],
            'look_direction': [0.0, 0.0, -1.0],
        }],
        'segments': [{
            'to_viewpoint': 202,
            'points': [
                {'positions_rad': [0.0] * 6},
                {'positions_rad': [0.1, 0.2, -0.3, 0.4, 0.5, -0.6]},
            ],
        }],
        'rejection_codes': [],
        'diagnostic': 'planned',
        'planning_diagnostics': {'attempted_ray_ids': [2]},
    }

    value = add_tesseract_response(value, response)
    selected = {item['ray_id']: item for item in value['rays']}[2]

    assert selected['planned_joint_positions_rad'] == [
        0.1, 0.2, -0.3, 0.4, 0.5, -0.6]
    assert selected['robot_pose_source'] == 'tesseract_planned_endpoint'
    assert selected['camera_position_m'] == [0.2, 0.0, 0.5]


def test_store_merges_stages_and_writes_self_contained_html(tmp_path):
    store = RayMissionDiagnosticsStore(tmp_path)
    base = snapshot()
    enriched = add_bridge_request(base, 'request-1', [2])

    json_path, html_path = store.record(enriched)
    # A later planner freshness update must not erase downstream evidence.
    store.record(base)

    with open(json_path, 'r', encoding='utf-8') as stream:
        document = json.load(stream)
    rays = {
        item['ray_id']: item for item in document['generations'][0]['rays']}
    assert rays[2]['bridge_status'] == 'shortlisted'
    assert document['generations'][0]['requests'][0]['request_id'] == 'request-1'
    with open(html_path, 'r', encoding='utf-8') as stream:
        rendered = stream.read()
    assert 'Spherical direction map' in rendered
    assert 'Target-centred top view' in rendered
    assert '3D PiPER URDF and L515 ray replay' in rendered
    assert 'piper_description.xacro' in rendered
    assert 'all ray cameras' in rendered
    assert 'mission-a' in rendered
    assert 'https://' not in rendered


def test_store_sanitizes_session_directory(tmp_path):
    store = RayMissionDiagnosticsStore(tmp_path)
    assert store.session_dir('../../unsafe').parent == tmp_path


def test_store_indexes_the_immutable_seed_population(tmp_path):
    generated = [ray(0, [1.0, 0.0, 0.0]), ray(1, [0.0, 1.0, 0.0])]
    value = planner_generation_snapshot(
        'mission-seed', 0, [0.0, 0.0, 0.2], 'base_link', 'ray_nbv',
        generated, generated, generated, generated, selection_ready=True)
    json_path, _html_path = RayMissionDiagnosticsStore(tmp_path).record(value)

    with open(json_path, encoding='utf-8') as stream:
        document = json.load(stream)

    assert document['journal_complete'] is True
    assert document['ray_population_validation'] == 'complete'
    assert document['ray_population_index'] == [{
        'session_id': 'mission-seed',
        'generation': 0,
        'ray_count': 2,
        'sha256': value['ray_population_sha256'],
    }]


def test_store_reports_qualified_population_reset_and_only_new_culls(tmp_path):
    generated = [
        ray(0, [1.0, 0.0, 0.0]),
        ray(1, [0.0, 1.0, 0.0]),
        ray(2, [0.0, 0.0, 1.0]),
    ]
    bootstrap = planner_generation_snapshot(
        'two-phase', 0, [0.40, 0.0, 0.10], 'base_link', 'ray_nbv',
        generated, generated, generated, generated)
    bootstrap = add_prequalification(bootstrap, [
        dict(generated[0], prequalified=False, reachable=False,
             reject_reasons=['CAPABILITY_MAP_NO_SUPPORT']),
        dict(generated[1], prequalified=True, reachable=True,
             reject_reasons=[]),
        dict(generated[2], prequalified=True, reachable=True,
             reject_reasons=[]),
    ], {})
    envelope = {'planning_anchor_m': [0.35, -0.01, 0.14]}
    qualified = planner_generation_snapshot(
        'two-phase', 1, envelope['planning_anchor_m'], 'base_link',
        'ray_nbv', generated, [generated[1], generated[2]],
        [generated[1], generated[2]], [generated[1], generated[2]],
        target_envelope=envelope)
    later = planner_generation_snapshot(
        'two-phase', 2, envelope['planning_anchor_m'], 'base_link',
        'ray_nbv', generated, [generated[2]], [generated[2]], [generated[2]],
        target_envelope=envelope)
    store = RayMissionDiagnosticsStore(tmp_path)
    store.record(bootstrap)
    store.record(qualified)
    json_path, _html_path = store.record(later)

    with open(json_path, encoding='utf-8') as stream:
        document = json.load(stream)
    upgrade_index = next(
        index for index, event in enumerate(document['events'])
        if event['stage'] == 'upgrade_population')
    upgrade = document['events'][upgrade_index]
    assert upgrade['population_reset'] is True
    assert upgrade['ray_population_phase'] == 'qualified'
    assert len(upgrade['ray_deltas']) == 3
    assert (
        document['ray_population_index'][0]['sha256']
        != document['ray_population_index'][1]['sha256'])
    qualified_culls = [
        event for event in document['events']
        if event['ray_population_phase'] == 'qualified'
        and event['stage'] in ('cull', 'cull_used_redundant')]
    assert qualified_culls[0]['newly_culled_ray_ids'] == [0]
    assert qualified_culls[1]['newly_culled_ray_ids'] == [1]


def test_store_groups_reacquisition_sessions_under_one_mission(tmp_path):
    store = RayMissionDiagnosticsStore(tmp_path)
    first = snapshot()
    first['mission_id'] = 'mission-42'
    second = snapshot()
    second['mission_id'] = 'mission-42'
    second['session_id'] = 'mission-42-acquisition-2'

    json_path, _html_path = store.record(first)
    store.record(second)

    assert 'mission-42' in json_path
    with open(json_path, 'r', encoding='utf-8') as stream:
        document = json.load(stream)
    assert document['session_ids'] == [
        'mission-42-acquisition-2', 'mission-a']


def test_historical_capture_replay_is_command_free_and_preserves_pose(tmp_path):
    dataset = tmp_path / 'datasets' / 'active_scan' / 'scan_20260801_120000'
    frames = dataset / 'frames'
    frames.mkdir(parents=True)
    (dataset / 'manifest.json').write_text('{}\n', encoding='utf-8')
    metadata = {
        'frame_index': 0,
        'planned_viewpoint_count': 360,
        'reachable_viewpoint_count': 351,
        'camera_transform': {
            'available': True,
            'header': {'frame_id': 'base_link'},
            'translation_m': [0.31, 0.01, 0.22],
            'matrix_4x4': [
                [1.0, 0.0, 0.0, 0.31],
                [0.0, 1.0, 0.0, 0.01],
                [0.0, 0.0, 1.0, 0.22],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        'joint_state': {
            'available': True,
            'position': [0.1, 0.8, -1.0, 0.2, 0.3, -0.4, 0.0, 0.0],
        },
        'view_selection': {
            'available': True,
            'request_id': 'recorded-plan',
            'camera_position_m': [0.30, 0.0, 0.20],
            'look_direction': [1.0, 0.0, 0.0],
            'ray_standoff_m': 0.35,
            'ray_id': 123,
            'nbv_rank': 4,
            'view_selection_generation': 0,
            'view_selection_policy': 'ray_nbv',
            'candidate_diagnostics': {'attempted_ray_ids': [123]},
        },
    }
    metadata_path = frames / 'view_000_metadata.yaml'
    metadata_path.write_text(yaml.safe_dump(metadata), encoding='utf-8')

    result = replay_historical_dataset(dataset, tmp_path / 'reports')

    assert result['command_free'] is True
    assert result['replayed_capture_count'] == 1
    with open(result['json_path'], encoding='utf-8') as stream:
        document = json.load(stream)
    generation = document['generations'][0]
    replayed = generation['rays'][0]
    assert generation['generated_ray_count'] == 360
    assert generation['historical_replay']['partial_candidate_population']
    assert replayed['ray_id'] == 123
    assert replayed['achieved_joint_positions_rad'] == metadata[
        'joint_state']['position'][:6]
    assert replayed['achieved_camera_position_m'] == [0.31, 0.01, 0.22]


def test_legacy_capture_replay_derives_ray_without_inventing_culls(tmp_path):
    metadata = {
        'frame_index': 2,
        'planned_viewpoint_count': 1050,
        'reachable_viewpoint_count': 1050,
        'camera_transform': {
            'available': True,
            'header': {'frame_id': 'base_link'},
            'translation_m': [0.1, 0.0, 0.2],
            'matrix_4x4': [
                [1.0, 0.0, 0.0, 0.1],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.2],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        'target_3d': {
            'available': True,
            'point': {'x': 0.0, 'y': 0.0, 'z': 0.3},
        },
        'joint_state': {
            'available': True,
            'position': [0.0, 0.8, -1.0, 0.0, 0.2, 0.0],
        },
    }

    replay = historical_replay_snapshot(
        tmp_path / 'scan_legacy', metadata, 'view_002_metadata.yaml')

    assert replay['view_selection_policy'] == 'legacy_achieved_capture_pose'
    assert replay['historical_replay']['legacy_pose_only'] is True
    assert replay['historical_replay']['partial_candidate_population'] is True
    assert replay['rays'][0]['planner_reasons'] == []
