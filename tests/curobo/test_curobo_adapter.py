"""Pure cuRobo adapter tests; no CUDA or cuRobo import is permitted here."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from motion_planning.curobo.adapter import (
    attach_digest,
    CuroboCandidateExhausted,
    CuroboCollisionRejected,
    CuroboContractError,
    CuroboPlanningBudgetExceeded,
    normalize_trajectory,
    obstacle_cuboids,
    prepend_bootstrap_recovery,
    target_ray_standoff_samples,
    trajectory_segment,
    validate_request,
    validate_trajectory_position_limits,
    worker_rejection_code,
)
from motion_planning.curobo.worker import (
    camera_path_visibility_rejection,
    CuroboBackend,
    planning_budgets_for_request,
    quaternion_optical_z_wxyz,
    Worker,
)


ROOT = Path(__file__).resolve().parents[2]


def request_fixture():
    request = {
        'schema_version': 5,
        'planner_backend': 'curobo',
        'plan_kind': 'MULTIVIEW_SCAN',
        'start_state': {
            'joint_names': [
                'joint1', 'joint2', 'joint3',
                'joint4', 'joint5', 'joint6'],
            'positions_rad': [0.0] * 6,
        },
        'planning': {
            'planner': 'MotionGen',
            'pipeline': 'CUROBO_V1',
        },
        'limits': {
            'position_rad': [[-1.0, 1.0] for _ in range(6)],
        },
        'scene': {
            'target_center_m': [0.4, 0.0, 0.12],
            'candidate_views': [{
                'id': 7,
                'camera_position_m': [0.1, 0.0, 0.12],
                'look_direction': [1.0, 0.0, 0.0],
            }],
            'obstacles': [{
                'id': 'box', 'type': 'box',
                'minimum_m': [0.2, -0.1, 0.0],
                'maximum_m': [0.3, 0.1, 0.2],
            }],
        },
    }
    return attach_digest(request, 'request_sha256')


def test_request_and_world_conversion_preserve_authoritative_geometry():
    request = validate_request(request_fixture())
    cuboids = obstacle_cuboids(request, floor_z_m=0.005)
    assert cuboids[0]['pose'][:3] == pytest.approx([0.25, 0.0, 0.1])
    assert cuboids[0]['dims'] == pytest.approx([0.1, 0.2, 0.2])
    assert cuboids[-1]['name'] == 'configured_support_floor'
    assert cuboids[-1]['pose'][2] == pytest.approx(-0.055)
    assert cuboids[-1]['dims'] == pytest.approx([4.0, 4.0, 0.10])


@pytest.mark.parametrize('mutation', [
    lambda value: value.update(planner_backend='tesseract'),
    lambda value: value.update(plan_kind='OCCLUSION_PROBE'),
    lambda value: value['start_state'].update(joint_names=['joint2'] * 6),
    lambda value: value['start_state'].update(
        positions_rad=[0.0, 0.0, math.nan, 0.0, 0.0, 0.0]),
    lambda value: value['scene'].update(candidate_views=[]),
])
def test_invalid_or_unsupported_requests_fail_closed(mutation):
    request = request_fixture()
    request.pop('request_sha256')
    mutation(request)
    request = attach_digest(request, 'request_sha256')
    with pytest.raises(CuroboContractError):
        validate_request(request)


def test_request_digest_correlation_is_mandatory():
    request = request_fixture()
    request['scene']['target_center_m'][0] = 0.5
    with pytest.raises(CuroboContractError, match='request_sha256 mismatch'):
        validate_request(request)


def test_out_of_limit_bootstrap_is_explicitly_unsupported():
    request = request_fixture()
    request.pop('request_sha256')
    request['start_state']['positions_rad'][2] = 1.01
    request = attach_digest(request, 'request_sha256')
    with pytest.raises(CuroboContractError, match='unsupported cuRobo bootstrap'):
        validate_request(request)


def test_native_path_is_fixed_rate_slowed_and_preserves_endpoints():
    points = normalize_trajectory(
        [[0.0] * 6, [0.12, 0.0, 0.0, 0.0, 0.0, 0.0]],
        native_dt_sec=0.01,
        speed_percent=5.0,
    )
    assert points[0]['positions_rad'] == [0.0] * 6
    assert points[-1]['positions_rad'][0] == pytest.approx(0.12)
    assert len(points) == 11
    assert [
        right['time_from_start_s'] - left['time_from_start_s']
        for left, right in zip(points, points[1:])
    ] == pytest.approx([0.05] * 10)
    assert max(
        abs(right['positions_rad'][0] - left['positions_rad'][0])
        for left, right in zip(points, points[1:])
    ) <= 5.0 * 0.05 * 0.05 + 1e-12
    assert all(
        right['time_from_start_s'] > left['time_from_start_s']
        for left, right in zip(points, points[1:]))


def test_native_vertices_are_retained_without_corner_shortcuts():
    native = [
        [0.0] * 6,
        [0.04, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.04, -0.04, 0.0, 0.0, 0.0, 0.0],
    ]
    points = normalize_trajectory(
        native, native_dt_sec=0.05, speed_percent=5.0)
    emitted = [point['positions_rad'] for point in points]

    first_vertex_index = next(
        index for index, point in enumerate(emitted)
        if point == pytest.approx(native[1]))
    assert first_vertex_index > 0
    assert first_vertex_index < len(emitted) - 1
    assert all(abs(point[1]) <= 1e-12 for point in emitted[:first_vertex_index])
    assert all(
        point[0] == pytest.approx(native[1][0])
        for point in emitted[first_vertex_index:])


def test_native_timing_is_not_accelerated_above_fixed_rate():
    points = normalize_trajectory(
        [[0.0] * 6, [0.001] * 6],
        native_dt_sec=0.12,
        speed_percent=100.0,
    )
    assert len(points) == 4
    assert points[-1]['time_from_start_s'] == pytest.approx(0.15)


def test_sixth_joint_uses_its_lower_movej_velocity_bound():
    points = normalize_trajectory(
        [[0.0] * 6, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0485364]],
        native_dt_sec=0.05,
        speed_percent=5.0,
    )
    maximum_step = max(
        abs(right['positions_rad'][5] - left['positions_rad'][5])
        for left, right in zip(points, points[1:]))
    assert maximum_step <= 3.0 * 0.05 * 0.05 + 1e-12
    assert all(
        right['time_from_start_s'] - left['time_from_start_s']
        == pytest.approx(0.05)
        for left, right in zip(points, points[1:]))


def _forward_at_degrees(angle_deg):
    angle = math.radians(float(angle_deg))
    return [math.cos(angle), math.sin(angle), 0.0]


def test_first_alignment_visibility_rejects_a_worsening_curobo_path():
    reason = camera_path_visibility_rejection(
        [[0.0, 0.0, 0.0]] * 3,
        [_forward_at_degrees(11.1),
         _forward_at_degrees(20.1),
         _forward_at_degrees(0.0)],
        [1.0, 0.0, 0.0],
        initial_alignment=True,
        final_aim_deg=5.0,
    )
    assert reason == (
        'initial target-alignment path worsens beyond its 11.1-degree '
        'acquired aim at sample 1 (20.1 degrees)')


def test_first_alignment_visibility_accepts_nonworsening_entry_and_endpoint():
    reason = camera_path_visibility_rejection(
        [[0.0, 0.0, 0.0]] * 4,
        [_forward_at_degrees(30.0),
         _forward_at_degrees(25.0),
         _forward_at_degrees(15.0),
         _forward_at_degrees(5.0)],
        [1.0, 0.0, 0.0],
        initial_alignment=True,
        final_aim_deg=5.0,
    )
    assert reason == ''


def test_wxyz_camera_quaternion_conversion_returns_optical_z():
    forwards = quaternion_optical_z_wxyz([
        [1.0, 0.0, 0.0, 0.0],
        [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0],
    ])
    assert forwards[0] == pytest.approx([0.0, 0.0, 1.0])
    assert forwards[1] == pytest.approx([1.0, 0.0, 0.0])


def test_curobo_pose_planning_retries_after_visibility_rejection():
    class Scalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    result = SimpleNamespace(
        success=Scalar(True), status='SUCCESS', total_time=0.25,
        goalset_index=Scalar(0))
    backend = object.__new__(CuroboBackend)
    backend.tensor_args = SimpleNamespace(to_device=lambda value: value)
    backend.Pose = lambda **kwargs: kwargs
    backend.MotionGenPlanConfig = lambda **kwargs: kwargs
    backend.motion_gen = SimpleNamespace(
        plan_goalset=lambda *_args, **_kwargs: result)
    backend._joint_state = lambda value: value
    backend._restore_collision_constraints = lambda: None
    backend._path = lambda *_args, **_kwargs: [
        {'positions_rad': [0.0] * 6},
        {'positions_rad': [0.1] * 6},
    ]
    rejections = iter([
        'initial target-alignment path worsens',
        '',
    ])
    backend._target_visibility_rejection = (
        lambda *_args, **_kwargs: next(rejections))
    request = {
        'plan_kind': 'MULTIVIEW_SCAN',
        'planning': {
            'effective_speed_percent': 5.0,
            'roll_samples_rad': [0.0, 1.0],
        },
        'limits': {'position_rad': [[-1.0, 1.0]] * 6},
        'scene': {'target_center_m': [0.4, 0.0, 0.12]},
        'scan_session': {'accepted_views': 0},
    }
    candidate = {
        'id': 7,
        'camera_position_m': [0.1, 0.0, 0.12],
        'look_direction': [1.0, 0.0, 0.0],
    }

    selected, _points, _duration = backend._plan_pose(
        [0.0] * 6, candidate, request)

    assert selected['curobo_roll_rad'] == pytest.approx(1.0)
    diagnostics = selected['curobo_attempt_diagnostics']
    assert len(diagnostics) == 2
    assert diagnostics[0]['path_qualification'] == 'rejected'
    assert diagnostics[1]['path_qualification'] == 'accepted'


def test_target_ray_samples_stay_in_disjoint_capability_intervals():
    samples = target_ray_standoff_samples({
        'candidate_geometry': 'target_ray',
        'ray_min_standoff_m': 0.28,
        'ray_max_standoff_m': 0.80,
        'ray_standoff_m': 0.31,
        'ray_scoring_standoff_m': 0.32,
        'ray_preferred_max_standoff_m': 0.50,
        'ray_capability_intervals_m': [[0.30, 0.36], [0.52, 0.60]],
    }, maximum_samples=9)
    assert samples[0] == pytest.approx(0.31)
    assert len(samples) == 9
    assert all(
        0.30 <= value <= 0.36 or 0.52 <= value <= 0.60
        for value in samples)
    assert any(value == pytest.approx(0.30) for value in samples)
    assert any(value == pytest.approx(0.60) for value in samples)


def test_curobo_target_ray_uses_one_goalset_and_reports_selected_standoff():
    class Scalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    result = SimpleNamespace(
        success=Scalar(True), status='SUCCESS', total_time=0.2,
        goalset_index=Scalar(2))
    calls = []
    backend = object.__new__(CuroboBackend)
    backend.tensor_args = SimpleNamespace(to_device=lambda value: value)
    backend.Pose = lambda **kwargs: kwargs
    backend.MotionGenPlanConfig = lambda **kwargs: kwargs
    backend.motion_gen = SimpleNamespace(
        plan_goalset=lambda *args, **kwargs: (
            calls.append((args, kwargs)) or result))
    backend._joint_state = lambda value: value
    backend._restore_collision_constraints = lambda: None
    backend._path = lambda *_args, **_kwargs: [
        {'positions_rad': [0.0] * 6},
        {'positions_rad': [0.1] * 6},
    ]
    backend._target_visibility_rejection = lambda *_args, **_kwargs: ''
    request = {
        'plan_kind': 'MULTIVIEW_SCAN',
        'planning': {
            'effective_speed_percent': 5.0,
            'roll_samples_rad': [0.0],
        },
        'limits': {'position_rad': [[-1.0, 1.0]] * 6},
        'scene': {'target_center_m': [0.4, 0.0, 0.12]},
        'scan_session': {'accepted_views': 1},
    }
    candidate = {
        'id': 1000007,
        'candidate_geometry': 'target_ray',
        'ray_id': 7,
        'camera_position_m': [0.10, 0.0, 0.12],
        'look_direction': [1.0, 0.0, 0.0],
        'ray_direction': [-1.0, 0.0, 0.0],
        'ray_min_standoff_m': 0.28,
        'ray_max_standoff_m': 0.60,
        'ray_standoff_m': 0.30,
        'ray_scoring_standoff_m': 0.31,
        'ray_preferred_max_standoff_m': 0.50,
    }

    selected, _points, duration = backend._plan_pose(
        [0.0] * 6, candidate, request)

    assert len(calls) == 1
    goal_pose = calls[0][0][1]
    assert len(goal_pose['position'][0]) == 54
    assert selected['ray_standoff_m'] == pytest.approx(0.50)
    assert selected['camera_position_m'] == pytest.approx([-0.10, 0.0, 0.12])
    assert selected['curobo_goalset_pose_count'] == 9
    assert selected['curobo_attempt_diagnostics'][0][
        'native_padded_goalset_size'] == 54
    assert duration == pytest.approx(0.2)


def test_curobo_target_ray_fallback_preserves_requested_nominal_aim():
    class Scalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    results = iter([
        SimpleNamespace(
            success=Scalar(False), status='IK_FAIL', total_time=0.1,
            goalset_index=None),
        SimpleNamespace(
            success=Scalar(True), status='SUCCESS', total_time=0.2,
            goalset_index=Scalar(0)),
    ])
    backend = object.__new__(CuroboBackend)
    backend.tensor_args = SimpleNamespace(to_device=lambda value: value)
    backend.Pose = lambda **kwargs: kwargs
    backend.MotionGenPlanConfig = lambda **kwargs: kwargs
    backend.motion_gen = SimpleNamespace(
        plan_goalset=lambda *_args, **_kwargs: next(results))
    backend._joint_state = lambda value: value
    backend._restore_collision_constraints = lambda: None
    backend._path = lambda *_args, **_kwargs: [
        {'positions_rad': [0.0] * 6},
        {'positions_rad': [0.1] * 6},
    ]
    backend._target_visibility_rejection = lambda *_args, **_kwargs: ''
    request = {
        'plan_kind': 'MULTIVIEW_SCAN',
        'planning': {
            'effective_speed_percent': 5.0,
            'roll_samples_rad': [0.0],
        },
        'limits': {'position_rad': [[-1.0, 1.0]] * 6},
        'scene': {'target_center_m': [0.4, 0.0, 0.12]},
        'scan_session': {'accepted_views': 1},
    }
    nominal = [1.0, 0.0, 0.0]
    fallback = [math.cos(math.radians(4.0)), math.sin(math.radians(4.0)), 0.0]
    candidate = {
        'id': 1000007,
        'candidate_geometry': 'target_ray',
        'ray_id': 7,
        'camera_position_m': [0.10, 0.0, 0.12],
        'look_direction': nominal,
        'fallback_look_directions': [fallback],
        'ray_direction': [-1.0, 0.0, 0.0],
        'ray_min_standoff_m': 0.28,
        'ray_max_standoff_m': 0.60,
        'ray_standoff_m': 0.30,
        'ray_scoring_standoff_m': 0.31,
        'ray_preferred_max_standoff_m': 0.50,
    }

    selected, _points, duration = backend._plan_pose(
        [0.0] * 6, candidate, request)

    assert selected['nominal_look_direction'] == pytest.approx(nominal)
    assert selected['look_direction'] == pytest.approx(fallback)
    assert selected['aim_fallback_used'] is True
    assert selected['aim_offset_deg'] == pytest.approx(4.0)
    assert duration == pytest.approx(0.3)


def test_curobo_planning_budgets_match_closed_loop_policy():
    request = {
        'plan_kind': 'MULTIVIEW_SCAN',
        'planning': {
            'min_viewpoints': 1,
            'max_viewpoints': 1,
            'include_return_home': False,
        },
    }
    assert planning_budgets_for_request(request) == pytest.approx((90.0, 3.0))
    request['planning']['include_return_home'] = True
    assert planning_budgets_for_request(request) == pytest.approx((150.0, 5.0))


def test_typed_curobo_failures_do_not_parse_backend_name_as_unavailable():
    assert worker_rejection_code(CuroboCollisionRejected(
        'cuRobo start state is in collision')) == 'PLANNING_COLLISION_REJECTED'
    assert worker_rejection_code(CuroboCandidateExhausted(
        'cuRobo exhausted ray shortlist')) == 'PLANNER_EXHAUSTED'
    assert worker_rejection_code(CuroboPlanningBudgetExceeded(
        'cuRobo request budget expired')) == 'PLANNER_TIMEOUT'


def test_curobo_candidate_loop_retries_a_visibility_failed_view():
    backend = object.__new__(CuroboBackend)
    backend._update_world = lambda _request: None
    backend._bootstrap_recovery = lambda _start, _request: None
    calls = []

    def plan_pose(_start, candidate, _request):
        calls.append(candidate['id'])
        if candidate['id'] == 1:
            raise CuroboContractError('visibility rejected')
        return candidate, [
            {'positions_rad': [0.0] * 6},
            {'positions_rad': [0.1] * 6},
        ], 0.2

    backend._plan_pose = plan_pose
    request = {
        'plan_kind': 'MULTIVIEW_SCAN',
        'start_state': {'positions_rad': [0.0] * 6},
        'planning': {
            'min_viewpoints': 1,
            'max_viewpoints': 1,
            'include_return_home': False,
        },
        'scene': {'candidate_views': [{'id': 1}, {'id': 2}]},
    }

    selected, segments, duration = backend.plan(request)

    assert calls == [1, 2]
    assert [item['id'] for item in selected] == [2]
    assert len(segments) == 1
    assert duration == pytest.approx(0.2)


def test_float32_joint_limit_overshoot_fails_at_curobo_boundary():
    limits = [[-1.0, 1.0] for _ in range(6)]
    positions = [[0.0] * 6, [0.0, 0.0, 0.0, 0.0, 1.000000028, 0.0]]
    with pytest.raises(
            CuroboContractError,
            match=(
                r'trajectory point 1 joint5=1\.000000028 is outside '
                r'\[-1, 1\] by 2\.8.*e-08 rad')):
        validate_trajectory_position_limits(positions, limits)
    with pytest.raises(CuroboContractError, match='point 1 joint5'):
        normalize_trajectory(
            positions, native_dt_sec=0.05, speed_percent=5.0,
            position_limits=limits)


def test_position_limit_validation_accepts_interior_scheduled_path():
    limits = [[-1.0, 1.0] for _ in range(6)]
    points = normalize_trajectory(
        [[0.0] * 6, [0.0, 0.0, 0.0, 0.0, 0.995, 0.0]],
        native_dt_sec=0.05,
        speed_percent=5.0,
        position_limits=limits,
    )
    assert points[-1]['positions_rad'][4] == pytest.approx(0.995)
    assert all(
        max(abs(b - a) for a, b in zip(
            left['positions_rad'], right['positions_rad'])) <= 0.05
        for left, right in zip(points, points[1:]))


def test_bootstrap_recovery_is_prepended_and_declared_generically():
    planned = normalize_trajectory(
        [[0.0, 0.0, -0.06, 0.0, 0.4, 0.0],
         [0.0, 0.1, -0.20, 0.0, 0.5, 0.0]],
        native_dt_sec=0.05,
        speed_percent=5.0,
    )
    recovery = [
        [0.0, 0.0, 0.0, 0.0, 0.4, 0.0],
        [0.0, 0.0, -0.03, 0.0, 0.4, 0.0],
        [0.0, 0.0, -0.06, 0.0, 0.4, 0.0],
    ]
    points, endpoint = prepend_bootstrap_recovery(
        planned, recovery, speed_percent=5.0)
    segment = trajectory_segment(points, bootstrap_recovery={
        'end_point': endpoint,
        'joint_numbers': [3],
        'delta_rad': [-0.06],
    })
    assert points[0]['positions_rad'] == recovery[0]
    assert points[endpoint]['positions_rad'] == recovery[-1]
    assert points[-1]['positions_rad'] == planned[-1]['positions_rad']
    assert all(
        right['time_from_start_s'] > left['time_from_start_s']
        for left, right in zip(points, points[1:]))
    assert segment['bootstrap_recovery_used'] is True
    assert segment['bootstrap_recovery_joint'] == 3
    assert segment['bootstrap_recovery_delta_rad'] == pytest.approx(-0.06)


def test_curobo_bootstrap_exception_is_rough_acquisition_only():
    backend = object.__new__(CuroboBackend)
    backend._start_state_status = lambda position: (
        (True, 'None') if position[2] <= -0.06 else
        (False, 'MotionGenStatus.INVALID_START_STATE_SELF_COLLISION'))
    request = request_fixture()
    request['plan_kind'] = 'ROUGH_ACQUISITION'
    request['scene']['observation_mode'] = 'bootstrap_static'
    start = [0.0] * 6
    recovery = CuroboBackend._bootstrap_recovery(backend, start, request)
    assert recovery['joint_numbers'] == [3]
    assert recovery['delta_rad'] == pytest.approx([-0.06])
    assert recovery['positions'][0] == start
    assert recovery['positions'][-1][2] == pytest.approx(-0.06)

    request['plan_kind'] = 'MULTIVIEW_SCAN'
    request['scene']['observation_mode'] = 'perception_snapshot'
    with pytest.raises(CuroboContractError, match='outside bootstrap scope'):
        CuroboBackend._bootstrap_recovery(backend, start, request)


@pytest.mark.parametrize('bad', [
    [], [[0.0] * 6], [[0.0] * 6, [math.inf] * 6],
])
def test_malformed_native_trajectory_fails_closed(bad):
    with pytest.raises(CuroboContractError):
        normalize_trajectory(bad, 0.05, 5.0)


def test_collision_qualification_requires_model_evidence_and_operator_opt_in(
        monkeypatch):
    worker = SimpleNamespace(backend=SimpleNamespace(
        model_provenance={'hardware_qualified': False}))
    monkeypatch.setenv('PIPER_CUROBO_COLLISION_MODEL_QUALIFIED', '1')
    assert Worker.collision_model_qualified(worker) is False

    worker.backend.model_provenance['hardware_qualified'] = True
    monkeypatch.setenv('PIPER_CUROBO_COLLISION_MODEL_QUALIFIED', '0')
    assert Worker.collision_model_qualified(worker) is False

    monkeypatch.setenv('PIPER_CUROBO_COLLISION_MODEL_QUALIFIED', '1')
    assert Worker.collision_model_qualified(worker) is True


def test_worker_health_stays_bounded_when_model_provenance_is_large():
    provenance = {
        'hardware_qualified': False,
        'collision_spheres': ['sphere'] * 4000,
    }
    worker = SimpleNamespace(
        generation_id='a' * 32,
        backend_error='',
        backend=SimpleNamespace(
            version='0.7.8',
            robot_config_sha256='b' * 64,
            model_provenance=provenance,
            environment={'cuda_available': True},
        ),
        model_hashes={
            'srdf_sha256': 'c' * 64,
            'collision_manifest_sha256': 'd' * 64,
        },
        collision_model_qualified=lambda: False,
    )

    health = Worker.health(worker)
    diagnostics = Worker.diagnostics(worker)
    encoded_health = json.dumps(
        health, sort_keys=True, separators=(',', ':')).encode('utf-8')

    assert len(encoded_health) <= 16 * 1024
    assert 'model_provenance' not in health
    assert diagnostics['model_provenance'] == provenance
    assert health['robot_config_sha256'] == diagnostics['robot_config_sha256']
    assert health['collision_manifest_sha256'] == (
        diagnostics['collision_manifest_sha256'])


def test_worker_converts_invalid_success_output_to_structured_failure(
        tmp_path):
    request_path = (
        ROOT / 'benchmarks/planner_backends/results/'
        '20260901_tabletop_controlled_replay/raw/curobo/'
        'recorded_multiview_00_to_01/0002.request.json')
    request = json.loads(request_path.read_text(encoding='utf-8'))
    limits = request['limits']['position_rad']
    invalid = list(request['start_state']['positions_rad'])
    invalid[4] = float(limits[4][1]) + 2.8e-8
    points = normalize_trajectory(
        [request['start_state']['positions_rad'], invalid],
        native_dt_sec=0.05,
        speed_percent=5.0,
    )

    class FakeSpool:
        def __init__(self):
            self.response = None

        def claim_next(self):
            return request['request_id'], request

        def write_response(self, _request_id, response):
            self.response = response

        def path(self, _queue, request_id):
            return tmp_path / ('%s.json' % request_id)

    worker = object.__new__(Worker)
    worker.spool = FakeSpool()
    worker.backend = SimpleNamespace(
        version='0.7.8',
        plan=lambda _request: (
            [request['scene']['candidate_views'][0]],
            [trajectory_segment(points)],
            0.1,
        ),
    )
    worker.collision_model_qualified = lambda: True

    assert Worker.process_once(worker) is True
    assert worker.spool.response['status'] == 'failed'
    assert worker.spool.response['rejection_codes'] == ['PLANNING_FAILED']
    assert 'normalized output failed the generic contract' in (
        worker.spool.response['diagnostic'])
    assert 'joint5=' in worker.spool.response['diagnostic']
