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
    position_limit_reentry_path,
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


@pytest.mark.parametrize('rate', [20.0, 100.0, 200.0])
def test_transport_rate_preserves_baseline_duration_vertices_and_recovery(rate):
    """Higher rate subdivides the approved geometry, not its movement time."""
    import numpy as np
    from piper_mobile_manipulation.execution.validation import validate_timed_execution_path

    source = [[0.0] * 6, [0.1, 0, 0, 0, 0, 0], [0.1, 0.1, 0, 0, 0, 0]]
    baseline = normalize_trajectory(source, 0.04, 5)
    points = normalize_trajectory(source, 0.04, 5, command_rate_hz=rate)
    factor = int(rate / 20)
    assert len(points) == (len(baseline) - 1) * factor + 1
    assert points[-1]['time_from_start_s'] == baseline[-1]['time_from_start_s']
    assert [p['positions_rad'] for p in points[::factor]] == [
        p['positions_rad'] for p in baseline]
    q = np.asarray([p['positions_rad'] for p in points])
    times = np.asarray([p['time_from_start_s'] for p in points])
    assert np.diff(times) == pytest.approx(1.0 / rate)
    assert np.max(np.abs(np.diff(q, axis=0))) <= 0.25 / rate + 1e-10
    validate_timed_execution_path(
        q, np.zeros_like(q), np.zeros_like(q), times, rate,
        velocity_limits_rad_s=[1.] * 6, acceleration_limits_rad_s2=[1.] * 6,
        speed_percent=5)
    joined, boundary = prepend_bootstrap_recovery(
        points, [[-0.01, 0, 0, 0, 0, 0], source[0]], 5,
        command_rate_hz=rate)
    assert joined[boundary]['positions_rad'] == source[0]
    assert np.diff([p['time_from_start_s'] for p in joined]) == pytest.approx(1 / rate)


@pytest.mark.parametrize('rate', [0, 50, 201, float('nan'), float('inf')])
def test_transport_rate_rejects_unsupported_values(rate):
    with pytest.raises(CuroboContractError, match='command rate'):
        normalize_trajectory([[0.] * 6, [0.1] * 6], .04, 5, command_rate_hz=rate)


@pytest.mark.parametrize('rate', [100., 200.])
def test_uniform_time_scale_preserves_native_speed_profile_and_all_vertices(rate):
    import numpy as np
    increments = np.array([.011, .013, .015, .013, .011])
    source = np.zeros((6, 6))
    source[1:, 0] = np.cumsum(increments)
    old = normalize_trajectory(source.tolist(), .05, 5, command_rate_hz=rate)
    new = normalize_trajectory(source.tolist(), .05, 5, command_rate_hz=rate,
                               uniform_time_scale=True)
    q = np.array([p['positions_rad'] for p in new])
    stride = (len(q) - 1) // (len(source) - 1)
    assert q[::stride] == pytest.approx(source)
    assert np.diff([p['time_from_start_s'] for p in new]) == pytest.approx(1 / rate)
    # One scale preserves acceleration/deceleration instead of creating
    # speed dips at independently rounded .05 -> .10 second boundaries.
    speeds = increments / (stride / rate)
    assert speeds / increments == pytest.approx([speeds[0] / increments[0]] * 5)
    old_q = np.array([p['positions_rad'] for p in old])
    old_velocity = np.diff(old_q[:, 0]) * rate
    new_velocity = np.diff(q[:, 0]) * rate
    assert np.max(np.abs(np.diff(new_velocity))) < np.max(np.abs(np.diff(old_velocity)))
    assert np.max(np.abs(new_velocity)) <= .25 + 1e-10
    assert new[-1]['time_from_start_s'] >= old[-1]['time_from_start_s']
    assert np.all(q[:, 1:] == 0.)


def test_uniform_time_scale_retains_detours_and_lower_wrist_speed():
    source = [[0.] * 6, [.02, 0, 0, 0, 0, .04], [.02, .01, 0, 0, 0, .04]]
    points = normalize_trajectory(source, .05, 5, command_rate_hz=100.,
                                   uniform_time_scale=True)
    stride = (len(points) - 1) // 2
    assert [p['positions_rad'] for p in points[::stride]] == source
    assert all(p['positions_rad'][1] == 0 for p in points[:stride])
    assert max(abs(b['positions_rad'][5] - a['positions_rad'][5]) * 100
               for a, b in zip(points, points[1:])) <= .15 + 1e-10


def test_uniform_time_scale_point_bound_fails_before_allocation(monkeypatch):
    import motion_planning.curobo.adapter as adapter
    monkeypatch.setattr(adapter, 'MAXIMUM_SCHEDULED_POINTS', 30)
    with pytest.raises(CuroboContractError, match='point limit'):
        normalize_trajectory([[0.] * 6, [.1] * 6, [.101] * 6], .05, 5,
                             command_rate_hz=100., uniform_time_scale=True)


def test_offline_transport_tool_reports_density_and_preserves_source(tmp_path):
    import csv
    import subprocess
    import sys

    source = tmp_path / 'response.json'
    points = normalize_trajectory([[0.] * 6, [.1] * 6], .05, 5)
    payload = dict(status='success', trajectory_binding=dict(
        execution=dict(command_rate_hz=20., effective_speed_percent=5.)),
        segments=[dict(points=points)])
    source.write_text(json.dumps(payload))
    original = source.read_bytes()
    output = tmp_path / 'report'
    command = [sys.executable, str(ROOT / 'tools/diagnostics/benchmark_stream_transport.py'),
               str(source), '--output', str(output)]
    subprocess.run(command, check=True, capture_output=True)
    with (output / 'transport.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert {row['duration_sec'] for row in rows} == {str(points[-1]['time_from_start_s'])}
    assert source.read_bytes() == original
    assert subprocess.run(command, capture_output=True).returncode != 0


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


def test_cad_holder_gate_rejects_recorded_sub_five_mm_floor_clearance():
    class ArrayValue:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    def link_poses(values, _links):
        count = len(values)
        return SimpleNamespace(
            position=ArrayValue([[0.0, 0.0, 0.059064]] * count),
            quaternion=ArrayValue([[1.0, 0.0, 0.0, 0.0]] * count),
        )

    backend = object.__new__(CuroboBackend)
    backend.floor_z_m = 0.005
    backend.collision_manifest = {
        'external_floor_clearance': {
            'enabled': True,
            'floor_z_m': 0.005,
            'clearance_m': 0.005,
            'label': 'camera holder/L515',
            'origin_link6_m': [0.0, 0.0, 0.0],
            'size_m': [0.10, 0.10, 0.10],
        },
    }
    backend.tensor_args = SimpleNamespace(to_device=lambda value: value)
    backend.motion_gen = SimpleNamespace(kinematics=SimpleNamespace(
        get_link_poses=link_poses))
    backend.last_planning_diagnostics = {}

    with pytest.raises(
            CuroboCollisionRejected,
            match=(
                r'camera holder/L515 envelope floor clearance 0\.004064m '
                r'is below 0\.005000m at sample 0')):
        backend._external_attached_validation([[0.0] * 6], [])


def test_cad_holder_gate_matches_expanded_obstacle_box_clearance():
    class ArrayValue:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    backend = object.__new__(CuroboBackend)
    backend.floor_z_m = 0.0
    backend.collision_manifest = {
        'external_floor_clearance': {
            'enabled': True,
            'floor_z_m': 0.0,
            'clearance_m': 0.005,
            'label': 'camera holder/L515',
            'origin_link6_m': [0.0, 0.0, 0.0],
            'size_m': [0.10, 0.10, 0.10],
        },
    }
    backend.tensor_args = SimpleNamespace(to_device=lambda value: value)
    backend.motion_gen = SimpleNamespace(kinematics=SimpleNamespace(
        get_link_poses=lambda values, _links: SimpleNamespace(
            position=ArrayValue([[0.0, 0.0, 0.20]] * len(values)),
            quaternion=ArrayValue(
                [[1.0, 0.0, 0.0, 0.0]] * len(values)))))
    backend.last_planning_diagnostics = {}
    obstacle = {
        'id': 'close_box',
        'type': 'box',
        'minimum_m': [0.052, -0.01, 0.19],
        'maximum_m': [0.06, 0.01, 0.21],
    }

    with pytest.raises(
            CuroboCollisionRejected,
            match=(
                'camera holder/L515 envelope intersects obstacle close_box '
                'clearance at sample 0')):
        backend._external_attached_validation([[0.0] * 6], [obstacle])


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
    # This test isolates post-plan retry/selection; prechecks have separate tests.
    backend._endpoint_precheck = lambda start, goals, request, candidate: goals
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
    backend._attached_tool_external_rejection = (
        lambda *_args, **_kwargs: '')
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


def test_curobo_pose_planning_retries_after_holder_floor_rejection():
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
    # This test isolates post-plan retry/selection; prechecks have separate tests.
    backend._endpoint_precheck = lambda start, goals, request, candidate: goals
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
        'camera holder/L515 envelope floor clearance 0.004064m is below '
        '0.005000m at sample 193',
        '',
    ])
    backend._attached_tool_external_rejection = (
        lambda *_args, **_kwargs: next(rejections))
    backend._target_visibility_rejection = lambda *_args, **_kwargs: ''
    request = {
        'plan_kind': 'MULTIVIEW_SCAN',
        'planning': {
            'effective_speed_percent': 5.0,
            'roll_samples_rad': [0.0, 1.0],
        },
        'limits': {'position_rad': [[-1.0, 1.0]] * 6},
        'scene': {
            'target_center_m': [0.4, 0.0, 0.12],
            'obstacles': [],
        },
        'scan_session': {'accepted_views': 1},
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
    assert diagnostics[0]['path_qualification'] == 'rejected'
    assert '0.004064m' in diagnostics[0]['path_qualification_reason']
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
    # This test isolates post-plan retry/selection; prechecks have separate tests.
    backend._endpoint_precheck = lambda start, goals, request, candidate: goals
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
    backend._attached_tool_external_rejection = (
        lambda *_args, **_kwargs: '')
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


@pytest.mark.parametrize('fallback_count', [1, 5])
def test_curobo_target_ray_fallback_preserves_requested_nominal_aim(fallback_count):
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
    # This test isolates post-plan retry/selection; prechecks have separate tests.
    backend._endpoint_precheck = lambda start, goals, request, candidate: goals
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
    backend._attached_tool_external_rejection = (
        lambda *_args, **_kwargs: '')
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
        'fallback_look_directions': [fallback] + [
            [math.cos(math.radians(angle)), math.sin(math.radians(angle)), 0.0]
            for angle in range(1, fallback_count)],
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


def test_bounded_cone_batch_balances_directions_without_growing_cuda_shape():
    goals = [{'look_direction': [1, direction, 0], 'distance': distance, 'roll': roll}
             for distance in range(9) for roll in range(6) for direction in range(5)]
    selected = CuroboBackend._bounded_aim_goals(goals)
    assert len(selected) == 54
    for direction in range(5):
        subset = [g for g in selected if g['look_direction'][1] == direction]
        assert len(subset) in (10, 11)
        assert {g['distance'] for g in subset} >= {0, 8}
    assert all(goal in goals for goal in selected)
    assert len(CuroboBackend._padded_goalset(selected)) == 54
    assert CuroboBackend._bounded_aim_goals(goals[:10]) == goals[:10]


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
    backend._position_limit_reentry = lambda _start, _request: None
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


def test_curobo_plan_prepends_limit_reentry_to_the_common_timed_path():
    backend = object.__new__(CuroboBackend)
    backend._update_world = lambda _request: None
    actual = [0.0, 0.8, -0.7, 0.0, 0.7, 3.136623084]
    projected = [0.0, 0.8, -0.7, 0.0, 0.7, math.pi - 0.0051]
    reentry = {
        'positions': [actual, projected],
        'joint_numbers': [6],
        'delta_rad': [projected[5] - actual[5]],
        'start_status': 'MotionGenStatus.INVALID_START_STATE_JOINT_LIMITS',
        'validation': 'raw_limit_valid_dense_curobo_collision_qualified',
    }
    backend._position_limit_reentry = (
        lambda _start, _request: reentry)
    backend._bootstrap_recovery = lambda _start, _request: None
    backend._target_visibility_rejection = lambda *_args, **_kwargs: ''
    backend._attached_tool_external_rejection = (
        lambda *_args, **_kwargs: '')
    planned = normalize_trajectory(
        [projected, [0.0, 0.8, -0.7, 0.0, 0.7, 3.0]],
        native_dt_sec=0.05,
        speed_percent=5.0,
    )

    def plan_pose(start, candidate, _request):
        assert start == pytest.approx(projected)
        return candidate, planned, 0.2

    backend._plan_pose = plan_pose
    request = {
        'plan_kind': 'MULTIVIEW_SCAN',
        'start_state': {'positions_rad': actual},
        'planning': {
            'min_viewpoints': 1,
            'max_viewpoints': 1,
            'include_return_home': False,
            'effective_speed_percent': 5.0,
            'shortlisted_ray_count': 0,
        },
        'limits': {'position_rad': [[-math.pi, math.pi]] * 6},
        'scene': {
            'target_center_m': [0.4, 0.0, 0.12],
            'candidate_views': [{'id': 1}],
        },
    }

    selected, segments, duration = backend.plan(request)

    assert selected == [{'id': 1}]
    assert segments[0]['points'][0]['positions_rad'] == pytest.approx(actual)
    assert segments[0]['points'][1]['positions_rad'] == pytest.approx(
        projected)
    assert segments[0]['points'][-1]['positions_rad'][5] == pytest.approx(3.0)
    assert segments[0]['bootstrap_recovery_used'] is False
    assert backend.last_planning_diagnostics[
        'position_limit_reentry_used'] is True
    assert backend.last_planning_diagnostics[
        'position_limit_reentry_joint_numbers'] == [6]
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


def test_position_limit_reentry_covers_recorded_joint6_feedback_boundary():
    limits = [[-math.pi, math.pi] for _ in range(6)]
    clips = [0.005, 0.0, 0.0, 0.005, 0.005, 0.005]
    start = [0.0, 0.8, -0.7, 0.0, 0.7, 3.136623084]

    path = position_limit_reentry_path(
        start, limits, clips,
        maximum_clip_excursion_rad=0.001,
        interior_offset_rad=0.0001,
        maximum_step_rad=0.00025,
    )

    assert path[0] == pytest.approx(start)
    assert path[-1][5] == pytest.approx(math.pi - 0.005 - 0.0001)
    assert all(
        math.pi - point[5] >= math.pi - start[5] - 1e-12
        for point in path)
    assert all(
        max(abs(right - left) for left, right in zip(first, second))
        <= 0.00025 + 1e-12
        for first, second in zip(path, path[1:]))


def test_position_limit_reentry_is_not_a_general_limit_bypass():
    limits = [[-math.pi, math.pi] for _ in range(6)]
    clips = [0.005, 0.0, 0.0, 0.005, 0.005, 0.005]
    comfortably_inside = [0.0, 0.8, -0.7, 0.0, 0.7, 3.0]
    assert position_limit_reentry_path(
        comfortably_inside, limits, clips) == ()

    outside_raw_limit = list(comfortably_inside)
    outside_raw_limit[5] = math.pi + 1e-6
    with pytest.raises(
            CuroboContractError,
            match='joint6 is outside the raw limit'):
        position_limit_reentry_path(
            outside_raw_limit, limits, clips)

    tighter_than_observed_policy = list(comfortably_inside)
    tighter_than_observed_policy[5] = math.pi - 0.0039
    with pytest.raises(
            CuroboContractError,
            match='joint6 exceeds the bounded feedback excursion'):
        position_limit_reentry_path(
            tighter_than_observed_policy, limits, clips,
            maximum_clip_excursion_rad=0.001)


def test_curobo_limit_reentry_requires_collision_free_dense_transition():
    backend = object.__new__(CuroboBackend)
    start = [0.0, 0.8, -0.7, 0.0, 0.7, 3.136623084]
    backend._start_state_status = lambda positions: (
        (False, 'MotionGenStatus.INVALID_START_STATE_JOINT_LIMITS')
        if positions == start else (True, 'None'))
    checked = []
    backend._collision_path_without_position_clip = (
        lambda positions: checked.append(positions) or True)
    request = request_fixture()
    request['limits']['position_rad'] = [
        [-math.pi, math.pi] for _ in range(6)]

    recovery = CuroboBackend._position_limit_reentry(
        backend, start, request)

    assert recovery['joint_numbers'] == [6]
    assert recovery['positions'][0] == pytest.approx(start)
    assert recovery['positions'][-1][5] < math.pi - 0.005
    assert checked and checked[0] == recovery['positions']

    backend._collision_path_without_position_clip = lambda _positions: False
    with pytest.raises(
            CuroboCollisionRejected,
            match='position-limit re-entry is in collision'):
        CuroboBackend._position_limit_reentry(backend, start, request)


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
        model_provenance={
            'hardware_qualified': False,
            'hardware_qualification': {
                'hardware_qualified': True,
                'scope': 'supervised_5_percent_target_scan',
                'floor_profile': 'tabletop',
                'free_motion_speed_percent': 5.0,
                'contact_speed_percent': 5.0,
                'real_motion_requires_explicit_opt_in': True,
            },
        }))
    monkeypatch.setenv('PIPER_FLOOR_PROFILE', 'tabletop')
    monkeypatch.setenv('PIPER_VIEWPOINT_SPEED_PERCENT', '5')
    monkeypatch.setenv('PIPER_CONTACT_SPEED_PERCENT', '5')
    monkeypatch.setenv('PIPER_CUROBO_COLLISION_MODEL_QUALIFIED', '1')
    assert Worker.collision_model_qualified(worker) is False

    worker.backend.model_provenance['hardware_qualified'] = True
    monkeypatch.setenv('PIPER_CUROBO_COLLISION_MODEL_QUALIFIED', '0')
    assert Worker.collision_model_qualified(worker) is False

    monkeypatch.setenv('PIPER_CUROBO_COLLISION_MODEL_QUALIFIED', '1')
    assert Worker.collision_model_qualified(worker) is True
    for free, contact in ((50, 5), (5, 50), (1, 1), (100, 100)):
        monkeypatch.setenv('PIPER_VIEWPOINT_SPEED_PERCENT', str(free))
        monkeypatch.setenv('PIPER_CONTACT_SPEED_PERCENT', str(contact))
        assert Worker.collision_model_qualified(worker) is True
    # Historical physical evidence is preserved, not relabelled as a 50% test.
    assert worker.backend.model_provenance['hardware_qualification']['free_motion_speed_percent'] == 5.0


@pytest.mark.parametrize(('name', 'value'), [
    ('PIPER_FLOOR_PROFILE', 'ground'),
    ('PIPER_VIEWPOINT_SPEED_PERCENT', 'nan'),
    ('PIPER_VIEWPOINT_SPEED_PERCENT', '0'),
    ('PIPER_VIEWPOINT_SPEED_PERCENT', '101'),
    ('PIPER_CONTACT_SPEED_PERCENT', 'inf'),
    ('PIPER_CONTACT_SPEED_PERCENT', '-1'),
    ('PIPER_CONTACT_SPEED_PERCENT', 'invalid'),
])
def test_collision_qualification_rejects_runtime_outside_physical_scope(
        monkeypatch, name, value):
    worker = SimpleNamespace(backend=SimpleNamespace(
        model_provenance={
            'hardware_qualified': True,
            'hardware_qualification': {
                'hardware_qualified': True,
                'scope': 'supervised_5_percent_target_scan',
                'floor_profile': 'tabletop',
                'free_motion_speed_percent': 5.0,
                'contact_speed_percent': 5.0,
                'real_motion_requires_explicit_opt_in': True,
            },
        }))
    monkeypatch.setenv('PIPER_CUROBO_COLLISION_MODEL_QUALIFIED', '1')
    monkeypatch.setenv('PIPER_FLOOR_PROFILE', 'tabletop')
    monkeypatch.setenv('PIPER_VIEWPOINT_SPEED_PERCENT', '5')
    monkeypatch.setenv('PIPER_CONTACT_SPEED_PERCENT', '5')
    monkeypatch.setenv(name, value)
    assert Worker.collision_model_qualified(worker) is False


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
    diagnostics = worker.spool.response['planning_diagnostics']
    assert diagnostics['worker_request_wall_sec'] >= 0
    assert diagnostics['worker_timing']['stages']['response_validation']['calls'] == 1


@pytest.mark.parametrize('boundary', [-2.967, 2.967])
@pytest.mark.parametrize('rate', [20.0, 100.0, 200.0])
def test_exact_float32_boundary_is_canonicalized(boundary, rate):
    import struct
    native = struct.unpack('f', struct.pack('f', boundary))[0]
    limits = [[-2.967, 2.967] for _ in range(6)]
    positions = [[0.0] * 6, [native] * 6]
    points = normalize_trajectory(
        positions, .05, 50., limits, command_rate_hz=rate)
    assert points[-1]['positions_rad'] == [boundary] * 6
    assert positions[-1] == [native] * 6
    validate_trajectory_position_limits(
        [point['positions_rad'] for point in points], limits)
    with pytest.raises(CuroboContractError):
        normalize_trajectory(
            [[0.0] * 6, [native + math.copysign(1e-8, native)] * 6],
            .05, 50., limits, command_rate_hz=rate)


def test_powered_reentry_keeps_raw_feedback_separate():
    from piper_mobile_manipulation.execution.motion import (
        CONTROLLER_COMMAND_LIMITS, URDF_JOINT_LIMITS)
    raw = URDF_JOINT_LIMITS.tolist()
    powered = CONTROLLER_COMMAND_LIMITS.tolist()
    start = [0., -.002, -.7, 0., .5, 0.]
    path = position_limit_reentry_path(
        start, raw, [0.] * 6, command_limits=powered)
    assert path[0] == start
    assert path[-1][1] > 0.
    assert all(a[1] <= b[1] for a, b in zip(path, path[1:]))
    start[1] = -.028179526329040527
    with pytest.raises(CuroboContractError, match='bounded feedback excursion'):
        position_limit_reentry_path(
            start, raw, [0.] * 6, command_limits=powered)


def test_worker_rejects_recorded_negative_joint2_output():
    from motion_planning.curobo.adapter import JOINT_NAMES
    from types import SimpleNamespace
    import numpy as np
    from piper_mobile_manipulation.execution.motion import URDF_JOINT_LIMITS

    class Tensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.array([[0., .1, -.7, 0., .5, 0.],
                             [0., -.028179526329040527, -.7, 0., .5, 0.]])

    path = SimpleNamespace(position=Tensor(), joint_names=JOINT_NAMES)
    result = SimpleNamespace(
        success=SimpleNamespace(item=lambda: True),
        get_interpolated_plan=lambda: path, interpolation_dt=.05)
    backend = object.__new__(CuroboBackend)
    with pytest.raises(CuroboContractError, match='joint2'):
        backend._path(result, 50., URDF_JOINT_LIMITS.tolist(), 200.)


@pytest.mark.parametrize('kind,mode,joint2', [
    ('ROUGH_ACQUISITION', 'bootstrap_static', -0.0005001),
    ('MULTIVIEW_SCAN', 'bootstrap_static', -0.000139552),
    ('RETURN_HOME', 'bootstrap_static', -0.000139552),
    ('ROUGH_ACQUISITION', 'live', -0.000139552),
    ('ROUGH_ACQUISITION', 'bootstrap_static', 0.),
])
def test_joint2_noise_admission_is_narrow_and_acquisition_only(kind, mode, joint2):
    backend = object.__new__(CuroboBackend)
    start = [0., joint2, 0., 0., .4, 0.]
    result = backend._acquisition_start_with_joint2_noise(
        start, {'plan_kind': kind, 'scene': {'observation_mode': mode}})
    assert result == start
    assert result is not start
