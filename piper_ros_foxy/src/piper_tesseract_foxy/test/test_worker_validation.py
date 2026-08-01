import numpy as np
import pytest
from types import SimpleNamespace

import piper_tesseract_foxy.worker as worker_module
from piper_tesseract_foxy.contract import ContractError
from piper_tesseract_foxy.worker import (
    quiesce_bootstrap_recovery_prefix,
    reverse_sdk_movej_points,
    sdk_movej_waypoint_trajectory,
    subdivide_joint_segment,
    TesseractBackend,
)


def test_reverse_sdk_movej_points_preserves_reverse_order_and_durations():
    points = [
        {
            'time_from_start_s': when,
            'positions_rad': [position] * 6,
            'velocities_rad_s': [0.0] * 6,
            'accelerations_rad_s2': [0.0] * 6,
        }
        for when, position in ((0.0, 0.0), (1.0, 0.2), (3.0, 0.5))
    ]
    reversed_points = reverse_sdk_movej_points(points)
    assert [item['positions_rad'][0] for item in reversed_points] == [0.5, 0.2, 0.0]
    assert [item['time_from_start_s'] for item in reversed_points] == [0.0, 2.0, 3.0]
    assert all(item['velocities_rad_s'] == [0.0] * 6 for item in reversed_points)
    assert all(item['accelerations_rad_s2'] == [0.0] * 6 for item in reversed_points)


def test_subdivide_joint_segment_bounds_total_joint_change():
    points = subdivide_joint_segment(
        np.zeros(6),
        np.asarray([0.01, -0.02, 0.0, 0.03, 0.0, 0.0]),
        0.01,
    )
    assert np.allclose(points[0], np.zeros(6))
    assert np.allclose(points[-1], [0.01, -0.02, 0.0, 0.03, 0.0, 0.0])
    assert all(
        np.sum(np.abs(second - first)) <= 0.0100000001
        for first, second in zip(points[:-1], points[1:])
    )


def test_subdivide_joint_segment_rejects_invalid_step():
    with pytest.raises(ContractError, match='maximum L1 joint step'):
        subdivide_joint_segment(np.zeros(6), np.ones(6), 0.0)


def test_sdk_movej_target_keeps_joint6_free_and_zero_derivatives():
    source = [
        {
            'time_from_start_s': 0.0,
            'positions_rad': [0.0] * 6,
            'velocities_rad_s': [0.0] * 6,
            'accelerations_rad_s2': [0.0] * 6,
        },
        {
            'time_from_start_s': 1.0,
            'positions_rad': [0.2, 0.0, 0.0, 0.0, 0.0, 0.4],
            'velocities_rad_s': [0.0] * 6,
            'accelerations_rad_s2': [0.0] * 6,
        },
    ]
    fast, fast_validation = sdk_movej_waypoint_trajectory(
        source, 100.0, 100.0, 0.10,
        [[-1.0, 1.0]] * 6, [1.0] * 6, [2.0] * 6)
    slow, _ = sdk_movej_waypoint_trajectory(
        source, 10.0, 100.0, 0.10,
        [[-1.0, 1.0]] * 6, [1.0] * 6, [2.0] * 6)
    assert fast[-1]['positions_rad'][5] == pytest.approx(0.4)
    assert slow[-1]['positions_rad'][5] == pytest.approx(0.4)
    assert slow == fast
    assert fast_validation == fast
    assert all(point['velocities_rad_s'] == [0.0] * 6 for point in fast)
    assert all(point['accelerations_rad_s2'] == [0.0] * 6 for point in fast)
    assert len(fast) == 2
    assert fast[0]['positions_rad'] == pytest.approx([0.0] * 6)
    assert fast[1]['positions_rad'] == pytest.approx(
        [0.2, 0.0, 0.0, 0.0, 0.0, 0.4])


def test_sdk_movej_waypoint_order_stamps_respect_transport_rate():
    source = [
        {
            'time_from_start_s': 0.0,
            'positions_rad': [0.0] * 6,
            'velocities_rad_s': [0.0] * 6,
            'accelerations_rad_s2': [0.0] * 6,
        },
        {
            'time_from_start_s': 0.002,
            'positions_rad': [0.001] * 6,
            'velocities_rad_s': [0.0] * 6,
            'accelerations_rad_s2': [0.0] * 6,
        },
        {
            'time_from_start_s': 0.012,
            'positions_rad': [0.002] * 6,
            'velocities_rad_s': [0.0] * 6,
            'accelerations_rad_s2': [0.0] * 6,
        },
    ]
    emitted, _ = sdk_movej_waypoint_trajectory(
        source, 100.0, 100.0, 0.10,
        [[-1.0, 1.0]] * 6, [1.0] * 6, [2.0] * 6)
    assert len(emitted) == 2
    intervals = [
        current['time_from_start_s'] - previous['time_from_start_s']
        for previous, current in zip(emitted[:-1], emitted[1:])
    ]
    assert min(intervals) >= 0.01 - 1e-9


def test_sdk_movej_allows_only_the_bound_bootstrap_start_outside_limits():
    source = [
        {
            'time_from_start_s': 0.0,
            'positions_rad': [0.0, 0.0, 0.0327, 0.0, 0.0, 0.0],
            'velocities_rad_s': [0.0] * 6,
            'accelerations_rad_s2': [0.0] * 6,
        },
        {
            'time_from_start_s': 1.0,
            'positions_rad': [0.0, 0.0, -0.01, 0.0, 0.0, 0.0],
            'velocities_rad_s': [0.0] * 6,
            'accelerations_rad_s2': [0.0] * 6,
        },
    ]
    limits = [[-1.0, 1.0] for _ in range(6)]
    limits[2] = [-1.0, 0.0]
    with pytest.raises(ContractError, match='position limit'):
        sdk_movej_waypoint_trajectory(
            source, 5.0, 100.0, 0.10,
            limits, [1.0] * 6, [2.0] * 6)
    emitted, _ = sdk_movej_waypoint_trajectory(
        source, 5.0, 100.0, 0.10,
        limits, [1.0] * 6, [2.0] * 6,
        bootstrap_start_limit_tolerance_rad=0.04)
    assert emitted[0]['positions_rad'][2] == pytest.approx(0.0327)
    assert emitted[1]['positions_rad'][2] == pytest.approx(-0.01)


def test_bootstrap_recovery_quiescing_prevents_anticipated_other_joint_motion():
    source = [
        {
            'time_from_start_s': 0.0,
            'positions_rad': [0.0] * 6,
            'velocities_rad_s': [0.0, 0.0, 0.04, 0.0, 0.0, 0.0],
            'accelerations_rad_s2': [0.0] * 6,
        },
        {
            'time_from_start_s': 0.5,
            'positions_rad': [0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
            # ISP may anticipate the normal multi-joint path at this knot.
            'velocities_rad_s': [0.02, -0.03, 0.05, 0.0, 0.0, 0.0],
            'accelerations_rad_s2': [0.01, -0.02, 0.03, 0.0, 0.0, 0.0],
        },
        {
            'time_from_start_s': 1.5,
            'positions_rad': [0.1, -0.1, 0.2, 0.0, 0.0, 0.0],
            'velocities_rad_s': [0.0] * 6,
            'accelerations_rad_s2': [0.0] * 6,
        },
    ]
    boundary = quiesce_bootstrap_recovery_prefix(
        source, [0.02, 0.0, 0.0, 0.0, 0.0, 0.0], 1)
    assert boundary == 1
    assert source[0]['velocities_rad_s'] == [0.0] * 6
    assert source[1]['velocities_rad_s'] == [0.0] * 6
    assert source[1]['accelerations_rad_s2'] == [0.0] * 6

    emitted, _ = sdk_movej_waypoint_trajectory(
        source, 100.0, 100.0, 0.10,
        [[-1.0, 1.0]] * 6, [1.0] * 6, [2.0] * 6,
        mandatory_waypoints=[[0.02, 0.0, 0.0, 0.0, 0.0, 0.0]])
    emitted_boundary = next(
        index for index, point in enumerate(emitted)
        if np.allclose(
            point['positions_rad'],
            [0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
            atol=1e-9,
        ))
    assert emitted_boundary == 1
    assert np.max(np.abs(np.asarray([
        point['positions_rad'][1:]
        for point in emitted[:emitted_boundary + 1]
    ]))) <= 1e-9
    assert all(
        emitted[index]['positions_rad'][0]
        <= emitted[index + 1]['positions_rad'][0] + 1e-9
        for index in range(emitted_boundary)
    )


def test_acquisition_worker_accepts_one_of_five_planned_looks():
    backend = SimpleNamespace(
        reset_scene=lambda: None,
        add_obstacles=lambda _obstacles: None,
        find_bootstrap_recovery=lambda _request: None,
    )
    points = [
        {'positions_rad': [0.0] * 6},
        {'positions_rad': [0.01] * 6},
    ]

    def plan_candidate(
            _current, candidate, _rolls, _step, _limits, _margin,
            _bootstrap_recovery):
        if candidate['id'] != 0:
            raise ContractError('synthetic unreachable look')
        return 0.0, points, {
            'minimum_clearance_m': 0.1,
            'limiting_link_pair': 'none/none',
        }

    backend.plan_candidate = plan_candidate
    request = {
        'plan_kind': 'ROUGH_ACQUISITION',
        'scene': {
            'obstacles': [],
            'candidate_views': [
                {
                    'id': index,
                    'camera_position_m': [0.4, 0.0, 0.3],
                    'look_direction': [1.0, 0.0, 0.0],
                    'required_first': index == 0,
                }
                for index in range(5)
            ],
        },
        'planning': {
            'min_viewpoints': 1,
            'max_viewpoints': 5,
            'max_execution_joint_step_rad': 0.10,
            'roll_samples_rad': [0.0],
            'effective_speed_percent': 100.0,
            'command_rate_hz': 100.0,
            'timing_policy': 'sdk_movej_targets_v1',
        },
        'limits': {
            'position_rad': [[-1.0, 1.0] for _ in range(6)],
            'joint_margin_rad': 0.03,
            'max_velocity_rad_s': [3.0] * 6,
            'max_acceleration_rad_s2': [5.0] * 6,
        },
        'start_state': {'positions_rad': [0.0] * 6},
    }

    selected, segments = TesseractBackend.plan(backend, request)

    assert [item['id'] for item in selected] == [0]
    assert len(segments) == 1


def test_worker_planning_budget_expires_before_bridge_timeout(monkeypatch):
    ticks = iter([0.0, worker_module.WORKER_PLANNING_BUDGET_SEC + 0.1])
    monkeypatch.setattr(worker_module.time, 'monotonic', lambda: next(ticks))
    backend = SimpleNamespace(
        reset_scene=lambda: None,
        add_obstacles=lambda _obstacles: None,
        find_bootstrap_recovery=lambda _request: None,
        plan_candidate=lambda *_args: pytest.fail(
            'no candidate should start after the internal deadline'),
    )
    request = {
        'plan_kind': 'ROUGH_ACQUISITION',
        'scene': {
            'obstacles': [],
            'candidate_views': [{
                'id': 0,
                'camera_position_m': [0.4, 0.0, 0.3],
                'look_direction': [1.0, 0.0, 0.0],
            }],
        },
        'planning': {
            'min_viewpoints': 1,
            'max_viewpoints': 1,
            'max_execution_joint_step_rad': 0.10,
            'roll_samples_rad': [0.0],
            'effective_speed_percent': 5.0,
            'command_rate_hz': 100.0,
        },
        'limits': {
            'position_rad': [[-1.0, 1.0] for _ in range(6)],
            'joint_margin_rad': 0.03,
            'max_velocity_rad_s': [3.0] * 6,
            'max_acceleration_rad_s2': [5.0] * 6,
        },
        'start_state': {'positions_rad': [0.0] * 6},
    }

    with pytest.raises(ContractError, match='before the bridge 180-second timeout'):
        TesseractBackend.plan(backend, request)


def test_worker_budget_is_checked_between_candidate_planner_attempts(
        monkeypatch):
    ticks = iter([0.0, 146.0])
    monkeypatch.setattr(worker_module.time, 'monotonic', lambda: next(ticks))
    backend = TesseractBackend.__new__(TesseractBackend)
    backend.planning_deadline_monotonic = 150.0
    backend.ik_joint_goals = lambda *_args: [
        np.asarray([0.1] * 6, dtype=float),
        np.asarray([0.2] * 6, dtype=float),
    ]
    attempts = []

    def reject_attempt(*_args):
        attempts.append(True)
        raise ContractError('synthetic planner failure')

    backend.plan_segment_to_joint_goal = reject_attempt
    candidate = {
        'id': 0,
        'camera_position_m': [0.4, 0.0, 0.3],
        'look_direction': [1.0, 0.0, 0.0],
    }

    with pytest.raises(
            ContractError, match='before the bridge 180-second timeout'):
        backend.plan_candidate(
            [0.0] * 6, candidate, [0.0], 0.10,
            [[-1.0, 1.0] for _ in range(6)], 0.03)

    assert len(attempts) == 1


def test_worker_retries_candidates_after_a_success_changes_start_state():
    backend = SimpleNamespace(
        reset_scene=lambda: None,
        add_obstacles=lambda _obstacles: None,
        find_bootstrap_recovery=lambda _request: None,
    )
    calls = []

    def plan_candidate(
            current, candidate, _rolls, _step, _limits, _margin,
            _bootstrap_recovery):
        calls.append((candidate['id'], list(current)))
        if candidate['id'] == 0 and current != [1.0] * 6:
            raise ContractError('requires the bridge viewpoint first')
        if candidate['id'] == 2:
            raise ContractError('synthetic unreachable look')
        goal = [1.0] * 6 if candidate['id'] == 1 else [2.0] * 6
        return 0.0, [
            {'positions_rad': list(current)},
            {'positions_rad': goal},
        ], {
            'minimum_clearance_m': 0.1,
            'limiting_link_pair': 'none/none',
        }

    backend.plan_candidate = plan_candidate
    request = {
        'plan_kind': 'ROUGH_ACQUISITION',
        'scene': {
            'obstacles': [],
            'candidate_views': [
                {
                    'id': index,
                    'camera_position_m': [0.4, 0.0, 0.3],
                    'look_direction': [1.0, 0.0, 0.0],
                    'required_first': index == 1,
                }
                for index in range(3)
            ],
        },
        'planning': {
            'min_viewpoints': 1,
            'max_viewpoints': 2,
            'max_execution_joint_step_rad': 0.10,
            'roll_samples_rad': [0.0],
            'effective_speed_percent': 100.0,
            'command_rate_hz': 100.0,
            'timing_policy': 'sdk_movej_targets_v1',
        },
        'limits': {
            'position_rad': [[-3.0, 3.0] for _ in range(6)],
            'joint_margin_rad': 0.03,
            'max_velocity_rad_s': [3.0] * 6,
            'max_acceleration_rad_s2': [5.0] * 6,
        },
        'start_state': {'positions_rad': [0.0] * 6},
    }

    selected, segments = TesseractBackend.plan(backend, request)

    assert [item['id'] for item in selected] == [1, 0]
    assert [item['to_viewpoint'] for item in segments] == [1, 0]
    assert calls[:2] == [
        (1, [0.0] * 6),
        (0, [1.0] * 6),
    ]


def test_multiview_worker_appends_collision_validated_return_home_segment():
    capture_points = [
        {'positions_rad': [0.0] * 6},
        {'positions_rad': [0.1] * 6},
    ]
    home = [0.0, 0.0, -0.026, -0.039, 0.346, 0.107]
    return_calls = []
    backend = SimpleNamespace(
        reset_scene=lambda: None,
        add_obstacles=lambda _obstacles: None,
        find_bootstrap_recovery=lambda _request: None,
        find_terminal_home_recovery=lambda _request, _home: None,
        plan_candidate=lambda *_args: (
            0.0,
            capture_points,
            {
                'minimum_clearance_m': 0.1,
                'limiting_link_pair': 'none/none',
            },
        ),
    )

    def plan_return(start, goal, maximum_step):
        return_calls.append((list(start), list(goal), maximum_step))
        return [
            {'positions_rad': list(start)},
            {'positions_rad': list(goal)},
        ], {
            'minimum_clearance_m': 0.1,
            'limiting_link_pair': 'none/none',
        }

    backend.plan_segment_to_joint_goal = plan_return
    request = {
        'plan_kind': 'MULTIVIEW_SCAN',
        'scene': {
            'obstacles': [],
            'candidate_views': [{
                'id': 4,
                'camera_position_m': [0.4, 0.0, 0.3],
                'look_direction': [1.0, 0.0, 0.0],
            }],
        },
        'planning': {
            'min_viewpoints': 1,
            'max_viewpoints': 1,
            'max_execution_joint_step_rad': 0.10,
            'roll_samples_rad': [0.0],
            'effective_speed_percent': 5.0,
            'command_rate_hz': 100.0,
            'return_home_positions_rad': home,
        },
        'limits': {
            'position_rad': [[-1.0, 1.0] for _ in range(6)],
            'joint_margin_rad': 0.03,
            'max_velocity_rad_s': [3.0] * 6,
            'max_acceleration_rad_s2': [5.0] * 6,
        },
        'start_state': {'positions_rad': [0.0] * 6},
    }

    selected, segments = TesseractBackend.plan(backend, request)

    assert [item['id'] for item in selected] == [4]
    assert len(segments) == 2
    assert segments[-1]['is_return_home'] is True
    assert segments[-1]['to_viewpoint'] == -2
    assert segments[-1]['points'][-1]['positions_rad'] == home
    assert return_calls == [([0.1] * 6, home, 0.10)]


def test_multiview_worker_reverses_qualified_folded_home_recovery():
    capture_points = [
        {'positions_rad': [0.0] * 6},
        {'positions_rad': [0.4] * 6},
    ]
    home = [0.0, 0.0, 0.0, 0.0, 0.43869236, 0.0]
    entry = [0.0, 0.04, -0.04, 0.0, 0.43869236, 0.0]
    recovery = {
        'positions': [home, entry],
        'bootstrap_recovery_end_positions_rad': entry,
    }
    calls = []
    backend = SimpleNamespace(
        reset_scene=lambda: None,
        add_obstacles=lambda _obstacles: None,
        find_bootstrap_recovery=lambda _request: None,
        find_terminal_home_recovery=lambda _request, _home: recovery,
        plan_candidate=lambda *_args: (
            0.0,
            capture_points,
            {'minimum_clearance_m': 0.1, 'limiting_link_pair': 'none/none'},
        ),
    )

    def plan_return(start, goal, maximum_step, supplied_recovery):
        calls.append((list(start), list(goal), maximum_step, supplied_recovery))
        return [
            {
                'time_from_start_s': when,
                'positions_rad': list(position),
                'velocities_rad_s': [0.0] * 6,
                'accelerations_rad_s2': [0.0] * 6,
            }
            for when, position in (
                (0.0, home), (1.0, entry), (4.0, [0.4] * 6))
        ], {
            'minimum_clearance_m': 0.01,
            'limiting_link_pair': 'link2/link5',
            'bootstrap_recovery_used': True,
            'bootstrap_recovery_end_point': 1,
        }

    backend.plan_segment_to_joint_goal = plan_return
    request = {
        'plan_kind': 'MULTIVIEW_SCAN',
        'scene': {
            'obstacles': [],
            'candidate_views': [{
                'id': 4,
                'camera_position_m': [0.4, 0.0, 0.3],
                'look_direction': [1.0, 0.0, 0.0],
            }],
        },
        'planning': {
            'min_viewpoints': 1,
            'max_viewpoints': 1,
            'max_execution_joint_step_rad': 0.10,
            'roll_samples_rad': [0.0],
            'effective_speed_percent': 5.0,
            'command_rate_hz': 100.0,
            'return_home_positions_rad': home,
        },
        'limits': {
            'position_rad': [[-1.0, 1.0] for _ in range(6)],
            'joint_margin_rad': 0.03,
            'max_velocity_rad_s': [3.0] * 6,
            'max_acceleration_rad_s2': [5.0] * 6,
        },
        'start_state': {'positions_rad': [0.0] * 6},
    }

    _selected, segments = TesseractBackend.plan(backend, request)

    assert calls == [(entry, [0.4] * 6, 0.10, recovery)]
    assert [
        point['positions_rad'] for point in segments[-1]['points']
    ] == [[0.4] * 6, entry, home]
    assert [
        point['time_from_start_s'] for point in segments[-1]['points']
    ] == [0.0, 3.0, 4.0]
    assert segments[-1]['is_return_home'] is True


class FakeRecoveryBackend:
    """Small collision oracle for fail-closed recovery-policy tests."""

    def __init__(self, contacts):
        self.contacts = contacts

    def collision_policy(self):
        return 0.1, 0.2, 0.01, {}

    def motion_allowances(self):
        return {None: 0.0}

    def contact_minimums(self, position, _report):
        return self.contacts(np.asarray(position, dtype=float))

    clearance_violations = TesseractBackend.clearance_violations


def recovery_policy():
    return {
        'allowed_contacts': {('link1', 'link5'): 0.02},
        'monotonic_tolerance_m': 0.0,
    }


def test_bootstrap_recovery_rejects_unapproved_contact():
    backend = FakeRecoveryBackend(
        lambda _q: {('link1', 'link5'): -0.01, ('link2', 'link6'): -0.001})
    with pytest.raises(ContractError, match='unapproved contact'):
        TesseractBackend.validate_bootstrap_recovery_path(
            backend, [np.zeros(6), np.asarray([0.01, 0, 0, 0, 0, 0])],
            recovery_policy())


def test_bootstrap_recovery_rejects_excessive_penetration():
    backend = FakeRecoveryBackend(
        lambda _q: {('link1', 'link5'): -0.021})
    with pytest.raises(ContractError, match='exceeds bounded penetration'):
        TesseractBackend.validate_bootstrap_recovery_path(
            backend, [np.zeros(6), np.asarray([0.01, 0, 0, 0, 0, 0])],
            recovery_policy())


def test_bootstrap_recovery_rejects_worsening_contact():
    backend = FakeRecoveryBackend(
        lambda q: {('link1', 'link5'): -0.005 - float(q[0])})
    with pytest.raises(ContractError, match='worsens'):
        TesseractBackend.validate_bootstrap_recovery_path(
            backend, [np.zeros(6), np.asarray([0.01, 0, 0, 0, 0, 0])],
            recovery_policy())


def test_bootstrap_recovery_endpoint_must_reach_normal_clearance():
    backend = FakeRecoveryBackend(
        lambda q: {('link1', 'link5'): -0.01 + 0.05 * float(q[0])})
    with pytest.raises(ContractError, match='endpoint does not reach normal'):
        TesseractBackend.validate_bootstrap_recovery_path(
            backend, [np.zeros(6), np.asarray([0.01, 0, 0, 0, 0, 0])],
            recovery_policy())


def test_bootstrap_recovery_policy_is_acquisition_only():
    backend = SimpleNamespace(manifest={
        'bootstrap_start_recovery': {
            'enabled': True,
            'plan_kind': 'ROUGH_ACQUISITION',
            'observation_mode': 'bootstrap_static',
            'search_step_rad': 0.01,
            'maximum_single_joint_delta_rad': 0.15,
            'maximum_start_limit_violation_rad': 0.04,
            'allowed_start_limit_joints': [3],
            'monotonic_tolerance_m': 0.0002,
            'allowed_start_contacts': [{
                'links': ['link1', 'link5'],
                'maximum_penetration_m': 0.01,
            }],
        },
    })
    assert TesseractBackend.bootstrap_recovery_policy(backend, {
        'plan_kind': 'MULTIVIEW_SCAN',
        'scene': {'observation_mode': 'perception_snapshot'},
    }) is None


def test_bootstrap_limit_recovery_moves_only_joint3_inward():
    backend = SimpleNamespace(
        bootstrap_recovery_policy=lambda _request: {
            'step_rad': 0.01,
            'maximum_delta_rad': 0.15,
            'maximum_start_limit_violation_rad': 0.04,
            'allowed_start_limit_joints': [3],
            'monotonic_tolerance_m': 0.0002,
            'allowed_contacts': {('link1', 'link5'): 0.01},
        },
        collision_policy=lambda: (0.005, 0.05, 0.001, {}),
        contact_minimums=lambda _position, _report: {},
        clearance_violations=lambda _minimums: {},
    )
    backend.validate_bootstrap_recovery_path = (
        lambda positions, policy:
        TesseractBackend.validate_bootstrap_recovery_path(
            backend, positions, policy))
    request = {
        'start_state': {
            'positions_rad': [0.0, 0.0, 0.0327, 0.0, 0.0, 0.0],
        },
        'limits': {
            'position_rad': [
                [-1.0, 1.0], [-1.0, 1.0], [-1.0, 0.0],
                [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0],
            ],
        },
    }
    recovery = TesseractBackend.find_bootstrap_recovery(backend, request)
    assert recovery['bootstrap_recovery_joint'] == 3
    assert recovery['bootstrap_recovery_delta_rad'] == pytest.approx(-0.04)
    assert recovery['positions'][0][2] == pytest.approx(0.0327)
    assert recovery['positions'][-1][2] == pytest.approx(-0.0073)


def test_bootstrap_limit_recovery_binds_one_dual_joint_target():
    backend = SimpleNamespace(
        bootstrap_recovery_policy=lambda _request: {
            'step_rad': 0.01,
            'maximum_delta_rad': 0.15,
            'maximum_start_limit_violation_rad': 0.04,
            'allowed_start_limit_joints': [2, 3],
            'maximum_recovery_joints': 2,
            'monotonic_tolerance_m': 0.0002,
            'allowed_contacts': {('link1', 'link5'): 0.01},
        },
        collision_policy=lambda: (0.005, 0.05, 0.001, {}),
        contact_minimums=lambda _position, _report: {},
        clearance_violations=lambda _minimums: {},
    )
    backend.validate_bootstrap_recovery_path = (
        lambda positions, policy:
        TesseractBackend.validate_bootstrap_recovery_path(
            backend, positions, policy))
    request = {
        'start_state': {
            'positions_rad': [0.0, -0.001, 0.035, 0.0, 0.0, 0.0],
        },
        'limits': {
            'position_rad': [
                [-1.0, 1.0], [0.0, 1.0], [-1.0, 0.0],
                [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0],
            ],
            'joint_margin_rad': 0.03,
        },
    }

    recovery = TesseractBackend.find_bootstrap_recovery(backend, request)

    assert recovery['bootstrap_recovery_joint'] == 0
    assert recovery['bootstrap_recovery_joints'] == [2, 3]
    assert recovery['bootstrap_recovery_deltas_rad'] == pytest.approx(
        [0.031, -0.065])
    assert recovery['positions'][0][1:3] == pytest.approx([-0.001, 0.035])
    assert recovery['positions'][-1][1:3] == pytest.approx([0.03, -0.03])
