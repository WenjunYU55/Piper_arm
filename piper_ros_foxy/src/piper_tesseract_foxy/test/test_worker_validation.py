import numpy as np
import pytest
from types import SimpleNamespace

import piper_tesseract_foxy.worker as worker_module
from piper_tesseract_foxy.contract import ContractError
from piper_tesseract_foxy.worker import (
    attached_box_floor_clearance_rejection,
    camera_transform_path_rejection,
    planning_budgets_for_request,
    quiesce_bootstrap_recovery_prefix,
    reverse_sdk_movej_points,
    sdk_movej_waypoint_trajectory,
    subdivide_joint_segment,
    TesseractBackend,
)


def optical_transform(angle_deg=0.0, z_m=0.0):
    angle = np.deg2rad(float(angle_deg))
    transform = np.eye(4)
    transform[:3, 2] = [np.sin(angle), 0.0, np.cos(angle)]
    transform[2, 3] = float(z_m)
    return transform


def test_worker_camera_path_rejects_off_axis_intermediate_fk():
    rejection = camera_transform_path_rejection(
        [optical_transform(0.0), optical_transform(25.0)],
        [0.0, 0.0, 1.0])

    assert 'leaves the 20.0-degree camera boresight cone' in rejection


def test_worker_camera_path_accepts_compact_visible_route():
    assert camera_transform_path_rejection(
        [optical_transform(0.0), optical_transform(19.5)],
        [0.0, 0.0, 1.0]) == ''


def test_worker_attached_holder_box_rejects_floor_grazing_transform():
    transform = np.eye(4)
    transform[2, 3] = 0.004

    assert attached_box_floor_clearance_rejection(
        transform,
        [0.0, 0.0, 0.0],
        [0.10, 0.10, 0.004],
        0.0,
        0.005,
        'camera holder/L515',
    ) == (
        'camera holder/L515 floor clearance 0.002000m is below 0.005000m')


def test_worker_tries_next_ik_goal_after_visibility_rejection():
    rejected_goal = np.asarray([0.1, 0, 0, 0, 0, 0], dtype=float)
    accepted_goal = np.asarray([0.0, 0.2, 0, 0, 0, 0], dtype=float)
    attempted = []

    class FakeRobot:
        @staticmethod
        def fk(_group, joints, tip_link=None):
            del tip_link
            # The nearer synthetic IK branch rotates off-axis; the second
            # branch preserves visibility and must be attempted next.
            angle = 25.0 if 0.05 < float(joints[0]) < 0.15 else 10.0
            if abs(float(joints[0])) < 1e-9:
                angle = 0.0
            return SimpleNamespace(matrix=optical_transform(angle))

    def plan_segment(_start, goal, _step, _bootstrap):
        attempted.append(np.asarray(goal).tolist())
        return [
            {'positions_rad': [0.0] * 6},
            {'positions_rad': np.asarray(goal).tolist()},
        ], {'minimum_clearance_m': 0.1}

    backend = SimpleNamespace(
        ik_joint_goals=lambda *_args: [rejected_goal, accepted_goal],
        ensure_planning_time=lambda _context: None,
        plan_segment_to_joint_goal=plan_segment,
        robot=FakeRobot(),
    )

    _roll, points, _validation = TesseractBackend.plan_candidate(
        backend, np.zeros(6), {}, [0.0], 0.1,
        [[-1.0, 1.0]] * 6, 0.0, None, [0.0, 0.0, 1.0])

    assert attempted == [rejected_goal.tolist(), accepted_goal.tolist()]
    assert points[-1]['positions_rad'][1] == pytest.approx(0.2)


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
        source, 100.0, 20.0, 0.05,
        [[-1.0, 1.0]] * 6, [1.0] * 6, [2.0] * 6)
    slow_source = [dict(point) for point in source]
    slow_source[1] = dict(slow_source[1])
    slow_source[1]['time_from_start_s'] = 4.0
    slow, _ = sdk_movej_waypoint_trajectory(
        slow_source, 10.0, 20.0, 0.05,
        [[-1.0, 1.0]] * 6, [1.0] * 6, [2.0] * 6)
    assert fast[-1]['positions_rad'][5] == pytest.approx(0.4)
    assert slow[-1]['positions_rad'][5] == pytest.approx(0.4)
    assert len(slow) > len(fast)
    assert slow[-1]['time_from_start_s'] == pytest.approx(1.35)
    assert fast_validation == source
    assert all(point['velocities_rad_s'] == [0.0] * 6 for point in fast)
    assert all(point['accelerations_rad_s2'] == [0.0] * 6 for point in fast)
    assert len(fast) == 9
    assert fast[0]['positions_rad'] == pytest.approx([0.0] * 6)
    assert fast[-1]['positions_rad'] == pytest.approx(
        [0.2, 0.0, 0.0, 0.0, 0.0, 0.4])


@pytest.mark.parametrize(
    'speed_percent,expected_duration,expected_step',
    [
        (5.0, 1.2, 0.0125),
        (20.0, 0.3, 0.05),
        # The hard step ceiling deliberately caps rates above the currently
        # qualified 20-percent stream envelope.
        (100.0, 0.3, 0.05),
    ],
)
def test_sdk_schedule_uses_speed_scaled_controller_velocity(
        speed_percent, expected_duration, expected_step):
    source = [
        {
            'time_from_start_s': 0.0,
            'positions_rad': [0.0] * 6,
            'velocities_rad_s': [0.0] * 6,
            'accelerations_rad_s2': [0.0] * 6,
        },
        {
            'time_from_start_s': 0.1,
            'positions_rad': [0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
            'velocities_rad_s': [0.0] * 6,
            'accelerations_rad_s2': [0.0] * 6,
        },
    ]
    emitted, _ = sdk_movej_waypoint_trajectory(
        source,
        speed_percent,
        20.0,
        0.05,
        [[-1.0, 1.0]] * 6,
        [3.0] * 6,
        [1000.0] * 6,
    )
    assert emitted[-1]['time_from_start_s'] == pytest.approx(
        expected_duration)
    maximum_step = max(
        abs(current['positions_rad'][0] - previous['positions_rad'][0])
        for previous, current in zip(emitted[:-1], emitted[1:])
    )
    assert maximum_step == pytest.approx(expected_step)


def test_sdk_schedule_does_not_infer_acceleration_from_movej_setpoints():
    source = [
        {
            'time_from_start_s': 0.0,
            'positions_rad': [0.0] * 6,
            'velocities_rad_s': [0.0] * 6,
            'accelerations_rad_s2': [0.0] * 6,
        },
        {
            'time_from_start_s': 0.05,
            'positions_rad': [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            'velocities_rad_s': [0.0] * 6,
            'accelerations_rad_s2': [0.0] * 6,
        },
        {
            'time_from_start_s': 0.10,
            'positions_rad': [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            'velocities_rad_s': [0.0] * 6,
            'accelerations_rad_s2': [0.0] * 6,
        },
    ]
    emitted, _ = sdk_movej_waypoint_trajectory(
        source,
        50.0,
        20.0,
        0.05,
        [[-1.0, 1.0]] * 6,
        [3.0] * 6,
        [2.0] * 6,
    )
    positions = np.asarray([
        point['positions_rad'] for point in emitted], dtype=float)
    times = np.asarray([
        point['time_from_start_s'] for point in emitted], dtype=float)
    velocities = np.diff(positions, axis=0) / np.diff(times)[:, None]
    accelerations = np.diff(velocities, axis=0) / 0.05
    assert np.max(np.abs(velocities)) <= 2.5 + 1e-6
    # Position-target corner acceleration is handled by MoveJ and must not
    # inflate the explicit model-speed schedule.
    assert emitted[-1]['time_from_start_s'] == pytest.approx(0.15)
    assert np.max(np.abs(accelerations)) > 1.0


def test_sdk_schedule_does_not_treat_isp_derivatives_as_movej_commands():
    source = [
        {
            'time_from_start_s': 0.0,
            'positions_rad': [0.0] * 6,
            'velocities_rad_s': [0.0] * 6,
            'accelerations_rad_s2': [0.0] * 6,
        },
        {
            'time_from_start_s': 0.10,
            'positions_rad': [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            'velocities_rad_s': [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'accelerations_rad_s2': [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    ]
    emitted, _ = sdk_movej_waypoint_trajectory(
        source,
        50.0,
        20.0,
        0.05,
        [[-1.0, 1.0]] * 6,
        [3.0] * 6,
        [2.0] * 6,
    )
    interval_velocity = max(
        abs(current['positions_rad'][0] - previous['positions_rad'][0]) / 0.05
        for previous, current in zip(emitted[:-1], emitted[1:])
    )
    assert interval_velocity <= 2.5 + 1e-6
    assert all(point['velocities_rad_s'] == [0.0] * 6 for point in emitted)
    assert all(point['accelerations_rad_s2'] == [0.0] * 6 for point in emitted)


def test_sdk_schedule_does_not_multiply_live_limit_at_five_percent():
    source = [
        {
            'time_from_start_s': 0.0,
            'positions_rad': [0.0] * 6,
            'velocities_rad_s': [0.0] * 6,
            'accelerations_rad_s2': [0.0] * 6,
        },
        {
            'time_from_start_s': 0.05,
            'positions_rad': [0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
            'velocities_rad_s': [0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
            'accelerations_rad_s2': [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        {
            'time_from_start_s': 0.10,
            'positions_rad': [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            'velocities_rad_s': [0.0] * 6,
            'accelerations_rad_s2': [0.0] * 6,
        },
    ]
    emitted, _ = sdk_movej_waypoint_trajectory(
        source,
        5.0,
        20.0,
        0.05,
        [[-1.0, 1.0]] * 6,
        [0.3] * 6,
        [0.5] * 6,
    )
    maximum_velocity = max(
        abs(current['positions_rad'][0] - previous['positions_rad'][0]) / 0.05
        for previous, current in zip(emitted[:-1], emitted[1:])
    )
    assert maximum_velocity == pytest.approx(0.25)
    assert emitted[-1]['time_from_start_s'] == pytest.approx(0.4)


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
        source, 100.0, 20.0, 0.10,
        [[-1.0, 1.0]] * 6, [1.0] * 6, [2.0] * 6)
    assert len(emitted) >= 2
    intervals = [
        current['time_from_start_s'] - previous['time_from_start_s']
        for previous, current in zip(emitted[:-1], emitted[1:])
    ]
    assert min(intervals) >= 0.05 - 1e-9


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
            source, 5.0, 20.0, 0.10,
            limits, [1.0] * 6, [2.0] * 6)
    emitted, _ = sdk_movej_waypoint_trajectory(
        source, 5.0, 20.0, 0.10,
        limits, [1.0] * 6, [2.0] * 6,
        bootstrap_start_limit_tolerance_rad=0.04)
    assert emitted[0]['positions_rad'][2] == pytest.approx(0.0327)
    assert emitted[-1]['positions_rad'][2] == pytest.approx(-0.01)


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
        source, 100.0, 20.0, 0.10,
        [[-1.0, 1.0]] * 6, [1.0] * 6, [2.0] * 6,
        mandatory_waypoints=[[0.02, 0.0, 0.0, 0.0, 0.0, 0.0]])
    emitted_boundary = next(
        index for index, point in enumerate(emitted)
        if np.allclose(
            point['positions_rad'],
            [0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
            atol=1e-9,
        ))
    assert 0 < emitted_boundary < len(emitted) - 1
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
            'command_rate_hz': 20.0,
            'timing_policy': 'tesseract_stream_v3',
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
            'command_rate_hz': 20.0,
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


def test_automatic_one_view_replan_has_a_tight_bounded_budget():
    request = {
        'plan_kind': 'MULTIVIEW_SCAN',
        'planning': {
            'min_viewpoints': 1,
            'max_viewpoints': 1,
            'include_return_home': False,
        },
    }
    assert planning_budgets_for_request(request) == (45.0, 3.0)

    request['planning']['include_return_home'] = True
    assert planning_budgets_for_request(request) == (150.0, 5.0)


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
            'command_rate_hz': 20.0,
            'timing_policy': 'tesseract_stream_v3',
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
            'command_rate_hz': 20.0,
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


def test_closed_loop_worker_omits_unused_embedded_return_home_segment():
    capture_points = [
        {'positions_rad': [0.0] * 6},
        {'positions_rad': [0.1] * 6},
    ]
    backend = SimpleNamespace(
        reset_scene=lambda: None,
        add_obstacles=lambda _obstacles: None,
        find_bootstrap_recovery=lambda _request: None,
        find_terminal_home_recovery=lambda *_args: pytest.fail(
            'closed-loop one-view must not plan an unused embedded home'),
        plan_segment_to_joint_goal=lambda *_args: pytest.fail(
            'closed-loop one-view must not plan an unused embedded home'),
        plan_candidate=lambda *_args: (
            0.0,
            capture_points,
            {'minimum_clearance_m': 0.1, 'limiting_link_pair': 'none/none'},
        ),
    )
    request = {
        'plan_kind': 'MULTIVIEW_SCAN',
        'scene': {
            'obstacles': [],
            'target_center_m': [0.4, 0.0, 0.0],
            'candidate_views': [{
                'id': 4,
                'camera_position_m': [0.4, 0.0, 0.3],
                'look_direction': [1.0, 0.0, 0.0],
            }],
        },
        'planning': {
            'min_viewpoints': 1,
            'max_viewpoints': 1,
            'include_return_home': False,
            'max_execution_joint_step_rad': 0.10,
            'roll_samples_rad': [0.0],
            'effective_speed_percent': 5.0,
            'command_rate_hz': 20.0,
            'return_home_positions_rad': [0.0] * 6,
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
    assert len(segments) == 1
    assert segments[0].get('is_return_home') is not True


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
            'command_rate_hz': 20.0,
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


def test_ik_goal_clearance_prescreen_uses_motion_bounded_policy():
    backend = FakeRecoveryBackend(
        lambda q: {('link2', 'link5'): float(q[0])})

    assert not TesseractBackend.state_meets_required_clearance(
        backend, np.asarray([0.099, 0, 0, 0, 0, 0]))
    assert TesseractBackend.state_meets_required_clearance(
        backend, np.asarray([0.101, 0, 0, 0, 0, 0]))


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


def test_powered_start_home_policy_requires_explicit_static_scene_flag():
    backend = SimpleNamespace(manifest={
        'powered_start_home_recovery': {
            'enabled': True,
            'plan_kind': 'RETURN_HOME',
            'observation_mode': 'perception_snapshot',
            'required_scene_flag': 'startup_home_static',
            'search_step_rad': 0.01,
            'maximum_single_joint_delta_rad': 0.10,
            'maximum_start_limit_violation_rad': 0.0,
            'allowed_start_limit_joints': [2, 3],
            'allowed_recovery_joints': [3],
            'monotonic_tolerance_m': 0.0002,
            'allowed_start_contacts': [{
                'links': ['link2', 'link5'],
                'maximum_penetration_m': 0.01,
            }],
        },
    })
    request = {
        'plan_kind': 'RETURN_HOME',
        'scene': {'observation_mode': 'perception_snapshot'},
        'limits': {'bootstrap_start_limit_tolerance_rad': 0.0},
    }
    assert TesseractBackend.bootstrap_recovery_policy(
        backend, request, 'powered_start_home_recovery') is None
    request['scene']['startup_home_static'] = True
    policy = TesseractBackend.bootstrap_recovery_policy(
        backend, request, 'powered_start_home_recovery')
    assert policy['allowed_recovery_joints'] == [3]


def test_configured_home_direct_policy_cannot_escape_return_home_stage():
    backend = SimpleNamespace(manifest={
        'configured_home_direct_joint_move': {
            'enabled': True,
            'plan_kind': 'RETURN_HOME',
            'maximum_start_limit_violation_rad': 0.3,
            'allowed_start_limit_joints': [1, 2, 3, 4, 5, 6],
            'allowed_home_stages': [
                'CONFIGURED_HOME', 'STARTUP_WRIST', 'ROUGH_HOME',
                'STORAGE_WRIST'],
        },
    })
    request = {
        'plan_kind': 'RETURN_HOME',
        'planning': {'home_stage': 'STORAGE_WRIST'},
    }
    assert TesseractBackend.configured_home_direct_policy(
        backend, request) == 'STORAGE_WRIST'

    request['plan_kind'] = 'MULTIVIEW_SCAN'
    assert TesseractBackend.configured_home_direct_policy(
        backend, request) is None

    request['plan_kind'] = 'RETURN_HOME'
    request['planning']['home_stage'] = 'UNDECLARED_STAGE'
    with pytest.raises(ContractError, match='not authorized'):
        TesseractBackend.configured_home_direct_policy(backend, request)


def test_configured_home_direct_binds_the_exact_stage_goal():
    start = [0.0, -0.1, 0.2, 0.0, 0.4, 0.0]
    storage = [0.0, -0.1, 0.2, 0.0, 0.4, -3.0]
    backend = SimpleNamespace(
        configured_home_direct_policy=lambda _request: 'STORAGE_WRIST',
        manifest={'configured_home_direct_joint_move': {
            'maximum_start_limit_violation_rad': 0.3,
            'allowed_start_limit_joints': [1, 2, 3, 4, 5, 6],
        }, 'validation_max_joint_l1_step_rad': 10.0},
        execution_position_limits=[[-3.2, 3.2]] * 6,
        command_rate_hz=100.0,
        external_floor_clearance_policy=lambda: {'enabled': True},
        external_floor_clearance_rejection=lambda _joints, _stage: '',
    )

    points, evidence = TesseractBackend.plan_configured_home_direct(
        backend, {'limits': {
            'configured_home_start_limit_tolerance_rad': 0.3,
        }}, start, storage)

    assert points[-1]['positions_rad'] == storage
    assert evidence['configured_home_goal_positions_rad'] == storage
    assert evidence['home_stage'] == 'STORAGE_WRIST'


def test_configured_home_direct_accepts_relaxed_start_but_not_relaxed_goal():
    request = {'limits': {
        'configured_home_start_limit_tolerance_rad': 0.3,
    }}
    backend = SimpleNamespace(
        configured_home_direct_policy=lambda _request: 'STARTUP_WRIST',
        manifest={'configured_home_direct_joint_move': {
            'maximum_start_limit_violation_rad': 0.3,
            'allowed_start_limit_joints': [1, 2, 3, 4, 5, 6],
        }, 'validation_max_joint_l1_step_rad': 10.0},
        execution_position_limits=[[-1.0, 1.0]] * 6,
        command_rate_hz=100.0,
        external_floor_clearance_policy=lambda: {'enabled': True},
        external_floor_clearance_rejection=lambda _joints, _stage: '',
    )
    start = [0.0, 0.0, 0.0, 0.0, 0.0, -1.03168]
    goal = [0.0] * 6
    points, _evidence = TesseractBackend.plan_configured_home_direct(
        backend, request, start, goal)
    assert points[0]['positions_rad'][5] == -1.03168

    start[5] = -1.300001
    with pytest.raises(ContractError, match='start exceeds'):
        TesseractBackend.plan_configured_home_direct(
            backend, request, start, goal)

    start[5] = 0.0
    goal[5] = 1.000001
    with pytest.raises(ContractError, match='goal exceeds'):
        TesseractBackend.plan_configured_home_direct(
            backend, request, start, goal)


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
