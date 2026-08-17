import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from builtin_interfaces.msg import Time

from piper_mobile_manipulation.home_pose import (
    load_home_pose,
    save_home_pose,
    staged_home_targets,
)
from piper_mobile_manipulation.plan_authorizer import (
    direct_home_stage_rejection,
    direct_home_stage_targets,
)
from piper_mobile_manipulation.scan_viewpoint_executor_node import (
    ScanViewpointExecutorNode,
    sdk_command_path,
)
from piper_mobile_manipulation.target_scan_mission_node import (
    discard_failed_zero_capture_dataset,
)


def test_schema4_profile_round_trip_includes_terminal_pre_home(tmp_path):
    path = tmp_path / 'home.json'
    rough = [0.0, 0.0, 0.0, 0.0, 0.4, 0.0]
    pre_home = [0.0, 0.4, -0.5, 0.0, 0.6, 0.0]
    save_home_pose(
        path, rough, storage_joint6_rad=-math.pi,
        staged_home_configured=True,
        pre_home_positions_rad=pre_home,
        pre_home_configured=True)
    loaded = load_home_pose(path)
    targets = staged_home_targets(loaded, [0.0] * 6)
    assert loaded['schema_version'] == 4
    assert targets['pre_home_positions_rad'] == pre_home


def test_startup_direct_home_adds_positive_wrap_bridge():
    current = np.asarray([0.0, 0.0, 0.0, 0.0, 0.4, -3.5])
    goal = current.copy()
    goal[5] = 0.0
    targets = direct_home_stage_targets('STARTUP_WRIST', current, goal)
    assert len(targets) == 2
    assert targets[0][5] == 3.2 - 2.0 * math.pi
    assert targets[1][5] == 0.0


def test_direct_home_stage_rejects_wrong_pre_home_endpoint():
    current = np.zeros(6)
    rough = np.zeros(6)
    pre_home = np.asarray([0.0, 0.4, -0.5, 0.0, 0.6, 0.0])
    limits = np.asarray([[-4.2, 4.2]] * 6)
    assert 'pre-home endpoint' in direct_home_stage_rejection(
        'PRE_HOME', np.zeros(6), current, rough, limits,
        pre_home=pre_home)


def test_direct_movej_uses_only_final_anchor_after_full_path_validation():
    path = [np.zeros(6), np.full(6, 0.1), np.full(6, 0.2)]
    vectors = [np.zeros(6) for _item in path]
    command_path, _velocities, _accelerations, times, streaming = (
        sdk_command_path(
            path, vectors, vectors, [0.0, 0.5, 1.0], 'DIRECT_MOVEJ'))
    assert len(command_path) == 1
    assert np.allclose(command_path[0], path[-1])
    assert times == [1.0]
    assert streaming is False


def test_tesseract_detour_keeps_streamed_waypoints():
    path = [np.zeros(6), np.full(6, 0.1), np.full(6, 0.2)]
    vectors = [np.zeros(6) for _item in path]
    command_path, _velocities, _accelerations, _times, streaming = (
        sdk_command_path(
            path, vectors, vectors, [0.0, 0.5, 1.0],
            'TESSERACT_STREAM'))
    assert len(command_path) == 2
    assert streaming is True


def test_startup_wrist_command_is_explicitly_tagged_for_driver_gate():
    published = []
    executor = SimpleNamespace(
        command_pub=SimpleNamespace(publish=published.append),
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=Time)),
        is_configured_home_direct=lambda: True,
        plan_configured_home_stages=['STARTUP_WRIST'],
        current_view=0,
        execution_speed_percent=lambda: 25.0,
    )
    ScanViewpointExecutorNode.publish_joint_command(
        executor, np.zeros(6))
    assert published[0].header.frame_id == (
        'piper_scan_executor_startup_wrist')


def test_hold_command_tag_takes_priority_over_startup_stage():
    published = []
    executor = SimpleNamespace(
        command_pub=SimpleNamespace(publish=published.append),
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=Time)),
        is_configured_home_direct=lambda: True,
        plan_configured_home_stages=['STARTUP_WRIST'],
        current_view=0,
        execution_speed_percent=lambda: 25.0,
    )
    ScanViewpointExecutorNode.publish_joint_command(
        executor, np.zeros(6), explicit_hold=True)
    assert published[0].header.frame_id == 'piper_scan_executor_hold'


def test_startup_wrist_direct_requires_exact_authenticated_stage():
    executor = SimpleNamespace(
        is_configured_home_direct=lambda: True,
        plan_configured_home_stages=['STARTUP_WRIST'],
    )
    assert ScanViewpointExecutorNode.is_startup_wrist_direct(executor)

    executor.plan_configured_home_stages = ['ROUGH_HOME']
    assert not ScanViewpointExecutorNode.is_startup_wrist_direct(executor)
    executor.plan_configured_home_stages = ['STARTUP_WRIST']
    executor.is_configured_home_direct = lambda: False
    assert not ScanViewpointExecutorNode.is_startup_wrist_direct(executor)


def test_startup_wrist_refreshes_j1_to_j5_at_command_boundary():
    published = []
    original = np.asarray([0.07, -0.0486, 0.0256, 0.055, 0.348, -3.083])
    measured = np.asarray([0.071, 0.0, 0.0, 0.056, 0.351, -3.21])
    executor = SimpleNamespace(
        current_path=[original.copy()],
        path_index=0,
        is_startup_wrist_direct=lambda: True,
        current_joints=lambda: measured.copy(),
        publish_joint_command=lambda target: published.append(
            np.asarray(target).copy()),
        command_target=None,
        command_sent_at=0.0,
        max_command_interval_sec=0.0,
        command_samples_sent=0,
        motion_started_at=None,
        waypoint_started_at=0.0,
        waypoint_last_progress_at=0.0,
        waypoint_best_error=math.inf,
        current_waypoint_error=math.inf,
        max_waypoint_error=0.0,
        total_joint_error=lambda target: float(abs(
            measured[5] - np.asarray(target)[5])),
        max_joint_error=lambda target: float(abs(
            measured[5] - np.asarray(target)[5])),
        publish_status=lambda: None,
    )

    ScanViewpointExecutorNode.publish_next_waypoint(executor, 10.0)

    assert len(published) == 1
    assert published[0][:5] == pytest.approx(measured[:5])
    assert published[0][5] == pytest.approx(original[5])
    assert executor.current_path[0] == pytest.approx(published[0])
    assert executor.command_target == pytest.approx(published[0])


def test_startup_wrist_progress_and_convergence_are_j6_only():
    current = np.asarray([0.07, 0.0, 0.0, 0.05, 0.39, -3.20])
    target = np.asarray([0.07, -0.035, 0.017, 0.05, 0.39, -3.10])
    executor = SimpleNamespace(
        current_joints=lambda: current.copy(),
        is_startup_wrist_direct=lambda: True,
    )

    assert ScanViewpointExecutorNode.max_joint_error(
        executor, target) == pytest.approx(0.10)
    assert ScanViewpointExecutorNode.total_joint_error(
        executor, target) == pytest.approx(0.10)

    executor.is_startup_wrist_direct = lambda: False
    assert ScanViewpointExecutorNode.max_joint_error(
        executor, target) == pytest.approx(0.10)
    assert ScanViewpointExecutorNode.total_joint_error(
        executor, target) == pytest.approx(0.152)


def test_startup_wrist_bridge_advances_despite_nonowned_joint_residual():
    target = np.asarray([
        0.07954464, -0.056274344, 0.02546824,
        -0.020566476, 0.349385876, -3.083185307,
    ])
    measured = np.asarray([
        0.07954464, 0.0, 0.0,
        -0.020566476, 0.351514044, -3.083185307,
    ])
    events = []
    executor = SimpleNamespace(
        command_target=target.copy(),
        current_path=[target.copy()],
        path_index=1,
        current_joints=lambda: measured.copy(),
        is_startup_wrist_direct=lambda: True,
        current_waypoint_error=math.inf,
        max_waypoint_error=0.0,
        waypoint_best_error=0.084216768,
        waypoint_last_progress_at=0.0,
        waypoint_started_at=0.0,
        get_parameter=lambda name: SimpleNamespace(value={
            'waypoint_progress_epsilon_rad': 0.0001,
            'waypoint_reached_tolerance_rad': 0.025,
            'waypoint_timeout_sec': 90.0,
            'waypoint_progress_timeout_sec': 20.0,
        }[name]),
        abort_or_finish_captures=lambda reason: events.append(
            ('abort', reason)),
        abort_return_in_progress=False,
        record_retrace_target=lambda value: events.append(
            ('record', np.asarray(value).copy())),
        returning_home=lambda: True,
        begin_return_home_settle=lambda: events.append(('settle',)),
        publish_status=lambda: None,
    )
    executor.max_joint_error = lambda value: (
        ScanViewpointExecutorNode.max_joint_error(executor, value))
    executor.total_joint_error = lambda value: (
        ScanViewpointExecutorNode.total_joint_error(executor, value))

    ScanViewpointExecutorNode.feedback_gated_moving_tick(executor, 10.0)

    assert [event[0] for event in events] == ['record', 'settle']


def test_startup_wrist_home_settle_proves_only_stationary_j6():
    first = np.asarray([0.00, -0.048, 0.033, 0.00, 0.39, -0.020])
    second = np.asarray([0.31, 0.000, 0.000, -0.02, 0.35, -0.020])
    target = np.asarray([0.00, -0.048, 0.033, 0.00, 0.39, 0.000])
    executor = SimpleNamespace(
        latest_joint_state=SimpleNamespace(position=first.tolist()),
        updated={'joints': 1.0},
        command_target=target,
        home_settle_previous_joints=None,
        home_settle_last_joint_update=-1e9,
        home_settle_last_sample_ok=False,
        fresh=lambda key, timeout: key == 'joints' and timeout == 1.0,
        current_joints=lambda: np.asarray(
            executor.latest_joint_state.position, dtype=float),
        is_startup_wrist_direct=lambda: True,
        get_parameter=lambda name: SimpleNamespace(value={
            'home_joint_feedback_timeout_sec': 1.0,
            'home_goal_tolerance_rad': 0.3,
            'home_motion_tolerance_rad': 0.3,
        }[name]),
    )

    assert not ScanViewpointExecutorNode.home_joints_settled(executor)
    executor.latest_joint_state.position = second.tolist()
    executor.updated['joints'] = 2.0
    assert ScanViewpointExecutorNode.home_joints_settled(executor)
    assert executor.home_settle_previous_joints == pytest.approx(second)


def test_startup_wrist_settle_publishes_no_intermediate_hold():
    holds = []
    states = []
    executor = SimpleNamespace(
        current_path=[np.zeros(6)],
        current_path_velocities=[np.zeros(6)],
        current_path_accelerations=[np.zeros(6)],
        current_path_times=[0.0],
        settle_started=1.0,
        home_settle_previous_joints=np.zeros(6),
        home_settle_last_joint_update=1.0,
        home_settle_last_sample_ok=True,
        publish_hold=lambda: holds.append(True),
        is_startup_wrist_direct=lambda: True,
        set_state=lambda state, reason: states.append((state, reason)),
        abort_return_in_progress=False,
    )

    ScanViewpointExecutorNode.begin_return_home_settle(executor)

    assert holds == []
    assert states[0][0] == 'SETTLING_HOME'

    executor.current_path = [np.zeros(6)]
    executor.current_path_velocities = [np.zeros(6)]
    executor.current_path_accelerations = [np.zeros(6)]
    executor.current_path_times = [0.0]
    executor.is_startup_wrist_direct = lambda: False
    ScanViewpointExecutorNode.begin_return_home_settle(executor)
    assert holds == [True]


def test_direct_home_service_retains_attached_tool_clearance_gate():
    configuration = SimpleNamespace(
        motion=SimpleNamespace(
            return_home_positions_rad=(0.0,) * 6,
            pre_home_positions_rad=(0.0, 0.4, -0.5, 0.0, 0.6, 0.0),
            plan_start_tolerance_rad=0.025,
        ),
        safety=SimpleNamespace(
            configured_home_feedback_limit_tolerance_rad=0.3,
            motion_limits_timeout_sec=1.0,
        ),
    )
    executor = SimpleNamespace(
        state='IDLE',
        real_motion_enabled=lambda: True,
        command_pub=object(),
        mission_authorization_valid=lambda: True,
        mission_task_id='task-1234',
        mission_sha256='a' * 64,
        current_joints=lambda: np.full(6, 0.1),
        configuration=configuration,
        joint_limits=np.asarray([[-math.pi, math.pi]] * 6),
        arm_status_reasons=lambda: [],
        fresh=lambda *_args: True,
        latest_motion_limits=SimpleNamespace(
            valid=True, limits_sha256='b' * 64),
        validate_attached_tool_external_path=lambda *_args: [
            'camera holder/L515 floor clearance below 0.005 m'],
        clear_plan=lambda: None,
        speed_percent=lambda: 5.0,
        begin_runtime_refresh=lambda *_args, **_kwargs: None,
    )
    request = SimpleNamespace(
        task_id='task-1234', mission_sha256='a' * 64,
        home_stage='ROUGH_HOME', joint_goal_positions_rad=[0.0] * 6)
    response = SimpleNamespace()
    ScanViewpointExecutorNode.execute_home_stage_cb(
        executor, request, response)
    assert response.accepted is False
    assert 'attached-tool clearance gate' in response.message


def test_direct_home_service_does_not_depend_on_tesseract_limit_telemetry():
    configuration = SimpleNamespace(
        motion=SimpleNamespace(
            return_home_positions_rad=(0.0,) * 6,
            pre_home_positions_rad=(0.0, 0.4, -0.5, 0.0, 0.6, 0.0),
            plan_start_tolerance_rad=0.025,
        ),
        safety=SimpleNamespace(
            configured_home_feedback_limit_tolerance_rad=0.3,
        ),
    )
    refreshes = []
    executor = SimpleNamespace(
        state='IDLE',
        real_motion_enabled=lambda: True,
        command_pub=object(),
        mission_authorization_valid=lambda: True,
        mission_task_id='task-1234',
        mission_sha256='a' * 64,
        current_joints=lambda: np.full(6, 0.1),
        configuration=configuration,
        joint_limits=np.asarray([[-math.pi, math.pi]] * 6),
        arm_status_reasons=lambda: [],
        validate_attached_tool_external_path=lambda *_args: [],
        clear_plan=lambda: None,
        speed_percent=lambda: 5.0,
        begin_runtime_refresh=lambda *args, **kwargs: refreshes.append(
            (args, kwargs)),
    )
    request = SimpleNamespace(
        task_id='task-1234', mission_sha256='a' * 64,
        home_stage='ROUGH_HOME', joint_goal_positions_rad=[0.0] * 6)
    response = SimpleNamespace()

    ScanViewpointExecutorNode.execute_home_stage_cb(
        executor, request, response)

    assert response.accepted
    assert executor.plan_motion_limits_sha256 == ''
    assert executor.runtime_motion_limits_sha256 == ''
    assert executor.plan_collision_model_qualified is False
    assert refreshes


def test_zero_capture_cleanup_requires_identity_and_no_completed_frame(tmp_path):
    root = tmp_path / 'active_scan'
    scan = root / 'scan_20260815_120000'
    frames = scan / 'frames'
    frames.mkdir(parents=True)
    metadata = {'task_id': 'task-1234', 'mission_sha256': 'a' * 64}
    (scan / 'metadata.yaml').write_text(
        yaml.safe_dump(metadata), encoding='utf-8')
    manifest = {'capture_count': 0}
    (scan / 'manifest.json').write_text(
        json.dumps(manifest), encoding='utf-8')
    removed, _reason = discard_failed_zero_capture_dataset(
        scan, root, 'task-1234', 'a' * 64)
    assert removed is True
    assert not scan.exists()


def test_zero_capture_cleanup_refuses_completed_capture(tmp_path):
    root = tmp_path / 'active_scan'
    scan = root / 'scan_20260815_120001'
    frames = scan / 'frames'
    frames.mkdir(parents=True)
    metadata = {'task_id': 'task-1234', 'mission_sha256': 'a' * 64}
    (scan / 'metadata.yaml').write_text(
        yaml.safe_dump(metadata), encoding='utf-8')
    (frames / 'view_000_metadata.yaml').write_text('{}\n', encoding='utf-8')
    removed, reason = discard_failed_zero_capture_dataset(
        scan, root, 'task-1234', 'a' * 64)
    assert removed is False
    assert 'completed captures' in reason
    assert scan.exists()
