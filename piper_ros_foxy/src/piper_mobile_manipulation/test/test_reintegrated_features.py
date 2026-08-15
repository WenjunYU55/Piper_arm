import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
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
