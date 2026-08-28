"""Native GUI responsibility-boundary regressions."""

from types import SimpleNamespace

import pytest

from piper_gui.app import run_gui
from piper_gui.ros_client import MissionActionClient
from piper_gui.view_model import (
    MissionResultView,
    MissionUiPhase,
    MissionViewModel,
    validate_mission_request,
)


class ImmediateThread:
    def __init__(self, target, args):
        self.target = target
        self.args = args

    def start(self):
        self.target(*self.args)


class FakeFuture:
    def __init__(self, result=None, error=None, immediate=True):
        self.value = result
        self.error = error
        self.immediate = immediate
        self.callbacks = []

    def result(self):
        if self.error is not None:
            raise self.error
        return self.value

    def add_done_callback(self, callback):
        self.callbacks.append(callback)
        if self.immediate:
            callback(self)

    def complete(self, result):
        self.value = result
        for callback in list(self.callbacks):
            callback(self)


class FakeGoalHandle:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.result_future = FakeFuture(immediate=False)
        self.cancel_count = 0

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancel_count += 1
        return FakeFuture(SimpleNamespace())


class FakeActionClient:
    def __init__(self, available=True, handle=None):
        self.available = available
        self.handle = handle or FakeGoalHandle()
        self.goals = []
        self.feedback_callback = None

    def wait_for_server(self, timeout_sec):
        assert timeout_sec == 5.0
        return self.available

    def send_goal_async(self, goal, feedback_callback):
        self.goals.append(goal)
        self.feedback_callback = feedback_callback
        return FakeFuture(self.handle)


def _result(**overrides):
    values = {
        'outcome': 0,
        'reason': 'complete',
        'failure_code': '',
        'retryable': False,
        'safe_shutdown': True,
        'capture_count': 12,
        'dataset_path': '/scan',
        'manifest_sha256': 'a' * 64,
        'mesh_job_id': 'mesh-1',
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_view_model_validates_input_and_only_reflects_action_lifecycle():
    view_model = MissionViewModel()
    request = view_model.begin_submission(('0.4', '0', '0.2'), '  cube  ')
    assert request.coordinates == (0.4, 0.0, 0.2)
    assert request.target_label == 'cube'
    assert view_model.state.phase == MissionUiPhase.SUBMITTING
    assert not view_model.state.can_start

    view_model.goal_accepted()
    assert view_model.state.can_cancel
    feedback = SimpleNamespace(
        phase='CAPTURE', reason='accepted', accepted_captures=3,
        required_captures=8)
    view_model.apply_feedback(feedback)
    assert '3 accepted' in view_model.state.status
    assert view_model.cancellation_requested()
    assert view_model.state.phase == MissionUiPhase.CANCELLING


@pytest.mark.parametrize('coordinates', [
    ('bad', 0, 0), (float('nan'), 0, 0), (0, 0), (0, 0, 0, 0),
])
def test_view_model_rejects_invalid_coordinates(coordinates):
    with pytest.raises(ValueError):
        validate_mission_request(coordinates, 'cube')


def test_view_model_preserves_result_fields_without_retry_decisions():
    view_model = MissionViewModel()
    view_model.begin_submission((0.4, 0.0, 0.2), 'cube')
    result = MissionResultView(
        task_id='task', outcome='FAILED', reason='wording may change',
        failure_code='SENSOR_UNAVAILABLE', retryable=True,
        safe_shutdown=True, capture_count=0, dataset_path='',
        manifest_sha256='', mesh_job_id='')
    view_model.apply_result(result)
    assert view_model.state.phase == MissionUiPhase.IDLE
    assert view_model.state.last_result is result
    assert 'SENSOR_UNAVAILABLE' in view_model.state.status
    assert result.reconstruction_payload is None


def test_success_result_exposes_only_required_reconstruction_payload():
    result = MissionResultView(
        task_id='task', outcome='SUCCEEDED', reason='done', failure_code='',
        retryable=False, safe_shutdown=True, capture_count=9,
        dataset_path='/scan', manifest_sha256='a' * 64,
        mesh_job_id='mesh')
    assert result.reconstruction_payload == {
        'task_id': 'task', 'outcome': 'SUCCEEDED', 'safe_shutdown': True,
        'dataset_path': '/scan', 'manifest_sha256': 'a' * 64,
        'mesh_job_id': 'mesh',
    }


def test_ros_client_maps_action_callbacks_and_cancellation():
    action = FakeActionClient()
    events = []
    client = MissionActionClient(
        action_client=action,
        goal_builder=lambda task_id, request: (task_id, request),
        outcome_names={0: 'SUCCEEDED'},
        event_sink=events.append,
        task_id_factory=lambda: 'task-1',
        thread_factory=lambda target, args: ImmediateThread(target, args),
    )
    request = validate_mission_request((0.4, 0, 0.2), 'cube')
    assert client.submit(request)
    assert [event.kind for event in events] == ['accepted']
    assert client.active_task_id == 'task-1'

    action.feedback_callback(SimpleNamespace(feedback=SimpleNamespace(
        phase='SCAN', reason='moving', accepted_captures=1,
        required_captures=8)))
    assert events[-1].kind == 'feedback'
    assert client.cancel()
    assert action.handle.cancel_count == 1
    assert events[-1].kind == 'cancel_requested'

    action.handle.result_future.complete(SimpleNamespace(result=_result()))
    assert events[-1].kind == 'result'
    assert events[-1].payload.outcome == 'SUCCEEDED'
    assert client.active_task_id == ''


def test_ros_client_clears_failed_or_rejected_submission():
    for action in (
            FakeActionClient(available=False),
            FakeActionClient(handle=FakeGoalHandle(accepted=False))):
        events = []
        client = MissionActionClient(
            action, lambda task_id, request: request, {0: 'SUCCEEDED'},
            events.append, task_id_factory=lambda: 'task',
            thread_factory=lambda target, args: ImmediateThread(target, args))
        assert client.submit(validate_mission_request((0, 0, 0), 'cube'))
        assert events[-1].kind == 'submission_failed'
        assert client.active_task_id == ''


def test_gui_bootstrap_lifecycle_without_tk_or_ros():
    calls = []

    class Runtime:
        def init(self): calls.append('init')
        def shutdown(self): calls.append('runtime_shutdown')

    class Node:
        def destroy_node(self): calls.append('destroy_node')

    class Executor:
        def add_node(self, _node): calls.append('add_node')
        def spin(self): calls.append('spin')
        def shutdown(self): calls.append('executor_shutdown')

    class Root:
        def mainloop(self): calls.append('mainloop')

    class App:
        def shutdown(self): calls.append('app_shutdown')

    run_gui(
        Runtime(), lambda _events: Node(),
        lambda _root, _node, _events: App(), Root, Executor)
    assert calls[0] == 'init'
    assert 'mainloop' in calls
    assert calls[-4:] == [
        'app_shutdown', 'executor_shutdown', 'destroy_node',
        'runtime_shutdown']


def test_native_gui_has_no_production_process_or_scan_controller_logic():
    source = open('piper_gui/native_app.py', encoding='utf-8').read()
    assert source.count('subprocess.Popen') == 1
    assert 'joint_preview.launch.py' in source
    assert 'start_new_session' not in source
    assert 'os.killpg' not in source
    assert 'PrepareAcquisition' not in source
    assert 'ApproveScanExecution' not in source
    assert 'RequestTesseractPlan' not in source
    assert '/scan_viewpoint_executor/cancel' not in source


def test_native_gui_exposes_all_staged_home_recording_controls():
    source = open('piper_gui/native_app.py', encoding='utf-8').read()

    assert 'Record Rough / Ready Home' in source
    assert 'Record Pre-Home (Shutdown Only)' in source
    assert 'Record Current J6 as Storage' in source


def test_native_gui_defaults_reconstruction_reference_to_35mm_cube():
    source = open('piper_gui/native_app.py', encoding='utf-8').read()

    assert "tk.StringVar(value='35') for _axis in range(3)" in source
    assert 'expected cube is 35 mm' in source
    assert 'Build Raw + Cleaned' in source
    assert 'Open Cleaned' in source
    assert 'Open Raw' in source
    assert 'Open Measured Points' in source
    assert "value='captured'" in source
    assert 'load_existing_reconstruction_outputs' in source


def test_native_gui_preserves_pre_home_when_other_stages_are_recorded():
    source = open('piper_gui/native_app.py', encoding='utf-8').read()

    assert 'pre_home_positions_rad=existing_pre_home' in source
    assert "pre_home_positions_rad=profile.get(" in source


def test_commissioning_disable_uses_feedback_service_without_hold_gate():
    source = open('piper_gui/native_app.py', encoding='utf-8').read()
    request_source = source.split(
        '    def request_safe_disable(self) -> None:', 1)[1].split(
        '    def use_feedback(self) -> None:', 1)[0]

    assert 'call_enable_async(False)' in request_source
    assert 'publish_joint_target' not in request_source
    assert '_poll_safe_disable_settle' not in source
    assert 'current-feedback hold did not settle' not in source
    assert 'Commissioning Disable (No Home)' in source


def test_commissioning_motion_starts_locked_until_graph_ownership_is_proved():
    source = open('piper_gui/native_app.py', encoding='utf-8').read()
    ros_source = open('piper_gui/ros_node.py', encoding='utf-8').read()
    ros_init = ros_source.split('class PiperGuiRos(Node):', 1)[1].split(
        '    def feedback_callback', 1)[0]
    restore = source.split(
        '    def _restore_manual_controls_if_unowned(self):', 1)[1].split(
        '    def report_tracked_robot_homed', 1)[0]

    assert 'self.manual_commands_enabled = False' in ros_init
    assert 'self.enable_manual_command_publisher()' not in ros_init
    assert 'self.enable_button.configure(state="disabled")' in source
    assert 'self.set_manual_motion_enabled(False)' in source
    assert 'resolution_timeout_sec=0.5' in restore
    assert 'disable_manual_command_publisher()' in restore
    assert 'explicit no-home Disable remains available' in restore
