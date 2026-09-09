"""Clearance-aware stream geometry and command admission regressions."""
from types import SimpleNamespace
import math
import numpy as np
import pytest
from piper_mobile_manipulation.execution.clearance import (
    clearance_margin, segment_box_distance, tracking_limit,
)
from piper_mobile_manipulation.execution.motion import PiperScanKinematics, CollisionBox
from piper_mobile_manipulation.scan_viewpoint_executor_node import ScanViewpointExecutorNode


def test_segment_box_distance_interior_crossing_and_parallel_gap():
    lo, hi = np.zeros(3), np.ones(3)
    assert segment_box_distance([-1, .5, .5], [2, .5, .5], lo, hi) == 0.
    assert segment_box_distance([-1, 2, .5], [2, 2, .5], lo, hi) == pytest.approx(1.)
    assert segment_box_distance([2, 2, 2], [2, 2, 2], lo, hi) == pytest.approx(math.sqrt(3))


def test_closing_motion_tightens_early_without_loosening_open_limit():
    limit = lambda d, v=0.: tracking_limit(d, v, .25, .03, .30, .10)
    assert limit(.3) == .30
    assert limit(0.) == .03
    assert .03 < limit(.05) < .30
    assert limit(.05, .2) == .03
    assert limit(.01, 10.) == .03
    with pytest.raises(ValueError):
        limit(float('nan'))


def test_recorded_near_table_pose_violates_measured_envelope():
    kin = PiperScanKinematics(np.eye(4))
    recorded = [-.27378358, 2.5943589, -1.997547328, -1.647533468, .035393876, -1.363231156]
    margin, _ = clearance_margin(kin, recorded, [], .005, .025, .060,
                                 [-.029750002, 0., .0375], [.1395, .10572671, .053], .005)
    assert margin <= -.0338
    with pytest.raises(ValueError):
        clearance_margin(kin, [float('nan')] * 6, [], .005, .025, .060,
                         [0., 0., 0.], [.1, .1, .1], .005)


def test_external_box_inside_holder_is_not_open_space():
    kin = PiperScanKinematics(np.eye(4))
    q = [0., .8, -.7, 0., .7, 0.]
    xyz = kin.forward(q)[:3, 3]
    box = CollisionBox('test', xyz - .1, xyz + .1)
    margin, _ = clearance_margin(kin, q, [box], .005, .025, .060,
                                 [-.029750002, 0., .0375], [.1395, .10572671, .053], .005)
    assert margin < 0.


def test_clearance_fault_stops_stream_before_any_target_is_published():
    failures = []
    def guard(_now):
        raise ValueError('measured collision clearance violated: holder/table')
    fake = SimpleNamespace(
        motion_started_at=0., stream_wall_started_at=0.,
        get_parameter=lambda name: SimpleNamespace(value=.30),
        stream_clearance_limit=guard,
        abort_or_finish_captures=failures.append,
        publish_joint_command=lambda _: pytest.fail('published after collision fault'))
    ScanViewpointExecutorNode.streaming_moving_tick(fake, .005)
    assert len(failures) == 1 and 'collision monitor' in failures[0]


def test_real_monitor_adapts_limit_and_preserves_direct_home(monkeypatch):
    import piper_mobile_manipulation.scan_viewpoint_executor_node as module
    monkeypatch.setattr(module, 'runtime_telemetry_snapshot', lambda _: None)
    monkeypatch.setattr(module, 'clearance_margin',
                        lambda _k, q, *args, **kwargs: (float(q[0]), 'test/table'))
    parameters = dict(trajectory_following_error_rad=.3,
                      trajectory_clearance_monitor_enabled=True,
                      trajectory_near_collision_error_rad=.03,
                      trajectory_clearance_transition_m=.1,
                      trajectory_clearance_reaction_sec=.25,
                      floor_z_m=.005, link_radius_m=.025, self_clearance_m=.06,
                      camera_holder_envelope_center_link6_m=[0., 0., 0.],
                      camera_holder_envelope_size_m=[.1, .1, .1],
                      camera_holder_external_clearance_m=.005)
    q = np.array([.2, 0., 0., 0., 0., 0.])
    node = SimpleNamespace(
        plan_kind='MULTIVIEW_SCAN', plan_backend='tesseract', kinematics=None, path_index=0,
        current_path=[q.copy()], current_joints=lambda: q.copy(),
        stream_clearance_state=dict(boxes=(), ready=True, recovery_end=0),
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        get_logger=lambda: SimpleNamespace(info=lambda _: None))
    assert ScanViewpointExecutorNode.stream_clearance_limit(node, 1.) == .3
    q[0] = .05
    assert ScanViewpointExecutorNode.stream_clearance_limit(node, 1.1) == .03
    q[0] = -.01
    with pytest.raises(ValueError, match='collision clearance'):
        ScanViewpointExecutorNode.stream_clearance_limit(node, 1.2)
    node.stream_clearance_state = dict(boxes=(), ready=False, recovery_end=10)
    with pytest.raises(ValueError, match='collision clearance'):
        ScanViewpointExecutorNode.stream_clearance_limit(node, 1.25)
    node.plan_kind = 'RETURN_HOME'
    assert ScanViewpointExecutorNode.stream_clearance_limit(node, 1.3) == .3


def test_nominal_wrist_self_margin_does_not_slow_open_space():
    # The normal wrist proxy has ~31 mm spare margin even away from the table.
    # A 100 mm transition would therefore tighten every ordinary command.
    assert tracking_limit(.031, 0., .25, .03, .30, .025) == .30
    assert tracking_limit(.031, .1, .25, .03, .30, .025) < .30


@pytest.mark.parametrize('backend', ['curobo', ''])
def test_other_backends_bypass_clearance_monitor_before_geometry(backend):
    # No scene/FK is provided: cuRobo must retain its existing limit even when
    # the new monitor is enabled in the shared execution configuration.
    def parameter(name):
        assert name == 'trajectory_following_error_rad'
        return SimpleNamespace(value=.30)
    node = SimpleNamespace(plan_backend=backend, get_parameter=parameter)
    assert ScanViewpointExecutorNode.stream_clearance_limit(node, 1.) == .30
