"""Characterization tests for coherent immutable telemetry snapshots."""

from dataclasses import FrozenInstanceError
import threading
from types import SimpleNamespace

import pytest

from piper_mobile_manipulation.telemetry_store import TelemetryStore
from piper_mobile_manipulation.scan_viewpoint_executor_node import (
    ScanViewpointExecutorNode,
)
from piper_mobile_manipulation.target_scan_mission_node import (
    TargetScanMissionNode,
)


class FakeClock:
    """Deterministic monotonic clock double."""

    def __init__(self, now=0.0):
        self.now = float(now)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)


def test_empty_store_has_one_coherent_empty_snapshot():
    clock = FakeClock(12.5)
    snapshot = TelemetryStore(clock).snapshot()

    assert snapshot.captured_at == 12.5
    assert snapshot.revision == 0
    assert snapshot.arm.joints is None
    assert snapshot.arm.status is None
    assert snapshot.arm.motion_limits is None
    assert snapshot.perception.camera is None
    assert snapshot.perception.target is None
    assert snapshot.perception.tracking is None
    assert snapshot.perception.target_status is None
    assert snapshot.perception.obstacles is None
    assert snapshot.mission.readiness is None
    assert snapshot.mission.plan is None
    assert snapshot.mission.execution is None
    assert snapshot.mission.capture is None
    assert snapshot.mission.scan_history is None
    assert snapshot.mission.reachable_scan is None
    assert snapshot.mission.workflow is None


def test_partial_telemetry_preserves_receipt_source_and_frame_metadata():
    store = TelemetryStore(FakeClock(99.0))
    joints = SimpleNamespace(position=[1.0] * 6)

    store.update_joints(
        joints, received_at=4.25, source_stamp_ns=123456,
        frame_id='base_link')
    snapshot = store.snapshot()

    assert snapshot.arm.joints.value.position == [1.0] * 6
    assert snapshot.arm.joints.received_at == 4.25
    assert snapshot.arm.joints.source_stamp_ns == 123456
    assert snapshot.arm.joints.frame_id == 'base_link'
    assert snapshot.arm.status is None
    assert snapshot.perception.target is None


def test_complete_telemetry_has_explicit_domain_channels():
    store = TelemetryStore(FakeClock(10.0))
    updates = (
        (store.update_joints, 'joints'),
        (store.update_arm_status, 'status'),
        (store.update_motion_limits, 'limits'),
        (store.update_camera, 'camera'),
        (store.update_target, 'target'),
        (store.update_tracking, 'tracking'),
        (store.update_target_status, 'LOCKED'),
        (store.update_obstacles, 'obstacles'),
        (store.update_readiness, 'readiness'),
        (store.update_plan, 'plan'),
        (store.update_execution, 'execution'),
        (store.update_capture, 'capture'),
        (store.update_scan_history, 'history'),
        (store.update_reachable_scan, 'reachable'),
        (store.update_workflow, 'workflow'),
    )
    for index, (update, value) in enumerate(updates):
        update(value, received_at=float(index))

    snapshot = store.snapshot()

    assert snapshot.revision == len(updates)
    assert snapshot.arm.joints.value == 'joints'
    assert snapshot.arm.status.value == 'status'
    assert snapshot.arm.motion_limits.value == 'limits'
    assert snapshot.perception.camera.value == 'camera'
    assert snapshot.perception.target.value == 'target'
    assert snapshot.perception.tracking.value == 'tracking'
    assert snapshot.perception.target_status.value == 'LOCKED'
    assert snapshot.perception.obstacles.value == 'obstacles'
    assert snapshot.mission.readiness.value == 'readiness'
    assert snapshot.mission.plan.value == 'plan'
    assert snapshot.mission.execution.value == 'execution'
    assert snapshot.mission.capture.value == 'capture'
    assert snapshot.mission.scan_history.value == 'history'
    assert snapshot.mission.reachable_scan.value == 'reachable'
    assert snapshot.mission.workflow.value == 'workflow'


def test_age_and_stale_calculation_use_injected_clock_without_thresholds():
    clock = FakeClock(5.0)
    store = TelemetryStore(clock)
    store.update_camera('healthy')
    clock.advance(0.75)

    observation = store.snapshot().perception.camera

    assert observation.age_at(clock()) == pytest.approx(0.75)
    assert observation.is_stale_at(clock(), 0.75) is False
    clock.advance(0.0001)
    assert observation.is_stale_at(clock(), 0.75) is True


def test_snapshot_dataclasses_are_frozen_and_values_are_defensive_copies():
    store = TelemetryStore(FakeClock(1.0))
    mutable = {'samples': [1, 2]}
    store.update_capture(mutable)
    snapshot = store.snapshot()

    with pytest.raises(FrozenInstanceError):
        snapshot.captured_at = 2.0
    with pytest.raises(FrozenInstanceError):
        snapshot.mission.capture.received_at = 2.0

    mutable['samples'].append(3)
    snapshot.mission.capture.value['samples'].append(4)
    next_snapshot = store.snapshot()

    assert snapshot.mission.capture.value['samples'] == [1, 2, 4]
    assert next_snapshot.mission.capture.value['samples'] == [1, 2]


def test_runtime_snapshot_excludes_bulk_evidence_and_stays_defensive():
    class CountedCopy:
        copies = 0

        def __deepcopy__(self, _memo):
            type(self).copies += 1
            return type(self)()

    store = TelemetryStore(FakeClock(2.0))
    store.update_joints({'positions': [0.1] * 6})
    store.update_arm_status('status')
    store.update_motion_limits('limits')
    store.update_camera('camera')
    store.update_target('target')
    store.update_tracking('tracking')
    store.update_target_status('LOCKED')
    store.update_obstacles('obstacles')
    store.update_workflow({'state': 'SCAN_READY'})
    store.update_reachable_scan(CountedCopy())
    copies_after_update = CountedCopy.copies

    snapshot = store.runtime_snapshot()

    assert CountedCopy.copies == copies_after_update
    assert snapshot.arm.joints.value['positions'] == [0.1] * 6
    assert snapshot.perception.target.value == 'target'
    assert snapshot.mission.workflow.value == {'state': 'SCAN_READY'}
    assert snapshot.mission.readiness is None
    assert snapshot.mission.plan is None
    assert snapshot.mission.execution is None
    assert snapshot.mission.capture is None
    assert snapshot.mission.scan_history is None
    assert snapshot.mission.reachable_scan is None

    snapshot.arm.joints.value['positions'].append(0.2)
    assert store.runtime_snapshot().arm.joints.value['positions'] == [0.1] * 6


def test_execution_tick_never_copies_bulk_ray_evidence_for_motion():
    class RuntimeOnlyStore(TelemetryStore):
        def snapshot(self):
            raise AssertionError('full snapshot entered the execution tick')

    store = RuntimeOnlyStore(FakeClock(5.0))
    store.update_obstacles(SimpleNamespace(
        instances=[], scene_blocked=False))
    calls = []
    executor = SimpleNamespace(
        telemetry_store=store,
        state='MOVING',
        plan_kind='MULTIVIEW_SCAN',
        current_view=0,
        plan_collision_model_qualified=True,
        acquisition_scene_snapshot_validated=False,
        abort_return_bootstrap_static_scene=False,
        is_acquisition=lambda: False,
        is_return_home=lambda: False,
        is_startup_home_static=lambda: False,
        returning_home=lambda: False,
        runtime_reasons=lambda policy, **kwargs: calls.append(
            ('gate', policy, kwargs['telemetry_snapshot'])) or [],
        moving_tick=lambda telemetry_snapshot=None: calls.append(
            ('moving', telemetry_snapshot)),
    )

    ScanViewpointExecutorNode.execution_tick(executor)

    assert calls[0][0] == 'gate'
    assert calls[1][0] == 'moving'
    assert calls[0][2] is calls[1][1]


def test_snapshot_consistency_keeps_value_timestamp_and_revision_together():
    clock = FakeClock(1.0)
    store = TelemetryStore(clock)
    store.update_joints({'sample': 1}, received_at=1.0)
    first = store.snapshot()
    store.update_joints({'sample': 2}, received_at=2.0)
    second = store.snapshot()

    assert first.arm.joints.value == {'sample': 1}
    assert first.arm.joints.received_at == 1.0
    assert first.arm.joints.revision <= first.revision
    assert second.arm.joints.value == {'sample': 2}
    assert second.arm.joints.received_at == 2.0
    assert second.arm.joints.revision <= second.revision


def test_concurrent_updates_never_produce_torn_observations():
    store = TelemetryStore(FakeClock(1000.0))
    barrier = threading.Barrier(3)
    errors = []

    def writer(update, prefix):
        barrier.wait()
        for sample in range(500):
            update(
                {'prefix': prefix, 'sample': sample},
                received_at=float(sample), source_stamp_ns=sample)

    threads = [
        threading.Thread(target=writer, args=(store.update_joints, 'joint')),
        threading.Thread(target=writer, args=(store.update_target, 'target')),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    while any(thread.is_alive() for thread in threads):
        snapshot = store.snapshot()
        for observation in (
                snapshot.arm.joints, snapshot.perception.target):
            if observation is None:
                continue
            sample = observation.value['sample']
            if (
                    observation.received_at != float(sample)
                    or observation.source_stamp_ns != sample
                    or observation.revision > snapshot.revision):
                errors.append((observation, snapshot.revision))
    for thread in threads:
        thread.join()

    assert errors == []


def test_selective_clear_operations_do_not_erase_unrelated_observations():
    store = TelemetryStore(FakeClock(1.0))
    store.update_joints('joints')
    store.update_camera('camera')
    store.update_plan('plan')
    store.update_capture('capture')

    store.clear_plan()
    after_plan_clear = store.snapshot()
    assert after_plan_clear.mission.plan is None
    assert after_plan_clear.mission.capture.value == 'capture'

    store.clear_mission_runtime()
    after_runtime_clear = store.snapshot()
    assert after_runtime_clear.mission.capture is None
    assert after_runtime_clear.arm.joints.value == 'joints'
    assert after_runtime_clear.perception.camera.value == 'camera'

    store.clear_arm_feedback()
    store.clear_camera()
    final = store.snapshot()
    assert final.arm.joints is None
    assert final.perception.camera is None


def test_executor_callback_updates_legacy_mirror_and_store_together():
    clock = FakeClock(42.0)
    executor = SimpleNamespace(
        telemetry_store=TelemetryStore(clock),
        updated={},
        now=clock,
        latest_tracked_target=None,
    )
    message = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=7, nanosec=8),
            frame_id='base_link'),
        position=SimpleNamespace(x=0.4, y=0.0, z=0.1),
    )

    ScanViewpointExecutorNode.tracked_target_cb(executor, message)
    snapshot = executor.telemetry_store.snapshot()

    assert executor.latest_tracked_target is message
    assert executor.updated['tracked_target'] == 42.0
    assert snapshot.perception.target.value.position.x == 0.4
    assert snapshot.perception.target.received_at == 42.0
    assert snapshot.perception.target.source_stamp_ns == 7_000_000_008
    assert snapshot.perception.target.frame_id == 'base_link'


def test_mission_arm_status_callback_accepts_headerless_piper_status():
    mission = SimpleNamespace(
        _lock=threading.RLock(),
        latest_arm_status=None,
        latest_arm_status_at=0.0,
        telemetry_store=TelemetryStore(),
    )
    message = SimpleNamespace(
        err_code=0,
        motor_feedback_valid=True,
        motor_1_driver_enabled=False,
    )

    TargetScanMissionNode.arm_status_cb(mission, message)
    observation = mission.telemetry_store.snapshot().arm.status

    assert mission.latest_arm_status is message
    assert mission.latest_arm_status_at > 0.0
    assert observation.value.err_code == 0
    assert observation.frame_id == ''


def test_mission_readiness_decision_matches_legacy_fields():
    now = 1e12
    readiness = SimpleNamespace(
        worker_ready=True,
        acquisition_ready=True,
        multiview_ready=False,
        manipulation_ready=False,
        acquisition_blockers=[],
        multiview_blockers=['planner warming'],
        manipulation_blockers=['not qualified'],
    )
    legacy = SimpleNamespace(
        latest_readiness=readiness,
        latest_readiness_at=now,
    )
    migrated = SimpleNamespace(
        latest_readiness=readiness,
        latest_readiness_at=now,
        telemetry_store=TelemetryStore(lambda: now),
    )
    migrated.telemetry_store.update_readiness(
        readiness, received_at=now)

    legacy_result = TargetScanMissionNode.readiness_rejection(
        legacy, 'multiview')
    migrated_result = TargetScanMissionNode.readiness_rejection(
        migrated, 'multiview')

    assert migrated_result == legacy_result == 'planner warming'


def test_executor_freshness_decision_matches_legacy_boundary_semantics():
    clock = FakeClock(10.0)
    parameters = {
        'data_timeout_sec': 2.0,
    }
    legacy = SimpleNamespace(
        now=clock,
        updated={'joints': 8.0},
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
    )
    migrated = SimpleNamespace(
        now=clock,
        updated={'joints': 8.0},
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        telemetry_store=TelemetryStore(clock),
    )
    migrated.telemetry_store.update_joints(
        SimpleNamespace(position=[0.0] * 6), received_at=8.0)

    assert ScanViewpointExecutorNode.fresh(legacy, 'joints') is True
    assert ScanViewpointExecutorNode.fresh(migrated, 'joints') is True

    clock.advance(0.0001)
    assert ScanViewpointExecutorNode.fresh(legacy, 'joints') is False
    assert ScanViewpointExecutorNode.fresh(migrated, 'joints') is False


def test_executor_scan_freshness_retains_full_snapshot_contract():
    clock = FakeClock(10.0)
    executor = SimpleNamespace(
        now=clock,
        updated={'scan': 9.0},
        get_parameter=lambda _name: SimpleNamespace(value=2.0),
        telemetry_store=TelemetryStore(clock),
    )
    executor.telemetry_store.update_reachable_scan(
        {'generation': 4}, received_at=9.0)

    assert ScanViewpointExecutorNode.fresh(executor, 'scan') is True
    clock.advance(1.0001)
    assert ScanViewpointExecutorNode.fresh(executor, 'scan') is False


def test_executor_joint_decision_uses_one_snapshot_and_matches_legacy():
    joints = SimpleNamespace(position=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    legacy = SimpleNamespace(latest_joint_state=joints)
    migrated = SimpleNamespace(
        latest_joint_state=joints,
        telemetry_store=TelemetryStore(lambda: 1.0),
    )
    migrated.telemetry_store.update_joints(joints, received_at=1.0)

    legacy_values = ScanViewpointExecutorNode.current_joints(legacy)
    migrated_values = ScanViewpointExecutorNode.current_joints(migrated)

    assert migrated_values.tolist() == legacy_values.tolist()


def test_capture_settle_decision_reuses_its_camera_snapshot(monkeypatch):
    clock = FakeClock(3.0)
    executor = SimpleNamespace(telemetry_store=TelemetryStore(clock))
    executor.telemetry_store.update_camera(
        SimpleNamespace(healthy=True), received_at=3.0)
    received = []

    def settled(_executor, settle_at_current=False, snapshot=None):
        received.append(snapshot)
        return snapshot.perception.camera.value.healthy

    monkeypatch.setattr(
        ScanViewpointExecutorNode, 'joints_settled', settled)

    assert ScanViewpointExecutorNode.capture_pose_settled(executor) is True
    assert len(received) == 1
    assert received[0].perception.camera.value.healthy is True
