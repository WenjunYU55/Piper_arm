"""Pure no-hardware tests for the Phase 5 shadow safety evaluator."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from piper_mobile_manipulation.failure_model import FailureCode
from piper_mobile_manipulation.safety_evaluator import (
    SafetyAuthorization,
    SafetyComparisonLogger,
    SafetyEvaluator,
    SafetyInputs,
    SafetyMode,
    SafetyProfile,
)
from piper_mobile_manipulation.scan_viewpoint_executor_node import (
    ScanViewpointExecutorNode,
)
from piper_mobile_manipulation.telemetry_store import (
    ArmTelemetry,
    MissionTelemetry,
    PerceptionTelemetry,
    TelemetryObservation,
    TelemetrySnapshot,
    TelemetryStore,
)


NOW = 100.0


def observation(value, age=0.1):
    return TelemetryObservation(value=value, received_at=NOW - age)


def healthy_status(**updates):
    values = {
        'err_code': 0,
        'motor_feedback_valid': True,
        'motor_faults': [],
        'motor_watchdog_reason': '',
    }
    for index in range(1, 7):
        values['joint_%d_angle_limit' % index] = False
        values['communication_status_joint_%d' % index] = False
        values['motor_%d_driver_enabled' % index] = True
    values.update(updates)
    return SimpleNamespace(**values)


def snapshot(
        joint_age=0.1, status_age=0.1, limits_age=0.1,
        camera_age=0.1, tracking_age=0.1, target_age=0.1,
        obstacle_age=0.1, joints=True, status=None, camera=True,
        tracking=True, target_status='LOCKED', obstacles=True,
        limits=True, workflow=True):
    joint_value = SimpleNamespace(position=[0.0] * 6)
    limit_value = SimpleNamespace(
        valid=True, limits_sha256='a' * 64,
        max_velocity_rad_s=[1.0] * 6,
        max_acceleration_rad_s2=[1.0] * 6)
    camera_value = SimpleNamespace(
        healthy=True, state='HEALTHY', reason='')
    tracking_value = SimpleNamespace(
        lifecycle_state='TRACKING', camera_settled=True,
        prediction_only=False, measurement_age_sec=0.1,
        recommended_speed_scale=1.0)
    obstacle_value = SimpleNamespace(
        scene_blocked=False, blocking_reason='', instances=[])
    return TelemetrySnapshot(
        captured_at=NOW,
        revision=1,
        arm=ArmTelemetry(
            joints=observation(joint_value, joint_age) if joints else None,
            status=observation(
                healthy_status() if status is None else status,
                status_age) if status is not False else None,
            motion_limits=observation(
                limit_value, limits_age) if limits else None),
        perception=PerceptionTelemetry(
            camera=observation(
                camera_value, camera_age) if camera else None,
            tracking=observation(
                tracking_value, tracking_age) if tracking else None,
            target_status=observation(
                target_status, target_age)
            if target_status is not None else None,
            obstacles=observation(
                obstacle_value, obstacle_age) if obstacles else None),
        mission=MissionTelemetry(
            workflow=observation(
                {'state': 'SCAN_READY'} if workflow else {'state': 'IDLE'})),
    )


@pytest.fixture
def evaluator():
    return SafetyEvaluator(SafetyProfile(
        data_timeout_sec=2.0,
        motion_limits_timeout_sec=3.0,
        max_tracking_measurement_age_sec=0.75,
        min_tracking_speed_scale=0.10,
        configured_speed_percent=30.0,
        max_target_drift_before_approval_m=0.015,
        joint_feedback_limit_tolerance_rad=0.005,
        configured_home_feedback_limit_tolerance_rad=0.3,
        hold_joint_feedback_timeout_sec=1.0,
    ))


@pytest.mark.parametrize('mode,inputs', [
    (SafetyMode.PLAN_VALIDATION, SafetyInputs()),
    (SafetyMode.ACQUISITION_APPROVAL,
     SafetyInputs(joints_settled=True)),
    (SafetyMode.ACQUISITION_MOTION,
     SafetyInputs(static_obstacle_scene_authorized=True)),
    (SafetyMode.SCAN_APPROVAL,
     SafetyInputs(joints_settled=True, workflow_required=True)),
    (SafetyMode.SCAN_MOTION,
     SafetyInputs(approved_obstacle_snapshot=True)),
    (SafetyMode.SCAN_CAPTURE,
     SafetyInputs(joints_settled=True)),
    (SafetyMode.RETURN_HOME, SafetyInputs()),
    (SafetyMode.HOLD_CURRENT, SafetyInputs(joints_settled=True)),
])
def test_all_observed_modes_accept_nominal_evidence(evaluator, mode, inputs):
    decision = evaluator.evaluate(mode, snapshot(), inputs)
    assert decision.permitted is True
    assert decision.failure is None


@pytest.mark.parametrize('mode', [
    SafetyMode.ACQUISITION_APPROVAL,
    SafetyMode.SCAN_APPROVAL,
    SafetyMode.SCAN_MOTION,
    SafetyMode.RETURN_HOME,
    SafetyMode.HOLD_CURRENT,
])
def test_stale_joint_feedback_rejects_every_joint_dependent_mode(
        evaluator, mode):
    decision = evaluator.evaluate(
        mode, snapshot(joint_age=3.1),
        SafetyInputs(
            joints_settled=True,
            static_obstacle_scene_authorized=True,
            approved_obstacle_snapshot=True))
    assert decision.permitted is False
    assert 'joints data missing or stale' in decision.reasons
    assert decision.failure_code == FailureCode.CONTROL_UNTRUSTWORTHY


def test_hold_uses_unchanged_one_second_joint_feedback_timeout(evaluator):
    assert evaluator.evaluate(
        SafetyMode.HOLD_CURRENT, snapshot(joint_age=1.0),
        SafetyInputs(joints_settled=True)).permitted
    assert not evaluator.evaluate(
        SafetyMode.HOLD_CURRENT, snapshot(joint_age=1.0001),
        SafetyInputs(joints_settled=True)).permitted


def test_camera_staleness_blocks_scan_but_not_return_home(evaluator):
    stale = snapshot(camera_age=2.1)
    scan = evaluator.evaluate(
        SafetyMode.SCAN_MOTION, stale,
        SafetyInputs(approved_obstacle_snapshot=True))
    home = evaluator.evaluate(SafetyMode.RETURN_HOME, stale)
    assert not scan.permitted
    assert 'camera_clock data missing or stale' in scan.reasons
    assert home.permitted


def test_target_lost_blocks_scan_approval_but_not_approved_motion(evaluator):
    lost = snapshot(target_status='LOST')
    approval = evaluator.evaluate(
        SafetyMode.SCAN_APPROVAL, lost,
        SafetyInputs(joints_settled=True))
    in_flight = evaluator.evaluate(
        SafetyMode.SCAN_MOTION, lost,
        SafetyInputs(approved_obstacle_snapshot=True))
    assert not approval.permitted
    assert approval.reacquisition_required
    assert 'target_status=LOST' in approval.reasons
    assert in_flight.permitted


def test_tracking_measurement_age_and_prediction_only_are_explicit(evaluator):
    data = snapshot()
    tracking = data.perception.tracking.value
    tracking.measurement_age_sec = 0.76
    tracking.prediction_only = True
    decision = evaluator.evaluate(
        SafetyMode.SCAN_APPROVAL, data,
        SafetyInputs(joints_settled=True))
    assert not decision.permitted
    assert 'tracking is prediction-only' in decision.reasons
    assert 'tracking measurement is stale' in decision.reasons
    assert decision.reacquisition_required


def test_obstacles_missing_rules_follow_operating_mode(evaluator):
    no_scene = snapshot(obstacles=False)
    acquisition_bootstrap = evaluator.evaluate(
        SafetyMode.ACQUISITION_MOTION, no_scene,
        SafetyInputs(static_obstacle_scene_authorized=True))
    acquisition_later = evaluator.evaluate(
        SafetyMode.ACQUISITION_MOTION, no_scene)
    capture = evaluator.evaluate(
        SafetyMode.SCAN_CAPTURE, no_scene,
        SafetyInputs(joints_settled=True))
    assert acquisition_bootstrap.permitted
    assert not acquisition_later.permitted
    assert capture.permitted


def test_approved_obstacle_snapshot_waives_age_not_unsafe_geometry(evaluator):
    stale = snapshot(obstacle_age=5.0)
    assert evaluator.evaluate(
        SafetyMode.SCAN_MOTION, stale,
        SafetyInputs(approved_obstacle_snapshot=True)).permitted
    stale.perception.obstacles.value.scene_blocked = True
    stale.perception.obstacles.value.blocking_reason = 'clutter'
    decision = evaluator.evaluate(
        SafetyMode.SCAN_MOTION, stale,
        SafetyInputs(approved_obstacle_snapshot=True))
    assert not decision.permitted
    assert 'scene_blocked: clutter' in decision.reasons


def test_motion_limits_missing_or_invalid_rejects_motion(evaluator):
    missing = evaluator.evaluate(
        SafetyMode.RETURN_HOME, snapshot(limits=False))
    invalid_data = snapshot()
    invalid_data.arm.motion_limits.value.valid = False
    invalid = evaluator.evaluate(SafetyMode.RETURN_HOME, invalid_data)
    assert not missing.permitted
    assert 'motion_limits data missing or stale' in missing.reasons
    assert not invalid.permitted
    assert invalid.failure_code == FailureCode.CONTROL_UNTRUSTWORTHY


def test_absent_mission_authorization_rejects(evaluator):
    decision = evaluator.evaluate(
        SafetyMode.SCAN_APPROVAL, snapshot(),
        SafetyInputs(joints_settled=True),
        SafetyAuthorization(required=True, granted=False))
    assert not decision.permitted
    assert decision.failure_code == FailureCode.CONTROL_UNTRUSTWORTHY


def test_target_drift_rejects_and_requests_replan(evaluator):
    decision = evaluator.evaluate(
        SafetyMode.SCAN_APPROVAL, snapshot(),
        SafetyInputs(joints_settled=True, target_drift_m=0.016))
    assert not decision.permitted
    assert decision.replan_required
    assert decision.failure_code == FailureCode.NO_REACHABLE_PLAN


@pytest.mark.parametrize('field', [
    'planner_result_valid', 'plan_schema_valid',
    'collision_model_qualified', 'path_valid',
])
def test_invalid_planner_or_path_evidence_rejects(evaluator, field):
    values = {field: False}
    decision = evaluator.evaluate(
        SafetyMode.PLAN_VALIDATION, snapshot(), SafetyInputs(**values))
    assert not decision.permitted
    assert decision.failure_code == FailureCode.NO_REACHABLE_PLAN


def test_motor_control_unavailable_rejects_motion_but_matches_legacy_hold(
        evaluator):
    unavailable = snapshot(status=healthy_status(
        motor_feedback_valid=False,
        motor_1_driver_enabled=False,
        motor_2_driver_enabled=False,
        motor_3_driver_enabled=False,
        motor_4_driver_enabled=False,
        motor_5_driver_enabled=False,
        motor_6_driver_enabled=False))
    motion = evaluator.evaluate(
        SafetyMode.SCAN_MOTION, unavailable,
        SafetyInputs(approved_obstacle_snapshot=True))
    hold = evaluator.evaluate(
        SafetyMode.HOLD_CURRENT, unavailable,
        SafetyInputs(joints_settled=True))
    assert not motion.permitted
    # publish_hold currently proves fresh joints and real-motion authority;
    # it does not independently inspect arm-status motor flags.
    assert hold.permitted


def test_capture_requires_settled_pose_and_healthy_camera(evaluator):
    unsettled = evaluator.evaluate(
        SafetyMode.SCAN_CAPTURE, snapshot(),
        SafetyInputs(joints_settled=False))
    camera_bad = snapshot()
    camera_bad.perception.camera.value.healthy = False
    unhealthy = evaluator.evaluate(
        SafetyMode.SCAN_CAPTURE, camera_bad,
        SafetyInputs(joints_settled=True))
    assert not unsettled.permitted
    assert not unhealthy.permitted


def test_decision_and_evidence_are_immutable(evaluator):
    decision = evaluator.evaluate(SafetyMode.RETURN_HOME, snapshot())
    with pytest.raises(FrozenInstanceError):
        decision.permitted = False
    with pytest.raises(FrozenInstanceError):
        decision.evidence[0].passed = False


def test_comparison_logger_records_agreement_and_structured_disagreement(
        evaluator):
    logger = SafetyComparisonLogger(maximum_records=2)
    accepted = evaluator.evaluate(SafetyMode.RETURN_HOME, snapshot())
    agreement = logger.record('runtime', (), accepted)
    rejected = evaluator.evaluate(
        SafetyMode.RETURN_HOME, snapshot(joint_age=3.0))
    disagreement = logger.record('runtime', (), rejected)
    assert agreement.agreement
    assert not disagreement.agreement
    assert disagreement.permission_agreement is False
    assert logger.summary() == {
        'comparisons': 2, 'agreements': 1, 'disagreements': 1}
    payload = disagreement.as_dict()
    assert payload['mode'] == 'RETURN_HOME'
    assert payload['telemetry_ages']['joints'] == pytest.approx(3.0)


def test_comparison_log_is_bounded_and_does_not_affect_decision(evaluator):
    logger = SafetyComparisonLogger(maximum_records=1)
    decision = evaluator.evaluate(SafetyMode.RETURN_HOME, snapshot())
    logger.record('first', (), decision)
    logger.record('second', (), decision)
    assert len(logger.records()) == 1
    assert logger.records()[0].context == 'second'
    assert decision.permitted


def test_executor_runtime_gate_records_shadow_without_changing_legacy_result(
        executor_runtime_harness, evaluator):
    harness = executor_runtime_harness
    harness.parameters['data_timeout_sec'] = 2.0
    harness.parameters['home_joint_feedback_timeout_sec'] = 1.0
    harness.state = 'MOVING'
    harness.plan_collision_model_qualified = True
    harness.returning_home = lambda: False
    harness.real_motion_enabled = lambda: True
    harness.telemetry_store = TelemetryStore(clock=lambda: NOW)
    harness.telemetry_store.update_joints(
        SimpleNamespace(position=[0.0] * 6), received_at=NOW - 0.1)
    harness.telemetry_store.update_arm_status(
        healthy_status(), received_at=NOW - 0.1)
    harness.telemetry_store.update_motion_limits(
        harness.latest_motion_limits, received_at=NOW - 0.1)
    harness.telemetry_store.update_camera(
        harness.latest_camera_timestamp_health, received_at=NOW - 0.1)
    harness.telemetry_store.update_tracking(
        harness.latest_tracking_health, received_at=NOW - 0.1)
    harness.telemetry_store.update_target_status(
        harness.latest_target_status, received_at=NOW - 0.1)
    harness.telemetry_store.update_obstacles(
        harness.latest_obstacles, received_at=NOW - 0.1)
    harness.telemetry_store.update_workflow(
        {'state': 'SCAN_READY'}, received_at=NOW - 0.1)
    harness.safety_evaluator = evaluator
    harness.safety_comparison_logger = SafetyComparisonLogger()

    reasons = ScanViewpointExecutorNode.runtime_reasons(
        harness,
        require_settled=False,
        require_workflow=False,
        allow_untracked=True,
        allow_stale_obstacles=True,
        enforce_tracking_speed_allowance=False,
        enforce_target_status=False,
        enforce_tracking_motion_state=False,
        shadow_mode=SafetyMode.SCAN_MOTION,
    )

    assert reasons == []
    comparison = harness.safety_comparison_logger.records()[-1]
    assert comparison.agreement
    assert comparison.legacy_permitted
    assert comparison.shadow_permitted
    assert harness.logged == []


def test_executor_shadow_disagreement_is_logged_but_legacy_stays_authoritative(
        executor_runtime_harness, evaluator):
    harness = executor_runtime_harness
    harness.parameters['data_timeout_sec'] = 2.0
    harness.parameters['home_joint_feedback_timeout_sec'] = 1.0
    harness.state = 'MOVING'
    harness.plan_collision_model_qualified = False
    harness.returning_home = lambda: False
    harness.real_motion_enabled = lambda: True
    harness.telemetry_store = TelemetryStore(clock=lambda: NOW)
    harness.telemetry_store.update_joints(
        SimpleNamespace(position=[0.0] * 6), received_at=NOW - 0.1)
    harness.telemetry_store.update_arm_status(
        healthy_status(), received_at=NOW - 0.1)
    harness.telemetry_store.update_motion_limits(
        harness.latest_motion_limits, received_at=NOW - 0.1)
    harness.telemetry_store.update_camera(
        harness.latest_camera_timestamp_health, received_at=NOW - 0.1)
    harness.telemetry_store.update_obstacles(
        harness.latest_obstacles, received_at=NOW - 0.1)
    harness.safety_evaluator = evaluator
    harness.safety_comparison_logger = SafetyComparisonLogger()

    legacy_reasons = ScanViewpointExecutorNode.runtime_reasons(
        harness,
        require_settled=False,
        require_workflow=False,
        allow_untracked=True,
        allow_stale_obstacles=True,
        enforce_tracking_speed_allowance=False,
        enforce_target_status=False,
        enforce_tracking_motion_state=False,
        shadow_mode=SafetyMode.SCAN_MOTION,
    )

    # collision-model qualification is enforced by approval/trajectory gates,
    # not by the legacy runtime_reasons function itself. Shadow mode detects
    # that boundary disagreement but cannot change its authoritative return.
    assert legacy_reasons == []
    comparison = harness.safety_comparison_logger.records()[-1]
    assert not comparison.agreement
    assert comparison.legacy_permitted
    assert not comparison.shadow_permitted
    assert harness.logged[-1].startswith('SAFETY_SHADOW_DISAGREEMENT ')
