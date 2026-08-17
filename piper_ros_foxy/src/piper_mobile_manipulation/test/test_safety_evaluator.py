"""No-hardware tests for named executor runtime-gate policies."""

from dataclasses import FrozenInstanceError

import pytest

from piper_mobile_manipulation.safety_evaluator import (
    ObstacleAuthority,
    RuntimeGatePolicy,
    SafetyMode,
    runtime_gate_policy,
)


@pytest.mark.parametrize(
    'mode',
    tuple(SafetyMode),
)
def test_every_safety_mode_has_one_complete_named_policy(mode):
    policy = runtime_gate_policy(mode)

    assert isinstance(policy, RuntimeGatePolicy)
    assert policy.mode is mode
    assert isinstance(policy.obstacle_authority, ObstacleAuthority)


def test_return_home_uses_only_live_control_evidence():
    policy = runtime_gate_policy(SafetyMode.RETURN_HOME)

    assert not policy.require_workflow
    assert not policy.require_tracking
    assert not policy.require_camera
    assert not policy.require_motion_limits
    assert policy.obstacle_authority is ObstacleAuthority.STATIC_BOOTSTRAP


def test_hold_current_has_the_same_independent_evidence_boundary():
    policy = runtime_gate_policy(
        SafetyMode.HOLD_CURRENT,
        require_settled=True,
        settle_at_current_hold=True,
    )

    assert policy.require_settled
    assert policy.settle_at_current_hold
    assert not policy.require_workflow
    assert not policy.require_tracking
    assert not policy.require_camera
    assert not policy.require_motion_limits


def test_scan_approval_requires_live_visual_and_planning_evidence():
    policy = runtime_gate_policy(
        SafetyMode.SCAN_APPROVAL,
        obstacle_authority=ObstacleAuthority.LIVE,
    )

    assert policy.require_workflow
    assert policy.require_tracking
    assert policy.require_camera
    assert policy.require_motion_limits
    assert policy.obstacle_authority is ObstacleAuthority.LIVE


@pytest.mark.parametrize(
    'mode',
    (SafetyMode.ACQUISITION_MOTION, SafetyMode.SCAN_MOTION),
)
def test_issued_eye_in_hand_motion_is_not_revoked_by_visual_evidence(mode):
    policy = runtime_gate_policy(
        mode, obstacle_authority=ObstacleAuthority.APPROVED_SNAPSHOT)

    assert not policy.require_workflow
    assert not policy.require_tracking
    assert not policy.require_camera
    assert policy.require_motion_limits
    assert policy.obstacle_authority is ObstacleAuthority.APPROVED_SNAPSHOT


def test_capture_requires_camera_but_not_live_tracking_during_request():
    policy = runtime_gate_policy(SafetyMode.SCAN_CAPTURE)

    assert policy.require_camera
    assert not policy.require_tracking
    assert not policy.require_workflow
    assert policy.require_motion_limits


def test_acquisition_approval_requires_camera_and_limits_only():
    policy = runtime_gate_policy(SafetyMode.ACQUISITION_APPROVAL)

    assert policy.require_camera
    assert policy.require_motion_limits
    assert not policy.require_tracking
    assert not policy.require_workflow
    assert policy.obstacle_authority is ObstacleAuthority.STATIC_BOOTSTRAP


def test_policy_options_are_preserved_without_hidden_boolean_defaults():
    policy = runtime_gate_policy(
        SafetyMode.SCAN_CAPTURE,
        require_settled=True,
        obstacle_authority=ObstacleAuthority.APPROVED_SNAPSHOT,
        settle_at_current_hold=True,
    )

    assert policy.require_settled
    assert policy.settle_at_current_hold
    assert policy.obstacle_authority is ObstacleAuthority.APPROVED_SNAPSHOT


def test_runtime_gate_policy_is_immutable():
    policy = runtime_gate_policy(SafetyMode.RETURN_HOME)

    with pytest.raises(FrozenInstanceError):
        policy.require_camera = True


def test_string_mode_is_accepted_but_unknown_mode_is_rejected():
    assert runtime_gate_policy('RETURN_HOME').mode is SafetyMode.RETURN_HOME
    with pytest.raises(ValueError):
        runtime_gate_policy('NOT_A_MODE')


def test_obsolete_shadow_evaluator_is_not_a_second_authority():
    import piper_mobile_manipulation.safety_evaluator as module

    assert not hasattr(module, 'SafetyEvaluator')
    assert not hasattr(module, 'SafetyComparisonLogger')
