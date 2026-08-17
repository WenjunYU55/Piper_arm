"""
Named, immutable runtime-gate policies for the viewpoint executor.

The executor remains responsible for evaluating live ROS evidence. This module
only maps each operating phase to the evidence categories it is allowed to use,
so callers cannot construct contradictory combinations of boolean switches.
"""

from dataclasses import dataclass
from enum import Enum


class SafetyMode(str, Enum):
    """Named executor contexts with distinct existing evidence policy."""

    PLAN_VALIDATION = 'PLAN_VALIDATION'
    ACQUISITION_APPROVAL = 'ACQUISITION_APPROVAL'
    ACQUISITION_MOTION = 'ACQUISITION_MOTION'
    SCAN_APPROVAL = 'SCAN_APPROVAL'
    SCAN_MOTION = 'SCAN_MOTION'
    SCAN_CAPTURE = 'SCAN_CAPTURE'
    RETURN_HOME = 'RETURN_HOME'
    HOLD_CURRENT = 'HOLD_CURRENT'


class ObstacleAuthority(str, Enum):
    """Describe which obstacle evidence owns one execution segment."""

    LIVE = 'LIVE'
    APPROVED_SNAPSHOT = 'APPROVED_SNAPSHOT'
    STATIC_BOOTSTRAP = 'STATIC_BOOTSTRAP'


@dataclass(frozen=True)
class RuntimeGatePolicy:
    """One named runtime policy instead of interacting boolean switches."""

    mode: SafetyMode
    require_settled: bool
    require_workflow: bool
    require_tracking: bool
    require_camera: bool
    require_motion_limits: bool
    obstacle_authority: ObstacleAuthority
    settle_at_current_hold: bool = False


def runtime_gate_policy(
        mode: SafetyMode, *, require_settled: bool = False,
        obstacle_authority: ObstacleAuthority = ObstacleAuthority.LIVE,
        settle_at_current_hold: bool = False) -> RuntimeGatePolicy:
    """Return the complete gate policy for one explicit executor phase."""
    selected = mode if isinstance(mode, SafetyMode) else SafetyMode(mode)
    if selected == SafetyMode.RETURN_HOME:
        return RuntimeGatePolicy(
            selected, bool(require_settled), False, False, False, False,
            ObstacleAuthority.STATIC_BOOTSTRAP,
            bool(settle_at_current_hold))
    if selected == SafetyMode.HOLD_CURRENT:
        return RuntimeGatePolicy(
            selected, bool(require_settled), False, False, False, False,
            ObstacleAuthority.STATIC_BOOTSTRAP,
            bool(settle_at_current_hold))
    if selected == SafetyMode.ACQUISITION_APPROVAL:
        return RuntimeGatePolicy(
            selected, bool(require_settled), False, False, True, True,
            ObstacleAuthority.STATIC_BOOTSTRAP,
            bool(settle_at_current_hold))
    if selected == SafetyMode.SCAN_APPROVAL:
        return RuntimeGatePolicy(
            selected, bool(require_settled), True, True, True, True,
            obstacle_authority, bool(settle_at_current_hold))
    if selected in (
            SafetyMode.ACQUISITION_MOTION,
            SafetyMode.SCAN_MOTION):
        # Target/tracking are approval and post-settle observation evidence.
        # They are not allowed to revoke an already-issued eye-in-hand MoveJ
        # merely because the camera is moving.
        return RuntimeGatePolicy(
            selected, bool(require_settled), False, False, False, True,
            obstacle_authority, bool(settle_at_current_hold))
    if selected == SafetyMode.SCAN_CAPTURE:
        return RuntimeGatePolicy(
            selected, bool(require_settled), False, False, True, True,
            obstacle_authority, bool(settle_at_current_hold))
    return RuntimeGatePolicy(
        selected, bool(require_settled), False, False, False, True,
        obstacle_authority, bool(settle_at_current_hold))
