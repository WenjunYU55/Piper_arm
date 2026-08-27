"""
Thread-safe immutable snapshots of runtime ROS observations.

The store owns observations only.  Mission phases, active plans, controller
commands, retry counters, and service futures remain owned by their existing
state machines.
"""

from copy import deepcopy
from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Generic, Optional, TypeVar


T = TypeVar('T')


@dataclass(frozen=True)
class TelemetryObservation(Generic[T]):
    """One value and the unchanged receipt/source metadata used to age it."""

    value: T
    received_at: float
    source_stamp_ns: Optional[int] = None
    frame_id: str = ''
    revision: int = 0

    def age_at(self, now: float) -> float:
        """Return monotonic receipt age without embedding a freshness limit."""
        return float(now) - float(self.received_at)

    def is_stale_at(self, now: float, maximum_age: float) -> bool:
        """Match the existing strict-greater-than freshness semantics."""
        return self.age_at(now) > float(maximum_age)


@dataclass(frozen=True)
class ArmTelemetry:
    """Feedback observations produced by the arm/controller boundary."""

    joints: Optional[TelemetryObservation[Any]] = None
    status: Optional[TelemetryObservation[Any]] = None
    motion_limits: Optional[TelemetryObservation[Any]] = None


@dataclass(frozen=True)
class PerceptionTelemetry:
    """Camera, target, tracking, and obstacle observations."""

    camera: Optional[TelemetryObservation[Any]] = None
    target: Optional[TelemetryObservation[Any]] = None
    tracking: Optional[TelemetryObservation[Any]] = None
    target_status: Optional[TelemetryObservation[str]] = None
    obstacles: Optional[TelemetryObservation[Any]] = None


@dataclass(frozen=True)
class MissionTelemetry:
    """ROS observations consumed by mission/executor orchestration."""

    readiness: Optional[TelemetryObservation[Any]] = None
    plan: Optional[TelemetryObservation[Any]] = None
    execution: Optional[TelemetryObservation[Any]] = None
    capture: Optional[TelemetryObservation[Any]] = None
    scan_history: Optional[TelemetryObservation[Any]] = None
    reachable_scan: Optional[TelemetryObservation[Any]] = None
    workflow: Optional[TelemetryObservation[Any]] = None


@dataclass(frozen=True)
class TelemetrySnapshot:
    """A coherent defensive copy of all observations at one instant."""

    captured_at: float
    revision: int
    arm: ArmTelemetry
    perception: PerceptionTelemetry
    mission: MissionTelemetry


class TelemetryStore:
    """Serialize callback updates and provide immutable decision snapshots."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.RLock()
        self._revision = 0
        self._joints = None
        self._arm_status = None
        self._motion_limits = None
        self._camera = None
        self._target = None
        self._tracking = None
        self._target_status = None
        self._obstacles = None
        self._readiness = None
        self._plan = None
        self._execution = None
        self._capture = None
        self._scan_history = None
        self._reachable_scan = None
        self._workflow = None

    def _observation(
            self, value: Any, received_at: Optional[float],
            source_stamp_ns: Optional[int], frame_id: str
    ) -> TelemetryObservation[Any]:
        self._revision += 1
        return TelemetryObservation(
            value=value,
            received_at=(
                float(self._clock())
                if received_at is None else float(received_at)),
            source_stamp_ns=(
                None if source_stamp_ns is None else int(source_stamp_ns)),
            frame_id=str(frame_id),
            revision=self._revision,
        )

    def _update(
            self, field: str, value: Any,
            received_at: Optional[float] = None,
            source_stamp_ns: Optional[int] = None,
            frame_id: str = '') -> None:
        owned_value = deepcopy(value)
        with self._lock:
            setattr(self, field, self._observation(
                owned_value, received_at, source_stamp_ns, frame_id))

    def update_joints(
            self, value: Any, received_at: Optional[float] = None,
            source_stamp_ns: Optional[int] = None,
            frame_id: str = '') -> None:
        self._update(
            '_joints', value, received_at, source_stamp_ns, frame_id)

    def update_arm_status(
            self, value: Any, received_at: Optional[float] = None,
            source_stamp_ns: Optional[int] = None,
            frame_id: str = '') -> None:
        self._update(
            '_arm_status', value, received_at, source_stamp_ns, frame_id)

    def update_motion_limits(
            self, value: Any, received_at: Optional[float] = None,
            source_stamp_ns: Optional[int] = None,
            frame_id: str = '') -> None:
        self._update(
            '_motion_limits', value, received_at, source_stamp_ns, frame_id)

    def update_camera(
            self, value: Any, received_at: Optional[float] = None,
            source_stamp_ns: Optional[int] = None,
            frame_id: str = '') -> None:
        self._update(
            '_camera', value, received_at, source_stamp_ns, frame_id)

    def update_target(
            self, value: Any, received_at: Optional[float] = None,
            source_stamp_ns: Optional[int] = None,
            frame_id: str = '') -> None:
        self._update(
            '_target', value, received_at, source_stamp_ns, frame_id)

    def update_tracking(
            self, value: Any, received_at: Optional[float] = None,
            source_stamp_ns: Optional[int] = None,
            frame_id: str = '') -> None:
        self._update(
            '_tracking', value, received_at, source_stamp_ns, frame_id)

    def update_target_status(
            self, value: str, received_at: Optional[float] = None,
            source_stamp_ns: Optional[int] = None,
            frame_id: str = '') -> None:
        self._update(
            '_target_status', value, received_at, source_stamp_ns, frame_id)

    def update_obstacles(
            self, value: Any, received_at: Optional[float] = None,
            source_stamp_ns: Optional[int] = None,
            frame_id: str = '') -> None:
        self._update(
            '_obstacles', value, received_at, source_stamp_ns, frame_id)

    def update_readiness(
            self, value: Any, received_at: Optional[float] = None,
            source_stamp_ns: Optional[int] = None,
            frame_id: str = '') -> None:
        self._update(
            '_readiness', value, received_at, source_stamp_ns, frame_id)

    def update_plan(
            self, value: Any, received_at: Optional[float] = None,
            source_stamp_ns: Optional[int] = None,
            frame_id: str = '') -> None:
        self._update(
            '_plan', value, received_at, source_stamp_ns, frame_id)

    def update_execution(
            self, value: Any, received_at: Optional[float] = None,
            source_stamp_ns: Optional[int] = None,
            frame_id: str = '') -> None:
        self._update(
            '_execution', value, received_at, source_stamp_ns, frame_id)

    def update_capture(
            self, value: Any, received_at: Optional[float] = None,
            source_stamp_ns: Optional[int] = None,
            frame_id: str = '') -> None:
        self._update(
            '_capture', value, received_at, source_stamp_ns, frame_id)

    def update_scan_history(
            self, value: Any, received_at: Optional[float] = None,
            source_stamp_ns: Optional[int] = None,
            frame_id: str = '') -> None:
        self._update(
            '_scan_history', value, received_at, source_stamp_ns, frame_id)

    def update_reachable_scan(
            self, value: Any, received_at: Optional[float] = None,
            source_stamp_ns: Optional[int] = None,
            frame_id: str = '') -> None:
        self._update(
            '_reachable_scan', value, received_at, source_stamp_ns, frame_id)

    def update_workflow(
            self, value: Any, received_at: Optional[float] = None,
            source_stamp_ns: Optional[int] = None,
            frame_id: str = '') -> None:
        self._update(
            '_workflow', value, received_at, source_stamp_ns, frame_id)

    def clear_plan(self) -> None:
        """Clear only plan evidence while retaining every other channel."""
        with self._lock:
            self._revision += 1
            self._plan = None

    def clear_mission_runtime(self) -> None:
        """Mirror the coordinator's existing between-mission cache reset."""
        with self._lock:
            self._revision += 1
            self._readiness = None
            self._plan = None
            self._execution = None
            self._capture = None
            self._scan_history = None

    def clear_arm_feedback(self) -> None:
        """Mirror a new driver generation without clearing other telemetry."""
        with self._lock:
            self._revision += 1
            self._joints = None
            self._arm_status = None

    def clear_camera(self) -> None:
        """Mirror a new camera generation without clearing other telemetry."""
        with self._lock:
            self._revision += 1
            self._camera = None

    @staticmethod
    def _copy(value):
        return None if value is None else deepcopy(value)

    def snapshot(self) -> TelemetrySnapshot:
        """Capture all channel references under one lock, then copy them."""
        with self._lock:
            captured_at = float(self._clock())
            revision = self._revision
            observations = (
                self._joints, self._arm_status, self._motion_limits,
                self._camera, self._target, self._tracking,
                self._target_status, self._obstacles,
                self._readiness, self._plan, self._execution,
                self._capture, self._scan_history,
                self._reachable_scan, self._workflow,
            )
        copied = tuple(self._copy(item) for item in observations)
        return TelemetrySnapshot(
            captured_at=captured_at,
            revision=revision,
            arm=ArmTelemetry(
                joints=copied[0], status=copied[1],
                motion_limits=copied[2]),
            perception=PerceptionTelemetry(
                camera=copied[3], target=copied[4], tracking=copied[5],
                target_status=copied[6], obstacles=copied[7]),
            mission=MissionTelemetry(
                readiness=copied[8], plan=copied[9], execution=copied[10],
                capture=copied[11], scan_history=copied[12],
                reachable_scan=copied[13], workflow=copied[14]),
        )
