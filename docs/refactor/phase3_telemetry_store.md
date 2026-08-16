# Phase 3 telemetry snapshots

## Scope

Phase 3 introduces one pure-Python observation boundary without changing the
mission engine, executor state machine, ROS graph, QoS, public messages,
timeouts, numerical thresholds, motion, Tesseract, perception, or shutdown
policy.

`piper_mobile_manipulation/telemetry_store.py` owns callback observations only.
It deliberately does not own mission phases, active plans, current paths,
command targets, service futures, retry counters, process ownership, capture
state, or home/disable proofs.

## Domain model

- `TelemetryObservation` binds one defensively copied value to its monotonic
  receipt time, optional ROS source stamp, frame ID, and store revision.
- `ArmTelemetry` contains joints, arm status, and controller motion limits.
- `PerceptionTelemetry` contains camera health, tracked target, tracking
  health, target status, and obstacle geometry.
- `MissionTelemetry` contains Tesseract readiness, correlated plan/execution
  status, capture/session history, reachable scan payload, and workflow status.
- `TelemetrySnapshot` is a frozen, revisioned composition captured under one
  re-entrant lock at one clock instant.
- `TelemetryStore` serializes updates, owns its copies, and returns defensive
  copies. Mutating an input after update or a returned nested ROS/dictionary
  value cannot mutate the stored observation or a later snapshot.

The store does not define freshness thresholds. `TelemetryObservation.age_at`
and `is_stale_at` apply the existing caller-provided bounds with the existing
strict `age > maximum` stale rule. Clock injection exists only for
deterministic tests.

## Production migration

The mission and executor create independent stores because each node receives
and decides from its own ROS subscriptions. Every accepted callback now writes
its established compatibility field and the store from the same receipt-time
sample. Joint, target, tracking, camera, obstacle, arm-status, motion-limit,
plan, execution, and readiness headers retain source stamp/frame metadata when
present.

The following decisions now use one coherent snapshot:

- mission readiness, joint freshness, stable joint stream, vision health,
  capture feedback, plan/execution correlation, configured-home feedback,
  motor-control guard, motor-loss six-disabled proof, terminal hold, and
  shutdown feedback;
- executor plan/controller-limit binding, tracking speed at plan validation,
  approval target drift, runtime freshness and safety gates, motion-recovery
  obstacle authority, target-lock acquisition, correlated obstacle scenes,
  joint/home settling, camera/capture settling, workflow acceptance, obstacle
  boxes, hold publication, current joints, arm status, and command speed.

The ROS subscriptions and their QoS arguments were not edited. The store is an
in-process implementation detail and adds no ROS topic, service, action,
parameter, process, environment variable, file format, or TF frame.

## Compatibility fields retained for later removal

Phase 3 intentionally keeps mirrors so older pure-Python characterization
harnesses and low-risk diagnostic code remain compatible while the migration
is reviewed.

`target_scan_mission_node.py` retains:

- `latest_readiness[_at]`, `latest_plan[_at]`,
  `latest_execution[_at]`, `latest_capture[_at]`, and
  `latest_scan_history[_at]`;
- `latest_joints[_at]`, `latest_joint_source_ns`,
  `latest_arm_status[_at]`, and `latest_camera_health[_at]`;
- `latest_scan_target_center`, which is derived coverage/session state rather
  than an independent callback channel and should move only with later
  session-state work.

`scan_viewpoint_executor_node.py` retains:

- `latest_scan`, `latest_joint_state`, `latest_arm_status`,
  `latest_motion_limits`, `latest_tracking_health`, `latest_tracked_target`,
  `latest_camera_timestamp_health`, `latest_target_status`,
  `latest_obstacles`, and `latest_workflow`;
- `updated`, the legacy string-key receipt-time mirror used by existing test
  seams and remaining diagnostics.

These mirrors are callback-written compatibility surfaces, not authoritative
inputs at the migrated safety decisions. They should be removed incrementally
only after their remaining diagnostic/session consumers and Phase 1 harnesses
use snapshots directly. Executor-owned `plan_*`, path, command, capture,
recovery, and retry fields are not telemetry debt and must not be folded into
the store.

## Frozen behavior and validation

No configured number changed. In particular, the existing 0.25/0.5/1.0-second
mission freshness checks, executor `data_timeout_sec`,
`motion_limits_timeout_sec`, `max_tracking_measurement_age_sec`, joint/home
feedback timeouts, settle windows, target-drift bound, speed limits, capture
budgets, and home tolerances remain owned by their original callers/config.

Tests cover empty, partial, and complete stores; age/staleness with a fake
clock; concurrent writers; revision/value/timestamp consistency; frozen
dataclasses and defensive copies; selective clearing; callback metadata; and
old/new freshness, readiness, and joint-decision equivalence.

The first supervised reintegration attempt on 2026-08-16 exposed one missing
characterization case: `PiperStatusMsg` has no ROS `Header`, while the migrated
mission callback had directly dereferenced `msg.header`. The coordinator
failed before enable or motion. The callback now preserves the intended
receipt-time-only status observation with empty frame/source metadata, and a
headerless-message regression prevents recurrence. All six motors were proved
disabled before the orphaned, mission-started driver group was terminated.

Validation on 2026-08-14 is software-only. No arm, camera, GPU worker, or
physical ROS process was started.

- Phase 1/Phase 2 characterization baseline before Phase 3: 107 passed.
- Phase 3 telemetry-store tests: 14 passed after the headerless arm-status
  regression was added.
- Phase 1 characterization, telemetry, and scan-motion regression selection:
  210 passed.
- Complete `piper_mobile_manipulation/test` suite: 458 passed.
- Five-package ROS 2 Foxy `colcon build --symlink-install`: passed.
- Registered colcon aggregate: 812 tests, 0 errors, 118 pre-existing lint
  assertion failures, and 1 skip. The failure count is unchanged from the
  Phase 2 baseline; functional registered tests pass.
- Wider repository functional selection: 195 passed and 2 skipped. The one
  top-level failure is the already documented repository-wide PEP 257 harness
  scanning generated/vendored trees, not a Phase 3 functional regression.
- New files pass `ament_flake8` and `ament_pep257`; changed Python files pass
  byte compilation.
- Command-free rootless core and compact qualification paths: passed.
