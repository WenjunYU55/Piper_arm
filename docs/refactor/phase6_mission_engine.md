# Phase 6 mission-engine extraction

> Historical phase record. On 2026-08-17 the uncalled frozen mission bodies
> were removed and `MissionEngine` became the only executable mission/shutdown
> implementation. Public compatibility delegates remain and call it directly.

## Scope and authority

Phase 6 separates the admitted mission workflow from ROS action mechanics.
`piper_mobile_manipulation/mission_engine.py` owns the current startup,
preflight, enable/hold, staged startup home, acquisition, target lock,
occlusion probe, adaptive scan, and terminal home/hold/disable/cleanup order.
It imports no `rclpy`, ROS message, action, service, QoS, or TF type.

The public `/piper/run_target_scan` action and every topic, service, message,
parameter, timeout, threshold, speed, limit, TF meaning, perception algorithm,
Tesseract rule, and terminal result string remain unchanged.

## Application model

- `CancellationToken` is thread-safe application cancellation state. The ROS
  cancel callback forwards into the token; production mission guards no longer
  poll the ROS goal handle directly.
- `MissionContext` owns one `MissionSession`, its cancellation token, resolved
  domain target, process-generation ownership, and observed phase sequence.
- `MissionResult` is the immutable application outcome before the node maps it
  to the existing ROS result and action-handle terminal transition.
- `MissionEngine` receives one injected operations adapter and an injectable
  clock. Its phase dispatch table uses the existing `MissionPhase` values.

The straightforward `_MissionNodeOperations` adapter remains in
`target_scan_mission_node.py`. It translates engine operations to the existing
ROS clients, telemetry helpers, process supervisor, arm calls, planner,
perception workflow, capture evidence, and status/feedback publishers. Safety
decisions remain with their current owners. The later 2026-08-17 cleanup
removed the unused duplicate shadow evaluator and retained the executor as the
single gate/command authority through named `RuntimeGatePolicy` inputs.

## Handler structure

The engine dispatches the existing phases through small handlers:

1. `STARTING`: bind home, start the owned generation, and prove readiness.
2. `PREFLIGHT`: validate wrist direction, motion opt-in and speed qualification.
3. `ENABLE_AND_HOLD`: enable all axes and prove current-position hold.
4. `RETURNING_HOME`: execute `STARTUP_WRIST` then `ROUGH_HOME`.
5. `ROUGH_ACQUISITION`: run at most five one-look-at-a-time attempts, entering
   `TARGET_LOCK` for each correlated execution.
6. `OCCLUSION_PROBE`: preserve `SCAN_READY`; preserve `PLAN_READY` as
   `NEEDS_OPERATOR` because contact execution remains unqualified.
7. `VIEW_PLANNING`: preserve one-view planning, target-drift replans,
   `VIEW_REJECTED` replacement, 8-to-24 bounds, feature completion, and safe
   frontier exhaustion.
8. Terminal handling: preserve motor-loss no-command cleanup, direct home,
   storage wrist, settled hold, feedback-confirmed disable, authorization
   revocation, and owned-process cleanup.

## ROS boundary

`TargetScanMissionNode.execute_cb` retains goal normalization/admission,
idempotent cached results, queue semantics, ROS goal-handle transitions,
result conversion, durable result/mesh-job writes, and queue release. For an
admitted task it constructs one context, adapter and engine, then consumes the
returned application result.

Queued cancellation remains command-free. Active cancellation is forwarded by
`cancel_cb` into the task token. Cancellation during committed terminal home,
hold, disable or cleanup remains non-interrupting, as characterized before the
extraction.

## Legacy compatibility retained

The former Phase 5 `run_pipeline` and `safe_shutdown` bodies were initially
retained as uncalled frozen evidence. They were removed on 2026-08-17 after
repository reference search and characterization proved production used only
`MissionEngine`. The public Python method names remain compatibility delegates
to the engine; no runtime fallback exists.

## Equivalence and tests

`test_mission_engine.py` uses only pure Python fakes and covers the successful
frozen Phase 1 trace, every active major failure/cancellation phase,
non-interrupting terminal cancellation, deadline expiry, capture replacement,
target-drift reacquisition, repeated acquisition looks, and repeated missions
without engine-owned state leakage. Phase 1 mission characterization and
direct shutdown tests also execute against engine-backed compatibility paths.

## Validation result

Validation is software-only. No arm, camera, GPU worker, driver, or robot stack
was started.

- Pre-change Phase 1 mission and mission-core baseline: 80 passed.
- New pure `test_mission_engine.py`: 34 passed.
- Engine, Phase 1 mission and mission-core focused selection: 117 passed.
- Complete `piper_mobile_manipulation/test`: 541 passed.
- Normal five-package Foxy `colcon build --symlink-install`: passed.
- Registered colcon aggregate: 903 assertions, 0 errors, 116 existing lint
  failures and 1 existing hardware-dependent skip. All functional registered
  targets passed. The remaining failures are 93 Flake8 findings, 21 PEP257
  findings and their two failed parent targets; Phase 5 recorded 118, so Phase
  6 introduced no lint debt.
- Root GUI/reconstruction/calibration selection: 58 passed, 1 skipped.
- Heavy worker, SAM2 worker and target-selection suites: 5/5, 6/6 and 19/19.
- New production/test files pass Flake8, PEP257 and byte compilation. All AI
  YAML parses and `git diff --check` pass.
