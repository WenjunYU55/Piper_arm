# Phase 7 viewpoint-executor component extraction

## Scope and authority

Phase 7 extracts four ROS-free application components from
`scan_viewpoint_executor_node.py`. The executor node remains the sole optional
`/joint_ctrl_single` publisher and retains every existing ROS subscription,
publisher, service, client, QoS profile, timer, message conversion, telemetry
callback, geometry/path check, and explicit execution-state transition.

No public ROS interface, parameter, threshold, speed, motion limit, trajectory
point, trajectory timestamp, settle rule, capture requirement, perception
algorithm, Tesseract rule, return-home behavior, hold behavior, or safety
authority changed.

The pre-extraction ownership and coupling map is recorded in
`phase7_executor_responsibility_map.md`.

## Extracted components

### `PlanAuthorizer`

`plan_authorizer.py` receives immutable exact plan/mission identity, expiry,
target availability/drift, capture-dependency readiness, and already-computed
path evidence. It returns a typed `PlanAuthorizationDecision` and preserves the
existing human-readable service detail at the ROS boundary. Tesseract proposal
failure, stale plan, wrong mission authorization, stale target, target drift,
dependency loss, and fresh-path rejection are distinct machine statuses.

Geometry, collision qualification, telemetry safety, and path construction are
not moved into this component. Their established owners remain authoritative.

### `TrajectoryRunner`

`trajectory_runner.py` freezes one already-authorized six-joint segment and
returns typed monitoring actions for the existing schedule and feedback
evidence: wait, publish exactly one due point, advance, complete, cancel,
invalid feedback, timeout, no progress, following-error failure, or schedule
overrun.

The executor still publishes every command. It continues to use the exact
Tesseract points/times, refuses bursts and shortcuts, feedback-gates the final
endpoint, and applies the unchanged values declared in
`scan_execution_params.yaml`/node parameters.

### `CaptureCoordinator`

`capture_coordinator.py` owns the settled status-propagation ordering and the
existing capture-result decision. The ROS node still owns service futures,
calls, timeout state, status publication, and accepted/rejected scan history.
String-only service detail is translated once by the Phase 2 compatibility
adapter; the coordinator consumes a typed `Failure` and preserves the ten
attempt same-view readiness bound, replacement-view result, and fatal abort.

### `RecoveryPolicy`

`executor_recovery.py` explicitly represents `RETRY`, `REACQUIRE`, `REPLAN`,
and `ABORT` across runtime, acquisition, planning, trajectory, and capture
contexts. It accepts only typed `Failure` values and bases decisions on
`FailureCode`/`FailureTag`; changing `Failure.detail` cannot change recovery.

The runtime freshness hold and capture retry/replacement paths now delegate to
this typed policy. Existing ROS-only untyped producer responses continue to
enter through `legacy_failure_adapter` until their public interfaces can carry
typed fields.

## Deliberately retained node logic

- Tesseract ROS message parsing and schema-v5 normalization;
- controller-limit stability and exact hash binding;
- dense robot/tool/target visibility path validation;
- acquisition request and image-stamp correlation;
- `TelemetryStore` adapters and Phase 5 safety-shadow comparisons;
- ROS service futures and status/history publication;
- return-home evidence, settle proof, hold, and terminal state mapping.

Moving these together would combine separate safety authorities and exceed the
behavior-preserving scope. They are recorded as a later incremental cut in
`docs/ai/60-debt.yaml`.

## Tests

`test_phase7_viewpoint_components.py` covers valid/stale/wrong-mission plans,
stale target, target drift, planner/path failure, successful trajectory
progression, every monitored trajectory failure, cancellation, capture
success/retry/replacement/failure, status propagation, the recovery decision
matrix, and wording-independent typed recovery.

The existing Phase 1 executor and scan-motion tests continue to exercise the
node integration and public result wording.

## Validation result

Validation was software-only. No arm, camera, GPU worker, driver, perception
pipeline, planner worker, or mission stack was started.

- Pre-change focused executor baseline: 195 passed.
- Phase 7 plus focused executor regression: 219 passed.
- Complete `piper_mobile_manipulation/test`: 565 passed.
- New production/test files: Flake8 and PEP257 pass.
- Normal five-package Foxy `colcon build --symlink-install`: passed.
- Registered colcon aggregate: 933 assertions, 0 errors, 116 existing lint
  failures and 1 existing hardware-dependent skip. Every functional target,
  including the 24 new Phase 7 component checks and the four public-interface
  characterization checks, passed. The lint debt remains 93 Flake8 findings,
  21 PEP257 findings, and their two failed parent targets, unchanged from
  Phase 6.
- Root GUI/reconstruction/calibration selection: 58 passed, 1 skipped.
- Heavy worker, SAM2 worker and target-selection suites: 5/5, 6/6 and 19/19
  passed using their established Python environments.
- All AI YAML parses, byte compilation, focused undefined-name lint and
  `git diff --check` pass.
