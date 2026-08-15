# Phase 2 typed failure model

## Scope

Phase 2 separates machine decisions from operator-facing explanations without
changing any ROS action, message, service, state transition, threshold, retry
budget, motion, perception, Tesseract, home, hold, disable, or cleanup policy.

The central implementation is
`piper_mobile_manipulation/failure_model.py`:

- `FailureCode` is the stable tracked-robot result classification.
- `FailureTag` represents finer internal decisions such as retrying an
  unchanged plan, waiting for visual reacquisition, replacing a rejected view,
  accepting a proved terminal-home status, or blocking automatic home.
- immutable `Failure` carries the code, detail, tags, retryability,
  operator requirement, outcome, and any machine-readable blocker.
- `legacy_failure_adapter` is the single compatibility boundary allowed to
  translate string-only legacy service, status, and exception text.
- `as_failure` leaves typed failures unchanged and invokes the adapter only for
  an untyped legacy value.

`MissionFailure` remains the runtime exception used by the coordinator and
retains its old public attributes. It now owns a typed `failure` and exposes
the same string through `str(exception)`. Existing action results still emit
ordinary string `failure_code` and human-readable reason fields.

## Decisions migrated

The mission coordinator no longer searches explanation strings to decide:

- public failure code;
- plan-approval retry and visual-reacquisition waits;
- command-free plan-request reacquisition;
- target-drift replanning;
- adaptive safe-view exhaustion and empty-frontier handling;
- whether a planning rejection permits a fresh home qualification;
- already-pending/already-active service handling;
- hold acknowledgement and configured-home completion.

The executor no longer searches explanation strings to decide:

- same-view RGB-D readiness retry versus fresh visual rejection;
- terminal-home hold behavior;
- runtime freshness hold versus hard abort;
- transient obstacle-transform gaps;
- automatic-home command/feedback blockers;
- duplicate self-collision-clearance findings on legacy retrace.

The GUI and its pure automation helper now use typed adapter output for
pending workflow requests, workflow-already-active handling, cancel/home
completion and retry, safe-disable hold acknowledgement, and Step 4/5
operator-recovery blockers.

## Deliberately retained legacy parsing

All behavior-affecting legacy phrase matching for the migrated mission path is
confined to `legacy_failure_adapter`. It must temporarily remain because Foxy
interfaces such as `std_srvs/Trigger.message`,
`ScanExecutionStatus.reason`, plan rejection messages, obstacle
`validity_reason`, and several workflow JSON payloads expose only strings and
cannot carry new fields without a prohibited ROS interface change.

The installed legacy `safe_servo_node.py` still recognizes `error`/`fault` in
its old `std_msgs/String /arm_status` input. It is not part of the autonomous
mission command path and its input is incompatible with the current typed
`PiperStatusMsg /arm_status` owner. Migrating or removing that obsolete
commissioning surface is intentionally deferred rather than expanding this
phase into a node/interface rewrite.

Lowercasing of declared enum/state values, labels, hashes, filenames, boolean
parameters, and commands remains. Those are structured value normalization,
not classification of human-readable failure explanations.

## Compatibility guarantees

- Every pre-Phase-2 phrase maps to the same public failure code.
- The ordering that makes camera startup timeout
  `SENSOR_UNAVAILABLE`, rather than `DEADLINE_EXPIRED`, is preserved.
- `visual replacement budget exhausted` deliberately remains
  `MISSION_FAILED`, as characterized in Phase 1.
- Existing emitted explanations are unchanged.
- Existing helper functions remain import-compatible and accept both strings
  and typed failures.
- No public ROS definition was edited.

## Required regression set

Run `test_failure_model.py`, all three Phase 1 characterization targets,
`test_mission_core.py`, `test_scan_motion.py`, the GUI automation tests, the
whole ROS package suite, root/reconstruction/calibration tests, AI worker tests,
and the normal five-package build. No hardware or GPU is required for the new
typed-failure tests.

## Validation result (2026-08-14)

- Pre-change Phase 1 characterization: 79 passed.
- Post-change Phase 1 characterization plus typed failures: 107 passed
  (79 existing plus 28 Phase 2 tests).
- Complete mobile-manipulation pytest directory: 445 passed.
- Root, reconstruction, and calibration suite: 58 passed, 1 existing skip.
- Heavy worker, SAM2 worker, and target selection: 5/5, 6/6, and 19/19.
- Normal Foxy build: all five packages passed in 12.5 seconds.
- Registered colcon aggregate: 796 tests, 0 errors, 118 failures, 1 skip.
  Every functional target passed. The failures are the unchanged baseline
  style debt: flake8 reports 93 findings and pep257 reports 23; their two
  failed parent targets account for the aggregate total. The new failure model
  and its tests add no style finding.

All validation was software-only. No arm, camera, GPU, or physical process was
enabled or commanded.
