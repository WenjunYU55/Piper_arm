# Phase 9 typed configuration boundary

> Historical phase record. On 2026-08-17 the duplicate `SafetyEvaluator` and
> its `SafetyProfile` were removed. Executor thresholds remain in the same typed
> configuration groups; `RuntimeGatePolicy` contains only named evidence
> categories and no numerical configuration.

Phase 9 centralizes the autonomous coordinator and viewpoint executor static
configuration without changing any ROS parameter name, default, launch/YAML
override, unit, motion limit, safety threshold, or public interface.

## Runtime boundary

`piper_mobile_manipulation/configuration.py` is the source of the exact ROS
defaults for the two production boundaries covered by this phase:

- 16 `target_scan_mission` parameters;
- 82 `scan_viewpoint_executor` parameters.

Each node now declares all of its parameters, reads every value exactly once,
validates the resulting set, and retains immutable responsibility-specific
dataclasses. Production execution no longer calls `Node.get_parameter()` from
either node. `configured_value()` reads only the frozen startup mapping in
production. Its direct ROS-parameter fallback exists solely for Phase 1-7
unbound-method characterization doubles that do not construct a ROS node.

The typed groups are:

- coordinator: `MissionConfig`, `ProcessConfig`, `MissionMotionConfig`,
  `MissionCaptureConfig`, and `MissionWorkflowConfig`;
- executor: `ExecutorInterfaceConfig`, `MotionConfig`, `TrackingConfig`,
  `CaptureConfig`, `SafetyConfig`, `PlanningConfig`, and `ExecutorConfig`.

`MissionEngine` receives motion, capture, and workflow configuration
explicitly. At the Phase 9 boundary `SafetyEvaluator` received an immutable
`SafetyProfile`; that duplicate shadow path was removed on 2026-08-17. The
authoritative executor still receives the same typed groups and unchanged
values.

## State separation

The configuration module contains only static configuration: ROS names, file
paths, thresholds, timeouts, limits, retry counts, speed percentages, feature
flags, and authorization phrases.

The following remain outside configuration:

- mission input: task ID/type, target label/profile/confidence, rough pose,
  covariance, deadline, and mission hash;
- runtime telemetry: joints, arm status, camera health, target/tracking,
  obstacles, workflow, motion limits, and receipt/source timestamps;
- derived state: plans, hashes, paths, scores, coverage, retries, decisions,
  phases, capture history, process ownership, and shutdown proofs.

## Other repository configuration sources inspected

The package still has 15 YAML files under
`piper_mobile_manipulation/config/`. Their launch-time values remain ROS
overrides and are intentionally not copied into Python defaults. In
particular, `scan_execution_params.yaml` continues to override the executor's
default configured-home joint vector.

Twenty-nine Python files elsewhere in `piper_mobile_manipulation` still contain
node-local parameter access. They are perception, tracking, capture,
occlusion, workflow, visualization, and legacy servo boundaries outside this
phase. Moving them together would mix parameters with different semantics and
was deliberately deferred.

Environment and shell inputs were retained unchanged:

- operator/static startup: `PIPER_WORKSPACE`, CAN settings, auto-enable,
  gripper, joint command topic, ROS domain, enable timeout, joint bounds,
  FastDDS profile, and Tesseract image/runtime/spool/model paths;
- per-mission child input: task ID/hash, target label/profile/prompt, exact
  return-home joints, speed/capture bounds, and authorization flags;
- OS/runtime: `XDG_RUNTIME_DIR`, inherited process environment, and ROS/RMW
  variables.

Calibration YAML, `piper_home_pose.json`, `piper_joint_bounds.json`, robot
description files, Tesseract collision/plugin YAML, and captured mission data
remain their existing authoritative file inputs. They were not reinterpreted
as generic application configuration.

## Duplicated-constant audit

An AST scan of Python production and launch files found many repeated literal
values. The most frequent nontrivial values included `2`, `10`, `6`, `3`,
`0.5`, `0.1`, `5`, `0.05`, `1e-6`, `2.0`, `0.2`, `0.25`, `0.001`, `8`,
`0.02`, `10.0`, `30.0`, `0.03`, `0.005`, `0.4`, `0.08`, `15.0`, `0.3`, and
`0.75`.

These are not automatically consolidated. Examples of deliberately distinct
semantics include:

- `0.005 rad` endpoint stillness, home motion tolerance, and feedback-limit
  allowance;
- `0.03 m` depth/geometry thresholds versus `0.03 rad` home tolerance;
- `0.25 s` status propagation, TF timeout, and unrelated perception timing;
- `30 s` recovery, home settling, acquisition handoff, and visual
  reacquisition budgets;
- values named `min_valid_depth_ratio`, `data_timeout_sec`, `debug`, and
  `enable_real_arm_motion` occurring in multiple nodes but belonging to
  different trust boundaries.

Only coordinator/executor values with identical existing ownership were moved.
No cross-node semantic deduplication was attempted.

## Compatibility

The Phase 1 module-level timeout/retry exports remain available from
`target_scan_mission_node.py`, but their values now come from
`MissionWorkflowConfig`. ROS YAML/launch overrides still win at declaration
time. List-valued parameters are frozen as tuples only after ROS has supplied
the final value.
