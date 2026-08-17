# Phase 5 safety evaluator shadow baseline

> Historical phase record, superseded 2026-08-17. The non-authoritative
> `SafetyEvaluator` and `SafetyComparisonLogger` were removed rather than
> promoted after live logs showed high-volume duplicate RETURN_HOME
> disagreements. `safety_evaluator.py` now contains only `SafetyMode`,
> `ObstacleAuthority`, immutable `RuntimeGatePolicy`, and its phase mapping.
> `scan_viewpoint_executor_node.py` remains the one authoritative gate and
> command owner.

## Scope and authority

Phase 5 adds a pure, typed safety/readiness evaluator around existing executor
behavior. It does not replace any legacy decision. Tesseract proposal
validation, `runtime_reasons`, path validation, hold publication, abort logic,
home, disable, process cleanup and action results retain their Phase 4
authority and ordering.

The implementation is
`piper_mobile_manipulation/safety_evaluator.py`. It reads one immutable
`TelemetrySnapshot` plus explicit frozen inputs. It does not import or call
ROS, read node parameters, access a clock, publish commands, mutate telemetry,
or select a mission transition.

## Named operating modes

The modes reflect contexts that actually exist in the executor:

| Mode | Existing behavior represented |
|---|---|
| `PLAN_VALIDATION` | Command-free Tesseract proposal/schema/limit evidence |
| `ACQUISITION_APPROVAL` | Settled first-look approval; camera required, target lock and semantic obstacle scene not yet required |
| `ACQUISITION_MOTION` | Approved rough-target look; later looks require semantic obstacles unless the exact bootstrap-static exception applies |
| `SCAN_APPROVAL` | Fresh camera, target/tracking, obstacle, workflow, controller and arm evidence before exact approval |
| `SCAN_MOTION` | Already approved scan segment; tracking is diagnostic, while arm/camera/limits and the approval-bound obstacle policy remain active |
| `SCAN_CAPTURE` | Stationary capture phase; missing obstacle telemetry may wait, but fresh camera/joints and the separate workflow/capture gates remain required |
| `RETURN_HOME` | Dedicated home evidence; camera, target, tracking and semantic obstacles are not runtime dependencies, but feedback/limits/arm/motor authority remain mandatory |
| `HOLD_CURRENT` | Existing `publish_hold` boundary: fresh finite joint feedback plus real command authority |

`PREFLIGHT`, `OCCLUSION_MANIPULATION` and `DISABLE_ONLY` are deliberately not
executor modes. Preflight/disable are coordinator-owned. Contact manipulation
is proposal-only and unqualified, so inventing an executable safety mode would
misdocument current behavior.

## Extracted rules

`SafetyEvaluator` records:

- immutable telemetry receipt ages and unchanged strict
  `age > maximum` staleness semantics;
- required-channel freshness by mode;
- controller-limit validity and shape compatibility;
- six finite joints and existing feedback-only limit tolerances;
- arm error, angle-limit, communication, six-driver enable, motor-fault and
  watchdog evidence;
- camera timestamp health;
- missing/stale, transform-transient, blocked and invalid obstacle evidence;
- settled-joint proof supplied explicitly by the existing stateful window;
- scan-approval tracking state, prediction-only status, measurement age,
  speed allowance and target status;
- workflow/capture readiness, mission authorization, target drift, plan
  validity, collision qualification and path validity supplied explicitly by
  their existing owners.

No threshold was moved into `TelemetryStore` or changed. `SafetyProfile` is
constructed once from the executor's existing parameter values.

## Shadow comparison

`SafetyComparisonLogger` stores at most 256 immutable comparisons. Each record
contains context, mode, legacy/shadow permission, typed failure codes, reason
lists, telemetry ages, and separate permission/code/text agreement flags.

The executor records comparisons at:

1. command-free Tesseract proposal validation;
2. exact approval authorization/schema/collision/target-drift/service/path
   boundaries;
3. every return from the compound `runtime_reasons` safety gate; and
4. every `publish_hold` outcome when a telemetry snapshot exists.

The legacy value is returned before and after comparison unchanged. A mismatch
emits a JSON `SAFETY_SHADOW_DISAGREEMENT` warning. No comparison value is read
by command, approval, state-transition, recovery, home, disable or result
logic.

## Known disagreements and retained behavior

- Collision-model qualification is enforced by the existing approval and
  fresh-path gates, not inside `runtime_reasons`. A deliberately synthetic
  unqualified runtime call therefore produces a shadow disagreement while the
  legacy function still returns its original value. Such a call cannot gain
  normal hardware approval.
- `HOLD_CURRENT` preserves the existing narrow `publish_hold` contract. It
  does not add an independent arm-status/motor-health read to that helper;
  broader powered-motion guards remain authoritative around it.
- Scan approval enforces target/tracking evidence. Tracking and target status
  intentionally become diagnostic during an exact already-issued scan segment
  and do not retroactively cancel it. This distinction is encoded rather than
  normalized away.
- Transform-only obstacle validity text still passes through the Phase 2
  `legacy_failure_adapter`, because the current ROS obstacle message exposes a
  human-readable validity reason. It is comparison/input compatibility, not a
  new behavior-affecting string parser.

These are observable boundaries for later equivalence review, not Phase 5
policy changes.

## Tests

`test_safety_evaluator.py` covers all eight modes, fresh/stale joints and
camera, target loss, prediction-only/stale tracking, missing/stale/unsafe
obstacles, unavailable/invalid motion limits, absent mission authorization,
target drift, invalid planner/schema/collision/path evidence, motor authority,
capture settling, immutable decisions, bounded comparison storage, executor
shadow agreement and a deliberate logged disagreement.

## Validation result

Validation on 2026-08-14 was software-only. No arm, camera, GPU worker, robot
driver or mission stack was started.

- Pre-change focused characterization baseline: 203 passed.
- New `test_safety_evaluator.py`: 33 passed.
- Combined evaluator/Phase-1-executor/scan-motion selection: 195 passed.
- Complete `piper_mobile_manipulation/test`: 504 passed.
- Normal five-package Foxy `colcon build --symlink-install`: passed.
- Registered colcon aggregate: 864 assertions, 0 errors, 118 pre-existing
  lint failures and 1 pre-existing hardware-dependent skip. Every registered
  functional target, including the new safety evaluator, passed; the lint-debt
  count is unchanged from Phase 4.
- Root GUI/reconstruction/calibration selection: 58 passed, 1 skipped.
- Heavy worker, SAM2 worker and target-selection suites: 5/5, 6/6 and 19/19
  passed.
- New production/test files pass `ament_flake8`, `ament_pep257` and byte
  compilation. AI YAML parsing and `git diff --check` pass.

A command-free rootless Tesseract qualification rerun was attempted even
though Phase 5 changes no model/planner code. The execution wrapper detached
twice during the core suite before producing a terminal report; both exact
test-only process trees were identified and stopped. The Phase 4 completed
core/compact qualification remains the latest count-bearing evidence, so this
attempt is not reported as a new pass or failure.
