# PiPER supervised physical requalification checklist

> 2026-08-17 scope extension: this checklist must also qualify the later
> mission-shutdown/runtime-gate simplification. In particular, prove that a
> controllable non-motor failure cannot veto fresh staged home, that stale
> camera/tracking/workflow/obstacle or Tesseract-limit telemetry is not direct
> home authority, and that autonomous startup/terminal shutdown does not issue
> a redundant hold-service request. Do not physically induce a motor fault;
> the motor-loss no-command route is software/simulation tested only.

## Authority and stop rule

This checklist is the only progression authorized by the final equivalence
audit. It does not itself authorize motion. A trained operator must explicitly
authorize each hardware stage, remain at the emergency stop, and stop on any
unexpected direction, contact, vibration, following error, stale telemetry,
publisher overlap, or process leak.

Complete stages in order. Do not skip a stage because a later test appears more
convenient. A failed or ambiguous stage blocks every later stage until the
cause is understood, corrected under a separately reviewed task, and all
affected earlier stages are repeated.

For every stage record:

- date/time, operator and observer;
- Git commit plus `git diff --stat` of the exact tested tree;
- ROS domain ID and launch/command used;
- parameter dump, home-profile hash, motion-limit hash, worker generation, and
  collision-model qualification;
- relevant logs, action result, scan manifest, and incident notes;
- explicit `PASS`, `FAIL`, or `NOT RUN`.

## Global prerequisites

- [ ] Mechanical inspection is complete: camera holder, L515, gripper, J5/J6,
  cables, fasteners, tracks/base mount, and table clearance are secure.
- [ ] The calibrated rough-home, mission-ready J6, and storage J6 values are
  intentionally selected and their file hash is recorded.
- [ ] The active hand-eye calibration and robot model are the intended files;
  this audit did not recalibrate or alter them.
- [ ] The workspace is cleared and guarded; no person is inside arm reach.
- [ ] Emergency stop and physical power isolation are reachable and tested.
- [ ] One operator watches the arm and one terminal/log observer is preferred.
- [ ] Real motion remains disabled until the checklist explicitly reaches an
  enable/motion stage.
- [ ] Exactly one possible `/joint_ctrl_single` publisher is confirmed before
  any enable.
- [ ] Contact/occlusion manipulation remains disabled. This checklist does not
  qualify push, pick/place, branch removal, or human-adjacent operation.

## 1. Build and tests only

Motors, camera, driver, GPU worker, GUI, coordinator, and robot stack remain
off.

- [ ] Verify the tested revision and cleanly identify all working-tree changes.
- [ ] Source ROS 2 Foxy and run the five-package symlink build.
- [ ] Run the complete mobile suite.
- [ ] Run root GUI/reconstruction/calibration tests.
- [ ] Run all three isolated perception suites.
- [ ] Run `colcon test` and confirm every functional target passes; record the
  known style-only findings separately.
- [ ] Run core and compact rootless Tesseract qualification and confirm
  `real_arm_motion=false`.
- [ ] Confirm the holder-floor incident remains rejected below 5 mm.

Exit criterion: results equal or improve on
`final_equivalence_report.md`, with no new functional failure.

## 2. ROS graph with no hardware

Use fake/disabled adapters only. Do not open CAN and do not create a real joint
command publisher.

- [ ] Start the coordinator/gateway and the intended no-hardware composition.
- [ ] Capture `ros2 node list`, `ros2 topic list -t`, `ros2 service list -t`,
  and `ros2 action list -t`.
- [ ] Compare the graph to `external_contracts.md`.
- [ ] Inspect QoS for plan/history/sensor channels.
- [ ] Submit a fake mission and confirm the public feedback/result schema.
- [ ] Exercise queued cancel, active cancel, deadline, and fake child crash.
- [ ] Confirm no `/joint_ctrl_single` command reaches hardware and no process
  outside the owned fake generation is signalled.

Exit criterion: graph/contracts match, simulated terminal routes match the
characterization suite, and cleanup leaves no owned child.

## 3. Recorded data

No live camera and no enabled motors.

- [ ] Replay an approved synchronized RGB, depth, intrinsics, masks, TF/joint
  timestamps, target, obstacle, and capture sequence.
- [ ] Verify target acquisition/lock, stable landmark behavior, occlusion
  classification, capture acceptance/rejection, and scan history.
- [ ] Compare selected viewpoints, plan IDs/hashes, retry/replan/reacquisition
  decisions, capture requests, terminal result, and shutdown route with the
  recorded expectation using numeric tolerances.
- [ ] Run reconstruction on accepted captures and inspect scale, alignment,
  completeness, and manifest hashes.

Exit criterion: no unexplained decision or geometric deviation. If a complete
mission recording is unavailable, create one only during a later successful
supervised run; do not pretend the vendor RealSense bag is a mission replay.

## 4. Live sensors with motors disabled

The arm remains physically disabled throughout.

- [ ] Start only the intended sensor/perception/TF stack.
- [ ] Verify one RGB and one depth stream, expected resolution/frame rate, and
  healthy timestamp diagnostics.
- [ ] Confirm joint and arm-status channels report the physically disabled
  state and that no enable request is sent.
- [ ] Place a non-contact test target in view and verify GroundingDINO/SAM,
  depth support, target frame, tracking health, obstacles, and target stability.
- [ ] Observe stationary-target jitter and reject background-depth/mask-layer
  contamination rather than accepting a moving landmark.
- [ ] Stop the stack through its owner and verify the L515 is actually released
  and all owned camera/perception processes exit.

Exit criterion: fresh coherent telemetry and clean process ownership with zero
motor command/enable activity.

## 5. Arm enabled, no autonomous trajectory

No acquisition, scan, home, or other autonomous path is approved.

- [ ] Clear the workspace and station the operator at emergency stop.
- [ ] Start the driver and inspect all-six motor feedback, limits, faults,
  watchdog state, and command-publisher count.
- [ ] Enable at the approved commissioning setting only.
- [ ] Command/prove hold at the current measured joint positions without a
  trajectory.
- [ ] Verify fresh settled feedback, then perform the approved safe-disable
  transaction without moving away from the current safe pose.
- [ ] Confirm all six axes disabled and no mechanical drop/contact occurs.

Exit criterion: enable, current-position hold proof, and disable are reliable;
any axis dropout is an immediate stop and requires operator recovery.

## 6. Single low-speed supervised motion

Use the established low commissioning speed, a small collision-free motion,
and no target/contact task.

- [ ] Confirm exact plan identity, motion-limit hash, collision qualification,
  fresh telemetry, and one command publisher.
- [ ] Review start, end, dense path, direction, duration, and maximum joint step
  before approval.
- [ ] Execute one small motion while watching direction and clearance.
- [ ] Confirm smooth newest-due 20 Hz command behavior, bounded following
  error, settled endpoint, and no unplanned stop/reversal.
- [ ] Hold and safely return to the starting commissioning pose.

Exit criterion: the physical path matches the reviewed request and controller
feedback without contact, chatter, or timing fault.

## 7. Single viewpoint

Use one visible, unobstructed, non-contact target and cap execution to one
approved viewpoint/capture.

- [ ] Acquire and lock the measured target from a conservative start.
- [ ] Review the selected viewpoint, target distance/boresight, joint path,
  holder/table clearance, and expected camera direction.
- [ ] Execute only that viewpoint.
- [ ] Confirm settled tracking, one capture request, acceptance decision,
  achieved camera FK, persisted RGB-D/mask bundle, and manifest integrity.
- [ ] Cancel before another viewpoint and verify the intended safe shutdown.

Exit criterion: one plan, one trajectory, and one capture are correlated and
the cancel route completes safely.

## 8. Return-home test

This is a dedicated safety gate. Start from a reviewed collision-free pose
reached in Stage 6 or 7.

- [ ] Confirm current joints, fresh all-six motor authority, home-profile hash,
  holder/L515 external-clearance evidence, and no unexpected object/person in
  the route. Record current limits diagnostically; they are not authority for
  the native direct-home service.
- [ ] Review the three terminal stages: current pose to terminal-only pre-home,
  then rough home, then storage J6 in the configured decreasing direction.
- [ ] Execute at the approved low speed under emergency-stop supervision.
- [ ] Confirm pre-home and rough-home feedback within their configured tolerances.
- [ ] Confirm storage-J6 feedback, retention of the final controller target,
  all-six disable, authority revocation, and process cleanup in that order.
- [ ] Confirm logs show no autonomous current-position hold-service request
  between the final configured-home proof and disable.
- [ ] Trigger one controllable perception/planning failure with motor authority
  intact and confirm the original failure is reported but cannot block fresh
  PRE_HOME, ROUGH_HOME, STORAGE_WRIST and disable.
- [ ] Specifically watch J5/J6, gripper-to-J1, holder/table clearance, and any
  unexpected J6 branch/wrap behavior.

Exit criterion: exact staged home, retained final target, disable, and cleanup are
proved, including the non-motor failure route. Any wrong J6 direction or contact is a safety-relevant failure and
blocks all scanning.

## 9. Short scan

Use a simple cube-like target, no physical occluder, conservative speed, and a
small capture/view cap.

- [ ] Run the production action through startup, acquisition, lock, clear
  occlusion probe, several viewpoint/capture cycles, and terminal shutdown.
- [ ] Confirm each rejected plan/view is replaced according to the bounded
  typed recovery policy.
- [ ] Verify accepted captures use actual camera FK and measured target state.
- [ ] Inspect Y-side/elevation diversity, target stability, capture quality,
  and the generated dataset/manifest.
- [ ] Confirm terminal rough home, storage wrist, hold, disable, cleanup, and
  action result.

Exit criterion: the short scan is coherent and safe even if coverage is
intentionally insufficient because of the cap.

## 10. Complete scan

Use the same simple non-contact target and production 8/24 capture limits.

- [ ] Run one complete production mission without manual control overlap.
- [ ] Verify adaptive planning stops only on achieved useful coverage or the
  documented bounded terminal condition.
- [ ] Check both feasible Y sides, azimuth/elevation diversity, measured
  surface novelty, duplicate rejection, and capture quality.
- [ ] Inspect all stored bundles and run reconstruction after the tracked-base
  homed trigger or the approved offline equivalent.
- [ ] Confirm the final action result, safe-shutdown flag, capture count,
  manifest hash, mesh job ID, and process-health record.
- [ ] Confirm the system accepts a new mission after completion.

Exit criterion: one complete mission passes every behavior and shutdown check;
no contact manipulation is introduced.

## 11. Repeated complete missions

Run at least three complete missions one after another with the same controlled
target, then one bounded cancellation between them.

- [ ] Confirm no state, authorization, plan, telemetry, capture counter,
  process handle, camera process, command publisher, or dataset directory leaks
  between missions.
- [ ] Confirm every mission independently performs startup and staged shutdown.
- [ ] Confirm queue/deduplication and rescan behavior.
- [ ] Confirm failed zero-capture missions remove empty scan directories while
  successful datasets remain intact.
- [ ] Compare phase sequences, coverage, viewpoint choices, execution timing,
  and shutdown evidence across runs; explain material variance.
- [ ] Perform one operator cancel and confirm the characterized cancel route.

Exit criterion: repeated missions are safe, independent, restartable, and
leave only the intended persistent coordinator/gateway/RViz processes.

## Final sign-off

- [ ] Every stage is `PASS` with linked evidence.
- [ ] No unexplained or safety-relevant deviation remains.
- [ ] Any observed baseline oddity is explicitly accepted or assigned a
  separate corrective task; it was not silently changed during testing.
- [ ] The operator, safety reviewer, and software reviewer sign the tested
  revision and configuration.

Only after this sign-off may the system be called physically requalified for
the exact tested non-contact scanning configuration. Occlusion manipulation,
outdoor/mobile-base operation, new TF integration, new payloads, or altered
speed/calibration each require separate qualification.
