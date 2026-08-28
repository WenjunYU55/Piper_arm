# Historical supervised cube workflow verification handoff

> Historical snapshot through 30 July 2026. Do not use this file as the current
> operator procedure; use `OPERATOR_COMMANDS.md`.

Status updated 2026-07-30. Rough acquisition retains the collision-qualified
15-degree primary sweep. An exact post-settle `target_mask_missing` result with
`obstacle_count: 0` now supplies the executor's timestamped typed empty scene
directly, so a missed duplicate scene relay cannot abort the remaining looks.
Any detected obstacle still requires the separately depth-projected typed
scene before another move.

Status updated 2026-07-30. False semantic target seeding is now fail-closed. The expensive detector
does not run automatically at GPU-pipeline startup: rough acquisition requests it only after a
viewpoint settles, while later heavy requests remain limited to loss/low-confidence recovery and
correlated post-settle scan captures. The target-only GroundingDINO caption is `green cube .`.
Acceptance requires semantic confidence at least 0.60, calibrated-green occupancy at least 0.15,
a cube-like box, and a separately validated refined SAM2 mask. Tracked-mask fallback applies the
same appearance gate. Across 63 recorded confirmed physical detections, confidence averaged 0.8435
and the minimum was 0.7947 under the former mixed prompt. Replaying the final target-only prompt
accepted 63/63 with mean 0.8751 and minimum 0.8342. The absent-scene replay scored 0.5627 with
10.3 percent green occupancy and was rejected as `target_mask_missing`; a recorded real cube was
accepted at 0.8484 with a 97.0-percent-green refined mask. Hand/person/finger are the only semantic
unsafe labels; configured non-hand clutter is candidate-movable but is never moved automatically.
Invalid/untrusted geometry remains fail-closed.

Status updated 2026-07-30. The command-free missing-target regression at
`[0.25, 0.0, 0.0]` now passes. Rough acquisition retains its five primary
looks and supplies fifteen bounded 0.30 m fallback candidates for Tesseract to
select from when the primary centerline set has no usable IK. The exact live
start also had J2 and J3 just outside their normal planning margins. A rough
acquisition may therefore declare at most those two joints in one combined
bootstrap target; every declared delta remains individually bounded and
directional, undeclared joints must remain fixed, and the dense path must
monotonically leave only the specifically qualified folded-start contacts.
The core qualification selected five views for both the all-zero centerline
case and the exact dual-limit live start. A rebuilt live GUI Step 2 reached
`PROPOSAL_READY` with five views, a hardware-qualified collision model, and
zero command samples. The arm remained disabled and Step 3 was not confirmed.

Status updated 2026-07-29. The native GUI completed the former endpoint-only
route end to end at 5 percent: rough-coordinate
acquisition, authoritative measured-lock handoff, one exact 13-view Tesseract
proposal/approval, 13 physical viewpoints, 13 full-resolution model updates,
and 13 synchronized RGB/depth/mask/metadata records. The accepted dataset is
`datasets/active_scan/scan_20260729_150616`, bound throughout to plan
`7474484cfd3ddb50`. The live run also validated correlated RGB-D frame pinning
during heavy inference and bounded stationary waiting for a transient obstacle
stream gap; fresh obstacle geometry is still mandatory before every new motion.

The next-plan software contract now maximizes capture diversity inside the same
fully qualified 120–175 degree sector. It builds a 21-candidate dome at -45,
-55, and -65 degrees camera pitch, selects a camera-space-diverse 13-view
subset, and orders that subset as a smooth nearest-neighbour route from the
calibrated current camera position. This removes the former endpoint
pendulum without restoring the four wider-sector poses that previously lacked
bounded IK. MULTIVIEW_SCAN also includes one Tesseract/OMPL planned and
collision-validated return to the operator-recorded powered feedback pose,
`[0.000366362, 0.0, 0.0, 0.0, 0.43869236, 0.0]` rad. The first operator screenshot is authoritative;
its raw disabled feedback was `[0.000366362, -0.02888726, 0.00624495, 0.0, 0.43869236, 0.0]`,
with J2/J3 normalized to their powered limits. The powered pose must be reached and verified before disable. That segment is in the same trajectory
hash and approval, runs only after all 13 records exist, never captures, and
ends in a current-position hold rather than automatic disable. If return-only
telemetry fails after all 13 records exist, the executor holds the current pose
and completes the capture session with a return warning; it does not schedule
Step-4/5 recovery.

Status updated 2026-07-25. Rough-coordinate acquisition and the normal Tesseract scan route have
now completed a supervised physical acceptance run. A `base_link` hint at
`[0.38, -0.12, 0.0]` moved the camera through the guarded acquisition path at 5 percent, obtained a
new measured GroundingDINO/SAM2 lock, and handed off to the normal workflow. Qualified plan
`709b2b86435c9537`, trajectory hash
`eadcfd404cf53e8b583202c89dce9adfe33beb01134080916e3118d4e8967311`, then moved the physical arm
through five collision-validated viewpoints at 5 percent. The workflow accepted and modeled all
five full-resolution clouds and `/supervised_cube_workflow/finish_scan` returned `scan complete`.
This is successful supervised guarded acceptance, not functional-safety certification and not
permission to remove operator approval, collision, tracking, freshness, or cancellation gates.

The latest full restart brought up the disabled PiPER driver, feedback TF, accepted hand-eye TF,
CUDA perception/tracking, the rootless Tesseract worker, and the Foxy proposal stack. Tracking was
healthy and stable; the reachability filter continuously reported 11 of 18 candidates safe. After
the proposal wrapper was limited to Foxy/PiPER overlays and its nodes were staggered, canonical
requests reached the isolated worker and returned structured results. Request
`41586d716de1afb2d83c9c30d4d2fd06` was correctly rejected because exact live J2 was approximately
`-0.034 rad`, below the then-current stale mode-0 lower limit of `0.0 rad`. No arm command publisher was introduced and
no physical movement occurred during this proposal test.

GUI enable is fixed. The driver enable service now uses the installed SDK's feedback-confirmed
EnablePiper/DisablePiper handshake at 10 ms instead of a 500 ms EnableArm/gripper retry loop, with one gripper
command only after success. Three focused tests, rebuild, and a supported live disable/re-enable cycle passed;
all six enable bits were true afterward with normal status and no motion publisher. An earlier 0xFC zero-offset
role command is recorded but was not the cause because acceptance passed without a power cycle.

Continued testing passed stable 200.05 Hz feedback, CAN-control status, ERROR-ACTIVE CAN, zero command
publishers, exact live TF/FK comparison, Tesseract functional/qualification tests, and proposal-only fail-closed
graph checks. After explicit workspace/cable/E-stop confirmation, a GUI-equivalent 5-percent command moved
J6 by +0.02029 rad and returned it within 0.00210 rad of its measured start with no arm fault. The controller
normalized the slightly negative J2 start to 0.0 during the first command. On 2026-07-25 the
production Xacro and planning limit constant were aligned to the existing valid controller-coordinate
J2 lower bound `-0.044796192 rad`, measured from the enabled compact pose on
2026-07-30. The compact-start failure was then
resolved with deterministic hash-verified 30 mm link1/link2/link5 collision pieces and a
first-bootstrap-segment-only monotonic recovery inside the exact trajectory hash. The recorded
start for `[0.33,-0.14,0.0]` uses J3 `-0.05 rad`, recovery boundary point 55, 352 total points, and
5608 validation samples. Foxy independently requires normal 60 mm proxy clearance at the boundary.
No pair is globally disabled and J6 remains free.

## Verified

- GPU perception runs on CUDA at approximately 7–13 FPS and tracks the labelled green cube.
- The target landmark is normally `LOCKED` and valid, with approximately 1–4 mm measurement error
  and 2–3 px projection error. An occasional single-frame insufficient-depth rejection was seen.
- The pen is detected as a blocking obstacle with valid depth and a stable `base_link` transform.
- The observed pen centroid was approximately `(0.755, 0.158, 0.259)` m in `base_link`.
- The workflow package tests and build passed before live verification.
- The coordinator is dry-run only and has no arm-command publisher.
- A separate `scan_viewpoint_executor_node.py` now accepts only mandatory-Tesseract complete
  six-joint plans, applies conservative bounds/trajectory/capsule/AABB revalidation, exact typed approval, cancellation hold,
  TrackingHealth gating, low-speed incremental joint commands, and automatic capture sequencing.
- The focused build passed; the expanded scan-motion suite has 9 passing tests, and an isolated
  proposal-only launch started the planner/filter/workflow/executor stack with no
  `/joint_ctrl_single` topic advertised.
- The command-free `piper_tesseract_foxy` bridge, canonical hashed spool contract, model generator,
  isolated CPU worker scaffold, typed messages/services, and executor routing are implemented. The
  two Foxy packages build, 15 focused contract/model/approval tests pass, and a proposal-only graph
  check showed neither the bridge nor executor publishes `/joint_ctrl_single`.
- Rootless Tesseract 0.35.0.6 runs and passed model load, mode-0 FK, timed trajectory, a 0.35 rad
  J6-change smoke, conservative camera/mount/local-cable geometry checks, bounded adaptive
  exact-path Bullet validation, and deterministic OMPL-detour regression. The checked-in manifest
  is `qualified_for_hardware: true`; its 2026-07-23 declared supervised guarded qualification was
  repeated twice with identical seed-42 output on 2026-07-24. On 2026-07-25 the 0.005 m global
  margin was retained while named positive compact-arm pair margins were added for the valid
  all-zero pose; an exact `[0.33, -0.14, 0.0]` rough-acquisition regression selected view 1 and
  passed 3501 adaptive collision samples.
- On 2026-07-27 the separate exact folded-start compact suite passed with the 30 mm decomposition,
  J3 `-0.05 rad` recovery through point 55, 352 points, and 5608 samples. The core suite also passed
  with `collision_model_qualified_for_hardware: true`.
- The subsequent Step 2 child-process exit was traced to an obsolete generated repository-root
  overlay, not Tesseract planning or the arm. It lacked `scan_capture` and the four
  `bootstrap_recovery_*` fields in the installed `TesseractPlan`. The root build/install/log outputs
  were removed, both GUI/scan launchers now use the canonical environment preflight, the three
  canonical packages rebuilt, 79 focused tests passed, and core plus compact qualification passed.
  A clean live proposal-only Step 1/Step 2 then returned hardware-qualified plan
  `4f8fc210be5b17a9` with three viewpoints and 443 points for `[0.33,-0.14,0.0]`.
  `scan_capture` remained graph-live, the executor reached `PROPOSAL_READY`, and
  `/joint_ctrl_single` had zero publishers. No approval or motion occurred; all temporary validation
  processes were stopped. The next physical regression remains one separately approved supervised
  5-percent acquisition, not another interface repair.
- Rough-coordinate acquisition is implemented: one atomic typed request binds a unique session ID
  to a fresh finite `base_link` point and yields five distinct camera-facing orbit poses at the
  lesser of the live camera-target range and the configured 0.45 m maximum. Exact duplicate retries
  are idempotent, changed data cannot reuse the ID, and `source_request_id` is preserved through
  candidates, schema-v5 planning, the executor, and GUI. A bounded one-in-flight Foxy delivery retry
  stops after one accepted `ROUGH_ACQUISITION` request whose first segment uses `bootstrap_static`, a
  GUI-selected 1-100 percent SDK execution setting, post-settle
  image/request-correlated GroundingDINO, and a measured-lock
  workflow handoff. The native GUI now owns one worker/scan stack, enforces command-publisher
  exclusivity with a bounded Foxy identity wait plus a zero-publisher/live-owned-stack/executor-node
  proof for a still-UNKNOWN sole endpoint, and requires separate exact confirmations: acquisition
  first, then an explicit current-lock plan request and a distinct 13-view confirmation. A fresh
  authoritative measured lock may be explicitly adopted after terminal acquisition; this starts or
  restarts workflow assessment and waits for `SCAN_READY` without changing the terminal acquisition
  or reusing its approval. There is no reusable 15-minute approval. The current rebuild,
  GUI/mobile tests, Tesseract tests, and rootless qualification runs pass.
- Step-5 capture diversity is session-scoped. The executor records only saved and workflow-accepted
  camera poses, look directions, actual joints, and plan identity. A retry under the same workflow
  session requests exactly `13 - accepted_views`, excludes near-duplicate poses, and requires a new
  exact GUI approval. The planner offers remaining candidates across three elevations and the bridge
  selects a diverse subset before creating a smooth camera-space route. Successful capture-session
  finalization deletes the memory even when the optional return segment ends in a held-pose warning.
- Controller motion-limit snapshots are stabilized across the driver's asynchronous CAN query:
  a new valid hash must persist for seven seconds and three samples. Invalid/stale data still blocks,
  while a persistent fresh valid six-joint generation may replace the runtime hash for the
  position-only SDK MoveJ interface. Planning and approval remain bound to their original snapshot;
  the rollover cannot change the selected speed or approved joint targets.
- The later acquisition child exit/service disappearance is repaired separately from planning:
  the worker writes an atomic heartbeat, the bridge publishes typed acquisition/multiview
  readiness, scan subscriptions are stable rather than recreated on stale input, and every
  nonvisual scan child is launch-critical. Step 1 waits for live parents, nodes, services,
  heartbeat/readiness, and exact command ownership; Step 2 stays disabled otherwise and accepts
  only its own session result. The affected packages build, 58 focused stability tests and a
  143-test wider suite pass, both rootless qualifications pass, and an isolated proposal-only
  launch kept all critical nodes alive until intentional shutdown. No arm approval or motion was
  issued during this stability repair.
- A 2026-07-28 live GUI run exposed a Step-2 response regression despite that isolated validation:
  `PrepareAcquisition` timed out twice and Step 3 never enabled. A real GUI/service regression
  reproduced Fast DDS rejecting an 84-byte service sample against a forced 55-byte XML endpoint
  history; retaining Foxy's native reallocating endpoint QoS made the round trip pass. Step 2 also
  uses a re-entrant two-thread GUI executor, a fresh client endpoint per 8-second attempt, the exact
  same immutable session/payload for automatic and operator retries, generation/attempt callback
  filtering, and a 185-second correlated-plan deadline. Full external-stack live GUI acceptance is
  still required.
- The first acquisition segment no longer requires a preexisting SAM2 obstacle scene. It omits only
  perception obstacles while retaining the qualified robot/camera/cable model, floor proxy, limits,
  feedback, arm/camera health, exact approval, and hold gates. After settling, target-missing results
  must produce a correlated typed empty scene or obstacle-only SAM2/depth scene before another look.
- On 2026-07-24 the acquisition route physically moved at 5 percent, obtained a fresh measured lock,
  and handed off to the normal workflow. The subsequent exact Tesseract plan completed five physical
  viewpoints with all five captures accepted and modeled.
- Live acceptance fixes include consistent-sample depth re-anchoring after real viewpoint changes,
  a two-thread landmark executor for timestamped TF, a 0.75-second SAM2 measurement-age gate,
  matching best-effort sensor QoS for `/piper/target_cloud`, capture status/cloud ordering tolerance,
  and read-only workflow diagnostics.

## Current stopping point

The ROS/Foxy-to-Tesseract boundary, rough-coordinate acquisition, measured-lock handoff, 13-view
execution, and 13 capture/model handoffs were live-accepted at 5 percent on 2026-07-29 using the
former one-SDK-target-per-view executor. Do not fake or clip future start snapshots, weaken limits, or
bypass either exact approval. On 2026-07-27 the GUI handoff was split into explicit acquisition and
scan phases. A 2026-07-29 live 5-percent acquisition then showed that the timed stream sent 7,281
targets even though PiPER MoveJ consumes only positions plus aggregate speed. Schema v5 now binds
fresh driver-published controller limits, selected speed, and `tesseract_stream_v1`; the isolated
worker now preserves and validates the full OMPL/ISP path as a 20 Hz position schedule. Foxy does
not pause at intermediate points, uses a 0.30 rad following guard, and waits at the final endpoint.
The authoritative Step 4
lock is now the same TrackedTarget/TrackingHealth/LOCKED tuple used by acquisition, with
target_landmark retained only as diagnostics. The earlier endpoint-only adapter completed the
supervised 5-percent 13-view run; physical acceptance of the current stream is pending. The
first live 100-percent acquisition attempt hit the sustained following-error hold before ACQUIRED,
although a fresh settled measured lock appeared afterward. Step 4 now explicitly adopts that
  current lock into a new workflow/scan phase. If the GUI-owned worker/scan stack is stopped, the
  explicit action may restart it from direct fresh settled measured tracking, but must wait for the
  workflow's authoritative lock validation before adoption or planning. Step 4 now has bounded
  generation-owned phases; it uses the diagnostic service, a 15-second SCAN_READY deadline,
  multiview readiness/blockers, a 12-second request-queue deadline, and a 185-second result deadline.
  Workflow PLAN_READY instructs the operator to clear clutter and retry rather than exposing the
  removal workflow. The current targeted command
  passes its software tests, including the GUI/scan Fast DDS transport contract; this recovery is not
  acceptance of 100-percent motion.

The Automatic Scan tab is isolated from the manual tab's cached sessions, delayed recovery timers,
plan callbacks, and terminal execution messages. Its ACQUIRED handoff waits for a fresh stable
multiview-readiness generation for one second before the only request is queued. During execution,
temporary target-tracker reacquisition does not cancel an approved view; the arm must still be
stationary and the RGB-D clock healthy. Completion and non-safety abort use the approved return or
exact executed reverse route, with a 30-second saved-home proof before hold, disable, and shutdown.
On 2026-07-27 the GUI/Step-1 crash was independently traced with GDB to Foxy's internal
`ParticipantEntitiesInfo` graph deserializer replaying corrupt shared-memory state. It was not a
Tesseract, perception, CAN, or arm-motion fault. The GUI and supervised scan wrapper now use
`fastdds_gui_udp_only.xml` with effective `ROS_LOCALHOST_ONLY=0`; the profile itself limits UDP to
`127.0.0.1`, disables builtin/shared-memory transport, and retains variable-size histories. A
bounded live Step-1 regression held GUI RSS to 71.5 MiB, preserved both GPU worker PIDs, started all
scan nodes, sent no approval or motion, and cleaned up all owned children.
The executor also requires a
service-triggered synchronized RGB/raw-depth/depth-PNG/mask/metadata record at every accepted view;
that addition passed focused tests/build and the 13-record live audit. The 2026-07-30 GUI run saved
`view_000` through `view_012` in
`datasets/active_scan/scan_20260730_221116`; measured joint metadata proved 13 distinct poses but the
old maximin order looked like a two-endpoint pendulum. The three-elevation replacement subsequently
passed the real ROS-free Tesseract backend with 13 views and one return segment in 29.9 seconds.
Repeat the fixed-cube
13-view run for repeatability, inspect the saved records/clouds, quantify coverage/registration, exercise
cancel-to-hold during an intentional
acceptance test, and verify the bounded no-lock acquisition sweep. The GUI lifecycle checks that
stopping it leaves driver/camera/perception running, restores manual publishing only after executor
shutdown, and cannot approve a second scan also remain mandatory regressions.

The earlier pen centroid near `x=0.755 m` was outside the configured `workspace_x_max=0.70 m`; this
is historical scene evidence, not a permanent planner blocker. Invalid obstacle geometry still
fails closed. The 2026-07-29 run validates the declared supervised 13-view path only; higher speed,
dynamic targets, changed collision assets, or unattended use remain unqualified.

Safe shutdown starts with GUI **Cancel and Home**, which holds, retraces executed approved
endpoints to configured home, proves the final hold, disables, and stops the managed stack.
The exact reverse of the first rough-acquisition segment reuses that segment's qualified
bootstrap-static scene when perception obstacle geometry has not been created yet; this exception
does not apply to later acquisition looks or multiview motion. Once the executor reports
`configured home reached`, the shutdown hold cannot start the reverse history a second time.
GUI **Disable** also explicitly
commands the fresh current feedback pose, requires at most 0.025 rad target
error and 0.005 rad sample motion for one full second, and only then calls the
disable service. If the eight-second proof fails it leaves the motors enabled.
Require `disable -> True`; do not send a different zero/home command immediately
before disabling.

The coordinator process started, but the first `/supervised_cube_workflow/start` call was made before
the process existed. A later call appeared to wait and was not diagnosed before stopping.

## Resume remotely (no arm motion)

1. Start the perception and coordinator terminals with `ROS_DOMAIN_ID=42` and
   `ROS_LOCALHOST_ONLY=1`. Never run `scripts/robot/enable_piper.sh` for this test.
2. Start monitors for `/piper/supervised_workflow_status` and `/piper/removal_plan` before calling
   `/supervised_cube_workflow/start` because they carry event messages.
3. Confirm the service exists with:

   ```bash
   ros2 node list
   ros2 service list | grep supervised_cube_workflow
   ros2 service type /supervised_cube_workflow/start
   ```

4. Call it with a bounded wait:

   ```bash
   timeout 10s ros2 service call /supervised_cube_workflow/start std_srvs/srv/Trigger '{}'
   echo "exit code: $?"
   ```

5. Expected result with the current scene: an invalid dry-run plan whose reason says the obstacle
   center is outside the configured workspace. The arm must remain disabled.

If the service times out with exit code 124, save the three diagnostic outputs above and the
coordinator terminal log.

## Resume in the lab

1. Keep the cube fixed. With the arm disabled, move only the pen toward the robot base until its
   stable `base_centroid.x` lies between 0.10 and 0.70 m.
2. Verify `valid: true` and `scene_blocked: true` on `/piper/obstacle_instances_3d`.
3. Repeat the coordinator dry-run and inspect its proposed plan and RViz markers.
4. Approve only after manually checking workspace clearance. For initial validation, move the pen
   by hand and use `confirm_action_complete`; do not use arm motion.
5. At `SCAN_READY`, run `run_supervised_viewpoint_execution.sh` in proposal-only mode. The current
   workstation orbit supplies thirteen ordered 8-degree candidates and Tesseract must select five valid
   collision-aware views while `/joint_ctrl_single` has no executor publisher. Valid unsafe objects
   remain `scene_blocked` for this supervised workflow but are passed to Tesseract as collision boxes;
   invalid obstacle geometry still blocks planning.
6. Complete the TF/static-cube/small-J1-J2 gates in `OPERATOR_COMMANDS.md`.
7. For SDK MoveJ waypoint-adapter acceptance, use the GUI flow at 5 percent: approve only the exact acquisition
   hash, verify measured lock and `SCAN_READY`, then explicitly select Prepare Scan from Current
   Lock. The rough acquisition candidate set includes ±45-degree yaw/pitch and diagonal
   yaw/pitch looks so the configured 0.30 m coordinate tolerance is actually searched. Do not rely
   on an automatic post-lock plan.
8. Inspect the fresh correlated 13-view plan and all six Tesseract joint trajectories, allow J6
   to move normally, then use the separate 13-view confirmation. Require target drift no more
   than 15 mm, fresh matching `/piper/motion_limits`, one exact arm-only target at a time,
   feedback-gated advancement, waypoint/no-progress and cancel-to-hold behavior, no automation
   gripper command, settled capture, and clean status. Real arm obstacle removal remains a separate
   safety phase.

## J6 planning policy

On 2026-07-17 the operator confirmed J6 is fully safe for collision-aware planning and
execution. On 2026-07-24 the fixed-J6/J1-J5 backend and helper were removed. Tesseract is now
mandatory and treats J1-J6 equally, with no J6 lock or J6-specific cost, while camera-cable
clearance remains scene geometry. The old
pre-zero measured J6 bounds remain invalid historical data, and the failed analytical hand-eye
correction remains rejected; this update does not claim replacement bounds or a new calibration run.
