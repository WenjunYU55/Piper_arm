# PiPER eye-in-hand system handoff

Status updated through 2026-07-29 from the checked-out source, architecture records, test results,
operator confirmations, the historical five-view acceptance, and the successful exact 13-view
physical acceptance session. Runtime processes are
transient; use `OPERATOR_COMMANDS.md` to inspect them rather than relying on this handoff as a process
manifest.

## Executive state

This repository is an eye-in-hand RGB-D perception and supervised multi-view capture system for an
AgileX PiPER arm with an Intel RealSense L515. Its strongest working areas are camera timestamp
safety, GroundingDINO/SAM2 perception, base-frame target tracking, explicit arm control, and
fail-closed process ownership. The operator reports that perception/tracking are solid and that the
arm joints move accurately.

Collision-aware viewpoint planning is implemented. A command-free
Foxy/Tesseract adapter, hashed filesystem boundary, isolated CPU worker scaffold, and guarded
six-joint executor route now exist. The Tesseract backend treats J1-J6 equally with no J6 lock or
special cost. The former fixed-J6 solver and backend were removed on 2026-07-24, so Tesseract is
mandatory. Rootless Tesseract 0.35.0.6 ran locally on
2026-07-20 through 2026-07-23 and passed model load, mode-0 FK, explicit OMPL-to-ISP timed planning,
a 0.35 rad J6-change smoke, a complete live five-view proposal around a detected foreground box,
conservative camera/mount/local-cable geometry, bounded adaptive exact-path Bullet validation, and
a deterministic OMPL detour regression. The checked-in collision manifest is qualified for
supervised guarded execution on 2026-07-23 and was requalified twice with identical seed-42 output
on 2026-07-24. On 2026-07-25 it retained the 5 mm global margin, added named positive compact-pair
margins for the valid all-zero pose, and passed the exact cold-start rough-acquisition regression.
Later on 2026-07-25 the recorded nonzero folded start was fixed with deterministic, hash-verified
30 mm link1/link2/link5 collision pieces plus an acquisition-only monotonic recovery embedded in
the exact approved trajectory. The exact `[0.33,-0.14,0.0]` regression now uses J3 `-0.05 rad` through
point 55 of a 352-point smooth path and passes 5608 validation samples. Core and compact qualification both
returned hardware-qualified true; no pair was globally disabled and J6 remains free.
On 2026-07-24, exact qualified plan
`709b2b86435c9537` with trajectory hash
`eadcfd404cf53e8b583202c89dce9adfe33beb01134080916e3118d4e8967311` completed five physical
viewpoints at 5 percent and all five capture/model handoffs succeeded. Qualification and this
acceptance are not permission to bypass exact operator approval or any runtime gate.

Rough-coordinate target acquisition is also implemented. A fresh finite `base_link` hint plus an
explicit start creates five distinct 0.45 m camera poses (center, left, right, camera-up, and
camera-down), keeps optical +Z pointed at the hint, requests a schema-v5 `ROUGH_ACQUISITION`
Tesseract plan, and can execute only through the guarded sole publisher at the GUI-selected
1-100 percent session speed.
The first exact segment is `bootstrap_static` and deliberately omits DINO/SAM obstacle output so
an unseen cube cannot prevent the initial camera-facing move; the qualified robot/camera/cable
model, floor proxy, limits, feedback, health, approval, and hold gates remain. Each settled pose
then requires a post-settle image and exact GroundingDINO request/status correlation. No later
look begins until the matching typed semantic scene is available; obstacle-only results use IDs
2+ while target ID 1 remains reserved, and a zero-obstacle result emits an exact-stamp empty scene.
a new stable measured lock within 0.30 m stops the sweep, starts the workflow, waits for
`SCAN_READY`, and hands back to normal tracked-target planning.

The native GUI exposes this as **Acquire & Scan**. It owns only one locked Tesseract worker and
one scan-stack child, disables its manual publisher, rejects competing command publishers, and
never enables motors. A Foxy endpoint reported as `_UNKNOWN_` receives a bounded two-second
identity-resolution wait; a still-unknown sole endpoint is accepted only with the GUI's prior
zero-publisher proof, live owned stack, and graph-present executor. Missing evidence or an additional
publisher still fails closed.
Acquisition and scanning now require two separate confirmations: the first binds only the displayed
target-matching acquisition execution hash. If acquisition terminates but perception subsequently
has a fresh authoritative measured lock, explicit **Prepare Scan from Current Lock** may adopt that
lock as a new scan phase, start/restart workflow assessment, and wait for `SCAN_READY`; it does not
relabel the acquisition or reuse its approval. The action then issues one fresh correlated request and the second confirmation
binds only its exact 13-view execution hash. There is no reusable 15-minute authorization. The
workflow and GUI now use the same fresh valid stable `base_link` TrackedTarget,
TrackingHealth TRACKING/camera-settled/nonprediction, and exact `LOCKED` status as the
authoritative Step 4 lock. The target-landmark route remains diagnostic and cannot independently
block this transition.
Step 2 now treats its service acknowledgement as a bounded generation-owned phase: each
`PrepareAcquisition` call uses a fresh endpoint and an 8-second deadline, the first timeout repeats
only the exact immutable session/payload, and the second returns to an operator retry without
inventing another session. Its matching plan has a 185-second deadline. The 2026-07-28 timeout was
also reproduced with the real GUI and acquisition nodes: forcing Foxy endpoint QoS from XML selected
a fixed 55-byte history and rejected the 84-byte service sample. The launchers now keep the
loopback-only participant XML while retaining native reallocating endpoint QoS. That command-free
round trip passes in software; full external-stack live acceptance remains pending.
Step 4 uses explicit generation-owned phases, fresh diagnostic service state, a 15-second workflow
deadline, multiview readiness/blockers, and 12/185-second queue/result deadlines. Movable clutter
stops at workflow `PLAN_READY` with a clear-workspace instruction; no removal approval is exposed.
The isolated worker has a 150-second internal planning budget, checks it
between individual OMPL attempts and collision samples, and reserves five
seconds to emit a correlated result before the bridge's 180-second boundary.
Normal completion uses the hash-bound return-home segment. A non-safety abort at a reached
endpoint may reverse only already executed approved targets; any safety,
telemetry, collision, obstacle, hardware, progress, emergency-stop, or
operator-cancel blocker holds.
focused Foxy rebuild, GUI/mobile tests, Tesseract tests, and seed-42 rootless qualification runs
pass. On 2026-07-24
the rough-coordinate route moved at 5 percent, obtained a post-settle measured lock, and handed off
successfully to the five-view scan.

The later 2026-07-25 Step 2 process exit was an environment/interface mismatch, not a planning or
arm fault. An obsolete generated `/home/prl/Piper_arm/install` overlay shadowed the canonical
workspace, did not contain `scan_capture`, and exposed an older `TesseractPlan` without the four
bootstrap-recovery fields. The verified generated root `build/install/log` trees were removed.
`start_gui.sh` and the supervised scan wrapper now use `source_piper_foxy_environment.sh`, which
clears inherited overlay variables and verifies exact canonical package prefixes, the capture import,
and the installed recovery schema before starting children. The canonical packages rebuilt, 79
focused tests passed, and both core and compact-start Tesseract qualification passed without real
motion. A clean live proposal-only Step 1/Step 2 then returned valid hardware-qualified plan
`4f8fc210be5b17a9` for `[0.33,-0.14,0.0]`, with three six-joint camera-facing viewpoints, 443
trajectory points, and hash
`50534f4fedae546b76f3690c524e8b9aa18e122c40186156b0f626c015020f35`.
`scan_capture` remained live, the executor reached `PROPOSAL_READY`, and `/joint_ctrl_single` had
zero publishers. No approval or motion occurred, and all temporary validation processes were
stopped. No joint limit, collision, command, or J6 policy was altered by this repair.

On 2026-07-27, a separate Foxy middleware defect caused two GUI participants on domain 42 to grow
to approximately 125 GiB before the kernel OOM killer terminated them; host starvation then made
perception supervision restart. GDB traced the allocation to
`rmw_dds_common::ParticipantEntitiesInfo` graph deserialization, not an application message,
Tesseract, GPU inference, CAN, or arm motion. `start_gui.sh` and
`run_supervised_viewpoint_execution.sh` now load `fastdds_gui_udp_only.xml`, preserve Foxy's
variable-size endpoint histories, force effective `ROS_LOCALHOST_ONLY=0` so Foxy does not replace
the profile, and restrict UDPv4 to `127.0.0.1`. Driver/camera/perception may remain at
`ROS_LOCALHOST_ONLY=1`; communication remains local and compatible on domain 42. An unused-domain
probe opened no `/dev/shm` files, idle GUI RSS stabilized near 70 MiB, and a bounded live Step-1
launch peaked at 71.5 MiB while all scan nodes started, GroundingDINO/SAM2 PIDs remained unchanged,
no approval/motion occurred, no OOM was logged, and every temporary GUI-owned process was removed.

The 2026-07-24 bootstrap-static update rebuilt both focused ROS packages, passed 57 focused
Foxy/Tesseract tests, 10 isolated heavy/SAM2 worker tests, and the rootless Tesseract qualification.
The broader behavioral run passed 138 tests with one skip; the only failure was the existing
repository-wide PiPER pep257 test scanning vendored environments and generated trees.

The fixed 5-percent automation restriction has since been replaced by one GUI-selected session
speed from 1 through 100 percent, default 5, applied to acquisition and the later scan. The
executor clamps only to the PiPER SDK's 1-100 percent range. The historical physical acceptance
above remains at 5 percent; higher selections are implemented and unit-tested but not yet physically
re-characterized. The supervised launch also now starts `scan_capture_node.py` in service mode.
After workflow quality accepts each settled MULTIVIEW_SCAN view, the executor requires one fresh
synchronized RGB PNG, raw depth NPY, 16-bit millimetre depth PNG, mask PNG, and YAML
intrinsics/joints/plan metadata record under `datasets/active_scan` before advancing. This capture
path passed focused tests and build. On 2026-07-29 the native GUI completed
plan `7474484cfd3ddb50` as exactly 13 collision-qualified viewpoints at 5
percent. Dataset `datasets/active_scan/scan_20260729_150616` contains 13
complete records plus scan metadata; every record has `GOOD` quality,
`CLEAR` occlusion, zero closer-depth pixels, identical RGB/depth stamps, and
the same plan/session correlation. J1 covered 56.95 degrees. The run finished
with a GUI compact-pose command, successful disable response, and verified
camera/process shutdown.

On 2026-07-29 a live 5-percent `timed_movej_v2` acquisition reached measured `SCAN_READY`, but its
7,281 tiny targets exposed the hardware-interface mismatch: PiPER SDK MoveJ consumes six joint
positions plus one aggregate speed percentage and does not consume Tesseract qdot/qddot. The
replacement `sdk_movej_targets_v1` keeps the fresh typed/hash-bound controller limits and selected
speed in schema v5, uses OMPL/ISP to establish all-six-joint goals, writes exact-zero derivative
placeholders, and adaptively collision-validates the actual direct SDK-interpolated segment.
Foxy sends one six-position arm-only target per viewpoint; only a folded-start rough acquisition
may use one separately proven bootstrap target first. Dense 0.025 rad validation samples are never
published. Each endpoint is published once; feedback must enter 0.025 rad. A 90-second absolute
target deadline or 20-second total-joint no-progress interval aborts to hold. The driver caches
unchanged `MotionCtrl_2` mode/speed and never emits `GripperCtrl` for the
six-position automation form. The packages build, the focused software suite passes 184 tests,
and both rootless collision qualifications pass with real motion false. The 2026-07-24 5-percent
run remains historical acceptance for the former executor; the new target adapter received its
supervised 13-view physical acceptance on 2026-07-29.
A live 2026-07-27 run at 100 percent hit the sustained following-error hold before acquisition could
publish `ACQUIRED`, although perception later reported a valid settled measured lock. The workflow
therefore remained `IDLE` and the old GUI kept Step 4 disabled. Explicit current-lock adoption now
recovers that exact state through a new workflow/plan/approval phase. Step 4 can start a stopped
GUI-owned worker/scan stack from direct fresh settled measured tracking, but still waits for the
workflow's full authoritative lock validation before adoption or planning. The current targeted
command passes 104 tests, including the GUI/scan transport contract; this is not evidence that
100-percent dynamics are accepted.

Work on 2026-07-20 fixed GUI enable. The repository service used a 500 ms hand-written EnableArm/gripper loop
instead of the installed SDK's 10 ms feedback-confirmed handshake. It now uses EnablePiper/DisablePiper at
10 ms and sends gripper control once after success. Three focused tests, the Foxy rebuild, and a supported live
disable/re-enable cycle passed; final feedback had all six motors enabled, normal status, ERROR-ACTIVE can0,
and zero joint-command publishers. An official 0xFC zero-offset role command was sent earlier with all motors
disabled, but the code-only fix passed without a power cycle, so it was not the cause and should be rechecked
after the next intentional reboot.

Continued acceptance measured 200.05 Hz stable feedback, normal CAN-control status, ERROR-ACTIVE can0, zero
joint-command publishers, and live TF/mode-0 FK agreement to numerical precision. Tesseract functional,
qualification, and command-free proposal-boundary checks pass. After the operator confirmed a clear workspace,
free camera cable, and emergency-stop access, a GUI-equivalent manual command moved J6 at 5 percent from
0.006018 rad to 0.026306 rad (+0.02029 rad measured), then returned it to 0.008111 rad (0.00210 rad error).
Final arm `err_code` and `motion_status` were zero, CAN remained ERROR-ACTIVE, and the temporary publisher exited.
The controller normalized the slightly negative J2 start to 0.0 during the first command. The J2
model bound and compact-start planner/proxy mismatch were subsequently resolved as described above.
The manual smoke itself remains only driver/J6 acceptance; collision qualification comes from the
separate isolated core/compact suites.

The worktree is intentionally dirty, with substantial modified and untracked tracking, timestamp,
GUI, robot-description, executor, test, calibration-metadata, and documentation work. Preserve it;
do not reset, clean, or overwrite it when resuming.

## Repository map

- `piper_ros_foxy/src/piper`: real PiPER ROS 2 driver, CAN ownership, feedback, enable service, and
  bounded joint/end-pose command translation.
- `piper_ros_foxy/src/piper_description`: mode-0 URDF, meshes, feedback-only live TF/RViz launch,
  and the separate draggable preview model.
- `piper_ros_foxy/src/piper_mobile_manipulation`: ROS messages, perception geometry, target tracking,
  scan planning, rough-coordinate target acquisition, dry-run workflow, timestamp safety, and the
  guarded viewpoint executor.
- `piper_ros_foxy/src/piper_tesseract_foxy`: command-free Foxy bridge, canonical spool contract,
  model builder, ROS-free planning worker, model manifest, and tests.
- `motion_planning/tesseract`: pinned Ubuntu 24.04 Tesseract 0.35 OCI and rootless Bubblewrap setup,
  isolation, smoke, qualification, and run scripts for workstation development.
- `L515_camera`: pinned L515 build/runtime wrappers, hand-eye calibration, GPU pipeline supervision,
  viewers, timestamp recovery, and operator launch scripts.
- `AI_perception_tests`: ROS-free GroundingDINO/SAM2 workers, offline datasets, and model tests.
- `docs/ai`: primary architecture memory. Read it before broad repository searches or edits.
- `OPERATOR_COMMANDS.md`: authoritative runtime and real-arm operating procedures.

The required architecture read order is `docs/ai/00-index.yaml`, `05-admin.yaml`,
`10-system-map.yaml`, and `30-contracts.yaml`, followed by modules, flows, guardrails, and debt.

## Runtime architecture

The normal data path is:

1. `start_piper.sh` owns CAN and starts the driver with `auto_enable=false`.
2. The driver publishes `/joint_states_single`, `/arm_status`, and `/end_pose`.
3. `display_live_robot.launch.py` publishes the feedback-driven robot TF tree without commanding it.
4. `run_hand_eye_tf.sh` combines six-joint mode-0 FK with the accepted camera-to-link6 transform.
5. `run_gpu_vision_pipeline.sh` starts the L515, timestamp watchdog, ROS/GPU filesystem bridges,
   GroundingDINO, SAM2, target cloud, and ROS geometry/tracking nodes in a supervised order.
6. GroundingDINO performs initialization and event-driven semantic refresh. SAM2 maintains live
   target and obstacle masks between refreshes.
7. Depth projection and timestamped TF produce target measurements in `base_link`; the target
   tracker filters and predicts them and publishes `TrackedTarget` and `TrackingHealth`.
8. Obstacle, landmark, quality, occlusion, and target-cloud nodes build the supervised scan inputs.
9. If tracking is absent, a separate source may publish a rough finite point in `base_link`.
   `scan_target_acquisition_node.py` acts only after an explicit start, constructs the bounded
   camera-facing five-look sweep, and requests a `ROUGH_ACQUISITION` proposal. Receipt of a hint
   never commands motion. After acquisition it starts the workflow but does not request the scan;
   the GUI owns the later explicit current-lock request. The GUI can publish the same hint and
   manage the worker/scan stack while leaving driver/camera/perception externally owned.
10. `supervised_cube_workflow_node.py` coordinates obstacle review and capture but remains dry-run and
   has no arm-command publisher.
11. The separate viewpoint executor accepts only command-free Tesseract proposals. Only an explicitly
    motion-enabled restart creates `/joint_ctrl_single`; approval additionally requires the exact
    schema-v5 sdk_movej_targets_v1 trajectory/controller-limit hash and a qualified collision model.

Foxy ROS nodes and Python 3.10+ CUDA workers are deliberately separated. GPU workers do not import
ROS and communicate through atomic filesystem spools, so they cannot command the arm.

## What is working

### Arm, robot model, and command boundary

- The PiPER driver publishes feedback at approximately 200 Hz and accepts named `joint1` through
  `joint6` commands plus the gripper. An executor-owned 200 Hz timer replaced the unmanaged
  `Node.create_rate()` loop after a 2026-07-25 live stall left CAN/endpoints active but stopped
  `/joint_states_single` and `/arm_status` delivery.
- Motor enabling is explicit; driver startup does not auto-enable by default.
- GUI, driver bounds, feedback-driven TF/RViz, and the six-joint draggable digital twin exist.
- The operator confirms that the joints, including J6, move accurately and that J6 is operationally
  safe.
- Exactly one hardware command publisher is permitted during automatic motion. The GPU pipeline,
  workflow, planners, viewers, and preview editor must remain command-free.

### Camera and timestamp safety

- The pinned camera stack uses librealsense 2.50.0 and realsense-ros 4.0.4.
- Camera startup sets and verifies RGB and depth global time after device initialization.
- The watchdog requires 15 consecutive monotonic frames within 0.5 seconds of ROS time before
  declaring the camera healthy.
- Stale, future, backwards, or missing timestamps force tracking speed to zero and independently
  invalidate scan execution.
- The documented five-minute stationary test passed with 1,490 healthy samples, zero recovery
  requests, and zero measured joint-position span. A supervised J1 sweep also retained healthy
  timestamps and tracking.
- Pipeline-owned recovery restarts the complete camera/perception stack only after fresh joint
  positions prove the arm stationary; orphan cleanup targets only validated recorded process groups.

### Perception and tracking

- GroundingDINO provides semantic target/obstacle initialization and recovery.
- SAM2 provides continuous multi-object tracking at approximately 7–13 FPS on the validated CUDA
  system.
- The target landmark has been observed normally `LOCKED`, with approximately 1–4 mm measurement
  error and 2–3 pixel projection error in the recorded validation.
- Target measurements are transformed at their image timestamps and tracked in `base_link` rather
  than gated incorrectly in the moving camera frame.
- Tracking recovery is motion-aware, serialized, bounded for normal recovery, and then retried
  slowly while absent. Low-confidence refresh uses persistence and hysteresis.
- The documented J1 motion run retained 100% target validity, 98% stability, and at least 0.599
  confidence. Settled target position returned below 1 mm from baseline after returning the arm.

### Calibration

- The deployed eye-in-hand calibration is an accepted OpenCV Park solve using 12 fitting and three
  held-out samples.
- The controller uses the modified-DH mode-0 convention; the optional two-degree mode-1 correction
  must not be introduced.
- A proposed analytical J6-frame correction failed physical testing and was removed. The original
  unadjusted transform remains deployed and matched the saved post-zero snapshots to about 10.8 mm
  and 1.37 degrees.
- J6 was confirmed operationally safe on 2026-07-17. The old pre-zero measured J6 bounds remain
  invalid historical data, and no replacement bounds or new calibration result are claimed.

### Scan workflow and current executor

- Candidate orbit generation, workspace filtering, scan quality, occlusion, synchronized RGB-D-mask
  capture, obstacle instances, and target-cloud accumulation are implemented.
- The supervised workflow is explicitly dry-run and cannot publish arm commands.
- The separate executor implements typed proposals/status, exact approval, expiry, target-drift
  rejection, low-speed commands, feedback convergence, health gating, cancellation-to-hold, settling,
  and automatic capture sequencing.
- Proposal-only mode creates no executor `/joint_ctrl_single` publisher.
- The former fixed-J6/J1-J5 planner has been removed. Tesseract is mandatory, consumes complete
  six-joint samples, and permits J6 to move normally.
- The Foxy executor retains a conservative capsule/AABB recheck as defense in depth. The Tesseract
  collision manifest was qualified on 2026-07-23 and requalified twice with identical seed-42
  output on 2026-07-24 for supervised guarded execution. On 2026-07-25 the 5 mm global margin was
  retained while named positive compact-shoulder and folded link2/link4 pair margins were qualified
  for the valid all-zero pose. The exact `[0.33, -0.14, 0.0]` cold-start acquisition regression
  selected a plan and passed 3501 adaptive samples. Conservative L515/mount/local-cable envelopes,
  bounded exact-path Bullet validation, deterministic detour regression, and the model's declared
  limitations remain authoritative.
- Rough-coordinate acquisition accepts only an atomic typed request containing a unique session ID
  and a fresh finite `PointStamped` in `base_link`. Exact duplicate retries are idempotent, changed
  target data cannot reuse the session, and the same `source_request_id` must reach the GUI before
  exact plan approval. It retains arm/camera/obstacle/collision/waypoint-progress gates, uses only the
  GUI-selected 1-100 percent SDK session speed, and never captures while searching. Semantic work must use
  a post-settle frame and exact request ID; target association requires a newer stable measured
  lock within 0.30 m of the hint.
- Worker heartbeat plus typed acquisition/multiview readiness now gate GUI startup. The scan bridge
  and reachability filter keep stable lifetime-owned subscriptions, every nonvisual scan child is
  launch-critical, and the GUI disables automation and cleans up its exact owned groups if either
  parent exits. This repairs the 2026-07-27 acquisition-service disappearance without changing
  planner geometry, motion limits, J6 freedom, or approval rules.

## Validation evidence

- The scan-motion, reprojection, workflow, obstacle-geometry, and J6 utility group ran with 28 tests
  passing on 2026-07-17.
- Tracking recovery ran with 15 tests passing; target landmark geometry ran with five tests passing.
- Five camera-watchdog tests passed before the sandbox blocked ROS node creation with
  `getifaddrs: Operation not permitted`; that interruption was an environment restriction, not an
  assertion failure. Repository records also contain prior complete unit and live watchdog results.
- Safety-critical driver, GPU pipeline, orphan cleanup, watchdog, and viewpoint-executor shell
  wrappers pass Bash syntax validation.
- A focused package build and isolated proposal-only launch previously passed, with no executor
  joint-command publisher.
- On 2026-07-21 a complete live restart first rejected exact live J2 near -0.034 rad against the
  mode-0 lower bound of 0.0 rad. After a separately supervised 5-percent normalization, the held
  collision-free start was near `[0.0101, 0.0504, -0.0504, -0.0741, 0.482, 0.0729]`. The planner
  then used seven candidates, valid foreground collision-box geometry, explicit OMPL-to-ISP output,
  and durable typed handoff to return five six-joint segments. Request
  `e16b303dfe0d528fc73d9843ea747484` reached `PROPOSAL_READY` with trajectory hash
  `2d9fb108a7b270f8e6d3b0412f2c996bb96322b9dfc673a092c5935d097a5aea`. No Tesseract command was
  published. This was historical proposal-only evidence before the 2026-07-23 collision
  qualification.
- On 2026-07-24 a live rough hint at `[0.38, -0.12, 0.0]` produced four of five workspace-safe
  camera-facing looks and reached the isolated worker. It correctly rejected exact live J2
  `-0.03164 rad` against the unchanged 0.0 rad lower limit. The replay exposed and fixed a Foxy
  logger call-site severity error in rejected-then-accepted handoff logging; focused tests and the
  package rebuild pass. The state was not clipped and the proposal-only graph retained zero
  `/joint_ctrl_single` publishers.
- The 2026-07-24 update rebuilt both Foxy packages; 87
  motion/acquisition/workflow/tracking/GUI tests and 28 Tesseract tests passed. The isolated
  Tesseract 0.35.0.6 smoke and qualification reran PASS with mode-0 FK error 2.78e-16, J6 changing
  by 0.350 rad, and the midpoint-collision detour regression passing.
- On 2026-07-24 a rough hint at `[0.38, -0.12, 0.0]` completed the guarded physical acquisition
  route at 5 percent, obtained a measured GroundingDINO/SAM2 lock, and handed off to the normal
  workflow. Exact plan `709b2b86435c9537` then completed five collision-validated physical
  viewpoints; executor diagnostics ended `IDLE` with all viewpoints reached, workflow diagnostics
  reported five accepted and five modeled views, and finishing the scan returned `scan complete`.
- Live acceptance exposed and fixed post-motion depth latching, landmark TF callback starvation,
  an overly short SAM2 measurement-age gate, target-cloud QoS incompatibility, and capture-cloud/
  status ordering. The current repeatable orbit supplies and requires exactly thirteen candidates
  spanning 55 degrees.

## Current limitations and stopping point

- Tesseract 0.35.0.6 executes in the checked-in rootless Ubuntu 24.04/Bubblewrap route despite the
  host's incompatible Focal glibc. The 2026-07-20 smoke passed KDL/Bullet plugin loading, mode-0 FK
  comparison, finite timed output, and free-J6 planning. Collision qualification passed on
  2026-07-23 and repeated identically twice after the seed-before-planner correction on 2026-07-24;
  formal SBOM/lifecycle/fuzz/crash soak and physical timed-driver characterization remain.
- A 2026-07-17 proposal found 12 workspace-safe candidates but rejected all because the nearest
  initial camera displacement was 0.380 m versus the configured 0.250 m ingress limit.
- That Cartesian ingress gate should become a secondary guard after Tesseract can plan safe intermediate
  configurations; do not simply increase it to force acceptance.
- The previous missing-SAM2/LOCKED blocker was cleared in the 2026-07-21 integrated run. CUDA
  perception, tracking health, target stability, obstacle geometry, reachability, isolated planning,
  and Foxy proposal acceptance have all been live. Exact J2 can still start slightly below the
  planning-model lower limit after a restart and must be normalized or reconciled without clipping.
  Supervised 5-percent physical execution first completed five views using the former
  stop-per-sample executor. On 2026-07-29 the position-only `sdk_movej_targets_v1` path then
  completed exact 13-view plan `7474484cfd3ddb50` through the GUI with 13 accepted full-resolution
  clouds and synchronized captures. Deliberate cancellation-to-hold acceptance still remains; J6
  and the qualified collision manifest are not blockers.
- Rough-coordinate acquisition is software-complete and physically accepted through a measured-lock
  handoff. The bounded no-lock sweep, explicit current-lock request/two-confirmation lifecycle, and
  GUI stop/compact-disable behavior all completed in the 2026-07-29 acceptance. Cold-restart and
  deliberate cancellation regressions remain.
- The accumulated target model is a PointCloud2 cloud, not TSDF or mesh reconstruction.
- Real automatic obstacle manipulation, reinforcement learning, dynamic-target following, and
  unsupervised real-arm motion remain out of scope.

## Safety invariants

- Never run two CAN owners.
- Never run a manual GUI/reset publisher and the motion-enabled automatic executor together. The
  GUI Acquire & Scan path is allowed only because it destroys its manual publisher first and
  verifies the executor is the sole publisher.
- Proposal-only execution must have no joint-command publisher.
- The executor never enables motors; the operator enables them separately.
- Real motion requires opt-in, a fresh exact plan approval, fresh arm/tracking/timestamp/obstacle
  state, fresh matching controller motion limits, a clear scene, and a `SCAN_READY` workflow.
- Default speed is 5%; the GUI and executor accept the PiPER SDK range from 1% through 100%.
  The recorded physical acceptance remains at 5%, so higher-speed dynamics are not yet qualified.
- Any stale input, arm error, tracking loss, timestamp fault, waypoint timeout/no-progress
  condition, capture failure, or cancellation must stop progression and request a current-position
  hold.
- The physical emergency stop/power procedure is authoritative.

## Recommended continuation

1. Preserve the working Foxy perception, tracking, driver, calibration, and single command-owner
   architecture.
2. Preserve the 2026-07-23 qualification report and exact model/runtime hashes; finish the formal
   SBOM, lifecycle, malformed-spool, and crash-recovery soak without weakening the qualified scope.
3. Retain proposal-only regressions for fresh, stale, wrong-frame, unreachable, occluded, and
   collision-blocked rough hints.
4. Repeat the accepted measured-lock acquisition and 13-view scan from a cold start, then complete the deliberate
   cancellation-to-hold and bounded no-lock failure sweeps.
5. Exercise the implemented frozen-scene canonical filesystem spool and verify exact model,
   calibration, limit, request, response, and trajectory hashes.
6. Keep execution in the existing guarded Foxy command boundary and retain exact approval, expiry,
   target drift, health, feedback-gated waypoint, cancellation, and settled-capture gates.
7. For every changed model/runtime, revalidate proposal-only first and then repeat supervised
   low-speed physical acceptance; do not infer acceptance from the 2026-07-24 run.
8. Inspect the saved 13-view cloud and add isolated masked TSDF reconstruction after repeatable
   13-view capture quality is demonstrated.

## Resume references

- Architecture and risk map: `docs/ai/`
- Supported commands and smoke checks: `OPERATOR_COMMANDS.md`
- Detailed supervised workflow history: `SUPERVISED_WORKFLOW_HANDOFF.md`
- Camera/perception operation: `L515_camera/README.md`
- Installation and dependency boundaries: `README.md` and `CLEAN_INSTALL.md`
