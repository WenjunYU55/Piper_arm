# Current architecture baseline

> Baseline document. For the current post-refactor ownership and the
> user-authorized 2026-08-17 mission-shutdown/runtime-gate simplification, read
> `docs/refactor/final_architecture.md` and `docs/ai/10-system-map.yaml` first.
> The Phase 5 shadow evaluator described below no longer exists.

## Scope and provenance

This document freezes the observable architecture at commit
`4945b480d8494fa840c0d2bc993c72834934a37f` on branch
`rewrite/core-integration-v2` (the same commit as `origin/main`) on
2026-08-14. It is Phase 0 documentation only: no production behavior, ROS
interface, calibration, motion limit, speed, TF convention, perception
algorithm, or Tesseract behavior was changed.

Generated trees (`piper_ros_foxy/build`, `install`, `log`), downloaded model
checkpoints/environments, runtime spools, and datasets were inspected as
artifacts but are not source-of-truth code. The primary architecture map remains
`docs/ai/`; this document is the human-readable refactor baseline.

## Repository map

| Area | Responsibility |
|---|---|
| `piper_ros_foxy/src/piper` | PiPER CAN/SDK driver, enable service, joint command input, joint/status/motion-limit feedback |
| `piper_ros_foxy/src/piper_msgs` | Driver messages and `Enable` service |
| `piper_ros_foxy/src/piper_description` | Xacro/URDF, visual and collision meshes, RViz configurations, live/preview robot-state launch |
| `piper_ros_foxy/src/piper_mobile_manipulation` | Mission orchestration, perception geometry, tracking, acquisition, scan planning/execution/capture/quality, occlusion policy, reconstruction job contract |
| `piper_ros_foxy/src/piper_tesseract_foxy` | Foxy bridge, typed spool contract, isolated Tesseract worker, collision model and qualification |
| `L515_camera` | RealSense startup/shutdown, GPU worker wrappers, hand-eye TF/calibration tools, camera viewing and watchdog recovery |
| `AI_perception_tests` | GroundingDINO/SAM2 workers, temporal tracking, offline experiments and worker tests |
| `motion_planning/tesseract` | Isolated Ubuntu 24.04/Tesseract runtime, rootless launch/build/qualification scripts |
| `reconstruction` | Offline target-mask RGB-D TSDF reconstruction and tests |
| `deployment` | System-level deployment assets |
| `piper_gui_native.py` and `piper_gui/` | Tk presentation, production `RunTargetScan` client, read-only diagnostics, manual/preview commissioning; no autonomous workflow or production process ownership |
| `piper_gui_automation.py` | Archived pre-Phase-8 pure workflow characterization; not imported by the production GUI |
| root `*.sh`/`*.py` | Operator startup, shutdown, diagnostics, calibration and qualification entry points |
| `docs/ai` | Machine-readable ownership, contracts, flows, guardrails, debt and maintenance routing |

The ROS workspace contains five packages: `piper`, `piper_description`,
`piper_msgs`, `piper_mobile_manipulation`, and `piper_tesseract_foxy`.

## ROS nodes

### Hardware, model and planning

| Node | Executable/source | Role |
|---|---|---|
| `piper_ctrl_single_node` | `piper_single_ctrl` / `piper_ctrl_single_node.py` | Sole PiPER SDK/CAN adapter; enable/disable, MoveJ input, 200 Hz feedback and controller-derived limits |
| `piper_live_robot_state_publisher` | `robot_state_publisher` | Publishes the live robot model from `/joint_states_single` |
| `piper_gui_joint_editor` | `piper_joint_preview_node.py` | Preview-only joint-state bridge |
| `tesseract_plan_bridge` | `tesseract_plan_bridge` / `bridge_node.py` | Snapshots current ROS state, validates readiness, hashes requests, exchanges typed spool jobs with the isolated worker |
| `tesseract_plan_worker` | `worker.py` in isolated runtime | OMPL/ISP planning, IK, collision validation and timing |

### Mission, acquisition and scan

| Node | Source | Role |
|---|---|---|
| `target_scan_gateway` | `target_scan_gateway_node.py` | Always-on tracked-robot-facing action/service gateway, disk spool and deferred reconstruction trigger |
| `target_scan_mission` | `target_scan_mission_node.py` plus pure `mission_engine.py` | Node owns ROS queue/action/feedback/result and durable boundaries; MissionEngine owns the existing admitted startup, retry, acquisition, scan and terminal shutdown sequence through injected operations |
| `scan_target_acquisition` | `scan_target_acquisition_node.py` | Converts one typed rough target request into centered then compact cardinal looks |
| `scan_viewpoint_planner` | `scan_viewpoint_planner_node.py` | Generates target-centered scan candidates and coverage payloads |
| `viewpoint_reachability_filter` | `viewpoint_reachability_filter_node.py` | Command-free coarse filtering; Tesseract remains authoritative |
| `scan_viewpoint_executor` | `scan_viewpoint_executor_node.py` | Sole optional `/joint_ctrl_single` publisher for scans; validates/approves/streams plans, holds, settles and invokes capture |
| `scan_capture` | `scan_capture_node.py` | Persists synchronized RGB, depth, mask, camera info, TF, joints and metadata after service-mode acceptance |
| `scan_quality` | `scan_quality_node.py` | Scores settled RGB-D/mask observations |
| `active_scan_debug_overlay` | `active_scan_debug_overlay_node.py` | Visual diagnostics only |
| `supervised_cube_workflow` | `supervised_cube_workflow_node.py` | Measured-lock/occlusion assessment and command-free manipulation proposals |

Ray-NBV missions also write observational schema-v2 evidence below
`datasets/active_scan/ray_diagnostics/<mission-id>/`. Its canonical JSON and
append-only event journal records the frozen generated pool, planner/history
and information culls, workspace/capability prequalification, bridge retries,
Tesseract outcomes, accepted captures, exact compressed target-model snapshots
and terminal result. The evidence is not consumed by planning, execution,
capture or mission safety; schema-v1 final records remain loadable and are not
expanded into invented intermediate history.

The GUI's Open Ray Review control reuses one freely resizable, ROS-free
PyQt5/VTK child through stdin JSON. Its Mission Process tab parses every
checked-in URDF visual and synchronizes ray/rank/target/achieved-pose evidence
to an event strip. Its Capability Map tab loads the committed map read-only and
does not claim IK or trajectory evidence. HTML remains a compatibility export;
its old kinematic drawing is labelled schematic. Historical datasets that
predate full rejected-ray persistence are visibly labelled partial, and the
replay never invents missing cull evidence.

This observational path is separate from the behavioral permanent-cull
feedback. The reachability filter and Tesseract bridge publish complete,
revision-bound source snapshots on private reliable transient-local
`/piper/ray_hard_culls`. The planner accepts a snapshot only when its mission,
session, frame, ray count, and canonical population SHA-256 match the frozen
pool it generated. Coarse workspace/capability rejections and explicit worker
`permanent_infeasible_ray_ids` are the only allowed sources. Matching rays are
removed before later NBV ranking; contextual IK, collision, path, visibility,
timeout, and shortlist failures remain generation-scoped. Accepted captures
alone change coverage.

The frozen ray request is target-relative and no longer uses the target
center's distance from `base_link` as a standoff cap: generation retains the
configured 0.28 m minimum, preferred band through 0.50 m, and 0.80 m maximum.
The coarse workspace filter still owns analytic interval eligibility. In
capability-map enforce mode, the sparse atlas preserves support for each
sampled standoff and derives ordered contiguous runs; the filter retains the
original requested bounds while narrowing active min/max to the first through
last supported run. The bridge carries those run records and chooses its
representative seed from a supported run. Because the outer active envelope
can span a gap between runs, it remains prequalification only: the worker and
executor still prove exact IK, joint limits, collision, path, aim, visibility,
and runtime safety for the actual selected endpoint.

### Perception and geometry

| Node/process | Source | Role |
|---|---|---|
| RealSense ROS wrapper | `L515_camera/start_l515.sh` and wrapper launch | Color, aligned depth, camera info and camera-frame TF |
| heavy worker | `AI_perception_tests/heavy_model_worker.py` | GroundingDINO language grounding plus SAM2 refinement; spool-based request/response |
| SAM2 live worker | `AI_perception_tests/sam2_live_worker.py` | Temporal multi-object mask propagation and reseeding |
| `heavy_refresh_bridge` | `heavy_refresh_bridge_node.py` | Correlates ROS RGB-D requests with heavy-worker jobs and publishes masks/status |
| `sam2_live_bridge` | `sam2_live_bridge_node.py` | ROS/spool temporal tracking bridge and tracking health |
| `motion_compensated_prompt` | `motion_compensated_prompt_node.py` | Reprojects target support during eye-in-hand motion |
| `sam2_mask_to_detection` | `mask_to_detection_node.py` | Mask-to-2D detection geometry |
| `sam2_depth_to_3d` | `depth_to_3d_node.py` | Mask/depth/camera-info synchronization and camera-frame `Target3D` |
| `sam2_target_tracker` | `target_tracker_node.py` | Timestamped base-frame target tracking and health/status |
| `target_landmark` | `target_landmark_node.py` | Stable base-frame landmark from eroded mask/depth observations |
| `object_frame_broadcaster` | `object_frame_broadcaster_node.py` | Tracked object TF frames |
| `obstacle_instance_3d` | `obstacle_instance_3d_node.py` | Heavy/SAM2 obstacle masks to typed 3D instances |
| `sam2_scan_quality` | `scan_quality_node.py` | Quality evidence for the active mask |
| `sam2_occlusion_checker` | `occlusion_checker_node.py` | Depth-layer/visibility-based occlusion state |
| `target_cloud` | `target_cloud_node.py` | Target/scene point clouds and optional refined cloud persistence |
| `camera_timestamp_watchdog` | `camera_timestamp_watchdog_node.py` | Camera clock, frame liveness and stationary recovery request |

Legacy/fake commissioning nodes remain installed: `target_handoff`,
`tf_target_transform`, `target_error`, `fake_visual_servo`,
`manipulation_target`, `safe_servo`, `manipulation_state_machine`, and
`fake_arm_interface`. They are not the autonomous scan command path.

## Launch files and composition

The workspace has the following launch files:

- Driver/model: `start_single_piper.launch.py`,
  `display_live_robot.launch.py`, `joint_preview.launch.py`.
- Manipulation/diagnostic: `fake_manipulation.launch.py`,
  `full_system.launch.py`, `full_visual_servo.launch.py`,
  `handoff_only.launch.py`, `tracking_only.launch.py`.
- Perception/scan: `gpu_geometry.launch.py`, `active_scan_debug.launch.py`,
  `active_scan_capture_debug.launch.py`,
  `supervised_cube_workflow.launch.py`,
  `supervised_viewpoint_execution.launch.py`.
- Always-on mission endpoints: `target_scan_gateway.launch.py` and
  `target_scan_mission.launch.py`.
- Planning: `tesseract_foxy.launch.py`.

`supervised_viewpoint_execution.launch.py` composes the Tesseract bridge,
reachability filter, workflow, executor, viewpoint planner, acquisition,
capture and overlay. `gpu_geometry.launch.py` composes the ROS-side perception
geometry/tracking nodes. Root and `L515_camera/*.sh` wrappers supply the camera
and two isolated GPU workers.

The autonomous coordinator starts, in dependency order:

1. `start_piper.sh` (`driver` process group, with auto-enable disabled).
2. `L515_camera/run_gpu_vision_pipeline.sh` (`vision`).
3. `L515_camera/run_hand_eye_tf.sh` (`hand_eye`).
4. `motion_planning/tesseract/run_worker.sh` (`tesseract_worker`).
5. `L515_camera/run_supervised_viewpoint_execution.sh` (`scan_stack`).

Each is started in a new process group and logged below the configured process
log root. Cleanup is reverse-order SIGINT, a five-second wait, SIGTERM, then a
three-second wait. A still-live command owner is deliberately not SIGKILLed;
shutdown becomes `NEEDS_OPERATOR`.

As of Phase 8, the GUI does not start `tesseract_worker`, `scan_stack`, camera,
perception or any other production mission child. Automatic Scan submits and
cancels `/piper/run_target_scan` and displays its feedback/result. The only GUI
child is the preview-only RViz joint editor. Direct manual controls, settled
manual disable, home-profile recording and confirmed preview mirroring remain
explicitly labelled commissioning.

## Configuration and calibration sources

Phase 9 moves the 16 coordinator defaults and 82 viewpoint-executor defaults
to `piper_mobile_manipulation/configuration.py`. Those two nodes declare and
read their unchanged ROS parameters once at startup, validate them, then use
immutable grouped configuration. The pure MissionEngine receives motion,
capture and workflow groups explicitly. See `phase9_configuration.md`.

All 424 direct `declare_parameter` call sites across 34 Python files were
inventoried at Phase 0. Parameters outside the two Phase 9 boundaries retain
their node-local ownership; deployed overrides remain concentrated in:

- `piper_mobile_manipulation/config/{camera,camera_timestamp_watchdog,fake_visual_servo,frames,manipulation,obstacle_instance_3d,occlusion_checker,safety,scan_capture,scan_execution,scan_planning,scan_quality,supervised_cube_workflow,target_error,tracking}_params.yaml`;
- `piper_tesseract_foxy/config/tesseract_bridge_params.yaml`;
- `piper_tesseract_foxy/model/{collision_model,contact_manager_plugins,piper_plugins}.yaml`;
- `piper_home_pose.json` and `piper_joint_bounds.json`;
- `piper_description/urdf/piper_description.xacro` and generated URDF;
- `L515_camera/calibration/hand_eye/session_20260808_straight_mount/calibration_result.yaml`.

The active hand-eye file is accepted eye-in-hand PARK calibration with the
datasheet/factory mechanical registration. Its contract is
`T_link6_camera_optical`; TF frame semantics are documented in
`docs/ai/30-contracts.yaml` and must not be reinterpreted during refactoring.

## Mission state model

As of 2026-08-20, a terminal operator-recovery result revokes mission
authorization and releases the exact non-command perception/planning groups.
It retains a possibly powered driver and scan executor. A later admission can
reconcile the supervisor's exact previous handles only when the driver is
absent or fresh typed feedback proves all six motors disabled.

`mission_core.MissionPhase` defines:

`LISTENING -> QUEUED -> GOAL_LATCHED -> STARTING -> PREFLIGHT ->
ENABLE_AND_HOLD -> RETURNING_HOME (startup wrist) -> RETURNING_HOME (rough
home) -> ROUGH_ACQUISITION <-> TARGET_LOCK -> OCCLUSION_PROBE ->
OCCLUSION_CLEARANCE (reserved) -> VIEW_PLANNING <-> CAPTURING ->
RETURNING_HOME -> HOLDING -> DISABLING -> STOPPING -> SUCCEEDED|FAILED|
NEEDS_OPERATOR`.

The enum does not enforce an adjacency graph: `MissionSession.transition`
rejects only transitions out of terminal phases. The orchestrator's control
flow is therefore the actual transition authority.

As of Phase 5, executor plan/runtime/hold safety is also evaluated by the pure
`SafetyEvaluator` using one immutable telemetry snapshot and explicit named
mode. This is shadow-only: legacy gates remain the sole command and transition
authority. Structured disagreements are diagnostic and cannot affect ROS
results or motion.

As of Phase 6, `MissionEngine` owns the admitted high-level mission and terminal
sequence without importing ROS. `target_scan_mission_node.py` converts and
admits action goals, forwards cancellation into a per-task application token,
adapts existing ROS/subsystem operations, publishes unchanged feedback and
converts `MissionResult` into the unchanged public/durable result. The former
Phase 5 pipeline and shutdown bodies remain renamed, non-authoritative
compatibility evidence; production `execute_cb` invokes the engine.

### Receipt, admission and queuing

1. A `RunTargetScan` goal arrives through the mission node directly or through
   the gateway spool. Goals are normalized and hashed.
2. Up to eight pending missions are admitted. The queue coalesces for one
   second, then runs the closest rough target first with arrival-order ties.
3. A repeated task ID with the same mission hash returns the cached result; a
   different hash conflicts. Only one mission may own arm resources.
4. A queued cancel returns `CANCELLED`, `retryable=true`,
   `safe_shutdown=true`, with no process or arm ownership.

### Startup and authority

1. Old runtime caches are cleared and a new owned process generation is
   required; live groups from a previous generation block admission.
2. The rough target is transformed once to `base_link`. There is no general
   static qualified-workspace gate; a target within the 0.10 m base exclusion
   radius returns `REPOSITION_REQUIRED`.
3. Driver service, coherent settled joints, camera/GPU markers, current hand-eye
   TF, Tesseract worker generation and stable acquisition readiness are proved.
4. Real motion and the configured speed profile must both be explicitly
   enabled. A mission-hash authorization is issued to the executor.
5. The driver enables all axes; the coordinator immediately commands and proves
   current-position hold.
6. Startup home is staged: current measured J6 to mission-ready J6
   (`STARTUP_WRIST`), then all six joints to rough home (`ROUGH_HOME`). Both are
   fresh, correlated, direct `RETURN_HOME` transactions.

### Target acquisition and reacquisition

The acquisition loop performs at most five distinct looks. The first is the
minimum-translation centered look from the current camera pose. Subsequent
looks are compact cardinal offsets (configured 15 degrees), replanned one at a
time from current measured arm/camera state. The maximum permitted standoff is
bounded by the current camera-to-hint distance so the camera is not sent
through or behind a close hint. Each plan is Tesseract-qualified and separately
approved. A fresh frame, GroundingDINO/SAM result, tracking lock and obstacle
scene are required. `ACQUIRED` exits immediately; otherwise a fresh look is
requested. Five misses terminate as retryable `TARGET_NOT_FOUND`.

Between scan views, a transient `LOW_CONFIDENCE`, `LOST`, or `SEARCHING` result
does not authorize motion. The arm remains held for up to 30 seconds while
SAM2/heavy perception attempts to restore a measured lock. A changed target
center invalidates approval and triggers a fresh plan, up to eight target-drift
replans.

### Occlusion assessment

After lock, `supervised_cube_workflow` requires correlated fresh perception and
3D obstacle evidence. `SCAN_READY` continues. `PLAN_READY` means beneficial
contact removal is required, but the current autonomous manipulation executor
is intentionally unqualified; the mission terminates `NEEDS_OPERATOR` rather
than contacting an obstacle. Pick/push/place is proposal-only in this baseline.

### Viewpoint planning, capture and retry

The automatic path is closed-loop next-best-view, one view per transaction:

1. The planner generates target-centered candidates over the configured
   azimuth, pitch and distance region around the latest measured target.
2. Authoritative voxel NBV scores the complete configured candidate region
   against cumulative accepted voxel coverage. A six-degree threshold rejects
   directions redundant with accepted views; it is not a maximum movement.
3. The bridge preserves information order, reserves direction diversity and
   sends at most 12 candidates to Tesseract. Every candidate uses exact current
   target aim first, with one optional hash-bound fallback no more than five
   degrees away only after exact-aim alternatives fail.
4. The executor revalidates hashes, plan age, start state, target drift,
   tracking, camera clock, joints, all-six motor state, controller limits,
   obstacle geometry, target visibility and dense path safety before motion.
5. It streams time-indexed MoveJ position targets at 20 Hz without stopping at
   every Tesseract vertex, settles, requests fresh perception, then calls the
   RGB-D capture service.
6. After settling, actual achieved camera FK is recorded independently of
   capture acceptance and final target aim must be within five degrees. Capture
   is accepted only if synchronized data, TF, valid target/mask/depth,
   `GOOD` score >= 0.65 and `CLEAR` occlusion all correlate with the settled
   view. Only persistence adds accepted coverage or reconstruction geometry.
7. A transient fresh visual rejection holds for exactly one correlated heavy
   refresh. Persistent rejection excludes the achieved pose and replans from
   its actual FK; the mission replans
   up to eight replacement views. Transport/liveness gaps receive up to ten
   bounded in-pose capture-readiness retries in the executor and do not consume
   the visual replacement budget.

The adaptive scan seeds at eight captures and never exceeds 24. Completion
requires measured feature coverage: at least two accepted views on each
reachable Y side, 120 degrees of azimuth, 25 degrees of elevation, and three
consecutive surface gains no greater than 0.02. A zero-view Tesseract result is
accepted as information or feasibility exhaustion only after the seed floor and sufficient
coverage are already proved. Otherwise it is retryable insufficient quality.

### Cancellation, failure and shutdown

Action cancellation is always accepted. The coordinator SIGINT handler and an
optional gateway heartbeat/cancel flag enter the same bounded failure path.
The normal terminal sequence is:

1. Stop active execution and prove an exact current-feedback hold.
2. Request one fresh direct current-state-to-rough-home transaction. It does
   not reverse or reuse the scan route.
3. Rotate J6 from rough home to the configured storage angle.
4. Prove a final current-position hold.
5. Call `/enable_srv` with false and prove feedback-confirmed disable.
6. Revoke mission authority and stop only mission-owned child process groups.

Dedicated home stages intentionally bypass robot self-collision checking for
the operator-configured folded pose, but retain joint limits, hashes, feedback,
timing, camera-holder external floor clearance, final hold, disable and cleanup.

If any motor axis drops while powered, automatic home is forbidden. The driver
watchdog owns all-axis disable; the coordinator waits for six-disabled feedback
and stops child processes. Failure to prove hold, home, storage wrist, disable,
or process cleanup produces `NEEDS_OPERATOR`, never a false safe-shutdown
claim. Motion-safety failures can deliberately leave the arm enabled in a
proved current-position hold for operator recovery.

### Terminal outputs and reconstruction

Terminal action outcomes are `SUCCEEDED`, `FAILED`, `CANCELLED`, `BUSY`,
`UNSUPPORTED_TARGET_PROFILE`, `NEEDS_OPERATOR`, or `REPOSITION_REQUIRED`.
Every result includes stable failure code, retryability, `safe_shutdown`,
dataset path/hash, capture count, optional mesh job ID and JSON summary.

Successful scan capture writes an immutable manifest. The gateway does not
start reconstruction at arm shutdown. After the tracked robot calls
`/piper/report_tracked_robot_homed` with matching task/mesh/manifest identity,
the gateway schedules `reconstruction/tsdf_reconstruct.py`. Status is published
on transient-local `/piper/mesh_job_status` and queried by mesh job ID. Empty
zero-capture scan folders are removed on failed missions; partial captures are
retained as failure evidence unless a stricter contract says otherwise.

## Existing tests

ROS package tests cover driver enable/watchdog/feedback, URDF/meshes, message
lint, acquisition, capture, scan motion/session/coverage, home profiles,
startup gates, occlusion policy, reconstruction job contracts, tracking,
Tesseract contracts/model/collision/worker validation and style gates. Root
tests cover GUI automation/transport, acquisition, J6/home and motion-limit
contracts. AI worker tests cover target selection, heavy-worker responses and
SAM2 temporal behavior. Reconstruction has an offline TSDF test. Exact baseline
results are in `baseline_test_results.md`.

Phase-specific additions are recorded in `phase2_typed_failures.md`,
`phase3_telemetry_store.md`, `phase4_process_supervisor.md`,
`phase5_safety_evaluator.md`, and `phase6_mission_engine.md`.
