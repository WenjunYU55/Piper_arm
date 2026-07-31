# L515 Camera Helpers

This folder contains helper files for testing the Intel RealSense L515 before target handoff is ready.

Files:

- `realsense_l515_version_notes.md`: L515-specific SDK/firmware guidance from RealSense release notes.
- `fetch_realsense_sources.sh`: clones the selected librealsense and realsense-ros source tags.
- `install_realsense_build_deps.sh`: installs system build dependencies that need sudo.
- `build_realsense_ws.sh`: applies the local L515/Foxy patch and builds the RealSense source workspace.
- `source_l515_environment.sh`: sources ROS 2 Foxy plus the local RealSense and PiPER overlays.
- `check_l515_ros.sh`: checks whether RealSense ROS, this ROS package, and camera topics are visible.
- `start_l515_camera.sh`: starts aligned RGB-D and applies the L515 Short Range preset.
- `run_heavy_refresh_bridge.sh`: snapshots heavy-refresh requests into filesystem jobs and publishes returned masks.
- `run_heavy_model_worker.sh`: runs GroundingDINO/SAM2 in the isolated Python 3.10 environment.
- `run_sam2_live_bridge.sh`: spools live RGB frames and publishes GPU SAM2 masks back into ROS.
- `run_sam2_live_worker.sh`: runs incremental SAM2.1 video propagation in the isolated CUDA environment.
- `run_gpu_vision_pipeline.sh`: starts the complete read-only CUDA vision pipeline and supervises
  stationary-only recovery from camera timestamp faults. Startup requires the watchdog node and
  health publisher to appear before perception starts, with three bounded watchdog-only attempts.
- `stop_gpu_vision_pipeline.sh`: stops only validated process groups recorded by the GPU wrapper,
  including orphaned workers left if the wrapper exits before its Ctrl+C trap completes.
- `run_camera_timestamp_watchdog.sh`: publishes typed camera clock health and may request a complete
  vision restart only after fresh arm feedback proves the arm stationary.
- `run_gpu_geometry.sh`: converts the SAM2 target mask into 2D/3D tracking and occlusion inputs.
- `run_target_cloud.sh`: accumulates the masked L515 depth into a target point cloud.
- `capture_hand_eye_sample.py`: captures a strict full-board ChArUco hand-eye sample.
- `solve_hand_eye.py`: solves and independently validates PiPER eye-in-hand calibration.
- `run_hand_eye_tf.sh`: publishes the accepted dynamic `base_link` to camera TF.
- `run_supervised_viewpoint_execution.sh`: starts typed scan proposals by default and optionally the
  separately approved guarded viewpoint executor. Tesseract is mandatory and the command-free
  bridge accepts complete all-six-joint plans; the former fixed-J6 fallback has been removed.
  The proposal wrapper calls the canonical `source_piper_foxy_environment.sh`, which clears inherited
  overlays and preflights the scan-capture import and recovery-bearing ROS schema before sourcing only
  Foxy plus the PiPER overlay. It uses a validated staggered node startup because it does not link
  RealSense and simultaneous Foxy/Fast DDS startup previously exposed arm/joint endpoints without
  delivering callbacks. It then loads the repository `fastdds_gui_udp_only.xml`, sets
  `RMW_FASTRTPS_USE_QOS_FROM_XML=0`, and forces effective `ROS_LOCALHOST_ONLY=0`. The first value
  retains Foxy's native reallocating endpoint histories for variable-size ROS services; the XML
  controls only the participant transport. This is still local-only because the XML allows only
  `127.0.0.1`; `ROS_LOCALHOST_ONLY=0` is required because Foxy's value `1` replaces the XML
  participant transports and re-enables stale shared-memory graph delivery. Missing or stale safety
  feedback remains fail-closed.
- `run_fixed_board_validation.sh`: interactively checks fixed-board repeatability across arm poses.
- `view_l515_camera.sh`: opens a simple image viewer for a camera topic.
- `view_l515_showimage.sh`: opens ROS 2 `image_tools/showimage` for a camera topic.
- `view_l515_opencv.sh`: opens a direct OpenCV viewer for a camera topic.
- `view_l515_rviz.sh`: opens RViz with color, aligned depth, and detection debug image displays.

Heavy-refresh mask topics:

```text
/piper/heavy_target_mask
/piper/heavy_obstacle_mask
/piper/candidate_movable_obstacle_mask
/piper/unsafe_obstacle_mask
/piper/sam2_target_mask
```

## Complete GPU vision pipeline

Prepare the isolated CUDA environment once:

```bash
./AI_perception_tests/groundingdino_test/setup_gpu_env.sh
```

Start the complete system with one command:

```bash
./L515_camera/run_gpu_vision_pipeline.sh
```

GroundingDINO does not run merely because the GPU pipeline started. Rough acquisition requests one
correlated target detection only after the arm reaches a viewpoint and the camera settles. After a
validated seed exists, SAM2 tracks it continuously and GroundingDINO runs again on loss, sustained
low confidence, or a separately correlated post-settle scan-capture refinement. Both
models are required to run on CUDA; the workers fail instead of silently falling back to CPU. The
rolling SAM2 state resets every eight frames using the latest masks, bounding GPU memory on the
validated RTX 3090. Live SAM2 inference defaults to 384 pixels wide and its masks are restored to the
native 640x480 RGB-D resolution with nearest-neighbour resizing. Override this with
`PIPER_SAM2_INFERENCE_WIDTH`; use `640` for native-resolution live inference.

View the output:

```bash
./L515_camera/view_l515_opencv.sh /piper/sam2_target_mask
./L515_camera/view_l515_opencv.sh /piper/sam2_obstacle_mask
./L515_camera/view_l515_opencv.sh /piper/sam2_object_ids
```

Request one full-resolution GroundingDINO/SAM2 cloud capture while the camera is stationary, then
save the accumulated L515 target cloud:

```bash
export ROS_DOMAIN_ID=42
source L515_camera/source_l515_environment.sh
ros2 topic pub --once /piper/target_cloud_request std_msgs/msg/String "{data: capture}"
ros2 topic pub --once /piper/target_cloud_request std_msgs/msg/String "{data: save}"
```

The delayed full-resolution mask is matched to its original cached RGB-D frame and eroded by one
pixel before projection, reducing background leakage. Live upscaled masks still accumulate by
default. For a refinement-only high-quality cloud, start with
`PIPER_CLOUD_ACCUMULATE_LIVE=false` and issue `capture` once at each stationary viewpoint.

Clouds are written under `datasets/target_clouds`. Points are fused in `base_link` by default and a
timestamped camera-to-base transform is mandatory. The explicit production command is:

```bash
export ROS_LOCALHOST_ONLY=0 L515_ROS_LOCALHOST_ONLY=0
PIPER_CLOUD_FRAME=base_link PIPER_CLOUD_REQUIRE_TF=true \
  ./L515_camera/run_gpu_vision_pipeline.sh
```

Status and performance are published on `/piper/sam2_tracking_status`; cloud status is published on
`/piper/target_cloud_status`. This pipeline is read-only and does not publish real arm commands.
If Ctrl+C returns but a worker remains, run `./L515_camera/stop_gpu_vision_pipeline.sh`. It validates
the atomic `/tmp/piper_vision_recovery/process_groups.txt` manifest instead of using broad `pkill`.

Camera measurement time is guarded on `/piper/camera_timestamp_health`. The watchdog samples the
small per-frame color CameraInfo header rather than deserializing full RGB frames. The state becomes
`HEALTHY` only after 15 consecutive stamps are monotonic and within 0.5 seconds of ROS time. Missing,
future, stale, or backwards stamps force SAM2 `TrackingHealth` to `CAMERA_CLOCK_INVALID` with speed
scale zero and independently block the scan executor. The watchdog never rewrites measurement stamps.
Both camera startup scripts set and verify librealsense global time on the live RGB and depth sensors
after the device opens, so the L515 hardware clock is continuously corrected into host time instead
of accumulating drift from its startup anchor. Startup fails closed if either control remains disabled.
When this wrapper owns the camera and latest-sample joint positions remain within 0.001 rad for 0.75 seconds,
it restarts camera plus the complete perception stack with 2, 5, 10, then 30 second
capped retry delays. `PIPER_REUSE_EXISTING_CAMERA=1` disables automatic camera restart because the
external camera owner must be restarted separately.
The watchdog retains and self-heals a reliable depth-1 subscription for this 200 Hz feedback; the
hand-eye TF publisher also requests reliable depth-1 latest-sample QoS. At startup hand-eye first
resolves and freezes the RealSense static `camera_link -> camera_color_optical_frame` edge, then
publishes the dynamic base-to-camera transform from current joint feedback. Missing static TF fails
closed. This prevents a reliable 200 Hz backlog from making the stationary world appear to move and
catch up in RViz.
`run_hand_eye_tf.sh` loads the repository loopback UDP-only Fast DDS profile. This avoids Foxy
retaining a visible camera graph endpoint while failing to replay the one-shot `/tf_static` sample
to a later listener. It does not change the RealSense TF publication rate or introduce duplicate
dynamic camera edges.

Monitor the gate with:

```bash
ros2 topic echo /piper/camera_timestamp_health
```

Eye-in-hand recovery is motion-aware. The base-frame tracker uses each image timestamp and does not
gate valid observations by camera-relative pixel, depth, or mask-area changes. During arm motion, a
depth-supported target mask is reprojected through timestamped TF to seed SAM2 locally. A heavy
GroundingDINO refresh is deferred until the camera settles, is single-flight, and is limited to two
normal attempts per latched loss episode. If those attempts fail, tracking remains `ABSENT` and makes
one serialized heavy retry every 30 seconds by default until valid tracking recovers. Set
`PIPER_ABSENT_RETRY_SEC` to change that slow interval; values at or below zero disable the periodic
ABSENT retry. General periodic semantic refresh and mask-area-triggered refresh remain disabled.
Separately, valid `/piper/tracked_target` confidence below 0.60 for one continuous second requests one
motion-gated, single-flight heavy refresh. It rearms only after confidence reaches 0.70. Override these
defaults with `PIPER_LOW_CONFIDENCE_REFRESH_THRESHOLD`,
`PIPER_LOW_CONFIDENCE_REFRESH_DURATION_SEC`, and `PIPER_LOW_CONFIDENCE_REFRESH_HYSTERESIS`.
Typed recovery state and the safe speed recommendation are published on
`/piper/tracking_health`; the safe servo holds unless tracking is `TRACKING` or `LOCKED`.

The semantic target query is exactly `green cube .`; mixing obstacle terms into the same
GroundingDINO caption is prohibited. A new target seed additionally requires GroundingDINO
confidence of at least 0.60, at least 15 percent calibrated-green pixels, and a bounded cube-like
box aspect ratio. The same green check applies to the refined SAM2 mask and to any tracked-mask
fallback. Replaying the final target-only prompt over 63 recorded confirmed physical detections
accepted 63/63: confidence averaged 0.8751 and the minimum was 0.8342. The absent-scene replay
scored 0.5627 and contained only 10.3 percent green pixels, so it failed both independent gates.

## Supervised automatic viewpoints

The automatic viewpoint path is separate from the read-only GPU pipeline and dry-run workflow. Its
default mode creates no `/joint_ctrl_single` publisher:

```bash
./L515_camera/run_supervised_viewpoint_execution.sh
```

It consumes the accepted eye-in-hand calibration and base-frame stationary-cube tracking to produce a
typed `/piper/scan_execution_plan`. Real motion requires the explicit environment opt-in, exact current
plan ID and exact execution trajectory hash plus `EXECUTE APPROVED SCAN` confirmation, a separately enabled arm,
clear obstacle state, settled TrackingHealth, and `SCAN_READY`. Tesseract preserves all six joints;
the fixed-J6 fallback has been removed. The GUI selects one execution speed from 1 through 100 percent,
default 5. The executor clamps only to the PiPER SDK's 1-100 percent range; recorded physical
acceptance remains at 5 percent. Since 2026-07-27 the driver publishes fresh hash-bound controller
velocity/acceleration limits and schema-v5 planning binds them with the selected speed, 100 Hz
maximum rate, and `timed_movej_v2`. The isolated worker generates and collision-validates the exact
C2 all-six-joint q/qdot/qddot/t schedule. Foxy never re-times it and streams at most one due point
per 10 ms tick without skip, burst, or feedback pause. Sustained time-aligned following error,
command-loop gaps, or changed limits abort to hold. The driver still caches unchanged mode/speed and
gripper writes. This new timing path has offline/build qualification but not yet physical acceptance.
Each accepted view triggers a full-resolution workflow capture plus one fresh synchronized raw
RGB/depth/mask/metadata record under `datasets/active_scan`. Follow the staged procedure in
`OPERATOR_COMMANDS.md`. The Tesseract route completed supervised five-view physical acceptance on
2026-07-24; the Foxy capsule/AABB recheck is still not full mesh/world collision planning.

When the cube is outside the camera view, `/scan_target_acquisition/prepare` atomically accepts one
unique session ID and one fresh finite `geometry_msgs/PointStamped` in `base_link`. It creates a
command-free set of five distinct bounded orbit poses and requests a hash-bound
`ROUGH_ACQUISITION` Tesseract proposal. An exact duplicate request is idempotent; an ID cannot be
reused for changed coordinates, and `source_request_id` is preserved through the typed plan to the
GUI. The configured 0.45 m standoff is a maximum: when the
live camera is already closer, the center look keeps its current position/radius and the
left/right/camera-up/camera-down alternatives sweep around that live camera-to-hint direction.
Camera optical +Z faces the supplied point throughout. Preparing a request never approves motion.
The Foxy bridge handoff has a bounded 10-second command-free republish/retry window, permits only
one plan call in flight, and stops after one accepted Tesseract request. Bridge/reachability
subscriptions remain stable for node lifetime; stale callbacks block planning without DDS graph
churn. The worker emits an atomic heartbeat and the bridge publishes
`/piper/tesseract_readiness`. The first plan is schema-v5 `bootstrap_static` and intentionally carries no DINO/SAM
obstacle boxes; this breaks the startup cycle when no cube/seed is yet visible. The qualified
robot/camera/mount/cable model, floor proxy, limits, feedback, arm status, camera-clock health,
exact approval, and hold behavior remain active.
The 2026-07-25 rootless qualification also retains an exact all-zero cold-start request for
`[0.33, -0.14, 0.0]`: the 0.005 m global margin remains active, named positive compact-arm pair
margins preserve the valid zero pose without allowing penetration, and the resulting acquisition
path passed 3501 adaptive collision samples.
The later exact nonzero folded-start regression uses the production 30 mm link1/link2/link5
decomposition and a trajectory-hash-bound acquisition-only recovery. It selected J3 `-0.05 rad`,
ended recovery at point 55 of 352, and passed 5608 validation samples. The Foxy executor independently
requires monotonic proxy-clearance improvement and ordinary 60 mm clearance at that boundary.
No collision pair is disabled and all six joints, including J6, remain available to normal planning.

After each approved move at the selected 1-100 percent execution speed settles, the heavy bridge waits for a new post-settle image and
correlates every GroundingDINO status by request ID and image stamp. Defaults are 10 seconds for a
fresh frame/idle worker, 60 seconds for GroundingDINO, and 10 seconds for a newer measured
`LOCKED` stable target within 0.30 m of the rough hint. Target-not-found waits up to 15 seconds for
the correlated typed scene before any next pose: zero obstacles emits an exact-stamp empty scene;
obstacle-only detections reserve target ID 1 and track IDs 2+ through SAM2/depth projection.
Missing, stale, or invalid geometry holds. `ACQUIRED` starts the supervised workflow and waits for
`SCAN_READY`, but it does not request normal tracked-target planning.
If acquisition is terminal but perception subsequently reports a fresh authoritative measured
lock, explicit **Prepare Scan from Current Lock** may adopt that lock as a new scan phase. It
can start a stopped GUI-owned worker/scan stack from fresh settled measured TrackingHealth, then
waits for the workflow to validate the full authoritative tuple before adoption. It starts/restarts
workflow assessment and waits for `SCAN_READY`; it does not convert the acquisition to success or
reuse its approval.
The workflow and GUI use the same fresh valid stable `base_link` TrackedTarget,
`TrackingHealth` TRACKING/camera-settled/nonprediction, and exact `LOCKED` status as the
authoritative Step 4 lock. `/piper/target_landmark_status` remains diagnostic and does not
independently block the current-lock transition.

The native GUI **Acquire & Scan** tab is the supported operator front end. Driver/camera/perception
stay externally owned; the GUI owns one locked Tesseract worker and one scan stack, removes its
manual command publisher, and requires the executor to be the sole command publisher. It waits up
to two seconds for Foxy's empty or `_UNKNOWN_` identity to resolve. A still-unknown sole endpoint
is accepted only after the GUI's zero-publisher prelaunch proof and while its owned stack is live
and `/scan_viewpoint_executor` is graph-present; missing evidence or another endpoint fails closed.
The first confirmation covers only the exact acquisition execution hash. With a fresh measured
lock, the operator must select **Prepare Scan from Current Lock**; the GUI waits for workflow
`SCAN_READY` (starting its owned scan stack first when needed), then the operator inspects its
correlated exact five-view plan and uses a separate
five-view confirmation. There is no reusable 15-minute
authorization. The GUI never enables motors. See `OPERATOR_COMMANDS.md`.

J6 policy was updated by operator confirmation on 2026-07-17: J6 is fully safe. The Tesseract route
treats all six joints equally, with no J6 lock or special cost, and includes camera-cable clearance as
ordinary planning-scene geometry.
The old pre-zero measured J6 bounds remain invalid history, and the rejected analytical hand-eye
correction must not be reinstated; no replacement bounds or new calibration test are claimed here.

Rootless Tesseract 0.35.0.6 and the checked-in augmented collision model passed the recorded
workstation qualification on 2026-07-23 and deterministic seed-42 requalification on 2026-07-24.
The collision-piece and compact-recovery change passed separate core and exact compact suites with
`collision_model_qualified_for_hardware: true` on 2026-07-25.
After the canonical-overlay repair later that day, a clean live proposal-only rough request for
`[0.33, -0.14, 0.0]` reached `PROPOSAL_READY` with three qualified viewpoints and 443 points while
`/joint_ctrl_single` had zero publishers. No approval or motion occurred.
Rough acquisition and five-view physical execution passed
the supervised 5-percent acceptance on 2026-07-24. Any change to collision geometry, ACM, margins,
planning, timing, or validation invalidates the recorded qualification until the full suite is
repeated.

## Eye-in-hand calibration

The deployed calibration is:

```text
calibration/hand_eye/session_20260701_local/calibration_result.yaml
```

It is accepted from 12 fitting and 3 held-out validation samples. A proposed +67.875-degree J6-frame
adjustment reproduced the saved-sample algebra but failed a physical three-pose test on 2026-07-15.
The deployed transform is therefore the original unadjusted solve: replaying those physical snapshots
with it produced about 10.8 mm maximum translation drift and 1.37 degrees maximum rotation drift.
Repeat the live fixed-board validation after restarting the TF publisher before automatic motion.
Do not use
`session_20260629_resample3`, which was rejected. Reproduce the solve from the compact committed
sample metadata:

```bash
python3 L515_camera/solve_hand_eye.py \
  L515_camera/calibration/hand_eye/session_20260701_local
```

Start the runtime TF publisher after the PiPER driver and camera:

```bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
./L515_camera/run_hand_eye_tf.sh
```

The publisher refuses any calibration whose status is not `accepted`. It reads
`/joint_states_single`, computes PiPER modified-DH mode-0 FK, and publishes dynamic
`base_link -> camera_link`; RealSense supplies the remaining static optical-frame transform.

Validate the physical chain with the fixed ChArUco board left stationary:

```bash
./L515_camera/run_fixed_board_validation.sh
```

Stop the arm at each substantially different viewpoint, press Enter to average ten strict full-board
detections, collect at least three poses, then enter `q`. The historical physical test used five poses
and measured maximum drift of 8.63 mm and 0.59 degrees against limits of 15 mm and 1.5 degrees. That
report predates the J6 zero change. The failed shifted-transform report is
`fixed_board_validation_post_j6_20260715_gui.yaml`; repeat the test with the restored deployed
transform.
Neither the TF publisher nor validator commands arm motion.

The ROS 2 package itself remains in:

```text
/home/prl/Piper_arm/piper_ros_foxy/src/piper_mobile_manipulation
```

The GPU geometry launch publishes one atomic obstacle snapshot per SAM2 frame on
`/piper/obstacle_instances_3d`. Each record contains camera and `base_link` geometry;
`scene_blocked` remains true for invalid, unsafe, unknown, or non-whitelisted objects.
Hand/person/finger are the only semantic unsafe labels. Configured pen, paper, tissue, wire, cable,
and cardboard detections are candidate-movable. They are never moved automatically and may still
stop the workflow with an instruction to clear the workspace. Invalid depth, unknown geometry, or
an untrusted obstacle scene remains fail-closed regardless of its semantic label.

That `scene_blocked` flag remains authoritative for the supervised obstacle-manipulation workflow.
The Tesseract viewpoint bridge makes a narrower planning distinction: a valid unsafe/non-whitelisted
instance is forwarded as collision-box geometry so OMPL can route around it, while missing depth,
invalid bounds, stale source data, or unavailable TF still blocks the planning snapshot. The
obstacle node may rebuild its read-only TF listener after the known Foxy/Fast-DDS callback stall;
it publishes blocked geometry until a fresh timestamped transform is available.

The geometry launch also maintains a stationary target landmark in `base_link`:

- `/piper/target_landmark` is the conservative world-frame reference point.
- `/piper/target_landmark_projection` is its predicted pixel in the current view.
- `/piper/target_landmark_status` reports agreement and whether a rescan is needed.

Meaningful viewpoint changes request a fresh full target mask. Target cloud points
are fused in `base_link`; a timestamped transform is required, so no camera-frame
fallback is used for multi-view accumulation.

`/piper/target_cloud` uses best-effort sensor QoS. The supervised workflow subscription must use
the same QoS and retain a capture cloud until the matching
`accumulating`/`full_resolution_refinement` status is accepted. Inspect the read-only state with:

```bash
ros2 service call /supervised_cube_workflow/diagnostic_state std_srvs/srv/Trigger '{}'
```

On 2026-07-24 the live rough-coordinate route obtained a fresh measured lock and the subsequent
qualified Tesseract plan completed five physical viewpoints at 5 percent. All five target clouds
were accepted and modeled. This validates the supervised guarded path only; exact approval,
collision qualification, camera-clock/tracking health, and hold-on-abort remain mandatory.

To validate a fixed marker after starting the normal perception and hand-eye TF terminals:

```bash
source /home/prl/Piper_arm/L515_camera/source_l515_environment.sh
ros2 run piper_mobile_manipulation obstacle_repeatability_validator.py \
  --ros-args -p scenario:=clear_view -p expected_label:=pen
```

At each stationary GUI-positioned viewpoint, capture one sample. Collect 5–8, then finalize:

```bash
ros2 service call /obstacle_repeatability_validator/capture_sample std_srvs/srv/Trigger '{}'
ros2 service call /obstacle_repeatability_validator/finalize std_srvs/srv/Trigger '{}'
```

Reports are written under `/tmp/piper_obstacle_validation` by default. For unsafe-object and
unknown-object scenarios, also pass `-p expect_scene_blocked:=true` and use a distinct scenario
name. Neither node publishes any motion command.

Run:

```bash
cd /home/prl/Piper_arm/L515_camera
./check_l515_ros.sh
```

Fetch source code from GitHub:

```bash
./fetch_realsense_sources.sh
```

Default source pair:

```text
librealsense v2.50.0
realsense-ros 4.0.4
```

Then, in separate terminals:

```bash
./start_l515_camera.sh
./view_l515_camera.sh
```

The camera launcher defaults to L515 visual preset `5` (Short Range), which is
intended for close targets and reduces near-field receiver saturation. If sunlight,
halogen lighting, or another strong IR source is present, try Low Ambient Light:

```bash
L515_VISUAL_PRESET=3 ./start_l515_camera.sh
```

Measure target distance from the front glass. The L515's specified minimum is
25 cm, so the preset improves stability around 30 cm but cannot make measurements
below 25 cm reliable. Prefer matte targets during diagnosis; glossy, black, or
steeply angled surfaces can still produce holes. To confirm the active setting:

```bash
ros2 param get /camera/camera depth_module.visual_preset
```

`realsense_ws/src`, `build`, `install`, and `log` are generated locally and intentionally not committed. Run `fetch_realsense_sources.sh` and `build_realsense_ws.sh` to recreate them on a new machine.

If `rqt_image_view` prints DDS deserialization errors, use the lighter viewers:

```bash
./view_l515_showimage.sh /camera/color/image_raw
./view_l515_opencv.sh /camera/color/image_raw
```

To source the same environment manually:

```bash
source /home/prl/Piper_arm/L515_camera/source_l515_environment.sh
```

For RViz:

```bash
./view_l515_rviz.sh
```
