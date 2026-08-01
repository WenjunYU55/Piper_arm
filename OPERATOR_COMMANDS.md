# Operator Commands

Command reference for running the PiPER + L515 perception stack from:

```bash
cd /home/prl/Piper_arm
```

## Headless autonomous mission

Same-computer deployment:

```bash
PIPER_MISSION_ENABLE_REAL_MOTION=1 \
PIPER_MISSION_SPEEDS_QUALIFIED=1 \
./run_target_scan_mission.sh
```

Do not set `PIPER_MISSION_SPEEDS_QUALIFIED=1` until the installed arm has
passed the staged 30% free-motion and 10% contact qualification. With that gate
false the mission may start proposal infrastructure but refuses enable and
shuts the never-enabled stack down safely.

Goals use `/piper/run_target_scan`, task type `SCAN_3D`, a unique 8–128
character task ID, the initially qualified `green_cube` profile, confidence at
least 0.60, a fresh `odom` or local `base_link` pose, and a 60–1200 second
deadline. Only one task is active. An exact duplicate replays its durable
result; changed reuse of a task ID is rejected.

Two-computer deployment:

```bash
PIPER_MISSION_ENABLE_REAL_MOTION=1 \
PIPER_MISSION_SPEEDS_QUALIFIED=1 \
PIPER_MISSION_REQUIRE_GATEWAY_HEARTBEAT=1 \
PIPER_MISSION_SPOOL_ROOT=/mnt/piper_target_scan_missions \
./run_target_scan_mission.sh
```

```bash
PIPER_TRACKED_ROBOT_ROS_DOMAIN_ID=42 \
PIPER_GATEWAY_BASE_FRAME=piper_base_link \
PIPER_MISSION_SPOOL_ROOT=/mnt/piper_target_scan_missions \
./run_target_scan_gateway.sh
```

For two computers, `/mnt/piper_target_scan_missions` must be the same secured
shared filesystem directory on both hosts, owned by the deployment account and
unavailable to untrusted users. Atomic rename must be supported. A plain local
`/tmp` path only supports same-computer deployment.

The gateway writes a 2 Hz heartbeat. Five seconds without it triggers the
hold/disable shutdown path. The tracked robot may power down only when the
result says `safe_shutdown=true`; `NEEDS_OPERATOR` means a safe hold or disable
could not be proved. The GUI's separate **Automatic Scan** tab uses the same
action: enter rough XYZ and press **Start Complete Automated Scan** once. It
does not automate the five commissioning buttons; the mission action owns the
complete lifecycle directly.

Automatic startup is strictly sequential. The listener waits up to 30 seconds
for the driver service and up to 15 seconds for two continuous seconds of
current-generation, coherent, settled joint feedback. It then waits up to 120
seconds for healthy camera timestamps plus the GroundingDINO and SAM2 CUDA
ready markers, 20 seconds for a newly stamped `base_link -> camera_link`
transform, 45 seconds for a new healthy Tesseract worker generation, and 90
seconds for typed `acquisition_ready`. Immediately before enable it again
requires two settled seconds of fresh joint feedback within a 15-second
window. A process exit, stale generation, malformed feedback, or timeout stops
the sequence with that exact blocker; later sections are not started early.

## Current live acceptance (2026-07-29)

The native GUI completed one exact 13-view `sdk_movej_targets_v1` scan at 5% from
rough coordinate `[0.25, -0.25, 0.0]`. Tesseract plan
`7474484cfd3ddb50` produced 13 collision-qualified J1-J6 targets over the
reachable 120–175 degree sector. All 13 workflow cloud refinements and all 13
synchronized RGB/depth/mask/metadata records completed under
`datasets/active_scan/scan_20260729_150616`; the records share one plan ID and
one `PROPOSAL_READY` occlusion session, all quality labels are `GOOD`, all
occlusion states are `CLEAR`, and RGB/depth stamp delta is zero.

Two live retryable blockers were fixed during acceptance. The target-cloud node
now pins the exact RGB-D frame selected by a correlated heavy-model request, so
the strict 80 ms match is not lost while inference runs. The executor permits a
transient missing obstacle sample only while the arm is stationary during
settle/capture; the bounded pre-motion runtime refresh still requires fresh
obstacle geometry before the next command. After the run, the GUI sent the
operator-selected compact target `[-0.008, 0.0, -0.010, 0.017, 0.457, 0.035]`
at 5%, received disable success, and the camera/perception, driver, TF,
Tesseract, workflow, executor, and GUI processes were stopped.

The local `main` baseline for this implementation is:

```text
5ebd8e8 Stabilize supervised 13-view scan pipeline
```

The autonomous mission changes described above are working-tree changes until
they are reviewed, committed, and pushed; do not infer remote GitHub state from
this operator document.

## Safety rules

- Do not enable real robot motion during perception, TF, or scan-proposal validation. Motion-enabled
  execution is a separate stage with explicit opt-in and exact plan approval.
- `./start_piper.sh` starts the PiPER driver and CAN interface, but should not auto-enable the arm.
- `./enable_piper.sh`, `./reset_piper.sh`, `./reset_arm.sh`, and GUI joint commands can move the real robot.
- `disable_piper.sh` is only a software disable request. Use the physical emergency stop/power procedure for emergencies.
- Keep the workspace clear before enabling the arm.
- Never run the GUI, reset scripts, or another `/joint_ctrl_single` publisher while the automatic
  viewpoint executor is motion-enabled.
- The automatic executor never enables motors. Its software cancel requests a position hold; the
  physical emergency stop/power procedure remains authoritative.

Current live status recorded 2026-07-20: GUI enable is fixed. The driver was sending the enable handshake
only every 500 ms and interleaving gripper commands; it now follows the installed official SDK pattern by
retrying feedback-confirmed `EnablePiper`/`DisablePiper` every 10 ms and sends gripper control only once after
success. Three focused tests, the Foxy rebuild, and a supported live disable/re-enable cycle passed. Final
feedback had all six motors enabled, normal status, `can0` `ERROR-ACTIVE`, and zero `/joint_ctrl_single`
publishers. An official `0xFC` zero-offset role command was sent earlier during diagnosis, but the successful
code-only retest occurred without a power cycle, so that command was not the cause; repeat read-only and
enable/disable checks after the next intentional reboot.

Historical manual acceptance on 2026-07-20 passed at 200.05 Hz with normal CAN-control status,
`can0` `ERROR-ACTIVE`, exclusive
command ownership, and live TF matching mode-0 FK to numerical precision. After explicit workspace, cable,
and emergency-stop confirmation, a GUI-equivalent J6 smoke at 5 percent moved from 0.006018 rad to 0.026306
rad and returned to 0.008111 rad with no arm fault. The controller normalized the slightly negative J2 start
to 0.0 on the first command. That run validated manual driver and J6 motion only. Tesseract was
subsequently collision-qualified on 2026-07-23, requalified with deterministic seed handling on
2026-07-24, requalified for exact all-zero rough-acquisition egress with named compact-pair margins
on 2026-07-25, and completed the supervised five-view physical
acceptance on 2026-07-24. Reconcile the J2 limit/zero behavior and the conservative Foxy 53 mm versus 60 mm
proxy mismatch rather than lowering automatic-planning thresholds.

On 2026-07-25 a live Step 2 attempt exposed a driver feedback stall: CAN and the driver endpoint
remained active while `/joint_states_single` and `/arm_status` delivered no samples. The driver now
uses an executor-owned 200 Hz timer; the former unmanaged `Node.create_rate()` loop was removed.
Startup checks must observe actual messages on both topics with an application-level subscriber;
endpoint counts and Foxy CLI `topic hz` alone are insufficient.

After the physical checks, run the passive preflight before starting ROS:

```bash
./check_piper_can.sh
```

It never reconfigures CAN or transmits a frame. It passes only when `can0` is `ERROR-ACTIVE` and
valid traffic arrives during the observation window.

## Joint 6 zero diagnosis and calibration

The J6 utility connects directly to CAN, so first stop the `start_piper.sh` driver terminal and the GUI.
Do not run two PiPER SDK/ROS driver processes against the same CAN interface.

Run a 10-second read-only diagnostic and optionally save the firmware and telemetry:

```bash
./calibrate_joint6_zero.sh --report /tmp/piper_j6_diagnostic.json
```

Persistent calibration requires an operator at the arm:

```bash
./calibrate_joint6_zero.sh --calibrate --report /tmp/piper_j6_calibration.json
```

The calibration path checks live feedback, disables only motor 6, asks the operator to align J6 using
physical neutral marks or a fixture, checks that it is stationary and fault-free, and requires the exact
phrase `SET JOINT 6 ZERO` twice before sending the SDK zero command. It does not enable any motor
and leaves J6 disabled. After verifying zero feedback, it marks the previous J6 bounds invalid without
deleting their recorded samples; the GUI and driver then use the SDK J6 fallback limit. Keep the
wrist/camera supported and the emergency stop ready.

After calibration, do not reuse the old J6 entries in `piper_joint_bounds.json`. First verify feedback
near zero, restart the arm, test slow approaches to zero from both directions, and only then record new
bounds. A reset is not part of this procedure because a PiPER reset immediately removes motor power.

Operational planning policy updated 2026-07-17: the operator has confirmed J6 is fully safe for
normal collision-aware planning and execution. The old pre-zero measured bounds remain invalid
historical data, and the failed analytical hand-eye correction remains rejected; this policy does not
claim replacement measurements or a new calibration run. The fixed-J6 planner was removed on
2026-07-24. Tesseract is mandatory, treats J1-J6 equally, uses no J6 lock or J6-specific cost, and
treats camera-cable clearance as planning-scene geometry.

## Standard environment

Use this in every manual ROS terminal:

```bash
cd /home/prl/Piper_arm
```

```bash
source source_piper_foxy_environment.sh
```

```bash
export ROS_DOMAIN_ID=42
```

```bash
export ROS_LOCALHOST_ONLY=1
```

The helper deliberately clears inherited colcon/Python/library overlay paths, then sources ROS 2
Foxy and only the canonical `piper_ros_foxy/install` workspace. It also verifies the exact
`piper_mobile_manipulation` and `piper_tesseract_foxy` prefixes, imports `scan_capture`, and checks
the recovery-bearing schema-v5 `TesseractPlan` plus `PiperMotionLimits` interfaces. Never source
`/home/prl/Piper_arm/install/setup.bash`; that obsolete generated root overlay caused the
2026-07-25 Step 2 process exit and has been removed.

Use `ROS_LOCALHOST_ONLY=1` for the driver, camera, perception, TF, RViz, and ordinary manual
single-machine checks. There is one deliberate Foxy exception: `start_gui.sh` and
`run_supervised_viewpoint_execution.sh` override it to `0` after loading
`fastdds_gui_udp_only.xml`. That XML allows only `127.0.0.1` UDP and disables builtin/shared-memory
transport, so these processes remain local-only and interoperate with the other domain-42 nodes.
Do not override those two launchers back to `1`: Foxy would replace the XML transport list and
silently re-enable the broken shared-memory discovery path.

## Copy-paste one-line startup commands

These are the preferred commands when pasting into terminals. Use one separate terminal per command.

### Complete end-to-end GUI test, in order

Run the optional Tesseract qualification once before starting the live stack. It tests only the
isolated planner/model/validator and reports `"real_arm_motion": false`; it does not start ROS,
enable the arm, open CAN, or move hardware:

```bash
cd /home/prl/Piper_arm
XDG_RUNTIME_DIR=/run/user/$(id -u) \
./motion_planning/tesseract/qualify_rootless_worker.sh
```

Terminal 1 — PiPER driver (leave running):

```bash
cd /home/prl/Piper_arm
PIPER_ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=1 ./start_piper.sh
```

Terminal 2 — feedback-driven robot TF and RViz (leave running):

```bash
cd /home/prl/Piper_arm
source L515_camera/source_l515_environment.sh
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
ros2 launch piper_description display_live_robot.launch.py
```

Terminal 3 — hand-eye TF (leave running):

```bash
cd /home/prl/Piper_arm
source L515_camera/source_l515_environment.sh
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
./L515_camera/run_hand_eye_tf.sh
```

Terminal 4 — L515, camera-clock watchdog, GroundingDINO, SAM2, depth/tracking, and target-cloud
pipeline (leave running):

```bash
cd /home/prl/Piper_arm
source L515_camera/source_l515_environment.sh
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
./L515_camera/run_gpu_vision_pipeline.sh
```

Terminal 5 — GUI:

```bash
cd /home/prl/Piper_arm
source source_piper_foxy_environment.sh
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
./start_gui.sh
```

The launcher deliberately prints `Using UDP-only Fast DDS transport for the GUI-owned process
tree.` Its effective `ROS_LOCALHOST_ONLY` is `0`, while the XML profile itself enforces loopback.
This prevents Foxy from replaying stale shared-memory graph state. On 2026-07-27, the prior path
twice drove GUI RSS to approximately 125 GiB and the kernel OOM killer terminated it; the resulting
host starvation made perception supervision restart. GDB located the allocation in the internal
`ParticipantEntitiesInfo` graph message, not Tesseract, the GPU models, or an arm topic. The repaired
live Step-1 test peaked at 71.5 MiB, kept GroundingDINO/SAM2 PIDs unchanged, started the complete
scan stack, and issued no approval or motion.

In **Acquire & Scan**, enter rough XYZ in metres in `base_link`, select an execution speed from
1–100% (use 5% for the new timed-adapter acceptance), then use **Start Tesseract Scan Stack**,
**Prepare Acquisition Plan**, and **Confirm Acquisition & Search**. Enable the arm separately with
the GUI Enable control only after the workspace/cable and exact acquisition plan are acceptable.
After measured lock and `SCAN_READY`, use **Prepare Scan from Current Lock**, inspect the fresh
13-view plan, then **Confirm 13-View Scan**. New plans build 21 candidates over the qualified
120–175 degree sector at camera pitches -45, -55, and -65 degrees. The bridge chooses a
camera-space-diverse 13-view subset, then orders that subset as a nearest-neighbour route from the
calibrated current camera position. This removes the former left/right endpoint pendulum while
retaining both horizontal and vertical capture baselines. The exact plan also contains a
collision-checked final return to the operator-recorded low-drop home pose:
`[0.000366362, 0.0, 0.0, 0.0, 0.43869236, 0.0]` rad while powered. The first operator screenshot is authoritative; its disabled raw feedback was
`[0.000366362, -0.02888726, 0.00624495, 0.0, 0.43869236, 0.0]`, with J2/J3 normalized to their representable powered limits. Shutdown must reproduce and verify the powered pose before disable. It returns
and holds only after all 13 synchronized records have been saved; it does not capture at home or
disable the arm. If return-only telemetry fails after the 13th record, the executor holds the
current feedback pose and reports the capture session complete with a return warning instead of
restarting Step 4/5. A non-safety abort at a reached viewpoint may retrace only the already
executed, separately approved acquisition/scan targets to the original loaded pose. Any obstacle,
collision, stale telemetry, hardware, joint-limit, command-progress, or emergency-stop condition
holds instead. **Cancel and Home** first holds an in-flight command, then retraces only executed
approved endpoints to the configured home, proves the final hold, disables, stops the owned
processes, and reports that the task failed and should be retried. The GUI
reuses the approval-bound bootstrap-static scene only when reversing the first rough-acquisition
segment before obstacle geometry exists; every later return still requires its normal obstacle
authority. A terminal `configured home reached` cancel is hold-only and cannot retrace away from
home. The mission launch file itself applies the loopback UDP profile, including for direct
`ros2 launch` invocation. The GUI
**Disable** control then resends the fresh current feedback as the sole owner's joint target,
requires target error at most 0.025 rad and sample motion at most 0.005 rad for one full second, and
only then calls the feedback-confirmed disable service. If that proof fails within eight seconds,
the motors remain enabled. Require `disable -> True`; never command another zero/home pose
immediately before disabling. The GUI does not enable the arm automatically.

Terminal 6 — optional monitoring:

```bash
cd /home/prl/Piper_arm
source L515_camera/source_l515_environment.sh
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
ros2 topic echo /piper/scan_execution_status
```

In additional sourced terminals, monitor `/piper/supervised_workflow_status` and
`/piper/scan_capture_status`. Before preparing either plan, verify the driver has published a valid
controller-limit sample:

```bash
ros2 topic echo --once /piper/motion_limits
```

Require `valid: true`, six J1-J6 velocity/acceleration values, and a 64-character
`limits_sha256`. Missing, stale, changed, or malformed limits block planning/execution. After
completion, inspect the raw records:

```bash
find /home/prl/Piper_arm/datasets/active_scan -type f | sort
```

Each accepted viewpoint now requires five related files: RGB PNG, raw depth NPY, 16-bit
millimetre depth PNG, mask PNG, and YAML metadata containing intrinsics, joint state, plan/view
identity, and selected speed. The accumulated target-cloud save remains a separate workflow output.

PiPER driver:

```bash
cd /home/prl/Piper_arm && PIPER_ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=1 ./start_piper.sh
```

Read-only live arm model and TF tree (safe to run beside the disabled driver and GUI):

```bash
cd /home/prl/Piper_arm
source L515_camera/source_l515_environment.sh
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
ros2 launch piper_description display_live_robot.launch.py
```

This consumes `/joint_states_single` and publishes the real `world -> base_link -> link1 ... link6`
chain. It does not publish arm commands. The former fake-joint-state, command-coupled RViz launch
was removed on 2026-07-24; do not recreate that command path.

If an RViz window is already running, publish only the robot TF/model topic into that ROS graph:

```bash
ros2 launch piper_description display_live_robot.launch.py use_rviz:=false
```

Hand-eye TF:

```bash
bash -lc 'cd /home/prl/Piper_arm && source source_piper_foxy_environment.sh && export ROS_DOMAIN_ID=42 && export ROS_LOCALHOST_ONLY=1 && ./L515_camera/run_hand_eye_tf.sh'
```

Full GPU perception pipeline:

```bash
bash -lc 'cd /home/prl/Piper_arm && source source_piper_foxy_environment.sh && export ROS_DOMAIN_ID=42 && export ROS_LOCALHOST_ONLY=1 && ./L515_camera/run_gpu_vision_pipeline.sh'
```

Full GPU perception pipeline, reusing an already-running camera:

```bash
bash -lc 'cd /home/prl/Piper_arm && source source_piper_foxy_environment.sh && export ROS_DOMAIN_ID=42 && export ROS_LOCALHOST_ONLY=1 && PIPER_REUSE_EXISTING_CAMERA=1 ./L515_camera/run_gpu_vision_pipeline.sh'
```

Active scan debug:

```bash
bash -lc 'cd /home/prl/Piper_arm && source source_piper_foxy_environment.sh && export ROS_DOMAIN_ID=42 && export ROS_LOCALHOST_ONLY=1 && ros2 launch piper_mobile_manipulation active_scan_debug.launch.py'
```

Supervised automatic-scan proposal, with no joint-command publisher:

```bash
bash -lc 'cd /home/prl/Piper_arm && export ROS_DOMAIN_ID=42 && export ROS_LOCALHOST_ONLY=1 && ./L515_camera/run_supervised_viewpoint_execution.sh'
```

Open a sourced shell for manual ROS checks:

```bash
bash -lc 'cd /home/prl/Piper_arm && source source_piper_foxy_environment.sh && export ROS_DOMAIN_ID=42 && export ROS_LOCALHOST_ONLY=1 && exec bash'
```

## Clean reset

Stop running terminals with `Ctrl+C`, then use one sourced terminal.

Reset ROS CLI discovery:

```bash
ros2 daemon stop
```

Check old processes:

```bash
pgrep -af 'realsense2_camera|rs_launch.py|run_gpu_vision_pipeline|run_heavy_model_worker|heavy_model_worker.py|run_sam2_live_worker|sam2_live_worker.py|run_heavy_refresh_bridge|sam2_live_bridge_node|target_tracker_node|scan_viewpoint_planner|viewpoint_reachability_filter|scan_viewpoint_executor|supervised_cube_workflow|active_scan_debug'
```

If Terminal 4 Ctrl+C stops the camera but leaves its GroundingDINO/SAM2 or geometry children, use the
recorded process-group cleanup instead of a broad `pkill`:

```bash
cd /home/prl/Piper_arm
./L515_camera/stop_gpu_vision_pipeline.sh
```

This does not target the PiPER driver, GUI, hand-eye TF, RViz, or separately launched scan executor.

If old vision/planning processes remain, stop their terminals or kill the specific stale process. Avoid killing PiPER unless you intend to restart the driver.

Archive stale perception queues:

```bash
mkdir -p /tmp/piper_reset_backup
```

```bash
mv /tmp/piper_heavy_refresh /tmp/piper_reset_backup/piper_heavy_refresh_$(date +%s) 2>/dev/null || true
```

```bash
mv /tmp/piper_sam2_live /tmp/piper_reset_backup/piper_sam2_live_$(date +%s) 2>/dev/null || true
```

Restart ROS CLI daemon:

```bash
ros2 daemon start
```

## Clean rebuild after reverting code

Use this if ROS nodes crash with message type-support errors such as:

```text
undefined symbol: piper_mobile_manipulation__msg__scene_object__convert_to_py
UnsupportedTypeSupport: Could not import 'rosidl_typesupport_c'
```

That means the source code and generated ROS build/install files do not match. Stop the relevant ROS nodes, then run:

```bash
cd /home/prl/Piper_arm/piper_ros_foxy
```

```bash
rm -rf build/piper_mobile_manipulation install/piper_mobile_manipulation log
```

```bash
source /opt/ros/foxy/setup.bash
```

```bash
colcon build --packages-select piper_mobile_manipulation --symlink-install
```

Then open new terminals and source the environment again before starting nodes.

## Recommended startup sequence

Use separate terminals.

### Terminal 1 — PiPER driver

```bash
cd /home/prl/Piper_arm
```

```bash
source source_piper_foxy_environment.sh
```

```bash
export ROS_DOMAIN_ID=42
```

```bash
export ROS_LOCALHOST_ONLY=1
```

```bash
PIPER_ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=1 ./start_piper.sh
```

Expected:

```text
auto_enable is False
```

Do not run `enable_piper.sh` for dry-run perception validation.

### Terminal 2 — hand-eye TF

```bash
cd /home/prl/Piper_arm
```

```bash
source source_piper_foxy_environment.sh
```

```bash
export ROS_DOMAIN_ID=42
```

```bash
export ROS_LOCALHOST_ONLY=1
```

```bash
./L515_camera/run_hand_eye_tf.sh
```

Expected:

```text
Publishing base_link -> camera_link
```

The launcher selects the repository loopback UDP-only Fast DDS profile. This repairs the Foxy
failure mode where `/camera/camera` remains visible in the graph but a later hand-eye process does
not receive its retained `/tf_static` sample. It does not alter RealSense's original 0 Hz static-TF
mode or publish duplicate dynamic camera edges.

The final optical-frame check is still `base_link -> camera_color_optical_frame` after the camera is running, because RealSense provides the static camera-frame-to-optical-frame TF.

### Terminal 3 — full GPU perception pipeline

```bash
cd /home/prl/Piper_arm
```

```bash
source source_piper_foxy_environment.sh
```

```bash
export ROS_DOMAIN_ID=42
```

```bash
export ROS_LOCALHOST_ONLY=1
```

```bash
./L515_camera/run_gpu_vision_pipeline.sh
```

This starts:

```text
L515 camera if no reusable camera is active
camera_timestamp_watchdog
heavy_refresh_bridge_node
heavy_model_worker with PIPER_HEAVY_DEVICE=cuda
sam2_live_worker with PIPER_SAM2_DEVICE=cuda
sam2_live_bridge_node
GPU geometry / depth-to-3D nodes
target cloud node
```

Do not also start `run_heavy_model_worker.sh` manually while this is running.

Before trusting tracking or starting a scan proposal, verify the typed camera clock gate:

```bash
ros2 topic echo /piper/camera_timestamp_health
```

Expected after startup is `state: HEALTHY`, `healthy: true`, an absolute `offset_sec` no greater than
0.5, and at least 15 consecutive healthy frames. Any missing, stale, future, or backwards image stamp
forces `CAMERA_CLOCK_INVALID`, tracking speed zero, and scan rejection. The watchdog never changes
camera timestamps. If this pipeline owns the camera, it requests a complete camera/perception restart
only after latest-sample joint positions remain within 0.001 rad for 0.75 seconds, using 2/5/10/30-second
capped backoff. This avoids false motion from noisy at-rest PiPER velocity telemetry. If
the arm is moving or feedback is unknown, recovery waits. With `PIPER_REUSE_EXISTING_CAMERA=1`, restart
the external camera owner manually.

If the camera is already running and publishing, reuse it:

```bash
PIPER_REUSE_EXISTING_CAMERA=1 ./L515_camera/run_gpu_vision_pipeline.sh
```

### Terminal 4 — active scan debug

```bash
cd /home/prl/Piper_arm
```

```bash
source source_piper_foxy_environment.sh
```

```bash
export ROS_DOMAIN_ID=42
```

```bash
export ROS_LOCALHOST_ONLY=1
```

```bash
ros2 launch piper_mobile_manipulation active_scan_debug.launch.py
```

This is dry-run/debug. It does not enable real robot motion.

## Camera-first debugging

If tracking does not work, verify the camera before debugging SAM2 or heavy refresh.

Check publishers:

```bash
ros2 topic info /camera/color/image_raw --verbose
```

```bash
ros2 topic info /camera/aligned_depth_to_color/image_raw --verbose
```

```bash
ros2 topic info /camera/color/camera_info --verbose
```

Expected:

```text
Publisher count: 1
```

Check image rate:

```bash
ros2 topic hz /camera/color/image_raw
```

If publisher count is `0` while RealSense processes exist, stop the GPU pipeline and start only the camera:

```bash
./L515_camera/start_l515_camera.sh
```

Then check:

```bash
ros2 topic hz /camera/color/image_raw
```

If camera works alone, restart the full pipeline with camera reuse:

```bash
PIPER_REUSE_EXISTING_CAMERA=1 ./L515_camera/run_gpu_vision_pipeline.sh
```

## Verification commands

Check live nodes:

```bash
ros2 node list | grep -E 'camera|heavy|sam2|target|scan|reach'
```

Expected core vision nodes:

```text
/camera/camera
/heavy_refresh_bridge_node
/sam2_live_bridge_node
/sam2_target_tracker
/object_frame_broadcaster
/sam2_depth_to_3d
/target_cloud_node
```

Check camera-to-base TF:

```bash
ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame
```

Check heavy refresh:

```bash
ros2 topic echo /piper/heavy_refresh_status
```

Loss recovery makes two normal heavy attempts, then changes to `ABSENT` and retries once every 30
seconds while the camera is settled. Override the slow interval before starting the GPU pipeline:

Valid tracked-target confidence below 0.60 continuously for one second also requests one serialized
heavy refresh while settled. It rearms only after confidence reaches 0.70; transient dips do not reset
SAM2. The threshold can be changed with `PIPER_LOW_CONFIDENCE_REFRESH_THRESHOLD`.

This 0.60 value is the downstream tracked/depth confidence gate. New semantic seeds independently
require GroundingDINO confidence of at least 0.60. Replaying the final target-only prompt over 63
recorded confirmed-cube acquisition/capture requests accepted all 63; the mean was 0.8751 and the
minimum was 0.8342. The absent-scene replay was 0.5627 with only 10.3 percent green pixels.
GroundingDINO uses the single target-only caption `green cube .`, then validates green pixel
fraction, cube-like box proportions, and the refined SAM2 mask. Pipeline startup does not issue an
uncorrelated initial request.

```bash
export PIPER_ABSENT_RETRY_SEC=30.0
```

The slow retry remains single-flight. Do not reduce this interval while diagnosing GPU load or a
heavy-refresh storm.

Check SAM2 status:

```bash
ros2 topic echo /piper/sam2_tracking_status
```

Check target 3D:

```bash
ros2 topic echo /piper/target_3d
```

Check filtered/predicted tracker output:

```bash
ros2 topic echo /piper/tracked_target
```

Check persistent object TF:

```bash
ros2 run tf2_ros tf2_echo base_link tracked_object_frame
```

Check predicted object TF:

```bash
ros2 run tf2_ros tf2_echo base_link predicted_object_frame
```

Check target cloud:

```bash
ros2 topic echo /piper/target_cloud_status
```

Check active scan outputs:

```bash
ros2 topic echo /piper/scan_viewpoints
```

```bash
ros2 topic echo /piper/reachable_scan_viewpoints
```

Note: in the reverted current GitHub version, the newer `/piper/target/raw_base`, `/piper/target/filtered_base`, `/piper/target/predicted_base`, and `tracked_object_frame` outputs from `fa32da6` are not active.

## Forcing a heavy refresh

Use this only after camera topics are publishing.

```bash
ros2 topic pub --once /piper/heavy_refresh_request std_msgs/msg/String "{data: '{\"request_id\":\"manual_initial_cube\",\"reason\":\"manual_initial_cube\",\"tracking\":{\"tracking_confidence\":0.0},\"dry_run\":true,\"real_arm_motion\":false}'}"
```

Then monitor:

```bash
ros2 topic echo /piper/heavy_refresh_status
```

Good signs:

```text
queued
published
sam2_seed_queued
```

Check output masks:

```bash
ros2 topic echo /piper/heavy_target_mask
```

```bash
ros2 topic echo /piper/sam2_target_mask
```

## Viewing masks and point clouds

View RGB:

```bash
./L515_camera/view_l515_opencv.sh /camera/color/image_raw
```

View SAM2 target mask:

```bash
./L515_camera/view_l515_opencv.sh /piper/sam2_target_mask
```

View heavy target mask:

```bash
./L515_camera/view_l515_opencv.sh /piper/heavy_target_mask
```

Open RViz:

```bash
./L515_camera/view_l515_rviz.sh
```

Use fixed frame:

```text
base_link
```

Useful displays:

```text
/piper/target_cloud
/piper/target_landmark
/piper/supervised_workflow_markers
TF
```

## Target cloud capture

At each stationary viewpoint:

```bash
ros2 topic pub --once /piper/target_cloud_request std_msgs/msg/String "{data: capture}"
```

Check status:

```bash
ros2 topic echo /piper/target_cloud_status
```

Save:

```bash
ros2 topic pub --once /piper/target_cloud_request std_msgs/msg/String "{data: save}"
```

Clear:

```bash
ros2 topic pub --once /piper/target_cloud_request std_msgs/msg/String "{data: clear}"
```

Saved PLY files are written under:

```text
datasets/target_clouds
```

## Supervised cube workflow

This coordinator is dry-run only. The operator performs any proposed movement manually.

Start after PiPER, hand-eye TF, and GPU perception are stable:

```bash
./L515_camera/run_supervised_cube_workflow.sh
```

Monitor:

```bash
ros2 topic echo --full-length /piper/supervised_workflow_status
```

```bash
ros2 topic echo --full-length /piper/removal_plan
```

Start workflow:

```bash
ros2 service call /supervised_cube_workflow/start std_srvs/srv/Trigger '{}'
```

Approve a dry-run proposal after review:

```bash
ros2 service call /supervised_cube_workflow/approve_plan std_srvs/srv/Trigger '{}'
```

Confirm manual action complete:

```bash
ros2 service call /supervised_cube_workflow/confirm_action_complete std_srvs/srv/Trigger '{}'
```

Capture a scan view:

```bash
ros2 service call /supervised_cube_workflow/capture_view std_srvs/srv/Trigger '{}'
```

Finish:

```bash
ros2 service call /supervised_cube_workflow/finish_scan std_srvs/srv/Trigger '{}'
```

Abort:

```bash
ros2 service call /supervised_cube_workflow/abort std_srvs/srv/Trigger '{}'
```

## Supervised automatic scan viewpoints

This is a separate executor; the supervised cube coordinator above remains dry-run and never
publishes arm commands. Tesseract is the only planning backend and preserves complete J1-J6 paths;
the former fixed-J6/J1-J5 numerical IK fallback has been removed. The Foxy executor independently
rechecks limits and a conservative capsule/AABB proxy, then gates motion on fresh arm, tracking,
target, obstacle, and workflow state. These checks do not replace the qualified Tesseract collision
model or a physically present operator. J6 moves normally under the same available limits and
collision checks as J1-J5, while camera-cable clearance belongs in scene geometry.

Complete the disabled-arm TF, static-cube perception, and small supervised J1/J2 checks first. Keep
the cube stationary, clear all other objects, stop the GUI/reset tools, and keep the physical emergency
stop reachable.

### Proposal-only validation

Start the proposal stack after the driver, hand-eye TF, and GPU pipeline are healthy:

```bash
./L515_camera/run_supervised_viewpoint_execution.sh
```

This wrapper calls `source_piper_foxy_environment.sh`, which clears inherited overlays, sources only
ROS 2 Foxy and `piper_ros_foxy/install`, and preflights exact package/import/message identity. It
does not source the RealSense overlay because these proposal nodes consume camera-derived messages
without linking RealSense packages. Its internal nodes start in a staggered order to preserve live
callback delivery on this Foxy/Fast DDS host. Do not collapse the timers or add overlays without
repeating the contaminated-shell preflight plus live arm-status, joint-feedback, and
zero-command-publisher checks.

Start the workflow and wait for `SCAN_READY`:

```bash
ros2 service call /supervised_cube_workflow/start std_srvs/srv/Trigger '{}'
```

Monitor the typed proposal and status:

```bash
ros2 topic echo /piper/scan_execution_plan
```

```bash
ros2 topic echo /piper/scan_execution_status
```

With the PiPER driver running, `/joint_ctrl_single` will exist because the driver subscribes to it.
Confirm proposal-only mode shows no executor publisher:

```bash
ros2 topic info /joint_ctrl_single --verbose
```

The plan must say `valid: true`, `dry_run: true`, `real_arm_motion: false`, and identify
`planner_backend: tesseract`. Every proposal must list bounded six-joint trajectory points and a
64-character `trajectory_sha256`; J6 may move normally. Stop this
proposal-only launch with `Ctrl+C` before any physical acceptance launch.

### Tesseract workstation proposal setup

The verified workstation route does not require Podman or administrator access. It downloads the
Canonical Ubuntu 24.04 root filesystem, verifies its pinned SHA-256, installs the pinned Python 3.10
and Tesseract 0.35.0.6 cohort under the ignored `.runtime` directory, and runs it with Bubblewrap:

```bash
./motion_planning/tesseract/setup_rootless_worker.sh
./motion_planning/tesseract/qualify_rootless_worker.sh
./motion_planning/tesseract/run_worker.sh
```

The latest 2026-07-27 qualification passed model load, KDL plugin discovery, a mode-0 FK cross-check
with maximum matrix error `2.78e-16`, and a 137-point smooth timed proposal whose J6 changed by `0.35 rad`.
The worker now applies canonical OMPL seed 42 before constructing planners; two complete runs
returned identical detour and trajectory metrics without weakening collision margins.
The planning process has a fresh network namespace, minimal synthetic `/dev`, no ROS/DDS, and no
host CAN, camera, or GPU access. Podman remains an optional equivalent packaging route.
When started from the GUI, `run_worker.sh` remains as Bubblewrap's supervising parent so
`--die-with-parent` follows the GUI-owned process group rather than the short-lived GUI startup
thread. An immediate `tesseract_worker exited -9` status indicates this lifecycle contract is not
present in the running checkout.

In another terminal, start the mandatory-Tesseract Foxy proposal stack and explicitly request a frozen plan:

```bash
./L515_camera/run_supervised_viewpoint_execution.sh
```

```bash
ros2 service call /tesseract_plan_bridge/request_plan \
  piper_mobile_manipulation/srv/RequestTesseractPlan \
  "{force_refresh: false}"
```

Monitor `/piper/tesseract_plan_status`, `/piper/tesseract_plan`, and
`/piper/scan_execution_plan`. The checked-in model became hardware-qualified on 2026-07-23 and was
requalified twice with identical seed-42 output on 2026-07-24. On 2026-07-25 an exact all-zero
rough-acquisition request for `[0.33, -0.14, 0.0]` exposed two nonpenetrating compact-arm
clearances below the generic margin. The 0.005 m global margin was retained, named positive
base/link2 and link2/link4 pair margins with conservative motion bounds were qualified, and the
complete suite passed with a 51-point/3501-sample zero-start acquisition regression plus the
deterministic OMPL-detour regression. Any model, geometry, ACM, margin,
planner, timing, or validator change invalidates that qualification until the complete qualification
suite passes again. Supervised physical acceptance completed on 2026-07-24: rough-coordinate
acquisition obtained a fresh measured lock and a qualified Tesseract plan completed five viewpoints
at 5 percent, with all five capture/model handoffs succeeding. Continue to use the same exact
approval and safety gates.

On 2026-07-25 the nonzero folded start that previously failed was also qualified. The planning
model now uses hash-verified overlapping 30 mm collision pieces for link1, link2, and link5 instead
of convexifying each complete concave link as one hull. For only the first `bootstrap_static`
`ROUGH_ACQUISITION` segment, Tesseract may include a bounded, monotonically improving recovery
prefix inside the exact trajectory hash. The exact recorded start and target
`[0.33, -0.14, 0.0]` produced J3 `-0.05 rad`, boundary point 55, 352 total points, and 5608
validation samples. Foxy independently verifies that its proxy reaches the normal 60 mm
threshold at the boundary. This does not disable any collision pair and does not constrain J6.

The current worker contract is exactly `OMPL_ISP`: explicit multi-seed six-joint look-at IK,
RRTConnect/OMPL, ISP geometric-knot generation, bounded SDK MoveJ waypoint subdivision, and
adaptive exact-polyline revalidation. The earlier
`OMPL_TrajOpt_ISP` label is rejected because the high-level freespace composition aborted during
live qualification after raw OMPL had succeeded. `/piper/tesseract_plan` is reliable
transient-local depth 1 so Foxy discovery cannot silently lose the one-shot proposal. When `ros2
topic echo` itself hits the known Fast-DDS deserialization fault, query the read-only executor state:

```bash
ros2 service call /scan_viewpoint_executor/diagnostic_state std_srvs/srv/Trigger '{}'
```

The workstation planner emits thirteen candidates across the qualified 55-degree sector and
requires all thirteen synchronized captures. Within one workflow session, accepted camera poses
are remembered and a retry plans only the remaining, non-duplicate views; the memory is cleared
after workflow completion. Valid unsafe/non-movable obstacle instances are sent to
Tesseract as collision boxes; only missing or invalid obstacle geometry blocks request creation.
The obstacle geometry node rebuilds a stalled read-only TF listener at a bounded rate and remains
blocked until fresh `base_link -> camera_color_optical_frame` data arrives.

The 2026-07-21 live restart produced 11 of 18 workspace-safe candidates and carried two canonical
requests through this service and worker boundary. The worker correctly rejected the exact live
start because J2 was approximately `-0.034 rad`, below the then-current stale planning lower limit
of `0.0 rad`.
The 2026-07-24 rough-coordinate replay at `[0.38, -0.12, 0.0]` likewise produced four of five
workspace-safe looks and reached the worker, which rejected exact live J2 `-0.03164 rad`. The
rejected-then-accepted Foxy handoff logger bug found during that replay is fixed; the request state
was not clipped or altered and `/joint_ctrl_single` remained at zero publishers.
On 2026-07-25 the production Xacro and planning constant were aligned to the existing valid
controller-coordinate J2 lower bound `-0.044796192 rad`, measured from the
enabled compact pose on 2026-07-30.
If `PLANNING_FAILED: start_state joint2 is outside limits` now appears, do not clip the snapshot:
verify that the rebuilt Xacro/worker request carries that bound and that feedback is genuinely
within it. A hard-limit-valid start can still be collision-invalid. Never disable a nonadjacent
pair or clip the snapshot. For the qualified folded start, inspect the typed recovery metadata and
require it only on acquisition segment zero; otherwise treat the proposal as invalid.

Later on 2026-07-21 the arm was held at an in-limit collision-free start, the pipeline was corrected
to OMPL then ISP, and final clean-topology request `e16b303dfe0d528fc73d9843ea747484` selected five of seven
six-joint viewpoints around the tracked cube and a detected foreground collision box. Foxy reached
`PROPOSAL_READY` with trajectory hash
`2d9fb108a7b270f8e6d3b0412f2c996bb96322b9dfc673a092c5935d097a5aea`; proposal mode still showed
zero `/joint_ctrl_single` publishers. This is proposal acceptance only; no Tesseract trajectory moved
the arm during that run.

The later 2026-07-24 physical acceptance used exact plan `709b2b86435c9537`, worker source trajectory hash
`eadcfd404cf53e8b583202c89dce9adfe33beb01134080916e3118d4e8967311`, and 5 percent speed. It
completed all five physical viewpoints. Read-only executor diagnostics ended `IDLE` with reason
`all approved viewpoints reached and captured`; workflow diagnostics reported five accepted and
five modeled views, and `finish_scan` returned `scan complete`. That run used the former
stop-per-sample executor; it is historical collision/workflow evidence, not physical acceptance of
the 2026-07-27 timed command adapter.

### Rough-coordinate acquisition

Use this when another system knows an approximate cube position but the camera does not yet see it.
The normal operator path is now the GUI **Acquire & Scan** tab:

> **Active live-validation item (2026-07-28):** a live Step 2 call timed out twice waiting for
> `/scan_target_acquisition/prepare`, so Step 3 never enabled. A real GUI-to-service regression
> reproduced Fast DDS rejecting an 84-byte service sample against a 55-byte fixed history when
> XML endpoint QoS was forced. The launchers now retain Foxy's native reallocating endpoint QoS;
> the same command-free round trip passes in software. Repeat the full external-stack GUI smoke
> before considering the live issue resolved.

1. Start the PiPER driver, camera, and GPU perception using their normal separate terminals.
2. Open `./start_gui.sh`, enter rough X/Y/Z in metres in `base_link`, and select
   **1. Start Tesseract Scan Stack**.
3. Select an execution speed from 1 through 100 percent (default 5). The GUI removes its manual command
   publisher and starts one isolated worker plus one Tesseract scan stack at that speed. It refuses
   another automation stack or command publisher. The selected value is bound into the planner
   request together with fresh controller velocity/acceleration limits; neither the bridge nor
   executor invents replacement dynamics. The recorded physical acceptance remains at 5 percent;
   higher settings require deliberate dynamic characterization.
   Step 1 stays pending for up to 25 seconds while it verifies both owned parent processes, every
   critical scan node and required service, a fresh ready worker heartbeat, acquisition readiness,
   and sole executor command ownership. The GUI displays the remaining blockers. Do not continue
   if it times out: any critical child exit now shuts down the scan launch and disables automation.
4. Select **2. Prepare Acquisition Plan**. Review the displayed plan kind, target, pose count,
   plan ID, hash, tracking, GroundingDINO, and workflow status.
   Step 2 remains disabled until the startup/readiness contract is fresh. It sends the XYZ and one
   unique `acq-*` session ID together through the typed prepare service. GUI service responses use
   a re-entrant two-thread executor. Each call has an 8-second deadline and a fresh client endpoint;
   if the first call times out, the GUI retires that endpoint and retries only the exact request
   once. After a second timeout Step 2 becomes clickable again and preserves the same immutable
   session/XYZ for the operator retry. Do not change the coordinates without **Cancel and Home**.
   The correlated acquisition plan must arrive within 185 seconds. Late acknowledgements, old
   stack generations, and plans from another session cannot enable Step 3.
   Foxy can retain the executor publisher as `_UNKNOWN_` in the GUI-local graph cache even after
   independent graph tools resolve it. The GUI waits up to two seconds, then accepts a sole unknown
   endpoint only when it previously proved zero publishers, still owns the live stack, and sees
   `/scan_viewpoint_executor`; missing evidence or an additional endpoint is rejected.
   A failed proposal must remain labelled `ROUGH_ACQUISITION`; if it says `MULTIVIEW_SCAN`, the
   executor is stale and the GUI-owned scan stack must be restarted after rebuilding/sourcing.
5. Enable the arm separately only after the plan and physical workspace/cable are acceptable.
6. Select **3. Confirm Acquisition & Search**. This first dialog authorizes only the displayed
   target-matching acquisition plan and its exact execution hash. It does not authorize a scan and
   the GUI does not enable the arm.
7. Normally, wait for `ACQUIRED` and a fresh measured `TRACKING`/`LOCKED` state.
   `/supervised_cube_workflow/diagnostic_state` must report `measured_lock_ready: true`; this is
   derived from the same stable `TrackedTarget`, `TrackingHealth`, and exact `LOCKED` evidence used
   by acquisition. The older target-landmark state is diagnostic and no longer independently
   blocks this handoff.
   Then select **4. Prepare Scan from Current Lock**. If acquisition previously terminated but the
   camera still reports fresh settled measured tracking, this explicit action remains available even
   if the GUI-owned worker/scan stack was stopped. It first starts that stack, waits for the
   workflow's authoritative measured-lock assessment, then adopts only the current lock,
   starts/restarts workflow assessment, and waits for `SCAN_READY`; it does not change the failed
   acquisition result or reuse its approval. This is the only action that makes the fresh normal
   scan request; acquisition never auto-requests it. Step 4 uses explicit phases and a fresh
   diagnostic-service response, not cached workflow status. Workflow assessment has a 15-second
   deadline, request queueing has 12 seconds, and the correlated result has 185 seconds. Step 4
   requires `multiview_ready`; acquisition readiness cannot substitute. Its `multiview_blockers`
   are displayed verbatim. `PLAN_READY` means movable clutter was detected: clear the workspace and
   retry. The GUI does not expose the removal workflow.
8. Inspect the correlated plan and require exactly 13 collision-qualified views. Select
   **5. Confirm 13-View Scan** to authorize only that displayed plan ID and execution hash.
   There is no reusable 15-minute approval.
9. Use **Cancel and Home** for any concern. It returns over executed approved endpoints, proves
   the final home hold, disables, and stops the managed stack. If fresh motion-safety evidence
   blocks the return, it holds enabled and reports that operator attention is required. Manual
   **Disable** remains available; it explicitly sends the
   exact fresh feedback pose, proves it settled for one second, and only then requests motor
   disable. Require `disable -> True`. **Stop Scan
   Stack** cancels and stops only GUI-owned worker/scan processes; the driver, camera, and perception
   remain running.

The **Automatic Scan** tab isolates its one-button mission from all delayed callbacks and retry
timers owned by the manual Acquire & Scan tab. After `ACQUIRED`, it waits for one continuous second
of fresh `multiview_ready` evidence (up to 30 seconds) before queuing the single plan request. A
temporary tracker reacquisition at an already approved viewpoint does not discard the scan: capture
still requires a stationary arm and healthy synchronized camera clock. Normal completion or a
non-safety abort returns over the hash-bound/already-executed route, allows up to 30 seconds for the
saved-home position proof, holds for one second, disables, and then stops every mission-owned process.

Manual ROS input remains available for integration testing. The coordinate must be a fresh finite
`geometry_msgs/PointStamped` in `base_link`, submitted atomically with a unique session ID through
`PrepareAcquisition`. Calling this command-free service does not approve or move the arm. It creates
five distinct positions on the bounded orbit—center, left, right, camera-up, and camera-down—and
aims camera optical +Z at the coordinate from every pose:

```bash
STAMP_SEC="$(date +%s)"
SESSION_ID="acq-manual-${STAMP_SEC}"
ros2 service call /scan_target_acquisition/prepare \
  piper_mobile_manipulation/srv/PrepareAcquisition \
  "{session_id: '${SESSION_ID}', rough_target: {header: {stamp: {sec: ${STAMP_SEC}, nanosec: 0}, frame_id: base_link}, point: {x: 0.45, y: 0.0, z: 0.15}}}"
```

Replace the example XYZ with the actual rough coordinate. Never reuse a session ID for changed
coordinates. Repeating the exact command is safe and idempotent. Monitor:

```bash
ros2 topic echo /piper/tesseract_readiness
ros2 topic echo /piper/reachable_acquisition_viewpoints
ros2 topic echo /piper/tesseract_plan_status
ros2 topic echo /piper/scan_execution_plan
ros2 topic echo /piper/scan_execution_status
```

Proposal-only mode still has no `/joint_ctrl_single` publisher. For a separately supervised
motion-enabled acceptance, approve the exact `plan_id` and `trajectory_sha256` using the normal
executor approval service. Acquisition uses the selected 1-100 percent session speed, never captures, and may run without
target tracking only for its exact hash-bound `ROUGH_ACQUISITION` plan. The first approved segment
is intentionally planned as schema-v5 `bootstrap_static`: it ignores current GroundingDINO/SAM2
target and obstacle output so the camera can aim at a cube that was initially outside view. This is
not an unchecked move—the qualified robot/camera/mount/cable collision model, floor proxy, joint
limits, feedback convergence, arm status, camera timestamp health, exact approval, and
cancel-to-hold remain active.

After that first segment settles, each look requests
GroundingDINO on a frame captured after settling. The executor allows 10 seconds for that fresh
frame/idle worker, 60 seconds for the exact GroundingDINO request, then 10 seconds for a new measured
`LOCKED` target. The measured stable `base_link` target must be from the processed frame or newer
and within 0.30 m of the rough coordinate. “Target not found” does not move immediately: the
executor waits up to 15 seconds for an obstacle scene correlated to that result. A result with zero
obstacles is already an authoritative correlated clear result, so the executor constructs the same
timestamped typed empty scene locally and advances even if the separate scene relay was missed. If
obstacles were detected, target ID 1 remains
reserved and SAM2 tracks obstacle IDs 2+ for normal depth projection. Only then is the next exact
Tesseract segment checked against those boxes. Worker, clock, scene, correlation, or health failures
abort and hold.

Hand/person/finger are the only semantic unsafe obstacle classes. The live obstacle prompt is
bounded to pen plus hand/finger: pen remains operator-cleared movable clutter and hand remains a
fail-closed blocker. Paper, tissue, wire, cable, and cardboard are no longer requested from
GroundingDINO because low-confidence desk/cable masks produced oversized false collision boxes.
Invalid or untrusted geometry for any emitted instance continues to block.

Measured lock normally produces `ACQUIRED`, holds current position, starts the supervised workflow
once, and waits for `SCAN_READY`; it does not request a normal proposal. Exhausting the approved
sweep produces `ACQUISITION_FAILED` and a hold. If acquisition terminates but a fresh authoritative
measured lock is present, the GUI may enable **Prepare Scan from Current Lock**. That explicit
action starts the GUI-owned worker/scan stack first if needed, waits for authoritative workflow
validation, adopts the current lock as a new scan phase, starts/restarts the workflow, waits for
`SCAN_READY`, and creates one fresh request through `/tesseract_plan_bridge/request_plan`.
It never relabels the acquisition or reuses its approval.
Only its correlated exact 13-view result may receive the second confirmation. Any failure,
cancellation, process exit, retry, changed plan, or replan requires new exact confirmations.

### Feedback-gated SDK MoveJ target execution

PiPER's SDK MoveJ boundary accepts six joint positions and one aggregate speed percentage. It does
not consume Tesseract's per-joint velocity or acceleration arrays. A live 5-percent acquisition on
2026-07-29 reached measured `SCAN_READY`, but the former timed adapter sent 7,281 tiny targets and
was visibly slow/choppy. A later 70-target waypoint acquisition still looked segmented, while one
GUI SDK target returned to the compact pose smoothly in about two seconds.
`sdk_movej_targets_v1` now matches the actual hardware interface:

- OMPL and ISP still establish feasible all-J1-J6 goals; J6 remains ordinary and free;
- the worker binds one final position target per viewpoint and adaptively collision-validates the
  direct SDK-interpolated joint segment;
- only a folded-start rough acquisition may add one separately proven bootstrap target;
- 0.025 rad validation samples are internal checks and are never hardware commands;
- qdot/qddot fields remain in schema v5 for message compatibility but must be exact zero, accurately
  declaring that they are not controller inputs;
- the approval hash covers every position target, zero derivative placeholder, order stamp, model,
  calibration, position/controller limits, selected aggregate speed, rate ceiling, and policy;
- Foxy publishes each six-position arm-only endpoint exactly once and waits for measured feedback
  to enter 0.025 rad before capture or the next viewpoint; repeatedly publishing the same endpoint
  can restart PiPER's SDK interpolation and is forbidden;
- a target exceeding 90 seconds, showing no 0.001 rad total-joint improvement for 20 seconds, or encountering a
  stale, invalid, or malformed controller limits abort to current-position hold; a later
  stable fresh valid hash is accepted for position-only SDK MoveJ without changing the
  selected aggregate speed or approved geometric path;
- endpoint capture requires final joint-position convergence plus a healthy synchronized camera
  clock; post-approval tracker reacquisition is diagnostic and does not cancel the 13-view route;
- the PiPER driver sends every `JointCtrl` target, caches unchanged `MotionCtrl_2` mode/speed, and
  never sends `GripperCtrl` for the six-position automation form.

The 2026-07-30 target-only regression captured 13 synchronized records at 5 percent. It exposed a
post-capture obstacle-telemetry timeout during the return-only segment; the capture result is now
preserved and the arm is held at its current position instead of starting Step 4/5 recovery. The
diverse-dome replacement has passed the ROS-free real Tesseract backend with 13 selected viewpoints
plus the collision-validated return segment. The isolated worker has a 150-second internal planning
budget, checks it before every OMPL attempt and throughout adaptive collision validation, and
reserves five seconds to serialize the result before the bridge's 180-second timeout and the GUI's
185-second result deadline. Higher-speed behavior remains unqualified.

### First physical acceptance: one viewpoint at 5 percent

Start with one viewpoint only. This creates the joint-command publisher but still does not enable the
arm and still cannot move until an exact fresh plan is approved:

```bash
PIPER_ENABLE_REAL_VIEWPOINT_MOTION=1 \
PIPER_VIEWPOINT_SPEED_PERCENT=5 \
PIPER_VIEWPOINT_MAX_VIEWS=1 \
PIPER_VIEWPOINT_MIN_VIEWS=1 \
./L515_camera/run_supervised_viewpoint_execution.sh
```

Call workflow `start` again and wait for a fresh `SCAN_READY` and `PROPOSAL_READY`. Inspect the full
plan in RViz and the typed topic. A plan expires after 60 seconds and is rejected if the target has
moved more than 15 mm. Only after checking the planned joint target, workspace, cables, and camera
clearance should the operator enable the arm:

```bash
./enable_piper.sh
```

Approve the exact displayed plan ID within its freshness window:

```bash
ros2 service call /scan_viewpoint_executor/approve \
  piper_mobile_manipulation/srv/ApproveScanExecution \
  "{plan_id: '<PLAN_ID>', trajectory_sha256: '<TRAJECTORY_SHA256>', confirmation: 'EXECUTE APPROVED SCAN'}"
```

Cancel immediately on unexpected motion or tracking degradation:

```bash
ros2 service call /scan_viewpoint_executor/cancel std_srvs/srv/Trigger '{}'
```

Use the physical emergency stop/power procedure if software does not respond. After the one-view test,
disable the arm and review `/piper/scan_execution_status`, TF stability, J6 feedback, target drift, and
the accepted capture before attempting 13 views.

### Thirteen-view automatic capture

Repeat the proposal-only review, then use motion opt-in with the default 13-view minimum/maximum:

```bash
PIPER_ENABLE_REAL_VIEWPOINT_MOTION=1 \
PIPER_VIEWPOINT_SPEED_PERCENT=5 \
./L515_camera/run_supervised_viewpoint_execution.sh
```

After exact approval, the executor moves incrementally, waits for feedback and settled tracking at
each view, calls `/supervised_cube_workflow/capture_view` for quality/full-resolution target-cloud
acceptance, then calls `/scan_capture/capture_view`. That second service writes synchronized RGB,
raw depth NPY, 16-bit millimetre depth PNG, mask, camera intrinsics, joint state, and execution
metadata under `datasets/active_scan`. Either capture failure aborts and holds instead of silently
skipping a viewpoint. The executor waits for `accepted_views` and `modeled_views` to advance before
moving again and leaves the arm at the final approved view. Finish and save the accumulated target
cloud after 13 accepted views:

```bash
ros2 service call /supervised_cube_workflow/finish_scan std_srvs/srv/Trigger '{}'
```

## Real PiPER arm commands

Start driver:

```bash
./start_piper.sh
```

Enable real arm:

```bash
./enable_piper.sh
```

Disable real arm:

```bash
./disable_piper.sh
```

Move to all-zero joint target:

```bash
./reset_piper.sh
```

Move to saved home/reset pose:

```bash
./reset_arm.sh
```

Start GUI:

```bash
./start_gui.sh
```

The **Acquire & Scan** tab is the supported rough-coordinate automation UI. It manages the
Tesseract worker and viewpoint stack only; start the driver and camera/perception separately.
Manual Send, Live Send, and preview-mirror motion are locked while automation owns
`/joint_ctrl_single`. The worker uses a nonblocking lock under `XDG_RUNTIME_DIR` (or `/tmp`) so a
second worker cannot silently compete.

The acquisition center look uses the current camera-to-hint direction and preserves the live
camera position when it is already inside the configured 0.45 m maximum standoff. The four
primary fallback looks sweep 15 degrees left/right/up/down at that effective radius. When those
five primary candidates do not yield bounded IK, the same request also carries fifteen
deterministic, deduplicated looks on a compact 0.30 m orbit. Tesseract still selects exactly five
views. This lets a valid centerline hint such as `[0.25, 0.0, 0.0]` search for an initially absent
target without weakening collision checks or moving until the exact proposal is approved.

The folded-start declaration may contain one or two joints, but only J2/J3 are permitted and every
declared joint must start within 0.04 rad of its normal limit. Two violations produce one combined
bootstrap target, not two independent moves. Each delta is bounded by 0.15 rad, every undeclared
joint remains fixed, and dense validation must monotonically leave only the specifically qualified
folded-start contacts before the ordinary path begins.

The command-free Foxy handoff retries a dropped reachable-acquisition callback for at most 10 seconds;
only rejected calls that queued no plan are retried, with one service call in flight and one
accepted Tesseract request. A preexisting obstacle callback is deliberately not required for the
first bootstrap plan. A timeout still leaves confirmation disabled and sends no command.
The worker heartbeat and `/piper/tesseract_readiness` must remain fresh. Scan bridge and
reachability subscriptions are lifetime-owned and are not recreated when input is stale; blockers
clear only after real callbacks resume. Any critical scan child exit shuts down the scan launch so
the GUI can disable the remaining automation controls and clean up its owned worker.

The **Graphical** tab controls a motion-free 3D digital twin made from the real STL link meshes:

1. Keep the arm disabled and select **Open 3D Joint Editor**.
2. Select **Load Live Arm into 3D Preview**.
3. In RViz, use the Interact tool and drag the large orange rotation ring on joint1 through joint6.
4. Review the six preview angles, model pose, physical workspace, cables, and camera clearance.
5. For the first physical test, change only one joint slightly and set 5% speed.
6. Enable the arm separately, then select **Confirm: Mirror 3D Preview on Real Arm**.

The editor publishes only `/piper_gui/preview_set`, `/piper_gui/preview_joint_states`, interactive
markers, and `preview_` TF frames. Dragging cannot command the real arm. The separate mirror action is
capped at 10%, preserves the current gripper position, refuses stale feedback or arm errors, validates
a conservative robot/floor path, and requires confirmation. It does not use live obstacle boxes, so a
clear workspace and physical supervision remain mandatory. Never use it while the automatic viewpoint
executor is motion-enabled.

Only use real motion commands when the arm workspace is clear.

## GitHub version commands

Show recent commits:

```bash
git log --oneline --decorate --graph -10
```

Current rollback commit:

```text
5d995cf Revert "Add base-frame perception recovery workflow"
```

Reverted commit:

```text
fa32da6 Add base-frame perception recovery workflow
```

Previous supervised dry-run version:

```text
dac547e Add supervised dry-run cube workflow
```

View a file from a previous commit:

```bash
git show dac547e:path/to/file
```

Temporarily browse an old version:

```bash
git checkout dac547e
```

Return to current main:

```bash
git checkout main
```
