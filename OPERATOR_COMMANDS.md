# PiPER Scan — Operator Quick Start

## Passive results campaign

The GUI **Results Campaign** tab records all cuRobo missions first and then the
same target-position sequence with Tesseract. These are `MATCHED_RUNS`, because
backend-block execution is confounded with elapsed time and environment. It records
missions without publishing ROS data or changing mission behaviour. Full design,
interpretation and output details are in
[`docs/experiments/results_campaign.md`](docs/experiments/results_campaign.md).

Command-line collection and report generation are also available:

```bash
cd /home/prl/Piper_arm
python3 tools/record_results_campaign.py --project-root /home/prl/Piper_arm --campaign piper-poster-blocked-20260902
python3 tools/build_results_campaign_report.py --project-root /home/prl/Piper_arm --campaign piper-poster-blocked-20260902 --run-reconstruction
```

These commands are offline/file-only. They do not start ROS, cameras, planners,
CAN, drivers, or robot motion.

This is the authoritative operating procedure for the one-button adaptive scan
workflow. Run commands from the repository root:

```bash
cd ~/Piper_arm
```

> [!CAUTION]
> Software cancel is not an emergency stop. Keep the physical emergency-stop
> method ready whenever the arm can be powered. Real motion is disabled by
> default and must remain disabled for an unqualified planner, collision model,
> floor profile, mount, speed, or workspace.

Current operating boundary:

| Selection | Status |
|---|---|
| Tesseract + tabletop profile | Current supervised reference; real motion still requires all explicit 5% opt-ins and every runtime gate. |
| Tesseract + tracked-robot ground profile | Proposal-only; `qualified_for_hardware: false`. |
| cuRobo + tabletop profile | Hardware-qualified for supervised 5% target-scan missions; requires the independent collision-model and real-motion opt-ins plus every runtime gate. |

Historical incidents, qualification runs, and detailed acceptance evidence are
kept in [`docs/historical/system_handoff_2026_08_11.md`](docs/historical/system_handoff_2026_08_11.md) and
[`docs/ai/80-problem-log.yaml`](docs/ai/80-problem-log.yaml), not in this quick
start.

## One-time host setup

Install the CAN boot service once on a new computer:

```bash
./scripts/setup/install_piper_can_service.sh
```

Enter the sudo password only during this installation. It configures `can0`
at 1 Mbps automatically after future boots and USB reconnects. It does not
start the ROS driver or enable the arm.

Confirm CAN after a boot:

```bash
systemctl is-active piper-can@can0.service
ip -details link show can0
```

The expected output is `active`, `state UP`, `can state ERROR-ACTIVE`, and
`bitrate 1000000`.

## Preflight

Before every session:

```bash
cd ~/Piper_arm
./verify_installation.sh
./scripts/robot/check_piper_can.sh
```

`verify_installation.sh` proves the installed software contract, not physical
safety. Inspect the arm, camera mount, cable, support plane, target workspace,
saved home path, and all six motor states before a powered run.

## Normal startup

Use three terminals. Do not separately run `start_piper.sh`, the camera/GPU
pipeline, hand-eye TF, or a planner worker. The coordinator owns and starts the
selected generation in the required order.

### Terminal 1 — proposal-only coordinator

```bash
cd ~/Piper_arm
./run_target_scan_mission.sh
```

This is the default and recommended first startup. It starts the coordinator
without permission to enable or move the arm. Tesseract is the default planner.

### Terminal 1 — one coordinator with GUI planner selection

You do **not** need separate coordinator nodes for Tesseract and cuRobo. Start
the same coordinator with both backends available, then use **Motion planner
for next mission** in the GUI and press **Apply for Next Mission** while the
mission is idle:

```bash
cd ~/Piper_arm
PIPER_PLANNER_BACKEND=tesseract \
PIPER_CUROBO_PYTHON=/home/prl/.venvs/piper-curobo-v0.7.8/bin/python \
PIPER_CUROBO_COLLISION_MODEL_QUALIFIED=1 \
PIPER_FLOOR_PROFILE=saved \
PIPER_MISSION_ENABLE_REAL_MOTION=1 \
PIPER_MISSION_SPEEDS_QUALIFIED=1 \
PIPER_MISSION_FREE_MOTION_SPEED_PERCENT=5 \
PIPER_MISSION_CONTACT_SPEED_PERCENT=5 \
./run_target_scan_mission.sh
```

`PIPER_PLANNER_BACKEND` is the coordinator's default only. The GUI writes an
explicit `planner_backend` into each mission goal, so its applied selection
wins for that mission and is then frozen until the mission finishes. The
cuRobo interpreter and qualification variables merely make cuRobo available;
they do not select cuRobo and do not grant it a separate execution path.

`PIPER_FLOOR_PROFILE=saved` makes the coordinator read the GUI's saved
**Collision environment for next mission** when each mission starts. That
control chooses the support-plane profile, not the planner backend. For real
cuRobo motion, the saved environment must currently be `tabletop`; `ground`
remains unqualified and fails closed. Set `PIPER_FLOOR_PROFILE=tabletop` only
when you intentionally want to override and lock out the GUI floor selection.

### Terminal 1 — supervised physical motion

Use this instead of the proposal-only command only after the tabletop Tesseract
profile, saved home, full installed geometry, current workspace, and 5% speed
gate have been checked for the exact session:

```bash
cd ~/Piper_arm
PIPER_PLANNER_BACKEND=tesseract \
PIPER_FLOOR_PROFILE=tabletop \
PIPER_MISSION_ENABLE_REAL_MOTION=1 \
PIPER_MISSION_SPEEDS_QUALIFIED=1 \
PIPER_MISSION_FREE_MOTION_SPEED_PERCENT=5 \
PIPER_MISSION_CONTACT_SPEED_PERCENT=5 \
./run_target_scan_mission.sh
```

For the hardware-qualified cuRobo scope, use the same supervised 5% gates and
add both the explicit isolated interpreter and collision-model opt-in:

```bash
cd ~/Piper_arm
PIPER_PLANNER_BACKEND=curobo \
PIPER_CUROBO_PYTHON=/home/prl/.venvs/piper-curobo-v0.7.8/bin/python \
PIPER_CUROBO_COLLISION_MODEL_QUALIFIED=1 \
PIPER_FLOOR_PROFILE=tabletop \
PIPER_MISSION_ENABLE_REAL_MOTION=1 \
PIPER_MISSION_SPEEDS_QUALIFIED=1 \
PIPER_MISSION_FREE_MOTION_SPEED_PERCENT=5 \
PIPER_MISSION_CONTACT_SPEED_PERCENT=5 \
./run_target_scan_mission.sh
```

The cuRobo opt-in does not enable motion by itself. Do not use the unqualified
`ground` floor profile for physical execution.
The worker also verifies that the generated, hash-bound qualification record
matches `tabletop` and exactly 5% free/contact speed. Any mismatch reports the
planner outside its hardware-qualified scope before a plan can be dispatched.

This launcher is a singleton. A second invocation exits with code 73 before
starting ROS nodes or mission children. Before submitting a GUI goal, require
exactly one `/piper/run_target_scan` action server.

The combined PiPER/L515/Bunker model is always loaded and shown in RViz. In the
GUI, **Motion planner for next mission** selects Tesseract or cuRobo and
**Collision environment for next mission** selects only the support plane:
`Tabletop floor (z = +0.005 m)` or `Tracked-robot ground (z = -0.466 m)`.
Apply selections while idle; the coordinator snapshots them when the next
mission starts. The explicit backend in the GUI mission goal takes precedence
over the launcher's default backend; clients that omit the field use that
default. The tabletop Tesseract manifest and current cuRobo model are
hardware-qualified for their declared supervised scopes. The ground profile is
not. cuRobo remains a non-conservative sphere approximation and is not claimed
collision-equivalent to Tesseract. For a command-line override, start the coordinator with
`PIPER_FLOOR_PROFILE=tabletop` or `PIPER_FLOOR_PROFILE=ground`; omit it to use
the saved GUI choice. Planner selection is similarly frozen for the mission and
never falls back automatically.

### Terminal 2 — GUI

```bash
cd ~/Piper_arm
./start_gui.sh
```

### Terminal 3 — RViz

```bash
cd ~/Piper_arm
source ./source_piper_foxy_environment.sh
export ROS_DOMAIN_ID=42
export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/fastdds_gui_udp_only.xml"
export RMW_FASTRTPS_USE_QOS_FROM_XML=0
export ROS_LOCALHOST_ONLY=0

ros2 launch piper_description display_live_robot.launch.py \
  fastdds_profile:="$FASTRTPS_DEFAULT_PROFILES_FILE"
```

RViz is feedback-only. Before a mission starts the driver, its RobotModel can
show missing transforms or a collapsed arm. This is expected; it should become
live during the driver-start phase. The fixed Bunker chassis and sensor station
remain visible even while the arm driver is stopped.

## Measure the camera/perception distance range

Use this command-free diagnostic when establishing the nearest and farthest
reliable range for GroundingDINO, SAM2 and L515 depth. It starts the camera and
GPU perception only; it does not start the PiPER driver, enable motors, publish
joint commands or start a mission.

Terminal 1:

```bash
cd ~/Piper_arm
ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=0 \
./L515_camera/run_gpu_vision_pipeline.sh
```

After requesting/detecting the target, open the correctly scaled four-panel
viewer in Terminal 2:

```bash
cd ~/Piper_arm
./L515_camera/view_range_debug_dashboard.sh
```

The dashboard shows raw RGB, RGB with the live SAM target mask, aligned depth
on a fixed diagnostic 0.15-4.00 m colour scale, an uncapped raw-mask median,
the production `Target3D` depth/quality values, and native L515 confidence
scaled over its real 0-15 range. Set
`PIPER_RANGE_DASHBOARD_MAX_DEPTH_M` before launching to use another display
maximum. The production `Target3D` value is admitted only through the current
3.00 m hard target-observation ceiling; the separately labelled raw-mask value
does not apply that gate. Press `q`
or Escape in the window to close only the viewer. Stop the perception stack
cleanly with:

```bash
cd ~/Piper_arm
./L515_camera/stop_gpu_vision_pipeline.sh
```

The aligned-depth panel deliberately shows one raw frame at a time and may
flicker on invalid, moving, reflective or mixed-depth pixels. Mission captures
do not use a single raw frame: they confidence-filter and temporally median 20
new native depth frames at the settled viewpoint.

To measure reliability within and around the current 3.00 m software ceiling,
keep the perception pipeline and dashboard running and start:

```bash
cd ~/Piper_arm
./L515_camera/record_perception_range_test.sh
```

Use a tape measure from the L515 optical face to the visible target surface.
At each requested distance, hold the target stationary, enter the distance in
metres, and capture three repetitions. Start with 0.25, 0.35, 0.50, 0.75,
1.00, 1.50, 2.00, 2.50, 2.75 and 3.00 m. Optional samples beyond 3.00 m are
raw diagnostic evidence only and are not production-admissible. Bracket the
first reliability failure in 0.10 m steps. This characterizes the selected
target and lighting, not every possible object or environment.

Every sample requests a new request-correlated GroundingDINO/SAM2 result and
analyses its exact aligned-depth artifact over 0.15-9.00 m without changing the
production gates. Results are updated after each sample under
`datasets/perception_range_tests/PerceptionRange - HH-MM - DD-MM-YYYY/` as:

- `perception_range_results.csv` — canonical raw table;
- `perception_range_results.xlsx` — Excel-readable table;
- `perception_range_plot.html` — plotted measured-versus-reference depth and
  pass/fail points.

The diagnostic `usable` result requires a detected/accepted target mask, no
frame-border contact, at least 50 coherent target points, at least 0.50 raw
depth and selected-layer support, selected depth standard deviation no greater
than 0.03 m, and absolute error no greater than 0.03 m or 5% of the reference
distance, whichever is larger. Choose an operational maximum only after three
repetitions pass at that distance and the next outward bracket fails; do not
raise production limits from one successful frame.

## Run one scan

1. Clear the entire direct current-to-home sweep, check the camera cable, and keep the emergency stop ready. Dedicated home stages skip only intentional folded robot self-collision; the full installed holder/L515 must retain at least 5 mm table/floor and external-obstacle clearance throughout the sweep.
2. In the GUI, open **Automatic Scan**.
3. Confirm the saved home shown by the GUI.
4. Optionally set **Camera-on-ray tolerance for next mission** from 1.0° to
   90.0° and press its **Apply for Next Mission** button. Rough acquisition
   and the first target lock remain capped at 5°; later scan viewpoints use
   the selected planned/path/achieved aim gate. It is frozen at mission startup.
5. Enter rough target X/Y/Z in metres in `base_link`.
6. Press **Start Complete Automated Scan** once.
7. Do not start other driver, camera, perception, TF, planning, or joint-command
   processes while the mission is active.

Pressing `Ctrl+C` in the coordinator terminal requests the same bounded cancel
path as the GUI: configured home, settled hold, feedback-confirmed disable, and
owned-child cleanup. The launch file allows 180 seconds for this recovery. Use
the physical emergency-stop procedure for an actual emergency; do not use a
second coordinator or force-kill mission children.

The coordinator starts the disabled driver, proves feedback, starts camera/GPU
perception, hand-eye TF, the frozen planner backend and the scan stack, enables
the arm, performs
rough acquisition, runs a separate request-correlated semantic occlusion probe,
then plans and captures at most one synchronized RGB-D view per measured-state
transaction until 8-24 are accepted and feature/coverage gates pass. Eight is only the model-seed floor; measured convergence selects the terminal count. Each next automatic view is selected by cumulative global information gain across the configured target-centric region; the bounded Tesseract handoff retains global leaders plus one informative continuity fallback and imposes no maximum angular step. The executor rejects a joint
path that turns the camera off the measured target. The mission then returns to
the saved home through direct ROUGH_HOME and STORAGE_WRIST requests with
robot self-collision validation explicitly exempted but CAD-derived
holder/L515 external clearance still mandatory, holds, disables,
and stops its child stack.

## Inspect an offline cross-view reconstruction

In the GUI's **Reconstruction Validation** tab, select a completed dataset and
choose **Constrained superposition**. Choose **Projected colour depth
(legacy)** to reproduce the existing sparse colour-plane input, or **Native
L515 depth (dense)** to reverse-correlate the same accepted confidence-qualified
samples onto their contiguous native grid. The two choices write separate
`target_mesh.ply` and `target_mesh.native_depth.ply` outputs. For the recorded
fine-grid comparison use a 1.5 mm mesh voxel and 6 mm TSDF band; the unchanged
compatibility defaults are 3 mm and 15 mm. **Mesh repair** defaults to **None
(measured TSDF only)**. Select **Conservative measured-wall repair (6 mm)** to
write another independent `*.wall_repaired.ply` comparison. It fills only
bounded side-wall TSDF openings, records the added triangles as interpolated,
keeps the raw TSDF untouched and refuses to erase object-sized open boundaries.
It is not measured evidence and cannot promote a failed dimensional quality
result. **Build Raw + Cleaned** remains
command-free and does not move the arm. After it finishes, use **Open All
Capture Overlays** to inspect every accepted viewpoint together, with a
different colour per viewpoint. Use **Open Superposition Overlay** to inspect
all of those full capture clouds after the constrained whole-view transforms
have placed them into capture 0's fixed frame. Capture 0 never moves. Later
captures may translate by whatever distance the overlap solve requires; their
camera-origin-centred rotation is regularized toward the minimum necessary
value and hard-capped at 3°. Overlap may connect through an intermediate
capture, so a view of another side does not need to match capture 0 directly.
**Open Consensus Points** shows only
surface locations supported by at least two distinct viewpoints; it uses one
representative per viewpoint, median/MAD outlier rejection, then an
equal-weight mean. The consensus is not the TSDF mesh and does not turn a
FAIL-quality reconstruction into a pass. The constrained inspection buttons become
available only after rebuilding in `constrained_superposition` or `auto` with
the current reconstruction code.

**Open Textured Model** opens the derived constrained OBJ/MTL/PNG model. It
starts from every confidence-qualified measured point in the superposition
overlay. Multi-view scans retain points near cross-capture consensus, connect
only adjacent source-depth pixels, and fine-voxel average the aligned surfaces.
Each resulting triangle takes RGB from one best depth-consistent source capture,
so independent views are not blurred together. Grey triangles had no qualified
source image. This output remains diagnostic and does not turn reconstruction
quality `FAIL` into a pass.

## Queue multiple targets

The tracked-robot gateway may keep at most eight target actions outstanding.
It snapshots each fresh odometry target into the arm's local base frame when
the request arrives. The coordinator waits one second for near-simultaneous
requests, then starts the closest target; equal-distance targets keep arrival
order. A newly closer target never interrupts an active scan. The active target
must finish capture validation, direct home, disable, owned-child cleanup, and
its durable result before the closest remaining target starts with fresh
runtime caches and a new dataset. Canceling a queued target removes only that
target and does not cancel the active arm mission.

For a disabled/proposal-only queue smoke, submit far then near within one
second and require `QUEUED` feedback followed by near-first dispatch and only
one child-process generation. Do not perform this as a powered test until all
hardware gates in this document pass.

Completed datasets are written below:

```text
/home/prl/Piper_arm/datasets/active_scan/
```

## Cancel or recover a failed scan

Use the GUI's **Cancel and Home** button. Wait for all of the following before
closing terminals or removing arm power:

1. Feedback reaches the saved home.
2. The GUI proves the final hold.
3. Disable reports success.
4. The mission reports `safe_shutdown=true`.

If the result is `NEEDS_OPERATOR` or `safe_shutdown=false`, do not assume a
home move is safe. Keep the arm supported, use the physical emergency-stop/power
procedure when required, stop command ownership, and verify all six per-motor
enable flags. After contact, a motor fault, stale/nonfinite feedback, or a
holder-clearance rejection, do not use the GUI to force home and do not reset or
re-enable. Inspect the mechanism first. A high-level `arm err_code=0` does not
override per-motor FOC collision/current/temperature/driver/stall faults.

## Contact or per-motor fault procedure

1. Stop motion with the physical emergency procedure appropriate to the event;
   software cancel is not an emergency stop.
2. Prevent every publisher/coordinator from issuing another command.
3. Read all six low-speed FOC enable flags and fault fields. Treat a single
   disabled/faulted motor with other motors enabled as an unsafe partial enable.
4. Use the explicit all-axis disable path and prove all six flags are false.
5. Do not clear faults, power-cycle as a reset, re-enable, or command home until
   the affected joint, mount, cable, support-plane contact, and full next path
   have been inspected.
6. Resume only after explicit authorization, initially at five percent with the
   emergency stop ready and the full holder/L515 path visibly clear.

For an emergency, use the physical emergency-stop/power procedure; software
cancel is not an emergency stop.

## Normal shutdown

After the GUI confirms home, hold, and disable:

1. Press `Ctrl+C` in the coordinator terminal if it is still running.
2. Close the GUI.
3. Press `Ctrl+C` in the RViz terminal.

The CAN boot service may remain active; it does not enable or command the arm.
