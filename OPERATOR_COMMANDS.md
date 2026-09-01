# PiPER Scan — Operator Quick Start

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
| cuRobo | Command-free planning only; current collision approximation is `hardware_qualified: false`. |

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

Do not substitute `curobo` or `ground` in this physical-motion command. Those
paths are not currently hardware-qualified and must fail closed.

This launcher is a singleton. A second invocation exits with code 73 before
starting ROS nodes or mission children. Before submitting a GUI goal, require
exactly one `/piper/run_target_scan` action server.

The combined PiPER/L515/Bunker model is always loaded and shown in RViz. In the
GUI, **Motion planner for next mission** selects Tesseract or cuRobo and
**Collision environment for next mission** selects only the support plane:
`Tabletop floor (z = +0.005 m)` or `Tracked-robot ground (z = -0.466 m)`.
Apply selections while idle; the coordinator snapshots them when the next
mission starts. The tabletop Tesseract manifest is hardware-qualified for its
declared supervised scope. The ground profile and current cuRobo approximation
are not. For a command-line override, start the coordinator with
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
