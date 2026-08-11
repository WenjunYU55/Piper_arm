# PiPER Scan — Operator Quick Start

This is the complete command list for the current one-button adaptive scan
workflow. Run every command from:

```bash
cd /home/prl/Piper_arm
```

> **Current hardware state — 11 August 2026:** The earlier J6 startup fault,
> J5 table contact and powered J5 dropout remain required incident history.
> Their fail-closed containment is active. The latest operator-run 50-percent
> task completed STARTUP_WRIST, 24 accepted captures, direct rough/storage
> home, stable hold, all-six disable and child cleanup with
> `safe_shutdown=true`; it failed only the independent coverage result at
> 113.5/120 degrees azimuth and without measured convergence. Repeat the
> read-only CAN/all-six preflight before every mission. This single run does
> not qualify 100-percent dynamics.

## One-time host setup

Install the CAN boot service once on a new computer:

```bash
./install_piper_can_service.sh
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

## Normal startup

Use three terminals. Do not separately run `start_piper.sh`, the camera/GPU
pipeline, hand-eye TF, or the Tesseract worker. The coordinator owns and starts
all of them in the required order.

### Terminal 1 — mission coordinator

```bash
cd /home/prl/Piper_arm
PIPER_MISSION_ENABLE_REAL_MOTION=1 \
PIPER_MISSION_SPEEDS_QUALIFIED=1 \
PIPER_MISSION_FREE_MOTION_SPEED_PERCENT=5 \
PIPER_MISSION_CONTACT_SPEED_PERCENT=5 \
./run_target_scan_mission.sh
```

This launcher is a singleton. A second invocation exits with code 73 before
starting ROS nodes or mission children. Before submitting a GUI goal, require
exactly one `/piper/run_target_scan` action server.

### Terminal 2 — GUI

```bash
cd /home/prl/Piper_arm
./start_gui.sh
```

### Terminal 3 — RViz

```bash
cd /home/prl/Piper_arm
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
live during the driver-start phase.

## Run one scan

1. Clear the entire direct current-to-home sweep, check the camera cable, and keep the emergency stop ready. Dedicated home stages skip only intentional folded robot self-collision; the full installed holder/L515 must retain at least 5 mm table/floor and external-obstacle clearance throughout the sweep.
2. In the GUI, open **Automatic Scan**.
3. Confirm the saved home shown by the GUI.
4. Enter rough target X/Y/Z in metres in `base_link`.
5. Press **Start Complete Automated Scan** once.
6. Do not start other driver, camera, perception, TF, planning, or joint-command
   processes while the mission is active.

Pressing `Ctrl+C` in the coordinator terminal requests the same bounded cancel
path as the GUI: configured home, settled hold, feedback-confirmed disable, and
owned-child cleanup. The launch file allows 180 seconds for this recovery. Use
the physical emergency-stop procedure for an actual emergency; do not use a
second coordinator or force-kill mission children.

The coordinator starts the disabled driver, proves feedback, starts camera/GPU
perception, hand-eye TF, Tesseract and the scan stack, enables the arm, performs
rough acquisition, runs a separate request-correlated semantic occlusion probe,
then plans and captures at most one synchronized RGB-D view per measured-state
transaction until 8-24 are accepted and feature/coverage gates pass. Eight is only the model-seed floor; measured convergence selects the terminal count. Each next automatic view is limited
to a compact target-centred direction step, and the executor rejects a joint
path that turns the camera off the measured target. The mission then returns to
the saved home through direct ROUGH_HOME and STORAGE_WRIST requests with
robot self-collision validation explicitly exempted but CAD-derived
holder/L515 external clearance still mandatory, holds, disables,
and stops its child stack.

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

## Proposal-only software check

To test startup without permitting arm enable or motion:

```bash
cd /home/prl/Piper_arm
./run_target_scan_mission.sh
```

Do not set the real-motion variables for a proposal-only check.
