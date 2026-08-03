# PiPER Scan — Operator Quick Start

This is the complete command list for the current one-button, 13-view scan
workflow. Run every command from:

```bash
cd /home/prl/Piper_arm
```

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

1. Clear the arm workspace and check the camera cable and emergency stop.
2. In the GUI, open **Automatic Scan**.
3. Confirm the saved home shown by the GUI.
4. Enter rough target X/Y/Z in metres in `base_link`.
5. Press **Start Complete Automated Scan** once.
6. Do not start other driver, camera, perception, TF, planning, or joint-command
   processes while the mission is active.

The coordinator starts the disabled driver, proves feedback, starts camera/GPU
perception, hand-eye TF, Tesseract and the scan stack, enables the arm, performs
rough acquisition, requests one 13-view plan, captures 13 synchronized RGB-D
views, returns to the saved home, holds, disables, and stops its child stack.

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

If the result is `NEEDS_OPERATOR` or `safe_shutdown=false`, do not kill the
coordinator or driver and do not power off the arm. Keep the arm supported and
use the GUI to return to the saved home and disable it. Stop remaining processes
only after disable success.

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
