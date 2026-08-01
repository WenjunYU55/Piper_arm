# PiPER Arm, L515 Camera, and Offline Perception

## Autonomous target-scan mission

The production entry point is the typed ROS 2 action `/piper/run_target_scan`.
It accepts one labelled rough target pose, runs bounded acquisition and a
13-view scan, returns to the approved home pose, proves a settled current-pose
hold, disables the motors, stops every PiPER-owned child, and only then reports
a successful dataset result. The GUI now opens with a separate **Automatic
Scan** tab: after entering rough XYZ, one button submits the whole mission
through the same action. The five-step **Acquire & Scan** tab remains available
only as the commissioning harness.

The automatic mission advances by observed readiness, not startup sleeps. It
requires a fresh, settled joint stream from the driver generation it started;
healthy camera timestamps and ready GroundingDINO/SAM2 CUDA workers; a new
hand-eye transform; a new healthy Tesseract worker generation; and finally
typed acquisition readiness. Each barrier has a bounded timeout and reports
the component that failed to become ready. The arm is not enabled until all of
those barriers and a second pre-enable joint-stream stability check pass.

The listener is command-free unless real motion is explicitly selected:

```bash
./run_target_scan_mission.sh
```

The 30% transit/10% contact speed profile has a separate deployment gate and
must remain unqualified until its staged physical acceptance is recorded.

For two-host deployment, the tracked-robot network sees only
`run_target_scan_gateway.sh`. The gateway snapshots
`odom -> piper_base_link` and uses a private hashed filesystem spool to reach
the loopback-only motion domain. On two computers, both launchers must receive
the same secured shared path through `PIPER_MISSION_SPOOL_ROOT`; their default
local `/tmp` path is only for one computer. Automatic leaf/branch contact remains
fail-closed until the installed gripper/contact collision model passes physical
qualification; a hand/person is always a terminal blocker.

For the current whole-system architecture, validated behavior, limitations, safety boundaries, and
recommended continuation point, see [`SYSTEM_HANDOFF.md`](SYSTEM_HANDOFF.md).

For a fresh machine, runtime commands, generated-asset policy, and CPU/GPU/Jetson selection, see
[`CLEAN_INSTALL.md`](CLEAN_INSTALL.md).

For day-to-day operation commands and what each script does, see
[`OPERATOR_COMMANDS.md`](OPERATOR_COMMANDS.md).

For the current Ubuntu 22.04 host, use the Docker-based Foxy environment documented in
[`DOCKER_FOXY_COMMANDS.md`](DOCKER_FOXY_COMMANDS.md).

This repository contains four separate dependency surfaces:

1. The PiPER ROS 2 workspace in `piper_ros_foxy/`.
2. Intel RealSense L515 source-build helpers in `L515_camera/`.
3. Offline AI experiments in `AI_perception_tests/`.
4. The isolated CPU Tesseract 0.35 worker in `motion_planning/tesseract/`, connected to Foxy only
   through the command-free `piper_tesseract_foxy` filesystem adapter.

Do not install the offline AI packages into the ROS Python environment.

## Supported host

The scripts target Ubuntu 20.04 (Focal), ROS 2 Foxy, and Python 3.8 for ROS nodes. The optional Grounded-SAM-2 environment requires Python 3.10 or newer. A PiPER arm also requires a SocketCAN-compatible USB-CAN adapter; camera workflows require an Intel RealSense L515.

ROS 2 Foxy must already be installed at `/opt/ros/foxy`. Foxy is end-of-life, so use a dedicated compatible host or container and do not substitute another ROS distribution without porting and testing the launch files and dependencies.

## Install the PiPER ROS stack

From the repository root:

```bash
chmod +x install_host_dependencies.sh
./install_host_dependencies.sh
source /opt/ros/foxy/setup.bash
cd piper_ros_foxy
colcon build --symlink-install
cd ..
source source_piper_foxy_environment.sh
```

`source_piper_foxy_environment.sh` is the supported runtime loader for GUI and scan tools. It clears
inherited overlay paths, sources Foxy plus the canonical `piper_ros_foxy/install`, and verifies the
installed scan packages and recovery-bearing message schema. Do not create or source a generated
repository-root `install/`; it can shadow the canonical ROS interfaces.

The installer installs the ROS, Python, GUI, build, and CAN packages used by the checked-in code. It also installs the tested `piper_sdk==0.6.1` and Python 3.8-compatible `python-can==4.5.0` with pip because the SDK has no ROS dependency key and Ubuntu 20.04's Python CAN package is too old. It then runs `rosdep` against every package manifest.

Verify the dependency declarations:

```bash
source /opt/ros/foxy/setup.bash
rosdep check --from-paths piper_ros_foxy/src --ignore-src --rosdistro foxy
```

Real-arm convenience launchers are included as explicit `.sh` / `.py` tools only:

- `start_piper.sh` starts the PiPER ROS driver and CAN interface, but does not auto-enable the arm by default.
- `enable_piper.sh` and `disable_piper.sh` call the PiPER enable service.
- `reset_piper.sh` / `reset_piper.py` and `reset_arm.sh` / `reset_arm.py` publish joint commands and can move the real arm.
- `start_gui.sh` / `piper_gui_native.py` opens manual/Graphical controls plus a publisher-exclusive
  Acquire & Scan tab for rough-coordinate acquisition and one exact 13-view session.
- `calibrate_bounds.sh` / `piper_calibrate_bounds.py` records measured joint limits into `piper_joint_bounds.json`.
- `calibrate_joint6_zero.sh` / `piper_joint6_zero.py` diagnoses joint-six feedback and, only with
  `--calibrate` plus two typed confirmations, writes the physically aligned J6 position as controller zero.
- `L515_camera/run_supervised_viewpoint_execution.sh` starts a separate proposal-first scan executor.
  It has no joint-command publisher by default; real motion requires launch opt-in, an exact fresh-plan
  approval, healthy tracking/obstacles/workflow, and a separately enabled arm. See `OPERATOR_COMMANDS.md`.

The selectable Tesseract backend has Foxy interfaces, a command-free bridge, an isolated rootless
Ubuntu 24.04 worker, model builder, tests, and a collision manifest initially qualified on
2026-07-23 and requalified with deterministic seed handling on 2026-07-24 for its
declared supervised guarded scope. Automatic motion still requires launch opt-in, a fresh exact
approval, and every executor health gate. On 2026-07-29 rough-coordinate acquisition and an exact
13-view Tesseract plan completed supervised physical acceptance at 5 percent, with all 13
capture/model handoffs succeeding. J6 is fully safe by operator confirmation and the Tesseract path
treats J1-J6 equally with no J6-specific lock or cost. The bounded acquisition cone aims camera
optical +Z around the rough hint;
the first exact segment uses a schema-v5 static bootstrap without DINO/SAM obstacle output, while
retaining robot/camera/cable/floor collision and runtime gates. GroundingDINO is then bound to a
post-settle frame and exact request. A second look cannot start until the matching typed semantic
scene is ready, and a measured stable lock within 0.30 m starts the normal workflow. The GUI manages
only the worker/scan stack and never enables motors. Acquisition uses one exact confirmation;
after measured lock plus `SCAN_READY`, explicit Prepare Scan from Current Lock creates one fresh
correlated 13-view request, which requires a separate exact confirmation. There is no reusable
15-minute authorization.

Automation speed is now selected in the GUI from 1 through 100 percent, default 5, for both rough
acquisition and the later scan. The executor clamps only to the PiPER SDK's 1-100 percent range.
It now creates a hash-bound, collision-validated all-six-joint SDK MoveJ target path and issues one
arm-only target per viewpoint. A folded acquisition start may use one separately proven bootstrap
target first. Dense collision samples are validation-only and are never sent to the arm. PiPER
MoveJ uses aggregate speed rather than Tesseract qdot/qddot, and automation cannot command the
gripper. The packages build, 258 focused tests and both rootless collision qualifications pass.
The target-only adapter completed the 13-view physical acceptance at 5 percent; higher-speed
dynamics remain unqualified. At every
accepted settled scan viewpoint, the supervised stack additionally records synchronized RGB, raw
depth, a 16-bit millimetre depth PNG, mask, intrinsics, joints, and plan/view metadata under
`datasets/active_scan`. GUI Safe Disable commands and verifies a settled current-feedback hold
before requesting motor disable. The historical live acceptance remains at 5 percent; higher
selections still require a live repeatability audit.

The no-extension wrapper shortcuts are intentionally not included. Use the `.sh` filenames directly on a fresh clone.
The L515 perception and temporal tracking pipeline and the supervised cube coordinator remain
read-only. Only the separately opted-in and approved viewpoint executor can publish slow scan targets.

## Install the L515 camera stack

The L515 integration builds pinned source versions: librealsense `v2.50.0` and realsense-ros `4.0.4`.

```bash
cd L515_camera
./fetch_realsense_sources.sh
./install_realsense_build_deps.sh
./install_l515_host_fixes.sh
./build_realsense_ws.sh
./check_l515_ros.sh
cd ..
./verify_installation.sh
```

The install scripts require `sudo`. Fetching sources requires network access. See `L515_camera/README.md` and `L515_camera/realsense_l515_version_notes.md` before changing SDK, ROS driver, kernel, or firmware versions.

The source build disables librealsense's optional examples and graphical examples. The ROS camera driver does not require them, and disabling them avoids unrelated OpenGL, GLFW, and GTK dependencies on a clean robot host.

`verify_installation.sh` checks the host version, ROS environments, overlays, commands, Python imports and pinned versions, and installed ROS packages. It does not prove that the arm, CAN adapter, L515, USB permissions, firmware, or network are physically working.

## Basic offline perception tools

The static analysis scripts do not need ROS or model frameworks:

```bash
python3 -m venv AI_perception_tests/.venv
AI_perception_tests/.venv/bin/python -m pip install -r AI_perception_tests/requirements_basic.txt
```

## Optional Grounded-SAM-2 tests

Use a separate Python 3.10 environment:

```bash
python3.10 -m venv AI_perception_tests/groundingdino_test/envs/grounded_sam2_py310
AI_perception_tests/groundingdino_test/envs/grounded_sam2_py310/bin/python -m pip install --upgrade pip
SAM2_BUILD_CUDA=0 AI_perception_tests/groundingdino_test/envs/grounded_sam2_py310/bin/python -m pip install -r AI_perception_tests/groundingdino_test/requirements_ai.txt
AI_perception_tests/groundingdino_test/fetch_ai_assets.sh
AI_perception_tests/groundingdino_test/check_env.sh
```

`python3.10` is not supplied by the standard Ubuntu 20.04 repositories; provide it through an isolated Conda environment, pyenv, or another maintained Python distribution. `SAM2_BUILD_CUDA=0` provides the reproducible CPU installation. CUDA installations depend on the host GPU, driver, CUDA toolkit, and the matching PyTorch wheel; validate those separately before enabling CUDA. Model checkpoints are not committed; `fetch_ai_assets.sh` downloads the two required checkpoints and checks out the tested source revision.

## Dependency files

- ROS packages: each `piper_ros_foxy/src/*/package.xml`
- Host and CAN tools: `install_host_dependencies.sh`
- L515 build tools: `L515_camera/install_realsense_build_deps.sh`
- Basic offline analysis: `AI_perception_tests/requirements_basic.txt`
- Grounded-SAM-2: `AI_perception_tests/groundingdino_test/requirements_ai.txt`

Generated ROS build directories, downloaded model repositories, virtual environments, model weights, captures, and analysis outputs are intentionally not dependencies committed to this repository.
