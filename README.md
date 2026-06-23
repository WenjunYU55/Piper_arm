# PiPER Arm, L515 Camera, and Offline Perception

This repository contains three separate dependency surfaces:

1. The PiPER ROS 2 workspace in `piper_ros_foxy/`.
2. Intel RealSense L515 source-build helpers in `L515_camera/`.
3. Offline AI experiments in `AI_perception_tests/`.

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
source install/setup.bash
```

The installer installs the ROS, Python, GUI, build, and CAN packages used by the checked-in code. It also installs the tested `piper_sdk==0.6.1` and Python 3.8-compatible `python-can==4.5.0` with pip because the SDK has no ROS dependency key and Ubuntu 20.04's Python CAN package is too old. It then runs `rosdep` against every package manifest.

Verify the dependency declarations:

```bash
source /opt/ros/foxy/setup.bash
rosdep check --from-paths piper_ros_foxy/src --ignore-src --rosdistro foxy
```

Start the arm only after its workspace is built and the USB-CAN adapter and arm are connected:

```bash
./start_piper
```

The startup script checks `python-can`, SciPy, `piper_sdk`, the ROS overlay, and the CAN interface before launching the controller. See `piper_ros_foxy/README(EN).MD` for arm and CAN-specific operation.

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
