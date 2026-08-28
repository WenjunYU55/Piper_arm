# Clean Installation

This procedure recreates the PiPER arm and Intel RealSense L515 system on a clean host. Run commands as
your normal user; the installers request `sudo` only for host changes.

For normal operation after installation, use [`OPERATOR_COMMANDS.md`](OPERATOR_COMMANDS.md).

## 1. Supported host

- Ubuntu 20.04 (Focal), x86_64
- ROS 2 Foxy installed at `/opt/ros/foxy`
- Python 3.8 for ROS and PiPER
- Intel RealSense L515 on a USB 3 port
- SocketCAN-compatible USB-CAN adapter for the real arm
- Internet access and a user with `sudo` permission

ROS 2 Foxy is end-of-life. This repository is pinned to Foxy-era dependencies and is not validated on
another Ubuntu or ROS release. Use a dedicated Ubuntu 20.04 host.

## 2. Install ROS 2 Foxy

Skip this section when `/opt/ros/foxy/setup.bash` already exists.

Configure the locale and ROS apt repository:

```bash
sudo apt-get update
sudo apt-get install -y curl gnupg2 locales lsb-release software-properties-common
sudo locale-gen en_GB en_GB.UTF-8
sudo update-locale LC_ALL=en_GB.UTF-8 LANG=en_GB.UTF-8
sudo add-apt-repository -y universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu focal main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
```

Install Foxy:

```bash
sudo apt-get update
sudo apt-get install -y ros-foxy-desktop python3-argcomplete
source /opt/ros/foxy/setup.bash
ros2 --help >/dev/null
```

Optionally source Foxy in new shells automatically:

```bash
grep -qxF 'source /opt/ros/foxy/setup.bash' ~/.bashrc || \
  echo 'source /opt/ros/foxy/setup.bash' >> ~/.bashrc
```

## 3. Clone and install host dependencies

```bash
cd ~
git clone https://github.com/WenjunYU55/Piper_arm.git
cd Piper_arm
./scripts/setup/install_host_dependencies.sh
```

The installer installs build, ROS, GUI, CAN, rootless-worker, and Python dependencies, including
`bubblewrap`, `bzip2`, `can-utils`, and `ethtool`. It pins `piper_sdk==0.6.1` and
`python-can==4.5.0`; do not replace these with Ubuntu's older `python3-can` version.

## 4. Build the PiPER workspace

```bash
cd ~/Piper_arm/piper_ros_foxy
source /opt/ros/foxy/setup.bash
colcon build --symlink-install
cd ..
source source_piper_foxy_environment.sh
```

The build must finish with all five packages successful: `piper_description`, `piper_msgs`,
`piper_mobile_manipulation`, `piper_tesseract_foxy`, and `piper`.
The helper must report no error. It rejects stale inherited overlays and verifies that the scan
capture module and recovery-bearing Tesseract message come from this canonical workspace. Do not
run colcon from `~/Piper_arm` or source `~/Piper_arm/install/setup.bash`.

Confirm the autonomous mission interfaces and launchers:

```bash
ros2 interface show piper_mobile_manipulation/action/RunTargetScan
ros2 interface show piper_mobile_manipulation/srv/AuthorizeMission
./verify_installation.sh
```

For two computers, install the same interface package on the tracked-robot
side, use a wired DDS network, and prefix the arm mount as `piper_base_link`.
Only `run_target_scan_gateway.sh` joins that network; never expose local driver,
camera, planning, or command topics. Configure CAN at boot through a narrowly
scoped system unit or sudo rule so headless startup cannot wait for a password.
Mount one deployment-owned, mode-0700 shared directory at the same path on both
hosts and export that path as `PIPER_MISSION_SPOOL_ROOT` for both launchers.
The filesystem must support atomic rename. The default local `/tmp` spool is
valid only when the gateway and mission run on the same computer.

## 5. Install and qualify the isolated Tesseract worker

The planning worker uses a networkless Bubblewrap runtime with Tesseract 0.35.0.6. It does not join
ROS, access CAN, use the camera, or publish arm commands. Install Micromamba using the official
Linux x86-64 package when it is not already available:

```bash
mkdir -p "$HOME/.local/bin"
temporary_dir="$(mktemp -d)"
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xj -C "$temporary_dir" bin/micromamba
install -m 0755 "$temporary_dir/bin/micromamba" "$HOME/.local/bin/micromamba"
rm -r "$temporary_dir"
export PATH="$HOME/.local/bin:$PATH"
micromamba --version
```

Add `$HOME/.local/bin` to future login shells if it is not already present:

```bash
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc || \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

Create and qualify the pinned runtime:

```bash
cd ~/Piper_arm
./motion_planning/tesseract/setup_rootless_worker.sh
./motion_planning/tesseract/qualify_rootless_worker.sh
```

Both the core and compact qualification suites must pass. The runtime is generated under
`motion_planning/tesseract/.runtime/` and is intentionally not committed.

## 6. Build and configure the L515

Disconnect the L515 before installing its udev rules.

```bash
cd ~/Piper_arm/L515_camera
./fetch_realsense_sources.sh
./install_realsense_build_deps.sh
./install_l515_host_fixes.sh
```

The host-fix installer pauses while installing the RealSense udev rules. Follow its prompt, wait 10
seconds after it completes, then reconnect the L515 directly to a USB 3 port.

Confirm that the SDK can access the camera:

```bash
./diagnose_l515_usb.sh
```

The output must identify `Intel RealSense L515`, show USB type 3.x, and list its serial and firmware.
An `RS2_USB_STATUS_ACCESS` error means the camera was not reconnected after installing the udev rule.

Build the pinned camera stack:

```bash
./build_realsense_ws.sh
./check_l515_ros.sh
cd ..
```

The source pair is pinned to librealsense `v2.50.0` and realsense-ros `4.0.4`. Build warnings from this
older source are expected; a failed package or nonzero command exit is not.

## 7. Configure the PiPER CAN adapter

Connect the USB-CAN adapter and arm, then identify its interface:

```bash
ip -brief link
```

Install the narrowly scoped boot/hot-plug service for the default `can0`
interface and PiPER's 1 Mbps bitrate:

```bash
./scripts/setup/install_piper_can_service.sh
ip -details link show can0
systemctl status --no-pager piper-can@can0.service
```

The installer requests `sudo` once while provisioning the host. Normal GUI,
coordinator, and `start_piper.sh` runs do not request a password. The service
is enabled at boot and the udev rule starts it again when the USB-CAN adapter
is hot-plugged. It configures SocketCAN only; it does not start the ROS driver
or enable any arm motor.

The result must contain `UP`, `can state ERROR-ACTIVE`, and `bitrate 1000000`. Use
`PIPER_CAN_PORT=can1 ./scripts/setup/install_piper_can_service.sh` during provisioning and
`PIPER_CAN_PORT=can1` with runtime scripts if the adapter appears as `can1`.

`start_piper.sh` reuses an interface that is already UP at the exact bitrate.
Headless startup fails with a provisioning instruction instead of waiting for
an impossible password prompt. An interactive direct run retains the legacy
setup fallback for development hosts only.

## 8. Install the scan perception environment

Do not install GroundingDINO, SAM2, or their dependencies into Foxy's Python 3.8 environment. The
full GUI scan pipeline requires an NVIDIA driver, a CUDA-capable GPU, and the isolated Python 3.10
environment below:

```bash
cd ~/Piper_arm
export PATH="$HOME/.local/bin:$PATH"
micromamba create --yes \
  --prefix AI_perception_tests/groundingdino_test/envs/python310_base \
  --channel conda-forge --strict-channel-priority \
  python=3.10.20 pip
PYTHON310="$PWD/AI_perception_tests/groundingdino_test/envs/python310_base/bin/python3.10" \
  ./AI_perception_tests/groundingdino_test/setup_gpu_env.sh
```

The GPU setup is pinned to the versions in `setup_gpu_env.sh` and must finish by printing the CUDA
device name. For offline CPU-only perception development, use `setup_cpu_env.sh` instead; that does
not qualify the real-time GPU scan launcher.

## 9. Optional offline TSDF reconstruction

Keep Open3D outside Foxy's Python environment:

```bash
python3 -m venv reconstruction/.venv
reconstruction/.venv/bin/pip install -r reconstruction/requirements.txt
reconstruction/.venv/bin/python reconstruction/tsdf_reconstruct.py \
  datasets/active_scan/<completed-scan> --output /tmp/target_mesh.ply
```

The prototype requires exactly 13 RGB/depth/mask records, calibrated
intrinsics, and the timestamped `base_link -> camera_color_optical_frame` 4x4
matrix saved with each view. Reconstruction is an asynchronous post-scan job;
mesh failure does not change the already completed arm shutdown result.

## 10. Verify the installation

Run the software checks:

```bash
cd ~/Piper_arm
./verify_installation.sh
```

Every line must report `PASS`. This verifies the OS, Foxy, the canonical PiPER and RealSense
overlays, the isolated Tesseract runtime, Python imports and versions, and ROS package discovery.
Also validate the isolated AI environment:

```bash
./AI_perception_tests/groundingdino_test/check_env.sh
```

Run a live camera test:

```bash
./L515_camera/start_l515_camera.sh
```

Wait for `RealSense Node Is Up!`. In another terminal:

```bash
cd ~/Piper_arm
export ROS_DOMAIN_ID=42
source L515_camera/source_l515_environment.sh
ros2 topic list | sort
```

At minimum, verify these topics exist:

```text
/camera/color/image_raw
/camera/color/camera_info
/camera/aligned_depth_to_color/image_raw
/camera/depth/image_rect_raw
/camera/imu
```

Stop the camera with `Ctrl+C` in its terminal.

## 10. Start the supported GUI system

Camera and read-only perception commands are listed in order in
[`OPERATOR_COMMANDS.md`](OPERATOR_COMMANDS.md#read-only-l515-perception-runtime). The camera workflow
does not move the arm.

Use four terminals. Start the real PiPER driver without automatically enabling motion:

```bash
cd ~/Piper_arm
./start_piper.sh
```

Leave that terminal running. Wait until it reports that the PiPER node has started, then use a second
terminal for the accepted hand-eye transform:

```bash
cd ~/Piper_arm
./L515_camera/run_hand_eye_tf.sh
```

Start the L515, camera timestamp watchdog, GroundingDINO, SAM2, and geometry pipeline in the third
terminal:

```bash
cd ~/Piper_arm
./L515_camera/run_gpu_vision_pipeline.sh
```

When camera and perception health are ready, start the GUI in the fourth terminal:

```bash
cd ~/Piper_arm
./start_gui.sh
```

Do not use `scripts/robot/enable_piper.sh` for the supervised scan workflow. Clear and support the workspace,
prepare an emergency-stop method, then use the GUI Enable button. The Acquire & Scan tab performs a
separately approved rough-coordinate acquisition followed by a separately approved correlated
13-view plan. Neither approval is reusable.

At completion or before shutdown, press GUI **Disable**. The GUI must first command the fresh
current-feedback pose, prove target error no greater than 0.025 rad and sample motion no greater than
0.005 rad continuously for one second, and only then report a successful motor disable. If that
eight-second proof fails, the motors remain enabled; do not stop the driver until the arm is safely
supported and Disable succeeds.

After the arm is disabled, stop the managed scan stack from the GUI, then stop camera and perception:

```bash
cd ~/Piper_arm
./L515_camera/stop_gpu_vision_pipeline.sh
```

Close the GUI and stop the hand-eye and driver terminals only after the disable acknowledgement.

## Troubleshooting

### Missing `diagnostic_updater`

```bash
sudo apt-get install ros-foxy-diagnostic-updater
```

Then rerun `./L515_camera/build_realsense_ws.sh`.

### Missing `libusb.h` or `config.h`

```bash
sudo apt-get install libusb-1.0-0-dev libudev-dev
```

Then rerun `./L515_camera/build_realsense_ws.sh`.

### Camera access denied

Rerun `./L515_camera/install_l515_host_fixes.sh`, disconnect the L515, wait 10 seconds, reconnect it,
and run `./L515_camera/diagnose_l515_usb.sh`.

### ROS camera topics are absent

Use `ROS_DOMAIN_ID=42` in every terminal communicating with the camera and source
`L515_camera/source_l515_environment.sh` before running `ros2` commands.

### `ethtool` or `can-utils` is missing

Rerun the host dependency installer, or install both packages directly:

```bash
sudo apt-get update
sudo apt-get install -y ethtool can-utils
```

Then rerun `./verify_installation.sh` before starting PiPER.

### `/enable_srv` is unavailable

Run `./start_piper.sh` first and leave it running. Wait for the PiPER node to start, then run
`./start_gui.sh` with the same `PIPER_ROS_DOMAIN_ID` value. For supervised operation, enable and
disable only through the GUI.

### Step 2 reports a service timeout

Restart the GUI so it loads the current code, and confirm that every terminal uses
`ROS_DOMAIN_ID=42`, `fastdds_gui_udp_only.xml`, `RMW_FASTRTPS_USE_QOS_FROM_XML=0`, and
`ROS_LOCALHOST_ONLY=0`. Always start the GUI through `./start_gui.sh`; do not source an old
repository-root colcon overlay. Step 2 retries its immutable request through a fresh Foxy client
endpoint after eight seconds and preserves the session for an operator retry after two failures.

### Tesseract readiness is blocked

Rerun:

```bash
export PATH="$HOME/.local/bin:$PATH"
./motion_planning/tesseract/qualify_rootless_worker.sh
source source_piper_foxy_environment.sh
```

Do not bypass acquisition or multiview readiness. The worker has a 150-second internal budget,
checks it before every OMPL attempt and during adaptive collision validation, and reserves five
seconds to serialize its response before the bridge's 180-second timeout.

### Generated files

ROS build directories, RealSense sources, Python environments, model checkouts, weights, captures, and
logs are intentionally ignored. Recreate them with the installers and build commands above; do not
commit them.
