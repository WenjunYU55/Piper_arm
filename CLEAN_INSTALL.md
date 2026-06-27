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
./install_host_dependencies.sh
```

The installer installs build, ROS, GUI, CAN, and Python dependencies, including `can-utils` and
`ethtool`, which are required by `start_piper.sh`. It pins `piper_sdk==0.6.1` and
`python-can==4.5.0`; do not replace these with Ubuntu's older `python3-can` version.

## 4. Build the PiPER workspace

```bash
cd ~/Piper_arm/piper_ros_foxy
source /opt/ros/foxy/setup.bash
colcon build --symlink-install
source install/setup.bash
cd ..
```

The build must finish with all four packages successful: `piper_description`, `piper_msgs`,
`piper_mobile_manipulation`, and `piper`.

## 5. Build and configure the L515

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

## 6. Configure the PiPER CAN adapter

Connect the USB-CAN adapter and arm, then identify its interface:

```bash
ip -brief link
```

For the default `can0` interface and PiPER's 1 Mbps bitrate:

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
ip -details link show can0
```

The result must contain `UP`, `can state ERROR-ACTIVE`, and `bitrate 1000000`. Use
`PIPER_CAN_PORT=can1` with runtime scripts if the adapter appears as `can1`.

CAN configuration is not persistent across reboot. `start_piper.sh` checks and configures the selected
interface, requesting `sudo` when necessary.

## 7. Verify the installation

Run the software checks:

```bash
cd ~/Piper_arm
./verify_installation.sh
```

Every line must report `PASS`. This verifies the OS, Foxy, both workspace overlays, Python imports and
versions, and ROS package discovery.

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

## 8. Start the system

Camera and read-only perception commands are listed in order in
[`OPERATOR_COMMANDS.md`](OPERATOR_COMMANDS.md#read-only-l515-perception-runtime). The camera workflow
does not move the arm.

To start the real PiPER driver without automatically enabling motion:

```bash
cd ~/Piper_arm
./start_piper.sh
```

Leave that terminal running. Wait until it reports that the PiPER node has started, then use a second
terminal to enable the arm. Only enable it after the workspace is clear and an emergency-stop method is
available:

```bash
./enable_piper.sh
```

## 9. Optional AI environment

Do not install GroundingDINO, SAM2, or their Python dependencies into Foxy's Python 3.8 environment.
Provide Python 3.10 through Conda, pyenv, or another isolated distribution, then run:

```bash
cd ~/Piper_arm
./AI_perception_tests/groundingdino_test/setup_cpu_env.sh
```

This creates the ignored environment under `AI_perception_tests/groundingdino_test/envs/`, checks out
the pinned Grounded-SAM-2 source, downloads model weights, and validates imports. NVIDIA and Jetson
installations require PyTorch builds matching their exact CUDA or JetPack version.

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
`./enable_piper.sh` in a second terminal with the same `PIPER_ROS_DOMAIN_ID` value.

### Generated files

ROS build directories, RealSense sources, Python environments, model checkouts, weights, captures, and
logs are intentionally ignored. Recreate them with the installers and build commands above; do not
commit them.
