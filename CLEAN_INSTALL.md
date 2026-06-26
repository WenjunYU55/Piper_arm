# Clean Installation and Device Profiles

This guide reproduces the current read-only L515 perception stack on a fresh device. Real PiPER arm
motion remains disabled in the temporal/heavy-model workflow.

For day-to-day command usage after installation, see [`OPERATOR_COMMANDS.md`](OPERATOR_COMMANDS.md).

## Supported control host

- Ubuntu 20.04
- ROS 2 Foxy installed at `/opt/ros/foxy`
- Python 3.8 for ROS nodes
- Intel RealSense L515
- Python 3.10 in a separate environment for GroundingDINO/SAM2

ROS 2 Foxy is end-of-life. Use an isolated compatible host or container; changing Ubuntu or ROS versions
requires a port and new validation.

## Clone and host dependencies

The repository can be cloned anywhere. Runtime scripts derive the repository root automatically.

```bash
git clone https://github.com/WenjunYU55/Piper_arm.git
cd Piper_arm
chmod +x install_host_dependencies.sh verify_installation.sh
./install_host_dependencies.sh
```

## Build PiPER ROS packages

```bash
source /opt/ros/foxy/setup.bash
cd piper_ros_foxy
colcon build --symlink-install
source install/setup.bash
cd ..
```

## Optional real-arm tools

The repository includes explicit `.sh` / `.py` tools for real PiPER operation:

```bash
./start_piper.sh
./enable_piper.sh
./disable_piper.sh
./reset_piper.sh
./reset_arm.sh
./start_gui.sh
./calibrate_bounds.sh
```

These are intentionally separate from the read-only L515 perception workflow. `start_piper.sh` does not
auto-enable the arm by default, but after the arm is enabled, `reset_piper.sh`, `reset_arm.sh`, the GUI,
and direct joint-topic commands can move the real robot.

No-extension wrappers such as `start_piper`, `reset_arm`, or `start_gui` are intentionally omitted. Use
the `.sh` names directly so a fresh clone is explicit and easier to audit.

## Build the pinned L515 driver

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

The source pair is pinned to librealsense `v2.50.0` and realsense-ros `4.0.4` with the checked-in Foxy/L515
patch.

## CPU AI environment

Do not install AI packages into ROS Foxy's Python 3.8 environment. Provide `python3.10` using Conda,
pyenv, or another maintained isolated distribution, then run:

```bash
chmod +x AI_perception_tests/groundingdino_test/setup_cpu_env.sh
./AI_perception_tests/groundingdino_test/setup_cpu_env.sh
```

This creates `AI_perception_tests/groundingdino_test/envs/grounded_sam2_py310`, installs pinned CPU
dependencies, checks out the pinned Grounded-SAM-2 revision, downloads GroundingDINO and SAM2.1 Hiera
Tiny weights, and verifies imports and assets. These generated dependencies are intentionally ignored by
Git.

Run the software tests and rebuild the ROS package:

```bash
cd AI_perception_tests
python3 -m unittest -v test_temporal_tracking.py test_heavy_model_worker.py
cd ../piper_ros_foxy
source /opt/ros/foxy/setup.bash
colcon build --packages-select piper_mobile_manipulation
cd ..
```

## Device selection

| Device | Live target tracking | Heavy perception | Notes |
|---|---|---|---|
| CPU-only x86 host | Adaptive Lab appearance, depth and Lucas-Kanade; calibrated HSV is fallback only | Event-driven GroundingDINO + SAM2 image masks on CPU | Current validated live configuration |
| NVIDIA desktop GPU | Same lightweight live tracker | Set `PIPER_HEAVY_DEVICE=cuda` in a CUDA-compatible isolated environment | Validate CUDA/PyTorch versions locally |
| Jetson today | Same lightweight tracker until the Jetson path is validated | Isolated CUDA heavy worker with JetPack-compatible PyTorch | Do not install AI dependencies into Foxy Python |
| Jetson planned | SAM2 video target and obstacle mask propagation | GroundingDINO for initialization/reacquisition | Offline quality validated; live Jetson integration is not yet implemented |

Do not describe fixed HSV as the primary CPU tracker. The current CPU path learns a robust Lab appearance
model from each SAM2 seed and uses HSV only when an appearance model is unavailable.

The SAM2.1 Hiera Tiny CPU video benchmark achieved mean IoU `0.917` but only `0.113 FPS` and used about
`2.65 GB` RAM. This is why SAM2 video is not enabled in the CPU live loop.

### Jetson AI environment

JetPack determines the compatible CUDA, PyTorch and torchvision versions. Install the NVIDIA-supported
PyTorch build for the exact JetPack release first; do not install the CPU `requirements_ai.txt` file on a
Jetson. Then install the remaining Grounded-SAM-2 dependencies and the pinned source revision documented
in `AI_perception_tests/groundingdino_test/fetch_ai_assets.sh`.

For the existing image worker after CUDA validation:

```bash
export GROUNDED_SAM2_PYTHON=/path/to/jetson/python
export PIPER_HEAVY_DEVICE=cuda
./L515_camera/run_heavy_model_worker.sh
```

If ROS/Foxy and the AI worker run on different machines, both must mount the same filesystem spool and use
the same value:

```bash
export PIPER_HEAVY_REFRESH_SPOOL=/mnt/shared/piper_heavy_refresh
```

The current boundary is filesystem-based; it is not a network RPC service.

## Read-only runtime

Use separate terminals, in this order:

```bash
./L515_camera/start_l515_camera.sh
```

```bash
./L515_camera/run_heavy_refresh_bridge.sh
```

```bash
./L515_camera/run_heavy_model_worker.sh
```

```bash
export ROS_DOMAIN_ID=42
source L515_camera/source_l515_environment.sh
ros2 topic echo /piper/heavy_refresh_status
```

```bash
./L515_camera/run_temporal_tracking_readonly.sh /piper/heavy_target_mask
```

```bash
./L515_camera/view_l515_opencv.sh /piper/temporal_tracking_debug_image
```

Heavy snapshot mask topics:

```text
/piper/heavy_target_mask
/piper/heavy_obstacle_mask
/piper/candidate_movable_obstacle_mask
/piper/unsafe_obstacle_mask
```

Candidate-movable masks are advisory only. Hands, people, fingers, wires, cables, generic tools, and
unknown objects remain blocked and must never become manipulation targets.

## Generated files not committed

- ROS `build/`, `install/`, and `log/`
- RealSense source/build workspace
- Python environments and caches
- Grounded-SAM-2 checkout
- Model weights and checkpoints
- captures, datasets, benchmark images, and AI output reports
- `/tmp/piper_heavy_refresh`

Run `fetch_ai_assets.sh` and the build steps above to recreate them.
