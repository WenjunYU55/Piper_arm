# Validated environments

This page records facts that have been checked on real devices. It separates
the proven workstation from planned deployment targets so an untested platform
is never presented as supported.

## Reference workstation

Verified on **28 August 2026** at `/home/prl/Piper_arm`.

### Hardware

| Component | Verified value |
|---|---|
| Architecture | x86_64 |
| Host | Ubuntu 20.04.6 LTS (Focal), Linux 5.15.0-139-generic |
| CPU | Intel Core i9-10900X, 10 cores / 20 threads |
| Memory | 125 GiB usable RAM |
| GPU | NVIDIA GeForce RTX 3090 |
| GPU memory | 24,576 MiB |
| GPU compute capability | 8.6 |
| NVIDIA driver | 570.133.07 |
| Driver-reported CUDA capability | 12.8 |
| CUDA compiler | 12.8.93 at `/home/prl/.local/cuda-12.8/bin/nvcc` |
| RGB-D camera | Intel RealSense L515, firmware 01.05.08.01 |
| Arm interface | SocketCAN `can0`, UP / ERROR-ACTIVE, 1 Mbps |
| Docker | Not installed and not part of the reference install |

The USB and CAN devices were present during this audit. Their presence does not
replace the mission's live freshness, health, transform, collision, and motor
authority checks.

### Software environments

The system deliberately uses separate Python environments.

| Surface | Verified value |
|---|---|
| ROS | ROS 2 Foxy at `/opt/ros/foxy` |
| ROS/system Python | Python 3.8.10 |
| PiPER SDK | 0.6.1 |
| python-can | 4.5.0 |
| RealSense | librealsense 2.50.0, realsense-ros 4.0.4 |
| Perception Python | Python 3.10.20 |
| PyTorch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| PyTorch CUDA / cuDNN | CUDA 12.8 / cuDNN 9.19.0 |
| GroundingDINO / SAM 2 | Imports and CUDA device check passed |
| Reconstruction | Open3D 0.19.0 in an isolated environment |
| Tesseract | 0.35.0.6 isolated rootless runtime |
| cuRobo | 0.7.8, commit `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` |
| Warp | 1.11.1 |

GroundingDINO/SAM 2 and cuRobo share compatible host GPU facts, but they do not
run inside Foxy's Python interpreter. The planner workers communicate through
validated files and cannot command the robot.

### Verification evidence

These checks passed on the reference workstation:

```bash
cd /home/prl/Piper_arm
./verify_installation.sh
./AI_perception_tests/groundingdino_test/check_env.sh
```

`verify_installation.sh` passed its host, Foxy, PiPER workspace, RealSense
workspace, command, Python import/version, ROS-package, interface, launcher,
and Tesseract-runtime checks. The perception check imported GroundingDINO and
SAM 2, found the pinned assets, reported PyTorch CUDA availability, and selected
the RTX 3090.

The following read-only commands record the main host facts:

```bash
. /etc/os-release && printf '%s %s\n' "$NAME" "$VERSION_ID"
uname -srmo
lscpu | grep -E '^(Architecture|CPU\(s\)|Model name):'
free -h
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
/home/prl/.local/cuda-12.8/bin/nvcc --version
ip -details link show can0
```

### What this proves

- The checked-in installation verifier passes against the installed software.
- The pinned perception environment imports and sees the reference GPU.
- The isolated cuRobo runtime imports and can perform command-free GPU planning.
- The L515 and CAN adapter are visible to the host.

### What this does not prove

- A clean-room reinstall was not performed as part of every documentation
  update; the clean-install script chain remains the reproducibility procedure.
- A software check does not qualify a new mount, collision model, floor profile,
  arm, cable routing, speed, or workspace for physical motion.
- cuRobo's current 69-sphere moving-link model is hardware-qualified for the
  supervised 5% target-scan scope reported by the operator on 2026-09-02. Its
  non-conservative geometry and measured coverage gaps remain limitations; it
  is not collision-equivalent to Tesseract.
- The tracked-robot ground profile is not physically qualified.

## Jetson deployment target

Jetson is **planned and unverified**. Do not use the workstation instructions as
if they were a proven Jetson install.

| Field | Status |
|---|---|
| Jetson model | TBD |
| JetPack / L4T | TBD |
| Ubuntu | TBD |
| CUDA / cuDNN / TensorRT | TBD |
| ROS deployment method | TBD |
| PyTorch distribution | TBD; must use a Jetson-compatible NVIDIA build |
| GroundingDINO / SAM 2 | Not benchmarked |
| cuRobo | Compatibility not validated |
| Tesseract | Current rootless image is amd64 and is not Jetson-compatible |
| L515 USB and firmware behavior | Not validated |
| SocketCAN adapter behavior | Not validated |
| Memory, thermals, and sustained latency | Not measured |
| End-to-end mission | Not tested |

Known portability blockers include the current Tesseract amd64 root filesystem,
the `linux-64` Micromamba bootstrap, and standard x86 CUDA/PyTorch wheels. A
Jetson release must provide architecture-correct worker environments without
modifying or weakening the common mission and safety path.

### Evidence required before marking Jetson supported

1. Record the exact Jetson model, JetPack/L4T, Ubuntu, kernel, CUDA, cuDNN,
   TensorRT, ROS, Python, PyTorch, and firmware versions.
2. Reproduce the ROS workspace, camera, perception, reconstruction, and chosen
   planner environments from a clean device.
3. Pass ordinary unit and contract tests without GPU planner imports in Foxy.
4. Pass command-free camera, perception, planner, process-cleanup, and dataset
   tests under sustained load.
5. Validate USB bandwidth, CAN recovery, thermals, memory pressure, and mission
   latency.
6. Repeat collision-model and staged physical qualification at the documented
   low-speed gate before any wider use.

Only after that evidence exists should this page gain a dated Jetson row marked
validated.

## Updating this page

Record the observation date, exact commands, device versions, and pass/fail
results. Mark unknowns as unknown. Never infer support from upstream marketing,
a package import, or a successful plan on another machine.
