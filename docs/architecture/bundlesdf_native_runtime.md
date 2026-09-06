# Dedicated native BundleSDF runtime

This follows the existing cuRobo integration pattern: an explicit Python 3.10
interpreter, local CUDA compiler, and separately installed native libraries.
Docker is optional. ROS Foxy Python 3.8, the cuRobo environment, mission processes,
and planner selection are not modified.

The setup creates these default paths outside the repository:

- `~/.venvs/piper-bundlesdf/`: Python environment.
- `~/.local/share/piper-bundlesdf/upstream/`: pinned BundleSDF checkout and extensions.
- `~/.local/share/piper-bundlesdf/native/usr/`: privately extracted/built native libraries.
- `~/.local/share/piper-bundlesdf/python-requirements.lock`: installed Python versions.
- `~/.local/share/piper-bundlesdf/native-debs.lock`: downloaded native package hashes.

```bash
cd /home/prl/Piper_arm
bash tools/bundlesdf/setup_native.sh
# Normal wrapper now selects this native runtime by default:
python3 tools/bundlesdf/run.py /absolute/session.complete --output /tmp/bundlesdf-new-run
# Docker remains explicitly available:
# python3 tools/bundlesdf/run.py /absolute/session.complete --output /tmp/another-run --runtime docker
```

Setup variables:

| Variable | Default |
|---|---|
| PIPER_BUNDLESDF_RUNTIME | ~/.local/share/piper-bundlesdf |
| PIPER_BUNDLESDF_ENV | ~/.venvs/piper-bundlesdf |
| PIPER_BUNDLESDF_BASE_PYTHON | ~/Piper_arm/AI_perception_tests/groundingdino_test/envs/grounded_sam2_py310/bin/python |
| PIPER_BUNDLESDF_CUDA_HOME | ~/.local/cuda-12.8 |
| PIPER_BUNDLESDF_BUILD_JOBS | 4 |
| PIPER_BUNDLESDF_PYTHON | ~/.venvs/piper-bundlesdf/bin/python (runtime launcher override) |

The launcher replaces `PYTHONPATH` and `LD_LIBRARY_PATH` so a sourced Foxy overlay
cannot contaminate the offline process. Reusing the existing CUDA compiler is
read-only. Setup downloads/extracts YAML, ZeroMQ and GLUT packages into its private
prefix; it does not run apt installation. Existing host PCL, Boost and graphics
build dependencies are checked before compilation.

Versions: BundleSDF `ffa67d425240b5b76d2e387a7dd3d3735a7cf1a1`, PyTorch 2.6.0+cu124,
torchvision 0.21.0, Kaolin 0.18.0 with the **matching** torch-2.6.0/cu124 wheel,
CUDA compiler 12.8, Eigen 3.4.0 and OpenCV 4.11.0 with CUDA modules. Upstream's
Dockerfile uses a Kaolin wheel for torch 2.5.1 despite installing torch 2.6; that
ABI mismatch is avoided. Source archives are SHA256 checked. Python dependencies
are declared in `native-requirements.txt`; the exact installed transitive versions
are recorded in the external lock file and copied into each native run.

PyTorch3D v0.7.9 is pinned to `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba` and installed
using its supported `PYTORCH3D_NO_EXTENSION=1` switch: BundleSDF imports only its
pure tensor Lie-group transforms. This does not supply PyTorch3D CUDA renderers or
ops. BundleSDF's own `common`, `gridencoder`, and BundleTrack extensions are built
and exercised separately; they are not mocked or replaced.

The small tracked `patches/native-build.patch` records upstream build adjustments.
The base upstream commit remains pinned; the wrapper saves `upstream-local.patch`
with each run, plus exact configuration, command and dependency records. Native
libraries and generated builds never enter Git. Upstream's noncommercial
research/evaluation licence still applies.

LoFTR weights come from the Kornia-maintained mirror at immutable revision
`0fb3b456cf176746709c38fa9cf80695a1c82dac`, with SHA256
`21f5bec5968178e8bc8b7633441836fe5de4f47d861dd2cd7dc38e271b0479ec`.
They are stored only under the external checkout. The official milk example
remains the preferred first smoke dataset; a quota-exceeded download must not be
mistaken for a valid archive.

For an independently reproducible synthetic smoke input:

```bash
python3 tools/bundlesdf/synthetic_smoke.py --output /tmp/bundlesdf-synthetic-new --frames 30
# The command prints its native session path. Use that exact path below.
python3 tools/bundlesdf/run.py /tmp/bundlesdf-synthetic-new/SESSION.complete \
  --output /tmp/bundlesdf-synthetic-result
```

This renders a fixed textured 14 cm cube using analytic pinhole/AABB intersections,
with 30 camera poses and matching raw depth, masks, intrinsics and camera-to-base
transforms. It enters the native recording writer through its real acceptance
gates. It is labelled synthetic and is not a physical arm/camera test. Successful
reconstruction requires subprocess completion, a mesh with vertices/faces, and
estimated object trajectory files. Input validation or GPU imports alone are not
reported as reconstruction success.

See [executed commands, failures and GPU evidence](../experiments/bundlesdf_native_validation.md).

Both BundleTrack and mycuda must use private Eigen 3.4.0. Host Eigen 3.3.7
compiled a no-op UV kernel on this workstation. Setup diagnostics execute a
known-triangle UV test as well as real Kaolin GPU operations and mapping API checks.
The wrapper repeats native diagnostics before reconstruction and validates finite
mesh coordinates, face indices, rigid `T_camera_object` poses and texture output.
These structural checks do not qualify real-object accuracy.
