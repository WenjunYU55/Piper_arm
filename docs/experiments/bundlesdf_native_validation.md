# Native BundleSDF validation — 2026-09-07

This follow-up replaces the earlier Docker installation blocker with an isolated
native runtime, following the existing cuRobo explicit-interpreter approach.
Development started on `feature/bundlesdf-rootless` from clean `main` / `origin/main`
commit `8e591e1f69f841d82bcacd85ff1513dedb7b2b66`. The [exact changed-file list](bundlesdf_native_files.txt) is tracked alongside this record.
No ROS package, mission process,
planner selection, controller or camera configuration changed in this follow-up.
The active saved planner remains cuRobo.

## Environment and reproducibility

See [installation commands and complete setup variables](../architecture/bundlesdf_native_runtime.md).
BundleSDF base commit: `ffa67d425240b5b76d2e387a7dd3d3735a7cf1a1`.
Python 3.10.20, Torch 2.6.0+cu124, Kaolin 0.18.0, CUDA compiler 12.8.93,
RTX 3090, C++ OpenCV 4.11.0 CUDA, Python OpenCV 4.10.0, private Eigen 3.4.0.
Native environment, weights, source/build trees and outputs stay outside Git.
The complete installer rerun finished with exit 0, including GPU diagnostics.
The tracked build patch and exact installed dependency records accompany each run.
Upstream algorithm sources are unchanged.

Commands executed from the feature worktree:

```bash
bash tools/bundlesdf/setup_native.sh
source /home/prl/Piper_arm/piper_ros_foxy/install/setup.bash
python3 -m pytest -q tests/reconstruction
python3 -m flake8 --max-line-length=99 \
  tools/bundlesdf/native_diagnostics.py tools/bundlesdf/output_validation.py \
  tools/bundlesdf/run.py tools/bundlesdf/synthetic_smoke.py \
  tests/reconstruction/test_bundlesdf_native.py
bash -n tools/bundlesdf/setup_native.sh tools/bundlesdf/python_native.sh
```

Reconstruction tests: **70 passed**, including the actual DDS replay on an isolated
ROS domain, native launcher isolation, analytic camera/depth geometry and output
corruption checks. Lint and shell syntax passed. An initial unsourced invocation
failed collection; prepending source-only packages then failed generated-message
imports. Sourcing the built workspace fixed both environment errors. No tests
were disabled. Existing baseline/regression results remain in
[the initial implementation record](reconstruction_recording_validation.md);
colcon and the full planner suite were not rerun for this offline-only follow-up.

## GPU smoke input and exact command

```bash
python3 tools/bundlesdf/synthetic_smoke.py \
  --output /tmp/piper-bundlesdf-native-evidence/synthetic --frames 30
python3 tools/bundlesdf/run.py \
  /tmp/piper-bundlesdf-native-evidence/synthetic/reconstruction_ae145bd3e9f743d5a2d8b1f9c95b9009.complete \
  --output /tmp/piper-bundlesdf-native-evidence/smoke-3
```

Input is an explicitly labelled analytic 14 cm textured cube, 30 RGB-D views,
640×480, calibrated pinhole projection, millimetre uint16 aligned depth, binary
masks and known camera-to-base transforms. Actual native recorder gates accepted
**30**, rejected **0**, dropped **0**, with **2** synthetic mission captures.
The pinned upstream data loader independently read all 30 converted frames.
This is synthetic GPU reconstruction evidence, not physical-system validation.

## Failures found and fixed

- Missing private native headers/libraries and mismatched Python CMake cache were
  resolved through explicit dependency paths, without changing host packages.
- Old pybind11 headers produced a `MapFrame` lacking `.keys()`: first reconstruction
  failed on frame 1. The build now uses pybind11 2.13.6 and diagnostics check the API.
- Second reconstruction completed tracking and mesh export, but output inspection
  found only **one coloured texture pixel**. The UV kernel silently did no work
  when compiled against host Eigen 3.3.7. A sentinel-output triangle probe and CUDA
  disassembly reproduced it. Building against private Eigen 3.4.0 produced the
  expected UV `[25,25]`. Installation diagnostics now execute that test; the wrapper
  rejects the observed degenerate-texture failure rather than trusting exit status.
- Official milk sample download remains quota-blocked on Google Drive, including
  curl and gdown attempts. The official example has **not** been reconstructed.
  Its absence is not described as a successful supplied-example test.

Logs are under `/tmp/piper-bundlesdf-native-evidence/`, including failed attempts,
installer replays, GPU diagnostics, pytest output and per-run `bundlesdf.log`.
Temporary evidence is local and may be removed by normal `/tmp` cleanup.

## Limits

No Piper arm, camera, motion, tracking-load or real-object accuracy qualification
was performed. Limited synthetic viewpoints do not establish watertightness,
complete hidden surfaces or metric reconstruction accuracy. The output validator
checks usable artifact structure, finite geometry, rigid object-to-camera poses
and nondegenerate texture; it does not certify reconstruction quality.

## Final executed result

The third reconstruction command exited **0** after all 30 tracked frames and
2,001 final SDF optimization iterations. The actual OBJ has **93,530 vertices**
and **119,499 faces**; all **30** saved object-to-camera transforms passed rigid
transform validation. Its 512×512 texture has **82,045 nonzero pixels**, and a
rendered preview has 60,376 nonzero pixels. The preview was visually inspected.
The exported mesh is not watertight; no metric-accuracy claim is made.

Mesh SHA256: `c1cb0db2d176b89db9e760f9395ac7513ba9aad501adafad94d62114898b49dd`.

A retained copy outside Git is at:

`/home/prl/.local/share/piper-bundlesdf/evidence/2026-09-07-native-smoke/`

- `result/textured_mesh.obj`, `material.mtl`, `material_0.png`: mesh and texture.
- `result/ob_in_cam/*.txt`: 30 estimated `T_camera_object` transforms.
- `mesh-preview.png`, `mesh-inspection.json`, `result_validation.json`: output checks.
- `input/`, `validation.json`: converted input and validation report.
- `bundlesdf.log`, installer/test/diagnostic logs: execution evidence.
- `run.json`, upstream configurations, `upstream-local.patch`, dependency locks and
  `runtime.json`: exact command, pin and native-build provenance.

The upstream texture exporter still warns when casting unobserved texture texels;
these become black. The corrected CUDA probe and nondegenerate output check pass.
This is a successful synthetic GPU reconstruction, not a supplied-example or
physical-camera reconstruction. The final wrapper also adds the already-executed
native diagnostic as a preflight; that sequencing addition was made while this
smoke command was running, and does not imply a fourth full reconstruction run.
