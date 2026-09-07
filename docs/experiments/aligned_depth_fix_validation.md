# Aligned-depth frameset correction — 2026-09-07

Started from clean `main` at `55f9ed02ab36413bb1116fb425e2aa998bc7c44e` after fetching
remote refs. Development branch: `fix/aligned-rgbd-framesets`.

## Defect and implementation

Installed RealSense ROS 4.0.4 (`8abb4657c0add15f87b0edbfb67eaba2c1c2c439`)
published any depth returned by enabled alignment as color-aligned. SDK 2.50.0
(`c94410a420b74e5fb6a414bd12215c05ddd82b69`) passes depth-only framesets through
without alignment. The resulting pixels could therefore have native-depth
geometry despite the topic and frame ID claiming color geometry.

The tracked camera patch requires color from the same original frameset, matching
processed-depth/color dimensions, finite intrinsics and distortion, and identity
extrinsics. Rejected aligned output generates a five-second throttled warning;
native depth publication continues. Build verifies the patch or fails on source
drift. No camera stream profiles, TF, robot activation, motion, NBV or stationary
native-depth/confidence median algorithm changed.

Live reconstruction uses bounded approximate candidate delivery for accounting,
but the shared admission gate saves only exact timestamp pairs. Native validation,
offline extraction and pre-SAM2 validation enforce the same rule. Approximate first
pins fail preparation; later ambiguous candidates have a named rejection counter.
This is conservative recovery for the legacy ROS archive, not a claim that exact
timestamps alone certify calibration. Old contaminated native datasets now fail
validation with guidance to recover the source archive.

## Executed checks

- Baseline reconstruction/GUI suite: **271 passed in 24.82 s**.
- Baseline camera build failed before edits: missing `diagnostic_updater` development
  package. Downloaded the existing Foxy package with `apt-get download` and extracted
  it under `/tmp/piper-camera-deps/root` for build-time use. No system package or
  driver process was started. The project's existing dependency installer already
  lists this package; normal builds should install the documented dependencies.
- Final reconstruction/GUI suite: **273 passed in 26.59 s**. Package CTest: **48/50**; unchanged **99 flake8 and 10 pep257** findings remain.
- Existing camera compatibility patches initially failed application due to CRLF/context whitespace and stale hunk counts. The build script now handles whitespace and recounts hunks, verifies already-applied patches, and refuses incompatible source. Their existing contents were not changed. Final camera rebuild with both compatibility patches and the guard passed in **9.72 s**.
- Actual SDK software-device regression passed for complete pair, missing color,
  missing depth and native-depth/color calibration mismatch. The test includes the
  same header installed by the production patch; it is not an alternate guard.
- Patched `realsense2_camera` build passed in **17.7 s**. `ldd` reports no missing
  library or temporary runtime dependency from the staged header-only updater.
- Native validation rejects the historical contaminated 2,041-frame dataset with source-archive recovery guidance; the recovered 1,633-frame dataset passes.
- Serialized bag regression verifies 11 ms nearby depth rejection and refusal of
  an ambiguous accepted initial pin. Actual ROS replay tests exercise rejection
  accounting through the public sensor interfaces.

Commands:

```bash
source piper_ros_foxy/install/setup.bash
python3 -m pytest -q tests/reconstruction tests/gui reconstruction/test_gui_support.py
L515_camera/test_aligned_depth_guard.sh
source L515_camera/source_l515_environment.sh
export CMAKE_PREFIX_PATH="/tmp/piper-camera-deps/root/opt/ros/foxy:$CMAKE_PREFIX_PATH"
cd L515_camera/realsense_ws
colcon build --packages-select realsense2_camera --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
cd ../../piper_ros_foxy
colcon build --packages-select piper_mobile_manipulation --symlink-install
colcon test --packages-select piper_mobile_manipulation --event-handlers console_cohesion+
```

No physical camera or arm validation occurred. The new installed library loads on
the next normal camera startup. Stationary hardware validation must confirm valid
aligned output, continuity under missing color, and continued native-depth capture.

## Real recording recovery and GPU rerun

Source: `datasets/mission_recordings/scan_recording_f62ed65bbeb4459cb35a9b36e64869f9.closed`.
Corrected extractor retained **1,633** pairs; rejected **408 ambiguous aligned-depth
pairs** and **41 unmatched RGB frames**. It preserves the three accepted mission
captures and incomplete/unknown terminal status. Actual offline SAM2 accepted all
1,633 frames with no quality rejections in **153.847 s**, retaining at most 16 temporal images. No offline backpressure drops occurred.

```bash
source piper_ros_foxy/install/setup.bash
python3 tools/bundlesdf/extract_video.py \
  datasets/mission_recordings/scan_recording_f62ed65bbeb4459cb35a9b36e64869f9.closed \
  /home/prl/.local/share/piper-bundlesdf/evidence/aligned-fix-extracted
tools/bundlesdf/python_sam2.sh tools/bundlesdf/segment_video.py \
  /home/prl/.local/share/piper-bundlesdf/evidence/aligned-fix-extracted \
  /home/prl/.local/share/piper-bundlesdf/evidence/aligned-fix-session
python3 tools/bundlesdf/reconstruct_job.py \
  /home/prl/.local/share/piper-bundlesdf/evidence/aligned-fix-session \
  --output /home/prl/.local/share/piper-bundlesdf/evidence/aligned-fix-reconstruction \
  --job-dir /home/prl/.local/share/piper-bundlesdf/jobs/aligned-fix-reconstruction \
  --frame-stride 1 --allow-incomplete
```

BundleSDF pin remains `ffa67d425240b5b76d2e387a7dd3d3735a7cf1a1`. The fresh full-sequence
GPU job completed successfully on 2026-09-07 at approximately 19:06:58 BST,
including final global refinement. Output: **4,506 vertices, 6,299 faces**, OBJ/MTL/PNG,
1,633 saved pose matrices, 75,284 nonzero texture pixels. The log contains **zero
explicit tracking failures and zero reference-recovery messages**, versus 15 and 24
in the contaminated run. This is not a claim that every estimated pose is accurate.

Two opposing offscreen views were inspected. The observed surfaces are now
recognizably cube-like and substantially more coherent than the previous separated
sheets. Holes/open surfaces and dark texture regions remain. Structural output and
viewer loading **PASS**; complete mesh/dimensional accuracy remains **unqualified**.
The mesh extents are 50.494 × 54.505 × 47.728 mm; these are measured output extents,
not validation against a calibrated physical reference. No cosmetic hole filling
or superposition correction was applied.

Artifacts reside under
`/home/prl/.local/share/piper-bundlesdf/evidence/aligned-fix-reconstruction/`:
`result/textured_mesh.obj`, material/texture files, `result/ob_in_cam/`,
`mesh-preview.png`, `mesh-preview-opposite.png`, `mesh-inspection.json`,
`result_validation.json`, configuration/provenance and `bundlesdf.log`.
Mesh SHA256: `6b26d68d79f715855032cb862dbcd8151d3d02f6c0c16652a24f69decab10fae`.
The actual viewer loader passed `--check-only`.

Exact upstream command is in `run.json`: isolated `python_native.sh run_custom.py
--mode run_video --video_dir <output>/input --out_folder <output>/result
--use_segmenter 0 --use_gui 0 --debug_level 2`. No BundleSDF settings, pose priors or
model weights were changed for this comparison.
Logs are `/tmp/piper-aligned-*.log` and durable job/output logs. The official example
remains download-quota-blocked as previously documented; it was not executed.

## Change inventory

Implementation commit: `8e25a1e` — Reject unaligned depth framesets and require
coherent reconstruction pairs. A following evidence commit records completion
and clarifies the software-device test output. The integration destination remains
`main`, via normal fast-forward and push. No published history is rewritten.

```text
M	L515_camera/README.md
M	L515_camera/build_realsense_ws.sh
A	L515_camera/patches/realsense-ros-4.0.4-aligned-depth-guard.patch
A	L515_camera/test_aligned_depth_guard.cpp
A	L515_camera/test_aligned_depth_guard.sh
M	docs/ai/00-index.yaml
M	docs/ai/05-admin.yaml
M	docs/ai/10-system-map.yaml
M	docs/ai/20-modules.yaml
M	docs/ai/30-contracts.yaml
M	docs/ai/40-flows.yaml
M	docs/ai/50-guardrails.yaml
M	docs/ai/86-reconstruction-recording.yaml
M	docs/ai/88-automatic-scan-recording.yaml
M	docs/ai/91-dense-offline-video.yaml
A	docs/ai/92-aligned-depth-framesets.yaml
M	docs/ai/project-structure.txt
M	docs/architecture/dense_offline_video.md
M	docs/architecture/reconstruction_recording.md
A	docs/experiments/aligned_depth_fix_validation.md
M	piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/reconstruction_recorder_node.py
M	piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/reconstruction_recording/core.py
M	tests/reconstruction/test_bag_import.py
M	tests/reconstruction/test_dense_video.py
M	tests/reconstruction/test_recording.py
M	tools/bundlesdf/dataset.py
M	tools/bundlesdf/extract_video.py
M	tools/bundlesdf/replay.py
M	tools/bundlesdf/segment_video.py
```
