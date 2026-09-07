# Dense offline recording workflow validation — 2026-09-07

Development started on `fix/recorded-session-recovery` at
`d1941c89920f1313972eed75ab507df2c04f8d4e`, preserving all pending recording,
calibration and GUI recovery changes. `git fetch --all --prune` succeeded; the only
remaining remote branch is `origin/main` at that same commit (the old
`origin/curobo-integration` branch no longer exists). No robot, driver, camera or
mission process was started or restarted.

## Changes

- `extract_video.py` reads archived SQLite ROS messages without a node or DDS,
  prioritizes exact RGB-D matches, reserves accepted raw pins, rejects changing
  calibration and preserves image-time TF. It does not require live masks or MOVING.
- `segment_video.py` runs every extracted image through the actual existing SAM2
  temporal predictor, preserving the accepted initial mask as conditioning authority.
  Memory stays bounded through the existing compactor. It preserves the recorded
  angle, mask-envelope, depth, blur and expansion gates without live rate throttling.
- `python_sam2.sh` isolates the existing perception environment from Foxy.
- `reconstruct_job.py` wires archive extraction, temporal masks, validation and
  upstream reconstruction to the GUI's Reconstruct action. The old callback
  importer remains a diagnostic option.
- GUI controls expose Reconstruct, Validate dataset, Cancel reconstruction,
  partial-scan permission and stride, with responsive process polling. Active
  missions block job startup in the GUI.
- Result validation now reports explicit upstream tracking failures separately
  from structurally valid saved pose matrices and mesh output.

No live capture counts, robot activation/motion, NBV, safety, live sampling,
camera settings or planner logic changed in this follow-up.

## Actual saved-camera preparation

Input archive:

`datasets/mission_recordings/scan_recording_f62ed65bbeb4459cb35a9b36e64869f9.closed`

The extractor indexed 4,358 RGB, 6,557 depth and three raw accepted-view pins,
plus calibration, TF and mission events. It recovered **2,041 synchronized frames**
after the first accepted seed, rejecting **41 unmatched RGB timestamps** under the
recorded 80 ms maximum synchronization delta. No camera frame was invented or
reused as another frame; earlier pre-seed imagery is not part of the frozen target
sequence. All recovered frames have image-time TF. The archive lacks a terminal
workflow event; preparation therefore preserves UNKNOWN/incomplete mission status.

Actual isolated GPU SAM2 preparation:

- First 100-frame prefix: **100 accepted**, zero quality rejections, 9.800 s.
- Full sequence: **2,041 processed and accepted**, zero quality rejections,
  **191.403 s**, maximum **16 retained temporal images**.
- Mission capture count remains **3**. Offline disk backpressure drops: **0**;
  this is offline processing, not a measurement of transport drops during recording.
- Mask intersection-over-union against independently saved stationary masks:
  initial **1.0**, second **0.97044**, third **0.97545**. These reference masks
  are not human-labelled ground truth. A 12-frame overlay contact sheet was inspected.
- Dense timestamps span **123.207 s**, median interval **0.033331 s**, p95
  **0.133239 s**, longest gap **0.332924 s**, no gaps over one second.
  Median spacing near 30 FPS does not mean the archive retained a sustained 30 FPS.
- Previous filtered input: **141 frames**, median interval **0.299349 s**,
  longest gap **23.485207 s**. Many long gaps were stationary pauses; gap length
  alone is not proof of pose-tracking failure.

SAM2 provenance: existing checkout `b7a9c29f196edff0eb54dbe14588d7ae5e3dde28`,
SAM2.1 tiny, existing isolated Torch `2.11.0+cu128`; exact checkpoint, predictor and
compactor SHA256 values are in the prepared manifest. BundleSDF retains its
separate Torch/CUDA environment and pin `ffa67d425240b5b76d2e387a7dd3d3735a7cf1a1`.

## Commands executed

```bash
source piper_ros_foxy/install/setup.bash
python3 -m pytest -q tests/reconstruction tests/gui reconstruction/test_gui_support.py
# Baseline: 253 passed. Final implementation suite: 271 passed in 24.42 s.
python3 tools/bundlesdf/extract_video.py \
  datasets/mission_recordings/scan_recording_f62ed65bbeb4459cb35a9b36e64869f9.closed \
  /home/prl/.local/share/piper-bundlesdf/evidence/2026-09-07-dense-video-extracted
tools/bundlesdf/python_sam2.sh tools/bundlesdf/segment_video.py \
  /home/prl/.local/share/piper-bundlesdf/evidence/2026-09-07-dense-video-extracted \
  /home/prl/.local/share/piper-bundlesdf/evidence/2026-09-07-dense-video-session --limit-frames 100
tools/bundlesdf/python_sam2.sh tools/bundlesdf/segment_video.py \
  /home/prl/.local/share/piper-bundlesdf/evidence/2026-09-07-dense-video-extracted \
  /home/prl/.local/share/piper-bundlesdf/evidence/2026-09-07-dense-video-full
tools/bundlesdf/python_native.sh tools/bundlesdf/check_upstream_loader.py \
  /home/prl/.local/share/piper-bundlesdf/evidence/2026-09-07-scan-3761faaf5bd3/input \
  --upstream /home/prl/.local/share/piper-bundlesdf/upstream
cd piper_ros_foxy
colcon build --packages-select piper_mobile_manipulation --symlink-install
colcon test --packages-select piper_mobile_manipulation --event-handlers console_cohesion+
colcon test-result --verbose
```

The unmodified pinned reader loaded all **2,041 converted frames at 640×480**.
Build passed. CTest passed **48/50**, with the same **99 flake8 and 10 pep257**
pre-existing findings. Aggregate `colcon test-result` counts their wrapper failures
too: 1,625 tests, zero errors, 111 failures, two skipped. These are not called green.
Changed-file lint, shell syntax, Python compilation and architecture YAML parsing pass.
The seven existing SAM2 live-worker unit tests also pass in the isolated perception
environment (`tools/bundlesdf/python_sam2.sh AI_perception_tests/test_sam2_live_worker.py`);
these use fake predictors to verify conditioning and memory contracts, not live hardware.
An actual owned child process was cancelled through the GUI control in its lifecycle test.

Deterministic tests explicitly use fake segmentation only for orchestration/gate
cases; actual model inference above is separately measured GPU evidence. Tests
cover unique/exact-first synchronization, 16UC1 and 32FC1, all 30 Hz fixture frames,
no later live masks or MOVING events, timestamped TF, depth/edge/expansion rejection,
interrupted usable partials, independent next jobs, overwrite refusal, job failure,
stride validation, cancellation status, GUI literal arguments and active-mission guard.

## GUI-to-GPU run

An isolated Tk panel (not the running mission GUI) selected the actual archive and
invoked its real **Reconstruct** button with stride 1 and partial scan enabled.
The exact job command it launched is:

```bash
/usr/bin/python3 /home/prl/Piper_arm/tools/bundlesdf/reconstruct_job.py \
  /home/prl/Piper_arm/datasets/mission_recordings/scan_recording_f62ed65bbeb4459cb35a9b36e64869f9.closed \
  --output /home/prl/.local/share/piper-bundlesdf/evidence/2026-09-07-scan-3761faaf5bd3 \
  --job-dir /home/prl/.local/share/piper-bundlesdf/jobs/2026-09-07-scan-3761faaf5bd3 \
  --frame-stride 1 --allow-incomplete
```

This independently repeated extraction and actual SAM2 propagation (193.093 s,
2,041 accepted) and entered upstream `run_custom.py --mode run_video` with final
global refinement included by upstream. The command exited successfully on
2026-09-07 at approximately 17:12:59 BST. The GUI reported the exported mesh and
warned about **15 explicitly failed tracking frames**. GPU usage was checked with
`nvidia-smi`: RTX 3090, 70% utilization at the sampled instant, two BundleSDF CUDA
processes using approximately 3.3 GiB. This was an actual GPU reconstruction.

**Structural output PASS; visual reconstruction quality FAIL.** The OBJ contains
12,107 vertices and 16,125 faces, with a PNG texture (54,372 nonzero pixels),
2,041 saved pose matrices and 24 reference-recovery messages. Pose-file count is
not a count of successful tracks. Failed frame IDs are 384, 615, 636, 694, 712,
758, 818, 843, 956, 962, 1020, 1046, 1084, 1100 and 1103. Opposing offscreen views
show extended sheets, holes and large black regions rather than a faithful cube.
The dense pipeline fixes missing temporal input but has not demonstrated accurate
mesh reconstruction on this archive. No result is promoted to calibrated accuracy.

Artifacts are retained at
`/home/prl/.local/share/piper-bundlesdf/evidence/2026-09-07-scan-3761faaf5bd3/`:
`result/textured_mesh.obj`, its MTL/PNG, `result/ob_in_cam/`, `bundlesdf.log`,
`run.json`, input/configuration snapshots, `result_validation.json`,
`mesh-inspection.json`, `mesh-preview.png` and `mesh-preview-opposite.png`.
Mesh SHA256: `c3083ad26b0af5a147b4eafe3fcecde549fc81c8bcee48f23896cebaeea9fe99`.
The actual viewer loader passed `--check-only`; offscreen rendering also completed.
The exact upstream invocation (cwd: pinned upstream checkout) was:

```bash
/home/prl/Piper_arm/tools/bundlesdf/python_native.sh run_custom.py --mode run_video \
  --video_dir /home/prl/.local/share/piper-bundlesdf/evidence/2026-09-07-scan-3761faaf5bd3/input \
  --out_folder /home/prl/.local/share/piper-bundlesdf/evidence/2026-09-07-scan-3761faaf5bd3/result \
  --use_segmenter 0 --use_gui 0 --debug_level 2
```

This real-archive run is not the download-blocked official milk example.

Logs: `/tmp/piper-dense-*.log` plus durable job stage logs and output `bundlesdf.log`.
The official supplied milk sample remains download-quota-blocked as documented in
the upstream audit; its reconstruction has not run. No new physical arm/camera
qualification or calibrated ground-truth mesh accuracy test was performed.

## Separate depth/transform diagnostic

A read-only diagnostic fused 103 sampled frames using recorded image-time
`T_base_camera` and calibrated rays, independently of BundleSDF's estimated object
poses. Its 77,240 voxel-deduplicated points still show overlapping sheets and
extended surfaces. This indicates that increasing input frame count alone cannot
be assumed to fix all geometry errors; it does not identify a specific calibration,
depth or TF fault. No speculative depth correction or robot-pose prior was applied.
Diagnostic NPZ, JSON and PNG are retained beside the standalone dense native session
in `/home/prl/.local/share/piper-bundlesdf/evidence/2026-09-07-dense-video-full`.
This cloud is not a BundleSDF result or calibrated ground truth.

## Implementation commits and file inventory

Implementation commit: `069ab95531a0f1aea47d36d7f051b7a7df080bab` (abbreviated `069ab95`),
“Reconstruct recorded RGB-D sequences with offline SAM2 tracking”. The repository's
active remote integration destination is `main`; the deleted remote cuRobo branch
and existing local worktree are not recreated or reset. This evidence document is
committed separately after the real GPU job completed.

Exact implementation file inventory (`A` added, `M` modified):

```text
M	docs/ai/00-index.yaml
M	docs/ai/05-admin.yaml
M	docs/ai/10-system-map.yaml
M	docs/ai/20-modules.yaml
M	docs/ai/30-contracts.yaml
M	docs/ai/40-flows.yaml
M	docs/ai/50-guardrails.yaml
M	docs/ai/86-reconstruction-recording.yaml
M	docs/ai/87-bundlesdf-native-runtime.yaml
M	docs/ai/88-automatic-scan-recording.yaml
A	docs/ai/89-offline-distortion.yaml
A	docs/ai/91-dense-offline-video.yaml
M	docs/ai/project-structure.txt
M	docs/architecture/bundlesdf_native_runtime.md
A	docs/architecture/dense_offline_video.md
M	docs/architecture/reconstruction_recording.md
A	docs/experiments/bundlesdf_upstream_workflow_audit.md
A	docs/experiments/dense_offline_video_validation.md
A	docs/experiments/offline_distortion_validation.md
A	docs/experiments/saved_camera_bundlesdf_recovery.md
M	piper_gui/bundlesdf_results.py
M	piper_gui/native_app.py
M	piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/reconstruction_recorder_node.py
M	piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/reconstruction_recording/core.py
M	tests/gui/test_bundlesdf_results.py
A	tests/reconstruction/test_bag_import.py
A	tests/reconstruction/test_dense_video.py
A	tests/reconstruction/test_distortion.py
A	tests/reconstruction/test_reconstruct_job.py
M	tests/reconstruction/test_recording.py
M	tests/reconstruction/test_ros_replay.py
M	tools/bundlesdf/dataset.py
A	tools/bundlesdf/distortion.py
A	tools/bundlesdf/extract_video.py
A	tools/bundlesdf/import_bag.py
M	tools/bundlesdf/output_validation.py
A	tools/bundlesdf/patches/headless-tracking.patch
A	tools/bundlesdf/python_sam2.sh
A	tools/bundlesdf/reconstruct_job.py
M	tools/bundlesdf/replay.py
M	tools/bundlesdf/replay_bag.sh
A	tools/bundlesdf/segment_video.py
M	tools/bundlesdf/setup_native.sh
```
