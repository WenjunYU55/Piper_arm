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
# Baseline: 253 passed. Implementation suite: 269 passed, including pre-GPU timestamp/calibration rejection.
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
global refinement included by upstream. **GPU reconstruction is in progress at
this record's initial writing; completion and mesh quality are not yet claimed.**
The final execution result must be appended after the command exits.

Logs: `/tmp/piper-dense-*.log` plus durable job stage logs and output `bundlesdf.log`.
The official supplied milk sample remains download-quota-blocked as documented in
the upstream audit; its reconstruction has not run. No new physical arm/camera
qualification or calibrated ground-truth mesh accuracy test was performed.
