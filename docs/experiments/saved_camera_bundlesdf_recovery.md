# Saved-camera BundleSDF recovery — 2026-09-07

Started from clean `main` at `d1941c89920f1313972eed75ab507df2c04f8d4e`;
implementation branch `fix/recorded-session-recovery`. No camera, driver, planner,
mission process or robot motion was started. The retained camera recording is:

`datasets/mission_recordings/scan_recording_f62ed65bbeb4459cb35a9b36e64869f9.closed`

It contains 97,547 SQLite messages, matching its archive manifest, including
4,358 color images, 6,557 aligned depths, 971 masks and three raw stationary
keyframes. The mission task is `gui-sim-f5956c842de64128b5f97592c3c179c6`;
that identifier's prefix does not classify the camera imagery as synthetic.
The recorded mission ended in operator cancellation after three accepted views.
Recovered data therefore remains explicitly incomplete.

## Failures and corrections

1. The recorder assumed pinhole CameraInfo and rejected nonzero distortion.
   The recorded camera has K approximately fx=598.87, fy=599.38, cx=330.25,
   cy=239.76 at 640x480, with plumb_bob coefficients
   `[0.16905, -0.51425, -0.0021469, 0.0013191, 0.4754114]`.
   New native frames preserve raw pixels and these calibration fields. Only the
   offline converter rectifies the aligned tuple: RGB linear, depth and binary
   mask nearest, unchanged K. Unsupported, nonfinite or changing calibration rejects.
2. The original TF rotation-member traceback was emitted during SIGINT, followed
   by another interrupt during worker join. It did not establish malformed saved
   transforms. All three recorded keyframes passed actual timestamped TF lookup
   and quality admission in the diagnostic. Signals now request shutdown without
   injecting Python exceptions into native TF conversion or worker joins.
3. Foxy saved no offered QoS for the late mission-history publisher. DDS replay
   offered volatile durability to a transient-local subscriber, delivering no
   accepted-view history. The wrapper now writes explicit playback QoS. A second
   replay received history but lost the first best-effort large keyframe: zero
   accepted, missing_initial_keyframe=2. This replay was not called successful.
4. `import_bag.py` avoids replay transport loss by reading SQLite in recorded
   order and invoking the real recorder callbacks, synchronizer, TF listener,
   acceptance gates and writer. Only this offline feeder waits for bounded
   admission capacity. It is callback recovery, not a DDS or new physical test.
   It retained 141 accepted frames including all three stationary views, with
   zero backpressure drops. Rejection counters were:
   rgb_depth_unsynchronised=3, sample_rate_throttled=39, stale_mask=32,
   unmatched_sensor_frame=2335. Counters include unmatched sensor/cache events,
   not only complete RGB-D candidate tuples.
5. The first real BundleSDF run hit upstream's `pdb.set_trace()` on insufficient
   reference matches at frame 28, then exited with `BdbQuit`. No completed mesh
   was claimed. `headless-tracking.patch` replaces only this debugger pause with
   logging. Existing reference recovery, matching thresholds and FAIL handling
   are retained. The full upstream diff is saved with each run.

## Executed commands and checks

```bash
source piper_ros_foxy/install/setup.bash
python3 -m pytest -q tests/reconstruction
# Baseline: 80 passed.
python3 -m pytest -q tests/reconstruction tests/gui reconstruction/test_gui_support.py
# Final: 248 passed, including distorted and pinhole public DDS replay,
# repeated shutdown signals and serialized ROS bag callback recovery.
cd piper_ros_foxy
colcon build --packages-select piper_mobile_manipulation --symlink-install
# Passed.
colcon test --packages-select piper_mobile_manipulation --event-handlers console_cohesion+
# 48/50 CTest checks passed; unchanged 99 flake8 and 10 pep257 findings.
cd ..
ROS_DOMAIN_ID=198 ROS_LOCALHOST_ONLY=1 python3 tools/bundlesdf/import_bag.py \
  datasets/mission_recordings/scan_recording_f62ed65bbeb4459cb35a9b36e64869f9.closed/bag \
  /tmp/piper-real-recovery-direct
python3 tools/bundlesdf/run.py \
  /tmp/piper-real-recovery-direct/reconstruction_1917af2a71f44029850a4293696b4342.incomplete \
  --allow-incomplete \
  --output /home/prl/.local/share/piper-bundlesdf/evidence/2026-09-07-real-scan-142623-headless
tools/bundlesdf/python_native.sh tools/bundlesdf/check_upstream_loader.py \
  /home/prl/.local/share/piper-bundlesdf/evidence/2026-09-07-real-scan-142623-headless/input \
  --upstream /home/prl/.local/share/piper-bundlesdf/upstream
# Unmodified pinned reader loaded all 141 640x480 frames.
```

BundleSDF base pin: `ffa67d425240b5b76d2e387a7dd3d3735a7cf1a1`.
The first failed run is retained separately in `2026-09-07-real-scan-142623`.
The headless run retains configuration, local patch, runtime verification, logs,
input validation and `native-session.incomplete`, a durable copy of the recovery.
Raw archives, datasets, weights and generated outputs are outside Git.


## Completed GPU output and inspection

The headless run exited **0**, exported `result/textured_mesh.obj` with its MTL
and texture PNG, and passed the wrapper's structural validation:

- 14,730 vertices and 19,475 triangular faces.
- 141 estimated rigid `T_camera_object` poses.
- 43,211 nonzero texture pixels out of 262,144.
- Mesh SHA256 `7f909ebe4b916f908e826b24d7262c52911fb5c9707aaaf325dd898d562b1f60`.

This is **successful execution, but poor visual reconstruction quality**. The
camera-facing preview shows the green target centrally, plus substantial spurious
surrounding surfaces; other regions are dark/untextured. The mesh axis-aligned
extent is approximately 100 x 60.1 x 100 mm, not a verified object dimension.
Accuracy remains unqualified. No artifact was removed or hidden to promote this
result to a good object model. The unchanged exported mesh and both front/back
previews are retained for inspection.

The interactive textured viewer was launched through the same `viewer_command`
used by the GUI; `viewer.log` confirms loading all 14,730 vertices and 19,475 faces.
The existing mission GUI was not restarted. To reopen, use Reconstruction
Validation -> BundleSDF results -> Refresh, select the output folder below, then
Open BundleSDF Model:

`/home/prl/.local/share/piper-bundlesdf/evidence/2026-09-07-real-scan-142623-headless`

A durable `native-session.incomplete`, recovery report, input, exact runtime and
upstream patch, validation reports, 141 poses, mesh and logs are retained there.
No new physical arm/camera run or dimensional accuracy validation was performed.
