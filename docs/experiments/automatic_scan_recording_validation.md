# Automatic scan recording validation — 2026-09-07

Started from clean `main` at `24a10d3`; developed on
`feature/automatic-scan-recording`. Normal supervised scans now default to a
whole-scan ROS bag plus the existing filtered native reconstruction session.
Stationary median captures, their accepted count, NBV, target freezing, motion
planning and safety policies are unchanged. Both recording paths can be disabled
explicitly. No live mission or camera was started or restarted during this work.

See [operation, parameters and replay commands](../architecture/reconstruction_recording.md)
and [ownership and lifecycle contracts](../architecture/recording-and-frame-contracts.md).

## Executed checks

```bash
source piper_ros_foxy/install/setup.bash
python3 -m pytest -q tests/reconstruction
# Baseline: 70 passed.
cd piper_ros_foxy
colcon build --packages-select piper_mobile_manipulation --symlink-install
# Baseline: 1 package built in 1.25 s. Final: 1 package built in 1.18 s.
colcon test --packages-select piper_mobile_manipulation --event-handlers console_cohesion+
# 48/50 CTest checks passed; unchanged flake8 and pep257 checks failed.
cd ..
python3 -m pytest -q tests/reconstruction tests/gui reconstruction/test_gui_support.py
# Final: 238 passed in 11.90 s.
python3 -m flake8 --max-line-length=99 \
  piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/mission_archive.py \
  tests/reconstruction/test_mission_archive.py
bash -n tools/bundlesdf/replay_bag.sh
python3 -m py_compile \
  piper_ros_foxy/src/piper_mobile_manipulation/launch/supervised_viewpoint_execution.launch.py \
  piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/mission_archive.py
```

The two package lint failures retain the previously recorded **99 flake8** and
**10 docstring** findings. New-file lint, shell syntax and Python compile passed.
Tests cover all four production archive/native flag combinations, raw-keyframe
retention in archive-only mode, unique sessions, clean SIGINT exit codes, low disk,
failed writer startup, bounded best-effort sensor QoS and mismatched SQLite counts.

## Actual ROS record and replay

The existing public-interface `Replay` harness published a synthetic nominal scan
on isolated ROS domain 193 with real motion disabled. The new supervisor recorded
it using actual `ros2 bag record`, then closed on SIGINT. It retained:

- **29 declared messages and 29 stored SQLite rows**.
- Four RGB, four aligned depth (16UC1 and 32FC1), four masks, four CameraInfo.
- Two complete raw stationary tuples, two accepted-history events.
- Six dynamic TF messages, two motion-state messages, one final completion event.
- No static transform messages: this synthetic harness uses direct dynamic TF.

The source native recorder saved **6 frames, 2 mission captures**. The corrected
bag is at:

`/tmp/piper-auto-archive-nominal-v2/bags/scan_recording_04dc705e9ab548588369fcffa135770b.closed/bag`

Executed offline commands:

```bash
ROS_DOMAIN_ID=194 ROS_LOCALHOST_ONLY=1 tools/bundlesdf/replay_bag.sh \
  /tmp/piper-auto-archive-nominal-v2/bags/scan_recording_04dc705e9ab548588369fcffa135770b.closed/bag \
  /tmp/piper-auto-archive-nominal-replayed-v2
python3 tools/bundlesdf/dataset.py \
  /tmp/piper-auto-archive-nominal-replayed-v2/reconstruction_f95ada0f65f14b9e88f2914c68b66ee6.complete \
  --validate-only
python3 tools/bundlesdf/dataset.py \
  /tmp/piper-auto-archive-nominal-replayed-v2/reconstruction_f95ada0f65f14b9e88f2914c68b66ee6.complete \
  --output /tmp/piper-auto-archive-converted
tools/bundlesdf/python_native.sh tools/bundlesdf/check_upstream_loader.py \
  /tmp/piper-auto-archive-converted \
  --upstream /home/prl/.local/share/piper-bundlesdf/upstream
```

Playback produced **one complete session, 6 accepted frames, 0 rejected, 0 writer
drops and 2 mission captures**. Native validation and conversion passed; the
unmodified reader from pinned BundleSDF commit
`ffa67d425240b5b76d2e387a7dd3d3735a7cf1a1` loaded all six converted 640×480 frames.
This is actual ROS bag/replay evidence with synthetic sensors, not physical proof.
A new GPU reconstruction was not run for this recording-only change; the existing
native GPU mesh smoke is documented separately.

## Failures found and fixed

1. Foxy `ros2cli` returns 2 after handled KeyboardInterrupt. Normal closure now
   recognizes that only for requested shutdown with a valid bag index.
2. The old replay wrapper passed an invalid empty `task_id:=` ROS argument and
   ignored recorder startup failure. Empty overrides are now omitted and startup
   failure is surfaced. Replay inherits archived task identity, frozen angle and
   available quality/recorder configuration; zero accepted frames is an error.
3. Foxy's eight-message rosbag cache reported 29 messages but left only 24 in
   SQLite on shutdown, losing the final accepted capture and completion event.
   Earlier mixed-session tests also produced incomplete tails. The archive now
   disables that cache, keeps bounded DDS histories in its separate writer process,
   and verifies SQLite row counts before labelling the archive closed. The corrected
   full round trip above retained the final capture and completion event.
4. Running the upstream reader check in system Python lacked imageio; running it
   through the installed isolated BundleSDF interpreter passed.

Logs and test assets are under `/tmp/piper-auto-*`, outside Git. Raw transport
loss totals are **unknown**, as Foxy does not expose an accurate session-level
counter; only stored row counts and native recorder counters are claimed.

Real-camera bandwidth, storage contention, live-mask quality and physical arm
execution timing remain untested. Recording all available evidence does not make
invalid frames usable or reveal surfaces the camera never observed. Archive
`.closed` means storage closure, not mission success; failed/cancelled mission
outcomes remain in the recorded events.
