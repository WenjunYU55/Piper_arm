# Tesseract and cuRobo command-free benchmark

## Safety and scope

This benchmark is entirely simulated/offline at the motion-planning boundary.
It starts isolated ROS-free planner workers and never starts ROS, a PiPER
driver, CAN, a camera, a controller, or a motor process. `real_arm_motion` and
`physical_result_claimed` are both false in the report.

The comparison does not claim that cuRobo is physically qualified. Tesseract
uses the exact configured moving and fixed collision geometry. cuRobo 0.7.8
uses exact fixed meshes but a 69-sphere articulated approximation marked
`hardware_qualified: false`. Every successful proposal from either backend was
therefore replayed through Tesseract's exact dense-path validator.

## Frozen corpus

The schema-v5 corpus is generated from achieved camera/joint geometry in the
recorded ray session `replay_scan_20260821_184024`. It contains:

- one recorded rough-acquisition request;
- four consecutive recorded multiview transitions;
- one deliberately blocked multiview negative control; and
- one configured return-home policy control.

Only volatile transaction identity, timestamps, and the backend-specific
planner selector change. Start state, target, camera request, calibration,
model hashes, joint limits, floor, obstacles, speed and timing policy remain
identical. Return home is excluded from algorithm success/timing because the
production Tesseract contract intentionally emits a direct, limit-checked
MoveJ target without ordinary collision planning.

## Result

The final run used one warm-up followed by three repetitions of all seven
fixtures on the reference RTX 3090 host.

| Metric | Tesseract 0.35.0.6 | cuRobo 0.7.8 |
| --- | ---: | ---: |
| Positive recorded requests | 15 | 15 |
| Positive planning success | 15/15 | 15/15 |
| Blocked controls rejected | 3/3 | 3/3 |
| Exact-validated successful paths | 15/15 | 15/15 |
| Median request wall time | 19.646 s | 0.630 s |
| Mean request wall time | 21.697 s | 2.298 s |
| Median scheduled trajectory duration | 4.750 s | 18.209 s |
| Median joint-space path length | 2.803 rad | 3.790 rad |
| Median request wall plus scheduled duration | 25.046 s | 25.048 s |

cuRobo generated proposals much faster, but its median path and scheduled
trajectory were longer. Consequently the median offline planning-plus-motion
proxy was effectively equal. Scheduled duration is not measured controller
execution time, and this result excludes perception, settling, capture,
startup, return home, disable and cleanup.

Evidence is retained under
`benchmarks/planner_backends/results/20260901_tabletop_controlled_replay/`.
Requests, responses and logs use portable relative paths; the private
executable runtimes are removed after successful completion.

## Reproduction

Generate the corpus from the immutable recorded diagnostics:

```bash
/usr/bin/python3 tools/create_planner_benchmark_corpus.py \
  --ray-diagnostics \
  /home/prl/Piper_arm/datasets/ray_diagnostics/replay_scan_20260821_184024/ray_mission_diagnostics.json \
  --output benchmarks/planner_backends/recorded_reference_corpus.json
```

Run the controlled replay:

```bash
/usr/bin/python3 tools/benchmark_motion_planners.py \
  --output benchmarks/planner_backends/results/new_run \
  --repetitions 3 --warmups 1
```

The output directory must be absent or empty. The tool owns only the two
workers it starts, terminates their exact process groups, and escalates only
within those owned groups.

## Mission efficiency instrumentation

New mission results contain additive `action_summary.phase_timing` intervals
and the frozen `action_summary.planner_backend`. The public ROS action schema,
state sequence, thresholds and motion behavior are unchanged. Saved result
payloads can be summarized with:

```bash
/usr/bin/python3 tools/summarize_mission_efficiency.py \
  /path/to/saved/results \
  --output /tmp/mission_efficiency.json
```

This reports phase totals, total duration, accepted captures, seconds per
capture, captures per minute, completion and safe-shutdown rates. All 35 saved
historical results lack the new timing field, so a backend-level full-mission
comparison requires new paired trials. Those later trials should
be labelled separately as simulation or supervised physical evidence.

## Interpretation boundary

- The planner result is `CONTROLLED_REPLAY`.
- It supports an offline timing and proposal-feasibility comparison.
- It does not qualify cuRobo for hardware.
- It does not compare physical tracking, controller following or collision
  outcomes.
- It does not establish complete mission efficiency until paired mission
  phase traces exist.
