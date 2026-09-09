## Clearance-aware streaming and timing overhead — 8 September 2026

Normal Tesseract streams now retain the existing 0.30-rad maximum lead
in open space and tighten toward 0.03 rad within 0.025 m remaining collision-proxy
margin. Closing speed, feedback age and a configurable 0.25-second reaction estimate
bring tightening forward. `execution/clearance.py` uses measured FK and the next
approved target against the existing arm/holder, floor and frozen obstacle snapshot.
cuRobo bypasses this new monitor using the accepted plan_backend.
A measured negative margin enters collision-fault abort handling before another
point is emitted. This supplements approval; it is not a physical stopping-distance
or collision-avoidance guarantee. Dedicated RETURN_HOME is unchanged. Qualified
bootstrap self proximity stays exempt until the recovery cursor has passed and the
measured pose is fully clear; external/table monitoring remains active throughout.

Tesseract now rounds directly at the requested rate, avoiding unnecessary 20 Hz
pre-rounding while retaining native durations, vertices and speed/step ceilings.
The same isolated native 3.52843-second path now schedules 3.66 seconds / 733 positions
instead of 5.90 seconds / 1181 positions. cuRobo normalization is unchanged.
Feedback holds can still increase actual elapsed time. Campaign Runs expose relevant
execution source/configuration hashes; recorded durations are not adjusted.

Evidence: `datasets/experiment_results/clearance_stream_20260908/`. Supervised
physical tracking, hold/cancel/home and duration qualification remain pending.
The dated entries below describe earlier stages, superseded by this section.

## Native timing and immediate feedback — 8 September 2026

Tesseract now respects native segment durations instead of replacing them with
a distance-only schedule. At 100/200 Hz, a common native-path stretch is derived
from continuous segment rates; rounding tiny ISP intervals cannot amplify the
whole-path stretch. Baseline intervals account for native time, velocity and step
limits; higher transport rates subdivide those intervals. Source geometry and
collision validation remain unchanged.

The shared following check now applies from the first point. The one-second
parameter bounds recovery after a hold, not an unchecked startup period. This
also closes the sub-second-move exemption for cuRobo; its normalizer is unchanged.
The 0.30-rad entry limit is unchanged and is not a tight collision-clearance proof.

An isolated real Tesseract plan with native duration 3.52843 seconds previously
produced a 1.45-second 200 Hz schedule. It now produces 5.90 seconds, with identical
source geometry and 1181 positions instead of 291. This is planning evidence,
not physical tracking validation or a measured whole-mission runtime estimate.
See datasets/experiment_results/tesseract_timing_fix_20260908/README.md for the
regressions, provenance audit and separate mission-failure diagnosis.

## Tesseract transport correction — 8 September 2026

Normal Tesseract scan proposals labelled DIRECT_MOVEJ now send every approved
position sample and timestamp through the same timed-stream executor as cuRobo.
The label remains planner evidence that the straight chord passed dense validation;
it no longer causes the executor to discard intermediate samples. Detours retain
their approved geometry. Dedicated configured-home transactions are unchanged.

The saved selection is **200 Hz commands / 400 Hz executor timer**. Existing
following-error threshold, recovery and settling settings remain unchanged
at the operator's request to use the cuRobo streaming behavior. Measured envelope
monitoring is not implemented by this correction. The nominal rate and endpoint
success do not establish physical path accuracy; qualify commanded-versus-measured
motion, hold/cancel and table clearance in a supervised run. No robot was moved
or live process restarted for the change. Older saved selections below are history.

# PiPER transport-rate experiment

## Status — 6 September 2026

The code default and standard comparison remain **20 Hz**, with a 200 Hz executor timer.
The saved next-mission experiment is now **100 Hz commands / 200 Hz timer**.
100 Hz and 200 Hz are implemented as explicit experimental choices, **not
physically qualified smooth-motion modes**. No robot, ROS node, CAN interface
or camera was started for this change. The pre-change repository was committed
and pushed to main as `ef54d8c` first.

The reason for this experiment is a community PiPER ros2_control implementation
using position-command MoveJ at 200 Hz:
https://github.com/justagist/piper_cpp/tree/main/piper_cpp_ros
That is evidence of an implementation, not manufacturer certification or proof
that command rate caused this arm's shaking. The community plugin is not installed;
its activation/home/shutdown defaults are not appropriate to copy into this stack.

## Implementation

### Startup incident and approval transaction repair

Task `gui-sim-4697e21310804c61a5c7ceed05ba56d5` selected 200 Hz with
a 400 Hz executor timer. The 4,540-command acquisition passed proposal checks,
but approval-time path validation occupied about 11.5 seconds. The coordinator
stopped waiting after its former 10-second service limit; cancelling that local
future did not cancel the server callback. The executor subsequently started
the acquisition while the coordinator attempted terminal recovery. The result
was NEEDS_OPERATOR, zero captures and safe_shutdown=false. Perception shutdown
then caused the later post-settle frame timeout; that was not the first failure.

Mission-policy approval now has a 60-second bounded client wait and a 55-second
absolute acceptance deadline in the existing confirmation string. The executor
parses the deadline before validation and rechecks it before starting runtime
refresh. A monotonic elapsed-time check also prevents a backward wall-clock
adjustment during validation from extending acceptance. Late queued requests
and validation overruns cannot authorize motion. No collision samples or safety
checks were removed. The deadline is same-host Unix nanoseconds; deployments
with a remote coordinator require synchronized clocks. Legacy/manual confirmation
strings remain accepted for compatibility; restart BOTH coordinator and scan
stack to use the new bounded mission transaction.

This is a bounded budget and transaction fix, not a collision-speed optimization.
The 200 Hz physical stream actually achieved 132.3 Hz, with 1,714 late ticks:
22.700 seconds scheduled versus 34.317 seconds wall time. Do not describe
200 Hz as achieved or smoother. The saved user selection is not silently reset.
An earlier 19:28 mission failed for a different runtime-obstacle reason and did
complete safe shutdown; it is not this incident.

The approval regression replays an 11.5-second validation delay without sleeping or
commands, checks rejection at/after 55 seconds and on expired queue entry, and
runs the real MissionEngine through eight simulated captures and simulated
home/disable/cleanup using fake peripheral operations. This proves application
transaction behaviour, not physical tracking, perception or a real ROS E2E.

- `configuration.py` reads and validates the authoritative execution YAML rate.
  Mission launch freezes and supplies the same rate to bridge and executor.
  Supported values are exactly 20, 100 and 200 Hz. The executor timer must be at
  least twice the command rate. There is no active-mission rate switching.
- The generic request and response validators bind the selected rate into the
  existing request/trajectory hashes and enforce its minimum command interval.
  The legacy `tesseract_stream_v3` request still requires 20 Hz; high-rate valid
  Tesseract plans are not published on the legacy plan transport.
- cuRobo's normalizer previously hard-coded 20 Hz regardless of request. Its
  initial experiment subdivided each baseline interval. The smoothness repair
  below first assigns one common native-segment duration for experimental
  100/200 Hz, then inserts exact on-segment samples. Source vertices and detours
  remain fixed; total duration may increase. Bootstrap
  and joint-limit re-entry prefixes use the same requested rate and retain their
  correctly recomputed evidence indices. The 60,000-point bound is unchanged.
- Tesseract already accepts a rate in its normalizer. Generic validation and
  launch now permit the additional rates. Its native-duration-preserving baseline now subdivides into 100/200 Hz samples
  without shortening the 20 Hz schedule. Continuous-rate native scaling preserves
  speed ceilings before per-segment rounding; every source vertex remains.
- TrajectoryRunner uses binary search on the admitted numeric schedule rather
  than rebuilding a complete tuple every control tick. Next-unsent selection,
  no bursts/skips, late-schedule stretching and feedback holds are unchanged.

This is **linear on-path densification**, not spline smoothing, MIT control,
CPV control, jerk-limited retiming or a promise of continuous velocity at path
corners. Derivative fields remain zero transport placeholders, not actual
joint velocity commands. Do not feed those placeholders to a cubic controller
as desired zero velocity at every intermediate point.

## Trial settings (do not use as a qualification claim)

GUI: reopen the application, go to **Automatic Scan**, and find **Trajectory
stream rate for next mission** below the camera-on-ray tolerance controls.
Select **20 Hz (standard)**, **100 Hz (experimental)** or **200 Hz
(experimental)**, then click its **Apply for Next Mission** button. The GUI
atomically writes the command/timer pair in the same authoritative YAML listed
below; it does not maintain a second setting. Controls and the handler refuse
changes while a mission is active. The next mission starts a fresh scan stack
and reads the saved rates; an active stack never receives a live parameter update.
The GUI automatically restores the timer to 200 Hz when selecting 20 or 100 Hz.
All unrelated YAML settings and comments are preserved.
GUI addition validation: 164 GUI/configuration tests passed, native application
and persistence helper compile, all AI YAMLs parse, and diff checks pass. The
saved selection remains 20 Hz; no GUI or robot-control process was launched.

In `piper_ros_foxy/src/piper_mobile_manipulation/config/scan_execution_params.yaml`:

| Mode | trajectory_command_rate_hz | executor_tick_rate_hz |
| --- | ---: | ---: |
| Normal / rollback | 20.0 | 200.0 |
| First supervised experiment | 100.0 | 200.0 |
| Later experiment, only if 100 Hz tracking/timing is good | 200.0 | 400.0 |

The saved 100 Hz setting is for the next supervised experiment. Stop/restart the mission stack
after changing configuration; all other operator commands and physical opt-ins
are unchanged. This setting does not enable motors or grant authorization.
Do not infer that the existing 5-percent collision-model qualification also
qualifies the higher-rate execution transport.

Before promotion, compare the same authorized path at the same speed with:
actual achieved command rate, maximum command interval, lateness events,
measured following error, endpoint settling, hold/cancel behaviour and observed
shaking. Both backends still use the same executor. Inspect command-versus-joint
feedback traces before attributing shaking to timing, firmware or mechanics.

Higher point counts also increase serialization and some dense validation work.
Do not assume faster planning or that Python/ROS will sustain 200 Hz. Pure
scheduler timing excludes runtime gates, ROS delivery, SDK calls and the CAN bus.

## Reproducible offline checks

Validation on this host: 1,284 ordinary mobile/Tesseract/cuRobo tests passed,
17 skipped (including the 16 opt-in GPU cases); all 16 GPU cases then passed
separately against `/run/user/1000/piper_curobo_model/piper_curobo.yml` and the
frozen requests in `/tmp/piper_curobo_runtime_qualification_20260902_v3/raw/curobo`.
Both affected ROS packages built. AI YAML parsing and `git diff --check` passed.
The GPU run emitted existing PyTorch deprecation/sparse-tensor warnings.
No physical test or full rootless Tesseract mission qualification was run;
Tesseract ordinary contract, collision and worker regression tests passed.

```bash
source /opt/ros/foxy/setup.bash
source piper_ros_foxy/install/setup.bash
python3 -m pytest -q piper_ros_foxy/src/piper_mobile_manipulation/test \
  piper_ros_foxy/src/piper_tesseract_foxy/test tests/curobo
```

The opt-in GPU suite is documented in `motion_planner_backends.md`. Its two
additional tests run real acquisition planning at 100/200 Hz without commanding
the arm. Run it in the pinned cuRobo Python environment with only the repository
and ROS-free Tesseract source package on PYTHONPATH, not Foxy's Python environment.

For a recorded **uniform 20 Hz** successful response:

```bash
python3 tools/diagnostics/benchmark_stream_transport.py /path/to/response.json \
  --output /path/to/new-comparison-directory
```

The tool refuses existing output directories and historical sparse schedules.
It checks source-point and duration retention, and writes hashed-source CSV
with command step sizes and scheduler-only timing. The archived 2 September
runtime-qualification acquisition was rejected by this tool because its
schedule was sparse; it was not relabelled as a current 20 Hz capture.

Physical smoothness, tracking bandwidth and CAN scheduling remain unmeasured
for the new modes. No live-default promotion is made by this work.

## Smoothness repair — 6 September 2026

The latest inspected mission, `gui-sim-a537a7fb27804589bddca4be9e892b46`,
logged 165.2–196.9 Hz average stream rates and maximum command gaps up to
28.4 ms. One segment logged 161 hold/resume pairs near the 0.30-rad following
threshold. The observed CAN interface counters showed no errors or drops.
These are delivery/following observations, not proof of cuRobo geometry failure.

Two local changes address those observations:

1. Following recovery is latched. Entry remains at the configured threshold
   (default 0.30 rad from the first point; one second bounds recovery only). Resume requires both
   the held and next-unsent target errors at or below 90% of that threshold
   (0.27 rad by default), demonstrated by three independently newer joint
   source stamps. Duplicate timestamps and missing observations cannot count;
   timer ticks alone are not evidence. The recovery deadline remains one
   configured grace duration from first entry, including time spent just below
   the entry threshold. It cannot restart on every oscillation. Every held
   interval, including the last, pauses the schedule. New paths reset recovery.
2. Experimental cuRobo normalization uses the largest required segment tick
   count for every native segment. That is one uniform time dilation of the
   already-qualified sampled path. Independent duration rounding previously
   introduced speed dips at otherwise smooth native segment boundaries. The
   change retains every vertex and all speed/step/point-count limits and cannot
   accelerate the legacy schedule. Standard 20 Hz and bounded bootstrap/re-entry
   prefixes keep their original timing. No spline corners are inserted.

The installed cuRobo 0.7.8 `retime_trajectory()` reconstructs an interpolated
path from the optimized trajectory. Direct substitution would require a new
geometry qualification rather than preserving the current sampled-path proof.
This repair therefore implements uniform dilation in the existing adapter.
`filter_robot_command` filters derivatives, which PiPER's position-only MoveJ
boundary does not consume; enabling it alone would not repair position delivery.
Neither flag was added. The existing JSON diagnostics now report timing policy,
native/scheduled counts and durations, rate, and available per-joint native
velocity/acceleration/jerk peaks. Public derivative arrays remain zero placeholders.

Validation evidence:

- Final ordinary mobile/Tesseract/cuRobo/GUI suite: **1,444 passed, 20 skipped**
  (19 opt-in GPU cases plus one environment skip). JUnit result:
  `/tmp/piper_stream_smoothness_regression.xml` (temporary, not archival).
- Self-contained pinned GPU selection: **9 passed, 10 deselected**, including
  20/100/200 Hz fallback collision/visibility qualification and the same-native
  timing comparison. The ten remaining cases require unavailable frozen requests.
- Both affected ROS packages built successfully.
- `colcon test`: functional suites passed after fixing the earlier rate-test
  repository import path under package-local execution (Tesseract: 201 passed).
  The overall mobile package remains red on **99 existing flake8 findings and
  10 existing docstring findings** outside this repair. No lint checks were
  disabled. Added code passed targeted fatal-error lint and Python compilation;
  all AI YAMLs parsed and `git diff --check` passed.
- A single real GPU solve produced 31 native samples. Reusing that exact result
  produced a 4.4-second legacy schedule and 6.0-second uniform schedules at both
  100 Hz (601 points) and 200 Hz (1,201 points). Every native vertex and the
  speed ceilings were verified. The longer duration is the cost of preserving
  the native speed profile; this does not establish smooth physical motion.
- Regression tests cover boundary oscillation, duplicate/missing feedback,
  three-sample recovery, deadline expiry, nonfinite feedback, native timing
  ratios, detour retention, wrist speed, and the point-count ceiling.

The earlier frozen GPU request corpus under `/tmp/piper_curobo_runtime_qualification_20260902_v3`
is absent after reboot. Tests that require those acquisition/multiview/home
fixtures cannot be presented as replayed. Self-contained real-model GPU tests
cover same-path retiming, collision controls, fallback qualification and model
bounds. Software-only tests cannot establish sustainable 100 Hz delivery or
remove controller-level periodic jitter. A supervised same-path comparison and
hold/cancel/home checks remain necessary before promotion.
