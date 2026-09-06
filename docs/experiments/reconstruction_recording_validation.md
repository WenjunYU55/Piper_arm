> Historical first checkpoint. See [native BundleSDF follow-up](bundlesdf_native_validation.md)
> for the subsequent isolated GPU runtime and reconstruction evidence.

# Reconstruction recording: executed validation, 2026-09-06

Implementation is validated through unit tests, actual ROS DDS replay, synthetic
rosbag playback and pinned upstream data loading. **No BundleSDF mesh was produced;
no physical Piper arm/camera test was run.** cuRobo is the active planner and its
saved selection was preserved. Tesseract checks cover retained compatibility code.

## Git and environment evidence

The original `curobo-integrated` history was already merged into `main`; the remote
integration branch had been deleted. The user agreed to continue on `main`.
`git fetch --all --prune` confirmed `origin/main` as the only remote branch.
The user's 34 existing modified files were backed up first in commit
`b1d1e54de1ba55677801ad2bde1b64c57083995d` (`Preserve stream-rate controls and bounded
execution recovery`) and pushed before reconstruction changes. Previous main was
`ef54d8c535485835a15e5654bc9bbe92fbe8d9e1`.

Development branch `feature/bundlesdf-recording` was created from the synchronized
`main` commit b1d1e54 in `/home/prl/.codex/worktrees/piper-bundlesdf`. Its baseline
status was clean. The original checkout remained clean during development.
Ubuntu 20.04, Foxy/system Python 3.8; separate existing cuRobo v0.7.8 Python 3.10
venv and RTX 3090 were used for command-free GPU tests. No weights, datasets,
meshes, build products or raw logs are included in this change.

Local raw evidence: `/tmp/piper-bundlesdf-evidence/`. These temporary files are
machine-local; this tracked report and automated tests preserve the reproducible
record. Exact implementation file list is in `reconstruction_recording_files.txt`.
Operator commands and complete parameters/defaults are in
[reconstruction_recording.md](../architecture/reconstruction_recording.md).

## Results

| Executed check | Result |
|---|---|
| Baseline colcon build | 5 packages passed, 59.9 s |
| Baseline broad pytest | 1483 passed, 22 skipped, 2 failed |
| Final broad pytest | 1550 passed, 22 skipped, 1 failed, 36.92 s |
| New recorder/visibility unit cases plus actual DDS test | 66 passed, 8.38 s |
| Focused regression | 560 passed, 16.34 s |
| Final colcon build | 5 packages passed, 25.2 s after dependency declaration |
| Final colcon test | All functional suites passed; 99 existing flake8 and 10 existing docstring findings remain |
| colcon test-result aggregate | 1611 tests, 0 errors, 111 failures, 2 skipped; includes aggregate lint test failures as well as findings |
| New source/tests flake8; changed shell syntax | Passed |
| Architecture YAML parsing; git diff whitespace check | Passed |
| Rootless Tesseract core | Failed thin-obstacle detour; exact failure reproduced in detached baseline |
| Independent rootless compact suite | PASS; command-free real Tesseract qualification |
| cuRobo GPU selected suite | 8 passed, 1 failed for missing saved request fixture, 10 deselected |
| Additional GPU feedback-boundary case | 1 failed for same missing fixture, 18 deselected |
| Actual DDS replay | Three complete scans at 20/10/5 degrees plus three aborted second missions passed |
| Synthetic rosbag replay | Six sessions saved; clean SIGINT handler shutdown |
| Interrupted rosbag playback | Five-frame incomplete journal saved and successfully converted |
| Pinned BundleSDF reader class | Six converted frames loaded at 480x640 with scaled intrinsics |
| BundleSDF setup/wrapper smoke | Failed: Docker unavailable; no isolated BundleSDF installation |
| Actual upstream reconstruction command | Failed at import: no torch in invoked Python; no mesh |
| Official example download | HTTP 200 HTML quota-exceeded response, not a dataset; example could not run |

Baseline failure 1 was missing ignored `piper_home_pose.json` in the isolated
checkout. Copying the existing local configuration (still ignored) fixed this
fixture/environment issue. Failure 2 persists unchanged:
`tests/planning/test_ray_hard_culls.py::test_later_generation_ranks_only_survivors_of_hard_culls`
expects ray `2`, but receives an empty ranking. No test was suppressed.
Baseline colcon aggregate was 1607 tests / 113 failures / 2 skipped before restoring
that local fixture. A preliminary test command ran before build setup existed;
it failed setup sourcing and was rerun after build. Another colcon invocation
without the installed overlay exposed a sibling-package import error; sourcing
the built overlay fixed it, and the final functional colcon suites passed.

Rootless core fails at adaptive point 9.379: link5/scene_000_thin_midpoint_regression
clearance 0.006589 m is below 0.005000 m plus 0.001600 m motion bound. The exact
baseline command reproduced that failure. Compact qualification was run separately
because the combined script stops at the core failure.

GPU cases requiring `PIPER_CUROBO_GPU_MULTIVIEW_REQUEST` could not run without their
frozen request file. Eight other cases really executed with the GPU, including
bounded fallback, collision/world reuse, blocked geometry, native trajectory timing,
folded self-collision and internal limit checks. They did not command a robot.

The static capability atlas's worker provenance hash was updated after checking
that only movement-planning methods changed; static sampling, collision and FK
methods are unchanged. Atlas array hashes were preserved exactly:

- keys: `9c2c2e99de5aa936c03cda4ad7dff07c5ecbc66f058f86d393c2f2c8d5253f37`
- maximum_tool_minimum_z_m: `a154669252776bed0be6d9e6b44441a6caa42c3cfe49d057d746834ed95f495a`

## Measured frames and scenarios

The release DDS replay uses the actual recorder node, synchronizer, TF buffer,
parameter overrides, control/sensor queues and session writer. Its raw keyframe
publisher uses the same best-effort sensor QoS as production capture. It publishes
public synthetic sensor/status interfaces; it does not simulate physical arm
control or claim planner feasibility. Three-consumer path boundary tests separately
exercise the actual executor, Tesseract and cuRobo visibility helpers.

| Scan angle | Complete-session frames | Mission captures | Rejected | Dropped |
|---|---:|---:|---:|---:|
| 20 degrees | 6 | 2 | 6 | 0 |
| 10 degrees | 6 | 2 | 3 | 0 |
| 5 degrees | 6 | 2 | 3 | 0 |
| Three following aborted missions | 1 each | 1 each | 0 | 0 |

Total **21 accepted, 12 rejected, zero backpressure drops**. Rejection reasons:
`empty_or_tiny_mask: 3`, `target_clipped: 3`, `outside_angle_tolerance: 3`,
`missing_tf: 1`, `rgb_depth_unsynchronised: 1`, `stale_mask: 1`.
Stationary raw initial/final frames and intermediate moving frames are saved;
mission captures remain two per complete scan. Unit tests additionally cover all
four edges, exact angle boundaries, behind/invalid targets, empty/tiny/missing and
expanding masks, blur, depth 16UC1/32FC1/non-finite values, calibration changes,
duplicate/out-of-order stamps, atomic finalization, corrupt/missing artifacts,
disk errors, and multi-mission reset. Forced slow-writer unit coverage measures
one `writer_queue_full` drop and verifies submission does not wait for the writer.

Recorded synthetic bag: `/tmp/piper-bundlesdf-evidence/synthetic-bag/`.
Playback into `bag-replay-2` produces **23 accepted, 10 rejected, zero dropped**:
three complete sessions [6,7,7] and three one-frame failed sessions. This older
mixed-angle bag predates history-angle metadata; the replay explicitly uses
20 degrees for all sessions, so two candidates previously outside 10/5 degrees
are admitted honestly under 20 degrees. Current live history binds the angle and
rejects mismatched recorder configuration.

SIGINT playback into `interrupted-replay` produces **5 accepted, 5 rejected, zero
dropped**, state `incomplete`, mission count 1. Rejects are empty mask, missing TF,
outside angle, unsynchronized depth and clipping (one each). Its five-frame partial
dataset validates/converts with explicit `--allow-incomplete`. No compatible real
rosbag was found in the repository/local assets searched. Existing stationary
median-depth datasets are not falsely treated as raw synchronized reconstruction.

## Exact build/test commands

Commands below ran from the development worktree unless noted. Logs/XML use the
matching names in `/tmp/piper-bundlesdf-evidence/`.

```bash
source /opt/ros/foxy/setup.bash
cd piper_ros_foxy
colcon build --symlink-install --event-handlers console_cohesion+
source install/setup.bash
colcon test --event-handlers console_cohesion+
colcon test-result --test-result-base build --all
cd ..
ROS_DOMAIN_ID=143 PIPER_MISSION_ENABLE_REAL_MOTION=0 python3 -m pytest -q \
  piper_ros_foxy/src/piper_mobile_manipulation/test \
  piper_ros_foxy/src/piper_tesseract_foxy/test tests reconstruction \
  --junitxml=/tmp/piper-bundlesdf-evidence/verified-pytest.xml
ROS_DOMAIN_ID=173 ROS_LOCALHOST_ONLY=1 python3 tools/bundlesdf/replay.py \
  --output /tmp/piper-bundlesdf-evidence/replay-release
ROS_DOMAIN_ID=175 ROS_LOCALHOST_ONLY=1 PIPER_REPLAY_TASK_ID=replay-task \
  bash tools/bundlesdf/replay_bag.sh /tmp/piper-bundlesdf-evidence/synthetic-bag \
  /tmp/piper-bundlesdf-evidence/bag-replay-2
ROS_DOMAIN_ID=178 ROS_LOCALHOST_ONLY=1 PIPER_REPLAY_TASK_ID=replay-task \
  timeout --signal=INT --kill-after=15s 5s bash tools/bundlesdf/replay_bag.sh \
  /tmp/piper-bundlesdf-evidence/synthetic-bag /tmp/piper-bundlesdf-evidence/interrupted-replay
PIPER_TESSERACT_RUNTIME=/home/prl/Piper_arm/motion_planning/tesseract/.runtime \
  PIPER_TESSERACT_SPOOL=/tmp/piper-bundlesdf-evidence/tesseract-spool \
  ./motion_planning/tesseract/qualify_rootless_worker.sh
```

Independent compact command after exporting those same runtime/spool variables:

```bash
source motion_planning/tesseract/rootless_common.sh
"${ROOTLESS_BWRAP[@]}" /opt/tesseract/bin/python -m piper_tesseract_foxy.qualification \
  --urdf /workspace/motion_planning/tesseract/.runtime/piper_planning.urdf \
  --srdf "$COLLISION_SRDF_CONTAINER" --manifest "$COLLISION_MANIFEST_CONTAINER" \
  --calibration /workspace/L515_camera/calibration/hand_eye/session_20260808_straight_mount/calibration_result.yaml \
  --home-pose /workspace/piper_home_pose.json --suite compact
PYTHONPATH="$PWD:$PWD/piper_ros_foxy/src/piper_tesseract_foxy:$PWD/piper_ros_foxy/src/piper_mobile_manipulation" \
  PIPER_RUN_CUROBO_GPU_TESTS=1 \
  PIPER_CUROBO_ROBOT_CONFIG=/run/user/1000/piper_curobo_model/piper_curobo.yml \
  /home/prl/.venvs/piper-curobo-v0.7.8/bin/python -m pytest -q \
  tests/curobo/test_curobo_gpu_integration.py \
  -k 'rigid_endpoint or bounded_fallback or world_reuse or blocking_dynamic or same_native or known_folded or internal_limit' \
  --junitxml=/tmp/piper-bundlesdf-evidence/curobo-gpu.xml
# Same environment for the additional case:
# -k real_model_reenters_recorded_joint6_feedback_boundary
```

## BundleSDF evidence and outputs

Pinned official commit: `ffa67d425240b5b76d2e387a7dd3d3735a7cf1a1`.
Read upstream README, LICENSE.txt, run_custom.py and YcbineoatReader; no upstream
implementation was copied into ROS. Native sessions and converted RGB/depth/mask
PNG datasets exist under `replay-release` and `interrupted-converted`. CPU reader
check loaded six frames, 480x640, K with focal lengths 800 and centre [320,240].
The diagnostic executes the unmodified reader class extracted by AST so unrelated
CUDA imports are not required. That is a data-format test, not reconstruction.

```bash
bash tools/bundlesdf/setup.sh /tmp/piper-bundlesdf-upstream
PYTHONPATH=/tmp/piper-bundlesdf-reader-deps python3 tools/bundlesdf/check_upstream_loader.py \
  /tmp/piper-bundlesdf-evidence/replay-2/converted_20 --upstream /tmp/piper-bundlesdf-upstream
# Actually attempted upstream reconstruction, from /tmp/piper-bundlesdf-upstream:
python3 run_custom.py --mode run_video \
  --video_dir /tmp/piper-bundlesdf-evidence/replay-2/converted_20 \
  --out_folder /tmp/piper-bundlesdf-evidence/upstream-smoke \
  --use_segmenter 0 --use_gui 0 --debug_level 2
```

Setup failed because Docker is unavailable. Upstream command failed immediately
with `ModuleNotFoundError: No module named 'torch'`. The wrapper attempt retained
`bundlesdf-wrapper-smoke/failure.json` after successful native validation/conversion
and reported missing Docker. The official milk download returned quota-exceeded
HTML, preventing the required official-data-first smoke. **No output mesh or
estimated BundleSDF object trajectory exists.** Installing a dedicated environment,
weights and native extensions and successfully running the official example and
native-data reconstruction remain required. GPU hardware exists; software/sample
availability is the blocker here. Do not install BundleSDF into Foxy Python.

Physical validation still requires the Piper arm and RGB-D camera: real synchronized
mask delivery, exposure/blur calibration, image-time TF, moving-frame acceptance,
CPU/storage load, tight-cone path feasibility, and regression of all existing
stationary capture, safety and mission behavior. Reconstruction defaults off until
the operator opts in. Existing baseline failures remain visible and unresolved;
this report does not claim every requested acceptance criterion has passed.
