# Baseline build and test results

## Test context

- Date: 2026-08-14 (Europe/London)
- Commit: `4945b480d8494fa840c0d2bc993c72834934a37f`
- Branch: `rewrite/core-integration-v2`
- Upstream comparison: commit matches `origin/main`
- Platform used by ROS tests: Ubuntu host, ROS 2 Foxy, Python 3.8.10
- Safety isolation: no PiPER, RealSense, GPU perception, Tesseract or RViz
  runtime nodes were running; ROS tests used `ROS_DOMAIN_ID=143` and
  `PIPER_MISSION_ENABLE_REAL_MOTION=0`.
- Hardware motion: not invoked. This Phase 0 run is software/build and
  command-free qualification only.
- Pre-existing source changes: the working tree already contained the
  documentation-only `docs/ai/00-index.yaml`, `05-admin.yaml`, and new feature
  bookmark changes. No production changes were made for this baseline.

## ROS build

Command:

```bash
cd /home/prl/Piper_arm/piper_ros_foxy
source /opt/ros/foxy/setup.bash
colcon build --symlink-install --event-handlers console_cohesion+
```

Result: **PASS** (exit 0), five packages finished in 2.79 seconds:

- `piper_msgs`
- `piper_description`
- `piper_mobile_manipulation`
- `piper`
- `piper_tesseract_foxy`

This is the normal whole-workspace ROS build. The docs also provide a focused
two-package build for mobile manipulation/Tesseract work, but the broader build
was used for this baseline.

## ROS package test suite

Command:

```bash
cd /home/prl/Piper_arm/piper_ros_foxy
source /opt/ros/foxy/setup.bash
source install/setup.bash
ROS_DOMAIN_ID=143 PIPER_MISSION_ENABLE_REAL_MOTION=0 \
  colcon test --event-handlers console_cohesion+
colcon test-result --verbose
```

Aggregate result: **NOT CLEAN**.

```text
679 tests, 0 errors, 118 failures, 1 skipped
```

The only failing test executables are the existing
`piper_mobile_manipulation` style gates:

- `flake8`: 95 findings (principally E501 line length, plus unused import,
  unused local and blank-line findings).
- `pep257`: 23 findings (22 D213 multi-line summary placement and one D401
  imperative-mood finding).

All registered behavioral pytest groups passed. Specifically:

- `piper`: 29 passed, 1 intentional copyright test skipped.
- `piper_description`: 10 passed.
- `piper_tesseract_foxy`: 99 passed.
- `piper_msgs`: CMake/XML lint passed.
- `piper_mobile_manipulation`: all 16 registered behavioral pytest targets
  passed; only aggregate flake8 and pep257 targets failed.

The style failures are part of the baseline and were deliberately not fixed in
Phase 0 because doing so would modify production files. Phase 1 should not
claim a clean suite until these are handled in a separately reviewed,
behavior-neutral patch.

## Root, reconstruction and calibration tests

Command:

```bash
cd /home/prl/Piper_arm
source /opt/ros/foxy/setup.bash
source piper_ros_foxy/install/setup.bash
ROS_DOMAIN_ID=143 PIPER_MISSION_ENABLE_REAL_MOTION=0 \
  python3 -m pytest -q tests \
    reconstruction/test_tsdf_reconstruct.py \
    L515_camera/test_validate_fixed_board.py
```

Result: **PASS**.

```text
58 passed, 1 skipped in 1.91 seconds
```

The skip is an existing environment/live-hardware-dependent case, not a new
failure.

## AI perception worker tests

Commands correspond to `AI_perception_tests/README.md`:

```bash
AI_perception_tests/groundingdino_test/envs/grounded_sam2_py310/bin/python \
  AI_perception_tests/test_heavy_model_worker.py
AI_perception_tests/groundingdino_test/envs/grounded_sam2_py310/bin/python \
  AI_perception_tests/test_sam2_live_worker.py
python3 -m pytest -q \
  AI_perception_tests/test_groundingdino_target_selection.py
```

Results: **PASS**.

- Heavy worker: 5/5 passed.
- SAM2 live worker: 6/6 passed.
- GroundingDINO target selection: 19/19 passed.

The isolated Python 3.10 environment does not contain `pytest`, so the
standalone target-selection pytest was correctly run with the repository's
system Python. This is an environment/tooling detail, not a test failure.

## Command-free Tesseract qualification

Command:

```bash
cd /home/prl/Piper_arm
./motion_planning/tesseract/qualify_rootless_worker.sh
```

Result: **PASS** for both suites with `real_arm_motion=false` and Tesseract
backend `0.35.0.6`.

- Core qualification: PASS; collision model reports hardware-qualified;
  includes model/FK, six-joint timing, 5% MoveJ timing, thin-obstacle detour,
  zero/centerline/dual-limit acquisition, folded/powered/staged home and the
  August 11 holder-floor incident rejection.
- Compact qualification: PASS; the recorded compact start used the expected
  bounded J3 recovery, selected a collision-qualified acquisition view and
  completed 5,261 validation samples.
- The 5% timing regression emitted maximum J6 velocity `0.15 rad/s`.
- The known holder-floor incident was correctly rejected because measured
  clearance `0.001245 m` is below the required `0.005 m`.

## Coverage limitations

This run establishes import/build, pure logic, interface-contract, worker,
collision-model and offline reconstruction baselines. It does **not** establish
a fresh physical end-to-end success rate, camera image quality, CAN behavior,
real-time DDS load, actual trajectory smoothness, perception accuracy, target
coverage, motor disable behavior or real process cleanup. Those require the
existing supervised hardware smoke groups in `docs/ai/40-flows.yaml` and
explicit operator authorization; Phase 0 intentionally did not move or enable
the arm.

## Baseline verdict

- Build baseline: green.
- Behavioral software baseline: green for every executed functional suite.
- Command-free planning qualification: green.
- Overall test baseline: yellow because 118 pre-existing style assertions fail.
- Hardware end-to-end baseline: not rerun in Phase 0; historical evidence only.
