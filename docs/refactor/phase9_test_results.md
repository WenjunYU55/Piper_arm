# Phase 9 test and build results

Validated on 2026-08-14 without starting the camera, arm driver, planner
service, perception stack, GUI, or any other hardware-facing process.

## Configuration regression tests

- `test/test_configuration.py`: 15 passed.
- The tests independently record and compare all 16 coordinator defaults and
  all 82 viewpoint-executor defaults.
- Covered startup validation, ROS override retention, one-time parameter
  reads, immutable groups, frozen runtime lookup, explicit MissionEngine
  injection, invalid values, units, and separation from mission input,
  telemetry, and derived state.
- `configuration.py` and `test_configuration.py` pass `ament_flake8` and
  `ament_pep257` independently.

## Functional regression results

- Entire `piper_mobile_manipulation` Python suite: 580 passed.
- Root GUI, reconstruction, and calibration suites: 69 passed, 1 skipped.
- AI-worker selection suites: 30 passed (5 heavy-model, 6 SAM2-live, and 19
  GroundingDINO-selection tests).
- Aggregate ROS/colcon functional tests include 951 recorded test cases. All
  functional pytest, message, XML, and CMake-lint cases passed.

## Build

The normal selected-stack build completed successfully for:

- `piper_msgs`
- `piper_description`
- `piper`
- `piper_mobile_manipulation`
- `piper_tesseract_foxy`

Command:

```text
colcon build --symlink-install --packages-select piper_msgs piper_description piper piper_mobile_manipulation piper_tesseract_foxy
```

## Known pre-existing lint baseline

The aggregate `colcon test` report is 951 tests, 0 errors, 116 failures, and 1
skip. The 116 failures are repository-wide style debt already recorded before
Phase 9: 95 `ament_flake8` findings and 21 `ament_pep257` findings. Phase 9 did
not increase that count, and its new configuration module and test module are
independently clean.

## Tesseract qualification

The command-free rootless qualification was run because the coordinator and
executor are high-risk routing files. It does not connect to physical
hardware. Both the core and compact suites passed; the backend version was
`0.35.0.6`, collision-model qualification remained true, and
`real_arm_motion` remained false.
