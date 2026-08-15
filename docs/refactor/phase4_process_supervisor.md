# Phase 4 process supervision

## Scope

Phase 4 extracts autonomous mission child-process mechanics from
`target_scan_mission_node.py` into the pure-Python
`piper_mobile_manipulation/process_supervisor.py` module. It does not change
the mission state machine, startup order, readiness gates, commands,
environment values, process ownership, shutdown sequencing, ROS interfaces,
motion, safety thresholds, perception, or Tesseract behavior.

The persistent coordinator and RViz remain outside mission ownership. The
supervisor never discovers, adopts, or signals processes by executable name;
it manages only exact handles it started with `start_new_session=True`.

## Domain model

- `ProcessSpec` is an immutable name, command, and copied environment.
- `ProcessHandle` binds that specification to its exact `Popen` handle,
  process-group leader, generation log, and log offset.
- `ShutdownReport` records attempted names, graceful stops, TERM stops,
  opt-in forced kills, remaining groups, exit status, and diagnostics.
- `ProcessSupervisor` constructs inherited environments, starts owned process
  groups, reads generation-scoped logs, reports health/unexpected exits,
  rejects a new generation while any old group remains live, stops one owned
  group, and cleans up all owned groups in reverse dependency order.

`ManagedProcessSet` remains as an import-compatible alias in the mission module
for Phase 0/1 tests and downstream tooling. Its implementation is now solely
`ProcessSupervisor`.

## Preserved autonomous process contract

The mission still starts exactly:

1. `driver`: `start_piper.sh`
2. `vision`: `L515_camera/run_gpu_vision_pipeline.sh`
3. `hand_eye`: `L515_camera/run_hand_eye_tf.sh`
4. `tesseract_worker`: `motion_planning/tesseract/run_worker.sh`
5. `scan_stack`:
   `L515_camera/run_supervised_viewpoint_execution.sh`

All exact `PIPER_*` environment overrides remain built at the same point from
the same goal, home profile, motion flags, speeds, and capture bounds.
Generation log names and the public process-health JSON fields remain
`pid`, `running`, `returncode`, and `log`.

Cleanup remains reverse-order SIGINT, a shared 5-second wait, SIGTERM, then a
shared 3-second wait. The autonomous coordinator constructs the supervisor
with forced kill disabled. A surviving command owner therefore remains live,
its log remains open, cleanup returns false, and mission shutdown reports
`NEEDS_OPERATOR`; it is never converted into a false safe-shutdown result.

The abstraction supports an explicit opt-in SIGKILL stage so another owner can
express its existing policy, but the coordinator does not enable it.

## Tests

`test_process_supervisor.py` uses injected fake `Popen`, clock, sleep, and
process-group signalling boundaries. It never starts the robot stack or any
real child. It covers:

- successful startup and exact Popen/environment/log behavior;
- startup failure without leaked ownership or log descriptor;
- immediate exit and crash after startup;
- graceful SIGINT termination;
- SIGINT timeout and SIGTERM escalation;
- autonomous no-SIGKILL timeout reporting;
- explicit opt-in forced kill;
- reverse-order multi-process cleanup;
- cleanup after partial startup;
- repeated generations without ownership leakage;
- refusal to signal an unowned process; and
- immutable/copy-owning process specifications.

## Remaining direct subprocess ownership

These paths were inspected and deliberately remain outside the autonomous
mission extraction because their ownership and escalation policies differ:

- `piper_gui_native.py` owns its manual Tesseract/scan groups and uses
  SIGINT/SIGTERM/SIGKILL; its preview process has a separate direct-child
  policy. It is a candidate for a later policy-explicit migration.
- `target_scan_gateway_node.py` uses bounded `subprocess.run` for offline TSDF
  reconstruction after the tracked base reports home. It is a job execution
  boundary, not a mission-owned live process group.
- `L515_camera/run_gpu_vision_pipeline.sh` owns nested camera/perception
  groups, restart generations, and a validated process manifest.
- `L515_camera/stop_gpu_vision_pipeline.sh` validates allowlisted names,
  process-group leadership, and command paths before signalling its manifest.
- `motion_planning/tesseract/run_worker.sh` owns the worker singleton lock and
  Bubblewrap/Podman parent-lifetime contract.
- coordinator SIGINT handling in `target_scan_mission_node.py` remains because
  it is mission cancellation authority, not child-process signal mechanics.

Moving these paths into the same policy without separate characterization
would risk unrelated process termination, command-publisher overlap, broken
camera recovery, or reconstruction cancellation changes.

## Validation

Validation is software-only. No arm, camera, GPU worker, robot driver, or real
mission child is started.

- Pre-change mission/process/telemetry characterization selection: 93 passed.
- New fake-process supervisor suite: 13 passed.
- Combined focused regression selection: 106 passed.
- Complete `piper_mobile_manipulation/test` suite: 471 passed.
- Five-package ROS 2 Foxy `colcon build --symlink-install`: passed.
- Registered colcon aggregate: 828 tests, 0 errors, 118 pre-existing lint
  assertion failures, and 1 skip. The Phase 3/Phase 2 lint-debt count is
  unchanged and all functional registered targets pass.
- Wider root/driver/description/Tesseract selection: 195 passed and 2 skipped;
  its only failure is the documented top-level PiPER PEP 257 harness scanning
  generated/vendored trees (129,219 findings), not a Phase 4 regression.
- Heavy worker, SAM2 worker, and target selection: 5/5, 6/6, and 19/19 passed.
- Command-free rootless Tesseract core and compact qualification: passed with
  `real_arm_motion=false`.
- New production/test files pass `ament_flake8`, `ament_pep257`, and Python
  byte compilation. AI YAML parsing and `git diff --check` pass.
