# Core stack modularization

## Scope

This ownership refactor starts from `feature/size-aware-target-envelope` at
`bfd0686`. It changes no ROS action, service, topic, message, QoS, parameter,
default, TF meaning, speed, threshold, mission phase, capture rule, Tesseract
schema-v5 field, executable entrypoint, or physical-motion policy.

The software baseline before editing was 935 passing mobile-manipulation and
Tesseract tests. The Phase-1 characterization suites remain the golden mission
phase/effect and executor decision traces.

## Resulting ownership

- `MissionEngine` remains the sole mission and terminal-shutdown authority.
- `mission_artifacts.py` owns calibration identity and guarded failed-dataset
  discovery/deletion. `target_scan_mission_node.py` re-exports the established
  functions.
- `mission_ros_operations.py` owns the engine-to-ROS operations adapter and
  previous-generation admission cleanup. The mission node remains the action,
  queue, spool, callback, result, and lifecycle façade.
- `ExecutorSession` is the only production owner of plan, motion, acquisition,
  capture, recovery, achieved-pose, home, and mission-authorization state.
  Existing executor attribute names are descriptors backed by that session;
  this preserves characterization harnesses without duplicate storage.
- Plan normalization, runtime/path safety classifications, and home-settling
  decisions have focused ROS-free owners. The executor node remains the only
  ROS command and status publisher.
- `bridge_candidates.py` owns candidate-policy validation, target-envelope
  adaptation, and the bounded information-ranked ray shortlist. NBV information
  ranking remains on the bridge/planner side.
- `contract_core.py`, `contract_validation.py`, `contract_request.py`,
  `contract_response.py`, `contract_hashing.py`, and `contract_spool.py` own
  errors, shared schema primitives, request validation, response validation,
  canonical bytes/hashes, and atomic queues. `contract.py` re-exports every
  legacy import as a compatibility façade.
- `WorkerOrchestrator` in `worker_components.py` owns the live request-level
  planning flow and routes scene setup, aim/IK, and trajectory/home operations
  through its composed components. `TesseractBackend.plan()` is a compatibility
  delegate; `worker.py` and its executable remain stable, and the backend
  retains exact feasibility authority.

Compatibility is one-way: old module imports point to the focused owner.
There is no fallback mission engine, second executor state machine, second
command publisher, or alternative contract serializer.

## Qualification boundary

Software tests, builds, canonical-byte fixtures, legacy-import checks, and
command-free Tesseract qualification may be completed in this environment.
Motors-disabled graph checks and every enabled-arm stage require the trained
operator and staged checklist in `physical_requalification_checklist.md`.
They are not implied by a software pass.

## Verification record

Verification on 2026-08-25 produced the following evidence:

- The complete mobile-manipulation and Tesseract Python suites pass: 943 tests.
- The repository GUI, reconstruction, and calibration suites pass: 149 tests,
  with one hardware-dependent skip. The isolated perception workers and model
  selection tests pass: 38 tests.
- The five affected ROS packages build successfully with `colcon build`.
- All 21 changed Python files pass Python 3.8 compilation, `ament_flake8`, and
  `ament_pep257`; `git diff --check` and every `docs/ai/*.yaml` parse pass.
- Package-level `colcon test` passes every functional target. The collected
  workspace result is 1,242 tests with 84 reported failures and one skip. The
  findings are the package's existing 68 repository-wide flake8 findings and
  14 existing docstring findings, plus the two parent CTest targets that report
  those lint failures; none is in a changed file. The Tesseract package records
  165 passing tests independently.
- Public message, service, action, package, and console-entrypoint definitions
  are byte-for-byte unchanged from `bfd0686`.
- The compact rootless Tesseract qualification passes with
  `real_arm_motion: false` and the collision model hardware-qualified flag.
- The deterministic two-million-sample capability map retained exactly the
  previous `keys` and `maximum_tool_minimum_z_m` arrays. Its generator source
  provenance now includes `worker_components.py`; the regenerated map contains
  1,685,619 collision-qualified configurations and 1,479,561 occupied bins.
- A routing regression proves the live orchestrator uses the collision-scene
  and aim components; all existing direct `TesseractBackend.plan()` callers
  continue through its thin compatibility delegate.

The rootless core qualification is not recorded as a pass. Its synthetic
dual-limit acquisition case exhausts the internal 150-second planning budget
before starting a candidate. The exact same failure was reproduced in a clean
detached worktree at the source baseline `bfd0686`, establishing that it was
not introduced by this extraction, but it remains a qualification blocker.

## Release status

The software refactor is implemented on
`refactor/core-stack-modularization`, but release qualification is incomplete.
The motors-disabled ROS graph, enabled arm without autonomous motion, one
low-speed viewpoint, return-home/cancellation, short scan, complete scan, and
repeated mission require explicit operator supervision and have not been run.
Because the plan requires every software and physical gate before publication,
`reintegrate/selected-archived-features` has not been advanced and nothing has
been pushed. The unrelated untracked `l515_attached_assembly_colored.ply` also
remains untouched and excluded.
