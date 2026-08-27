# Behaviour-preserving architecture cleanup

## Branch and baseline

This cleanup is implemented only for
`reintegrate/selected-archived-features`. Its integration commit preserves the
old reintegration tip and `origin/main` as parents while using the exact
`origin/main` tree at `904dc39e96d5ad36b659cb240b1ad2ab0845775e` as the
behavior baseline. No main-branch commit is changed.

The architecture audit and target design are governed by
`current_architecture.md`, `external_contracts.md`, `safety_invariants.md`,
`refactor_risks.md`, and `docs/ai/`.

## Phase 0 baseline

- Five ROS packages build with `colcon build --symlink-install`.
- The complete mobile source test directory produced 848 passes, one skip and
  one missing ignored local-home-profile fixture. Linking the same deployed
  local profile into the isolated worktree made that exact test pass.
- Tesseract Python tests: 162 passed.
- Driver: 69 passed and one copyright skip.
- Robot description: 18 passed.
- Root GUI, reconstruction and L515 calibration selection: 193 passed and one
  existing environment-dependent skip.
- GroundingDINO target selection: 25 passed.
- No hardware-facing process or command was started.

The command-free rootless core Tesseract qualification loaded the production
model and passed model, six-joint timing, five-percent timing, thin-obstacle
detour, zero-start and centerline-zero-start stages. The unchanged baseline
then exceeded its internal 150-second planning budget in the dual-limit-start
case. This is baseline performance evidence, not a cleanup regression.

## Phase 1: characterization and test registration

All existing `piper_mobile_manipulation/test/test_*.py` files are now
registered in CMake. `test_architecture_boundaries.py` additionally freezes:

- the named ROS-free mission, execution, planning, perception and
  infrastructure owners;
- the ROS-free Tesseract worker and contract boundary; and
- absence of internal import cycles in both production Python packages.

Focused Phase 1 validation passes 48 tests. The complete mobile source suite
passes 852 tests with one hardware-dependent skip, and all seven newly
registered CMake targets pass through the installed ROS overlay with 55 test
results. Runtime source, interfaces, configuration and behavior are unchanged.

## Incremental change rule

Each later phase adds the new responsibility owner first, retains an explicit
old-path facade, migrates callers, runs the linked tests, and updates AI docs.
Compatibility removal is a separate evidence-gated cleanup. Safety checks at
distinct trust boundaries are not deduplicated as part of structural work.

## Phase 2: responsibility packages

The existing pure owners moved intact into two cohesive packages:

- `mission/{core,engine,spool}.py` owns mission state, progression and durable
  handoff;
- `infrastructure/{failure_model,telemetry_store,process_supervisor}.py` owns
  cross-cutting typed failures, observations and exact child-process mechanics.

All production consumers import the new owners. The six former module paths
remain explicit, one-way facades with fixed `__all__` lists, and tests prove
that every facade export is the identical owner object. Focused mission,
infrastructure, external-contract and compatibility validation passes 333
tests. The complete mobile suite passes 854 tests with one hardware-dependent
skip; changed Python lint/compile, the five-package build and all AI YAML parsing
pass. Public ROS interface, launch and configuration trees are byte-identical
to the baseline. No spool format, state transition, signal escalation, safety
decision or hardware behavior changed.

## Phase 3: mission adapter cleanup, first seam

`mission/resources.py` now owns the exact calibration provenance hash, guarded
failed zero-capture dataset discovery/removal, and previous-generation cleanup
selection. `mission/engine.py` now also owns construction from typed config and
the characterization fallback. The ROS node imports these owners and retains
the established resource-function names for downstream compatibility.

Dedicated resource characterization plus mission regression passes 187 tests;
the complete mobile suite passes 859 tests with one hardware-dependent skip,
and changed-file lint and the focused package build pass. Action, queue, topic,
service, state-transition, process-signal, dataset-format, launch and hardware
behavior are unchanged.

## Phase 4: execution responsibility package

The existing ROS-free authorization, capture coordination, execution-mode,
motion validation, recovery, trajectory monitoring and schedule-validation
modules moved intact under `execution/`. Production consumers use those owners;
the seven former module paths remain explicit one-way facades whose exports are
identity-tested. Only inward imports and five docstring layout corrections were
made; numerical and state behavior is unchanged.

Focused execution, mission and external-contract validation passes 373 tests.
The complete mobile suite passes 860 tests with one hardware-dependent skip,
and changed-file flake8/pep257 pass. The executor node remains the sole command
publisher and all safety, timing, capture and recovery gates are retained.

## Phase 5: planning responsibility package

The existing ROS-free capability map, object coverage/NBV, generation identity,
persisted surface coverage, permanent ray cull and bounded ray modules moved
intact under `planning/`. Production imports use the owners; six old module
paths remain explicit identity-tested facades. White-box NBV tests now patch the
owner module so test injection follows implementation ownership.

Focused planning validation passes 161 tests with one existing replay skip;
the complete mobile suite passes 861 tests with one hardware-dependent skip,
and changed-file lint passes. Candidate IDs, hashes, cull lifetime, ranking,
coverage, capability schema, ROS contracts and Tesseract authority are unchanged.

## Phase 6: perception responsibility package

The existing ROS-free acquisition, trusted target-envelope, landmark geometry,
obstacle geometry and occlusion policy modules moved intact under `perception/`.
Production nodes use the owners and five old paths remain explicit facades.
Obstacle and occlusion label canonicalization were deliberately not deduplicated:
they serve different accepted-label and policy contexts.

Focused perception, capture, session and planner validation passes 199 tests;
the complete mobile suite passes 862 tests with one hardware-dependent skip,
and changed-file lint passes. Exact timestamps, seed/envelope hashes, crop gates,
obstacle classifications, repeated occlusion proof and ROS contracts are unchanged.

## Phase 7: Tesseract boundary cleanup

The command-free bridge now delegates its existing pure candidate validation,
shortlist construction and typed endpoint classification to
`candidate_selection.py`. The schema-v5 canonical request/response contract is
owned by `protocol/contract.py`, while atomic permission-bounded queue and
heartbeat mechanics are owned by `protocol/spool.py`. The former `contract.py`
path is an explicit facade, and the bridge retains identical helper attributes
for compatibility.

No function body, shortlist constant, schema version, canonical encoding,
hash, queue name, path, permission, heartbeat operation, planning budget or
worker algorithm changed. `worker.py` remains byte-identical, including its
old-path facade import, because the committed camera capability map binds the
exact worker source hash. The complete Tesseract suite passes 163 tests,
including facade and former-bridge export identity checks; touched files pass
flake8/pep257 and the dependency-ordered three-package build passes.

## Phase 8a: native GUI ROS adapter

The complete `PiperGuiRos` class moved intact from the Tk launcher to
`piper_gui/ros_node.py`. `piper_gui_native.py` imports and exposes the identical
class, so established imports and bootstrap behavior remain compatible. ROS
topics, services, actions, QoS, event tuple names, goal conversion and command
messages are unchanged; Tk presentation and explicit commissioning controls
remain in the native module.

All 106 GUI and reconstruction-control tests pass. Source-characterization
tests inspect both production modules, the class identity is regression-tested,
and touched files pass flake8, pep257 and byte compilation.

## Phase 8b: reconstruction input provenance

`reconstruction/input_provenance.py` now owns immutable capture-set, manifest,
artifact, calibration, confidence-schema, exact camera-transform and optional
offline-mask input admission. `tsdf_reconstruct.py` exposes identical aliases
for every moved symbol, while `offline_resegment.py` imports the owner directly.
Fusion, registration, component filtering, quality selection, output writing and
CLI orchestration remain in `tsdf_reconstruct.py`.

AST comparison proves all 15 moved function bodies are unchanged. Fifty-four
reconstruction tests pass, direct CLI help works outside the repository working
directory, and the touched files pass flake8, pep257 and byte compilation.
