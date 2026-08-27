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
