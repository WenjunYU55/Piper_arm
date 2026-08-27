# Architectural cleanup: final equivalence report

## Verdict

The behaviour-preserving architectural cleanup is software-complete against
`origin/main` commit `904dc39e96d5ad36b659cb240b1ad2ab0845775e`.
No ROS interface, launch/configuration file, operator shell entrypoint, runtime
asset path, mission/safety/planner/control policy, dataset schema,
reconstruction output contract, or GUI behavior was intentionally changed.

Work was performed in an isolated worktree and commit series for
`reintegrate/selected-archived-features`. The local and remote `main` branch
were never checked out for editing, rewritten, or pushed.

This is a structural-equivalence result, not a new physical qualification.
No robot, CAN command, camera, GUI mission, or hardware-facing ROS graph was
started during the cleanup.

## Final repository tree

```text
Piper_arm/
├── ARCHITECTURE.md
├── docs/
│   ├── ai/                         authoritative YAML routing and guardrails
│   ├── architecture/               asset policy
│   ├── historical/                 dated scan/session notes
│   └── refactor/                   audit, phase and validation evidence
├── piper_ros_foxy/src/
│   ├── piper_msgs/                 ROS interfaces
│   ├── piper_description/          URDF and runtime geometry
│   ├── piper/                      driver + pure joint-state policy
│   ├── piper_mobile_manipulation/
│   │   └── piper_mobile_manipulation/
│   │       ├── mission/
│   │       ├── infrastructure/
│   │       ├── perception/
│   │       ├── planning/
│   │       ├── execution/
│   │       └── ROS nodes and compatibility facades
│   └── piper_tesseract_foxy/
│       └── piper_tesseract_foxy/
│           ├── protocol/
│           ├── candidate_selection.py
│           ├── bridge_node.py
│           └── worker.py
├── piper_gui/                      GUI model/adapter/viewer components
├── reconstruction/                provenance + reconstruction pipeline
├── L515_camera/                    camera and calibration tooling
├── AI_perception_tests/            isolated perception workers/tests
├── motion_planning/tesseract/      rootless planner tooling
├── tools/                          diagnostics and replay utilities
├── tests/{gui,driver,planning}/    cross-package development tests
└── stable root operator scripts and configuration
```

`docs/ai/project-structure.txt` contains the detailed file-level map.

## Before and after

| Area | Before | After |
|---|---|---|
| Mobile package | Mission, infrastructure, execution, planning and perception owners were flat peers of ROS nodes. | Cohesive responsibility packages own pure logic; old flat paths are explicit facades. |
| Mission node | ROS action/lifecycle code plus calibration, failed-dataset and prior-process resource mechanics. | ROS admission/transport remains; resource decisions moved to `mission/resources.py`; `MissionEngine` remains the sole progression owner. |
| Tesseract | Bridge contained candidate policy; one large contract owned validation and spool mechanics. | Pure candidate policy is separate; `protocol/contract.py` and `protocol/spool.py` have distinct ownership; worker stays cohesive. |
| GUI | Native Tk module also defined the complete ROS node. | `piper_gui/ros_node.py` owns ROS transport; the native module retains presentation and re-exports the class. |
| Reconstruction | Input admission/provenance was mixed with Open3D fusion and output orchestration. | `input_provenance.py` owns immutable admission; fusion/registration/output remain together. |
| Driver | Deterministic joint mapping and coherent CAN-cycle decisions lived inside the hardware node. | Pure policy is testable in `joint_state_policy.py`; all CAN/SDK/timing/command behavior remains in the driver. |
| Repository root | Thirteen development tests and two dated notes were mixed with operator entrypoints. | Tests are grouped by responsibility; dated notes are historical; supported operator paths stay unchanged. |

## Old-to-new module mapping

| Established import/path | Responsibility owner |
|---|---|
| `mission_core.py`, `mission_engine.py`, `mission_spool.py` | `mission/core.py`, `mission/engine.py`, `mission/spool.py` |
| `failure_model.py`, `telemetry_store.py`, `process_supervisor.py` | `infrastructure/failure_model.py`, `infrastructure/telemetry_store.py`, `infrastructure/process_supervisor.py` |
| `plan_authorizer.py`, `capture_coordinator.py`, `scan_execution_modes.py` | `execution/authorization.py`, `execution/capture.py`, `execution/modes.py` |
| `scan_motion.py`, `executor_recovery.py`, `trajectory_runner.py`, `scan_trajectory.py` | `execution/motion.py`, `execution/recovery.py`, `execution/trajectory.py`, `execution/validation.py` |
| `capability_map.py`, `nbv_coverage.py`, `view_generation.py` | `planning/capability.py`, `planning/coverage.py`, `planning/generation.py` |
| `surface_coverage.py`, `ray_hard_culls.py`, `viewpoint_rays.py` | `planning/measured_surface.py`, `planning/ray_culls.py`, `planning/rays.py` |
| `target_acquisition.py`, `target_envelope.py` | `perception/acquisition.py`, `perception/target_envelope.py` |
| `target_landmark_geometry.py`, `obstacle_geometry.py`, `occlusion_policy.py` | `perception/landmark_geometry.py`, `perception/obstacle_geometry.py`, `perception/occlusion.py` |
| Tesseract `contract.py` | `protocol/contract.py` plus `protocol/spool.py` |
| Candidate helpers in `bridge_node.py` | `candidate_selection.py` |
| `PiperGuiRos` in `piper_gui_native.py` | `piper_gui/ros_node.py` |
| Reconstruction input helpers in `tsdf_reconstruct.py` | `reconstruction/input_provenance.py` |
| Deterministic driver joint helpers in `piper_ctrl_single_node.py` | `piper/joint_state_policy.py` |
| Root `test_*.py` files | `tests/gui`, `tests/driver`, and `tests/planning` |

## Compatibility wrappers retained

The 24 former flat mobile modules in the table remain explicit import-only
facades with fixed `__all__` surfaces. The Tesseract `contract.py` facade
exports both protocol validation and `Spool`; the bridge retains its candidate
helper attributes. `piper_gui_native.py`, `tsdf_reconstruct.py`, and
`piper_ctrl_single_node.py` re-export their moved public objects. Tests assert
object identity rather than merely checking that imports resolve.

These wrappers prevent a repository-wide flag day and can be removed only in
a later, separately tested migration.

## Intentionally retained large modules

| Module | Lines before → after | Reason retained |
|---|---:|---|
| `scan_viewpoint_executor_node.py` | 5,082 → 5,084 | Safety-critical ROS integration and sole command authority. Existing pure policies were organised, but moving live callbacks/state without deeper characterization would add risk. |
| Tesseract `worker.py` | 3,051 → 3,051 | Cohesive ROS-free planning backend; its exact source hash participates in capability-map provenance. |
| `target_scan_mission_node.py` | 2,832 → 2,683 | Still a large ROS/action/service adapter; the first safe resource seam moved, while lifecycle callbacks remain coupled to Foxy action behavior. |
| `piper_gui_native.py` | 2,341 → 2,040 | Cohesive Tk view/controller shell after ROS extraction; further widget extraction needs GUI interaction tests. |
| `ray_mission_diagnostics.py` | 1,726 → 1,726 | One append-only diagnostic schema, replay and compatibility owner; splitting the schema lifecycle would reduce cohesion. |
| `piper_ctrl_single_node.py` | 1,999 → 1,710 | Hardware boundary retains CAN, SDK, enable/disable, watchdog and timing assumptions by design. |
| Tesseract `bridge_node.py` | 2,387 → 1,658 | Remaining code is the ROS snapshot/spool/planning adapter; candidate policy was extracted. |
| `scan_capture_node.py` | 1,621 → 1,621 | Transactional exact-time RGB-D persistence and callback scheduling remain tightly coupled. |
| `scan_viewpoint_planner_node.py` | 1,487 → 1,487 | ROS integration of independently owned planning components; algorithm owners already live under `planning/`. |
| `tsdf_reconstruct.py` | 1,805 → 1,351 | Open3D registration, fusion, quality selection and atomic output form one CLI transaction. |

File size alone was never used as a split criterion.

## Removed dead or duplicate code

No production behavior was deleted in this cleanup. Moved implementations were
removed from their former files only after identity/AST characterization and
replacement by one-way facades. Existing apparently similar obstacle,
occlusion and safety checks were retained where they protect different trust
boundaries. No experimental module or asset was deleted merely by inference.

## Large assets

- 115 STL files total about 84 MB and remain at their URDF/planning paths.
- The qualified 8.1 MB capability map remains tracked with its convergence and
  source/model hash contract.
- 213 hand-eye calibration records remain as deployed results and provenance.
- Model checkpoints, generated datasets, bags, point clouds, build/install/log
  output and caches remain untracked.
- Git LFS is an optional future repository-administration change, not part of
  this refactor.

See `docs/architecture/asset_policy.md` for per-class decisions.

## Final validation

| Check | Result |
|---|---|
| Complete mobile-manipulation source suite | 863 passed, 1 hardware-dependent skip |
| Complete Tesseract Python suite | 163 passed |
| Driver functional + description + relocated GUI/driver/planning + L515/reconstruction + GroundingDINO | 305 passed, 1 environment-dependent skip |
| Heavy perception worker / SAM2 live worker | 6 passed / 7 passed |
| Five-package symlink build | PASS |
| Registered `colcon test` | All functional, XML and CMake checks pass; package-wide mobile style targets reproduce 76 existing Flake8/PEP257 findings |
| Added-file Flake8 at the repository's 99-column convention | PASS |
| Python byte compilation | PASS |
| AI architecture YAML | 12 files parsed |
| Git whitespace check | PASS |
| ROS message/service/action, launch, config, shell and XML diff from baseline | Empty |
| Rootless Tesseract compact qualification | PASS, backend 0.35.0.6, collision qualified, `real_arm_motion=false` |
| Rootless Tesseract core qualification | All early stages pass; unchanged `dual_limit_start_acquisition` exceeds its existing internal 150-second budget |

The core qualification limit was reproduced before structural edits at the
same stage and is therefore planner-performance debt, not a cleanup regression.

## Remaining technical debt

1. Further executor and mission-node thinning requires new callback/state
   characterization and supervised requalification; it was deliberately not
   forced into this cleanup.
2. Package-wide mobile Flake8/PEP257 remains red on historical files even
   though functional suites pass and added files are clean.
3. The rootless core dual-limit start case needs a separate performance
   investigation without weakening the bounded planning timeout.
4. Compatibility facades remain until downstream imports are migrated.
5. The native Tk view, capture adapter, planner adapter and diagnostic journal
   have possible later seams, but each needs responsibility-specific tests.
6. Optional mesh Git LFS and old calibration-session archival require separate
   repository/provenance migrations.
7. Physical robot behavior was not requalified by this software-only cleanup.

## Rollback points

Every coherent phase is independently committed: baseline characterization;
mission/infrastructure packaging; mission resources; execution; planning;
perception; Tesseract protocol/candidate boundaries; GUI adapter;
reconstruction provenance; driver policy; and repository layout/assets. A
phase can therefore be reverted without rewriting the protected main branch.
