# Piper Arm Architecture

This repository is organised around responsibility boundaries while retaining
its established ROS 2 Foxy, operator-script, dataset and hardware interfaces.
The detailed machine-readable routing map is in `docs/ai/`; start with
`docs/ai/00-index.yaml`.

## Repository structure

```text
Piper_arm/
├── piper_ros_foxy/src/
│   ├── piper_msgs/                 ROS interfaces
│   ├── piper_description/          URDF, meshes and robot-description tests
│   ├── piper/                      real PiPER CAN/SDK driver
│   ├── piper_mobile_manipulation/  mission, perception, planning and execution
│   └── piper_tesseract_foxy/       generic Foxy planner bridge and Tesseract adapter
├── piper_gui/                      GUI model, ROS adapter and ray review
├── reconstruction/                 immutable-input reconstruction pipeline
├── L515_camera/                    camera launch and calibration tooling
├── AI_perception_tests/            isolated heavy-perception workers and tests
├── motion_planning/tesseract/      rootless Tesseract runtime tooling
├── motion_planning/curobo/         isolated ROS-free cuRobo adapter and worker
├── tools/                          diagnostics, validation and replay utilities
├── tests/                          cross-package GUI, driver and planning tests
├── docs/ai/                        authoritative architecture YAML
├── docs/refactor/                  audit, phase and equivalence records
├── docs/architecture/              maintainer-facing policies
└── *.sh                            stable operator entrypoints
```

## Dependency direction

```text
ROS nodes / GUI / shell / filesystem adapters
                    ↓
         mission orchestration
                    ↓
 perception → planning → execution policy
                    ↓
       infrastructure and pure utilities
```

Domain packages do not import ROS nodes. ROS nodes own parameters, ROS
entities, message conversion, callback timing and lifecycle. They delegate
deterministic policy to ordinary Python modules. Hardware and subprocess
boundaries stay explicit rather than being hidden behind a framework.

## Responsibility owners

### Mission and infrastructure

- `piper_mobile_manipulation/mission/core.py` owns mission state and contracts.
- `mission/engine.py` is the sole owner of mission progression and shutdown.
- `mission/resources.py` owns mission-bound calibration and failed-dataset
  lifecycle decisions; `mission/spool.py` owns durable handoff.
- `infrastructure/failure_model.py`, `telemetry_store.py`, and
  `process_supervisor.py` own typed failures, immutable observations, and exact
  child-process mechanics respectively.
- `target_scan_mission_node.py` remains the ROS action/admission adapter. It
  must not grow a second mission workflow.

### Perception and planning

- `perception/` owns target acquisition/envelope, landmark geometry, obstacle
  geometry and occlusion policy.
- `planning/` owns capability lookup, measured coverage/NBV ranking,
  generation identity, permanent ray culls and bounded ray geometry.
- Perception measurements are evidence. Planning may predict geometry but must
  not treat predicted surfaces as measured reconstruction input.
- The frozen mission-selected backend remains the authority for exact IK,
  collision and path feasibility. NBV does not know which backend is selected.

### Execution and safety

- `execution/` owns plan authorization, motion/path validation, trajectory
  scheduling/monitoring, settled capture handoff and recovery decisions.
- `scan_viewpoint_executor_node.py` remains the sole autonomous joint-command
  publisher and the runtime ROS/safety integration boundary.
- `safety_evaluator.py` provides named immutable evidence profiles. It does not
  replace executor, planner or driver authority.
- `piper_ctrl_single_node.py` owns CAN/SDK lifecycle, enable/disable, command
  timing, motor watchdogs and feedback. `piper/joint_state_policy.py` contains
  only deterministic joint mapping and coherent-cycle admission.

### Motion planners, GUI and reconstruction

- `piper_tesseract_foxy/bridge_node.py` is the generic ROS snapshot/spool
  router retained in its historical package for Foxy compatibility;
  `candidate_selection.py` owns pure shortlisting; `protocol/` owns schema-v5
  validation and atomic spool transfer. `motion_planning/tesseract/` and
  `motion_planning/curobo/` are isolated backends. Both produce `MotionPlan`;
  neither has a PiPER command interface.
- `piper_gui/ros_node.py` owns GUI ROS transport, while `piper_gui_native.py`
  owns Tk presentation and explicit commissioning controls.
- `reconstruction/input_provenance.py` owns immutable input admission.
  `tsdf_reconstruct.py` owns registration, fusion, quality selection and output.

## Main runtime pipeline

```text
RunTargetScan goal
  → gateway/coordinator admission
  → MissionEngine startup and readiness
  → enabled, feedback-proved startup wrist and rough home
  → target acquisition, tracking and occlusion evidence
  → candidate rays, NBV ranking and reachability filtering
  → frozen Tesseract or cuRobo IK/collision/path qualification
  → generic MotionPlan normalization and validation
  → executor authorization, trajectory and settled capture
  → accepted RGB-D/mask evidence updates coverage
  → repeat until completion or bounded failure
  → pre-home, rough home, storage wrist, disable and owned-process cleanup
  → immutable reconstruction job/output
```

ROS names, schemas, QoS, launch/config semantics, mission phases, safety gates,
trajectory behavior and dataset formats are external contracts. Consult
`docs/ai/30-contracts.yaml`, `40-flows.yaml`, and `50-guardrails.yaml` before
changing them.

## Compatibility policy

Former flat mobile-package modules remain import-only facades for the new
`mission`, `infrastructure`, `execution`, `planning`, and `perception` owners.
Legacy Tesseract ROS aliases and the old Tesseract contract path, GUI
`PiperGuiRos` export, reconstruction input
exports and driver policy exports are also retained. Remove a facade only in a
separate change after all callers migrate and its `docs/ai/60-debt.yaml`
evidence gate is satisfied.

## Adding functionality

- New target measurement or tracking method: `perception/`, with a thin ROS
  adapter beside existing perception nodes.
- New NBV/ray strategy: `planning/`; preserve planner feasibility ownership.
- New motion planner: an isolated ROS-free backend adapter producing the
  generic `MotionPlan`; keep the generic bridge/protocol contract explicit.
- New execution/recovery policy: `execution/`; do not duplicate command
  authority outside the executor.
- New reconstruction method: `reconstruction/`, consuming admitted immutable
  capture inputs.
- New GUI feature: presentation in `piper_gui_native.py` or a focused
  `piper_gui/` component; ROS transport belongs in `piper_gui/ros_node.py`.

Large runtime assets are governed by `docs/architecture/asset_policy.md`.
Cross-package development tests belong under `tests/`; package-specific tests
remain with their package.

See `docs/architecture/motion_planner_backends.md` for planner selection,
contracts, environment isolation, qualification and launch commands.
