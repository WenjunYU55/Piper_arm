# System overview

This document explains the runtime boundaries of the PiPER active-view scanning
system. For package-level ownership and dependency rules, see
[`ARCHITECTURE.md`](../../ARCHITECTURE.md). For operator commands, use
[`OPERATOR_COMMANDS.md`](../../OPERATOR_COMMANDS.md).

## Design rule

ROS, GUI, and CLI components adapt external inputs into application requests.
Mission orchestration owns progression. Perception and NBV decide what target
view should be attempted next. A selected planner decides whether the robot can
reach that view. The common authorization and execution path decides whether a
plan may command the arm.

Planner workers are command-free. They cannot publish PiPER commands or bypass
the shared safety boundary.

## Runtime and process boundaries

```mermaid
flowchart TB
    OP["Operator GUI / coordinator"]

    subgraph FOXY["ROS 2 Foxy control environment"]
        MISSION["Target-scan coordinator<br/>MissionEngine"]
        SUP["ProcessSupervisor<br/>generation ownership · health · bounded cleanup"]
        VISIONROS["L515 camera, timestamp watchdog,<br/>Foxy bridges, tracking, geometry, and occlusion"]
        NBV["Target model, measured coverage,<br/>NBV and candidate rays"]
        BRIDGE["Generic MotionPlannerBridge<br/>ROS snapshot + contract validation"]
        MOTION["Generic MotionPlan"]
        NORMALIZE["Common trajectory normalization<br/>and execution validation"]
        SEP["ScanExecutionPlan"]
        AUTH["PlanAuthorizer"]
        GATES["Runtime safety gates<br/>freshness · drift · following error · timeout"]
        EXEC["scan_viewpoint_executor<br/>TrajectoryRunner"]
        DRIVER["PiPER CAN/SDK driver"]
    end

    subgraph PERCEPTION["Isolated Python 3.10 CUDA perception environment"]
        HEAVY["GroundingDINO + SAM 2<br/>heavy refresh worker"]
        LIVE["SAM 2 live worker"]
        PN["ROS-free; no motor interface"]
    end

    PSPOOL[("Permission-bounded<br/>perception spool")]

    subgraph TESS["Backend option A: rootless Tesseract runtime"]
        TW["ROS-free worker<br/>IK · collision · path planning"]
    end

    subgraph CUROBO["Backend option B: explicit cuRobo Python/CUDA environment"]
        CW["ROS-free MotionGen worker<br/>IK · collision · path planning"]
        CQ["Current model<br/>hardware_qualified = true<br/>supervised 5% scope"]
    end

    TSPOOL[("Tesseract spool")]
    CSPOOL[("cuRobo spool")]
    SELECT{"Frozen backend<br/>one worker only"}
    SAFE["Planner workers have no motor publisher<br/>and cannot command PiPER directly"]

    OP -->|RunTargetScan| MISSION
    MISSION --> SUP
    SUP -. starts and stops .-> VISIONROS
    SUP -. starts and stops .-> SELECT
    SUP -. starts and stops .-> DRIVER

    VISIONROS <--> PSPOOL
    PSPOOL <--> HEAVY
    PSPOOL <--> LIVE
    VISIONROS --> NBV
    NBV --> MISSION
    NBV --> BRIDGE

    SELECT -->|Tesseract mission| TW
    SELECT -->|cuRobo mission| CW
    TW <--> TSPOOL
    CW <--> CSPOOL
    BRIDGE <--> TSPOOL
    BRIDGE <--> CSPOOL

    MISSION -->|RequestMotionPlan| BRIDGE
    BRIDGE --> MOTION --> NORMALIZE --> SEP --> AUTH --> GATES --> EXEC
    EXEC -->|/joint_ctrl_single| DRIVER

    HEAVY -.-> PN
    LIVE -.-> PN
    TW -.-> SAFE
    CW -.-> SAFE
    CW -.-> CQ
```

`ProcessSupervisor` owns the top-level vision launcher. That launcher owns its
nested camera, timestamp watchdog, bridge, geometry, GroundingDINO, and SAM 2
process groups. The selected planner is the only planner worker started for a
mission generation.

## Mission lifecycle

```mermaid
stateDiagram-v2
    [*] --> STARTING
    STARTING --> PREFLIGHT: driver, camera, perception, hand-eye, planner, scan stack ready
    PREFLIGHT --> ENABLE_AND_HOLD: motion opt-ins and authorization pass
    ENABLE_AND_HOLD --> STARTUP_WRIST: post-enable feedback proved
    STARTUP_WRIST --> ROUGH_HOME: wrist branch proved
    ROUGH_HOME --> ROUGH_ACQUISITION: rough home proved

    ROUGH_ACQUISITION --> TARGET_LOCK: approved acquisition look
    TARGET_LOCK --> ROUGH_ACQUISITION: target absent, bounded retry
    TARGET_LOCK --> WORKFLOW_READY: measured target acquired
    WORKFLOW_READY --> VIEW_PLANNING: multiview evidence ready

    VIEW_PLANNING --> CAPTURING: generic plan approved
    CAPTURING --> VIEW_PLANNING: view accepted or pose excluded
    VIEW_PLANNING --> TERMINAL_RECOVERY: complete, exhausted, failed, or cancelled
    CAPTURING --> TERMINAL_RECOVERY: complete, failed, or cancelled
    ROUGH_ACQUISITION --> TERMINAL_RECOVERY: acquisition failure or cancel

    TERMINAL_RECOVERY --> REQUIRED_WRIST: STARTUP_WRIST if not already proved
    REQUIRED_WRIST --> PRE_HOME: fresh current-state qualification
    PRE_HOME --> FINAL_ROUGH_HOME
    FINAL_ROUGH_HOME --> STORAGE_WRIST
    STORAGE_WRIST --> DISABLE
    DISABLE --> CLEANUP
    CLEANUP --> RESULT
    RESULT --> [*]

    TERMINAL_RECOVERY --> MOTOR_CONTROL_LOST: motor authority unavailable
    MOTOR_CONTROL_LOST --> DISABLE_PROOF: issue no further motion command
    DISABLE_PROOF --> CLEANUP: all six disabled proved
```

Cancellation and controllable failures enter the same bounded terminal recovery
path. A motor-control-loss state issues no further motion command and requires
disable evidence. Software cancellation remains distinct from the physical
emergency-stop procedure.

## Responsibility boundaries

| Area | Owner | Must not own |
|---|---|---|
| Mission progression | `mission/engine.py` with the target-scan ROS adapter | Planner algorithms or PiPER command encoding |
| Perception and tracking | L515/semantic/tracking ROS adapters plus isolated GPU workers | Motion authorization |
| NBV and coverage | `piper_mobile_manipulation/planning/` | Planner backend selection or robot commands |
| Planner selection | Typed next-mission configuration frozen into the mission | Mid-mission fallback |
| Planner-native logic | Isolated Tesseract or cuRobo worker | ROS motor topics, CAN, or mission state |
| Plan authorization | `PlanAuthorizer` and common execution validation | Backend-native tensors or planner shortcuts |
| Automatic commands | `scan_viewpoint_executor` and `TrajectoryRunner` | Perception model inference or NBV scoring |
| Process lifetime | `ProcessSupervisor` and owned child launchers | Discovery/adoption of unrelated processes |

## Dataset boundary

Each accepted view persists synchronized RGB, aligned depth, mask, intrinsics,
joint state, camera pose, target/session correlation, planner provenance, and
selection diagnostics under `datasets/active_scan/`. Reconstruction consumes a
completed dataset asynchronously; reconstruction failure does not retroactively
change the robot mission's shutdown result.

## Where to extend the system

- Add a perception method behind the existing perception/target contract.
- Add an NBV strategy under `planning/` without importing a planner backend.
- Add a motion planner through the generic request/result and `MotionPlan`
  boundary, then reuse common normalization, authorization, and execution.
- Add reconstruction methods under `reconstruction/`; do not add robot-command
  dependencies there.
- Add operator controls through the GUI model/client boundary and persist only
  next-mission settings while a mission is idle.
