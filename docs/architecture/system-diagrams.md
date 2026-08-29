# System diagrams

These diagrams are rendered directly by GitHub from Mermaid source. They
describe the current repository interfaces and intentionally distinguish
implemented command paths from mechanical or future integration boundaries.

## Hardware and software architecture

```mermaid
flowchart LR
  subgraph TRACKED["Tracked robot / operator network"]
    CLIENT["Tracked mission client"]
    ODOM["Tracked localisation<br/>odom to base_link"]
    BASE["Bunker Pro 2 tracked base"]
  end

  subgraph GATEWAY["Tracked-to-arm isolation boundary"]
    GW["target_scan_gateway_node<br/>goal admission and TF snapshot"]
    MSPOOL["Private mission spool<br/>atomic goals, status and results"]
  end

  subgraph HOST["Arm host: Ubuntu 20.04 / ROS 2 Foxy"]
    GUI["PiPER operator GUI"]
    MISSION["target_scan_mission_node<br/>MissionEngine"]

    subgraph PERCEPTION["RGB-D perception"]
      RS["RealSense ROS wrapper"]
      GEOM["Target, scene, quality<br/>and occlusion geometry"]
      AI["GroundingDINO and SAM2<br/>isolated workers"]
    end

    subgraph PLANNING["View and motion planning"]
      NBV["Measured coverage<br/>and next-best-view ranking"]
      BRIDGE["Foxy Tesseract bridge"]
      TESS["Isolated Tesseract 0.35 worker<br/>IK, collision and path checks"]
    end

    EXEC["scan_viewpoint_executor<br/>sole autonomous joint publisher"]
    DRIVER["piper_ctrl_single_node<br/>CAN and motor authority"]
    CAPTURE["scan_capture_node"]
    DATASET["Validated RGB-D, mask,<br/>pose and robot-state dataset"]
    RECON["Offline TSDF / bounded GICP<br/>reconstruction"]
  end

  subgraph HARDWARE["Arm hardware"]
    CAN["USB-CAN / SocketCAN can0"]
    ARM["PiPER 6-DOF arm"]
    L515["Intel RealSense L515<br/>eye-in-hand sensor"]
  end

  CLIENT --> GW
  ODOM --> GW
  GW <--> MSPOOL
  MSPOOL <--> MISSION
  GUI --> MISSION

  MISSION --> RS
  MISSION --> EXEC
  MISSION --> DRIVER
  L515 --> RS
  RS --> GEOM
  GEOM <--> AI
  GEOM --> NBV
  NBV --> BRIDGE
  BRIDGE <--> TESS
  BRIDGE --> EXEC
  EXEC --> DRIVER
  DRIVER <--> CAN
  CAN <--> ARM
  EXEC --> CAPTURE
  RS --> CAPTURE
  CAPTURE --> DATASET
  DATASET --> NBV
  DATASET --> RECON

  ARM --- L515
  BASE -. fixed mechanical mount .-> ARM
```

The dashed base-to-arm link is deliberate: the tracked chassis provides the
mounting frame, task request and pose transform, but this repository does not
command the base. Mounted-chassis collision TF and brake authority remain
deferred integration work.

## Autonomous target-scan flow

```mermaid
flowchart TD
  GOAL["RunTargetScan goal<br/>label and rough target pose"]
  SNAP["Optional gateway snapshots<br/>odom to piper_base_link"]
  READY["Start owned processes and prove readiness<br/>camera, calibration, joints, AI and Tesseract"]
  START["Enable after fresh all-six-axis evidence<br/>startup wrist and rough home"]
  ACQUIRE["Acquire target<br/>L515 RGB-D, GroundingDINO, SAM2 and aligned depth"]
  RANK["Generate and rank candidate views<br/>accepted measured coverage / NBV"]
  PLAN["Tesseract qualification<br/>exact IK, collision and path feasibility"]
  AUTH{"Fresh plan identity<br/>and safety gates valid?"}
  MOVE["Executor sends approved MoveJ path"]
  SETTLE["Prove convergence and settled feedback"]
  CAPTURE["Persist synchronized RGB, depth, mask,<br/>intrinsics, camera pose, joints and metadata"]
  QUALITY{"Fresh GOOD target<br/>and acceptable occlusion?"}
  ACCEPT["Atomically accept view<br/>and update measured coverage"]
  DONE{"Completion proof?<br/>8 to 24 views and convergence<br/>or typed safe-frontier exhaustion"}
  RECOVER{"Bounded retry,<br/>reacquire or replan"}
  HOME["Pre-home, rough home and storage wrist"]
  DISABLE["Settled hold, all-six disable<br/>and owned-process cleanup"]
  RESULT["Immutable dataset and mission result"]
  RECON["Offline reconstruction"]

  GOAL --> SNAP --> READY --> START --> ACQUIRE --> RANK --> PLAN --> AUTH
  AUTH -->|yes| MOVE --> SETTLE --> CAPTURE --> QUALITY
  AUTH -->|no| RECOVER
  QUALITY -->|yes| ACCEPT --> DONE
  QUALITY -->|no| RECOVER
  RECOVER -->|budget remains| RANK
  RECOVER -->|terminal or unsafe| HOME
  DONE -->|more information required| RANK
  DONE -->|complete| HOME
  HOME --> DISABLE --> RESULT --> RECON
```

Tesseract proposes collision-qualified motion; the executor remains the sole
autonomous joint-command publisher, and the PiPER driver owns enable/disable,
CAN timing, motor health and feedback. GroundingDINO/SAM2 perception does not
have a command path to the gripper.

## CAD-to-robot relationship

```mermaid
flowchart LR
  ASM["FullCase.SLDASM<br/>editable enclosure assembly"]
  PARTS["SolidWorks part sources"]
  DXF["DXF panel profiles<br/>laser cutting, mm"]
  STL["STL fabrication exports<br/>3D printing, mm scale"]
  MF["Seven-plate 3MF slicer project"]
  ENC["Tracked-platform enclosure<br/>frame, panels and battery retention"]
  OPTIONAL["Optional platform sensing<br/>ZED and LiDAR mounts"]
  WRIST["PiPER wrist L515 holder"]
  L515["Current L515 RGB-D scan sensor"]
  RUNTIME["Qualified URDF / Tesseract meshes<br/>separate runtime ownership"]

  ASM --> PARTS
  PARTS --> DXF
  PARTS --> STL
  STL --> MF
  DXF --> ENC
  STL --> ENC
  STL --> OPTIONAL
  STL --> WRIST
  WRIST --> L515
  WRIST -. design source only .-> RUNTIME
```

The CAD snapshot supports manufacture and design review. Runtime meshes remain
at their established URDF/Tesseract paths and may only be regenerated through
the repository's hash-checked geometry and qualification workflow.

## Evidence sources

- [`README.md`](../../README.md)
- [`ARCHITECTURE.md`](../../ARCHITECTURE.md)
- [`docs/ai/10-system-map.yaml`](../ai/10-system-map.yaml)
- [`docs/ai/40-flows.yaml`](../ai/40-flows.yaml)
- [`integration/track_robot_description/README.md`](../../integration/track_robot_description/README.md)
- [`CAD/enclosure-v4/README.md`](../../CAD/enclosure-v4/README.md)
