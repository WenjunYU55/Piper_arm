# System and feedback diagrams

These diagrams are based on a source-level audit rather than a conceptual feature list. The whole-system map describes the integrated `main` working tree. Software integration and physical hardware qualification remain separate claims.

## Audited repository states

| Ref | Audited commit | What the diagrams claim |
|---|---|---|
| `main` working tree | integrated local main | Production perception/NBV, frozen Tesseract or cuRobo selection, generic `MotionPlan`, common execution, capture, terminal handling and reconstruction |

The cuRobo software path is integrated and its current collision model is
hardware-qualified for supervised 5% target-scan missions. The separate
collision-model opt-in, all common runtime gates, and default-disabled real
motion remain unchanged.

## Visual vocabulary

| Mark | Meaning |
|---|---|
| Graphite solid arrow | Data, evidence or a command-free proposal |
| Blue solid arrow | Mission control or lifecycle ownership |
| Green dashed arrow | Feedback, retry, reacquisition, replanning or readiness return |
| Red solid arrow | Physical motor command; intentionally appears only at the executor-to-driver boundary |
| Gray dashed arrow | Optional backend or auxiliary path |
| Blue / violet / teal boxes | Physical inputs / perception / measured state and immutable data |
| Amber / red / graphite boxes | Planning / safety and recovery / physical actuation and infrastructure |

## One detailed system diagram

<div align="center">
  <a href="../assets/readme/architecture/system-overview.svg">
    <img src="../assets/readme/architecture/system-overview.svg" alt="Detailed PiPER architecture from mission request through perception, NBV, planner backend, execution feedback, capture and reconstruction" width="1000">
  </a>
  <br>
  <sub>Click for the full-resolution SVG.</sub>
</div>

The map keeps five ownership facts visible:

1. The selected planner backend is frozen before mission admission and exactly one worker is supervised.
2. Perception, NBV and planner workers remain command-free.
3. `scan_viewpoint_executor` is the sole autonomous joint publisher.
4. The PiPER driver alone owns MoveJ, SocketCAN, enable/disable and motor feedback.
5. Only accepted immutable observations advance coverage or reconstruction.

## Perception and reacquisition

<div align="center">
  <a href="../assets/readme/architecture/perception-pipeline.svg"><img src="../assets/readme/architecture/perception-pipeline.svg" alt="Target perception, measured geometry, tracking degradation and reacquisition feedback" width="1000"></a>
</div>

Fresh L515 time, mask identity and ambiguity-qualified depth are independent gates. A short tracker outage may publish `LOW_CONFIDENCE` prediction, but planning requires a fresh measured lock. Lost/invalid evidence returns through hold and correlated heavy reacquisition; it cannot update coverage.

## Accepted-only NBV loop

<div align="center">
  <a href="../assets/readme/architecture/viewpoint-planning-pipeline.svg"><img src="../assets/readme/architecture/viewpoint-planning-pipeline.svg" alt="Next-best-view planning with accept, retry, reject, retire and replan feedback" width="1000"></a>
</div>

The diagram separates the effects that were previously collapsed:

- accepted observation → immutable commit → new history generation → measured-coverage rebuild;
- retryable observation → hold achieved FK → one same-pose heavy refresh → re-admit;
- rejected observation → no coverage update → exclude the view and replan;
- exact planner rejection → optionally retire a hard-infeasible ray and request another candidate;
- target loss → hold → reacquire measured target → produce a fresh plan that must be authorized again.

## Planner backend and transport

<div align="center">
  <a href="../assets/readme/architecture/planner-backend-pipeline.svg"><img src="../assets/readme/architecture/planner-backend-pipeline.svg" alt="Frozen Tesseract or cuRobo backend, worker readiness, validated response, generic transport and unchanged common execution" width="1000"></a>
</div>

The generic bridge publishes backend-neutral `MotionPlan`, `MotionPlanStatus`, `PlannerReadiness` and provenance while retaining Tesseract aliases only in Tesseract mode. Worker heartbeat, generation, schema, backend and model hashes must match the frozen request.

The cuRobo worker uses MotionGen 0.7.8 `plan_single` for camera poses and `plan_single_js` for home motions. Fixed Bunker geometry uses exact meshes, while moving links use the audited 69-sphere model. The operator reported successful physical end-to-end qualification on 2026-09-02, limited to supervised 5% target-scan motion; `conservative_geometry` remains false and the documented coverage gaps prevent any claim of Tesseract-equivalent geometry. There is no automatic Tesseract fallback.

## Execution feedback and recovery

<div align="center">
  <a href="../assets/readme/architecture/execution-safety-pipeline.svg"><img src="../assets/readme/architecture/execution-safety-pipeline.svg" alt="Plan validation, authorization, runtime physical feedback, hold-refresh-resume and terminal recovery" width="1000"></a>
</div>

Common plan validation checks six finite joints, time order, 20 Hz scheduling, maximum step, speed-scaled MoveJ limits and fresh matching hashes. Authorization then checks mission identity, TTL, backend, target drift, dependencies and the complete path.

During execution, joints, arm status, controller limits, camera timing, target tracking, scene quality, following error, timeout, settle state and holder/floor clearance return to the executor. Transient evidence follows hold → refresh → re-authorize → resume of the exact interrupted stage. Cancellation or hard fault follows the bounded home/disable recovery sequence. Motor-authority loss allows no further command.

## Capture admission, rejection and reconstruction

<div align="center">
  <a href="../assets/readme/architecture/capture-reconstruction-pipeline.svg"><img src="../assets/readme/architecture/capture-reconstruction-pipeline.svg" alt="Settled capture, confidence-qualified burst, atomic commit, rejection feedback, safe terminal state and reconstruction" width="1000"></a>
</div>

The capture service uses the exact mask/RGB stamp and exactly 20 new native depth/confidence frames. Admission requires confidence grade ≥ 8, at least 0.50 support, per-pixel median depth, calibrated intrinsics/TF, achieved FK, plan provenance, and fresh quality/occlusion evidence. Atomic artifacts and their manifest SHA form one schema-2 observation; partial files never count.

After the safe mission terminal and optional tracked-base-home correlation, reconstruction validates immutable input, then runs target-only TSDF by default with optional bounded GICP and target-excluded scene pose-graph refinement. Reconstruction failure is reported without changing the mission result.

## Hardware and compute boundaries

<div align="center">
  <a href="../assets/readme/architecture/hardware-topology.svg"><img src="../assets/readme/architecture/hardware-topology.svg" alt="Robot hardware, isolated compute environments and motor-command boundary" width="1000"></a>
</div>

The eye-in-hand L515 is the qualified active scan sensor. ZED and LiDAR parts under [`CAD/`](../../CAD/) are mechanical provision, not current runtime inputs. The tracked base remains stationary and externally controlled; this repository sends no chassis command.

## Implementation evidence

Primary system descriptions:

- [`ARCHITECTURE.md`](../../ARCHITECTURE.md)
- [`docs/architecture/system-overview.md`](system-overview.md)
- [`docs/ai/10-system-map.yaml`](../ai/10-system-map.yaml)
- [`docs/ai/40-flows.yaml`](../ai/40-flows.yaml)

Planner integration evidence:

- [planner backend contract](motion_planner_backends.md)
- [integrated architecture](../../ARCHITECTURE.md)
- [generic planner ROS interfaces](../../piper_ros_foxy/src/piper_mobile_manipulation/msg/)
- [cuRobo tests](../../tests/curobo/)

Subsystem evidence:

- [`integration/track_robot_description/README.md`](../../integration/track_robot_description/README.md)
- [`L515_camera/README.md`](../../L515_camera/README.md)
- [`CAD/enclosure-v4/README.md`](../../CAD/enclosure-v4/README.md)
- [`reconstruction/`](../../reconstruction/)

## Regeneration and review

The committed SVGs are generated deterministically with the Python standard library:

```bash
python3 docs/architecture/diagrams/generate_diagrams.py
```

After any architecture change, regenerate all figures, render them to raster images for visual inspection, and verify that branch status, command ownership and every state-changing feedback path are still explicit. See the [diagram-source rules](diagrams/README.md).
