# Final refactor architecture

> 2026-08-20 focused tracking repair: `depth_to_3d_node.py` now requires a
> fresh correlated semantic mask and never creates a target measurement from
> an unmasked ROI. The existing `target_tracker_node.py` remains the sole
> target estimate, using near-static prediction, Mahalanobis innovation gating
> and bounded prediction-only output through the existing status interfaces.
> NBV, Tesseract, capture coverage, motion and mission sequencing are unchanged.

> 2026-08-20 live cadence/shortlist qualification: measured valid-target gaps
> reached 3.8868 seconds, so the same authoritative stationary tracker now
> retains prediction for five seconds with 0.01 process noise and gates with
> base measurement noise rather than confidence-scaled noise. The existing
> Tesseract bridge keeps global NBV leaders and reserves one of its existing
> fallback slots for a nearby positive-information continuity candidate. This
> changes neither NBV gain/coverage nor Tesseract feasibility ownership.

> 2026-08-20 NBV repair: target measurement remains owned by
> `depth_to_3d_node.py` and timestamped base-frame estimation by
> `target_tracker_node.py`; `nbv_coverage.py` owns cumulative accepted coverage
> and information ranking; the Tesseract bridge/worker own bounded exact-aim
> feasibility; the executor owns achieved physical pose and retry handoff; the
> capture node alone validates persistence. Accepted reconstruction stays fixed
> in `base_link`. These remain separate owners rather than a new manager.

> 2026-08-17 update: the later integration simplification made
> `MissionEngine` the sole mission/shutdown authority, removed the uncalled
> frozen mission bodies, and replaced the duplicate shadow evaluator with the
> small immutable `RuntimeGatePolicy` mapping. Direct home now re-qualifies
> from current arm/control evidence on every owned controllable failure; only
> confirmed motor-control loss suppresses commands. This post-audit change is
> software-tested (609 mobile tests, 82 affected driver/GUI/diagnostic tests,
> changed-file lint clean, normal five-package Foxy build) but requires
> supervised physical requalification. Historical package-wide lint debt
> remains outside the changed files.

## Scope

This is the Phase 10 architecture after the behavior-preserving Phase 0--9
extractions. It records ownership and dependency direction; it does not replace
the public interface catalog in `external_contracts.md` or the numerical rules
in `safety_invariants.md`. No ROS name, QoS policy, message field, parameter
default, threshold, motion command, TF meaning, calibration value, perception
algorithm, or Tesseract behavior changed during Phase 10.

## Module structure

### Domain and application layer (ROS-free)

| Module | Ownership |
|---|---|
| `infrastructure/failure_model.py` | Typed `FailureCode`, tags, immutable failure evidence, and the one-way adapter from legacy text-only boundaries. |
| `infrastructure/telemetry_store.py` | Thread-safe telemetry updates and immutable, time-coherent `TelemetrySnapshot` values. |
| `configuration.py` | Typed immutable startup configuration for the mission coordinator and viewpoint executor, preserving all ROS names/defaults/overrides. |
| `mission/core.py` | Mission phases, session state, queue/deduplication records, and result primitives. |
| `mission/engine.py` | The admitted autonomous mission sequence and terminal shutdown sequence. |
| `mission/resources.py` | Mission-bound calibration provenance, guarded failed zero-capture dataset lifecycle, and previous-generation cleanup target selection. |
| `mission/spool.py` | Atomic, permission-bounded mission task, status and result persistence. |
| `infrastructure/process_supervisor.py` | Exact owned-process generations, environment construction, health checks, reverse-order group shutdown, and shutdown reports. |
| `safety_evaluator.py` | Pure named immutable runtime-gate policy mapping. It contains no ROS, thresholds, telemetry evaluator, command or second permission authority. |
| `execution/authorization.py` | Exact mission/plan identity, expiry, target drift, dependency evidence, typed authorization decisions, and configured-home stage/endpoint policy. |
| `execution/trajectory.py` | Pure scheduling and feedback decisions for one already-authorized Tesseract trajectory. |
| `execution/capture.py` | Settling-to-capture sequencing and typed capture retry/replacement/abort decisions. |
| `execution/recovery.py` | Typed retry, reacquire, replan, and abort policy. |
| `execution/modes.py`, `execution/motion.py`, `execution/validation.py` | Execution-mode semantics, conservative kinematics/path validation, and exact scheduled-trajectory validation. |
| Scan/perception helpers | Geometry, acquisition, coverage, scan history, motion validation, occlusion policy, capture persistence, and reconstruction job contracts. |
| `planning/capability.py`, `planning/coverage.py`, `planning/generation.py` | Capability lookup, accepted object-centric NBV evidence/ranking, and immutable proposal-generation identity. |
| `planning/measured_surface.py`, `planning/ray_culls.py`, `planning/rays.py` | Persisted measured coverage, permanent ray-elimination protocol, and bounded target-centred ray geometry. |

The former root-level names remain explicit import-only compatibility facades;
they contain no duplicate implementation or policy.

### ROS boundary layer

| Boundary | Ownership |
|---|---|
| `target_scan_gateway_node.py` | Always-on external action/service endpoint, durable spool handoff, result replay, and deferred reconstruction trigger. |
| `target_scan_mission_node.py` | ROS action admission/queueing, message conversion, feedback/result publication, service adapters, telemetry callbacks, and dependency injection into `MissionEngine`. |
| `scan_viewpoint_executor_node.py` | Sole optional autonomous joint-command publisher; ROS plan normalization, exact path/geometry revalidation, mission-authorized direct home service, telemetry adapters, command publication, capture service calls, and status/history publication. |
| Acquisition/planning/perception/capture nodes | Their existing topic/service adapters and subsystem-specific algorithms; they do not own the mission sequence. |
| PiPER driver | CAN/SDK authority, arm enable service, command input, and low-level feedback/limit publication. |
| Tesseract bridge/worker | Typed spool planning contract, collision/IK/path qualification, bounded pass-through smoothing with dense fallback, and DIRECT_MOVEJ-versus-streamed-detour evidence. It no longer owns production configured-home planning. |

### GUI boundary

| Module | Ownership |
|---|---|
| `piper_gui/view_model.py` | Immutable, Tk/ROS-free presentation state and operator-input validation. |
| `piper_gui/ros_client.py` | `RunTargetScan` goal, feedback, cancellation, and result mapping. |
| `piper_gui/app.py` | GUI/ROS bootstrap. |
| `piper_gui_native.py` | Tk presentation, read-only diagnostics, and clearly separated commissioning controls. Its only child process is the preview-only RViz joint editor. |

The GUI is not an autonomous controller. It does not own production retries,
replanning, occlusion decisions, scan sequencing, mission safety, or production
child-process lifecycle.

## Dependency direction

```text
tracked robot / GUI
        |
        v
ROS gateway/action nodes  --->  ROS message/service/action types
        |
        v
MissionEngine  ---> typed configuration / failures / cancellation / session
        |              |
        |              +---> TelemetryStore snapshots
        +---> ProcessSupervisor
        +---> ROS operation adapters
                         |
                         v
                 viewpoint executor ROS node
                         |
                         +---> PlanAuthorizer
                         +---> TrajectoryRunner
                         +---> CaptureCoordinator
                         +---> RecoveryPolicy
                         +---> RuntimeGatePolicy (named evidence categories)
                         |
                         v
              driver / Tesseract / perception / capture
```

Dependencies point inward from ROS/Tk/subprocess adapters to typed application
components. Domain/application modules do not import `rclpy`, Tk, or
`subprocess`. `ProcessSupervisor` is the deliberate exception for owned process
mechanics; it has no ROS or mission-policy authority.

## Ownership rules

### Mission ownership

`MissionEngine` owns the established sequence: startup, readiness, enable,
startup wrist, rough home, acquisition, target lock, occlusion probe,
view planning/capture iteration, terminal pre-home, rough home, storage wrist,
the status-only `HOLDING` compatibility phase, disable, process cleanup, and
result classification. Autonomous startup/shutdown does not call the redundant
hold service. The mission ROS node owns
admission, queuing, action mechanics, durable boundaries, and calls to concrete
ROS dependencies. It must not grow a second workflow.

### Safety ownership

The executor gates remain authoritative. `RuntimeGatePolicy` selects one
complete named evidence profile so call sites cannot assemble contradictory
boolean combinations; it does not evaluate or authorize motion itself. Tesseract
retains IK/collision/path authority; the executor retains live identity,
telemetry, limit, attached-tool, target, timing, and command-publication gates;
the driver retains low-level all-axis enable/fault watchdog authority. Return
home retains its narrowly scoped self-collision exception, live control,
configured target/stage, attached-tool external-clearance and convergence
proofs. It does not depend on moving-camera perception or a Tesseract
controller-limit hash.

### Process ownership

`ProcessSupervisor` owns only process groups explicitly started for the active
mission generation. It never discovers or adopts processes by name and cleans
up in reverse dependency order with the existing escalation policy. The GUI
preview child, camera wrapper manifest, Tesseract singleton lock, gateway
reconstruction subprocess, and persistent gateway/coordinator/RViz processes
remain with their distinct owners; merging them would change lifecycle policy.

If operator recovery is required before home or disable is proved,
`MissionEngine` revokes mission authorization and releases only the exact
non-command vision, hand-eye, and Tesseract-worker groups. The driver and scan
executor remain available for powered recovery. Before another mission is
admitted, the ROS adapter may retry cleanup of the supervisor's exact previous
handles when no driver remains, or when fresh typed feedback proves all six
motors disabled. It never adopts a process by name or infers disable from stale
state.

### Telemetry ownership

ROS callbacks write observations and original receipt/source timestamps to a
node-local `TelemetryStore`. A decision takes one immutable snapshot and uses
the unchanged freshness rules against that snapshot. Plans, command targets,
retry counters, process ownership, capture history, and shutdown proofs are
derived/session state and do not belong in telemetry.

### Closed-loop view-generation ownership

`nbv_coverage.py` owns measured voxel evidence and normalized marginal
information scoring (unknown plus direction-novel surface support divided by
projected target support, followed by accepted-view angular novelty);
`view_generation.py` owns immutable session/accepted-count generation
identity. The viewpoint planner publishes a labelled generation, the existing
preliminary filter preserves it, and the Tesseract bridge emits a receipt only
after caching it. MissionEngine waits for a post-barrier matching receipt
before requesting one plan. The bridge bounds membership across distinct view
directions; the worker remains the sole IK/collision feasibility authority and
tries candidates in NBV-rank order. `scan_capture_node.py` persists selected
policy/rank/gain only through exact `plan_id` correlation.

### GUI ownership

The GUI owns presentation, presentation state, ROS client transport, read-only
diagnostics, and explicit manual commissioning. Production mission and safety
behavior is reached only through the existing action interface. Commissioning
publishing is locked while an autonomous action owns the arm.

## Phase 10 cleanup disposition

Repository reference search and the complete characterization suite proved the
following production code inert and it was removed:

- `utils/transforms.py`, whose two accessors had no repository import or call;
- unused `point_distance`, `make_pose_stamped`, and `point_from_xyz` helpers
  from `utils/geometry.py`;
- an unused velocity-array calculation in the coordinator's final-hold proof;
- an unused qualification-only import of `reverse_sdk_movej_points`.

Four pre-existing formatting-only lint findings in touched files were also
removed; these edits do not change an expression or control-flow branch.

The `tf2_geometry_msgs` import in `target_tracker_node.py` is intentionally
retained and annotated: importing it registers geometry conversions with tf2.
Repeated node-local parsing/parameter helpers were not consolidated because
their callback, configuration, or synchronization ownership differs.

The following apparent cleanup candidates are intentionally retained:

- `latest_*` and executor receipt-time mirrors still consumed by diagnostics
  and characterization seams;
- `ManagedProcessSet` and configuration exports used by compatibility tests;
- `piper_gui_automation.py`, retained as archived Phase 1 characterization for
  the review period specified in `docs/ai/60-debt.yaml`;
- legacy/fake commissioning ROS nodes, whose installed ROS interfaces remain
  documented external contracts;
- vendor, calibration, model, generated-message, and isolated-runtime code.

## Change rule after Phase 10

New behavior belongs in the owner above and must preserve dependency direction.
The uncalled `_legacy_run_pipeline`/`_legacy_safe_shutdown` bodies and duplicate
`SafetyEvaluator`/`SafetyComparisonLogger` were removed on 2026-08-17 after
reference search and focused characterization; their public compatibility
delegates and the one authoritative executor gate remain.

Removing a retained compatibility surface requires satisfying its explicit
evidence gate in `docs/ai/60-debt.yaml`, updating `docs/ai`, and rerunning the
linked characterization, build, and command-free planning checks. The
2026-08-17 policy cleanup did not promote a new safety authority; it removed
the unused duplicate and requires separate physical requalification.

## Phase 10 validation record

Validation on 2026-08-15 was software-only. No arm, camera, GPU worker, driver,
GUI, coordinator, or other hardware-facing process was started.

- complete `piper_mobile_manipulation` suite: 580 passed;
- root GUI/reconstruction/calibration selection: 69 passed, 1 existing
  hardware-dependent skip;
- isolated perception suites: 5 heavy-worker, 6 SAM2-worker, and 19 target
  selection tests passed;
- normal five-package Foxy `colcon build --symlink-install`: passed;
- registered `colcon test`: every functional, message, XML, and CMake-lint
  target passed; the two existing mobile-package style targets remain nonzero;
- style baseline: 87 Flake8 and 21 PEP257 findings, improved from the Phase 9
  baseline of 95 and 21; all Phase 10-touched Python files pass both linters;
- no repository type checker or automatic formatter is configured; Python 3.8
  byte compilation, AI YAML parsing, and `git diff --check` passed;
- both rootless Tesseract qualifications passed with backend `0.35.0.6`,
  collision-model qualification true, and `real_arm_motion=false`.
