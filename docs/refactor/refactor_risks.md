# Phase 0 refactor risk assessment

Phase 8 update: the Phase 0 GUI complexity/process/retry findings below remain
historical characterization, but the production GUI no longer imports or runs
that alternate Step 1-5 controller. `piper_gui/native_app.py` delegates action
lifecycle to `piper_gui/ros_client.py`, presentation state to
`piper_gui/view_model.py`, and owns only a preview RViz child. See
`phase8_gui_responsibility_map.md`.

Phase 9 update: the coordinator and executor now use immutable typed startup
configuration. The legacy `configured_value` fallback is test-only; restoring
runtime `get_parameter()` calls would reintroduce mixed-epoch decisions.
Cross-node numerical values must not be consolidated merely because their
literals match; see `phase9_configuration.md`.

## Method

The three primary files were parsed with the Python AST. Function size is
physical source lines. The complexity indicator below is a deliberately simple
branch count (function entry plus `if`, loop, `try`, conditional expression,
comprehension and boolean branches); it is a comparison aid, not a formal
McCabe score. Repository interfaces, launch/config sources, tests and
`docs/ai` ownership/guardrails were then traced manually.

## File-level findings

| File | LOC | Functions/methods | `latest_*` names | Highest-risk function |
|---|---:|---:|---:|---|
| `target_scan_mission_node.py` | 2,684 | 95 | 18 | `run_pipeline`: 313 LOC, branch indicator 33 |
| `scan_viewpoint_executor_node.py` | 3,878 | 114 | 10 | `tesseract_plan_cb`: 517 LOC, branch indicator 143 |
| `piper_gui/native_app.py` | 3,715 | 102 | 11 | `drain_events`: 549 LOC, branch indicator 127 |

### Long/high-complexity functions

Mission coordinator hotspots:

- `run_pipeline` (313 LOC/33): startup, acquisition, occlusion, NBV, capture
  completion and retry policy in one method.
- `safe_shutdown` (150/32): motor-loss, never-enabled, held-home, storage,
  disable and child cleanup branches.
- `execute_cb` (149/26): admission, result caching, exception translation,
  shutdown and action terminal state.
- `prove_return_home_for_shutdown` (135/37): hold, service selection, plan
  correlation, process health, executor interpretation and feedback proof.
- `start_processes` (119/7), `prove_current_hold` (60/15),
  `failure_code_for_reason` (30/24).

Executor hotspots:

- `tesseract_plan_cb` (517/143): typed contract validation, trajectory
  conversion, plan-kind exceptions, collision/visibility/timing checks and
  cache mutation.
- `prepare_current_view` (186/46): live safety, current path preparation,
  capture/acquisition behavior and retry state.
- `runtime_reasons` (147/51): live joints/motors/limits/tracking/camera/target/
  obstacle/plan gates.
- `approve_cb` (129/38), `heavy_refresh_status_cb` (108/21), motion tick
  methods, abort-return setup and `execution_tick`.

GUI hotspots:

- `drain_events` (549/127): presentation, ROS callback interpretation,
  generation filtering, automation state transitions and error recovery.
- `handle_scan_plan` (139/29), `_build_automatic_scan` (128),
  `_start_automation_processes` (116/16), `prepare_acquisition` (84/12),
  `_run_step45_auto_recovery` (78/13), `request_safe_disable` (61/8).

## Mutable latest-state coupling

The coordinator independently stores `latest_joints`, `latest_capture`,
`latest_plan`, `latest_execution`, `latest_arm_status`, `latest_readiness`,
`latest_camera_health`, `latest_scan_history`, their receipt times/source
stamps, and a scan target center. Correctness depends on lock discipline,
generation clearing, receipt-time comparisons and correlations spread across
callbacks and blocking loops.

The executor stores `latest_tracking_health`, camera health, joints, obstacles,
workflow, target status, scan candidates, motion limits, arm status and tracked
target, plus many plan/path/current-view fields without one immutable state
object. The GUI has its own latest plan/workflow/tracking/readiness/status and
feedback clocks.

This is a refactor risk because a seemingly mechanical move can retain a stale
object but lose its matching timestamp, clear one field without its generation,
or evaluate fields from different ROS callback epochs. The replacement seam
must preserve atomic snapshots and monotonic-time semantics before removing
individual attributes.

## String-based classification

Behavior currently depends on reason text in all three layers. Examples:

- `failure_code_for_reason` maps substrings such as `camera`, `timed out`,
  `quality`, `occlusion`, `not found`, `plan`, `collision`, `hold` and
  `disable` to stable result codes.
- Plan retry helpers require prefixes/fragments such as `execution blocked:`,
  `target_status=low_confidence`, `planning failed: tesseract proposal
  rejected`, and `target moved ... after planning; refresh the plan`.
- Home completion recognizes executor state `ABORTED` plus reason fragment
  `configured home reached`.
- Capture retry distinguishes exact/prefix strings such as `missing target_3d`,
  `quality_rejected: scan quality is stale`, and
  `occlusion_rejected: settled target view is ...`.
- GUI recovery classifies plan/executor/workflow message strings separately.

Changing punctuation or wording can therefore change retries, home authority,
failure code, GUI controls or tracked-robot response. Typed enums/codes are a
future goal, but Phase 1 must first lock current strings with characterization
tests and introduce typed classification behind compatibility adapters.

## Boolean-heavy control and implicit state machines

The mission session combines phase with booleans including
`return_home_proved`, `storage_wrist_proved`, `startup_wrist_completed`,
`startup_home_completed`, `perception_scene_established`,
`current_hold_proved`, `disabled_proved`, `processes_stopped`, `arm_enabled`
and `motor_control_lost_reason`. Some combinations are invalid but are not
represented by a type.

The executor contains flags for returning home, startup home, closed-loop mode,
approval, capture, runtime refresh/recovery, hold/resume, abort/retrace,
mission authority and collision snapshots. `ACTIVE_STATES` is a parallel state
set. GUI `AutomationSession`, local phase enums, attempt generations, button
states and process flags form a third state machine.

`MissionSession.transition` prevents leaving a terminal state but does not
validate adjacency. State-machine correctness therefore lives in procedural
call order and boolean combinations. Do not replace this with a new state
machine in one patch; first record legal traces and terminal proofs.

Phase 5 names the executor's observed boolean combinations with `SafetyMode`
and evaluates them in shadow from immutable telemetry. This reduces ambiguity
for comparison but does not remove, replace or authorize any legacy check.
Promotion from shadow remains a separate high-risk change.

Phase 6 moves the admitted mission and shutdown sequence into a ROS-free
`MissionEngine`. The ROS node retains admission, queue/cache semantics, TF
conversion, action feedback/results and durable writes. The highest remaining
migration risk is divergence between the engine and the former procedural
bodies, so those bodies remain under non-authoritative legacy names and tests
compare the exact successful trace plus failures, cancellation, retries and
repeated missions. No runtime fallback selects the legacy implementation.

## Process supervision

As of Phase 4, the coordinator delegates its five exact named process groups
to `ProcessSupervisor`, including environment inheritance, generation-scoped
logs, health/exit reporting, reverse shutdown, SIGINT/SIGTERM and deliberately
no SIGKILL for a live command owner. `ManagedProcessSet` remains only as an
import-compatible alias. The GUI independently owns two groups and uses
SIGINT/SIGTERM/SIGKILL. Root camera scripts have their own process-group
manifest and cleanup rules. Reconstruction is launched separately by the
gateway.

Ownership, generation, signal order, timeout and kill policy differ for safety
reasons. Extracting a generic process manager without explicit policies could
kill the wrong generation, leave a camera process alive, restore manual command
ownership early, or falsely mark safe shutdown.

## Duplicated/coupled logic across the three primary files

| Concern | Mission | Executor | GUI | Refactor hazard |
|---|---|---|---|---|
| Command publisher ownership | Relies on mission-owned stack | Creates/destroys sole publisher | Enumerates publishers, destroys/restores manual publisher | Short overlap can create two arm command sources |
| Startup/readiness | Waits for driver/vision/TF/worker/readiness | Publishes/validates readiness dependencies | Starts manual stack and polls nodes/services/readiness | Different timeout/generation semantics |
| Target acquisition | Five-look coordinator loop | Acquisition state and heavy-refresh phases | Manual acquisition phases and retry threads | Three retry state machines can diverge |
| Plan correlation | Request IDs, plan kind, receipt time | Plan/source/hash/TTL/start snapshot | Session/attempt generations and plan matching | Late Foxy responses may enter a new attempt |
| Approval | Mission hash confirmation | Hash and live safety checks | Literal manual confirmation | Literal strings and authority modes differ intentionally |
| Cancellation/hold | Action/SIGINT/heartbeat -> safe shutdown | Cancel/hold services and abort state | Cancel/Home button and safe-disable hold | Duplicating home or disabling before proof |
| Home | Staged startup/terminal orchestration | Direct-home validation/execution and reason strings | Records profile and has manual safe-disable/home controls | Calibration/direction/collision exception blast radius |
| Motor safety | Guard and motor-loss no-home path | Runtime motor reasons and command suppression | Displays status and gates disable | Aggregate versus per-axis semantics |
| Retry classification | Reason substring helpers and budgets | Capture/recovery classifications | GUI auto-recovery helpers | Message wording is behavior |
| Mission state | `MissionPhase` plus session booleans | Executor string states and flags | `AutomationSession`, acquisition/Step4 phases and buttons | No single source of truth; not all are equivalent |
| Process cleanup | Five autonomous groups, no SIGKILL | ROS node cleanup and publisher destruction | Two GUI-owned groups, can SIGKILL | Incorrect abstraction can weaken safety |
| Speed limits | Exports selected speed/profile gate | Validates plan speed and publishes commands | Collects manual speed and exports environment | Double scaling or bypass of qualification |

## Existing safety-check duplication that must remain during extraction

The bridge, worker and executor all validate plan schema/counts, joint names,
limits, speed/timing and hashes. The executor and mission both check joint/motor
freshness and current-state convergence. The capture node independently checks
executor state, target/mask/depth, quality and occlusion even after the executor
settles. The mission verifies capture history/coverage again. Home stages are
checked by profile helpers, bridge, worker, executor and coordinator feedback.

This defense in depth is not accidental dead code. Phase 1 can centralize pure
calculations but must keep checks at trust boundaries unless tests prove the
boundary is unchanged.

## Ten highest-risk refactor areas

1. **Home, hold, disable and motor-loss shutdown.** It can move real hardware,
   contains the intentional home collision exception, and decides whether an
   operator must intervene.
2. **Executor plan ingestion (`tesseract_plan_cb`).** One 517-line callback
   binds hashes, model qualification, timing, start state and all path arrays;
   partial extraction can authorize a mismatched plan.
3. **Sole command-publisher ownership.** Mission, executor and GUI coordinate
   dynamically; lifecycle ordering errors create competing motion commands.
4. **Asynchronous `latest_*` snapshots and generation correlation.** Stale ROS
   messages, late service futures or old transient-local plans can be mistaken
   for current authority.
5. **String-based failure/retry/home classification.** A harmless wording
   cleanup can alter motion, retry budget, terminal failure code or safe-home
   interpretation.
6. **Tesseract bridge/worker/executor contract and timing.** It spans Python
   3.8/3.10 environments, disk spool schemas, hashes, QoS and exact MoveJ
   sampling; it must remain cross-version compatible.
7. **Capture acceptance and adaptive completion.** Acceptance is jointly owned
   by executor, capture, quality, occlusion, achieved history and measured
   surface coverage; duplicate or missing updates can falsely complete a scan.
8. **Process generation startup/cleanup.** Camera, GPU workers, hand-eye,
   Tesseract and ROS nodes have ordered readiness and different escalation
   policies; leaks or cross-generation kills affect hardware control.
9. **Perception timestamp/depth/mask association.** Mixing latest data with an
   archived heavy result causes target jitter, wrong obstacles and invalid
   reconstruction while appearing syntactically valid.
10. **GUI event loop and manual/autonomous state overlap.** The 549-line event
    reducer combines user-visible state, generation filtering, recovery and
    command ownership, and has tests but no pure typed reducer yet.

## Phase 1 sequencing recommendation (not implemented)

The safest first phase is characterization and pure extraction, not a mission
rewrite:

1. Add golden tests for ROS names/types/QoS, JSON payloads, failure reason/code
   mapping, state traces, process escalation and terminal result proofs.
2. Extract side-effect-free classifiers/snapshot validators while leaving each
   trust-boundary call site intact.
3. Introduce immutable snapshot/result objects behind compatibility adapters;
   do not change message schemas.
4. Separate GUI event reduction from widgets only after event-generation tests
   exist.
5. Defer subprocess unification, executor state-machine replacement, home logic
   changes and ROS interface cleanup to later, independently qualified phases.

Large rewrites, topic/message renames, threshold consolidation and removal of
apparently duplicate safety checks are explicitly out of Phase 1's first cut.
