# Final behavioural-equivalence audit

> Audit boundary notice (2026-08-17): this report describes the 2026-08-15
> refactor revision. A later user-authorized shutdown/runtime-gate correction
> removed the duplicate shadow evaluator and frozen mission bodies, made fresh
> direct-home qualification independent of the original failure and Tesseract
> motion-limit telemetry, and removed redundant autonomous startup and terminal
> hold-service calls after configured home is proved. The earlier READY verdict does not
> physically qualify those later changes; use the updated physical checklist
> and treat supervised requalification as pending.

> Historical note: this report describes the behavior-preserving Phase 0--10
> refactor before the selected-feature reintegration on 2026-08-15. The branch
> `reintegrate/selected-archived-features` intentionally adds ExecuteHomeStage,
> schema-v4 PRE_HOME, positive-only startup J6 handling, motion-adapter and
> reconstruction behavior. Those differences are documented in
> `reintegration_report.md` and require supervised physical requalification;
> the earlier "No difference" statements are not claims about that branch.

## Verdict

**READY FOR SUPERVISED PHYSICAL REQUALIFICATION**

The final refactored working tree is behaviourally equivalent to the
known-working baseline within the software, deterministic replay, static
contract, build, and command-free planning evidence available in this
repository. No `UNEXPLAINED` or `SAFETY-RELEVANT` refactor difference was
found.

This verdict authorizes only the staged, supervised checklist in
`physical_requalification_checklist.md`. It is not a claim that the robot is
already physically requalified, and it does not authorize an autonomous arm
motion. No arm, camera, driver, GPU worker, GUI, coordinator, or hardware-facing
ROS process was started by this audit.

## Comparison basis

- Baseline commit: `4945b480d8494fa840c0d2bc993c72834934a37f`
- Baseline subject: `Fix Tesseract MoveJ stream timing and validation`
- Baseline identity: the Phase 0 record identifies this commit as the
  known-working refactor baseline and as the then-current `origin/main`.
- Refactored subject: the uncommitted Phase 1--10 working-tree delta layered on
  that exact commit.
- Audit date: 2026-08-15, Europe/London.
- Runtime compatibility: ROS 2 Foxy and Python 3.8.10.

The audit used four complementary forms of evidence:

1. exact Git comparison against the baseline commit;
2. exact public-interface and configuration characterization tests;
3. identical-input deterministic baseline/refactor mission replays; and
4. complete software tests, ROS build/test, and rootless Tesseract
   qualification.

Exact equality was used for interface text, identifiers, phase sequences,
decisions, action outcomes, hashes, and integer limits. Existing
`pytest.approx`/NumPy tolerance checks were used for floating-point geometry,
timing, telemetry age, and trajectory calculations. No floating-point result
was rejected solely for representational round-off.

## Direct identical-input comparison

The Phase 1 deterministic mission harness was run in isolated Python processes
against both the Git baseline package and the final refactored package. The
same normalized goals and subsystem responses were supplied to each version.
The comparison included action outcome, failure code, retryability,
`safe_shutdown`, capture count, mesh job ID presence, full mission phase trace,
ordered subsystem/arm events, process cleanup events, and action terminal
transition.

| Replay | Baseline vs refactor | Phases | Events | Terminal result |
|---|---|---:|---:|---|
| Successful eight-capture mission | exact match | 29 | 116 | `SUCCEEDED` |
| Target absent for five acquisition looks | machine decisions/trace exact; one benign reason-wording difference | 21 | 70 | `FAILED / TARGET_NOT_FOUND` |
| Nine rejected views after eight replacement replans | exact match | 31 | 124 | `FAILED / MISSION_FAILED` |
| No reachable multiview plan | exact match | 14 | 48 | `FAILED / NO_REACHABLE_PLAN` |

The replay was then expanded to 19 scenarios: the four above plus camera
unavailable/stale, joint feedback unavailable/stale, stale arm status, target
reacquisition failure, planner failure, trajectory failure, capture failure,
child crash, deadline, occlusion plan ready, insufficient coverage, queued
cancel, and cancellation during planning. Eighteen scenarios matched in every
serialized result and trace field. The target-absence scenario differed only
as follows:

- baseline reason: `target not found after five distinct closed-loop looks`;
- refactored reason: `target not found after 5 distinct closed-loop looks`.

The cause is explicit: the extracted `MissionEngine` formats the unchanged
`maximum_looks` value instead of retaining the baseline's hard-coded English
word. The public failure code remains `TARGET_NOT_FOUND`, `retryable` remains
true, every decision and shutdown event is identical, and no code branches on
this detail. The durable failure `result_sha256` also changes because it
correctly covers the changed reason text. This is classified `BENIGN`: it is a
public diagnostic-text/hash difference, but not a machine, motion, safety, or
state-transition difference. This audit does not alter it.

The successful trace retained:

`STARTING -> PREFLIGHT -> ENABLE_AND_HOLD -> RETURNING_HOME -> RETURNING_HOME
-> ROUGH_ACQUISITION -> TARGET_LOCK -> OCCLUSION_PROBE ->
(VIEW_PLANNING -> CAPTURING) x 8 -> RETURNING_HOME -> RETURNING_HOME ->
HOLDING -> DISABLING -> STOPPING`.

The same run also retained the ordered external effects: owned-process startup,
readiness, authority grant, enable, settled hold, startup wrist, rough home,
acquisition, measured lock, occlusion probe, plan/execute/capture loop, rough
home, storage wrist, motor disable, authority revocation, reverse process
cleanup, result spool, and mesh-job spool.

The repository does not contain an immutable full-mission ROS bag with action,
joint, target, obstacle, planner, capture, and shutdown channels. The bundled
RealSense bag is a vendor camera test resource, not a PiPER mission recording.
Therefore a dual live ROS-graph replay was not claimed. The deterministic
harness and existing recorded scan/reconstruction tests are the strongest
repeatable no-hardware inputs available.

## Required equivalence matrix

| # | Contract | Evidence and result | Difference classification |
|---:|---|---|---|
| 1 | Public ROS actions | All 31 repository `.action`, `.msg`, and `.srv` definitions are byte-identical. `/piper/run_target_scan` is unchanged. | No difference |
| 2 | Action goal schemas | `RunTargetScan` constants and goal fields are byte-identical. | No difference |
| 3 | Action feedback | Feedback fields, phase names, reasons, counters, health JSON, and shutdown phase are unchanged; phase replay matched exactly. | No difference |
| 4 | Action results | Result fields/constants and machine outcomes are unchanged. One target-absence reason changes `five` to `5`, with the corresponding self-consistent result hash change. | `BENIGN` |
| 5 | Topics | Mission/executor subscription and publication names and message types match the baseline; interface characterization is green. | No difference |
| 6 | Services | Mission/executor service names and service types match the baseline; all service schemas are byte-identical. | No difference |
| 7 | QoS | Mission and executor QoS construction is unchanged, including depth-1 reliable/transient-local plan/history channels and sensor-data QoS where used. | No difference |
| 8 | Parameter names | Existing names are preserved. The 16 mission and 82 executor parameters moved to typed loaders are frozen by exact-name tests; all other parameter-owning sources are unchanged. | No difference |
| 9 | Parameter defaults | Exact-value old-vs-new default tests pass for all 98 migrated parameters; launch/config defaults outside those nodes are byte-identical. | No difference |
| 10 | TF frames | Launch, URDF/Xacro, calibration, hand-eye, and TF semantic sources in the protected comparison set are unchanged. | No difference |
| 11 | Motion limits | Driver bounds, recorded limits, URDF limits, live-limit hashes, Tesseract model, and limit checks are unchanged. | No difference |
| 12 | Speed limits | Free/contact/manual percentages, `1..100%` validation, 20 Hz command rate, velocity/acceleration caps, and tracking scale semantics are unchanged. | No difference |
| 13 | Timeouts | Exact mission/executor timeout characterization passes; no default changed. | No difference |
| 14 | Freshness thresholds | Snapshot migration retains source/receipt timestamps and strict age semantics; legacy/new decision-equivalence and fake-clock tests pass. | No difference |
| 15 | Target drift thresholds | The `0.015 m` approval drift limit and bounded drift replan behavior are unchanged and characterized. | No difference |
| 16 | Capture acceptance thresholds | `GOOD >= 0.65`, `CLEAR`, depth/mask/settle, persistence, and achieved-FK requirements are unchanged. | No difference |
| 17 | Retry limits | Five acquisition looks, six occlusion actions, eight quality replacements, eight target-drift replans, ten readiness retries, and 8/24 capture limits are unchanged. | No difference |
| 18 | Startup sequence | Driver, vision, hand-eye, Tesseract worker, scan stack; readiness; authority; enable/hold; startup wrist; rough home are unchanged. | No difference |
| 19 | Shutdown sequence | Stop/hold as applicable, fresh rough home, storage wrist, final settled hold, disable, authority revoke, and owned-process cleanup remain in the baseline order. | No difference |
| 20 | Process ownership | Exact mission-owned handles, generation isolation, commands, environment, reverse cleanup, SIGINT 5 s and SIGTERM 3 s policy are preserved. GUI production-process ownership was intentionally removed. | `EXPECTED` |
| 21 | Cancellation behavior | Queued and active-stage cancellation paths match characterization. The retained late-terminal-cancel behavior is unchanged. | No difference |
| 22 | Failure classification | Machine decisions now use typed failures; the compatibility boundary preserves public strings/codes. Wording-independence and legacy-code tests pass. | `EXPECTED` |
| 23 | Return-home behavior | Startup J6 then rough home and terminal rough home then storage J6 remain exact; loss of trustworthy motor authority still forbids automatic home. | No difference |
| 24 | Hold behavior | Hold remains a current-feedback target plus fresh settled proof, not silence. Active and shutdown paths are characterized. | No difference |
| 25 | Disable behavior | Disable remains conditional on the required home/storage/hold proofs; motor-control loss still forbids service disable. | No difference |
| 26 | Mission-state transitions | Baseline/refactor replay is exact and the Phase 6 legacy-vs-engine trace test passes. | No difference |

## Planning, viewpoint, capture, and safety decisions

- `scan_viewpoint_planner_node.py`, `target_acquisition.py`, `scan_motion.py`,
  capture/perception implementations, Tesseract worker, launch files,
  robot models, collision meshes, and calibration inputs are byte-identical to
  the baseline. Consequently the proposal and trajectory algorithms receive
  and produce the same values for identical state and deterministic seed.
- Rootless Tesseract core and compact qualifications pass with backend
  `0.35.0.6`, collision qualification true, and `real_arm_motion=false`.
  The compact recorded input still selects viewpoint 0, produces 91 trajectory
  points and 5,261 validation samples. The 5% regression emits maximum J6
  velocity `0.15000000000000013 rad/s`. The August 11 holder-floor incident is
  still rejected at `0.001245 m < 0.005 m`.
- Plan authorization, stale/wrong mission plans, target drift, stale target,
  planner/path failure, trajectory completion/failure/cancel, capture
  success/retry/failure, and typed recovery decisions all pass Phase 7 tests.
- The new safety evaluator remains shadow-only. Legacy gates alone remain
  authoritative for commands. The one documented synthetic disagreement is an
  unqualified collision model supplied to the broad runtime helper: the new
  shadow evaluator rejects it while the legacy runtime helper permits because
  the same collision qualification is enforced earlier at approval/path
  validation. This is an `EXPECTED` diagnostic difference and cannot alter a
  command.

## Exhaustive difference classification

All changed production paths are accounted for by the following groups.

| Difference group | Files/behavior | Classification | Rationale |
|---|---|---|---|
| Typed failure, telemetry, process, safety, mission, executor, and configuration components | New application modules and adapter wiring in the two primary ROS nodes | `EXPECTED` | The requested Phase 2--9 extractions; legacy/public decisions are protected by exact and characterization tests. |
| Mission/executor internal delegation | `target_scan_mission_node.py`, `scan_viewpoint_executor_node.py` | `EXPECTED` | ROS names/QoS and authoritative behavior remain at the boundary; deterministic sequence and decision tests match. |
| GUI client architecture | `piper_gui/`, `piper_gui_native.py`, `piper_gui_automation.py`, `start_gui.sh` | `EXPECTED` | The GUI now uses the production action instead of owning a second autonomous workflow. Commissioning controls remain separate. |
| Test registration and test seams | Mobile `CMakeLists.txt`, characterization files, test-only changes | `EXPECTED` | Adds regression evidence without runtime behavior. |
| Documentation | `docs/refactor/`, `docs/ai/` | `EXPECTED` | Architecture and requalification records only. |
| AST-identical formatting/import-registration notes | `target_tracker_node.py`, Tesseract bridge/contract | `BENIGN` | Parsed executable AST is unchanged. |
| Proven-unused cleanup | three geometry helpers, `utils/transforms.py`, one inert final-hold calculation, one unused qualification import | `BENIGN` | Repository reference search plus the full tests prove no production caller. |
| Style-result count | Baseline 95 Flake8/23 PEP257 findings; final 87/21 findings | `BENIGN` | Only lint debt decreased; all functional targets pass. Aggregate CTest duplicates the two failing style executables. |
| Target-absence diagnostic text | action result reason spells the configured five-look limit as `5` rather than `five`; derived result hash follows the payload | `BENIGN` | Typed code, retryability, decisions, phases, arm/process events, and safe shutdown are identical; no behavior parses the detail. |

No changed or added production path remains outside these groups.

### Difference totals

- `EXPECTED`: internal ownership/dependency changes required by Phases 2--9,
  test registration, and documentation.
- `BENIGN`: behavior-neutral cleanup/formatting and the documented
  human-readable `five` to `5` failure-reason/hash variation.
- `UNEXPLAINED`: **none**.
- `SAFETY-RELEVANT`: **none**.

## Automated validation results

All commands used `PIPER_MISSION_ENABLE_REAL_MOTION=0`; ROS tests used
`ROS_DOMAIN_ID=143` where applicable.

| Validation | Result |
|---|---|
| Direct mobile package pytest | **580 passed** |
| Root GUI/reconstruction/calibration pytest | **69 passed, 1 existing hardware-dependent skip** |
| Heavy perception worker | **5 passed** |
| SAM2 live worker | **6 passed** |
| GroundingDINO target selection | **19 passed** |
| Five-package `colcon build --symlink-install` | **PASS**, 5 packages |
| Registered functional ROS tests | **PASS** |
| Rootless Tesseract core qualification | **PASS** |
| Rootless Tesseract compact qualification | **PASS** |

`colcon test-result --verbose` reports 944 assertions, 0 errors, 110 failures,
and 1 skip. The 110 count consists solely of the two already-known style
executables and their aggregate CTest records: 87 Flake8 findings, 21 PEP257
findings, and 2 aggregate failures. There is no functional test failure. This
is improved from, and does not introduce a regression against, the yellow
Phase 0 style baseline.

## Retained baseline behavior and evidence limits

The following are not refactor differences and therefore do not block
behavioural equivalence, but must remain visible during requalification:

- cancellation arriving after productive work while terminal shutdown is in
  progress does not interrupt home/hold/disable/cleanup and retains the
  successful wire result;
- exhaustion of the visual replacement budget retains the generic
  `MISSION_FAILED` public failure code;
- safety shadow mode is not authorized to command or veto the robot;
- legacy pipeline/shutdown bodies remain in place until a separately
  authorized supervised trace has been collected;
- there is no repository-owned complete mission bag for dual ROS-graph replay;
- camera quality, DDS timing under load, CAN behavior, real controller
  smoothness, actual J6 direction, physical clearances, and end-to-end scan
  quality require the staged physical procedure.

## Blockers

There are **no software-equivalence blockers** and no unexplained or
safety-relevant refactor differences.

Physical production use remains blocked until every mandatory stage in
`physical_requalification_checklist.md` has passed and its evidence has been
recorded. A failure at any stage stops progression and changes the operational
verdict to `NOT READY` pending diagnosis.
