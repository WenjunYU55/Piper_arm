# Phase 7 executor responsibility map

## Scope

This map describes the active responsibilities in
`scan_viewpoint_executor_node.py` before Phase 7 extraction. It is a
behavioral map, not a proposed policy change.

## Current responsibility map

| Responsibility | Current owner/path | State and contract coupled to it |
|---|---|---|
| ROS construction and configuration | `ScanViewpointExecutorNode.__init__` | Declares every existing parameter, creates unchanged subscriptions, publishers, services, clients, QoS profiles, and the 200 Hz executor timer. |
| Telemetry ingestion | `scan_cb`, `joint_cb`, `arm_status_cb`, `motion_limits_cb`, `tracking_health_cb`, `tracked_target_cb`, `camera_timestamp_health_cb`, `target_status_cb`, `obstacle_cb`, `workflow_cb`, `heavy_refresh_status_cb` | Updates `TelemetryStore` plus temporary `latest_*` compatibility mirrors. |
| Tesseract proposal normalization and validation | `tesseract_plan_cb` | Validates plan kind/count, identity, schema-v5 timing, controller-limit hash, speed, six-joint points, recovery/home evidence, collision qualification, target/path data, and publishes `PROPOSAL_READY` or rejection. |
| Exact plan and mission authorization | `approve_cb`, `authorize_mission_cb`, `mission_authorization_valid` | Binds plan ID, trajectory SHA-256, confirmation phrase or mission policy identity/deadline, current telemetry, target drift, capture-service readiness, and fresh path validation. |
| Phase 5 safety shadow comparison | `record_*_safety_shadow`, `runtime_reasons`, `publish_hold` | Legacy executor checks remain authoritative; `SafetyEvaluator` only records comparison evidence. |
| Runtime state dispatch | `tick`, `execution_tick` | Dispatches proposal expiry, limit refresh, runtime refresh/recovery, movement, settling, acquisition, capture, return-home, and workflow-finish states. |
| Runtime telemetry hold/recovery | `begin_runtime_refresh`, `begin_runtime_recovery`, `waiting_for_runtime_refresh_tick` | Holds current joints, freezes/resumes the scheduled stream, preserves approval-bound scene authority, and applies the existing bounded recovery timeout. |
| Trajectory execution and monitoring | `moving_tick`, `feedback_gated_moving_tick`, `streaming_moving_tick`, `publish_next_waypoint` | Executes one approved six-joint segment, preserves the 20 Hz schedule and source vertices, monitors command lag, following error, endpoint convergence, waypoint timeout, and no-progress timeout. |
| Segment preparation and live revalidation | `prepare_current_view`, `validate_path`, `validate_attached_tool_external_path`, `validation_path` | Rechecks exact start, limits, dense collision/tool clearance, target visibility/standoff, recovery declarations, and trajectory schedule immediately before execution. |
| Settle detection | `settling_tick`, `joints_settled`, `home_joints_settled`, `capture_pose_settled` | Applies unchanged endpoint, motion-window, duration, and timeout thresholds before acquisition, capture, or home completion. |
| Acquisition and target reacquisition | `request_acquisition_refresh`, `publish_acquisition_refresh`, `waiting_for_fresh_frame_tick`, `waiting_for_grounding_dino_tick`, `waiting_for_tracking_lock_tick`, `waiting_for_obstacle_scene_tick`, `advance_acquisition_view` | Correlates request/image stamps, waits for GroundingDINO/SAM and a measured target lock, validates rough-hint distance, obtains the semantic obstacle scene, and advances at most the existing configured looks. |
| Capture coordination | `capturing_tick`, `capturing_rgbd_tick`, `wait_capture_tick`, `record_accepted_view`, `record_rejected_view`, `advance_view` | Publishes settled capture authority, sequences workflow and RGB-D services, preserves propagation/response timeouts and ten same-view readiness retries, records accepted/rejected achieved views, and advances or finishes. |
| Return-home execution | `returning_home`, `record_retrace_target`, `begin_return_home_settle`, `return_home_settling_tick`, `complete_return_home`, `try_start_abort_return` | Preserves dedicated direct-home evidence, approved retrace compatibility, configured endpoint proof, stable home hold, and terminal home result wording. |
| Recovery and terminal decisions | `runtime_gate_action`, capture classifications, `handle_return_home_failure`, `abort_or_finish_captures`, `abort_motion`, `_terminal_abort` | Chooses current hold, bounded telemetry wait, same-view capture retry, view rejection/replan, acquisition next look, dedicated-home handoff, or abort. Some compatibility inputs still enter through the Phase 2 legacy failure adapter. |
| ROS output mapping | `publish_joint_command`, `publish_hold`, `set_state`, `publish_plan`, `publish_status`, `publish_scan_history`, service callbacks | Converts internal state into the existing joint command, latched plan/history, status, and service response contracts. |
| Mutable execution session | fields initialized in `__init__`, reset by `clear_plan` | Plan identity/data, current segment schedule, command metrics, settle windows, capture futures/retries, acquisition correlation, retrace/home state, scan history, and mission authorization currently share one node object. |

## Coupling hotspots

1. `tesseract_plan_cb` both parses an external ROS proposal and performs
   application-level plan validation before mutating the active plan.
2. `approve_cb` combines exact authorization, telemetry safety, target drift,
   service readiness, path validation, retrace initialization, and runtime
   state transition.
3. `execution_tick` and its movement helpers combine scheduling decisions with
   ROS command publication and terminal state mutation.
4. Capture response classification, retry accounting, accepted-view history,
   and workflow service orchestration are interleaved.
5. Recovery choices are spread across runtime refresh, acquisition, capture,
   return-home, and abort helpers instead of being represented by one typed
   decision.
6. `clear_plan` must reset fields owned by every responsibility above, making
   repeated-mission state leakage a refactor risk.

## Phase 7 extraction boundary

The safe extraction boundary is four application components with explicit
inputs and typed outputs:

- `PlanAuthorizer`: exact mission/plan identity, expiry, target drift, and
  already-computed plan/service/safety evidence. It does not validate geometry
  or call ROS.
- `TrajectoryRunner`: the state of one already-authorized trajectory schedule
  and pure next-step monitoring decisions. The ROS node remains the sole joint
  command publisher.
- `CaptureCoordinator`: settle-to-service sequencing, response timeout, and
  the existing bounded same-view retry accounting. The ROS node retains ROS
  service futures and scan-history publication.
- `RecoveryPolicy`: typed selection among retry, reacquire, replan, and abort.
  It consumes `Failure`/`FailureTag` evidence and never searches human-readable
  detail.

The ROS node remains responsible for ROS entity lifecycle, message conversion,
telemetry callbacks, command/status publication, and the authoritative Phase 5
safety-shadow integration. No public interface, QoS, numerical threshold,
motion path, command cadence, capture ordering, or recovery outcome is changed
by this boundary.
