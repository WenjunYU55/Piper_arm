# Phase 8 GUI responsibility audit

Date: 2026-08-14

## Pre-change responsibility map

| Category | Previous owner in `piper_gui_native.py` | Phase 8 disposition |
|---|---|---|
| A. Presentation | Tk tabs, labels, dialogs, status text | Retained |
| B. GUI state | widget state, manual targets, preview state, mission display state | Retained; mission display state moved to `piper_gui/view_model.py` |
| C. ROS communications | joint/status subscriptions, manual command publisher, services, scan services, action client | Retained only for production action, reconstruction handshake, read-only diagnostics and explicit commissioning interfaces |
| D. Preview/debug | RViz preview child, preview topics, conservative mirror confirmation | Retained and labelled commissioning |
| E. Mission business logic | Step 1-5 acquisition/workflow/13-view state machines, plan correlation, retry/replan classification | Removed from the production GUI |
| F. Production process control | GUI-owned Tesseract worker and scan-stack process groups with signal escalation | Removed from the production GUI |
| G. Safety logic | publisher ownership, scan approval, cancel/home retry, automation shutdown | Removed where the production mission owns it |

## Removed alternate autonomous controller

The GUI no longer creates clients for `PrepareAcquisition`,
`RequestTesseractPlan`, workflow start/diagnostic, executor approval, or executor
cancel. It no longer subscribes to scan plans/status for decisions, starts the
Tesseract worker or supervised scan stack, advances acquisition/workflow
states, approves a trajectory, classifies autonomous retries, requests an
autonomous return home, disables after autonomous shutdown, or terminates
production mission children.

`piper_gui_automation.py` is no longer imported by production GUI code. It is
retained temporarily as an archived Phase 1 characterization fixture; it has no
runtime owner or side effects.

## Current architecture

- `piper_gui_native.py` contains Tk presentation plus explicit commissioning
  controls and the ROS node adapter for joint/status/diagnostic topics,
  `/enable_srv`, `/piper/run_target_scan`, and
  `/piper/report_tracked_robot_homed`.
- `piper_gui/view_model.py` is Tk/ROS-free immutable presentation state. It
  validates operator input and reflects action lifecycle, feedback and result;
  it does not select production phases or recovery.
- `piper_gui/ros_client.py` maps the action goal, feedback, cancellation and
  result callbacks into typed GUI events. It does not interpret failure text or
  choose a retry.
- `piper_gui/app.py` owns only GUI/ROS lifecycle bootstrap and is testable with
  fakes.

The automatic tab submits exactly the existing `RunTargetScan` interface. The
mission node/MissionEngine remains the only owner of startup, readiness,
enable, acquisition, occlusion policy, viewpoint planning, capture, retry,
replan, home, hold, disable and mission-child cleanup.

## Retained commissioning behavior

Direct joint target publishing, manual enable, settled-hold-before-manual-
disable, home-profile recording, the preview-only RViz child and confirmed
preview mirroring remain available. They are deliberately labelled
commissioning and are locked while the action is active. These controls serve
manual hardware commissioning; the mission action does not own their standalone
operator workflow.

The Diagnostics tab retains read-only displays for heavy refresh, tracking,
workflow, capture and Tesseract readiness with the existing topic names and
depth-10 QoS. Those values never authorize GUI motion or state transitions.

## Compatibility

No ROS topic, service, action, message field, frame, QoS depth, deadline, speed,
motion limit, calibration value, Tesseract behavior, perception algorithm or
robot command format changed. The automatic goal still uses `SCAN_3D`,
`base_link`, confidence `1.0`, deadline `1200.0` seconds and diagonal rough
position covariance `0.01`.

## Phase 8 tests

`test_piper_gui_phase8.py` protects input validation, immutable view state,
feedback/result display, reconstruction eligibility, action callback mapping,
cancellation, rejected/unavailable action servers, lifecycle cleanup, and the
absence of production scan-service/process control from the native GUI.

The old GUI source assertions were inverted where they explicitly required the
now-prohibited Step 1-5 controller. Pure historical helper tests remain as
characterization evidence.
