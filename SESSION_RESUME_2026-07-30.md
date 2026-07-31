# End-to-end scan test resume point — 2026-07-30

## Safe paused state

- A later GUI-only run deliberately used the wrong rough hint
  `(0.25, -0.25, 0.0)`. It recovered a measured lock near
  `(0.305, -0.200, 0.029)`, generated a valid 13-view plan, and saved four
  GOOD/CLEAR synchronized views in
  `datasets/active_scan/scan_20260730_224512`.
- The remainder exposed an oversized low-confidence cardboard/cable scene and
  a worker response just beyond 180 seconds. The four records remain
  preserved. The live obstacle prompt is now pen plus hand/finger only. The
  worker has a 150-second internal planning budget, checks that budget before
  every OMPL attempt and throughout adaptive collision validation, and
  reserves five seconds to emit its correlated response before the bridge's
  180-second limit.
- GUI Safe Disable now sends the fresh current feedback pose, proves it
  settled for one second, and only then calls the disable service.
- Camera, GroundingDINO, SAM2, geometry, and managed scan/Tesseract processes
  are stopped. The arm is disabled. The PiPER driver and hand-eye TF publisher
  remain running. Do not start or move hardware until the operator explicitly
  authorizes the next physical test.
- The 2026-07-30 GUI run saved all 13 synchronized records in
  `datasets/active_scan/scan_20260730_221116`.
- The post-capture return encountered stale obstacle telemetry. GUI
  `Cancel / Hold` was selected and feedback was allowed to settle at the exact
  current pose before GUI `Disable`.
- GUI service status confirmed `disable -> True`.
- Managed Tesseract/scan stack is stopped through the GUI.
- The arm is disabled. Do not move it without a fresh GUI enable, fresh Step-2
  plan, and exact approval.

## Implemented and tested

- Session-scoped accepted-view memory, duplicate-pose filtering,
  remaining-count replanning, workflow session IDs, and reset after capture
  session finalization.
- A 21-candidate dome now covers seven azimuths and camera pitches -45, -55,
  and -65 degrees. The bridge chooses a diverse 13-view subset, then orders it
  as a smooth camera-space route instead of alternating orbit endpoints.
- Post-capture return failures hold the current pose and finish with a warning;
  they no longer erase 13 successful captures or start Step-4/5 recovery.
- Step-4 bounded state machine and correlated request handling.
- Stable motion-limit hash confirmation.
- PiPER SDK endpoint publication is one-shot per feedback-gated waypoint. Each
  target is the exact approved Tesseract endpoint; no intermediate positions
  are generated.
- Absolute MoveJ timeout increased from 20 to 90 seconds while the existing
  no-progress abort is twenty seconds at 5% speed. Waypoint arrival and the
  established loaded-arm settle gate both use 0.025 rad.
- Rough acquisition expanded to a bounded 45-degree yaw/pitch search with
  45-degree yaw / 30-degree pitch diagonal looks. A regression proves the
  `right_up` look is within 5 degrees of the actual cube direction when the
  entered hint is `(0.25, 0, 0)` and the cube is near `(0.25, -0.25, 0)`.
- GPU launch now uses the same UDP-only Fast DDS transport profile as the GUI.
- The final offline regression run passed 258 GUI, workflow, acquisition,
  capture, executor, driver, bridge, and Tesseract tests with one intentional
  skip. Twenty-one CPU perception tests and six SAM2 GPU-environment worker
  tests also passed. Both affected ROS packages rebuilt successfully; 23
  architecture/configuration YAML files, Python compilation, and
  `git diff --check` passed.

## Live findings

- The old one-shot SDK command caused no measurable motion. Reasserting the
  identical endpoint fixed it; RViz and measured joint feedback followed.
- The old 20-second limit aborted a still-progressing 5-percent move. The
  90-second limit plus five-second stall detector fixes that failure mode.
- The old 15-degree axis-only acquisition cone completed correlated
  GroundingDINO refreshes but could not see the deliberately offset cube.
- The expanded diagonal acquisition code is loaded in the workspace but has
  not yet received its physical GUI approval because the test was paused.

## Resume sequence

1. Keep the workspace clear and support the arm. Start the L515/GPU pipeline
   and `run_hand_eye_tf.sh`; verify both are healthy.
2. Use the GUI only: Step 1, then Step 2 with `(0.25, 0, 0)` at 5 percent.
3. Inspect and approve Step 3. Require a correlated GroundingDINO/SAM result
   and a measured lock near `(0.25, -0.25, 0)`.
4. Use Step 4 and require one correlated 13-view plan.
5. Approve Step 5 through the GUI. Require thirteen synchronized captures,
   session memory growth without duplicate measured poses, and viewpoints from
   all three elevation bands.
6. At the terminal state, press GUI **Disable**. The button must request the
   current-feedback hold from the sole command owner, retain the pre-request
   pose as its proof target, verify at most 0.025 rad target error and 0.005 rad
   sample motion continuously for one second, and only then report a successful
   disable. If that eight-second proof fails, motors remain enabled. Do not
   command another zero/home pose before disabling.
7. Inspect the dataset metadata and compute pose-distance/angle diversity.
8. Run the complete GUI/mobile/Tesseract regression suites after the physical
   pass.
