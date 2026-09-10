# cuRobo startup boundary repair and stationary-target diagnosis — 2026-09-09

## Startup regression and repair

The failed rough-acquisition mission `gui-sim-4c5ff28eee9e4057b9d640867a603cb8` recorded Joint 2 at -0.000139552 rad (about -0.008 degrees) around its request. The local powered-limit intersection introduced after GitHub checkpoint b5f7154 routes that negative feedback into position-limit re-entry. Re-entry checks collision before the normal folded-self recovery, producing PLANNING_COLLISION_REJECTED at home. A representative home state with this recorded Joint 2 value produces no correction in checkpoint 5773244 or b5f7154, but a correction in the pre-repair local code. The exact failed request was consumed and was unavailable; the measured Joint 2 value was recovered from retained TF.

The worker now qualifies only initial ROUGH_ACQUISITION/bootstrap_static Joint 2 feedback in [-0.0005, 0) rad for a planning start of zero. The original request and feedback remain unchanged. The measured geometry and two intervening/end samples are checked against the current world; only the existing acquisition folded-self exception is allowed, and a self-colliding measured transition requires the normal declared folded-start recovery from the zero boundary. Native solver bounds stay powered-command bounds. This is not global rounding, a geometry offset, or a new collision exemption. Larger offsets, subsequent scan moves, other joints and dedicated home remain on existing paths. Diagnostics record the measured/planning values and bound.

Seven actual cuRobo/CUDA checks passed: recorded noise and the exact tolerance boundary reach a valid folded recovery; blocked geometry rejects and restores constraints; powered-negative Joint 2 targets remain invalid; Joint 5 inset and changed-world checks remain valid; a complete native acquisition plan from the reproduced home state contains only nonnegative Joint 2 targets. The full ordinary cuRobo/mobile/Tesseract/campaign/GUI run passed 1575 tests with 25 skips. No arm command or physical qualification was performed. Newly started workers load the repair.

## Stationary cube reported as moving

The user confirms the cube did not move during `gui-sim-86a175f6f2f944d3be04cbcfcbb9fe26`. Recorded streams completed at 192.2 and 194.6 Hz, with normal settling. The first attempted view was rejected as target LOST. A subsequent view saved frame 000; the workflow then changed to INITIALIZING for first-capture semantic occlusion assessment. Shortly afterward it aborted on tracked displacement 0.035 m and landmark residual 0.039 m.

Offline replay used exact mask/aligned-depth timestamp pairs from the saved archive, the existing TargetLandmarkNode.measurement algorithm, recorded intrinsics and image-time TF. No ROS nodes or commands were created. Of the retained matching samples, 56 produced valid measurements and two had empty target masks. This is a replay of retained pairs, not a reconstruction of every original approximate-sync callback or Kalman state.

| Retained observation group | Samples | Median base_link XYZ (m) |
|---|---:|---|
| Before first scan movement, timestamps 1788992309.0–1788992310.3 | 8 | 0.843866, -0.009070, 0.319871 |
| Settled first saved capture, timestamps 1788992328.0–1788992330.5 | 6 | 0.833621, -0.016393, 0.355020 |

The apparent displacement is **37.34 mm**, with delta **(-10.24, -7.32, +35.15) mm**. Thus the saved camera/depth/TF pipeline reproduces the position change despite the stationary object; this is not merely a stale GUI status or invented error code.

The supposed independent corroboration is not independent of the sensing error: the tracker consumes target_3d derived from mask/depth and the landmark estimates the median of masked depth points from the same RGB-D stream. The tracker resets its filter when receiving a valid measurement after LOST. The landmark stores/updates a view-dependent visible-surface point rather than a proven invariant object center. Both can therefore agree on a pose-dependent measurement shift without physical object motion.

A second, definite control-flow issue makes this terminal at capture one: target_motion_is_terminal exempts SCAN_READY and WAIT_CAPTURE when allow_target_motion_during_scan is true, but the deferred first-capture occlusion step returns to INITIALIZING. The existing strict motion check therefore reactivates while this first capture is pending. Its 20 mm physical threshold plus 15 mm surface uncertainty is exceeded by this incident.

The evidence establishes a false physical-motion interpretation and the state transition that makes it abort. It does **not** isolate the exact shares of visible-surface/depth error, hand-eye/robot calibration error and synchronization effects. Full-resolution live target_3d, filtered target and landmark-status streams are not retained in the archive. No calibration or guard threshold was changed, and this target-jump issue is diagnosed but not repaired in this patch. A follow-up should preserve the active scan/first-capture lifecycle explicitly and require a geometric object-motion check that is robust to viewpoint changes, rather than increasing a blanket tolerance.
