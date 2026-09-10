# Invalid J1/J2 feedback during storage home — 10 September 2026

Incident: gui-sim-4348f1d4a22f4f14aa6405a10a562f9a. Four captures, ray frontier exhaustion, rough home completed, STORAGE_WRIST started 1789053612.799. At 1789053615.138, return-home bounds check rejected J1=299.364613 rad and J2=10537.162912 rad. J5 collision_status/driver_error_status appeared at 1789053615.772; driver watchdog disabled all axes. Subsequent five-second passive CAN check found stationary plausible joints and all six enable bits clear; J5 retained faults. Mission remains NEEDS_OPERATOR with safe_shutdown=false; later disabled feedback does not retroactively prove the interrupted home transaction.

## Exact payload fingerprint

Inverse driver scaling (0.017444/1000 rad per millidegree) maps the printed values uniquely to raw integers 17161466 and 604056576. Their concatenated bytes are `01 05 dc fa 24 01 2c 00`. Interpreted with the installed manufacturer's 0x473 motor-limit response layout, these encode motor 1, maximum +150.0 degrees, minimum -150.0 degrees, maximum speed raw 300. Interpreted as joint-pair integers they reproduce 299.364612904 and 10537.162911744 rad exactly to the log precision.

This is strong evidence of a motor-limit reply payload entering the joint-feedback path, not physical joint motion or random numeric jitter. The driver periodically sends read-only 0x472 limit queries. Its dedicated SocketCAN reader filters joint IDs and checks IDs again before decoding; it does not call the SDK's joint parser. Therefore the numerical fingerprint alone does not establish whether controller/adapter traffic carried the wrong ID or another software boundary corrupted the association. Original raw incident frames and outgoing command bytes were not retained. No claim that a normal correctly labelled 0x473 frame bypassed the current ID check.

## Confirmed unsafe software behavior

The raw decoder accepts any eight-byte signed integer pair. Coherent feedback checks age, inter-pair skew, sequence and finiteness, but not physical plausibility. The executor rejects out-of-range values at its runtime safety gate. However, `_terminal_abort` then calls `publish_hold`; `energized_hold_target` checks only six finite values and clips them to powered command bounds. Offline execution with the bad pair yields J1=2.6179 and J2=3.14 rad (approximately +150 and +180 degrees). This converts invalid measurements into potentially large position commands. It is a confirmed defect and a plausible causal explanation for the reported disturbance; without the transmitted command or a captured hold snapshot it is not proof that this exact clipped pair reached the controller during the incident.

## History and follow-up

Reader ID-check/dispatch lines blame to d738a78 (1 August); hold helper lines to 4a2acdc (27 August). Only one impossible joint-feedback bounds event was found in retained aggregate executor logs. Recent campaign, recording and perception edits did not introduce these code paths; missing older raw traffic prevents proving no previous unlogged event.

Required repair: reject physically implausible feedback before it can become motion authority; never turn corrupt/out-of-range feedback into a clipped hold target; define explicit invalid-feedback stop behavior independently of a fresh measured hold. Preserve bounded legitimate encoder-boundary handling and J6 multi-turn semantics. Add bounded raw RX/command diagnostic capture for fault evidence. Validate command-free bad-frame/abort tests before any re-enable or physical qualification. Do not simply disable home safety checks or round the numbers.

This investigation made no production code changes and sent no CAN commands. Supporting bounded passive capture and log excerpts are in /tmp/piper_home_feedback_20260910 (temporary storage).

Follow-up passive 15-second capture: 3,000 frames per joint-pair ID, 36 existing host limit queries, 18 limit responses. Correctly labelled 0x473 traffic contained the exact payload `0105dcfa24012c00`. No impossible joint pairs recurred and no joint-control frames were observed in this capture. This confirms that the fingerprint matches an actual controller reply, but does not identify the incident misassociation point.

## Follow-up: recorded movement after the abort

Read-only extraction from `datasets/mission_recordings/scan_recording_c8a1751a9d4d405a8fa83be5b3dcf9a4.inprogress/bag/bag_0.db3` shows the following relative-link TF rotations. Times below are recorded transform header times; they are feedback-derived, not independent optical metrology. J2 is expressed as change from the pre-abort rotation to avoid confusing the fixed URDF rotation with joint zero.

| Seconds after logged abort | J1 angle | J2 change from pre-abort |
| --- | --- | --- |
| -0.032 | 0.00 deg | 0.00 deg |
| +0.018 | 0.00 deg | 0.00 deg |
| +0.073 | 0.57 deg | +0.61 deg |
| +0.183 | 4.21 deg | +4.23 deg |
| +0.393 | 20.48 deg | +20.43 deg |
| +0.608 | 38.33 deg | +38.27 deg |

The executor logged SETTLING_HOME at 1789053615.121794, approximately 16 ms before the invalid-feedback abort. Both axes then accelerated positively, with the motor-fault watchdog firing approximately 634 ms after the abort. This materially strengthens the explanation that the invalid-feedback hold path caused the reported unexpected motion. It still does not recover the exact transmitted command or prove which snapshot publish_hold consumed. The retained TF is a second recorded representation of feedback, not independent proof of encoder correctness.

## Follow-up: packet association and upstream comparison

The installed python-can 4.5.0 SocketCAN reader obtains CAN ID and payload from one immutable recvmsg byte buffer and creates a new Message. The dedicated driver reader checks the ID before decoding and protects pair storage with a lock. SDK command encoders and SendCanMessage construct local messages. These inspected paths do not reveal a shared ID/payload buffer race; this is evidence against an ordinary Python decoding mix-up, not an exhaustive exclusion of host or adapter faults. The adapter uses gs_usb (USB ID 1d50:606f). A normal 0x473 reply cannot pass the explicit joint-ID check.

AgileX acknowledges an MCU interrupt-processing defect associated with MIT-mode twitching in [issue 67](https://github.com/agilexrobotics/piper_sdk/issues/67#issuecomment-3163133732), and identifies firmware 1.7-6 and later as fixing that reported issue [here](https://github.com/agilexrobotics/piper_sdk/issues/67#issuecomment-3709104813). The controller previously reported S-V1.5-8. However, this application's home commands use MoveJ (move_mode 0x01), not the issue's MIT control. The upstream report does not demonstrate that a 0x473 response is emitted as 0x2A5, and therefore cannot establish the origin of this incident or justify claiming a firmware update will cure it.

Current conclusion: strong evidence identifies the invalid reading as a motor-limit payload and the post-abort movement as consistent with our confirmed unsafe hold conversion. The exact controller/USB/host boundary that misassociated the payload remains unresolved because the original raw frame was not captured. No production changes, runtime restarts, new limit queries, firmware changes or motion commands were performed in this follow-up.

## Implemented repair and validation

The driver now validates classic standard joint-frame flags, length and physical plausibility before pair storage or J6 wrapping. J1-J5 use model bounds with the existing 0.3-rad direct-home feedback allowance; J6 retains the qualified raw multi-turn interval of +/-545 degrees with the same allowance. This is a corruption guard, not expanded planning or command authority. It does not attempt to identify every possible corrupt payload that happens to encode plausible joint angles.

Any invalid joint frame latches INVALID_JOINT_FEEDBACK until driver restart. Joint feedback publication and position commands are blocked; a subsequent good frame does not clear the latch. The existing feedback watchdog requests all-axis disable independently of measured position or motor fault flags, including when motor enable status is unavailable. Requests repeat at 0.1-second intervals until all six motors report disabled; the fault reason remains visible afterward. Removing motor power is not a guarantee against gravity-driven motion or a controlled braking trajectory. Final Cartesian/joint command emission is serialized with the latch; previously transmitted commands cannot be recalled. Enable handshakes also reject the latch during retries and retain their existing disable rollback.

The shared energized_hold_target helper now rejects out-of-range inputs before clipping. Existing bounded droop correction remains; the exact incident values cannot generate any hold command. Startup J6 reference and controller-turn mapping remain unchanged. Planner source, geometry, timing and firmware were not modified for this repair.

The new can_fault_trace owner retains up to 12000 pre-fault and 12000 post-fault SocketCAN records, with a one-second post-fault interval, then writes the first fault of each driver lifetime in a disk-only background thread. It retains the newest five completed files in `~/.ros/piper_can_faults/`. Captured fields include actual IDs/payloads, frame flags and DLC, kernel and monotonic timestamps, and local-TX versus RX. Filters cover all three MoveJ command frames (0x155-0x157), mode/Cartesian/gripper commands, enable/disable, status and limit replies. This captures observations, not proof of controller receipt. Storage failures cannot suppress stopping. Graceful shutdown flushes pending evidence; abrupt process death can lose it.

Validation:
- 1371 driver, mobile, Tesseract and driver-tool behavioral tests passed; two skipped. Legacy lint tests were excluded from this behavioral run.
- A subsequent 178-test focused run passed after the final trace-ID completeness regression, covering actual receive-loop corrupt/malformed/flagged frames, first-fault evidence, disk failure, retention bounds, unknown-status stopping, command/enable rejection, hold and terminal-abort behavior.
- Driver, description and mobile packages built successfully in the normal piper_ros_foxy workspace. Changed Python files compile; diff whitespace check and targeted lint passed with existing style exclusions.
- Five-second passive live observation decoded 1000 frames from each joint-pair ID (3000 total), with no new-guard rejection. All six low-speed motor enable bits were false and no 0x155-0x157 joint commands were observed. This check opened only a passive SocketCAN reader and did not start/restart a driver, enable motors or transmit commands.

Deployment remains pending: both the live driver and scan executor must be restarted to load the repair before further powered testing. No running process was restarted by this repair task. Supervised low-speed startup/home/cancel and physical stopping qualification remain outstanding. The original packet-misassociation boundary remains unproven; this repair prevents its identified unsafe consequence and adds evidence for future diagnosis.
