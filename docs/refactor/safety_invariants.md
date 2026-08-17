# Safety invariants and numerical baseline

These values are recorded, not endorsed or changed. A Phase 1 extraction must
preserve the effective value, units, comparison direction, clock source,
evaluation point and failure behavior. Node defaults can differ from launch
YAML overrides; the deployed `supervised_viewpoint_execution`/GPU launch path
and the files named below are authoritative for that composition.

## Non-negotiable behavioral invariants

1. There is exactly one live arm command publisher. Autonomous execution
   destroys/blocks the GUI manual publisher before creating the executor
   publisher; manual ownership returns only after all scan publishers stop.
2. No motion occurs unless real motion, a physically qualified speed profile,
   mission authority, a matching plan/hash, live all-six motor feedback,
   current limits, settled joints/tracking/camera health, valid obstacle scene,
   collision qualification and exact-path checks all agree.
3. Aggregate arm status is insufficient. Every powered guard requires valid
   low-level motor feedback and all six driver-enabled flags with no motor
   fault/watchdog reason.
4. Cancellation, deadline, process exit, unexpected exception and normal
   completion all enter the same staged shutdown transaction.
5. Autonomous startup and terminal shutdown do not call the redundant hold
   service. PiPER position control retains the latest commanded target; each
   home stage is proved from feedback and the final storage target is retained
   until immediate feedback-confirmed disable. Executor-local recovery holds
   remain unchanged.
6. Disable is attempted only after required rough-home and storage proofs.
   Failure to prove home, storage, disable or owned-process cleanup is
   `NEEDS_OPERATOR` and never `safe_shutdown=true`.
7. If any motor axis drops while powered, automatic home is forbidden. Wait
   for driver-owned six-axis disable, then clean up processes.
8. Home is a fresh current-state plan. It never reverses or reuses a prior
   scan trajectory. Startup is J6 mission-ready then rough home; terminal home
   is pre-home, rough home, storage J6, retained final target, disable and cleanup.
9. Dedicated configured-home stages may bypass robot self-collision only for
   the exact hash-bound home transaction. Joint limits, camera-holder external
   floor clearance, live all-six motor authority, timing, feedback convergence,
   disable and cleanup remain mandatory. Camera/tracking/workflow/obstacle-topic
   freshness and Tesseract controller-limit hash are not direct-home gates.
10. Normal acquisition/scan motion always retains Tesseract IK, joint-limit,
    robot/attached-model collision, obstacle and target-visibility checks.
11. A heavy result is correlated to its archived image/depth/intrinsics/stamp.
    It is never combined with latest live depth or TF.
12. Only accepted, persisted, achieved camera FK contributes to coverage.
    Desired/rejected viewpoints do not.
13. Contact remains command-free: `manipulation_model_qualified=false` until
    gripper/TCP/attached-object/allowed-contact paths are separately qualified.
14. Hand/person/unknown/unprojectable obstacles block; semantic uncertainty is
    never converted into contact authorization.
15. TF frame meanings, optical-axis conventions and the accepted hand-eye
    composition are immutable in this phase.

## Driver, feedback and controller limits

| Item | Value | Source/behavior |
|---|---:|---|
| Default joint bounds J1/J3/J4 | `[-2.8, 2.8] rad` | Driver fallback only |
| Default J2/J5 bounds | `[-2.1, 2.1] rad` | Driver fallback only |
| Default J6 bounds | `[-pi, pi] rad` | Driver fallback/URDF authority |
| Default gripper | `[0, 0.08] m` | Driver fallback |
| Recorded valid J1 | `[-2.707029696, 2.697714600] rad` | `piper_joint_bounds.json` |
| Recorded valid J2 | `[-0.044796192, 3.378955132] rad` | Same; controller command is clamped to `[0, pi]` while powered |
| Recorded valid J3 | `[-3.024126728, 0.032201624] rad` | Same; controller command is clamped to `[-2.967, 0]` while powered |
| Recorded valid J4 | `[-1.784835192, 1.786719144] rad` | Same |
| Recorded valid J5 | `[-1.296752072, 1.332599492] rad` | Same |
| J6 recorded sample | invalidated; software authority remains `[-pi, pi]` | Must not revive stale wider sample |
| Gripper recorded | `[0.00035, 0.06986] m` | Same |
| Enable retry period | `0.01 s` | Driver all-axis handshake |
| Enable service timeout | `15 s` default | Driver parameter/launch |
| CAN pair max age/skew | `0.10 s / 0.03 s` | Six-joint coherent sample gate |
| Feedback warning gap | `0.25 s` | Driver diagnostic |
| Motion-limit query/maximum age | `5 s / 10 s` | Driver parameters |
| Motion-limit protocol cap | `3 rad/s`, `5 rad/s^2` | Driver and Tesseract contract reject larger/nonpositive values |
| URDF nominal velocity | J1--J5 `5 rad/s`, J6 `3 rad/s` | Tesseract MoveJ timing model; live controller limits remain required hash evidence |
| Driver telemetry | approximately `200 Hz` | Current architecture contract |

## Mission admission, queue and lifecycle

| Item | Value |
|---|---:|
| Default/maximum mission deadline | `1200 s` |
| Accepted deadline range | `60..1200 s` |
| Task ID | `8..128` safe characters |
| Target prompt | maximum 12 words and 96 normalized characters |
| Supported profile confidence floor | `0.60` |
| Rough target age/future tolerance | `5.0 s / 0.5 s` |
| Rough-target XYZ standard deviation | each `<= 0.30 m` |
| Base-link exclusion radius | `< 0.10 m` returns reposition required |
| Pending missions | maximum `8` |
| Queue coalescing | `1.0 s`, then closest first |
| Optional gateway heartbeat stale | `5.0 s`; on-disk wall heartbeat accepted up to `2.0 s` old |
| Required/maximum captures | `8 / 24` |
| Acquisition looks | maximum `5` |
| Occlusion/contact actions | maximum `6` (contact execution currently blocked) |
| Quality replacement replans | maximum `8` |
| Target-drift replans | maximum `8` |
| Acquisition service call | `8 s` |
| Workflow occlusion assessment | `75 s` coordinator; `70 s` correlated probe configuration |
| Plan request queue | `12 s` |
| Plan result/home execution | `185 s` |
| Transient approval retry | `5 s` |
| Between-view visual reacquisition | `30 s` |
| Driver service startup | `30 s` |
| Vision startup | `120 s` |
| Hand-eye TF startup | `20 s` |
| Tesseract worker startup | `45 s` |
| Pre-enable readiness stable/timeout | acquisition `2 s / 90 s`; multiview `1 s / 30 s` |
| Joint stream stability | usually `2 s` within `15 s`; freshness `0.25 s` during startup |
| Mission service availability probe | at most `5 s` of each call timeout |
| Enable/disable service call | `20 s` |
| Hold/cancel/authorization calls | `8 s` |
| Process stop escalation | SIGINT `5 s`, then SIGTERM `3 s`; autonomous owner never SIGKILLs a live command process |

## Motion speed, timing and tracking gates

| Item | Value | Meaning |
|---|---:|---|
| Autonomous transit/free speed | `30%` configured mission default/launch | Must also have `motion_speed_profile_qualified=true` |
| Contact speed | `10%` | Proposal contract only; autonomous contact is unqualified |
| Manual scan default | `5%` | GUI/launch default |
| Allowed SDK speed percent | clamped/validated `1..100%` | Selected percent is the whole speed contract; tracking cannot increase it |
| Tesseract command rate | `20 Hz` | Time-indexed newest-due MoveJ target, no burst replay |
| Executor safety tick | `200 Hz` | Independent of command issue rate |
| Adjacent command ceiling | `0.05 rad` | Hard stream target bound |
| Following error | `0.30 rad`, `1.0 s` grace | Any joint |
| Plan start tolerance | `0.025 rad` |
| Joint/waypoint goal tolerance | `0.025 rad` |
| Progress epsilon/timeout | `0.001 rad / 20 s` |
| Waypoint timeout | `90 s` |
| Endpoint settled window | `0.005 rad` for `1.5 s`, timeout `15 s` |
| Diagnostic velocity-settled value | `0.20 rad/s` | Retained for compatibility; position stability is authoritative |
| Home goal/motion tolerance | `0.030 rad / 0.005 rad` |
| Home feedback gap/settle/timeout | `1.0 s / 1.0 s / 30 s` |
| Normal feedback-limit tolerance | `0.005 rad` |
| Configured-home start relaxation | maximum `0.3 rad`, feedback-only, direct home only |
| Motion-limit freshness/change promotion | `3 s`; new hash for `7 s` and `3` samples |
| Runtime refresh/recovery | `3 s / 30 s` |
| Plan maximum age | `300 s` |
| Maximum target drift before approval | `0.015 m` |
| Runtime data timeout | `2.0 s` |
| Maximum tracking measurement age | `0.75 s` |
| Minimum tracking speed-scale diagnostic | `0.10` |
| Motor guard grace/status age | `0.5 s / 0.5 s` |
| Motor-loss six-disabled wait | `2.0 s` |
| Autonomous startup/terminal hold service | not called; configured-home feedback proof remains authoritative |
| GUI safe-disable hold | target `0.025 rad`, motion `0.005 rad`, stable `1.0 s`, timeout `8 s`, polling `0.1 s` |
| GUI manual safe-disable speed | clamped `1..5%` |

Home profile schema 3 fixes rough home to
`[0, 0, 0, 0, 0.399345492, 0] rad`, mission-ready J6 to `0`, and storage J6
to `-3.139536232 rad`. Startup J6 direction is increasing and terminal storage
direction decreasing. These are calibrated operator data, not refactor
constants to normalize or recompute.

## Target visibility, acquisition and NBV

| Item | Effective value |
|---|---:|
| Rough hint age/future tolerance/TF timeout | `5.0 s / 0.1 s / 0.25 s` |
| Acquisition maximum/fallback standoff | `0.45 m / 0.28 m`; capped by current hint distance |
| Acquisition cardinal sweep | `15 deg` |
| Acquisition handoff retry/timeout | `0.50 s / 30 s` |
| Acquisition fresh frame/Grounding/tracking/scene | `10 / 60 / 10 / 15 s` |
| Lock position tolerance around rough hint | `0.30 m` |
| Scan target boresight | maximum `20 deg` along every dense path sample |
| Scan target distance | minimum `0.22 m` along every dense path sample |
| Candidate azimuth region | `180 deg` centered at `180 deg`, `7.5 deg` grid |
| Candidate radius | base `0.30 m`; offsets `0, .03, .06, .09, .12, .15 m`; diagnostic max `0.80 m` |
| Candidate pitch | base `-50 deg`; offsets `+35,+25,+15,+5,-5,-15,-25 deg` |
| Generated candidate cap | `25` azimuth samples before pitch/radius expansion; automatic bridge shortlist `36` |
| Local automatic frontier | `6..30 deg` from achieved camera direction |
| Aim relaxation | maximum `12 deg` from target-centered nominal |
| Duplicate pose/look | `0.012 m` and `2 deg`, both must be close |
| Target plan translation/min period/refresh | `0.01 m / 0.50 s / 0.50 s` |
| Coverage Y-side definition | normalized lateral fraction `>= +0.35` or `<= -0.35` |
| Coverage per side | at least `2` accepted achieved views |
| Coverage total floor | geometric helper `9`; mission capture seed `8` |
| Coverage span | at least `120 deg` azimuth and `25 deg` elevation |
| Side non-regression tolerance | `0.02` normalized lateral exposure |
| Measured surface voxels | `0.010 m`, pixel stride `3`, at least `100` occupied voxels |
| Measured surface convergence | `3` consecutive novel fractions `<= 0.02` |
| Surface depth range | `(0.10, 1.50) m` |

The static reach values (`0.20..0.75 m` arm reach, `0.25..0.80 m`
camera-target distance, `0.40 m` height change and the rectangular workspace)
are currently diagnostic because `enforce_static_reach_bounds=false` and
`enforce_static_workspace=false`. Removing them is still an interface change;
turning them on is a behavior change.

## Perception, synchronization and quality

### Camera/depth geometry

| Item | Effective value |
|---|---:|
| Depth accepted range | `0.24..1.20 m` in camera config; obstacle geometry `0.25..1.20 m` |
| Valid pixels | at least `20` (landmark `50`) |
| Camera config valid-depth ratio | `0.0`; quality/obstacle/landmark gate `0.40` |
| Depth crop/ROI | `10 px / 10 px` |
| Depth percentile | `50` (median) |
| Depth stddev rejection/confidence reference | `0.20 m / 0.15 m` camera config |
| Depth jump/reacquisition | `0.20 m`; `3` samples within `0.03 m` |
| Smoothing alpha | `0.2` |
| Mask age/erosion | `0.20 s / 2 px` |
| RGB-D sync queue/slop | `10 / 0.08 s`; heavy/obstacle queues `20 / 0.08 s` |
| Capture bundle age/slop | `1.0 s / 0.08 s` |
| Capture camera TF timeout | `0.25 s` |
| Capture diagnostic timeout | `1.0 s` |
| Capture propagation delay | `0.25 s` |
| Capture timeout/finalize timeout | `20 s / 10 s` |
| Maximum interval frames | `30`; interval `2.0 s` (service mode overrides autonomous path) |

### Quality and occlusion

| Item | Effective value |
|---|---:|
| GOOD/ACCEPTABLE quality | `>=0.65 / >=0.40` |
| Good depth stddev | `<=0.03 m` |
| Quality valid depth/mask/edge | `>=0.40`, `>=100 px`, `40 px` edge margin |
| Quality valid depth range | `0.15..1.20 m` |
| Quality evaluation/stale | `1.0 s / 1.0 s` |
| Autonomous capture | requires GOOD `>=0.65` and CLEAR |
| Occlusion evaluation/stale | `0.25 s / 1.0 s` |
| Occlusion closer-depth margin | `0.04 m` deployed (`0.03 m` planner compatibility value) |
| Occluder persistence/area/dilation | `3 frames / 200 px / 10 px` deployed |
| Occlusion valid depth/mask | `>=0.20 / >=40 px` |
| Partial/heavy closer ratio | `0.12 / 0.60` |
| State/lost confirmations | `1 / 2` |
| Reference initialization | `8 frames`, minimum `300 px`, alpha `0.05` |
| Partial/heavy visible ratio | `0.94 / 0.44` |

### Tracking and temporal recovery

| Item | Effective value |
|---|---:|
| Tracker prediction horizon | `0.15 s` deployed |
| Missed/minimum track frames | `10 / 5` |
| Stable speed/time | `0.03 m/s / 0.4 s` |
| Process/measurement noise | `0.05 / 0.03` |
| Tracker TF timeout | `0.2 s` |
| Minimum measurement/track confidence | `0.03 / 0.20` deployed |
| Depth/pixel/3D/area gates | `0.15 m / 80 px / 0.10 m / 0.5..2.0` (camera-space gates disabled) |
| Maximum target speed | `1.0 m/s` |
| Low-confidence/lost timeout | `0.5 s / 1.0 s` |
| SAM2 frame rate/seed cache | `10 Hz / 60 s` |
| Refresh cooldown/ack/lost/absent/no-mask | `5 / 3 / 10 / 30 / 8 s` |
| SAM2 minimum target area | `100 px` |
| Moving/settled velocity | `0.08 / 0.03 rad/s` |
| Motion window/delta | `0.75 s / 0.012 rad`; settled delta `0.009 rad` |
| Camera settle | `0.5 s` |
| Reacquisition attempts/valid frames | `2 / 5` |
| Low-confidence threshold/duration/hysteresis | `0.60 / 1.0 s / 0.10` |
| Degraded scale | `0.25` diagnostic |
| Measurement/motion-prompt age | `0.75 / 0.25 s`; prompt grace `0.75 s` |
| Timestamp maximum offset/backward tolerance | `0.5 s / 0.001 s` |
| Timestamp healthy/unhealthy frames | `15 / 5` |
| Frame/startup/joint timeouts | `2 / 3 / 1 s` |
| Watchdog stationary position/time | `0.001 rad / 0.75 s`; velocity fallback `0.03 rad/s` |

## Collision and attached-model thresholds

| Item | Value |
|---|---:|
| Proposal/hardware clearance | `0.005 m / 0.005 m` |
| Dense validation joint L1 interval | `0.001 rad` |
| Clearance reporting distance | `0.050 m` |
| Holder/L515 floor | `z=0`, clearance `0.005 m` |
| Holder envelope center/size in link6 | `[-0.029750002,0,0.0375] m`; `[0.1395,0.10572671,0.053] m` |
| Executor link radius/self-clearance | `0.025 m / 0.060 m` |
| Acquisition bootstrap search/max delta/joints/start-limit | `0.01 rad / 0.15 rad / 2 / 0.04 rad` |
| Powered-start home search/max delta/joints | `0.01 rad / 0.10 rad / 1` (J3 only) |
| Monotonic clearance tolerance | `0.0002 m` |
| Allowed acquisition start overlaps | gripper-base/link1 or link2 `0.0001 m`; link1/link5 `0.0065 m`; link2/link5 `0.008 m` |
| Powered-home allowed start overlaps | gripper-base/link1 or link2 `0.0001 m`; link1/link5 and link2/link5 `0.010 m` |
| Pair margins | base/link2 `0.0025 m`; link2/link4 `0.0005 m`; gripper-base/link5 `0.0005 m`; L515 cable/link5 `0.0015 m`; mount/link5 `0.002 m` |

These exceptions apply only to their named plan kind, observation mode, start
state, links and monotonic recovery. They are not global allowed-collision
pairs.

## Contact/removal proposal thresholds

These values remain documented even though autonomous execution is blocked:

- movable labels are pen/marker/stick; two observations and probe confirmation;
- closer-depth and target-overlap ratios each at least `0.05`;
- benefit is at least two unlocked viewpoints or `0.10` predicted surface gain;
- position uncertainty at most `0.025 m`; grasp width at most `0.070 m`;
- pick placement at least `0.120 m` from target, `0.080 m` from obstacles and
  `0.050 m` from a table edge;
- observed support radius/points/stddev: `0.055 m / 16 / 0.008 m`;
- drop search radius `0.180 m`, approach height `0.100 m`;
- push increments: first `0.010 m`, later `0.030 m`, maximum three increments;
- contact speed `10%` and maximum contact actions `6`.

## Calibration validation thresholds

The active calibration's fixed-board validation used five distinct stable arm
poses, 17 mm squares, 12 mm markers, translation limit `15 mm`, and rotation
limit `1.5 deg`. It reported mean/worst translation `5.73/12.22 mm` and
mean/worst rotation `0.87/1.25 deg`. Capture stability uses `0.001 rad` for
`0.75 s`. These values are provenance and must not be silently re-fitted during
code extraction.
