# Runtime qualification and limitations

This reference separates operator-reported physical qualification, software/offline evidence, and observed mission outcomes. A reported qualification is not independently verified, and later failed runs remain failed evidence. Software tests or rootless planning checks do not establish physical safety, motion fidelity, sustainable speed, or geometry equivalence.

## Four-box geometry record

The following is the complete recorded qualification declaration for the fixed four-cuboid scene. It identifies the operator report and its exact limited scope. `independent_verification` is false; no physical run ID or log is recorded. The runtime consumes the integrated model source, not this report.

```yaml
schema_version: 1
record_type: operator_reported_physical_qualification
report_date: '2026-09-05'
model: piper_four_fixed_world_cuboids
qualification:
  hardware_qualified: true
  basis: operator_reported_physical_test
  scope: supervised_5_percent_target_scan
  floor_profile: tabletop
  free_motion_speed_percent: 5.0
  contact_speed_percent: 5.0
  real_motion_requires_explicit_opt_in: true
  statement: Operator explicitly confirmed a successful physical test of the exact four-box geometry with the tabletop profile at 5-percent speed.
  independent_verification: false
  physical_run_id: null
  physical_log_path: null
geometry_identity:
  source_proposal: /home/prl/Downloads/PiPER_Split_Station_20260905_haOGAQ/proposed_fixed_cuboids.yaml
  source_proposal_sha256: 79123563669c06932ec1188f2771e0bb0de636fbb7050e38a8e99cb6a8ecc764
  moving_sphere_model: motion_planning/curobo/model/piper_collision_spheres.yaml
  moving_sphere_model_sha256: e9cc618a1a25e0731fe7c84e5c65278f061ad20cb63dbba5941d31a6484020a5
  frame: base_link
  dimensions_units: metres
  quaternion_order: wxyz
  cuboids:
    - name: bunker_chassis_collision_cuboid
      dims: [1.0673361602087021, 0.7885020109672546, 0.4552615548480749]
      pose: [-0.475544698536396, 0.000250011682510376, -0.22762978411465884, 1.0, 0.0, 0.0, 0.0]
    - name: sensor_station_housing
      dims: [0.5375019642372132, 0.46700200762939453, 0.30000199999999994]
      pose: [-0.6024999916553497, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0]
    - name: sensor_station_lidar_camera
      dims: [0.3236518832702637, 0.17525197144937515, 0.14755199876022335]
      pose: [-0.5291603803634644, -0.00010685995221138, 0.3737749993801117, 1.0, 0.0, 0.0, 0.0]
    - name: piper_base_collision_cuboid
      dims: [0.09850199845027924, 0.08000199821186066, 0.08600200292062761]
      pose: [-0.00925000011920929, 0.0, 0.04300000146031379, 1.0, 0.0, 0.0, 0.0]
deployment:
  status: integrated_default_at_next_worker_startup
  current_runtime_representation: four_cuboids_default_with_explicit_mesh_rollback
  runtime_consumes_this_record: false
  note: This record documents the operator report. The runtime source is motion_planning/curobo/model/piper_fixed_world_cuboids.yaml. Model preparation defaults to cuboids; existing running coordinators require restart. No automatic motion enablement.
  integration_owners: [motion_planning/curobo/generate_robot_config.py, motion_planning/curobo/worker.py, motion_planning/curobo/prepare_model.sh]
  integration_requirements:
    - Bind this exact geometry and moving-sphere identity through normal model provenance validation.
    - Preserve explicit real-motion opt-in and all common execution checks.
    - Select geometry before worker startup; an empty mesh update in pinned cuRobo does not remove already loaded meshes.
    - Run model-generation, malformed-provenance, planner and execution regression tests before deployment.
limitations:
  - No physical trial count or measured physical outcome beyond the operator statement is asserted.
  - Existing offline reports remain unchanged and retain their original evidence classification.
  - Offline testing found 11 additional rejected random states among 1306 samples; no equivalence of reachable workspace is claimed.
  - No scope extension to other geometry, speeds, mounting, calibration or ground profile is implied.
subsequent_saved_run:
  task_id: gui-sim-af49e2b4e79b405d9ad805afde303329
  source: /tmp/piper_target_scan_missions/results/gui-sim-af49e2b4e79b405d9ad805afde303329.json
  source_sha256: 9c5d1da654f462548aa7acb36a8f5748622bd459b499cad5495ee2d25fff562f
  dataset: datasets/active_scan/scan_20260905_233726
  outcome: FAILED
  accepted_captures: 5
  safe_shutdown: true
  elapsed_sec: 593.5588938320288
  view_planning_phase_sec: 333.1248873559525
  reason: RAY_FRONTIER_EXHAUSTED at generation 5 before the then-current eight-view minimum and coverage convergence. That minimum is historical; the current completion policy below supersedes it.
  evidence_class: HISTORICAL_OBSERVATIONAL
  operator_observation: Run felt faster.
  interpretation: Safe termination with captures, not a successful complete mission or a controlled speed comparison. Does not replace the preceding operator-reported qualification statement.
```

The source report states 11 additional rejected random states among 1,306 offline samples. This is not evidence of equivalent reachable workspace. It does not extend qualification to other geometry, speed, mounting, calibration, or floor profiles.

## Later observed run

The saved run `gui-sim-af49e2b4e79b405d9ad805afde303329` ended `FAILED` with five accepted captures and `safe_shutdown: true`, after 593.5588938320288 seconds, at `RAY_FRONTIER_EXHAUSTED` before the then-current eight-view minimum and coverage convergence. That minimum is historical; the current completion policy below supersedes it. The operator said it felt faster. This is a historical observed failure with safe shutdown, not a controlled speed comparison and not a replacement for the preceding operator-reported physical test. The original evidence is retained in the datasets path named in the declaration above.

## Current completion semantics and remaining limits

The current mission policy treats explicit full ray-population exhaustion as scan completion, including zero or fewer than eight captures and unconverged coverage. Capture count, elevation, occupied surface voxels, convergence, and achieved-history counts remain diagnostic. Passing coverage does not stop scanning while rays remain. At the existing 24-capture ceiling, a command-free next-plan check distinguishes full exhaustion from remaining work; remaining work fails before approval. Success still requires terminal recovery and shutdown proof. Therefore `SUCCEEDED` means that the available ray population was exhausted and shutdown completed safely; it does not certify reconstruction or surface coverage. See the current [NBV policy](../architecture/nbv-policy.md), [mission engine](../../piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/mission/engine.py), [failure model](../../piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/infrastructure/failure_model.py), and [ray retry validation](../experiments/nbv_scoring_retry_validation.md).

The following current findings remain material to qualification:

- Invalid joint feedback repair is implemented offline; live restart and supervised powered qualification were still pending in the latest incident record. The packet misassociation cause was not proven. See [home feedback incident evidence](../experiments/home_invalid_feedback_20260910.md) and [driver source](../../piper_ros_foxy/src/piper/piper/piper_ctrl_single_node.py).
- With `allow_target_motion_during_scan` enabled, INITIALIZING keeps the active-scan displacement exception only while a capture occlusion probe is pending and post-action verification is false. Genuine startup, post-action verification, terminal states, and disabled movement allowance remain strict. This exception applies to first and later saved-capture probes; real target movement can pass its displacement check during capture verification. It is offline validated only, with physical behavior unqualified. See [`target_motion_is_terminal`](../../piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/supervised_cube_workflow_node.py).
- Do not mechanically remove an installed end load while its wired tool remains connected and powered; severe flailing was observed in a prior incident. Keep the installed load in place during powered checks.
- Camera freshness analysis found older masks and fewer accepted tracker updates at higher-resolution 15 FPS runs, but exact callback order and per-stage latency were not isolated. Camera stall or GPU overload was not established. See [camera freshness evidence](../experiments/camera_profile_freshness_20260910.md).
- Physical tracking, braking and sustainable command rate remain unqualified where documented; estimated reaction allowances are not stopping-distance guarantees.

## Immutable baseline records

The [four-box qualification declaration](https://github.com/WenjunYU55/Piper_arm/blob/d03cbfe40605cd43d2cdfaaa31f2384962ee60ae/docs/ai/85-four-box-qualification.yaml)
and the [capture-probe displacement policy](https://github.com/WenjunYU55/Piper_arm/blob/d03cbfe40605cd43d2cdfaaa31f2384962ee60ae/docs/ai/50-guardrails.yaml#L1762)
are linked at the reviewed immutable baseline. Current behavior and limits are
summarized in this reference and [safety invariants](safety-invariants.md#occlusion-probe-and-session-assumption).

For physical staging and explicit stop conditions, use [physical requalification](physical-requalification.md). For software-only evidence boundaries, consult the retained individual [experiment records](../experiments/).
