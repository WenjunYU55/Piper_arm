# Motion-planner backends

## Scope and safety status

The scan mission has one backend-neutral planning path. `planner_backend` is
either `tesseract` or `curobo`, defaults to `tesseract`, is validated before
mission admission, and is copied into the immutable mission goal/hash. There
is no automatic fallback and no mid-mission switch.

Tesseract is the regression baseline. The cuRobo adapter is real MotionGen
integration against cuRobo v0.7.8. On 28 August 2026 the reference host passed
native-extension import, CUDA tensor execution, model warm-up, command-free
free-space planning, worker readiness, and bounded cleanup. The rigid PiPER
base and fixed Bunker world now use exact hash-bound triangle meshes; moving
links and attachments use a reviewed, hash-bound 69-sphere Isaac/Lula
approximation because cuRobo 0.7.8 does not support articulated triangle-mesh
collision. That approximation remains marked `conservative_geometry: false`.
On 1 September 2026 the operator explicitly promoted the reviewed model to
`hardware_qualified: true` for supervised 5% testing after the command-free
comparison suite. This is not a claim of Tesseract-equivalent geometry. No
planner in this architecture directly commands the robot.

## Runtime architecture

```text
RunTargetScan / MissionEngine / NBV candidate
                    |
                    v
       /motion_planner/request_*
                    |
          generic Foxy bridge
                    |
       schema-v5 private spool request
                    |
       +------------+------------+
       |                         |
Tesseract ROS-free worker   cuRobo ROS-free worker
       |                         |
       +------------+------------+
                    |
          validated MotionPlan
                    |
   common schedule normalization/validation
                    |
           ScanExecutionPlan
                    |
       PlanAuthorizer + runtime gates
                    |
    common executor / TrajectoryRunner
                    |
             PiPER driver
```

The selected worker is the only planner worker started for a mission.
`ProcessSupervisor` owns its process group, generation, heartbeat, bounded
termination, and cleanup. The cuRobo script uses `exec` with the exact
`PIPER_CUROBO_PYTHON` path, so process-group termination also owns CUDA work.
The cuRobo worker refreshes a compact `worker_health.json` within the
coordinator's 16 KiB bounded-input contract. Full environment and
collision-model provenance is retained separately in atomic
`worker_diagnostics.json`; that diagnostic record is not readiness authority.

## Generic ROS boundary

Production interfaces:

- `RequestMotionPlan.srv`
- `/motion_planner/request_plan`
- `/motion_planner/request_acquisition_plan`
- `/motion_planner/request_return_home_plan`
- `/motion_planner/request_startup_home_plan`
- `/piper/motion_plan` (`MotionPlan.msg`)
- `/piper/motion_plan_status` (`MotionPlanStatus.msg`)
- `/piper/planner_readiness` (`PlannerReadiness.msg`)
- `/piper/view_generation`
- `/piper/motion_plan_provenance`

The bridge keeps the old `RequestTesseractPlan`, `TesseractPlan`, status,
readiness, view-generation, provenance and `/tesseract_plan_bridge/request_*`
names only for Tesseract compatibility. Compatibility messages are not
published in cuRobo mode. The production mission and executor consume only the
generic interfaces.

## `MotionPlan` and `ScanExecutionPlan`

`MotionPlan` is the command-free planner transport. It carries backend and
version, transaction identities and hashes, plan kind, target/view metadata,
collision qualification, complete timed six-joint trajectories, startup
recovery evidence, execution schedule policy, and comparable planning metrics.

The common executor independently validates `MotionPlan` against current
controller limits, target/scene freshness, mission backend, hashes, joint
order, finite values, timestamps, command rate, joint-step ceiling,
speed-scaled MoveJ limits, collision qualification, and bootstrap evidence.
Only then does it publish `ScanExecutionPlan`, the authorization-facing summary
of the normalized executable plan. `ScanExecutionPlan` is not a second planner
result and cannot restore information rejected at the `MotionPlan` boundary.

### Former `TesseractPlan` field classification

| Fields | Classification | Current owner |
|---|---|---|
| plan/request IDs and hashes, plan kind, backend/version, validity, collision qualification, rejection codes, target/view geometry, trajectories, clearance/link diagnostics | generic planner output | `MotionPlan` |
| planning duration, candidate/feasible/success counts | generic comparable diagnostics | `MotionPlan` and provenance |
| execution speed, command rate, timing policy | PiPER execution policy bound by the request and revalidated by the executor | `MotionPlan` then `ScanExecutionPlan` |
| bootstrap recovery endpoints/joints/deltas/evidence | generic acquisition/start-state safety evidence | `MotionPlan`; both backends support the bounded in-limit folded-start escape, while cuRobo explicitly rejects out-of-limit bootstrap |
| planner-native attempt traces | backend diagnostic metadata | private response/provenance, never the executor |
| duplicated Tesseract transport | compatibility only | `TesseractPlan` alias publisher in Tesseract mode |

`TIMED_STREAM` and `timed_stream_v1` are the generic execution mode/policy.
`TESSERACT_STREAM`, `tesseract_stream_v3`, `validate_tesseract_point`, and
`validate_timed_tesseract_path` remain parsing/import aliases only.

## Backend selection and GUI

The Automatic Scan tab exposes exactly **Tesseract** and **cuRobo** under
“Motion planner for next mission”. **Apply for Next Mission** persists the
validated value in `config/planner_backend.yaml`. Starting the single automatic
scan path reads it into a frozen `MissionRequest`, sends it in
`RunTargetScan.Goal.planner_backend`, and the mission includes it in its
canonical hash. The controls are disabled while a mission is active. Editing
the persisted value cannot change the active goal.

Headless selection uses the same mission field, with
`PIPER_PLANNER_BACKEND=tesseract|curobo` as the launch/default source. Unknown
or missing persisted values fail closed; an empty legacy action field resolves
to the typed `tesseract` default.

## Backend behavior

| Plan kind | Tesseract | cuRobo |
|---|---|---|
| `MULTIVIEW_SCAN` | supported | MotionGen pose plans plus optional joint-space home |
| `ROUGH_ACQUISITION` | supported, including qualified folded-start recovery | supported inside all six limits, including the acquisition-only bounded folded-start escape |
| `RETURN_HOME` | supported | MotionGen joint-space plan |
| `OCCLUSION_PROBE` | not in the active schema-v5 worker contract | explicitly unsupported |
| `OCCLUDER_PICK_PLACE` | not production-qualified | explicitly unsupported |
| `OCCLUDER_PUSH` | not production-qualified | explicitly unsupported |

Unsupported kinds return a structured failure. They are never approximated and
never trigger another backend.

## cuRobo implementation and model

The worker uses the verified v0.7.8 APIs:

- `MotionGenConfig.load_from_robot_config`
- `MotionGen` and `MotionGenPlanConfig`
- `plan_single` for target-facing camera poses
- `plan_single_js` for home poses
- `Pose`, `JointState`, `WorldConfig`, `Cuboid`, and fixed-world `Mesh`
- `MotionGenResult.get_interpolated_plan()` and `interpolation_dt`

cuRobo tensors are reordered to the canonical `joint1..joint6` order and
converted to ordinary finite lists before the response is written. Native
paths are slowed/subdivided to the unchanged 20 Hz PiPER schedule and then pass
the common validator. Tensor/CUDA types never enter ROS, mission, NBV,
authorization, or execution code.

`prepare_model.sh` first invokes the existing Tesseract model builder, so both
backends start from the same current Xacro, calibrated L515 frame, SRDF,
collision manifest, joint order and limits. The cuRobo conversion represents:

- moving PiPER, gripper, holder and L515 geometry as a reviewed 49-sphere
  model extracted from the final operator-saved Isaac/Lula USD;
- cable and mount envelopes as 20 circumscribed regular-cell spheres that
  cover their complete canonical boxes;
- the rigidly bolted PiPER base as its exact fixed `base_link.STL` world mesh,
  so the intentional base/Bunker mounting overlap is not treated as a
  robot-versus-world collision;
- the fixed Bunker chassis and sensor station as their exact base-frame STL
  meshes, not the previous 62 overlapping AABB cuboids;
- the selected support floor as a world cuboid;
- current authoritative perception obstacles as request-bound world cuboids.

Differences from Tesseract are material: Tesseract uses the exact configured
mesh/convex model for both moving and fixed geometry and reports comparable
clearances. cuRobo now uses exact fixed base/Bunker meshes, but articulated links
remain non-conservative spheres and clearance remains unavailable (`-1`). The
current audit reports 69 spheres and a worst per-owner sampled-surface gap of
48.3 mm. Link 5 has 52.5 percent sampled coverage and a 34.4 mm maximum gap.
A measured 7 mm Link-1 self-collision buffer closes every observed
state-level miss while retaining the zero, neutral and qualified-scan poses.
The low count avoids cuRobo's large self-collision kernel, but the sparse
coverage—especially around link 5—means collision equivalence is not claimed.
cuRobo still lacks Tesseract's qualified out-of-limit bootstrap recovery. For
the configured in-limit rough-home start, only `ROUGH_ACQUISITION` with
`bootstrap_static` may search a single joint-2 or joint-3 escape of at most
0.15 rad. Intermediate states may retain only cuRobo's start-self-collision
classification; the endpoint and all later MotionGen planning must pass the
ordinary constraints. The prefix and its joint/delta/boundary evidence then
pass through the existing common executor bootstrap validation.

The generated config binds the pinned cuRobo version/commit and SHA-256 hashes
of its generated URDF, SRDF, collision manifest, reviewed sphere YAML and
source Isaac USD, every source moving-link mesh, the rigid base mesh, and both
fixed Bunker meshes.
Schema-2 loading fails before CUDA initialization if the audit is incomplete,
the sphere counts/owners disagree, or a runtime source asset changed. Hardware collision
qualification additionally requires both `hardware_qualified: true` in
reviewed model provenance and
`PIPER_CUROBO_COLLISION_MODEL_QUALIFIED=1`; the environment flag alone cannot
grant motion authority.

## Pinned runtime and host validation

- cuRobo: `v0.7.8`
- exact commit: `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`
- host GPU: NVIDIA GeForce RTX 3090, 24,576 MiB
- driver: 570.133.07
- driver-reported CUDA runtime: 12.8
- current isolated Python: 3.10.20
- current PyTorch: 2.11.0+cu128
- torch CUDA: 12.8; CUDA available: yes
- cuDNN: 9.19.0
- installed cuRobo: 0.7.8 at commit
  `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`
- installed Warp: 1.11.1 (pinned because 1.13+ removes the `warp.torch`
  API used by cuRobo 0.7.8)
- current CUDA compiler: 12.8.93 at `/home/prl/.local/cuda-12.8/bin/nvcc`

The v0.7.8 documentation supports Ubuntu 20.04/22.04, Volta-or-newer NVIDIA
GPUs with at least 4 GB, Python 3.8–3.10, and PyTorch 1.15+ (2.x recommended).
The RTX 3090 and Python 3.10 meet those published bounds. The exact
PyTorch-2.11/CUDA-12.8 combination is not recorded as qualified until the
pinned source builds and the GPU suite passes. v0.8 is not substituted because
it changes the public planner API.

## Deterministic environment setup

Do not install into Foxy's Python and do not alter the driver or system Python.
This workstation uses a toolkit-only, non-root CUDA install. The driver, ROS,
system Python and system CUDA packages were not changed:

```bash
/home/prl/.local/cuda-12.8/bin/nvcc --version
```

Create an isolated environment from the available Python 3.10 interpreter:

```bash
/home/prl/Piper_arm/AI_perception_tests/groundingdino_test/envs/grounded_sam2_py310/bin/python \
  -m venv /home/prl/.venvs/piper-curobo-v0.7.8
CUROBO_PYTHON=/home/prl/.venvs/piper-curobo-v0.7.8/bin/python
"$CUROBO_PYTHON" -m pip install --upgrade pip setuptools wheel ninja
"$CUROBO_PYTHON" -m pip install torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$CUROBO_PYTHON" -m pip install -c \
  /home/prl/Piper_arm/motion_planning/curobo/constraints.txt \
  warp-lang==1.11.1
git lfs install
git clone https://github.com/NVlabs/curobo.git \
  /home/prl/.venvs/curobo-src-v0.7.8
git -C /home/prl/.venvs/curobo-src-v0.7.8 checkout \
  d64c4b005459db10c5dd867d8b30a87d5bda9bdb
CUDA_HOME=/home/prl/.local/cuda-12.8 \
PATH=/home/prl/.local/cuda-12.8/bin:"$PATH" \
  "$CUROBO_PYTHON" -m pip install -e \
  /home/prl/.venvs/curobo-src-v0.7.8 --no-build-isolation \
  -c /home/prl/Piper_arm/motion_planning/curobo/constraints.txt
```

Verify before configuring the mission:

```bash
CUROBO_PYTHON=/home/prl/.venvs/piper-curobo-v0.7.8/bin/python
env -u PYTHONPATH "$CUROBO_PYTHON" -c 'import curobo, torch, warp; print(curobo.__version__, warp.__version__, torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

Qualification on 2026-08-31 converted the saved 54-sphere Isaac/Lula edit
into 49 moving-link spheres plus 20 exact-cover cable/mount envelope spheres.
Camera holder/L515 spheres are transformed into `l515_attached_assembly`; the
rigid base is an exact fixed-world mesh. The 2026-09-01 operator refinement
removed the dominant link3/link5 false-positive cluster. A deterministic
2,004-pose articulated self-collision comparison against exact Tesseract then
produced zero state-level false negatives and 18 conservative false positives
without adding any ignored pair. Real CUDA
planning passes neutral-to-scan and reverse, a deliberately blocking world is
rejected, and the known folded self collision remains rejected afterward.

The adapter also restores world- and self-collision constraints before and
after every attempt. This contains a pinned cuRobo v0.7.8 invalid-start path
which otherwise leaves self-collision disabled after classifying a world
collision in a persistent worker.

Earlier qualification on 2026-08-28 proved native extension import, CUDA tensor
execution, MotionGen model warm-up, a command-free free-space joint plan,
healthy worker publication and bounded Ctrl-C cleanup. The generated PiPER
model locks gripper joints 7/8 at zero for six-axis arm planning. The corrected
sphere pruning retains legitimate 40 mm grid neighbours instead of collapsing
them through rounded keys, while removing near-duplicate centres largest-first.
The exact Bunker mesh world now passes command-free plans from the canonical
neutral pose to a previously qualified scan pose and back, and a deliberately
blocking dynamic box is rejected. The folded start remains a narrowly scoped
bootstrap-recovery problem, not evidence for ignoring cuRobo collisions
elsewhere. A command-free 1 September 2026 replay from the configured
rough-home joints found a 0.06 rad joint-3 escape, passed the common 60 mm
proxy-clearance gate, and then produced a normally collision-checked MotionGen
acquisition path.

The 2026-09-01 articulated self-collision comparison used 2,000 seeded joint
samples plus four reference poses. It found zero state-level false negatives,
43 mutually colliding states, 1,943 mutually clear states, and 18 conservative cuRobo
rejections. This is useful command-free evidence, not a proof over continuous
configuration space.

The model is operator-promoted to `hardware_qualified: true` for supervised 5%
testing. Moving-link spheres still have per-owner sampled-surface gaps up to
48.3 mm, Link 5 sampled coverage is only 52.5 percent with a 34.4 mm gap,
conservative false positives remain, and the configured-home policy is not
equivalent across backends. These limitations remain visible in provenance and
must not be described as collision-model equivalence.

## Controlled planner replay

On 1 September 2026, five recorded positive requests were replayed three times
per backend after warm-up. Both backends solved 15/15, both rejected 3/3
deliberately blocked controls, and all 30 successful paths passed Tesseract's
exact geometry validator. Median request wall time was 19.646 seconds for
Tesseract and 0.630 seconds for cuRobo. Median scheduled trajectory duration
was 4.750 seconds and 18.209 seconds respectively, leaving the median
planning-plus-scheduled-duration proxy effectively equal at 25.046 versus
25.048 seconds.

This is command-free `CONTROLLED_REPLAY`, not physical evidence. It establishes
that cuRobo is a substantially faster proposal generator for these requests,
but not that it executes a mission faster or more safely. Full methods,
artifacts and limitations are in
`docs/experiments/planner_backend_benchmark.md`.

Version and installation references:

- [cuRobo v0.7.8 source](https://github.com/NVlabs/curobo/tree/v0.7.8)
- [cuRobo v0.7.8 release](https://github.com/NVlabs/curobo/releases/tag/v0.7.8)
- [Version-matched installation guide](https://curobo.org/get_started/1_install_instructions.html)

## Running and testing

Tesseract mode retains the established defaults:

```bash
cd /home/prl/Piper_arm
PIPER_PLANNER_BACKEND=tesseract ./run_target_scan_mission.sh
```

cuRobo command-free startup (it remains unready for hardware until qualified):

```bash
cd /home/prl/Piper_arm
PIPER_PLANNER_BACKEND=curobo \
PIPER_CUROBO_PYTHON=/home/prl/.venvs/piper-curobo-v0.7.8/bin/python \
PIPER_CUROBO_CUDA_HOME=/home/prl/.local/cuda-12.8 \
PATH=/home/prl/.local/cuda-12.8/bin:"$PATH" \
./run_target_scan_mission.sh
```

Real arm motion remains disabled unless the existing independent mission motion
opt-ins are also supplied. Selecting cuRobo does not enable motors or motion.

Ordinary tests do not import cuRobo:

```bash
source /opt/ros/foxy/setup.bash
source piper_ros_foxy/install/setup.bash
python3 -m pytest -q \
  piper_ros_foxy/src/piper_mobile_manipulation/test \
  piper_ros_foxy/src/piper_tesseract_foxy/test \
  tests/curobo tests/gui/test_piper_gui_planner_backend.py
```

GPU tests consume frozen, real bridge requests for all three plan kinds and
never publish robot commands:

```bash
PIPER_RUN_CUROBO_GPU_TESTS=1 \
PIPER_CUROBO_ROBOT_CONFIG=/tmp/piper_curobo_model/piper_curobo.yml \
PIPER_CUROBO_GPU_MULTIVIEW_REQUEST=/path/to/multiview.request.json \
PIPER_CUROBO_GPU_ACQUISITION_REQUEST=/path/to/acquisition.request.json \
PIPER_CUROBO_GPU_RETURN_HOME_REQUEST=/path/to/return_home.request.json \
/home/prl/.venvs/piper-curobo-v0.7.8/bin/python -m pytest -q \
  tests/curobo/test_curobo_gpu_integration.py
```

Plan comparison data is recorded in `MotionPlan`, motion-plan provenance and
ray diagnostics: backend/version, plan/request IDs, result and failure reason,
planning time, trajectory duration/point count/path length, candidate counts,
selected views and available clearance evidence.
