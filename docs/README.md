# Documentation

Start here for installation, operation and technical explanations. Current code
and configuration define behavior; use the operator guide for supported hardware
procedures and the qualification references for their limits.

## Get started

| Task | Document |
| --- | --- |
| Understand the project | [Project overview](../README.md) |
| Install a workstation | [Clean installation](../CLEAN_INSTALL.md) |
| Check supported versions | [Validated environments](reference/validated-environments.md) |
| Run, cancel, recover or shut down | [Operator commands](../OPERATOR_COMMANDS.md) |
| Inspect or manufacture the enclosure | [CAD and fabrication files](../CAD/README.md) |
| Use the optional legacy Docker setup | [Foxy Docker reference](reference/docker-foxy.md) |

## Understand the system

| Topic | Document |
| --- | --- |
| Responsibilities and dependency direction | [Architecture](../ARCHITECTURE.md) |
| Runtime pipeline and process boundaries | [System overview](architecture/system-overview.md) |
| Detailed system and feedback diagrams | [System diagrams](architecture/system-diagrams.md) |
| Component design and compatibility | [Component design](architecture/component-design.md) |
| ROS, configuration and persistence interfaces | [Interface contracts](reference/interface-contracts.md) |
| Command authority and runtime safety | [Safety invariants](reference/safety-invariants.md) |
| Physical qualification procedure | [Requalification checklist](reference/physical-requalification.md) |
| Qualified scope, known limits and incident evidence | [Runtime qualification](reference/runtime-qualification.md) |
| Canonical CAD, meshes and generated assets | [Asset policy](architecture/asset_policy.md) |

## Subsystems

| Area | Document |
| --- | --- |
| L515 build, calibration and camera checks | [Camera guide](../L515_camera/README.md) |
| GroundingDINO and SAM2 setup/workers | [Perception guide](../AI_perception_tests/README.md) |
| Tesseract/cuRobo selection, setup and plan contract | [Planner backends](architecture/motion_planner_backends.md) |
| NBV scoring, retries and completion | [NBV policy](architecture/nbv-policy.md) |
| Planner latency and cache behavior | [Planning latency](architecture/planning_latency_improvements.md) |
| Motion stream and GUI rates | [Trajectory stream rates](architecture/trajectory_stream_rates.md) |
| Recording, replay and reconstruction commands | [Reconstruction recording](architecture/reconstruction_recording.md) |
| Recorder, depth pairing, distortion and output frames | [Recording and frame contracts](architecture/recording-and-frame-contracts.md) |
| Offline GPU reconstruction | [Native BundleSDF setup](architecture/bundlesdf_native_runtime.md) |
| Temporal segmentation and dense inputs | [Dense offline video](architecture/dense_offline_video.md) |
| Tracked-platform description integration | [Integration contract](../integration/track_robot_description/README.md) |

## Develop, test and interpret results

- [Contribution and test guidance](../CONTRIBUTING.md)
- [Repository development tests](../tests/README.md)
- [Experiment evidence and historical source provenance](experiments/README.md)
- [Paired results-campaign methods and commands](experiments/results_campaign.md)
- [Method comparisons and evidence classification](chapter5_results_summary.md)

Experiment records describe particular software versions, inputs and trials.
They do not override current operator procedures or extend hardware qualification.
A successful software test is not permission to move the robot.
