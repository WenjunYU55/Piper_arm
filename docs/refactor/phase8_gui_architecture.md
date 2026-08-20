# Phase 8 GUI result

The native GUI is now a client of `/piper/run_target_scan`. Autonomous mission
and safety authority is not duplicated in Tk callbacks.

The runtime direction is:

```text
Tk input -> MissionViewModel -> MissionActionClient -> RunTargetScan action
                                                        |
Tk display <- typed client events <- feedback/result <--+
```

Commissioning controls are a separate set of direct operator tools. The sole
remaining `subprocess.Popen` in GUI code launches the preview-only RViz joint
editor. The GUI has no production child-process ownership.

An additive configuration-only control now selects either the existing
`legacy` viewpoint heuristic or `voxel_nbv` for the next mission. It atomically
changes only `view_selection_policy` in `scan_planning_params.yaml` while no
mission is active. The mission server still starts the planner stack and owns
all planning decisions; the GUI cannot modify a running stack and does not
implement either policy.

Validation is recorded in the Phase 8 completion report and the repository AI
architecture documents. No physical robot, camera, perception stack or
planning stack was started during Phase 8.
