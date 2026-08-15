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

Validation is recorded in the Phase 8 completion report and the repository AI
architecture documents. No physical robot, camera, perception stack or
planning stack was started during Phase 8.
