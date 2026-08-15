# Selected feature reintegration report

Date: 2026-08-15
Branch: `reintegrate/selected-archived-features`

This branch starts from the cleaned Phase 0--10 architecture and selectively
ports archived features rather than restoring the later monolithic codebase.
No physical robot, camera, GPU, or mission process was started during the work.

## Active ports

1. Configured home uses the typed, mission-authorized `ExecuteHomeStage`
   service and no longer calls Tesseract.
2. Home profile schema 4 adds terminal-only `PRE_HOME`. Startup remains
   `STARTUP_WRIST -> ROUGH_HOME`; terminal shutdown is
   `hold -> PRE_HOME -> ROUGH_HOME -> STORAGE_WRIST -> hold -> disable`.
3. The PiPER driver owns the startup-only positive J6 branch and qualified raw
   wrap bridge. Executor startup and hold commands carry explicit driver tags.
4. A failed zero-capture dataset is deleted only after writers stop and only
   when path, mission identity, manifest, capture, symlink, and unknown-artifact
   guards all pass.
5. A normal Tesseract path may use one native MoveJ endpoint only after the
   complete straight joint chord passes independent dense validation. A route
   requiring collision avoidance retains its full streamed detour.
6. Internal pass-through blending is bounded and the actual blended geometry
   is densely validated; any rejection falls back to the exact source polyline.
7. Offline reconstruction supports robot-pose masked TSDF, optional bounded
   GICP, provenance/quality reports, and command-free GUI preview/inspection.

Existing ROS interfaces remain compatible. `ExecuteHomeStage` is one additive
internal service. Existing numerical speed, limit, timeout, freshness, target
drift, capture, and retry defaults are unchanged.

## Deliberately deferred

- Reachable full-sphere NBV v2 is still needed to address the observed local
  frontier/same-sector coverage failure; cleaner code alone does not change
  that algorithm.
- NBV planning-budget recovery should follow the full-sphere port only if
  measured traces still show repeated candidate generation or timeouts.
- Shared depth-layer selection remains the appropriate fix for foreground/
  background depth ambiguity and target landmark jitter.
- Confidence-qualified L515 capture schema 2 remains a later sensor-provenance
  enhancement and should follow shared depth selection.

## Qualification boundary

Software validation completed with 592 mobile-manipulation tests, 102
Tesseract tests, 54 PiPER-driver tests plus one intentional skip, and 93
root GUI/calibration/description/reconstruction tests plus one intentional
skip passing. The five selected ROS packages build successfully. `colcon test`
reports every functional test green; its aggregate result remains nonzero only
for the known repository-wide mobile-package style debt (89 flake8 and 21
pep257 findings, improved from the Phase-0 baseline of 95 and 23).

Motion-affecting ports are not physically qualified by those results.
Use `physical_requalification_checklist.md` from the first stage; do not jump
directly to an autonomous scan.
