# Selected feature reintegration report

Date: 2026-08-16
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
8. One shared ambiguity-aware target-depth selector now serves live Target3D,
   stable landmark, occlusion fallback, and capture, rejecting close-score
   foreground/background ambiguity instead of switching layers.
9. L515 confidence-qualified capture schema 2 exact-correlates SAM mask/RGB,
   synchronizes native depth/confidence, projects one qualified target layer,
   and transactionally persists immutable provenance artifacts.

Existing ROS interfaces remain compatible. `ExecuteHomeStage` is one additive
internal service. Existing numerical speed, limit, timeout, freshness, target
drift, capture, and retry defaults are unchanged.

## Deliberately deferred

- Reachable full-sphere NBV v2 is still needed to address the observed local
  frontier/same-sector coverage failure; cleaner code alone does not change
  that algorithm.
- NBV planning-budget recovery should follow the full-sphere port only if
  measured traces still show repeated candidate generation or timeouts.
- Shared depth-layer selection and confidence-qualified schema 2 are now
  software-integrated in that order. A motors-disabled native L515 stream test
  passed on 2026-08-16 after fixing an eager-control USB startup race, but
  stationary target jitter and a persisted schema-2 capture still require
  live-sensor qualification before their physical effect is claimed.

## Qualification boundary

The original selected-port validation completed with 592 mobile-manipulation
tests. The feature-7/8 extension plus the live-discovered headerless
arm-status regression bring that suite to 605 passing tests. The complete
software matrix now passes 102 Tesseract
tests, 54 PiPER-driver tests plus one intentional skip, 93 root
GUI/calibration/description/reconstruction tests plus one intentional skip,
and 26 focused reconstruction/provenance tests. The complete five-package ROS
build succeeds.

`colcon test` reports every functional test green. Its aggregate result remains
nonzero only for known repository style debt: the mobile package currently has
86 flake8 and 21 pep257 findings, improved from the Phase-0 baseline of 95 and
23. The feature-7/8 files introduce no new linter findings.

Motion-affecting ports are not physically qualified by those results.
Use `physical_requalification_checklist.md` from the first stage; do not jump
directly to an autonomous scan.
