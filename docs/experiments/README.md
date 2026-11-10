# Experiment evidence

These records retain measured outcomes, reproduction methods, incident diagnosis
and qualification limits. They are distinct from the supported operating
procedure in [OPERATOR_COMMANDS.md](../../OPERATOR_COMMANDS.md) and the current
[runtime qualification reference](../reference/runtime-qualification.md).

## Methods and validation

- [Planner backend benchmark](planner_backend_benchmark.md)
- [Paired results campaign](results_campaign.md)
- [Reconstruction recording validation](reconstruction_recording_validation.md)
- [Automatic scan recording](automatic_scan_recording_validation.md)
- [Native BundleSDF validation](bundlesdf_native_validation.md)
- [BundleSDF GUI validation](bundlesdf_gui_validation.md)
- [Upstream BundleSDF workflow](bundlesdf_upstream_workflow_audit.md)
- [Dense offline video](dense_offline_video_validation.md)
- [Aligned-depth frameset validation](aligned_depth_fix_validation.md)
- [Offline distortion validation](offline_distortion_validation.md)
- [NBV scoring and retries](nbv_scoring_retry_validation.md)
- [Saved camera recovery](saved_camera_bundlesdf_recovery.md)
- [Movement recording compaction](movement_recording_compaction_20260909.md)

## Hardware and capture diagnosis

- [cuRobo start and target jumps](curobo_start_and_target_jump_20260909.md)
- [Invalid home feedback](home_invalid_feedback_20260910.md)
- [Joint 4 startup feedback](joint4_startup_feedback_20260910.md)
- [Joint 4 vendor evidence](joint4_vendor_report_20260910.md)
- [Camera profile freshness](camera_profile_freshness_20260910.md)

Keep the accompanying raw measurements and figures with their reports. An
operator statement, an observational run, command-free replay and a controlled
physical comparison are different evidence classes; a report's limitations travel
with its result.

## Historical source provenance

The method-comparison workbook, JSON and analysis generator retain their original
source-attribution strings. Some name historical handoffs or maintenance files
that are no longer part of the working tree. Resolve those paths against the
[immutable source snapshot](https://github.com/WenjunYU55/Piper_arm/tree/d03cbfe40605cd43d2cdfaaa31f2384962ee60ae)
used before the documentation cleanup. The same rule applies to dated changed-file
listings within validation reports. These are historical citations, not runtime
file dependencies or current setup instructions. Original measurements and
qualification results have not been rewritten.
