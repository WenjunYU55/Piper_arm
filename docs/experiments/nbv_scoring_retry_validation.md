# NBV scoring correction and retained exclusion policy — 8 September 2026

The 6–7 September cube scans followed their recorded information ranks, but scored
small targets at an approximately 1.64 m nominal ray midpoint. Supported camera
intervals were typically near 0.29 m. Four projected scoring pixels quantized the
gain and created many ties. The bridge intentionally retains exhausted-ray exclusions across later robot
starting poses to limit repeated planning; this policy is preserved.

The prefilter now supplies a separate nearest-supported scoring distance. The
coverage owner uses that distance without changing ray motion geometry or endpoint
search bounds. With no enforcing map, it uses the sensor/envelope ray minimum.
The 64x48 scorer and normalized information-first order remain unchanged.

The original failed-ray exclusion policy is retained at the user's direction:
correlated exhausted rays stay excluded across robot movement and accepted captures
until explicit population replacement or a new session. This intentionally bounds
runtime at the cost of potentially missing a ray that could succeed from a later
start. The proposed retry reset and its GUI/protocol changes were withdrawn. No
physical mission used that proposed policy. The scoring-distance correction remains.

## Offline evidence

[nbv_scoring_retry_20260908T043231Z](../../datasets/experiment_results/nbv_scoring_retry_20260908T043231Z/README.md) contains numerical CSV/JSON, the replay script, source hashes and validation logs.

Rescoring covered **696 candidates across 38 generations from nine scans**.
The original gain was reproduced exactly for every compared candidate. Projected
target support increased from **4 pixels to 59–87 pixels**. In the latest
`scan_20260907_142623`, cycle-one support became **67–80 pixels**; the number of
distinct gain values changed from 2 to 3, and cycle three changed from 3 to 5.
Some first-generation views still tie legitimately. No claim is made that every
new top-ranked view is reachable or that a mission will complete.

Historical snapshots omitted render/FOV settings; the replay explicitly assumes
the documented 64x48/20000 limits and derives FOV from the corresponding recorded
CameraInfo. Exact reproduction of old gain is asserted. The replay includes only
survivors with retained capability intervals, preserving the same membership.
Future snapshots now record FOV tangents, render dimensions and the voxel budget.
Original snapshots and scans were not rewritten.

## Validation and limits

- Full ordinary mobile, Tesseract, GUI and cuRobo suite: 1,482 passed, 20 skipped
  (before the final additional GUI static-cull regression). The final focused
  suite passed 274 tests; a 150-test planning/diagnostic subset also passed. These
  earlier logs include the subsequently withdrawn retry-policy change; final
  restoration validation is recorded below.
- Nine self-contained pinned real-GPU tests passed without hardware commands,
  covering fallback aiming, collision/world checks and native path timing.
  Ten fixture-dependent GPU cases were not selected: their old temporary request
  fixtures are unavailable; this change does not modify the GPU worker.
- All five ROS packages built. Affected-package functional colcon tests exclude
  the previously documented package-wide flake8/pep257 checks; current changed
  source is checked separately.
- No August 20 recorded scans were present under the current active_scan root;
  the nine available September scans provide the saved-data comparison instead.
- Approximate capability-map false negatives, unseen geometry and genuine score
  ties remain limitations. Exhausted-ray exclusions intentionally persist across changed starts and accepted counts.
- Physical coverage, latency and staged shutdown require a supervised follow-up.
  No mission, live-node restart, camera setting change or robot command was issued.

## Final retained scope

Supported-distance scoring and its provenance are retained. Bridge retry logic,
its tests, and Ray Review model/tests have been restored byte-for-byte to HEAD.
Only scoring metadata additions remain in ray_mission_diagnostics.py.

After restoring the intentional exclusion policy: **1,480 passed, 20 skipped**
in the complete mobile/Tesseract/GUI/ordinary-cuRobo run. All five ROS packages
built again; AI YAML unique-key validation and diff whitespace checks passed.
No physical mission or live application restart was performed.

## Tesseract partial-success bookkeeping — 9 September 2026

The live audit found repeated failed rays because Tesseract's successful response
contains typed candidate failures but no explicit exhausted-ray list. The bridge
now maps those failures through the exact pending request. It retires only rays
whose request alternatives all exhausted their aim searches and were not selected.
Generic failures, unknown IDs, successful/unattempted rays, stale generations and
changed populations cannot enter this new path. cuRobo's existing explicit list,
worker geometry, capability-map binding and scoring are unchanged. Evidence and
validation: `datasets/experiment_results/tesseract_retry_fix_20260909/`.
