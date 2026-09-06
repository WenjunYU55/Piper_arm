# Capability lookup and diagnostic latency

## Change (6 September 2026)

Mission launch now reads the GUI's authoritative `final_capture_aim_tolerance_deg`
once and passes that same float to the bridge, executor and both capability
consumers. The full GUI range of 1-90 degrees is supported. Changing the GUI
affects the next mission only. The internal `capability_direction_tolerance_deg`
parameter is a standalone default (5.0), not a second mission setting.

With the existing 10-degree bins, lookup adds approximately 7.07 degrees of
bin half-diagonal slack. A GUI value of 5 therefore gives a 12.07-degree
bin-centre threshold; exact capture admission remains 5 degrees. At a GUI
value of 15, lookup matches the former atlas slack, but later capture aiming
also follows the selected 15-degree tolerance. The first capture still uses
min(selected tolerance, 5 degrees), regardless of broader map support. Wider
lookup can forward more rays and increase planning time; it grants no motion
authority. No atlas regeneration is needed.

The NPZ, occupancy, floor data and source hashes are untouched. Legacy offline
loader calls without an override still use the original artifact policy.
Runtime diagnostics distinguish effective slack from artifact slack.

Recorded-pose replay retained 48/48 achieved and nominal-look poses from 12
post-build scans. On 26 distinct rays appearing in failed requests, support
fell from 26 to 7. Those are failed searches, not proven impossible poses.
The sample does not establish full-workspace recall; tighter lookup may reject
other valid views and affect final coverage. Exact IK, collisions, dense path
checks, final capture admission and authorization remain mandatory.

## Diagnostic persistence

Early capability exclusions now persist under the canonical ray-population
identity in StaticRayPrefilter. They are pruned before later history filtering
and NBV, rather than repeatedly passed through the query cache and reported
as a capability-cull operation. New mission/session or bootstrap-to-qualified
population replacement clears retention; another accepted capture does not.
Remaining unchanged queries still use the exact-input cache, and downstream
current-state admission remains mandatory. Carried exclusions remain visible
in the existing journal's carried-hard-cull count; no historical report is edited.

The 18:30 scan selected ranks 7, 6, 4 and 5 after the bootstrap view because
earlier ranks failed or were already excluded by the bridge. It achieved a
55.43-degree azimuth span versus 70.0 degrees in scan_20260905_233726.
All five achieved poses from that older scan still have support under the
five-degree lookup replay. This does not prove that every requested ray or
path from the old run remains feasible. No controlled causal attribution of
the span difference is available; latest request diagnostics record 21 IK-failed
MotionGen calls and 14 movement-cone-rejected paths out of 40 calls.

The generic bridge now submits private snapshots to a single background writer.
The existing RayMissionDiagnosticsStore still merges under its cross-process
lock and atomically writes the same JSON and HTML. Other diagnostic producers
and offline callers remain synchronous. No events are coalesced or hidden.
The FIFO has 64 entries; saturation applies backpressure rather than unbounded
memory growth. Disk/render failures are logged and later records continue.
Bridge destruction drains for up to two seconds, within normal supervisor
shutdown allowance. A timeout explicitly warns that the report may be incomplete;
hard termination or disk failure cannot guarantee delivery. The daemon writer
cannot prevent process exit. A live report can briefly lag planning.

A copied 7.98 MB historical journal took a median 0.486 seconds per synchronous
record (five trials). This work is removed from the normal bridge callback,
not eliminated from CPU consumption. Background Python serialization can still
contend for CPU/GIL, and other producers can contend for the journal lock.

The historical eight-request run spent 89.048 seconds in VIEW_PLANNING,
60.814 seconds inside instrumented bridge requests and 59.599 in the worker,
including 50.004 in MotionGen. The remaining 28.234 seconds are not all
coordinator CPU or removable overhead. World updates were only 0.029 seconds.
That run used the former framed policy; it is not a live cone-policy benchmark.

No planner retry budget, NBV scoring, collision model, freshness check, execution
limit, ROS interface or real-motion default changed. Live end-to-end improvement
requires a new matched test; the replay does not promise a particular capture time.

## Validation

Completed after GUI linkage: 1,377 tests passed, 15 skipped across complete mobile, Tesseract,
GUI and ordinary cuRobo suites. Both affected ROS packages built successfully.
The production override loader retained all 48 recorded poses. Five queue
submissions on the copied final-generation snapshot had median 0.033 ms
callback cost, and the merged report retained exactly the same 44 events.
This measures enqueue only, not all diagnostic preparation or mission speed.
The skips include opt-in GPU integration tests; no GPU or physical retest ran.

Ordinary tests cover override provenance, invalid policy rejection, unchanged
legacy loading, queued snapshot ownership and order, exact journal equivalence,
failure isolation and bounded shutdown. Full mobile/Tesseract/GUI suites and
package builds are run without starting ROS or robot processes. Physical testing
is separate. Offline audit scripts and original detailed results are under
`/home/prl/Downloads/PiPER_Direction_Timing_Audit_20260906_EDdigH`.
