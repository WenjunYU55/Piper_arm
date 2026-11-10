# NBV scoring, exhausted-ray retries and mission completion

This page describes how the next-best-view (NBV) shortlist is ranked, how exhausted ray evidence persists, and what establishes scan success. Scoring metadata does not authorize motion: exact planning, endpoint/path checks, operator authorization, capture validation and terminal recovery remain in force. The maintained source owners are [coverage scoring](../../piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/planning/coverage.py), [bridge retry handling](../../piper_ros_foxy/src/piper_tesseract_foxy/piper_tesseract_foxy/bridge_node.py), and [mission completion](../../piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/mission/engine.py).

## Scoring position and rank

Coverage scoring records `nbv_scoring_standoff_m`, `nbv_scoring_position_m` and `nbv_scoring_source`. When an enforcing capability atlas supplies a nearest supported interval, that interval supplies the scoring position; without an atlas, the ray minimum is used. This changes the projection point used to estimate information, not desired camera motion, ray identity/direction/population, or endpoint search bounds. The 64×48 projection and 0.02 minimum normalized gain remain. Normalized information remains primary; angular novelty, then absolute gain/travel, break ties. Bootstrap seed policy is unchanged. Ranking cache identity includes scoring standoff; the static atlas query cache remains exact-input and bounded. Only the six best surviving information-ranked rays are sent to exact planning.

Approximate atlas false negatives remain possible. Offline re-ranking does not prove a ray reachable, and equal scores can be valid for unknown or unobserved geometry. Scoring parameters belong to coverage/prequalification; atlas membership/enforcement is not changed by scoring metadata. See [scoring and retry validation](../experiments/nbv_scoring_retry_validation.md).

## Exhaustion correlation and lifetime

Exhausted ray IDs are retained for the lifetime of the same session and ray population, across robot movement and accepted captures. They reset only for a new session or explicit bootstrap-to-qualified population replacement. This is an intentional runtime-saving policy: a ray excluded after planner alternatives fail may become feasible from a later starting pose; the exclusion is not a claim of permanent geometric infeasibility. Correlation is strict: only IDs in the exact pending request and matching generation/population can be recorded. Infrastructure failures do not retire rays. Ray Review retains the same population-lifetime evidence.

For successful Tesseract responses that omit explicit `exhausted_ray_ids`, typed `AIM_VARIANTS_EXHAUSTED` candidates are mapped through the exact pending request. A ray is retired only if every requested alternative failed and none was selected. Successful, unattempted, shared/selected and foreign IDs are not retired. cuRobo's explicit exhaustion handling remains. Generic failures, worker geometry/source binding, NBV rank, collision and authorization gates are unchanged. Permanent typed endpoint and capability culls, visual rejection quarantine, and accepted-direction retirement remain separate mechanisms.

## Completion and operational bounds

Positive/negative Y achieved-view counts are diagnostics; default completion does not require two views on each side. The current successful completion predicate is explicit full-population `RAY_FRONTIER_EXHAUSTED` followed by the existing safe terminal recovery. Capture count, 25-degree elevation span, measured surface minimum, convergence and achieved-history counts are reported but advisory. Passing those coverage measures does not stop a scan while rays remain. A shortlist exhaustion is retried; it is not proof that the full ray population is exhausted.

Do not infer full-ray exhaustion from the broader legacy `SAFE_VIEW_EXHAUSTED` tag, planner timeout, no-positive-information alone, generic IK failure, invalid obstacle geometry, tracking/capture rejection or cancellation. Unexpected executor `COMPLETE` without full-ray proof cannot report success. Existing limits remain: maximum 24 captures, mission deadlines and bounded retries. At the capture ceiling, the system performs a command-free next-plan check: full ray exhaustion can succeed, but a remaining plan fails `MISSION_FAILED` before approval. No extra capture is authorized beyond the bound.

Success still requires the existing safe terminal recovery: home/hold, motor disable, process cleanup, collision/joint-limit checks and capture-validity checks. `SUCCEEDED` means the available ray population was exhausted and shutdown completed safely; it does not mean surface coverage or reconstruction passed. Zero captures may satisfy that mission boundary. Empty data is discarded, and reconstruction rejects missing input.

The September 11 offline retrospective reclassified 47 prior failed missions with explicit `RAY_FRONTIER_EXHAUSTED` and `safe_shutdown: true` across 13 campaigns; original terminal/evidence files were retained. Invalid-obstacle and joint-limit incidents remain failures. This was offline regression/campaign evidence only: the physical check is pending, and running mission coordinators need restart to load the policy. No live restart or robot command was performed. See the [NBV validation record](../experiments/nbv_scoring_retry_validation.md).

## Test entry points and qualification

Source-level owners for supporting policy are [coverage scoring](../../piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/planning/coverage.py), [session coverage memory](../../piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/scan_session_memory.py), [failure classification](../../piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/infrastructure/failure_model.py), and the [bridge](../../piper_ros_foxy/src/piper_tesseract_foxy/piper_tesseract_foxy/bridge_node.py). Relevant focused suites include:

```bash
source piper_ros_foxy/install/setup.bash
python3 -m pytest -q piper_ros_foxy/src/piper_mobile_manipulation/test/test_nbv_coverage.py
python3 -m pytest -q piper_ros_foxy/src/piper_tesseract_foxy/test/test_bridge_feedback_freshness.py
python3 -m pytest -q piper_ros_foxy/src/piper_mobile_manipulation/test
```

Changes to exhaustion classification or completion require the full mobile, Tesseract and GUI campaign regressions, the five-package build, capture-ceiling and terminal-recovery cases, plus supervised physical qualification. Software validation does not establish physical mission behavior.
