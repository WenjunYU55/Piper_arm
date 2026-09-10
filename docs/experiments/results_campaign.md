# PiPER results campaigns

This facility records and analyses successive physical missions without changing
the production scan pipeline. The recorder is a low-priority, file-only sidecar:
it imports no ROS or planner runtime, publishes nothing, and has no camera, CAN,
driver, mission, safety, authorization, or execution interface.

## Default planner-comparison design

- Target: 35 × 35 × 35 mm cube; six-face surface reference.
- Target X: 0.30, 0.50, 0.70, 0.90 and 1.10 m from `base_link`.
- Target Y: 0.00 m.
- Target Z: 0.00, 0.12 and 0.30 m.
- Backends: one Tesseract and one cuRobo mission per position.
- Total: 15 matched positions and 30 missions.
- Order: run both planners at each position, starting cuRobo then Tesseract at
  X=0.30 m, reversing to Tesseract then cuRobo at X=0.50 m, and repeating.
  Sweep X at each height; alternating pair parity continues across height boundaries.
- Evidence class: `MATCHED_RUNS`.
- Repetition limitation: N=1 for each backend/location cell. This supports a
  matched engineering comparison but not per-cell confidence intervals.
- Order limitation: alternating the first planner reduces systematic order bias
  but does not eliminate lighting, placement or temporal drift. N=1 still limits
  the comparison. Existing backend-blocked campaigns retain their original confounders.
- 30 cm is a physically confirmed position. 50 cm and the other planned
  distances are not pre-labelled as confirmed.
- 120 seconds is a visual runtime reference, not a qualified requirement.

The generated `campaign.json` freezes this design. Reusing a campaign ID with a
different mode resumes the existing frozen design; select a new ID to create a different design. Invalid edited definitions fail validation.

## GUI workflow

1. Open the **Results Campaign** tab.
2. Select a campaign mode. Keep its suggested campaign ID or enter a new safe ID, then select
   **Create / Resume**.
3. Select **Prepare Next Trial**. This fills the existing Automatic Scan XYZ and
   planner widgets, plus target label, camera profile and aim tolerance for sweeps.
   It does not persist settings, submit a mission or command anything.
4. Review the cube position. In **Automatic Scan**, use the existing
   **Apply for Next Mission** planner control, apply the prepared camera profile
   and aim tolerance, then use the normal mission button.
5. The GUI records the task ID only after the existing action client reports a
   successful local submission. A recorder error is shown as a warning and does
   not block or change the mission.
6. Repeat. **Create / Resume** and **Prepare Next Trial** continue from the first
   scheduled trial without a matching terminal record or explicit skip. An explicitly prepared retry takes priority until completed or skipped.
7. Select **Generate Excel / CSV / Figures** when desired. Reconstruction replay
   is optional and runs offline on stored captures.

An operator may run a mission outside the loaded schedule. It is preserved but
marked as excluded rather than forced into a planned cell.

## Command-line workflow

Start or resume the file-only recorder:

```bash
cd /home/prl/Piper_arm
python3 tools/record_results_campaign.py \
  --project-root /home/prl/Piper_arm \
  --campaign piper-paired-20cm-20260908
```

Refresh stored evidence once:

```bash
python3 tools/record_results_campaign.py \
  --project-root /home/prl/Piper_arm \
  --campaign piper-paired-20cm-20260908 \
  --once
```

Generate results without rerunning reconstruction:

```bash
python3 tools/build_results_campaign_report.py \
  --project-root /home/prl/Piper_arm \
  --campaign piper-paired-20cm-20260908
```

Run all five reconstruction modes on each matching immutable capture set and
write every derived mesh beneath the new report directory:

```bash
python3 tools/build_results_campaign_report.py \
  --project-root /home/prl/Piper_arm \
  --campaign piper-paired-20cm-20260908 \
  --run-reconstruction
```

## Sources and outputs

The collector reads existing mission result JSON, schema-2 capture manifests and
frame YAML, heavy-perception result/SAM2 YAML, RayProcesses JSON, and existing
reconstruction quality JSON. Small normalized evidence snapshots are stored at:

```text
datasets/experiment_campaigns/<campaign>/trials/<trial>/attempts/<task>/
```

The campaign root also contains two continuously refreshed bookmark ledgers:

- `campaign_bookmarks.json`: the canonical structured index;
- `campaign_bookmarks.csv`: the same attempt and explicit-skip index for immediate
  inspection in Excel.

Each bookmark records the declared trial, task ID, planner, target coordinates,
submission/terminal/evidence state, outcome, safe shutdown, capture count,
dataset and ray-diagnostic identity, first-acquisition identity, source-manifest
hash, Git branch/commit/dirty state, collision-model qualification, and whether
the attempt is included or excluded. The ledger is atomically regenerated
after submission, terminal result, submission failure, and evidence collection.
It therefore remains useful even before a full report is generated.

Reports are written to a new directory beneath
`datasets/experiment_results/`. Existing directories are never overwritten; a
timestamp suffix is added. Each report contains:

- `PiPER_results_campaign.xlsx`;
- one CSV per workbook sheet;
- `campaign_bookmarks.json` and `campaign_bookmarks.csv` copied into the report;
- `evidence_manifest.csv`, which records every source/configuration file and
  SHA-256 used for each included or excluded mission;
- acquisition, coverage, mission-result, runtime and reconstruction figures as
  300 dpi PNG and vector PDF;
- `report_manifest.json`, containing SHA-256 and size for every generated
  artifact, and a local `README.md`.

The workbook records submission-to-capture timing, each capture transaction
duration, phase totals, planner candidate timing, exact configuration/model
hashes, first Grounding DINO confidence, first SAM2 score, depth/support quality,
system-native measured 10 mm voxel coverage, offline six-face cube coverage,
azimuth/elevation spans, qualified point-cloud dimensions, and reconstruction
quality.

Planner-model qualification is copied from the hash-bound configuration at
submission. The current cuRobo model records operator-reported physical
qualification on 2026-09-02 for supervised 5% target-scan missions. Results
must retain the exact tabletop and 5% free/contact speed settings, that scope,
and the non-conservative sphere-model limitation; this status is not evidence
of Tesseract-equivalent collision geometry.

## Interpretation limits

- Grounding DINO confidence and SAM2 score are not detection accuracy.
- Depth spread is stability/contamination evidence, not metric sensor accuracy.
- Six-face coverage samples a known 35 mm, axis-aligned cube at 1 mm and counts
  samples within 2 mm of persisted qualified depth points. Report this model and
  its assumption with the result.
- Point-cloud extents use a 1st-to-99th percentile estimator and are not
  independent metrology.
- Reconstruction methods are `CONTROLLED_REPLAY` only when they process the same
  accepted capture set. A lower registration residual never overrides a failed
  known-dimension quality gate.
- Failed and cancelled missions remain results; they are not silently removed.
  Schedule mismatches are listed in `exclusions.csv`.
- Existing blocked campaigns (including `piper-poster-blocked-20260902`) and
  the original Tesseract-first alternating design remain load-compatible and
  retain their saved order and attempts. Use `piper-paired-20cm-20260908` for the new order.

## Order update validation (8 September 2026)

The new campaign contains 30 adjacent paired trials at the unchanged 15 positions.
All 166 campaign and GUI tests pass, including report CSV/XLSX/figure generation
and historical schedule resumption without definition rewrites. Python compile,
AI YAML parsing, schedule CSV validation and diff whitespace checks pass.
The new campaign is `piper-paired-20cm-20260908`; no mission was started.

### Skipping a scheduled trial

Use **Skip Trial** in the Results Campaign tab to skip the current pending trial
and prepare the next entry. Each click skips one entry; check the prepared planner
because alternating order can place two Tesseract entries consecutively. Review
and apply the next planner using the normal controls before starting a mission.
The button is unavailable during an active mission, and unresolved recorded
submissions also block skipping after a restart.

The choice persists in `trials/<trial_id>/skip.json` with the immutable expected
trial, timestamp and operator reason. Campaign bookmarks show `SKIPPED`. Reports
include `skipped_trials.csv`, a `Skipped_Trials` workbook sheet and hashed copies
of skip receipts. Skips are separate from completed scans and cannot supply a
measured outcome or matched planner comparison. No capture files are deleted.

## Additional cuRobo-only modes

| Mode | Conditions and order | Position, metres in base_link | Trials |
| --- | --- | --- | --- |
| Camera resolution / FPS | 640×480, 960×540, 1280×720, 1920×1080; each at 15 then 30 FPS | (0.30, 0, 0.30) | 8 |
| Camera aim tolerance | At each X: 1°, 5°, 10°, 15°, 20°, 25° | X 0.30/0.50/0.70/0.90/1.10, Y 0, Z 0.30 | 30 |
| Objects | Green cube benchmark, cup, stick, flower | (0.30, 0, 0) | 4 |

All three use cuRobo. Camera and object trials use 5° aim; only the tolerance
sweep varies it. The existing first-capture min(selected, 5°) gate remains.
The two non-camera sweeps freeze the saved RGB profile at campaign creation.
Depth profile and other launch settings are not varied by this feature. Z uses
the existing base_link convention; no additional table-height transform is inferred.

Each new mode uses a distinct suggested ID and schema-2 `campaign.json`.
Saved conditions are immutable on resume. `Scheduled_Trials` in the workbook and
`scheduled_trials.csv` retain the full schedule. Each normalized row carries the
condition, while separate charts retain each trial/attempt instead of collapsing
the sweep into the old two-planner spatial grid. Evidence is labelled
`PHYSICAL_PARAMETER_SWEEP`; one attempt per condition does not establish variance.

The passive submission record reads and hashes the saved RGB and executor files.
A mismatched label, camera profile or aim tolerance is retained but excluded from
the intended condition. Saved profile FPS is requested FPS, not measured throughput.
Actual images and runtime logs remain the evidence for achieved camera behavior.

Only the nominal cube has a declared reference. Cup, stick and flower retain
observed extents and capture/timing evidence, with reference dimensional errors
and cube-surface coverage unavailable. Optional reconstruction omits expected
cube dimensions for those objects. Existing source quality reports remain source
reports and are not silently revalidated as object-specific accuracy.

## Retry the last attempt

Use **Retry Last Attempt** while no mission is active. Success, failure, cancellation and submission failure are eligible. It prepares the same scheduled target, backend and camera/object/aim conditions; review the controls, use Apply for Next Mission where needed, then press the normal Start Mission button. The button never starts or moves the robot.

The selected retry persists in `active_trial.json` with `retry_of_task_id`. A mismatched previous attempt is matched by its actual submitted conditions, so retry does not silently adopt the next slot's position. If those conditions match no scheduled trial, preparation reports the problem. Unresolved submissions block retry preparation.

Each new attempt has a new task ID and preserves prior submission, terminal and evidence files. Once a matching retry reaches a terminal outcome, reports use it for that trial and list earlier matching attempts as superseded exclusions; the bookmarks retain the complete history and retry linkage. A failed retry is still the latest result, rather than selecting only successes. A mismatched or unfinished retry does not replace the previous measured result.

Prepare Next Trial and reopening the campaign retain a pending retry. Skip Trial on a pending retry cancels only that retry preparation, records a `retry_skips` receipt and resumes the normal schedule without marking the already completed trial skipped.

## Campaign trial table

Create / Resume displays all scheduled trials with planner, XYZ, sweep conditions, status, latest matching outcome and attempt count. **Done** means a matching mission ended, including a failed mission; the outcome column distinguishes success and failure. Mismatched submissions do not complete a trial.

Select a previously attempted row and click **Retry Selected Trial** to prepare its most recent matching attempt, regardless of outcome. Review and apply settings, then use the normal Start Mission control. Previous evidence is retained and a completed matching retry replaces earlier matching attempts in report summaries. Pending retries survive reopening; normal remaining trials resume afterwards. **Retry Last Attempt** remains available. Unattempted entries use **Prepare Next Trial**. Active or unresolved missions block retry preparation. **Refresh Table** reloads recorded state; campaign status events also refresh the table.

Execution-policy provenance (8 September 2026): configuration snapshots now include
`execution/clearance.py`, `execution/trajectory.py` and the Tesseract worker alongside
the execution YAML. Runs expose available abbreviated hashes in
`execution_policy_sources`, so timing comparisons can identify changed execution
policies. Historical rows without these snapshots leave the field empty; elapsed
mission durations are never rewritten to compensate for tracking pauses.

### Recovery of a missed GUI completion callback (2026-09-09)

The passive collector now reconciles a missing campaign terminal.json from the matching finalized mission result. Previously only the GUI result callback wrote that record: the observer could repeatedly collect a failed mission result while its trial remained In progress. The latest Tesseract retry bb4f5971ec2c49adb449f7800b7158e9 exhibited this inconsistency after its 20-minute deadline, seven captures and safe shutdown. The current GUI started after mission completion; the exact loss mechanism of the original callback is not retained.

Recovery requires exact task identity, a terminal outcome, explicit shutdown status and a valid capture count. It preserves the source result, records its path/hash/modification time, leaves existing terminal records untouched and uses normal campaign completion rules for retries, active-trial clearing and bookmark updates. Collection never commands the arm or changes a mission result.

Validation: 188 campaign and GUI tests passed. The actual saved retry was recovered through collect_task; its row is Done / FAILED, capture count seven, active retry cleared, and original mission result bytes unchanged. Refresh Table displays the repaired durable row; restarting the GUI loads the new collector into its managed sidecar.
