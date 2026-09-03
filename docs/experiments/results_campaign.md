# PiPER backend-blocked results campaign

This facility records and analyses successive physical missions without changing
the production scan pipeline. The recorder is a low-priority, file-only sidecar:
it imports no ROS or planner runtime, publishes nothing, and has no camera, CAN,
driver, mission, safety, authorization, or execution interface.

## Declared design

- Target: 35 × 35 × 35 mm cube; six-face surface reference.
- Target X: 0.30, 0.50, 0.70, 0.90 and 1.10 m from `base_link`.
- Target Y: 0.00 m.
- Target Z: 0.00, 0.12 and 0.30 m.
- Backends: one Tesseract and one cuRobo mission per position.
- Total: 15 matched positions and 30 missions.
- Order: all 15 cuRobo missions first, followed by all 15 Tesseract missions
  in the same position order.
- Evidence class: `MATCHED_RUNS`.
- Repetition limitation: N=1 for each backend/location cell. This supports a
  matched engineering comparison but not per-cell confidence intervals.
- Order limitation: elapsed time, lighting, target placement drift and other
  environmental drift are confounded with planner backend. Do not describe
  the backend comparison as controlled or paired physical evidence.
- 30 cm is a physically confirmed position. 50 cm and the other planned
  distances are not pre-labelled as confirmed.
- 120 seconds is a visual runtime reference, not a qualified requirement.

The generated `campaign.json` freezes this design. Reusing a campaign ID with a
different design fails instead of silently mixing evidence.

## GUI workflow

1. Open the **Results Campaign** tab.
2. Keep the default campaign ID or enter a new safe ID, then select
   **Create / Resume**.
3. Select **Prepare Next Trial**. This fills the existing Automatic Scan XYZ and
   planner widgets but does not persist the planner, submit a mission, or command
   anything.
4. Review the cube position. In **Automatic Scan**, use the existing
   **Apply for Next Mission** planner control, then the normal mission button.
5. The GUI records the task ID only after the existing action client reports a
   successful local submission. A recorder error is shown as a warning and does
   not block or change the mission.
6. Repeat. **Create / Resume** and **Prepare Next Trial** continue from the first
   scheduled trial without a matching terminal record.
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
  --campaign piper-poster-blocked-20260902
```

Refresh stored evidence once:

```bash
python3 tools/record_results_campaign.py \
  --project-root /home/prl/Piper_arm \
  --campaign piper-poster-blocked-20260902 \
  --once
```

Generate results without rerunning reconstruction:

```bash
python3 tools/build_results_campaign_report.py \
  --project-root /home/prl/Piper_arm \
  --campaign piper-poster-blocked-20260902
```

Run all five reconstruction modes on each matching immutable capture set and
write every derived mesh beneath the new report directory:

```bash
python3 tools/build_results_campaign_report.py \
  --project-root /home/prl/Piper_arm \
  --campaign piper-poster-blocked-20260902 \
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
- `campaign_bookmarks.csv`: the same one-row-per-attempt index for immediate
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
- The original empty alternating campaign, `piper-poster-20260902`, is retained
  unchanged for provenance. Use `piper-poster-blocked-20260902` for this run.
