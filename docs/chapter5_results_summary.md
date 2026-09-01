# Chapter 5 method comparison and integration ablation

## Scope and evidence protocol

This report evaluates alternatives that were actually implemented or tested in
the PiPER target-centric RGB-D acquisition project. It does not introduce
synthetic competitors and does not use detector confidence as a substitute for
accuracy. The numerical source tables are in
[`chapter5_method_comparison.xlsx`](chapter5_method_comparison.xlsx), and the
machine-readable analysis plus source provenance is in
[`chapter5_method_comparison.json`](chapter5_method_comparison.json).

The evidence was updated on 1 September 2026 from the `curobo-integration`
worktree at commit `36d12f5`, with the analysis state explicitly marked dirty
because this benchmark and its diagnostic timing changes are the work under
evaluation. No robot, CAN interface, camera, ROS graph, or motor process was
started. The new computation was command-free replay of saved RGB-D artifacts,
ray diagnostics, capability-map benchmarks, offline reconstruction, and frozen
motion-planning requests through isolated Tesseract and cuRobo workers.

Every comparison is assigned one of the following strengths:

| Strength | Meaning in this report |
| --- | --- |
| `CONTROLLED_REPLAY` | The same stored inputs or fixed validation set were processed by both configurations. |
| `PAIRED_PHYSICAL` | Two deliberately paired physical trials under matched conditions. No such result was found in the current archive. |
| `MATCHED_RUNS` | Closely matched physical runs, but not randomized and not fully controlled. |
| `HISTORICAL_OBSERVATIONAL` | Development-run evidence with material confounding factors. |
| `EXPLORATORY` | Useful feasibility evidence that cannot support a final performance ranking. |

The project currently contains 154 accepted observations in 45 non-empty scan
directories. Of these, 152 are confidence-qualified schema-2 captures. The
archive also contains a 63-image prompt replay, a seven-image offline
resegmentation replay, seven full-sphere capability-filter traces containing
2,520 initial rays, a fixed 2,520-ray capability-map convergence benchmark, and
a 2,004-state exploratory cuRobo/Tesseract collision-model comparison.

## 1. Semantic target acquisition

The June historical system used a calibrated green HSV detector. The later
system implemented Grounding DINO for open-vocabulary acquisition followed by
SAM2 refinement. No matched HSV-versus-Grounding-DINO image benchmark survives,
so accuracy, precision, recall, and false-positive-rate comparisons between
those two methods would be unsupported.

A genuine controlled comparison does survive for Grounding DINO prompting. The
same 63 confirmed-positive physical images were replayed using the former mixed
target/obstacle prompt and the final target-only prompt `green cube .`.

| Configuration | N | Mean confidence | Minimum confidence | Result |
| --- | ---: | ---: | ---: | --- |
| Former mixed prompt | 63 | 0.8435 | 0.7947 | Confirmed detections recorded |
| Target-only prompt | 63 | 0.8751 | 0.8342 | 63/63 accepted |

The mean confidence increased by 0.0316 (3.75% relative), and the minimum
increased by 0.0395. This is evidence that the target-only prompt provided the
more robust usable semantic hypothesis of the tested prompt configurations. It
is not an accuracy comparison because the replay set contains confirmed
positive images. The separately recorded absent scene scored 0.5627 and failed
the green-appearance gate, but one negative example cannot establish a
false-positive rate.

**Selection:** the target-only Grounding DINO prompt was retained because it
separates target semantics from obstacle vocabulary and produced a stronger
downstream acquisition hypothesis on the controlled positive replay.

## 2. Target mask strategy

The implemented mask history includes live SAM2 masks, temporally propagated
SAM2 masks, event-triggered Grounding-DINO/SAM2 refresh, and an optional offline
fresh image-mode Grounding-DINO/SAM2 resegmentation path for reconstruction. No
manual pixel ground truth exists, so IoU between two automatic masks measures
agreement, not segmentation accuracy.

On the same seven RGB frames from `scan_20260821_220003`, fresh offline masks
had a mean IoU of 0.882 and a median IoU of 0.949 against the captured live
masks. The fresh result retained 57,219 of 65,570 captured-mask pixels after
intersection with captured qualified support, removing 12.7%. View 004 was the
clear outlier: IoU was 0.627 and 4,983 captured pixels associated with the
shadow/table region were removed.

That narrowing improved the selected reconstruction OBB from
58.79 × 52.88 × 48.29 mm to 51.48 × 49.00 × 45.77 mm and reduced median
point-to-mesh residual from 0.558 to 0.530 mm. However, dominant-component
coherence fell from 90.5% to 86.65%, and both outputs remained `FAIL` against
the measured 35 mm cube.

**Selection:** the captured live mask remains the online authority because it
is synchronized with acquisition and tracking. Fresh offline resegmentation is
retained as an optional diagnostic reconstruction configuration, not promoted
as universally superior.

## 3. Depth and 3D target localisation

The strongest controlled ablation uses the exact same stored aligned depth for
three implemented support strategies:

1. rectangular detector ROI;
2. semantic SAM mask;
3. confidence- and depth-component-qualified target support.

For the 152 captures where all three forms are available:

| Support strategy | N | Median spatial depth standard deviation | Interpretation |
| --- | ---: | ---: | --- |
| Rectangular ROI | 152 | 65.27 mm | Includes target plus surrounding/background depth |
| Raw SAM mask | 152 | 30.50 mm | Removes much rectangular-ROI contamination |
| Qualified target component | 152 | 7.03 mm | Retains one confidence-qualified depth layer |

Across all 154 ROI/mask observations, the corresponding medians were 57.84 mm
and 30.20 mm. A median 25.45% of valid ROI depth pixels lay outside the SAM
mask. In schema 2, 111/152 observations (73.0%) contained more than one
candidate depth layer. The final qualification retained a median 53.15% of raw
mask-valid depth pixels.

These results demonstrate improved geometric stability and reduced layer
mixing; they do not demonstrate metric accuracy because no per-pixel depth
ground truth exists. Nevertheless, they directly explain the retained
integration contract: a missing or stale semantic mask produces no target
measurement, and raw ROI/background depth cannot correct the tracker.

**Selection:** segmentation-mask support plus confidence and depth-component
qualification was the best-performing tested configuration for supplying
stable target geometry downstream.

## 4. Temporal target estimation

The repository genuinely contains both historical raw/loosely filtered target
updates and the final near-static Kalman estimator with Mahalanobis innovation
gating, bounded prediction-only operation, and a five-second loss expiry. Unit
and fault-replay tests prove that corrupted updates are rejected and short
measurement outages do not immediately erase the track.

However, the local datasets retain accepted capture states rather than the
continuous synchronized raw Target3D and filtered TrackedTarget time series.
Therefore frame-to-frame jitter, RMSE, MAE, recovery time, and raw-versus-filtered
physical performance cannot be calculated scientifically from the current
archive. No `Tracking_Comparison` worksheet was created for that reason.

**Selection status:** the filtered and gated tracker is retained by its safety
and continuity contract, but a quantitative physical superiority claim remains
pending a recorded stationary and moving-target trace.

## 5. View generation and scan strategy

The implemented progression is documented in Git and historical records:

- a dry-run fixed arc with an HSV target;
- a 21-candidate, three-elevation dome with a diverse 13-view subset and
  nearest-neighbour route;
- exact-point `voxel_nbv` with cumulative accepted-coverage scoring;
- frozen full-sphere `ray_nbv` with bounded standoff, accepted-only coverage,
  size-aware target envelope, persistent ray culls, and capability-map
  prequalification.

The fixed-arc and July dome RGB-D datasets needed for a same-input replay are
not present, so they are historical baselines rather than controlled numerical
comparators.

### Closest matched voxel/ray runs

The closest stored comparison is `scan_20260820_230845` (voxel NBV) versus
`scan_20260821_220003` (ray NBV). The target and general setup are closely
matched, but the code changed between dates; this is therefore `MATCHED_RUNS`.

| Metric | Voxel NBV | Ray NBV |
| --- | ---: | ---: |
| Accepted captures | 6 | 7 |
| Measured 10 mm target voxels | 111 | 164 |
| Mean post-seed novel fraction | 0.261 | 0.392 |
| Azimuth span | 60.0° | 72.6° |
| Elevation span | 30.0° | 36.3° |
| Adjacent transitions below 15° | 4/5 | 2/6 |
| Capture-time span | 412.5 s | 292.0 s |

In this matched comparison, ray NBV accumulated 47.7% more measured voxels,
50.4% greater mean post-seed novel fraction, and fewer near-duplicate adjacent
views.

### Historical aggregate check

The complete historical archive prevents cherry-picking. For scans containing
at least three captures, only three voxel-NBV scans (12 captures) exist versus
21 ray-NBV scans (115 captures). Median covered voxels were 111 for voxel and
106 for ray; median post-seed novel fraction was 0.411 and 0.400 respectively.
Ray runs had a wider median azimuth span (32.5° versus 15.0°) and a lower median
adjacent redundancy rate (0.111 versus 0.500).

This aggregate does **not** prove a general coverage-efficiency advantage for
ray NBV. It shows that the retained ray implementation improved angular
diversity and reduced immediate repetition, while coverage per scan remained
strongly affected by target pose, capture count, feasibility, and the evolving
integration.

**Selection:** ray NBV was retained because it is the most suitable of the
implemented alternatives for full-sphere, variable-standoff, size-aware,
capability-filtered planning. The evidence does not justify calling it globally
best. None of the 45 stored accepted-capture datasets satisfied the current
three-view measured-convergence calculation during this offline replay, so a
current complete-scan stopping proof remains pending.

## 6. Viewpoint feasibility filtering

### Capability-map configuration

The capability map was evaluated on the same 87 achieved poses and 2,520
validation rays at five sampling levels:

| Joint samples | Final-supported-ray recall | Known-pose recall | Median query |
| ---: | ---: | ---: | ---: |
| 100,000 | 60.17% | 91.95% | 0.235 ms |
| 250,000 | 70.35% | 98.85% | 0.249 ms |
| 500,000 | 77.62% | 98.85% | 0.258 ms |
| 1,000,000 | 90.41% | 100.00% | 0.268 ms |
| 2,000,000 | 100.00% | 100.00% | 0.280 ms |

The two-million-sample atlas was therefore the best-performing tested atlas
configuration under the fixed replay. Its “100%” is agreement with the densest
sampled reference, not proof of exact reachability.

### Candidate-load ablation

Seven stored initial full-sphere populations generated 2,520 rays. The analytic
workspace gate rejected 33, the capability map rejected 2,224, and 263 survived
cheap prequalification. Compared with workspace-only filtering, the map reduced
the rays passed toward expensive planning from 2,487 to 263 (89.4%).

Among ray records that later carried explicit Tesseract outcomes, 25 were
selected and 161 were culled. Those repeated generation-level records are not
an independent plan-success-rate sample, but they demonstrate why coarse
prequalification cannot replace exact IK, collision, path, and visibility
validation.

**Selection:** the 2M capability atlas is retained only as a rapid coarse gate;
Tesseract remains the exact feasibility authority.

## 7. Motion-planning method and configuration

Tesseract is the final physically validated backend. A separate
`curobo-integration` branch implemented cuRobo 0.7.8 and replayed 2,004 sampled
joint states against the exact Tesseract collision model. The cuRobo sphere
approximation produced zero state-level false negatives in that sample and 18
conservative false positives. It remains `hardware_qualified=false`, has
sampled-surface gaps up to 48.3 mm, and has no matched physical planning trial.

A new `CONTROLLED_REPLAY` benchmark then sent the same five recorded positive
planning scenarios—rough acquisition plus four multiview transitions—through
both ROS-free workers three times after one warm-up. It also included three
blocked negative controls and a return-home policy control. Return home is not
counted as a planner-algorithm comparison because the production Tesseract path
uses the deliberately direct configured-home policy.

| Offline metric | Tesseract 0.35.0.6 | cuRobo 0.7.8 |
| --- | ---: | ---: |
| Recorded positive requests solved | 15/15 | 15/15 |
| Blocked controls rejected | 3/3 | 3/3 |
| Successful paths passing exact Tesseract revalidation | 15/15 | 15/15 |
| Median end-to-end planner request wall time | 19.646 s | 0.630 s |
| Median scheduled trajectory duration | 4.750 s | 18.209 s |
| Median joint-space path length | 2.803 rad | 3.790 rad |
| Median planner-wall + scheduled-trajectory proxy | 25.046 s | 25.048 s |

cuRobo was therefore much faster at producing proposals on the RTX 3090, but
several of its proposals were materially longer. The median combined offline
proxy was effectively unchanged. This is not physical mission time: controller
following, settling, perception, capture, startup and shutdown are absent.

**Selection:** Tesseract was retained because it has the strongest physically
validated integration with this system. The evidence does not show that
Tesseract is algorithmically faster or globally superior to cuRobo, while the
offline result is not authority to execute cuRobo paths on hardware.

## 8. Capture quality gating

The saved schema-2 evidence supports an ablation of mask and target-depth
qualification inside already accepted captures. It shows that confidence and
component selection reduced median within-support depth spread from 30.50 mm
to 7.03 mm while retaining a median 53.15% of mask-valid pixels.

The archive cannot measure how many invalid captures would have entered without
the final admission gates because rejected observations are intentionally not
persisted. Consequently, precision, recall, rejected-frame counts, and
reconstruction quality “without all gates” are not reported. A future
diagnostic-only rejected-frame archive would be required for that experiment.

**Selection:** the final gate chain was retained because it prevents mixed or
uncorrelated geometry from entering the authoritative dataset, not because a
false-admission rate has already been measured.

## 9. Reconstruction comparison

A fresh controlled replay ran all five implemented registration modes on the
same seven immutable captures from `scan_20260821_220003`, using the same
captured masks, 1 mm voxel, 35 × 35 × 35 mm reference, calibration, and quality
criteria.

| Registration mode | Mean dimensional error | Median residual | P90 residual | Quality |
| --- | ---: | ---: | ---: | --- |
| Robot pose | 16.59 mm | 2.486 mm | 9.883 mm | FAIL |
| Bounded sequential GICP | 16.51 mm | 2.260 mm | 12.122 mm | FAIL |
| Bounded multiway GICP | 16.35 mm | 0.674 mm | 6.690 mm | FAIL |
| Constrained superposition | 16.44 mm | 0.712 mm | 8.044 mm | FAIL |
| Static-scene pose graph | 12.47 mm | 2.524 mm | 10.299 mm | FAIL |

Auto selected bounded multiway GICP because it materially reduced residual
without exceeding the bounded pose-correction contract. The static-scene mode
had the smallest mean dimension error but worse residual evidence, illustrating
why a single metric cannot choose the reconstruction. Most importantly, every
result still failed the measured cube-size quality gate.

**Selection:** bounded auto-selection was retained because it preserves the
robot-pose baseline and accepts a refinement only when the full quality evidence
supports it. No registration method is established as globally best, and the
current archive does not establish successful final reconstruction.

## 10. Integration ablation conclusions

The valid same-input ablations show the following contributions:

| Removed component | Same-input consequence | Strength |
| --- | --- | --- |
| Semantic target mask | Median depth spread rises from 30.20 to 57.84 mm when the rectangular ROI is used | Controlled replay |
| Confidence/depth-layer gate | Median depth spread rises from 7.03 to 30.50 mm | Controlled replay |
| Capability atlas | Initial rays passed after cheap filtering rise from 263 to 2,487 | Historical candidate replay |
| Bounded multiway registration | Median residual rises from 0.674 to 2.486 mm under robot-pose-only fusion | Controlled replay |

The following requested ablations cannot currently be performed without
inventing missing information: raw versus Kalman tracking, labelled mask
accuracy, detector precision/recall, single-frame versus 20-frame temporal
noise, rejected-observation admission, a physical Tesseract-versus-cuRobo trial,
and a complete simple-pipeline-versus-final-pipeline mission replay. They are
listed explicitly in the workbook’s `Unsupported_Comparisons` sheet.

The action result now records additive monotonic phase totals for startup,
preflight, acquisition, planning, capture, return home, hold, disable and
cleanup, together with the frozen planner backend. This does not change the ROS
action schema or mission decisions. It enables future paired missions to report
total time, seconds per accepted capture, captures per minute and phase shares.
All 35 saved historical action results predate that field, so a complete
backend-by-backend mission efficiency number cannot be reconstructed honestly
from the current archive.

## Final method-selection table

| Subsystem | Alternatives Evaluated | Evaluation Metric | Best Performing Tested Approach | Evidence | Reason Retained |
| --- | --- | --- | --- | --- | --- |
| Semantic acquisition | Mixed prompt; target-only prompt | Confirmed-positive availability and confidence | Target-only Grounding DINO prompt | 63/63 accepted; mean 0.8751 vs 0.8435 | Cleaner, stronger semantic handoff |
| Target depth support | ROI; SAM mask; qualified mask depth | Depth spread and layer contamination | Confidence/layer-qualified mask depth | Median spread 65.27 → 30.50 → 7.03 mm | Prevents background layers driving geometry |
| Temporal target estimate | Raw updates; filtered/gated prediction | Fault-replay safety contract | Near-static Kalman with innovation gating | Corrupted updates rejected in tests; empirical trace missing | Bounded continuity and fail-closed loss behavior |
| View planning | Fixed/dome; voxel NBV; ray NBV | Coverage, novel gain, diversity, redundancy | Ray NBV in the selected matched comparison | 164 vs 111 voxels; fewer adjacent duplicates | Full-sphere, bounded, size-aware feasibility integration |
| Feasibility | Workspace only; capability atlas; Tesseract | Recall, query time, candidate reduction | 2M atlas + exact Tesseract | 100% reference recall; 0.280 ms; 263/2520 survive | Cheap cull without replacing exact planning |
| Motion planning | Tesseract; cuRobo | Same-request offline timing, exact revalidation, physical qualification | Tesseract for final hardware system; cuRobo remains offline | 19.646 vs 0.630 s median proposal time, but 25.046 vs 25.048 s combined proxy; cuRobo unqualified | Retain the strongest physically validated integration while preserving the faster offline candidate |
| Reconstruction | Robot pose; GICP; superposition; scene graph | Residual, dimensions, components, quality | Bounded auto-selection for this dataset | Auto selected multiway; all modes still FAIL | Retains baseline and fails closed on poor geometry |

## Why the final integrated methods were retained

### Target-only Grounding DINO plus SAM2

1. **Alternatives tested:** the historical HSV path, a mixed semantic prompt,
   and the final target-only Grounding DINO prompt with SAM2 refinement.
2. **Evidence:** the controlled 63-positive replay improved mean and minimum
   detector confidence, while the SAM mask materially reduced depth mixing.
3. **Best tested configuration:** target-only Grounding DINO followed by SAM2
   was the most suitable implemented semantic-to-pixel pipeline.
4. **Reason retained:** it supports task text while producing pixel support for
   geometric qualification.
5. **Cannot conclude:** no labelled dataset supports detector precision/recall
   or segmentation IoU against ground truth.

### Confidence-qualified target geometry

1. **Alternatives tested:** rectangular ROI depth, raw SAM-mask depth, and
   confidence/layer-qualified mask depth.
2. **Evidence:** median depth spread fell from 65.27 to 30.50 to 7.03 mm on the
   same 152 observations; 73.0% contained multiple depth candidates.
3. **Best tested configuration:** qualified target-mask depth.
4. **Reason retained:** it prevents clean background or shadow depth from
   becoming authoritative target geometry.
5. **Cannot conclude:** stability is not metric accuracy without depth ground
   truth.

### Near-static filtered target state

1. **Alternatives tested:** loose/raw updates and the final prediction/correction
   model with innovation gating.
2. **Evidence:** software fault replays protect missing, stale, and corrupted
   input behavior.
3. **Best tested configuration:** the gated near-static estimator is the safer
   tested integration contract.
4. **Reason retained:** it bridges short observation loss without accepting an
   implausible correction indefinitely.
5. **Cannot conclude:** the missing continuous trace prevents a quantitative
   physical jitter or accuracy comparison.

### Full-sphere ray NBV

1. **Alternatives tested:** fixed arc/dome routes, exact-point voxel NBV, and
   bounded ray NBV.
2. **Evidence:** the closest matched ray run had more measured voxels, wider
   angular span, higher post-seed gain, and fewer adjacent duplicate views;
   the full historical aggregate confirms lower redundancy but not higher
   median coverage.
3. **Best tested configuration:** ray NBV was the most suitable tested
   configuration for the final size-aware, variable-standoff pipeline.
4. **Reason retained:** it integrates full-sphere directional choice with a
   continuous bounded standoff and accepted-only cumulative coverage.
5. **Cannot conclude:** the observational archive does not prove global NBV
   superiority or current completion efficiency.

### Capability atlas plus Tesseract

1. **Alternatives tested:** workspace-only culling, capability-map sample
   levels, exact Tesseract feasibility, and exploratory cuRobo integration.
2. **Evidence:** the 2M atlas reached full reference recall at 0.280 ms and
   reduced seven initial pools from 2,487 workspace-valid rays to 263 coarse
   survivors; Tesseract then rejected most attempted rays exactly.
3. **Best tested configuration:** 2M atlas as a coarse prefilter and Tesseract
   as exact authority.
4. **Reason retained:** it reduces computational load while preserving the
   physically validated collision/path backend.
5. **Cannot conclude:** map support is not exact IK. The matched offline planner
   benchmark shows faster cuRobo proposals but does not establish physical
   execution performance or mission-level superiority.

### Capture admission and reconstruction selection

1. **Alternatives tested:** raw mask depth versus confidence-qualified support;
   captured versus fresh offline masks; robot pose, bounded GICP, constrained
   superposition, and static-scene registration.
2. **Evidence:** the capture gate sharply reduced depth-layer spread; bounded
   multiway registration sharply reduced residual on the controlled replay.
3. **Best tested configuration:** strict schema-2 admission followed by bounded
   auto-selection was the most defensible tested integration.
4. **Reason retained:** invalid geometry is excluded before fusion, and no
   refinement can silently override the kinematic baseline or quality gate.
5. **Cannot conclude:** all available reconstruction outputs remain `FAIL`, so
   the experiment does not yet establish a successful final 3D model.

## Overall interpretation

The current evidence supports the thesis claim that several final integration
choices were retained because they were the most suitable of the implemented
and tested configurations: target-only prompting, semantic mask support,
confidence-qualified depth, full-sphere bounded rays, a converged capability
atlas before exact Tesseract validation, strict capture admission, and bounded
reconstruction auto-selection.

It does **not** yet support the stronger statement that the complete final
system—through adaptive scan completion and quality-qualified reconstruction—
has been demonstrated to work end to end. The archive contains substantial
physical acquisition and safe-shutdown evidence, but no current dataset passed
the replayed measured-convergence criterion and all stored/fresh reconstruction
quality results remain `FAIL`. Chapter 5 should present that as the remaining
experimental boundary rather than convert integration progress into an
unsupported success claim.
