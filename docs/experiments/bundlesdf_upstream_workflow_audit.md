# BundleSDF upstream workflow audit — 2026-09-07

The user requested a source-level check of the creators' continuous-camera demo
against this integration. This is an audit, not a claim of completed dense-video
recovery or improved reconstruction quality. No motion, captures, frame-selection
policy, calibration or runtime code was changed during this audit.

## Authoritative sources inspected

- [Official repository and custom-data instructions](https://github.com/NVlabs/BundleSDF)
- [Creators' project/demo page](https://bundlesdf.github.io/)
- Local verified pin `ffa67d425240b5b76d2e387a7dd3d3735a7cf1a1`:
  `run_custom.py`, `segmentation_utils.py`,
  `BundleTrack/scripts/data_reader.py`, `bundlesdf.py`,
  `BundleTrack/src/Bundler.cpp`, `config.yml`, `BundleTrack/config_ho3d.yml`.

`git ls-remote https://github.com/NVlabs/BundleSDF.git HEAD` returned the same
commit as the local pinned checkout. The five entrypoint/segmentation/loader/config
files match their upstream Git versions. Existing local modifications are native
build bindings and the documented replacement of an interactive debugger with a
warning; reconstruction algorithms and matching thresholds remain unchanged.

## Comparison

| Concern | Upstream behaviour | Current Piper integration |
|---|---|---|
| Camera recording | RGB-D video; project demonstrates a RealSense milk jug and moving-camera static scenes | Whole-scan archive retains RGB, aligned depth, calibration and available masks |
| Custom input | Ordered matching RGB/depth/mask PNG names and 3×3 K; depth uint16 millimetres | Matches; pinned reader loaded all 141 converted frames |
| Segmentation | Paper assumes first-frame segmentation; public release omits XMem integration, with a stub that reads saved mask PNGs | Uses saved live SAM2 masks, with exact RGB timestamp matching; no dense offline propagation over the full recording |
| Tracking input | `--stride 1` processes every supplied frame | Wrapper stride defaults to 1, but only 141 mission-recorder-selected frames are supplied upstream |
| Pose | Upstream estimates object-in-camera transforms from the sequence | Same; saved robot camera poses are diagnostic sidecars, not replacement object poses |
| Global refinement | README lists a separate command; at the actual pinned commit `run_one_video` already calls global refinement | Included automatically; actual output contains `final/nerf/config.yml` with 2,000 steps and a textured mesh |
| Accuracy | Demonstrated on supplied examples; no guarantee for arbitrary bad inputs | Structural artifact checks passed, but saved real mesh visual quality failed |

The important mismatch is **continuous recording to dense segmented input**.
`import_bag.py` deliberately invokes the real mission recorder, which starts at
first accepted capture, admits moving frames only in the recorded MOVING state,
pins accepted captures, requires exact-stamped masks, and applies 8 Hz sampling
plus quality gates. This is not a general full-recording preprocessing pipeline.
It should not have been described as equivalent to the creators' full video demo.

The existing recorder should remain available for its original contract. A separate
offline preparation path needs to recover synchronized RGB-D in timestamp order,
initialize the target from the accepted mask, run segmentation across the sequence
in an isolated environment, validate the resulting masks/depth, and feed usable
consecutive frames to upstream. It must never reuse a stale mask as if it belonged
to a new frame, invent missing camera frames, or relax robot safety. This path is
**not implemented by this audit**.

## Evidence from the previous real reconstruction

The 141 selected images span **120.0124 seconds**. Median spacing is **0.2993 s**,
p95 **0.8086 s**, maximum **23.4852 s**. Six gaps exceed one second.
Time gaps alone are not proof of tracking failure: most large gaps cover a
stationary pause. The 7.4937-second gap from frame 98 to 99 includes **22.307 mm**
and **6.678 degrees** of recorded camera change. Upstream's log then reports an
estimated 33.489 mm neighbour displacement and rejects frame 99. This is a useful
diagnostic association, not proof that frame rate alone caused the failure.

The actual GPU log explicitly marks frames **4, 47, 99 and 137** as `FAIL` and
contains six insufficient-reference-match recovery messages. A directory with 141
pose matrices does not establish that all 141 frames tracked successfully.
Global refinement used 43 selected keyframes, as recorded in the earlier report.
The generated mesh remains visually unacceptable; no new mesh was produced here.

Machine-readable source checks, spacing/camera-motion measurements and retained
download error are stored outside Git:

`/home/prl/.local/share/piper-bundlesdf/evidence/2026-09-07-upstream-workflow-audit/`

The renewed official example attempt was:

```bash
/home/prl/.venvs/piper-bundlesdf/bin/python -m gdown \
  'https://drive.google.com/uc?id=1akutk_Vay5zJRMr3hVzZ7s69GT4gxuWN' \
  -O /tmp/piper-official-milk-audit.zip
```

It exited **1**, reporting Google Drive's download quota. No official-example GPU
reconstruction was run or claimed. The prior actual synthetic and retained-camera
GPU results remain evidence of execution, not validation of the creators' example
or real-object accuracy. Runtime code was unchanged in this audit, so no fresh
build/test pass is claimed; the preceding distortion follow-up's 253 passing tests
do not prove dense-video workflow completeness.
