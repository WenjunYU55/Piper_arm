# Recording, depth and frame contracts

This page collects the cross-cutting rules for sensor evidence and reconstruction. The operator workflow, recorder defaults and commands are in [reconstruction recording](reconstruction_recording.md); the isolated Python/CUDA boundary is in [native BundleSDF runtime](bundlesdf_native_runtime.md); whole-archive temporal segmentation is in [dense offline video](dense_offline_video.md). These systems record or interpret evidence offline and do not own robot motion or mission capture counts.

## Recording lifecycle and resource bounds

The quality-filtered reconstruction recorder and the whole-scan archive are separate outputs. The former starts with accepted capture zero, preserves that exact raw synchronized tuple as the seed, and admits moving frames through the recorder gates. The latter starts with the scan stack and retains sensor evidence around motion, including frames that fail reconstruction quality. Neither recorder has command authority; a recording failure does not fail launch or alter motion. Existing stationary RGB and median-depth mission captures remain authoritative for mission capture counts.

The movement-window recorder buffers at most one second and 128 MiB of idle RGB/depth/mask messages. A MOVING event flushes the pre-roll once; the first subsequent non-MOVING event starts one second of post-roll. Overlapping windows do not duplicate messages. Camera calibration, transforms, mission events and accepted keyframe pins remain in the archive. Sensor delivery is best effort (history depth 8); slow storage can lose messages. Foxy does not provide a reliable transport-drop total, so this count is unknown, never reported as zero.

The archive uses a Foxy-compatible SQLite writer with bounded 0.2-second transactions. The default free-space reserve is 2 GiB. UUID `.inprogress` outputs are never reused; normal recorder closure verifies SQLite row counts against metadata before `.closed`. That suffix means the archive closed, not that the mission succeeded. SIGINT/SIGTERM is forwarded to the owned recorder group with an eight-second graceful wait; low disk, error or forced termination preserves partial output as incomplete/in progress when possible. A hard kill can leave an inspectable `.inprogress` directory.

Extraction is read-only and chooses movement intervals with one-second shoulders, then retains exact RGB/depth pairs and accepted pins. Legacy archives without execution status use a labelled full-sequence fallback. Destructive compaction is a distinct explicit operation: [compact_recordings.py](../../tools/bundlesdf/compact_recordings.py) is plan-only unless `--apply` is passed. It operates only on closed/incomplete archives with metadata and movement evidence; refuses open databases and pending journals; matches exact header stamps globally across split databases; and never synthesizes a partner. It replaces the smallest retained segment first, verifies retained rows by SHA256 and SQLite `quick_check`, fsyncs and atomically replaces. Its durable progress journal supports resuming the same invocation; extraction refuses unfinished compaction. Source timestamps and mission outcome are preserved.

## Exact pairing, depth and distortion

An aligned RGB-D frame is valid for reconstruction only when RGB and depth header stamps are exactly equal. Approximate synchronization can deliver candidates for rejection accounting, but neither live admission, native dataset validation nor legacy archive extraction may save an approximate pair. `synchronization_slop_sec` is compatibility/provenance metadata and does not relax this rule. The driver guard also requires color in the original frameset and processed depth with matching finite color intrinsics, dimensions, distortion and identity depth-to-color extrinsics; invalid aligned output is skipped. Native depth publication remains available, and no TF or camera profile is changed. The camera was software-tested only; stationary real-camera validation and later scan remain unqualified. See the [aligned depth validation](../experiments/aligned_depth_fix_validation.md).

The recorder retains raw RGB, aligned depth and masks, plus the supported `plumb_bob` (five coefficient) or `rational_polynomial` (eight coefficient) calibration. Unsupported, non-finite or changing calibration rejects the frame. Rectification is offline and uses the recorded color model and unchanged intrinsic matrix: linear RGB interpolation, nearest-neighbour depth/mask sampling, zero borders. Never apply color distortion coefficients to a native depth-camera grid. Depth is axial Z, not radial range, and rectification does not rescale it. Target depth support is checked after remapping; planar residuals are diagnostics, not a fitted correction. Raw session files are unchanged. See [distortion validation](../experiments/offline_distortion_validation.md).

For the filtered recorder, the ROS synchronizer's 100 ms diagnostic candidate window is distinct from the configured admission limit (default 0.08 s, permitted `(0, 0.1]`). Masks must match the RGB stamp exactly; there is no rectangular fallback. Moving sample rate defaults to 8 Hz (permitted 5–10 Hz). Non-finite depth inside the target rejects; non-finite values outside it may remain in native data and convert to zero for BundleSDF. Native depth retains original 16UC1 or 32FC1 precision; 32FC1 is metres, and 16UC1 scale defaults to 0.001 m/unit.

## Temporal segmentation and coordinate provenance

The GUI archive path extracts synchronized movement-window frames after the accepted-history seed, then runs the existing SAM2.1 tiny temporal predictor in its separate perception environment. It segments every extracted frame, including frames without usable depth or TF; generated masks do not replace the accepted seed. The seed's exact pin and frozen target authorize the sequence. The normal temporal-memory limit is 16 retained images, preserving the conditioning frame. Frame stride is applied only after propagation. Saved quality thresholds are reused; offline preparation does not add MOVING-only or 8 Hz admission. Missing image-time TF prevents reconstruction admission but does not prevent mask propagation. Partial/truncated jobs remain explicitly incomplete. Saved pose count is not successful tracking count, and structural mesh checks do not establish accuracy.

The native camera pose is `T_base_camera`, mapping optical-camera points into `base_link` (`p_base = R_base_camera p_camera + t_base_camera`). It is not an object pose. TF lookup uses the image stamp exactly; no latest-transform fallback exists. Capture reconstructions already use recorded base-frame geometry and are not rotated or refit. Reports and per-geometry `.frame.json` sidecars bind frame identity, units and geometry SHA256. Missing/non-base parent identity remains explicitly unverified.

BundleSDF's native mesh and estimated poses remain in the upstream model frame. A separate base-frame export is allowed only with explicit `base_link` identity, finite right-handed rigid transforms, unique same-image correspondence and the pinned upstream initial gauge:

```text
T_base_object = T_base_camera[00000000.png] @ T_camera_object[00000000.txt]
```

No later-pose averaging, PCA, face fitting, scaling or reflection is used. If provenance is missing or unsupported, retain the native result and mark orientation unavailable. A sidecar hash prevents re-transforming an already oriented export. Z-up means robot-base +Z, not independently measured gravity; display arrows show axes and robot direction, not the physical base origin. See [coordinate-frame implementation](../../reconstruction/coordinate_frames.py).

## Ownership and verification

Parameter and behavior ownership remains with the source components: [recorder configuration](../../piper_ros_foxy/src/piper_mobile_manipulation/config/reconstruction_recording_params.yaml), [movement-window recorder](../../piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/movement_bag_recorder.py), [archive lifecycle](../../piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/mission_archive.py), [admission core](../../piper_ros_foxy/src/piper_mobile_manipulation/piper_mobile_manipulation/reconstruction_recording/core.py), [dense extraction](../../tools/bundlesdf/extract_video.py), and [coordinate helpers](../../reconstruction/coordinate_frames.py). BundleSDF dependencies and weights stay outside ROS Foxy; see [native runtime](bundlesdf_native_runtime.md).

For software regressions, source the Foxy overlay and run the maintained reconstruction suite and recorder checks:

```bash
source piper_ros_foxy/install/setup.bash
python3 -m pytest -q tests/reconstruction
```

These checks do not qualify physical camera timing, arm motion, or real-object accuracy. The [recording validation](../experiments/automatic_scan_recording_validation.md), [dense-video validation](../experiments/dense_offline_video_validation.md), [native validation](../experiments/bundlesdf_native_validation.md), and [distortion validation](../experiments/offline_distortion_validation.md) retain their respective evidence and limitations.
