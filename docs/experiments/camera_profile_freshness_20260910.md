# Camera-profile freshness diagnosis — 10 September 2026

Read-only comparison of seven retained CameraProfiles_Final attempts. No camera, GPU job or robot command was started. Mission windows begin at TARGET_LOCK and end at terminal RETURNING_HOME; capture counts are saved result counts.

| Profile | Captures | Recorded colour CameraInfo Hz | Accepted tracker log events/s | Stale-mask log events | Median mask receipt age (ms) | 95th percentile age (ms) |
|---|---:|---:|---:|---:|---:|---:|
| 640x480 @ 15 | 7 | 13.85 | 4.38 | 47 | 294 | 360 |
| 640x480 @ 30 | 7 | 27.59 | 3.91 | 50 | 374 | 506 |
| 960x540 @ 15 | 8 | 13.86 | 2.72 | 155 | 316 | 406 |
| 960x540 @ 30 | 6 | 28.09 | 2.30 | 69 | 430 | 629 |
| 1280x720 @ 15 | 4 | 13.88 | 0.94 | 99 | 373 | 500 |
| 1280x720 @ 15 | 4 | 13.92 | 1.08 | 48 | 364 | 491 |
| 1280x720 @ 30 | 0 | 29.07 | 0.37 | 9 | unavailable | unavailable |

## Findings

- The two 1280x720/15 failures retain colour CameraInfo at 13.88 and 13.92 Hz; maximum recorded interarrival gaps are 0.402 and 0.407 seconds. No >0.5-second gap in these observation windows. This argues against a sustained camera-wide stall, but CameraInfo is a lightweight proxy and cannot prove delivery to every full-image subscriber.
- Mask receipt age includes source clock offset, camera delivery, processing, transport and recorder scheduling. At 15 FPS, CameraInfo median receipt age increases only from 104 to 113 to 130 ms, whereas masks rise from 294 to 316 to 364–373 ms. This supports increased downstream latency, without isolating CPU, GPU, I/O or DDS contributions.
- Mask statistics cover retained movement windows plus their shoulders only. Idle masks at the final failure are intentionally not stored. They are not a replay of the failing callbacks or unbiased whole-mission latency statistics.
- Stale-mask rejections already occurred in lower-resolution runs. Counts are log events, not a calibrated frame-drop percentage. This was not newly introduced at 720p; higher-resolution runs produce fewer accepted target updates and hit the fatal recovery boundary.
- `depth_to_3d_node.py` synchronizes detection, depth and CameraInfo, but holds only `latest_mask_msg`. It rejects absolute source mask/depth stamp separation >0.20 seconds. It has no exact mask-to-source-depth buffer, so delayed or differently ordered mask delivery makes validity intermittent. Mask receipt age is not the same quantity as this 200 ms gate.
- `sam2_live_bridge_node.py` JPEG-encodes native frames in the colour callback, writes native images, reads and publishes five full-resolution mask/ID outputs per result, and publishes health on a 200 ms timer. These callbacks share the default single-threaded spin. Higher resolutions increase this work even though the neural input is resized.
- `sam2_live_worker.py` defaults to inference width 384; 960x540 and 1280x720 both infer at 384x216. The surviving 102 result payloads from the later 1280x720/30 run show median inference 82.1 ms and p95 93.2 ms. These exclude decoding, resizing, output encoding and transfer. They do not measure inference during the earlier failing 15 FPS windows.
- In the second 720p/15 failure, valid tracking was logged at 1789051811.037, 125 ms before the executor aborted at 1789051811.163 still reporting prediction-only/low speed scale. A 200 ms health timer plus separate target-status propagation makes that race plausible; exact callback ordering was not retained.

## Diagnosis and next work

Supported mechanism: latency-sensitive live mask/depth association produces intermittent invalid target measurements; the short post-validation deadline can turn transient or already-recovering tracking into terminal failure. Higher resolution correlates with worse downstream mask latency and accepted-update cadence. Neither camera hardware failure nor GPU saturation is established.

Priorities: (1) correlate masks with bounded buffered source-frame depth/CameraInfo and retain image-time TF plus an explicit wall-age gate; delayed historical pairs must not authorize current motion; (2) return recoverable pre-motion visual failure to bounded stationary reacquisition and fresh replanning; (3) publish health on relevant transitions and measure frame receipt, encode/write, inference, result publication and executor receipt latency. Keep archived source timing intact and compare profiles without physical movement before runtime qualification. No production behavior changed by this investigation.
