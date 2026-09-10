# Offline movement recording compaction — 2026-09-09

The user explicitly requested permanent removal of idle footage from existing recordings.
27 archives were compacted, reclaiming **207.69 GiB**. Total bag database storage across all 33 archives changed from **242.38 GiB** to **34.69 GiB**. No full-size backup of the removed payloads remains.

Selection uses recorded MOVING intervals with one second before and after. Exact RGB/depth header-stamp matching is performed globally across bag splits, retaining a matching partner even if its arrival falls just outside the window. All accepted keyframe messages, calibration, transforms and mission events are byte-identical to their source. Mission success/failure and incomplete status are unchanged. Original timestamps remain; extraction creates one ordered sequence with idle gaps omitted.

Each replacement passed SQLite quick_check, exact topic metadata comparison, and SHA256 comparison of every retained message ID, topic ID, timestamp and serialized payload before the original segment was atomically replaced. A final independent scan of all 27 archives verified matching RGB/depth timestamp sets, planned frameset totals, per-segment message counts, bag metadata, manifest topic counts and storage totals. Every archive has a durable movement_compaction.json report with original metadata and selection/progress evidence.

Validation: 9 focused tests passed, covering cross-split/shoulder pairing, evidence retention, interruption resume, verification-failure preservation, low disk, live/no-event rejection, unfinished-extraction blocking, movement windows and serialized extraction. Native ros2 bag info read a compacted archive. Real extraction from scan_recording_503ead1879014b8c9ec1d74f02c8a475.closed produced **330 strictly timestamp-ordered, exact RGB-D pairs**, with zero unsynchronized-pair rejects. One pre-existing ambiguous aligned-depth sample remained rejected by the existing admission check. The temporary validation images were removed after inspection. This validates storage and extraction, not reconstruction accuracy or a new GPU mesh.

Five archives with no execution events were left untouched: 19905873d6a848058d1930e2033d9e58, 27e3ecb0fd2343b8a0b977bd0ba50328, 836b9699753e45cdbbd78a2d147f8a58, e075e0f1ed064913b0087079f0242099, f25370f84a374a8d919d48136ccd995a. The unfinished archive 387c434cfea34179afd3fe2cd45f7ce4 has no bag metadata and was also left untouched. Missing accepted seeds in some historical recordings are not repaired by storage compaction.

Future recording uses the separate movement_bag_recorder process, saving buffered pre-roll, motion and post-roll directly instead of first writing the entire idle sensor stream. Its prior isolated ROS capture/playback check retained and replayed all 79 expected synthetic frames. No physical arm motion or live-system restart was performed for this offline task.

| Archive | Before GiB | After GiB | Freed GiB | Paired frames | Movement windows |
|---|---:|---:|---:|---:|---:|
| scan_recording_01a1eafc4bb243bca1d44d2d21cbb8ed.closed | 3.86 | 1.54 | 2.32 | 1054 | 3 |
| scan_recording_249c44eb71da43879972fb52e136dbd7.closed | 4.77 | 1.16 | 3.61 | 755 | 8 |
| scan_recording_250f813fa205425694756cf66fefc0e4.closed | 0.49 | 0.04 | 0.45 | 26 | 1 |
| scan_recording_3f0d10c203af4aacb01491c4488b877f.closed | 6.95 | 1.18 | 5.77 | 750 | 10 |
| scan_recording_4b670f4ac9844a83bafb25a99542edd9.incomplete | 0.32 | 0.07 | 0.26 | 45 | 1 |
| scan_recording_4eeb5e61040d45f6b384f473f67aad93.closed | 6.90 | 1.41 | 5.49 | 908 | 9 |
| scan_recording_503ead1879014b8c9ec1d74f02c8a475.closed | 22.44 | 0.83 | 21.62 | 500 | 10 |
| scan_recording_514389daf43a42f3a1927014d693fec1.closed | 14.90 | 0.67 | 14.22 | 426 | 7 |
| scan_recording_54f1852a9fac44eb9977e8dae4bf88c8.closed | 3.20 | 0.67 | 2.54 | 439 | 6 |
| scan_recording_79c526c02867477db7b6a902dc5957a0.closed | 4.20 | 0.97 | 3.23 | 629 | 8 |
| scan_recording_7ff9872d9ef2427c8027a3d5dd8425e5.closed | 7.00 | 1.29 | 5.71 | 828 | 12 |
| scan_recording_81bf38bf83ee4c7180d6a29d6ae5f23a.closed | 13.13 | 0.63 | 12.50 | 393 | 8 |
| scan_recording_86ccbfd3628b4303905f276acb0f286c.closed | 14.78 | 0.91 | 13.87 | 573 | 11 |
| scan_recording_8813e7f7897e45bab0dfa2f3ba0ab3fb.closed | 2.05 | 1.32 | 0.73 | 916 | 2 |
| scan_recording_99d9214ab852461fa46635ee44c2f9b8.closed | 2.36 | 1.09 | 1.26 | 760 | 2 |
| scan_recording_a121779e656c4664b0b16c733d410894.closed | 0.60 | 0.27 | 0.33 | 187 | 2 |
| scan_recording_a15744ab83d64555b033f6e0b320abb4.incomplete | 11.11 | 0.67 | 10.44 | 424 | 6 |
| scan_recording_a42731e498be48ab9661f26576aded5c.closed | 28.74 | 0.96 | 27.78 | 591 | 10 |
| scan_recording_a9e391e52a0e4c34840b6ad3ca92dca3.closed | 26.39 | 0.76 | 25.62 | 466 | 8 |
| scan_recording_b22e866e22c44d87aecfa426bded2194.closed | 0.59 | 0.30 | 0.29 | 206 | 2 |
| scan_recording_b2fdc8f2d8834b16b38e22c31a18f552.closed | 18.01 | 1.00 | 17.01 | 632 | 8 |
| scan_recording_d17e9b84327943cfbdfc0c7b240cef74.closed | 10.25 | 4.00 | 6.24 | 2636 | 9 |
| scan_recording_ddcbaa5b26094161af174c9f53eddddb.closed | 2.69 | 0.61 | 2.08 | 405 | 5 |
| scan_recording_e22bbb2105c449ed87487036e6e49766.closed | 4.27 | 0.93 | 3.34 | 607 | 8 |
| scan_recording_f04c67819db94786be284c991b936c41.closed | 7.79 | 0.59 | 7.20 | 389 | 4 |
| scan_recording_f1599702b64d4bbfb2f37027d322988f.closed | 10.96 | 2.25 | 8.71 | 1421 | 19 |
| scan_recording_f62ed65bbeb4459cb35a9b36e64869f9.closed | 7.81 | 2.75 | 5.05 | 1818 | 8 |
