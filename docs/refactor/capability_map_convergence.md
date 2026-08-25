# Capability-map convergence

The atlas stores occupied 20 mm camera-position cells plus 10-degree optical-direction bins. It stores neither raw joint samples nor a solid outer workspace volume.

| Samples | Occupied 5D bins | Validation support | Final-support recall | Known poses | Median query | Artifact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 83,691 | 207/2,520 | 60.17% | 80/87 | 0.243 ms | 535,146 B |
| 250,000 | 207,043 | 242/2,520 | 70.35% | 86/87 | 0.257 ms | 1,270,919 B |
| 500,000 | 406,842 | 267/2,520 | 77.62% | 86/87 | 0.269 ms | 2,418,557 B |
| 1,000,000 | 787,065 | 311/2,520 | 90.41% | 87/87 | 0.291 ms | 4,509,074 B |
| 2,000,000 | 1,479,561 | 344/2,520 | 100.00% | 87/87 | 0.298 ms | 8,130,884 B |

**Selected:** `capability_map_2000000.npz` (passed enforcement qualification).

The two-million-sample checkpoint is the comparison reference, not automatically the runtime artifact. Tesseract remains authoritative for exact IK, collision, path and visibility validation.
