# Capability-map convergence

The atlas stores occupied 20 mm camera-position cells plus 10-degree optical-direction bins. It stores neither raw joint samples nor a solid outer workspace volume.

| Samples | Occupied 5D bins | Validation support | Final-support recall | Known poses | Median query | Artifact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 82,808 | 53/720 | 60.23% | 8/9 | 0.321 ms | 530,085 B |
| 250,000 | 204,760 | 68/720 | 77.27% | 9/9 | 0.333 ms | 1,258,498 B |
| 500,000 | 402,761 | 73/720 | 82.95% | 9/9 | 0.347 ms | 2,397,780 B |
| 1,000,000 | 780,005 | 82/720 | 93.18% | 9/9 | 0.361 ms | 4,478,009 B |
| 2,000,000 | 1,470,246 | 88/720 | 100.00% | 9/9 | 0.373 ms | 8,089,252 B |

**Selected:** `capability_map_2000000.npz` (passed enforcement qualification).

The two-million-sample checkpoint is the comparison reference, not automatically the runtime artifact. Tesseract remains authoritative for exact IK, collision, path and visibility validation.
