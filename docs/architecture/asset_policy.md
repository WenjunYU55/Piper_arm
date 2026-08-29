# Large-asset policy

This repository contains about 84 MB across 115 tracked STL files, an 8.1 MB
camera capability map, and 213 hand-eye calibration records. These files were
audited separately from the source refactor. None were moved or deleted.

| Asset class | Current owner | Classification | Disposition |
|---|---|---|---|
| PiPER visual meshes | `piper_description/meshes/*.STL` | Canonical runtime robot-description assets | Keep tracked at the current package paths. URDF resource lookup depends on them. Git LFS is optional for future repository-size reduction, but only as a separately tested repository migration. |
| L515, holder, and Bunker source meshes | `piper_description/meshes/` and `platform_sources/` | Canonical source and runtime visual/collision assets | Keep. Source and transformed runtime geometry have distinct provenance and are not duplicates. |
| Enclosure and sensor-mount manufacturing CAD | `CAD/enclosure-v4/` | User-supplied design snapshot: native SolidWorks, DXF, STL and 3MF files | Keep extracted and checksum-manifested for design/manufacturing traceability. These files do not replace qualified URDF/Tesseract meshes. Preserve the supplied relative paths until a validated SolidWorks Pack and Go revision repairs references. |
| Curated README media | `docs/assets/readme/media/` | Repository documentation assets derived from project evidence | Keep the two source screenshots and the web-optimised ray-process preview. The committed MP4 is a 720p/15 fps H.264 derivative; retain the full-resolution 1080p source outside Git and replace the preview only through a documented re-encode. |
| Decomposed collision cells | `planning_30mm/`, `platform_planning_150mm/` | Generated planning artifacts required at runtime | Keep at their current paths with their collision-mesh manifests. Regeneration is possible only through the documented, hash-checked geometry workflow; do not regenerate during ordinary cleanup. |
| Camera capability map | `piper_mobile_manipulation/config/piper_camera_capability_map.npz` | Generated, qualified runtime planning asset | Keep tracked. Its convergence record and implementation/model hashes make it part of the qualified planner contract. Consider a release artifact only after startup can fetch and verify it without changing offline operation. |
| Hand-eye sessions | `L515_camera/calibration/hand_eye/` | Calibration results plus reproducibility evidence | Keep the deployed result and the raw/validation evidence currently used for provenance. A later archival policy may move superseded raw sessions, but only after references and hash contracts are audited. |
| Joint bounds | `piper_joint_bounds.json` | Canonical measured safety configuration | Keep tracked at the root because operator tools and launch workflows use the stable path. The ignored local `piper_home_pose.json` remains machine-specific. |
| Model checkpoints | GroundingDINO/SAM environments | Downloaded runtime dependency | None are tracked. Continue downloading through setup tooling and keep them ignored. |
| Captures, reconstructions, ray diagnostics, ROS bags | `datasets/` and runtime output locations | Generated experiment data | Keep ignored by default. Commit only small, deliberately curated fixtures under an explicit test-data owner. |
| `build/`, `install/`, `log/`, caches and bytecode | workspace-local generated output | Reproducible temporary output | Never track. The canonical ROS build output remains under `piper_ros_foxy/`; root-level overlays can shadow generated interfaces and are unsupported. |

## Decision rules

1. A large binary is not removed merely because it is large.
2. Runtime paths are compatibility contracts for URDF, launch, Tesseract and
   calibration consumers.
3. Generated assets may be replaced only by a deterministic generator plus a
   manifest or hash-based equivalence check.
4. Git LFS migration is repository administration, not an architectural code
   refactor. It requires a clone/build/runtime check and explicit coordination
   because it rewrites how every checkout obtains binary content.
5. Experiment outputs stay outside source ownership; curated fixtures must be
   minimal and documented by the test that consumes them.
