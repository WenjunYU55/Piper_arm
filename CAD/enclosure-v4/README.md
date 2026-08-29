# Enclosure v4 CAD package

This package is an extracted, browsable copy of the supplied
`All_3DFiles.zip`. It describes the enclosure and sensor-mounting system for the
tracked robot carrying the PiPER arm.

## Snapshot and integrity

- Source archive SHA-256:
  `1c6a05fecd4bc616e5948283db232beedef352130b3425d95635fe6aabd70a71`
- Source archive size: 26,550,290 bytes
- Extracted contents: 33 files, 38,720,962 bytes
- Formats: 17 SolidWorks parts, one SolidWorks assembly, nine STL meshes, five
  DXF drawings, and one 3MF print project
- Integrity: the source ZIP passed CRC validation
- Units: the DXF and 3MF files declare millimetres; the STL files are unitless
  but were exported on the same millimetre scale

Per-file sizes and SHA-256 values are recorded in
[`MANIFEST.csv`](MANIFEST.csv).

## What the parts are used for

| Subsystem | Source and fabrication files | Role on the robot |
|---|---|---|
| Complete enclosure | `FullCase.SLDASM`, `FullCase.SLDPRT`, `FullCase.STL` | Houses and protects tracked-platform electronics and provides the structure to which panels and sensor mounts attach. The SLDASM is the editable master assembly. |
| Base, frame and access | `baseplate`, `FramePart`, `DoorParts` sources and STLs | Forms the structural base, vertical frame rails and removable/access-door hardware. |
| Acrylic panels | `Acrylic/*.SLDPRT` and matching `LazerCut/*.DXF` | Editable panel models and millimetre-scale laser-cut profiles for the front, side, top and plastic enclosure panels. |
| Battery retention | `attachnents/battery.SLDPRT`, `Prints/batteryHolder.STL` | Retains the mobile-platform battery inside the enclosure. |
| PiPER L515 holder | `attachnents/cameraHolder.SLDPRT`, `Prints/cameraHolder_L515.STL` | Three-piece eye-in-hand mount for the Intel RealSense L515 used by the active RGB-D perception and scanning stack. |
| ZED camera mount | `cameraHolder_ZED.SLDPRT`, `ZEDCameraTop.SLDPRT`, `ZEDCameraBottom.SLDPRT`, and ZED STLs | Optional tracked-platform stereo-camera support. It is mechanical provision only; the current target-scan runtime does not consume ZED data. |
| LiDAR mount | `LIDAR.SLDPRT`, `LIDAR_Top.SLDPRT`, `LIDAR_Bottom.SLDPRT`, `Prints/LIDAR.STL` | Optional two-piece LiDAR mount/protective housing for the tracked platform. It is not part of the current L515 target-scan data path. |
| Print project | `Prints/AllPrints_ProjectFile.3mf` | Seven-plate Bambu Studio layout containing the printable enclosure and mount pieces. It stores slicer settings, not ready-to-run G-code. |

## Directory map

```text
Case4/
├── FullCase.SLDASM             Editable master assembly
├── FullCase.SLDPRT             Derived/consolidated enclosure part
├── FullCase.STL                Whole-enclosure reference mesh
├── baseplate.SLDPRT            Structural base source
├── DoorParts.SLDPRT            Access-door source
├── FramePart.SLDPRT            Frame-rail source
├── Acrylic/                    Editable panel parts
├── LazerCut/                   Millimetre DXF panel profiles
├── attachnents/                Battery, L515, ZED and LiDAR native sources
└── Prints/                     Printable STL exports and the 3MF project
```

The original directory names `attachnents` and `LazerCut`, and filenames such
as `frount` and `Plastic_pannel`, are intentionally retained for
assembly-reference compatibility.

## Fabrication notes

- `FullCase.STL` is a 194,524-triangle, 34-shell reference mesh measuring
  approximately 537.5 × 447.55 × 467 mm. It predates the latest
  `FullCase.SLDASM` and ZED/LiDAR source revisions, so treat it as a legacy
  preview until it is re-exported from the current assembly.
- The DXF profiles explicitly use millimetres. Confirm acrylic thickness, kerf,
  hole tolerances, material and quantities before cutting; those choices are
  not fully encoded in the drawings.
- The 3MF was prepared for a Bambu Lab P1S with a 0.4 mm nozzle, PLA Matte,
  0.16 mm layers, five walls, 12% gyroid infill, automatic tree supports and a
  textured PEI plate. Re-slice and inspect every plate for the actual printer,
  material and loading conditions.
- The archive is a design snapshot, not a verified SolidWorks Pack and Go
  package. Check for unresolved external references before editing or releasing
  a new assembly revision.

## Safety and deployment status

This package does not qualify a part for autonomous motion. In particular, the
installed PiPER/L515 holder envelope must continue to match the qualified
runtime collision model and retain the repository's required minimum 5 mm
support-plane and external clearance throughout every validated path.

ZED and LiDAR mounts are documented as optional mechanical hardware because
their drivers and data are not integrated into the current production scan
mission. The tracked base is also not commanded by this repository; it supplies
a mission request and a pose snapshot through the gateway while remaining
stationary during arm execution.

No repository-wide licence is currently declared. Confirm ownership and
redistribution permission for any embedded vendor geometry before reuse or
redistribution.
