# Mechanical CAD

This directory contains editable mechanical sources and fabrication exports for
the tracked-platform enclosure and sensor-mounting hardware used with the PiPER
mobile-manipulation project.

## Available design package

- [`enclosure-v4/`](enclosure-v4/) — the enclosure assembly, panel drawings,
  printable structural parts, battery holder, PiPER/L515 camera holder, and
  optional ZED-camera and LiDAR mounts.

The supplied filenames and directory structure are preserved, including their
original spelling, because changing them may break external references in the
SolidWorks assembly. Use the corrected component names in documentation, not by
renaming the source files in place.

## Relationship to runtime geometry

These files are design and manufacturing sources. The collision-qualified
runtime meshes used by the URDF and Tesseract remain under
`piper_ros_foxy/src/piper_description/meshes/`. Do not replace those files from
this directory without regenerating the collision model, updating its manifest
and hashes, and repeating the documented clearance and motion qualification.

The L515 holder belongs to the current eye-in-hand scanning stack. The ZED and
LiDAR parts provide mechanical mounting options for the tracked platform but
are not inputs to the current L515-based target-scan pipeline.

See [the system diagrams](../docs/architecture/system-diagrams.md) for the
hardware/software boundary and
[the large-asset policy](../docs/architecture/asset_policy.md) for repository
rules.
