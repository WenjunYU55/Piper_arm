# Offline Test Sets

This folder contains offline-only validation sets for saved L515 RGB-D captures.

The `real_l515_baseline` set is a small validation/debug set, not a training database. It is intended for checking the current static analyser and, later, pretrained GroundingDINO and SAM2 models against real L515 images from the PiPER scene.

Capture folders are organised by symlink where possible so the original data remains under:

```text
/home/prl/Piper_arm/L515_camera/captures
```

Categories:

- `clear_cube`
- `partial_occlusion`
- `heavy_occlusion`
- `hand_blocker`
- `edge_cases`
- `lost_target`
- `unknown`
