# Architecture diagram sources

`generate_diagrams.py` is the authoritative source for the SVG figures embedded
in the root README and system-diagram guide. The generated assets are committed
under `docs/assets/readme/architecture/` so GitHub can render them without an
external service.

Regenerate every figure from the repository root:

```bash
python3 docs/architecture/diagrams/generate_diagrams.py
```

The generator uses only the Python standard library and writes deterministic
UTF-8 SVG files.

## Maintenance rules

- Keep the whole-system diagram predominantly vertical so labels remain
  readable at normal GitHub README width.
- Use one focused diagram per major responsibility: perception, view planning,
  guarded execution, capture/reconstruction and physical hardware.
- Blue represents physical inputs, violet perception, teal accepted state/data,
  amber planning, red safety, graphite actuation and gray optional hardware.
- Solid arrows show the main runtime progression. Dashed returns show feedback,
  replanning or recovery rather than a second command authority.
- Do not show the ZED camera, LiDAR or tracked-base drive as active inputs or
  commanded outputs of the current PiPER target-scan runtime.
- Update the accompanying prose and architecture evidence whenever a boundary
  changes.
