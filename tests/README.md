# Repository Development Tests

These tests exercise top-level adapters and cross-package behaviour without
being owned by a single ROS package.

- `gui/` covers the native GUI, GUI-to-ROS adapter, camera profile, and ray
  review presentation.
- `driver/` covers top-level hardware-driver tools and DDS transport contracts.
- `planning/` covers planning diagnostics that span the GUI and ROS package.

Run all repository development tests from the repository root:

```bash
python3 -m pytest -q tests
```

ROS-dependent tests require the built workspace overlays to be sourced first.
