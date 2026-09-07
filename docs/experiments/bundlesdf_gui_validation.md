# BundleSDF GUI result opening — 2026-09-07

The colourful cube shown in the preceding GPU smoke test is an analytic synthetic
object, not a live camera recording. This change makes its provenance explicit
in the Piper GUI and lets users open saved BundleSDF outputs from Reconstruction
Validation, following the existing separate interactive-viewer pattern.

Started from clean `main` at `e4c9cf3f85bc49994dd2652bb03fb17f322ddd85`;
developed on `feature/bundlesdf-gui-viewer`. The implementation adds
`piper_gui/bundlesdf_results.py`, `reconstruction/view_bundlesdf.py`, and
`tests/gui/test_bundlesdf_results.py`, wires the panel into `piper_gui/native_app.py`,
and updates AI routing/contracts/flows/guardrails and native-runtime instructions.

The panel discovers retained external results and supports arbitrary output-folder
browsing. It never labels unknown provenance as live. A selected result launches
through the dedicated BundleSDF interpreter; heavy mesh/texture loading stays
outside Tk and Foxy. Viewer failures return to the GUI status label. The tab now
uses the existing scroll-area helper to retain access to its TSDF controls.
No recording, reconstruction algorithm, mission state, ROS endpoint, planner,
safety policy or capture data changed. The view is a separate window opened from
the GUI, not an embedded Tk 3D canvas.

Executed validation:

```bash
source piper_ros_foxy/install/setup.bash
python3 -m pytest -q tests/gui reconstruction/test_gui_support.py
# Baseline: 155 passed. With the new tests: 158 passed.
python3 -m pytest -q tests/gui reconstruction/test_gui_support.py tests/reconstruction
# Final: 228 passed in 11.61 seconds.
python3 -m flake8 --max-line-length=99 piper_gui/bundlesdf_results.py \
  reconstruction/view_bundlesdf.py tests/gui/test_bundlesdf_results.py
python3 -m py_compile piper_gui/native_app.py piper_gui/bundlesdf_results.py \
  reconstruction/view_bundlesdf.py
```

A real Tk `PiperGuiApp` was opened on isolated ROS domain 179, localhost-only,
with `PIPER_MISSION_ENABLE_REAL_MOTION=0`. Its Reconstruction Validation tab was
selected and the actual `Open BundleSDF Model` button invoked. X11 confirmed both
`PiPER Control — isolated GUI viewer test` and
`BundleSDF: Synthetic — not a live recording` windows. After 20 seconds the test
closed only its own viewer and GUI, reporting `GUI_SMOKE_PASS` in
`/tmp/piper-bundlesdf-gui-smoke.log`. Existing mission and GUI processes were not
restarted. This is GUI/viewer integration evidence, not physical validation.

Tests exercise synthetic/unknown provenance, invalid folders, literal path
arguments, missing environment, actual Tk button invocation and child failure
reporting. Lint, Python compile and architecture YAML parsing passed. An initial
lint check found two long lines and an unused test import; these were fixed.
The preceding full GPU reconstruction is not rerun for this viewer-only change;
its existing textured output was loaded by the actual native viewer.

Usage: reopen the GUI when appropriate, select **Reconstruction Validation →
BundleSDF results**, choose an output folder, then **Open BundleSDF Model**.
The retained synthetic demo is automatically listed. Live camera reconstruction
quality remains untested.
