# Architecture diagram sources

`generate_diagrams.py` is the authoritative source for the SVG figures embedded in the root README and the system-diagram guide. Generated assets are committed under `docs/assets/readme/architecture/` so GitHub renders them without an external diagram service.

Regenerate every figure from the repository root:

```bash
python3 docs/architecture/diagrams/generate_diagrams.py
```

The generator uses only the Python standard library and writes deterministic UTF-8 SVG.

Run the structural checks after regeneration:

```bash
python3 docs/architecture/diagrams/validate_diagrams.py
```

## Diagram set

- `system-overview.svg`: one detailed end-to-end map covering mission ownership, perception, accepted-only NBV, backend selection, authorization/execution, capture feedback, terminal recovery and reconstruction.
- `perception-pipeline.svg`: measurement freshness, tracking degradation and visual reacquisition.
- `viewpoint-planning-pipeline.svg`: accepted coverage plus distinct accept/retry/reject/replan outcomes.
- `planner-backend-pipeline.svg`: frozen Tesseract or cuRobo selection and the generic motion contract.
- `execution-safety-pipeline.svg`: plan authorization, sole command ownership, physical feedback and bounded recovery.
- `capture-reconstruction-pipeline.svg`: settled burst admission, atomic commit, same-pose retry, exclusion and safe reconstruction.
- `hardware-topology.svg`: installed hardware, isolated compute environments and optional CAD provision.

## Maintenance rules

- Keep the whole-system map predominantly vertical and detailed enough to show all state-changing features. Give every numbered section a distinct header band, generous inter-row gaps and visible bottom padding; the master should be read section by section, not compressed into a poster. It is the canonical overview; focused diagrams explain individual loops rather than replacing it.
- Keep focused canvases at 1280 px and embed every diagram at up to 1000 px. At GitHub desktop width, body text must remain at least 11 px and edge labels at least 10 px.
- Draw connector paths first, cards second and opaque label capsules last. Feedback buses must stay outside lane headings and status badges must occupy their own header row.
- Never hide qualification status. An optional or unqualified path must use gray dashed edges and an explicit status badge. Hardware qualification must be stated independently from software availability.
- Keep planner workers command-free. Only `scan_viewpoint_executor` may be labelled as the autonomous joint publisher, and only the PiPER driver may be labelled as the CAN owner.
- Show feedback where it changes behavior: worker readiness, target loss, planner rejection, runtime safety, capture retry/rejection, accepted-history generation, terminal recovery and disable proof.
- Distinguish accepted evidence from prediction or achieved pose. Rejected captures may update physical FK but must never update measured coverage.
- Blue represents physical inputs, violet perception, teal measured state/data, amber planning, red safety/recovery, graphite actuation and gray optional or unqualified elements.
- Use graphite for data/evidence, blue for mission control, dashed green for feedback/retry and red only for motor commands.
- Do not show ZED, LiDAR or tracked-base drive as active inputs or commanded outputs of the current PiPER scan runtime.
- Update the prose, audited commit table and implementation-evidence links whenever an ownership or qualification boundary changes.

## Visual verification

Generated SVG is the deliverable, but it must be rendered before review. Run `validate_diagrams.py`, check the full system map at its native aspect ratio and inspect each focused diagram at normal GitHub README width. Confirm that text is legible, legends and labels are not clipped, status pills do not collide with titles, arrows terminate at the intended boxes, and feedback loops stay visually distinct from command flow.
