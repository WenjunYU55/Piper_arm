#!/usr/bin/env python3
"""Generate the repository's dependency-free SVG architecture diagrams."""

from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "assets" / "readme" / "architecture"

COLORS = {
    "blue": ("#e8f1fb", "#2f6b9a"),
    "violet": ("#f0eaf8", "#6f4c9b"),
    "teal": ("#e4f4f1", "#2a7f76"),
    "amber": ("#fff2d8", "#b57900"),
    "red": ("#fce8e6", "#b83a3a"),
    "graphite": ("#eceff1", "#4b5563"),
    "gray": ("#f7f7f7", "#9ca3af"),
}


def node(title, detail, color):
    return {"title": title, "detail": detail, "color": color}


def svg_text(x, y, value, css_class, anchor="middle"):
    return (
        f'<text x="{x}" y="{y}" class="{css_class}" '
        f'text-anchor="{anchor}">{escape(value)}</text>'
    )


def render_pipeline(filename, title, subtitle, stages, note, feedback=None):
    width = 900
    top = 126
    box_x = 135
    box_w = 630
    box_h = 78
    gap = 44
    stage_step = box_h + gap
    legend_y = top + len(stages) * stage_step + 10
    height = legend_y + 118

    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f'<title>{escape(title)}</title>',
        f'<desc>{escape(subtitle)} {escape(note)}</desc>',
        '<defs>',
        '  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
        '    <path d="M0,0 L0,6 L9,3 z" fill="#5b6470"/>',
        '  </marker>',
        '  <marker id="feedback" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
        '    <path d="M0,0 L0,6 L9,3 z" fill="#2a7f76"/>',
        '  </marker>',
        '  <style>',
        '    .title{font:600 28px Arial,Helvetica,sans-serif;fill:#1f2937}',
        '    .subtitle{font:14px Arial,Helvetica,sans-serif;fill:#5b6470}',
        '    .stage{font:600 16px Arial,Helvetica,sans-serif;fill:#1f2937}',
        '    .detail{font:13px Arial,Helvetica,sans-serif;fill:#374151}',
        '    .index{font:600 13px Arial,Helvetica,sans-serif;fill:#ffffff}',
        '    .edge-label{font:12px Arial,Helvetica,sans-serif;fill:#2a7f76}',
        '    .legend{font:12px Arial,Helvetica,sans-serif;fill:#4b5563}',
        '    .note{font:12px Arial,Helvetica,sans-serif;fill:#6b7280}',
        '  </style>',
        '</defs>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 42, title, "title"),
        svg_text(width / 2, 69, subtitle, "subtitle"),
    ]

    centers = []
    for index in range(len(stages)):
        y = top + index * stage_step
        centers.append((width / 2, y + box_h / 2))
        if index:
            previous_y = top + (index - 1) * stage_step + box_h
            out.append(
                f'<path d="M {width / 2} {previous_y + 5} L {width / 2} {y - 10}" '
                'fill="none" stroke="#5b6470" stroke-width="2" marker-end="url(#arrow)"/>'
            )

    for index, stage in enumerate(stages, start=1):
        y = top + (index - 1) * stage_step
        fill, stroke = COLORS[stage["color"]]
        out.extend(
            [
                f'<rect x="{box_x}" y="{y}" width="{box_w}" height="{box_h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>',
                f'<circle cx="{box_x - 37}" cy="{y + box_h / 2}" r="17" fill="{stroke}"/>',
                svg_text(box_x - 37, y + box_h / 2 + 5, str(index), "index"),
                svg_text(width / 2, y + 31, stage["title"], "stage"),
                svg_text(width / 2, y + 55, stage["detail"], "detail"),
            ]
        )

    if feedback:
        start_index, end_index, label = feedback
        start_y = centers[start_index][1]
        end_y = centers[end_index][1]
        out.append(
            f'<path d="M {box_x + box_w} {start_y} C 842 {start_y}, 842 {end_y}, {box_x + box_w} {end_y}" '
            'fill="none" stroke="#2a7f76" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#feedback)"/>'
        )
        out.append(svg_text(836, (start_y + end_y) / 2 - 8, label, "edge-label", "end"))

    legend = [
        ("blue", "hardware / input"),
        ("violet", "perception"),
        ("teal", "state / data"),
        ("amber", "planning"),
        ("red", "safety"),
        ("graphite", "actuation"),
    ]
    x = 92
    for color, label in legend:
        _, stroke = COLORS[color]
        out.append(f'<circle cx="{x}" cy="{legend_y}" r="6" fill="{stroke}"/>')
        out.append(svg_text(x + 12, legend_y + 4, label, "legend", "start"))
        x += 124
    out.append(svg_text(width / 2, legend_y + 40, note, "note"))
    out.append('</svg>')

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / filename).write_text("\n".join(out) + "\n", encoding="utf-8")


DIAGRAMS = [
    (
        "system-overview.svg",
        "PiPER Active RGB-D Scanning Architecture",
        "A vertical view of mission admission, perception, planning, guarded arm execution and reconstruction.",
        [
            node("Mission request", "Local GUI or tracked-robot gateway supplies a label and rough target pose", "blue"),
            node("MissionEngine", "Owns admission, readiness barriers, phase progression, recovery and shutdown", "teal"),
            node("Eye-in-hand sensing", "Intel RealSense L515 publishes RGB, aligned depth, confidence, intrinsics and TF", "blue"),
            node("Target and scene perception", "GroundingDINO, SAM2, target geometry, obstacles, quality and occlusion evidence", "violet"),
            node("Measured target state", "A stable base-frame target and accepted RGB-D evidence define the scan model", "teal"),
            node("Next-best-view planning", "Candidate rays are ranked by measured coverage, novelty and reachability", "amber"),
            node("Exact motion qualification", "The isolated Tesseract worker checks IK, collision and complete paths", "amber"),
            node("Guarded execution", "The executor validates freshness and safety before publishing autonomous joints", "red"),
            node("PiPER actuation and feedback", "The driver owns CAN, motor enable/disable, MoveJ timing and six-axis health", "graphite"),
            node("Settled capture and reconstruction", "Validated observations update coverage and feed offline TSDF / bounded GICP", "teal"),
        ],
        "The tracked base provides the mount, request and pose snapshot; this repository does not command chassis motion.",
        (9, 5, "accepted evidence updates NBV"),
    ),
    (
        "perception-pipeline.svg",
        "Target Perception and Geometric State",
        "How an open-label target becomes validated target, obstacle and occlusion evidence.",
        [
            node("Synchronized L515 streams", "RGB, aligned depth, confidence, CameraInfo and camera-frame timing", "blue"),
            node("Open-label acquisition", "GroundingDINO grounds the requested object in a fresh settled frame", "violet"),
            node("Dense target persistence", "SAM2 refines and propagates the selected target mask over time", "violet"),
            node("Depth qualification", "Mask-aware depth selection rejects unreliable, clipped or unsupported geometry", "violet"),
            node("Robot-frame target state", "Hand-eye calibration and TF express the measured target in base_link", "teal"),
            node("Scene and occlusion evidence", "Obstacle instances, quality, tracking health and occlusion state are published", "teal"),
        ],
        "Heavy AI workers remain isolated from the ROS 2 Foxy Python environment and have no motion-command authority.",
    ),
    (
        "viewpoint-planning-pipeline.svg",
        "Active Viewpoint and Motion Planning",
        "How measured geometry becomes a reachable, collision-qualified camera motion.",
        [
            node("Accepted observation history", "Only validated RGB-D, masks and capture poses contribute scan evidence", "teal"),
            node("Target envelope and measured surface", "Capture-bound geometry defines object scale, centre and visible coverage", "teal"),
            node("Candidate camera rays", "Configurable target-centred sector, hemisphere or sphere populations are generated", "amber"),
            node("Next-best-view ranking", "Marginal measured coverage, angular novelty and travel rank surviving candidates", "amber"),
            node("Reachability shortlist", "Capability-map evidence removes obviously unreachable camera candidates", "amber"),
            node("Tesseract exact qualification", "Fresh joints, limits, scene and hashes bind exact IK, collision and path checks", "amber"),
            node("Hash-bound execution plan", "The bridge returns a plan whose identity can be independently authorized", "red"),
        ],
        "Predicted geometry guides planning but never becomes measured reconstruction evidence.",
        (6, 2, "replan after rejection"),
    ),
    (
        "execution-safety-pipeline.svg",
        "Guarded Motion Execution and Recovery",
        "The command-authority chain from a qualified plan to a settled capture or safe shutdown.",
        [
            node("Qualified plan and mission identity", "Exact hashes bind the task, target, start state, limits, scene and path", "amber"),
            node("Authorization gates", "Plan freshness, calibration, target, obstacles, motor state and ownership are checked", "red"),
            node("Sole autonomous joint publisher", "scan_viewpoint_executor is the only production node allowed to send scan targets", "red"),
            node("PiPER driver and SocketCAN", "piper_ctrl_single_node owns enable/disable, watchdogs, command timing and feedback", "graphite"),
            node("Trajectory monitoring", "All-six-axis following, faults, endpoint convergence and a settled hold are proved", "red"),
            node("Capture or bounded recovery", "Accept the settled view, retry, reacquire, replan, hold or enter terminal home", "teal"),
            node("Safe terminal state", "Pre-home, rough home, storage wrist, feedback-confirmed disable and child cleanup", "graphite"),
        ],
        "Tesseract proposes motion; the executor authorizes it; the driver retains physical motor and CAN authority.",
        (5, 1, "bounded retry / recovery"),
    ),
    (
        "capture-reconstruction-pipeline.svg",
        "Multi-view Capture and 3D Reconstruction",
        "How settled observations become an immutable dataset and a target-only 3D product.",
        [
            node("Settled capture trigger", "The executor requests evidence only after motion convergence and safety checks", "red"),
            node("Synchronized observation", "RGB, raw depth, confidence depth, mask, intrinsics, joints and capture-time TF", "blue"),
            node("Transactional dataset admission", "Fresh GOOD target and acceptable occlusion evidence gate atomic persistence", "teal"),
            node("Measured coverage update", "Accepted views update the target surface model and scan-completion decision", "teal"),
            node("Pose registration", "Captured poses can be refined by bounded target GICP without changing provenance", "amber"),
            node("Target-only fusion", "Open3D TSDF integrates admitted masked RGB-D measurements", "teal"),
            node("Reconstruction products", "Raw and cleaned meshes, coloured clouds, quality metrics and provenance reports", "blue"),
        ],
        "Reconstruction starts from immutable accepted captures after the arm and, when applicable, tracked base are safe.",
        (3, 0, "request another view"),
    ),
    (
        "hardware-topology.svg",
        "Robot Hardware and Integration Boundaries",
        "The physical platform, current PiPER/L515 scan hardware and optional enclosure-mounted sensors.",
        [
            node("Bunker Pro 2 tracked platform", "Carries the enclosure, sensor station and arm mounting structure", "graphite"),
            node("Enclosure v4", "Protects compute and power hardware; CAD includes panels, frame, battery and sensor mounts", "blue"),
            node("PiPER arm mounting tree", "base_link to arm_base_link mount with the piper_base_link gateway identity frame", "graphite"),
            node("PiPER 6-DOF arm", "USB-CAN / SocketCAN connects the ROS driver to six-axis motor hardware", "graphite"),
            node("Eye-in-hand L515 assembly", "The qualified holder and Intel RealSense L515 provide current RGB-D sensing", "blue"),
            node("Optional ZED and LiDAR mounts", "CAD provides mechanical provision; their data is not used by this scan runtime", "gray"),
        ],
        "Tracked-base drive, brake authority and repositioning are outside the current PiPER_arm command boundary.",
    ),
]


def main():
    for args in DIAGRAMS:
        render_pipeline(*args)


if __name__ == "__main__":
    main()
