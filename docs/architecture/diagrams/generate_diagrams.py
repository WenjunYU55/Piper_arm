#!/usr/bin/env python3
"""Generate evidence-backed SVG architecture diagrams for the PiPER system."""

from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "assets" / "readme" / "architecture"

COLORS = {
    "input": ("#EAF3FF", "#2563A6"),
    "perception": ("#F2EAFF", "#7851A9"),
    "state": ("#E5F7F3", "#147D70"),
    "planning": ("#FFF3D9", "#B77900"),
    "safety": ("#FDE9E7", "#B33A3A"),
    "actuation": ("#EEF1F4", "#46515E"),
    "optional": ("#F7F7F8", "#929AA5"),
    "external": ("#EFF6FF", "#3B6EA5"),
}

EDGE_COLORS = {
    "data": "#52606D",
    "control": "#2563A6",
    "feedback": "#147D70",
    "command": "#B33A3A",
    "optional": "#929AA5",
}


def esc(value):
    return escape(str(value), quote=True)


def text(x, y, value, css_class, anchor="middle"):
    return (
        f'<text x="{x}" y="{y}" class="{css_class}" '
        f'text-anchor="{anchor}">{esc(value)}</text>'
    )


def multiline(x, y, lines, css_class="detail", anchor="middle", step=19):
    result = [
        f'<text x="{x}" y="{y}" class="{css_class}" text-anchor="{anchor}">'
    ]
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else step
        result.append(
            f'<tspan x="{x}" dy="{dy}">{esc(line)}</tspan>'
        )
    result.append("</text>")
    return "".join(result)


def point(node, side):
    x, y, width, height = (
        node["x"], node["y"], node["w"], node["h"]
    )
    return {
        "top": (x + width / 2, y),
        "bottom": (x + width / 2, y + height),
        "left": (x, y + height / 2),
        "right": (x + width, y + height / 2),
    }[side]


def box(node):
    fill, stroke = COLORS[node["color"]]
    x, y, width, height = (
        node["x"], node["y"], node["w"], node["h"]
    )
    padding = node.get("padding", 24)
    content_x = x + padding
    title_y = y + node.get("title_offset", 35)
    detail_y = y + node.get("detail_offset", 66)
    if node.get("stack_status"):
        title_y = y + 57
        detail_y = y + 88
    result = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'rx="14" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>',
        f'<rect x="{x}" y="{y}" width="7" height="{height}" '
        f'rx="3.5" fill="{stroke}" opacity="0.9"/>',
        text(content_x, title_y, node["title"], "node-title", "start"),
        multiline(
            content_x,
            detail_y,
            node.get("lines", ()),
            "detail",
            anchor="start",
            step=node.get("line_step", 18),
        ),
    ]
    status = node.get("status")
    if status:
        badge_width = max(96, 8.2 * len(status) + 22)
        result.extend([
            f'<rect x="{content_x}" y="{y + 13}" '
            f'width="{badge_width}" height="23" rx="11.5" fill="{stroke}"/>',
            text(
                content_x + badge_width / 2,
                y + 29,
                status,
                "badge",
            ),
        ])
    return result


def lane(y, height, title, subtitle="", x=90, width=1140):
    result = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" '
        'fill="#FAFBFC" stroke="#D7DDE5" stroke-width="1.2"/>',
        f'<path d="M {x} {y + 70} H {x + width}" stroke="#D7DDE5" '
        'stroke-width="1.2"/>',
        text(x + 24, y + 30, title, "lane-title", "start"),
    ]
    if subtitle:
        result.append(text(x + 24, y + 54, subtitle, "lane-subtitle", "start"))
    return result


def edge(nodes, item, labels_only=False):
    source = item.get(
        "src_at",
        point(nodes[item["src"]], item.get("src_side", "bottom")),
    )
    target = item.get(
        "dst_at",
        point(nodes[item["dst"]], item.get("dst_side", "top")),
    )
    points = [source] + list(item.get("via", ())) + [target]
    color = EDGE_COLORS[item.get("kind", "data")]
    dashed = item.get("kind") in ("feedback", "optional")
    marker = item.get("kind", "data")
    polyline = " ".join(f"{x},{y}" for x, y in points)
    label = item.get("label")
    if labels_only:
        if not label:
            return []
        label_x, label_y = item.get(
            "label_at",
            ((source[0] + target[0]) / 2, (source[1] + target[1]) / 2 - 7),
        )
        anchor = item.get("label_anchor", "middle")
        label_width = max(46, len(label) * 7.6 + 18)
        if anchor == "start":
            background_x = label_x - 7
        elif anchor == "end":
            background_x = label_x - label_width + 7
        else:
            background_x = label_x - label_width / 2
        return [
            f'<rect x="{background_x}" y="{label_y - 17}" '
            f'width="{label_width}" height="23" rx="5" '
            'class="edge-label-bg"/>',
            text(
            label_x,
            label_y,
            label,
            "edge-label",
            anchor,
            ),
        ]
    return [
        f'<polyline points="{polyline}" fill="none" stroke="{color}" '
        f'stroke-width="{item.get("width", 2.2)}" '
        'stroke-linecap="round" stroke-linejoin="round" '
        + ('stroke-dasharray="8 6" ' if dashed else "")
        + f'marker-end="url(#{marker}-arrow)"/>'
    ]


def svg_header(width, height, title, subtitle):
    result = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f'<title>{esc(title)}</title>',
        f'<desc>{esc(subtitle)}</desc>',
        "<defs>",
    ]
    for name, color in EDGE_COLORS.items():
        result.extend([
            f'<marker id="{name}-arrow" markerWidth="10" markerHeight="10" '
            'refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
            f'<path d="M0,0 L0,6 L9,3 z" fill="{color}"/>',
            "</marker>",
        ])
    result.extend([
        "<style>",
        ".title{font:700 40px Arial,Helvetica,sans-serif;fill:#17212B}",
        ".subtitle{font:18px Arial,Helvetica,sans-serif;fill:#52606D}",
        ".lane-title{font:700 20px Arial,Helvetica,sans-serif;fill:#24303C}",
        ".lane-subtitle{font:14px Arial,Helvetica,sans-serif;fill:#687684}",
        ".node-title{font:700 21px Arial,Helvetica,sans-serif;fill:#17212B}",
        ".detail{font:16.5px Arial,Helvetica,sans-serif;fill:#344250}",
        ".edge-label-bg{fill:#fff;fill-opacity:.96}",
        ".edge-label{font:700 14.5px Arial,Helvetica,sans-serif;fill:#344250}",
        ".badge{font:700 12.5px Arial,Helvetica,sans-serif;fill:#fff;letter-spacing:.35px}",
        ".legend{font:14.5px Arial,Helvetica,sans-serif;fill:#46515E}",
        ".note{font:15px Arial,Helvetica,sans-serif;fill:#52606D}",
        "</style>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        text(width / 2, 49, title, "title"),
        text(width / 2, 78, subtitle, "subtitle"),
    ])
    return result


def legend(y, width=1400):
    items = [
        ("data", "data / evidence"),
        ("control", "mission control"),
        ("feedback", "feedback / retry"),
        ("command", "motor command"),
        ("optional", "branch-only / optional"),
    ]
    if width >= 1600:
        x, spacing = 270, 260
    elif width >= 1300:
        x, spacing = 170, 225
    else:
        x, spacing = 65, 205
    result = []
    for kind, label in items:
        color = EDGE_COLORS[kind]
        result.extend([
            f'<line x1="{x}" y1="{y}" x2="{x + 38}" y2="{y}" '
            f'stroke="{color}" stroke-width="3" '
            + ('stroke-dasharray="8 6" ' if kind in ("feedback", "optional") else "")
            + f'marker-end="url(#{kind}-arrow)"/>',
            text(x + 48, y + 4, label, "legend", "start"),
        ])
        x += spacing
    return result


def write_svg(filename, body):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / filename).write_text("\n".join(body) + "\n", encoding="utf-8")




def render_system():
    """Render the whole system as a native two-column vertical map.

    Long feedback paths use dedicated outer buses.  The layout deliberately
    avoids the previous scaled 1400 px coordinate system, which made the
    canvas larger without making the information easier to read.
    """
    width, height = 1320, 5050
    title = "PiPER Active-View Scanning: Detailed System Architecture"
    subtitle = "One command path, accepted-only evidence, and explicit feedback from request to reconstruction"
    nodes = {
        "operator": dict(x=130, y=230, w=500, h=160, title="Operator, native GUI and E-stop", lines=("choose target and next-mission settings", "review rays, status and diagnostics", "start, observe or cancel", "emergency stop remains independent"), color="external"),
        "gateway": dict(x=690, y=230, w=500, h=160, title="Tracked-robot gateway", lines=("SCAN_3D request + rough target", "odom -> piper_base_link snapshot", "task heartbeat + home report", "base remains stationary and braked"), color="external"),
        "config": dict(x=220, y=415, w=880, h=155, title="Frozen mission contract", lines=("target profile + scan/NBV policy", "planner backend + speed + motion opt-ins", "validated before admission; immutable in flight", "canonical mission SHA-256 binds every downstream result"), color="state"),
        "coordinator": dict(x=130, y=675, w=500, h=165, title="RunTargetScan coordinator", lines=("queues and admits one immutable goal", "binds task, target and backend identity", "publishes typed action feedback", "returns mission result + dataset reference"), color="state"),
        "supervisor": dict(x=690, y=675, w=500, h=175, title="ProcessSupervisor", lines=("owns one exact process generation", "starts driver -> vision -> calibration", "starts exactly one selected planner", "monitors heartbeats and bounded cleanup", "never adopts unrelated processes"), color="safety"),
        "mission": dict(x=220, y=875, w=880, h=210, title="MissionEngine lifecycle", lines=("preflight -> enable/hold -> startup wrist", "rough home -> bounded target acquisition", "workflow ready -> one-view closed loop", "terminal PRE_HOME -> ROUGH_HOME -> STORAGE_WRIST", "prove all-six disable and revoke authorization", "motor-authority loss permits no new command"), color="safety"),
        "camera": dict(x=130, y=1200, w=500, h=160, title="Eye-in-hand L515 evidence", lines=("RGB + aligned/native depth", "4-bit confidence + CameraInfo", "source timestamps at 30 Hz", "settled capture uses new synchronized frames"), color="input"),
        "clock": dict(x=690, y=1200, w=500, h=185, title="Timing, TF and hand-eye calibration", lines=("camera timestamp watchdog", "accepted optical-frame transform", "base_link <-> optical frame", "fresh joints and camera pose", "fault may restart only owned vision work"), color="input"),
        "ai": dict(x=130, y=1395, w=500, h=185, title="Isolated CUDA perception", lines=("GroundingDINO acquire / correlated refresh", "SAM 2 seed + live mask propagation", "permission-bounded request/result spools", "current-generation readiness and heartbeat", "ROS-free workers; no motor interface"), color="perception"),
        "geometry": dict(x=690, y=1395, w=500, h=185, title="ROS perception geometry", lines=("mask -> detection -> qualified depth", "Target3D + Kalman target tracker", "target cloud, landmark and envelope", "obstacles + quality + occlusion", "freshness and correlation metadata"), color="perception"),
        "target_state": dict(x=130, y=1595, w=500, h=125, title="Measured target state", lines=("stable base-frame target + covariance", "TRACKING / LOW_CONFIDENCE / LOST"), color="state"),
        "scene_state": dict(x=690, y=1595, w=500, h=125, title="Scene safety evidence", lines=("obstacles, target envelope and framing", "occlusion, quality and tracking health"), color="state"),
        "coverage": dict(x=130, y=1905, w=500, h=210, title="Accepted-only scan memory", lines=("schema-2 captures + achieved FK", "5 mm measured coverage grid", "unknown / observed target surface", "accepted generation is explicit", "rejected views never add coverage"), color="state"),
        "nbv": dict(x=690, y=1905, w=500, h=210, title="Closed-loop NBV and ray policy", lines=("legacy, voxel_nbv or frozen ray_nbv", "seed from actual current FK", "rank marginal information before travel", "direction novelty + bounded standoff", "one accepted view per generation", "complete or exhaust the safe frontier"), color="planning"),
        "reach": dict(x=220, y=2140, w=880, h=165, title="Command-free prequalification", lines=("workspace/capability evidence removes hard culls", "target-facing aim with bounded orientation fallback", "shortlist at most 12 voxel points or 6 ray directions", "no motor command is possible at this stage"), color="planning"),
        "bridge": dict(x=220, y=2415, w=880, h=210, title="MotionPlannerBridge (ROS 2 Foxy)", lines=("freezes tesseract or curobo for the mission", "requires matching worker generation and readiness", "snapshots joints, limits, target, camera and scene", "binds robot/world/calibration hashes and requested views", "writes private command-free schema-v5 request", "publishes correlated status, blockers and provenance"), color="planning"),
        "tesseract": dict(x=130, y=2650, w=500, h=205, title="Tesseract 0.35 worker", lines=("ROS-free exact IK, collision and path", "configured moving + fixed meshes", "qualified baseline and startup recovery", "atomic request/response spool"), color="planning", status="MAIN + BRANCH"),
        "curobo": dict(x=690, y=2650, w=500, h=205, title="cuRobo 0.7.8 MotionGen worker", lines=("Python 3.10 + CUDA; ROS-free", "exact fixed Bunker meshes", "167-sphere articulated approximation", "hardware_qualified=false; motion blocked", "no automatic Tesseract fallback"), color="optional", status="BRANCH ONLY"),
        "transport": dict(x=220, y=2880, w=880, h=210, title="Validated planner transport", lines=("main: TesseractPlan", "integration branch: generic MotionPlan", "backend, request, plan and trajectory hashes", "complete timed six-joint trajectory + metrics", "collision qualification and rejection diagnostics", "planner output remains a proposal only"), color="state"),
        "normalize": dict(x=130, y=3200, w=500, h=165, title="Common plan normalization", lines=("six finite joints in canonical order", "20 Hz schedule + 0.05 rad step", "speed-scaled MoveJ limits", "fresh-start and trajectory hashes"), color="safety"),
        "authorize": dict(x=690, y=3200, w=500, h=165, title="PlanAuthorizer", lines=("live mission + backend authorization", "exact plan and trajectory identity", "TTL, target drift and dependencies", "fresh complete-path revalidation"), color="safety"),
        "executor": dict(x=130, y=3390, w=500, h=220, title="scan_viewpoint_executor", lines=("sole autonomous joint publisher", "TrajectoryRunner owns the 20 Hz schedule", "runtime freshness + scene gates", "following error, timeout and settle proof", "hold, refresh, replan or recover", "publishes ScanExecutionStatus"), color="safety", status="COMMAND OWNER"),
        "driver": dict(x=690, y=3390, w=500, h=220, title="PiPER driver / SocketCAN", lines=("/joint_ctrl_single -> MoveJ", "SocketCAN at 1 Mbps", "enable/disable + motor watchdogs", "six motors + coherent joint feedback"), color="actuation", status="CAN OWNER"),
        "settle": dict(x=130, y=3780, w=500, h=180, title="Settled capture request", lines=("actual FK + target-facing aim", "stationary joints + healthy clock", "fresh target/scene evidence", "executor enters CAPTURING_RGBD"), color="safety"),
        "burst": dict(x=690, y=3780, w=500, h=190, title="Confidence-qualified RGB-D burst", lines=("exact mask/RGB stamp identity", "20 new depth + confidence frames", "grade >=8 and >=0.50 support", "median depth + capture-time TF", "target-component ambiguity gate"), color="input"),
        "admission": dict(x=130, y=3985, w=500, h=190, title="Capture admission", lines=("GOOD target + fresh diagnostics", "CLEAR or correlated occlusion proof", "matching achieved FK and plan provenance", "atomic artifacts + manifest hashes", "decide ACCEPT, RETRY or REJECT"), color="safety"),
        "dataset": dict(x=690, y=3985, w=500, h=190, title="Immutable schema-2 dataset", lines=("RGB, raw/qualified depth and mask", "intrinsics, joints and capture-time TF", "view/plan provenance + quality", "target support and model seed", "transactional manifest SHA-256"), color="state"),
        "rejected": dict(x=130, y=4200, w=500, h=145, title="Retry / rejected observation", lines=("hold achieved FK; add no coverage", "one correlated heavy refresh", "then exclude and replan from actual pose"), color="safety"),
        "history": dict(x=690, y=4200, w=500, h=145, title="Accepted history generation", lines=("commit achieved pose + evidence", "rebuild measured coverage", "request the next qualified view"), color="state"),
        "recovery": dict(x=130, y=4515, w=500, h=175, title="Safe terminal recovery", lines=("cancel, completion or bounded failure", "PRE_HOME -> ROUGH_HOME -> STORAGE_WRIST", "disable all six and revoke authorization", "stop only mission-owned child processes"), color="safety"),
        "base_home": dict(x=690, y=4515, w=500, h=175, title="Tracked-base correlation", lines=("arm result may wait for base home", "bind task, job and manifest identity", "gateway accepts idempotent report", "this repository sends no chassis command"), color="external"),
        "reconstruct": dict(x=220, y=4715, w=880, h=190, title="Offline reconstruction", lines=("admit immutable inputs and verify every hash", "target-only Open3D TSDF; 3 mm voxels by default", "optional bounded GICP / scene pose graph", "mesh + cloud + quality metrics + provenance", "failure is separate and never rewrites mission safety"), color="state"),
    }
    for node in nodes.values():
        node.setdefault("line_step", 23)
        node.setdefault("title_offset", 37)
        node.setdefault("detail_offset", 70)
        if node.get("status"):
            node["stack_status"] = True

    edges = [
        dict(src="operator", dst="config", kind="control", src_at=(380, 390), dst_at=(480, 415), via=((380, 402), (480, 402))),
        dict(src="gateway", dst="config", kind="control", src_at=(940, 390), dst_at=(840, 415), via=((940, 402), (840, 402))),
        dict(src="config", dst="coordinator", kind="data", label="frozen goal + hash", src_at=(660, 570), dst_at=(380, 675), via=((660, 655), (380, 655)), label_at=(540, 578)),
        dict(src="coordinator", dst="supervisor", kind="control", src_side="right", dst_side="left"),
        dict(src="supervisor", dst="mission", kind="control", src_at=(940, 850), dst_at=(940, 875)),
        dict(src="mission", dst="camera", kind="control", label="ordered acquisition", src_at=(380, 1085), dst_at=(380, 1200), label_at=(380, 1102)),
        dict(src="camera", dst="clock", kind="data", src_side="right", dst_side="left"),
        dict(src="camera", dst="ai", kind="data", label="RGB-D frames", src_at=(380, 1350), dst_at=(380, 1395)),
        dict(src="clock", dst="geometry", kind="data", label="TF + camera health", src_at=(940, 1370), dst_at=(940, 1395)),
        dict(src="ai", dst="geometry", kind="data", src_side="right", dst_side="left"),
        dict(src="geometry", dst="target_state", kind="data", label="qualified Target3D", src_at=(850, 1570), dst_at=(380, 1595), via=((850, 1582), (380, 1582)), label_at=(615, 1578)),
        dict(src="geometry", dst="scene_state", kind="data", label="quality + obstacles", src_at=(1030, 1570), dst_at=(940, 1595), via=((1030, 1582), (940, 1582)), label_at=(985, 1578)),
        dict(src="coverage", dst="nbv", kind="data", src_side="right", dst_side="left"),
        dict(src="target_state", dst="nbv", kind="data", label="fresh target", src_at=(380, 1720), dst_at=(940, 1905), via=((380, 1765), (1210, 1765), (1210, 1885), (940, 1885)), label_at=(1110, 1758), label_anchor="end"),
        dict(src="scene_state", dst="reach", kind="data", label="safe scene", src_at=(940, 1720), dst_at=(940, 2140), via=((1220, 1720), (1220, 2120), (940, 2120)), label_at=(1208, 2030), label_anchor="end"),
        dict(src="nbv", dst="reach", kind="data", label="ranked shortlist", src_at=(940, 2115), dst_at=(760, 2140), via=((940, 2127), (760, 2127)), label_at=(850, 2123)),
        dict(src="reach", dst="bridge", kind="data", label="candidate views only", src_at=(660, 2305), dst_at=(660, 2415), label_at=(660, 2316)),
        dict(src="bridge", dst="tesseract", kind="data", label="main / selected backend", src_at=(520, 2625), dst_at=(380, 2650), via=((520, 2637), (380, 2637)), label_at=(450, 2633)),
        dict(src="bridge", dst="curobo", kind="optional", label="branch-only backend", src_at=(800, 2625), dst_at=(940, 2650), via=((800, 2637), (940, 2637)), label_at=(870, 2633)),
        dict(src="tesseract", dst="transport", kind="data", label="validated response", src_at=(380, 2855), dst_at=(520, 2880), via=((380, 2867), (520, 2867)), label_at=(450, 2863)),
        dict(src="curobo", dst="transport", kind="optional", label="MotionGen response", src_at=(940, 2855), dst_at=(800, 2880), via=((940, 2867), (800, 2867)), label_at=(870, 2863)),
        dict(src="transport", dst="normalize", kind="data", label="proposal only", src_at=(520, 3090), dst_at=(380, 3200), via=((520, 3180), (380, 3180)), label_at=(450, 3100)),
        dict(src="normalize", dst="authorize", kind="data", src_side="right", dst_side="left"),
        dict(src="authorize", dst="executor", kind="control", label="exact approval", src_at=(940, 3365), dst_at=(380, 3390), via=((940, 3377), (380, 3377)), label_at=(660, 3373)),
        dict(src="executor", dst="driver", kind="command", src_at=(630, 3470), dst_at=(690, 3470)),
        dict(src="driver", dst="executor", kind="feedback", src_at=(690, 3550), dst_at=(630, 3550)),
        dict(src="executor", dst="settle", kind="control", label="settled viewpoint", src_at=(380, 3610), dst_at=(380, 3780), label_at=(380, 3675)),
        dict(src="settle", dst="burst", kind="control", src_side="right", dst_side="left"),
        dict(src="burst", dst="admission", kind="data", label="qualified observation", src_at=(940, 3960), dst_at=(380, 3985), via=((940, 3972), (380, 3972)), label_at=(660, 3968)),
        dict(src="admission", dst="dataset", kind="data", src_side="right", dst_side="left"),
        dict(src="admission", dst="rejected", kind="feedback", label="RETRY / REJECT", src_at=(380, 4175), dst_at=(380, 4200)),
        dict(src="dataset", dst="history", kind="data", label="accepted generation", src_at=(940, 4175), dst_at=(940, 4200)),
        dict(src="history", dst="coverage", kind="feedback", label="F4 accepted", src_side="left", dst_side="left", via=((75, 4272), (75, 2010), (130, 2010)), label_at=(92, 4380), label_anchor="start"),
        dict(src="rejected", dst="ai", kind="feedback", label="F3 reacquire", src_side="left", dst_side="left", via=((45, 4272), (45, 1482), (130, 1482)), label_at=(62, 4355), label_anchor="start"),
        dict(src="rejected", dst="nbv", kind="feedback", src_at=(270, 4200), dst_side="left", via=((270, 4185), (60, 4185), (60, 2010), (690, 2010))),
        dict(src="bridge", dst="nbv", kind="feedback", label="retire failed ray", src_side="left", dst_side="left", via=((55, 2520), (55, 1980), (690, 1980)), label_at=(72, 2498), label_anchor="start"),
        dict(src="scene_state", dst="executor", kind="feedback", label="F2 live scene", src_side="right", dst_side="right", via=((1240, 1657), (1240, 3500), (630, 3500)), label_at=(1224, 3300), label_anchor="end"),
        dict(src="driver", dst="bridge", kind="feedback", label="F1 joint limits", src_side="right", dst_side="right", via=((1270, 3500), (1270, 2510), (1100, 2510)), label_at=(1254, 3100), label_anchor="end"),
        dict(src="driver", dst="mission", kind="feedback", label="motor / disable status", src_at=(1190, 3450), dst_side="right", via=((1290, 3450), (1290, 970), (1100, 970)), label_at=(1274, 1100), label_anchor="end"),
        dict(src="mission", dst="recovery", kind="control", label="complete / cancel / fail", src_side="right", dst_side="right", via=((1250, 980), (1250, 4495), (630, 4495)), label_at=(1234, 4412), label_anchor="end"),
        dict(src="recovery", dst="base_home", kind="control", src_side="right", dst_side="left"),
        dict(src="base_home", dst="reconstruct", kind="control", label="correlated base-home report", src_at=(940, 4690), dst_at=(800, 4715), via=((940, 4702), (800, 4702)), label_at=(870, 4698)),
        dict(src="dataset", dst="reconstruct", kind="data", label="immutable captures", src_side="right", dst_side="right", via=((1215, 4080), (1215, 4810), (1100, 4810)), label_at=(1198, 4450), label_anchor="end"),
    ]

    body = svg_header(width, height, title, subtitle)
    body.extend([
        "<style>",
        ".title{font-size:40px}",
        ".subtitle{font-size:18px}",
        ".legend{font-size:14px}",
        ".system-map .lane-title{font-size:24px}",
        ".system-map .lane-subtitle{font-size:16px}",
        ".system-map .node-title{font-size:21px}",
        ".system-map .detail{font-size:16.5px}",
        ".system-map .edge-label{font-size:15px}",
        ".system-map .badge{font-size:12.5px}",
        ".system-map .note{font-size:15px}",
        "</style>",
    ])
    body.extend(legend(112, width))
    body.append('<g class="system-map">')
    body.extend(lane(140, 420, "1  Request and freeze mission intent", "External task ownership; the tracked base is never commanded here."))
    body.extend(lane(585, 500, "2  Orchestrate one owned mission", "Admission, process generations and terminal lifecycle remain correlated."))
    body.extend(lane(1110, 680, "3  Produce measured target and scene evidence", "Timestamped RGB-D, calibration and isolated AI feed ROS safety evidence."))
    body.extend(lane(1815, 485, "4  Select the next useful and feasible view", "Accepted captures alone change measured coverage."))
    body.extend(lane(2325, 760, "5  Propose motion through one frozen backend", "Tesseract is on main; cuRobo is branch-only and hardware-unqualified."))
    body.extend(lane(3110, 555, "6  Authorize and execute through one command owner", "All planner backends traverse the same fail-closed physical boundary."))
    body.extend(lane(3690, 710, "7  Admit or reject the settled observation", "Acceptance, retry and rejection deliberately have different feedback effects."))
    body.extend(lane(4425, 500, "8  Recover safely, correlate base home and reconstruct", "Offline reconstruction consumes immutable evidence after the required safe state."))
    for item in edges:
        body.extend(edge(nodes, item))
    for node in nodes.values():
        body.extend(box(node))
    for item in edges:
        body.extend(edge(nodes, item, labels_only=True))
    body.extend([
        text(width / 2, 5010, "Red is the sole motor-command edge. Dashed green paths are measured feedback, retry or replanning.", "note"),
        "</g>",
        "</svg>",
    ])
    write_svg("system-overview.svg", body)


def render_flow(filename, title, subtitle, nodes, edges, height, footer):
    # The README displays focused diagrams at 900 px.  Widening the source
    # geometry without scaling the type gives every block a comfortable text
    # measure while keeping labels above 11 screen pixels on GitHub.
    source_width = 1100
    x_scale = 1.10
    x_margin = 35
    width = int(source_width * x_scale + 2 * x_margin)
    for node in nodes.values():
        node["x"] = x_margin + node["x"] * x_scale
        node["w"] *= x_scale
        node.setdefault("line_step", 23)
        if node.get("status"):
            node["stack_status"] = True
    for item in edges:
        if "via" in item:
            item["via"] = tuple(
                (x_margin + x * x_scale, y) for x, y in item["via"]
            )
        if "label_at" in item:
            label_x, label_y = item["label_at"]
            label_x = x_margin + label_x * x_scale
            item["label_at"] = (label_x, label_y)
            if label_x < 145:
                item.setdefault("label_anchor", "start")
            elif label_x > width - 145:
                item.setdefault("label_anchor", "end")
    body = svg_header(width, height, title, subtitle)
    body.extend(legend(108, width))
    for item in edges:
        body.extend(edge(nodes, item))
    for node_value in nodes.values():
        body.extend(box(node_value))
    for item in edges:
        body.extend(edge(nodes, item, labels_only=True))
    body.extend([
        text(width / 2, height - 24, footer, "note"),
        "</svg>",
    ])
    write_svg(filename, body)


def render_perception():
    nodes = {
        "l515": dict(x=380, y=155, w=340, h=145, title="L515 synchronized evidence", lines=("RGB + aligned/native depth + confidence", "CameraInfo, timestamps and camera-frame TF", "timestamp watchdog must remain healthy"), color="input"),
        "heavy": dict(x=80, y=365, w=300, h=165, title="GroundingDINO heavy worker", lines=("open-label acquisition / correlated refresh", "local-cache model readiness", "permission-bounded request/result spool"), color="perception"),
        "sam": dict(x=410, y=365, w=280, h=165, title="SAM 2 live worker", lines=("seed from grounded target", "propagate dense target mask", "publish confidence and identity"), color="perception"),
        "depth": dict(x=720, y=365, w=300, h=165, title="Ambiguity-aware depth", lines=("mask is support, not depth proof", "qualified component / confidence gate", "invalid or tied layers fail closed"), color="perception"),
        "target3d": dict(x=380, y=610, w=340, h=150, title="Measured Target3D", lines=("finite camera-frame point from fresh mask/depth", "source-stamped transform to base_link", "invalid evidence publishes an explicit reason"), color="state"),
        "tracker": dict(x=380, y=835, w=340, h=175, title="Timestamped Kalman tracker", lines=("innovation-gated measurement correction", "short outage: LOW_CONFIDENCE prediction", "long outage: LOST and reset", "settled measured lock required for planning"), color="state"),
        "geometry": dict(x=80, y=1085, w=300, h=175, title="Target geometry", lines=("landmark, cloud and envelope", "1 mm refined target cloud", "5 mm NBV evidence grid", "prediction is planning-only"), color="state"),
        "scene": dict(x=410, y=1085, w=280, h=175, title="Scene evidence", lines=("obstacle instance boxes", "camera timestamp and tracking health", "freshness + correlation metadata"), color="state"),
        "quality": dict(x=720, y=1085, w=300, h=175, title="Quality and occlusion", lines=("framing, silhouette and target quality", "CLEAR or labelled occlusion", "person/hand remains a terminal blocker"), color="safety"),
        "consumers": dict(x=300, y=1355, w=500, h=170, title="Consumers with independent gates", lines=("NBV + planner snapshot | executor runtime safety", "settled capture admission | GUI diagnostics", "none of these perception paths owns a motor publisher"), color="planning"),
    }
    edges = [
        dict(src="l515", dst="heavy", kind="data", label="settled RGB", src_side="bottom", dst_side="top"),
        dict(src="l515", dst="sam", kind="data", label="live frames", src_side="bottom", dst_side="top"),
        dict(src="l515", dst="depth", kind="data", label="depth + confidence", src_side="bottom", dst_side="top"),
        dict(src="heavy", dst="sam", kind="data", src_side="right", dst_side="left"),
        dict(src="sam", dst="depth", kind="data", src_side="right", dst_side="left"),
        dict(src="depth", dst="target3d", kind="data", label="qualified point", src_side="bottom", dst_side="right", via=((870, 575), (740, 575), (740, 685))),
        dict(src="target3d", dst="tracker", kind="data", label="measurement", src_side="bottom", dst_side="top"),
        dict(src="tracker", dst="geometry", kind="data", label="stable target", src_side="bottom", dst_side="top"),
        dict(src="tracker", dst="scene", kind="data", label="health + covariance", src_side="bottom", dst_side="top"),
        dict(src="depth", dst="quality", kind="data", label="support / ambiguity", src_side="right", dst_side="right", via=((1050, 447), (1050, 1172), (1020, 1172)), label_at=(1042, 810)),
        dict(src="sam", dst="quality", kind="data", label="mask / detection", src_side="right", dst_side="left", via=((705, 550), (705, 1172), (720, 1172)), label_at=(712, 820)),
        dict(src="geometry", dst="consumers", kind="data", src_side="bottom", dst_side="top"),
        dict(src="scene", dst="consumers", kind="data", src_side="bottom", dst_side="top"),
        dict(src="quality", dst="consumers", kind="data", src_side="bottom", dst_side="top"),
        dict(src="tracker", dst="heavy", kind="feedback", label="LOST / bounded reacquisition", src_side="left", dst_side="left", via=((45, 922), (45, 447), (80, 447)), label_at=(165, 915)),
        dict(src="quality", dst="heavy", kind="feedback", label="one correlated settled refresh", src_side="right", dst_side="right", via=((1060, 1172), (1060, 330), (230, 330), (230, 365)), label_at=(720, 323)),
        dict(src="target3d", dst="sam", kind="feedback", src_side="left", dst_side="left", via=((355, 685), (355, 447), (410, 447))),
    ]
    render_flow(
        "perception-pipeline.svg",
        "Target Perception, Tracking and Reacquisition",
        "The measurement path and the explicit feedback loops used when masks, depth or tracking degrade",
        nodes,
        edges,
        1580,
        "Short prediction gaps never become measured coverage; a recovered fresh measurement must pass the same gates again.",
    )


def render_nbv():
    nodes = {
        "accepted": dict(x=380, y=150, w=340, h=150, title="Accepted schema-2 capture", lines=("manifest-verified target depth/support", "capture-time camera transform + achieved FK", "rejected observations are excluded"), color="state"),
        "coverage": dict(x=380, y=365, w=340, h=160, title="Measured surface coverage", lines=("rebuild at exact accepted generation", "unknown / observed / visible voxels", "completion and novelty evidence"), color="state"),
        "candidate": dict(x=380, y=595, w=340, h=180, title="Candidate policy", lines=("legacy points | voxel_nbv | frozen ray_nbv", "seed from current FK when no model exists", "marginal information before travel", "duplicate directions and hard culls removed"), color="planning"),
        "prequal": dict(x=380, y=845, w=340, h=160, title="Prequalification and shortlist", lines=("workspace / capability evidence", "safe target-facing aim + bounded fallback", "12 voxel points or 6 ray directions max"), color="planning"),
        "planner": dict(x=380, y=1075, w=340, h=150, title="Exact planner backend", lines=("fresh scene, joints, limits and hashes", "first feasible complete collision-qualified path", "structured rejection remains correlated"), color="planning"),
        "execute": dict(x=380, y=1295, w=340, h=160, title="Authorize, execute and settle", lines=("common safety path; one view", "record actual FK even if capture is rejected", "final aim, target drift and framing checks"), color="safety"),
        "decision": dict(x=380, y=1525, w=340, h=150, title="Observation decision", lines=("ACCEPT -> immutable commit", "RETRY -> one same-pose refresh", "REJECT -> exclude and replan"), color="safety"),
        "complete": dict(x=760, y=365, w=280, h=160, title="Completion decision", lines=("bounded 8-24 view policy", "useful-face / convergence evidence", "or safe-frontier exhaustion"), color="state"),
        "reacquire": dict(x=40, y=1075, w=280, h=150, title="Visual reacquisition", lines=("hold current pose; no command", "recover fresh measured target", "new plan must be re-approved"), color="perception"),
    }
    edges = [
        dict(src="accepted", dst="coverage", kind="data", label="commit generation", src_side="bottom", dst_side="top"),
        dict(src="coverage", dst="candidate", kind="data", label="measured evidence", src_side="bottom", dst_side="top"),
        dict(src="coverage", dst="complete", kind="data", src_side="right", dst_side="left"),
        dict(src="complete", dst="candidate", kind="feedback", label="more surface required", src_side="bottom", dst_side="right", via=((900, 560), (740, 560), (740, 685)), label_at=(840, 553)),
        dict(src="candidate", dst="prequal", kind="data", src_side="bottom", dst_side="top"),
        dict(src="prequal", dst="planner", kind="data", label="bounded shortlist", src_side="bottom", dst_side="top"),
        dict(src="planner", dst="execute", kind="data", label="qualified proposal", src_side="bottom", dst_side="top"),
        dict(src="execute", dst="decision", kind="data", label="settled evidence", src_side="bottom", dst_side="top"),
        dict(src="decision", dst="accepted", kind="feedback", label="ACCEPT: update coverage", src_side="right", dst_side="right", via=((1060, 1600), (1060, 225), (720, 225)), label_at=(1052, 900)),
        dict(src="decision", dst="candidate", kind="feedback", label="REJECT: achieved FK, no coverage", src_side="left", dst_side="left", via=((20, 1600), (20, 685), (380, 685)), label_at=(170, 1593)),
        dict(src="planner", dst="candidate", kind="feedback", label="infeasible ray", src_side="right", dst_side="right", via=((1040, 1150), (1040, 685), (720, 685)), label_at=(1030, 1030), label_anchor="end"),
        dict(src="planner", dst="reacquire", kind="feedback", label="target lost", src_side="left", dst_side="right", label_at=(350, 1040)),
        dict(src="reacquire", dst="candidate", kind="feedback", label="fresh measurement", src_side="top", dst_side="left", via=((180, 1035), (340, 1035), (340, 685), (380, 685)), label_at=(250, 1028)),
    ]
    render_flow(
        "viewpoint-planning-pipeline.svg",
        "Closed-Loop Next-Best-View Planning",
        "One accepted observation changes coverage; rejection changes achieved state but never manufactures evidence",
        nodes,
        edges,
        1740,
        "The planner proves feasibility; NBV chooses what to try next; capture acceptance alone changes measured coverage.",
    )


def render_planner():
    nodes = {
        "select": dict(x=370, y=145, w=360, h=155, title="Next-mission backend selection", lines=("GUI/config: tesseract or curobo", "validated before goal admission", "frozen into RunTargetScan + mission hash"), color="external"),
        "supervisor": dict(x=370, y=365, w=360, h=155, title="ProcessSupervisor", lines=("starts exactly one selected worker", "owns process group + generation", "bounded stop also owns CUDA work"), color="safety"),
        "bridge": dict(x=370, y=590, w=360, h=190, title="Generic Foxy planner bridge", lines=("fresh joints, limits, target, camera + scene", "worker generation/heartbeat/readiness", "request ID + model/calibration hashes", "command-free schema-v5 spool request"), color="planning"),
        "tess": dict(x=55, y=860, w=400, h=180, title="Tesseract worker", lines=("exact configured moving/fixed geometry", "IK + collision + complete path", "baseline + qualified startup recovery", "main and integration branch"), color="planning", status="DEFAULT"),
        "curobo": dict(x=645, y=850, w=400, h=200, title="cuRobo 0.7.8 worker", lines=("MotionGen plan_single / plan_single_js", "exact fixed meshes + moving spheres", "CUDA/Python 3.10 isolated", "hardware_qualified=false", "integration branch only"), color="optional", status="NO ARM MOTION"),
        "response": dict(x=370, y=1125, w=360, h=175, title="Validated planner response", lines=("backend/version + request identity", "collision qualification + rejection codes", "timed six-joint trajectories + metrics", "atomic response consumed once"), color="state"),
        "transport": dict(x=370, y=1370, w=360, h=175, title="ROS planner transport", lines=("main: TesseractPlan", "integration branch: generic MotionPlan", "Tesseract aliases only in Tesseract mode", "status, readiness and provenance topics"), color="state"),
        "common": dict(x=370, y=1615, w=360, h=180, title="Unchanged common execution path", lines=("schedule normalization + validation", "ScanExecutionPlan + PlanAuthorizer", "runtime gates + TrajectoryRunner", "executor alone may publish joints"), color="safety"),
        "blocked": dict(x=40, y=1370, w=270, h=175, title="Structured failure", lines=("stale/malformed/unsupported/unready", "publish correlated rejection", "mission/NBV decides bounded next action", "never switch backend automatically"), color="safety"),
    }
    edges = [
        dict(src="select", dst="supervisor", kind="control", label="frozen backend", src_side="bottom", dst_side="top"),
        dict(src="supervisor", dst="bridge", kind="control", label="matching generation", src_side="bottom", dst_side="top"),
        dict(src="bridge", dst="tess", kind="data", label="private spool", src_side="bottom", dst_side="top"),
        dict(src="bridge", dst="curobo", kind="optional", label="branch-only spool", src_side="bottom", dst_side="top"),
        dict(src="tess", dst="response", kind="data", label="success / rejection", src_side="bottom", dst_side="left", via=((255, 1090), (350, 1090), (350, 1212)), label_at=(300, 1083)),
        dict(src="curobo", dst="response", kind="optional", label="success / rejection", src_side="bottom", dst_side="right", via=((845, 1090), (750, 1090), (750, 1212)), label_at=(800, 1083)),
        dict(src="response", dst="transport", kind="data", src_side="bottom", dst_side="top"),
        dict(src="transport", dst="common", kind="data", label="planner proposal", src_side="bottom", dst_side="top"),
        dict(src="response", dst="blocked", kind="feedback", label="invalid / failed", src_side="left", dst_side="right"),
        dict(src="blocked", dst="bridge", kind="feedback", label="new bounded request only", src_side="top", dst_side="left", via=((175, 1335), (25, 1335), (25, 685), (370, 685)), label_at=(150, 1328)),
        dict(src="tess", dst="bridge", kind="feedback", label="heartbeat + blockers", src_side="left", dst_side="left", via=((25, 950), (25, 685), (370, 685)), label_at=(170, 825)),
        dict(src="curobo", dst="bridge", kind="feedback", label="model qualification", src_side="right", dst_side="right", via=((1070, 950), (1070, 685), (730, 685)), label_at=(1060, 825), label_anchor="end"),
    ]
    render_flow(
        "planner-backend-pipeline.svg",
        "Frozen Planner Backend and Generic Motion Contract",
        "What is common, what is branch-only, and how readiness and rejection return to the mission",
        nodes,
        edges,
        1850,
        "Selecting cuRobo never grants motor authority; its current collision model intentionally blocks physical execution.",
    )


def render_execution():
    nodes = {
        "plan": dict(x=380, y=145, w=340, h=155, title="Planner proposal", lines=("plan/backend/request/trajectory hashes", "collision qualification + timed path", "command-free until independently approved"), color="planning"),
        "validate": dict(x=380, y=370, w=340, h=180, title="Normalize and validate", lines=("six finite joints + increasing timestamps", "20 Hz / 0.05 rad step / speed policy", "fresh matching controller limits", "fresh start and complete path semantics"), color="safety"),
        "authorize": dict(x=380, y=620, w=340, h=180, title="PlanAuthorizer", lines=("live task/hash/backend authorization", "exact plan + trajectory confirmation", "TTL, target drift, dependencies", "path and collision evidence"), color="safety"),
        "runtime": dict(x=380, y=870, w=340, h=200, title="Runtime safety profile", lines=("fresh joints, arm, limits and camera clock", "tracking and approved obstacle authority", "external holder/floor clearance", "motor faults, following error and timeout", "cancel token and mission deadline"), color="safety"),
        "runner": dict(x=380, y=1140, w=340, h=190, title="TrajectoryRunner + executor", lines=("sole autonomous /joint_ctrl_single publisher", "never burst or skip overdue samples", "monitor endpoint, progress and settling", "publish execution state after every change"), color="safety", status="COMMAND OWNER"),
        "driver": dict(x=380, y=1400, w=340, h=175, title="PiPER driver / SocketCAN", lines=("MoveJ targets + aggregate speed", "enable/disable and motor watchdogs", "joint, arm and motion-limit feedback", "all-six feedback proof"), color="actuation", status="CAN OWNER"),
        "capture": dict(x=780, y=1140, w=280, h=190, title="Settled capture", lines=("actual FK and target-facing aim", "stationary feedback + healthy clock", "CAPTURING_RGBD service gate"), color="state"),
        "refresh": dict(x=40, y=870, w=280, h=200, title="Transient recovery", lines=("publish current-position hold", "wait for exact fresh evidence", "resume unchanged stage only", "or replan after target drift"), color="state"),
        "terminal": dict(x=40, y=1400, w=280, h=175, title="Terminal recovery", lines=("PRE_HOME -> ROUGH_HOME", "-> STORAGE_WRIST -> disable all six", "motor authority loss: no new command", "revoke mission and clean children"), color="safety"),
    }
    edges = [
        dict(src="plan", dst="validate", kind="data", src_side="bottom", dst_side="top"),
        dict(src="validate", dst="authorize", kind="data", label="ScanExecutionPlan", src_side="bottom", dst_side="top"),
        dict(src="authorize", dst="runtime", kind="control", label="exact approval", src_side="bottom", dst_side="top"),
        dict(src="runtime", dst="runner", kind="control", label="dispatch allowed", src_side="bottom", dst_side="top"),
        dict(src="runner", dst="driver", kind="command", label="MoveJ samples", src_side="bottom", dst_side="top"),
        dict(src="driver", dst="runner", kind="feedback", label="joint + motor feedback", src_side="right", dst_side="right", via=((750, 1487), (750, 1235), (720, 1235)), label_at=(742, 1360)),
        dict(src="driver", dst="validate", kind="feedback", label="controller-limit generation", src_side="right", dst_side="right", via=((1080, 1487), (1080, 460), (720, 460)), label_at=(1072, 980)),
        dict(src="driver", dst="runtime", kind="feedback", label="fresh all-six health", src_side="left", dst_side="left", via=((345, 1487), (345, 970), (380, 970)), label_at=(330, 1375), label_anchor="end"),
        dict(src="runner", dst="capture", kind="data", src_side="right", dst_side="left"),
        dict(src="runtime", dst="refresh", kind="feedback", src_side="left", dst_side="right"),
        dict(src="refresh", dst="runtime", kind="feedback", label="fresh evidence -> resume", src_side="top", dst_side="left", via=((180, 835), (345, 835), (345, 970), (380, 970)), label_at=(250, 828)),
        dict(src="refresh", dst="plan", kind="feedback", label="target drift -> replan", src_side="left", dst_side="left", via=((20, 970), (20, 222), (380, 222)), label_at=(45, 700), label_anchor="start"),
        dict(src="runtime", dst="terminal", kind="feedback", label="hard fault / cancel", src_side="left", dst_side="right", via=((25, 970), (25, 1487), (40, 1487)), label_at=(140, 1130)),
        dict(src="runner", dst="terminal", kind="feedback", label="timeout / following failure", src_side="left", dst_side="right", via=((330, 1235), (330, 1487), (320, 1487)), label_at=(245, 1228)),
    ]
    render_flow(
        "execution-safety-pipeline.svg",
        "Guarded Execution, Feedback and Recovery",
        "The sole command path, its live feedback, and the split between recoverable evidence loss and terminal faults",
        nodes,
        edges,
        1640,
        "Planner output is necessary but never sufficient: authorization and live feedback remain independent authorities.",
    )


def render_capture():
    nodes = {
        "settled": dict(x=380, y=145, w=340, h=155, title="Executor-settled viewpoint", lines=("achieved FK + final aim/drift checks", "fresh target/scene + healthy clock", "CAPTURING_RGBD request identity"), color="safety"),
        "burst": dict(x=380, y=370, w=340, h=190, title="Synchronized observation", lines=("exact mask/RGB stamp", "20 new native depth/confidence frames", "per-pixel median with >=0.50 support", "intrinsics + capture-time transforms", "joints + plan provenance"), color="input"),
        "gate": dict(x=380, y=630, w=340, h=190, title="Semantic and quality gate", lines=("unambiguous target depth component", "GOOD target / fresh diagnostics", "CLEAR or exact correlated occlusion proof", "atomic write must finish completely"), color="safety"),
        "commit": dict(x=380, y=890, w=340, h=175, title="Immutable schema-2 commit", lines=("target-only support/depth + RGB/mask", "manifest and per-artifact SHA-256", "achieved pose + view selection metadata", "partial files never increment count"), color="state"),
        "coverage": dict(x=760, y=890, w=290, h=175, title="Accepted feedback", lines=("history generation advances", "measured coverage rebuilds", "NBV replans or completes"), color="state"),
        "retry": dict(x=40, y=630, w=280, h=190, title="Rejected / stale feedback", lines=("hold the achieved pose", "one correlated heavy refresh", "retry same observation once", "then exclude and replan"), color="safety"),
        "shutdown": dict(x=380, y=1140, w=340, h=165, title="Safe mission terminal", lines=("home stages + all-six disable", "mission result and dataset become durable", "optionally WAITING_FOR_BASE_HOME"), color="safety"),
        "base": dict(x=380, y=1375, w=340, h=155, title="Exact tracked-base-home report", lines=("task + job + manifest correlation", "wrong or repeated identities fail/idempotent", "gateway launches isolated reconstruction"), color="external"),
        "admit": dict(x=380, y=1600, w=340, h=165, title="Immutable input admission", lines=("manifest and every frame hash", "capture-time TF/calibration/confidence", "reject tamper, escape or incomplete data"), color="safety"),
        "register": dict(x=80, y=1840, w=300, h=180, title="Registration candidates", lines=("robot_pose baseline", "optional bounded target GICP", "optional target-excluded scene pose graph", "every correction remains bounded"), color="planning"),
        "fusion": dict(x=410, y=1840, w=280, h=180, title="Target-only TSDF", lines=("3 mm voxel / 15 mm truncation default", "qualified masked depth only", "predicted geometry never fused"), color="state"),
        "outputs": dict(x=720, y=1840, w=300, h=180, title="Validated products", lines=("raw + cleaned PLY meshes", "coloured measured clouds", "quality metrics + visual preview", "complete provenance report"), color="state"),
    }
    edges = [
        dict(src="settled", dst="burst", kind="control", label="capture service", src_side="bottom", dst_side="top"),
        dict(src="burst", dst="gate", kind="data", src_side="bottom", dst_side="top"),
        dict(src="gate", dst="commit", kind="data", label="ACCEPT", src_side="bottom", dst_side="top"),
        dict(src="gate", dst="retry", kind="feedback", src_side="left", dst_side="right"),
        dict(src="retry", dst="gate", kind="feedback", label="one fresh re-evaluation", src_side="top", dst_side="left", via=((180, 595), (345, 595), (345, 725), (380, 725)), label_at=(250, 588)),
        dict(src="retry", dst="settled", kind="feedback", label="exclude + new NBV", src_side="left", dst_side="left", via=((20, 725), (20, 222), (380, 222)), label_at=(45, 500), label_anchor="start"),
        dict(src="commit", dst="coverage", kind="feedback", src_side="right", dst_side="left"),
        dict(src="coverage", dst="settled", kind="feedback", label="next qualified view", src_side="right", dst_side="right", via=((1070, 977), (1070, 222), (720, 222)), label_at=(1062, 600)),
        dict(src="commit", dst="shutdown", kind="control", label="complete / terminal", src_side="bottom", dst_side="top"),
        dict(src="shutdown", dst="base", kind="control", src_side="bottom", dst_side="top"),
        dict(src="base", dst="admit", kind="control", src_side="bottom", dst_side="top"),
        dict(src="admit", dst="register", kind="data", src_side="bottom", dst_side="top"),
        dict(src="admit", dst="fusion", kind="data", label="validated frames", src_side="bottom", dst_side="top"),
        dict(src="register", dst="fusion", kind="data", src_side="right", dst_side="left"),
        dict(src="fusion", dst="outputs", kind="data", src_side="right", dst_side="left"),
    ]
    render_flow(
        "capture-reconstruction-pipeline.svg",
        "Settled Capture, Feedback and Reconstruction",
        "Why acceptance changes NBV state, rejection does not, and reconstruction waits for immutable safe-state evidence",
        nodes,
        edges,
        2075,
        "Reconstruction is asynchronous and command-free; its failure cannot retroactively change a safely completed arm mission.",
    )


def render_hardware():
    nodes = {
        "base": dict(x=380, y=145, w=340, h=155, title="Bunker Pro 2 tracked base", lines=("physical carrier + enclosure + arm mount", "external brake/localization authority", "stationary during PiPER motion", "no chassis command from this repository"), color="actuation"),
        "enclosure": dict(x=380, y=370, w=340, h=155, title="Enclosure v4 + compute", lines=("panels, frame, battery and mounts", "Ubuntu 20.04 / ROS 2 Foxy host", "RTX 3090 reference CUDA workstation"), color="input"),
        "arm": dict(x=380, y=595, w=340, h=180, title="PiPER 6-DOF arm", lines=("piper_base_link -> base_link mount", "driver -> USB-CAN -> SocketCAN", "six motor controllers + MoveJ", "joint/arm/limit feedback returns to ROS"), color="actuation"),
        "l515": dict(x=380, y=845, w=340, h=170, title="Qualified eye-in-hand L515", lines=("camera holder + calibrated optical frame", "RGB, depth, confidence and intrinsics", "camera motion is the active scan sensor"), color="input"),
        "foxy": dict(x=45, y=1110, w=300, h=175, title="ROS 2 Foxy control environment", lines=("mission, NBV, bridge and executor", "PiPER SDK / CAN driver", "Python 3.8; sole command boundary"), color="safety"),
        "ai": dict(x=400, y=1110, w=300, h=175, title="Isolated perception environment", lines=("Python 3.10 + CUDA", "GroundingDINO + SAM 2", "file spools; no motor interface"), color="perception"),
        "planner": dict(x=755, y=1110, w=300, h=175, title="Isolated planner environments", lines=("rootless Tesseract runtime", "cuRobo Python/CUDA on branch", "file spools; no motor interface"), color="planning"),
        "optional": dict(x=380, y=1370, w=340, h=170, title="Optional CAD sensor provision", lines=("ZED camera and LiDAR mounts", "mechanical files only", "not current perception inputs", "not collision-qualified by CAD alone"), color="optional", status="NOT IN RUNTIME"),
    }
    edges = [
        dict(src="base", dst="enclosure", kind="data", label="mount + power", src_side="bottom", dst_side="top"),
        dict(src="enclosure", dst="arm", kind="data", label="rigid mounting tree", src_side="bottom", dst_side="top"),
        dict(src="arm", dst="l515", kind="data", label="eye-in-hand transform", src_side="bottom", dst_side="top"),
        dict(src="l515", dst="foxy", kind="data", label="ROS camera streams", src_side="bottom", dst_side="top"),
        dict(src="l515", dst="ai", kind="data", label="RGB-D spool input", src_side="bottom", dst_side="top"),
        dict(src="arm", dst="planner", kind="data", label="URDF/SRDF + current state", src_side="right", dst_side="right", via=((1075, 685), (1075, 1197), (1055, 1197)), label_at=(1067, 930)),
        dict(src="foxy", dst="arm", kind="command", label="sole MoveJ command", src_side="top", dst_side="left", via=((195, 1075), (345, 1075), (345, 685), (380, 685)), label_at=(265, 1068)),
        dict(src="arm", dst="foxy", kind="feedback", label="joints / status / limits", src_side="left", dst_side="top", via=((20, 685), (20, 1075), (195, 1075)), label_at=(130, 678)),
        dict(src="ai", dst="foxy", kind="data", src_side="left", dst_side="right"),
        dict(src="planner", dst="foxy", kind="data", label="command-free plan", src_at=(755, 1285), dst_at=(345, 1285), via=((755, 1320), (345, 1320)), label_at=(550, 1312)),
        dict(src="enclosure", dst="optional", kind="optional", label="optional CAD", src_side="right", dst_side="right", via=((1070, 447), (1070, 1455), (720, 1455)), label_at=(1062, 1020), label_anchor="end"),
    ]
    render_flow(
        "hardware-topology.svg",
        "Hardware, Compute and Command Boundaries",
        "What is physically installed, what owns motor authority, and what remains optional CAD provision",
        nodes,
        edges,
        1595,
        "Design CAD supports fabrication; URDF/planner collision assets and calibration retain separate qualification ownership.",
    )


def main():
    render_system()
    render_perception()
    render_nbv()
    render_planner()
    render_execution()
    render_capture()
    render_hardware()


if __name__ == "__main__":
    main()

