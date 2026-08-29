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
    title_y = y + node.get("title_offset", 31)
    detail_y = y + node.get("detail_offset", 58)
    if node.get("stack_status"):
        title_y = y + 46
        detail_y = y + 78
    result = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'rx="13" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>',
        text(x + width / 2, title_y, node["title"], "node-title"),
        multiline(
            x + width / 2,
            detail_y,
            node.get("lines", ()),
            "detail",
            step=node.get("line_step", 18),
        ),
    ]
    status = node.get("status")
    if status:
        badge_width = max(88, 8 * len(status) + 18)
        result.extend([
            f'<rect x="{x + width - badge_width - 10}" y="{y + 9}" '
            f'width="{badge_width}" height="21" rx="10" fill="{stroke}"/>',
            text(
                x + width - badge_width / 2 - 10,
                y + 24,
                status,
                "badge",
            ),
        ])
    return result


def lane(y, height, title, subtitle=""):
    result = [
        f'<rect x="28" y="{y}" width="1344" height="{height}" rx="18" '
        'fill="#FAFBFC" stroke="#D7DDE5" stroke-width="1.2"/>',
        text(50, y + 30, title, "lane-title", "start"),
    ]
    if subtitle:
        result.append(text(50, y + 52, subtitle, "lane-subtitle", "start"))
    return result


def edge(nodes, item):
    source = point(nodes[item["src"]], item.get("src_side", "bottom"))
    target = point(nodes[item["dst"]], item.get("dst_side", "top"))
    points = [source] + list(item.get("via", ())) + [target]
    color = EDGE_COLORS[item.get("kind", "data")]
    dashed = item.get("kind") in ("feedback", "optional")
    marker = item.get("kind", "data")
    polyline = " ".join(f"{x},{y}" for x, y in points)
    result = [
        f'<polyline points="{polyline}" fill="none" stroke="{color}" '
        f'stroke-width="{item.get("width", 2.2)}" '
        + ('stroke-dasharray="8 6" ' if dashed else "")
        + f'marker-end="url(#{marker}-arrow)"/>'
    ]
    label = item.get("label")
    if label:
        label_x, label_y = item.get(
            "label_at",
            ((source[0] + target[0]) / 2, (source[1] + target[1]) / 2 - 7),
        )
        result.append(text(label_x, label_y, label, "edge-label"))
    return result


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
        ".title{font:700 34px Arial,Helvetica,sans-serif;fill:#17212B}",
        ".subtitle{font:15px Arial,Helvetica,sans-serif;fill:#52606D}",
        ".lane-title{font:700 18px Arial,Helvetica,sans-serif;fill:#24303C}",
        ".lane-subtitle{font:12px Arial,Helvetica,sans-serif;fill:#687684}",
        ".node-title{font:700 16px Arial,Helvetica,sans-serif;fill:#17212B}",
        ".detail{font:12.5px Arial,Helvetica,sans-serif;fill:#344250}",
        ".edge-label{font:700 11.5px Arial,Helvetica,sans-serif;fill:#344250;paint-order:stroke;stroke:#fff;stroke-width:5px;stroke-linejoin:round}",
        ".badge{font:700 10px Arial,Helvetica,sans-serif;fill:#fff;letter-spacing:.3px}",
        ".legend{font:12px Arial,Helvetica,sans-serif;fill:#46515E}",
        ".note{font:12.5px Arial,Helvetica,sans-serif;fill:#52606D}",
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
    # The system map is intentionally roomier than the focused diagrams.  Its
    # original 1400 px layout is scaled as a unit, then receives larger type so
    # the block contents remain readable at normal GitHub README width.
    scale = 1.25
    width, height = 1750, 4075
    title = "PiPER Active-View Scanning: Detailed System Architecture"
    subtitle = "Implementation map for main plus the explicitly labelled cuRobo integration path"
    nodes = {
        "operator": dict(x=60, y=180, w=280, h=110, title="Operator / supervisor", lines=("Native GUI, RViz and E-stop", "select next-mission settings", "start, observe or cancel"), color="external"),
        "gateway": dict(x=385, y=170, w=310, h=130, title="Tracked-robot gateway", lines=("SCAN_3D request + rough target", "odom -> piper_base_link snapshot", "2 Hz task heartbeat / home report", "base remains stationary and braked"), color="external"),
        "gui": dict(x=740, y=180, w=300, h=110, title="PiPER native GUI", lines=("Automatic scan + ray review", "planner for next mission", "status, diagnostics and results"), color="external"),
        "config": dict(x=1080, y=170, w=270, h=130, title="Frozen mission config", lines=("target label/profile", "floor + scan/NBV policy", "backend + speed + motion opt-ins", "canonical mission SHA-256"), color="state"),
        "coordinator": dict(x=60, y=410, w=335, h=155, title="RunTargetScan coordinator", lines=("queue and admission adapter", "immutable goal / backend identity", "publishes action feedback", "returns typed result + dataset"), color="state"),
        "supervisor": dict(x=445, y=390, w=405, h=195, title="ProcessSupervisor", lines=("owns exact process generations", "driver -> vision -> hand-eye", "one selected planner worker", "scan stack; heartbeats; bounded cleanup", "never adopts unrelated processes"), color="safety"),
        "mission": dict(x=900, y=380, w=440, h=215, title="MissionEngine lifecycle", lines=("preflight -> enable/hold -> startup wrist", "rough home -> bounded target acquisition", "workflow ready -> one-view closed loop", "terminal PRE_HOME -> ROUGH_HOME", "STORAGE_WRIST -> all-six disable", "motor loss: no further motion command"), color="safety"),
        "camera": dict(x=55, y=730, w=260, h=165, title="Eye-in-hand L515", lines=("RGB + aligned depth", "native depth + 4-bit confidence", "CameraInfo + source timestamps", "30 Hz synchronized evidence"), color="input"),
        "clock": dict(x=345, y=720, w=280, h=185, title="Timing, TF and calibration", lines=("camera timestamp watchdog", "accepted hand-eye transform", "base_link <-> optical frame", "fresh joints and camera pose", "fault can restart owned vision stack"), color="input"),
        "ai": dict(x=655, y=700, w=320, h=210, title="Isolated CUDA perception", lines=("GroundingDINO acquisition/refresh", "SAM 2 seed + live mask propagation", "permission-bounded file spools", "current-run worker readiness", "ROS-free workers; no motor interface"), color="perception"),
        "geometry": dict(x=1005, y=700, w=340, h=210, title="ROS perception geometry", lines=("mask -> detection -> qualified depth", "Target3D + Kalman tracked target", "target cloud / landmark / envelope", "obstacle boxes + quality + occlusion", "freshness and correlation metadata"), color="perception"),
        "target_state": dict(x=655, y=935, w=320, h=100, title="Measured target state", lines=("stable base-frame target + covariance", "TRACKING / LOW_CONFIDENCE / LOST"), color="state"),
        "scene_state": dict(x=1005, y=935, w=340, h=100, title="Scene safety evidence", lines=("obstacles, target envelope, framing", "occlusion, quality and tracking health"), color="state"),
        "coverage": dict(x=55, y=1150, w=330, h=180, title="Accepted-only scan memory", lines=("schema-2 captures + achieved FK", "0.005 m measured coverage grid", "unknown / observed target surface", "rejected views never add coverage"), color="state"),
        "nbv": dict(x=445, y=1130, w=430, h=220, title="Closed-loop NBV and ray policy", lines=("legacy, voxel_nbv or frozen ray_nbv", "seed from actual current FK", "rank marginal information before travel", "direction novelty + bounded standoff", "one accepted view per generation", "completion or safe-frontier exhaustion"), color="planning"),
        "reach": dict(x=935, y=1150, w=375, h=180, title="Command-free prequalification", lines=("capability-map / workspace evidence", "target-facing aim and <=5 deg fallback", "static hard culls + shortlist", "publishes reachable candidates"), color="planning"),
        "bridge": dict(x=55, y=1480, w=395, h=235, title="MotionPlannerBridge (ROS 2 Foxy)", lines=("freezes tesseract or curobo per mission", "requires worker heartbeat/readiness", "snapshots joints, controller limits, target", "camera clock, obstacles, hashes and views", "writes schema-v5 private request", "publishes status/readiness/provenance"), color="planning"),
        "tesseract": dict(x=505, y=1435, w=390, h=155, title="Tesseract 0.35 worker", lines=("ROS-free exact IK/collision/path", "configured moving + fixed meshes", "qualified baseline and bootstrap recovery", "atomic request/response spool"), color="planning", status="MAIN + BRANCH"),
        "curobo": dict(x=505, y=1620, w=390, h=180, title="cuRobo 0.7.8 MotionGen worker", lines=("Python 3.10 + CUDA; ROS-free", "exact fixed Bunker meshes", "167-sphere articulated approximation", "hardware_qualified=false; motion blocked", "no fallback to Tesseract"), color="optional", status="BRANCH ONLY"),
        "transport": dict(x=960, y=1500, w=370, h=220, title="Planner transport", lines=("main: TesseractPlan", "curobo branch: generic MotionPlan", "IDs + backend + hashes + plan kind", "complete timed six-joint trajectories", "collision qualification + diagnostics", "invalid/rejected results stay correlated"), color="state"),
        "normalize": dict(x=50, y=1940, w=300, h=180, title="Common plan normalization", lines=("joint order, finite values and times", "20 Hz schedule + 0.05 rad step", "speed-scaled MoveJ limits", "fresh-start and trajectory hashes"), color="safety"),
        "authorize": dict(x=390, y=1940, w=300, h=180, title="PlanAuthorizer", lines=("live mission authorization + TTL", "plan/trajectory/backend identity", "target drift + dependency readiness", "fresh path revalidation"), color="safety"),
        "executor": dict(x=730, y=1905, w=335, h=250, title="scan_viewpoint_executor", lines=("sole autonomous joint publisher", "TrajectoryRunner schedule", "runtime freshness + scene gates", "following-error / timeout / settle", "hold, refresh, replan or recovery", "publishes ScanExecutionStatus"), color="safety", status="COMMAND OWNER"),
        "driver": dict(x=1105, y=1940, w=245, h=180, title="PiPER driver", lines=("/joint_ctrl_single -> MoveJ", "SocketCAN at 1 Mbps", "enable/disable + watchdogs", "six motors + coherent feedback"), color="actuation", status="CAN OWNER"),
        "settle": dict(x=45, y=2320, w=285, h=150, title="Settled capture request", lines=("actual FK + final aim check", "stationary joints + healthy clock", "executor enters", "CAPTURING_RGBD"), color="safety"),
        "burst": dict(x=365, y=2300, w=300, h=190, title="Confidence-qualified burst", lines=("exact mask/RGB stamp identity", "20 new native depth frames", "grade >=8 and >=0.50 support", "per-pixel median; calibrated TF", "target component ambiguity gate"), color="input"),
        "admission": dict(x=705, y=2300, w=280, h=190, title="Capture admission", lines=("GOOD target + fresh diagnostics", "CLEAR or correlated", "semantic occlusion proof", "atomic artifacts + manifest hashes", "accept, retry or reject"), color="safety"),
        "dataset": dict(x=1025, y=2290, w=325, h=205, title="Immutable schema-2 dataset", lines=("RGB, raw/qualified depth, mask", "intrinsics, joints, capture-time TF", "plan/view provenance + quality", "target-only support and model seed", "transactional manifest SHA-256"), color="state"),
        "rejected": dict(x=365, y=2530, w=300, h=125, title="Rejected observation", lines=("hold achieved FK", "one bounded heavy refresh", "then exclude/replan; no coverage"), color="safety"),
        "history": dict(x=705, y=2530, w=280, h=125, title="Accepted history generation", lines=("commit achieved pose + evidence", "rebuild measured coverage", "request the next view"), color="state"),
        "recovery": dict(x=75, y=2735, w=360, h=165, title="Safe terminal recovery", lines=("cancel, completion or bounded failure", "PRE_HOME -> ROUGH_HOME", "-> STORAGE_WRIST -> disable all six", "revoke authorization + stop owned children"), color="safety"),
        "base_home": dict(x=515, y=2735, w=350, h=165, title="Tracked-base correlation", lines=("arm result may wait for base home", "exact task/job/manifest identity", "gateway accepts idempotent report", "repository still sends no chassis command"), color="external"),
        "reconstruct": dict(x=945, y=2720, w=385, h=195, title="Offline reconstruction", lines=("immutable input admission", "target-only Open3D TSDF (3 mm default)", "optional bounded GICP / scene pose graph", "mesh + cloud + metrics + provenance", "failure does not rewrite mission result"), color="state"),
    }
    for node_value in nodes.values():
        node_value.setdefault("line_step", 21)
        node_value.setdefault("title_offset", 34)
        node_value.setdefault("detail_offset", 65)
        if node_value.get("status"):
            node_value["stack_status"] = True
    edges = [
        dict(src="operator", dst="coordinator", kind="control", label="start / cancel", src_side="bottom", dst_side="top"),
        dict(src="gateway", dst="coordinator", kind="control", label="RunTargetScan", src_side="bottom", dst_side="top"),
        dict(src="gui", dst="config", kind="control", label="apply for next mission", src_side="right", dst_side="left"),
        dict(src="config", dst="coordinator", kind="data", label="frozen goal + hash", src_side="bottom", dst_side="top", via=((1215, 335), (228, 335)), label_at=(750, 329)),
        dict(src="coordinator", dst="supervisor", kind="control", label="own generation", src_side="right", dst_side="left"),
        dict(src="coordinator", dst="mission", kind="control", label="MissionContext", src_side="right", dst_side="left", via=((420, 620), (880, 620)), label_at=(650, 613)),
        dict(src="supervisor", dst="camera", kind="control", label="ordered startup", src_side="bottom", dst_side="top", via=((648, 650), (185, 650)), label_at=(410, 643)),
        dict(src="supervisor", dst="ai", kind="control", label="vision process group", src_side="bottom", dst_side="top"),
        dict(src="camera", dst="clock", kind="data", label="timestamps + frames", src_side="right", dst_side="left"),
        dict(src="camera", dst="ai", kind="data", label="settled RGB-D", src_side="right", dst_side="left", via=((330, 690), (640, 690)), label_at=(480, 683)),
        dict(src="clock", dst="geometry", kind="data", label="TF + calibration", src_side="right", dst_side="left", via=((640, 930), (990, 930)), label_at=(815, 923)),
        dict(src="ai", dst="geometry", kind="data", label="mask / detection", src_side="right", dst_side="left"),
        dict(src="geometry", dst="target_state", kind="data", label="Target3D / tracker", src_side="left", dst_side="right"),
        dict(src="geometry", dst="scene_state", kind="data", label="scene + diagnostics", src_side="bottom", dst_side="top"),
        dict(src="target_state", dst="nbv", kind="data", label="target centre", src_side="bottom", dst_side="top"),
        dict(src="scene_state", dst="reach", kind="data", label="safe scene", src_side="bottom", dst_side="top"),
        dict(src="coverage", dst="nbv", kind="data", label="accepted evidence", src_side="right", dst_side="left"),
        dict(src="nbv", dst="reach", kind="data", label="candidate rays/views", src_side="right", dst_side="left"),
        dict(src="reach", dst="bridge", kind="data", label="shortlist", src_side="bottom", dst_side="top", via=((1122, 1390), (252, 1390)), label_at=(690, 1383)),
        dict(src="scene_state", dst="bridge", kind="data", label="fresh target / scene snapshot", src_side="right", dst_side="right", via=((1370, 985), (1370, 1460), (450, 1460)), label_at=(1080, 1453)),
        dict(src="bridge", dst="tesseract", kind="data", label="schema-v5 spool", src_side="right", dst_side="left"),
        dict(src="bridge", dst="curobo", kind="optional", label="selected branch backend", src_side="right", dst_side="left", via=((475, 1775), (485, 1775)), label_at=(480, 1768)),
        dict(src="tesseract", dst="transport", kind="data", label="validated response", src_side="right", dst_side="left"),
        dict(src="curobo", dst="transport", kind="optional", label="MotionGen response", src_side="right", dst_side="left"),
        dict(src="transport", dst="normalize", kind="data", label="proposal only", src_side="bottom", dst_side="top", via=((1145, 1845), (200, 1845)), label_at=(680, 1838)),
        dict(src="normalize", dst="authorize", kind="data", label="ScanExecutionPlan", src_side="right", dst_side="left"),
        dict(src="authorize", dst="executor", kind="control", label="exact approval", src_side="right", dst_side="left"),
        dict(src="executor", dst="driver", kind="command", label="/joint_ctrl_single", src_side="right", dst_side="left"),
        dict(src="driver", dst="executor", kind="feedback", label="joints / arm / limits", src_side="top", dst_side="top", via=((1227, 1870), (898, 1870)), label_at=(1060, 1863)),
        dict(src="driver", dst="bridge", kind="feedback", label="current start + controller limits", src_side="right", dst_side="right", via=((1380, 2030), (1380, 1470), (450, 1470)), label_at=(1090, 1463)),
        dict(src="driver", dst="mission", kind="feedback", label="enable / disable proof", src_side="right", dst_side="right", via=((1390, 2030), (1390, 620), (1340, 620)), label_at=(1365, 1060)),
        dict(src="clock", dst="executor", kind="feedback", label="runtime freshness", src_side="left", dst_side="left", via=((25, 812), (25, 1875), (730, 1875)), label_at=(250, 1868)),
        dict(src="scene_state", dst="executor", kind="feedback", label="tracking / obstacle / quality gates", src_side="right", dst_side="right", via=((1360, 985), (1360, 1880), (1065, 1880)), label_at=(1200, 1873)),
        dict(src="executor", dst="settle", kind="control", label="settled view", src_side="bottom", dst_side="top", via=((898, 2200), (187, 2200)), label_at=(550, 2193)),
        dict(src="settle", dst="burst", kind="control", label="capture service", src_side="right", dst_side="left"),
        dict(src="burst", dst="admission", kind="data", label="qualified observation", src_side="right", dst_side="left"),
        dict(src="admission", dst="dataset", kind="data", label="atomic accept", src_side="right", dst_side="left"),
        dict(src="admission", dst="rejected", kind="feedback", label="retry / reject", src_side="bottom", dst_side="top"),
        dict(src="dataset", dst="history", kind="data", label="commit", src_side="bottom", dst_side="right", via=((1187, 2560), (1000, 2560)), label_at=(1090, 2553)),
        dict(src="history", dst="coverage", kind="feedback", label="new accepted generation", src_side="left", dst_side="left", via=((25, 2582), (25, 1240), (55, 1240)), label_at=(130, 2575)),
        dict(src="rejected", dst="ai", kind="feedback", label="bounded heavy refresh / reacquire", src_side="left", dst_side="left", via=((18, 2582), (18, 805), (655, 805)), label_at=(210, 2575)),
        dict(src="rejected", dst="nbv", kind="feedback", label="replan from achieved FK", src_side="top", dst_side="bottom", via=((515, 2505), (915, 2505), (915, 1380), (660, 1380)), label_at=(810, 2498)),
        dict(src="bridge", dst="nbv", kind="feedback", label="reject / retire ray; no fallback", src_side="left", dst_side="left", via=((35, 1598), (35, 1240), (445, 1240)), label_at=(190, 1591)),
        dict(src="bridge", dst="coordinator", kind="feedback", label="readiness + status blockers", src_side="left", dst_side="left", via=((20, 1598), (20, 487), (60, 487)), label_at=(160, 1591)),
        dict(src="mission", dst="recovery", kind="control", label="complete / cancel / fail", src_side="right", dst_side="right", via=((1390, 487), (1390, 2685), (435, 2685)), label_at=(1230, 2678)),
        dict(src="coverage", dst="recovery", kind="control", label="completion / exhaustion", src_side="left", dst_side="left", via=((30, 1240), (30, 2818), (75, 2818)), label_at=(175, 2708)),
        dict(src="executor", dst="recovery", kind="feedback", label="fault or cancellation", src_side="left", dst_side="left", via=((25, 2030), (25, 2818), (75, 2818)), label_at=(160, 2023)),
        dict(src="recovery", dst="base_home", kind="control", label="WAITING_FOR_BASE_HOME", src_side="right", dst_side="left"),
        dict(src="base_home", dst="reconstruct", kind="control", label="exact correlated report", src_side="right", dst_side="left"),
        dict(src="dataset", dst="reconstruct", kind="data", label="immutable captures", src_side="right", dst_side="right", via=((1370, 2392), (1370, 2818), (1330, 2818)), label_at=(1363, 2605)),
    ]

    # Reserve a genuine header band in every lane.  The original compact map
    # placed subtitles almost on top of the first row; these insertions move
    # nodes, edge waypoints and labels together while retaining their routing.
    header_insertions = (160, 375, 670, 1100, 1425, 1875, 2265, 2695)

    def expanded_y(value):
        return value + 35 * sum(value >= threshold for threshold in header_insertions)

    for node_value in nodes.values():
        node_value["y"] = expanded_y(node_value["y"])
    for item in edges:
        if "via" in item:
            item["via"] = tuple((x, expanded_y(y)) for x, y in item["via"])
        if "label_at" in item:
            label_x, label_y = item["label_at"]
            item["label_at"] = (label_x, expanded_y(label_y))

    body = svg_header(width, height, title, subtitle)
    body.extend([
        "<style>",
        ".title{font-size:42px}",
        ".subtitle{font-size:18px}",
        ".legend{font-size:14px}",
        ".system-map .lane-title{font-size:22px}",
        ".system-map .lane-subtitle{font-size:15px}",
        ".system-map .node-title{font-size:20px}",
        ".system-map .detail{font-size:17.5px}",
        ".system-map .edge-label{font-size:14px;stroke-width:6px}",
        ".system-map .badge{font-size:12px}",
        ".system-map .note{font-size:15.5px}",
        "</style>",
    ])
    body.extend(legend(122, width))
    body.append(f'<g class="system-map" transform="scale({scale})">')
    body.extend(lane(expanded_y(135), 230, "1  Mission request and frozen operator intent", "External task ownership; the tracked base is never commanded here."))
    body.extend(lane(expanded_y(350), 310, "2  Mission orchestration and process ownership", "One coordinator, one process generation and one autonomous command owner."))
    body.extend(lane(expanded_y(645), 445, "3  Eye-in-hand sensing, perception and measured scene state", "Only timestamp-correlated measurements become target or scene evidence."))
    body.extend(lane(expanded_y(1075), 340, "4  Accepted-only coverage and next-best-view selection", "Prediction may rank a view; it never becomes measured reconstruction input."))
    body.extend(lane(expanded_y(1400), 460, "5  Frozen motion-planner backend and command-free proposal", "Tesseract is on main; cuRobo is explicitly branch-only and hardware-unqualified."))
    body.extend(lane(expanded_y(1850), 405, "6  Common authorization, execution and physical feedback", "Every backend must traverse the same fail-closed execution boundary."))
    body.extend(lane(expanded_y(2240), 445, "7  Settled observation, immutable commit and closed-loop feedback", "Accepted and rejected observations deliberately have different state effects."))
    body.extend(lane(expanded_y(2670), 305, "8  Terminal safety, tracked-base correlation and reconstruction", "Reconstruction begins only from immutable data after the required safe-state evidence."))
    for item in edges:
        body.extend(edge(nodes, item))
    for node_value in nodes.values():
        body.extend(box(node_value))
    body.extend([
        text(
            width / 2,
            expanded_y(2962),
            "Command chain: planner proposal -> executor joint command -> PiPER driver CAN + feedback.",
            "note",
        ),
        "</g>",
        "</svg>",
    ])
    write_svg("system-overview.svg", body)


def render_flow(filename, title, subtitle, nodes, edges, height, footer):
    width = 1100
    body = svg_header(width, height, title, subtitle)
    body.extend(legend(108, width))
    for item in edges:
        body.extend(edge(nodes, item))
    for node_value in nodes.values():
        body.extend(box(node_value))
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
        dict(src="heavy", dst="sam", kind="data", label="target seed", src_side="right", dst_side="left"),
        dict(src="sam", dst="depth", kind="data", label="dense mask", src_side="right", dst_side="left"),
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
        dict(src="target3d", dst="sam", kind="feedback", label="measurement disagreement", src_side="left", dst_side="left", via=((355, 685), (355, 447), (410, 447)), label_at=(240, 677)),
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
        dict(src="coverage", dst="complete", kind="data", label="coverage metrics", src_side="right", dst_side="left"),
        dict(src="complete", dst="candidate", kind="feedback", label="more surface required", src_side="bottom", dst_side="right", via=((900, 560), (740, 560), (740, 685)), label_at=(840, 553)),
        dict(src="candidate", dst="prequal", kind="data", src_side="bottom", dst_side="top"),
        dict(src="prequal", dst="planner", kind="data", label="bounded shortlist", src_side="bottom", dst_side="top"),
        dict(src="planner", dst="execute", kind="data", label="qualified proposal", src_side="bottom", dst_side="top"),
        dict(src="execute", dst="decision", kind="data", label="settled evidence", src_side="bottom", dst_side="top"),
        dict(src="decision", dst="accepted", kind="feedback", label="ACCEPT: update coverage", src_side="right", dst_side="right", via=((1060, 1600), (1060, 225), (720, 225)), label_at=(1052, 900)),
        dict(src="decision", dst="candidate", kind="feedback", label="REJECT: achieved FK, no coverage", src_side="left", dst_side="left", via=((20, 1600), (20, 685), (380, 685)), label_at=(170, 1593)),
        dict(src="planner", dst="candidate", kind="feedback", label="retire hard-infeasible ray", src_side="right", dst_side="right", via=((750, 1150), (750, 685), (720, 685)), label_at=(742, 920)),
        dict(src="planner", dst="reacquire", kind="feedback", label="stale / lost target", src_side="left", dst_side="right"),
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
        dict(src="tess", dst="bridge", kind="feedback", label="0.5 s heartbeat + blockers", src_side="left", dst_side="left", via=((25, 950), (25, 685), (370, 685)), label_at=(170, 943)),
        dict(src="curobo", dst="bridge", kind="feedback", label="heartbeat + model qualification", src_side="right", dst_side="right", via=((1070, 950), (1070, 685), (730, 685)), label_at=(920, 943)),
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
        dict(src="driver", dst="runtime", kind="feedback", label="fresh all-six health", src_side="left", dst_side="left", via=((345, 1487), (345, 970), (380, 970)), label_at=(270, 1480)),
        dict(src="runner", dst="capture", kind="data", label="settled status", src_side="right", dst_side="left"),
        dict(src="runtime", dst="refresh", kind="feedback", label="stale but recoverable", src_side="left", dst_side="right"),
        dict(src="refresh", dst="runtime", kind="feedback", label="fresh evidence -> resume", src_side="top", dst_side="left", via=((180, 835), (345, 835), (345, 970), (380, 970)), label_at=(250, 828)),
        dict(src="refresh", dst="plan", kind="feedback", label="target drift -> replan", src_side="left", dst_side="left", via=((20, 970), (20, 222), (380, 222)), label_at=(155, 963)),
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
        dict(src="gate", dst="retry", kind="feedback", label="RETRY / REJECT", src_side="left", dst_side="right"),
        dict(src="retry", dst="gate", kind="feedback", label="one fresh re-evaluation", src_side="top", dst_side="left", via=((180, 595), (345, 595), (345, 725), (380, 725)), label_at=(250, 588)),
        dict(src="retry", dst="settled", kind="feedback", label="exclude + new NBV", src_side="left", dst_side="left", via=((20, 725), (20, 222), (380, 222)), label_at=(150, 718)),
        dict(src="commit", dst="coverage", kind="feedback", label="accepted generation", src_side="right", dst_side="left"),
        dict(src="coverage", dst="settled", kind="feedback", label="next qualified view", src_side="right", dst_side="right", via=((1070, 977), (1070, 222), (720, 222)), label_at=(1062, 600)),
        dict(src="commit", dst="shutdown", kind="control", label="complete / terminal", src_side="bottom", dst_side="top"),
        dict(src="shutdown", dst="base", kind="control", src_side="bottom", dst_side="top"),
        dict(src="base", dst="admit", kind="control", src_side="bottom", dst_side="top"),
        dict(src="admit", dst="register", kind="data", src_side="bottom", dst_side="top"),
        dict(src="admit", dst="fusion", kind="data", label="validated frames", src_side="bottom", dst_side="top"),
        dict(src="register", dst="fusion", kind="data", label="bounded poses", src_side="right", dst_side="left"),
        dict(src="fusion", dst="outputs", kind="data", label="mesh / cloud", src_side="right", dst_side="left"),
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
        dict(src="ai", dst="foxy", kind="data", label="mask / target / scene", src_side="left", dst_side="right"),
        dict(src="planner", dst="foxy", kind="data", label="command-free plan", src_side="left", dst_side="right"),
        dict(src="enclosure", dst="optional", kind="optional", label="mechanical provision", src_side="right", dst_side="right", via=((1070, 447), (1070, 1455), (720, 1455)), label_at=(1062, 950)),
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
