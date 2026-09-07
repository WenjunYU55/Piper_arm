"""Validate native raw RGB-D sessions and adapt to pinned BundleSDF custom data."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

import cv2
import numpy as np

from distortion import rectification_maps

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "piper_ros_foxy/src/piper_mobile_manipulation"))
from piper_mobile_manipulation.reconstruction_recording.core import (  # noqa: E402
    atomic_json,
    intrinsics_valid,
    distortion_valid,
)
from piper_mobile_manipulation.visibility import validate_transform  # noqa: E402

UPSTREAM_COMMIT = "ffa67d425240b5b76d2e387a7dd3d3735a7cf1a1"


def artifact(root, relative):
    """Reject absolute paths, escaping symlinks and missing artifacts."""
    relative = Path(relative)
    path = (root / relative).resolve()
    if (
        relative.is_absolute()
        or root.resolve() not in path.parents
        or not path.is_file()
    ):
        raise ValueError("missing or escaping artifact: " + str(relative))
    return path


def validate(session, minimum_frames=5, allow_incomplete=False):
    """Validate all persisted evidence before conversion or GPU execution."""
    root = Path(session).resolve()
    if root.suffix.lower() in (".mp4", ".avi", ".mov"):
        raise ValueError(
            (
                'RGB video alone is unsupported: aligned depth, calibration and '
                'target masks are required'
            )
        )
    if not root.is_dir():
        raise ValueError("input must be a native reconstruction session directory")
    with (root / "manifest.json").open() as stream:
        manifest = json.load(stream)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("dataset_type") != "piper_raw_rgbd_reconstruction"
    ):
        raise ValueError(
            "unsupported dataset schema; stationary median-depth scans are not raw sequences"
        )
    if not manifest.get("complete") and not allow_incomplete:
        raise ValueError(
            "incomplete session: inspect it, then explicitly use --allow-incomplete if appropriate"
        )
    entries = manifest.get("frames", [])
    if len(entries) != manifest.get("accepted") or len(entries) < minimum_frames:
        raise ValueError(
            "frame count mismatch or fewer than %d usable frames" % minimum_frames
        )
    if not entries or entries[0]["source"] != "first_capture":
        raise ValueError("initial synchronized accepted-capture frame is missing")
    target = np.asarray(manifest.get("coverage_target_center"), dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("invalid frozen target")
    previous = 0
    calibration = None
    expected_files = set()
    for i, entry in enumerate(entries):
        if (
            entry["index"] != i
            or not isinstance(entry["rgb_stamp_ns"], int)
            or entry["rgb_stamp_ns"] <= previous
        ):
            raise ValueError("nonmonotonic timestamps or frame indices")
        previous = entry["rgb_stamp_ns"]
        if entry["mask_stamp_ns"] != previous or entry["tf_stamp_ns"] != previous:
            raise ValueError("mask/TF does not match exact image time")
        delta = abs(previous - entry["depth_stamp_ns"]) * 1e-9
        if delta > manifest["configuration"]["synchronization_slop_sec"] + 1e-12:
            raise ValueError("RGB/depth synchronization limit exceeded")
        if abs(delta - entry["synchronization_delta_sec"]) > 1e-12:
            raise ValueError("incorrect synchronization metadata")
        paths = [artifact(root, entry[key]) for key in ("rgb", "depth", "mask")]
        expected_files.update(paths)
        rgb, depth, mask = (
            cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED),
            np.load(str(paths[1]), allow_pickle=False),
            cv2.imread(str(paths[2]), cv2.IMREAD_UNCHANGED),
        )
        if rgb is None or mask is None:
            raise ValueError("corrupt RGB/mask PNG")
        if (
            rgb.dtype != np.uint8
            or rgb.shape != (entry["height"], entry["width"], 3)
            or depth.shape != rgb.shape[:2]
            or mask.shape != depth.shape
        ):
            raise ValueError("RGB/depth/mask dimension mismatch")
        if (
            mask.dtype != np.uint8
            or not np.isin(mask, [0, 255]).all()
            or not mask.any()
        ):
            raise ValueError("invalid or empty binary mask")
        scale = float(entry["depth_scale_m"])
        if (
            not np.isfinite(scale)
            or scale <= 0
            or str(depth.dtype) != entry["depth_dtype"]
            or (entry["depth_encoding"], str(depth.dtype))
            not in (("16UC1", "uint16"), ("32FC1", "float32"))
            or (entry["depth_encoding"] == "32FC1" and scale != 1.0)
        ):
            raise ValueError("invalid depth encoding or units")
        selected = depth[mask > 0]
        if (
            not np.isfinite(selected).all()
            or np.count_nonzero(selected > 0) / len(selected)
            < manifest["configuration"]["minimum_valid_depth_ratio"]
        ):
            raise ValueError("invalid target depth")
        if not intrinsics_valid(entry["K"], entry["width"], entry["height"]):
            raise ValueError("invalid intrinsics")
        model = entry.get("distortion_model", "")
        coefficients = entry.get("distortion_coefficients", [])
        if not distortion_valid(model, coefficients):
            raise ValueError("invalid or unsupported distortion calibration")
        identity = (
            entry["width"],
            entry["height"],
            entry["base_frame"],
            entry["camera_optical_frame"],
            json.dumps(entry["K"]),
            model,
            tuple(coefficients),
        )
        if (
            not entry["base_frame"]
            or not entry["camera_optical_frame"]
            or (calibration is not None and calibration != identity)
        ):
            raise ValueError("intrinsics/resolution/frame convention changed")
        calibration = identity
        validate_transform(entry["T_base_camera"])
        frame_json = artifact(root, "%08d/frame.json" % i)
        if json.loads(frame_json.read_text()) != entry:
            raise ValueError("frame metadata differs from session manifest")
    actual_files = {
        p.resolve()
        for pattern in ("[0-9]*/rgb.png", "[0-9]*/depth.npy", "[0-9]*/mask.png")
        for p in root.glob(pattern)
    }
    if actual_files != expected_files:
        raise ValueError("artifact counts disagree with manifest")
    return (
        dict(
            valid=True,
            input=str(root),
            frames=len(entries),
            complete=bool(manifest["complete"]),
            pose_convention=manifest["pose_convention"],
            upstream_commit=UPSTREAM_COMMIT,
        ),
        manifest,
    )


def convert(
    session,
    output,
    stride=1,
    start=0,
    end=None,
    allow_incomplete=False,
    minimum_frames=5,
):
    """Write matching PNG basenames and cam_K.txt; preserve robot poses as sidecar only."""
    report, manifest = validate(
        session, minimum_frames=1, allow_incomplete=allow_incomplete
    )
    if stride < 1 or start < 0 or (end is not None and end <= start):
        raise ValueError("invalid frame selection")
    entries = manifest["frames"][start:end:stride]
    if len(entries) < minimum_frames:
        raise ValueError("selection has fewer than %d frames" % minimum_frames)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    # Upstream uses string.replace("rgb", "depth") on the entire path.
    # Guard its parent-path bug without modifying the pinned loader.
    if "rgb" in str(output):
        raise ValueError(
            'BundleSDF output parent path must not contain lowercase rgb; '
            'the upstream loader replaces that text globally'
        )
    temporary = output.with_name(
        "." + output.name + "." + uuid.uuid4().hex + ".inprogress"
    )
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        for folder in ("rgb", "depth", "masks"):
            (temporary / folder).mkdir()
        mapping = []
        for i, entry in enumerate(entries):
            filename = "%08d.png" % i
            root = Path(session).resolve()
            depth = (
                np.load(str(artifact(root, entry["depth"])), allow_pickle=False).astype(
                    np.float64
                )
                * entry["depth_scale_m"]
                * 1000.0
            )
            if np.any(depth[np.isfinite(depth)] > 65535):
                raise ValueError("depth exceeds BundleSDF uint16 millimetre range")
            depth[~np.isfinite(depth) | (depth < 0)] = 0
            encoded = np.rint(depth).astype(np.uint16)
            coefficients = np.asarray(entry.get("distortion_coefficients", []), dtype=float)
            if np.any(coefficients):
                # BundleSDF's loader is pinhole-only. Rectify the entire aligned
                # tuple into the same K; nearest sampling preserves depth edges
                # and binary semantic support. Native files remain untouched.
                K = np.asarray(entry["K"], dtype=float)
                maps = rectification_maps(K, coefficients, entry["width"], entry["height"])
                encoded = cv2.remap(encoded, *maps, cv2.INTER_NEAREST)
                for key, folder, interpolation in (
                    ("rgb", "rgb", cv2.INTER_LINEAR),
                    ("mask", "masks", cv2.INTER_NEAREST),
                ):
                    source = cv2.imread(str(artifact(root, entry[key])), cv2.IMREAD_UNCHANGED)
                    rectified = cv2.remap(source, *maps, interpolation)
                    if key == "mask" and not rectified.any():
                        raise ValueError("rectification removed the target mask")
                    if key == "mask":
                        ratio = np.count_nonzero(encoded[rectified > 0]) / np.count_nonzero(rectified)
                        if ratio < manifest["configuration"]["minimum_valid_depth_ratio"]:
                            raise ValueError("rectification left insufficient target depth")
                    if not cv2.imwrite(str(temporary / folder / filename), rectified):
                        raise OSError("rectified image PNG encoding failed")
            if not cv2.imwrite(str(temporary / "depth" / filename), encoded):
                raise OSError("depth PNG encoding failed")
            if not np.any(coefficients):
                shutil.copyfile(
                    str(artifact(root, entry["rgb"])), str(temporary / "rgb" / filename)
                )
                shutil.copyfile(
                    str(artifact(root, entry["mask"])), str(temporary / "masks" / filename)
                )
            mapping.append(
                dict(
                    filename=filename,
                    native_frame=entry["index"],
                    rgb_stamp_ns=entry["rgb_stamp_ns"],
                    T_base_camera=entry["T_base_camera"],
                )
            )
        np.savetxt(
            str(temporary / "cam_K.txt"), np.asarray(entries[0]["K"]), fmt="%.17g"
        )
        report.update(
            converted_frames=len(entries),
            calibration=dict(
                input_distortion_model=entries[0].get("distortion_model", ""),
                input_distortion_coefficients=entries[0].get("distortion_coefficients", []),
                input_K=entries[0]["K"], output_K=entries[0]["K"],
                applied=bool(np.any(entries[0].get("distortion_coefficients", []))),
                output_distortion_coefficients=[0.0] * 5,
                depth_convention="axial Z, not radial range; pixel remap does not change Z",
            ),
            rectification=(
                "Recorded Brown-Conrady distortion to pinhole with unchanged K; "
                "RGB linear, aligned depth and mask nearest; zero border"),
            depth_conversion=(
                'uint16 millimetres; round nearest, maximum quantization error 0.5 '
                'mm'
            ),
            stride=stride,
            start=start,
            end=end,
        )
        atomic_json(temporary / "validation.json", report)
        atomic_json(
            temporary / "robot_camera_poses.json",
            dict(
                pose_convention=manifest["pose_convention"],
                frames=mapping,
                upstream_use=(
                    'diagnostic only; not object-in-camera estimates or annotated '
                    'ground truth'
                ),
            ),
        )
        atomic_json(temporary / "input_configuration.json", manifest["configuration"])
        os.rename(str(temporary), str(output))
    except Exception:
        shutil.rmtree(str(temporary))
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_session", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--minimum-frames", type=int, default=5)
    args = parser.parse_args()
    try:
        if args.validate_only:
            report, _ = validate(
                args.input_session, args.minimum_frames, args.allow_incomplete
            )
        elif args.output:
            report = convert(
                args.input_session,
                args.output,
                args.frame_stride,
                args.start_frame,
                args.end_frame,
                args.allow_incomplete,
                args.minimum_frames,
            )
        else:
            parser.error("--output or --validate-only is required")
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, "Dataset rejected: %s\n" % error)


if __name__ == "__main__":
    main()
