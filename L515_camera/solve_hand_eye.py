#!/usr/bin/env python3
"""Solve and independently validate PiPER eye-in-hand calibration."""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def transform(rotation=None, translation=None):
    result = np.eye(4, dtype=float)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return result


def inverse(value):
    rotation = value[:3, :3].T
    return transform(rotation, -rotation @ value[:3, 3])


def rpy_matrix(rpy):
    return Rotation.from_euler("xyz", rpy).as_matrix()


def rotation_error_deg(first, second):
    delta = first[:3, :3].T @ second[:3, :3]
    return math.degrees(Rotation.from_matrix(delta).magnitude())


def mean_transform(values):
    translations = np.asarray([value[:3, 3] for value in values])
    rotations = Rotation.from_matrix(np.asarray([value[:3, :3] for value in values])).mean().as_matrix()
    return transform(rotations, translations.mean(axis=0))


def transform_dict(value):
    quaternion = Rotation.from_matrix(value[:3, :3]).as_quat()
    return {
        "translation_m": value[:3, 3].astype(float).tolist(),
        "quaternion_xyzw": quaternion.astype(float).tolist(),
        "matrix": value.astype(float).tolist(),
    }


class PiperModifiedDhFk:
    """PiPER SDK modified-DH FK, mode 0, expressed in metres."""

    def __init__(self):
        self.a = np.asarray([0, 0, 285.03, -21.98, 0, 0], dtype=float) / 1000.0
        self.alpha = np.asarray([0, -math.pi / 2, 0, math.pi / 2, -math.pi / 2, math.pi / 2])
        self.theta = np.asarray([0, -math.radians(174.22), -math.radians(100.78), 0, 0, 0])
        self.d = np.asarray([123, 0, 0, 250.75, 0, 91], dtype=float) / 1000.0

    @staticmethod
    def link(alpha, a, theta, d):
        ca, sa = math.cos(alpha), math.sin(alpha)
        ct, st = math.cos(theta), math.sin(theta)
        return np.asarray([
            [ct, -st, 0, a],
            [st * ca, ct * ca, -sa, -sa * d],
            [st * sa, ct * sa, ca, ca * d],
            [0, 0, 0, 1],
        ], dtype=float)

    def calculate(self, positions):
        result = np.eye(4)
        for index, angle in enumerate(positions[:6]):
            result = result @ self.link(
                self.alpha[index], self.a[index], angle + self.theta[index], self.d[index]
            )
        return result


def board_points(target):
    dictionary_id = getattr(cv2.aruco, target["dictionary"])
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard_create(
        int(target["squares_x"]),
        int(target["squares_y"]),
        float(target["square_length_m"]),
        float(target["marker_length_m"]),
        dictionary,
    )
    return np.asarray(board.chessboardCorners, dtype=np.float64)


def load_sample(path, fk):
    with path.open() as stream:
        data = yaml.safe_load(stream)
    target = data["target"]
    ids = np.asarray(target["charuco_ids"], dtype=int)
    image_points = np.asarray(target["charuco_corners_px"], dtype=np.float64)
    object_points = board_points(target)[ids]
    camera = data["camera"]
    camera_matrix = np.asarray(camera["k"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(camera["d"], dtype=np.float64)
    ok, rotation_vector, translation = cv2.solvePnP(
        object_points, image_points, camera_matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise RuntimeError("PnP failed for %s" % path)
    rotation, _ = cv2.Rodrigues(rotation_vector)
    camera_from_target = transform(rotation, translation.reshape(3))
    projected, _ = cv2.projectPoints(object_points, rotation_vector, translation, camera_matrix, distortion)
    reprojection = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    positions = np.asarray(data["joints"]["position_rad"][:6], dtype=float)
    return {
        "name": path.parent.name,
        "path": str(path),
        "base_from_link6": fk.calculate(positions),
        "camera_from_target": camera_from_target,
        "reprojection_rms_px": float(np.sqrt(np.mean(reprojection ** 2))),
    }


def load_group(path, fk):
    files = sorted(path.glob("capture_*/sample.yaml"))
    return [load_sample(file, fk) for file in files]


def solve(fitting):
    rotations_gripper_to_base = [item["base_from_link6"][:3, :3] for item in fitting]
    translations_gripper_to_base = [item["base_from_link6"][:3, 3] for item in fitting]
    rotations_target_to_camera = [item["camera_from_target"][:3, :3] for item in fitting]
    translations_target_to_camera = [item["camera_from_target"][:3, 3] for item in fitting]
    rotation, translation = cv2.calibrateHandEye(
        rotations_gripper_to_base,
        translations_gripper_to_base,
        rotations_target_to_camera,
        translations_target_to_camera,
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    return transform(rotation, translation.reshape(3))


def evaluate(samples, link6_from_camera, reference):
    rows = []
    for item in samples:
        estimate = item["base_from_link6"] @ link6_from_camera @ item["camera_from_target"]
        rows.append({
            "name": item["name"],
            "translation_error_mm": float(np.linalg.norm(estimate[:3, 3] - reference[:3, 3]) * 1000.0),
            "rotation_error_deg": float(rotation_error_deg(reference, estimate)),
            "reprojection_rms_px": item["reprojection_rms_px"],
            "base_from_target": estimate,
        })
    return rows


def summary(rows):
    translations = np.asarray([row["translation_error_mm"] for row in rows])
    rotations = np.asarray([row["rotation_error_deg"] for row in rows])
    return {
        "translation_rms_mm": float(np.sqrt(np.mean(translations ** 2))),
        "translation_max_mm": float(np.max(translations)),
        "rotation_rms_deg": float(np.sqrt(np.mean(rotations ** 2))),
        "rotation_max_deg": float(np.max(rotations)),
        "per_sample": [
            {key: row[key] for key in ("name", "translation_error_mm", "rotation_error_deg", "reprojection_rms_px")}
            for row in rows
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_root", type=Path)
    parser.add_argument("--translation-limit-mm", type=float, default=15.0)
    parser.add_argument("--rotation-limit-deg", type=float, default=1.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.session_root / "calibration_result.yaml"

    fk = PiperModifiedDhFk()
    fitting = load_group(args.session_root / "fitting", fk)
    validation = load_group(args.session_root / "validation", fk)
    if len(fitting) < 3:
        parser.error("at least 3 fitting samples are required")
    if not validation:
        parser.error("at least 1 held-out validation sample is required")

    link6_from_camera = solve(fitting)
    fit_target_estimates = [
        item["base_from_link6"] @ link6_from_camera @ item["camera_from_target"] for item in fitting
    ]
    base_from_target = mean_transform(fit_target_estimates)
    fit_summary = summary(evaluate(fitting, link6_from_camera, base_from_target))
    held_out_summary = summary(evaluate(validation, link6_from_camera, base_from_target))
    accepted = all(
        row["translation_error_mm"] <= args.translation_limit_mm
        and row["rotation_error_deg"] <= args.rotation_limit_deg
        for row in held_out_summary["per_sample"]
    )
    result = {
        "status": "accepted" if accepted else "rejected_by_held_out_validation",
        "model": "eye_in_hand",
        "method": "OpenCV CALIB_HAND_EYE_PARK",
        "sample_count": {"fitting": len(fitting), "held_out_validation": len(validation)},
        "fk": {
            "model": "piper_sdk_modified_dh_mode_0",
            "units": "metres and radians",
            "note": "Mode 0 matches the live controller /end_pose; mode 1 applies optional 2-degree J2/J3 offsets.",
        },
        "camera_to_link6": transform_dict(link6_from_camera),
        "target_to_base": transform_dict(base_from_target),
        "fitting_residuals": fit_summary,
        "held_out_validation": {
            "accepted": accepted,
            "acceptance_limits": {
                "translation_mm_per_sample": args.translation_limit_mm,
                "rotation_deg_per_sample": args.rotation_limit_deg,
            },
            **held_out_summary,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        yaml.safe_dump(result, stream, sort_keys=False)
    print("status:", result["status"])
    print("fitting translation RMS: %.3f mm" % fit_summary["translation_rms_mm"])
    print("fitting rotation RMS: %.3f deg" % fit_summary["rotation_rms_deg"])
    print("held-out translation errors:", [round(row["translation_error_mm"], 3) for row in held_out_summary["per_sample"]])
    print("held-out rotation errors:", [round(row["rotation_error_deg"], 3) for row in held_out_summary["per_sample"]])
    print("result:", output)
    return 0 if accepted else 2


if __name__ == "__main__":
    sys.exit(main())
