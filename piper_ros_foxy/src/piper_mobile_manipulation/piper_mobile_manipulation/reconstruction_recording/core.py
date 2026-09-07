"""Validate raw synchronized frames and persist bounded asynchronous sessions."""

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import queue
import threading
import uuid

import cv2
import numpy as np

from piper_mobile_manipulation.visibility import (
    DEFAULT_VISIBILITY_ANGLE_DEG,
    angle_degrees,
    inside_cone,
    validate_transform,
    visibility_tolerance,
)


@dataclass(frozen=True)
class RecorderConfig:
    """Frozen acquisition policy; native depth is never quantized by recording."""

    sample_rate_hz: float = 8.0
    synchronization_slop_sec: float = 0.08
    max_mask_age_sec: float = 1.0
    min_mask_area_px: int = 100
    edge_margin_px: int = 40
    minimum_valid_depth_ratio: float = 0.5
    blur_gate_enabled: bool = True
    minimum_blur_score: float = 20.0
    target_visibility_angle_tolerance_deg: float = DEFAULT_VISIBILITY_ANGLE_DEG
    recorder_visibility_angle_tolerance_deg: float = 0.0
    writer_queue_size: int = 8
    depth_scale_m: float = 0.001
    max_mask_area_growth_ratio: float = 2.0

    def __post_init__(self):
        """Reject malformed acquisition policy before accepting any frames."""
        for name, value in asdict(self).items():
            if not isinstance(value, (bool, int, float)) or not math.isfinite(
                float(value)
            ):
                raise ValueError("invalid recorder parameter: " + name)
        if not 5.0 <= self.sample_rate_hz <= 10.0:
            raise ValueError("sample_rate_hz must be within 5..10")
        if not 0 < self.synchronization_slop_sec <= 0.1 or self.max_mask_age_sec <= 0:
            raise ValueError("invalid timestamp tolerances")
        if (
            self.min_mask_area_px < 1
            or self.edge_margin_px < 0
            or self.writer_queue_size < 1
            or self.writer_queue_size > 256
            or not 0 < self.minimum_valid_depth_ratio <= 1
            or self.minimum_blur_score < 0
            or self.depth_scale_m <= 0
            or self.max_mask_area_growth_ratio < 1
        ):
            raise ValueError("invalid quality, depth scale or queue setting")
        for value in (
            self.min_mask_area_px,
            self.edge_margin_px,
            self.writer_queue_size,
        ):
            if int(value) != value:
                raise ValueError("pixel and queue settings must be integers")
        visibility_tolerance(self.target_visibility_angle_tolerance_deg)
        if self.recorder_visibility_angle_tolerance_deg != 0:
            visibility_tolerance(self.recorder_visibility_angle_tolerance_deg)
            if (
                self.recorder_visibility_angle_tolerance_deg
                > self.target_visibility_angle_tolerance_deg
            ):
                raise ValueError("recorder cone may only tighten the movement cone")

    @property
    def angle_limit(self):
        """Inherit the central movement setting unless explicitly tightened."""
        return (
            self.recorder_visibility_angle_tolerance_deg
            or self.target_visibility_angle_tolerance_deg
        )


@dataclass
class Frame:
    """One raw RGB/depth/mask observation and its timestamped provenance."""

    rgb: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    K: np.ndarray
    T_base_camera: np.ndarray
    rgb_stamp_ns: int
    depth_stamp_ns: int
    mask_stamp_ns: int
    mask_received_age_sec: float
    depth_encoding: str
    base_frame: str
    camera_frame: str
    target_id: str
    tf_stamp_ns: int
    source: str = "moving"
    mask_source: str = "sam2_target_mask"
    accepted_view: int = 0
    distortion_model: str = ""
    distortion_coefficients: tuple = ()


def distortion_valid(model, coefficients):
    """Support recorded ROS Brown-Conrady calibration, retaining legacy pinhole data."""
    values = np.asarray(coefficients, dtype=float)
    return bool(
        values.ndim == 1
        and np.isfinite(values).all()
        and ((model == "" and values.size == 0)
             or (model == "plumb_bob" and values.size == 5)
             or (model == "rational_polynomial" and values.size == 8))
    )


def intrinsics_valid(K, width, height):
    """Accept finite calibrated pinhole intrinsics for this exact RGB image."""
    K = np.asarray(K)
    return bool(
        K.shape == (3, 3)
        and np.isfinite(K).all()
        and width > 0
        and height > 0
        and K[0, 0] > 0
        and K[1, 1] > 0
        and 0 <= K[0, 2] < width
        and 0 <= K[1, 2] < height
        and np.allclose(K[2], [0, 0, 1])
        and abs(K[0, 1]) < 1e-8
        and abs(K[1, 0]) < 1e-8
    )


def mask_metrics(mask, rgb):
    """Return inclusive mask bounds, edge clearance and interior Laplacian variance."""
    selected = np.asarray(mask) > 0
    ys, xs = np.nonzero(selected)
    if not len(xs):
        raise ValueError("empty_mask")
    h, w = selected.shape
    box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    margin = min(box[0], box[1], w - 1 - box[2], h - 1 - box[3])
    # Erosion excludes the silhouette boundary, so a sharp mask edge cannot
    # make a blurred target look sharp. RGB input is explicitly RGB8.
    interior = cv2.erode(selected.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    laplacian = cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F)
    score = float(np.var(laplacian[interior])) if interior.any() else 0.0
    return box, margin, score


def atomic_json(path, data):
    """Replace a manifest only after flushing its complete finite JSON record."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


class RecordingSession:
    """
    Own one frozen target, quality admission and a drop-newest disk queue.

    Call submit from the recorder processing thread, never an arm callback.
    Only the writer thread encodes images, creates directories and writes files.
    A journal manifest is replaced after each committed frame for crash recovery.
    """

    def __init__(self, root, session_id, target, config, target_id=None):
        """Freeze session identity and start the bounded background writer."""
        self.config = config
        self.session_id = str(session_id)
        self.target_id = str(target_id or session_id)
        self.target = np.asarray(target, dtype=float)
        if self.target.shape != (3,) or not np.isfinite(self.target).all():
            raise ValueError("invalid frozen coverage_target_center")
        self.path = Path(root) / ("reconstruction_" + uuid.uuid4().hex + ".inprogress")
        self.final_path = None
        self.rejected = Counter()
        self.accepted = 0
        self.mission_capture_count = 0
        self.last_stamp = -1
        self.last_sample_stamp = -1
        self.calibration = None
        self.reference_mask_area = None
        self.frames = []
        self.error = ""
        self.state = "recording"
        self.closed = False
        self._queue = queue.Queue(maxsize=config.writer_queue_size)
        self._closing = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._write_loop, name="rgbd-writer", daemon=True
        )
        self._thread.start()

    def reject(self, reason):
        """Count ordinary rejection without per-frame error logging."""
        with self._lock:
            self.rejected[reason] += 1
        return False

    def submit(self, frame):
        """Admit immutable raw evidence; never wait for the disk queue."""
        if self.closed or self.error:
            return self.reject("recorder_unavailable")
        cfg = self.config
        if frame.source not in (
            "first_capture",
            "moving",
            "stationary_capture",
            "final_capture",
        ):
            return self.reject("invalid_frame_source")
        if self.accepted == 0 and frame.source != "first_capture":
            return self.reject("missing_initial_keyframe")
        if frame.rgb_stamp_ns <= self.last_stamp or frame.rgb_stamp_ns <= 0:
            return self.reject("duplicate_or_out_of_order")
        if abs(frame.rgb_stamp_ns - frame.depth_stamp_ns) > round(
            cfg.synchronization_slop_sec * 1e9
        ):
            return self.reject("rgb_depth_unsynchronised")
        if frame.mask is None:
            return self.reject("missing_mask")
        if (
            frame.mask_stamp_ns != frame.rgb_stamp_ns
            or frame.mask_received_age_sec < 0
            or frame.mask_received_age_sec > cfg.max_mask_age_sec
        ):
            return self.reject("stale_mask")
        if frame.target_id != self.target_id or frame.mask_source != "sam2_target_mask":
            return self.reject("wrong_target_or_mask_source")
        if (
            frame.rgb.ndim != 3
            or frame.rgb.shape[2] != 3
            or frame.rgb.dtype != np.uint8
            or frame.depth.shape != frame.rgb.shape[:2]
            or frame.mask.shape != frame.rgb.shape[:2]
        ):
            return self.reject("resolution_mismatch")
        h, w = frame.depth.shape
        if not intrinsics_valid(frame.K, w, h) or not distortion_valid(
            frame.distortion_model, frame.distortion_coefficients
        ):
            return self.reject("invalid_intrinsics")
        identity = (
            h,
            w,
            frame.base_frame,
            frame.camera_frame,
            tuple(np.asarray(frame.K).ravel()),
            frame.distortion_model,
            tuple(frame.distortion_coefficients),
        )
        if self.calibration is not None and identity != self.calibration:
            return self.reject("intrinsics_or_resolution_changed")
        if (
            not frame.base_frame
            or not frame.camera_frame
            or frame.tf_stamp_ns != frame.rgb_stamp_ns
        ):
            return self.reject("missing_tf")
        try:
            transform = validate_transform(frame.T_base_camera)
            angle = angle_degrees(transform[:3, 2], self.target - transform[:3, 3])
        except (ValueError, TypeError):
            return self.reject("invalid_tf_or_target_vector")
        if not inside_cone(angle, cfg.angle_limit):
            return self.reject("outside_angle_tolerance")
        area = int(np.count_nonzero(frame.mask))
        if area < cfg.min_mask_area_px:
            return self.reject("empty_or_tiny_mask")
        box, margin, blur = mask_metrics(frame.mask, frame.rgb)
        if margin <= cfg.edge_margin_px:
            return self.reject("target_clipped")
        if frame.depth_encoding == "16UC1" and frame.depth.dtype == np.uint16:
            scale = cfg.depth_scale_m
        elif frame.depth_encoding == "32FC1" and frame.depth.dtype == np.float32:
            scale = 1.0
        else:
            return self.reject("invalid_depth_encoding")
        target_depth = frame.depth[frame.mask > 0]
        if not np.isfinite(target_depth).all():
            return self.reject("nonfinite_depth")
        valid = target_depth > 0
        ratio = float(np.count_nonzero(valid)) / area
        if ratio < cfg.minimum_valid_depth_ratio:
            return self.reject("insufficient_depth")
        # Perspective-normalized area catches sudden mask leakage onto background.
        normalized_area = (
            area
            * float(np.median(target_depth[valid]) * scale) ** 2
            / (frame.K[0, 0] * frame.K[1, 1])
        )
        if (
            self.reference_mask_area is not None
            and normalized_area
            > self.reference_mask_area * cfg.max_mask_area_growth_ratio
        ):
            return self.reject("mask_expansion")
        if cfg.blur_gate_enabled and blur < cfg.minimum_blur_score:
            return self.reject("excessive_blur")
        if (
            frame.source == "moving"
            and frame.rgb_stamp_ns - self.last_sample_stamp
            < round(1e9 / cfg.sample_rate_hz)
        ):
            return self.reject("sample_rate_throttled")
        metadata = {
            "index": self.accepted,
            "source": frame.source,
            "accepted_view": frame.accepted_view,
            "rgb_stamp_ns": int(frame.rgb_stamp_ns),
            "depth_stamp_ns": int(frame.depth_stamp_ns),
            "mask_stamp_ns": int(frame.mask_stamp_ns),
            "tf_stamp_ns": int(frame.tf_stamp_ns),
            "synchronization_delta_sec": abs(frame.rgb_stamp_ns - frame.depth_stamp_ns)
            * 1e-9,
            "base_frame": frame.base_frame,
            "camera_optical_frame": frame.camera_frame,
            "T_base_camera": transform.tolist(),
            "K": np.asarray(frame.K).tolist(),
            "distortion_model": frame.distortion_model,
            "distortion_coefficients": list(frame.distortion_coefficients),
            "width": w,
            "height": h,
            "depth_encoding": frame.depth_encoding,
            "depth_scale_m": scale,
            "depth_dtype": str(frame.depth.dtype),
            "angular_offset_deg": angle,
            "target_bbox_xyxy": box,
            "mask_area_px": area,
            "edge_margin_px": margin,
            "blur_laplacian_variance": blur,
            "valid_target_depth_ratio": ratio,
            "target_id": frame.target_id,
            "mask_source": frame.mask_source,
            "normalized_mask_area_m2": normalized_area,
        }
        # Copy buffers before handing ownership to another thread; caller may reuse them.
        item = (
            metadata,
            frame.rgb.copy(),
            frame.depth.copy(),
            (frame.mask > 0).astype(np.uint8) * 255,
        )
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            return self.reject("writer_queue_full")
        self.calibration = identity
        if self.reference_mask_area is None:
            self.reference_mask_area = normalized_area
        self.last_stamp = frame.rgb_stamp_ns
        self.last_sample_stamp = frame.rgb_stamp_ns
        self.accepted += 1
        return True

    def manifest(self):
        """Expose disk-committed counts separately from admitted queue entries."""
        with self._lock:
            rejected = dict(self.rejected)
        return dict(
            schema_version=1,
            dataset_type="piper_raw_rgbd_reconstruction",
            session_id=self.session_id,
            target_id=self.target_id,
            coverage_target_center=self.target.tolist(),
            pose_convention=(
                'T_base_camera maps camera optical coordinates (+Z forward, +X '
                'right, +Y down) to base coordinates; translation metres'
            ),
            configuration=asdict(self.config),
            state=self.state,
            complete=self.state == "complete" and not self.error,
            error=self.error,
            accepted=len(self.frames),
            admitted=self.accepted,
            rejected_by_reason=rejected,
            rejected=sum(rejected.values()),
            dropped=sum(
                rejected.get(key, 0)
                for key in (
                    "writer_queue_full",
                    "input_queue_full",
                    "keyframe_queue_full",
                    "event_queue_full",
                )
            ),
            mission_capture_count=self.mission_capture_count,
            frames=list(self.frames),
        )

    def _persist_frame(self, item):
        metadata, rgb, depth, mask = item
        name = "%08d" % metadata["index"]
        temporary = self.path / ("." + name + ".tmp")
        destination = self.path / name
        temporary.mkdir()
        if not cv2.imwrite(
            str(temporary / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ):
            raise OSError("RGB PNG encoding failed")
        if not cv2.imwrite(str(temporary / "mask.png"), mask):
            raise OSError("mask PNG encoding failed")
        with (temporary / "depth.npy").open("wb") as stream:
            np.save(stream, depth, allow_pickle=False)
        metadata.update(
            rgb=name + "/rgb.png", depth=name + "/depth.npy", mask=name + "/mask.png"
        )
        atomic_json(temporary / "frame.json", metadata)
        os.rename(str(temporary), str(destination))
        self.frames.append(metadata)
        atomic_json(self.path / "manifest.json", self.manifest())

    def _write_loop(self):
        try:
            self.path.mkdir(parents=True, exist_ok=False)
            atomic_json(self.path / "manifest.json", self.manifest())
            while not self._closing.is_set() or not self._queue.empty():
                try:
                    item = self._queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    self._persist_frame(item)
                finally:
                    self._queue.task_done()
        except Exception as error:
            self.error = "%s: %s" % (type(error).__name__, error)
            self.state = "incomplete"
            self.reject("writer_error")
        finally:
            try:
                if self.state == "recording":
                    self.state = "incomplete"
                if not self.frames or self.frames[0]["source"] != "first_capture":
                    self.state = "incomplete"
                if self.state == "complete":
                    # Final workflow completion identifies which accepted view was last.
                    for entry in reversed(self.frames):
                        if entry["accepted_view"] == self.mission_capture_count:
                            entry["source"] = (
                                "final_capture" if entry["index"] else "first_capture"
                            )
                            atomic_json(
                                self.path / ("%08d" % entry["index"]) / "frame.json",
                                entry,
                            )
                            break
                atomic_json(self.path / "manifest.json", self.manifest())
                destination = self.path.with_name(
                    self.path.name.replace(".inprogress", "." + self.state)
                )
                os.rename(str(self.path), str(destination))
                self.final_path = destination
            except Exception as error:
                self.error = self.error or str(error)

    def close(self, state="incomplete", timeout=10.0):
        """Bound shutdown wait; a stuck writer leaves an inspectable in-progress journal."""
        if not self.closed:
            self.closed = True
            self.state = (
                state if state in ("complete", "cancelled", "failed") else "incomplete"
            )
            self._closing.set()
        self._thread.join(timeout)
        return not self._thread.is_alive()
