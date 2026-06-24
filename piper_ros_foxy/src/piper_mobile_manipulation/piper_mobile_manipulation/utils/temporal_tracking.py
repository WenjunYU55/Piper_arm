#!/usr/bin/env python3
"""Lightweight target-mask propagation and heavy-refresh policy for RGB-D sequences."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class TemporalTrackerConfig:
    max_features: int = 120
    min_features: int = 4
    feature_quality: float = 0.01
    feature_min_distance_px: float = 4.0
    feature_mask_dilation_px: int = 5
    lk_window_px: int = 21
    lk_max_level: int = 3
    max_forward_backward_error_px: float = 1.5
    ransac_reprojection_px: float = 2.5
    min_tracking_confidence: float = 0.50
    low_tracking_confidence: float = 0.25
    max_missed_frames: int = 5
    refresh_interval_frames: int = 90
    scene_change_threshold: float = 45.0
    min_mask_area_px: int = 100
    min_area_ratio: float = 0.50
    max_area_ratio: float = 2.00
    min_depth_valid_ratio: float = 0.40
    depth_margin_m: float = 0.03
    target_near_margin_px: int = 6
    min_depth_obstacle_support_px: int = 20
    obstacle_persistence_frames: int = 3
    enable_color_correction: bool = False
    hsv_lower: tuple[int, int, int] = (35, 80, 60)
    hsv_upper: tuple[int, int, int] = (88, 255, 255)
    color_search_margin_px: int = 80
    color_min_area_px: int = 100
    color_max_centroid_shift_px: float = 100.0
    color_min_area_ratio: float = 0.30
    color_max_area_ratio: float = 2.50
    color_depth_tolerance_m: float = 0.15
    require_color_correction: bool = True
    enable_adaptive_appearance: bool = True
    appearance_distance_threshold: float = 3.5
    appearance_min_chroma_sigma: float = 8.0
    appearance_max_chroma_sigma: float = 35.0
    appearance_update_rate: float = 0.05
    use_hsv_fallback: bool = True


@dataclass
class TemporalTrackResult:
    frame_index: int
    mode: str
    tracking_confidence: float
    target_valid: bool
    target_predicted: bool
    heavy_refresh_requested: bool
    heavy_refresh_reason: str
    feature_count: int
    feature_survival_ratio: float
    affine_inlier_ratio: float
    forward_backward_error_px: float
    mask_area_px: int
    mask_area_ratio: float
    depth_valid_ratio: float
    target_depth_m: float | None
    scene_change_score: float
    missed_frames: int
    obstacle_candidate: bool
    obstacle_persistent: bool
    obstacle_support_px: int
    obstacle_persistence_count: int
    color_correction_used: bool = False
    color_support_ratio: float = 0.0
    appearance_correction_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def depth_to_meters(depth: np.ndarray | None) -> np.ndarray | None:
    if depth is None:
        return None
    if np.issubdtype(depth.dtype, np.integer):
        return depth.astype(np.float32, copy=False) * 0.001
    depth_m = depth.astype(np.float32, copy=False)
    valid = depth_m[np.isfinite(depth_m) & (depth_m > 0.0)]
    if valid.size and float(np.median(valid)) > 20.0:
        return depth_m * 0.001
    return depth_m


def largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary > 0
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == component


def mask_depth_metrics(mask: np.ndarray, depth_m: np.ndarray | None) -> tuple[float, float | None]:
    area = int(np.count_nonzero(mask))
    if depth_m is None or depth_m.shape[:2] != mask.shape[:2] or area == 0:
        return 0.0, None
    values = depth_m[mask & np.isfinite(depth_m) & (depth_m > 0.0)]
    ratio = float(values.size / area)
    median = float(np.median(values)) if values.size else None
    return ratio, median


class TemporalMaskTracker:
    def __init__(self, config: TemporalTrackerConfig | None = None):
        self.config = config or TemporalTrackerConfig()
        self.reset()

    def reset(self) -> None:
        self.frame_index = -1
        self.previous_gray: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.points: np.ndarray | None = None
        self.reference_area = 0
        self.target_depth_m: float | None = None
        self.missed_frames = 0
        self.frames_since_refresh = 0
        self.obstacle_persistence = 0
        self.obstacle_refresh_latched = False
        self.heavy_refresh_latched = False
        self.appearance_center: np.ndarray | None = None
        self.appearance_scale: np.ndarray | None = None
        self.initialized = False

    def initialize(
        self,
        rgb_bgr: np.ndarray,
        target_mask: np.ndarray,
        depth: np.ndarray | None = None,
    ) -> TemporalTrackResult:
        gray = self._gray(rgb_bgr)
        mask = largest_component(target_mask)
        area = int(np.count_nonzero(mask))
        if area < self.config.min_mask_area_px:
            raise ValueError("initial target mask area %d < %d" % (area, self.config.min_mask_area_px))
        depth_m = depth_to_meters(depth)
        depth_ratio, target_depth = mask_depth_metrics(mask, depth_m)
        self.frame_index = 0
        self.previous_gray = gray
        self.mask = mask
        self.points = self._detect_features(gray, mask)
        self.reference_area = area
        self.target_depth_m = target_depth
        self.missed_frames = 0
        self.frames_since_refresh = 0
        self.obstacle_persistence = 0
        self.obstacle_refresh_latched = False
        self.heavy_refresh_latched = False
        self._initialize_appearance(rgb_bgr, mask)
        self.initialized = True
        return TemporalTrackResult(
            frame_index=0,
            mode="INITIALIZED",
            tracking_confidence=1.0,
            target_valid=True,
            target_predicted=False,
            heavy_refresh_requested=False,
            heavy_refresh_reason="",
            feature_count=self._point_count(self.points),
            feature_survival_ratio=1.0,
            affine_inlier_ratio=1.0,
            forward_backward_error_px=0.0,
            mask_area_px=area,
            mask_area_ratio=1.0,
            depth_valid_ratio=depth_ratio,
            target_depth_m=target_depth,
            scene_change_score=0.0,
            missed_frames=0,
            obstacle_candidate=False,
            obstacle_persistent=False,
            obstacle_support_px=0,
            obstacle_persistence_count=0,
        )

    def apply_heavy_refresh(
        self,
        rgb_bgr: np.ndarray,
        target_mask: np.ndarray,
        depth: np.ndarray | None = None,
    ) -> TemporalTrackResult:
        frame_index = self.frame_index
        obstacle_persistence = self.obstacle_persistence
        obstacle_refresh_latched = self.obstacle_refresh_latched
        result = self.initialize(rgb_bgr, target_mask, depth)
        self.frame_index = frame_index
        self.obstacle_persistence = obstacle_persistence
        self.obstacle_refresh_latched = obstacle_refresh_latched
        result.mode = "REFRESHED"
        result.frame_index = frame_index
        return result

    def step(self, rgb_bgr: np.ndarray, depth: np.ndarray | None = None) -> tuple[TemporalTrackResult, np.ndarray]:
        if not self.initialized or self.previous_gray is None or self.mask is None:
            raise RuntimeError("tracker must be initialized before step")
        self.frame_index += 1
        self.frames_since_refresh += 1
        current_gray = self._gray(rgb_bgr)
        depth_m = depth_to_meters(depth)
        scene_change = float(np.mean(cv2.absdiff(self.previous_gray, current_gray)))

        propagated, metrics = self._propagate(self.previous_gray, current_gray, self.mask, self.points)
        confidence = float(metrics["confidence"])
        color_correction_used = False
        appearance_correction_used = False
        color_support_ratio = 0.0
        if self.config.enable_color_correction:
            color_prediction = propagated if propagated is not None else self.mask
            corrected, color_support_ratio = self._color_correct(rgb_bgr, color_prediction, depth_m)
            if corrected is not None:
                propagated = corrected
                color_correction_used = True
                appearance_correction_used = self.appearance_center is not None
                confidence = max(confidence, 0.90)
                self._update_appearance(rgb_bgr, corrected)
            elif self.config.require_color_correction:
                propagated = None
                confidence = 0.0
        area = int(np.count_nonzero(propagated)) if propagated is not None else 0
        area_ratio = float(area / max(1, self.reference_area))
        area_valid = (
            area >= self.config.min_mask_area_px
            and self.config.min_area_ratio <= area_ratio <= self.config.max_area_ratio
        )
        if not area_valid:
            confidence = 0.0

        refresh_reasons = []
        if scene_change >= self.config.scene_change_threshold:
            refresh_reasons.append("scene_change")
        if self.frames_since_refresh >= self.config.refresh_interval_frames:
            refresh_reasons.append("periodic_refresh")

        accepted = propagated is not None and confidence >= self.config.low_tracking_confidence and area_valid
        if accepted:
            self.mask = propagated
            self.previous_gray = current_gray
            self.points = self._detect_features(current_gray, propagated)
            depth_ratio, measured_depth = mask_depth_metrics(propagated, depth_m)
            if measured_depth is not None and depth_ratio >= self.config.min_depth_valid_ratio:
                if self.target_depth_m is None:
                    self.target_depth_m = measured_depth
                else:
                    self.target_depth_m = 0.9 * self.target_depth_m + 0.1 * measured_depth
        else:
            depth_ratio, measured_depth = mask_depth_metrics(self.mask, depth_m)

        if confidence >= self.config.min_tracking_confidence and accepted:
            mode = "TRACKING"
            self.missed_frames = 0
        elif accepted:
            mode = "LOW_CONFIDENCE"
            self.missed_frames += 1
            refresh_reasons.append("low_tracking_confidence")
        else:
            self.missed_frames += 1
            refresh_reasons.append("tracking_failed")
            mode = "REFRESH_REQUIRED"

        if self.missed_frames > self.config.max_missed_frames:
            mode = "LOST"
            refresh_reasons.append("miss_timeout")

        obstacle_support = self._depth_obstacle_support(self.mask, depth_m)
        obstacle_candidate = obstacle_support >= self.config.min_depth_obstacle_support_px
        if obstacle_candidate:
            self.obstacle_persistence += 1
        else:
            self.obstacle_persistence = 0
            self.obstacle_refresh_latched = False
        obstacle_persistent = self.obstacle_persistence >= self.config.obstacle_persistence_frames
        if obstacle_persistent and not self.obstacle_refresh_latched:
            refresh_reasons.append("persistent_obstacle")
            self.obstacle_refresh_latched = True

        refresh_condition = bool(refresh_reasons) or mode in ("LOW_CONFIDENCE", "REFRESH_REQUIRED", "LOST")
        heavy_requested = refresh_condition and not self.heavy_refresh_latched
        if heavy_requested:
            self.heavy_refresh_latched = True
        target_valid = mode != "LOST" and self.mask is not None
        result = TemporalTrackResult(
            frame_index=self.frame_index,
            mode=mode,
            tracking_confidence=confidence,
            target_valid=target_valid,
            target_predicted=mode in ("LOW_CONFIDENCE", "REFRESH_REQUIRED"),
            heavy_refresh_requested=heavy_requested,
            heavy_refresh_reason=(";".join(dict.fromkeys(refresh_reasons)) if heavy_requested else ""),
            feature_count=int(metrics["feature_count"]),
            feature_survival_ratio=float(metrics["survival_ratio"]),
            affine_inlier_ratio=float(metrics["inlier_ratio"]),
            forward_backward_error_px=float(metrics["fb_error"]),
            mask_area_px=int(np.count_nonzero(self.mask)) if self.mask is not None else 0,
            mask_area_ratio=area_ratio,
            depth_valid_ratio=depth_ratio,
            target_depth_m=self.target_depth_m if self.target_depth_m is not None else measured_depth,
            scene_change_score=scene_change,
            missed_frames=self.missed_frames,
            obstacle_candidate=obstacle_candidate,
            obstacle_persistent=obstacle_persistent,
            obstacle_support_px=obstacle_support,
            obstacle_persistence_count=self.obstacle_persistence,
            color_correction_used=color_correction_used,
            color_support_ratio=color_support_ratio,
            appearance_correction_used=appearance_correction_used,
        )
        return result, self.mask.copy()

    def _color_correct(
        self,
        rgb_bgr: np.ndarray,
        predicted_mask: np.ndarray,
        depth_m: np.ndarray | None,
    ) -> tuple[np.ndarray | None, float]:
        """Associate a target-local HSV component with the optical-flow prediction."""
        color = self._appearance_binary_mask(rgb_bgr)
        color = cv2.morphologyEx(color, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        color = cv2.morphologyEx(color, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        margin = max(1, int(self.config.color_search_margin_px))
        search = np.zeros(predicted_mask.shape, dtype=bool)
        predicted_y, predicted_x = np.nonzero(predicted_mask)
        if predicted_x.size == 0:
            return None, 0.0
        x0 = max(0, int(predicted_x.min()) - margin)
        x1 = min(search.shape[1], int(predicted_x.max()) + margin + 1)
        y0 = max(0, int(predicted_y.min()) - margin)
        y1 = min(search.shape[0], int(predicted_y.max()) + margin + 1)
        search[y0:y1, x0:x1] = True
        color = ((color > 0) & search).astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(color, connectivity=8)
        if count <= 1:
            return None, 0.0

        predicted_area = max(1, int(np.count_nonzero(predicted_mask)))
        predicted_center = np.array(
            [float(np.mean(predicted_x)), float(np.mean(predicted_y))], dtype=np.float32
        )
        best_mask = None
        best_score = -math.inf
        best_support = 0.0
        for component in range(1, count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            area_ratio = float(area / max(1, self.reference_area))
            if area < self.config.color_min_area_px:
                continue
            if not self.config.color_min_area_ratio <= area_ratio <= self.config.color_max_area_ratio:
                continue
            distance = float(np.linalg.norm(centroids[component] - predicted_center))
            if distance > self.config.color_max_centroid_shift_px:
                continue
            candidate = labels == component
            if depth_m is not None and self.target_depth_m is not None:
                _, candidate_depth = mask_depth_metrics(candidate, depth_m)
                if (
                    candidate_depth is not None
                    and abs(candidate_depth - self.target_depth_m) > self.config.color_depth_tolerance_m
                ):
                    continue
            intersection = int(np.count_nonzero(candidate & predicted_mask))
            union = int(np.count_nonzero(candidate | predicted_mask))
            iou = float(intersection / union) if union else 0.0
            proximity = max(0.0, 1.0 - distance / max(1.0, self.config.color_max_centroid_shift_px))
            area_score = min(area, predicted_area) / max(area, predicted_area)
            score = 2.0 * iou + proximity + area_score
            if score > best_score:
                best_score = score
                best_mask = candidate
                best_support = float(intersection / predicted_area)
        return best_mask, best_support

    def _initialize_appearance(self, rgb_bgr: np.ndarray, mask: np.ndarray) -> None:
        self.appearance_center = None
        self.appearance_scale = None
        if not self.config.enable_adaptive_appearance:
            return
        values = self._lab_chroma_values(rgb_bgr, mask)
        if values.shape[0] < self.config.color_min_area_px:
            return
        center, scale = self._robust_appearance_statistics(values)
        self.appearance_center = center
        self.appearance_scale = scale

    def _update_appearance(self, rgb_bgr: np.ndarray, mask: np.ndarray) -> None:
        if self.appearance_center is None or self.appearance_scale is None:
            return
        values = self._lab_chroma_values(rgb_bgr, mask)
        if values.shape[0] < self.config.color_min_area_px:
            return
        center, scale = self._robust_appearance_statistics(values)
        rate = float(np.clip(self.config.appearance_update_rate, 0.0, 1.0))
        self.appearance_center = (1.0 - rate) * self.appearance_center + rate * center
        self.appearance_scale = (1.0 - rate) * self.appearance_scale + rate * scale

    def _appearance_binary_mask(self, rgb_bgr: np.ndarray) -> np.ndarray:
        if self.appearance_center is not None and self.appearance_scale is not None:
            lab = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            chroma = lab[:, :, 1:3]
            normalized = (chroma - self.appearance_center) / self.appearance_scale
            distance = np.sqrt(np.sum(normalized * normalized, axis=2))
            return (distance <= self.config.appearance_distance_threshold).astype(np.uint8) * 255
        if self.config.use_hsv_fallback:
            hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
            lower = np.asarray(self.config.hsv_lower, dtype=np.uint8)
            upper = np.asarray(self.config.hsv_upper, dtype=np.uint8)
            return cv2.inRange(hsv, lower, upper)
        return np.zeros(rgb_bgr.shape[:2], dtype=np.uint8)

    def _robust_appearance_statistics(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center = np.median(values, axis=0).astype(np.float32)
        scale = (1.4826 * np.median(np.abs(values - center), axis=0)).astype(np.float32)
        scale = np.clip(
            scale,
            self.config.appearance_min_chroma_sigma,
            self.config.appearance_max_chroma_sigma,
        )
        return center, scale

    @staticmethod
    def _lab_chroma_values(rgb_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        return lab[:, :, 1:3][mask > 0]

    def _propagate(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        previous_mask: np.ndarray,
        previous_points: np.ndarray | None,
    ) -> tuple[np.ndarray | None, dict[str, float]]:
        points = previous_points
        if self._point_count(points) < self.config.min_features:
            points = self._detect_features(previous_gray, previous_mask)
        initial_count = self._point_count(points)
        empty = {
            "confidence": 0.0,
            "feature_count": 0.0,
            "survival_ratio": 0.0,
            "inlier_ratio": 0.0,
            "fb_error": math.inf,
        }
        if points is None or initial_count < 2:
            return None, empty

        lk_params = {
            "winSize": (self.config.lk_window_px, self.config.lk_window_px),
            "maxLevel": self.config.lk_max_level,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        }
        next_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, points, None, **lk_params
        )
        if next_points is None or forward_status is None:
            return None, empty
        back_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray, previous_gray, next_points, None, **lk_params
        )
        if back_points is None or backward_status is None:
            return None, empty

        fb_errors = np.linalg.norm(points.reshape(-1, 2) - back_points.reshape(-1, 2), axis=1)
        valid = (
            (forward_status.reshape(-1) > 0)
            & (backward_status.reshape(-1) > 0)
            & np.isfinite(fb_errors)
            & (fb_errors <= self.config.max_forward_backward_error_px)
        )
        source = points.reshape(-1, 2)[valid]
        destination = next_points.reshape(-1, 2)[valid]
        valid_count = int(source.shape[0])
        if valid_count < 2:
            return None, empty

        affine, inliers = cv2.estimateAffinePartial2D(
            source,
            destination,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.ransac_reprojection_px,
        )
        if affine is None:
            return None, empty
        propagated = cv2.warpAffine(
            previous_mask.astype(np.uint8),
            affine,
            (current_gray.shape[1], current_gray.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
        survival_ratio = float(valid_count / max(1, initial_count))
        inlier_ratio = float(np.mean(inliers)) if inliers is not None and inliers.size else 0.0
        mean_fb = float(np.mean(fb_errors[valid])) if valid_count else math.inf
        fb_score = max(0.0, 1.0 - mean_fb / max(self.config.max_forward_backward_error_px, 1e-6))
        feature_score = min(1.0, valid_count / max(1, self.config.min_features * 2))
        confidence = 0.30 * survival_ratio + 0.30 * inlier_ratio + 0.20 * fb_score + 0.20 * feature_score
        return propagated, {
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "feature_count": float(valid_count),
            "survival_ratio": survival_ratio,
            "inlier_ratio": inlier_ratio,
            "fb_error": mean_fb,
        }

    def _detect_features(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        radius = max(0, int(self.config.feature_mask_dilation_px))
        feature_mask = mask.astype(np.uint8) * 255
        if radius > 0:
            kernel_size = radius * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            feature_mask = cv2.dilate(feature_mask, kernel)
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.config.max_features,
            qualityLevel=self.config.feature_quality,
            minDistance=self.config.feature_min_distance_px,
            mask=feature_mask,
            blockSize=5,
        )

    def _depth_obstacle_support(self, mask: np.ndarray, depth_m: np.ndarray | None) -> int:
        return int(np.count_nonzero(self.depth_obstacle_mask(mask, depth_m)))

    def depth_obstacle_mask(
        self, mask: np.ndarray, depth: np.ndarray | None
    ) -> np.ndarray:
        """Return closer-depth pixels over or immediately around the expected target."""
        depth_m = depth_to_meters(depth)
        if depth_m is None or self.target_depth_m is None or depth_m.shape[:2] != mask.shape[:2]:
            return np.zeros(mask.shape, dtype=bool)
        margin = max(1, int(self.config.target_near_margin_px))
        kernel_size = margin * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        near_target = cv2.dilate(mask.astype(np.uint8), kernel) > 0
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        # Include the target footprint itself: a direct occluder replaces the cube's
        # depth samples and was previously discarded by the `~mask` exclusion.
        closer = near_target & valid & (depth_m < self.target_depth_m - self.config.depth_margin_m)
        closer = cv2.morphologyEx(closer.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return closer > 0

    @staticmethod
    def _gray(rgb_bgr: np.ndarray) -> np.ndarray:
        if rgb_bgr is None or rgb_bgr.ndim != 3:
            raise ValueError("expected a BGR image")
        return cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _point_count(points: np.ndarray | None) -> int:
        return int(points.reshape(-1, 2).shape[0]) if points is not None else 0
