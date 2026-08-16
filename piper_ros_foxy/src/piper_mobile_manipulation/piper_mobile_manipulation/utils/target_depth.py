"""
Shared target-depth layer selection for live tracking and saved captures.

Segmentation masks are semantic support, not proof that every enclosed depth
sample belongs to the target. This module enumerates distinct depth modes and
selects one compact image component without trusting any single seed pixel.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthComponent:
    """Describe one ranked, coherent target-depth component."""

    mask: np.ndarray
    points: int
    depth_m: float
    depth_mad_m: float
    center_distance: float
    support_fraction: float
    score: float


def _histogram_peaks(values, bin_width_m, minimum_separation_m):
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    width = max(float(bin_width_m), 1e-4)
    if maximum - minimum < width:
        return [float(np.median(values))]
    edges = np.arange(minimum, maximum + 2.0 * width, width)
    counts, edges = np.histogram(values, bins=edges)
    smooth = np.convolve(
        counts.astype(float), np.asarray([1, 2, 3, 2, 1], dtype=float) / 9.0,
        mode='same')
    candidates = []
    for index, count in enumerate(smooth):
        left = smooth[index - 1] if index else -1.0
        right = smooth[index + 1] if index + 1 < smooth.size else -1.0
        if count > 0.0 and count >= left and count >= right:
            candidates.append((float(count), float(
                0.5 * (edges[index] + edges[index + 1]))))
    selected = []
    separation = max(float(minimum_separation_m), width)
    for _, depth in sorted(candidates, reverse=True):
        if all(abs(depth - existing) >= separation for existing in selected):
            selected.append(depth)
    return selected or [float(np.median(values))]


def _semantic_center(support, semantic_u, semantic_v, center_u, center_v):
    rows, columns = np.nonzero(support)
    u = columns.astype(float) if semantic_u is None else np.asarray(
        semantic_u, dtype=float)[support]
    v = rows.astype(float) if semantic_v is None else np.asarray(
        semantic_v, dtype=float)[support]
    wanted_u = float(np.median(u)) if center_u is None else float(center_u)
    wanted_v = float(np.median(v)) if center_v is None else float(center_v)
    span_u = max(float(np.max(u) - np.min(u)), 1.0)
    span_v = max(float(np.max(v) - np.min(v)), 1.0)
    normalizer = max(math.hypot(span_u, span_v) * 0.5, 1.0)
    return wanted_u, wanted_v, normalizer


def select_target_depth_component(
        support, depth_m, *, semantic_u=None, semantic_v=None,
        center_u=None, center_v=None, depth_bin_width_m=0.005,
        depth_band_m=0.030, minimum_peak_separation_m=0.025,
        minimum_points=20, minimum_support_fraction=0.0,
        ambiguity_margin=0.08, preferred_depth_m=None):
    """
    Select a coherent target layer and return its mask plus diagnostics.

    Candidate layers are histogram modes split into ordinary image-connected
    components. Ranking combines support, semantic centrality, foreground
    preference, compact depth and (when supplied) continuity with the previous
    accepted target depth. A close tie is rejected instead of silently
    jumping between surfaces.
    """
    valid = np.asarray(support, dtype=bool)
    depth = np.asarray(depth_m, dtype=float)
    if valid.ndim != 2 or valid.shape != depth.shape:
        raise ValueError('target support and depth must be matching 2D arrays')
    if semantic_u is not None and np.asarray(semantic_u).shape != valid.shape:
        raise ValueError('semantic_u shape does not match target support')
    if semantic_v is not None and np.asarray(semantic_v).shape != valid.shape:
        raise ValueError('semantic_v shape does not match target support')
    valid &= np.isfinite(depth) & (depth > 0.0)
    total = int(np.count_nonzero(valid))
    if total < int(minimum_points):
        raise ValueError('insufficient target depth support')

    wanted_u, wanted_v, center_scale = _semantic_center(
        valid, semantic_u, semantic_v, center_u, center_v)
    peaks = _histogram_peaks(
        depth[valid], depth_bin_width_m, minimum_peak_separation_m)
    raw = []
    band = max(float(depth_band_m), float(depth_bin_width_m))
    for peak in peaks:
        candidate = valid & (np.abs(depth - peak) <= band)
        labels_count, labels = cv2.connectedComponents(
            candidate.astype(np.uint8), connectivity=8)
        for label in range(1, labels_count):
            component = labels == label
            count = int(np.count_nonzero(component))
            if count < int(minimum_points):
                continue
            component_depth = depth[component]
            median = float(np.median(component_depth))
            mad = float(np.median(np.abs(component_depth - median)))
            rows, columns = np.nonzero(component)
            u = columns.astype(float) if semantic_u is None else np.asarray(
                semantic_u, dtype=float)[component]
            v = rows.astype(float) if semantic_v is None else np.asarray(
                semantic_v, dtype=float)[component]
            center_distance = math.hypot(
                float(np.median(u)) - wanted_u,
                float(np.median(v)) - wanted_v) / center_scale
            raw.append({
                'mask': component,
                'points': count,
                'depth_m': median,
                'depth_mad_m': mad,
                'center_distance': center_distance,
            })
    if not raw:
        raise ValueError('no coherent target depth component')

    # Overlapping mode bands can enumerate the same physical layer. Keep the
    # strongest representative before calculating the ambiguity margin.
    deduplicated = []
    for item in sorted(raw, key=lambda value: value['points'], reverse=True):
        duplicate = False
        for existing in deduplicated:
            intersection = int(np.count_nonzero(
                item['mask'] & existing['mask']))
            union = int(np.count_nonzero(item['mask'] | existing['mask']))
            if union and float(intersection) / float(union) >= 0.60:
                duplicate = True
                break
        if not duplicate:
            deduplicated.append(item)

    largest = max(item['points'] for item in deduplicated)
    near = min(item['depth_m'] for item in deduplicated)
    far = max(item['depth_m'] for item in deduplicated)
    depth_span = max(far - near, float(depth_bin_width_m))
    use_prior = preferred_depth_m is not None \
        and math.isfinite(float(preferred_depth_m)) \
        and float(preferred_depth_m) > 0.0
    ranked = []
    for item in deduplicated:
        support_score = float(item['points']) / float(largest)
        center_score = 1.0 - min(item['center_distance'], 1.0)
        foreground_score = 1.0 - (
            float(item['depth_m']) - near) / depth_span
        compact_score = 1.0 - min(item['depth_mad_m'] / 0.020, 1.0)
        base_score = (
            0.35 * support_score + 0.30 * center_score
            + 0.20 * foreground_score + 0.15 * compact_score)
        if use_prior:
            prior_score = 1.0 - min(
                abs(float(item['depth_m']) - float(preferred_depth_m)) / 0.10,
                1.0)
            # Continuity breaks otherwise close ties, but cannot preserve a
            # previously selected background layer against stronger current
            # foreground evidence.
            score = 0.85 * base_score + 0.15 * prior_score
        else:
            score = base_score
        ranked.append(DepthComponent(
            mask=item['mask'], points=item['points'],
            depth_m=item['depth_m'], depth_mad_m=item['depth_mad_m'],
            center_distance=item['center_distance'],
            support_fraction=float(item['points']) / float(total),
            score=score))
    ranked.sort(key=lambda item: item.score, reverse=True)
    winner = ranked[0]
    if winner.support_fraction < float(minimum_support_fraction):
        raise ValueError('selected target layer has insufficient support')
    margin = winner.score - ranked[1].score if len(ranked) > 1 else 1.0
    if len(ranked) > 1 and margin < float(ambiguity_margin):
        raise ValueError('ambiguous target depth layers')
    report = {
        'selected_depth_m': winner.depth_m,
        'selected_depth_mad_m': winner.depth_mad_m,
        'selected_points': winner.points,
        'selected_support_fraction': winner.support_fraction,
        'selected_score': winner.score,
        'score_margin': margin,
        'candidate_count': len(ranked),
        'candidate_depths_m': [item.depth_m for item in ranked],
        'candidate_scores': [item.score for item in ranked],
        'preferred_depth_m': (
            float(preferred_depth_m) if use_prior else None),
    }
    return winner.mask.copy(), report
