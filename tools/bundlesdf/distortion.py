"""Offline calibration diagnostics; never estimate depth corrections from a mesh."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def rectification_maps(K, coefficients, width, height):
    """Map a pinhole output pixel to its Brown-Conrady source pixel, with unchanged K."""
    return cv2.initUndistortRectifyMap(
        np.asarray(K, dtype=float), np.asarray(coefficients, dtype=float),
        np.eye(3), np.asarray(K, dtype=float), (width, height), cv2.CV_32FC1)


def plane_residuals(depth_m, K, support, coefficients=()):
    """Measure a user-selected planar patch in one frame, using axial Z in metres.

    This reports residuals rather than flattening data. Multiple object faces or
    foreground/background mixtures invalidate the planar-patch assumption.
    Distortion belongs to this depth grid, not an unrelated colour camera.
    """
    depth = np.asarray(depth_m, dtype=float)
    support = np.asarray(support, dtype=bool)
    if depth.ndim != 2 or support.shape != depth.shape:
        raise ValueError('depth and support must be matching two-dimensional arrays')
    y, x = np.nonzero(support & np.isfinite(depth) & (depth > 0))
    if len(x) < 100:
        raise ValueError('planar patch needs at least 100 valid depth pixels')
    pixels = np.stack((x, y), axis=1).astype(float).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(
        pixels, np.asarray(K, dtype=float),
        np.asarray(coefficients, dtype=float) if len(coefficients) else None,
    ).reshape(-1, 2)
    points = np.column_stack((rays, np.ones(len(rays)))) * depth[y, x, None]
    center = np.mean(points, axis=0)
    _, singular, axes = np.linalg.svd(points - center, full_matrices=False)
    if singular[1] < 1e-8:
        raise ValueError('planar patch is degenerate')
    residual_mm = (points - center) @ axes[-1] * 1000
    return dict(
        pixels=len(points), rms_mm=float(np.sqrt(np.mean(residual_mm ** 2))),
        absolute_p95_mm=float(np.percentile(np.abs(residual_mm), 95)),
        peak_to_peak_mm=float(np.ptp(residual_mm)),
        assumption='One physically planar surface; not a calibration or accuracy certificate',
    )


def diagnose(session, output, frame=0, plane_roi=None):
    """Save raw/corrected views and measured remap displacement without altering input."""
    from dataset import artifact, atomic_json, validate

    _, manifest = validate(session, minimum_frames=1, allow_incomplete=True)
    if frame < 0 or frame >= len(manifest['frames']):
        raise ValueError('frame index outside session')
    root, output = Path(session).resolve(), Path(output).resolve()
    entry = manifest['frames'][frame]
    rgb = cv2.imread(str(artifact(root, entry['rgb'])))
    mask = cv2.imread(str(artifact(root, entry['mask'])), 0)
    coefficients = entry.get('distortion_coefficients', [])
    maps = rectification_maps(
        entry['K'], coefficients or [0.] * 5, entry['width'], entry['height'])
    corrected = cv2.remap(rgb, *maps, cv2.INTER_LINEAR)
    corrected_mask = cv2.remap(mask, *maps, cv2.INTER_NEAREST)
    y, x = np.indices(mask.shape)
    displacement = np.hypot(maps[0] - x, maps[1] - y)
    target = displacement[corrected_mask > 0]
    report = dict(
        frame=frame, input=str(root), distortion_model=entry.get('distortion_model', ''),
        coefficients=coefficients, K=entry['K'],
        correction_applied=bool(np.any(coefficients)),
        maximum_displacement_px=float(displacement.max()),
        target_p95_displacement_px=float(np.percentile(target, 95)) if target.size else None,
        target_maximum_displacement_px=float(target.max()) if target.size else None,
        limitation='Pixel rectification only; does not correct depth bias, motion distortion or pose drift',
    )
    if plane_roi is not None:
        x0, y0, x1, y1 = plane_roi
        if not (0 <= x0 < x1 <= entry['width'] and 0 <= y0 < y1 <= entry['height']):
            raise ValueError('plane ROI must be inside the recorded image')
        support = np.zeros(mask.shape, bool)
        support[y0:y1, x0:x1] = True
        depth = np.load(artifact(root, entry['depth']), allow_pickle=False) * entry['depth_scale_m']
        report['plane_roi_xyxy'] = list(plane_roi)
        report['plane_with_calibration'] = plane_residuals(depth, entry['K'], support, coefficients)
        report['plane_ignoring_distortion'] = plane_residuals(depth, entry['K'], support)
    output.mkdir(parents=True, exist_ok=False)
    for name, pixels in (('raw.png', rgb), ('rectified.png', corrected)):
        if not cv2.imwrite(str(output / name), pixels):
            raise OSError('failed to write diagnostic image')
    panels = []
    for label, pixels in (('Raw recorded RGB', rgb), ('Calibration-rectified RGB', corrected)):
        panel = cv2.copyMakeBorder(pixels, 30, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(panel, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        panels.append(panel)
    if not cv2.imwrite(str(output / 'comparison.png'), np.hstack(panels)):
        raise OSError('failed to write comparison')
    atomic_json(output / 'distortion.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--plane-roi', nargs=4, type=int, metavar=('X0', 'Y0', 'X1', 'Y1'),
                        help='Optional single flat-surface patch in raw image coordinates')
    args = parser.parse_args()
    print(json.dumps(diagnose(args.session, args.output, args.frame, args.plane_roi), indent=2))


if __name__ == '__main__':
    main()
