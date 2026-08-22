#!/usr/bin/env python3
"""Create immutable-derived GroundingDINO/SAM2 masks for one scan dataset.

The source capture and manifest are never edited.  Each derived generation is
bound to the immutable manifest, target prompt and exact model assets.  The
original live mask is deliberately not copied into the model workspace, so a
failed fresh GroundingDINO detection cannot silently fall back to that mask.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECONSTRUCTION_ROOT = Path(__file__).resolve().parent
if str(RECONSTRUCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(RECONSTRUCTION_ROOT))
AI_SCRIPT_ROOT = (
    PROJECT_ROOT / 'AI_perception_tests' / 'groundingdino_test')
if str(AI_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SCRIPT_ROOT))

from run_groundingdino_on_capture import (  # noqa: E402
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_REPO_DIR,
    DEFAULT_TEXT_THRESHOLD,
    run_on_capture,
)
from sam2_refine_on_capture import (  # noqa: E402
    DEFAULT_GROUNDED_SAM2_REPO_DIR,
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CONFIG,
    refine_capture,
)
from tsdf_reconstruct import (  # noqa: E402
    canonical_sha256,
    load_metadata,
    manifest_artifact_index,
    metadata_paths_from_manifest,
    resolve_frame_artifacts,
    sha256_file,
    validate_manifest_integrity,
)


PIPELINE_VERSION = 2
MINIMUM_SEMANTIC_MASK_PIXELS = 100


def _atomic_write_json(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    temporary.replace(path)


def _model_identity():
    assets = {
        'groundingdino_config': Path(DEFAULT_CONFIG_PATH).resolve(),
        'groundingdino_checkpoint': Path(DEFAULT_CHECKPOINT_PATH).resolve(),
        'sam2_checkpoint': Path(DEFAULT_SAM2_CHECKPOINT).resolve(),
    }
    missing = [str(path) for path in assets.values() if not path.is_file()]
    if missing:
        raise ValueError(
            'offline segmentation model assets are missing: %s'
            % ', '.join(missing))
    return {
        name: {'path': str(path), 'sha256': sha256_file(path)}
        for name, path in assets.items()
    }


def _cache_identity(manifest, manifest_sha256, models):
    value = {
        'pipeline_version': PIPELINE_VERSION,
        'manifest_sha256': str(manifest_sha256),
        'target_label': str(manifest.get('target_label', '')),
        'target_profile': str(manifest.get('target_profile', '')),
        'target_prompt': str(manifest.get('target_prompt', '')),
        'models': models,
    }
    return value, canonical_sha256(value)


def _valid_cached_index(path, identity):
    try:
        with open(path, 'r', encoding='utf-8') as stream:
            value = json.load(stream)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get('identity') != identity:
        return None
    root = Path(path).resolve().parent
    frames = value.get('frames')
    if not isinstance(frames, list) or not frames:
        return None
    for frame in frames:
        if not isinstance(frame, dict):
            return None
        relative = Path(str(frame.get('mask_path', '')))
        if relative.is_absolute() or '..' in relative.parts:
            return None
        mask_path = (root / relative).resolve()
        try:
            mask_path.relative_to(root)
        except ValueError:
            return None
        if not mask_path.is_file() \
                or sha256_file(mask_path) != str(frame.get('mask_sha256', '')):
            return None
        comparison_relative = Path(str(frame.get('comparison_path', '')))
        comparison_path = (root / comparison_relative).resolve()
        try:
            comparison_path.relative_to(root)
        except ValueError:
            return None
        if comparison_relative.is_absolute() \
                or '..' in comparison_relative.parts \
                or not comparison_path.is_file() \
                or sha256_file(comparison_path) != str(
                    frame.get('comparison_sha256', '')):
            return None
    return value


def _prepare_model_capture(source_dir, rgb_path, depth_path):
    source_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(rgb_path), str(source_dir / 'rgb.png'))
    shutil.copy2(str(depth_path), str(source_dir / 'depth.npy'))
    # Do not create detection_mask.png.  Its absence makes the independent
    # GroundingDINO pass fail closed instead of falling back to the live mask.


def _fresh_target_mask(refinement, rgb_shape):
    if not isinstance(refinement, dict) or refinement.get('status') != 'ok':
        raise ValueError(
            'offline SAM2 refinement failed: %s'
            % str((refinement or {}).get('reason', 'unknown failure')))
    records = [
        record for record in refinement.get('masks', [])
        if isinstance(record, dict) and record.get('mask_role') == 'target'
        and record.get('prompt_source') == 'groundingdino']
    if len(records) != 1:
        raise ValueError(
            'fresh GroundingDINO/SAM2 produced %d authoritative target masks'
            % len(records))
    record = records[0]
    mask = cv2.imread(str(record.get('mask_png', '')), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != tuple(rgb_shape[:2]):
        raise ValueError('fresh target mask does not match the stored RGB')
    score = float(record.get('sam2_score', float('nan')))
    confidence = float(record.get('confidence', float('nan')))
    if not math.isfinite(score) or score <= 0.0 \
            or not math.isfinite(confidence) or confidence <= 0.0:
        raise ValueError('fresh target mask has invalid model confidence')
    box = np.asarray(record.get('box_xyxy_pixels', []), dtype=float)
    if box.shape != (4,) or not np.all(np.isfinite(box)):
        raise ValueError('fresh target mask has no finite GroundingDINO box')
    height, width = mask.shape
    x0 = max(0, min(width, int(math.floor(box[0]))))
    y0 = max(0, min(height, int(math.floor(box[1]))))
    x1 = max(0, min(width, int(math.ceil(box[2]))))
    y1 = max(0, min(height, int(math.ceil(box[3]))))
    if x1 <= x0 or y1 <= y0:
        raise ValueError('fresh GroundingDINO target box is empty')
    unclipped_pixels = int(np.count_nonzero(mask))
    clipped = np.zeros_like(mask)
    clipped[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    pixels = int(np.count_nonzero(clipped))
    if pixels < MINIMUM_SEMANTIC_MASK_PIXELS:
        raise ValueError(
            'fresh target mask has only %d pixels after box validation'
            % pixels)
    return clipped, {
        'groundingdino_confidence': confidence,
        'sam2_score': score,
        'groundingdino_box_xyxy_pixels': box.tolist(),
        'unclipped_mask_pixels': unclipped_pixels,
        'validated_mask_pixels': pixels,
        'pixels_removed_outside_groundingdino_box': (
            unclipped_pixels - pixels),
    }


def resegment_scan(scan_dir, device='cuda', detector=None, refiner=None):
    """Generate or reuse one manifest-bound offline mask generation."""
    scan = Path(scan_dir).expanduser().resolve()
    with open(scan / 'manifest.json', 'r', encoding='utf-8') as stream:
        manifest = json.load(stream)
    manifest_sha256 = validate_manifest_integrity(scan, manifest)
    metadata_paths = metadata_paths_from_manifest(
        scan, manifest, allow_partial_view_set=True)
    models = _model_identity()
    identity, generation = _cache_identity(
        manifest, manifest_sha256, models)
    base = scan / 'reconstruction' / 'offline_resegment'
    root = base / generation
    index_path = root / 'index.json'
    cached = _valid_cached_index(index_path, identity)
    if cached is not None:
        return index_path, cached
    if root.exists():
        raise ValueError(
            'offline segmentation cache generation failed validation: %s'
            % root)

    partial = base / (generation + '.partial')
    if partial.exists():
        shutil.rmtree(partial)
    (partial / 'masks').mkdir(parents=True)
    (partial / 'sources').mkdir()
    output_root = partial / 'model_outputs'
    artifact_index = manifest_artifact_index(scan, manifest)
    detector = detector or run_on_capture
    refiner = refiner or refine_capture
    frames = []
    prompt = str(manifest.get('target_prompt', '')).strip()
    label = str(manifest.get('target_label', '')).strip()
    profile = str(manifest.get('target_profile', '')).strip()
    if not prompt or not label or not profile:
        raise ValueError(
            'offline segmentation requires manifest target prompt, label and profile')

    try:
        for metadata_path in metadata_paths:
            metadata = load_metadata(metadata_path)
            artifacts = resolve_frame_artifacts(
                scan, metadata_path, metadata, manifest)
            stem = metadata_path.name[:-len('_metadata.yaml')]
            depth_relative = 'frames/%s_depth.npy' % stem
            depth_path = artifact_index.get(depth_relative)
            if depth_path is None or not depth_path.is_file():
                raise ValueError(
                    '%s is missing immutable aligned depth NPY' % stem)
            source_dir = partial / 'sources' / stem
            _prepare_model_capture(
                source_dir, artifacts['rgb'], depth_path)
            detection = detector(
                capture_dir=source_dir,
                prompt=prompt,
                output_root=output_root,
                repo_dir=Path(DEFAULT_REPO_DIR),
                config_path=Path(DEFAULT_CONFIG_PATH),
                checkpoint_path=Path(DEFAULT_CHECKPOINT_PATH),
                box_threshold=DEFAULT_BOX_THRESHOLD,
                text_threshold=DEFAULT_TEXT_THRESHOLD,
                device=str(device),
                obstacle_prompt='',
                target_label=label,
                target_profile=profile,
            )
            if detection.get('summary', {}).get('target_source') \
                    != 'groundingdino':
                raise ValueError(
                    '%s has no fresh GroundingDINO target detection' % stem)
            refinement = refiner(
                capture_dir=source_dir,
                groundingdino_boxes=Path(
                    detection['outputs']['boxes_yaml']),
                output_root=output_root,
                repo_dir=Path(DEFAULT_GROUNDED_SAM2_REPO_DIR),
                sam2_config=DEFAULT_SAM2_CONFIG,
                sam2_checkpoint=Path(DEFAULT_SAM2_CHECKPOINT),
                device=str(device),
                max_masks=1,
            )
            rgb = cv2.imread(str(artifacts['rgb']), cv2.IMREAD_COLOR)
            if rgb is None:
                raise ValueError('%s RGB artifact is unreadable' % stem)
            mask, metrics = _fresh_target_mask(refinement, rgb.shape)
            mask_path = partial / 'masks' / ('%s_mask.png' % stem)
            if not cv2.imwrite(str(mask_path), mask):
                raise OSError('failed to write derived mask for %s' % stem)
            original = cv2.imread(
                str(artifacts['mask']), cv2.IMREAD_GRAYSCALE)
            if original is None or original.shape != mask.shape:
                raise ValueError('%s live mask artifact is invalid' % stem)
            original_pixels = int(np.count_nonzero(original))
            intersection_pixels = int(np.count_nonzero(
                (original > 0) & (mask > 0)))
            overlay = rgb.copy()
            original_only = (original > 0) & (mask == 0)
            fresh_support = mask > 0
            overlay[original_only] = (
                0.45 * overlay[original_only]
                + 0.55 * np.asarray([0, 0, 255])
            ).astype(np.uint8)
            overlay[fresh_support] = (
                0.65 * overlay[fresh_support]
                + 0.35 * np.asarray([0, 255, 0])
            ).astype(np.uint8)
            overlay_path = partial / 'masks' / ('%s_comparison.png' % stem)
            if not cv2.imwrite(str(overlay_path), overlay):
                raise OSError(
                    'failed to write derived-mask comparison for %s' % stem)
            frames.append({
                'frame': metadata_path.name,
                'mask_path': mask_path.relative_to(partial).as_posix(),
                'mask_sha256': sha256_file(mask_path),
                'comparison_path': overlay_path.relative_to(partial).as_posix(),
                'comparison_sha256': sha256_file(overlay_path),
                'source_rgb_sha256': sha256_file(artifacts['rgb']),
                'source_live_mask_sha256': sha256_file(artifacts['mask']),
                'source_live_mask_pixels': original_pixels,
                'fresh_live_intersection_pixels': intersection_pixels,
                'fresh_to_live_iou': float(
                    intersection_pixels / max(1, np.count_nonzero(
                        (original > 0) | (mask > 0)))),
                'target_prompt': prompt,
                **metrics,
            })
        shutil.rmtree(partial / 'sources')
        shutil.rmtree(partial / 'model_outputs')
        index = {
            'schema_version': 1,
            'results_type': 'offline_groundingdino_sam2_target_masks',
            'identity': identity,
            'generation_sha256': generation,
            'device': str(device),
            'frame_count': len(frames),
            'frames': frames,
            'source_captures_immutable': True,
            'live_mask_used_as_model_fallback': False,
            'temporary_model_inputs_retained': False,
        }
        _atomic_write_json(partial / 'index.json', index)
        root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(partial), str(root))
        return index_path, index
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scan_dir')
    parser.add_argument(
        '--device', default=os.environ.get(
            'PIPER_OFFLINE_SEGMENT_DEVICE', 'cuda'))
    args = parser.parse_args()
    index_path, index = resegment_scan(args.scan_dir, device=args.device)
    print(json.dumps({
        'offline_resegment_index': str(index_path),
        'frame_count': int(index['frame_count']),
        'generation_sha256': str(index['generation_sha256']),
    }, sort_keys=True))


if __name__ == '__main__':
    main()
