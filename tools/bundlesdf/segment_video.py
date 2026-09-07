#!/usr/bin/env python3
"""Propagate an accepted target mask over every saved RGB-D frame in isolated SAM2."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import cv2
import numpy as np

from dataset import artifact, atomic_json, intrinsics_valid, distortion_valid
from piper_mobile_manipulation.reconstruction_recording.core import RecorderConfig, mask_metrics
from piper_mobile_manipulation.visibility import angle_degrees, inside_cone, validate_transform

ROOT = Path(__file__).resolve().parents[2]


def validate_extraction(data):
    """Fail before GPU loading if the prepared sensor sequence lost its timestamp/calibration contract."""
    if data.get('dataset_type') != 'piper_unsegmented_rgbd_video' or data.get('state') != 'ready':
        raise ValueError('Input extraction is not complete and usable')
    config = RecorderConfig(**data['configuration'])
    previous, calibration = 0, None
    if not data['frames'] or data['frames'][0].get('accepted_view') != 1:
        raise ValueError('Missing first accepted capture')
    for i, entry in enumerate(data['frames']):
        stamp = entry['rgb_stamp_ns']
        if entry['index'] != i or not isinstance(stamp, int) or stamp <= previous:
            raise ValueError('Nonmonotonic extracted image timestamps or indices')
        if abs(stamp-entry['depth_stamp_ns']) > round(config.synchronization_slop_sec*1e9):
            raise ValueError('Extracted RGB-D synchronization limit exceeded')
        if entry['tf_stamp_ns'] != stamp:
            raise ValueError('Extracted TF is not at image time')
        if not intrinsics_valid(entry['K'], entry['width'], entry['height']) or not distortion_valid(
                entry.get('distortion_model', ''), entry.get('distortion_coefficients', [])):
            raise ValueError('Invalid extracted camera calibration')
        identity = [entry[k] for k in (
            'K', 'width', 'height', 'base_frame', 'camera_optical_frame',
            'distortion_model', 'distortion_coefficients')]
        if calibration is not None and calibration != identity:
            raise ValueError('Extracted camera calibration changed')
        calibration, previous = identity, stamp
    return config


def qualify(entry, rgb, depth, mask, target, config, reference_area):
    """Keep geometric/quality gates; mask is inferred at this exact image, with no rate gate."""
    if entry.get('tf_error') or entry['T_base_camera'] is None:
        raise ValueError('missing_tf')
    transform = validate_transform(entry['T_base_camera'])
    angle = angle_degrees(transform[:3, 2], np.asarray(target) - transform[:3, 3])
    if not inside_cone(angle, config.angle_limit):
        raise ValueError('outside_angle_tolerance')
    if mask.shape != depth.shape or rgb.shape[:2] != mask.shape:
        raise ValueError('resolution_mismatch')
    area = int(np.count_nonzero(mask))
    if area < config.min_mask_area_px:
        raise ValueError('empty_or_tiny_mask')
    box, margin, blur = mask_metrics(mask, cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    if margin <= config.edge_margin_px:
        raise ValueError('target_clipped')
    selected = depth[mask > 0]
    if not np.isfinite(selected).all():
        raise ValueError('nonfinite_depth')
    valid = selected > 0
    ratio = float(np.count_nonzero(valid)) / area
    if ratio < config.minimum_valid_depth_ratio:
        raise ValueError('insufficient_depth')
    K = np.asarray(entry['K'])
    normalized_area = area * float(np.median(selected[valid]) * entry['depth_scale_m'])**2 / (K[0, 0]*K[1, 1])
    if reference_area is not None and normalized_area > reference_area * config.max_mask_area_growth_ratio:
        raise ValueError('mask_expansion')
    if config.blur_gate_enabled and blur < config.minimum_blur_score:
        raise ValueError('excessive_blur')
    return dict(angular_offset_deg=angle, target_bbox_xyxy=box, edge_margin_px=margin,
                mask_area_px=area, blur_laplacian_variance=blur, valid_target_depth_ratio=ratio,
                normalized_mask_area_m2=normalized_area)


class TemporalSAM2:
    """Use the existing streaming SAM2 model, preserving seed conditioning in bounded memory."""

    def __init__(self):
        import torch
        sys.path.insert(0, str(ROOT / 'AI_perception_tests'))
        from sam2_live_worker import Sam2LiveWorker, DEFAULT_CONFIG, DEFAULT_CHECKPOINT, SAM2_REPO

        if not torch.cuda.is_available():
            raise RuntimeError('SAM2 CUDA unavailable; use the isolated Grounded-SAM2 Python environment')
        self.torch = torch
        self.worker = Sam2LiveWorker.__new__(Sam2LiveWorker)
        self.worker.device, self.worker.config = 'cuda', DEFAULT_CONFIG
        self.worker.checkpoint = DEFAULT_CHECKPOINT
        self.worker.build()
        self.predictor = self.worker.predictor
        if not all(hasattr(self.predictor, name) for name in ('add_new_frame', 'infer_single_frame')):
            raise RuntimeError('SAM2 lacks the streaming API; use the existing Grounded-SAM-2 checkout')
        self.state = self.predictor.init_state(video_path=None, offload_state_to_cpu=False)
        self.maximum_retained_frames = 0
        self.provenance = dict(
            model='SAM2.1 tiny', config=DEFAULT_CONFIG, checkpoint=str(DEFAULT_CHECKPOINT),
            checkpoint_sha256=hashlib.sha256(DEFAULT_CHECKPOINT.read_bytes()).hexdigest(),
            repository=str(SAM2_REPO),
            repository_commit=subprocess.check_output(
                ['git', '-C', str(SAM2_REPO), 'rev-parse', 'HEAD'], text=True).strip(),
            predictor_sha256=hashlib.sha256(
                (SAM2_REPO / 'sam2/sam2_video_predictor.py').read_bytes()).hexdigest(),
            memory_compactor_sha256=hashlib.sha256(
                (ROOT / 'AI_perception_tests/sam2_live_worker.py').read_bytes()).hexdigest(),
            temporal_memory_limit=16, device='cuda', torch_version=torch.__version__,
        )

    def infer(self, rgb_bgr, seed=None):
        with self.torch.inference_mode(), self.torch.autocast(device_type='cuda', dtype=self.torch.bfloat16):
            i = self.predictor.add_new_frame(self.state, cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB))
            self.state['video_height'], self.state['video_width'] = rgb_bgr.shape[:2]
            if seed is not None:
                _, ids, logits = self.predictor.add_new_mask(self.state, frame_idx=i, obj_id=1, mask=seed > 0)
            else:
                _, ids, logits = self.predictor.infer_single_frame(self.state, i)
            if list(ids) != [1]:
                raise RuntimeError('SAM2 lost target object identity')
            mask = (logits[0, 0] > 0).cpu().numpy().astype(np.uint8)*255
        self.worker.compact_predictor_history(self.state, 16)
        self.maximum_retained_frames = max(self.maximum_retained_frames, self.state['num_frames'])
        return seed.copy() if seed is not None else mask


def prepare(extracted, output, predictor=None, limit_frames=None):
    extracted, output = Path(extracted).resolve(), Path(output).resolve()
    if output.exists() or output.with_name(output.name+'.incomplete').exists():
        raise FileExistsError('Refusing existing segmentation output')
    data = json.loads((extracted / 'extraction.json').read_text())
    config = validate_extraction(data)
    entries = data['frames']
    if limit_frames is not None:
        if limit_frames < 1:
            raise ValueError('limit_frames must be positive')
        entries = entries[:limit_frames]
    predictor = predictor or TemporalSAM2()
    temporary = output.with_name('.'+output.name+'.'+uuid.uuid4().hex+'.inprogress')
    temporary.mkdir(parents=True)
    records, rejects = [], Counter()
    reference_area = None
    started = time.monotonic()
    manifest = dict(
        schema_version=1, dataset_type='piper_raw_rgbd_reconstruction',
        session_id=data['session_id'], target_id=data['session_id'],
        coverage_target_center=data['coverage_target_center'],
        pose_convention='T_base_camera maps optical (+X right, +Y down, +Z forward) to base; metres',
        configuration=data['configuration'], complete=False, state='incomplete', accepted=0, frames=records,
        mission_capture_count=data['mission_capture_count'], source_bag=data['source_bag'],
        acquisition_policy='offline_dense_video; every synchronized image, no MOVING or rate gate',
        segmentation=predictor.provenance, rejected_by_reason={}, dropped=0,
    )

    def checkpoint(done=False):
        manifest.update(accepted=len(records), rejected_by_reason=dict(rejects),
                        rejected=sum(rejects.values()),
                        preparation_complete=done, processed_frames=processed,
                        preparation_duration_sec=time.monotonic()-started,
                        maximum_retained_sam2_frames=predictor.maximum_retained_frames)
        atomic_json(temporary / 'manifest.json', manifest)

    processed = 0
    try:
        checkpoint()
        for entry in entries:
            rgb_path, depth_path = artifact(extracted, entry['rgb']), artifact(extracted, entry['depth'])
            rgb = cv2.imread(str(rgb_path))
            depth = np.load(str(depth_path), allow_pickle=False)
            seed = cv2.imread(str(artifact(extracted, data['seed_mask'])), 0) if processed == 0 else None
            if rgb is None or (processed == 0 and (seed is None or not seed.any())):
                raise ValueError('Corrupt RGB or initial seed')
            # Every frame updates temporal memory even if its reconstruction depth/TF is rejected.
            mask = predictor.infer(rgb, seed)
            processed += 1
            try:
                metrics = qualify(
                    entry, rgb, depth, mask, data['coverage_target_center'], config, reference_area)
            except ValueError as error:
                rejects[str(error)] += 1
                if processed == 1:
                    raise ValueError('First accepted seed fails offline gate: '+str(error)) from error
                continue
            name = '%08d' % len(records)
            folder = temporary / name
            folder.mkdir()
            # Same-filesystem hard links keep native RGB/depth byte-for-byte; no source writes.
            for source, filename in ((rgb_path, 'rgb.png'), (depth_path, 'depth.npy')):
                try:
                    os.link(str(source), str(folder / filename))
                except OSError:
                    import shutil
                    shutil.copyfile(str(source), str(folder / filename))
            if not cv2.imwrite(str(folder / 'mask.png'), mask):
                raise OSError('Mask PNG encoding failed')
            record = dict(entry, **metrics)
            record.update(index=len(records), extracted_index=entry['index'], rgb=name+'/rgb.png',
                          depth=name+'/depth.npy', mask=name+'/mask.png', target_id=data['session_id'],
                          mask_source='sam2_offline_video', mask_stamp_ns=entry['rgb_stamp_ns'],
                          source='first_capture' if not records else 'recorded_video',
                          synchronization_delta_sec=abs(entry['rgb_stamp_ns']-entry['depth_stamp_ns'])*1e-9)
            atomic_json(folder / 'frame.json', record)
            records.append(record)
            if reference_area is None:
                reference_area = metrics['normalized_mask_area_m2']
            if processed % 50 == 0:
                checkpoint()
                print('SAM2 %d/%d frames, %d accepted, %d rejected' % (
                    processed, len(entries), len(records), sum(rejects.values())), flush=True)
        if len(records) < 5:
            raise ValueError('Fewer than five usable frames after segmentation')
        manifest['complete'] = bool(data['complete'] and processed == len(data['frames']))
        manifest['state'] = 'complete' if manifest['complete'] else 'incomplete'
        checkpoint(done=True)
        os.rename(str(temporary), str(output))
        print(json.dumps({k: v for k, v in manifest.items() if k != 'frames'}, indent=2), flush=True)
        return manifest
    except BaseException as error:
        manifest.update(error=str(error), complete=False, state='incomplete')
        checkpoint()
        os.rename(str(temporary), str(output.with_name(output.name+'.incomplete')))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('extracted', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--limit-frames', type=int,
                        help='Diagnostic prefix only; never marked a complete scan')
    args = parser.parse_args()

    def stop(_signal, _frame):
        raise KeyboardInterrupt('Offline SAM2 cancelled')

    signal.signal(signal.SIGTERM, stop)
    prepare(args.extracted, args.output, limit_frames=args.limit_frames)
