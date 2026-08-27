"""Immutable reconstruction input admission and provenance."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np


MINIMUM_CAPTURE_VIEWS = 1
MAXIMUM_CAPTURE_VIEWS = 24
MASK_SOURCES = ('captured', 'offline_resegment')


def canonical_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def camera_extrinsic_from_metadata(metadata):
    """Convert stored T_base_camera into Open3D's T_camera_base extrinsic."""
    transform = metadata.get('camera_transform', {})
    matrix = np.asarray(transform.get('matrix_4x4'), dtype=float)
    if not transform.get('available') or matrix.shape != (4, 4) \
            or not np.all(np.isfinite(matrix)):
        raise ValueError('frame has no finite timestamped camera transform')
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError('camera transform is not homogeneous')
    return np.linalg.inv(matrix)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_metadata(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError('PyYAML is required for scan metadata') from exc
    with open(path, 'r', encoding='utf-8') as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError('frame metadata is not an object')
    return value


def capture_set_provenance(manifest, allow_partial_view_set=False):
    """Validate count while leaving reconstruction quality to geometry."""
    count = int(manifest.get('capture_count', 0))
    if not MINIMUM_CAPTURE_VIEWS <= count <= MAXIMUM_CAPTURE_VIEWS:
        raise ValueError(
            'TSDF reconstruction requires %d-%d captured views'
            % (MINIMUM_CAPTURE_VIEWS, MAXIMUM_CAPTURE_VIEWS))
    return {
        'classification': 'VIEW_COUNT_ELIGIBLE',
        'capture_count': count,
        'ordinary_feature_minimum': MINIMUM_CAPTURE_VIEWS,
        'partial_view_set_explicitly_allowed': bool(allow_partial_view_set),
        'reason': (
            'view count is eligible; reconstruction quality and completeness '
            'are determined by the existing geometric evidence gates'),
    }


def validate_capture_set(
        manifest, metadata_paths, allow_partial_view_set=False):
    """Validate one bounded feature-driven capture set before integration."""
    capture_set = capture_set_provenance(
        manifest, allow_partial_view_set=allow_partial_view_set)
    count = int(capture_set['capture_count'])
    paths = list(metadata_paths)
    if len(paths) != count:
        raise ValueError(
            'manifest capture_count %d does not match %d frame metadata files'
            % (count, len(paths)))
    return paths


def validate_manifest_integrity(scan, manifest):
    """Verify the canonical manifest digest and every immutable artifact."""
    if not isinstance(manifest, dict):
        raise ValueError('manifest is not an object')
    expected = str(manifest.get('manifest_sha256', ''))
    unsigned = dict(manifest)
    unsigned.pop('manifest_sha256', None)
    actual = canonical_sha256(unsigned)
    if expected != actual:
        raise ValueError('manifest SHA-256 does not match its canonical payload')
    root = Path(scan).resolve()
    files = manifest.get('files')
    if not isinstance(files, list) or not files:
        raise ValueError('manifest contains no immutable artifacts')
    seen = set()
    for index, record in enumerate(files):
        if not isinstance(record, dict):
            raise ValueError('manifest file %d is invalid' % index)
        relative = Path(str(record.get('path', '')))
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('manifest file escapes the dataset root')
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError('manifest file escapes the dataset root') from exc
        key = relative.as_posix()
        if not key or key in seen:
            raise ValueError('manifest contains a duplicate or empty path')
        seen.add(key)
        if not candidate.is_file():
            raise ValueError('manifest artifact is missing: %s' % candidate)
        if int(record.get('bytes', -1)) != candidate.stat().st_size:
            raise ValueError('manifest artifact size changed: %s' % candidate)
        if str(record.get('sha256', '')) != sha256_file(candidate):
            raise ValueError('manifest artifact hash changed: %s' % candidate)
    return expected


def manifest_artifact_index(scan, manifest):
    root = Path(scan).resolve()
    index = {}
    for record in manifest.get('files', []):
        relative = Path(str(record.get('path', '')))
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError('manifest file escapes the dataset root') from exc
        index[relative.as_posix()] = candidate
    return index


def prepare_offline_mask_context(scan, manifest_sha256):
    """Run/reuse the isolated offline segmenter and validate its index."""
    scan = Path(scan).resolve()
    project_root = Path(__file__).resolve().parent.parent
    python = (
        project_root / 'AI_perception_tests' / 'groundingdino_test'
        / 'envs' / 'grounded_sam2_py310' / 'bin' / 'python')
    script = project_root / 'reconstruction' / 'offline_resegment.py'
    if not python.is_file() or not script.is_file():
        raise ValueError(
            'offline segmentation environment is unavailable; run the '
            'Grounded-SAM2 environment setup first')
    ai_root = project_root / 'AI_perception_tests' / 'groundingdino_test'
    environment = dict(os.environ)
    environment.update({
        'HF_HOME': str(ai_root / 'hf_cache'),
        'TRANSFORMERS_CACHE': str(ai_root / 'hf_cache' / 'transformers'),
        'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1',
        'MPLCONFIGDIR': str(Path('/tmp') / 'piper_offline_resegment_mpl'),
    })
    result = subprocess.run(
        [str(python), str(script), str(scan)],
        cwd=str(project_root), env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, check=False)
    if result.returncode != 0:
        raise ValueError(
            'offline GroundingDINO/SAM2 preprocessing failed: %s'
            % (result.stdout or 'no diagnostic output').strip()[-3000:])
    summary = None
    for line in reversed((result.stdout or '').splitlines()):
        try:
            candidate = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(candidate, dict) \
                and candidate.get('offline_resegment_index'):
            summary = candidate
            break
    if summary is None:
        raise ValueError(
            'offline segmentation completed without an index result')
    index_path = Path(str(summary['offline_resegment_index'])).resolve()
    allowed_root = (scan / 'reconstruction' / 'offline_resegment').resolve()
    try:
        index_path.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(
            'offline segmentation index escapes the scan dataset') from exc
    with open(index_path, 'r', encoding='utf-8') as stream:
        index = json.load(stream)
    if not isinstance(index, dict) \
            or str(index.get('identity', {}).get('manifest_sha256', '')) \
            != str(manifest_sha256) \
            or index.get('source_captures_immutable') is not True \
            or index.get('live_mask_used_as_model_fallback') is not False:
        raise ValueError(
            'offline segmentation index does not match the immutable scan')
    frames = index.get('frames')
    if not isinstance(frames, list) or not frames:
        raise ValueError('offline segmentation index contains no frame masks')
    by_frame = {}
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError('offline segmentation frame record is invalid')
        name = str(frame.get('frame', ''))
        if not name or name in by_frame:
            raise ValueError(
                'offline segmentation frame identity is missing or duplicated')
        by_frame[name] = frame
    return {
        'root': index_path.parent,
        'index_path': index_path,
        'index': index,
        'by_frame': by_frame,
    }


def load_offline_target_mask(
        context, metadata_path, source_rgb, expected_shape, cv2):
    """Load one hash-bound derived mask without trusting an arbitrary path."""
    record = context['by_frame'].get(metadata_path.name)
    if record is None:
        raise ValueError(
            '%s has no offline target mask' % metadata_path.name)
    root = Path(context['root']).resolve()
    relative = Path(str(record.get('mask_path', '')))
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('offline target mask path is not dataset relative')
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError('offline target mask escapes its generation') from exc
    if not path.is_file() \
            or sha256_file(path) != str(record.get('mask_sha256', '')):
        raise ValueError('offline target mask hash is invalid')
    if sha256_file(source_rgb) != str(record.get('source_rgb_sha256', '')):
        raise ValueError('offline target mask belongs to a different RGB input')
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != tuple(expected_shape):
        raise ValueError('offline target mask dimensions are invalid')
    unique = set(np.unique(mask).astype(int).tolist())
    if not unique.issubset({0, 255}) or not np.any(mask):
        raise ValueError('offline target mask is empty or non-binary')
    return mask, record


def metadata_paths_from_manifest(
        scan, manifest, allow_partial_view_set=False):
    artifacts = manifest_artifact_index(scan, manifest)
    paths = [
        path for relative, path in artifacts.items()
        if relative.startswith('frames/view_')
        and relative.endswith('_metadata.yaml')
    ]
    return validate_capture_set(
        manifest, sorted(paths),
        allow_partial_view_set=allow_partial_view_set)


def resolve_frame_artifacts(scan, metadata_path, metadata, manifest):
    """
    Resolve a frame only through manifest-listed dataset-relative paths.

    Historical metadata contains absolute paths.  Those strings are treated as
    provenance only; their basename must agree with the immutable frame name.
    """
    root = Path(scan).resolve()
    artifacts = manifest_artifact_index(root, manifest)
    suffix = '_metadata.yaml'
    if not metadata_path.name.endswith(suffix):
        raise ValueError('frame metadata filename is invalid')
    stem = metadata_path.name[:-len(suffix)]
    requested = {
        'rgb': ('rgb_file_path', 'frames/%s_rgb.png' % stem),
        'depth': ('depth_png_file_path', 'frames/%s_depth.png' % stem),
        'mask': ('mask_file_path', 'frames/%s_mask.png' % stem),
    }
    if int(metadata.get('capture_schema_version', 1)) >= 2:
        requested.update({
            'native_depth_npy': (
                'native_depth_file_path',
                'frames/%s_native_depth.npy' % stem),
            'native_depth': (
                'native_depth_png_file_path',
                'frames/%s_native_depth.png' % stem),
            'confidence': (
                'confidence_file_path',
                'frames/%s_confidence.png' % stem),
            'target_depth': (
                'target_depth_png_file_path',
                'frames/%s_target_depth.png' % stem),
            'target_support_mask': (
                'target_support_mask_file_path',
                'frames/%s_target_support_mask.png' % stem),
        })
    resolved = {}
    for name, (metadata_key, relative) in requested.items():
        recorded = Path(str(metadata.get(metadata_key, '')))
        if recorded.name != Path(relative).name:
            raise ValueError(
                '%s does not identify the manifest-listed %s artifact'
                % (metadata_path.name, name))
        if relative not in artifacts:
            raise ValueError(
                '%s is not listed in the immutable manifest' % relative)
        candidate = artifacts[relative]
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError('frame artifact escapes the dataset root') from exc
        resolved[name] = candidate
    return resolved


def confidence_capture_provenance(metadata, manifest, artifacts, cv2):
    """Validate schema-v2 confidence-qualified geometry or identify history."""
    schema = int(metadata.get('capture_schema_version', 1))
    if schema < 2:
        return {
            'mode': 'historical_aligned_depth_mask',
            'confidence_qualified': False,
        }
    required = {
        'native_depth_npy', 'native_depth', 'confidence', 'target_depth',
        'target_support_mask'}
    if not required.issubset(artifacts):
        raise ValueError('schema-v2 capture is missing confidence artifacts')
    synchronization = metadata.get('synchronization')
    quality = metadata.get('confidence_quality')
    policy = manifest.get('confidence_policy')
    if not isinstance(synchronization, dict) or not isinstance(quality, dict) \
            or not isinstance(policy, dict):
        raise ValueError('schema-v2 confidence provenance is incomplete')
    if synchronization.get('mask_rgb_exact') is not True:
        raise ValueError('schema-v2 mask is not exactly RGB-correlated')
    try:
        threshold = int(quality['confidence_threshold'])
        policy_threshold = int(policy['minimum_grade'])
        rgb_depth_delta = float(synchronization[
            'rgb_native_depth_delta_sec'])
        depth_confidence_delta = float(synchronization[
            'native_depth_confidence_delta_sec'])
        rgb_depth_limit = float(synchronization[
            'maximum_rgb_native_depth_delta_sec'])
        depth_confidence_limit = float(synchronization[
            'maximum_native_depth_confidence_delta_sec'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('schema-v2 confidence provenance is malformed') from exc
    if threshold != policy_threshold or threshold < 8 or threshold > 15:
        raise ValueError('schema-v2 confidence threshold is not qualified')
    if not all(np.isfinite(value) and value >= 0.0 for value in (
            rgb_depth_delta, depth_confidence_delta, rgb_depth_limit,
            depth_confidence_limit)):
        raise ValueError('schema-v2 synchronization bounds are invalid')
    if rgb_depth_delta > rgb_depth_limit + 1e-9 \
            or depth_confidence_delta > depth_confidence_limit + 1e-9:
        raise ValueError('schema-v2 capture exceeds synchronization bounds')
    confidence = cv2.imread(
        str(artifacts['confidence']), cv2.IMREAD_UNCHANGED)
    if confidence is None or confidence.ndim != 2 \
            or not np.issubdtype(confidence.dtype, np.integer) \
            or np.any(confidence < 0) or np.any(confidence > 15):
        raise ValueError('schema-v2 confidence artifact is invalid')
    try:
        native_array = np.load(
            artifacts['native_depth_npy'], allow_pickle=False)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError('schema-v2 native depth array is invalid') from exc
    native_png = cv2.imread(
        str(artifacts['native_depth']), cv2.IMREAD_UNCHANGED)
    target_depth = cv2.imread(
        str(artifacts['target_depth']), cv2.IMREAD_UNCHANGED)
    target_support = cv2.imread(
        str(artifacts['target_support_mask']), cv2.IMREAD_UNCHANGED)
    if native_array.ndim != 2 or native_array.dtype != np.uint16 \
            or native_png is None or native_png.dtype != np.uint16 \
            or native_png.shape != native_array.shape \
            or not np.array_equal(native_png, native_array):
        raise ValueError('schema-v2 native depth artifacts are inconsistent')
    if confidence.shape != native_array.shape:
        raise ValueError('schema-v2 confidence dimensions do not match native depth')
    if target_depth is None or target_depth.ndim != 2 \
            or target_depth.dtype != np.uint16 \
            or target_support is None or target_support.ndim != 2 \
            or target_support.shape != target_depth.shape:
        raise ValueError('schema-v2 target depth artifacts are invalid')
    unique_support = set(np.unique(target_support).astype(int).tolist())
    if not unique_support.issubset({0, 255}) \
            or not np.array_equal(target_support > 0, target_depth > 0):
        raise ValueError('schema-v2 target depth/support artifacts disagree')
    try:
        projected_points = int(quality['projected_output_points'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('schema-v2 projected point count is malformed') from exc
    if projected_points != int(np.count_nonzero(target_support)) \
            or projected_points < 20:
        raise ValueError('schema-v2 projected target point count is inconsistent')
    synchronized_target = metadata.get('synchronized_target_3d')
    if metadata.get('target_valid') is not True \
            or not isinstance(synchronized_target, dict) \
            or synchronized_target.get('valid') is not True:
        raise ValueError('schema-v2 synchronized target geometry is not valid')
    return {
        'mode': 'l515_confidence_qualified_native_projection',
        'confidence_qualified': True,
        'minimum_confidence_grade': threshold,
        'rgb_native_depth_delta_sec': rgb_depth_delta,
        'native_depth_confidence_delta_sec': depth_confidence_delta,
        'confidence_quality': quality,
    }


def calibration_provenance(manifest, metadata_values, allow_missing=False):
    identifiers = [str(manifest.get('calibration_sha256', '')).strip().lower()]
    identifiers.extend(
        str(value.get('calibration_sha256', '')).strip().lower()
        for value in metadata_values)
    present = {value for value in identifiers if value}
    if any(value and len(value) != 64 for value in identifiers):
        raise ValueError('calibration SHA-256 is malformed')
    if len(present) > 1:
        raise ValueError('capture frames do not share one calibration identity')
    if not present:
        if not allow_missing:
            raise ValueError(
                'capture has no calibration identity; use '
                '--allow-missing-calibration-id for diagnostic-only replay')
        return {
            'classification': 'DIAGNOSTIC_ONLY',
            'calibration_sha256': '',
            'reason': 'capture predates calibration identity binding',
        }
    identity = next(iter(present))
    if any(value != identity for value in identifiers):
        raise ValueError('calibration identity is missing from part of the capture set')
    return {
        'classification': 'CERTIFIED',
        'calibration_sha256': identity,
        'reason': 'manifest and every frame bind one calibration identity',
    }


def capture_schema_provenance(
        manifest, metadata_values, provenance, allow_historical=False):
    """Bind certification to one complete confidence-qualified capture schema."""
    try:
        schemas = [int(value.get('capture_schema_version', 1))
                   for value in metadata_values]
    except (TypeError, ValueError) as exc:
        raise ValueError('capture schema version is malformed') from exc
    if not schemas:
        raise ValueError('capture has no frame metadata')
    if len(set(schemas)) != 1:
        raise ValueError('capture frames mix incompatible schema versions')
    schema = schemas[0]
    result = dict(provenance)
    if schema < 2:
        if not allow_historical:
            raise ValueError(
                'capture predates confidence-qualified schema; use '
                '--allow-missing-calibration-id for diagnostic-only replay')
        prior_reason = str(result.get('reason', '')).rstrip('; ')
        result.update({
            'classification': 'DIAGNOSTIC_ONLY',
            'capture_schema_version': schema,
            'confidence_qualified': False,
            'reason': (
                prior_reason +
                ('; ' if prior_reason else '') +
                'capture predates confidence-qualified depth filtering'),
        })
        return result
    try:
        manifest_schema = int(manifest.get('capture_schema_version', 0))
    except (TypeError, ValueError) as exc:
        raise ValueError('manifest capture schema version is malformed') from exc
    if schema != 2 or manifest_schema != schema:
        raise ValueError('manifest and frames do not bind capture schema 2')
    if not isinstance(manifest.get('confidence_policy'), dict):
        raise ValueError('manifest has no confidence-qualified capture policy')
    result.update({
        'capture_schema_version': schema,
        'confidence_qualified': True,
    })
    return result
