"""Declared file-only campaign designs; no runtime configuration writes."""

from copy import deepcopy
import math


MODE_LABELS = {
    'planner_comparison': 'Planner comparison — existing paired grid',
    'camera_profiles': 'Camera resolution / FPS — fixed position',
    'aim_tolerance': 'Camera aim tolerance — distance sweep',
    'objects': 'Object comparison — cube, cup, stick, flower',
}
MODE_CAMPAIGN_IDS = {
    'planner_comparison': 'piper-paired-20cm-20260908',
    'camera_profiles': 'piper-camera-profiles-20260908',
    'aim_tolerance': 'piper-aim-tolerance-20260908',
    'objects': 'piper-objects-20260908',
}
DISTANCES_M = (0.30, 0.50, 0.70, 0.90, 1.10)
TOLERANCES_DEG = (1.0, 5.0, 10.0, 15.0, 20.0, 25.0)


def _profile(value):
    if (not isinstance(value, list) or len(value) != 3
            or any(type(item) is not int or item <= 0 for item in value)):
        raise ValueError('camera profile must be [width, height, fps] positive integers')
    return value


def mode_trials(mode, settings):
    """Regenerate and validate a frozen schema-2 schedule on resume."""
    if mode not in MODE_LABELS or mode == 'planner_comparison':
        raise ValueError('unknown sweep campaign mode')
    backend = settings['backend']
    if backend != 'curobo':
        raise ValueError('parameter and object sweeps use cuRobo only')
    camera = _profile(settings['camera_profile'])
    aim = float(settings['aim_tolerance_deg'])
    z = float(settings['height_m'])
    if not math.isfinite(z) or aim != 5.0:
        raise ValueError('invalid campaign height or aim tolerance')
    target_label = str(settings['cube_label']).strip()
    if not target_label:
        raise ValueError('a cube target label is required')
    conditions = []
    if mode == 'camera_profiles':
        # Missing distance identifies an existing 0.30 m frozen campaign.
        distance = float(settings.get('camera_distance_m', 0.30))
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError('camera campaign distance must be positive and finite')
        profiles = settings['camera_profiles']
        if not profiles or len(profiles) != len({tuple(_profile(p)) for p in profiles}):
            raise ValueError('camera comparison needs distinct profiles')
        for profile in profiles:
            conditions.append((distance, z, profile, aim, 'cube', target_label))
    elif mode == 'aim_tolerance':
        for distance in DISTANCES_M:
            for tolerance in TOLERANCES_DEG:
                conditions.append((distance, z, camera, tolerance, 'cube', target_label))
    else:
        # Missing distance preserves existing 0.30 m frozen object campaigns.
        distance = float(settings.get('object_distance_m', 0.30))
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError('object campaign distance must be positive and finite')
        for obj in ('cube', 'cup', 'stick', 'flower'):
            conditions.append((distance, 0.0, camera, aim, obj,
                               target_label if obj == 'cube' else obj))
    trials = []
    for sequence, (x, height, profile, tolerance, obj, label) in enumerate(conditions, 1):
        trials.append({
            'sequence': sequence, 'pair_index': sequence,
            'trial_id': '%03d_%s_%s' % (sequence, mode, backend),
            'backend': backend, 'x_m': x, 'y_m': 0.0, 'z_m': height,
            'campaign_mode': mode, 'camera_profile': deepcopy(profile),
            'aim_tolerance_deg': tolerance, 'object_key': obj,
            'target_label': label,
            'reference_dimensions_mm': [35.0] * 3 if obj == 'cube' else None,
            'reference_basis': 'nominal cube, not independent metrology' if obj == 'cube' else 'unknown',
        })
    return trials


def mode_definition(campaign_id, mode, settings, created_utc):
    settings = deepcopy(settings)
    if mode == 'camera_profiles':
        settings.setdefault('camera_distance_m', 0.70)
    elif mode == 'objects':
        settings.setdefault('object_distance_m', 0.70)
        settings['height_m'] = 0.0
        settings['height_basis'] = 'declared base_link Z 0.00 m'
    trials = mode_trials(mode, settings)
    return {
        'schema_version': 2, 'campaign_id': campaign_id, 'created_utc': created_utc,
        'evidence_class': 'PHYSICAL_PARAMETER_SWEEP',
        'experimental_unit': 'one physical mission',
        'campaign_mode': mode, 'settings': deepcopy(settings), 'trials': trials,
        'design': {
            'planned_missions': len(trials), 'backend_order': 'single_backend',
            'description': MODE_LABELS[mode],
            'limitation': ('One attempt per condition; no per-condition variance. '
                           'Order and time may confound comparisons.'),
            'coordinate_frame': 'base_link',
            'height_basis': settings.get('height_basis', 'base_link Z; table offset not inferred'),
        },
    }


def applied_settings(project_root):
    """Read saved launch inputs, distinct from measured stream properties."""
    from pathlib import Path
    import yaml
    root = Path(project_root)
    result = {'camera_profile': None, 'aim_tolerance_deg': None}
    try:
        values = (root / 'L515_camera/rgb_profile.conf').read_text().strip().split(',')
        result['camera_profile'] = _profile([int(value) for value in values])
    except (OSError, ValueError):
        pass
    try:
        path = root / 'piper_ros_foxy/src/piper_mobile_manipulation/config/scan_execution_params.yaml'
        document = yaml.safe_load(path.read_text())
        matches = []

        def visit(value):
            if isinstance(value, dict):
                if 'final_capture_aim_tolerance_deg' in value:
                    matches.append(float(value['final_capture_aim_tolerance_deg']))
                for child in value.values():
                    visit(child)
        visit(document)
        if len(matches) == 1 and 1.0 <= matches[0] <= 90.0:
            result['aim_tolerance_deg'] = matches[0]
    except (OSError, ValueError, TypeError, yaml.YAMLError):
        pass
    return result


def condition_mismatches(trial, target_label, applied):
    if 'campaign_mode' not in trial:
        return []
    errors = []
    if str(target_label).strip() != trial['target_label']:
        errors.append('target label differs from scheduled object')
    for key in ('camera_profile', 'aim_tolerance_deg'):
        if applied.get(key) != trial[key]:
            errors.append('saved %s differs from scheduled condition or is unavailable' % key)
    return errors
