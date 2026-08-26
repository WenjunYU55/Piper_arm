"""Pure protocol helpers for mission-scoped permanent ray elimination."""

from copy import deepcopy
import hashlib
import json
import math


PROTOCOL_VERSION = 1
HARD_CULL_SOURCES = ('prequalification', 'tesseract_endpoint')


def _vector(value):
    if isinstance(value, dict):
        values = [value.get(axis) for axis in ('x', 'y', 'z')]
    else:
        values = list(value or [])
    if len(values) != 3:
        raise ValueError('ray population vector must contain XYZ')
    result = [float(item) for item in values]
    if not all(math.isfinite(item) for item in result):
        raise ValueError('ray population vector is non-finite')
    return result


def canonical_ray_population(viewpoints):
    """Return the stable generated-ray fields used by diagnostics and IPC."""
    records = []
    for order, candidate in enumerate(viewpoints, start=1):
        ray_id = int(candidate.get('ray_id', candidate.get('index', -1)))
        if ray_id < 0:
            raise ValueError('ray population contains an invalid ray ID')
        direction = _vector(candidate.get(
            'direction', candidate.get('ray_direction')))
        norm = math.sqrt(sum(value * value for value in direction))
        if norm <= 1e-12:
            raise ValueError('ray population direction is zero length')
        azimuth = math.degrees(math.atan2(direction[1], direction[0]))
        if azimuth < 0.0:
            azimuth += 360.0
        elevation = math.degrees(math.asin(max(
            -1.0, min(1.0, direction[2] / norm))))
        records.append({
            'ray_id': ray_id,
            'generated_order': int(candidate.get('generated_order', order)),
            'direction': direction,
            'representative_position_m': _vector(candidate.get(
                'representative_position_m',
                candidate.get('desired_camera_position'))),
            'azimuth_deg': float(candidate.get('azimuth_deg', azimuth)),
            'elevation_deg': float(candidate.get('elevation_deg', elevation)),
            'minimum_standoff_m': float(candidate.get(
                'minimum_standoff_m', candidate.get('ray_min_standoff_m'))),
            'maximum_standoff_m': float(candidate.get(
                'maximum_standoff_m', candidate.get('ray_max_standoff_m'))),
            'scoring_standoff_m': float(candidate.get(
                'scoring_standoff_m', candidate.get(
                    'ray_scoring_standoff_m'))),
        })
    records.sort(key=lambda item: item['ray_id'])
    return records


def population_sha256(records):
    payload = json.dumps(
        records, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def ray_population_identity(
        viewpoints, mission_id, session_id, target_center, frame_id):
    records = canonical_ray_population(viewpoints)
    return {
        'schema_version': PROTOCOL_VERSION,
        'mission_id': str(mission_id),
        'session_id': str(session_id),
        'sha256': population_sha256(records),
        'ray_count': len(records),
        'target_center_m': _vector(target_center),
        'frame_id': str(frame_id),
    }


def population_key(identity):
    if not isinstance(identity, dict):
        raise ValueError('ray population identity is missing')
    digest = str(identity.get('sha256', ''))
    if len(digest) != 64 or any(value not in '0123456789abcdef'
                                for value in digest):
        raise ValueError('ray population SHA-256 is invalid')
    count = int(identity.get('ray_count', -1))
    if count < 0:
        raise ValueError('ray population count is invalid')
    return (
        str(identity.get('mission_id', '')),
        str(identity.get('session_id', '')),
        digest,
        count,
        str(identity.get('frame_id', '')),
    )


def stable_revision(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def hard_cull_snapshot(
        population, source, source_revision, generation, culls):
    if source not in HARD_CULL_SOURCES:
        raise ValueError('unsupported hard-cull source')
    population_key(population)
    normalized = []
    for item in culls:
        ray_id = int(item['ray_id'])
        if ray_id < 0:
            raise ValueError('hard-cull ray ID is invalid')
        normalized.append({
            'ray_id': ray_id,
            'stage': str(item.get('stage', source)),
            'reason_code': str(item.get('reason_code', '')),
            'reason': str(item.get('reason', '')),
            'evidence': deepcopy(item.get('evidence', {})),
        })
    return {
        'schema_version': PROTOCOL_VERSION,
        'population': deepcopy(population),
        'source': source,
        'source_revision': str(source_revision),
        'generation_observed': int(generation),
        'complete_snapshot': True,
        'culls': sorted(normalized, key=lambda item: item['ray_id']),
    }


class HardCullLedger:
    """Merge monotonic snapshots while rejecting a different ray universe."""

    def __init__(self):
        self._population_key = None
        self._sources = {}

    def reset(self, population=None):
        self._population_key = (
            None if population is None else population_key(population))
        self._sources = {}

    def update(self, payload):
        if not isinstance(payload, dict) or payload.get(
                'schema_version') != PROTOCOL_VERSION:
            raise ValueError('hard-cull snapshot schema is unsupported')
        key = population_key(payload.get('population'))
        source = str(payload.get('source', ''))
        revision = str(payload.get('source_revision', ''))
        if source not in HARD_CULL_SOURCES or not revision:
            raise ValueError('hard-cull snapshot source is invalid')
        if payload.get('complete_snapshot') is not True:
            raise ValueError('hard-cull payload must be a complete snapshot')
        if self._population_key is None:
            self._population_key = key
        if key != self._population_key:
            return False
        previous = self._sources.get(source)
        if previous is None or previous['revision'] != revision:
            entries = {}
        else:
            entries = dict(previous['entries'])
        for item in payload.get('culls', []):
            ray_id = int(item['ray_id'])
            if ray_id < 0 or ray_id >= key[3]:
                raise ValueError('hard-cull ray ID is outside the population')
            entries[ray_id] = deepcopy(item)
        self._sources[source] = {'revision': revision, 'entries': entries}
        return True

    def entries(self, population):
        if population_key(population) != self._population_key:
            return {}
        merged = {}
        for source in HARD_CULL_SOURCES:
            record = self._sources.get(source, {})
            for ray_id, item in record.get('entries', {}).items():
                merged[ray_id] = deepcopy(item)
        return merged


def prune_hard_culled_rays(viewpoints, entries, rejection_reasons=None):
    """Remove only ledger-proven rays before ranking."""
    hard = {int(key): value for key, value in (entries or {}).items()}
    survivors = []
    for item in viewpoints:
        ray_id = int(item.get('ray_id', item.get('index', -1)))
        evidence = hard.get(ray_id)
        if evidence is None:
            survivors.append(item)
            continue
        if rejection_reasons is not None:
            rejection_reasons[ray_id] = [str(evidence.get(
                'reason', 'permanently infeasible ray'))]
    return survivors
