"""Static capability prefilter before NBV; never authorizes motion.

Keep motion geometry untouched; supply a separate capability-supported
scoring distance so a distant nominal midpoint cannot alias a small target.
"""

from collections import OrderedDict
from copy import deepcopy
import json

from .capability import load_capability_map


def query_capability_ray(atlas, viewpoint, minimum, maximum, floor_z):
    target = [float(viewpoint['target_object_center'][k]) for k in ('x', 'y', 'z')]
    raw = viewpoint['ray_direction']
    direction = ([float(raw[k]) for k in ('x', 'y', 'z')]
                 if isinstance(raw, dict) else [float(v) for v in raw])
    return atlas.intersects_ray(
        target, direction,
        max(float(viewpoint['ray_min_standoff_m']), float(minimum)),
        min(float(viewpoint['ray_max_standoff_m']), float(maximum)),
        float(floor_z), float(atlas.metadata['tool_floor_clearance_m']))


class StaticRayPrefilter:
    """Bounded exact-input cache for an immutable, source-verified atlas."""

    def __init__(self, atlas, minimum, maximum, floor_z):
        self.atlas = atlas
        self.minimum, self.maximum, self.floor_z = minimum, maximum, floor_z
        self.cache = OrderedDict()
        self.population = None
        self.rejected_rays = {}

    def carried_culls(self, population):
        """Retain static exclusions until the canonical population changes."""
        if population != self.population:
            self.population = deepcopy(population)
            self.rejected_rays = {}
        return deepcopy(self.rejected_rays)

    @classmethod
    def load(cls, mode, path, root, minimum, maximum, floor_z,
             direction_tolerance_deg=None):
        if mode != 'enforce':
            return None
        atlas = load_capability_map(
            path, root, verify_sources=True,
            direction_tolerance_deg=direction_tolerance_deg)
        if not atlas.metadata.get('qualified_for_enforcement', False):
            return None
        return cls(atlas, minimum, maximum, floor_z)

    def partition(self, viewpoints):
        survivors, rejected = [], {}
        for item in viewpoints:
            if item.get('candidate_geometry') != 'target_ray':
                survivors.append(item)
                continue
            # Exact JSON identity, no rounded geometry and no current-state
            # evidence cached. Runtime admission remains downstream.
            key = json.dumps({k: item[k] for k in (
                'target_object_center', 'ray_direction',
                'ray_min_standoff_m', 'ray_max_standoff_m')}, sort_keys=True)
            if key not in self.cache:
                self.cache[key] = query_capability_ray(
                    self.atlas, item, self.minimum, self.maximum, self.floor_z)
                if len(self.cache) > 1024:
                    self.cache.popitem(last=False)
            result = self.cache[key]
            if result.supported:
                # The nearest supported interval maximizes target resolution.
                # This is scoring evidence, never an IK or motion endpoint.
                intervals = result.supported_intervals_m
                if not intervals:
                    raise ValueError('Supported capability ray has no intervals')
                survivors.append(dict(
                    item, nbv_scoring_standoff_m=min(float(a) for a, _b in intervals),
                    nbv_scoring_source='capability_supported_interval'))
            else:
                rejected[int(item['ray_id'] if 'ray_id' in item else item['index'])] = {
                    'stage': 'capability_prequalification',
                    'reason': ('CAPABILITY_MAP_NO_SUPPORT: bounded ray has no '
                               'nearby collision-qualified camera capability cell'),
                    'evidence': {
                        'checked_keys': int(result.checked_keys),
                        'direction_tolerance_deg': self.atlas.direction_tolerance_deg},
                }
        if self.population is not None:
            self.rejected_rays.update(deepcopy(rejected))
        return survivors, rejected
