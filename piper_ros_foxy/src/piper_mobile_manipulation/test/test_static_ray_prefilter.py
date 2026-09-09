"""Early atlas membership and scoring distance never change motion geometry."""
from copy import deepcopy
from types import SimpleNamespace

from piper_mobile_manipulation.planning.prequalification import StaticRayPrefilter


def test_partition_preserves_candidates_and_caches_exact_queries():
    calls = []
    class Atlas:
        metadata = {'tool_floor_clearance_m': 0.005}
        direction_tolerance_deg = 5.0
        def intersects_ray(self, target, direction, minimum, maximum, floor, clearance):
            calls.append((target, direction, minimum, maximum, floor, clearance))
            return SimpleNamespace(supported=direction[0] > 0, checked_keys=3,
                                   supported_intervals_m=((0.4, 0.5), (0.8, 0.9)))
    gate = StaticRayPrefilter(Atlas(), 0.25, 3.0, 0.005)
    candidates = [dict(index=i, ray_id=i, candidate_geometry='target_ray',
        target_object_center=dict(x=0.3, y=0, z=0.1),
        ray_direction=dict(x=sign, y=0, z=0),
        ray_min_standoff_m=0.28, ray_max_standoff_m=3.0,
        desired_camera_position=dict(x=1.8, y=0, z=0.1))
        for i, sign in enumerate((1, -1))]
    before = deepcopy(candidates)
    survivors, rejected = gate.partition(candidates)
    assert survivors[0]['nbv_scoring_standoff_m'] == 0.4
    assert survivors[0]['desired_camera_position'] == candidates[0]['desired_camera_position']
    assert survivors[0]['ray_min_standoff_m'] == 0.28
    assert survivors[0]['ray_max_standoff_m'] == 3.0
    assert set(rejected) == {1}
    assert candidates == before
    gate.partition(candidates)
    assert len(calls) == 2
    candidates[0]['target_object_center']['x'] += 1e-8
    gate.partition(candidates)
    assert len(calls) == 3


def test_nonray_and_disabled_mode_are_unchanged():
    assert StaticRayPrefilter.load('shadow', 'absent', 'absent', .25, 3, 0) is None
    gate = StaticRayPrefilter(None, .25, 3, 0)
    points = [{'index': 2, 'candidate_geometry': 'exact_point'}]
    assert gate.partition(points) == (points, {})


def test_capability_culls_survive_capture_generation_but_not_population_reset():
    from piper_mobile_manipulation.planning.ray_culls import prune_hard_culled_rays
    calls = []
    class Atlas:
        metadata = {'tool_floor_clearance_m': 0.005}
        direction_tolerance_deg = 5.0
        def intersects_ray(self, *args):
            calls.append(args)
            return SimpleNamespace(supported=False, checked_keys=3)
    gate = StaticRayPrefilter(Atlas(), .25, 3, .005)
    population = {'mission_id': 'm', 'session_id': 's', 'sha256': 'qualified'}
    ray = dict(index=1, ray_id=1, candidate_geometry='target_ray',
               target_object_center=dict(x=.3, y=0, z=0),
               ray_direction=dict(x=1, y=0, z=0),
               ray_min_standoff_m=.28, ray_max_standoff_m=3.)
    assert gate.carried_culls(population) == {}
    assert gate.partition([ray])[0] == []
    for _accepted_count in range(2, 6):
        carried = gate.carried_culls(dict(population))
        assert set(carried) == {1}
        survivors = prune_hard_culled_rays([ray], carried, {})
        assert gate.partition(survivors) == ([], {})
        carried.clear()  # Caller cannot mutate the retained evidence.
    assert len(calls) == 1
    assert gate.carried_culls(dict(population, sha256='replacement')) == {}
    assert gate.carried_culls(None) == {}
