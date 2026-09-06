"""World-upload reuse never caches request freshness or robot state."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from motion_planning.curobo.worker import CuroboBackend


def backend():
    value = object.__new__(CuroboBackend)
    value.fixed_platform_cuboids = []
    value.fixed_world_meshes = []
    value.floor_z_m = 0.005
    value.Cuboid = lambda **kw: kw
    value.Mesh = lambda **kw: kw
    value.WorldConfig = lambda **kw: kw
    value.uploads = []
    value.restores = []
    value.motion_gen = SimpleNamespace(update_world=value.uploads.append)
    value._restore_collision_constraints = lambda: value.restores.append(True)
    value.last_planning_diagnostics = {}
    return value


def request():
    return {'scene': {'obstacles': [{
        'id': 'box', 'minimum_m': [0.1, 0.1, 0.1],
        'maximum_m': [0.2, 0.2, 0.2], 'type': 'box'}]}}


def test_identical_world_skips_upload_but_restores_constraints():
    value = backend()
    first = request()
    value._update_world(first)
    assert value.last_planning_diagnostics['world_geometry_reused'] is False
    second = deepcopy(first)
    second.update(request_id='new', created_at_ns=999,
                  start_state={'positions_rad': [0.2]*6})
    value._update_world(second)
    assert len(value.uploads) == 1
    assert len(value.restores) == 2
    assert value.last_planning_diagnostics['world_geometry_reused'] is True
    assert first == request()


@pytest.mark.parametrize('change', ['move', 'resize', 'remove', 'add', 'floor'])
def test_every_geometry_change_uploads(change):
    value = backend()
    data = request()
    value._update_world(data)
    if change == 'move':
        data['scene']['obstacles'][0]['minimum_m'][0] += 1e-8
        data['scene']['obstacles'][0]['maximum_m'][0] += 1e-8
    elif change == 'resize':
        data['scene']['obstacles'][0]['maximum_m'][0] += 0.01
    elif change == 'remove':
        data['scene']['obstacles'] = []
    elif change == 'add':
        extra = deepcopy(data['scene']['obstacles'][0])
        extra['id'] = 'another'
        data['scene']['obstacles'].append(extra)
    else:
        value.floor_z_m += 0.01
    value._update_world(data)
    assert len(value.uploads) == 2
    assert value.last_planning_diagnostics['world_geometry_reused'] is False


def test_failed_upload_invalidates_previous_world():
    value = backend()
    value._update_world(request())
    def fail(world):
        raise RuntimeError('native upload failed')
    value.motion_gen.update_world = fail
    with pytest.raises(RuntimeError):
        value._update_world({'scene': {'obstacles': []}})
    assert value._loaded_world_identity is None
    value.motion_gen.update_world = value.uploads.append
    value._update_world(request())
    assert len(value.uploads) == 2


def test_bad_geometry_cannot_hit_cache():
    value = backend()
    value._update_world(request())
    bad = request()
    bad['scene']['obstacles'][0]['minimum_m'][0] = float('nan')
    with pytest.raises(ValueError):
        value._update_world(bad)
    assert len(value.uploads) == 1


def test_new_backend_never_inherits_loaded_world():
    first, second = backend(), backend()
    first._update_world(request())
    second._update_world(request())
    assert len(first.uploads) == len(second.uploads) == 1
