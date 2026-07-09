from types import SimpleNamespace

from piper_mobile_manipulation.supervised_workflow import choose_removal_plan, cloud_model


def p(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def obstacle(object_id=2, label='pen', center=(0.4, 0.1, 0.05), size=(0.05, 0.01, 0.01)):
    return SimpleNamespace(
        object_id=object_id, semantic_label=label, valid=True, validity_reason='ok',
        base_centroid=p(*center),
        base_bounds_min=p(*(center[i] - size[i] / 2 for i in range(3))),
        base_bounds_max=p(*(center[i] + size[i] / 2 for i in range(3))),
    )


def config():
    return dict(
        movable_whitelist=['pen'], target_clearance_m=0.04,
        drop_target_clearance_m=0.12, drop_obstacle_clearance_m=0.08,
        drop_search_radius_m=0.18, max_grasp_width_m=0.07,
        approach_height_m=0.10, pre_push_offset_m=0.08, push_distance_m=0.06,
        workspace_x_min=0.10, workspace_x_max=0.70,
        workspace_y_min=-0.40, workspace_y_max=0.40,
        workspace_z_min=0.02, workspace_z_max=0.60,
    )


def test_prefers_pick_for_graspable_pen():
    item = obstacle()
    plan = choose_removal_plan(item, (0.35, 0.0, 0.05), [item], config())
    assert plan['valid']
    assert plan['action'] == 'pick_and_place'
    assert plan['execute'] is False


def test_unknown_label_is_rejected():
    item = obstacle(label='knife')
    plan = choose_removal_plan(item, (0.35, 0.0, 0.05), [item], config())
    assert not plan['valid']


def test_large_pen_uses_outward_push():
    item = obstacle(center=(0.45, 0.0, 0.05), size=(0.09, 0.02, 0.02))
    plan = choose_removal_plan(item, (0.35, 0.0, 0.05), [item], config())
    assert plan['valid']
    assert plan['action'] == 'push'
    assert plan['push_end'][0] > plan['object_center'][0]


def test_obstacle_inside_target_clearance_is_rejected():
    item = obstacle(center=(0.36, 0.0, 0.05))
    plan = choose_removal_plan(item, (0.35, 0.0, 0.05), [item], config())
    assert not plan['valid']


def test_obstacle_outside_workspace_is_rejected():
    item = obstacle(center=(0.755, 0.16, 0.26))
    plan = choose_removal_plan(item, (0.55, 0.0, 0.05), [item], config())
    assert not plan['valid']
    assert plan['reason'] == 'obstacle center is outside configured workspace'


def test_cloud_model_rejects_outlier_for_center():
    points = [(0.3, 0.0, 0.1)] * 50 + [(0.31, 0.01, 0.11)] * 50 + [(10, 10, 10)]
    model = cloud_model(points, 'base_link', 5)
    assert model['valid']
    assert model['center'][0] < 0.32
    assert model['accepted_views'] == 5
