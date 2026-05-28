def point_in_workspace(point, params):
    return (
        params['workspace_x_min'] <= point.x <= params['workspace_x_max']
        and params['workspace_y_min'] <= point.y <= params['workspace_y_max']
        and params['workspace_z_min'] <= point.z <= params['workspace_z_max']
    )


def safety_reason_for_command(command, stable, visible, depth_valid, transform_valid, params):
    if params.get('require_valid_tf', True) and not transform_valid:
        return False, 'transform unavailable'
    if params.get('require_valid_depth', True) and not depth_valid:
        return False, 'depth invalid'
    if not visible:
        return False, 'target not visible'
    if command == 'grab' and params.get('require_stable_before_grab', True) and not stable:
        return False, 'target not stable for grab'
    return True, 'ok'
