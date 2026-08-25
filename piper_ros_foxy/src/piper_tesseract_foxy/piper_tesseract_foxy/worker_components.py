"""Explicit composition boundaries around the Tesseract backend."""


class CollisionSceneOperations:
    """Own request-local collision scene lifecycle operations."""

    def __init__(self, backend):
        self.backend = backend

    def reset(self):
        return self.backend.reset_scene()

    def add_obstacles(self, obstacles):
        return self.backend.add_obstacles(obstacles)

    def state_in_collision(self, joints):
        return self.backend.state_in_collision(joints)

    def state_meets_required_clearance(self, joints):
        return self.backend.state_meets_required_clearance(joints)

    def find_bootstrap_recovery(self, *args, **kwargs):
        return self.backend.find_bootstrap_recovery(*args, **kwargs)

    def find_terminal_home_recovery(self, *args, **kwargs):
        return self.backend.find_terminal_home_recovery(*args, **kwargs)


class AimSolver:
    """Own endpoint IK and target-aim solving entrypoints."""

    def __init__(self, backend):
        self.backend = backend

    def ray_ik_solutions(self, *args, **kwargs):
        return self.backend.ray_ik_solutions(*args, **kwargs)

    def joint_goals(self, *args, **kwargs):
        return self.backend.ik_joint_goals(*args, **kwargs)

    def plan_candidate_aims(self, *args, **kwargs):
        from piper_tesseract_foxy.worker import TesseractBackend

        return TesseractBackend.plan_candidate_aims(
            self.backend, *args, **kwargs)


class TrajectoryPlanner:
    """Own trajectory planning, timing, and final validation entrypoints."""

    def __init__(self, backend):
        self.backend = backend

    def plan_segment(self, *args, **kwargs):
        return self.backend.plan_segment(*args, **kwargs)

    def time_parameterize(self, instructions):
        return self.backend.time_parameterize(instructions)

    def final_validate(self, points):
        return self.backend.final_validate(points)

    def configured_home_direct_policy(self, request):
        return self.backend.configured_home_direct_policy(request)

    def plan_configured_home_direct(self, *args, **kwargs):
        return self.backend.plan_configured_home_direct(*args, **kwargs)

    def plan_dual_recovery_return_home(self, *args, **kwargs):
        return self.backend.plan_dual_recovery_return_home(*args, **kwargs)

    def plan_segment_to_joint_goal(self, *args, **kwargs):
        return self.backend.plan_segment_to_joint_goal(*args, **kwargs)


class WorkerOrchestrator:
    """Coordinate one validated request through the composed backend."""

    def __init__(self, backend):
        self.backend = backend
        self.collision_scene = CollisionSceneOperations(backend)
        self.aim_solver = AimSolver(backend)
        self.trajectory_planner = TrajectoryPlanner(backend)

    @property
    def last_planning_diagnostics(self):
        """Expose the backend's established diagnostic record."""
        return getattr(self.backend, 'last_planning_diagnostics', {})

    def _ensure_planning_time(self, context):
        from piper_tesseract_foxy.worker import TesseractBackend

        return TesseractBackend.ensure_planning_time(self.backend, context)

    def plan(self, request):
        """Coordinate one request through explicit planning components."""
        from piper_tesseract_foxy import worker as worker_module

        backend = self.backend
        planning_budget, segment_budget = (
            worker_module.planning_budgets_for_request(request))
        backend.planning_budget_sec = planning_budget
        backend.segment_planning_budget_sec = segment_budget
        # Scene cleanup/model loading is request setup, not an IK/OMPL
        # candidate attempt. The bridge's unchanged response timeout bounds
        # the complete transaction; start the tighter worker budget only once
        # the request-local collision scene is ready.
        backend.planning_deadline_monotonic = None
        backend.last_planning_diagnostics = {
            'shortlisted_candidates': len(
                request.get('scene', {}).get('candidate_views', [])),
            'shortlisted_rays': int(request.get(
                'planning', {}).get('shortlisted_ray_count', 0)),
            'expanded_ray_candidates': int(request.get(
                'planning', {}).get('expanded_ray_candidate_count', 0)),
            'candidate_attempts': 0,
            'exact_aim_attempts': 0,
            'fallback_aim_attempts': 0,
            'ray_ik_solver_attempts': 0,
            'failure_stage_counts': {},
            'candidate_failures': [],
            'attempted_ray_ids': [],
            'ray_failure_stage_counts': {},
        }
        self.collision_scene.reset()
        self.collision_scene.add_obstacles(
            request['scene'].get('obstacles', []))
        planning_deadline = worker_module.time.monotonic() + planning_budget
        backend.planning_deadline_monotonic = planning_deadline
        backend.execution_speed_percent = float(
            request['planning']['effective_speed_percent'])
        backend.command_rate_hz = float(
            request['planning']['command_rate_hz'])
        backend.execution_position_limits = request['limits']['position_rad']
        backend.execution_velocity_limits = request[
            'limits']['max_velocity_rad_s']
        backend.execution_acceleration_limits = request[
            'limits']['max_acceleration_rad_s2']
        backend.bootstrap_start_limit_tolerance_rad = float(
            request['limits'].get(
                'bootstrap_start_limit_tolerance_rad', 0.0))
        minimum_views = int(request['planning']['min_viewpoints'])
        maximum_views = int(request['planning']['max_viewpoints'])
        maximum_step = float(
            request['planning']['max_execution_joint_step_rad'])
        position_limits = request['limits']['position_rad']
        joint_margin = float(request['limits'].get('joint_margin_rad', 0.0))
        rolls = [
            float(value)
            for value in request['planning']['roll_samples_rad']]
        current = request['start_state']['positions_rad']
        if request.get('plan_kind') == 'RETURN_HOME' and (
                self.trajectory_planner.configured_home_direct_policy(
                    request) is not None):
            self._ensure_planning_time(
                'before configured direct return-home target')
            home = worker_module.finite_six(
                request['planning']['return_home_positions_rad'],
                'return home positions')
            points, validation = (
                self.trajectory_planner.plan_configured_home_direct(
                    request, current, home))
            return [], [{
                'from_viewpoint': -1,
                'to_viewpoint': -2,
                'is_return_home': True,
                'startup_home_static': bool(
                    request['scene'].get('startup_home_static', False)),
                'points': points,
                **validation,
            }]
        powered_start_recovery = (
            self.collision_scene.find_bootstrap_recovery(
                request, 'powered_start_home_recovery')
            if request.get('plan_kind') == 'RETURN_HOME' else None)
        bootstrap_recovery = (
            powered_start_recovery
            if request.get('plan_kind') == 'RETURN_HOME'
            else self.collision_scene.find_bootstrap_recovery(request))
        if bootstrap_recovery is not None:
            current = bootstrap_recovery[
                'bootstrap_recovery_end_positions_rad']
        selected = []
        segments = []
        failures = {}
        pending = list(request['scene']['candidate_views'])
        visibility_target = (
            request['scene'].get('target_center_m')
            if request.get('plan_kind') == 'MULTIVIEW_SCAN' else None)
        if request.get('plan_kind') == 'RETURN_HOME':
            self._ensure_planning_time(
                'before dedicated return-home qualification')
            home = worker_module.finite_six(
                request['planning']['return_home_positions_rad'],
                'return home positions')
            terminal_recovery = (
                self.collision_scene.find_terminal_home_recovery(
                    request, home))
            if (
                    powered_start_recovery is not None
                    and terminal_recovery is not None):
                points, validation = (
                    self.trajectory_planner.plan_dual_recovery_return_home(
                        powered_start_recovery, terminal_recovery,
                        maximum_step))
            elif terminal_recovery is None:
                points, validation = (
                    self.trajectory_planner.plan_segment_to_joint_goal(
                        current, home, maximum_step, powered_start_recovery,
                        'powered_start_home_recovery'))
            else:
                recovery_endpoint = worker_module.finite_six(
                    terminal_recovery[
                        'bootstrap_recovery_end_positions_rad'],
                    'terminal home recovery endpoint')
                points, validation = (
                    self.trajectory_planner.plan_segment_to_joint_goal(
                        recovery_endpoint, current, maximum_step,
                        terminal_recovery))
                points = worker_module.reverse_sdk_movej_points(points)
            segments.append({
                'from_viewpoint': -1,
                'to_viewpoint': -2,
                'is_return_home': True,
                'startup_home_static': bool(
                    request['scene'].get('startup_home_static', False)),
                'points': points,
                **validation,
            })
            return selected, segments
        if request.get('plan_kind') == 'ROUGH_ACQUISITION':
            selected_center_id = None
            centered = [
                candidate for candidate in pending
                if candidate.get('required_first') is True
            ]
            # Direct backend qualification helpers predate the transport
            # marker and already place the centered view first. Real worker
            # requests pass validate_request(), which requires the marker.
            if not centered and pending:
                centered = pending[:1]
            for candidate in centered:
                self._ensure_planning_time(
                    'before centered rough-coordinate candidate')
                try:
                    backend.last_planning_diagnostics['candidate_attempts'] += 1
                    accepted = self.aim_solver.plan_candidate_aims(
                        current, candidate, rolls, maximum_step,
                        position_limits, joint_margin, bootstrap_recovery)
                except (worker_module.ContractError, RuntimeError, ValueError) as error:
                    failures[int(candidate['id'])] = str(error)
                    continue
                (selected_candidate, roll, points, validation,
                 aim_fallback_used, aim_offset_deg) = accepted
                selected.append({
                    'id': int(candidate['id']),
                    'camera_position_m': selected_candidate[
                        'camera_position_m'],
                    'look_direction': selected_candidate['look_direction'],
                    'nominal_look_direction': candidate['look_direction'],
                    'aim_fallback_used': bool(aim_fallback_used),
                    'aim_offset_deg': float(aim_offset_deg),
                    'aim_attempt_diagnostics': selected_candidate.get(
                        'aim_attempt_diagnostics', {}),
                    'roll_rad': roll,
                    **{
                        key: selected_candidate[key]
                        for key in (
                            'view_selection_policy',
                            'view_selection_requested_policy',
                            'view_selection_generation',
                            'view_selection_session_id',
                            'nbv_rank',
                            'nbv_positive_information_gain',
                            'nbv_predicted_unknown_pixels',
                            'nbv_novel_surface_pixels',
                            'nbv_marginal_information_pixels',
                            'nbv_marginal_information_fraction',
                            'coverage_score',
                        )
                        if key in selected_candidate
                    },
                })
                segments.append({
                    'from_viewpoint': -1,
                    'to_viewpoint': int(candidate['id']),
                    'points': points,
                    **validation,
                })
                current = points[-1]['positions_rad']
                bootstrap_recovery = None
                failures.pop(int(candidate['id']), None)
                selected_center_id = int(candidate['id'])
                break
            if not selected:
                raise worker_module.ContractError(
                    'no centered rough-coordinate first view is reachable (%s)'
                    % '; '.join(
                        'view %s: %s' % item
                        for item in sorted(failures.items())))
            # ``required_first`` makes a centered candidate eligible for the
            # mandatory first look; it does not make every other centered
            # compact pose unusable later in the bounded search.
            pending = [
                candidate for candidate in pending
                if int(candidate['id']) != selected_center_id
            ]
        while pending and len(selected) < maximum_views:
            next_pending = []
            progress = False
            for candidate_index, candidate in enumerate(pending):
                self._ensure_planning_time(
                    'before starting a viewpoint candidate')
                ray_id = candidate.get('ray_id')
                if ray_id is not None:
                    attempted_rays = backend.last_planning_diagnostics[
                        'attempted_ray_ids']
                    if int(ray_id) not in attempted_rays:
                        attempted_rays.append(int(ray_id))
                try:
                    backend.last_planning_diagnostics['candidate_attempts'] += 1
                    accepted = self.aim_solver.plan_candidate_aims(
                        current, candidate, rolls, maximum_step,
                        position_limits, joint_margin, bootstrap_recovery,
                        visibility_target)
                except (
                        worker_module.ContractError,
                        RuntimeError,
                        ValueError) as error:
                    failures[int(candidate['id'])] = str(error)
                    diagnostics = backend.last_planning_diagnostics
                    aim_failures = [dict(item) for item in getattr(
                        error, 'evidence', ())]
                    permanent_endpoint_failure = bool(
                        aim_failures
                        and all(
                            str(item.get('stage', ''))
                            in worker_module.PERMANENT_ENDPOINT_FAILURE_STAGES
                            for item in aim_failures))
                    if ray_id is not None:
                        ray_counts = diagnostics[
                            'ray_failure_stage_counts'].setdefault(
                                str(int(ray_id)), {})
                        stage = str(getattr(
                            error, 'stage', 'PLANNING_FAILURE'))
                        ray_counts[stage] = int(
                            ray_counts.get(stage, 0)) + 1
                    if len(diagnostics['candidate_failures']) < len(
                            request['scene']['candidate_views']):
                        diagnostics['candidate_failures'].append({
                            'id': int(candidate['id']),
                            'nbv_rank': int(candidate.get('nbv_rank', 0)),
                            'camera_position_m': list(
                                candidate['camera_position_m']),
                            'stage': str(getattr(
                                error, 'stage', 'PLANNING_FAILURE')),
                            'aim_failures': aim_failures,
                            'permanent_endpoint_failure': bool(
                                permanent_endpoint_failure),
                            'detail': str(error),
                        })
                    next_pending.append(candidate)
                    continue
                (selected_candidate, roll, points, validation,
                 aim_fallback_used, aim_offset_deg) = accepted
                selected.append({
                    'id': int(candidate['id']),
                    'camera_position_m': selected_candidate[
                        'camera_position_m'],
                    'look_direction': selected_candidate['look_direction'],
                    'nominal_look_direction': candidate['look_direction'],
                    'aim_fallback_used': bool(aim_fallback_used),
                    'aim_offset_deg': float(aim_offset_deg),
                    'aim_attempt_diagnostics': selected_candidate.get(
                        'aim_attempt_diagnostics', {}),
                    'roll_rad': roll,
                    **{
                        key: selected_candidate[key]
                        for key in (
                            'view_selection_policy',
                            'view_selection_requested_policy',
                            'view_selection_generation',
                            'view_selection_session_id',
                            'nbv_rank',
                            'nbv_positive_information_gain',
                            'nbv_predicted_unknown_pixels',
                            'nbv_novel_surface_pixels',
                            'nbv_marginal_information_pixels',
                            'nbv_marginal_information_fraction',
                            'nbv_projected_object_pixels',
                            'nbv_direction_novelty_deg',
                            'nbv_camera_travel_m',
                            'coverage_score',
                            'ray_id',
                            'ray_standoff_m',
                            'ray_probe_index',
                            'ray_probe_phase',
                        )
                        if key in selected_candidate
                    },
                })
                backend.last_planning_diagnostics['selected_candidate'] = {
                    'id': int(candidate['id']),
                    'nbv_rank': int(candidate.get('nbv_rank', 0)),
                    'camera_position_m': list(
                        selected_candidate['camera_position_m']),
                    'aim_fallback_used': bool(aim_fallback_used),
                    'aim_offset_deg': float(aim_offset_deg),
                    **{
                        key: selected_candidate[key]
                        for key in (
                            'ray_id', 'ray_standoff_m', 'ray_probe_index',
                            'ray_probe_phase')
                        if key in selected_candidate
                    },
                }
                segments.append({
                    'from_viewpoint': (
                        -1 if len(selected) == 1
                        else int(selected[-2]['id'])),
                    'to_viewpoint': int(candidate['id']),
                    'points': points,
                    **validation,
                })
                current = points[-1]['positions_rad']
                bootstrap_recovery = None
                failures.pop(int(candidate['id']), None)
                pending = (
                    next_pending + pending[candidate_index + 1:])
                progress = True
                break
            if not progress:
                break
        if len(selected) < minimum_views:
            raise worker_module.CandidateExhausted(
                'only %d viewpoints planned; require at least %d of %d (%s)' % (
                    len(selected), minimum_views, maximum_views,
                    '; '.join(
                        'view %s: %s' % item
                        for item in sorted(failures.items()))
                    if failures else 'no candidates'))
        if (
                request.get('plan_kind') == 'MULTIVIEW_SCAN'
                and bool(request.get('planning', {}).get(
                    'include_return_home', True))):
            self._ensure_planning_time('before return-home qualification')
            home = worker_module.finite_six(
                request['planning']['return_home_positions_rad'],
                'return home positions')
            terminal_recovery = (
                self.collision_scene.find_terminal_home_recovery(
                    request, home))
            if terminal_recovery is None:
                points, validation = (
                    self.trajectory_planner.plan_segment_to_joint_goal(
                        current, home, maximum_step))
            else:
                # The folded home is a qualified bounded start corridor, not a
                # normal-clearance OMPL goal.  Plan and validate home->current,
                # then execute its exact rest-to-rest reverse current->home.
                recovery_endpoint = worker_module.finite_six(
                    terminal_recovery[
                        'bootstrap_recovery_end_positions_rad'],
                    'terminal home recovery endpoint')
                points, validation = (
                    self.trajectory_planner.plan_segment_to_joint_goal(
                        recovery_endpoint, current, maximum_step,
                        terminal_recovery))
                points = worker_module.reverse_sdk_movej_points(points)
            segments.append({
                'from_viewpoint': int(selected[-1]['id']),
                'to_viewpoint': -2,
                'is_return_home': True,
                'points': points,
                **validation,
            })
        return selected, segments
