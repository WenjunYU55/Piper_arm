"""
Mission-scoped, command-free diagnostics for target-ray selection.

The live planner, prequalification filter, and Tesseract bridge each own a
different part of the candidate lifecycle.  This module only records their
already-made decisions.  It never selects a candidate and is deliberately
free of ROS and plotting dependencies so a reporting failure cannot affect
motion behaviour.
"""

from copy import deepcopy
import fcntl
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

import numpy as np
import yaml

from piper_mobile_manipulation.ray_hard_culls import (
    canonical_ray_population,
    ray_universe_sha256,
)


SCHEMA_VERSION = 2
ARTIFACT_BASENAME = 'ray_mission_diagnostics'
_SAFE_SESSION = re.compile(r'[^A-Za-z0-9_.-]+')

# Compact kinematic representation of piper_description.xacro.  The modified
# DH values are the controller mode-0 chain and are regression-tested against
# that URDF in piper_description/test/test_robot_description.py.  Keeping only
# the kinematic chain makes every HTML report self-contained without copying
# roughly 35 MB of STL meshes into every mission artifact.
ROBOT_MODEL = {
    'name': 'PiPER with installed L515',
    'source': 'piper_description/urdf/piper_description.xacro',
    # HTML retains a compact drawing for compatibility.  The primary Ray
    # Review process parses and renders every checked-in URDF visual mesh.
    'rendering': 'schematic_urdf_kinematic_link_model',
    'joint_names': ['joint%d' % index for index in range(1, 7)],
    'modified_dh': {
        'a_m': [0.0, 0.0, 0.28503, -0.02198, 0.0, 0.0],
        'alpha_rad': [
            0.0, -math.pi / 2.0, 0.0, math.pi / 2.0,
            -math.pi / 2.0, math.pi / 2.0,
        ],
        'theta_offset_rad': [
            0.0, -math.radians(174.22), -math.radians(100.78),
            0.0, 0.0, 0.0,
        ],
        'd_m': [0.123, 0.0, 0.0, 0.25075, 0.0, 0.091],
    },
    'camera_model': 'Intel RealSense L515',
}


def _stable_event_id(session_id, generation, stage, correlation_id='',
                     revision=0):
    identity = '%s\0%d\0%s\0%s\0%d' % (
        session_id, int(generation), stage, correlation_id, int(revision))
    return 'evt-' + hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]


def _append_event(snapshot, stage, *, correlation_id='', ray_deltas=None,
                  selected_ray_ids=None, attempted_ray_ids=None,
                  newly_culled_ray_ids=None, reasons=None, metrics=None,
                  message='', extra=None, revision=None, identity_suffix=''):
    """Append one immutable, deterministically identified journal event."""
    result = deepcopy(snapshot)
    session_id = str(result.get('session_id', ''))
    generation = int(result.get('generation', 0))
    planner_revision = generation if revision is None else int(revision)
    event_id = _stable_event_id(
        session_id, generation, stage,
        str(correlation_id) + str(identity_suffix), planner_revision)
    events = list(result.get('events', []))
    if any(str(item.get('event_id')) == event_id for item in events):
        return result
    event = {
        'event_id': event_id,
        'timestamp_ns': time.time_ns(),
        'accepted_view_cycle': generation,
        'planner_revision': planner_revision,
        'stage': str(stage),
        'correlation_id': str(correlation_id),
        'ray_deltas': deepcopy(ray_deltas or {}),
        'selected_ray_ids': sorted(int(value) for value in selected_ray_ids or []),
        'attempted_ray_ids': sorted(int(value) for value in attempted_ray_ids or []),
        'newly_culled_ray_ids': sorted(
            int(value) for value in newly_culled_ray_ids or []),
        'reasons': deepcopy(reasons or {}),
        'metrics': deepcopy(metrics or {}),
        'message': str(message),
        'target_center_m': deepcopy(result.get('target_center_m')),
        'ray_population_phase': str(result.get(
            'ray_population_phase', '')),
        'ray_population_sha256': str(result.get(
            'ray_population_sha256', '')),
    }
    if extra:
        event.update(deepcopy(extra))
    events.append(event)
    result['events'] = events
    return result


def _finite_float(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _vector(mapping):
    if isinstance(mapping, dict):
        values = [_finite_float(mapping.get(axis)) for axis in ('x', 'y', 'z')]
    elif isinstance(mapping, (list, tuple)) and len(mapping) == 3:
        values = [_finite_float(value) for value in mapping]
    else:
        values = [0.0, 0.0, 0.0]
    return values


def _ray_angles(direction):
    x_value, y_value, z_value = _vector(direction)
    norm = math.sqrt(x_value * x_value + y_value * y_value + z_value * z_value)
    if norm <= 1e-12:
        return 0.0, 0.0
    azimuth = math.degrees(math.atan2(y_value, x_value))
    if azimuth < 0.0:
        azimuth += 360.0
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, z_value / norm))))
    return azimuth, elevation


def _candidate_id(candidate):
    return int(candidate.get('ray_id', candidate.get('index', -1)))


def planner_generation_snapshot(
        session_id, generation, target_center, frame_id, policy, generated,
        history_remaining, ranked, selected, planner_rejections=None,
        selection_ready=True, selection_reason='', remaining_views=1,
        mission_id='', persistent_culls=None, target_envelope=None):
    """Build the complete planner-owned ray snapshot for one generation."""
    generated = [dict(item) for item in generated]
    history_ids = {_candidate_id(item) for item in history_remaining}
    selected_ids = {_candidate_id(item) for item in selected}
    ranked_by_id = {_candidate_id(item): item for item in ranked}
    ranked_order = {
        _candidate_id(item): order for order, item in enumerate(ranked, start=1)}
    rejection_map = {
        int(key): [str(value) for value in values]
        for key, values in (planner_rejections or {}).items()
    }
    persistent_map = {
        int(key): deepcopy(value)
        for key, value in (persistent_culls or {}).items()
        if isinstance(value, dict)
    }
    rays = []
    for order, candidate in enumerate(generated, start=1):
        ray_id = _candidate_id(candidate)
        scored = ranked_by_id.get(ray_id, {})
        direction = _vector(candidate.get('ray_direction'))
        azimuth, elevation = _ray_angles(direction)
        planner_rank = int(scored.get(
            'nbv_rank', ranked_order.get(ray_id, order)))
        rank_kind = 'nbv' if 'nbv_rank' in scored else 'deterministic_seed'
        reasons = list(rejection_map.get(ray_id, []))
        if ray_id in persistent_map:
            planner_status = 'culled_hard'
            persistent = persistent_map[ray_id]
            reasons = [str(persistent.get(
                'reason', 'permanently infeasible ray'))]
        elif ray_id not in history_ids:
            planner_status = 'culled_history'
            if not reasons:
                reasons = ['duplicate or redundant accepted-view direction']
        elif not selection_ready:
            planner_status = 'withheld'
            reasons = [selection_reason or 'authoritative NBV model is not ready']
        elif ray_id not in selected_ids:
            planner_status = 'culled_information'
            if bool(scored.get('nbv_positive_information_gain', False)):
                reasons = reasons or ['below minimum useful information threshold']
            else:
                reasons = reasons or ['no positive predicted information gain']
        elif int(remaining_views) <= 0:
            planner_status = 'mission_complete'
            reasons = ['mission accepted-view limit reached']
        else:
            planner_status = 'remaining'
        ray = {
            'ray_id': ray_id,
            'generated_order': order,
            'direction': direction,
            'representative_position_m': _vector(
                candidate.get('desired_camera_position')),
            'azimuth_deg': azimuth,
            'elevation_deg': elevation,
            'minimum_standoff_m': _finite_float(
                candidate.get('ray_min_standoff_m')),
            'maximum_standoff_m': _finite_float(
                candidate.get('ray_max_standoff_m')),
            'scoring_standoff_m': _finite_float(
                candidate.get('ray_scoring_standoff_m')),
            'rank': planner_rank,
            'rank_kind': rank_kind,
            'planner_status': planner_status,
            'planner_reasons': reasons,
        }
        for source_key, destination_key in (
                ('ray_requested_min_standoff_m',
                 'requested_minimum_standoff_m'),
                ('ray_requested_max_standoff_m',
                 'requested_maximum_standoff_m'),
                ('ray_envelope_min_standoff_m',
                 'envelope_minimum_standoff_m'),
                ('ray_envelope_max_standoff_m',
                 'envelope_maximum_standoff_m'),
                ('target_envelope_supported', 'target_envelope_supported'),
                ('target_envelope_sha256', 'target_envelope_sha256'),
                ('target_envelope_rejection_reason',
                 'target_envelope_rejection_reason')):
            if source_key in candidate:
                ray[destination_key] = deepcopy(candidate[source_key])
        if planner_status == 'culled_hard':
            ray['cull_disposition'] = 'permanent'
            ray['hard_cull_evidence'] = deepcopy(persistent_map[ray_id])
        for key in (
                'nbv_rank_score', 'nbv_predicted_unknown_pixels',
                'nbv_novel_surface_pixels', 'nbv_marginal_information_pixels',
                'nbv_marginal_information_fraction',
                'nbv_projected_object_pixels', 'nbv_direction_novelty_deg',
                'nbv_camera_travel_m', 'nbv_positive_information_gain'):
            if key in scored:
                ray[key] = scored[key]
        rays.append(ray)
    result = {
        'schema_version': SCHEMA_VERSION,
        'mission_id': str(mission_id),
        'session_id': str(session_id),
        'generation': int(generation),
        'frame_id': str(frame_id),
        'target_center_m': _vector(target_center),
        'view_selection_policy': str(policy),
        'selection_ready': bool(selection_ready),
        'selection_reason': str(selection_reason),
        'remaining_views': int(remaining_views),
        'generated_ray_count': len(rays),
        'rays': rays,
        'requests': [],
        'events': [],
        'persistent_hard_culls': {
            str(key): deepcopy(value) for key, value in persistent_map.items()},
        'target_envelope': deepcopy(target_envelope),
        'ray_population_phase': (
            'qualified' if target_envelope is not None else 'bootstrap'),
    }
    generated_deltas = {}
    for ray in rays:
        generated_deltas[str(ray['ray_id'])] = {
            key: deepcopy(ray[key]) for key in (
                'ray_id', 'generated_order', 'direction',
                'representative_position_m', 'azimuth_deg', 'elevation_deg',
                'minimum_standoff_m', 'maximum_standoff_m',
                'scoring_standoff_m')}
        generated_deltas[str(ray['ray_id'])].update({
            key: deepcopy(ray[key]) for key in (
                'requested_minimum_standoff_m',
                'requested_maximum_standoff_m',
                'envelope_minimum_standoff_m',
                'envelope_maximum_standoff_m',
                'target_envelope_supported', 'target_envelope_sha256',
                'target_envelope_rejection_reason')
            if key in ray})
        generated_deltas[str(ray['ray_id'])]['generation'] = int(generation)
    canonical_population = canonical_ray_population(rays)
    generated_population = [dict(item, generation=int(generation))
                            for item in canonical_population]
    result['generated_ray_population'] = generated_population
    result['ray_population_sha256'] = ray_universe_sha256(
        rays, target_center, frame_id)
    result['ray_population_complete'] = (
        len(generated_population) == len(rays))
    population_created = bool(
        int(generation) == 0
        or (int(generation) == 1 and target_envelope is not None))
    if population_created:
        phase = result['ray_population_phase']
        result = _append_event(
            result,
            'generate' if int(generation) == 0 else 'upgrade_population',
            ray_deltas=generated_deltas,
            metrics={
                'generated_ray_count': len(rays),
                'input_ray_count': 0,
                'eliminated_ray_count': 0,
                'surviving_ray_count': len(rays),
                'ray_population_sha256': result['ray_population_sha256'],
            },
            message=(
                'Generated the temporary bootstrap target-ray population'
                if phase == 'bootstrap' else
                'Replaced bootstrap rays with the qualified permanent '
                'target-ray population'),
            extra={
                'target_envelope': deepcopy(target_envelope),
                'population_reset': (
                    phase == 'qualified' and int(generation) > 0),
            })
    history_culled = [
        ray for ray in rays if ray['planner_status'] == 'culled_history']
    hard_culled = [
        ray for ray in rays if ray['planner_status'] == 'culled_hard']
    initial_cycle = population_created
    culled_this_event = (
        history_culled + hard_culled if initial_cycle else history_culled)
    cull_input_count = (
        len(rays) if initial_cycle else len(rays) - len(hard_culled))
    cull_stage = 'cull' if initial_cycle else 'cull_used_redundant'
    result = _append_event(
        result, cull_stage,
        ray_deltas={str(ray['ray_id']): {
            'ray_id': ray['ray_id'],
            'status': ('culled' if ray['planner_status'] in (
                           'culled_history', 'culled_hard')
                       else 'surviving'),
            'culled': ray['planner_status'] in (
                'culled_history', 'culled_hard'),
            'cull_stage': (
                'history' if ray['planner_status'] == 'culled_history'
                else str(ray.get('hard_cull_evidence', {}).get(
                    'stage', 'hard_cull'))
                if ray['planner_status'] == 'culled_hard' else ''),
            'cull_disposition': (
                'permanent' if ray['planner_status'] == 'culled_hard'
                else ray.get('cull_disposition', '')),
            'reasons': (deepcopy(ray['planner_reasons'])
                        if ray['planner_status'] in (
                            'culled_history', 'culled_hard') else [])}
            for ray in rays},
        newly_culled_ray_ids=[ray['ray_id'] for ray in culled_this_event],
        reasons={str(ray['ray_id']): ray['planner_reasons']
                 for ray in culled_this_event},
        metrics={
            'input_ray_count': cull_input_count,
            'eliminated_ray_count': len(culled_this_event),
            'surviving_ray_count': cull_input_count - len(culled_this_event),
            'carried_hard_culled_count': (
                0 if initial_cycle else len(hard_culled)),
        })
    rank_stage = 'seed_rank' if int(generation) == 0 else 'nbv_rank'
    rankable = [ray for ray in rays if ray['planner_status'] not in (
        'culled_history', 'culled_hard')]
    if int(generation) > 0:
        result = _append_event(
            result, rank_stage,
            ray_deltas={str(ray['ray_id']): dict({
                'ray_id': ray['ray_id'], 'rank': ray['rank'],
                'rank_kind': ray['rank_kind']}, **{
                    key: deepcopy(ray[key]) for key in ray
                    if key.startswith('nbv_')}) for ray in rankable},
            metrics={
                'ranked_ray_count': len(rankable),
                'input_ray_count': len(rankable),
                'eliminated_ray_count': 0,
                'surviving_ray_count': len(rankable),
            })
    if int(generation) > 0:
        information_culled = [
            ray for ray in rays
            if ray['planner_status'] == 'culled_information']
        result = _append_event(
            result, 'information_cull',
            ray_deltas={str(ray['ray_id']): {
                'ray_id': ray['ray_id'], 'status': 'culled',
                'culled': True, 'cull_stage': 'information',
                'reasons': deepcopy(ray['planner_reasons'])}
                for ray in information_culled},
            newly_culled_ray_ids=[ray['ray_id'] for ray in information_culled],
            reasons={str(ray['ray_id']): ray['planner_reasons']
                     for ray in information_culled},
            metrics={
                'input_ray_count': len(rankable),
                'eliminated_ray_count': len(information_culled),
                'surviving_ray_count': (
                    len(rankable) - len(information_culled)),
            })
    return result


def add_prequalification(snapshot, viewpoints, filter_summary):
    """Add the coarse workspace/capability decision for planner survivors."""
    result = deepcopy(snapshot)
    by_id = {_candidate_id(item): item for item in viewpoints if isinstance(item, dict)}
    for ray in result.get('rays', []):
        candidate = by_id.get(int(ray['ray_id']))
        if candidate is None:
            continue
        accepted = bool(candidate.get(
            'prequalified', candidate.get('reachable', False)))
        ray['prequalification_status'] = 'remaining' if accepted else 'culled'
        ray['prequalification_reasons'] = [
            str(value) for value in candidate.get('reject_reasons', [])]
        if not accepted:
            ray['cull_disposition'] = str(candidate.get(
                'cull_disposition', 'retry_eligible'))
        capability = candidate.get('capability_map_prequalification')
        if isinstance(capability, dict):
            ray['capability_map'] = dict(capability)
        if candidate.get('ray_capability_bounded'):
            ray.update({
                'requested_minimum_standoff_m': _finite_float(candidate.get(
                    'ray_requested_min_standoff_m')),
                'requested_maximum_standoff_m': _finite_float(candidate.get(
                    'ray_requested_max_standoff_m')),
                'minimum_standoff_m': _finite_float(candidate.get(
                    'ray_min_standoff_m')),
                'maximum_standoff_m': _finite_float(candidate.get(
                    'ray_max_standoff_m')),
                'scoring_standoff_m': _finite_float(candidate.get(
                    'ray_scoring_standoff_m')),
                'representative_position_m': _vector(candidate.get(
                    'desired_camera_position')),
                'capability_intervals_m': deepcopy(candidate.get(
                    'ray_capability_intervals_m', [])),
                'capability_bounded': True,
            })
    result['prequalification'] = dict(filter_summary or {})
    deltas = {}
    culled = []
    reasons = {}
    for ray in result.get('rays', []):
        status = ray.get('prequalification_status')
        if status is None:
            continue
        ray_id = int(ray['ray_id'])
        rejected = status == 'culled'
        deltas[str(ray_id)] = {
            'ray_id': ray_id,
            'status': 'culled' if rejected else 'surviving',
            'culled': rejected,
            'cull_stage': 'prequalification' if rejected else '',
            'reasons': deepcopy(ray.get('prequalification_reasons', [])),
            'cull_disposition': ray.get('cull_disposition', ''),
            **({key: deepcopy(ray[key]) for key in (
                'requested_minimum_standoff_m',
                'requested_maximum_standoff_m',
                'minimum_standoff_m', 'maximum_standoff_m',
                'scoring_standoff_m', 'representative_position_m',
                'capability_intervals_m', 'capability_bounded')
               if key in ray}),
            **({'capability_map': deepcopy(ray['capability_map'])}
               if 'capability_map' in ray else {}),
        }
        if rejected:
            culled.append(ray_id)
            reasons[str(ray_id)] = deepcopy(
                ray.get('prequalification_reasons', []))
    result = _append_event(
        result, 'prequalify', ray_deltas=deltas,
        newly_culled_ray_ids=culled, reasons=reasons,
        metrics=dict(filter_summary or {}, **{
            'input_ray_count': len(deltas),
            'eliminated_ray_count': len(culled),
            'surviving_ray_count': len(deltas) - len(culled),
        }))
    if int(result.get('generation', 0)) == 0:
        rankable = [
            ray for ray in result.get('rays', [])
            if ray.get('planner_status') not in (
                'culled_history', 'culled_hard')
            and ray.get('prequalification_status') != 'culled']
        result = _append_event(
            result, 'seed_rank',
            ray_deltas={str(ray['ray_id']): {
                'ray_id': ray['ray_id'], 'rank': ray['rank'],
                'rank_kind': ray['rank_kind']} for ray in rankable},
            metrics={
                'ranked_ray_count': len(rankable),
                'input_ray_count': len(rankable),
                'eliminated_ray_count': 0,
                'surviving_ray_count': len(rankable),
            })
    return result


def add_bridge_request(
        snapshot, request_id, shortlisted_ray_ids, retired_ray_ids=None,
        transiently_exhausted_ray_ids=None):
    """Record the bounded bridge shortlist without changing its membership."""
    result = deepcopy(snapshot)
    shortlisted = {int(value) for value in shortlisted_ray_ids}
    retired = {int(value) for value in (retired_ray_ids or [])}
    transient = {
        int(value) for value in (transiently_exhausted_ray_ids or [])}
    for ray in result.get('rays', []):
        ray_id = int(ray['ray_id'])
        if ray.get('prequalification_status') == 'culled' or not str(
                ray.get('planner_status', '')).startswith('remaining'):
            continue
        if ray_id in retired:
            ray['bridge_status'] = 'culled_permanent_endpoint'
            ray['bridge_reasons'] = [
                'retired after permanent Tesseract endpoint infeasibility']
        elif ray_id in transient:
            ray['bridge_status'] = 'culled_generation_exhausted'
            ray['bridge_reasons'] = [
                'already exhausted by Tesseract in this accepted-view generation']
        elif ray_id in shortlisted:
            ray['bridge_status'] = 'shortlisted'
            ray['bridge_reasons'] = []
        else:
            ray['bridge_status'] = 'deferred_shortlist'
            ray['bridge_reasons'] = [
                'not attempted in this bounded Tesseract request; remains '
                'eligible for a later request']
    request = {
        'request_id': str(request_id),
        'status': 'queued',
        'shortlisted_ray_ids': sorted(shortlisted),
    }
    result['requests'] = _merge_requests(result.get('requests', []), [request])
    bridge_deltas = {}
    newly_culled = []
    reasons = {}
    for ray in result.get('rays', []):
        if 'bridge_status' not in ray:
            continue
        ray_id = int(ray['ray_id'])
        is_culled = str(ray['bridge_status']).startswith('culled')
        bridge_deltas[str(ray_id)] = {
            'ray_id': ray_id,
            'status': ('culled' if is_culled else 'deferred'
                       if ray['bridge_status'] == 'deferred_shortlist'
                       else 'surviving'),
            'culled': is_culled,
            'cull_stage': 'bridge' if is_culled else '',
            'reasons': deepcopy(ray.get('bridge_reasons', [])),
        }
        if is_culled:
            newly_culled.append(ray_id)
            reasons[str(ray_id)] = deepcopy(ray.get('bridge_reasons', []))
    return _append_event(
        result, 'plan', correlation_id=request_id,
        ray_deltas=bridge_deltas,
        attempted_ray_ids=shortlisted,
        newly_culled_ray_ids=newly_culled, reasons=reasons,
        metrics={
            'input_ray_count': len(bridge_deltas),
            'eliminated_ray_count': len(newly_culled),
            'surviving_ray_count': (
                len(bridge_deltas) - len(newly_culled)),
            'shortlisted_ray_count': len(shortlisted),
        },
        message='Queued one correlated Tesseract planning request',
        extra={'request_id': str(request_id), 'request_status': 'queued'})


def add_tesseract_response(snapshot, payload, request=None):
    """Add worker attempt, rejection, and selection evidence."""
    result = deepcopy(snapshot)
    diagnostics = payload.get('planning_diagnostics', {})
    attempted = {int(value) for value in diagnostics.get('attempted_ray_ids', [])}
    selected_items = {
        int(item['ray_id']): item
        for item in payload.get('selected_viewpoints', [])
        if item.get('ray_id') is not None}
    selected = set(selected_items)
    segment_endpoints = {
        int(segment['to_viewpoint']): segment['points'][-1]['positions_rad']
        for segment in payload.get('segments', [])
        if (
            isinstance(segment, dict)
            and isinstance(segment.get('points'), list)
            and segment['points']
            and isinstance(segment['points'][-1], dict)
            and isinstance(segment['points'][-1].get('positions_rad'), list)
            and segment.get('to_viewpoint') is not None
        )
    }
    candidate_to_ray = {
        int(item['id']): int(item['ray_id'])
        for item in (request or {}).get('scene', {}).get('candidate_views', [])
        if item.get('ray_id') is not None}
    failures = {}
    for item in diagnostics.get('candidate_failures', []):
        ray_id = item.get('ray_id')
        if ray_id is None:
            ray_id = candidate_to_ray.get(int(item.get('id', -1)))
        if ray_id is None:
            continue
        failures.setdefault(int(ray_id), []).append({
            'stage': str(item.get('stage', 'PLANNING_FAILURE')),
            'detail': str(item.get('detail', '')),
            'permanent_endpoint_failure': bool(
                item.get('permanent_endpoint_failure', False)),
        })
    for ray in result.get('rays', []):
        ray_id = int(ray['ray_id'])
        if ray_id in selected:
            selected_item = selected_items[ray_id]
            ray['tesseract_status'] = 'selected'
            ray['tesseract_reasons'] = []
            ray['camera_position_m'] = _vector(
                selected_item.get('camera_position_m'))
            ray['look_direction'] = _vector(
                selected_item.get('look_direction'))
            endpoint = segment_endpoints.get(int(selected_item.get('id', -1)))
            if endpoint is not None and len(endpoint) == 6:
                ray['planned_joint_positions_rad'] = [
                    _finite_float(value) for value in endpoint]
                ray['robot_pose_source'] = 'tesseract_planned_endpoint'
        elif ray_id in failures:
            ray['tesseract_status'] = 'culled'
            ray['tesseract_reasons'] = failures[ray_id]
        elif ray_id in attempted:
            ray['tesseract_status'] = 'attempted_not_selected'
            ray['tesseract_reasons'] = [{
                'stage': 'NOT_SELECTED',
                'detail': 'attempted but no selected endpoint was returned',
                'permanent_endpoint_failure': False,
            }]
    request = {
        'request_id': str(payload.get('request_id', '')),
        'status': str(payload.get('status', 'unknown')),
        'selected_ray_ids': sorted(selected),
        'attempted_ray_ids': sorted(attempted),
        'rejection_codes': [
            str(value) for value in payload.get('rejection_codes', [])],
        'diagnostic': str(payload.get('diagnostic', '')),
    }
    result['requests'] = _merge_requests(result.get('requests', []), [request])
    deltas = {}
    newly_culled = []
    reasons = {}
    affected_ray_ids = attempted | selected | set(failures)
    for ray in result.get('rays', []):
        status = ray.get('tesseract_status')
        ray_id = int(ray['ray_id'])
        if status is None or ray_id not in affected_ray_ids:
            continue
        rejected = status == 'culled'
        delta = {
            'ray_id': ray_id,
            'status': 'selected' if status == 'selected' else (
                'culled' if rejected else status),
            'culled': rejected,
            'cull_stage': 'tesseract' if rejected else '',
            'reasons': deepcopy(ray.get('tesseract_reasons', [])),
        }
        if rejected:
            delta['cull_disposition'] = (
                'permanent' if any(bool(item.get(
                    'permanent_endpoint_failure')) for item in ray.get(
                        'tesseract_reasons', []) if isinstance(item, dict))
                else 'retry_eligible')
        for key in (
                'camera_position_m', 'look_direction',
                'planned_joint_positions_rad', 'robot_pose_source'):
            if key in ray:
                delta[key] = deepcopy(ray[key])
        deltas[str(ray_id)] = delta
        if rejected:
            newly_culled.append(ray_id)
            reasons[str(ray_id)] = deepcopy(ray.get('tesseract_reasons', []))
    selected_evidence = []
    for item in payload.get('selected_viewpoints', []):
        selected_evidence.append({
            key: deepcopy(item[key]) for key in item
            if key in ('id', 'ray_id', 'camera_position_m', 'look_direction',
                       'planned_camera_roll_rad', 'camera_roll_rad')})
    return _append_event(
        result, 'plan', correlation_id=payload.get('request_id', ''),
        identity_suffix=':result', ray_deltas=deltas,
        selected_ray_ids=selected, attempted_ray_ids=attempted,
        newly_culled_ray_ids=newly_culled, reasons=reasons,
        metrics=dict(deepcopy(diagnostics), **{
            'input_ray_count': len(affected_ray_ids),
            'eliminated_ray_count': len(newly_culled),
            'surviving_ray_count': (
                len(affected_ray_ids) - len(newly_culled)),
            'selected_ray_count': len(selected),
        }),
        message=str(payload.get('diagnostic', '')),
        extra={
            'request_id': str(payload.get('request_id', '')),
            'request_status': str(payload.get('status', 'unknown')),
            'rejection_codes': [
                str(value) for value in payload.get('rejection_codes', [])],
            'planned_camera_evidence': selected_evidence,
            'standoff_probe_evidence': deepcopy(diagnostics.get(
                'standoff_probes', diagnostics.get('candidate_failures', []))),
        })


def add_request_rejection(snapshot, request_id, code, reason):
    result = deepcopy(snapshot)
    request = {
        'request_id': str(request_id),
        'status': 'rejected',
        'rejection_codes': [str(code)],
        'diagnostic': str(reason),
    }
    result['requests'] = _merge_requests(result.get('requests', []), [request])
    return _append_event(
        result, 'plan', correlation_id=request_id,
        identity_suffix=':rejection:' + str(code),
        reasons={'request': [str(code), str(reason)]}, message=str(reason),
        extra={
            'request_id': str(request_id), 'request_status': 'rejected',
            'rejection_codes': [str(code)],
        })


def capture_event_identity(history, metadata, accepted):
    """Resolve capture correlation from runtime history or frame evidence."""
    entries = history.get('accepted_entries', []) \
        if isinstance(history, dict) else []
    entry = entries[-1] if entries and isinstance(entries[-1], dict) else {}
    selection = metadata.get('view_selection', {}) \
        if isinstance(metadata, dict) else {}
    if not isinstance(selection, dict):
        selection = {}
    execution = metadata.get('scan_execution', {}) \
        if isinstance(metadata, dict) else {}
    if not isinstance(execution, dict):
        execution = {}
    capture_id = next((str(value) for value in (
        entry.get('plan_id'), selection.get('plan_id'),
        selection.get('request_id'), execution.get('plan_id'))
        if str(value or '').strip()), 'capture-%03d' % int(accepted))
    raw_ray_id = entry.get('ray_id')
    if raw_ray_id in (None, ''):
        raw_ray_id = selection.get('ray_id')
    try:
        ray_id = int(raw_ray_id)
    except (TypeError, ValueError):
        ray_id = None
    return capture_id, ray_id


def add_capture_event(snapshot, capture_id, accepted, ray_id=None,
                      achieved_camera_matrix_4x4=None, joint_names=None,
                      joint_positions=None, gripper_joint_names=None,
                      gripper_joint_positions=None, reason='',
                      coverage_snapshot_path='', artifact_bindings=None):
    """Record an achieved capture outcome without issuing any command."""
    selected = [] if ray_id is None else [int(ray_id)]
    robot_pose = {
        'source': 'achieved_capture' if accepted else 'last_achieved_pose',
        'joint_names': [str(value) for value in joint_names or []],
        'joint_positions_rad': [
            _finite_float(value) for value in joint_positions or []],
        'gripper_joint_names': [
            str(value) for value in gripper_joint_names or []],
        'gripper_joint_positions': [
            _finite_float(value) for value in gripper_joint_positions or []],
    }
    extra = {
        'capture_id': str(capture_id),
        'capture_accepted': bool(accepted),
        'capture_rejection_reason': '' if accepted else str(reason),
        'achieved_camera_matrix_4x4': deepcopy(
            achieved_camera_matrix_4x4),
        'robot_pose': robot_pose,
        'captured_ray_ids': selected if accepted else [],
        'artifact_bindings': deepcopy(artifact_bindings or []),
    }
    if accepted:
        extra['coverage_snapshot_path'] = str(coverage_snapshot_path)
    return _append_event(
        snapshot, 'capture', correlation_id=capture_id,
        identity_suffix=':accepted' if accepted else ':rejected',
        selected_ray_ids=selected,
        ray_deltas={str(ray_id): {
            'ray_id': int(ray_id),
            'status': 'captured' if accepted else 'capture_rejected',
            'reasons': [] if accepted else [str(reason)],
        }} if ray_id is not None else {},
        reasons={} if accepted else {'capture': [str(reason)]},
        message='Capture accepted' if accepted else str(reason), extra=extra)


def add_target_update_event(snapshot, capture_id, coverage_snapshot_path,
                            newly_measured_points_path='', reason=''):
    available = bool(coverage_snapshot_path)
    return _append_event(
        snapshot, 'update_target', correlation_id=capture_id,
        message='Target model updated' if available else (
            reason or 'target-model artifacts unavailable'),
        extra={'capture_id': str(capture_id), 'target_model': {
            'available': available,
            'snapshot_path': str(coverage_snapshot_path),
            'newly_measured_points_path': str(newly_measured_points_path),
            'reason': '' if available else (
                reason or 'target-model artifacts unavailable'),
        }})


def add_terminal_event(snapshot, outcome, reason='', failure_code=''):
    normalized = str(outcome).strip().lower()
    stage = {
        'complete': 'completed', 'completed': 'completed',
        'success': 'completed', 'succeeded': 'completed',
        'cancel': 'cancelled', 'cancelled': 'cancelled', 'canceled': 'cancelled',
        'failure': 'failed', 'failed': 'failed', 'error': 'failed',
    }.get(normalized, 'failed')
    return _append_event(
        snapshot, stage, correlation_id='terminal', message=str(reason),
        extra={'terminal_outcome': str(outcome),
               'failure_code': str(failure_code)})


def _merge_requests(existing, incoming):
    merged = {str(item.get('request_id', '')): deepcopy(item) for item in existing}
    for item in incoming:
        request_id = str(item.get('request_id', ''))
        current = merged.get(request_id, {})
        current.update(deepcopy(item))
        merged[request_id] = current
    return [merged[key] for key in sorted(merged)]


def _merge_generation(existing, incoming):
    merged = deepcopy(existing)
    for key, value in incoming.items():
        if key not in ('rays', 'requests', 'events'):
            merged[key] = deepcopy(value)
    rays = {int(item['ray_id']): deepcopy(item) for item in merged.get('rays', [])}
    for item in incoming.get('rays', []):
        ray_id = int(item['ray_id'])
        current = rays.get(ray_id, {})
        current.update(deepcopy(item))
        rays[ray_id] = current
    merged['rays'] = [rays[key] for key in sorted(rays)]
    merged['requests'] = _merge_requests(
        merged.get('requests', []), incoming.get('requests', []))
    events = {
        str(item.get('event_id')): deepcopy(item)
        for item in merged.get('events', []) if item.get('event_id')}
    for item in incoming.get('events', []):
        event_id = str(item.get('event_id', ''))
        if event_id and event_id not in events:
            events[event_id] = deepcopy(item)
    merged['events'] = sorted(events.values(), key=lambda item: (
        int(item.get('timestamp_ns', 0)), str(item.get('event_id', ''))))
    return merged


def _normalised(values):
    vector = _vector(values)
    length = math.sqrt(sum(value * value for value in vector))
    if length <= 1e-12:
        return [0.0, 0.0, 0.0]
    return [value / length for value in vector]


def _historical_target_center(view_selection, camera_position):
    look = _normalised(view_selection.get('look_direction'))
    standoff = _finite_float(view_selection.get('ray_standoff_m'))
    if standoff > 0.0 and any(abs(value) > 1e-12 for value in look):
        return [
            camera_position[index] + look[index] * standoff
            for index in range(3)
        ]
    return [0.0, 0.0, 0.0]


def _legacy_view_selection(metadata, camera_transform):
    matrix = camera_transform.get('matrix_4x4')
    target = metadata.get('target_3d', {})
    if not (
            bool(camera_transform.get('available', False))
            and isinstance(matrix, list) and len(matrix) == 4
            and isinstance(target, dict)
            and bool(target.get('available', False))):
        raise ValueError('historical capture has no selected-view evidence')
    camera = _vector(camera_transform.get('translation_m'))
    camera_target = _vector(target.get('point'))
    homogeneous = camera_target + [1.0]
    try:
        target_center = [
            sum(float(matrix[row][column]) * homogeneous[column]
                for column in range(4))
            for row in range(3)
        ]
    except (IndexError, TypeError, ValueError):
        raise ValueError('legacy capture camera transform is invalid')
    offset = [target_center[index] - camera[index] for index in range(3)]
    standoff = math.sqrt(sum(value * value for value in offset))
    frame_index = int(metadata.get('frame_index', 0))
    return {
        'available': True,
        'camera_position_m': camera,
        'look_direction': _normalised(offset),
        'ray_standoff_m': standoff,
        'ray_id': frame_index,
        'nbv_rank': 0,
        'view_selection_generation': frame_index,
        'view_selection_policy': 'legacy_achieved_capture_pose',
        'candidate_diagnostics': {'attempted_ray_ids': [frame_index]},
        'legacy_pose_only': True,
    }


def historical_replay_snapshot(dataset, metadata, source_path,
                               coverage_snapshot_path='',
                               coverage_error='target-model artifacts unavailable'):
    """Convert one immutable capture record into a diagnostic generation."""
    if not isinstance(metadata, dict):
        raise ValueError('historical capture metadata is not an object')
    camera_transform = metadata.get('camera_transform', {})
    view = metadata.get('view_selection')
    if not isinstance(view, dict) or not bool(view.get('available', False)):
        view = _legacy_view_selection(metadata, camera_transform)
    achieved_camera = _vector(camera_transform.get('translation_m'))
    desired_camera = _vector(view.get('camera_position_m'))
    camera_position = (
        achieved_camera
        if bool(camera_transform.get('available', False))
        else desired_camera
    )
    target_center = _historical_target_center(view, desired_camera)
    look_direction = _normalised(view.get('look_direction'))
    direction = [-value for value in look_direction]
    azimuth, elevation = _ray_angles(direction)
    joint_state = metadata.get('joint_state', {})
    joint_positions = joint_state.get('position', [])
    achieved_joints = []
    if bool(joint_state.get('available', False)) and len(joint_positions) >= 6:
        achieved_joints = [_finite_float(value) for value in joint_positions[:6]]
    capture_index = int(metadata.get('frame_index', 0))
    generation = int(view.get('view_selection_generation', capture_index))
    ray_id = int(view.get('ray_id', capture_index))
    nbv_rank = int(view.get('nbv_rank', 0))
    ray = {
        'ray_id': ray_id,
        'generated_order': capture_index + 1,
        'direction': direction,
        'representative_position_m': desired_camera,
        'camera_position_m': desired_camera,
        'achieved_camera_position_m': camera_position,
        'look_direction': look_direction,
        'azimuth_deg': azimuth,
        'elevation_deg': elevation,
        'minimum_standoff_m': _finite_float(view.get('ray_standoff_m')),
        'maximum_standoff_m': _finite_float(view.get('ray_standoff_m')),
        'scoring_standoff_m': _finite_float(view.get('ray_standoff_m')),
        'rank': nbv_rank if nbv_rank > 0 else capture_index + 1,
        'rank_kind': 'nbv' if nbv_rank > 0 else 'historical_selected_order',
        'planner_status': 'remaining',
        'planner_reasons': [],
        'prequalification_status': 'remaining',
        'prequalification_reasons': [],
        'bridge_status': 'shortlisted',
        'bridge_reasons': [],
        'tesseract_status': 'selected',
        'tesseract_reasons': [],
        'capture_index': capture_index,
        'historical_source_metadata': str(source_path),
        'robot_pose_source': 'achieved_joint_state_at_capture',
    }
    if achieved_joints:
        ray['achieved_joint_positions_rad'] = achieved_joints
    matrix = camera_transform.get('matrix_4x4')
    if (
            bool(camera_transform.get('available', False))
            and isinstance(matrix, list) and len(matrix) == 4):
        ray['achieved_camera_matrix_4x4'] = matrix
    for key in (
            'nbv_rank_score', 'nbv_predicted_unknown_pixels',
            'nbv_novel_surface_pixels', 'nbv_marginal_information_pixels',
            'nbv_marginal_information_fraction',
            'nbv_projected_object_pixels', 'nbv_direction_novelty_deg',
            'nbv_camera_travel_m', 'nbv_positive_information_gain'):
        if key in view:
            ray[key] = view[key]
    planned_count = max(1, int(metadata.get('planned_viewpoint_count', 1)))
    reachable_count = max(
        1, int(metadata.get('reachable_viewpoint_count', 1)))
    request_id = str(view.get('request_id', view.get('plan_id', '')))
    result = {
        'schema_version': SCHEMA_VERSION,
        'mission_id': 'replay_' + dataset.name,
        'session_id': 'replay_' + dataset.name,
        'generation': generation,
        'frame_id': str(camera_transform.get('header', {}).get(
            'frame_id', 'base_link')),
        'target_center_m': target_center,
        'view_selection_policy': str(
            view.get('view_selection_policy', 'historical')),
        'selection_ready': True,
        'selection_reason': 'accepted historical capture replay',
        'remaining_views': 0,
        'generated_ray_count': planned_count,
        'known_ray_count': 1,
        'rays': [ray],
        'events': [],
        'requests': [{
            'request_id': request_id,
            'status': 'historical_capture_accepted',
            'selected_ray_ids': [ray_id],
            'attempted_ray_ids': [
                int(value) for value in view.get(
                    'candidate_diagnostics', {}).get(
                        'attempted_ray_ids', [ray_id])],
        }],
        'historical_replay': {
            'dataset': dataset.name,
            'partial_candidate_population': True,
            'known_selected_rays': 1,
            'original_generated_ray_count': planned_count,
            'original_reachable_ray_count': reachable_count,
            'legacy_pose_only': bool(view.get('legacy_pose_only', False)),
            'limitation': (
                'The archived capture predates full ray diagnostics. It '
                'preserves the accepted ray, achieved arm joints, camera pose, '
                'rank and planning summary, but not every rejected ray identity.'
            ),
        },
    }
    # Historical captures contain one real selected/captured ray, not the full
    # original population.  Start with an explicitly partial snapshot event;
    # do not manufacture old cull or ranking stages.
    result = _append_event(
        result, 'legacy_snapshot', correlation_id=str(capture_index),
        ray_deltas={str(ray_id): deepcopy(ray)},
        selected_ray_ids=[ray_id],
        attempted_ray_ids=result['requests'][0]['attempted_ray_ids'],
        message=result['historical_replay']['limitation'],
        extra={'legacy_partial': True})
    result = add_capture_event(
        result, request_id or 'capture-%03d' % capture_index, True,
        ray_id=ray_id,
        achieved_camera_matrix_4x4=ray.get('achieved_camera_matrix_4x4'),
        joint_names=['joint%d' % index for index in range(1, 7)],
        joint_positions=achieved_joints,
        coverage_snapshot_path=coverage_snapshot_path)
    return add_target_update_event(
        result, request_id or 'capture-%03d' % capture_index,
        coverage_snapshot_path, reason=coverage_error)


def replay_historical_dataset(dataset_dir, output_root):
    """Render archived accepted-view evidence without running ROS or motion."""
    dataset = Path(dataset_dir).expanduser().resolve()
    if not dataset.is_dir() or not (dataset / 'manifest.json').is_file():
        raise ValueError('historical dataset is missing its manifest')
    metadata_paths = sorted((dataset / 'frames').glob('view_*_metadata.yaml'))
    if not metadata_paths:
        raise ValueError('historical dataset has no frame metadata')
    store = RayMissionDiagnosticsStore(output_root)
    accepted = 0
    skipped = []
    target_model_unavailable = []
    artifact_paths = None
    loaded = []
    measured_centers = []
    for metadata_path in metadata_paths:
        try:
            with metadata_path.open('r', encoding='utf-8') as stream:
                metadata = yaml.safe_load(stream) or {}
            loaded.append((metadata_path, metadata))
            target = metadata.get('target_3d', {})
            transform = metadata.get('camera_transform', {}).get('matrix_4x4')
            point = target.get('point', {})
            if bool(target.get('available', False)) and transform is not None:
                matrix = np.asarray(transform, dtype=float)
                camera_point = np.asarray([
                    float(point['x']), float(point['y']), float(point['z']), 1.0])
                if matrix.shape == (4, 4) and np.all(np.isfinite(matrix)):
                    measured_centers.append(matrix.dot(camera_point)[:3])
        except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError):
            loaded.append((metadata_path, None))
    frozen_center = (
        np.median(np.asarray(measured_centers), axis=0)
        if measured_centers else None)
    coverage_dir = (
        Path(output_root) / RayMissionDiagnosticsStore.safe_session_id(
            'replay_' + dataset.name) / 'coverage')
    for capture_number, (metadata_path, metadata) in enumerate(loaded, start=1):
        try:
            if metadata is None:
                raise ValueError('historical capture metadata is unavailable')
            coverage_path = ''
            coverage_error = 'target-model artifacts unavailable'
            if frozen_center is not None:
                try:
                    from piper_mobile_manipulation.nbv_coverage import (
                        ObjectCoverageModel, persist_coverage_snapshot)
                    model = ObjectCoverageModel()
                    coverage = model.rebuild_from_scan(
                        dataset, capture_number, frozen_center,
                        'replay_' + dataset.name)
                    output = coverage_dir / ('capture_%03d.npz' % capture_number)
                    capture_paths = [metadata_path]
                    for key in (
                            'target_depth_png_file_path',
                            'target_support_mask_file_path', 'depth_file_path'):
                        if metadata.get(key):
                            capture_paths.append(metadata[key])
                    persist_coverage_snapshot(
                        output, coverage, capture_artifacts=capture_paths,
                        configuration_artifacts=[dataset / 'manifest.json'],
                        dataset_root=dataset)
                    coverage_path = str(output)
                    coverage_error = ''
                except (KeyError, OSError, TypeError, ValueError,
                        json.JSONDecodeError, yaml.YAMLError) as error:
                    coverage_error = (
                        'target-model artifacts unavailable: %s' % error)
                    target_model_unavailable.append({
                        'metadata': metadata_path.name,
                        'reason': coverage_error,
                    })
            snapshot = historical_replay_snapshot(
                dataset, metadata, metadata_path.relative_to(dataset),
                coverage_path, coverage_error)
            artifact_paths = store.record(snapshot)
            accepted += 1
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            skipped.append({
                'metadata': metadata_path.name,
                'reason': str(exc),
            })
    if artifact_paths is None:
        raise ValueError('historical dataset has no replayable accepted views')
    return {
        'dataset': dataset.name,
        'replayed_capture_count': accepted,
        'skipped_capture_count': len(skipped),
        'skipped': skipped,
        'target_model_available_count': accepted - len(target_model_unavailable),
        'target_model_unavailable_count': len(target_model_unavailable),
        'target_model_unavailable': target_model_unavailable,
        'json_path': artifact_paths[0],
        'html_path': artifact_paths[1],
        'command_free': True,
    }


class RayMissionDiagnosticsStore:
    """Merge and atomically render one report per mission session."""

    def __init__(self, root):
        self.root = Path(os.path.expanduser(str(root)))

    @staticmethod
    def safe_session_id(session_id):
        value = _SAFE_SESSION.sub('_', str(session_id).strip()).strip('._')
        return value or 'unknown_session'

    def session_dir(self, session_id):
        return self.root / self.safe_session_id(session_id)

    def record(self, snapshot):
        session_id = str(snapshot.get('session_id', '')).strip()
        if not session_id:
            raise ValueError('ray diagnostic session_id is missing')
        mission_id = str(snapshot.get('mission_id', '')).strip()
        artifact_id = mission_id or session_id
        generation = int(snapshot.get('generation', -1))
        if generation < 0:
            raise ValueError('ray diagnostic generation is invalid')
        directory = self.session_dir(artifact_id)
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / (ARTIFACT_BASENAME + '.json')
        html_path = directory / (ARTIFACT_BASENAME + '.html')
        lock_path = directory / ('.' + ARTIFACT_BASENAME + '.lock')
        with lock_path.open('a+', encoding='utf-8') as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            document = self._load(json_path, artifact_id, mission_id)
            generations = {
                (str(item['session_id']), int(item['generation'])): item
                for item in document.get('generations', [])}
            generation_key = (session_id, generation)
            generations[generation_key] = _merge_generation(
                generations.get(generation_key, {}), snapshot)
            document['generations'] = [
                generations[key] for key in sorted(generations)]
            document['session_ids'] = sorted({
                str(item['session_id']) for item in document['generations']})
            document['events'] = sorted((
                deepcopy(event)
                for item in document['generations']
                for event in item.get('events', [])), key=lambda item: (
                    int(item.get('timestamp_ns', 0)),
                    str(item.get('event_id', ''))))
            document['events'] = _normalize_new_cull_transitions(
                document['events'])
            document['ray_population_index'] = [{
                'session_id': str(item['session_id']),
                'generation': int(item['generation']),
                'ray_count': len(item.get('generated_ray_population', [])),
                'sha256': str(item.get('ray_population_sha256', '')),
            } for item in document['generations']
                if item.get('ray_population_complete')]
            historical_partial = any(
                bool(item.get('historical_replay', {}).get(
                    'partial_candidate_population', False))
                for item in document['generations'])
            seed_populations = [
                item for item in document['generations']
                if int(item.get('generation', -1)) == 0
                and item.get('ray_population_complete')
                and len(item.get('generated_ray_population', []))
                == int(item.get('generated_ray_count', -1))]
            seed_generate_events = [
                event for event in document['events']
                if event.get('stage') == 'generate'
                and event.get('ray_deltas')]
            document['journal_complete'] = bool(
                not historical_partial
                and seed_populations
                and seed_generate_events)
            document['ray_population_validation'] = (
                'complete' if document['journal_complete'] else 'partial')
            document['robot_model'] = deepcopy(ROBOT_MODEL)
            document['updated_at_ns'] = time.time_ns()
            self._atomic_text(
                json_path, json.dumps(document, indent=2, sort_keys=True) + '\n')
            self._atomic_text(html_path, render_html(document))
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        return str(json_path), str(html_path)

    @staticmethod
    def _load(path, artifact_id, mission_id):
        if path.is_file():
            try:
                with path.open('r', encoding='utf-8') as stream:
                    value = json.load(stream)
                if value.get('artifact_id') == artifact_id:
                    schema = int(value.get('schema_version', 1))
                    if schema == SCHEMA_VERSION:
                        return value
                    if schema == 1:
                        # Preserve final generation compatibility records.  A
                        # v1 artifact has no event journal, so never fabricate
                        # intermediate events during upgrade.
                        value['schema_version'] = SCHEMA_VERSION
                        value['upgraded_from_schema_version'] = 1
                        value['events'] = []
                        value['journal_complete'] = False
                        for generation in value.get('generations', []):
                            generation.setdefault('events', [])
                        return value
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        return {
            'schema_version': SCHEMA_VERSION,
            'artifact_kind': 'ray_mission_diagnostics',
            'artifact_id': artifact_id,
            'mission_id': mission_id,
            'session_ids': [],
            'generations': [],
            'events': [],
            'journal_complete': True,
            'ray_population_index': [],
            'ray_population_validation': 'pending',
            'robot_model': deepcopy(ROBOT_MODEL),
        }

    @staticmethod
    def _atomic_text(path, content):
        descriptor, temporary = tempfile.mkstemp(
            prefix='.' + path.name + '.', suffix='.tmp', dir=str(path.parent))
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, str(path))
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise


def _normalize_new_cull_transitions(events):
    """Mark a ray newly culled only on an actual active-to-culled edge."""
    states = {}
    result = []
    for source in events:
        event = deepcopy(source)
        population = str(event.get('ray_population_sha256', '')).strip()
        if not population:
            population = '%s:%s' % (
                event.get('accepted_view_cycle', ''),
                event.get('ray_population_phase', 'legacy'))
        if event.get('population_reset') is True:
            states[population] = {}
        population_states = states.setdefault(population, {})
        transitioned = set()
        for raw_id, delta in event.get('ray_deltas', {}).items():
            if not isinstance(delta, dict):
                continue
            ray_id = int(delta.get('ray_id', raw_id))
            was_culled = bool(population_states.get(ray_id, False))
            status = str(delta.get('status', ''))
            is_culled = bool(
                delta.get('culled') is True or status == 'culled')
            becomes_active = bool(
                delta.get('culled') is False
                or status in ('surviving', 'remaining'))
            if is_culled and not was_culled:
                transitioned.add(ray_id)
            if is_culled:
                population_states[ray_id] = True
            elif becomes_active:
                population_states[ray_id] = False
        declared = {
            int(value) for value in event.get('newly_culled_ray_ids', [])}
        event['newly_culled_ray_ids'] = sorted(declared & transitioned)
        result.append(event)
    return result


def render_html(document):
    """Render a dependency-free interactive report from canonical JSON data."""
    title = 'Ray mission diagnostics — %s' % (
        document.get('mission_id') or document.get('artifact_id', ''))
    encoded = json.dumps(document, sort_keys=True, separators=(',', ':')).replace(
        '</', '<\\/')
    template = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#10151d;--panel:#18212c;--ink:#edf4fb;--muted:#9fb0c2;
--grid:#344354;--generated:#7b8794;--planner:#59636f;
--prequalification:#d16ba5;--bridge:#e3a72f;--tesseract:#e05252;
--remaining:#3e9bd6;--selected:#39b86b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px system-ui,sans-serif}
header{padding:20px 24px 10px} h1{font-size:22px;margin:0 0 6px}
.muted{color:var(--muted)}.notice{margin:0 24px 12px;padding:10px;
border:1px solid #a97625;background:#2d2517;border-radius:6px;color:#f1d69a}
.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:10px 24px}
select,input{background:#0d131a;color:var(--ink);border:1px solid var(--grid);
border-radius:5px;padding:7px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
gap:8px;padding:0 24px 12px}
.card,.panel{background:var(--panel);border:1px solid #283544;border-radius:8px}
.card{padding:10px}.card b{display:block;font-size:20px}
.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));
gap:12px;padding:0 24px 12px}
.panel{padding:10px;overflow:auto}.panel h2{font-size:15px;margin:0 0 8px}
svg{width:100%;min-width:330px;height:auto}
.scene-panel{margin:0 24px 12px}.scene-controls{padding:0 0 8px}
canvas{display:block;width:100%;height:auto;max-height:650px;background:#0d131a;
border:1px solid var(--grid);border-radius:5px;cursor:grab}
canvas:active{cursor:grabbing}.pose-info{padding:7px 0;color:var(--muted)}
.legend{display:flex;gap:12px;flex-wrap:wrap;padding:0 24px 12px}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:4px}
.table-wrap{margin:0 24px 24px;max-height:620px;overflow:auto}
table{width:100%;border-collapse:collapse;background:var(--panel)}
th{position:sticky;top:0;background:#202c39}
th,td{padding:7px 8px;border-bottom:1px solid #2b3948;text-align:left;
white-space:nowrap}
tr:hover{background:#223141}.reason{white-space:normal;min-width:280px}
circle{stroke:#091019;stroke-width:1}
</style></head><body>
<header><h1>__TITLE__</h1><div class="muted">
Every generated target ray, each cull stage, remaining candidates, and rank.
Select a ray to inspect its L515 pose. The legacy kinematic drawing below is
schematic; use Open Ray Review for full checked-in URDF visual meshes.</div></header>
<div class="toolbar">
<label>Accepted-view generation <select id="generation"></select></label>
<label>Status <select id="status"><option value="all">all</option>
<option value="selected">selected</option><option value="remaining">remaining</option>
<option value="culled">culled</option></select></label>
<label>Find ray <input id="search" placeholder="ray ID or reason"></label></div>
<div id="cards" class="cards"></div><div id="replayNotice"></div>
<div id="legend" class="legend"></div>
<div class="plots"><section class="panel">
<h2>Spherical direction map (azimuth × elevation)</h2>
<svg id="sphere" viewBox="0 0 720 380"></svg></section>
<section class="panel"><h2>Target-centred top view (X/Y)</h2>
<svg id="top" viewBox="0 0 480 480"></svg></section>
<section class="panel"><h2>Target-centred side view (X/Z)</h2>
<svg id="side" viewBox="0 0 480 480"></svg></section></div>
<section class="panel scene-panel"><h2>3D PiPER URDF and L515 ray replay</h2>
<div class="toolbar scene-controls">
<label>Ray <select id="raySelect"></select></label>
<button id="previousRay">Previous</button><button id="nextRay">Next</button>
<button id="playRays">Play</button>
<label><input id="allCameras" type="checkbox" checked> all ray cameras</label>
<button id="resetView">Reset view</button></div>
<div id="poseInfo" class="pose-info"></div>
<canvas id="scene3d" width="1120" height="620"></canvas>
<div class="muted">Drag to orbit; use the mouse wheel to zoom. The arm is a
kinematic link rendering derived from piper_description.xacro. Camera poses are
desired poses unless achieved capture evidence is present.</div></section>
<div class="table-wrap panel"><table><thead><tr><th>Rank</th><th>Ray</th>
<th>Final stage</th><th>Azimuth</th><th>Elevation</th><th>Marginal info</th>
<th>Novel pixels</th><th class="reason">Reason</th></tr></thead>
<tbody id="rows"></tbody></table></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const doc=JSON.parse(document.getElementById('data').textContent);
const colors={generated:'#7b8794',planner:'#59636f',
  prequalification:'#d16ba5',bridge:'#e3a72f',tesseract:'#e05252',
  remaining:'#3e9bd6',selected:'#39b86b'};
function state(r){
  if(r.tesseract_status==='selected')return ['selected','selected'];
  if(r.tesseract_status==='culled'||
      r.tesseract_status==='attempted_not_selected')return ['culled','tesseract'];
  if(r.bridge_status&&r.bridge_status!=='shortlisted')return ['culled','bridge'];
  if(r.prequalification_status==='culled')return ['culled','prequalification'];
  if(r.planner_status&&r.planner_status!=='remaining')return ['culled','planner'];
  return ['remaining','remaining'];
}
function reasons(r){
  let out=[];
  (r.planner_reasons||[]).forEach(x=>out.push(x));
  (r.prequalification_reasons||[]).forEach(x=>out.push(x));
  (r.bridge_reasons||[]).forEach(x=>out.push(x));
  (r.tesseract_reasons||[]).forEach(x=>
    out.push((x.stage?x.stage+': ':'')+(x.detail||x)));
  return out.join(' | ');
}
function svg(tag,attrs,text){
  const node=document.createElementNS('http://www.w3.org/2000/svg',tag);
  Object.entries(attrs||{}).forEach(([key,value])=>node.setAttribute(key,value));
  if(text!==undefined)node.textContent=text;
  return node;
}
function basePlot(node,width,height,xLabels,yLabels){
  node.replaceChildren();
  node.append(svg('rect',{x:42,y:15,width:width-57,height:height-55,
    fill:'#121923',stroke:'#344354'}));
  for(let index=0;index<=4;index++){
    let x=42+(width-57)*index/4,y=15+(height-55)*index/4;
    node.append(svg('line',{x1:x,y1:15,x2:x,y2:height-40,stroke:'#263544'}));
    node.append(svg('line',{x1:42,y1:y,x2:width-15,y2:y,stroke:'#263544'}));
    node.append(svg('text',{x:x,y:height-18,fill:'#9fb0c2','font-size':11,
      'text-anchor':'middle'},xLabels[index]));
    node.append(svg('text',{x:36,y:y+4,fill:'#9fb0c2','font-size':11,
      'text-anchor':'end'},yLabels[index]));
  }
}
function tooltip(r){
  return `ray ${r.ray_id} · rank ${r.rank} (${r.rank_kind}) · `+
    `az ${r.azimuth_deg.toFixed(1)}° · el ${r.elevation_deg.toFixed(1)}° · `+
    `${state(r)[0]} · ${reasons(r)}`;
}
function point(node,x,y,r,radius=4){
  let current=state(r);
  let circle=svg('circle',{cx:x,cy:y,r:radius,fill:colors[current[1]]});
  circle.append(svg('title',{},tooltip(r)));
  node.append(circle);
  if(current[0]==='selected'||r.rank<=12){
    node.append(svg('text',{x:x+6,y:y-5,fill:colors[current[1]],
      'font-size':9},String(r.rank)));
  }
}
function vecAdd(a,b){return a.map((value,index)=>value+b[index])}
function vecSub(a,b){return a.map((value,index)=>value-b[index])}
function vecScale(a,scale){return a.map(value=>value*scale)}
function vecNorm(a){let length=Math.hypot(...a)||1;return vecScale(a,1/length)}
function cross(a,b){return [a[1]*b[2]-a[2]*b[1],
  a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}
function matrixMultiply(a,b){
  return a.map((row,i)=>b[0].map((_value,j)=>
    row.reduce((sum,value,k)=>sum+value*b[k][j],0)));
}
function dhTransform(alpha,a,theta,d){
  let ca=Math.cos(alpha),sa=Math.sin(alpha),ct=Math.cos(theta),st=Math.sin(theta);
  return [[ct,-st,0,a],[st*ca,ct*ca,-sa,-sa*d],
    [st*sa,ct*sa,ca,ca*d],[0,0,0,1]];
}
function transformPoint(matrix,point=[0,0,0]){
  let value=[...point,1];
  return matrix.slice(0,3).map(row=>
    row.reduce((sum,item,index)=>sum+item*value[index],0));
}
function urdfPoints(ray){
  let joints=ray.achieved_joint_positions_rad||ray.planned_joint_positions_rad;
  if(!joints||joints.length!==6)return [];
  let model=doc.robot_model.modified_dh;
  let transform=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  let points=[[0,0,0]];
  joints.forEach((joint,index)=>{
    transform=matrixMultiply(transform,dhTransform(model.alpha_rad[index],
      model.a_m[index],joint+model.theta_offset_rad[index],model.d_m[index]));
    points.push(transformPoint(transform));
  });
  return points;
}
function cameraPosition(ray){
  return ray.achieved_camera_position_m||ray.camera_position_m||
    ray.representative_position_m;
}
let viewYaw=-0.72,viewPitch=0.48,viewZoom=1,dragStart=null,playTimer=null;
function generationRays(){
  let selectedGeneration=doc.generations[Number(generation.value)]||
    doc.generations.at(-1);
  return selectedGeneration.rays.slice().sort((a,b)=>
    a.rank-b.rank||a.ray_id-b.ray_id);
}
function selectedSceneRay(){
  let rays=generationRays();
  return rays.find(ray=>String(ray.ray_id)===raySelect.value)||rays[0];
}
function projector(points){
  let centre=[0,1,2].map(axis=>(Math.min(...points.map(p=>p[axis]))+
    Math.max(...points.map(p=>p[axis])))/2);
  let span=Math.max(...[0,1,2].map(axis=>
    Math.max(...points.map(p=>p[axis]))-Math.min(...points.map(p=>p[axis]))),.2);
  let scale=520/span*viewZoom,cy=Math.cos(viewYaw),sy=Math.sin(viewYaw);
  let cp=Math.cos(viewPitch),sp=Math.sin(viewPitch);
  return point=>{
    let value=vecSub(point,centre);
    let x=cy*value[0]-sy*value[1],y=sy*value[0]+cy*value[1];
    return [560+x*scale,310-(cp*value[2]-sp*y)*scale,
      sp*value[2]+cp*y];
  };
}
function canvasLine(context,project,a,b,color,width=1,alpha=1){
  let first=project(a),second=project(b);
  context.globalAlpha=alpha;context.strokeStyle=color;context.lineWidth=width;
  context.beginPath();context.moveTo(first[0],first[1]);
  context.lineTo(second[0],second[1]);context.stroke();context.globalAlpha=1;
}
function cameraGlyph(context,project,position,target,color,size){
  let forward=vecNorm(vecSub(target,position));
  let right=vecNorm(cross(forward,Math.abs(forward[2])>.9?[0,1,0]:[0,0,1]));
  let up=vecNorm(cross(right,forward));
  let centre=vecAdd(position,vecScale(forward,size));
  let corners=[[-1,-.7],[1,-.7],[1,.7],[-1,.7]].map(pair=>
    vecAdd(centre,vecAdd(vecScale(right,pair[0]*size*.7),
      vecScale(up,pair[1]*size*.7))));
  corners.forEach(corner=>canvasLine(context,project,position,corner,color,1.2,.85));
  corners.forEach((corner,index)=>canvasLine(context,project,corner,
    corners[(index+1)%4],color,1.2,.85));
}
function drawScene(){
  let canvas=scene3d,context=canvas.getContext('2d'),rays=generationRays();
  let ray=selectedSceneRay();
  context.clearRect(0,0,canvas.width,canvas.height);
  if(!ray){poseInfo.textContent='No ray is available';return}
  let generationItem=doc.generations[Number(generation.value)]||doc.generations.at(-1);
  let target=generationItem.target_center_m,arm=urdfPoints(ray);
  let points=[target,[0,0,0],...rays.map(cameraPosition),...arm];
  let project=projector(points),current=state(ray),camera=cameraPosition(ray);
  canvasLine(context,project,[-.35,0,0],[.55,0,0],'#7c8794',1,.55);
  canvasLine(context,project,[0,-.35,0],[0,.55,0],'#7c8794',1,.55);
  if(allCameras.checked)rays.forEach(item=>{
    let position=cameraPosition(item),itemState=state(item);
    canvasLine(context,project,position,target,colors[itemState[1]],1,.18);
    cameraGlyph(context,project,position,target,colors[itemState[1]],.012);
  });
  canvasLine(context,project,camera,target,colors[current[1]],3,.9);
  cameraGlyph(context,project,camera,target,'#f5f8fb',.035);
  if(arm.length){
    arm.slice(1).forEach((item,index)=>
      canvasLine(context,project,arm[index],item,'#b8c8d8',10,1));
    arm.forEach(item=>{
      let screen=project(item);context.fillStyle='#eaf3fb';context.beginPath();
      context.arc(screen[0],screen[1],5,0,Math.PI*2);context.fill();
    });
    canvasLine(context,project,arm.at(-1),camera,'#68c7ef',6,.9);
  }
  let targetScreen=project(target);context.fillStyle='#fff';context.beginPath();
  context.arc(targetScreen[0],targetScreen[1],8,0,Math.PI*2);context.fill();
  let poseSource=ray.robot_pose_source||'unavailable: ray has no valid IK endpoint';
  poseInfo.textContent=`ray ${ray.ray_id} · rank ${ray.rank} · ${current[0]} / `+
    `${current[1]} · arm ${poseSource} · L515 ${camera.map(v=>v.toFixed(3)).join(', ')} m`;
}
function populateRaySelector(){
  let previous=raySelect.value,rays=generationRays();raySelect.replaceChildren();
  rays.forEach(ray=>raySelect.add(new Option(
    `rank ${ray.rank} · ray ${ray.ray_id} · ${state(ray)[0]}`,ray.ray_id)));
  raySelect.value=rays.some(ray=>String(ray.ray_id)===previous)?previous:
    String((rays.find(ray=>state(ray)[0]==='selected')||rays[0]||{}).ray_id||'');
}
function stepRay(direction){
  let options=[...raySelect.options],index=Math.max(0,raySelect.selectedIndex);
  if(options.length)raySelect.selectedIndex=(index+direction+options.length)%options.length;
  drawScene();
}
function radial(node,firstAxis,secondAxis,rays){
  basePlot(node,480,480,['−1','−0.5','0','+0.5','+1'],
    ['+1','+0.5','0','−0.5','−1']);
  let centerX=253.5,centerY=217.5,scale=162.5;
  node.append(svg('circle',{cx:centerX,cy:centerY,r:7,fill:'#fff'}));
  node.append(svg('text',{x:centerX+10,y:centerY-10,fill:'#fff',
    'font-size':11},'target'));
  rays.forEach(r=>{
    let direction=r.direction,norm=Math.hypot(...direction)||1;
    let x=centerX+direction[firstAxis]/norm*scale;
    let y=centerY-direction[secondAxis]/norm*scale;
    node.append(svg('line',{x1:centerX,y1:centerY,x2:x,y2:y,
      stroke:colors[state(r)[1]],'stroke-opacity':.22}));
    point(node,x,y,r,3.5);
  });
}
function escaped(value){
  return String(value).replaceAll('&','&amp;').replaceAll('<','&lt;');
}
function render(){
  let selectedGeneration=doc.generations[Number(generation.value)]||
    doc.generations.at(-1);
  let statusFilter=status.value,query=search.value.toLowerCase();
  let rays=selectedGeneration.rays.filter(r=>
    (statusFilter==='all'||state(r)[0]===statusFilter)&&
    (!query||String(r.ray_id).includes(query)||
      reasons(r).toLowerCase().includes(query)));
  let counts={generated:selectedGeneration.generated_ray_count||
    selectedGeneration.rays.length,remaining:0,
    culled:0,selected:0};
  selectedGeneration.rays.forEach(r=>counts[state(r)[0]]++);
  cards.innerHTML=Object.entries(counts).map(([key,value])=>
    `<div class="card"><span class="muted">${key}</span><b>${value}</b></div>`
  ).join('')+`<div class="card"><span class="muted">policy</span>`+
    `<b style="font-size:14px">${selectedGeneration.view_selection_policy}</b></div>`;
  let replay=selectedGeneration.historical_replay;
  replayNotice.className=replay?'notice':'';
  replayNotice.textContent=replay?replay.limitation:'';
  basePlot(sphere,720,380,['0°','90°','180°','270°','360°'],
    ['+90°','+45°','0°','−45°','−90°']);
  rays.forEach(r=>point(sphere,42+r.azimuth_deg/360*663,
    15+(90-r.elevation_deg)/180*325,r));
  radial(top,0,1,rays);radial(side,0,2,rays);
  rows.innerHTML=rays.slice().sort((a,b)=>a.rank-b.rank||a.ray_id-b.ray_id)
    .map(r=>{
      let current=state(r);
      return `<tr data-ray="${r.ray_id}"><td>${r.rank} `+
        `<span class="muted">${r.rank_kind}</span></td>`+
        `<td>${r.ray_id}</td><td style="color:${colors[current[1]]}">`+
        `${current[0]} / ${current[1]}</td><td>${r.azimuth_deg.toFixed(1)}°</td>`+
        `<td>${r.elevation_deg.toFixed(1)}°</td>`+
        `<td>${Number(r.nbv_marginal_information_fraction||0).toFixed(4)}</td>`+
        `<td>${r.nbv_novel_surface_pixels||0}</td>`+
        `<td class="reason">${escaped(reasons(r))}</td></tr>`;
    }).join('');
  [...rows.children].forEach(row=>row.onclick=()=>{
    raySelect.value=row.dataset.ray;drawScene();
    scene3d.scrollIntoView({behavior:'smooth',block:'center'});
  });
  populateRaySelector();drawScene();
}
doc.generations.forEach((item,index)=>generation.add(new Option(
  `${item.session_id} · ${item.generation} accepted`,index)));
generation.value=doc.generations.length-1;
legend.innerHTML=Object.entries(colors).map(([key,value])=>
  `<span><i class="dot" style="background:${value}"></i>${key}</span>`).join('');
generation.onchange=status.onchange=search.oninput=render;
raySelect.onchange=allCameras.onchange=drawScene;
previousRay.onclick=()=>stepRay(-1);nextRay.onclick=()=>stepRay(1);
playRays.onclick=()=>{
  if(playTimer){clearInterval(playTimer);playTimer=null;playRays.textContent='Play'}
  else{playTimer=setInterval(()=>stepRay(1),700);playRays.textContent='Pause'}
};
resetView.onclick=()=>{viewYaw=-.72;viewPitch=.48;viewZoom=1;drawScene()};
scene3d.onmousedown=event=>dragStart=[event.clientX,event.clientY,viewYaw,viewPitch];
window.onmouseup=()=>dragStart=null;
window.onmousemove=event=>{
  if(!dragStart)return;viewYaw=dragStart[2]+(event.clientX-dragStart[0])*.008;
  viewPitch=Math.max(-1.4,Math.min(1.4,
    dragStart[3]+(event.clientY-dragStart[1])*.008));drawScene();
};
scene3d.onwheel=event=>{
  event.preventDefault();viewZoom=Math.max(.35,Math.min(4,
    viewZoom*(event.deltaY>0?.9:1.1)));drawScene();
};
render();
</script></body></html>"""
    return template.replace('__TITLE__', html.escape(title)).replace(
        '__DATA__', encoded)
