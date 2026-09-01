#!/usr/bin/env python3
"""Replay frozen planner requests through Tesseract and cuRobo safely.

The tool starts only ROS-free planner workers with private spools.  It has no
ROS, CAN, motor, controller, action, service, or joint-command interface.
"""

import argparse
import contextlib
import csv
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(
    ROOT / 'piper_ros_foxy/src/piper_tesseract_foxy'))

from motion_planning.benchmarking import (  # noqa: E402
    canonical_bytes, materialize_request, scenario_sha256,
    summarize_trials, trajectory_metrics)
from piper_tesseract_foxy.protocol.contract import (  # noqa: E402
    validate_request, validate_response)


BACKENDS = ('tesseract', 'curobo')


def read_json(path):
    with open(path, 'r', encoding='utf-8') as stream:
        return json.load(stream)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name('.%s.%d.tmp' % (path.name, os.getpid()))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(temporary), flags, 0o600)
    try:
        data = canonical_bytes(value)
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError('atomic benchmark write made no progress')
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(str(temporary), str(path))
    return path


def git_primary_root():
    try:
        completed = subprocess.run(
            ['git', '-C', str(ROOT), 'rev-parse', '--path-format=absolute',
             '--git-common-dir'], check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, timeout=5.0)
        common = Path(completed.stdout.strip()).resolve()
        if common.name == '.git':
            return common.parent
    except (OSError, subprocess.SubprocessError):
        pass
    return ROOT


def find_tesseract_runtime(explicit):
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    configured = os.environ.get('PIPER_TESSERACT_BENCHMARK_RUNTIME', '')
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend([
        ROOT / 'motion_planning/tesseract/.runtime',
        git_primary_root() / 'motion_planning/tesseract/.runtime',
    ])
    for candidate in candidates:
        root = candidate.resolve()
        if (root / 'rootfs/opt/tesseract/bin/python').exists():
            return root
    raise RuntimeError(
        'Tesseract rootless runtime is missing; provide '
        '--tesseract-runtime or run setup_rootless_worker.sh')


def process_log_tail(path, lines=30):
    try:
        values = path.read_text(
            encoding='utf-8', errors='replace').splitlines()
    except OSError:
        return ''
    return '\n'.join(values[-lines:])


def terminate_owned_process(process, timeout_sec=10.0):
    if process.poll() is not None:
        return process.returncode
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return process.poll()
    try:
        return process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait(timeout=5.0)


def wait_for_health(process, spool, timeout_sec, log_path):
    deadline = time.monotonic() + float(timeout_sec)
    path = spool / 'worker_health.json'
    last_error = 'health file not written'
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                'planner worker exited during startup with code %s\n%s'
                % (process.returncode, process_log_tail(log_path)))
        try:
            health = read_json(path)
            if health.get('worker_ready') is True:
                return health
            last_error = str(health.get('backend_error') or 'worker not ready')
            if health.get('backend_error'):
                raise RuntimeError(
                    'planner worker initialization failed: %s\n%s'
                    % (last_error, process_log_tail(log_path)))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            last_error = str(error)
        time.sleep(0.1)
    raise RuntimeError(
        'planner worker readiness timed out: %s\n%s'
        % (last_error, process_log_tail(log_path)))


@contextlib.contextmanager
def planner_worker(backend, work_root, args):
    runtime_root = work_root / ('%s_runtime' % backend)
    spool = runtime_root / ('%s_spool' % backend)
    spool.mkdir(parents=True, exist_ok=True, mode=0o700)
    log_path = work_root / ('%s_worker.log' % backend)
    environment = dict(os.environ)
    environment.update({
        'PIPER_ARM_ROOT': str(ROOT),
        'PIPER_FLOOR_PROFILE': args.floor_profile,
        'XDG_RUNTIME_DIR': str(runtime_root),
    })
    if backend == 'tesseract':
        source_runtime = find_tesseract_runtime(args.tesseract_runtime)
        private_runtime = spool / 'model_runtime'
        private_runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.symlink(
            str(source_runtime / 'rootfs'),
            str(private_runtime / 'rootfs'))
        environment.update({
            'PIPER_TESSERACT_RUNTIME': str(private_runtime),
            'PIPER_TESSERACT_URDF_CONTAINER': (
                '/spool/model_runtime/piper_planning.urdf'),
            'PIPER_TESSERACT_SPOOL': str(spool),
        })
        command = [str(ROOT / 'motion_planning/tesseract/run_worker.sh')]
    else:
        # Preserve a virtual-environment launcher path. Resolving its Python
        # symlink would bypass that environment's site-packages.
        curobo_python = Path(args.curobo_python).expanduser().absolute()
        if not curobo_python.is_file():
            raise RuntimeError(
                'cuRobo interpreter is missing: %s' % curobo_python)
        environment.update({
            'PIPER_CUROBO_PYTHON': str(curobo_python),
            'PIPER_CUROBO_SPOOL': str(spool),
        })
        if args.curobo_cuda_home:
            environment['PIPER_CUROBO_CUDA_HOME'] = str(
                Path(args.curobo_cuda_home).expanduser().resolve())
        command = [str(ROOT / 'motion_planning/curobo/run_worker.sh')]
    started = time.monotonic()
    with open(log_path, 'w', encoding='utf-8') as log_stream:
        process = subprocess.Popen(
            command, cwd=str(ROOT), env=environment,
            stdin=subprocess.DEVNULL, stdout=log_stream,
            stderr=subprocess.STDOUT, start_new_session=True)
        try:
            health = wait_for_health(
                process, spool, args.startup_timeout_sec, log_path)
            yield {
                'process': process,
                'spool': spool,
                'health': health,
                'startup_wall_sec': time.monotonic() - started,
                'log_path': log_path,
                'runtime_root': runtime_root,
            }
        finally:
            terminate_owned_process(process)


def wait_for_response(process, spool, request_id, timeout_sec, log_path):
    response_path = spool / 'responses' / ('%s.json' % request_id)
    deadline = time.monotonic() + float(timeout_sec)
    while time.monotonic() < deadline:
        if response_path.is_file():
            return read_json(response_path)
        if process.poll() is not None:
            raise RuntimeError(
                'planner worker exited while processing %s (code %s)\n%s'
                % (request_id, process.returncode,
                   process_log_tail(log_path)))
        time.sleep(0.02)
    raise TimeoutError('planner response timed out for %s' % request_id)


def response_artifact_path(raw_root, backend, fixture_name, run_index, kind):
    return raw_root / backend / fixture_name / ('%04d.%s.json' % (
        int(run_index), kind))


def execute_trial(
        worker, backend, fixture, run_index, warmup, args, raw_root):
    template = fixture['request_template']
    request = materialize_request(
        template, backend, run_index, ttl_sec=args.request_timeout_sec + 60.0)
    validate_request(request)
    if scenario_sha256(request) != fixture['scenario_sha256']:
        raise RuntimeError(
            'fixture scenario digest changed during materialization')
    request_path = worker['spool'] / 'requests' / (
        request['request_id'] + '.json')
    started = time.monotonic()
    atomic_json(request_path, request)
    try:
        response = wait_for_response(
            worker['process'], worker['spool'], request['request_id'],
            args.request_timeout_sec, worker['log_path'])
        wall = time.monotonic() - started
        validate_response(response, request)
        metrics = trajectory_metrics(response)
        status = str(response.get('status', 'failed'))
        diagnostic = str(response.get('diagnostic', ''))
        rejection_codes = list(response.get('rejection_codes', []))
    except TimeoutError as error:
        wall = time.monotonic() - started
        response = None
        metrics = {}
        status = 'timeout'
        diagnostic = str(error)
        rejection_codes = ['BENCHMARK_TIMEOUT']
    request_artifact = response_artifact_path(
        raw_root, backend, fixture['name'], run_index, 'request')
    atomic_json(request_artifact, request)
    response_artifact = ''
    if response is not None:
        response_path = response_artifact_path(
            raw_root, backend, fixture['name'], run_index, 'response')
        atomic_json(response_path, response)
        response_artifact = str(response_path)
    return {
        'backend': backend,
        'backend_version': str(worker['health'].get('backend_version', '')),
        'fixture': fixture['name'],
        'plan_kind': fixture['plan_kind'],
        'expected_role': fixture.get('expected_role', ''),
        'scenario_sha256': fixture['scenario_sha256'],
        'run_index': int(run_index),
        'warmup': bool(warmup),
        'request_id': request['request_id'],
        'request_sha256': request['request_sha256'],
        'status': status,
        'request_wall_sec': wall,
        'diagnostic': diagnostic,
        'rejection_codes': rejection_codes,
        'request_artifact': str(request_artifact),
        'response_artifact': response_artifact,
        'exact_collision_validation': (
            'pending' if status == 'success' else 'not_applicable'),
        **metrics,
    }


def selected_fixtures(corpus, names):
    fixtures = list(corpus.get('fixtures', []))
    if names:
        requested = set(names)
        fixtures = [
            item for item in fixtures if item.get('name') in requested]
        missing = requested - {item.get('name') for item in fixtures}
        if missing:
            raise ValueError(
                'unknown fixture names: %s' % ', '.join(sorted(missing)))
    if not fixtures:
        raise ValueError('benchmark corpus contains no selected fixtures')
    return fixtures


def run_backend(backend, fixtures, work_root, args, raw_root):
    rows = []
    with planner_worker(backend, work_root, args) as worker:
        run_index = 0
        warm_fixture = next(
            (item for item in fixtures
             if item.get('expected_role') != 'negative_control'), fixtures[0])
        for _index in range(args.warmups):
            rows.append(execute_trial(
                worker, backend, warm_fixture, run_index, True, args,
                raw_root))
            run_index += 1
        for _repeat in range(args.repetitions):
            for fixture in fixtures:
                rows.append(execute_trial(
                    worker, backend, fixture, run_index, False, args,
                    raw_root))
                run_index += 1
        metadata = {
            'backend': backend,
            'startup_wall_sec': worker['startup_wall_sec'],
            'health': worker['health'],
            'worker_log': str(worker['log_path']),
            'runtime_root': str(worker['runtime_root']),
        }
    return rows, metadata


def exact_validation(trials, workers, work_root):
    """Recheck every successful measured path with exact Tesseract geometry."""
    tesseract = next(
        (item for item in workers if item['backend'] == 'tesseract'), None)
    measured = [
        trial for trial in trials
        if not trial.get('warmup', False)
        and trial.get('status') == 'success'
        and trial.get('expected_role') == 'recorded_achieved_geometry'
    ]
    if not measured:
        return {
            'status': 'complete', 'reason': 'no successful path to validate',
            'result_count': 0,
        }
    if tesseract is None:
        return {
            'status': 'not_run',
            'reason': 'exact validation requires the Tesseract backend setup',
            'result_count': 0,
        }
    items = []
    for trial in measured:
        key = '%s|%s|%d' % (
            trial['backend'], trial['fixture'], trial['run_index'])
        trial['exact_validation_trial_key'] = key
        items.append({
            'trial_key': key,
            'request': read_json(Path(trial['request_artifact'])),
            'response': read_json(Path(trial['response_artifact'])),
        })
    validation_spool = work_root / 'exact_validation_spool'
    validation_spool.mkdir(parents=True, exist_ok=True, mode=0o700)
    source_model = (
        Path(tesseract['runtime_root'])
        / 'tesseract_spool/model_runtime/piper_planning.urdf')
    validation_model = validation_spool / 'model_runtime'
    validation_model.mkdir(parents=True, exist_ok=True, mode=0o700)
    shutil.copy2(
        str(source_model), str(validation_model / 'piper_planning.urdf'))
    atomic_json(validation_spool / 'validation_input.json', {
        'schema_version': 1,
        'real_arm_motion': False,
        'items': items,
    })
    environment = dict(os.environ)
    environment.update({
        'PIPER_ARM_ROOT': str(ROOT),
        'PIPER_TESSERACT_RUNTIME': str(
            Path(tesseract['runtime_root'])
            / 'tesseract_spool/model_runtime'),
        'PIPER_TESSERACT_URDF_CONTAINER': (
            '/spool/model_runtime/piper_planning.urdf'),
        'PIPER_TESSERACT_SPOOL': str(validation_spool),
        'XDG_RUNTIME_DIR': str(Path(tesseract['runtime_root'])),
    })
    completed = subprocess.run(
        [str(ROOT / (
            'motion_planning/tesseract/validate_benchmark_responses.sh'))],
        cwd=str(ROOT), env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, timeout=600.0)
    output_path = validation_spool / 'validation_output.json'
    if not output_path.is_file():
        raise RuntimeError(
            'exact benchmark validation did not produce a report '
            '(code %d):\n%s'
            % (completed.returncode, completed.stdout[-4000:]))
    report = read_json(output_path)
    by_key = {item['trial_key']: item for item in report['results']}
    for trial in measured:
        validation = by_key[trial['exact_validation_trial_key']]
        trial['exact_collision_validation'] = validation['status']
        trial['exact_validation_reason'] = validation['reason']
        trial['exact_validation_wall_sec'] = validation[
            'validation_wall_sec']
        trial['exact_validation_segments'] = validation['segment_reports']
    destination = work_root.parent / 'exact_collision_validation.json'
    atomic_json(destination, report)
    return {
        'status': 'complete',
        'validator_exit_code': completed.returncode,
        'result_count': len(report['results']),
        'passed': sum(
            item['status'] == 'passed' for item in report['results']),
        'failed': sum(
            item['status'] == 'failed' for item in report['results']),
        'report_path': str(destination),
    }


def write_csv(path, trials):
    scalar_fields = [
        'backend', 'backend_version', 'fixture', 'plan_kind', 'expected_role',
        'scenario_sha256', 'run_index', 'warmup', 'status',
        'request_wall_sec', 'backend_reported_planning_sec',
        'trajectory_duration_sec', 'trajectory_point_count',
        'joint_space_path_length_rad', 'maximum_joint_step_rad',
        'selected_viewpoint_count', 'candidate_viewpoints_considered',
        'candidate_viewpoints_rejected', 'feasible_viewpoints',
        'exact_collision_validation', 'diagnostic',
    ]
    with open(path, 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields)
        writer.writeheader()
        for trial in trials:
            writer.writerow({
                field: trial.get(field, '') for field in scalar_fields})


def make_report_portable(output, trials, workers, validation):
    """Retain evidence while removing the private executable runtime."""
    logs = output / 'logs'
    logs.mkdir(mode=0o700)
    for worker in workers:
        source = Path(worker['worker_log'])
        destination = logs / ('%s_worker.log' % worker['backend'])
        shutil.copy2(str(source), str(destination))
        worker['worker_log'] = str(destination.relative_to(output))
        worker.pop('runtime_root', None)
        worker['private_runtime_removed_after_success'] = True
    for trial in trials:
        for field in ('request_artifact', 'response_artifact'):
            if trial.get(field):
                trial[field] = str(
                    Path(trial[field]).resolve().relative_to(output))
    if validation.get('report_path'):
        validation['report_path'] = str(
            Path(validation['report_path']).resolve().relative_to(output))


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--corpus', type=Path,
        default=ROOT / (
            'benchmarks/planner_backends/recorded_reference_corpus.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', action='append', choices=BACKENDS)
    parser.add_argument('--fixture', action='append')
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--request-timeout-sec', type=float, default=190.0)
    parser.add_argument('--startup-timeout-sec', type=float, default=300.0)
    parser.add_argument('--floor-profile', choices=('tabletop', 'ground'),
                        default='tabletop')
    parser.add_argument('--tesseract-runtime', default='')
    parser.add_argument(
        '--curobo-python', default=os.environ.get(
            'PIPER_CUROBO_PYTHON',
            '/home/prl/.venvs/piper-curobo-v0.7.8/bin/python'))
    parser.add_argument(
        '--curobo-cuda-home', default=os.environ.get(
            'PIPER_CUROBO_CUDA_HOME', '/home/prl/.local/cuda-12.8'))
    return parser.parse_args()


def main():
    args = arguments()
    if args.repetitions < 1 or args.warmups < 0:
        raise SystemExit('repetitions must be >=1 and warmups must be >=0')
    corpus = read_json(args.corpus.resolve())
    fixtures = selected_fixtures(corpus, args.fixture)
    backends = tuple(args.backend or BACKENDS)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(
            'output directory must be absent or empty: %s' % output)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw_root = output / 'raw'
    work_root = output / 'runtime'
    raw_root.mkdir(mode=0o700)
    work_root.mkdir(mode=0o700)
    trials = []
    workers = []
    started = time.monotonic()
    for backend in backends:
        rows, metadata = run_backend(
            backend, fixtures, work_root, args, raw_root)
        trials.extend(rows)
        workers.append(metadata)
    validation = exact_validation(trials, workers, work_root)
    make_report_portable(output, trials, workers, validation)
    report = {
        'schema_version': 1,
        'comparison_strength': 'CONTROLLED_REPLAY',
        'real_arm_motion': False,
        'physical_result_claimed': False,
        'corpus_path': os.path.relpath(
            str(args.corpus.resolve()), str(output)),
        'corpus_sha256': corpus.get('corpus_sha256', ''),
        'floor_profile': args.floor_profile,
        'repetitions': args.repetitions,
        'warmups': args.warmups,
        'total_benchmark_wall_sec': time.monotonic() - started,
        'workers': workers,
        'trials': trials,
        'summary': summarize_trials(trials),
        'exact_validation': validation,
    }
    report_path = output / 'planner_benchmark.json'
    atomic_json(report_path, report)
    write_csv(output / 'planner_benchmark.csv', trials)
    shutil.rmtree(str(work_root))
    print(report_path)
    print(json.dumps(report['summary'], indent=2, sort_keys=True))
    print('real_arm_motion=false physical_result_claimed=false')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
