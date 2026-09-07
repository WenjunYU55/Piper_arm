#!/usr/bin/env python3
"""Run a GUI-selected archive or native session through the offline BundleSDF workflow."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess


def write_status(root, **values):
    """Publish small atomic progress records for a responsive GUI."""
    temporary = root / '.job.json.tmp'
    temporary.write_text(json.dumps(values, indent=2) + '\n')
    os.replace(str(temporary), str(root / 'job.json'))


def run_job(source, output, job_dir, allow_incomplete=False, stride=1, validate_only=False):
    """Keep recovery, validation and GPU execution off the GUI thread."""
    source, output, job_dir = [Path(p).resolve() for p in (source, output, job_dir)]
    if output.exists() or job_dir.exists():
        raise ValueError('Refusing existing output/job directory')
    if stride < 1:
        raise ValueError('Frame stride must be positive')
    job_dir.mkdir(parents=True)
    project = Path(__file__).resolve().parents[2]
    scripts = project / 'tools/bundlesdf'

    def stage(name, command):
        write_status(job_dir, state='running', stage=name, output=str(output))
        print(name, flush=True)
        with (job_dir / (name + '.log')).open('w') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError('%s failed; inspect %s' % (name, job_dir / (name + '.log')))

    try:
        bag = source / 'bag' if (source / 'recording.json').is_file() else source
        if (bag / 'metadata.yaml').is_file() and (bag.parent / 'recording.json').is_file():
            extraction, prepared = job_dir / 'extracted', job_dir / 'prepared-session'
            # The shell text is fixed; filesystem arguments are separate literals.
            command = [
                'bash', '-c', 'source "$1" && exec /usr/bin/python3 "$2" "$3" "$4"',
                'bundlesdf-bag-import', str(project / 'piper_ros_foxy/install/setup.bash'),
                str(scripts / 'extract_video.py'), str(bag), str(extraction)]
            stage('extract_rgbd_video', command)
            extraction_report = json.loads((extraction / 'extraction.json').read_text())
            if not extraction_report['complete'] and not allow_incomplete:
                raise ValueError('Incomplete mission; select Allow partial scan after inspecting the archive')
            stage('track_target_masks', [
                str(scripts / 'python_sam2.sh'), str(scripts / 'segment_video.py'),
                str(extraction), str(prepared)])
            source = prepared
            recovery_report = dict(
                evidence_class='RECORDED_RGBD_DENSE_OFFLINE_SAM2', source_bag=str(bag),
                extracted_frames=len(extraction_report['frames']),
                prepared_session=str(source),
                note='Every synchronized frame after accepted seed; temporal SAM2 before quality selection')
        elif not (source / 'manifest.json').is_file():
            raise ValueError('Select a raw reconstruction session or whole-scan recording; '
                             'an RGB video or stationary median-depth dataset is insufficient')
        manifest = json.loads((source / 'manifest.json').read_text())
        if manifest.get('source_bag') and 'recovery_report' not in locals():
            recovery_report = dict(evidence_class='RECORDED_RGBD_DENSE_OFFLINE_SAM2',
                                   source_bag=manifest['source_bag'], prepared_session=str(source))
        if len(manifest.get('frames', [])[::stride]) < 5:
            raise ValueError('Selected frame stride leaves fewer than five usable frames')
        command = ['/usr/bin/python3', str(scripts / 'run.py'), str(source),
                   '--output', str(output), '--frame-stride', str(stride)]
        if allow_incomplete:
            command.append('--allow-incomplete')
        stage('validate_input', command + ['--validate-only'])
        if not validate_only:
            stage('reconstruct', command)
            if 'recovery_report' in locals():
                (output / 'recovery.json').write_text(json.dumps(recovery_report, indent=2) + '\n')
        write_status(job_dir, state='succeeded', stage='validated' if validate_only else 'finished',
                     output=str(output), input=str(source), validate_only=validate_only)
    except BaseException as error:
        state = 'cancelled' if isinstance(error, KeyboardInterrupt) else 'failed'
        write_status(job_dir, state=state, error=str(error), output=str(output))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--job-dir', type=Path, required=True)
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()

    def stop(_number, _frame):
        raise KeyboardInterrupt('Offline reconstruction cancelled')

    signal.signal(signal.SIGTERM, stop)
    try:
        run_job(args.source, args.output, args.job_dir, args.allow_incomplete,
                args.frame_stride, args.validate_only)
    except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as error:
        parser.exit(2, str(error) + '\n')


if __name__ == '__main__':
    main()
