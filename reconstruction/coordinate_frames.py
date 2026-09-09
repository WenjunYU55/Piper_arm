"""Robot-base orientation for offline geometry; never infer axes from object shape."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np


BUNDLESDF_PIN = 'ffa67d425240b5b76d2e387a7dd3d3735a7cf1a1'
LEGEND = 'base_link | +Z UP (blue), +X red, +Y green | magenta: toward robot (XY)'
GEOMETRY_KEYS = ('mesh_path', 'raw_mesh_path', 'input_cloud_path',
                 'measured_cloud_path', 'consensus_cloud_path', 'textured_mesh_path')


def file_hash(path):
    """Hash original bytes without loading large clouds into memory."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rigid_transform(value):
    """Reject reflected, scaled or invalid transforms rather than guessing axes."""
    transform = np.asarray(value, dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1., atol=1e-4)):
        raise ValueError('A finite right-handed rigid transform is required')
    return transform


def base_frame():
    """Describe robot-relative axes without claiming gravity or metrology accuracy."""
    return dict(schema_version=1, frame_id='base_link', units='metres',
                up_axis='+Z (robot base up; not independently measured gravity)',
                axes=dict(X='red', Y='green', Z='blue'),
                robot_base_origin=[0., 0., 0.],
                robot_direction='from object toward base origin projected into base XY',
                accuracy_qualified=False)


def report_frame(report):
    """Recognize current and retained PiPER reports; arbitrary models stay unknown."""
    frame = report.get('coordinate_frame')
    if frame:
        if frame.get('frame_id') != 'base_link' or frame.get('units') != 'metres':
            raise ValueError('Report is not in base_link metres')
        return frame
    frames = report.get('frame_inputs', [])
    if not report.get('scan_dir') or not frames:
        raise ValueError('Coordinate frame unverified: no recorded base-camera poses')
    for entry in frames:
        rigid_transform(entry['T_base_camera'])
        name = entry.get('base_frame')
        if name is None:
            # Older reports omitted the parent frame ID; consult retained capture metadata.
            import yaml
            metadata = Path(report['scan_dir']) / 'frames' / entry.get('frame', '')
            value = yaml.safe_load(metadata.read_text())
            name = value.get('camera_transform', {}).get('header', {}).get('frame_id')
        if name != 'base_link':
            raise ValueError('Coordinate frame unverified: capture parent is not base_link')
    return dict(base_frame(), basis='legacy PiPER report with recorded T_base_camera')


def write_report_frames(report):
    """Attach frame/hash sidecars to every newly generated mesh and cloud variant."""
    try:
        frame = report_frame(report)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        frame = dict(frame_id='unverified', units='metres', reason=str(exc))
    report['coordinate_frame'] = frame
    for key in GEOMETRY_KEYS:
        if report.get(key):
            path = Path(report[key])
            data = dict(frame, geometry=path.name, geometry_sha256=file_hash(path),
                        basis='capture-time robot poses with selected offline refinement')
            sidecar = path.with_name(path.name + '.frame.json')
            temporary = sidecar.with_name('.' + sidecar.name + '.tmp')
            temporary.write_text(json.dumps(data, indent=2) + '\n')
            temporary.replace(sidecar)


def bundlesdf_frame(root):
    """Anchor the pinned model gauge using the SAME first image's two transforms."""
    root = Path(root)
    run_path = root / 'run.json'
    if json.loads(run_path.read_text()).get('upstream_commit') != BUNDLESDF_PIN:
        raise ValueError('BundleSDF model frame unverified: unsupported upstream revision')
    poses_path = root / 'input/robot_camera_poses.json'
    poses = json.loads(poses_path.read_text())
    matches = [f for f in poses.get('frames', []) if f.get('filename') == '00000000.png']
    if len(matches) != 1 or 'T_base_camera' not in poses.get('pose_convention', ''):
        raise ValueError('Missing or ambiguous first-image base-camera correspondence')
    first = matches[0]
    if not isinstance(first.get('rgb_stamp_ns'), int) or first['rgb_stamp_ns'] <= 0:
        raise ValueError('Missing first-image timestamp')
    source_files = [run_path, poses_path]
    parent = poses.get('base_frame')
    if parent is None:
        validation_path = root / 'input/validation.json'
        validation = json.loads(validation_path.read_text())
        source_files.append(validation_path)
        sessions = [Path(validation['input']), root / 'native-session.incomplete']
        for session in sessions:
            path = session / ('%08d' % first['native_frame']) / 'frame.json'
            if not path.is_file():
                continue
            entry = json.loads(path.read_text())
            if (entry.get('rgb_stamp_ns') != first['rgb_stamp_ns']
                    or not np.allclose(rigid_transform(entry['T_base_camera']),
                                       rigid_transform(first['T_base_camera']), atol=1e-8)):
                raise ValueError('Original frame differs from converted pose correspondence')
            parent = entry.get('base_frame')
            source_files.append(path)
            break
    if parent != 'base_link':
        raise ValueError('Coordinate frame unverified: recorded parent is not base_link')
    pose_path = root / 'result/ob_in_cam/00000000.txt'
    camera_object = rigid_transform(np.loadtxt(str(pose_path)))
    base_camera = rigid_transform(first['T_base_camera'])
    # The pinned upstream fixes the initial gauge, including its final NeRF pass.
    # Do not average later poses: that would hide tracking/hand-eye disagreement.
    transform = rigid_transform(base_camera @ camera_object)
    return dict(base_frame(), source_frame='bundlesdf_object',
                T_base_object=transform.tolist(),
                formula='T_base_object = T_base_camera[0] @ T_camera_object[0]',
                anchor=dict(filename=first['filename'], native_frame=first.get('native_frame'),
                            rgb_stamp_ns=first['rgb_stamp_ns']),
                basis='first recorded image; pinned BundleSDF initial-frame gauge',
                sources=[dict(path=str(p.resolve()), sha256=file_hash(p))
                         for p in source_files + [pose_path]])


def view_layout(points):
    """Choose a base-Z-up view from the robot side; provide nearby orientation arrows."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Nonempty finite geometry is required for orientation')
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    span = max(float(np.linalg.norm(high - low)), .01)
    toward = np.array([-center[0], -center[1], 0.])
    has_direction = np.linalg.norm(toward) > 1e-6
    horizontal = toward / np.linalg.norm(toward) if has_direction else np.array([-1., 0., 0.])
    # A slight side view keeps the toward-robot arrow legible rather than end-on.
    side = np.cross([0., 0., 1.], horizontal)
    front = horizontal * np.cos(np.pi / 6) + side * np.sin(np.pi / 6) + np.array([0., 0., .55])
    front /= np.linalg.norm(front)
    right = np.cross([0., 0., 1.], front)
    right /= np.linalg.norm(right)
    up = np.cross(front, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((right, up, front))
    pose[:3, 3] = center + front * span * 2.2
    origin = center - right * span * .65 - up * span * .35
    arrows = [(origin, axis * span * .30, color)
              for axis, color in zip(np.eye(3), ([1., 0., 0.], [0., 1., 0.], [0., .3, 1.]))]
    if has_direction:
        arrows.append((center + up * span * .65, horizontal * span * .35, [1., 0., 1.]))
    return dict(center=center, front=front, up=up, camera_pose=pose, arrows=arrows,
                span=span, legend=LEGEND if has_direction else
                LEGEND.split(' | magenta')[0] + ' | robot direction undefined on base Z axis')


def arrow_transform(origin, vector):
    """Place a renderer's +Z arrow along a base-frame vector."""
    z = np.asarray(vector) / np.linalg.norm(vector)
    guide = np.array([1., 0., 0.]) if abs(z[0]) < .9 else np.array([0., 1., 0.])
    x = np.cross(guide, z)
    x /= np.linalg.norm(x)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((x, np.cross(z, x), z))
    pose[:3, 3] = origin
    return pose


def export_bundlesdf(root, output):
    """Create a separate base-frame OBJ with original UVs/materials; never reconstruct."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite orientation export: ' + str(output))
    frame = bundlesdf_frame(root)
    transform = np.asarray(frame['T_base_object'])
    source = root / 'result/textured_mesh.obj'
    frame['source_geometry'] = dict(path=str(source), sha256=file_hash(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + output.name + '-', dir=str(output.parent)))
    try:
        target = temporary / source.name
        materials = []
        with source.open() as inp, target.open('w') as out:
            out.write('# PiPER base_link; metres; +Z robot-up; robot base origin (0,0,0)\n')
            for line in inp:
                parts = line.split()
                if parts and parts[0] == 'mtllib':
                    materials.extend(parts[1:])
                if parts and parts[0] in ('v', 'vn'):
                    xyz = np.asarray([float(v) for v in parts[1:4]])
                    if xyz.shape != (3,) or not np.isfinite(xyz).all():
                        raise ValueError('Invalid OBJ geometry')
                    xyz = transform[:3, :3] @ xyz
                    if parts[0] == 'v':
                        xyz += transform[:3, 3]
                    line = ' '.join([parts[0]] + ['%.17g' % v for v in xyz] + parts[4:]) + '\n'
                out.write(line)

        def copy_asset(path):
            resolved = path.resolve()
            if source.parent not in resolved.parents or not resolved.is_file():
                raise ValueError('Material/texture must be retained inside the result folder')
            destination = temporary / resolved.relative_to(source.parent)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(resolved), str(destination))
        if not materials:
            raise ValueError('Textured mesh has no material library')
        for name in materials:
            material = source.parent / name
            copy_asset(material)
            for line in material.read_text().splitlines():
                parts = line.split()
                if parts and parts[0].lower().startswith('map_'):
                    copy_asset(material.parent / parts[-1])
        frame['geometry'] = source.name
        frame['geometry_sha256'] = file_hash(target)
        (temporary / (source.name + '.frame.json')).write_text(json.dumps(frame, indent=2) + '\n')
        (temporary / 'README.md').write_text(
            '# Robot-base oriented reconstruction\n\n'
            'Coordinates: base_link, metres. +Z is robot-base up; X/Y/Z use red/green/blue. '
            'The robot base is at (0,0,0); the robot-facing direction is toward that origin '
            'in the XY plane. Configure other viewers for Z-up; some default to Y-up.\n\n'
            'This is one rigid transform of the native mesh, not a new reconstruction '
            'or accuracy correction. UVs, faces and textures are retained. The .frame.json '
            'records the exact source hashes, first-image correspondence and transform.\n')
        temporary.rename(output)
    except Exception:
        shutil.rmtree(str(temporary))
        raise
    return frame


def export_report(report_path, output):
    """Copy all retained report variants with explicit frame sidecars, without refusion."""
    report_path, output = Path(report_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite orientation export: ' + str(output))
    report = json.loads(report_path.read_text())
    report_frame(report)
    report = copy.deepcopy(report)
    reports = [report] + list((report.get('candidate_reports') or {}).values())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + output.name + '-', dir=str(output.parent)))
    try:
        for item in reports:
            report_frame(item)
            for key in GEOMETRY_KEYS + ('texture_material_path', 'texture_atlas_path'):
                if not item.get(key):
                    continue
                source = Path(item[key]).resolve()
                target = temporary / source.name
                if target.exists() and file_hash(target) != file_hash(source):
                    raise ValueError('Conflicting exported filenames: ' + source.name)
                if not target.exists():
                    shutil.copyfile(str(source), str(target))
                item[key] = str(target)
            write_report_frames(item)
            for key in GEOMETRY_KEYS + ('texture_material_path', 'texture_atlas_path'):
                if item.get(key):
                    item[key] = str(output / Path(item[key]).name)
        report['orientation_export_source'] = dict(path=str(report_path), sha256=file_hash(report_path))
        (temporary / report_path.name).write_text(json.dumps(report, indent=2) + '\n')
        (temporary / 'README.md').write_text(
            '# Robot-base oriented reconstruction\n\n'
            'All included meshes and clouds already use base_link coordinates in metres. '
            '+Z is robot-base up. The robot is at the origin; toward-robot is the direction '
            'from the object toward (0,0) in base XY. Set other viewers to Z-up. '
            'Per-file .frame.json sidecars identify axes and geometry hashes. '
            'Copied geometry and materials are byte-identical to their sources. '
            'No reconstruction or alignment was rerun; quality conclusions are unchanged.\n')
        temporary.rename(output)
    except Exception:
        shutil.rmtree(str(temporary))
        raise
    return report


def main():
    """Export retained BundleSDF geometry without modifying the original result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', action='store_true', help='Input is a PiPER quality report')
    args = parser.parse_args()
    if args.report:
        export_report(args.run, args.output)
    else:
        export_bundlesdf(args.run, args.output)
    print(args.output)


if __name__ == '__main__':
    main()
