#!/usr/bin/env python3
"""Display an exported BundleSDF textured mesh using the isolated native environment."""

import argparse
from pathlib import Path

from coordinate_frames import (
    arrow_transform, bundlesdf_frame, file_hash, view_layout,
)


def main():
    """Load immutable output only; this process has no ROS or robot interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mesh', type=Path)
    parser.add_argument('--title', default='BundleSDF saved reconstruction')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    import json
    import numpy as np
    import trimesh

    mesh = trimesh.load(str(args.mesh), force='mesh')
    if not len(mesh.vertices) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError('Reconstruction mesh is empty or non-finite')
    if mesh.visual.kind != 'texture' or mesh.visual.material.image is None:
        raise ValueError('Texture missing: keep the OBJ, MTL and PNG in their output folder')
    orientation = None
    frame_file = args.mesh.with_name(args.mesh.name + '.frame.json')
    if frame_file.is_file():
        frame = json.loads(frame_file.read_text())
        if (frame.get('frame_id') != 'base_link' or frame.get('units') != 'metres'
                or frame.get('geometry_sha256') != file_hash(args.mesh)):
            raise ValueError('Invalid or mismatched coordinate-frame sidecar')
        orientation = view_layout(mesh.vertices)
    else:
        try:
            frame = bundlesdf_frame(args.mesh.resolve().parent.parent)
            mesh.apply_transform(np.asarray(frame['T_base_object']))
            orientation = view_layout(mesh.vertices)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print('Base orientation unavailable: ' + str(exc), flush=True)
    title = args.title + ' | ' + (orientation['legend'] if orientation else
                                  'ORIENTATION UNVERIFIED (native model frame)')
    print('Loaded %d vertices and %d faces' % (len(mesh.vertices), len(mesh.faces)), flush=True)
    print(title, flush=True)
    if args.check_only:
        return
    import pyrender

    scene = pyrender.Scene(ambient_light=[1., 1., 1.], bg_color=[.12, .12, .12, 1.])
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    flags = {}
    if orientation:
        from PIL import Image, ImageDraw, ImageFont
        try:
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 32)
        except OSError:
            font = ImageFont.load_default()
        for index, (origin, vector, color) in enumerate(orientation['arrows']):
            length = np.linalg.norm(vector)
            shaft = trimesh.creation.cylinder(radius=length * .025, height=length * .75)
            shaft.apply_translation([0, 0, length * .375])
            tip = trimesh.creation.cone(radius=length * .065, height=length * .25)
            tip.apply_translation([0, 0, length * .75])
            arrow = trimesh.util.concatenate([shaft, tip])
            arrow.visual.vertex_colors = np.array(color + [1.]) * 255
            arrow.apply_transform(arrow_transform(origin, vector))
            scene.add(pyrender.Mesh.from_trimesh(arrow))
            # Textured billboards avoid the pinned pyrender font renderer's empty-glyph bug.
            label = Image.new('RGBA', (256, 64), (0, 0, 0, 0))
            ImageDraw.Draw(label).text((2, 0), ('X', 'Y', 'Z UP', 'ROBOT')[index],
                                      fill=tuple(int(c * 255) for c in color) + (255,), font=font)
            label = label.crop(label.getbbox())
            height = orientation['span'] * .075
            width = height * label.width / label.height
            quad = trimesh.Trimesh(
                vertices=[[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0]],
                faces=[[0, 1, 2], [0, 2, 3]], process=False)
            quad.visual = trimesh.visual.TextureVisuals(
                uv=[[0, 0], [1, 0], [1, 1], [0, 1]], image=label)
            pose = orientation['camera_pose'].copy()
            pose[:3, 3] = origin + vector + pose[:3, 0] * orientation['span'] * .025
            quad.apply_transform(pose)
            material = pyrender.MetallicRoughnessMaterial(
                baseColorTexture=label, metallicFactor=0., roughnessFactor=1.,
                alphaMode='BLEND', doubleSided=True)
            scene.add(pyrender.Mesh.from_trimesh(quad, material=material))
        scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3), pose=orientation['camera_pose'])
        flags['view_center'] = orientation['center']
    pyrender.Viewer(scene, viewport_size=(1000, 800), window_title=title,
                    use_raymond_lighting=True, **flags)


if __name__ == '__main__':
    main()
