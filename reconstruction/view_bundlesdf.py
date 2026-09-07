#!/usr/bin/env python3
"""Display an exported BundleSDF textured mesh using the isolated native environment."""

import argparse
from pathlib import Path


def main():
    """Load immutable output only; this process has no ROS or robot interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mesh', type=Path)
    parser.add_argument('--title', default='BundleSDF saved reconstruction')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    import numpy as np
    import trimesh

    mesh = trimesh.load(str(args.mesh), force='mesh')
    if not len(mesh.vertices) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError('Reconstruction mesh is empty or non-finite')
    if mesh.visual.kind != 'texture' or mesh.visual.material.image is None:
        raise ValueError('Texture missing: keep the OBJ, MTL and PNG in their output folder')
    print('Loaded %d vertices and %d faces' % (len(mesh.vertices), len(mesh.faces)), flush=True)
    if args.check_only:
        return
    import pyrender

    scene = pyrender.Scene(ambient_light=[1., 1., 1.], bg_color=[.12, .12, .12, 1.])
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    pyrender.Viewer(scene, viewport_size=(1000, 800), window_title=args.title,
                    use_raymond_lighting=True)


if __name__ == '__main__':
    main()
