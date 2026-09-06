"""Check actual reconstructed geometry, trajectory and texture artifacts."""

from pathlib import Path

import cv2
import numpy as np


def inspect_result(root):
    """Reject empty/corrupt output; these checks do not qualify reconstruction accuracy."""
    root = Path(root)
    vertices, faces = [], []
    with (root / 'textured_mesh.obj').open() as stream:
        for line in stream:
            if line.startswith('v '):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                faces.append([int(x.split('/')[0]) for x in line.split()[1:]])
    if not vertices or not faces or not np.isfinite(vertices).all():
        raise ValueError('empty or non-finite reconstructed mesh')
    if any(len(face) < 3 or any(i < 1 or i > len(vertices) for i in face) for face in faces):
        raise ValueError('invalid reconstructed mesh face indices')
    poses = sorted((root / 'ob_in_cam').glob('*.txt'))
    if not poses:
        raise ValueError('missing estimated object trajectory')
    for path in poses:
        pose = np.loadtxt(str(path))
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-4)):
            raise ValueError('invalid T_camera_object: ' + str(path))
    texture = cv2.imread(str(root / 'material_0.png'))
    if texture is None:
        raise ValueError('missing or corrupt mesh texture')
    colored = int(np.count_nonzero(np.any(texture != 0, axis=2)))
    # A known failed CUDA UV kernel mapped every observation to one texel.
    # This detects that failure, not subjective texture quality or coverage.
    if colored <= 1:
        raise ValueError('degenerate mesh texture: check native CUDA UV kernel diagnostic')
    return dict(success=True, vertices=len(vertices), faces=len(faces),
                estimated_pose_count=len(poses), pose_convention='T_camera_object',
                texture_nonzero_pixels=colored, texture_total_pixels=int(texture.shape[0]
                                                                         * texture.shape[1]),
                mesh=str(root / 'textured_mesh.obj'), accuracy_qualified=False)
