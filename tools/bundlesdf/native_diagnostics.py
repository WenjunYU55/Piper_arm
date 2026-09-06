#!/usr/bin/env python3
"""Check actual offline GPU/native imports and save version provenance, not a mesh claim."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import cv2
import kaolin
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runtime = Path(os.environ.get('PIPER_BUNDLESDF_RUNTIME',
                                  str(Path.home() / '.local/share/piper-bundlesdf')))
    import bundlesdf
    import my_cpp
    from mycuda import common, gridencoder
    from pytorch3d.transforms import so3_exp_map

    assert list(my_cpp.MapFrame().keys()) == [], "C++ pybind11 headers lack mapping API"

    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU is available to the isolated BundleSDF environment')
    identity = so3_exp_map(torch.zeros((1, 3), device='cuda'))
    assert torch.allclose(identity, torch.eye(3, device='cuda')[None])
    points = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int16, device='cuda')
    octree = kaolin.ops.spc.unbatched_points_to_octree(points, level=4)
    assert octree.numel() > 0
    # Import success does not prove that a compiled kernel writes its output.
    faces = torch.tensor([[0, 1, 2]], device='cuda')
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], device='cuda')
    locations = torch.tensor([[.25, .25, 0.]], device='cuda')
    face_ids = torch.tensor([0], device='cuda')
    texture_uv = torch.tensor([[0., 0.], [100., 0.], [0., 100.]], device='cuda')
    output_uv = torch.full((1, 2), -123., device='cuda')
    common.rayColorToTextureImageCUDA(
        faces, vertices, locations, face_ids, texture_uv, output_uv)
    torch.cuda.synchronize()
    assert torch.allclose(output_uv, torch.tensor([[25., 25.]], device='cuda')), (
        'Texture CUDA kernel failed known-triangle test; rebuild mycuda with private Eigen 3.4')
    checkout = runtime / 'upstream'
    patch = subprocess.check_output(['git', '-C', str(checkout), 'diff', 'HEAD'])
    result = dict(
        check='GPU kernels and native imports; not reconstruction validation',
        python=platform.python_version(), torch=torch.__version__,
        torch_cuda=torch.version.cuda, kaolin=kaolin.__version__, numpy=np.__version__,
        cuda_compiler=subprocess.check_output(
            [str(Path(os.environ['CUDA_HOME']) / 'bin/nvcc'), '--version'], text=True),
        cpp_opencv=subprocess.check_output(
            [str(runtime / 'native/usr/bin/opencv_version')], text=True).strip(),
        loftr_weights_sha256=hashlib.sha256(
            (checkout / 'BundleTrack/LoFTR/weights/outdoor_ds.ckpt').read_bytes()).hexdigest(),
        python_opencv=cv2.__version__, gpu=torch.cuda.get_device_name(),
        upstream_commit=subprocess.check_output(
            ['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip(),
        local_patch_sha256=hashlib.sha256(patch).hexdigest(),
        modules={name: module.__file__ for name, module in (
            ('bundlesdf', bundlesdf), ('my_cpp', my_cpp),
            ('common', common), ('gridencoder', gridencoder))},
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
