#!/usr/bin/env python3
"""Exercise the pinned upstream reader class without importing its CUDA toolchain."""

import argparse
import ast
import glob
import json
import logging
import os
from pathlib import Path
import subprocess

import cv2
import imageio
import numpy as np

from dataset import UPSTREAM_COMMIT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('converted', type=Path)
    parser.add_argument('--upstream', type=Path, required=True)
    args = parser.parse_args()
    pin = subprocess.check_output(
        ['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True
    ).strip()
    if pin != UPSTREAM_COMMIT:
        parser.error('upstream checkout does not match pin')
    path = args.upstream / 'BundleTrack/scripts/data_reader.py'
    tree = ast.parse(path.read_text())
    reader = next(node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == 'YcbineoatReader')
    module = ast.Module(body=[reader], type_ignores=[])
    namespace = dict(cv2=cv2, np=np, imageio=imageio, glob=glob, os=os, logging=logging)
    exec(compile(module, str(path), 'exec'), namespace)
    data = namespace['YcbineoatReader'](str(args.converted.resolve()), shorter_side=480)
    for index in range(len(data)):
        color, depth, mask = data.get_color(index), data.get_depth(index), data.get_mask(index)
        if color.shape[:2] != depth.shape or depth.shape != mask.shape:
            raise ValueError('upstream reader returned incompatible dimensions')
        if not np.isfinite(depth).all() or not np.any(mask):
            raise ValueError('upstream reader returned invalid depth or empty mask')
    print(json.dumps(dict(upstream_commit=pin, frames=len(data),
                          dimensions=[data.H, data.W], camera_matrix=data.K.tolist(),
                          check='unmodified reader class only; not GPU reconstruction'), indent=2))


if __name__ == '__main__':
    main()
