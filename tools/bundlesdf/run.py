#!/usr/bin/env python3
"""Convert a native session and run the pinned upstream in a separate GPU environment."""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

from dataset import UPSTREAM_COMMIT, atomic_json, convert, validate


def run(args):
    if args.validate_only:
        report, _ = validate(args.input_session, allow_incomplete=args.allow_incomplete)
        print(json.dumps(report, indent=2))
        return
    if args.output.exists():
        raise ValueError("refusing to overwrite existing output directory")
    args.output.mkdir(parents=True)
    args.created_output = True
    output = args.output.resolve()
    report = convert(
        args.input_session,
        output / "input",
        args.frame_stride,
        args.start_frame,
        args.end_frame,
        args.allow_incomplete,
    )
    atomic_json(output / "validation.json", report)
    checkout = args.upstream.resolve()
    pin = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if pin != UPSTREAM_COMMIT:
        raise ValueError("upstream commit differs from integration pin")
    # run_custom.py internally runs rm -rf via os.system on its output argument.
    # Only give it a new path of shell-safe characters inside this fresh run.
    for path in (checkout, output):
        if not re.fullmatch(r"[A-Za-z0-9_/.-]+", str(path)):
            raise ValueError(
                "pinned upstream requires paths without spaces or shell metacharacters"
            )
    command = [
        "python3",
        "run_custom.py",
        "--mode",
        "run_video",
        "--video_dir",
        str(output / "input"),
        "--out_folder",
        str(output / "result"),
        "--use_segmenter",
        "0",
        "--use_gui",
        "0",
        "--debug_level",
        "2",
    ]
    if args.python:
        command[0] = str(args.python.resolve())
    else:
        if not shutil.which("docker"):
            raise ValueError(
                (
                    'Docker missing: run tools/bundlesdf/setup.sh after installing '
                    'Docker/NVIDIA Container Toolkit, or supply --python from a '
                    'separately installed BundleSDF environment'
                )
            )
        command = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "-v",
            "%s:%s" % (checkout, checkout),
            "-v",
            "%s:%s" % (output, output),
            "-w",
            str(checkout),
            "piper-bundlesdf:" + UPSTREAM_COMMIT,
        ] + command
    if not (checkout / "BundleTrack/LoFTR/weights/outdoor_ds.ckpt").is_file():
        raise ValueError(
            (
                'Missing LoFTR outdoor_ds.ckpt: download the official weights '
                'linked in the pinned readme into BundleTrack/LoFTR/weights; do not '
                'install weights in ROS'
            )
        )
    atomic_json(
        output / "run.json",
        dict(
            upstream_commit=pin,
            command=command,
            upstream_config_files=["config.yml", "BundleTrack/config_ho3d.yml"],
        ),
    )
    for source, target in [
        ("config.yml", "upstream_nerf.yml"),
        ("BundleTrack/config_ho3d.yml", "upstream_bundletrack.yml"),
    ]:
        shutil.copyfile(str(checkout / source), str(output / target))
    with (output / "bundlesdf.log").open("w") as log:
        result = subprocess.run(
            command, cwd=str(checkout), stdout=log, stderr=subprocess.STDOUT
        )
    if result.returncode:
        raise RuntimeError(
            (
                'BundleSDF failed (exit %d); inspect bundlesdf.log for GPU/CUDA, '
                'missing weights or mycuda/BundleTrack native-extension errors; '
                'rebuild inside the isolated environment'
            )
            % result.returncode
        )
    mesh = output / "result/textured_mesh.obj"
    if not mesh.is_file():
        raise RuntimeError(
            "BundleSDF exited without textured_mesh.obj; reconstruction is not successful"
        )
    vertices = faces = 0
    with mesh.open() as stream:
        for line in stream:
            vertices += line.startswith("v ")
            faces += line.startswith("f ")
    if (
        not vertices
        or not faces
        or not list((output / "result/ob_in_cam").glob("*.txt"))
    ):
        raise RuntimeError("empty mesh or missing estimated object trajectory")
    atomic_json(
        output / "result_validation.json",
        dict(success=True, vertices=vertices, faces=faces, mesh=str(mesh)),
    )
    print(mesh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_session", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--upstream",
        type=Path,
        default=Path.home() / ".local/share/piper-bundlesdf/upstream",
    )
    parser.add_argument(
        "--python",
        type=Path,
        help="Python executable in an already-installed isolated BundleSDF environment",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        if getattr(args, "created_output", False):
            atomic_json(
                args.output / "failure.json",
                dict(success=False, error=str(error), upstream_commit=UPSTREAM_COMMIT),
            )
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
