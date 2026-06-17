#!/usr/bin/env python3
"""Check the offline GroundingDINO test environment without installing anything."""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import sys
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_DIR = SCRIPT_DIR / "GroundingDINO"
DEFAULT_CONFIG_PATH = DEFAULT_REPO_DIR / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
DEFAULT_CHECKPOINT_PATH = SCRIPT_DIR / "weights" / "groundingdino_swint_ogc.pth"


def torch_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "installed": False,
        "version": "",
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": "",
    }
    try:
        import torch
    except Exception as exc:
        status["error"] = str(exc)
        return status

    status["installed"] = True
    status["version"] = str(getattr(torch, "__version__", ""))
    status["cuda_available"] = bool(torch.cuda.is_available())
    if status["cuda_available"]:
        status["cuda_device_count"] = int(torch.cuda.device_count())
        try:
            status["cuda_device_name"] = str(torch.cuda.get_device_name(0))
        except Exception as exc:
            status["cuda_device_name"] = "unavailable: %s" % exc
    return status


def import_status(module_name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as exc:
        return {
            "module": module_name,
            "available": False,
            "origin": "",
            "error": str(exc),
        }
    return {
        "module": module_name,
        "available": spec is not None,
        "origin": str(spec.origin) if spec is not None and spec.origin else "",
    }


def add_repo_to_path(repo_dir: Path) -> None:
    if repo_dir.is_dir() and str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))


def check_env(config_path: Path, checkpoint_path: Path, repo_dir: Path) -> dict[str, Any]:
    add_repo_to_path(repo_dir)
    return {
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
            "version_info": {
                "major": sys.version_info.major,
                "minor": sys.version_info.minor,
                "micro": sys.version_info.micro,
            },
            "platform": platform.platform(),
        },
        "torch": torch_status(),
        "cuda_home": {
            "set": bool(os.environ.get("CUDA_HOME")),
            "value": os.environ.get("CUDA_HOME", ""),
        },
        "groundingdino_import": import_status("groundingdino"),
        "official_inference_import": import_status("groundingdino.util.inference"),
        "paths": {
            "repo_dir": {
                "path": str(repo_dir),
                "exists": repo_dir.is_dir(),
            },
            "config_path": {
                "path": str(config_path),
                "exists": config_path.is_file(),
            },
            "checkpoint_path": {
                "path": str(checkpoint_path),
                "exists": checkpoint_path.is_file(),
            },
        },
    }


def print_human(status: dict[str, Any]) -> None:
    print("Python:", status["python"]["version"])
    print("Python executable:", status["python"]["executable"])
    torch = status["torch"]
    print("torch installed:", torch["installed"])
    if torch["installed"]:
        print("torch version:", torch["version"])
        print("CUDA available:", torch["cuda_available"])
        print("CUDA devices:", torch["cuda_device_count"])
        if torch["cuda_device_name"]:
            print("CUDA device 0:", torch["cuda_device_name"])
    else:
        print("torch error:", torch.get("error", "not installed"))
    print("CUDA_HOME set:", status["cuda_home"]["set"])
    print("CUDA_HOME:", status["cuda_home"]["value"])
    print("GroundingDINO import:", status["groundingdino_import"]["available"])
    print("GroundingDINO inference import:", status["official_inference_import"]["available"])
    for key, value in status["paths"].items():
        print("%s exists: %s  %s" % (key, value["exists"], value["path"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check offline GroundingDINO environment.")
    parser.add_argument("--repo-dir", type=Path, default=Path(os.environ.get("GROUNDINGDINO_REPO_DIR", DEFAULT_REPO_DIR)))
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("GROUNDINGDINO_CONFIG_PATH", DEFAULT_CONFIG_PATH)))
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("GROUNDINGDINO_CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH)))
    parser.add_argument("--yaml", action="store_true", help="Print YAML instead of human-readable text.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status = check_env(args.config.expanduser().resolve(), args.checkpoint.expanduser().resolve(), args.repo_dir.expanduser().resolve())
    if args.yaml:
        print(yaml.safe_dump(status, sort_keys=False))
    else:
        print_human(status)
    missing_required = [
        not status["torch"]["installed"],
        not status["groundingdino_import"]["available"],
        not status["official_inference_import"]["available"],
        not status["paths"]["config_path"]["exists"],
        not status["paths"]["checkpoint_path"]["exists"],
    ]
    return 0 if not any(missing_required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
