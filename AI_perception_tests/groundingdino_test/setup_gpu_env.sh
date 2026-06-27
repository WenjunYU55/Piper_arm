#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON310:-python3.10}"
ENV_DIR="${GROUNDED_SAM2_ENV_DIR:-$SCRIPT_DIR/envs/grounded_sam2_py310}"
CUDA_INDEX="${PYTORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [ -x "$SCRIPT_DIR/envs/python310_base/bin/python3.10" ]; then
  PYTHON_BIN="$SCRIPT_DIR/envs/python310_base/bin/python3.10"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "An NVIDIA driver and nvidia-smi are required." >&2
  exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "$PYTHON_BIN is required in an isolated environment." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$ENV_DIR/bin/python" -m pip install --index-url "$CUDA_INDEX" \
  "torch==2.11.0" "torchvision==0.26.0"
"$ENV_DIR/bin/python" -m pip install \
  numpy==2.2.6 opencv-python==4.13.0.92 PyYAML==6.0.3 transformers==4.44.2 \
  addict==2.4.0 yapf==0.43.0 timm==1.0.27 supervision==0.29.0 \
  pycocotools==2.0.11 hydra-core==1.3.3 iopath==0.1.10 pillow==12.2.0 tqdm==4.68.3
# SAM2's fill-hole extension is optional. Avoid requiring a system nvcc toolchain;
# model inference still runs through CUDA-enabled PyTorch.
SAM2_BUILD_CUDA=0 "$ENV_DIR/bin/python" -m pip install \
  "sam-2 @ git+https://github.com/IDEA-Research/Grounded-SAM-2.git@b7a9c29f196edff0eb54dbe14588d7ae5e3dde28" \
  "groundingdino @ git+https://github.com/IDEA-Research/Grounded-SAM-2.git@b7a9c29f196edff0eb54dbe14588d7ae5e3dde28#subdirectory=grounding_dino"
"$SCRIPT_DIR/fetch_ai_assets.sh"
GROUNDED_SAM2_PYTHON="$ENV_DIR/bin/python" "$SCRIPT_DIR/check_env.sh"
"$ENV_DIR/bin/python" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

echo "GPU AI environment ready: $ENV_DIR"
