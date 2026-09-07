#!/usr/bin/env bash
# Ubuntu 20.04 native userspace, following the explicit cuRobo interpreter boundary.
# All downloaded libraries, headers, weights and environments stay outside ROS/Git.
set -euo pipefail
BUNDLESDF_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLESDF_RUNTIME="${PIPER_BUNDLESDF_RUNTIME:-$HOME/.local/share/piper-bundlesdf}"
BUNDLESDF_ENV="${PIPER_BUNDLESDF_ENV:-$HOME/.venvs/piper-bundlesdf}"
BUNDLESDF_CUDA="${PIPER_BUNDLESDF_CUDA_HOME:-$HOME/.local/cuda-12.8}"
BUNDLESDF_BASE_PYTHON="${PIPER_BUNDLESDF_BASE_PYTHON:-$HOME/Piper_arm/AI_perception_tests/groundingdino_test/envs/grounded_sam2_py310/bin/python}"
BUNDLESDF_JOBS="${PIPER_BUNDLESDF_BUILD_JOBS:-4}"
BUNDLESDF_PIN=ffa67d425240b5b76d2e387a7dd3d3735a7cf1a1
BUNDLESDF_NATIVE="$BUNDLESDF_RUNTIME/native/usr"
BUNDLESDF_UPSTREAM="$BUNDLESDF_RUNTIME/upstream"
# Installation must also ignore a caller's sourced Foxy overlay.
unset PYTHONPATH PYTHONHOME
export LD_LIBRARY_PATH="$BUNDLESDF_NATIVE/lib:$BUNDLESDF_NATIVE/lib/x86_64-linux-gnu:$BUNDLESDF_CUDA/lib64"
for executable in "$BUNDLESDF_BASE_PYTHON" "$BUNDLESDF_CUDA/bin/nvcc"; do
  test -x "$executable" || { echo "Missing executable: $executable; set PIPER_BUNDLESDF_BASE_PYTHON or PIPER_BUNDLESDF_CUDA_HOME." >&2; exit 2; }
done
"$BUNDLESDF_BASE_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10), "BundleSDF requires an isolated Python 3.10 base"'
# The existing ROS development workstation supplies these read-only host libraries.
for package in libpcl-dev libopencv-dev libboost-all-dev libeigen3-dev; do
  dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'install ok installed' || {
    echo "Missing host development prerequisite $package; install it or use the separate Docker route." >&2; exit 2;
  }
done
mkdir -p "$BUNDLESDF_RUNTIME" "$BUNDLESDF_RUNTIME/logs" "$BUNDLESDF_NATIVE"
if [ ! -x "$BUNDLESDF_ENV/bin/python" ]; then
  "$BUNDLESDF_BASE_PYTHON" -m venv "$BUNDLESDF_ENV"
fi
BUNDLESDF_PYTHON="$BUNDLESDF_ENV/bin/python"
"$BUNDLESDF_PYTHON" -m pip install pip==25.0.1 setuptools==75.8.2 wheel==0.45.1 ninja==1.11.1.3 cmake==3.25.2
"$BUNDLESDF_PYTHON" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"$BUNDLESDF_PYTHON" -m pip install -r "$BUNDLESDF_SCRIPTS/native-requirements.txt"
# Upstream's Dockerfile mixes torch 2.6 with a 2.5.1 Kaolin wheel. Use matching ABI.
"$BUNDLESDF_PYTHON" -m pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu124.html
if [ ! -d "$BUNDLESDF_UPSTREAM/.git" ]; then
  git clone https://github.com/NVlabs/BundleSDF.git "$BUNDLESDF_UPSTREAM"
  git -C "$BUNDLESDF_UPSTREAM" checkout --detach "$BUNDLESDF_PIN"
fi
test "$(git -C "$BUNDLESDF_UPSTREAM" rev-parse HEAD)" = "$BUNDLESDF_PIN"
if git -C "$BUNDLESDF_UPSTREAM" apply --check "$BUNDLESDF_SCRIPTS/patches/native-build.patch"; then
  git -C "$BUNDLESDF_UPSTREAM" apply "$BUNDLESDF_SCRIPTS/patches/native-build.patch"
else
  git -C "$BUNDLESDF_UPSTREAM" apply --reverse --check "$BUNDLESDF_SCRIPTS/patches/native-build.patch"
fi
if git -C "$BUNDLESDF_UPSTREAM" apply --check "$BUNDLESDF_SCRIPTS/patches/headless-tracking.patch"; then
  git -C "$BUNDLESDF_UPSTREAM" apply "$BUNDLESDF_SCRIPTS/patches/headless-tracking.patch"
else
  git -C "$BUNDLESDF_UPSTREAM" apply --reverse --check "$BUNDLESDF_SCRIPTS/patches/headless-tracking.patch"
fi
# Download and extract packages without apt installation or system mutation.
(cd "$BUNDLESDF_RUNTIME" && apt-get download libyaml-cpp-dev=0.6.2-4ubuntu1 libyaml-cpp0.6=0.6.2-4ubuntu1 libzmq3-dev=4.3.2-2ubuntu1 libzmq5=4.3.2-2ubuntu1 freeglut3-dev=2.8.1-3 freeglut3=2.8.1-3 libpgm-5.2-0=5.2.122~dfsg-3ubuntu1 libnorm1=1.5.8+dfsg2-2build1)
for archive in "$BUNDLESDF_RUNTIME"/*.deb; do dpkg-deb -x "$archive" "$BUNDLESDF_RUNTIME/native"; done
"$BUNDLESDF_PYTHON" - "$BUNDLESDF_NATIVE" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
# Ubuntu's yaml-cpp CMake exports have a hard-coded /usr prefix; relocate the
# extracted copy only. Host files remain untouched.
for path in (root / 'lib/x86_64-linux-gnu/cmake/yaml-cpp').glob('*.cmake'):
    text = path.read_text().replace('"/usr"', '"' + str(root) + '"')
    path.write_text(text.replace('"/usr/', '"' + str(root) + '/'))
PY
fetch_archive() {
  local name="$1" url="$2" digest="$3"
  if [ ! -f "$BUNDLESDF_RUNTIME/$name.tar.gz" ]; then
    curl -fL "$url" -o "$BUNDLESDF_RUNTIME/$name.tar.gz.part"
    mv "$BUNDLESDF_RUNTIME/$name.tar.gz.part" "$BUNDLESDF_RUNTIME/$name.tar.gz"
  fi
  printf '%s  %s\n' "$digest" "$BUNDLESDF_RUNTIME/$name.tar.gz" | sha256sum -c -
  if [ ! -d "$BUNDLESDF_RUNTIME/$name" ]; then
    tar -xzf "$BUNDLESDF_RUNTIME/$name.tar.gz" -C "$BUNDLESDF_RUNTIME"
  fi
}
fetch_archive eigen-3.4.0 https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz 8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
fetch_archive opencv-4.11.0 https://github.com/opencv/opencv/archive/refs/tags/4.11.0.tar.gz 9a7c11f924eff5f8d8070e297b322ee68b9227e003fd600d4b8122198091665f
fetch_archive opencv_contrib-4.11.0 https://github.com/opencv/opencv_contrib/archive/refs/tags/4.11.0.tar.gz 2dfc5957201de2aa785064711125af6abb2e80a64e2dc246aca4119b19687041
BUNDLESDF_CMAKE="$BUNDLESDF_ENV/bin/cmake"
"$BUNDLESDF_CMAKE" -S "$BUNDLESDF_RUNTIME/eigen-3.4.0" -B "$BUNDLESDF_RUNTIME/eigen-build" -DCMAKE_INSTALL_PREFIX="$BUNDLESDF_NATIVE"
"$BUNDLESDF_CMAKE" --install "$BUNDLESDF_RUNTIME/eigen-build"
BUNDLESDF_ARCH="$("$BUNDLESDF_PYTHON" -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')"
"$BUNDLESDF_CMAKE" -S "$BUNDLESDF_RUNTIME/opencv-4.11.0" -B "$BUNDLESDF_RUNTIME/opencv-build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$BUNDLESDF_NATIVE" \
  -DOPENCV_EXTRA_MODULES_PATH="$BUNDLESDF_RUNTIME/opencv_contrib-4.11.0/modules" \
  -DBUILD_LIST=core,imgproc,imgcodecs,highgui,features2d,calib3d,rgbd,cudev,cudaarithm,cudaimgproc,cudaoptflow,cudafeatures2d \
  -DBUILD_opencv_cudev=ON -DWITH_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR="$BUNDLESDF_CUDA" \
  -DCUDA_ARCH_BIN="$BUNDLESDF_ARCH" -DCUDA_ARCH_PTX= -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF \
  -DWITH_CUDNN=OFF -DWITH_NVCUVID=OFF -DWITH_NVCUVENC=OFF -DWITH_QT=OFF -DWITH_GTK=OFF \
  -DWITH_OPENEXR=OFF -DBUILD_JAVA=OFF
"$BUNDLESDF_CMAKE" --build "$BUNDLESDF_RUNTIME/opencv-build" -j"$BUNDLESDF_JOBS"
"$BUNDLESDF_CMAKE" --install "$BUNDLESDF_RUNTIME/opencv-build"
if [ ! -d "$BUNDLESDF_RUNTIME/pytorch3d/.git" ]; then
  git clone --branch v0.7.9 --depth 1 https://github.com/facebookresearch/pytorch3d.git "$BUNDLESDF_RUNTIME/pytorch3d"
fi
test "$(git -C "$BUNDLESDF_RUNTIME/pytorch3d" rev-parse HEAD)" = 33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba
# BundleSDF imports only pure tensor transforms; no PyTorch3D renderer/ops is used.
PYTORCH3D_NO_EXTENSION=1 "$BUNDLESDF_PYTHON" -m pip install --no-build-isolation "$BUNDLESDF_RUNTIME/pytorch3d"
BUNDLESDF_EIGEN_INCLUDE="$BUNDLESDF_NATIVE/include/eigen3" \
CUDA_HOME="$BUNDLESDF_CUDA" TORCH_CUDA_ARCH_LIST="$BUNDLESDF_ARCH" MAX_JOBS=2 \
  "$BUNDLESDF_PYTHON" -m pip install --no-build-isolation --no-deps -e "$BUNDLESDF_UPSTREAM/mycuda"
"$BUNDLESDF_CMAKE" -S "$BUNDLESDF_UPSTREAM/BundleTrack" -B "$BUNDLESDF_UPSTREAM/BundleTrack/build" \
  -U 'PYTHON_MODULE_*' -U 'PYTHON_VERSION*' \
  -DCMAKE_CUDA_COMPILER="$BUNDLESDF_CUDA/bin/nvcc" -DCUDA_TOOLKIT_ROOT_DIR="$BUNDLESDF_CUDA" \
  -DBUNDLESDF_CUDA_ARCHITECTURES="${BUNDLESDF_ARCH/./}" -DCMAKE_PREFIX_PATH="$BUNDLESDF_NATIVE" \
  -DEigen3_DIR="$BUNDLESDF_NATIVE/share/eigen3/cmake" \
  -DEIGEN_INCLUDE_DIR="$BUNDLESDF_NATIVE/include/eigen3" -DOpenCV_DIR="$BUNDLESDF_NATIVE/lib/cmake/opencv4" \
  -Dpybind11_DIR="$BUNDLESDF_ENV/lib/python3.10/site-packages/pybind11/share/cmake/pybind11" \
  -DPYBIND11_FINDPYTHON=ON -DPYTHON_EXECUTABLE="$BUNDLESDF_PYTHON" \
  -DPython_EXECUTABLE="$BUNDLESDF_PYTHON" -DCMAKE_LIBRARY_PATH="$BUNDLESDF_NATIVE/lib/x86_64-linux-gnu"
"$BUNDLESDF_CMAKE" --build "$BUNDLESDF_UPSTREAM/BundleTrack/build" -j2
mkdir -p "$BUNDLESDF_UPSTREAM/BundleTrack/LoFTR/weights"
BUNDLESDF_WEIGHTS="$BUNDLESDF_UPSTREAM/BundleTrack/LoFTR/weights/outdoor_ds.ckpt"
if [ ! -f "$BUNDLESDF_WEIGHTS" ]; then
  curl -fL https://huggingface.co/kornia/loftr/resolve/0fb3b456cf176746709c38fa9cf80695a1c82dac/loftr_outdoor.ckpt -o "$BUNDLESDF_WEIGHTS.part"
  mv "$BUNDLESDF_WEIGHTS.part" "$BUNDLESDF_WEIGHTS"
fi
printf '%s  %s\n' 21f5bec5968178e8bc8b7633441836fe5de4f47d861dd2cd7dc38e271b0479ec "$BUNDLESDF_WEIGHTS" | sha256sum -c -
"$BUNDLESDF_PYTHON" -m pip freeze --all > "$BUNDLESDF_RUNTIME/python-requirements.lock"
sha256sum "$BUNDLESDF_RUNTIME"/*.deb > "$BUNDLESDF_RUNTIME/native-debs.lock"
PIPER_BUNDLESDF_RUNTIME="$BUNDLESDF_RUNTIME" PIPER_BUNDLESDF_PYTHON="$BUNDLESDF_PYTHON" \
  PIPER_BUNDLESDF_CUDA_HOME="$BUNDLESDF_CUDA" "$BUNDLESDF_SCRIPTS/python_native.sh" \
  "$BUNDLESDF_SCRIPTS/native_diagnostics.py" --output "$BUNDLESDF_RUNTIME/runtime.json"
echo 'Environment ready. Run the official example and verify an actual output mesh before claiming reconstruction success.'
