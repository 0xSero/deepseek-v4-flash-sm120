#!/usr/bin/env bash
# Build the deepseek_v4_kernel CUDA extension for the sglang DeepSeek-V4
# Blackwell Docker image's Python 3.12 environment.
#
# NOTHING is persisted into the Docker image: we spin up a throwaway
# container, invoke setup.py build_ext (NOT `pip install`), and copy the
# compiled .so back to ./build-docker/deepseek_v4_kernel/.  The sglang
# image on disk is completely untouched.
#
# The build artifacts are later injected into a running sglang container
# via a read-only bind mount + PYTHONPATH; they never hit the image
# filesystem.
#
# Usage:
#   scripts/build_in_sglang_docker.sh [IMAGE]
# default IMAGE: lmsysorg/sglang:deepseek-v4-blackwell

set -euo pipefail
IMG="${1:-lmsysorg/sglang:deepseek-v4-blackwell}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build-docker"
OUT_PKG="${BUILD_DIR}/deepseek_v4_kernel"
# Optional: set CUTLASS_DIR to a CUTLASS checkout to build against the full
# library.  Otherwise the shim at csrc/common/cutlass_shim.h is used.
CUTLASS_DIR_OPT=()
if [[ -n "${CUTLASS_DIR:-}" ]]; then
    CUTLASS_DIR_OPT=(-e "CUTLASS_DIR=/cutlass" -v "${CUTLASS_DIR}:/cutlass:ro")
fi

# Clean previous output but keep the src read-only copy.
rm -rf "${BUILD_DIR}"
mkdir -p "${OUT_PKG}"

docker run --rm \
  --gpus all \
  -v "${PROJECT_DIR}:/src:ro" \
  -v "${BUILD_DIR}:/out" \
  "${CUTLASS_DIR_OPT[@]}" \
  -e TORCH_CUDA_ARCH_LIST=12.0 \
  -e DSV4_KERNEL_ARCH='arch=compute_120a,code=sm_120a' \
  -e NVCC_THREADS="${NVCC_THREADS:-16}" \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  --entrypoint bash \
  "${IMG}" -lc '
set -e
cp -r /src /tmp/dsv4
cd /tmp/dsv4
# Only run the CUDA extension build step; no package-level install.
python3 setup.py build_ext --inplace
# Sanity-check the resulting .so works.
python3 - <<PY
import importlib.util, sys, os
spec = importlib.util.spec_from_file_location(
    "deepseek_v4_kernel.cuda",
    next(p for p in os.listdir("deepseek_v4_kernel") if p.endswith(".so") and "cpython-312" in p) or "",
)
print("built:", os.listdir("deepseek_v4_kernel"))
PY
# Copy pure-Python + compiled .so back to /out; strip pycache and host .so.
mkdir -p /out/deepseek_v4_kernel
for f in __init__.py ops.py _patch.py sitecustomize_hook.py; do
  cp /tmp/dsv4/deepseek_v4_kernel/$f /out/deepseek_v4_kernel/$f
done
cp /tmp/dsv4/deepseek_v4_kernel/cuda.cpython-312-*.so /out/deepseek_v4_kernel/
# Top-level sitecustomize.py is auto-loaded by Python on every interpreter
# start so the patch follows sglang into every forked/spawned worker.
cp /tmp/dsv4/deepseek_v4_kernel/sitecustomize_hook.py /out/sitecustomize.py
chown -R $HOST_UID:$HOST_GID /out
'

echo "[deepseek_v4_kernel] Docker-Python 3.12 artifacts:"
ls "${OUT_PKG}"
