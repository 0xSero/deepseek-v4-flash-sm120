#!/usr/bin/env bash
# DeepSeek-V4-Flash on SM_120 — TP=4 sglang Docker launcher (rig-specific variant).
#
# This is a sibling of `launch_dsv4_flash_sm120.sh`, kept separate so we can
# tune for our 4× RTX Pro 6000 rig (PCIe Gen 5, no NVLink, model on /nvme/...)
# without diverging from the upstream reference launcher.
#
# Differences from the upstream launcher:
#   - MODEL_DIR defaults to /nvme/models/safetensors/DeepSeek-V4-Flash-FP8
#   - CONTAINER_NAME is suffixed -tp4 to coexist with the upstream launcher
#   - sglang Prometheus metrics enabled (latency buckets + token histograms)
#   - --attention-backend compressed left in (matches upstream); flip to
#     flashinfer for the experiment described in STATUS.md
#
# Env variables you can override on the command line:
#   MODEL_DIR        absolute path to DeepSeek-V4-Flash-FP8 checkpoint
#                    (default: /nvme/models/safetensors/DeepSeek-V4-Flash-FP8)
#   PORT             host port for the OpenAI-compatible server (default 8000)
#   CUDA_VISIBLE     GPU indices to expose (default: 0,1,2,3 -> the four RTX Pro 6000s)
#   DSV4_KERNEL_DIR  this repo root (default: the repo containing this script)
#
# Example:
#   MODEL_DIR=/nvme/models/safetensors/DeepSeek-V4-Flash-FP8 PORT=8000 ./launch_dsv4_flash_sm120_tp4.sh
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/nvme/models/safetensors/DeepSeek-V4-Flash-FP8}
PORT=${PORT:-${1:-8000}}
CONTAINER_NAME=${CONTAINER_NAME:-sglang-dsv4-flash-sm120-tp4}
CUDA_VISIBLE=${CUDA_VISIBLE:-0,1,2,3}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DSV4_KERNEL_DIR=${DSV4_KERNEL_DIR:-"$(cd "${SCRIPT_DIR}/.." && pwd)"}
DSV4_BUILD_DIR="${DSV4_KERNEL_DIR}/build-docker"

# Auto-build the kernel for the image's Python 3.12 if missing.
if [[ ! -f "${DSV4_BUILD_DIR}/deepseek_v4_kernel/cuda.cpython-312-x86_64-linux-gnu.so" ]]; then
  echo "[launcher] Building deepseek_v4_kernel against sglang image (one-time)..."
  "${DSV4_KERNEL_DIR}/scripts/build_in_sglang_docker.sh"
fi

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

exec docker run \
  --name "$CONTAINER_NAME" \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE}" \
  --shm-size=64g \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v "${MODEL_DIR}:/workspace/model:ro" \
  -v "${DSV4_BUILD_DIR}:/dsv4:ro" \
  -v "${DSV4_KERNEL_DIR}/w8a8-configs:/workspace/sglang/python/sglang/srt/layers/quantization/configs:ro" \
  -v "${DSV4_KERNEL_DIR}/moe-configs:/moe-configs:ro" \
  -e SGLANG_MOE_CONFIG_DIR=/moe-configs \
  -e PYTHONPATH=/dsv4 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e TORCH_CUDA_ARCH_LIST=12.0 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_P2P_DISABLE=0 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=lo \
  -e GLOO_SOCKET_IFNAME=lo \
  -e NCCL_DEBUG=WARN \
  -e NCCL_CUMEM_HOST_ENABLE=0 \
  \
  -e SGLANG_ENABLE_JIT_DEEPGEMM=0 \
  -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
  -e SGLANG_DSV4_FP4_EXPERTS=0 \
  -e SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 \
  -e SGLANG_OPT_USE_TILELANG_INDEXER=1 \
  -e SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 \
  -e SGLANG_OPT_USE_TILELANG_SWA_PREPARE=1 \
  -e SGLANG_OPT_USE_TILELANG_MHC_PRE=1 \
  -e SGLANG_OPT_USE_TILELANG_MHC_POST=1 \
  -e SGLANG_ENABLE_SPEC_V2=True \
  -e SGLANG_SET_CPU_AFFINITY=1 \
  -e SGLANG_ENABLE_THINKING=1 \
  -e SGLANG_REASONING_EFFORT=high \
  \
  lmsysorg/sglang:deepseek-v4-blackwell \
  python3 -m sglang.launch_server \
    --model-path /workspace/model \
    --host 0.0.0.0 --port "${PORT}" \
    --served-model-name deepseek-v4-flash \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --context-length 393216 \
    --mem-fraction-static 0.85 \
    --max-running-requests 8 \
    --kv-cache-dtype fp8_e4m3 \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4 \
    --fp8-gemm-backend triton \
    --moe-runner-backend triton \
    --attention-backend compressed \
    --chunked-prefill-size 8192 \
    --watchdog-timeout 3600 \
    --page-size 256 \
    --enable-metrics \
    --collect-tokens-histogram \
    --bucket-time-to-first-token 0.1 0.25 0.5 1 2 5 10 \
    --bucket-inter-token-latency 0.05 0.1 0.2 0.5 1 \
    --bucket-e2e-request-latency 1 5 10 30 60 120
