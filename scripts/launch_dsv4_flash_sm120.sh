#!/usr/bin/env bash
# DeepSeek-V4-Flash on SM120 (RTX PRO 6000 Blackwell Workstation) via SGLang Docker.
#
# The SGLang image is used unmodified. The custom SM120 FlashMLA sparse-decode
# patch is injected through a read-only bind mount plus PYTHONPATH/sitecustomize.
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/mnt/llm_models/DeepSeek-V4-Flash-FP8}
PORT=${PORT:-${1:-8000}}
CONTAINER_NAME=${CONTAINER_NAME:-sglang-dsv4-flash-sm120}
CUDA_VISIBLE=${CUDA_VISIBLE:-0,2,3,4}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DSV4_KERNEL_DIR=${DSV4_KERNEL_DIR:-"$(cd "${SCRIPT_DIR}/.." && pwd)"}
DSV4_BUILD_DIR="${DSV4_KERNEL_DIR}/build-docker"

if [[ ! -f "${DSV4_BUILD_DIR}/deepseek_v4_kernel/cuda.cpython-312-x86_64-linux-gnu.so" ]]; then
  echo "[launcher] Building deepseek_v4_kernel against the SGLang image (one-time)..."
  "${DSV4_KERNEL_DIR}/scripts/build_in_sglang_docker.sh"
fi

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

exec docker run \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --privileged \
  --shm-size=64g \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v "${MODEL_DIR}:/workspace/model:ro" \
  -v "${DSV4_BUILD_DIR}:/dsv4:ro" \
  -e PYTHONPATH=/dsv4 \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE}" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_HOME=/usr/local/cuda \
  -e TORCH_CUDA_ARCH_LIST=12.0 \
  -e OMP_NUM_THREADS=8 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_P2P_DISABLE=0 \
  -e NCCL_P2P_LEVEL=PIX \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=lo \
  -e GLOO_SOCKET_IFNAME=lo \
  -e NCCL_DEBUG=WARN \
  -e NCCL_CUMEM_HOST_ENABLE=0 \
  -e TORCH_NCCL_BLOCKING_WAIT=1 \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -e DSV4_KERNEL_STRICT=0 \
  -e DSV4_KERNEL_TRACE=0 \
  -e SGLANG_ENABLE_THINKING=1 \
  -e SGLANG_REASONING_EFFORT=max \
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
  lmsysorg/sglang:deepseek-v4-blackwell \
  python3 -m sglang.launch_server \
    --model-path /workspace/model \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --served-model-name deepseek-v4-flash \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --context-length 393216 \
    --mem-fraction-static 0.85 \
    --max-running-requests 8 \
    --kv-cache-dtype fp8_e4m3 \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4 \
    --attention-backend compressed \
    --fp8-gemm-backend triton \
    --moe-runner-backend triton \
    --chunked-prefill-size 8192 \
    --watchdog-timeout 3600 \
    --page-size 256 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 1 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 2 \
    --speculative-attention-mode decode \
    --cuda-graph-max-bs 32 \
    --enable-return-routed-experts
