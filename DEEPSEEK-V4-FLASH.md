# DeepSeek-V4-Flash on SM120 — corrected recipe

This document is intentionally short. The earlier version over-specified stale settings. The current working recipe is:

- SGLang image: `lmsysorg/sglang:deepseek-v4-blackwell`
- Model: `sgl-project/DeepSeek-V4-Flash-FP8`
- GPUs: 4x RTX PRO 6000 Blackwell / SM120
- Patch: mount this repo's `build-docker/` at `/dsv4` and set `PYTHONPATH=/dsv4`
- Chat encoding: **SGLang built-in `encoding_dsv4.py`**, not a Jinja template
- CUDA graphs: enabled, `--cuda-graph-max-bs 32`
- Spec decode: EAGLE enabled with one step, topk 1, two draft tokens

## Why this repo exists

Upstream FlashMLA sparse decode does not ship an SM120 implementation. Without this patch, DeepSeek-V4-Flash can fail on RTX PRO 6000 with:

```text
RuntimeError: Unsupported architecture for sparse decode fwd
```

`build-docker/sitecustomize.py` imports `deepseek_v4_kernel` at interpreter startup and monkey-patches the FlashMLA sparse-decode call for SM120.

## Build

```bash
git clone https://github.com/0xSero/deepseek-v4-flash-sm120.git
cd deepseek-v4-flash-sm120
git submodule update --init --recursive

docker pull lmsysorg/sglang:deepseek-v4-blackwell
scripts/build_in_sglang_docker.sh
```

## Launch

```bash
MODEL_DIR=/mnt/llm_models/DeepSeek-V4-Flash-FP8 PORT=8000 \
  scripts/launch_dsv4_flash_sm120.sh
```

The launch script uses this core SGLang command inside the container:

```bash
python3 -m sglang.launch_server \
  --model-path /workspace/model \
  --host 0.0.0.0 \
  --port 8000 \
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
```

Do **not** add `--chat-template`. DeepSeek's official V4 repos provide `encoding/encoding_dsv4.py`, not a Jinja template, and current SGLang calls its adapted encoder automatically for `deepseekv4`.

Do **not** add `--disable-cuda-graph` unless you are debugging a separate graph-capture issue.

## Required mounts and environment

```bash
-v "$MODEL_DIR:/workspace/model:ro"
-v "$PWD/build-docker:/dsv4:ro"
-e PYTHONPATH=/dsv4
-e CUDA_VISIBLE_DEVICES=0,2,3,4
-e NCCL_P2P_DISABLE=0
-e NCCL_P2P_LEVEL=PIX
-e SGLANG_ENABLE_SPEC_V2=True
-e SGLANG_ENABLE_JIT_DEEPGEMM=0
-e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
-e SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
-e SGLANG_DSV4_FP4_EXPERTS=0
-e SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
-e SGLANG_OPT_USE_TILELANG_INDEXER=1
-e SGLANG_OPT_USE_TILELANG_SWA_PREPARE=1
-e SGLANG_OPT_USE_TILELANG_MHC_PRE=1
-e SGLANG_OPT_USE_TILELANG_MHC_POST=1
```

## Smoke test

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","temperature":0,"max_tokens":32,
       "messages":[{"role":"user","content":"Say OK only."}]}'
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Unsupported architecture for sparse decode fwd` | Patch did not load. Verify `/dsv4/sitecustomize.py`, `PYTHONPATH=/dsv4`, and the `build-docker` mount. |
| Chat/tool/thinking formatting is wrong | Remove any custom Jinja template and keep `--tool-call-parser deepseekv4` / `--reasoning-parser deepseek-v4`. |
| OOM | Lower `--mem-fraction-static` or `--max-running-requests`. |
| 3090 gets used | Set `CUDA_VISIBLE_DEVICES=0,2,3,4`. |

## Current max-thinking benchmark on 4x RTX PRO 6000

Configuration: no Jinja template, SGLang built-in `encoding_dsv4`, `SGLANG_ENABLE_THINKING=1`, `SGLANG_REASONING_EFFORT=max`, EAGLE draft tokens 2, CUDA graphs enabled.

| Context | TTFT | Prefill tok/s | Decode tok/s | Needle accuracy |
|---:|---:|---:|---:|:---:|
| 8K | 11.337s | 701.6 | 37.59 | yes |
| 16K | 12.770s | 1250.0 | 37.99 | yes |
| 32K | 25.101s | 1273.1 | 35.50 | yes |
| 64K | 55.850s | 1145.1 | 30.33 | yes |
| 128K | 135.271s | 946.0 | 24.08 | yes |
| 196K | 246.863s | 793.8 | 19.33 | yes |
| 300K | 464.072s | 646.4 | 15.09 | yes |

These are max-thinking single-request measurements. The current correctness-first SM120 sparse-decode patch does not sustain 50 tok/s across the full context window yet. Hitting 50+ tok/s at 300K requires kernel work: split-KV / multi-CTA sparse decode, better SM120 tiling, and tuned W8A8/MoE configs.
