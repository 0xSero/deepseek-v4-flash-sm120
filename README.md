# deepseek-v4-flash-sm120

Runtime patch for serving **DeepSeek-V4-Flash FP8** on NVIDIA SM120 / RTX PRO 6000 Blackwell workstation GPUs with SGLang.

The stock `lmsysorg/sglang:deepseek-v4-blackwell` image contains FlashMLA sparse-decode kernels for SM90/SM100, but not SM120. On RTX PRO 6000 it can fail with:

```text
RuntimeError: Unsupported architecture for sparse decode fwd
```

This repo builds a small SM120 CUDA extension and injects it at runtime with:

```bash
-v ./build-docker:/dsv4:ro
-e PYTHONPATH=/dsv4
```

No SGLang image rebuild and no install inside the container are required.

## Correct current launch recipe

Important corrections:

- Do **not** pass `--chat-template` for DeepSeek-V4. SGLang has built-in DeepSeek-V4 OpenAI chat encoding via `encoding_dsv4.py` when `--tool-call-parser deepseekv4` or the DeepseekV4 architecture is detected.
- Do **not** disable CUDA graphs. Use `--cuda-graph-max-bs 32`.
- EAGLE speculative decoding works in the current recipe: one step, topk 1, two draft tokens.
- Use the SM120 patch mount and `PYTHONPATH=/dsv4`; otherwise FlashMLA sparse decode falls back to upstream and fails on SM120.

```bash
MODEL_DIR=/mnt/llm_models/DeepSeek-V4-Flash-FP8 PORT=8000 \
  scripts/launch_dsv4_flash_sm120.sh
```

Equivalent core SGLang flags:

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

## Build

```bash
git clone https://github.com/0xSero/deepseek-v4-flash-sm120.git
cd deepseek-v4-flash-sm120
git submodule update --init --recursive

docker pull lmsysorg/sglang:deepseek-v4-blackwell
scripts/build_in_sglang_docker.sh
```

The build writes the runtime package to `build-docker/`:

```text
build-docker/sitecustomize.py
build-docker/deepseek_v4_kernel/cuda.cpython-312-x86_64-linux-gnu.so
build-docker/deepseek_v4_kernel/_patch.py
```

## Model

Use the FP8 checkpoint:

```bash
export MODEL_DIR=/mnt/llm_models/DeepSeek-V4-Flash-FP8
huggingface-cli download sgl-project/DeepSeek-V4-Flash-FP8 \
  --local-dir "$MODEL_DIR" \
  --local-dir-use-symlinks False
```

DeepSeek's official V4 repos do **not** ship a Jinja chat template. They ship `encoding/encoding_dsv4.py`; current SGLang includes an adapted encoder at `sglang.srt.entrypoints.openai.encoding_dsv4` and uses it automatically for this model path/parser.

## Smoke test

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","temperature":0,"max_tokens":32,
       "messages":[{"role":"user","content":"Say OK only."}]}'
```

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Unsupported architecture for sparse decode fwd` | Patch did not load. Check `-v ./build-docker:/dsv4:ro`, `PYTHONPATH=/dsv4`, and `build-docker/sitecustomize.py`. |
| No `deepseek_v4_kernel.patch_flash_mla installed` log | `sitecustomize.py` was not imported; inspect `PYTHONPATH`. |
| Tool calls or thinking formatting looks wrong | Do not use a custom Jinja template; keep `--tool-call-parser deepseekv4` and `--reasoning-parser deepseek-v4`. |
| OOM near full context | Lower `--mem-fraction-static` or `--max-running-requests`. |

## License

Kernel, scripts, and docs are Apache-2.0. CUTLASS under `csrc/cutlass/` keeps its NVIDIA BSD-3 license. Model weights and SGLang image keep their upstream licenses.

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
