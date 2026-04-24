# deepseek-v4-flash-sm120

SM_120 (NVIDIA Blackwell workstation — RTX Pro 6000 / GeForce RTX 50xx)
sparse-decode kernel + runtime patch that makes
**[DeepSeek-V4-Flash (FP8)](https://huggingface.co/sgl-project/DeepSeek-V4-Flash-FP8)**
run end-to-end under the official
`lmsysorg/sglang:deepseek-v4-blackwell` Docker image — including chat
completions, parallel tool/function calling, and `reasoning_content`
splitting — **without modifying the image or installing anything system-wide**.

Upstream `flash_mla.cuda.sparse_decode_fwd` only ships SM_90 (WGMMA) and
SM_100 (TCGEN05) kernels, both of which fail on the Blackwell
workstation silicon.  This repo ships a portable CUDA-core SM_120
kernel with the exact `[nope_rope][scales]` page layout DSv4-Flash
uses, and a monkey-patch that swaps it into `flash_mla` at runtime via
`PYTHONPATH=/dsv4 + sitecustomize.py`.

## TL;DR

```bash
# prereqs: nvidia-driver >= 580, docker, nvidia-container-toolkit
git clone https://github.com/0xSero/deepseek-v4-flash-sm120.git
cd deepseek-v4-flash-sm120
git submodule update --init --recursive           # pulls in csrc/cutlass

# 1) Pull the sglang image (pinned tag).
docker pull lmsysorg/sglang:deepseek-v4-blackwell

# 2) Download the FP8 checkpoint (~274 GB) to $MODEL_DIR.
export MODEL_DIR=/path/to/DeepSeek-V4-Flash-FP8
huggingface-cli download sgl-project/DeepSeek-V4-Flash-FP8 \
    --local-dir "$MODEL_DIR" --local-dir-use-symlinks False

# 3) Build the cpython-3.12 .so for the image (ephemeral container).
scripts/build_in_sglang_docker.sh

# 4) Launch the OpenAI-compatible server on 4x SM_120 GPUs.
MODEL_DIR=$MODEL_DIR PORT=8000 scripts/launch_dsv4_flash_sm120.sh

# 5) Smoke test.
curl -s http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-v4-flash","temperature":0,"max_tokens":80,
         "messages":[{"role":"user","content":"Why is the sky blue?"}]}' \
    | jq -r .choices[0].message.content
```

Full reproducible recipe with exact versions, hardware, every CLI flag
and every env var: **[DEEPSEEK-V4-FLASH.md](./DEEPSEEK-V4-FLASH.md)**.

Benchmark methodology + results (prefill / decode throughput at
1K..256K tokens, NIAH accuracy, GSM8K reasoning with `thinking=true`):
**[BENCHMARKS.md](./BENCHMARKS.md)**.

## Verified working

| Feature                                     | Status |
| ------------------------------------------- | :----: |
| OpenAI-compatible chat completions          | yes    |
| Parallel function/tool calling              | yes    |
| `reasoning_content` split by parser         | yes    |
| 4x SM_120 tensor parallel, FP8 KV cache     | yes    |
| 393 216-token context window                | yes    |
| Keeps the sglang image untouched            | yes    |

## Repo layout

```
deepseek-v4-flash-sm120/
|-- csrc/                      CUDA kernel source (+ vendored CUTLASS)
|-- deepseek_v4_kernel/        Python package: wrapper + monkey-patch
|-- scripts/                   build_in_sglang_docker.sh + launch script
|-- bench/                     throughput / NIAH / GSM8K benchmarks
|-- tests/                     pytest correctness tests (reference oracle)
|-- setup.py  pyproject.toml   build config (sm_120a via CUDA 12.8+)
|-- DEEPSEEK-V4-FLASH.md       exhaustive recipe
|-- BENCHMARKS.md              benchmark results + methodology
```

## Unit tests (optional)

```bash
python3.10 -m venv .venv --system-site-packages
.venv/bin/pip install -U pip wheel ninja pytest
.venv/bin/python setup.py build_ext --inplace
.venv/bin/python -m pytest tests/test_sparse_decode.py -v   # 7 passed
```

## License

* Kernel / scripts / docs: Apache-2.0 (see `LICENSE`).
* Vendored `csrc/cutlass/`: NVIDIA BSD-3 (see `csrc/cutlass/LICENSE.txt`).
* Weights and upstream sglang image are governed by their own licenses.
