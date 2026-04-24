# DeepSeek-V4-Flash on RTX Pro 6000 Blackwell (SM_120) — exact recipe

End-to-end reproducible recipe for running **DeepSeek-V4-Flash (FP8)** with
**sglang 0.0.0.dev0 / `lmsysorg/sglang:deepseek-v4-blackwell`** on a
workstation with **4× NVIDIA RTX Pro 6000 Blackwell Workstation (SM_120)**.

Upstream `flash_mla.cuda.sparse_decode_fwd` only ships SM_90 (WGMMA) and
SM_100 (TCGEN05) kernels — both of which fail on Blackwell workstation
silicon.  This repo ships a **portable CUDA-core SM_120 sparse-decode
kernel** and a **runtime monkey-patch** that swaps it into
`flash_mla.flash_mla_with_kvcache` *without touching the sglang image*.
The container image stays read-only; only `PYTHONPATH` and
`sitecustomize.py` are injected.

What you get, verified in this tree:

| Feature                               | Working |
| ------------------------------------- | :-----: |
| OpenAI-compatible chat completions    | yes     |
| `finish_reason = stop / length`       | yes     |
| Parallel function / tool calling      | yes     |
| `reasoning_content` split by parser   | yes     |
| 393 216-token context, MLA sparse     | yes     |
| TP=4 across 4 SM_120 GPUs             | yes     |
| Keeps the sglang image untouched      | yes     |

Verified with `curl` probes at the bottom of this document.

> The kernel is a correctness-first V1: CUDA-core BF16 dot products,
> one CTA per request × per 16-head block, no split-KV, no CUDA
> graphs, no MTP / EAGLE draft.  It is fast enough for interactive
> use (single-request decode ≈ 5 tok/s on this setup); if you need
> throughput you'll want to port a WGMMA-style kernel.

---

## 1. Hardware

| Component      | Exact value used when writing this doc                               |
| -------------- | -------------------------------------------------------------------- |
| GPUs           | 4× **NVIDIA RTX PRO 6000 Blackwell Workstation Edition**, 97 887 MiB each |
| Compute cap    | **12.0** (a.k.a. `sm_120` / `sm_120a`)                               |
| Other GPU      | 1× RTX 3090 at PCI index 1 — **masked out** via `CUDA_VISIBLE_DEVICES=0,2,3,4` |
| NVIDIA driver  | **580.126.18**                                                       |
| CPU / RAM      | ≥ 32 cores, ≥ 256 GB system RAM recommended (model weights are 274 GB on disk, FP8 on GPU needs ~300 GB across 4 GPUs) |
| Host disk      | ≥ 300 GB free for the FP8 checkpoint                                 |

---

## 2. Host OS

| Item                  | Exact value                                     |
| --------------------- | ----------------------------------------------- |
| Distribution          | **Pop!_OS 22.04 LTS** (or any Ubuntu 22.04 base)|
| Kernel                | **6.12.10-76061203-generic**                    |
| Host CUDA toolkit     | **CUDA 12.8** (`nvcc 12.8.93`) — needed only for the host-side dev venv; inside the container we use the image's CUDA 12.9 |
| Docker                | **28.2.2**                                      |
| NVIDIA Container Tk   | installed (`--gpus all` works)                  |
| Python (host)         | **3.10.12** (for the host venv & unit tests)    |
| `git`, `curl`, `jq`   | any recent version                              |

Install NVIDIA container toolkit once:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 3. sglang Docker image

| Item                 | Exact value                                                   |
| -------------------- | ------------------------------------------------------------- |
| Image reference      | `lmsysorg/sglang:deepseek-v4-blackwell`                       |
| Image id used here   | `sha256:4d7396f62f797370bafaf6f57524344bee4948b149855ab5285b7b67790a160a` (2026-04-24) |
| Python inside image  | **3.12.3**                                                    |
| CUDA inside image    | **12.9** (`nvcc 12.9.86`)                                     |
| `torch`              | **2.9.1+cu129**                                               |
| `sglang`             | **0.0.0.dev0** (pre-release DSv4 build)                       |
| `flash_mla`          | **1.0.0+71c7379**                                             |
| `flashinfer-python`  | **0.6.8**                                                     |
| `triton`             | **3.5.1**                                                     |
| `deep_gemm`          | **2.4.2+7f2a703** (skipped on SM_120)                          |
| `tilelang`           | **0.1.8**                                                     |
| `xgrammar`           | **0.1.27**                                                    |
| `fastapi` / `uvicorn`| 0.128.0 / 0.40.0                                              |

Pull it once:

```bash
docker pull lmsysorg/sglang:deepseek-v4-blackwell
```

The image is **not modified** by anything in this repo.

---

## 4. Model checkpoint

DeepSeek-V4-Flash is published by `sgl-project` in native FP8 (e4m3 weights,
UE8M0 scales, 128×128 block quantisation):

| Item            | Value                                                                |
| --------------- | -------------------------------------------------------------------- |
| Hugging Face ID | `sgl-project/DeepSeek-V4-Flash-FP8`                                  |
| Architecture    | `DeepseekV4ForCausalLM`                                              |
| Layers          | 43                                                                   |
| Hidden size     | 4096                                                                 |
| Heads           | 64 attention, 1 KV (MQA) — 512-d head_dim                            |
| MoE             | 256 routed experts, 6/token, 1 shared                                |
| Vocab           | 129 280                                                              |
| Max position    | 1 048 576 (we launch with `--context-length 393216`)                 |
| Quant fmt       | `quant_method=fp8`, `fmt=e4m3`, `scale_fmt=ue8m0`, block [128, 128]  |
| Shards          | 46 × `.safetensors` (~6.2 GB each, ≈ 274 GB total)                   |
| Chat template   | `tool_chat_template_deepseekv4.jinja` (sha256 `f7b71796...aa27`)     |

Download once to a local directory (~300 GB free):

```bash
pip install --user "huggingface_hub[cli]"
export HF_HUB_ENABLE_HF_TRANSFER=1            # optional, faster
export MODEL_DIR=/mnt/llm_models/DeepSeek-V4-Flash-FP8
huggingface-cli download sgl-project/DeepSeek-V4-Flash-FP8 \
  --local-dir "$MODEL_DIR" \
  --local-dir-use-symlinks False
```

The `tool_chat_template_deepseekv4.jinja` ships inside the checkpoint — no
manual copy needed.

---

## 5. This repo (the SM_120 kernel)

### 5.1 Layout

```
deepseek-v4-kernel/
├── csrc/
│   ├── api/
│   │   ├── api.cpp                        # pybind11 module ("deepseek_v4_kernel.cuda")
│   │   ├── sparse_decode.cpp              # ATen wrapper, shape/dtype checks
│   │   └── sparse_decode.h
│   ├── common/
│   │   ├── defines.h                      # CUDA error macros, ceil_div
│   │   └── params.h                       # SparseAttnDecodeParams (binary-compatible with flash_mla)
│   ├── sm120/decode/
│   │   ├── sparse_decode.h                # kernel launch entry (C++ side)
│   │   ├── sparse_decode_instantiation.cu # dim3 grid/block dispatch
│   │   └── sparse_decode_kernel.cuh       # *** the CUDA-core SM_120 kernel ***
│   └── cutlass/                           # vendored CUTLASS (include-only, for bf16 types)
├── deepseek_v4_kernel/
│   ├── __init__.py                        # re-exports: sparse_decode_fwd, patch_flash_mla
│   ├── ops.py                             # Python wrapper of the pybind11 function
│   ├── _patch.py                          # runtime monkey-patch (flash_mla + sglang indexer)
│   └── sitecustomize_hook.py              # auto-load body for sitecustomize.py
├── scripts/
│   ├── build_in_sglang_docker.sh          # builds cpython-3.12 .so inside a throwaway container
│   ├── launch_dsv4_flash_sm120.sh         # the canonical launcher
│   └── sglang_entrypoint.py               # (optional) alternative launcher shim
├── tests/
│   ├── reference.py                       # pure-PyTorch oracle (matches sglang MODEL1 layout)
│   └── test_sparse_decode.py              # 7 parametrised correctness cases
├── setup.py                               # CUDAExtension: -gencode arch=compute_120a,code=sm_120a
├── pyproject.toml                         # torch>=2.5
└── DEEPSEEK-V4-FLASH.md                   # <-- this file
```

### 5.2 Kernel specifics

* **Model variant**: DeepSeek-V4-Flash = flash_mla `MODEL1` (NOT `V32`).
  * `HEAD_DIM_K = 512` = `HEAD_DIM_NOPE (448) + HEAD_DIM_ROPE (64)`
  * `HEAD_DIM_V = 512`
  * `QUANT_TILE = 64`, `NUM_SCALES = 8` (7 active + 1 padding)
* **KV page layout** (`page_block_size = P = 256`):
  ```
  [ 0 .. P*576 )          nope_rope section
      per token:  448 B FP8 e4m3 NoPE  |  128 B BF16 RoPE  =  576 B
  [ P*576 .. P*(576+8) )  scale section
      per token:  7 UE8M0 bytes + 1 pad = 8 B
                  scale_i = 2^(byte_i − 127)
                  covers NoPE[i*64 .. (i+1)*64)
  ```
  The sglang tensor's `.view(num_pages, P, 1, 584)` is **cosmetic** —
  the kernel reads raw bytes using `kv.stride(0)` (bytes-per-page) and
  the compile-time offsets above.
* **Grid**: `(batch * s_q, ceil(h_q / 16))`, 128 threads / CTA.
* **Shared memory**: ≈ 84 KB / CTA (fits the 99 KB/SM budget).
* **Sink + top-k mask**: same semantics as upstream `flash_mla`.
* **Q dtype**: BF16.  **KV dtype**: uint8 / fp8_e4m3fn (checked).

### 5.3 Building

Two builds are needed — host (for unit tests) and Docker-image-Python
(for runtime):

```bash
# One-time: host venv (Python 3.10, CUDA 12.8).  Only used for pytest.
cd deepseek-v4-kernel
python3.10 -m venv .venv --system-site-packages   # pulls in host torch
.venv/bin/pip install -U pip wheel ninja pytest
.venv/bin/python setup.py build_ext --inplace
.venv/bin/python -m pytest tests/test_sparse_decode.py -v     # must print 7 passed

# Runtime: Docker-Python 3.12 .so, built inside a throwaway container.
# Output goes to ./build-docker/ (mounted RO into the server container).
scripts/build_in_sglang_docker.sh
# -> build-docker/deepseek_v4_kernel/{__init__.py, ops.py, _patch.py,
#                                     sitecustomize_hook.py,
#                                     cuda.cpython-312-x86_64-linux-gnu.so}
# -> build-docker/sitecustomize.py   (auto-loaded on every interpreter start)
```

Neither step writes to the sglang image nor to the host system site.

---

## 6. Runtime patch (how the kernel actually gets called)

`deepseek_v4_kernel.patch_flash_mla()` performs three monkey-patches (all
idempotent; safe to call from every worker process):

1. `flash_mla.flash_mla_interface.flash_mla_with_kvcache` — every sparse
   decode call is routed to our CUDAExtension when:
   `indices is not None` **AND** `is_fp8_kvcache=True` **AND**
   `q.element_size() == 2` (bf16) **AND** `device is SM 12.x`.  Any other
   call falls through to the original function.
2. `vllm.third_party.flashmla.flash_mla_interface.flash_mla_with_kvcache`
   — same wrapper, for completeness when vllm is present.
3. `sglang.srt.layers.attention.{nsa.tilelang_kernel,
   compressed.indexer}.{tilelang_fp8_paged_mqa_logits,
   fp8_paged_mqa_logits_torch}` — squeezes a stray trailing
   `1`-dim on `seq_lens` that sglang introduces to match the deep_gemm
   (SM_100-only) signature.  Without this the tilelang fallback asserts
   `seq_lens.shape == (batch,)`.

The patch is auto-applied by `build-docker/sitecustomize.py`, which
Python imports on every interpreter start (and every forked/spawned
worker).  You can also apply it manually:

```python
import deepseek_v4_kernel
deepseek_v4_kernel.patch_flash_mla()
```

---

## 7. Launch script

`scripts/launch_dsv4_flash_sm120.sh` bind-mounts three things read-only into
`lmsysorg/sglang:deepseek-v4-blackwell` and starts the OpenAI server:

```
-v ${MODEL_DIR}:/workspace/model:ro         # 46 safetensor shards + config + template
-v ${DSV4_BUILD_DIR}:/dsv4:ro               # our cpython-3.12 .so + sitecustomize.py
-e PYTHONPATH=/dsv4                         # triggers sitecustomize.py auto-load
```

The sglang CLI flags it passes are the *minimum set* that avoids every
deep_gemm / mxfp4 / SM_100-only path:

| Flag                                     | Why                                                 |
| ---------------------------------------- | --------------------------------------------------- |
| `--tensor-parallel-size 4`               | 4× RTX Pro 6000                                     |
| `--context-length 393216`                | fits within KV budget with `mem-fraction-static 0.85` |
| `--mem-fraction-static 0.85`             | leaves headroom for activations                     |
| `--kv-cache-dtype fp8_e4m3`              | matches DSv4 MODEL1 layout expected by kernel       |
| `--attention-backend compressed`         | routes into the MLA sparse path the kernel patches  |
| `--page-size 256`                        | DSv4 expected page_block_size                       |
| `--chunked-prefill-size 8192`            | default OK                                          |
| `--fp8-gemm-backend triton`              | avoid deep_gemm (SM_100-only)                       |
| `--moe-runner-backend triton`            | avoid deep_gemm MoE                                 |
| `--tool-call-parser deepseekv4`          | parallel tool-calls work                            |
| `--reasoning-parser deepseek-v4`         | splits `<think>...</think>` into `reasoning_content`|
| `--chat-template /workspace/model/...`   | honors `chat_template_kwargs.thinking`              |
| `--disable-cuda-graph`                   | V1 kernel doesn't implement CUDA graph capture      |
| `--watchdog-timeout 3600`                | cold-start of 274 GB FP8 weights takes ~60 s/GPU    |

And the env variables that disable the paths our kernel replaces:

| Env var                                  | Effect                                              |
| ---------------------------------------- | --------------------------------------------------- |
| `CUDA_VISIBLE_DEVICES=0,2,3,4`           | skip the RTX 3090 at PCI index 1                    |
| `TORCH_CUDA_ARCH_LIST=12.0`              | keep JIT arch aligned with SM_120                   |
| `SGLANG_ENABLE_JIT_DEEPGEMM=0`           | skip deep_gemm JIT paths                            |
| `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`       | skip precompile                                     |
| `SGLANG_DSV4_FP4_EXPERTS=0`              | use the FP8 expert code path                        |
| `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0`       | use torch HC-prenorm                                |
| `SGLANG_OPT_USE_TILELANG_INDEXER=1`      | tilelang MQA logits instead of deep_gemm            |
| `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1`    | metadata-lookup fallback                            |
| `SGLANG_OPT_USE_TILELANG_SWA_PREPARE=1`  | tilelang SWA                                        |
| `SGLANG_OPT_USE_TILELANG_MHC_PRE=1`      | tilelang MHC pre                                    |
| `SGLANG_OPT_USE_TILELANG_MHC_POST=1`     | tilelang MHC post                                   |
| `SGLANG_ENABLE_SPEC_V2=True`             | leave spec-v2 enabled (EAGLE is not used)           |
| `SGLANG_SET_CPU_AFFINITY=1`              | pin TP workers to separate NUMA nodes               |
| `NCCL_P2P_DISABLE=0`, `NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME=lo`, `NCCL_DEBUG=WARN`, `NCCL_CUMEM_HOST_ENABLE=0` | Single-node NCCL settings |

> We **do not** enable `--speculative-algorithm` (EAGLE).  sglang's DSv4
> draft worker has a pre-existing shape bug (`req_pool_indices=[1]`,
> `seq_lens=[1]`, `out_cache_loc=[3]` mismatch in
> `deepseek_v4_backend_radix.py:424`).  Disabling it is the correct
> workaround today; re-enable once sglang fixes the draft-path shapes.

---

## 8. End-to-end bring-up (from zero)

```bash
# 0. Prereqs: driver ≥ 580, Docker, NVIDIA container toolkit already installed.
#    Checkpoint downloaded to $MODEL_DIR.

# 1. Clone the repo next to where you want the build artefacts.
git clone <this repo url> deepseek-v4-kernel
cd deepseek-v4-kernel
git submodule update --init --recursive       # pulls in csrc/cutlass

# 2. Pull the sglang image (pinned tag).
docker pull lmsysorg/sglang:deepseek-v4-blackwell

# 3. (Optional but recommended) Unit tests in a host venv.
python3.10 -m venv .venv --system-site-packages
.venv/bin/pip install -U pip wheel ninja pytest
.venv/bin/python setup.py build_ext --inplace
.venv/bin/python -m pytest tests/test_sparse_decode.py -v
#   expect: 7 passed

# 4. Build the cpython-3.12 .so for the image. Writes to ./build-docker/.
scripts/build_in_sglang_docker.sh
ls build-docker/deepseek_v4_kernel/
#   cuda.cpython-312-x86_64-linux-gnu.so  __init__.py  _patch.py
#   ops.py  sitecustomize_hook.py
ls build-docker/sitecustomize.py
#   build-docker/sitecustomize.py

# 5. Launch the server (cold start ~60 s/GPU while 274 GB FP8 is loaded).
MODEL_DIR=/mnt/llm_models/DeepSeek-V4-Flash-FP8 PORT=8000 \
  scripts/launch_dsv4_flash_sm120.sh

# 6. From another shell: wait for /health and hit the OpenAI endpoint.
until curl -sf http://127.0.0.1:8000/health >/dev/null; do sleep 5; done
echo "server is up"
```

---

## 9. Smoke tests (exactly what is verified in this tree)

### 9.1 Coherent completion

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","temperature":0.0,"max_tokens":80,
       "messages":[{"role":"user","content":"In one sentence, why is the sky blue?"}]}' | jq -r .choices[0].message.content
```

Expected (actual output observed here):

> The sky appears blue because sunlight is scattered in all directions by the
> gases and particles in Earth's atmosphere, and blue light is scattered
> more than other colors due to its shorter, smaller wavelengths.

### 9.2 Parallel tool calls

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"deepseek-v4-flash","temperature":0.0,"max_tokens":300,
  "messages":[{"role":"user","content":"What is the weather in Paris and Tokyo? Use tools."}],
  "tools":[{"type":"function","function":{"name":"get_weather","description":"Current weather",
    "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
  "tool_choice":"auto"}' | jq .choices[0].message.tool_calls
```

Expected: **two** tool calls, `{"city":"Paris"}` and `{"city":"Tokyo"}`,
`finish_reason = tool_calls`.

### 9.3 Reasoning content

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"deepseek-v4-flash","temperature":0.0,"max_tokens":600,
  "chat_template_kwargs":{"thinking":true},
  "messages":[{"role":"user","content":"A farmer has 17 sheep. All but 9 die. How many are left?"}]}' \
  | jq '{reasoning: .choices[0].message.reasoning_content, content: .choices[0].message.content}'
```

Expected: `reasoning_content` holds the chain-of-thought, `content` holds
the final answer (`9`), `finish_reason = stop`.

### 9.4 Kernel unit tests

```bash
.venv/bin/python -m pytest tests/test_sparse_decode.py -v
# test_matches_reference[64-64]   PASSED
# test_matches_reference[64-128]  PASSED
# test_matches_reference[256-64]  PASSED
# test_matches_reference[256-128] PASSED
# test_matches_reference[2048-64] PASSED
# test_matches_reference[2048-128]PASSED
# test_topk_length_mask           PASSED
```

---

## 10. Known limitations / gotchas

* **CUDA-graph disabled** — V1 kernel isn't graph-capturable
  (`cudaFuncSetAttribute` inside the launcher).  You pay a kernel-launch
  overhead every decode step.
* **No split-KV** — `num_sm_parts == 1`.  For very long contexts
  (> ~200 K tokens × large batch) decode will become launch-bound.
* **No MTP / EAGLE** — sglang's DSv4 draft worker has a separate shape
  bug; once fixed upstream, enable with `--speculative-algorithm eagle`
  and friends.
* **Only MQA** — `h_kv == 1` enforced in `sparse_decode.cpp`.
* **`mem-fraction-static 0.85`** leaves ~14 GB free per GPU for
  activations; cut it to 0.80 if you see OOM at peak concurrency.

---

## 11. Version pin summary (for `pip freeze`-style reproducibility)

| Component                | Pin                                           |
| ------------------------ | --------------------------------------------- |
| Host CUDA                | 12.8 (12.8.93)                                |
| Host PyTorch (unit tests)| 2.10.0+cu128                                  |
| Image tag                | `lmsysorg/sglang:deepseek-v4-blackwell` `sha256:4d7396f62f79...` |
| Image CUDA               | 12.9.86                                       |
| Image PyTorch            | 2.9.1+cu129                                   |
| Image sglang             | 0.0.0.dev0                                    |
| Image flash_mla          | 1.0.0+71c7379                                 |
| Image flashinfer-python  | 0.6.8                                         |
| Image triton             | 3.5.1                                         |
| Image tilelang           | 0.1.8                                         |
| Image xgrammar           | 0.1.27                                        |
| Model                    | `sgl-project/DeepSeek-V4-Flash-FP8`           |
| Chat template sha256     | `f7b717962a94afd9561357aa498fdd25c6eb00995a8d30dc65dcfa0e78e7aa27` |
| Driver                   | 580.126.18                                    |
| nvidia-container-toolkit | latest from NVIDIA stable repo                |

---

## 12. Files you must ship unchanged

Everything in this repository, **plus**:

1. The local copy of the FP8 checkpoint at `${MODEL_DIR}`.
2. `${DSV4_KERNEL_DIR}/build-docker/` produced by
   `scripts/build_in_sglang_docker.sh`.  This directory is the only
   thing mounted into the runtime container alongside the model.

Nothing else is required.  No `pip install` inside the container, no
edits to the sglang image, no rebuild of `flash_mla`.

---

## 13. Quick troubleshooting

| Symptom                                                    | Fix                                                                       |
| ---------------------------------------------------------- | ------------------------------------------------------------------------- |
| `RuntimeError: Unsupported architecture for sparse decode` | `deepseek_v4_kernel.patch_flash_mla` didn't run; check `PYTHONPATH=/dsv4` and that `/dsv4/sitecustomize.py` exists in the container |
| `RuntimeError: DeepSeek-V4-Flash expects 584 bytes / token`| KV-cache dtype / page-size mismatch; verify `--kv-cache-dtype fp8_e4m3` and `--page-size 256` |
| Garbled output (`"题 题 题 …"`)                             | Old kernel was built; re-run `scripts/build_in_sglang_docker.sh` and restart the container |
| `assert seq_lens.shape == (batch_size,)` in tilelang       | indexer fallback wrapper missing; make sure `_patch.py` is the one in this repo (it patches `tilelang_fp8_paged_mqa_logits`) |
| TP workers OOM at > 200 K context                          | Drop `--mem-fraction-static` to 0.80 and/or `--max-running-requests` to 4 |
| 3090 shows up in `torch.cuda.device_count()`               | `CUDA_VISIBLE_DEVICES=0,2,3,4` not exported — check `docker inspect`      |

To get verbose kernel tracing (one log line per decode call), set

```bash
-e DSV4_KERNEL_TRACE=1 -e DSV4_KERNEL_STRICT=1
```

in the launcher (already wired into `scripts/launch_dsv4_flash_sm120.sh`
as commented-out lines; flip them on if you need to debug a new sglang
build).

---

## 14. License / attribution

* CUTLASS is vendored under `csrc/cutlass/` (NVIDIA BSD-3).
* The kernel, monkey-patch, and scripts in this repo are Apache-2.0.
* The sglang image and the DeepSeek-V4-Flash weights are governed by
  their respective upstream licenses (DeepSeek Model License + sgl-project
  Apache-2.0 + sglang Apache-2.0).
