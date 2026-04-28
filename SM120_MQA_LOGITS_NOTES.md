# SM_120 (RTX Pro 6000 Blackwell Workstation) — DSv4-Flash Indexer Kernel

Notes on the work done to fix the prefill cliff in DeepSeek-V4-Flash on the RTX
Pro 6000 (SM_120) Blackwell Workstation card. Captured for the community —
several pieces here were not obvious from the upstream code and are worth
preserving.

> **Audience.** Anyone running DSv4-Flash (or any DeepSeek-V3.2 / V4 family
> sparse-attention model) on a non-B200 / non-Hopper Blackwell card and trying
> to figure out why prefill is slow.

## TL;DR

DSv4-Flash uses a sparse-attention indexer (`sparse_attn_indexer` →
`mqa_logits_kernel`) that ranks K candidates per query before the real
attention. On B200 this kernel uses tcgen05; on Hopper it uses wgmma; on
SM_120 the upstream repo dispatches to a **scalar fallback that does
byte-by-byte fp8→fp32 conversion with no tensor-core math.** That fallback
was 76% of prefill GPU time at 16k context — and worse, it scaled badly: PP
went *backwards* as context grew (1493 → 966 tok/s at 8k → 16k).

A vectorized rewrite using `uint4` 16-byte loads and the hardware
`__nv_fp8x4_e4m3 → float4` conversion (no tensor cores yet — that's the next
step) gave **~5× e2e prefill** at 16k and erased the backward-scaling cliff.

| Context | Before (tok/s) | After vec (tok/s) | Speedup |
|--------:|---------------:|------------------:|--------:|
|     4k  |        ~1700   |            2999   |   1.8×  |
|     8k  |         1493   |            5320   |   3.6×  |
|    16k  |          966   |            4876   |   5.0×  |
|    32k  |          —     |            4198   |    —    |
|    64k  |          —     |            3093   |    —    |

(TP=4, max-num-batched-tokens=4096, prefill-only firer with prompt salt to bypass
prefix cache. See `bench/` for the firer scripts.)

## Why the SM_120 dispatcher hard-codes a fallback

`csrc/apis/attention.hpp` dispatches by `arch_major`. The relevant branch:

```
arch_major == 9   -> sm90_fp8_mqa_logits   (wgmma — Hopper only)
arch_major == 10  -> sm100_fp8_mqa_logits  (tcgen05 — B200 only)
arch_major == 12  -> sm120_fp8_mqa_logits_fallback   <-- this one
```

The Hopper kernel uses `wgmma.mma_async` which doesn't exist on SM_120. The
B200 kernel uses `tcgen05.mma` (TMEM-based) which doesn't exist on SM_120
either. SM_120 is "Blackwell Workstation" but the tensor-core ISA is closer
to **super-Ada (SM_89)** than to B200 — it has the SM_89-class `mma.sync`
fp8 instructions but neither wgmma nor tcgen05.

So the fallback exists. The original fallback was correct but slow.

## What the original fallback did

`mqa_logits_kernel<out_t, kIsFP4=false>` in
`csrc/sm120_mqa_logits_fallback.cu`:

```cpp
for (int h = 0; h < num_heads; ++h) {        // 64 heads
    for (int d = 0; d < head_dim; ++d) {     // 128 dims, byte at a time
        dot += static_cast<float>(q[q_base + d]) *
               (static_cast<float>(kv[kv_base + d]) * kv_scale);
    }
    sum += fmaxf(dot, 0.0f) * weights[m * num_heads + h];
}
```

Per output element this is `64 × 128 = 8192` 1-byte fp8 loads, `8192` fp8→fp32
conversions in the math pipe, no vectorization. KV is reloaded for every head
even though it is constant across heads in MQA. `kv_scale` is reloaded per `d`.
Grid is capped at 4096 blocks of 256 threads — fine for occupancy on small
shapes, but the per-thread serial work is the bottleneck.

## What the vec kernel does

`mqa_logits_fp8_vec_kernel<out_t, kHeadDim=128>` (same file). Same grid/thread
structure. Per output element:

1. **Stage the KV row into registers once.** `head_dim==128` is `8 × uint4`
   per row. KV is the same across the inner head-loop in MQA, so reuse pays
   for itself: one 16-byte load instead of 16 byte loads × 64 heads.
2. **HW fp8x4 → float4 conversion.** Each `uint4` holds 16 fp8 values; cast a
   `uint32_t` view of each lane to `__nv_fp8x4_e4m3` and use the hardware
   `static_cast<float4>(...)` conversion. This is a single `cvt.rn.f32.e4m3x4`
   per group of 4 fp8 elements — no shifts, no manual exponent decoding.
3. **Hoist `kv_scale` out of the d-loop.** Apply once per dot.
   FP32-accumulate associativity is preserved well below the `1e-3` tolerance
   the unit test asserts.

Net: global-load instruction count drops 16×, and the math pipe spends its
time on multiplies + adds rather than fp8 decode.

```cpp
const uint4* kv_vec_ptr = reinterpret_cast<const uint4*>(
    kv_ptr + static_cast<int64_t>(n) * kHeadDim);
uint4 kv_vecs[kVecsPerHead];
#pragma unroll
for (int v = 0; v < kVecsPerHead; ++v) kv_vecs[v] = kv_vec_ptr[v];

for (int h = 0; h < num_heads; ++h) {
    /* uint4 q4 + uint4 k4 -> 4× float4 each via __nv_fp8x4_e4m3 cast */
    /* dot reduces all 16 lanes; sum += fmaxf(dot * kv_scale, 0) * w[m,h] */
}
```

The FP4 path and the paged path are still on the original scalar kernel — the
vec kernel is gated to `head_dim==128 && !kIsFP4` because that's what DSv4
actually hits. A future PR should bring those onto the same pattern.

## Build flow (the part that bit me)

This repo's venv ships **PyTorch 2.11.0+cu130**. The host has CUDA 12.9. The
CUDAExtension build refuses to mix them:

```
RuntimeError: The detected CUDA version (12.9) mismatches the version that
was used to compile PyTorch (13.0).
```

Do **not** install CUDA 13 on the host — the rig is also running Qwen,
Qwen-embedding, and other models that depend on CUDA 12.9 at runtime.

Instead, rebuild inside the prebuilt builder image (already present on the
rig):

```bash
docker run --rm --gpus all \
  -v /ai/vllm/DS4_SM120_Flash_VLLM_Experiment:/work \
  -w /work \
  vllm-deepseekv4-sm120-builder:latest \
  bash -c "python setup.py build_ext --inplace -j $(nproc)"
```

The `.so` lands in the source tree at
`deep_gemm/_C.cpython-312-x86_64-linux-gnu.so` and the editable install picks
it up automatically. The builder image carries CUDA 13.0 + matching torch
2.11.0+cu130 + ninja, so the host stays untouched.

NVCC arch flag is `-gencode=arch=compute_120f,code=sm_120f` (note the trailing
`f` — that's the family-PTX form, distinct from plain `sm_120`).

## Validation

`tests/test_attention.py::test_mqa_logits` enumerates 32 configs across:

- `seq_len ∈ {2048, 4096}`
- `seq_len_kv ∈ {4096, 8192}`
- `num_heads = 64`, `head_dim = 128`
- `compressed_logits ∈ {True, False}`
- `dtype ∈ {float32, bfloat16}`

Each config asserts `(out - ref).abs().max() < 1e-3` against an `einsum`
reference. All 32 pass with the vec kernel. The venv has no `pytest`, so run
directly:

```bash
cd /ai/vllm/DS4_SM120_Flash_VLLM_Experiment
python -c "import sys; sys.path.insert(0,'tests'); \
  from test_attention import test_mqa_logits; test_mqa_logits()"
```

## How to profile

vLLM v1 exposes a torch profiler via HTTP:

```bash
# Already wired into the launcher — see /ai/vllm/dsv4-flash-vllm-sm120.sh
curl -X POST http://localhost:8000/start_profile
# ... fire one prefill ...
curl -X POST http://localhost:8000/stop_profile
ls /tmp/vllm_profile_dsv4/profiler_out_0.txt   # rank 0 trace
```

The text profile has columns for `Self CUDA %` and `Self CUDA total`. Sort by
those to find what actually takes the time. In the original trace,
`vllm::sparse_attn_indexer` → `deep_gemm::sm120_fallback::mqa_logits_kernel`
sat at **20.4 s of 26.8 s (76.13%)** for a single 16k-token prefill.

After the vec kernel landed, that same kernel is roughly 4-5× lower; further
work below.

## FP8 MMA tile kernel — Q-stationary block design

The vec kernel still does pairwise multiplies in the math pipe. SM_120 has the
SM_89-class `mma.sync.aligned.m16n8k32.f32.e4m3.e4m3.f32` fp8 tensor-core
instruction. We added an MMA path gated behind `DG_SM120_MMA_INDEXER=1`. Two
iterations:

**v1 (per-tile, 2026-04-26 morning) — REJECTED.** 1 warp = 1 (m, 8-n) tile,
A-operand reads from gmem inside the inner mma loop. Numerics: passed
(diff ~7e-4). Kernel-only: 52–77 TFLOPS vs vec's ~22 TFLOPS. **E2E: regressed.**
At 8k context, prefill went 5320 → 1770 tok/s. Cause: Q[m, :, :] re-read from
gmem once per (m, 8-n) tile, blowing the L1 reuse window before warps in the
same SM could amortize.

**v2 (Q-stationary, 2026-04-26 afternoon) — SHIPPED.** 1 block = 1 m, 8 warps
share `Q[m, :, :]` in shared memory. Each warp picks an n-tile in stride
(8-tile chunks). Block layout:

- 256 threads, 8 warps
- `__shared__ uint32_t q_smem[64 × 36]`  (head dim 128 fp8 = 32 u32 + pad 4)
- `__shared__ float w_smem[64]`
- One block per m, grid = `dim3(seq_len)`, each block loads Q[m] once

The pad = 4 u32 (row stride 36) is what makes the A-operand reads
**bank-conflict-free**. Bank index for `q_smem[h][k_u32]` is
`(h × 36 + k_u32) mod 32 = (h × 4 + k_u32) mod 32`. With lane t = (g, l)
where t = 4g + l, the read offset becomes `(h_base × 4 + lane + const) mod 32`
— a unique bank per lane across the warp. Pad = 1 leaves a residual 4-way
conflict; pad = 4 eliminates it.

Numerics: 32/32 configs at diff ~6.8e-4 (`tests/test_mma_indexer.py`).

E2E sweep (TP=4, single sequential firer, prompt salt at start):

| Tokens | Vec t/s | MMA Q-stat t/s |     Δ |
|-------:|--------:|---------------:|------:|
|    860 |   2819  |          2698  |   −4% |
|   3320 |   3320  |          3318  |  flat |
|   6598 |   3144  |          3180  |   +1% |
|  13151 |   2907  |          3027  |   +4% |
|  26262 |   2621  |          2893  | **+10%** |

The 8k regression from v1 is gone. Long-context modestly improves. Why so
modest given the 3× kernel speedup? Because the indexer is no longer the
dominant fraction of prefill time after the vec kernel landed — Amdahl
caps the e2e gain at whatever was left to win.

The MMA kernel is enabled by default in `/ai/vllm/dsv4-flash-vllm-sm120.sh`
via `DG_SM120_MMA_INDEXER=1`. Set to `0` to fall back to the vec kernel.

## What's next

Indexer is no longer the bottleneck. Profile to find the new top-of-list —
expect MoE GEMMs, BF16 BMMs, or all-reduce traffic over PCIe. Updates here as
the next investigation lands.

## Things to watch on this rig

- 4× RTX Pro 6000 Workstation, **no NVLink, PCIe Gen 5.** All-reduce traffic
  goes over PCIe — `--enable-expert-parallel` does not help and tends to hurt.
- Worker process names collide across vLLM model launches (`VLLM::Worker_TP*`
  is just a `prctl(PR_SET_NAME)` on every multiproc worker). Use PID lineage
  from the launcher, not `pkill VLLM::`.
- `max-num-batched-tokens=4096` is the largest stable value; `8192` crashes
  with the current allocator under TP=4.
- Decode at ~51 tok/s is acceptable; prefill is the active bottleneck.

## File map

- `csrc/sm120_mqa_logits_fallback.cu` — kernels (this work).
- `csrc/apis/attention.hpp` — dispatch by arch_major; SM_120 → `_fallback`.
- `csrc/jit_kernels/impls/sm120_mqa_logits_fallback.hpp` — C++ entry-point
  declarations.
- `tests/test_attention.py::test_mqa_logits` — validation harness.
- `bench/` — scripts/launchers, including the variable-size prefill firer
  used for the throughput sweep.

## Questions worth asking

If you're reading this and you have an SM_120 box: please run the e2e sweep
yourself and post numbers. The 5× number is on this specific rig with this
specific launcher config; your prefill mix may differ.

If you're a kernel author: the FP4 path and paged path are still scalar.
They're a less hot path on DSv4 (fp8 indexer is what runs), but a community
PR that brings the same pattern to those would be welcome.
