# DeepSeek-V4-Flash on SM_120 — Resume Notes

Last updated: 2026-04-27 — split-K bf16 MMA path for sparse_mla_score
landed behind `DG_SM120_SCORE_MMA_SPLITK=1` (default OFF, **+5-7% prefill at
long ctx, no semantic regression**); the non-splitk SCORE_MMA stays default
OFF (regressed). hc_prenorm TF32 MMA kernel landed behind
`DG_SM120_HC_PRENORM_MMA=1`; TP=2 ruled out as VRAM-infeasible.

## 2026-04-27: sparse_mla_score split-K bf16 MMA — SHIPPED FLAG-OFF, REAL WIN

`sparse_mla_workspace_score_mma_splitk_kernel` + reduce kernel added to
`csrc/sm120_sparse_mla_decode.cu`. Splits head_dim=512 four ways across
blocks (128 elems per block), emits float32 partials, then a small reduce
kernel sums the splitk dim and applies softmax_scale + validity mask. Grid is
`(ceil(slots/8), batch, splitk=4)` → 512 blocks at B=8/topk=128 (vs 188 SMs,
~3 waves) where the original SCORE_MMA produced too few blocks (~128) to hide
launch + MMA setup. M=32 (all `active_heads` together in one block).
Triggers with `DG_SM120_FAST_SPARSE_MLA=1 DG_SM120_SCORE_MMA_SPLITK=1`.

Math validated: bit-exact OUT/LSE vs FAST_SPARSE_MLA reference across full
production shape (batch=8, heads=32, head_dim=512, topk=128); MMA fragment
layout verified via `kScalarCheck` template flag (FMA oracle reading the
SAME smem) — bit-exact with the `mma.sync` path. Activation proven via
intentional 1.0001f bug test (output diverged → bug reverted → bit-exact
again).

**Perf result (cold-cache, nonce-prefix bench, TP=4 on 4× Pro 6000)**:

| Tokens | FAST_SPARSE_MLA | + SCORE_MMA_SPLITK | Δ prefill |
|---:|---:|---:|---:|
|   860 | 3058 | 2860 | -6% |
|  3320 | 3427 | 3479 | +1.5% |
|  6595 | 3274 | 3377 | +3.1% |
| 13151 | 3135 | 3309 | +5.5% |
| 26265 | 3057 | 3255 | +6.5% |

Quality probe vs FAST_SPARSE_MLA baseline: 0/5 semantic regressions. Soft
token-level diffs only (CoT wording: "Let's think step by step" vs "Let's go
step by step" — same arithmetic). Needle code (`BLUEHORIZON-7142`)
correctly emitted. `factual_capital` actually *fixed* a bug in baseline
(Italy: Venice→Rome).

**Decision**: ship flag-gated, default OFF until production rig confirmation.
`DG_SM120_SCORE_MMA_SPLITK=1` is the recommended opt-in. Pairs with
`DG_SM120_FAST_SPARSE_MLA=1`. Activation gated on
`fast_path && active_heads <= 32`. Branch:
`feat/sm120-flag-gated-mma-kernels`, commit a7ccdb7.

Future work in benefit order: (a) cp.async double-buffer K to overlap loads
with MMA, (b) MHC_PRE tile widening so `--max-num-batched-tokens` can grow
past 4096 (8k still crashes in `vllm.mhc_pre.default` tilelang kernel).

## 2026-04-27 evening: NCCL env-var sweep — DEAD END (saved to memory)

Captured `NCCL_DEBUG=INFO` topology: NCCL 2.28.9/cuda13.0, 4× Pro 6000 over PCIe
Gen5 (no NVLink), Trees + Ring built at init, transport SHM/direct.

A/B against SPLITK baseline:
- `NCCL_ALGO=Tree` alone → CRASHED at init (`ncclAllGather` returned 5 /
  ncclInvalidUsage; Tree-only forbids the AllGather paths vLLM uses)
- `NCCL_ALGO=Ring,Tree` + `NCCL_PROTO=LL128` → -5-6% prefill, -3-10% decode
  regression across ctx 860/3320/6595/13151/26265
- `NCCL_NTHREADS=512` → neutral (within ±1.5% noise)

Conclusion: AllReduce on this rig is **PCIe-bandwidth-bound**, not
algo-bound. Ring is already the right choice; protocol/thread-count knobs
don't unlock more bandwidth. Memory: `feedback_nccl_config_dead_end.md`.

## 2026-04-27 evening: mhc_pre 8k batched-tokens crash — DIAGNOSED + workaround in place

Root cause: `compute_num_split` in `vllm/model_executor/layers/mhc.py` returns
`n_splits = n_sms // grid_size`. For large batched-token counts the
`grid_size` rises and `n_splits` can drop to 1; when a process has already
JIT-compiled the kernel for `n_splits=2`, the recompile to `n_splits=1`
hits an inductor PTX codegen bug on SM_120. Crash signature is during
mhc_pre forward at 8k batched tokens.

Workaround: env-flag-gated lower bound on `compute_num_split`.
`DG_SM120_MHC_NSPLITS_MIN=2` clamps `split_k = max(split_k, 2)`. Default
is 1 (today's behavior). Single edit in `mhc.py` lines 24-26. Validation
needs a vLLM restart with the env set + a 8k-batched-tokens prompt; not
yet tested in prod. No code other than `mhc.py` touched (kernel is
unchanged; we just suppress the n_splits=1 specialization).

## 2026-04-27 evening: flash-attention-style score+softmax+output fusion — DESIGN PARKED, IMPLEMENTATION DEFERRED

Largest remaining single prefill win per the post-SPLITK profile:

| GPU kernel                                  | ms   | % of GPU  |
|--------------------------------------------:|-----:|----------:|
| `sparse_mla_workspace_score (qstat)`        |  425 |    24%    |
| `sparse_mla_workspace_output (sstat)`       |  256 |    14%    |
| `sparse_mla_softmax_kernel`                 |   49 |     3%    |

41% of prefill GPU time across three kernels with the same data flow:
score = QK, softmax, output = PV. They write a fp32 `scores` tensor of
shape `[num_tokens, active_heads, topk]` to GDDR7 between kernels —
~54 MiB read+write at ctx=6595/topk=128. Classic flash-attention setup.

### Design

Single fused kernel `sparse_mla_prefill_flash_kernel`. **Grid**: 1 block
per `token` (collapse all `active_heads=32` into the M-dim, keeps MMA
fed). **Threads**: 128 (4 warps).

**Per-block smem layout**:
- `Q[active_heads, kHeadDim]`: 32×512×bf16 = 32 KiB (loaded once at start)
- `KV_tile[TILE_K, kHeadDim]`: TILE_K×512×bf16, where TILE_K=32
  → 32 KiB (overwritten per tile; sparse-MLA uses K==V from same `kv`
  tensor, so single load serves both QK and PV)
- `O_accum[active_heads, kHeadDim]`: 32×512×fp32 = 64 KiB (running output)
- `m[active_heads]`, `l[active_heads]`: tiny (256 B total)

Total: ~128 KiB. Pro 6000 SM_120 default block smem cap is ~99 KiB but
configurable to ~228 KiB via `cudaFuncAttributeMaxDynamicSharedMemorySize`.
Headroom is fine.

**Per-tile loop** (TILE_K=32 candidates per pass, ⌈topk/32⌉ tiles):
1. Cooperative gather: load `KV_tile` from `kv[indices[token, k]]` for
   k in tile; mask k≥limit / invalid index slots to a sentinel that
   produces -inf score.
2. QK MMA: `S[heads, TILE_K] = Q @ KV_tile.T` using
   `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` over kHeadDim
   in K-dim (32 chunks of K=16 → 32 fma-accumulated MMAs per tile).
   Same fragment layout as SCORE_MMA_SPLITK. M=32 → two m16 sub-tiles.
3. Online-softmax update: per-head `tile_max[h] = max_k S[h,k]`,
   `new_max = max(m[h], tile_max)`, `rescale = exp(m[h] - new_max)`,
   `P[h,k] = exp(S[h,k] - new_max)`, `tile_sum = sum_k P[h,k]`.
4. Output rescale + accumulate: `O_accum[h,d] = rescale*O_accum[h,d] +
   sum_k P[h,k] * KV_tile[k,d]` (PV MMA with K=TILE_K=32, M=32, N=8 along
   head_dim — covered by 64 MMAs along the head_dim).
5. `l[h] = rescale*l[h] + tile_sum`; `m[h] = new_max`.

**Final**: per-head `inv_denom = 1/l[h]`; `gate = sigmoid(lse - sink)`
when attn_sink is present; `lse = log(l) + m`; write
`out[token,h,d] = O_accum[h,d] * inv_denom * gate`.

### Expected savings

- Eliminates the `[num_tokens, active_heads, topk]` fp32 scratch
  (54 MiB of GDDR7 traffic at ctx=6595/topk=128).
- Eliminates two kernel launches + their device-side syncs.
- KV is read once per tile (not twice across score+output kernels).

Optimistic upper bound: 41% → ~20-25% of prefill, net ~10-15% e2e at
long ctx. Realistic target: ~6-10% at long ctx, given that
`active_heads=32, topk=128` is small enough that the SPLITK score
kernel already has decent occupancy and L2 reuse.

### Risks

- Per-block smem ~128 KiB requires `cudaFuncSetAttribute(MaxDynamicSmem)`
  bumped above the default; must verify Pro 6000 SM_120 supports the
  required size at this kernel's register/occupancy point.
- Block grid drops from `(topk/16) × active_heads × num_tokens` (split
  score kernel) to `num_tokens` blocks. At ctx=6595/TP=4 (~1650 tokens/
  shard), that's ~9 blocks/SM on 188 SMs — borderline. Below ctx=2k
  could occupancy-starve.
- Online-softmax with running `O_accum[heads, head_dim]` in smem (64 KiB)
  needs a coherent thread→element mapping; PV MMA writes back into the
  same smem tile, so block-sync points need careful placement.
- bf16 fragment layout for `m16n8k16` is well-trodden in SCORE_MMA_SPLITK;
  reusing it cuts risk. Add `kScalarCheck` template flag (FMA oracle on
  same smem) before trusting mma.sync output.

### Why deferred

This is multi-day kernel work (design, MMA layout, smem layout, scalar
oracle, parity, A/B, quality probe, default-flip decision). The smaller
mhc_pre clamp + NCCL findings + OUTPUT_MMA neutral result already
documented during this autonomous block. Not starting flash-fusion
without user sign-off on scope.

Next concrete step when resumed:
1. Build a scalar-FMA fused prototype (no MMA) at the same single-block
   shape — validates online-softmax + KV-tile-reuse memory pattern.
   Expected to run slower than SPLITK due to FMA throughput, but tells
   us whether the algorithm is correct end-to-end.
2. Add `mma.sync` for QK and PV under `kScalarCheck` template flag.
3. Bench cold-cache nonce-prefix sweep at ctx 860/3320/6595/13151/26265
   to decide if the kernel ships flag-on or flag-off.

## 2026-04-27: sparse_mla_workspace_output bf16 MMA — SHIPPED FLAG-OFF, NEUTRAL

`sparse_mla_workspace_output_mma_kernel` added to
`csrc/sm120_sparse_mla_decode.cu` and dispatched from
`launch_sparse_mla_decode_from_workspace_split` when
`DG_SM120_OUTPUT_MMA=1` (and fast_path + active_heads ≤ 32 +
candidate_slots ≤ 128). M=32 N=8 K=128, one block per (batch, head_dim/8)
= 8×64 = 512 blocks. Same fragment layout as SCORE_MMA_SPLITK; validated
bit-exact via `DG_SM120_OUTPUT_MMA_SCALAR_CHECK=1` FMA oracle (max_abs
3e-5 = 1 bf16 ULP, mean_abs 2e-10).

**Perf result (cold-cache, OUTPUT_MMA on top of SPLITK baseline, TP=4)**:

| Tokens | baseline pf/dc | +OUTPUT_MMA pf/dc | Δ prefill | Δ decode |
|---:|---:|---:|---:|---:|
|   860 | 2297 / 119.9 | 2336 / 119.8 | +1.7% | -0.1% |
|  3320 | 3267 / 108.9 | 3225 / 116.0 | -1.3% | +6.5% |
|  6595 | 3284 / 111.6 | 3234 / 107.0 | -1.5% | -4.1% |
| 13151 | 3235 / 105.4 | 3221 / 106.3 | -0.4% | +0.9% |
| 26265 | 3233 / 102.5 | 3206 / 97.2  | -0.8% | -5.2% |

Mixed signs, magnitudes within run-to-run noise — workspace_output is
**launch-overhead-bound, not flop-bound** (~33M FLOPs total per decode
step distributed across 512 blocks; MMA fragment-setup ≈ FMA cost at
this M/K). Decision: ship flag-gated default OFF, document as a
neutral exploration so we don't redo this. Don't re-bench unless
active_heads grows past 32 or topk past 128.

## 2026-04-27 night: sparse_mla_score bf16 MMA kernel — SHIPPED FLAG-OFF, NO WIN

## 2026-04-27 night: sparse_mla_score bf16 MMA kernel — SHIPPED FLAG-OFF, NO WIN

`sparse_mla_workspace_score_mma_kernel` added to
`csrc/sm120_sparse_mla_decode.cu` (templated on bf16 q/kv,
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, M=16 N=32 K=16,
4 warps × (M=16, N=8) sub-tiles, K=512 streamed in 64-elem chunks via
shared mem, kScalarCheck template flag for fragment-layout validation).
Triggers with `DG_SM120_FAST_SPARSE_MLA=1 DG_SM120_SCORE_MMA=1`.

Math validated: max_abs ≤ 1 bf16 ULP across 5 production shape configs
vs the qstat scalar reference; quality probe vs `fast_sparse_mla` baseline
shows 0/5 HARD-DIVERGE (all 5 prompts produce semantically correct output;
3/5 minor wording soft-diverge from ULP noise).

**Perf result (cold-cache, nonce-prefix bench)**:

| Tokens | FAST_SPARSE_MLA | + SCORE_MMA | Δ prefill | Δ decode |
|---:|---:|---:|---:|---:|
|   860 | 3058 | 2509 | -18% | flat |
|  3320 | 3352 | 2827 | -16% | flat |
|  6595 | 3254 | 2781 | -15% | flat |
| 13151 | 3132 | 2754 | -12% | flat |
| 26265 | 3063 | 2739 | -11% | flat |

The kernel is correct but slower than the qstat scalar path on this rig.
Likely culprits: launch grid is `(ceil(slots/32), ceil(heads/16), batch)` —
typically `1 × 2 × batch` blocks for active_heads=32 with topk≤128, vs the
qstat path's `(slots, heads, batch)` which exploits more parallelism;
single-buffered K-streaming with no async-copy overlap; m16n8k16 needs M=16
but active_heads=32 splits awkwardly. Decode path appears unaffected (this
shape also dominates only at large topk, which is rarer in tested decode).

**Decision**: ship flag-gated, default OFF. Math is sound and quality
passes; the kernel itself is just dispatch-shaped wrong for this rig's
typical (small-topk × small-heads) sparse-MLA decode shape. Future work:
either (a) tile differently (e.g. one block per slot-tile spanning all
heads) or (b) add cp.async double-buffering to overlap K loads. Not chasing
either tonight.

## 2026-04-27 night: TP=2 sweep ruled infeasible

Per-card VRAM at TP=4 + DSv4-Flash: ~78 GB used / 96 GB total. At TP=2 the
weights per rank would roughly double (~120 GB) and won't fit without enabling
expert parallel — which user has off by default per prior experiments. Skipped.
Hypothesis (less NCCL overhead) was sound but the comm savings on this rig
(~10pp NCCL → ~5pp) wouldn't recover the doubled per-rank compute even if
memory fit. Recorded the analysis in task #39 description.

## 2026-04-27 night: hc_prenorm TF32 MMA kernel

`hc_prenorm_block_reduce_mma_kernel` added to
`csrc/sm120_hc_prenorm_fallback.cu`. Gated by `DG_SM120_HC_PRENORM_MMA=1`
(default OFF). Re-tiles the original "1 block per row" kernel into
"1 block per 16-row M-tile × split", uses
`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`.

Block layout: 4 warps × 32 threads = 128. Each warp covers an M=16 × N=8 tile
of the output for its split. K is streamed in 128-element chunks via dynamic
smem (size = (16 + n) × 128 × 4 bytes; for n=24 → 20 KB). bf16 A converts to
tf32 losslessly (bf16 mantissa is 7 bits, tf32 holds 10). B is rounded to tf32
once on smem load (was rounded per-element on every read in the scalar path).

`sqr_sum` is computed alongside MMA: each warp owns 4 rows of the M=16 tile,
warp-shuffle-reduces sum-of-squares from a_smem after each K-chunk MMA pass.

**Fragment-layout bug caught and fixed**: first build had hard-diverged 4/5
quality probe prompts. Root cause: A/B fragment layout was the stride-2
contiguous (fp16 m16n8k16) pattern, but `m16n8k8 .f32.tf32.tf32.f32` actually
uses STRIDE-4 column layout — `a[0..3] = A[gID|gID+8][tid|tid+4]`,
`b[0..1] = B[tid|tid+4][gID]`. Caught by adding a `kScalarCheck` template
flag that swaps the `mma.sync` for hand-rolled scalar matmul reading the SAME
smem; bit-identical to scalar fallback at m=16,n=8,k=8 confirmed the fragment
fix. Memory: `feedback_mma_fragment_layout.md`.

E2E sweep (sequential firer, on top of FAST_SPARSE_MLA):

| Tokens | FAST_SPARSE_MLA | + HC_PRENORM_MMA | Δ |
|---:|---:|---:|---:|
|   860 | 2616 | 3010 | +15% |
|  3320 | 3361 | 3527 | +5% |
|  6595 | 3284 | 3439 | +5% |
| 13151 | 3115 | 3327 | +7% |
| 26265 | 3083 | 3231 | +5% |

**Quality**: probe vs fast_sparse_mla baseline shows 1/5 HARD-DIVERGE
(`cot_reasoning`: model emits JSON-like garbage at token 3, "broken structure")
and 1/5 needle truncation (`BLUEHORIZON-714` instead of `-7142` — factual
error). Source: legitimate tf32 vs f32 accumulation-order ULP noise that
greedy argmax tips into different tokens on tight-margin prompts. The math is
within ~2e-5 relative of scalar reference (verified at all production shapes
m∈{8,137,2048,4096}, n=24, k=16384) — kernel is correct, but the noise is
enough to flip downstream argmax on this model.

**Decision**: flag stays default OFF. Speed gain is real, but the HARD-DIVERGE
fails the user's quality bar. Available as opt-in `DG_SM120_HC_PRENORM_MMA=1`
for deployments that tolerate tf32-precision argmax flips.

## 2026-04-27 night: TP=4 + FAST baseline (locked-in reference)

| Tokens | tok/s |
|---:|---:|
|   860 | 2616 |
|  3320 | 3361 |
|  6595 | 3284 |
| 13151 | 3115 |
| 26265 | 3083 |

This is the floor for tonight's optimization measurements.

---

## 2026-04-26 late evening: FAST_SPARSE_MLA Q-stat smem caching

`sparse_mla_workspace_score_tiled_qstat_kernel` and
`sparse_mla_workspace_output_sstat_kernel` added to
`csrc/sm120_sparse_mla_decode.cu`. Gated by `DG_SM120_FAST_SPARSE_MLA=1`,
default OFF until user signs off. Pattern: cache `Q[b,h,:]` (2 KB static smem)
and `scores[b,h,:]` (≤16 KB dynamic smem) once per block to eliminate the
gmem-read redundancy in the inner accumulation loops. Multiplication and
accumulation order preserved at the source level.

Quality (5-prompt deterministic harness, `quality_probe.py`):
- 4/5 prompts: bit-identical tokens AND zero logprob drift
- needle (4845-token long-context): same answer token, divergence at token
  10 in post-answer continuation. Cause: compiler FMA-fusion / loop-unroll
  reordering inside the new smem-cached inner loop produces tiny float
  differences that, combined with TP=4 NCCL run-to-run noise, flip greedy
  argmax late. Within the kernel itself the math is bit-identical at the
  source level. Per user feedback, "soft-diverge" is acceptable.

E2E sweep (cache-busting, sequential):

| Tokens | MMA Q-stat (FAST off) | FAST_SPARSE_MLA on | Δ |
|---:|---:|---:|---:|
|   864 | 2698 | 2844 | +5% |
|  3316 | 3318 | 3405 | +3% |
|  6590 | 3180 | 3310 | +4% |
| 13148 | 3027 | 3183 | +5% |
| 26263 | 2893 | 3071 | +6% |

Stacks cleanly on MMA indexer — net +17% at 26k tokens vs. pre-MMA vec.

## Updated profile (post-FAST_SPARSE_MLA, 6595-token prefill, rank 0)

| GPU kernel                                  | ms   | % of GPU  |
|--------------------------------------------:|-----:|----------:|
| `sparse_mla_workspace_score (qstat)`        |  425 |    24%    |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`  |  271 |    15%    |
| `sparse_mla_workspace_output (sstat)`       |  256 |    14%    |
| `hc_prenorm_block_reduce_kernel`            |  189 |    11%    |
| `vectorized_gather_kernel` (aten::index)    |  122 |     7%    |
| `fill_scale_kernel`                         |  116 |     7%    |
| cutlass GEMM kernels (top 2)                |  137 |     8%    |
| `sparse_mla_softmax_kernel`                 |   49 |     3%    |
| `mqa_logits_fp8_mma_kernel`                 |    9 |    0.5%   |

Score went 513→425 ms (-17%), output went 373→256 ms (-31%). Indexer is
truly off the hot list now (0.5%).

## Next-session optimization candidates (in benefit order)

1. **`sparse_mla_workspace_score` → MMA (24% potential, ~+10% e2e)**: stack on
   top of qstat. Need to batch active_heads (32) together per block to
   feed M=32, N=32, K=512 bf16 MMA — current launch is 1 (b, h) per block
   which makes M=1 (GEMV-shaped, no tensor cores). Q-stationary smem layout
   already there. Target instruction:
   `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.
2. **`hc_prenorm_block_reduce_kernel` → TF32 MMA (11% potential, ~+5% e2e)**:
   tall-skinny GEMM with `n ≤ 32`. Currently scalar fp32 FMA (FMA-bound, not
   memory-bound — L2 already absorbs the B reuse). Target:
   `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`. Side-output
   `sqr_sum` = sum(A²) needs to coexist with the MMA path.
3. **`fill_scale_kernel` skip-fill window (7% potential, ~+4% e2e)**: the
   `DG_SM120_MOE_SKIP_SFA_FILL_MAX_M=16` cap is conservative for prefill
   where m-per-group routinely exceeds 16. Investigate whether
   `prepare_compact_grouped_kernel` writes every active scale slot, and if
   so raise/remove the cap. Pure env-flag tweak, no kernel work, but needs
   careful audit of the prepare-kernels' coverage.

## Decisions to confirm next session

- Whether to flip `DG_SM120_FAST_SPARSE_MLA=1` default ON. Current evidence:
  +3-6% at every size, no hard-quality regression, soft-diverge on long-
  context only. User has approved flag-default flips at their discretion.

---

(Earlier notes preserved below.)

## 2026-04-26: Indexer MMA work (vec → tensor cores) and where it leaves us

## 2026-04-26: Indexer MMA work (vec → tensor cores) and where it leaves us

`mqa_logits_fp8_mma_kernel` v2 (Q-stationary block) is shipped and ON by default
(`DG_SM120_MMA_INDEXER=1` in `/ai/vllm/dsv4-flash-vllm-sm120.sh`). 32/32 numerics
configs pass at diff ~6.8e-4. E2E sweep vs vec kernel (sequential firer,
cache-busting, TP=4):

| Tokens | Vec t/s | MMA q-stat | Δ |
|---:|---:|---:|---:|
|   860 | 2819 | 2698 | −4% |
|  3320 | 3320 | 3318 | flat |
|  6598 | 3144 | 3180 | +1% |
| 13151 | 2907 | 3027 | +4% |
| 26262 | 2621 | 2893 | **+10%** |

Full notes in `SM120_MQA_LOGITS_NOTES.md`. The earlier per-tile MMA design (v1)
regressed e2e despite passing numerics — it re-read Q from gmem per (m, 8-n)
tile. v2 stages Q[m, :, :] into shared memory once per block, 8 warps share it.

**Why so modest e2e gain given 3× kernel speedup?** Indexer is no longer the
dominant fraction of GPU time. Torch-profiler trace of one 8060-token prefill
on rank 0:

| GPU kernel                                  | ms   | % of GPU  |
|--------------------------------------------:|-----:|----------:|
| `sparse_mla_workspace_score_tiled_kernel`   |  513 |    23%    |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`  |  434 |    20%    |
| `sparse_mla_workspace_output_kernel`        |  373 |    17%    |
| `hc_prenorm_block_reduce_kernel`            |  232 |    11%    |
| `vectorized_gather_kernel`                  |  151 |     7%    |
| `fill_scale_kernel`                         |  142 |     6%    |
| cutlass GEMM kernels (top 2)                |  158 |     7%    |
| `sparse_mla_softmax_kernel`                 |   60 |     3%    |
| `mqa_logits_fp8_mma_kernel` (this PR)       |   14 |    0.6%   |

Total observed kernel time ~2200 ms over 2670 ms wall = 82% util. To reach the
user's 10k tok/s target (currently ~3k sequential), we'd need ~3.3× — not
reachable from indexer work alone.

## Next bottleneck: sparse_mla_workspace_score_tiled_kernel

In `csrc/sm120_sparse_mla_decode.cu:1325`. **Same scalar bf16 dot-product
pattern** that the original mqa_logits_kernel had — one `qv * kv` multiply per
d, no tensor cores. Computes `scores[b, h, j] = q[b, h, :] · kv_workspace[b, j, :]`.
Shape: `batch_size × active_heads (32) × candidate_slots (~128)`.

**Plausible fix**: same Q-stationary playbook with `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
(the bf16 tensor-core instruction available on SM_120). Q stationary in smem
across active_heads × d, warp-stride over candidate_slots. Estimated kernel
speedup 3–5× → e2e maybe +15-20%.

The `sparse_mla_workspace_output_kernel` immediately below it (line 1381) has
the same shape (sum over j of `scores[b,h,j] * kv_workspace[b,j,d]`) and would
benefit from the same treatment. Together: 23% + 17% = 40% of GPU time across
two kernels with the same structure and the same fix.

## All-reduce (PCIe Gen 5, no NVLink) is 20% of GPU time

`ncclDevKernel_AllReduce_Sum_bf16_RING_LL`. This is fundamental to TP=4 over
PCIe; symmetric memory and custom all-reduce are already off (right call for
this rig). Not much to do here without going to TP=2 + DP, or to a TP=1
single-GPU footprint (~150 GB unweighted DSv4 doesn't fit on a single 96 GB
card, so a non-starter today).

---

(Earlier sglang-era notes preserved below for context.)

Last updated: 2026-04-25 (storm shutdown)

## Where we are

- 4× RTX Pro 6000 Blackwell Workstation, TP=4, sglang docker image `lmsysorg/sglang:deepseek-v4-blackwell`
- Custom sparse-decode kernel injected via `PYTHONPATH=/dsv4` + sitecustomize hook (no image mods)
- Checkpoint: `/nvme/models/safetensors/DeepSeek-V4-Flash-FP8/` (274 GB, sgl-project repo)
- Launchers: `/ai/vllm/dsv4-flash-sm120.sh` → `scripts/launch_dsv4_flash_sm120.sh`

## Big win this session

Moved `cudaFuncSetAttribute` out of the per-call launcher into a one-time static lambda in
`csrc/sm120/decode/sparse_decode_instantiation.cu`. Runtime API calls aren't capturable into
CUDA graphs — having it inline forced `--disable-cuda-graph`. After the fix, decode jumped
**~10× (5 → 51 tok/s)** at context-0 single-stream. CUDA graphs now enabled (default).

## Current perf (post-fix)

| Scenario | Throughput |
|---|---|
| Single-stream decode @ ctx-0 | 51 tok/s |
| Single-stream decode @ ctx-8k | 25 tok/s |
| 4-concurrent decode @ ctx-0 | 188 tok/s aggregate |
| TTFT @ ctx-0 | 0.19 s |
| TTFT @ ctx-128k | 63 s |
| Prefill | 2071–3037 tok/s (nearly flat across 8k–128k) |

## Last edit (UNTESTED — storm shutdown)

`scripts/launch_dsv4_flash_sm120.sh`: `--chunked-prefill-size 8192` → `16384`.

If it OOMs on startup, revert to 8192. If it works and prefill improves, push to 32768 next.

## Next steps in order

1. **Test chunked-prefill 16384** (this edit). A/B vs old number. Push to 32768 if good.
2. **Try `--attention-backend flashinfer`** instead of `compressed`. FlashInfer's prefill
   kernels are gold-standard on Blackwell; `compressed` may not be Tensor-Core-optimal for
   SM_120 prefill (same kind of issue we hit on decode).
3. **Autotune MoE configs** in-container (~30–60 min):
   ```
   python benchmark/kernels/fused_moe_triton/benchmark_moe.py --model /workspace/model --tune
   ```
   Writes `E=256,N=512,...,Workstation_Edition,fp8_w8a8,block_shape=[128, 128].json`
   (and `_down` variant) to silence the warnings AND speed up MoE.
4. **MTP/EAGLE bug** at `deepseek_v4_backend_radix.py:424` — was on the V1 roadmap but never
   tackled. Speculative decode would help single-stream decode further.

## MoE config dead-end (logged so we don't re-investigate)

voipmonitor's `blackwell-llm-docker/configs/` ships:
- `E=512,N=256,...,Server_Edition,fp8_w8a8.json` — different model
- `E=257,N=256,...,Server_Edition.json` — DSv3-shape, half N
- `E=256,N=256,...,Server_Edition,fp8_w8a8.json` — half N, no block_shape
- `E=128,N=704,...` — different model (Qwen3-30B-A3B-ish)

We need `E=256,N=512,...,Workstation_Edition,fp8_w8a8,block_shape=[128, 128].json`.
Nobody has it pre-baked. sglang's lookup is exact-match on the filename string, so no
amount of renaming saves us. **Just autotune.**

## Encoding_dsv4 500 errors (NOT a config issue)

`TypeError: sequence item 0: expected str instance, list found` at
`encoding_dsv4.py:336` (`"\n\n".join(parts)`). DSv4's encoder doesn't accept the
multimodal-array `content` form. The benchmark client (3.1.0.162) was sending list-form
content. Either send plain string content, or patch sglang's encoder. Server-side flag
won't fix it.

## Model geometry crib sheet

- 43 layers, hidden 4096, 64 attention heads, 1 KV head (MQA), 512-dim heads
- 256 routed experts (6/token + 1 shared)
- FP8 e4m3 weights, UE8M0 scales, 128×128 blocks
- KV layout: 584 B/token = 448 NoPE FP8 + 64 RoPE BF16 + 8 UE8M0 scale bytes per page slot
- 274 GB checkpoint
- Decode is dispatch-bound (kernel-launch overhead dominated), not compute or bandwidth
  bound — that's why GPUs only pulled 150 W and inter-card BW was single-digit GB/s
  before the CUDA-graph fix.

## Files of interest

- `scripts/launch_dsv4_flash_sm120.sh` — canonical launcher (env vars + sglang flags)
- `csrc/sm120/decode/sparse_decode_kernel.cuh` — the V1 kernel; line ~340-ish has a
  scalar BF16 multiply loop in QK^T that uses NO Tensor Cores. Future kernel work.
- `csrc/sm120/decode/sparse_decode_instantiation.cu` — has the static lambda fix
- `deepseek_v4_kernel/_patch.py` — monkey-patch entry point
- `DEEPSEEK-V4-FLASH.md` — original doc (claims `tool_chat_template_deepseekv4.jinja`
  ships in checkpoint with sha256 f7b71796...aa27 — FALSE, file doesn't exist; we use
  the in-image `tool_chat_template_deepseekv32.jinja` instead)
