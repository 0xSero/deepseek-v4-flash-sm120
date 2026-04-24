// =====================================================================
// SM_120 sparse decode kernel for DeepSeek-V4-Flash.
//
// Upstream flash_mla.cuda.sparse_decode_fwd only ships SM_90 (WGMMA) and
// SM_100 (TCGEN05) classes.  RTX Pro 6000 Blackwell Workstation (SM_120)
// has neither, so we rebuild the kernel using portable CUDA-core BF16
// dot-products.  We keep the same on-disk packed KV format and query
// layout as the sglang deepseek_v4 attention backend.
//
// DeepSeek-V4-Flash sparse KV layout (sglang "nope_fp8_rope_bf16_pack",
// flash_mla MODEL1).  Per-page (page_block_size = P = 256):
//
//   [0                               ..  P * 576)       nope_rope section
//       per token:  [0 .. 448)  NoPE FP8 e4m3fn  (448 bytes)
//                   [448 .. 576) RoPE BF16       (64 * 2 = 128 bytes)
//
//   [P * 576                         ..  P * (576+8))   scale section
//       per token:  8 fp8_e8m0 (== UE8M0) bytes — 7 active + 1 pad
//                   scale_i = 2^(byte_i - 127)
//                   scale_i applies to NoPE[i*64 .. (i+1)*64)
//
// The tensor passed from sglang has shape (num_pages, P, 1, 584) viewed
// on top of a (num_pages, page_bytes_padded) uint8/fp8 storage.  The
// `.view()` shape is cosmetic — the true layout is as above and our
// kernel reads the raw bytes directly using stride_kv_block (bytes between
// pages) and the compile-time nope+rope / scale offsets.
//
// Q tensor: BF16 [b, s_q, h_q, 512] = [NoPE(448) | RoPE(64)].
// Output : BF16 [b, s_q, h_q, 512]; in MLA the value block equals the
// dequantised K row including RoPE (sglang applies the inverse RoPE to
// the output's trailing 64 elements in a post-processing step).
//
// Supported features (V1):
//   * d_qk = d_v = 512 (448 NoPE + 64 RoPE)
//   * h_q multiple of BLOCK_M_HEADS = 16; typically 64 per TP shard
//   * attn_sink, topk_length, extra_k_cache (SWA sidecar)
//
// Not yet implemented (surfaced with TORCH_CHECK in api/sparse_decode.cpp):
//   * split-KV (num_sm_parts > 1)
//   * s_q > 1  (MTP must be flattened into outer `b`)
//
// SMEM budget (per CTA, HEADS_PER_CTA = 16):
//   sQ        :  16 x 512  BF16 = 16 384 B
//   sK        :  32 x 512  BF16 = 32 768 B
//   sP        :  16 x  32  FP32 =  2 048 B
//   sO        :  16 x 512  FP32 = 32 768 B
//   running stats / pad                   256 B
//   ------------------------------------------
//   total                           ~84 KB  (SM_120: 99 KB/SM)
// =====================================================================
#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "common/cutlass_shim.h"
#include "common/defines.h"
#include "common/params.h"

namespace dsv4_kernel {
namespace sm120 {

static constexpr int BLOCK_M_HEADS = 16;
static constexpr int KV_CHUNK      = 32;
static constexpr int HEAD_DIM_NOPE = 448;
static constexpr int HEAD_DIM_ROPE = 64;
static constexpr int HEAD_DIM_QK   = HEAD_DIM_NOPE + HEAD_DIM_ROPE;  // 512
static constexpr int HEAD_DIM_V    = HEAD_DIM_QK;                    // 512

static constexpr int QUANT_TILE    = 64;                          // per-scale block
static constexpr int NUM_ACTIVE_SCALES = HEAD_DIM_NOPE / QUANT_TILE;  // 7
static constexpr int NUM_SCALE_SLOTS   = 8;                          // 7 + 1 padding
static constexpr int NOPE_BYTES    = HEAD_DIM_NOPE;                  // 448
static constexpr int ROPE_BYTES    = HEAD_DIM_ROPE * 2;              // 128
static constexpr int NOPE_ROPE_BYTES = NOPE_BYTES + ROPE_BYTES;      // 576
// Per-token bytes as reported by the sglang 4-D tensor (cosmetic view):
static constexpr int K_BYTES_PER_TOKEN =
    NOPE_ROPE_BYTES + NUM_SCALE_SLOTS;                               // 584

static constexpr int NUM_WARPS     = 4;
static constexpr int NUM_THREADS   = NUM_WARPS * 32;

// UE8M0 byte -> multiplicative scale.
__device__ __forceinline__ float ue8m0_to_scale(unsigned char b) {
    int e = static_cast<int>(b) - 127;
    return __powf(2.0f, static_cast<float>(e));
}

// FP8 e4m3 quad -> BF16 quad with a single (shared) FP32 scale.
__device__ __forceinline__ void fp8x4_to_bf16x4(uint32_t bits, float scale,
                                                 __nv_bfloat16 out[4]) {
    __nv_fp8_e4m3 f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        f.__x = static_cast<unsigned char>((bits >> (i * 8)) & 0xFFu);
        out[i] = __float2bfloat16_rn(static_cast<float>(f) * scale);
    }
}

struct SmemLayout {
    __nv_bfloat16 sQ[BLOCK_M_HEADS][HEAD_DIM_QK];   // 16 384 B
    __nv_bfloat16 sK[KV_CHUNK][HEAD_DIM_QK];         // 32 768 B
    float         sP[BLOCK_M_HEADS][KV_CHUNK];       //  2 048 B
    float         sO[BLOCK_M_HEADS][HEAD_DIM_V];     // 32 768 B
    float         sRowMax[BLOCK_M_HEADS];
    float         sRowSum[BLOCK_M_HEADS];
};

// -------------------------------------------------------------------
// Dequantise & load one KV chunk of up to KV_CHUNK tokens into smem.sK
// as BF16 [KV_CHUNK][512]. `kv_bytes_base` points at the start of the
// packed KV arena for the current "h_kv" (MQA, h_kv=1).
// -------------------------------------------------------------------
__device__ __forceinline__ void load_kv_chunk(
    SmemLayout &smem,
    const uint8_t *kv_bytes_base,
    int stride_kv_block,      // bytes between pages
    int stride_kv_row,        // bytes between tokens inside a page
    int page_block_size,
    const int *indices_base,
    int token_offset,
    int valid_tokens,
    int warp_id, int lane_id) {

    for (int t = warp_id; t < KV_CHUNK; t += NUM_WARPS) {
        // Out-of-range / invalid tokens → zero row so softmax masks them.
        if (t >= valid_tokens) {
            for (int v = lane_id; v < HEAD_DIM_QK; v += 32) {
                smem.sK[t][v] = __float2bfloat16_rn(0.0f);
            }
            continue;
        }
        int flat_idx = indices_base[token_offset + t];
        if (flat_idx < 0) {
            for (int v = lane_id; v < HEAD_DIM_QK; v += 32) {
                smem.sK[t][v] = __float2bfloat16_rn(0.0f);
            }
            continue;
        }
        int block_idx    = flat_idx / page_block_size;
        int row_in_block = flat_idx % page_block_size;

        // Base of the page (bytes between pages = stride_kv_block).
        const uint8_t *page_base =
            kv_bytes_base +
            static_cast<size_t>(block_idx) * static_cast<size_t>(stride_kv_block);

        // Per-token nope+rope segment (contiguous 576 B per token, within
        // the page's nope_rope section).
        const uint8_t *tok_nope_rope =
            page_base + static_cast<size_t>(row_in_block) * NOPE_ROPE_BYTES;

        // Scale section starts right after the page_block_size nope+rope
        // rows.  8 bytes per token (7 active + 1 padding).
        const unsigned char *scales_u8 =
            page_base + static_cast<size_t>(page_block_size) * NOPE_ROPE_BYTES +
            static_cast<size_t>(row_in_block) * NUM_SCALE_SLOTS;

        float s[NUM_ACTIVE_SCALES];
        #pragma unroll
        for (int i = 0; i < NUM_ACTIVE_SCALES; ++i) {
            s[i] = ue8m0_to_scale(scales_u8[i]);
        }

        // NoPE dequant: 4 elements / lane / iter, 32 lanes -> 128 elements,
        // 4 iters = 512; we only need 448 and clip beyond.
        const uint32_t *fp8_words =
            reinterpret_cast<const uint32_t *>(tok_nope_rope);
        #pragma unroll
        for (int iter = 0; iter < 4; ++iter) {
            int elem_start = iter * 128 + lane_id * 4;   // [0..511]
            if (elem_start >= HEAD_DIM_NOPE) break;
            uint32_t bits = fp8_words[iter * 32 + lane_id];
            int q_tile = elem_start / QUANT_TILE;        // 0..6
            float scale = s[q_tile];
            __nv_bfloat16 out[4];
            fp8x4_to_bf16x4(bits, scale, out);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int dst = elem_start + k;
                if (dst < HEAD_DIM_NOPE) {
                    smem.sK[t][dst] = out[k];
                }
            }
        }

        // RoPE BF16: 64 elements starting at byte 448.
        const __nv_bfloat16 *rope =
            reinterpret_cast<const __nv_bfloat16 *>(tok_nope_rope + NOPE_BYTES);
        if (lane_id * 2 + 1 < HEAD_DIM_ROPE) {
            smem.sK[t][HEAD_DIM_NOPE + lane_id * 2 + 0] = rope[lane_id * 2 + 0];
            smem.sK[t][HEAD_DIM_NOPE + lane_id * 2 + 1] = rope[lane_id * 2 + 1];
        }
    }
}

// -------------------------------------------------------------------
// Main kernel: Grid = (batch * s_q, ceil(h_q / HEADS_PER_CTA)), block = 128.
// -------------------------------------------------------------------
template <int HEADS_PER_CTA = BLOCK_M_HEADS>
__global__ __launch_bounds__(NUM_THREADS, 2)
void dsv4_sparse_decode_kernel(SparseAttnDecodeParams params) {
    extern __shared__ __align__(16) unsigned char _smem[];
    SmemLayout &smem = *reinterpret_cast<SmemLayout *>(_smem);

    const int bs_s_q    = blockIdx.x;
    const int head_bk   = blockIdx.y;
    const int batch_idx = bs_s_q / params.s_q;
    const int s_q_idx   = bs_s_q % params.s_q;
    const int head_base = head_bk * HEADS_PER_CTA;
    if (head_base >= params.h_q) return;
    const int heads_this_cta = min(HEADS_PER_CTA, params.h_q - head_base);

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    int my_topk = params.topk;
    if (params.topk_length) {
        my_topk = min(my_topk, params.topk_length[batch_idx]);
    }
    if (my_topk < 0) my_topk = 0;

    int my_extra_topk = 0;
    if (params.extra_kv) {
        my_extra_topk = params.extra_topk;
        if (params.extra_topk_length) {
            my_extra_topk = min(my_extra_topk, params.extra_topk_length[batch_idx]);
        }
        if (my_extra_topk < 0) my_extra_topk = 0;
    }

    // ---- Load Q ----
    const cutlass::bfloat16_t *q_base =
        params.q + static_cast<size_t>(batch_idx) * params.stride_q_b +
        s_q_idx * params.stride_q_s_q + head_base * params.stride_q_h_q;
    for (int h = 0; h < heads_this_cta; ++h) {
        const cutlass::bfloat16_t *row = q_base + h * params.stride_q_h_q;
        for (int t = tid; t < HEAD_DIM_QK; t += NUM_THREADS) {
            smem.sQ[h][t] = reinterpret_cast<const __nv_bfloat16 *>(row)[t];
        }
    }
    if (tid < BLOCK_M_HEADS) {
        smem.sRowMax[tid] = -INFINITY;
        smem.sRowSum[tid] = 0.0f;
        for (int v = 0; v < HEAD_DIM_V; ++v) smem.sO[tid][v] = 0.0f;
    }
    __syncthreads();

    const int *indices_base =
        params.indices + static_cast<size_t>(batch_idx) * params.stride_indices_b +
        s_q_idx * params.stride_indices_s_q;
    const int *extra_indices_base =
        params.extra_indices
            ? params.extra_indices +
                  static_cast<size_t>(batch_idx) * params.stride_extra_indices_b +
                  s_q_idx * params.stride_extra_indices_s_q
            : nullptr;

    auto process = [&](const uint8_t *kv_bytes_base, int stride_block,
                        int stride_row, int page_block,
                        const int *idx_base, int total_tokens) {
        if (total_tokens <= 0 || idx_base == nullptr) return;
        for (int token_offset = 0; token_offset < total_tokens;
             token_offset += KV_CHUNK) {
            int valid = min(KV_CHUNK, total_tokens - token_offset);
            load_kv_chunk(smem, kv_bytes_base, stride_block, stride_row,
                          page_block, idx_base, token_offset, valid,
                          warp_id, lane_id);
            __syncthreads();

            // --- QK^T (CUDA-core; heads<=16, chunk=32, depth=512) ---
            for (int cell = tid; cell < BLOCK_M_HEADS * KV_CHUNK;
                 cell += NUM_THREADS) {
                int h = cell / KV_CHUNK;
                int k = cell % KV_CHUNK;
                if (h >= heads_this_cta || k >= valid) {
                    smem.sP[h][k] = -INFINITY;
                    continue;
                }
                float acc = 0.0f;
                const __nv_bfloat16 *q_row = smem.sQ[h];
                const __nv_bfloat16 *k_row = smem.sK[k];
                #pragma unroll 4
                for (int d = 0; d < HEAD_DIM_QK; ++d) {
                    acc += __bfloat162float(q_row[d]) * __bfloat162float(k_row[d]);
                }
                smem.sP[h][k] = acc * params.sm_scale;
            }
            __syncthreads();

            // --- Online softmax + P @ V accumulate ---
            for (int h_local = warp_id; h_local < heads_this_cta;
                 h_local += NUM_WARPS) {
                float chunk_max = -INFINITY;
                #pragma unroll
                for (int k = 0; k < KV_CHUNK; ++k) {
                    chunk_max = fmaxf(chunk_max, smem.sP[h_local][k]);
                }
                float old_max = smem.sRowMax[h_local];
                float new_max = fmaxf(old_max, chunk_max);
                float rescale = (old_max == -INFINITY) ? 0.0f
                                                       : __expf(old_max - new_max);
                if (lane_id == 0) {
                    smem.sRowMax[h_local] = new_max;
                    smem.sRowSum[h_local] *= rescale;
                }
                for (int v = lane_id; v < HEAD_DIM_V; v += 32) {
                    smem.sO[h_local][v] *= rescale;
                }

                float local_sum = 0.0f;
                float exps[KV_CHUNK];
                #pragma unroll
                for (int k = 0; k < KV_CHUNK; ++k) {
                    float p = smem.sP[h_local][k];
                    float e = (p == -INFINITY) ? 0.0f : __expf(p - new_max);
                    exps[k] = e;
                    local_sum += e;
                }
                if (lane_id == 0) {
                    smem.sRowSum[h_local] += local_sum;
                }

                for (int v = lane_id; v < HEAD_DIM_V; v += 32) {
                    float acc = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < KV_CHUNK; ++k) {
                        if (k < valid) {
                            acc += exps[k] * __bfloat162float(smem.sK[k][v]);
                        }
                    }
                    smem.sO[h_local][v] += acc;
                }
            }
            __syncthreads();
        }
    };

    // Main KV loop.
    process(reinterpret_cast<const uint8_t *>(params.kv),
            params.stride_kv_block, params.stride_kv_row,
            params.page_block_size, indices_base, my_topk);
    // Optional SWA sidecar.
    if (params.extra_kv) {
        process(reinterpret_cast<const uint8_t *>(params.extra_kv),
                params.stride_extra_kv_block, params.stride_extra_kv_row,
                params.extra_page_block_size, extra_indices_base,
                my_extra_topk);
    }

    // --- Epilogue: normalise, apply attn_sink, writeback ---
    for (int h_local = warp_id; h_local < heads_this_cta; h_local += NUM_WARPS) {
        float row_max = smem.sRowMax[h_local];
        float row_sum = smem.sRowSum[h_local];
        bool  has_sink = params.attn_sink != nullptr;
        float sink    = has_sink ? params.attn_sink[head_base + h_local] : 0.0f;

        float lse_val;
        if (row_sum == 0.0f) {
            lse_val = has_sink ? sink : -INFINITY;
            for (int v = lane_id; v < HEAD_DIM_V; v += 32) {
                smem.sO[h_local][v] = 0.0f;
            }
        } else {
            float log_sum = logf(row_sum) + row_max;
            if (has_sink) {
                float sink_scale = 1.0f / (1.0f + __expf(sink - log_sum));
                for (int v = lane_id; v < HEAD_DIM_V; v += 32) {
                    smem.sO[h_local][v] =
                        (smem.sO[h_local][v] / row_sum) * sink_scale;
                }
                float m = fmaxf(log_sum, sink);
                lse_val = m + logf(__expf(log_sum - m) + __expf(sink - m));
            } else {
                for (int v = lane_id; v < HEAD_DIM_V; v += 32) {
                    smem.sO[h_local][v] = smem.sO[h_local][v] / row_sum;
                }
                lse_val = log_sum;
            }
        }
        if (lane_id == 0) {
            params.lse[static_cast<size_t>(batch_idx) * params.stride_lse_b +
                       s_q_idx * params.stride_lse_s_q + head_base + h_local] =
                lse_val;
        }
    }
    __syncthreads();

    // Write out BF16 output [b, s_q, h_q, 512].
    cutlass::bfloat16_t *out_base =
        params.out + static_cast<size_t>(batch_idx) * params.stride_o_b +
        s_q_idx * params.stride_o_s_q + head_base * params.stride_o_h_q;
    for (int h_local = 0; h_local < heads_this_cta; ++h_local) {
        cutlass::bfloat16_t *row_out = out_base + h_local * params.stride_o_h_q;
        for (int v = tid; v < HEAD_DIM_V; v += NUM_THREADS) {
            reinterpret_cast<__nv_bfloat16 *>(row_out)[v] =
                __float2bfloat16_rn(smem.sO[h_local][v]);
        }
    }
}

}  // namespace sm120
}  // namespace dsv4_kernel
